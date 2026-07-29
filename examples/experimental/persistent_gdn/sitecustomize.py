# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in TensorRT-LLM patch for the persistent GDN RF/TMEM prototype.

Set ``TRTLLM_PERSISTENT_GDN_EXTENSION`` to the compiled extension path.  The
first standard batch-1 decode step runs the shipping implementation so all 24
layer states exist.  At the next step those states are loaded once into the
resident service; every later GDN layer reads and updates that RF/TMEM copy.

This is intentionally a batch-1, non-speculative integration experiment.  At
the configured token limit it releases RF/TMEM without writing the resident
state back to global memory.
"""

import importlib.util
import os
import struct
import time

_EXTENSION_PATH = os.environ.get("TRTLLM_PERSISTENT_GDN_EXTENSION")
_ACTIVATION_FILE = os.environ.get("TRTLLM_PERSISTENT_GDN_ACTIVATION_FILE")
_RESIDENT_TOKEN_LIMIT = int(os.environ.get("TRTLLM_PERSISTENT_GDN_RESIDENT_TOKENS", "0"))
_TOKEN_LOG_INTERVAL = max(1, int(os.environ.get("TRTLLM_PERSISTENT_GDN_TOKEN_LOG_INTERVAL", "1")))


if _EXTENSION_PATH:
    import torch

    from tensorrt_llm._torch.modules.mamba import gdn_mixer as _gdn_mixer

    _MODULE_NAME = "persistent_gdn_state_r192"
    _SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _EXTENSION_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError(f"cannot load persistent GDN extension: {_EXTENSION_PATH}")
    _EXTENSION = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_EXTENSION)

    _ORIGINAL = _gdn_mixer.fused_sigmoid_gating_delta_rule_update
    _SNAPSHOTS = []
    _CONTROL = None
    _CALLS = 0
    _TOKENS_RESIDENT = 0
    _CAPTURE_ENABLED = _ACTIVATION_FILE is None
    _DISCARDED = False

    def _wait_until_ready(control):
        deadline = time.monotonic() + 30.0
        while True:
            snapshot = bytes(control.cpu().tolist())
            ready = struct.unpack_from("<I", snapshot, 8)[0]
            if ready == 128:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"persistent GDN initialized only {ready}/128 CTAs")
            time.sleep(0.005)

        # Prime the stream-memory command path once before model work resumes.
        _EXTENSION.debug_command(control, 1, 0, 0.0)
        deadline = time.monotonic() + 5.0
        while True:
            snapshot = bytes(control.cpu().tolist())
            epoch, done = struct.unpack_from("<II", snapshot)
            if done >= epoch:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"persistent GDN command prime stalled at {done}/{epoch}")
            time.sleep(0.005)

    def _is_supported_standard_decode(
        q, k, v, a, b, initial_state_source, initial_state_indices, cu_seqlens
    ):
        return (
            q.is_cuda
            and q.dtype == torch.bfloat16
            and q.numel() == 16 * 128
            and k.numel() == 16 * 128
            and v.numel() == 32 * 128
            and a.numel() == 32
            and b.numel() == 32
            and initial_state_indices.numel() == 1
            and cu_seqlens is not None
            and cu_seqlens.numel() == 2
            and initial_state_source.shape[-3:] == (32, 128, 128)
        )

    def _resident_gdn_update(
        A_log,
        a,
        dt_bias,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        initial_state_source,
        initial_state_indices,
        scale=None,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        output=None,
    ):
        global _CALLS, _CONTROL, _TOKENS_RESIDENT, _CAPTURE_ENABLED, _DISCARDED

        # TensorRT-LLM runs internal model/JIT probes before accepting the
        # benchmark request.  The host creates this marker only after the
        # worker announces that initialization is complete, so probe calls do
        # not shift the 24-layer mapping or seed the service with dummy state.
        if not _CAPTURE_ENABLED:
            if _ACTIVATION_FILE and os.path.exists(_ACTIVATION_FILE):
                _CAPTURE_ENABLED = True
                _CALLS = 0
                _SNAPSHOTS.clear()
                print("PERSISTENT_GDN_CAPTURE_ENABLED", flush=True)
            else:
                return _ORIGINAL(
                    A_log=A_log,
                    a=a,
                    dt_bias=dt_bias,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q=q,
                    k=k,
                    v=v,
                    b=b,
                    initial_state_source=initial_state_source,
                    initial_state_indices=initial_state_indices,
                    scale=scale,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=cu_seqlens,
                    output=output,
                )

        del softplus_beta, softplus_threshold, scale

        if not _is_supported_standard_decode(
            q,
            k,
            v,
            a,
            b,
            initial_state_source,
            initial_state_indices,
            cu_seqlens,
        ):
            raise RuntimeError(
                "persistent GDN prototype requires batch=1, T=1, H/HV/K/V=16/32/128/128"
            )
        if not use_qk_l2norm_in_kernel:
            raise RuntimeError("persistent GDN prototype requires Q/K L2 normalization")
        if _DISCARDED:
            raise RuntimeError(
                "persistent GDN received another token after discard; "
                "increase TRTLLM_PERSISTENT_GDN_RESIDENT_TOKENS"
            )

        layer = _CALLS % 24
        _CALLS += 1

        if len(_SNAPSHOTS) < 24:
            result = _ORIGINAL(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=initial_state_source,
                initial_state_indices=initial_state_indices,
                scale=128**-0.5,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                output=output,
            )
            cache_slot = int(initial_state_indices.item())
            _SNAPSHOTS.append(initial_state_source[cache_slot].clone())
            if len(_SNAPSHOTS) == 24:
                print(
                    "PERSISTENT_GDN_CAPTURED: 24 layer states after decode token 1",
                    flush=True,
                )
            return result

        if _CONTROL is None:
            packed_state = torch.stack(_SNAPSHOTS).float().contiguous()
            _CONTROL = _EXTENSION.start(packed_state.view(24, 4096, 128))
            _wait_until_ready(_CONTROL)
            print(
                "PERSISTENT_GDN_READY: 128 CTAs, "
                "state=RF(12 rows/layer)+TMEM(20 rows/layer), "
                "global_state_writeback=disabled",
                flush=True,
            )

        result = _EXTENSION.gdn_decode(
            _CONTROL,
            layer,
            q.reshape(16, 128).contiguous(),
            k.reshape(16, 128).contiguous(),
            v.reshape(32, 128).contiguous(),
            a.reshape(32).contiguous(),
            b.reshape(32).contiguous(),
            A_log.contiguous(),
            dt_bias.contiguous(),
        ).view(1, 1, 32, 128)
        if output is not None:
            output.copy_(result)
            result = output
        if layer == 23:
            _TOKENS_RESIDENT += 1
            if (
                _TOKENS_RESIDENT == 1
                or _TOKENS_RESIDENT % _TOKEN_LOG_INTERVAL == 0
                or (_RESIDENT_TOKEN_LIMIT > 0 and _TOKENS_RESIDENT == _RESIDENT_TOKEN_LIMIT)
            ):
                print(
                    f"PERSISTENT_GDN_TOKEN: completed resident decode token {_TOKENS_RESIDENT + 1}",
                    flush=True,
                )
            if _RESIDENT_TOKEN_LIMIT > 0 and _TOKENS_RESIDENT == _RESIDENT_TOKEN_LIMIT:
                _EXTENSION.discard(_CONTROL)
                _CONTROL = None
                _DISCARDED = True
                print(
                    "PERSISTENT_GDN_DISCARDED: RF/TMEM released, global_state_writeback=0 bytes",
                    flush=True,
                )
        return result

    _gdn_mixer.fused_sigmoid_gating_delta_rule_update = _resident_gdn_update
    print(
        f"PERSISTENT_GDN_PATCHED: extension={_EXTENSION_PATH}",
        flush=True,
    )
