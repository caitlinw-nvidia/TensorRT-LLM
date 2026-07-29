# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoTuned Marlin/V6 selection for dense W4A16 NVFP4 prefill."""

from __future__ import annotations

import os
from typing import ClassVar

import torch
import torch.nn.functional as F

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._utils import get_sm_version

_MARLIN = "marlin"
_V6 = "v6"
_FALLBACK = -1

_V6_KERNEL = None
_TRACE_KEYS: set[tuple] = set()


def _get_v6_kernel():
    """Construct PR #6's V6 kernel lazily.

    Keeping this import lazy lets installations without the external b12x
    package import TensorRT-LLM and continue to use Marlin.
    """
    global _V6_KERNEL
    if _V6_KERNEL is None:
        from b12x.gemm.w4a16._cute_prefill_kernel import DenseGemmW4A16CutePrefillKernel

        _V6_KERNEL = DenseGemmW4A16CutePrefillKernel()
    return _V6_KERNEL


def _v6_is_supported(x: torch.Tensor, w_fp4: torch.Tensor) -> bool:
    if (
        get_sm_version() not in (120, 121)
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or w_fp4.dim() != 2
    ):
        return False
    m, k = (int(v) for v in x.shape)
    n = int(w_fp4.shape[0])
    try:
        from b12x.gemm.w4a16._cute_prefill_kernel import DenseGemmW4A16CutePrefillKernel
    except ImportError:
        return False
    return DenseGemmW4A16CutePrefillKernel.is_supported(m, k, n)


def _run_v6(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale_swizzled_u8: torch.Tensor,
    w_alpha: torch.Tensor,
) -> torch.Tensor:
    return _get_v6_kernel()(x.contiguous(), w_fp4, w_blockscale_swizzled_u8, w_alpha)


def _run_marlin(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    marlin_weight: torch.Tensor,
    marlin_scale: torch.Tensor,
    marlin_global_scale: torch.Tensor,
) -> torch.Tensor:
    size_n = int(w_fp4.shape[0])
    size_k = int(w_fp4.shape[1]) * 2
    size_k_pad = fp4_utils.pad_up(size_k, 64)
    size_n_pad = fp4_utils.pad_up(size_n, 128)

    x_bf16 = x.bfloat16()
    if size_k_pad != size_k:
        x_bf16 = F.pad(x_bf16, (0, size_k_pad - size_k))
    output = torch.ops.trtllm.marlin_nvfp4_gemm(
        x_bf16,
        marlin_weight,
        scale_a=None,
        scale_b=marlin_scale,
        alpha=None,
        weight_global_scale=marlin_global_scale,
        bias=None,
        out_dtype=torch.bfloat16,
        size_n=size_n_pad,
        size_k=size_k_pad,
        output_buffer_kind=0,
    )
    if size_n_pad != size_n:
        output = output[..., :size_n].contiguous()
    return output


class W4A16NVFP4PrefillRunner(TunableRunner):
    """Compare V6 and Marlin after both kernels' one-time preparation."""

    HEURISTIC_MIN_M: ClassVar[int] = 256
    tuning_config = TuningConfig(
        # V6 caches packed weights by data pointer. Cold-L2 cloning would
        # create new pointers and accidentally time repacking, not the GEMM.
        use_cold_l2_cache=False,
        use_cuda_graph=False,
    )

    def unique_id(self):
        return ("w4a16_nvfp4_prefill_marlin_v6_v1",)

    @staticmethod
    def _unpack(inputs: list[torch.Tensor]):
        (
            x,
            w_fp4,
            w_blockscale_swizzled_u8,
            w_alpha,
            marlin_weight,
            marlin_scale,
            marlin_global_scale,
        ) = inputs
        return (
            x,
            w_fp4,
            w_blockscale_swizzled_u8,
            w_alpha,
            marlin_weight,
            marlin_scale,
            marlin_global_scale,
        )

    def heuristic_backend(self, inputs: list[torch.Tensor]) -> str:
        x, w_fp4, *_ = self._unpack(inputs)
        if int(x.shape[0]) >= self.HEURISTIC_MIN_M and _v6_is_supported(x, w_fp4):
            return _V6
        return _MARLIN

    def resolve_backend(self, inputs: list[torch.Tensor], tactic) -> str:
        if tactic == _FALLBACK:
            return self.heuristic_backend(inputs)
        if tactic not in (_MARLIN, _V6):
            raise ValueError(f"unknown W4A16 prefill tactic: {tactic!r}")
        return tactic

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> list:
        del profile, kwargs
        x, w_fp4, *_ = self._unpack(inputs)
        fallback_backend = self.heuristic_backend(inputs)
        tactics = [_FALLBACK]
        if _v6_is_supported(x, w_fp4):
            # The fallback already represents one backend. Add only the other
            # backend so each implementation is measured exactly once.
            tactics.append(_MARLIN if fallback_backend == _V6 else _V6)
        return tactics

    def forward(
        self,
        inputs: list[torch.Tensor],
        *,
        tactic=_FALLBACK,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        (
            x,
            w_fp4,
            w_blockscale_swizzled_u8,
            w_alpha,
            marlin_weight,
            marlin_scale,
            marlin_global_scale,
        ) = self._unpack(inputs)

        if do_preparation:
            # Pack/compile V6 and warm Marlin outside the measured region.
            _run_marlin(x, w_fp4, marlin_weight, marlin_scale, marlin_global_scale)
            if _v6_is_supported(x, w_fp4):
                return _run_v6(x, w_fp4, w_blockscale_swizzled_u8, w_alpha)
            return _run_marlin(x, w_fp4, marlin_weight, marlin_scale, marlin_global_scale)

        backend = self.resolve_backend(inputs, tactic)
        if backend == _V6:
            if not _v6_is_supported(x, w_fp4):
                raise ValueError(
                    f"V6 does not support M={x.shape[0]}, K={x.shape[1]}, N={w_fp4.shape[0]}"
                )
            return _run_v6(x, w_fp4, w_blockscale_swizzled_u8, w_alpha)
        return _run_marlin(x, w_fp4, marlin_weight, marlin_scale, marlin_global_scale)


_RUNNER = W4A16NVFP4PrefillRunner()


def _trace_dispatch(inputs: list[torch.Tensor], backend: str, source: str) -> None:
    if os.environ.get("TRTLLM_W4A16_NVFP4_PREFILL_TRACE_DISPATCH") != "1":
        return
    x, w_fp4, *_ = inputs
    key = (backend, source, tuple(x.shape), tuple(w_fp4.shape))
    if key in _TRACE_KEYS:
        return
    _TRACE_KEYS.add(key)
    print(
        "[w4a16-prefill-dispatch] "
        f"backend={backend} source={source} "
        f"M={x.shape[0]} K={x.shape[1]} N={w_fp4.shape[0]}",
        flush=True,
    )


@torch.library.custom_op(
    "trtllm::w4a16_nvfp4_prefill",
    mutates_args=(),
    device_types="cuda",
)
def _w4a16_nvfp4_prefill_tunable(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale_swizzled_u8: torch.Tensor,
    w_alpha: torch.Tensor,
    marlin_weight: torch.Tensor,
    marlin_scale: torch.Tensor,
    marlin_global_scale: torch.Tensor,
) -> torch.Tensor:
    inputs = [
        x,
        w_fp4,
        w_blockscale_swizzled_u8,
        w_alpha,
        marlin_weight,
        marlin_scale,
        marlin_global_scale,
    ]
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm::w4a16_nvfp4_prefill",
        [_RUNNER],
        _RUNNER.tuning_config,
        inputs,
    )
    backend = runner.resolve_backend(inputs, tactic)
    _trace_dispatch(inputs, backend, "autotuner")
    return runner(inputs, tactic=tactic)


@_w4a16_nvfp4_prefill_tunable.register_fake
def _(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale_swizzled_u8: torch.Tensor,
    w_alpha: torch.Tensor,
    marlin_weight: torch.Tensor,
    marlin_scale: torch.Tensor,
    marlin_global_scale: torch.Tensor,
) -> torch.Tensor:
    del (w_blockscale_swizzled_u8, w_alpha, marlin_weight, marlin_scale, marlin_global_scale)
    return torch.empty((x.shape[0], w_fp4.shape[0]), dtype=torch.bfloat16, device=x.device)


def w4a16_nvfp4_prefill(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale_swizzled_u8: torch.Tensor,
    w_alpha: torch.Tensor,
    marlin_weight: torch.Tensor,
    marlin_scale: torch.Tensor,
    marlin_global_scale: torch.Tensor,
) -> torch.Tensor:
    """Run auto, heuristic, forced-V6, or forced-Marlin prefill dispatch."""
    inputs = [
        x,
        w_fp4,
        w_blockscale_swizzled_u8,
        w_alpha,
        marlin_weight,
        marlin_scale,
        marlin_global_scale,
    ]
    mode = os.environ.get("TRTLLM_W4A16_NVFP4_PREFILL_BACKEND", "auto").lower()
    if mode == "auto":
        return _w4a16_nvfp4_prefill_tunable(*inputs)
    if mode == "heuristic":
        tactic = _FALLBACK
    elif mode in (_MARLIN, _V6):
        tactic = mode
    else:
        raise ValueError(
            "TRTLLM_W4A16_NVFP4_PREFILL_BACKEND must be one of "
            f"auto, heuristic, marlin, or v6; got {mode!r}"
        )
    backend = _RUNNER.resolve_backend(inputs, tactic)
    _trace_dispatch(inputs, backend, mode)
    return _RUNNER(inputs, tactic=tactic)


__all__ = ["W4A16NVFP4PrefillRunner", "w4a16_nvfp4_prefill"]
