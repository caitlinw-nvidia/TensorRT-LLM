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

import math
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.w4a16_nvfp4_m1 import (
    W4A16NVFP4CuteM1Runner,
    w4a16_nvfp4_cute_m1_gemv,
)
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import (
    W4A16NVFP4LinearMethod,
    get_quant_method,
    get_sm_version,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _run_w4a16_cutlass3_reference_case(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    assert k % 32 == 0
    assert n % 32 == 0
    act, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(m, n, k, dtype)

    expected = torch.empty((m, n), device="cuda", dtype=dtype)
    for start in range(0, m, 16):
        stop = min(start + 16, m)
        expected[start:stop, :] = torch.ops.trtllm.w4a16_nvfp4_gemm(
            act[start:stop, :],
            weight,
            weight_scale,
            weight_scale_2,
            dtype,
            bias=None,
        )

    actual = torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm(
        act,
        weight,
        weight_scale,
        weight_scale_2,
        dtype,
        bias=None,
    )
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)


def _make_w4a16_nvfp4_case(m: int, n: int, k: int, dtype: torch.dtype):
    act = torch.randn((m, k), device="cuda", dtype=dtype)
    weight = torch.empty((n, k // 2), device="cuda", dtype=fp4_utils.float4_e2m1x2)
    weight_u8 = torch.randint(
        0,
        256,
        (n, k // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    weight.copy_(weight_u8.view(fp4_utils.float4_e2m1x2))

    scale_cols = fp4_utils.pad_up(k // 16, 4)
    scale_rows = fp4_utils.pad_up(n, 128)
    weight_scale_linear = torch.randint(
        1,
        120,
        (scale_rows, scale_cols),
        device="cuda",
        dtype=torch.uint8,
    )
    weight_scale = torch.ops.trtllm.block_scale_interleave(weight_scale_linear).view(
        fp4_utils.float4_sf_dtype
    )
    weight_scale_2 = torch.ones((1,), device="cuda", dtype=torch.float32)
    return act, weight, weight_scale, weight_scale_2


def _bf16_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Return distance in ordered BF16 encodings for finite values."""

    def ordered_encoding(value: torch.Tensor) -> torch.Tensor:
        bits = value.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        return torch.where(
            (bits & 0x8000) != 0,
            0xFFFF - bits,
            0x8000 + bits,
        )

    return (ordered_encoding(actual) - ordered_encoding(expected)).abs()


def _w4a16_accuracy_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()

    actual_f32 = actual.float()
    expected_f32 = expected.float()
    abs_error = (actual_f32 - expected_f32).abs()
    ulp_error = _bf16_ulp_distance(actual, expected).float()
    expected_l2 = expected_f32.norm().clamp_min(torch.finfo(torch.float32).tiny)
    expected_amax = expected_f32.abs().max().clamp_min(torch.finfo(torch.float32).tiny)

    return {
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
        "relative_l2_error": (abs_error.norm() / expected_l2).item(),
        "normalized_max_error": (abs_error.max() / expected_amax).item(),
        "exact_match_fraction": (actual == expected).float().mean().item(),
        "p99_bf16_ulp": torch.quantile(ulp_error, 0.99).item(),
        "within_two_bf16_ulp": (ulp_error <= 2).float().mean().item(),
    }


def _assert_w4a16_rounding_accuracy(
    actual_name: str,
    actual: torch.Tensor,
    expected_name: str,
    expected: torch.Tensor,
    *,
    relative_l2_limit: float = 1.0e-4,
    normalized_max_limit: float = 2.0e-3,
    p99_bf16_ulp_limit: float = 1.0,
    within_two_bf16_ulp_limit: float = 0.999,
) -> dict[str, float]:
    metrics = _w4a16_accuracy_metrics(actual, expected)
    print(
        f"{actual_name}-vs-{expected_name}: "
        + " ".join(f"{name}={value:.8e}" for name, value in metrics.items())
    )

    # These gates allow the expected FP32 reassociation differences while
    # catching scale-layout errors or distributed numerical drift. The ULP
    # gate checks local BF16 agreement; relative L2 checks aggregate error.
    assert metrics["relative_l2_error"] <= relative_l2_limit
    assert metrics["normalized_max_error"] <= normalized_max_limit
    assert metrics["p99_bf16_ulp"] <= p99_bf16_ulp_limit
    assert metrics["within_two_bf16_ulp"] >= within_two_bf16_ulp_limit
    return metrics


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
@pytest.mark.parametrize(
    "shape",
    [(32, 256, 256), (128, 512, 1024), (256, 1024, 2048)],
)
def test_w4a16_nvfp4_cutlass3_bf16_matches_cuda_core(shape):
    m, n, k = shape
    _run_w4a16_cutlass3_reference_case(m, n, k, torch.bfloat16)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
def test_w4a16_nvfp4_gemm_large_m_chunks_cuda_core():
    m, n, k = 32, 64, 64
    act, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(m, n, k, torch.bfloat16)
    expected = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for start in range(0, m, 16):
        stop = min(start + 16, m)
        expected[start:stop, :] = torch.ops.trtllm.w4a16_nvfp4_gemm(
            act[start:stop, :],
            weight,
            weight_scale,
            weight_scale_2,
            torch.bfloat16,
            bias=None,
        )

    actual = torch.ops.trtllm.w4a16_nvfp4_gemm(
        act,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )

    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
def test_w4a16_nvfp4_cute_m1_matches_cuda_core(capfd):
    torch.manual_seed(11)
    m, k, n = 1, 2688, 4096
    input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
        m, n, k, torch.bfloat16
    )
    expected = torch.ops.trtllm.w4a16_nvfp4_gemm(
        input_tensor,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )

    scale_rows = fp4_utils.pad_up(n, 128)
    scale_cols = fp4_utils.pad_up(k // 16, 4)
    env = {"TRTLLM_W4A16_NVFP4_M1_TRACE_DISPATCH": "1"}
    with patch.dict(os.environ, env):
        actual = w4a16_nvfp4_cute_m1_gemv(
            input_tensor,
            weight,
            weight_scale.view(scale_rows, scale_cols),
            weight_scale_2,
            out=None,
        )

    captured = capfd.readouterr()
    assert "kernel=cute_m1_gemv M=1 K=2688 N=4096" in captured.out
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
def test_w4a16_nvfp4_cute_m1_autotuner_matches_cuda_core():
    """Profile real CuTe tactics, reuse the cache, and check numerical accuracy."""
    torch.manual_seed(41)
    m, k, n = 1, 2688, 4608
    input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
        m, n, k, torch.bfloat16
    )
    expected = torch.ops.trtllm.w4a16_nvfp4_gemm(
        input_tensor,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )
    scale_rows = fp4_utils.pad_up(n, 128)
    scale_cols = fp4_utils.pad_up(k // 16, 4)
    weight_scale_2d = weight_scale.view(scale_rows, scale_cols)
    inputs = [input_tensor, weight, weight_scale_2d, weight_scale_2]

    tuner = AutoTuner.get()
    old_warmup = tuner.warmup
    old_repeat = tuner.repeat
    old_stream_delay = tuner.stream_delay_micro_secs
    tuner.clear_cache()
    try:
        tuner.warmup = 1
        tuner.repeat = 3
        tuner.stream_delay_micro_secs = 10
        with autotune():
            tuned = w4a16_nvfp4_cute_m1_gemv(
                input_tensor,
                weight,
                weight_scale_2d,
                weight_scale_2,
                out=None,
            )

        runner = W4A16NVFP4CuteM1Runner()
        _, selected_tactic = tuner.choose_one(
            "trtllm::w4a16_nvfp4_cute_m1_gemv",
            [runner],
            runner.tuning_config,
            inputs,
        )
        cached = w4a16_nvfp4_cute_m1_gemv(
            input_tensor,
            weight,
            weight_scale_2d,
            weight_scale_2,
            out=None,
        )
        print(f"W4A16 M=1 AutoTuner selected tactic={selected_tactic}")

        assert selected_tactic in runner.get_valid_tactics(inputs, None)
        assert (
            tuner.stats.tuned_op_profiled_configs[
                "trtllm::w4a16_nvfp4_cute_m1_gemv"
            ]
            >= 1
        )
        torch.testing.assert_close(tuned, expected, atol=0.08, rtol=0.08)
        torch.testing.assert_close(cached, tuned, atol=0.0, rtol=0.0)
    finally:
        tuner.warmup = old_warmup
        tuner.repeat = old_repeat
        tuner.stream_delay_micro_secs = old_stream_delay
        tuner.clear_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
def test_w4a16_nvfp4_cute_m1_lm_head_tactic_matches_cuda_core():
    """Cover the padded Nano3.5 vocabulary projection used during decode."""
    torch.manual_seed(43)
    m, k, n = 1, 2688, 131072
    input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
        m, n, k, torch.bfloat16
    )
    expected = torch.ops.trtllm.w4a16_nvfp4_gemm(
        input_tensor,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )

    scale_rows = fp4_utils.pad_up(n, 128)
    scale_cols = fp4_utils.pad_up(k // 16, 4)
    inputs = [
        input_tensor,
        weight,
        weight_scale.view(scale_rows, scale_cols),
        weight_scale_2,
    ]
    runner = W4A16NVFP4CuteM1Runner()
    assert runner.get_valid_tactics(inputs, None) == [-1, 16, 28, 32]

    fallback = runner(inputs, tactic=-1)
    tuned = runner(inputs, tactic=28)

    _assert_w4a16_rounding_accuracy(
        "lm_head_fallback", fallback, "cuda_core", expected
    )
    _assert_w4a16_rounding_accuracy("lm_head_tuned", tuned, "cuda_core", expected)
    torch.testing.assert_close(tuned, fallback, atol=0.08, rtol=0.08)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
@pytest.mark.parametrize(
    ("actual_name", "expected_name"),
    [
        pytest.param("cute_m1", "cuda_core", id="cute-vs-cuda-core"),
        pytest.param("pamela_cutlass", "cuda_core", id="pamela-cutlass-vs-cuda-core"),
        pytest.param("cute_m1", "pamela_cutlass", id="cute-vs-pamela-cutlass"),
    ],
)
def test_w4a16_nvfp4_m1_kernel_accuracy(actual_name, expected_name):
    torch.manual_seed(13)
    m, k, n = 1, 2688, 4096
    input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
        m, n, k, torch.bfloat16
    )
    scale_rows = fp4_utils.pad_up(n, 128)
    scale_cols = fp4_utils.pad_up(k // 16, 4)
    # Pamela's CUTLASS dispatcher has no M=1 specialization. GEMM rows are
    # independent, so run a supported M=32 problem and compare its first row.
    cutlass_input = torch.cat(
        (
            input_tensor,
            torch.zeros((31, k), device="cuda", dtype=torch.bfloat16),
        ),
        dim=0,
    )

    outputs = {
        "cuda_core": torch.ops.trtllm.w4a16_nvfp4_gemm(
            input_tensor,
            weight,
            weight_scale,
            weight_scale_2,
            torch.bfloat16,
            bias=None,
        ),
        "pamela_cutlass": torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm(
            cutlass_input,
            weight,
            weight_scale,
            weight_scale_2,
            torch.bfloat16,
            bias=None,
        )[:1],
        "cute_m1": w4a16_nvfp4_cute_m1_gemv(
            input_tensor,
            weight,
            weight_scale.view(scale_rows, scale_cols),
            weight_scale_2,
            out=None,
        ),
    }
    actual = outputs[actual_name]
    expected = outputs[expected_name]
    abs_error = (actual.float() - expected.float()).abs()
    atol = rtol = 0.08
    relative_l2_error = abs_error.norm() / expected.float().norm()
    within_tolerance = (abs_error <= atol + rtol * expected.float().abs()).float().mean()
    print(
        f"{actual_name}-vs-{expected_name}: "
        f"max_abs_error={abs_error.max().item():.8f} "
        f"mean_abs_error={abs_error.mean().item():.8f} "
        f"relative_l2_error={relative_l2_error.item():.8e} "
        f"within_tolerance={within_tolerance.item():.4%}"
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
@pytest.mark.parametrize(
    ("projection_name", "k", "n", "global_scale", "seed"),
    (
        pytest.param("mamba_in_proj", 2688, 10304, 0.0137, 19, id="large-n"),
        pytest.param("mamba_out_proj", 4096, 2688, 0.317, 23, id="large-k"),
        pytest.param("attn_qkv", 2688, 4608, 1.37, 29, id="balanced"),
    ),
)
def test_w4a16_nvfp4_cute_m1_rounding_accuracy(
    projection_name, k, n, global_scale, seed
):
    """Bound error introduced by moving block/global scaling after partial sums."""
    torch.manual_seed(seed)
    input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
        1, n, k, torch.bfloat16
    )
    weight_scale_2.fill_(global_scale)
    scale_rows = fp4_utils.pad_up(n, 128)
    scale_cols = fp4_utils.pad_up(k // 16, 4)

    # The CUTLASS dispatcher starts at M=32. Matrix rows are independent, so
    # padding with zero rows gives the same first-row problem as the M=1 paths.
    cutlass_input = torch.zeros((32, k), device="cuda", dtype=torch.bfloat16)
    cutlass_input[:1].copy_(input_tensor)

    cuda_core = torch.ops.trtllm.w4a16_nvfp4_gemm(
        input_tensor,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )
    cutlass = torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm(
        cutlass_input,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )[:1]
    cute_m1 = w4a16_nvfp4_cute_m1_gemv(
        input_tensor,
        weight,
        weight_scale.view(scale_rows, scale_cols),
        weight_scale_2,
        out=None,
    )

    prefix = f"{projection_name}[K={k},N={n},global_scale={global_scale}]"
    cute_vs_cuda_core = _assert_w4a16_rounding_accuracy(
        f"{prefix}/cute_m1", cute_m1, "cuda_core", cuda_core
    )
    cutlass_vs_cuda_core = _assert_w4a16_rounding_accuracy(
        f"{prefix}/cutlass",
        cutlass,
        "cuda_core",
        cuda_core,
        relative_l2_limit=5.0e-3,
        normalized_max_limit=1.0e-2,
        p99_bf16_ulp_limit=32.0,
        within_two_bf16_ulp_limit=0.90,
    )
    _assert_w4a16_rounding_accuracy(
        f"{prefix}/cute_m1",
        cute_m1,
        "cutlass",
        cutlass,
        relative_l2_limit=5.0e-3,
        normalized_max_limit=1.0e-2,
        p99_bf16_ulp_limit=32.0,
        within_two_bf16_ulp_limit=0.90,
    )
    assert (
        cute_vs_cuda_core["relative_l2_error"]
        <= cutlass_vs_cuda_core["relative_l2_error"]
    )


def _benchmark_cuda_event_ms(fn, *, warmup=25, iterations=100, repeats=7):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / iterations)

    return sorted(samples_ms)[len(samples_ms) // 2], min(samples_ms), samples_ms


_W4A16_M1_ABLATION_ENV_KEYS = (
    "TRTLLM_W4A16_NVFP4_M1_ABLATION",
    "TRTLLM_W4A16_NVFP4_M1_K_GROUPS",
    "TRTLLM_W4A16_NVFP4_M1_OUTPUTS_PER_WARP",
    "TRTLLM_W4A16_NVFP4_M1_SCALE_PLACEMENT",
    "TRTLLM_W4A16_NVFP4_M1_SPLITK_COLS",
    "TRTLLM_W4A16_NVFP4_M1_SPLITK_MAX_N",
    "TRTLLM_W4A16_NVFP4_M1_SPLITK_WARPS",
    "TRTLLM_W4A16_NVFP4_M1_THREADS",
    "TRTLLM_W4A16_NVFP4_M1_TILE_N",
    "TRTLLM_W4A16_NVFP4_M1_TRACE_DISPATCH",
    "TRTLLM_W4A16_NVFP4_M1_WARPS",
)


def _set_w4a16_m1_ablation_env(config: dict[str, str]) -> None:
    for key in _W4A16_M1_ABLATION_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(config)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
@pytest.mark.parametrize(
    ("projection_name", "k", "n"),
    (
        ("mamba_in_proj", 2688, 10304),
        ("mamba_out_proj", 4096, 2688),
        ("shared_fc1", 2688, 3712),
        ("shared_fc2", 3712, 2688),
        ("attn_qkv", 2688, 4608),
    ),
)
def test_w4a16_nvfp4_m1_kernel_timing(projection_name, k, n):
    torch.manual_seed(17)
    m = 1
    input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
        m, n, k, torch.bfloat16
    )
    scale_rows = fp4_utils.pad_up(n, 128)
    scale_cols = fp4_utils.pad_up(k // 16, 4)
    weight_scale_2d = weight_scale.view(scale_rows, scale_cols)
    cutlass_input = torch.cat(
        (
            input_tensor,
            torch.zeros((31, k), device="cuda", dtype=torch.bfloat16),
        ),
        dim=0,
    )

    def run_cute_m1():
        return w4a16_nvfp4_cute_m1_gemv(
            input_tensor,
            weight,
            weight_scale_2d,
            weight_scale_2,
            out=None,
        )

    def run_cuda_core_m1():
        return torch.ops.trtllm.w4a16_nvfp4_gemm(
            input_tensor,
            weight,
            weight_scale,
            weight_scale_2,
            torch.bfloat16,
            bias=None,
        )

    def run_pamela_cutlass_m32():
        return torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm(
            cutlass_input,
            weight,
            weight_scale,
            weight_scale_2,
            torch.bfloat16,
            bias=None,
        )

    cuda_core_output = run_cuda_core_m1()
    torch.testing.assert_close(run_cute_m1(), cuda_core_output, atol=0.08, rtol=0.08)
    torch.testing.assert_close(
        run_pamela_cutlass_m32()[:1], cuda_core_output, atol=0.08, rtol=0.08
    )

    timings = {}
    for name, fn in (
        ("cute_m1", run_cute_m1),
        ("cuda_core_m1", run_cuda_core_m1),
        ("pamela_cutlass_m32_padded", run_pamela_cutlass_m32),
    ):
        median_ms, min_ms, samples_ms = _benchmark_cuda_event_ms(fn)
        timings[name] = median_ms
        print(
            f"{projection_name} M={m} K={k} N={n} {name}: "
            f"median_ms={median_ms:.6f} min_ms={min_ms:.6f} "
            f"samples_ms={[round(sample, 6) for sample in samples_ms]}"
        )

    print(
        f"{projection_name} M={m} K={k} N={n} cute_m1_speedup_over: "
        f"cuda_core_m1={timings['cuda_core_m1'] / timings['cute_m1']:.4f}x "
        "pamela_cutlass_m32_padded="
        f"{timings['pamela_cutlass_m32_padded'] / timings['cute_m1']:.4f}x"
    )
    assert all(latency_ms > 0.0 for latency_ms in timings.values())


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
@pytest.mark.skip(
    reason="Historical multi-topology ablation; production now has one output-per-warp kernel."
)
def test_w4a16_nvfp4_cute_m1_ablation_timing():
    """Measure one-at-a-time differences from the tuned M=1 CuTe kernel.

    `cuda_core_like` is the faithful combined C++ organization: kStepK=32,
    tileN=2, 128 threads, activation reuse across both outputs, per-weight
    scaling, and shared-memory reduction of four warp partials.
    """
    shapes = (
        ("mamba_in_proj", 2688, 10304, 0.0137, 19),
        ("mamba_out_proj", 4096, 2688, 0.317, 23),
        ("shared_fc1", 2688, 3712, 0.731, 31),
        ("shared_fc2", 3712, 2688, 0.093, 37),
        ("attn_qkv", 2688, 4608, 1.37, 29),
    )
    variants = (
        ("optimized", {}),
        (
            "per_weight_scaling",
            {"TRTLLM_W4A16_NVFP4_M1_SCALE_PLACEMENT": "per_weight"},
        ),
        ("k32_fragment", {"TRTLLM_W4A16_NVFP4_M1_K_GROUPS": "2"}),
        (
            "one_output_w32",
            {
                "TRTLLM_W4A16_NVFP4_M1_K_GROUPS": "1",
                "TRTLLM_W4A16_NVFP4_M1_OUTPUTS_PER_WARP": "1",
                "TRTLLM_W4A16_NVFP4_M1_WARPS": "32",
            },
        ),
        (
            "two_outputs_w16",
            {
                "TRTLLM_W4A16_NVFP4_M1_K_GROUPS": "1",
                "TRTLLM_W4A16_NVFP4_M1_OUTPUTS_PER_WARP": "2",
                "TRTLLM_W4A16_NVFP4_M1_WARPS": "16",
            },
        ),
        (
            "two_outputs_w1",
            {
                "TRTLLM_W4A16_NVFP4_M1_K_GROUPS": "1",
                "TRTLLM_W4A16_NVFP4_M1_OUTPUTS_PER_WARP": "2",
                "TRTLLM_W4A16_NVFP4_M1_WARPS": "1",
            },
        ),
        (
            "four_outputs_w8",
            {
                "TRTLLM_W4A16_NVFP4_M1_K_GROUPS": "1",
                "TRTLLM_W4A16_NVFP4_M1_OUTPUTS_PER_WARP": "4",
                "TRTLLM_W4A16_NVFP4_M1_WARPS": "8",
            },
        ),
        (
            "one_warp_one_column",
            {
                "TRTLLM_W4A16_NVFP4_M1_K_GROUPS": "1",
                "TRTLLM_W4A16_NVFP4_M1_OUTPUTS_PER_WARP": "1",
                "TRTLLM_W4A16_NVFP4_M1_WARPS": "1",
            },
        ),
        (
            "four_warp_split_k16",
            {
                "TRTLLM_W4A16_NVFP4_M1_K_GROUPS": "1",
                "TRTLLM_W4A16_NVFP4_M1_OUTPUTS_PER_WARP": "1",
                "TRTLLM_W4A16_NVFP4_M1_SPLITK_COLS": "1",
                "TRTLLM_W4A16_NVFP4_M1_SPLITK_MAX_N": str(2**31 - 1),
                "TRTLLM_W4A16_NVFP4_M1_SPLITK_WARPS": "4",
            },
        ),
        (
            "cuda_core_like",
            {"TRTLLM_W4A16_NVFP4_M1_ABLATION": "cuda_core_like"},
        ),
    )
    paired_effects = (
        ("scale_placement", "per_weight_scaling", "optimized"),
        ("k_fragment_32", "k32_fragment", "optimized"),
        ("two_outputs_per_warp", "two_outputs_w16", "one_output_w32"),
        ("output_tile_n2", "two_outputs_w1", "two_outputs_w16"),
        ("four_outputs_per_warp", "four_outputs_w8", "one_output_w32"),
        (
            "four_warp_shared_reduction",
            "four_warp_split_k16",
            "one_warp_one_column",
        ),
        ("combined_cuda_core_like", "cuda_core_like", "optimized"),
    )

    saved_env = {key: os.environ.get(key) for key in _W4A16_M1_ABLATION_ENV_KEYS}
    effect_ratios = {effect_name: [] for effect_name, _, _ in paired_effects}
    try:
        for projection_name, k, n, global_scale, seed in shapes:
            torch.manual_seed(seed)
            input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
                1, n, k, torch.bfloat16
            )
            weight_scale_2.fill_(global_scale)
            scale_rows = fp4_utils.pad_up(n, 128)
            scale_cols = fp4_utils.pad_up(k // 16, 4)
            weight_scale_2d = weight_scale.view(scale_rows, scale_cols)
            shared_out = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)

            cuda_core = torch.ops.trtllm.w4a16_nvfp4_gemm(
                input_tensor,
                weight,
                weight_scale,
                weight_scale_2,
                torch.bfloat16,
                bias=None,
            )

            def run_cute():
                return w4a16_nvfp4_cute_m1_gemv(
                    input_tensor,
                    weight,
                    weight_scale_2d,
                    weight_scale_2,
                    out=shared_out,
                )

            # Compile and check every variant before any timing. This excludes
            # CuTe compilation and first-launch costs from the event samples.
            accuracy = {}
            for variant_name, env in variants:
                _set_w4a16_m1_ablation_env(env)
                shared_out.fill_(float("nan"))
                run_cute()
                torch.cuda.synchronize()
                actual = shared_out.clone()
                metrics = _assert_w4a16_rounding_accuracy(
                    f"{projection_name}/{variant_name}",
                    actual,
                    "cuda_core",
                    cuda_core,
                    relative_l2_limit=1.0e-3,
                    normalized_max_limit=5.0e-3,
                    p99_bf16_ulp_limit=4.0,
                    within_two_bf16_ulp_limit=0.99,
                )
                accuracy[variant_name] = metrics

            # Warm every already-compiled variant using the same output buffer.
            for variant_name, env in variants:
                _set_w4a16_m1_ablation_env(env)
                for _ in range(50):
                    run_cute()
            torch.cuda.synchronize()

            samples = {variant_name: [] for variant_name, _ in variants}
            iterations = 500
            repeats = 11
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for repeat in range(repeats):
                offset = repeat % len(variants)
                ordered_variants = variants[offset:] + variants[:offset]
                for variant_name, env in ordered_variants:
                    _set_w4a16_m1_ablation_env(env)
                    start.record()
                    for _ in range(iterations):
                        run_cute()
                    end.record()
                    end.synchronize()
                    samples[variant_name].append(start.elapsed_time(end) / iterations)

            def run_cuda_core():
                return torch.ops.trtllm.w4a16_nvfp4_gemm(
                    input_tensor,
                    weight,
                    weight_scale,
                    weight_scale_2,
                    torch.bfloat16,
                    bias=None,
                )

            cuda_median, cuda_min, cuda_samples = _benchmark_cuda_event_ms(
                run_cuda_core, warmup=50, iterations=500, repeats=11
            )
            timings = {}
            for variant_name, _ in variants:
                ordered = sorted(samples[variant_name])
                median_ms = ordered[len(ordered) // 2]
                q1_ms = ordered[len(ordered) // 4]
                q3_ms = ordered[(3 * len(ordered)) // 4]
                timings[variant_name] = median_ms
                print(
                    "ABLATION_RESULT "
                    f"projection={projection_name} K={k} N={n} "
                    f"variant={variant_name} median_ms={median_ms:.6f} "
                    f"min_ms={min(ordered):.6f} iqr_ms={q3_ms - q1_ms:.6f} "
                    f"slowdown_vs_optimized={median_ms / timings['optimized']:.6f} "
                    f"speedup_vs_cpp_cuda_core={cuda_median / median_ms:.6f} "
                    f"relative_l2={accuracy[variant_name]['relative_l2_error']:.8e} "
                    f"p99_bf16_ulp={accuracy[variant_name]['p99_bf16_ulp']:.3f} "
                    f"exact_fraction={accuracy[variant_name]['exact_match_fraction']:.8f}"
                )
            print(
                "ABLATION_RESULT "
                f"projection={projection_name} K={k} N={n} variant=cpp_cuda_core "
                f"median_ms={cuda_median:.6f} min_ms={cuda_min:.6f} "
                f"samples_ms={[round(value, 6) for value in cuda_samples]}"
            )

            for effect_name, changed_name, control_name in paired_effects:
                ratio = timings[changed_name] / timings[control_name]
                effect_ratios[effect_name].append(ratio)
                print(
                    "ABLATION_EFFECT "
                    f"projection={projection_name} effect={effect_name} "
                    f"changed={changed_name} control={control_name} "
                    f"latency_ratio={ratio:.6f} slowdown_percent={(ratio - 1.0) * 100.0:.3f}"
                )

        ranked = []
        for effect_name, ratios in effect_ratios.items():
            geomean_ratio = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            ranked.append((geomean_ratio, effect_name))
        ranked.sort(reverse=True)
        for rank, (geomean_ratio, effect_name) in enumerate(ranked, start=1):
            print(
                "ABLATION_RANK "
                f"rank={rank} effect={effect_name} geomean_latency_ratio={geomean_ratio:.6f} "
                f"geomean_slowdown_percent={(geomean_ratio - 1.0) * 100.0:.3f}"
            )
    finally:
        for key in _W4A16_M1_ABLATION_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in saved_env.items():
            if value is not None:
                os.environ[key] = value


def test_get_quant_method_returns_w4a16_nvfp4_linear_method():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    method = get_quant_method(quant_config)

    assert isinstance(method, W4A16NVFP4LinearMethod)


def test_w4a16_nvfp4_linear_uses_high_precision_activation_without_fp4_quantize():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 4), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((5, 2), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.25], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=3,
        pre_quant_scale=None,
    )
    captured = {}

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured["out_dtype"] = out_dtype
        captured["bias"] = bias
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    def fail_fp4_quantize(*args, **kwargs):
        raise AssertionError("W4A16 NVFP4 must not quantize activations")

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True):
        with patch("torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True):
            output = method.apply(module, input_tensor, bias)

    assert captured["input"] is input_tensor
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert captured["bias"] is None
    expected = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
def test_w4a16_nvfp4_linear_uses_local_cute_m1_gemv(capfd):
    torch.manual_seed(12)
    m, k, n = 1, 64, 64
    input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
        m, n, k, torch.bfloat16
    )
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        dtype=torch.bfloat16,
        out_features=n,
        pre_quant_scale=None,
        scaling_vector_size=16,
    )
    expected = (
        torch.ops.trtllm.w4a16_nvfp4_gemm(
            input_tensor,
            weight,
            weight_scale,
            weight_scale_2,
            torch.bfloat16,
            bias=None,
        )
        + bias
    )

    method = W4A16NVFP4LinearMethod()
    env = {"TRTLLM_W4A16_NVFP4_M1_TRACE_DISPATCH": "1"}
    with patch.dict(os.environ, env):
        output = method.apply(module, input_tensor, bias)

    captured = capfd.readouterr()
    assert "kernel=cute_m1_gemv M=1 K=64 N=64" in captured.out
    torch.testing.assert_close(output, expected, atol=0.08, rtol=0.08)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
def test_w4a16_nvfp4_linear_can_disable_local_cute_m1_gemv():
    torch.manual_seed(13)
    m, k, n = 1, 64, 64
    input_tensor, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(
        m, n, k, torch.bfloat16
    )
    module = SimpleNamespace(
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        dtype=torch.bfloat16,
        out_features=n,
        pre_quant_scale=None,
        scaling_vector_size=16,
    )
    expected = torch.ops.trtllm.w4a16_nvfp4_gemm(
        input_tensor,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )

    method = W4A16NVFP4LinearMethod()
    with patch.dict(os.environ, {method.DISABLE_CUTE_M1_ENV: "1"}):
        with patch(
            "tensorrt_llm._torch.modules.linear._w4a16_nvfp4_cute_m1_gemv",
            side_effect=AssertionError("CuTe M=1 must be disabled"),
        ):
            output = method.apply(module, input_tensor, bias=None)

    torch.testing.assert_close(output, expected)


def test_w4a16_nvfp4_linear_restores_high_rank_input_shape():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 3, 4), dtype=torch.float16)
    module = SimpleNamespace(
        weight=torch.empty((7, 2), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.float16,
        out_features=5,
        pre_quant_scale=None,
    )

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        assert input_arg.shape == (6, 4)
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True):
        output = method.apply(module, input_tensor, bias=None)

    assert output.shape == (2, 3, 5)


def test_w4a16_nvfp4_linear_uses_chunked_w4a16_op_for_large_m():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 16), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((5, 8), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=3,
        pre_quant_scale=None,
    )
    captured = {}

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured["out_dtype"] = out_dtype
        captured["bias"] = bias
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    def fail_fp4_quantize(*args, **kwargs):
        raise AssertionError("large-M W4A16 path must not quantize activations")

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True):
        with patch("torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True):
            output = method.apply(module, input_tensor, bias=None)

    assert captured["input"] is input_tensor
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert output.shape == (17, 3)


def test_w4a16_nvfp4_linear_uses_cutlass3_op_for_large_bf16_m_when_enabled():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((32, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=3,
        pre_quant_scale=None,
    )
    captured = {}

    def fake_w4a16_nvfp4_cutlass_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured["out_dtype"] = out_dtype
        captured["bias"] = bias
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    def fail_w4a16_gemm(*args, **kwargs):
        raise AssertionError("CUTLASS3 W4A16 prefill must not call the default W4A16 op")

    def fail_fp4_quantize(*args, **kwargs):
        raise AssertionError("CUTLASS3 W4A16 prefill must not quantize activations")

    with patch.dict(os.environ, {"TRTLLM_W4A16_NVFP4_CUTLASS3": "1"}):
        with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120):
            with patch(
                "torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fail_w4a16_gemm, create=True
            ):
                with patch(
                    "torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm",
                    side_effect=fake_w4a16_nvfp4_cutlass_gemm,
                    create=True,
                ):
                    with patch(
                        "torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True
                    ):
                        output = method.apply(module, input_tensor, bias=None)

    assert captured["input"] is input_tensor
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert captured["bias"] is None
    assert output.shape == (17, 3)


def test_w4a16_nvfp4_linear_cutlass3_restores_high_rank_input_shape():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 9, 32), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((32, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=3,
        pre_quant_scale=None,
    )
    captured = {}

    def fake_w4a16_nvfp4_cutlass_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input_shape"] = input_arg.shape
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    def fail_w4a16_gemm(*args, **kwargs):
        raise AssertionError("large-M CUTLASS3 path must not call the default W4A16 op")

    with patch.dict(os.environ, {"TRTLLM_W4A16_NVFP4_CUTLASS3": "1"}):
        with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120):
            with patch(
                "torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fail_w4a16_gemm, create=True
            ):
                with patch(
                    "torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm",
                    side_effect=fake_w4a16_nvfp4_cutlass_gemm,
                    create=True,
                ):
                    output = method.apply(module, input_tensor, bias=bias)

    assert captured["input_shape"] == (18, 32)
    assert output.shape == (2, 9, 3)
    expected = torch.tensor([2.0, 3.0, 4.0], dtype=torch.bfloat16).expand(2, 9, 3)
    torch.testing.assert_close(output, expected)


def test_w4a16_nvfp4_cutlass3_prefill_requires_supported_shape():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((6, 16), dtype=torch.uint8),
        dtype=torch.bfloat16,
    )

    with patch.dict(os.environ, {"TRTLLM_W4A16_NVFP4_CUTLASS3": "1"}):
        with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120):
            assert not method._can_use_cutlass3_w4a16_prefill(module, input_tensor, m=17)


def test_w4a16_nvfp4_linear_cutlass3_unsupported_shape_uses_default_w4a16_op():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((6, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=3,
        pre_quant_scale=None,
    )
    captured = {}

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured["out_dtype"] = out_dtype
        captured["bias"] = bias
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    def fail_cutlass_gemm(*args, **kwargs):
        raise AssertionError("unsupported CUTLASS3 shape must use the default W4A16 op")

    def fail_fp4_quantize(*args, **kwargs):
        raise AssertionError("unsupported CUTLASS3 W4A16 path must not quantize activations")

    with patch.dict(os.environ, {"TRTLLM_W4A16_NVFP4_CUTLASS3": "1"}):
        with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120):
            with patch(
                "torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True
            ):
                with patch(
                    "torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm",
                    side_effect=fail_cutlass_gemm,
                    create=True,
                ):
                    with patch(
                        "torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True
                    ):
                        output = method.apply(module, input_tensor, bias=None)

    assert captured["input"] is input_tensor
    assert captured["bias"] is None
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert output.shape == (17, 3)


def test_w4a16_nvfp4_post_load_preserves_checkpoint_weight_global_scale():
    method = W4A16NVFP4LinearMethod()
    module = SimpleNamespace(
        input_scale=None,
        inv_input_scale=None,
        alpha=None,
        weight_scale_2=torch.empty([1], dtype=torch.float32),
        tmp_nvfp4_input_scales_list=[torch.tensor(1.0, dtype=torch.float32)],
        tmp_nvfp4_weight_scale_2_list=[torch.tensor(0.25, dtype=torch.float32)],
    )

    method.process_weights_after_loading_vanilla(module)

    assert module.input_scale is None
    assert module.inv_input_scale is None
    assert module.alpha is None
    torch.testing.assert_close(module.weight_scale_2, torch.tensor([0.25], dtype=torch.float32))
    assert not hasattr(module, "tmp_nvfp4_input_scales_list")
    assert not hasattr(module, "tmp_nvfp4_weight_scale_2_list")


def test_lm_head_uses_w4a16_nvfp4_quant_method_for_packed_lm_head():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    lm_head = LMHead(
        num_embeddings=3, embedding_dim=16, dtype=torch.float16, quant_config=quant_config
    )

    assert isinstance(lm_head.quant_method, W4A16NVFP4LinearMethod)
    assert lm_head.weight.dtype == torch.uint8
    assert lm_head.weight.shape == (3, 8)
    assert lm_head.weight_scale.shape == (128 * 4,)
    assert lm_head.weight_scale_2.shape == (1,)


def test_lm_head_w4a16_nvfp4_forward_dispatches_to_w4a16_op():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    lm_head = LMHead(
        num_embeddings=3, embedding_dim=16, dtype=torch.float16, quant_config=quant_config
    )
    input_tensor = torch.ones((2, 16), dtype=torch.float16)
    captured = {}

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured["out_dtype"] = out_dtype
        captured["bias"] = bias
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True):
        output = lm_head(input_tensor)

    assert captured["input"] is input_tensor
    assert captured["weight"] is lm_head.weight
    assert captured["weight_scale"] is lm_head.weight_scale
    assert captured["weight_scale_2"] is lm_head.weight_scale_2
    assert captured["out_dtype"] is torch.float16
    assert captured["bias"] is None
    assert output.shape == (2, 3)
