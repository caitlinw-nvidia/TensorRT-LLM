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

"""Microbenchmark FlashInfer GDN against the CUDA C++ persistent rewrite.

The traffic-matched resident mode reloads one layer from a BF16 global-memory
state pool before every update and writes the updated BF16 state back after
every update. This removes RF/TMEM persistence from the comparison while
retaining the rewrite's CTA decomposition, arithmetic, and command mechanism.
"""

import json
import os
import statistics
import struct
import time
from pathlib import Path

import torch
from test_persistent_state import build

LAYERS = 24
HV = 32
H = 16
V = 128
K = 128


def wait_until_ready(control: torch.Tensor) -> None:
    deadline = time.monotonic() + 30
    while True:
        snapshot = bytes(control.cpu().tolist())
        ready = struct.unpack_from("<I", snapshot, 8)[0]
        if ready == 128:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only {ready}/128 resident CTAs initialized")
        time.sleep(0.01)


def event_time_us(function, iterations: int) -> float:
    # Device-wide synchronization would wait forever for the intentionally
    # long-lived service kernel. Synchronize only the command stream.
    torch.cuda.current_stream().synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000.0 / iterations


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_us": ordered[0],
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
        "max_us": ordered[-1],
    }


def main() -> None:
    torch.cuda.set_device(0)
    torch.manual_seed(20260729)

    iterations = int(os.environ.get("GDN_BENCH_ITERATIONS", "200"))
    trials = int(os.environ.get("GDN_BENCH_TRIALS", "20"))
    warmup = int(os.environ.get("GDN_BENCH_WARMUP", "20"))
    assert iterations > 0 and trials > 0 and warmup > 0

    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule

    print("building/loading CUDA C++ extension", flush=True)
    extension = build()
    print("CUDA C++ extension ready", flush=True)
    device = torch.device("cuda")

    initial_state = (
        torch.randn(1, HV, V, K, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    service_initial = torch.zeros(LAYERS, HV, V, K, device=device, dtype=torch.float32)
    service_initial[0].copy_(initial_state[0].float())

    q = torch.randn(1, 1, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 1, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 1, HV, V, device=device, dtype=torch.bfloat16)
    a = torch.randn(1, 1, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(1, 1, HV, device=device, dtype=torch.bfloat16)
    a_log = torch.empty(HV, device=device, dtype=torch.float32).uniform_(-4.0, -2.0)
    dt_bias = torch.empty(HV, device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
    indices = torch.zeros(1, device=device, dtype=torch.int32)

    baseline_output = torch.empty(1, 1, HV, V, device=device, dtype=torch.bfloat16)
    resident_output = torch.empty(HV, V, device=device, dtype=torch.bfloat16)
    baseline_state = initial_state.clone()
    resident_state = initial_state[0].clone()

    def baseline() -> None:
        gated_delta_rule(
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=baseline_state,
            initial_state_indices=indices,
            output=baseline_output,
            use_qk_l2norm_in_kernel=True,
            scale=K**-0.5,
        )

    def start_service():
        control = extension.start(service_initial.view(LAYERS, HV * V, K))
        wait_until_ready(control)
        return control

    # JIT/allocator warmup and one-call correctness check. The baseline runs
    # without the service alive because the persistent service intentionally
    # occupies 128 of the B300's 148 SMs.
    baseline_state.copy_(initial_state)
    print("warming FlashInfer baseline", flush=True)
    for _ in range(warmup):
        baseline()
    torch.cuda.current_stream().synchronize()
    print("FlashInfer baseline warmup complete", flush=True)

    baseline_state.copy_(initial_state)
    baseline()
    torch.cuda.current_stream().synchronize()
    print("FlashInfer one-call reference complete", flush=True)
    expected_output = baseline_output.clone()
    expected_state = baseline_state.clone()

    print("starting persistent service for correctness", flush=True)
    control = start_service()
    print("persistent service ready for correctness", flush=True)

    def resident_compute_only() -> None:
        extension.gdn_decode_into(
            control,
            0,
            q.view(H, K),
            k.view(H, K),
            v.view(HV, V),
            a.view(HV),
            b.view(HV),
            a_log,
            dt_bias,
            resident_output,
        )

    def resident_write() -> None:
        extension.gdn_decode_gmem(
            control,
            0,
            q.view(H, K),
            k.view(H, K),
            v.view(HV, V),
            a.view(HV),
            b.view(HV),
            a_log,
            dt_bias,
            resident_state,
            False,
            resident_output,
        )

    def resident_round_trip() -> None:
        extension.gdn_decode_gmem(
            control,
            0,
            q.view(H, K),
            k.view(H, K),
            v.view(HV, V),
            a.view(HV),
            b.view(HV),
            a_log,
            dt_bias,
            resident_state,
            True,
            resident_output,
        )

    print("running resident compute-only probe", flush=True)
    resident_compute_only()
    torch.cuda.current_stream().synchronize()
    print("resident compute-only probe complete", flush=True)

    print("running resident write-only probe", flush=True)
    resident_write()
    torch.cuda.current_stream().synchronize()
    print("resident write-only probe complete", flush=True)

    resident_state.copy_(initial_state[0])
    print("running one traffic-matched resident correctness call", flush=True)
    resident_round_trip()
    torch.cuda.current_stream().synchronize()
    print("traffic-matched resident correctness call complete", flush=True)
    extension.discard(control)
    print("persistent service stopped after correctness call", flush=True)
    output_error = (resident_output.float() - expected_output.view(HV, V).float()).abs()
    state_error = (resident_state.float() - expected_state[0].float()).abs()
    torch.testing.assert_close(
        resident_output.float(),
        expected_output.view(HV, V).float(),
        atol=0.02,
        rtol=0.02,
    )
    torch.testing.assert_close(
        resident_state.float(),
        expected_state[0].float(),
        atol=0.02,
        rtol=0.02,
    )
    print("one-call correctness check passed", flush=True)

    samples = {
        "flashinfer_cutlass_cute_bf16_round_trip": [],
        "cuda_cpp_service_noop_lower_bound": [],
        "cuda_cpp_resident_compute_only": [],
        "cuda_cpp_resident_write_only": [],
        "cuda_cpp_bf16_round_trip": [],
    }

    # Alternate baseline-first and resident-first trials to reduce order and
    # thermal bias. Service initialization/teardown and state resets occur
    # outside every timed interval.
    for trial in range(trials):
        resident_first = trial % 2 == 1

        def measure_baseline() -> None:
            baseline_state.copy_(initial_state)
            for _ in range(warmup):
                baseline()
            samples["flashinfer_cutlass_cute_bf16_round_trip"].append(
                event_time_us(baseline, iterations)
            )

        def measure_resident() -> None:
            nonlocal control
            control = start_service()

            for _ in range(warmup):
                extension.debug_command(control, 1, 0, 0.0)
            samples["cuda_cpp_service_noop_lower_bound"].append(
                event_time_us(
                    lambda: extension.debug_command(control, 1, 0, 0.0),
                    iterations,
                )
            )

            resident_state.copy_(initial_state[0])
            resident_round_trip()
            for _ in range(warmup):
                resident_compute_only()
            samples["cuda_cpp_resident_compute_only"].append(
                event_time_us(resident_compute_only, iterations)
            )

            resident_state.copy_(initial_state[0])
            resident_round_trip()
            for _ in range(warmup):
                resident_write()
            samples["cuda_cpp_resident_write_only"].append(
                event_time_us(resident_write, iterations)
            )

            resident_state.copy_(initial_state[0])
            for _ in range(warmup):
                resident_round_trip()
            samples["cuda_cpp_bf16_round_trip"].append(
                event_time_us(resident_round_trip, iterations)
            )
            extension.discard(control)

        if resident_first:
            measure_resident()
            measure_baseline()
        else:
            measure_baseline()
            measure_resident()

        print(f"completed trial {trial + 1}/{trials}", flush=True)

    metrics = {name: summarize(values) for name, values in samples.items()}
    baseline_median = metrics["flashinfer_cutlass_cute_bf16_round_trip"]["median_us"]
    for name, values in metrics.items():
        values["speedup_vs_flashinfer"] = baseline_median / values["median_us"]
        values["change_vs_flashinfer_pct"] = ((values["median_us"] / baseline_median) - 1.0) * 100.0
    noop_median = metrics["cuda_cpp_service_noop_lower_bound"]["median_us"]
    compute_median = metrics["cuda_cpp_resident_compute_only"]["median_us"]
    write_median = metrics["cuda_cpp_resident_write_only"]["median_us"]
    round_trip_median = metrics["cuda_cpp_bf16_round_trip"]["median_us"]

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch_version": torch.__version__,
        "iterations_per_trial": iterations,
        "trials": trials,
        "warmup_calls": warmup,
        "shape": {
            "batch": 1,
            "tokens": 1,
            "query_heads": H,
            "value_heads": HV,
            "k": K,
            "v": V,
            "state_dtype": "bfloat16",
        },
        "state_bytes_per_direction": HV * V * K * 2,
        "correctness": {
            "output_max_abs": output_error.max().item(),
            "output_mean_abs": output_error.mean().item(),
            "state_max_abs": state_error.max().item(),
            "state_mean_abs": state_error.mean().item(),
        },
        "metrics": metrics,
        "derived_median_deltas_us": {
            "compute_above_minimal_service_noop": compute_median - noop_median,
            "bf16_writeback_above_compute": write_median - compute_median,
            "bf16_reload_above_writeback": round_trip_median - write_median,
        },
        "raw_samples_us": samples,
    }

    print(json.dumps(result, indent=2), flush=True)
    output_path = os.environ.get("GDN_MICROBENCH_JSON")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
