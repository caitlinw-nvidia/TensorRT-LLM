<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Qwen3.5-9B persistent GDN full E2E result

Date: 2026-07-29

## Configuration

- GPU: one B300 SXM6 AC (`umb-b300-dp-146`)
- TensorRT-LLM image: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc21`
- Backend: PyTorch, TP1, batch/concurrency 1
- Workload: one 1,024-token prompt generating 1,024 output tokens
- Baseline: shipping TensorRT-LLM GDN decode
- Resident: token 1 establishes state with the shipping path; the following
  1,022 decode updates use the 128-CTA RF/TMEM service
- Warmup requests: zero for both variants
- Repetitions: four matched A/B pairs; the fourth pair reversed execution
  order

## Result

| Metric | Shipping baseline | Resident RF/TMEM | Change |
|---|---:|---:|---:|
| Mean TPOT | 25.9638 ms | 23.9693 ms | **7.68% lower** |
| Mean total latency | 26,793.9341 ms | 24,714.1316 ms | **7.76% lower** |
| Mean output throughput | 38.2208 token/s | 41.4351 token/s | **8.41% higher** |
| TPOT standard deviation | 0.3378 ms | 0.1923 ms | |

The mean TPOT ratio is `25.9638 / 23.9693 = 1.0832x`. Every pair favored the
resident implementation:

| Pair | Order | Baseline TPOT | Resident TPOT | Reduction |
|---|---|---:|---:|---:|
| 1 | baseline, resident | 25.8632 ms | 23.8008 ms | 7.97% |
| 2 | baseline, resident | 25.6642 ms | 23.8739 ms | 6.98% |
| 3 | baseline, resident | 25.8790 ms | 23.9624 ms | 7.41% |
| 4 | resident, baseline | 26.4489 ms | 24.2402 ms | 8.35% |

The reversed-order pair rules out the simple explanation that resident always
benefited from running second.

## Updated interpretation

A subsequent traffic-matched kernel microbenchmark found that the CUDA C++
rewrite takes 54.4070 us when it reloads and writes BF16 state, versus 15.9312
us for the FlashInfer CUTLASS/CuTe kernel. The rewrite is therefore 3.42x
slower at the isolated GDN operation.

The E2E improvement below must not be interpreted as evidence that the
rewritten GDN computation is faster. It is a system-level result that still
needs to be reconciled with the kernel measurement. See
`MICROBENCH_RESULTS.md`.

## Persistence and writeback

Every resident run logged:

```text
PERSISTENT_GDN_CAPTURED: 24 layer states after decode token 1
PERSISTENT_GDN_READY: 128 CTAs, state=RF(12 rows/layer)+TMEM(20 rows/layer), global_state_writeback=disabled
PERSISTENT_GDN_TOKEN: completed resident decode token 1023
PERSISTENT_GDN_DISCARDED: RF/TMEM released, global_state_writeback=0 bytes
```

Thus each run completed all 1,022 resident updates. The command interface
contains no recurrent-state address, so each layer reads and updates its state
directly in RF/TMEM. Teardown uses `kStopWithoutEvict`; it does not materialize
the final state in global memory.

## Accuracy

The kernel-level reference test remains:

```text
output error: max=0.0001220703 mean=1.292544e-09
state error: max=2.384186e-07 mean=3.558968e-09
PASS: 48 row-sharded resident GDN decode commands match reference
```

All eight full benchmark cases completed with the requested output length of
1,024 and exit code zero.

## Profiling note

A separate attempt to stop a focused Nsight capture during the full decode was
excluded from timing. `cudaProfilerStop` waited for the intentionally
long-lived persistent kernel to exit, while that kernel was waiting for later
decode commands. This is a profiler-control deadlock, not an E2E execution
failure. The earlier short E2E Nsight trace already established that there
were zero shipping GDN launches after the persistent service started.

## Artifacts

Raw logs and aggregate CSV are in
`results/persistent_gdn_full_e2e_20260729/`.
