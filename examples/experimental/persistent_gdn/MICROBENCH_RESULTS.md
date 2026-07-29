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

# GDN decode kernel microbenchmark

Date: 2026-07-29

## Question

How much of the persistent-GDN result comes from the CUDA C++ rewrite itself,
rather than retaining state in RF/TMEM?

To remove the persistence advantage, the matched CUDA C++ mode reloads one
layer from a BF16 global-memory state pool before each update and writes the
updated BF16 state back after each update. This matches the state datatype and
global-memory traffic of the FlashInfer baseline.

## Configuration

- GPU: one B300 SXM6 AC, 148 SMs.
- Image: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc21`.
- Shape: batch 1, one token, 16 query/key heads, 32 value heads, K=V=128.
- State: `[32, 128, 128]` BF16, 1 MiB read and 1 MiB write per round trip.
- Baseline: FlashInfer CUTLASS/CuTe
  `gdn_decode_bf16state_mtp_ilp4_kernel`.
- Timing: CUDA events on the command stream, 20 warmups, 200 calls per trial,
  and 20 trials.
- Order: baseline-first and resident-first trials alternated.
- Persistent service initialization and teardown were outside timed regions.

## Result

| Mode | Median | Trial stdev | Change versus FlashInfer |
|---|---:|---:|---:|
| FlashInfer BF16 read + compute + write | 15.9312 us | 0.2098 us | reference |
| CUDA C++ service no-op lower bound | 23.9130 us | 0.0926 us | +50.10% |
| CUDA C++ resident compute only | 48.4366 us | 0.2283 us | +204.04% |
| CUDA C++ resident compute + BF16 write | 49.9479 us | 0.1805 us | +213.52% |
| CUDA C++ BF16 read + compute + write | 54.4070 us | 0.2282 us | **+241.51%** |

The traffic-matched CUDA C++ rewrite is therefore `54.4070 / 15.9312 =
3.42x` slower than the FlashInfer kernel.

Median differences between the CUDA C++ modes were:

- Recurrence above the minimal service no-op: 24.5236 us.
- BF16 state writeback above compute-only: 1.5114 us.
- BF16 state reload above write-only: 4.4590 us.

The no-op uses fewer descriptor writes than a GDN command, so the first
difference is an approximate decomposition rather than an isolated kernel
duration.

## Correctness

The one-call traffic-matched comparison produced:

```text
output max absolute error: 7.6293945e-06
output mean absolute error: 1.8628725e-09
state max absolute error: 0.00048828125
state mean absolute error: 2.2278801e-09
```

The existing 24-layer, two-token regression also passed after adding the
global-memory modes:

```text
output error: max=0.0001220703 mean=1.292544e-09
state error: max=2.384186e-07 mean=3.558968e-09
PASS: 48 row-sharded resident GDN decode commands match reference
```

## Interpretation

The CUDA C++ rewrite is not a faster GDN kernel. Even with state already
resident, its measured command latency is 48.44 us versus 15.93 us for the
complete FlashInfer read/compute/write operation. The persistent design only
wins if avoiding repeated state movement or system-level overlap compensates
for its slower CTA decomposition and command protocol.

Consequently, the earlier 7.68% E2E TPOT reduction cannot be attributed to a
faster rewritten GDN kernel. That E2E change must be analyzed as a system-level
effect, and reconciled with this kernel result before treating it as a
production performance claim.

These are hot steady-state measurements over a repeatedly used 1 MiB state,
so cache residency may reduce the apparent cost of global-memory state
traffic. The benchmark isolates one GDN layer, not the complete 24-layer
decode iteration.
