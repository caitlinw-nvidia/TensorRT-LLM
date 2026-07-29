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

# Persistent GDN state in RF/TMEM

This directory contains an opt-in prototype that keeps the complete
Qwen3.5-9B gated-delta-network recurrent state resident in registers and TMEM
across autoregressive decode tokens on B300.

It is intentionally separate from the production TensorRT-LLM dispatch path.
`sitecustomize.py` monkey-patches the PyTorch GDN call only when
`TRTLLM_PERSISTENT_GDN_EXTENSION` is set. This makes the experiment easy to
inspect and reproduce without enabling an incomplete batch-1 specialization
for other users.

## Design

- 128 persistent CTAs partition all 4,096 state rows.
- Each CTA owns 32 rows from each of the model's 24 GDN layers.
- Twelve rows per layer are retained in 144 named FP32 registers per thread.
- Twenty rows per layer are retained in 480 live TMEM columns per CTA.
- The 48 MiB state split is 18 MiB in RF and 30 MiB in TMEM.
- Q, K, V, gates, layer parameters, and output are passed with each command.
  No decode command carries a recurrent-state pointer.
- State is read from global memory once during initialization. The E2E path
  releases RF/TMEM with `kStopWithoutEvict`, producing zero final state
  writeback.

`register_state_generated.cuh` uses explicitly named scalar variables because
a runtime-indexed C array can be lowered to local memory. Regenerate it with:

```bash
python examples/experimental/persistent_gdn/generate_register_state.py
```

## Files

- `persistent_state.cu`: long-lived service, command queue, TMEM accessors,
  and the row-sharded GDN recurrence.
- `sitecustomize.py`: opt-in routing from TensorRT-LLM's PyTorch GDN function.
- `test_gdn_decode.py`: 24 layers by two tokens against an FP32 PyTorch
  reference.
- `bench_gdn_decode.py`: kernel-only FlashInfer versus CUDA C++ benchmark,
  including BF16 global-memory write-only and read/write modes.
- `test_persistent_state.py`: synthetic RF/TMEM retention test and extension
  builder.
- `run_qwen35_9b_full_ab.sh`: matched shipping-versus-resident E2E runner.
- `tmem_allocation_probe.cu`: legal B300 TMEM allocation probe.
- `sweep_register_caps.sh`: register-cap compilation sweep.
- `RESULTS.md`: full four-pair ISL-1024/OSL-1024 measurements.
- `MICROBENCH_RESULTS.md`: traffic-matched GDN kernel measurements.

## Requirements and limitations

The tested environment was:

- One B300 SXM6 AC (SM103a).
- `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc21`.
- Qwen3.5-9B, TP1, batch/concurrency 1.

The prototype assumes standard non-speculative decode, exactly 24 GDN layers,
and fixed Qwen3.5-9B dimensions. It is not a production-ready scheduler for
multiple requests, request preemption, migration, or variable state shapes.

## Accuracy test

Inside the tested container on B300:

```bash
cd examples/experimental/persistent_gdn
export TORCH_EXTENSIONS_DIR=/tmp/persistent_gdn_build
export GDN_MAX_REGS=192
python test_gdn_decode.py
```

The measured maximum errors were `0.0001220703` for BF16 output and
`2.384186e-07` for FP32 state.

## Kernel microbenchmark

Inside the same container on B300:

```bash
cd examples/experimental/persistent_gdn
export TORCH_EXTENSIONS_DIR=/tmp/persistent_gdn_build
export GDN_MAX_REGS=192
export GDN_MICROBENCH_JSON=/tmp/gdn_microbench.json
python bench_gdn_decode.py
```

The traffic-matched mode reloads the BF16 state from global memory and writes
it back for every command. See `MICROBENCH_RESULTS.md` for the measured
FlashInfer comparison and important interpretation caveats.

## Full E2E A/B

Provide absolute host paths and run on the allocated GPU node:

```bash
export MODEL=/path/to/Qwen3.5-9B
export DATASET=/path/to/isl1024_osl1024.json
export CONFIG=/path/to/qwen9b_benchmark.yaml
export WORK_DIR=/path/to/writable/artifacts

examples/experimental/persistent_gdn/run_qwen35_9b_full_ab.sh
```

The runner builds the extension if needed, launches an unmodified shipping
baseline, launches the opt-in resident path, and writes `metrics.csv` plus both
raw logs under `$WORK_DIR/full_e2e`.

Do not place `cudaProfilerStop` in the middle of this experiment. That API
waits for the intentionally long-lived service kernel to finish, while the
service kernel is waiting for later decode commands. Use a full-lifetime trace
or profile a finite synthetic test instead.
