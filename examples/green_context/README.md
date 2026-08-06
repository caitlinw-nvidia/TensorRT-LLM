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

# Green-context CUDA graph prototype

This directory is a standalone prototype for splitting a GPU's SM resources
between two CUDA green contexts and using their streams from PyTorch. It proves
that a regular main stream can fork work onto two green-context streams and join
them within one captured CUDA graph.

The opt-in Python implementation in
`tensorrt_llm/_torch/green_context.py` is wired into PyExecutor startup and
shutdown. Set `TRTLLM_ENABLE_GREEN_CONTEXT=1` to provision two equal green
contexts and register their streams in the model's per-forward extra attributes
as `green_context_streams`. Enabling the flag only provisions the resources; a
model call site must explicitly route work to the streams. This preserves the
regular full-SM execution stream for all other kernels.

The benchmark programs use synthetic elementwise and no-op kernels, so no model
checkpoint is required.

## Components

- `greenContextTwoStreams.cu` is a Driver API smoke test. It creates two green
  contexts, launches one kernel on each stream, and validates the output.
- `greenContextPair.cpp` exposes a small C ABI that owns two green contexts and
  their streams. Python wraps the stream handles with
  `torch.cuda.ExternalStream`.
- `test_external_stream_graph.py` validates capture and replay of a
  main-stream-to-two-green-streams fork/join graph.
- `benchmark_external_stream_graph.py` compares regular and green-context
  streams with synthetic elementwise work.
- `benchmark_noop_fork_join.py` isolates the graph fork/join cost by comparing
  serial, regular two-stream, and green-context two-stream no-op graphs.
- `benchmark_executor_green_context.py` exercises the implementation owned by
  PyExecutor and measures its context lifecycle and CUDA graph fork/join costs.

## Requirements

- A CUDA toolkit and driver that expose the green-context Driver API used here.
  The prototype was validated with CUDA 13.2.
- A GPU that supports SM resource partitioning.
- PyTorch with `torch.cuda.ExternalStream` and CUDA graph support for the Python
  programs.

Before running a GPU program, verify that the selected device is idle:

```bash
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
```

Build the programs and shared libraries from this directory:

```bash
./build.sh
```

Then run the smoke tests and benchmarks:

```bash
./greenContextTwoStreams
python test_external_stream_graph.py
python benchmark_noop_fork_join.py
python benchmark_external_stream_graph.py
python benchmark_executor_green_context.py
```

The code initializes PyTorch's CUDA primary context first; it does not call
`cuCtxCreate`. It obtains the device's SM resource, rounds each half-partition
down to the co-scheduling alignment, and leaves any unsplittable SMs in the
reported remainder. For example, a 148-SM B200 produced two 72-SM green
contexts plus a four-SM remainder.

The timing programs report CUDA-event GPU spans and host graph-submission time
as JSON. They include warmup and interleave benchmark cases to reduce ordering
bias. Results are microbenchmarks and should not be treated as model-level
TensorRT-LLM performance.
