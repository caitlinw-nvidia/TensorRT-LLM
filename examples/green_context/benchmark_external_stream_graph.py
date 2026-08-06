#!/usr/bin/env python3
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

import ctypes
import gc
import json
import statistics
import time
from pathlib import Path

import torch

WARMUP_REPLAYS = 500
GPU_SAMPLES = 3000
HOST_REPLAYS = 10000


def check(status: int, library: ctypes.CDLL, operation: str) -> None:
    if status != 0:
        message = library.gcLastError().decode("utf-8", errors="replace")
        raise RuntimeError(f"{operation} failed with status {status}: {message}")


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "mean_us": statistics.fmean(ordered),
        "median_us": statistics.median(ordered),
        "p10_us": ordered[int(0.10 * (count - 1))],
        "p90_us": ordered[int(0.90 * (count - 1))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def capture_event_graph(streams: list[torch.cuda.Stream]):
    main_stream = torch.cuda.Stream(device=0)
    fork_event = torch.cuda.Event()
    done_events = [torch.cuda.Event(), torch.cuda.Event()]

    # Materialize reusable event handles before capture.
    with torch.cuda.stream(main_stream):
        fork_event.record()
    for stream, event in zip(streams, done_events):
        with torch.cuda.stream(stream):
            stream.wait_event(fork_event)
            event.record()
    for event in done_events:
        main_stream.wait_event(event)
    main_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    start_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=main_stream):
        fork_event.record()
        for stream, event in zip(streams, done_events):
            with torch.cuda.stream(stream):
                stream.wait_event(fork_event)
                event.record()
        for event in done_events:
            main_stream.wait_event(event)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - start_ns) / 1.0e6
    return graph, main_stream, capture_ms, None


def capture_work_graph(streams: list[torch.cuda.Stream], element_count: int):
    main_stream = torch.cuda.Stream(device=0)
    fork_event = torch.cuda.Event()
    done_events = [torch.cuda.Event(), torch.cuda.Event()]
    input0 = torch.full((element_count,), 1.25, dtype=torch.float32, device="cuda")
    input1 = torch.full((element_count,), 2.5, dtype=torch.float32, device="cuda")
    branch0 = torch.empty_like(input0)
    branch1 = torch.empty_like(input0)
    combined = torch.empty_like(input0)

    # Warm the exact Torch kernels before graph capture.
    with torch.cuda.stream(streams[0]):
        torch.mul(input0, 2.0, out=branch0)
    with torch.cuda.stream(streams[1]):
        torch.add(input1, 3.0, out=branch1)
    torch.cuda.synchronize()

    with torch.cuda.stream(main_stream):
        fork_event.record()
    for stream, event in zip(streams, done_events):
        with torch.cuda.stream(stream):
            stream.wait_event(fork_event)
            event.record()
    for event in done_events:
        main_stream.wait_event(event)
    main_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    start_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=main_stream):
        fork_event.record()
        with torch.cuda.stream(streams[0]):
            streams[0].wait_event(fork_event)
            torch.mul(input0, 2.0, out=branch0)
            done_events[0].record()
        with torch.cuda.stream(streams[1]):
            streams[1].wait_event(fork_event)
            torch.add(input1, 3.0, out=branch1)
            done_events[1].record()
        main_stream.wait_event(done_events[0])
        main_stream.wait_event(done_events[1])
        torch.add(branch0, branch1, out=combined)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - start_ns) / 1.0e6

    expected = 1.25 * 2.0 + 2.5 + 3.0
    graph.replay()
    torch.cuda.synchronize()
    actual_min = combined.min().item()
    actual_max = combined.max().item()
    assert actual_min == expected and actual_max == expected
    tensors = (input0, input1, branch0, branch1, combined)
    return graph, main_stream, capture_ms, tensors


def measure_gpu_interleaved(cases: dict[str, tuple]) -> dict[str, dict]:
    for graph, main_stream, _, _ in cases.values():
        with torch.cuda.stream(main_stream):
            for _ in range(WARMUP_REPLAYS):
                graph.replay()
    torch.cuda.synchronize()

    samples = {name: [] for name in cases}
    names = list(cases)
    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in names
    }
    for iteration in range(GPU_SAMPLES):
        order = names if iteration % 2 == 0 else list(reversed(names))
        for name in order:
            graph, main_stream, _, _ = cases[name]
            start, end = events[name]
            with torch.cuda.stream(main_stream):
                start.record()
                graph.replay()
                end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)
    return {name: summarize(values) for name, values in samples.items()}


def measure_host_enqueue(
    graph: torch.cuda.CUDAGraph, main_stream: torch.cuda.Stream
) -> dict[str, float]:
    with torch.cuda.stream(main_stream):
        for _ in range(WARMUP_REPLAYS):
            graph.replay()
    torch.cuda.synchronize()

    start_ns = time.perf_counter_ns()
    with torch.cuda.stream(main_stream):
        for _ in range(HOST_REPLAYS):
            graph.replay()
    elapsed_ns = time.perf_counter_ns() - start_ns
    torch.cuda.synchronize()
    return {
        "count": HOST_REPLAYS,
        "mean_enqueue_us": elapsed_ns / HOST_REPLAYS / 1000.0,
    }


def query_stream(library: ctypes.CDLL, handle: int) -> tuple[int, int]:
    sm_count = ctypes.c_uint()
    context_id = ctypes.c_uint64()
    check(
        library.gcQueryStream(
            ctypes.c_uint64(handle), ctypes.byref(sm_count), ctypes.byref(context_id)
        ),
        library,
        "gcQueryStream",
    )
    return sm_count.value, context_id.value


def main() -> None:
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    library = ctypes.CDLL(str(Path(__file__).with_name("libgreen_context_pair.so")))
    library.gcLastError.restype = ctypes.c_char_p
    library.gcCreatePair.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.gcCreatePair.restype = ctypes.c_int
    library.gcQueryStream.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.gcQueryStream.restype = ctypes.c_int
    library.gcDestroyPair.restype = ctypes.c_int

    handles = (ctypes.c_uint64 * 2)()
    sm_counts = (ctypes.c_uint * 6)()
    context_ids = (ctypes.c_uint64 * 2)()
    torch.cuda.synchronize()
    setup_start_ns = time.perf_counter_ns()
    check(
        library.gcCreatePair(0, handles, sm_counts, context_ids),
        library,
        "gcCreatePair",
    )
    setup_ms = (time.perf_counter_ns() - setup_start_ns) / 1.0e6

    wrap_start_ns = time.perf_counter_ns()
    green_streams = [torch.cuda.ExternalStream(handles[i], device=0) for i in range(2)]
    external_stream_wrap_ms = (time.perf_counter_ns() - wrap_start_ns) / 1.0e6
    regular_streams = [torch.cuda.Stream(device=0) for _ in range(2)]
    stream_queries = [query_stream(library, handles[i]) for i in range(2)]

    all_results = {}
    workloads = (
        ("empty_graph_control", 0),
        ("elementwise_1k", 1 << 10),
        ("elementwise_64k", 1 << 16),
        ("elementwise_1m", 1 << 20),
    )
    for workload, element_count in workloads:
        capture = (
            capture_event_graph
            if element_count == 0
            else (lambda streams: capture_work_graph(streams, element_count))
        )
        cases = {
            "regular_streams": capture(regular_streams),
            "green_context_streams": capture(green_streams),
        }
        gpu = measure_gpu_interleaved(cases)
        host = {
            name: measure_host_enqueue(graph, main_stream)
            for name, (graph, main_stream, _, _) in cases.items()
        }
        regular_median = gpu["regular_streams"]["median_us"]
        green_median = gpu["green_context_streams"]["median_us"]
        all_results[workload] = {
            "element_count": element_count,
            "capture_ms": {name: case[2] for name, case in cases.items()},
            "gpu_span": gpu,
            "host_enqueue": host,
            "green_minus_regular_median_us": green_median - regular_median,
            "green_over_regular_ratio": green_median / regular_median,
        }
        torch.cuda.synchronize()
        del cases
        gc.collect()

    post_queries = [query_stream(library, handles[i]) for i in range(2)]
    assert post_queries == stream_queries
    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "green_context_setup_ms": setup_ms,
        "external_stream_wrap_ms": external_stream_wrap_ms,
        "device_sms": sm_counts[2],
        "min_partition_sms": sm_counts[3],
        "coscheduled_alignment_sms": sm_counts[4],
        "remainder_sms": sm_counts[5],
        "green_contexts": [
            {
                "index": i,
                "id": context_ids[i],
                "sm_count": sm_counts[i],
                "post_benchmark_query": post_queries[i],
            }
            for i in range(2)
        ],
        "warmup_replays": WARMUP_REPLAYS,
        "gpu_samples": GPU_SAMPLES,
        "host_replays": HOST_REPLAYS,
        "workloads": all_results,
    }
    print(json.dumps(result, indent=2))

    torch.cuda.synchronize()
    del green_streams
    gc.collect()
    check(library.gcDestroyPair(), library, "gcDestroyPair")


if __name__ == "__main__":
    main()
