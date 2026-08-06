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

WARMUP_REPLAYS = 1000
GPU_SAMPLES = 5000
HOST_REPLAYS = 10000


def check_gc(status: int, library: ctypes.CDLL, operation: str) -> None:
    if status != 0:
        message = library.gcLastError().decode("utf-8", errors="replace")
        raise RuntimeError(f"{operation} failed with status {status}: {message}")


def launch_noop(library: ctypes.CDLL, stream: torch.cuda.Stream) -> None:
    status = library.launchNoop(ctypes.c_uint64(stream.cuda_stream))
    if status != 0:
        raise RuntimeError(f"launch_noop failed with CUDA runtime status {status}")


def materialize_events(
    main_stream: torch.cuda.Stream,
    branch_streams: list[torch.cuda.Stream],
    fork_event: torch.cuda.Event,
    done_events: list[torch.cuda.Event],
) -> None:
    with torch.cuda.stream(main_stream):
        fork_event.record()
    for stream, event in zip(branch_streams, done_events):
        with torch.cuda.stream(stream):
            stream.wait_event(fork_event)
            event.record()
    for event in done_events:
        main_stream.wait_event(event)
    main_stream.synchronize()


def capture_serial(noop_library: ctypes.CDLL):
    main_stream = torch.cuda.Stream(device=0)
    graph = torch.cuda.CUDAGraph()
    start_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=main_stream):
        launch_noop(noop_library, main_stream)
        launch_noop(noop_library, main_stream)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - start_ns) / 1.0e6
    return graph, main_stream, capture_ms, ()


def capture_fork_join(noop_library: ctypes.CDLL, branch_streams: list[torch.cuda.Stream]):
    main_stream = torch.cuda.Stream(device=0)
    fork_event = torch.cuda.Event()
    done_events = [torch.cuda.Event(), torch.cuda.Event()]
    materialize_events(main_stream, branch_streams, fork_event, done_events)

    graph = torch.cuda.CUDAGraph()
    start_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=main_stream):
        fork_event.record()
        for stream, event in zip(branch_streams, done_events):
            with torch.cuda.stream(stream):
                stream.wait_event(fork_event)
                launch_noop(noop_library, stream)
                event.record()
        for event in done_events:
            main_stream.wait_event(event)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - start_ns) / 1.0e6
    keepalive = (branch_streams, fork_event, done_events)
    return graph, main_stream, capture_ms, keepalive


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


def measure_gpu_interleaved(cases: dict[str, tuple]) -> dict[str, dict]:
    for graph, main_stream, _, _ in cases.values():
        with torch.cuda.stream(main_stream):
            for _ in range(WARMUP_REPLAYS):
                graph.replay()
    torch.cuda.synchronize()

    names = list(cases)
    samples = {name: [] for name in names}
    timing_events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in names
    }
    for iteration in range(GPU_SAMPLES):
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            graph, main_stream, _, _ = cases[name]
            start, end = timing_events[name]
            with torch.cuda.stream(main_stream):
                start.record()
                graph.replay()
                end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)
    return {name: summarize(values) for name, values in samples.items()}


def measure_host_submit(
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
        "mean_submit_us": elapsed_ns / HOST_REPLAYS / 1000.0,
    }


def query_stream(gc_library: ctypes.CDLL, handle: int) -> tuple[int, int]:
    sm_count = ctypes.c_uint()
    context_id = ctypes.c_uint64()
    check_gc(
        gc_library.gcQueryStream(
            ctypes.c_uint64(handle), ctypes.byref(sm_count), ctypes.byref(context_id)
        ),
        gc_library,
        "gcQueryStream",
    )
    return sm_count.value, context_id.value


def main() -> None:
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    directory = Path(__file__).parent
    gc_library = ctypes.CDLL(str(directory / "libgreen_context_pair.so"))
    gc_library.gcLastError.restype = ctypes.c_char_p
    gc_library.gcCreatePair.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    gc_library.gcCreatePair.restype = ctypes.c_int
    gc_library.gcQueryStream.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    gc_library.gcQueryStream.restype = ctypes.c_int
    gc_library.gcDestroyPair.restype = ctypes.c_int

    noop_library = ctypes.CDLL(str(directory / "libnoop_kernel.so"))
    noop_library.launchNoop.argtypes = [ctypes.c_uint64]
    noop_library.launchNoop.restype = ctypes.c_int

    handles = (ctypes.c_uint64 * 2)()
    sm_counts = (ctypes.c_uint * 6)()
    context_ids = (ctypes.c_uint64 * 2)()
    check_gc(
        gc_library.gcCreatePair(0, handles, sm_counts, context_ids),
        gc_library,
        "gcCreatePair",
    )
    green_streams = [torch.cuda.ExternalStream(handles[i], device=0) for i in range(2)]
    regular_streams = [torch.cuda.Stream(device=0) for _ in range(2)]
    initial_queries = [query_stream(gc_library, handles[i]) for i in range(2)]

    # Every case contains exactly two instances of the same one-thread kernel.
    cases = {
        "serial_two_noops": capture_serial(noop_library),
        "regular_stream_fork_join": capture_fork_join(noop_library, regular_streams),
        "green_context_fork_join": capture_fork_join(noop_library, green_streams),
    }
    gpu = measure_gpu_interleaved(cases)
    host = {
        name: measure_host_submit(graph, main_stream)
        for name, (graph, main_stream, _, _) in cases.items()
    }

    serial = gpu["serial_two_noops"]["median_us"]
    regular_fork = gpu["regular_stream_fork_join"]["median_us"]
    green_fork = gpu["green_context_fork_join"]["median_us"]
    post_queries = [query_stream(gc_library, handles[i]) for i in range(2)]
    assert post_queries == initial_queries

    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_sms": sm_counts[2],
        "green_context_sm_counts": [sm_counts[0], sm_counts[1]],
        "green_context_ids": [context_ids[0], context_ids[1]],
        "remainder_sms": sm_counts[5],
        "kernel": "two one-block, one-thread no-op kernels per graph",
        "warmup_replays": WARMUP_REPLAYS,
        "gpu_samples": GPU_SAMPLES,
        "host_replays": HOST_REPLAYS,
        "capture_ms": {name: case[2] for name, case in cases.items()},
        "gpu_span": gpu,
        "host_submit": host,
        "derived": {
            "regular_fork_join_minus_serial_us": regular_fork - serial,
            "green_fork_join_minus_serial_us": green_fork - serial,
            "green_minus_regular_fork_join_us": green_fork - regular_fork,
        },
        "post_benchmark_green_context_queries": post_queries,
    }
    print(json.dumps(result, indent=2))

    torch.cuda.synchronize()
    del cases
    del green_streams
    gc.collect()
    check_gc(gc_library.gcDestroyPair(), gc_library, "gcDestroyPair")


if __name__ == "__main__":
    main()
