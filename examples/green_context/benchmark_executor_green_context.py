#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the production PyExecutor green-context stream implementation."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


def load_green_context_pair() -> type:
    """Load the implementation without requiring a built TensorRT-LLM wheel."""
    module_path = Path(__file__).parents[2] / "tensorrt_llm" / "_torch" / "green_context.py"
    spec = importlib.util.spec_from_file_location("trtllm_executor_green_context", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load green-context module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GreenContextPair


def summarize(values: list[float]) -> dict[str, float]:
    """Return stable summary statistics for a set of microsecond samples."""
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


def materialize_events(
    main_stream: torch.cuda.Stream,
    branch_streams: tuple[torch.cuda.Stream, torch.cuda.Stream],
    fork_event: torch.cuda.Event,
    done_events: tuple[torch.cuda.Event, torch.cuda.Event],
) -> None:
    """Create event state before CUDA graph capture."""
    with torch.cuda.stream(main_stream):
        fork_event.record(main_stream)
        for stream, done_event in zip(branch_streams, done_events, strict=True):
            with torch.cuda.stream(stream):
                stream.wait_event(fork_event)
                done_event.record(stream)
        for done_event in done_events:
            main_stream.wait_event(done_event)
    main_stream.synchronize()


def execute_regular_fork_join(
    branch_streams: tuple[torch.cuda.Stream, torch.cuda.Stream],
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    fork_event: torch.cuda.Event,
    done_events: tuple[torch.cuda.Event, torch.cuda.Event],
) -> tuple[Any, Any]:
    """Run the same topology as the production helper on regular streams."""
    parent_stream = torch.cuda.current_stream()
    fork_event.record(parent_stream)
    results = []
    for fn, stream, done_event in zip((fn0, fn1), branch_streams, done_events, strict=True):
        with torch.cuda.stream(stream):
            stream.wait_event(fork_event)
            results.append(fn())
            done_event.record(stream)
    for done_event in done_events:
        parent_stream.wait_event(done_event)
    return results[0], results[1]


def make_ops() -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    """Create two identical one-element kernels with preallocated outputs."""
    inputs = (
        torch.ones(1, device="cuda"),
        torch.ones(1, device="cuda"),
    )
    outputs = (
        torch.empty_like(inputs[0]),
        torch.empty_like(inputs[1]),
    )

    def op0() -> torch.Tensor:
        return torch.add(inputs[0], 1.0, out=outputs[0])

    def op1() -> torch.Tensor:
        return torch.add(inputs[1], 1.0, out=outputs[1])

    return op0, op1


def capture_serial() -> tuple:
    """Capture two kernels serially on one regular stream."""
    main_stream = torch.cuda.Stream()
    op0, op1 = make_ops()
    graph = torch.cuda.CUDAGraph()
    started_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=main_stream):
        op0()
        op1()
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    return graph, main_stream, capture_ms, (op0, op1)


def capture_regular_fork_join() -> tuple:
    """Capture a parent-to-two-regular-streams fork/join."""
    main_stream = torch.cuda.Stream()
    branch_streams = (torch.cuda.Stream(), torch.cuda.Stream())
    fork_event = torch.cuda.Event()
    done_events = (torch.cuda.Event(), torch.cuda.Event())
    materialize_events(main_stream, branch_streams, fork_event, done_events)
    op0, op1 = make_ops()

    graph = torch.cuda.CUDAGraph()
    started_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=main_stream):
        execute_regular_fork_join(branch_streams, op0, op1, fork_event, done_events)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    keepalive = (branch_streams, fork_event, done_events, op0, op1)
    return graph, main_stream, capture_ms, keepalive


def capture_green_fork_join(pair: Any) -> tuple:
    """Capture the production green-context helper in a CUDA graph."""
    main_stream = torch.cuda.Stream()
    fork_event = torch.cuda.Event()
    done_events = (torch.cuda.Event(), torch.cuda.Event())
    materialize_events(main_stream, pair.streams, fork_event, done_events)
    op0, op1 = make_ops()

    graph = torch.cuda.CUDAGraph()
    started_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=main_stream):
        pair.execute_in_parallel(op0, op1, fork_event, done_events)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    keepalive = (pair, fork_event, done_events, op0, op1)
    return graph, main_stream, capture_ms, keepalive


def measure_gpu_interleaved(
    cases: dict[str, tuple], warmup_replays: int, samples: int
) -> dict[str, dict[str, float]]:
    """Measure graph spans in a rotating order to reduce ordering bias."""
    for graph, main_stream, _, _ in cases.values():
        with torch.cuda.stream(main_stream):
            for _ in range(warmup_replays):
                graph.replay()
    torch.cuda.synchronize()

    names = list(cases)
    values = {name: [] for name in names}
    timing_events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in names
    }
    for iteration in range(samples):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            graph, main_stream, _, _ = cases[name]
            start, end = timing_events[name]
            with torch.cuda.stream(main_stream):
                start.record()
                graph.replay()
                end.record()
            end.synchronize()
            values[name].append(start.elapsed_time(end) * 1000.0)
    return {name: summarize(case_values) for name, case_values in values.items()}


def measure_host_submit(
    graph: torch.cuda.CUDAGraph,
    main_stream: torch.cuda.Stream,
    warmup_replays: int,
    replays: int,
) -> dict[str, float]:
    """Measure asynchronous host graph-submission cost."""
    with torch.cuda.stream(main_stream):
        for _ in range(warmup_replays):
            graph.replay()
    torch.cuda.synchronize()

    started_ns = time.perf_counter_ns()
    with torch.cuda.stream(main_stream):
        for _ in range(replays):
            graph.replay()
    elapsed_ns = time.perf_counter_ns() - started_ns
    torch.cuda.synchronize()
    return {
        "count": replays,
        "mean_submit_us": elapsed_ns / replays / 1000.0,
    }


def measure_pair_lifecycle(pair_type: type, samples: int) -> dict[str, Any]:
    """Measure startup-only pair creation and destruction on an initialized GPU."""
    creation_us = []
    destruction_us = []
    last_info = None
    for _ in range(samples):
        pair = pair_type.create()
        creation_us.append(pair.info.creation_time_ms * 1000.0)
        last_info = pair.info
        started_ns = time.perf_counter_ns()
        pair.close()
        destruction_us.append((time.perf_counter_ns() - started_ns) / 1000.0)
    return {
        "samples": samples,
        "creation": summarize(creation_us),
        "destruction": summarize(destruction_us),
        "last_info": last_info,
    }


def parse_args() -> argparse.Namespace:
    """Parse benchmark controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-replays", type=int, default=1000)
    parser.add_argument("--gpu-samples", type=int, default=5000)
    parser.add_argument("--host-replays", type=int, default=10000)
    parser.add_argument("--lifecycle-samples", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    """Run lifecycle and CUDA graph timing measurements."""
    args = parse_args()
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    pair_type = load_green_context_pair()

    lifecycle = measure_pair_lifecycle(pair_type, args.lifecycle_samples)
    pair = pair_type.create()
    cases = {
        "serial_two_kernels": capture_serial(),
        "regular_stream_fork_join": capture_regular_fork_join(),
        "green_context_fork_join": capture_green_fork_join(pair),
    }
    gpu = measure_gpu_interleaved(cases, args.warmup_replays, args.gpu_samples)
    host = {
        name: measure_host_submit(graph, main_stream, args.warmup_replays, args.host_replays)
        for name, (graph, main_stream, _, _) in cases.items()
    }

    serial_us = gpu["serial_two_kernels"]["median_us"]
    regular_us = gpu["regular_stream_fork_join"]["median_us"]
    green_us = gpu["green_context_fork_join"]["median_us"]
    info = pair.info
    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_sms": info.device_sm_count,
        "green_context_sm_counts": info.stream_sm_counts,
        "green_context_ids": info.context_ids,
        "remainder_sms": info.remainder_sm_count,
        "kernel": "two one-element torch.add kernels per graph",
        "capture_ms": {name: case[2] for name, case in cases.items()},
        "gpu_span": gpu,
        "host_submit": host,
        "lifecycle": lifecycle,
        "derived": {
            "regular_fork_join_minus_serial_us": regular_us - serial_us,
            "green_fork_join_minus_serial_us": green_us - serial_us,
            "green_minus_regular_fork_join_us": green_us - regular_us,
        },
    }
    print(json.dumps(result, indent=2, default=lambda item: item.__dict__))

    torch.cuda.synchronize()
    del cases
    pair.close()


if __name__ == "__main__":
    main()
