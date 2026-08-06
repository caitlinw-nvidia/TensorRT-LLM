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
from pathlib import Path

import torch


def check(status: int, library: ctypes.CDLL, operation: str) -> None:
    if status != 0:
        message = library.gcLastError().decode("utf-8", errors="replace")
        raise RuntimeError(f"{operation} failed with status {status}: {message}")


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
    # Materialize PyTorch's primary context before provisioning green contexts.
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
    # [group0, group1, device total, minimum, alignment, remainder]
    sm_counts = (ctypes.c_uint * 6)()
    context_ids = (ctypes.c_uint64 * 2)()
    check(
        library.gcCreatePair(0, handles, sm_counts, context_ids),
        library,
        "gcCreatePair",
    )

    external_streams = [torch.cuda.ExternalStream(handles[i], device=0) for i in range(2)]
    initial_queries = [query_stream(library, handles[i]) for i in range(2)]
    for i, (queried_sms, queried_id) in enumerate(initial_queries):
        assert queried_sms == sm_counts[i]
        assert queried_id == context_ids[i]
    assert context_ids[0] != context_ids[1]

    count = 1 << 20
    input0 = torch.empty(count, dtype=torch.float32, device="cuda")
    input1 = torch.empty_like(input0)
    branch0 = torch.empty_like(input0)
    branch1 = torch.empty_like(input0)
    combined = torch.empty_like(input0)

    # Warm the exact Torch kernels before capture so compilation/allocation is
    # excluded from the graph.
    input0.fill_(1.0)
    input1.fill_(2.0)
    with torch.cuda.stream(external_streams[0]):
        torch.mul(input0, 2.0, out=branch0)
    with torch.cuda.stream(external_streams[1]):
        torch.add(input1, 3.0, out=branch1)
    torch.cuda.synchronize()

    main_stream = torch.cuda.Stream(device=0)
    fork_event = torch.cuda.Event()
    done_events = [torch.cuda.Event(), torch.cuda.Event()]

    # Materialize the events before capture. They may be reused and re-recorded
    # as captured event nodes below.
    with torch.cuda.stream(main_stream):
        fork_event.record()
    for stream, event in zip(external_streams, done_events):
        with torch.cuda.stream(stream):
            stream.wait_event(fork_event)
            event.record()
    for event in done_events:
        main_stream.wait_event(event)
    main_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main_stream):
        fork_event.record()
        with torch.cuda.stream(external_streams[0]):
            external_streams[0].wait_event(fork_event)
            torch.mul(input0, 2.0, out=branch0)
            done_events[0].record()
        with torch.cuda.stream(external_streams[1]):
            external_streams[1].wait_event(fork_event)
            torch.add(input1, 3.0, out=branch1)
            done_events[1].record()
        main_stream.wait_event(done_events[0])
        main_stream.wait_event(done_events[1])
        torch.add(branch0, branch1, out=combined)

    replay_checks = []
    for iteration in range(1, 6):
        lhs = float(iteration)
        rhs = float(iteration * 10)
        input0.fill_(lhs)
        input1.fill_(rhs)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = lhs * 2.0 + rhs + 3.0
        actual_min = combined.min().item()
        actual_max = combined.max().item()
        if actual_min != expected or actual_max != expected:
            raise AssertionError(
                f"replay {iteration}: expected {expected}, range=[{actual_min}, {actual_max}]"
            )
        replay_checks.append({"iteration": iteration, "expected": expected, "actual": actual_min})

    post_replay_queries = [query_stream(library, handles[i]) for i in range(2)]
    assert post_replay_queries == initial_queries

    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_sms": sm_counts[2],
        "min_partition_sms": sm_counts[3],
        "coscheduled_alignment_sms": sm_counts[4],
        "remainder_sms": sm_counts[5],
        "green_contexts": [
            {
                "index": i,
                "id": context_ids[i],
                "stream_handle": handles[i],
                "sm_count": sm_counts[i],
                "post_replay_sm_count": post_replay_queries[i][0],
                "post_replay_context_id": post_replay_queries[i][1],
            }
            for i in range(2)
        ],
        "cuda_graph_replays": replay_checks,
    }
    print(json.dumps(result, indent=2))

    torch.cuda.synchronize()
    del graph
    del external_streams
    gc.collect()
    check(library.gcDestroyPair(), library, "gcDestroyPair")


if __name__ == "__main__":
    main()
