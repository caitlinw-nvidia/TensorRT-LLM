# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel_on_streams,
    with_multi_stream,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA is required for stream tests")


def test_explicit_two_stream_fork_join_is_cuda_graph_safe():
    parent = torch.cuda.Stream()
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    fork_event = torch.cuda.Event()
    done0 = torch.cuda.Event()
    done1 = torch.cuda.Event()

    x = torch.randn(32, 64, device="cuda")
    weight0 = torch.randn(64, 48, device="cuda")
    weight1 = torch.randn(64, 40, device="cuda")
    output0 = torch.empty(32, 48, device="cuda")
    output1 = torch.empty(32, 40, device="cuda")
    branch_streams = []

    def fn0():
        branch_streams.append(torch.cuda.current_stream().cuda_stream)
        torch.mm(x, weight0, out=output0)
        return output0

    def fn1():
        branch_streams.append(torch.cuda.current_stream().cuda_stream)
        torch.mm(x, weight1, out=output1)
        return output1

    parent.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(parent), with_multi_stream(True):
        maybe_execute_in_parallel_on_streams(
            fn0,
            fn1,
            fork_event,
            done0,
            done1,
            stream0,
            stream1,
        )
    torch.cuda.current_stream().wait_stream(parent)
    torch.cuda.synchronize()

    assert branch_streams[-2:] == [stream0.cuda_stream, stream1.cuda_stream]
    torch.testing.assert_close(output0, x @ weight0)
    torch.testing.assert_close(output1, x @ weight1)

    graph = torch.cuda.CUDAGraph()
    branch_streams.clear()
    with torch.cuda.graph(graph, stream=parent), with_multi_stream(True):
        maybe_execute_in_parallel_on_streams(
            fn0,
            fn1,
            fork_event,
            done0,
            done1,
            stream0,
            stream1,
        )

    assert branch_streams == [stream0.cuda_stream, stream1.cuda_stream]
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output0, x @ weight0)
    torch.testing.assert_close(output1, x @ weight1)
