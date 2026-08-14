import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch

GDN_STREAM_ATTR = "gdn_stream"
GEMM_STREAM_ATTR = "gemm_stream"


class do_multi_stream_local(threading.local):

    def __init__(self):
        self.do_multi_stream = False


_local = do_multi_stream_local()


def set_do_multi_stream(enable: bool):
    _local.do_multi_stream = enable


def do_multi_stream() -> bool:
    return _local.do_multi_stream


@contextmanager
def with_multi_stream(enable: bool):
    prev_do_multi_stream = _local.do_multi_stream
    set_do_multi_stream(enable)
    try:
        yield
    finally:
        set_do_multi_stream(prev_do_multi_stream)


def maybe_execute_in_parallel(
        fn0: Callable,
        fn1: Callable,
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        aux_stream: Optional[torch.cuda.Stream] = None,
        disable_on_compile: bool = False) -> tuple[Any, Any]:
    """Utility function to run two functions in two cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0 and fn1 do not support inputs.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream
        event0 (torch.cuda.Event): cuda event for fn0
        event1 (torch.cuda.Event): cuda event for fn1
        aux_stream (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
            Multi-stream is disabled when aux_stream is None.
        disable_on_compile (bool): if True, disable multi-stream when
            torch.compile is tracing. Callers that are not inside a custom op
            should set this to True so that stream/event ops are not captured
            by dynamo. Callers inside custom ops (e.g. attention, MoE) should
            leave this as False since custom ops are opaque to the compiler.

    Returns:
        tuple[Any, Any]: the return values of fn0() and fn1()
    """

    multi_stream = (do_multi_stream() and aux_stream is not None and
                    not (disable_on_compile and torch.compiler.is_compiling()))

    if multi_stream:
        event0.record()
        result0 = fn0()

        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)


def maybe_execute_in_parallel_on_streams(
        fn0: Callable,
        fn1: Callable,
        fork_event: torch.cuda.Event,
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        stream0: Optional[torch.cuda.Stream] = None,
        stream1: Optional[torch.cuda.Stream] = None,
        disable_on_compile: bool = False) -> tuple[Any, Any]:
    """Run two independent functions on two explicitly supplied CUDA streams.

    The current stream is the parent stream. It records ``fork_event`` before
    either branch starts and waits for both completion events before returning.
    The fork/join is safe to capture in a CUDA graph with ordinary streams or
    green-context streams from the same device.

    As with :func:`maybe_execute_in_parallel`, eager execution remains
    sequential unless ``with_multi_stream(True)`` is active. TRT-LLM enables
    that context during CUDA graph warmup and capture.
    """

    multi_stream = (do_multi_stream() and stream0 is not None
                    and stream1 is not None and
                    not (disable_on_compile and torch.compiler.is_compiling()))

    if not multi_stream:
        return fn0(), fn1()

    parent_stream = torch.cuda.current_stream()
    fork_event.record(parent_stream)
    with torch.cuda.stream(stream0):
        fork_event.wait()
        result0 = fn0()
        event0.record()

    with torch.cuda.stream(stream1):
        fork_event.wait()
        result1 = fn1()
        event1.record()

    parent_stream.wait_event(event0)
    parent_stream.wait_event(event1)
    return result0, result1
