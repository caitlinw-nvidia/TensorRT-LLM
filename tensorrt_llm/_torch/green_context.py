# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA green-context stream ownership for the PyTorch executor."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


GREEN_CONTEXT_ENABLE_ENV = "TRTLLM_ENABLE_GREEN_CONTEXT"
GREEN_CONTEXT_INFO_ATTR = "green_context_info"
GREEN_CONTEXT_STREAMS_ATTR = "green_context_streams"


def _check_cuda(result: tuple[Any, ...], operation: str) -> Any:
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with CUDA driver status {status}")
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return tuple(values)


def green_context_enabled() -> bool:
    """Return whether experimental PyExecutor green contexts are enabled."""
    return os.environ.get(GREEN_CONTEXT_ENABLE_ENV, "0").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


@dataclass(frozen=True)
class GreenContextInfo:
    """Driver-reported metadata for a pair of green-context streams."""

    device_index: int
    device_sm_count: int
    stream_sm_counts: tuple[int, int]
    remainder_sm_count: int
    minimum_partition_sm_count: int
    coscheduled_alignment_sm_count: int
    context_ids: tuple[int, int]
    creation_time_ms: float


class GreenContextPair:
    """Own two equal SM partitions, their green contexts, and their streams.

    The owner must outlive all eager launches and captured CUDA graphs that use
    :attr:`streams`. Call :meth:`close` only after those graphs have been reset
    and all device work has completed.
    """

    def __init__(
        self,
        contexts: tuple[Any, Any],
        driver_streams: tuple[Any, Any],
        streams: tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream],
        resources: tuple[Any, Any],
        remainder: Any,
        info: GreenContextInfo,
    ) -> None:
        self._contexts = contexts
        self._driver_streams = driver_streams
        self._streams = streams
        self._resources = resources
        self._remainder = remainder
        self.info = info
        self._closed = False

    @property
    def streams(
        self,
    ) -> tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream]:
        """Return the two PyTorch wrappers around the green-context streams."""
        if self._closed:
            raise RuntimeError("Green-context streams have already been closed")
        return self._streams

    def execute_in_parallel(
        self,
        fn0: Callable[[], Any],
        fn1: Callable[[], Any],
        fork_event: torch.cuda.Event,
        done_events: tuple[torch.cuda.Event, torch.cuda.Event],
    ) -> tuple[Any, Any]:
        """Fork the current stream onto both green streams, then join it.

        Callers own the events so model layers can allocate them once during
        initialization. Events must be materialized before CUDA graph capture,
        following the same rule as TRT-LLM's existing multi-stream helpers.
        """
        parent_stream = torch.cuda.current_stream(self.info.device_index)
        fork_event.record(parent_stream)

        results = []
        for fn, stream, done_event in zip((fn0, fn1), self.streams, done_events, strict=True):
            with torch.cuda.stream(stream):
                stream.wait_event(fork_event)
                results.append(fn())
                done_event.record(stream)

        for done_event in done_events:
            parent_stream.wait_event(done_event)
        return results[0], results[1]

    @classmethod
    def create(cls, device_index: Optional[int] = None) -> "GreenContextPair":
        """Create two equal, disjoint green-context SM partitions.

        Any SMs that cannot be split equally at the device's co-scheduling
        alignment remain unassigned. For example, a 148-SM B200 produces two
        72-SM streams and a four-SM remainder.
        """
        if device_index is None:
            device_index = torch.cuda.current_device()

        torch.cuda.set_device(device_index)
        torch.cuda.init()
        started_ns = time.perf_counter_ns()

        _check_cuda(cuda.cuInit(0), "cuInit")
        device = _check_cuda(cuda.cuDeviceGet(device_index), "cuDeviceGet")
        all_sms = _check_cuda(
            cuda.cuDeviceGetDevResource(device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
            "cuDeviceGetDevResource",
        )
        total_sm_count = all_sms.sm.smCount
        minimum_partition_sm_count = all_sms.sm.minSmPartitionSize
        alignment_sm_count = all_sms.sm.smCoscheduledAlignment
        requested_sm_count = total_sm_count // 2 // alignment_sm_count * alignment_sm_count
        if requested_sm_count < minimum_partition_sm_count:
            raise RuntimeError(
                "The device does not have enough SMs for two green contexts: "
                f"device={total_sm_count}, minimum_partition="
                f"{minimum_partition_sm_count}, alignment={alignment_sm_count}"
            )

        groups, group_count, remainder = _check_cuda(
            cuda.cuDevSmResourceSplitByCount(
                2,
                all_sms,
                0,
                requested_sm_count,
            ),
            "cuDevSmResourceSplitByCount",
        )
        if group_count != 2 or len(groups) != 2:
            raise RuntimeError(
                "CUDA did not create the requested two green-context SM groups: "
                f"group_count={group_count}, groups={len(groups)}"
            )

        contexts = []
        driver_streams = []
        try:
            for group in groups:
                descriptor = _check_cuda(
                    cuda.cuDevResourceGenerateDesc([group], 1),
                    "cuDevResourceGenerateDesc",
                )
                context = _check_cuda(
                    cuda.cuGreenCtxCreate(
                        descriptor,
                        device,
                        cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
                    ),
                    "cuGreenCtxCreate",
                )
                contexts.append(context)
                stream = _check_cuda(
                    cuda.cuGreenCtxStreamCreate(
                        context,
                        cuda.CUstream_flags.CU_STREAM_NON_BLOCKING,
                        0,
                    ),
                    "cuGreenCtxStreamCreate",
                )
                driver_streams.append(stream)

            streams = (
                torch.cuda.ExternalStream(int(driver_streams[0]), device=device_index),
                torch.cuda.ExternalStream(int(driver_streams[1]), device=device_index),
            )
            context_ids = (
                int(_check_cuda(cuda.cuGreenCtxGetId(contexts[0]), "cuGreenCtxGetId")),
                int(_check_cuda(cuda.cuGreenCtxGetId(contexts[1]), "cuGreenCtxGetId")),
            )
            stream_sm_counts = (
                _check_cuda(
                    cuda.cuStreamGetDevResource(
                        driver_streams[0],
                        cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
                    ),
                    "cuStreamGetDevResource",
                ).sm.smCount,
                _check_cuda(
                    cuda.cuStreamGetDevResource(
                        driver_streams[1],
                        cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
                    ),
                    "cuStreamGetDevResource",
                ).sm.smCount,
            )
            if len(set(context_ids)) != 2:
                raise RuntimeError(
                    "CUDA returned the same ID for two separately created green contexts"
                )
            if stream_sm_counts != tuple(group.sm.smCount for group in groups):
                raise RuntimeError(
                    "Green-context stream resources do not match their SM groups: "
                    f"streams={stream_sm_counts}, "
                    f"groups={tuple(group.sm.smCount for group in groups)}"
                )

            remainder_sm_count = (
                remainder.sm.smCount
                if remainder.type == cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
                else 0
            )
            info = GreenContextInfo(
                device_index=device_index,
                device_sm_count=total_sm_count,
                stream_sm_counts=stream_sm_counts,
                remainder_sm_count=remainder_sm_count,
                minimum_partition_sm_count=minimum_partition_sm_count,
                coscheduled_alignment_sm_count=alignment_sm_count,
                context_ids=context_ids,
                creation_time_ms=(time.perf_counter_ns() - started_ns) / 1.0e6,
            )
            return cls(
                contexts=tuple(contexts),
                driver_streams=tuple(driver_streams),
                streams=streams,
                resources=tuple(groups),
                remainder=remainder,
                info=info,
            )
        except (RuntimeError, TypeError, ValueError):
            for stream in reversed(driver_streams):
                _check_cuda(cuda.cuStreamDestroy(stream), "cuStreamDestroy")
            for context in reversed(contexts):
                _check_cuda(cuda.cuGreenCtxDestroy(context), "cuGreenCtxDestroy")
            raise

    def close(self) -> None:
        """Destroy streams first, followed by their owning green contexts."""
        if self._closed:
            return

        for stream in self._streams:
            stream.synchronize()
        for stream in reversed(self._driver_streams):
            _check_cuda(cuda.cuStreamDestroy(stream), "cuStreamDestroy")
        for context in reversed(self._contexts):
            _check_cuda(cuda.cuGreenCtxDestroy(context), "cuGreenCtxDestroy")
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except (RuntimeError, AttributeError):
            # Interpreter shutdown may unload CUDA before this object is
            # collected. Deterministic executor shutdown calls close directly.
            pass
