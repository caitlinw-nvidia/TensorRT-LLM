# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA green-context stream ownership for GDN/GEMM overlap."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from cuda.core import Context, SMResource, Stream

GREEN_CONTEXT_ENABLE_ENV = "TRTLLM_ENABLE_GDN_GREEN_CONTEXT"
GDN_SM_COUNT_ENV = "TRTLLM_GDN_GREEN_CONTEXT_SM_COUNT"
GDN_SM_RATIO_MULTIPLIER_ENV = "TRTLLM_GDN_GREEN_CONTEXT_RATIO_MULTIPLIER"
GDN_GREEN_CONTEXT_POOL_ATTR = "gdn_green_context_pool"


def green_context_enabled() -> bool:
    """Return whether the experimental GDN green-context pair is enabled."""
    return os.environ.get(GREEN_CONTEXT_ENABLE_ENV, "0").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def get_requested_gdn_sm_count() -> Optional[int]:
    """Return the optional explicit GDN SM count from the environment."""
    value = os.environ.get(GDN_SM_COUNT_ENV)
    if value is None or value == "":
        return None
    try:
        sm_count = int(value)
    except ValueError as error:
        raise ValueError(f"{GDN_SM_COUNT_ENV} must be a positive integer, got {value!r}") from error
    if sm_count <= 0:
        raise ValueError(f"{GDN_SM_COUNT_ENV} must be a positive integer, got {sm_count}")
    return sm_count


def get_gdn_sm_ratio_multiplier() -> float:
    """Return the optional multiplier applied to the traffic heuristic."""
    value = os.environ.get(GDN_SM_RATIO_MULTIPLIER_ENV, "1.0")
    try:
        multiplier = float(value)
    except ValueError as error:
        raise ValueError(
            f"{GDN_SM_RATIO_MULTIPLIER_ENV} must be a positive number, got {value!r}"
        ) from error
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            f"{GDN_SM_RATIO_MULTIPLIER_ENV} must be a positive finite number, got {value!r}"
        )
    return multiplier


@dataclass(frozen=True)
class GdnGreenContextWorkload:
    """Shape and dtype inputs to the GDN/Z-GEMM traffic heuristic."""

    batch_size: int
    num_tokens: int
    num_v_heads: int
    head_size: int
    hidden_size: int
    key_head_size: Optional[int] = None
    gemm_element_bytes: int = 2
    state_element_bytes: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("num_tokens", self.num_tokens),
            ("num_v_heads", self.num_v_heads),
            ("head_size", self.head_size),
            ("hidden_size", self.hidden_size),
            ("gemm_element_bytes", self.gemm_element_bytes),
            ("state_element_bytes", self.state_element_bytes),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.key_head_size is not None and self.key_head_size <= 0:
            raise ValueError(f"key_head_size must be positive, got {self.key_head_size}")


@dataclass(frozen=True)
class GdnGreenContextDecision:
    """Result of mapping one workload to a legal green-context split."""

    gdn_sm_count: int
    estimated_gemm_bytes: int
    estimated_gdn_bytes: int
    gdn_to_gemm_ratio: float
    ratio_multiplier: float


def calculate_gdn_green_context_decision(
    workload: GdnGreenContextWorkload,
    total_sm_count: int,
    minimum_sm_count: int,
    alignment_sm_count: int,
    ratio_multiplier: float = 1.0,
) -> GdnGreenContextDecision:
    """Choose an SM split using the green-context sweep's traffic model.

    The independent Z projection is modeled as a memory-bound ``M x K`` by
    ``K x N`` GEMM, where ``M`` is the number of input tokens and
    ``N = num_v_heads * head_size``. GDN is modeled as one read and one write
    of its recurrent ``[batch, heads, value_head_size, key_head_size]`` state.
    Their byte ratio is converted to the same resource ratio and rounded down
    to a legal CGA2-aligned partition.
    """
    if total_sm_count <= 0:
        raise ValueError(f"total_sm_count must be positive, got {total_sm_count}")
    if minimum_sm_count <= 0:
        raise ValueError(f"minimum_sm_count must be positive, got {minimum_sm_count}")
    if alignment_sm_count <= 0:
        raise ValueError(f"alignment_sm_count must be positive, got {alignment_sm_count}")
    if not math.isfinite(ratio_multiplier) or ratio_multiplier <= 0.0:
        raise ValueError(
            f"ratio_multiplier must be a positive finite number, got {ratio_multiplier}"
        )

    gemm_m = workload.num_tokens
    gemm_n = workload.num_v_heads * workload.head_size
    gemm_k = workload.hidden_size
    estimated_gemm_bytes = (
        gemm_m * gemm_n + gemm_n * gemm_k + gemm_m * gemm_k
    ) * workload.gemm_element_bytes
    estimated_gdn_bytes = (
        workload.batch_size
        * workload.num_v_heads
        * workload.head_size
        * (workload.key_head_size or workload.head_size)
        * workload.state_element_bytes
        * 2
    )
    gdn_to_gemm_ratio = estimated_gdn_bytes / estimated_gemm_bytes
    weighted_ratio = gdn_to_gemm_ratio * ratio_multiplier

    # Retain the sweep's four-SM rounding while also honoring any stricter
    # driver-reported co-scheduling alignment.
    legal_alignment = math.lcm(4, alignment_sm_count)
    minimum_gdn = ((minimum_sm_count + legal_alignment - 1) // legal_alignment) * legal_alignment
    maximum_gdn = ((total_sm_count - minimum_sm_count) // legal_alignment) * legal_alignment
    if minimum_gdn > maximum_gdn:
        raise RuntimeError(
            "The device cannot form two legal green-context partitions: "
            f"device={total_sm_count}, minimum={minimum_sm_count}, "
            f"alignment={legal_alignment}"
        )

    requested_gdn = int(total_sm_count * weighted_ratio / (1.0 + weighted_ratio))
    requested_gdn = requested_gdn // legal_alignment * legal_alignment
    gdn_sm_count = min(max(requested_gdn, minimum_gdn), maximum_gdn)
    return GdnGreenContextDecision(
        gdn_sm_count=gdn_sm_count,
        estimated_gemm_bytes=estimated_gemm_bytes,
        estimated_gdn_bytes=estimated_gdn_bytes,
        gdn_to_gemm_ratio=gdn_to_gemm_ratio,
        ratio_multiplier=ratio_multiplier,
    )


@dataclass(frozen=True)
class GreenContextInfo:
    """Driver-reported metadata for the GDN and GEMM partitions."""

    device_index: int
    device_sm_count: int
    stream_sm_counts: tuple[int, int]
    remainder_sm_count: int
    minimum_partition_sm_count: int
    coscheduled_alignment_sm_count: int


class GreenContextPair:
    """Own disjoint GDN/GEMM green contexts and their streams.

    The first context receives the requested GDN SM count. The second context
    receives the largest valid structured group from the remaining SMs. The
    ordinary PyExecutor execution stream is not replaced and can therefore use
    the full device after it has joined both green streams.
    """

    def __init__(
        self,
        contexts: tuple[Context, Context],
        core_streams: tuple[Stream, Stream],
        streams: tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream],
        resources: tuple[SMResource, SMResource],
        remainder: Optional[SMResource],
        info: GreenContextInfo,
    ) -> None:
        self._contexts = contexts
        self._core_streams = core_streams
        self._streams = streams
        self._resources = resources
        self._remainder = remainder
        self.info = info
        self._closed = False

    @property
    def streams(
        self,
    ) -> tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream]:
        """Return the GDN and GEMM PyTorch stream wrappers, in that order."""
        if self._closed:
            raise RuntimeError("Green-context streams have already been closed")
        return self._streams

    @staticmethod
    def _default_gdn_sm_count(total_sm_count: int, minimum_sm_count: int) -> int:
        # Start with the 25% carve-out used by the GDN overlap experiment and
        # retain four-SM granularity for its CGA2 configuration.
        alignment = 4
        requested = total_sm_count // 4 // alignment * alignment
        return max(requested, minimum_sm_count)

    @classmethod
    def create(
        cls,
        device_index: Optional[int] = None,
        gdn_sm_count: Optional[int] = None,
    ) -> "GreenContextPair":
        """Create disjoint GDN and GEMM green-context streams."""
        from cuda.core import ContextOptions, Device, SMResourceOptions, StreamOptions

        if device_index is None:
            device_index = torch.cuda.current_device()
        if gdn_sm_count is None:
            gdn_sm_count = get_requested_gdn_sm_count()

        torch.cuda.set_device(device_index)
        torch.cuda.init()
        device = Device(device_index)
        device.set_current()
        sm_resource = device.resources.sm
        total_sm_count = sm_resource.sm_count
        minimum_sm_count = sm_resource.min_partition_size
        if gdn_sm_count is None:
            gdn_sm_count = cls._default_gdn_sm_count(total_sm_count, minimum_sm_count)
        if gdn_sm_count < minimum_sm_count:
            raise RuntimeError(
                "The requested GDN green context is smaller than the device "
                f"minimum: requested={gdn_sm_count}, minimum={minimum_sm_count}"
            )
        if total_sm_count - gdn_sm_count < minimum_sm_count:
            raise RuntimeError(
                "The requested GDN green context leaves too few SMs for GEMM: "
                f"device={total_sm_count}, requested={gdn_sm_count}, "
                f"minimum_remainder={minimum_sm_count}"
            )

        groups, remainder = sm_resource.split(
            SMResourceOptions(
                count=(gdn_sm_count, 0),
                coscheduled_sm_count=(2, 2),
            )
        )
        if len(groups) != 2:
            raise RuntimeError(
                "CUDA did not create the requested GDN/GEMM resource pair: "
                f"group_count={len(groups)}"
            )

        contexts: list[Context] = []
        core_streams: list[Stream] = []
        try:
            for group in groups:
                context = device.create_context(ContextOptions(resources=[group]))
                contexts.append(context)
                core_streams.append(context.create_stream(StreamOptions(nonblocking=True)))

            streams = (
                torch.cuda.ExternalStream(int(core_streams[0].handle), device=device_index),
                torch.cuda.ExternalStream(int(core_streams[1].handle), device=device_index),
            )
            stream_sm_counts = (
                core_streams[0].resources.sm.sm_count,
                core_streams[1].resources.sm.sm_count,
            )
            if stream_sm_counts != tuple(group.sm_count for group in groups):
                raise RuntimeError(
                    "Green-context stream resources do not match their SM groups: "
                    f"streams={stream_sm_counts}, "
                    f"groups={tuple(group.sm_count for group in groups)}"
                )

            info = GreenContextInfo(
                device_index=device_index,
                device_sm_count=total_sm_count,
                stream_sm_counts=stream_sm_counts,
                remainder_sm_count=(remainder.sm_count if remainder is not None else 0),
                minimum_partition_sm_count=minimum_sm_count,
                coscheduled_alignment_sm_count=sm_resource.coscheduled_alignment,
            )
            return cls(
                contexts=(contexts[0], contexts[1]),
                core_streams=(core_streams[0], core_streams[1]),
                streams=streams,
                resources=(groups[0], groups[1]),
                remainder=remainder,
                info=info,
            )
        except (RuntimeError, TypeError, ValueError):
            for stream in reversed(core_streams):
                stream.close()
            for context in reversed(contexts):
                context.close()
            raise

    def close(self) -> None:
        """Destroy streams before releasing their owning green contexts."""
        if self._closed:
            return
        for stream in self._streams:
            stream.synchronize()
        for stream in reversed(self._core_streams):
            stream.close()
        for context in reversed(self._contexts):
            context.close()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except (RuntimeError, AttributeError):
            # CUDA may already be unloaded during interpreter shutdown.
            pass


class GreenContextPool:
    """Cache capture-stable green-context pairs by heuristic SM count.

    CUDA graph replay must use the same streams that were present during
    capture. A graph warmup creates the pair selected for that shape, capture
    reuses it, and the pool keeps every selected pair alive until executor
    shutdown. Workloads that round to the same GDN SM count share one pair.
    """

    def __init__(self, device_index: Optional[int] = None) -> None:
        from cuda.core import Device

        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.set_device(device_index)
        torch.cuda.init()
        device = Device(device_index)
        device.set_current()
        sm_resource = device.resources.sm

        self.device_index = device_index
        self.device_sm_count = sm_resource.sm_count
        self.minimum_partition_sm_count = sm_resource.min_partition_size
        self.alignment_sm_count = math.lcm(4, sm_resource.coscheduled_alignment)
        self._requested_gdn_sm_count = get_requested_gdn_sm_count()
        self._ratio_multiplier = get_gdn_sm_ratio_multiplier()
        self._pairs: dict[int, GreenContextPair] = {}
        self._closed = False

    def decision_for(self, workload: GdnGreenContextWorkload) -> GdnGreenContextDecision:
        """Return the effective heuristic or explicitly overridden decision."""
        decision = calculate_gdn_green_context_decision(
            workload,
            total_sm_count=self.device_sm_count,
            minimum_sm_count=self.minimum_partition_sm_count,
            alignment_sm_count=self.alignment_sm_count,
            ratio_multiplier=self._ratio_multiplier,
        )
        if self._requested_gdn_sm_count is None:
            return decision
        return GdnGreenContextDecision(
            gdn_sm_count=self._requested_gdn_sm_count,
            estimated_gemm_bytes=decision.estimated_gemm_bytes,
            estimated_gdn_bytes=decision.estimated_gdn_bytes,
            gdn_to_gemm_ratio=decision.gdn_to_gemm_ratio,
            ratio_multiplier=decision.ratio_multiplier,
        )

    def get_pair(self, workload: GdnGreenContextWorkload) -> GreenContextPair:
        """Return a cached pair, creating it during graph warmup if needed."""
        if self._closed:
            raise RuntimeError("The green-context pool has already been closed")
        decision = self.decision_for(workload)
        pair = self._pairs.get(decision.gdn_sm_count)
        if pair is not None:
            return pair
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "A GDN green-context split was first requested during CUDA "
                "graph capture. Run the graph warmup for this batch shape "
                "before capture."
            )

        pair = GreenContextPair.create(
            device_index=self.device_index,
            gdn_sm_count=decision.gdn_sm_count,
        )
        if pair.info.stream_sm_counts[0] != decision.gdn_sm_count:
            pair.close()
            raise RuntimeError(
                "CUDA created a different GDN partition than the heuristic "
                f"requested: requested={decision.gdn_sm_count}, "
                f"actual={pair.info.stream_sm_counts[0]}"
            )
        self._pairs[decision.gdn_sm_count] = pair
        logger.info(
            "[GDN green context] selected split for workload "
            f"batch={workload.batch_size}, tokens={workload.num_tokens}, "
            f"heads={workload.num_v_heads}, value_head_size={workload.head_size}, "
            f"key_head_size={workload.key_head_size or workload.head_size}, "
            f"hidden={workload.hidden_size}: "
            f"gdn/gemm_sms={pair.info.stream_sm_counts}, "
            f"traffic_ratio={decision.gdn_to_gemm_ratio:.4f}, "
            f"ratio_multiplier={decision.ratio_multiplier:.3f}, "
            f"remainder_sms={pair.info.remainder_sm_count}."
        )
        return pair

    def get_streams(
        self, workload: GdnGreenContextWorkload
    ) -> tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream]:
        """Return the cached GDN and GEMM streams for ``workload``."""
        return self.get_pair(workload).streams

    @property
    def cached_sm_counts(self) -> tuple[int, ...]:
        """Return GDN partition sizes currently retained by the pool."""
        return tuple(self._pairs)

    def close(self) -> None:
        """Destroy every stream pair after CUDA graphs have been released."""
        if self._closed:
            return
        for pair in reversed(tuple(self._pairs.values())):
            pair.close()
        self._pairs.clear()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except (RuntimeError, AttributeError):
            # CUDA may already be unloaded during interpreter shutdown.
            pass
