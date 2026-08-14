# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm._torch.green_context import (
    GDN_SM_COUNT_ENV,
    GDN_SM_RATIO_MULTIPLIER_ENV,
    GREEN_CONTEXT_ENABLE_ENV,
    GdnGreenContextWorkload,
    GreenContextPair,
    calculate_gdn_green_context_decision,
    get_gdn_sm_ratio_multiplier,
    get_requested_gdn_sm_count,
    green_context_enabled,
)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Y"])
def test_green_context_enabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(GREEN_CONTEXT_ENABLE_ENV, value)
    assert green_context_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "", "unexpected"])
def test_green_context_disabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(GREEN_CONTEXT_ENABLE_ENV, value)
    assert not green_context_enabled()


def test_green_context_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GREEN_CONTEXT_ENABLE_ENV, raising=False)
    assert not green_context_enabled()


def test_requested_gdn_sm_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GDN_SM_COUNT_ENV, "36")
    assert get_requested_gdn_sm_count() == 36


@pytest.mark.parametrize("value", ["0", "-4", "not-an-integer"])
def test_invalid_requested_gdn_sm_count(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(GDN_SM_COUNT_ENV, value)
    with pytest.raises(ValueError, match=GDN_SM_COUNT_ENV):
        get_requested_gdn_sm_count()


def test_default_gdn_sm_count_uses_quarter_of_device() -> None:
    assert GreenContextPair._default_gdn_sm_count(148, 8) == 36
    assert GreenContextPair._default_gdn_sm_count(24, 8) == 8


def test_gdn_sm_ratio_multiplier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GDN_SM_RATIO_MULTIPLIER_ENV, "1.5")
    assert get_gdn_sm_ratio_multiplier() == 1.5


@pytest.mark.parametrize("value", ["0", "-1", "nan", "not-a-number"])
def test_invalid_gdn_sm_ratio_multiplier(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(GDN_SM_RATIO_MULTIPLIER_ENV, value)
    with pytest.raises(ValueError, match=GDN_SM_RATIO_MULTIPLIER_ENV):
        get_gdn_sm_ratio_multiplier()


@pytest.mark.parametrize(
    ("batch_size", "expected_gdn_sms"),
    [(1, 12), (2, 20), (4, 40), (8, 68), (16, 100)],
)
def test_green_context_traffic_heuristic_tracks_decode_batch(
    batch_size: int, expected_gdn_sms: int
) -> None:
    decision = calculate_gdn_green_context_decision(
        GdnGreenContextWorkload(
            batch_size=batch_size,
            num_tokens=batch_size,
            num_v_heads=64,
            head_size=128,
            hidden_size=4096,
        ),
        total_sm_count=208,
        minimum_sm_count=8,
        alignment_sm_count=2,
    )
    assert decision.gdn_sm_count == expected_gdn_sms
    assert decision.estimated_gdn_bytes > 0
    assert decision.estimated_gemm_bytes > 0


def test_green_context_traffic_heuristic_applies_ratio_multiplier() -> None:
    workload = GdnGreenContextWorkload(
        batch_size=1,
        num_tokens=1,
        num_v_heads=64,
        head_size=128,
        hidden_size=4096,
    )
    baseline = calculate_gdn_green_context_decision(workload, 208, 8, 2)
    weighted = calculate_gdn_green_context_decision(workload, 208, 8, 2, ratio_multiplier=2.0)
    assert baseline.gdn_sm_count == 12
    assert weighted.gdn_sm_count == 20
