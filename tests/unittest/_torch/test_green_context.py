# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm._torch.green_context import GREEN_CONTEXT_ENABLE_ENV, green_context_enabled


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
