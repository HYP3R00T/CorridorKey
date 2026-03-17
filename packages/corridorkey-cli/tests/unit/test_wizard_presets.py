"""Unit tests for wizard engine preset mapping."""

from __future__ import annotations

import pytest
from corridorkey_cli.commands.wizard import _resolve_engine_preset


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("speed", ("auto", "speed", "fp16", 1024)),
        ("balanced", ("auto", "auto", "auto", 1536)),
        ("quality", ("auto", "speed", "bf16", 2048)),
        ("max_quality", ("auto", "speed", "fp32", 2560)),
        ("lowvram", ("auto", "lowvram", "fp16", 1024)),
    ],
)
def test_resolve_engine_preset_defaults(preset: str, expected: tuple[str, str, str, int | None]) -> None:
    assert _resolve_engine_preset(preset) == expected


def test_resolve_engine_preset_uses_device_override() -> None:
    device, opt_mode, precision, img_size = _resolve_engine_preset("quality", default_device="cuda")
    assert device == "cuda"
    assert opt_mode == "speed"
    assert precision == "bf16"
    assert img_size == 2048


def test_resolve_engine_preset_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        _resolve_engine_preset("manual")
