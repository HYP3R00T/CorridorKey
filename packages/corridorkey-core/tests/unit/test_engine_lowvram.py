"""Unit tests for lowvram tiled-refiner hook wiring in CorridorKeyEngine."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import torch
from corridorkey_core.engine import CorridorKeyEngine


class _DummyRefiner(torch.nn.Module):
    def forward(self, img: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        # The real refiner outputs [B, 4, H, W] delta logits.
        return torch.zeros_like(coarse_pred)


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.refiner = _DummyRefiner()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        alpha_logits = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
        fg_logits = torch.zeros((b, 3, h, w), device=x.device, dtype=x.dtype)

        alpha_coarse = torch.sigmoid(alpha_logits)
        fg_coarse = torch.sigmoid(fg_logits)

        rgb = x[:, :3, :, :]
        coarse = torch.cat([alpha_coarse, fg_coarse], dim=1)
        delta = self.refiner(rgb.float(), coarse.float()).to(x.dtype)

        alpha = torch.sigmoid(alpha_logits + delta[:, 0:1])
        fg = torch.sigmoid(fg_logits + delta[:, 1:4])
        return {"alpha": alpha, "fg": fg}


def _make_engine(img_size: int = 32) -> CorridorKeyEngine:
    engine = CorridorKeyEngine.__new__(CorridorKeyEngine)
    engine.device = torch.device("cpu")
    engine.img_size = img_size
    engine.model_precision = torch.float32
    engine.mixed_precision = False
    engine.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    engine.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    engine._checkerboard_cache = {}
    engine._bypass_tiled_refiner_hook = False
    engine._tile_refiner = True
    engine.model = _DummyModel()
    return engine


class TestLowVramRefinerHook:
    """Ensure tiled-refiner hook receives both refiner inputs correctly."""

    def test_hook_passes_4ch_coarse_to_tiled_refiner(self):
        """The lowvram hook must pass [B,4,H,W] coarse_pred to _run_refiner_tiled."""
        engine = _make_engine(32)
        spy = MagicMock(return_value=torch.zeros((1, 4, 32, 32), dtype=torch.float32))
        cast(Any, engine)._run_refiner_tiled = spy

        image = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.random.rand(32, 32).astype(np.float32)

        engine.process_frame(
            image,
            mask,
            auto_despeckle=False,
            despill_strength=0.0,
            source_passthrough=False,
        )

        assert spy.call_count == 1
        rgb_arg, coarse_arg = spy.call_args[0]
        assert rgb_arg.shape == (1, 3, 32, 32)
        assert coarse_arg.shape == (1, 4, 32, 32)

    def test_lowvram_hook_does_not_recurse_when_tiling_calls_refiner(self):
        """Lowvram hook must bypass itself during internal tiled-refiner calls."""
        engine = _make_engine(32)

        image = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.random.rand(32, 32).astype(np.float32)

        result = engine.process_frame(
            image,
            mask,
            auto_despeckle=False,
            despill_strength=0.0,
            source_passthrough=False,
        )

        assert result["alpha"].shape == (32, 32, 1)
        assert result["fg"].shape == (32, 32, 3)
