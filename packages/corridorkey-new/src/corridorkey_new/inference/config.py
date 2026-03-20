"""Inference stage — configuration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

OptimizationMode = Literal["auto", "speed", "lowvram"]

# VRAM threshold below which lowvram mode is selected automatically.
_VRAM_LOWVRAM_THRESHOLD_GB = 12.0

# Tiled refiner defaults (lowvram mode).
REFINER_TILE_SIZE = 512
REFINER_TILE_OVERLAP = 128


@dataclass
class InferenceConfig:
    """Configuration for the inference stage.

    Attributes:
        checkpoint_path: Path to the .pth model checkpoint file.
        device: PyTorch device string ("cuda", "cuda:0", "mps", "cpu").
        img_size: Square resolution the model runs at. Must match the
            resolution the checkpoint was trained at (default 2048).
        use_refiner: Whether to enable the CNN refiner module.
        mixed_precision: Run the forward pass under fp16 autocast.
            Ignored on CPU (autocast is a no-op there).
        model_precision: Weight dtype. float32 is safe everywhere;
            float16 saves VRAM on CUDA but may reduce accuracy.
        optimization_mode: Refiner execution strategy.
            "auto"    — probe VRAM; < 12 GB → lowvram, else → speed.
            "speed"   — full-frame refiner pass.
            "lowvram" — tiled refiner (512×512, 128px overlap).
    """

    checkpoint_path: Path
    device: str = "cpu"
    img_size: int = 2048
    use_refiner: bool = True
    mixed_precision: bool = True
    model_precision: torch.dtype = field(default=torch.float32)
    optimization_mode: OptimizationMode = "auto"
