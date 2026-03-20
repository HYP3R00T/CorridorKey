"""Inference stage — orchestrator.

Public entry point: ``run_inference(frame, model, config) -> InferenceResult``.

Owns:
  - fp16 autocast
  - tiled refiner (lowvram mode)
  - VRAM probing for "auto" optimization mode
  - converting raw model output to InferenceResult

Does NOT own:
  - model loading (loader.py)
  - despeckle / despill / compositing (postprocessor stage)
  - writing to disk (writer stage)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from corridorkey_new.inference.config import (
    _VRAM_LOWVRAM_THRESHOLD_GB,
    REFINER_TILE_OVERLAP,
    REFINER_TILE_SIZE,
    InferenceConfig,
)
from corridorkey_new.inference.contracts import InferenceResult
from corridorkey_new.preprocessor.orchestrator import PreprocessedFrame

logger = logging.getLogger(__name__)


def run_inference(
    frame: PreprocessedFrame,
    model: nn.Module,
    config: InferenceConfig,
) -> InferenceResult:
    """Run model inference on a single preprocessed frame.

    Args:
        frame: Output of the preprocessing stage. tensor is [1, 4, H, W]
            on config.device, already ImageNet-normalised.
        model: Loaded GreenFormer in eval mode.
        config: Inference configuration.

    Returns:
        InferenceResult with alpha and fg tensors on device, plus FrameMeta.
    """
    tile_refiner = _should_tile_refiner(config)
    device_type = torch.device(config.device).type

    hook_handle = None
    if tile_refiner:
        refiner = getattr(model, "refiner", None)
        if refiner is not None:
            hook_handle = refiner.register_forward_hook(_make_tiled_refiner_hook(frame.tensor, config))

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=config.mixed_precision and device_type != "cpu",
        ),
    ):
        output = model(frame.tensor)

    if hook_handle is not None:
        hook_handle.remove()

    return InferenceResult(
        alpha=output["alpha"],
        fg=output["fg"],
        meta=frame.meta,
    )


# ---------------------------------------------------------------------------
# Optimization mode resolution
# ---------------------------------------------------------------------------


def _should_tile_refiner(config: InferenceConfig) -> bool:
    """Return True if the refiner should run in tiled mode."""
    if not config.use_refiner:
        return False
    mode = config.optimization_mode
    if mode == "lowvram":
        return True
    if mode == "speed":
        return False
    # auto — probe VRAM
    vram_gb = _probe_vram_gb(config.device)
    if vram_gb > 0 and vram_gb < _VRAM_LOWVRAM_THRESHOLD_GB:
        logger.info("Auto mode: %.1f GB VRAM detected — using tiled refiner", vram_gb)
        return True
    logger.info("Auto mode: %.1f GB VRAM detected — using full-frame refiner", vram_gb)
    return False


def _probe_vram_gb(device: str) -> float:
    """Return total VRAM in GB for the given device. Returns 0.0 on failure."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.total / (1024**3)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Tiled refiner hook
# ---------------------------------------------------------------------------


def _make_tiled_refiner_hook(tensor: torch.Tensor, config: InferenceConfig):
    """Return a forward hook that replaces the refiner with a tiled pass."""

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if len(inputs) != 2:
            raise RuntimeError(f"Tiled refiner hook expected 2 inputs (rgb, coarse_pred), got {len(inputs)}")
        rgb, coarse = inputs
        return _run_refiner_tiled(module, rgb, coarse, config)

    return hook


def _run_refiner_tiled(
    refiner: nn.Module,
    rgb: torch.Tensor,
    coarse: torch.Tensor,
    config: InferenceConfig,
) -> torch.Tensor:
    """Run the CNN refiner in overlapping tiles with cosine-weighted blending.

    Splits the [1, C, H, W] inputs into tiles of REFINER_TILE_SIZE with
    REFINER_TILE_OVERLAP overlap, runs the refiner on each tile, and blends
    the results back using a cosine weight window.

    Args:
        refiner: The CNNRefinerModule.
        rgb: [1, 3, H, W] float tensor.
        coarse: [1, 4, H, W] float tensor.
        config: InferenceConfig (for device).

    Returns:
        [1, 4, H, W] delta logits tensor, same dtype as coarse.
    """
    _, _, h, w = rgb.shape
    tile_size = REFINER_TILE_SIZE
    overlap = REFINER_TILE_OVERLAP
    stride = tile_size - overlap

    output = torch.zeros_like(coarse)
    weight_sum = torch.zeros(1, 1, h, w, device=rgb.device, dtype=torch.float32)

    # 1-D cosine window — smooth blend at tile edges.
    window_1d = torch.hann_window(tile_size, device=rgb.device, dtype=torch.float32)
    window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)  # [tile, tile]
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, tile, tile]

    y_starts = list(range(0, h - tile_size + 1, stride))
    if not y_starts or y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))

    x_starts = list(range(0, w - tile_size + 1, stride))
    if not x_starts or x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))

    for y in y_starts:
        for x in x_starts:
            y2, x2 = y + tile_size, x + tile_size
            rgb_tile = rgb[:, :, y:y2, x:x2].float()
            coarse_tile = coarse[:, :, y:y2, x:x2].float()

            with torch.inference_mode():
                delta_tile = refiner(rgb_tile, coarse_tile)

            w_tile = window_2d.expand_as(delta_tile[:, :1])
            output[:, :, y:y2, x:x2] += (delta_tile * w_tile).to(output.dtype)
            weight_sum[:, :, y:y2, x:x2] += w_tile.to(weight_sum.dtype)

    # Avoid division by zero in any uncovered corner (shouldn't happen).
    weight_sum = weight_sum.clamp(min=1e-6)
    return output / weight_sum.to(output.dtype)
