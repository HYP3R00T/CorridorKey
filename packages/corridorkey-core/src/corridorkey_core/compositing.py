"""Image compositing utilities for the CorridorKey keying pipeline.

Provides color space conversion (linear/sRGB), alpha compositing, green spill
removal, matte cleanup, and preview generation. All functions support both
numpy arrays and PyTorch tensors unless otherwise noted.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import cv2
import numpy as np
import torch


def _is_tensor(x: np.ndarray | torch.Tensor) -> bool:
    return isinstance(x, torch.Tensor)


def _if_tensor(is_tensor: bool, tensor_func: Callable, numpy_func: Callable) -> Callable:
    return tensor_func if is_tensor else numpy_func


def _power(x: np.ndarray | torch.Tensor, exponent: float) -> np.ndarray | torch.Tensor:
    # Dispatches to torch.pow or np.power depending on input type.
    power = _if_tensor(_is_tensor(x), torch.pow, np.power)
    return power(x, exponent)


def _where(
    condition: np.ndarray | torch.Tensor, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    # Dispatches to torch.where or np.where depending on input type.
    where = _if_tensor(_is_tensor(x), torch.where, np.where)
    return where(condition, x, y)


def _clamp(x: np.ndarray | torch.Tensor, min: float) -> np.ndarray | torch.Tensor:
    # Clamps values to a minimum of 0.0. Dispatches to torch or numpy.
    if isinstance(x, torch.Tensor):
        return x.clamp(min=0.0)
    return np.clip(x, 0.0, None)


_torch_stack = functools.partial(torch.stack, dim=-1)
_numpy_stack = functools.partial(np.stack, axis=-1)

# sRGB transfer function constants (IEC 61966-2-1)
# Reference: https://www.color.org/chardata/rgb/srgb.xalter
# Linear values at or below this use the linear segment
_SRGB_LINEAR_THRESHOLD = 0.0031308
# Encoded values at or below this use the linear segment (= _SRGB_LINEAR_THRESHOLD * 12.92)
_SRGB_ENCODED_THRESHOLD = 0.04045
# Slope of the linear segment
_SRGB_LINEAR_SCALE = 12.92
# Exponent for the power curve (encoding: linear -> sRGB)
_SRGB_GAMMA = 1.0 / 2.4
# Scale factor for the power curve
_SRGB_ALPHA = 1.055
# Offset for the power curve
_SRGB_BETA = 0.055


def linear_to_srgb(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert linear light values to sRGB using the IEC 61966-2-1 piecewise transfer function.

    Supports both numpy arrays and PyTorch tensors. Values below zero are clamped.
    """
    x = _clamp(x, 0.0)
    mask = x <= _SRGB_LINEAR_THRESHOLD
    return _where(mask, x * _SRGB_LINEAR_SCALE, _SRGB_ALPHA * _power(x, _SRGB_GAMMA) - _SRGB_BETA)


def srgb_to_linear(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert sRGB encoded values to linear light using the IEC 61966-2-1 piecewise transfer function.

    Supports both numpy arrays and PyTorch tensors. Values below zero are clamped.
    """
    x = _clamp(x, 0.0)
    mask = x <= _SRGB_ENCODED_THRESHOLD
    return _where(mask, x / _SRGB_LINEAR_SCALE, _power((x + _SRGB_BETA) / _SRGB_ALPHA, 2.4))


def premultiply(fg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Multiply foreground color by alpha to produce a premultiplied image.

    Args:
        fg: Color array with shape [..., C] or [C, ...].
        alpha: Alpha array with shape [..., 1] or [1, ...].
    """
    return fg * alpha


def composite_straight(
    fg: np.ndarray | torch.Tensor, bg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Composite a straight (unpremultiplied) foreground over a background.

    Formula: FG * Alpha + BG * (1 - Alpha)
    """
    return fg * alpha + bg * (1.0 - alpha)


def composite_premul(
    fg: np.ndarray | torch.Tensor, bg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Composite a premultiplied foreground over a background.

    Formula: FG + BG * (1 - Alpha)
    """
    return fg + bg * (1.0 - alpha)


def despill(
    image: np.ndarray | torch.Tensor, green_limit_mode: str = "average", strength: float = 1.0
) -> np.ndarray | torch.Tensor:
    """Remove green spill from an RGB image using a luminance-preserving method.

    Excess green is redistributed equally to red and blue channels to neutralize
    the spill without darkening the subject.

    Args:
        image: RGB float array in range 0-1.
        green_limit_mode: How to compute the green limit. "average" uses (R+B)/2,
            "max" uses max(R, B).
        strength: Blend factor between original and despilled result (0.0 to 1.0).
    """
    if strength <= 0.0:
        return image

    tensor = _is_tensor(image)
    _maximum = _if_tensor(tensor, torch.max, np.maximum)
    _stack = _if_tensor(tensor, _torch_stack, _numpy_stack)

    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]

    limit = _maximum(r, b) if green_limit_mode == "max" else (r + b) / 2.0

    if isinstance(image, torch.Tensor):
        diff: torch.Tensor = g - limit  # type: ignore[assignment]
        spill_amount = torch.clamp(diff, min=0.0)
    else:
        spill_amount = np.maximum(g - limit, 0.0)

    g_new = g - spill_amount
    r_new = r + (spill_amount * 0.5)
    b_new = b + (spill_amount * 0.5)

    despilled = _stack([r_new, g_new, b_new])

    if strength < 1.0:
        return image * (1.0 - strength) + despilled * strength

    return despilled


def clean_matte(alpha_np: np.ndarray, area_threshold: int = 300, dilation: int = 15, blur_size: int = 5) -> np.ndarray:
    """Remove small disconnected regions from a predicted alpha matte.

    Useful for eliminating tracking markers, noise islands, or other small
    artifacts that the model incorrectly classified as foreground.

    Args:
        alpha_np: Float array with shape [H, W] or [H, W, 1] in range 0.0-1.0.
        area_threshold: Minimum pixel area for a connected component to be kept.
        dilation: Radius in pixels to dilate the cleaned mask before blending.
        blur_size: Radius in pixels for Gaussian blur applied after dilation.
    """
    # Needs to be 2D for connected components analysis
    is_3d = False
    if alpha_np.ndim == 3:
        is_3d = True
        alpha_np = alpha_np[:, :, 0]

    # Binarize at 0.5 to get a uint8 mask for OpenCV
    mask_8u = (alpha_np > 0.5).astype(np.uint8) * 255

    # Label each connected foreground region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)

    cleaned_mask = np.zeros_like(mask_8u)

    # Keep regions above the area threshold; label 0 is background and is always skipped
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            cleaned_mask[labels == i] = 255

    # Dilate to recover edges lost during binarization
    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned_mask = cv2.dilate(cleaned_mask, kernel)

    # Blur to soften the hard edges of the cleaned mask
    if blur_size > 0:
        b_size = int(blur_size * 2 + 1)
        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (b_size, b_size), 0)

    safe_zone = cleaned_mask.astype(np.float32) / 255.0

    # Multiply the original soft alpha by the safe zone to zero out removed regions
    result_alpha = alpha_np * safe_zone

    if is_3d:
        result_alpha = result_alpha[:, :, np.newaxis]

    return result_alpha


def create_checkerboard(
    width: int, height: int, checker_size: int = 64, color1: float = 0.2, color2: float = 0.4
) -> np.ndarray:
    """Create a grayscale checkerboard pattern for compositing previews.

    Values are in linear light (not gamma-encoded). Convert with srgb_to_linear
    before use if your pipeline expects linear input.

    Args:
        width: Output width in pixels.
        height: Output height in pixels.
        checker_size: Side length of each checker tile in pixels.
        color1: Linear brightness of the dark tiles (0.0-1.0).
        color2: Linear brightness of the light tiles (0.0-1.0).

    Returns:
        Float array with shape [H, W, 3] in range 0.0-1.0.
    """
    x = np.arange(width)
    y = np.arange(height)

    x_tiles = x // checker_size
    y_tiles = y // checker_size

    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)

    # Even sum = color1, odd sum = color2
    checker = (x_grid + y_grid) % 2

    bg_img = np.where(checker == 0, color1, color2).astype(np.float32)

    return np.stack([bg_img, bg_img, bg_img], axis=-1)
