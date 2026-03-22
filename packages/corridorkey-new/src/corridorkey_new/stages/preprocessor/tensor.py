"""Preprocessing stage — tensor construction.

Converts NumPy arrays from the reader into PyTorch tensors in channel-first
layout and moves them to the target device. All subsequent transforms operate
on these tensors directly on the device.

This is the boundary between NumPy (CPU disk I/O) and PyTorch (device compute).

BGR→RGB reorder
---------------
OpenCV reads images as BGR. Rather than reordering on CPU (a full memcopy),
we transfer the BGR tensor to the device and then reorder channels there as
a near-zero-cost index operation: ``img_t[:, [2, 1, 0], :, :]``.
On CUDA this is a strided view — no data is copied.
"""

from __future__ import annotations

import numpy as np
import torch


def to_tensors(
    image: np.ndarray,
    alpha: np.ndarray,
    device: str,
    bgr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert image and alpha NumPy arrays to device tensors.

    Transfers data to the device first, then reorders BGR→RGB on-device if
    needed. This avoids a full-resolution CPU memcopy just to swap channels.

    Args:
        image: float32 [H, W, 3], sRGB or linear, BGR or RGB channel order.
        alpha: float32 [H, W, 1], linear, range 0.0–1.0.
        device: PyTorch device string ("cuda", "mps", "cpu").
        bgr: If True, reorder image channels BGR→RGB on-device after transfer.
            Pass the value returned by ``_read_frame_pair``.

    Returns:
        Tuple of (image [1, 3, H, W] RGB, alpha [1, 1, H, W]) on the device.
    """
    # np.ascontiguousarray is a no-op if already contiguous, otherwise a
    # single copy — makes the intent explicit and avoids a hidden copy inside
    # torch.from_numpy when the array is non-contiguous (e.g. after a slice).
    img_t = (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        .unsqueeze(0)
        .float()
        .to(device)
    )
    alp_t = (
        torch.from_numpy(np.ascontiguousarray(alpha.transpose(2, 0, 1)))
        .unsqueeze(0)
        .float()
        .to(device)
    )

    if bgr:
        # Reorder BGR→RGB on-device. On CUDA this is a strided view — no copy.
        img_t = img_t[:, [2, 1, 0], :, :]

    return img_t, alp_t


def to_tensor(
    image: np.ndarray,
    alpha: np.ndarray,
    device: str,
    bgr: bool = False,
) -> torch.Tensor:
    """Build the final model input tensor by concatenating image and alpha.

    Kept for callers that use the single-tensor API. Prefer ``to_tensors``
    when transforms need to be applied between construction and concatenation.

    Args:
        image: float32 [H, W, 3], ImageNet-normalised.
        alpha: float32 [H, W, 1], linear, range 0.0–1.0.
        device: PyTorch device string ("cuda", "mps", "cpu").
        bgr: If True, reorder image channels BGR→RGB on-device after transfer.

    Returns:
        float32 tensor [1, 4, H, W] on the specified device.
    """
    img_t, alp_t = to_tensors(image, alpha, device, bgr=bgr)
    return torch.cat([img_t, alp_t], dim=1)  # [1, 4, H, W]
