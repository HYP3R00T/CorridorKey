"""Preprocessing stage — ImageNet normalisation.

Normalises the image with ImageNet mean and std before inference.
This is a model input contract — the weights were trained exclusively
on inputs in this distribution.

Operates on a PyTorch tensor so the computation runs on whatever device the
tensor lives on (CUDA, MPS, or CPU).

The alpha hint is never normalised — it is passed through as-is.

In-place ops
------------
``sub_`` and ``div_`` modify the tensor in-place, eliminating two intermediate
allocations compared to ``(image - mean) / std``. This is safe here because
the tensor is freshly created by ``to_tensors`` and is not referenced anywhere
else at the point normalisation runs.
"""

from __future__ import annotations

import torch

# ImageNet mean and std — model input contract, do not change.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def normalise_image(image: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std normalisation to an sRGB image tensor in-place.

    Uses ``sub_`` / ``div_`` to avoid allocating intermediate tensors.
    The input tensor is modified and returned — do not use the original
    reference after calling this function.

    Args:
        image: float32 tensor [B, 3, H, W] or [3, H, W], sRGB, range 0.0–1.0.

    Returns:
        The same tensor, normalised in-place. Values will be outside [0, 1] —
        that is expected and correct.
    """
    mean = torch.tensor(_MEAN, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    if image.ndim == 3:
        image = image.unsqueeze(0)
        image.sub_(mean).div_(std)
        return image.squeeze(0)
    image.sub_(mean).div_(std)
    return image
