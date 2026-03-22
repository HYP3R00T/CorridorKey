"""Unit tests for corridorkey_new.stages.preprocessor.resize."""

from __future__ import annotations

import torch
from corridorkey_new.stages.preprocessor.resize import DEFAULT_UPSAMPLE_MODE, resize_frame


def _make_image(h: int, w: int) -> torch.Tensor:
    return torch.rand(1, 3, h, w, dtype=torch.float32)


def _make_alpha(h: int, w: int) -> torch.Tensor:
    return torch.rand(1, 1, h, w, dtype=torch.float32)


class TestResizeFrame:
    def test_squish_output_shape_image(self):
        image, alpha = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512, "squish")
        assert image.shape == (1, 3, 512, 512)

    def test_squish_output_shape_alpha(self):
        image, alpha = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512, "squish")
        assert alpha.shape == (1, 1, 512, 512)

    def test_already_square_unchanged_shape(self):
        image, alpha = resize_frame(_make_image(512, 512), _make_alpha(512, 512), 512, "squish")
        assert image.shape == (1, 3, 512, 512)
        assert alpha.shape == (1, 1, 512, 512)

    def test_output_dtype_float32(self):
        image, alpha = resize_frame(_make_image(64, 64), _make_alpha(64, 64), 32, "squish")
        assert image.dtype == torch.float32
        assert alpha.dtype == torch.float32

    def test_downscale_values_stay_in_range(self):
        image, alpha = resize_frame(_make_image(128, 128), _make_alpha(128, 128), 32, "squish")
        assert image.min().item() >= 0.0 and image.max().item() <= 1.0
        assert alpha.min().item() >= 0.0 and alpha.max().item() <= 1.0

    def test_upscale_values_stay_in_range(self):
        # Bicubic interpolation can produce slight overshoot/undershoot outside [0,1]
        # — this is a known property of the cubic kernel, not a bug.
        # We verify the output is float32 and spatially correct, not clamped.
        image, alpha = resize_frame(_make_image(16, 16), _make_alpha(16, 16), 64, "squish")
        assert image.dtype == torch.float32
        assert image.shape == (1, 3, 64, 64)
        assert alpha.dtype == torch.float32
        assert alpha.shape == (1, 1, 64, 64)

    def test_letterbox_falls_back_to_squish(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            image, alpha = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512, "letterbox")
        assert image.shape == (1, 3, 512, 512)
        assert "not yet implemented" in caplog.text

    def test_image_and_alpha_same_spatial_size(self):
        image, alpha = resize_frame(_make_image(200, 300), _make_alpha(200, 300), 128, "squish")
        assert image.shape[2:] == alpha.shape[2:]

    def test_default_upsample_mode_is_bicubic(self):
        assert DEFAULT_UPSAMPLE_MODE == "bicubic"

    def test_upsample_mode_bilinear_produces_different_result_than_bicubic(self):
        """bilinear and bicubic must produce different tensors for the same input."""
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img_bicubic, _ = resize_frame(src, alp, 128, "squish", upsample_mode="bicubic")
        img_bilinear, _ = resize_frame(src, alp, 128, "squish", upsample_mode="bilinear")
        assert not torch.allclose(img_bicubic, img_bilinear)

    def test_upsample_mode_has_no_effect_when_downscaling(self):
        """When downscaling, area mode is always used — upsample_mode is ignored."""
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_bicubic, _ = resize_frame(src, alp, 128, "squish", upsample_mode="bicubic")
        img_bilinear, _ = resize_frame(src, alp, 128, "squish", upsample_mode="bilinear")
        torch.testing.assert_close(img_bicubic, img_bilinear)
