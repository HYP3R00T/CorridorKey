"""Property-based tests for corridorkey_new.preprocessor.resize."""

from __future__ import annotations

import numpy as np
from corridorkey_new.preprocessor.resize import resize_frame
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

_sizes = st.integers(4, 32)
_img_sizes = st.sampled_from([16, 32, 64])


def _make_image(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.float32)


def _make_alpha(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 1), dtype=np.float32)


class TestResizeFrameProperties:
    @given(_sizes, _sizes, _img_sizes)
    def test_output_image_shape(self, h: int, w: int, img_size: int):
        """Image output is always [img_size, img_size, 3]."""
        img, _ = resize_frame(_make_image(h, w), _make_alpha(h, w), img_size, "squish")
        assert img.shape == (img_size, img_size, 3)

    @given(_sizes, _sizes, _img_sizes)
    def test_output_alpha_shape(self, h: int, w: int, img_size: int):
        """Alpha output is always [img_size, img_size, 1]."""
        _, alpha = resize_frame(_make_image(h, w), _make_alpha(h, w), img_size, "squish")
        assert alpha.shape == (img_size, img_size, 1)

    @given(_sizes, _sizes, _img_sizes)
    def test_output_dtype_float32(self, h: int, w: int, img_size: int):
        """Both outputs are always float32."""
        img, alpha = resize_frame(_make_image(h, w), _make_alpha(h, w), img_size, "squish")
        assert img.dtype == np.float32
        assert alpha.dtype == np.float32

    @given(
        _sizes,
        _sizes,
        _img_sizes,
        arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(4, 16), st.integers(4, 16), st.just(3)),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
    )
    def test_values_stay_in_range(self, h: int, w: int, img_size: int, raw_img: np.ndarray):
        """Resize never produces values outside [0, 1] for inputs in [0, 1]."""
        alpha = np.zeros((raw_img.shape[0], raw_img.shape[1], 1), dtype=np.float32)
        img_out, alpha_out = resize_frame(raw_img, alpha, img_size, "squish")
        assert float(img_out.min()) >= -1e-6
        assert float(img_out.max()) <= 1.0 + 1e-6

    @given(_img_sizes)
    def test_already_square_same_size_unchanged_shape(self, img_size: int):
        """Resizing a square image to its own size preserves shape."""
        img = _make_image(img_size, img_size)
        alpha = _make_alpha(img_size, img_size)
        img_out, alpha_out = resize_frame(img, alpha, img_size, "squish")
        assert img_out.shape == (img_size, img_size, 3)
        assert alpha_out.shape == (img_size, img_size, 1)

    @given(_sizes, _sizes, _img_sizes)
    def test_image_and_alpha_same_spatial_dims(self, h: int, w: int, img_size: int):
        """Image and alpha always share the same H and W after resize."""
        img_out, alpha_out = resize_frame(_make_image(h, w), _make_alpha(h, w), img_size, "squish")
        assert img_out.shape[:2] == alpha_out.shape[:2]

    @given(_sizes, _sizes, _img_sizes)
    def test_letterbox_falls_back_to_squish(self, h: int, w: int, img_size: int):
        """letterbox strategy produces same shape as squish (fallback)."""
        img = _make_image(h, w)
        alpha = _make_alpha(h, w)
        squish_img, squish_alpha = resize_frame(img, alpha, img_size, "squish")
        letter_img, letter_alpha = resize_frame(img, alpha, img_size, "letterbox")
        assert squish_img.shape == letter_img.shape
        assert squish_alpha.shape == letter_alpha.shape
