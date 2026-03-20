"""Property-based tests for corridorkey_new.preprocessor.normalise."""

from __future__ import annotations

import numpy as np
from corridorkey_new.preprocessor.normalise import normalise_image
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

_image = arrays(
    dtype=np.float32,
    shape=st.tuples(
        st.integers(1, 16),
        st.integers(1, 16),
        st.just(3),
    ),
    elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)


class TestNormaliseImageProperties:
    @given(_image)
    def test_output_dtype_float32(self, image: np.ndarray):
        """Output is always float32."""
        assert normalise_image(image).dtype == np.float32

    @given(_image)
    def test_output_shape_preserved(self, image: np.ndarray):
        """Shape is never changed."""
        assert normalise_image(image).shape == image.shape

    @given(_image)
    def test_output_can_be_negative(self, image: np.ndarray):
        """Normalised values are not bounded to [0, 1] — this is expected."""
        # Just verify it doesn't raise and returns finite values
        result = normalise_image(image)
        assert np.all(np.isfinite(result))

    @given(_image)
    def test_linear(self, image: np.ndarray):
        """normalise(a) - normalise(b) == normalise(a - b) (linearity of subtraction)."""
        # Shift by a constant and verify the difference is preserved
        shifted = np.clip(image + 0.1, 0.0, 1.0).astype(np.float32)
        diff_normalised = normalise_image(shifted) - normalise_image(image)
        normalised_diff = normalise_image(shifted - image + np.array([0.485, 0.456, 0.406], dtype=np.float32))
        # Both should be close (within float32 precision)
        assert np.allclose(diff_normalised, normalised_diff, atol=1e-5)

    @given(
        st.integers(1, 8),
        st.integers(1, 8),
    )
    def test_imagenet_mean_maps_to_zero(self, h: int, w: int):
        """An image filled with ImageNet mean values normalises to ~0."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        image = np.broadcast_to(mean, (h, w, 3)).copy()
        result = normalise_image(image)
        assert np.allclose(result, 0.0, atol=1e-6)
