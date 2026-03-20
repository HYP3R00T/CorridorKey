"""Property-based tests for corridorkey_new.preprocessor.colorspace."""

from __future__ import annotations

import numpy as np
from corridorkey_new.preprocessor.colorspace import linear_to_srgb
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Strategy: float32 arrays shaped [H, W, 3] with values in [0, 1]
_image = arrays(
    dtype=np.float32,
    shape=st.tuples(
        st.integers(1, 16),
        st.integers(1, 16),
        st.just(3),
    ),
    elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)


class TestLinearToSrgbProperties:
    @given(_image)
    def test_output_range_0_1(self, image: np.ndarray):
        """sRGB output is always in [0, 1]."""
        result = linear_to_srgb(image)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    @given(_image)
    def test_output_dtype_float32(self, image: np.ndarray):
        """Output is always float32."""
        assert linear_to_srgb(image).dtype == np.float32

    @given(_image)
    def test_output_shape_preserved(self, image: np.ndarray):
        """Shape is never changed."""
        assert linear_to_srgb(image).shape == image.shape

    @given(_image)
    def test_monotone(self, image: np.ndarray):
        """Brighter linear input always produces brighter sRGB output (monotone)."""
        # Scale the image by 0.5 — every pixel gets darker
        darker = linear_to_srgb(image * 0.5)
        brighter = linear_to_srgb(image)
        assert np.all(darker <= brighter + 1e-6)

    @given(
        arrays(
            dtype=np.float32,
            shape=(4, 4, 3),
            elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
        )
    )
    def test_out_of_range_input_clamped(self, image: np.ndarray):
        """Out-of-range linear values are clamped, not propagated."""
        result = linear_to_srgb(image)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0
