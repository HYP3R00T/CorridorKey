# Property-Based Tests

Property-based tests verify that a function satisfies a mathematical invariant across many randomly generated inputs, rather than a single fixed example. This catches edge cases that hand-written tests miss.

This project uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing.

## When to Use Property-Based Tests

Property-based tests are well suited for:

- Mathematical invariants in color math (roundtrips, monotonicity, range constraints)
- Shape contracts that must hold for any valid input size
- Functions where the correct output can be described as a rule rather than a specific value

They are not suited for:

- Tests that require a specific expected output value
- Tests that depend on model weights or hardware

## Installing Hypothesis

Hypothesis is already in the dev dependencies for `corridorkey-core`. Run `uv sync --all-groups` to install it if you haven't already.

```toml
[dependency-groups]
dev = [
  "pytest>=9.0.2",
  "pytest-cov>=7.0.0",
  "hypothesis>=6.151.9",
]
```

## Examples

### Color space roundtrip

The sRGB transfer function must be its own inverse. For any value in `[0, 1]`, converting to linear and back must return the original value.

```python
from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
from corridorkey_core.compositing import linear_to_srgb, srgb_to_linear

@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_srgb_linear_roundtrip(value: float):
    x = np.array([value], dtype=np.float32)
    assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-5)
```

### Output range constraint

`linear_to_srgb` must always return values in `[0, 1]` for any non-negative input.

```python
@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_linear_to_srgb_output_range(value: float):
    x = np.array([value], dtype=np.float32)
    result = linear_to_srgb(x)
    assert 0.0 <= result[0] <= 1.0
```

### Compositing identity

Compositing a foreground over any background with full alpha must return the foreground exactly.

```python
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_composite_full_alpha_identity(fg_val: float, bg_val: float):
    fg = np.full((4, 4, 3), fg_val, dtype=np.float32)
    bg = np.full((4, 4, 3), bg_val, dtype=np.float32)
    alpha = np.ones((4, 4, 1), dtype=np.float32)
    result = composite_straight(fg, bg, alpha)
    assert np.allclose(result, fg, atol=1e-6)
```

## Suppressing Hypothesis Output in CI

Hypothesis prints a summary when it finds a failing example. This is useful locally but noisy in CI. Add a `settings` profile to `conftest.py` if needed:

```python
from hypothesis import settings
settings.register_profile("ci", max_examples=50)
settings.load_profile("ci")
```

## Related

- [Testing Overview](index.md)
- [Unit Tests](unit-tests.md)
