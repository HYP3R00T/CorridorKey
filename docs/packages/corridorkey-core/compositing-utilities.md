# Compositing Utilities

`corridorkey_core.compositing` provides pure math functions for color space conversion, alpha compositing, green spill removal, matte cleanup, and checkerboard generation. All functions are used internally by `process_frame` but are also importable for custom pipelines.

All functions accept both NumPy arrays and PyTorch tensors unless noted otherwise.

## Color Space Conversion

The pipeline works in linear light internally and converts to sRGB for display and model input. Both functions implement the IEC 61966-2-1 piecewise transfer function.

```python
from corridorkey_core.compositing import linear_to_srgb, srgb_to_linear
import numpy as np

linear = np.array([0.0, 0.5, 1.0], dtype=np.float32)
srgb   = linear_to_srgb(linear)   # [0.0, ~0.735, 1.0]
back   = srgb_to_linear(srgb)     # [0.0, 0.5, 1.0]
```

Values below zero are clamped to zero. The functions are exact inverses of each other within float32 tolerance.

## Premultiplication

`premultiply` multiplies foreground color by alpha. Use this before writing to EXR or compositing in premultiplied mode.

```python
from corridorkey_core.compositing import premultiply

fg_premul = premultiply(fg, alpha)  # fg * alpha
```

## Alpha Compositing

Two compositing functions implement the Porter-Duff over operator:

`composite_straight` expects a straight (unpremultiplied) foreground:

```text
result = fg * alpha + bg * (1 - alpha)
```

`composite_premul` expects a premultiplied foreground:

```text
result = fg + bg * (1 - alpha)
```

```python
from corridorkey_core.compositing import composite_straight, composite_premul

# Straight foreground over background
comp = composite_straight(fg, bg, alpha)

# Premultiplied foreground over background
comp = composite_premul(fg_premul, bg, alpha)
```

## Green Spill Removal

`despill` removes the green color cast that reflects off a green screen onto the subject. It redistributes excess green equally to the red and blue channels.

```python
from corridorkey_core.compositing import despill

# Default: average mode, full strength
fg_clean = despill(fg)

# Weaker despill
fg_clean = despill(fg, strength=0.5)

# Max mode: limits green to max(R, B) instead of (R+B)/2
fg_clean = despill(fg, green_limit_mode="max", strength=1.0)
```

`green_limit_mode` controls how the green ceiling is computed:

| Mode | Formula | When to use |
|---|---|---|
| `"average"` | `(R + B) / 2` | Default. Balanced removal for most footage. |
| `"max"` | `max(R, B)` | More aggressive. Use when average leaves visible spill. |

`strength=0.0` is a no-op. `strength=1.0` applies full despill. Values between blend the original and despilled result.

## Matte Cleanup

`clean_matte` removes small disconnected foreground regions from a predicted alpha matte. This eliminates tracking markers, noise islands, and other small artifacts the model incorrectly classified as foreground.

```python
from corridorkey_core.compositing import clean_matte

# Remove regions smaller than 400 pixels
alpha_clean = clean_matte(alpha, area_threshold=400)

# Tighter cleanup with less edge dilation
alpha_clean = clean_matte(alpha, area_threshold=200, dilation=10, blur_size=3)
```

The function accepts `[H, W]` or `[H, W, 1]` input and returns the same shape. It does not accept PyTorch tensors - NumPy only.

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `area_threshold` | `300` | Minimum pixel area to keep a connected region |
| `dilation` | `15` | Pixels to dilate the cleaned mask before blending |
| `blur_size` | `5` | Gaussian blur radius applied after dilation |

## Checkerboard Generation

`create_checkerboard` generates a grey checkerboard background for transparency previews.

```python
from corridorkey_core.compositing import create_checkerboard

bg = create_checkerboard(width=1920, height=1080, checker_size=128, color1=0.15, color2=0.55)
# Returns [H, W, 3] float32
```

The output is in linear light. Convert with `linear_to_srgb` if your display pipeline expects sRGB.

## Related

- [compositing reference](../../api/corridorkey-core/compositing.md)
- [Output contract](output-contract.md)
