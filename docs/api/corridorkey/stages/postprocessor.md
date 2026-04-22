# Postprocessor

Converts inference tensors to numpy arrays, upscales to the original frame resolution, and applies despill, despeckle, source passthrough, and hint sharpening.

## Reference

::: corridorkey.stages.postprocessor.orchestrator.postprocess_frame

::: corridorkey.stages.postprocessor.contracts.ProcessedFrame

## Usage

```python
from corridorkey import load_config
from corridorkey.stages.postprocessor.orchestrator import postprocess_frame

config = load_config()
postprocess_config = config.to_postprocess_config()

processed = postprocess_frame(
    inference_result,
    postprocess_config,
    output_dir=manifest.output_dir,
)

print(processed.alpha.shape)      # [H, W, 1] float32 — original resolution
print(processed.fg.shape)         # [H, W, 3] float32
print(processed.processed.shape)  # [H, W, 4] float32 — premultiplied RGBA
print(processed.comp.shape)       # [H, W, 3] uint8 — checkerboard preview
```

## ProcessedFrame arrays

All arrays are numpy, at the original source frame resolution.

| Field | Shape | dtype | Description |
|---|---|---|---|
| `alpha` | `[H, W, 1]` | float32 | Alpha matte, values in [0, 1] |
| `fg` | `[H, W, 3]` | float32 | Straight sRGB foreground, values in [0, 1] |
| `processed` | `[H, W, 4]` | float32 | Premultiplied RGBA in linear light |
| `comp` | `[H, W, 3]` | uint8 | Checkerboard preview composite, sRGB |

## Postprocessing steps

Applied in order:

1. Transfer tensors from GPU to CPU
2. Upscale alpha and foreground to source resolution
3. `hint_sharpen` — apply binarised alpha hint mask to remove upscaling tails
4. `source_passthrough` — replace model FG in opaque interior regions with original source pixels
5. `auto_despeckle` — remove small disconnected alpha islands
6. `despill` — suppress green channel in foreground
7. Composite — multiply foreground by alpha to produce premultiplied RGBA
8. `debug_dump` — write intermediate PNG snapshots (when enabled)
