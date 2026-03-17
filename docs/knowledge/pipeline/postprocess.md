# Postprocess

Stage 5 takes the raw model predictions and produces four final outputs ready for delivery. It runs entirely on CPU in NumPy. No GPU access, no filesystem access.

## What It Does

1. Despeckle - removes small disconnected alpha islands using connected component analysis.
2. Despill - suppresses green channel contamination in the foreground.
3. Composite - builds a checkerboard preview by compositing the foreground over a generated background.
4. Premultiply - multiplies foreground RGB by alpha to produce premultiplied RGBA.
5. Upsample - resizes all four outputs from model resolution back to original source resolution.
6. Source passthrough - in fully opaque interior regions, replaces the model's foreground prediction with original source pixels.

## Input

A `RawPrediction` from stage 4, plus:

- `source_image` - the original source frame `[H, W, 3]` float32 sRGB. Used only when source passthrough is enabled.
- `source_is_linear` - whether the source was originally in linear light.
- `params` - a `PostprocessParams` controlling each step.
- `stem` - filename stem carried through to the output.

## Output

A `ProcessedFrame` with four arrays at original source resolution:

- `alpha` - alpha matte `[H, W, 1]`, float32, linear, values 0.0-1.0.
- `fg` - foreground RGB `[H, W, 3]`, float32, sRGB straight (unpremultiplied).
- `comp` - checkerboard composite preview `[H, W, 3]`, float32, sRGB.
- `processed` - linear premultiplied RGBA `[H, W, 4]`, float32. Primary compositing deliverable.

## Despeckle

Connected component analysis finds all connected regions in the alpha matte. Regions below `despeckle_size` pixels in area are removed. Dilation and blur are applied after to smooth the cleaned matte.

Controlled by `auto_despeckle` (bool) and `despeckle_size` (int, pixel area).

## Despill

Green light bounces off the backdrop and tints the subject, especially at edges and on reflective surfaces. Despill suppresses the green channel relative to the average of red and blue.

Controlled by `despill_strength` (float, 0.0-1.0). 0.0 means no despill. 1.0 means full correction.

## Source Passthrough

The model's foreground prediction is accurate at edges but can introduce subtle colour shifts in the interior - face, body, clothing. The original source pixels are always the ground truth for those regions.

Source passthrough blends original source pixels into fully opaque interior regions. Only the edge transition band uses the model's fg prediction, where green screen separation actually matters.

A blend mask is computed from the alpha matte:

1. The alpha matte is eroded inward by `edge_erode_px` pixels to define the interior region.
2. The eroded mask is blurred by `edge_blur_px` pixels to create a smooth transition.
3. Interior pixels (blend mask near 1.0) take source pixels. Edge pixels (blend mask near 0.0) take model fg.

Controlled by `source_passthrough` (bool), `edge_erode_px` (int), and `edge_blur_px` (int).

## Colour Space in Outputs

- `alpha` - linear. Alpha is always linear.
- `fg` - sRGB straight. The model predicts in sRGB. Despill operates in sRGB. Source passthrough blends in sRGB.
- `comp` - sRGB. The checkerboard is generated in sRGB. Compositing happens in linear then converts back to sRGB for the preview.
- `processed` - linear premultiplied. The fg is converted from sRGB to linear before premultiplication. This is the standard format for NLE and VFX compositing.

## Source Code

- Stage function: `stage_5_postprocess` in [corridorkey-core/stages.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/stages.py)
- Compositing utilities: [corridorkey-core/compositing.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/compositing.py)
- Contract: `PostprocessParams` and `ProcessedFrame` in [corridorkey-core/stages.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/stages.py)

## Related Documents

- [Infer](infer.md) - Stage 4, which produces the raw predictions this stage consumes.
- [Write outputs](write-outputs.md) - Stage 6, which writes the ProcessedFrame this stage produces.
- [PostprocessParams contract](../contracts/postprocess-params.md)
- [ProcessedFrame contract](../contracts/processed-frame.md)
- [Source passthrough](../improvements/source-passthrough.md)
