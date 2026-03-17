# RawPrediction

Output of stage 4 (infer). Input to stage 5 (postprocess).

Carries the raw alpha and foreground predictions at model resolution, before any postprocessing has been applied.

## Fields

| Field | Type | Description |
|---|---|---|
| `alpha` | `np.ndarray [img_size, img_size, 1]` float32 | Predicted alpha matte, linear, values 0.0-1.0. At model resolution. |
| `fg` | `np.ndarray [img_size, img_size, 3]` float32 | Predicted foreground RGB, sRGB straight (unpremultiplied), values 0.0-1.0. At model resolution. |
| `img_size` | int | Resolution of the prediction (matches the model's training resolution, 2048). |
| `source_h` | int | Original frame height in pixels. Used by stage 5 for upsampling. |
| `source_w` | int | Original frame width in pixels. Used by stage 5 for upsampling. |

## Notes

Both arrays are at model resolution (`img_size x img_size`), not at source resolution. Stage 5 upsamples them back to `source_h x source_w` after postprocessing.

`alpha` and `fg` are the direct outputs of the model's decoder heads after the CNN refiner has applied its corrections. No despeckle, no despill, no compositing has been applied yet.

## Related Documents

- [Infer](../pipeline/infer.md) - Stage 4, which produces this contract.
- [Postprocess](../pipeline/postprocess.md) - Stage 5, which consumes this contract.
