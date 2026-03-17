# FrameData

Output of stage 1 (load frame). Input to stage 3 (preprocess).

Carries the source image and alpha hint mask as normalised float32 arrays, along with metadata needed by downstream stages.

## Fields

| Field | Type | Description |
|---|---|---|
| `image` | `np.ndarray [H, W, 3]` float32 | RGB image, sRGB, values 0.0-1.0. Always sRGB regardless of source colour space. |
| `mask` | `np.ndarray [H, W, 1]` float32 | Alpha hint mask, linear, values 0.0-1.0. |
| `source_h` | int | Original frame height in pixels. |
| `source_w` | int | Original frame width in pixels. |
| `is_linear` | bool | True if the source image was originally in linear light (for example, EXR). The image field always contains sRGB - this flag records the origin. |
| `stem` | str | Filename stem of the source frame (for example, "frame_000001"). Used for output file naming. |

## Notes

The `image` field always contains sRGB. If the source was in linear light, stage 1 converts it before returning. Downstream stages do not need to check `is_linear` for colour space handling. The flag is carried through for reference only.

The `mask` field is always `[H, W, 1]` - a single channel with an explicit trailing dimension. Stage 3 expects this shape.

## Related Documents

- [Load frame](../pipeline/load-frame.md) - Stage 1, which produces this contract.
- [Preprocess](../pipeline/preprocess.md) - Stage 3, which consumes this contract.
