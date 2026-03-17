# ProcessedFrame

Output of stage 5 (postprocess). Input to stage 6 (write outputs).

Carries all four final output arrays at original source resolution, ready to be written to disk.

## Fields

| Field | Type | Description |
|---|---|---|
| `alpha` | `np.ndarray [H, W, 1]` float32 | Alpha matte, linear, values 0.0-1.0. |
| `fg` | `np.ndarray [H, W, 3]` float32 | Foreground RGB, sRGB straight (unpremultiplied), values 0.0-1.0. |
| `comp` | `np.ndarray [H, W, 3]` float32 | Checkerboard composite preview, sRGB, values 0.0-1.0. |
| `processed` | `np.ndarray [H, W, 4]` float32 | Linear premultiplied RGBA. Primary compositing deliverable. |
| `source_h` | int | Frame height in pixels. |
| `source_w` | int | Frame width in pixels. |
| `stem` | str | Filename stem carried through from stage 1 for output file naming. |

## Notes

All arrays are at original source resolution (`source_h x source_w`). Stage 5 upsamples from model resolution before returning this contract.

`comp` is a preview only. It is not suitable for compositing. The checkerboard background makes transparency visible for review purposes.

`processed` is the primary deliverable. It is in linear premultiplied RGBA, which is the standard format for NLE and VFX compositing tools.

`fg` is straight (unpremultiplied) sRGB. It is useful if you want to handle premultiplication yourself downstream, or if you need the foreground in sRGB for a specific workflow.

## Source Code

Defined in [corridorkey-core/stages.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/stages.py).

## Related Documents

- [Postprocess](../pipeline/postprocess.md) - Stage 5, which produces this contract.
- [Write outputs](../pipeline/write-outputs.md) - Stage 6, which consumes this contract.
