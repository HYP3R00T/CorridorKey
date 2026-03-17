# PostprocessParams

Parameters controlling stage 5 (postprocess) behaviour. Constructed by the caller and passed in alongside the `RawPrediction`.

This is not a data carrier between stages. It is the configuration bag for stage 5. The GUI and CLI construct it from user settings. The service layer converts `InferenceParams` to `PostprocessParams` via `inference_params_to_postprocess`.

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `despill_strength` | float | 1.0 | Green spill suppression strength. 0.0 means no despill. 1.0 means full correction. |
| `auto_despeckle` | bool | True | Remove small disconnected alpha islands. |
| `despeckle_size` | int | 400 | Minimum connected region area in pixels to keep. Regions smaller than this are removed. |
| `source_passthrough` | bool | False | Use original source pixels in fully opaque interior regions. |
| `edge_erode_px` | int | 3 | Pixels to erode the interior mask inward before blending. Safety buffer against green spill at edges. |
| `edge_blur_px` | int | 7 | Gaussian blur radius for the transition seam between source pixels and model fg. |
| `fg_is_straight` | bool | True | Whether the fg prediction is straight (unpremultiplied). Always true for the current model. |

## Relationship to InferenceParams

`InferenceParams` is the application-layer configuration object used by the CLI and GUI. `PostprocessParams` is the core-layer equivalent used by stage 5.

The bridge function `inference_params_to_postprocess` in [corridorkey/service.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/service.py) converts between them. This is the only place that mapping lives.

## Source Code

Defined in [corridorkey-core/stages.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/stages.py).

## Related Documents

- [Postprocess](../pipeline/postprocess.md) - Stage 5, which uses these parameters.
- [Source passthrough](../improvements/source-passthrough.md) - Detailed explanation of the source passthrough parameters.
- [Configuration](../configuration/index.md) - All parameters with defaults and valid ranges.
