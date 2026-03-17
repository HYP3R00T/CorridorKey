# Contracts

Data structures that flow between pipeline stages. Each contract defines exactly what one stage produces and the next stage consumes.

Contracts are defined as Python dataclasses. They carry no behaviour - only data. Stages communicate only through their contracts. What happens inside a stage is an implementation detail.

## Contracts in This Section

| Contract | Produced by | Consumed by | Description |
|---|---|---|---|
| [FrameData](frame-data.md) | Stage 1 (load frame) | Stage 3 (preprocess) | Source image and mask arrays with metadata. |
| [PreprocessedTensor](preprocessed-tensor.md) | Stage 3 (preprocess) | Stage 4 (infer) | Model-ready tensor on the target device. |
| [RawPrediction](raw-prediction.md) | Stage 4 (infer) | Stage 5 (postprocess) | Raw alpha and foreground at model resolution. |
| [PostprocessParams](postprocess-params.md) | Caller (service or GUI) | Stage 5 (postprocess) | Parameters controlling postprocessing behaviour. |
| [ProcessedFrame](processed-frame.md) | Stage 5 (postprocess) | Stage 6 (write outputs) | Four final output arrays at source resolution. |

## Why Four Contracts for Six Stages

Stage 2 (generate masks) writes files to disk and does not pass a contract to stage 3. Stage 1 reads those files directly.

`PostprocessParams` is not a data carrier between stages. It is the configuration bag that controls how stage 5 behaves. It is constructed by the caller (the service layer or GUI) and passed in alongside the `RawPrediction`.

So the count is: four data contracts flowing between stages, plus one params contract for stage 5 configuration.

## Package Locations

Contracts that cross the package boundary between `corridorkey` and `corridorkey-core` are defined in the package that produces them:

- `FrameData` and `WriteConfig` are defined in `corridorkey.stages` (produced by the application layer).
- `PreprocessedTensor`, `RawPrediction`, `PostprocessParams`, and `ProcessedFrame` are defined in `corridorkey_core.stages` (produced by the core layer).

The application layer imports core contracts by name when it needs to bridge between them. See `inference_params_to_postprocess` and `output_config_to_write_config` in [corridorkey/service.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/service.py).
