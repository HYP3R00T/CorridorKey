# Contracts

Data structures that flow between pipeline stages. Each contract defines exactly what one stage produces and the next stage consumes.

Contracts carry no behaviour - only data. Stages communicate only through their contracts. What happens inside a stage is an implementation detail.

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

`PostprocessParams` is not a data carrier between stages. It is the configuration bag that controls how stage 5 behaves. It is constructed by the caller and passed in alongside the `RawPrediction`.

So the count is: four data contracts flowing between stages, plus one params contract for stage 5 configuration.
