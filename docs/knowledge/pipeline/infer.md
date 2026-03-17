# Infer

Stage 4 runs the model forward pass and returns raw alpha and foreground predictions at model resolution.

This stage is the GPU-bound core of the pipeline. Everything before it is preparation. Everything after it is refinement and delivery.

## What It Does

1. Casts the input tensor to the model's precision dtype (fp16, bf16, or fp32).
2. Optionally registers a forward hook on the CNN refiner to scale or tile its output.
3. Runs the model forward pass under `torch.autocast` and `torch.inference_mode`.
4. Extracts alpha and foreground tensors from the model output dict.
5. Moves both to CPU and converts to NumPy float32 arrays.
6. Returns a `RawPrediction`.

## Input

A `PreprocessedTensor` from stage 3, plus:

- `engine` - a loaded `CorridorKeyEngine` (or MLX adapter). Created by `create_engine()`.
- `refiner_scale` - multiplier on the CNN refiner's delta corrections. Default 1.0.

## Output

A `RawPrediction` with:

- `alpha` - predicted alpha matte `[img_size, img_size, 1]`, float32, linear, values 0.0-1.0.
- `fg` - predicted foreground RGB `[img_size, img_size, 3]`, float32, sRGB straight, values 0.0-1.0.
- `img_size` - model resolution.
- `source_h`, `source_w` - original frame dimensions carried through for stage 5.

## Model Architecture

The model is GreenFormer, built on the Hiera encoder from Meta.

The forward pass runs in three steps:

1. Hiera encoder processes the `[1, 4, 2048, 2048]` input tensor and produces four feature maps at scales `[112, 224, 448, 896]` channels with spatial sizes `[512, 256, 128, 64]` at 2048 input resolution.
2. Alpha decoder and foreground decoder run in parallel. Each combines all four feature map scales using a Feature Pyramid Network approach and upsamples to full model resolution. Output is coarse alpha and coarse foreground.
3. CNN refiner takes the original RGB input alongside the coarse predictions and outputs delta corrections. The final result is `sigmoid(coarse_logits + delta_logits * refiner_scale)`.

The refiner corrects edge precision. The decoders understand what is foreground vs background. The refiner corrects exactly where the boundary is.

## Floating Point Precision

The forward pass runs under `torch.autocast`. PyTorch decides per-operation whether to use fp16 or fp32 based on numerical stability requirements. Matrix multiplications and convolutions run in fp16 (or bf16 on supported hardware). Softmax, layer norm, and exponentials run in fp32.

The model precision (weight dtype) is set at engine creation time via the `precision` parameter on `create_engine()`. See [precision modes](../improvements/precision-modes.md) for details.

## refiner_scale

`refiner_scale` controls how much of the CNN refiner's correction is applied.

| Value | Effect |
|---|---|
| 1.0 (default) | Full refiner correction. Best quality. Use for all production work. |
| 0.5 | Half correction. Useful if the refiner is over-correcting on certain footage. |
| 0.0 | Refiner output discarded. Raw decoder output only. Fastest. |

Do not use values above 1.0 in production. They amplify corrections beyond the training distribution and risk artifacts.

## Optimization Mode

`optimization_mode` controls two things: whether the CNN refiner tiles, and whether `torch.compile` is active. The encoder and decoders are unaffected.

| Mode | Refiner tiling | torch.compile | When to use |
|---|---|---|---|
| auto (default) | Probes VRAM, picks speed or lowvram | On | Right choice for most users |
| speed | Off - full 2048x2048 in one pass | On | GPUs with enough VRAM |
| lowvram | On - 512x512 tiles, 128px overlap | On | GPUs with limited VRAM (8 GB class) |

MPS (Apple Silicon via PyTorch) always forces `lowvram` and disables `torch.compile`. Triton does not support Metal.

See [optimization modes](../improvements/optimization-modes.md) for full details.

## Source Code

- Stage function: `stage_4_infer` in [corridorkey-core/stages.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/stages.py)
- Engine: `CorridorKeyEngine` in [corridorkey-core/inference_engine.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/inference_engine.py)
- Engine factory: `create_engine` in [corridorkey-core/engine_factory.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/engine_factory.py)
- Contract: `RawPrediction` in [corridorkey-core/stages.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/stages.py)

## Related Documents

- [Preprocess](preprocess.md) - Stage 3, which produces the tensor this stage consumes.
- [Postprocess](postprocess.md) - Stage 5, which consumes the raw predictions this stage produces.
- [RawPrediction contract](../contracts/raw-prediction.md)
- [Optimization modes](../improvements/optimization-modes.md)
- [Precision modes](../improvements/precision-modes.md)
