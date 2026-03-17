# Preprocess

Stage 3 takes the image and mask arrays from stage 1 and prepares a single tensor ready for model inference.

This is the first stage in `corridorkey-core`. It has no filesystem access. Input and output are in-memory arrays and tensors only.

## What It Does

1. Resizes the image to `img_size x img_size` (default 2048) using bilinear interpolation.
2. Resizes the mask to the same dimensions.
3. Applies ImageNet normalisation to the image: `(pixel - mean) / std`.
4. Concatenates the normalised image and mask into a single four-channel array.
5. Transposes to channels-first layout for the model.
6. Moves the tensor to the target compute device.

## Input

- The image and mask arrays from stage 1.
- The target square resolution (default 2048, must match the model's training resolution).
- The compute device to place the tensor on.
- The original frame dimensions, carried through for upsampling in stage 5.

## Output

A `PreprocessedTensor` contract carrying:

- A float32 tensor `[1, 4, img_size, img_size]` on the target device.
- The resolution the tensor was prepared at.
- The device identifier.
- The original frame dimensions carried through for stage 5.

## ImageNet Normalisation

The model's encoder (Hiera) was pretrained on ImageNet. Its weights only understand inputs in the normalised distribution it was trained on. Normalisation is an input contract, not an optimisation.

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
normalised = (pixel - mean) / std
```

The mask is not normalised. It is concatenated as-is as the fourth channel.

## Resize Strategy

The current implementation squishes the frame to `img_size x img_size` regardless of aspect ratio. This distorts non-square footage but the model is robust enough that mild distortion does not break results.

Letterboxing (padding the shorter dimension to preserve aspect ratio) is a planned improvement. See [improvements](../improvements/index.md) for details.

## Model Resolution

The model was trained at 2048x2048. This is the resolution all input frames are resized to before inference. VRAM usage scales with the square of this value. For high-resolution footage, tiling is the correct approach rather than increasing the resolution.

## Related Documents

- [Load frame](load-frame.md) - Stage 1, which produces the arrays this stage consumes.
- [Infer](infer.md) - Stage 4, which consumes the tensor this stage produces.
- [PreprocessedTensor contract](../contracts/preprocessed-tensor.md)
- [Improvements - letterboxing and tiling](../improvements/index.md)
