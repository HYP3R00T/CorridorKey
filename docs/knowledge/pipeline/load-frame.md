# Load Frame

Stage 1 reads one image frame and its corresponding alpha hint mask from disk and returns them as normalised float32 arrays.

This stage is the only place in the pipeline where files are read. Everything downstream works with in-memory arrays.

## What It Does

1. Reads the source image using OpenCV with `IMREAD_UNCHANGED` to preserve bit depth.
2. Strips the alpha channel if the image has four channels (RGBA).
3. Converts from BGR (OpenCV default) to RGB.
4. If the source is an EXR file and `input_is_linear` is true, converts from linear light to sRGB using the IEC 61966-2-1 transfer function.
5. If the source is a standard image (PNG, TIFF, JPEG), normalises from uint8 0-255 to float32 0.0-1.0.
6. Reads the mask with `IMREAD_ANYDEPTH` to support 8-bit, 16-bit, and float masks.
7. Normalises the mask to float32 0.0-1.0 based on its dtype.
8. Reduces the mask to a single channel and reshapes to `[H, W, 1]`.
9. Returns a `FrameData` carrying both arrays and metadata.

## Input

- `image_path` - path to the source image (PNG, EXR, TIFF, or any format OpenCV supports).
- `mask_path` - path to the alpha hint mask (any single-channel image).
- `input_is_linear` - set true if the source is in linear light (for example, EXR from a VFX pipeline).
- `stem` - filename stem used for output naming downstream. Defaults to the image filename stem.

## Output

A `FrameData` with:

- `image` - RGB float32 `[H, W, 3]`, values 0.0-1.0, always in sRGB regardless of source colour space.
- `mask` - grayscale float32 `[H, W, 1]`, values 0.0-1.0.
- `source_h`, `source_w` - original frame dimensions in pixels.
- `is_linear` - records whether the source was originally in linear light.
- `stem` - filename stem carried through for output naming.

## Colour Space

The image field always contains sRGB. If the source was in linear light, the conversion happens here. All downstream stages receive sRGB and do not need to know the original colour space. The `is_linear` flag is carried through for reference only.

## Mask Values

The mask is a trimap:

- 1.0 (white) - definite foreground.
- 0.0 (black) - definite background.
- 0.5 (grey) - unknown, the model decides.

The model focuses its attention on the unknown region. A reasonable unknown band around subject edges is approximately 20-40 pixels wide at 2048 resolution.

## Source Code

- Stage function: `stage_1_load_frame` in [corridorkey/stages.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/stages.py)
- Contract: `FrameData` in the same file.

## Related Documents

- [Generate masks](generate-masks.md) - Stage 2, which produces the mask files this stage reads.
- [Preprocess](preprocess.md) - Stage 3, which consumes the arrays this stage produces.
- [FrameData contract](../contracts/frame-data.md)
