# Write Outputs

Stage 6 writes all enabled output images for one processed frame to disk. It is the only stage that writes files.

## What It Does

For each enabled output type, it:

1. Converts the array to the correct bit depth for the target format (full float for EXR, 8-bit for PNG).
2. Writes the file to disk.
3. Raises an error if the write fails.

All colour space conversions happened in stage 5. This stage only writes.

## Input

A `ProcessedFrame` from stage 5, along with configuration describing which output types to write and where.

## Output Types

| Output | Default format | Colour space | Description |
|---|---|---|---|
| fg | EXR | Linear | Foreground RGB, straight (unpremultiplied). |
| matte | EXR | Linear | Alpha matte, single channel. |
| comp | PNG | sRGB | Checkerboard composite. Preview only, not for production use. |
| processed | EXR | Linear premultiplied | RGBA. Primary deliverable for NLE and VFX. |

Each output can be independently enabled or disabled via `WriteConfig`.

## EXR vs PNG

EXR stores float16 half-precision values. Full dynamic range, no clamping, no quality loss. EXR is the correct format for all production outputs.

PNG is 8-bit. Values are clamped to 0-255. Suitable for the comp preview. Not suitable for compositing.

## EXR Compression

All EXR outputs share the same compression setting from `WriteConfig.exr_compression`.

| Compression | Type | Best for |
|---|---|---|
| dwaa (default) | Lossy wavelet | General use. Small files, fast, visually lossless. |
| piz | Lossless wavelet | Archival or maximum fidelity. |
| zip | Lossless deflate | Compatibility with older tools. |
| none | Uncompressed | Debugging and scratch space only. |

DWAA is the correct default. The compression loss is below the threshold of visual perception for compositing work.

## Output Directory Structure

Each output type writes to its own subdirectory under the clip's Output folder:

```text
Output/
  FG/         frame_000001.exr, frame_000002.exr, ...
  Matte/      frame_000001.exr, frame_000002.exr, ...
  Comp/       frame_000001.png, frame_000002.png, ...
  Processed/  frame_000001.exr, frame_000002.exr, ...
```

Directories are created by `ensure_output_dirs` in `validators.py` before processing begins.

## No Video Reassembly

The pipeline writes frame sequences only. If a video file is needed, run FFmpeg over the output sequence after processing. For the comp preview:

```shell
ffmpeg -framerate 24 -i Output/Comp/%06d.png -c:v libx264 -crf 18 preview.mp4
```

For a ProRes 4444 delivery of the foreground:

```shell
ffmpeg -framerate 24 -i Output/FG/%06d.exr -c:v prores_ks -profile:v 4444 fg.mov
```

## Related Documents

- [Postprocess](postprocess.md) - Stage 5, which produces the ProcessedFrame this stage writes.
- [ProcessedFrame contract](../contracts/processed-frame.md)
- [Configuration](../configuration/index.md) - Output format and compression settings.
