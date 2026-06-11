# Clips Layout

CorridorKey expects clips in a specific directory structure. This page describes the full layout and what each folder contains.

## Standard layout

```text
clips/
  ClipName/
    Input/          source frames or video file
    AlphaHint/      (optional) pre-generated alpha matte frames
    Output/         written by CorridorKey — do not put source files here
```

`clips/` is the root you pass to `ck`. Each subfolder is treated as one clip.

## Input

`Input/` holds the source footage. Two formats are accepted:

**Image sequence** — JPEG or PNG files named with a numeric suffix:

```text
Input/
  frame_000000.jpg
  frame_000001.jpg
  ...
```

**Video file** — a single video file in any format FFmpeg can decode:

```text
Input/
  footage.mp4
```

When a video file is found, CorridorKey extracts it to a `Frames/` directory before processing. The original video is not modified.

## AlphaHint (optional)

`AlphaHint/` holds pre-generated alpha matte frames. These are used to guide the model and sharpen edges. If this folder is absent, CorridorKey will ask you to provide an alpha generator plugin before it can proceed.

Alpha hint frames must be grayscale PNG files with the same count and naming as the input frames.

## Output

`Output/` is created by CorridorKey on first write. It contains four subdirectories:

| Folder | Contents |
|---|---|
| `alpha/` | Alpha matte — grayscale, one channel |
| `fg/` | Foreground colour — straight sRGB |
| `processed/` | Premultiplied RGBA — primary compositor output |
| `comp/` | Checkerboard preview composite — sRGB, for visual review |

Which outputs are written depends on your [configuration](configuration.md). All four are enabled by default.

## Multiple clips

Pass a root directory and CorridorKey will scan all immediate subfolders:

```shell
ck /path/to/clips
```

You can also pass a single clip folder directly:

```shell
ck /path/to/clips/MyShot
```

Or a single video file:

```shell
ck /path/to/footage.mp4
```

## Resuming

If a clip's `Output/` directory already contains the expected number of output frames, CorridorKey skips it automatically and reports it as already complete.
