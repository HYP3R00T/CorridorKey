# Quick Start

This guide takes you from a fresh install to a processed clip.

## 1. Run first-time setup

```shell
ck init
```

This checks your environment and downloads the inference model if it is not already present. You only need to do this once.

## 2. Organise your clips

CorridorKey expects a specific directory layout. Each clip lives in its own folder:

```text
clips/
  MyShot/
    Input/
      frame_000000.jpg
      frame_000001.jpg
      ...
    AlphaHint/        (optional — pre-generated alpha matte)
      frame_000000.png
      ...
```

Video files are also accepted inside `Input/`:

```text
clips/
  MyShot/
    Input/
      footage.mp4
```

See [Clips Layout](clips-layout.md) for the full specification.

## 3. Process your clips

```shell
ck /path/to/clips
```

The wizard will show your current config, ask you to pick an engine preset, and then process all clips it finds.

To skip the prompts and use your config defaults:

```shell
ck /path/to/clips --yes
```

## 4. Find your outputs

Outputs are written to `ClipName/Output/` alongside the input:

```text
clips/
  MyShot/
    Output/
      alpha/        alpha matte frames
      fg/           foreground colour frames
      processed/    premultiplied RGBA (primary compositor output)
      comp/         checkerboard preview composite
```

## What to do if something goes wrong

- Run `ck init` to verify your environment.
- Check [Troubleshooting](troubleshooting.md) for common errors.
- Adjust quality settings with [Engine Presets](presets.md).
