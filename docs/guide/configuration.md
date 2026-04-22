# Configuration

CorridorKey loads configuration from multiple sources. Values from higher-priority sources override lower-priority ones.

## Priority order (lowest to highest)

1. Built-in defaults
2. `~/.config/corridorkey/corridorkey.yaml` — global user config
3. `./corridorkey.yaml` — project-local config (current working directory)
4. `CK_*` environment variables

## Config file

Run `ck config --write` to write the current resolved config to disk:

```shell
ck config --write
```

The file is written to `~/.config/corridorkey/corridorkey.yaml`. Edit it with any text editor.

To view the resolved config with source attribution (which value came from which source):

```shell
ck config
```

## Full reference

```yaml
device: auto  # auto / cuda / cuda:N / rocm / rocm:N / mps / cpu / all

logging:
  level: INFO
  dir: ~/.config/corridorkey/logs

preprocess:
  img_size: 0             # 0=auto, or 512 / 1024 / 1536 / 2048
  image_upsample_mode: bicubic   # bicubic / bilinear
  sharpen_strength: 0.3   # 0.0–1.0
  half_precision: false
  source_passthrough: true

inference:
  checkpoint_path: ~/.config/corridorkey/models/CorridorKey_v1.0.pth
  use_refiner: true
  mixed_precision: true
  model_precision: auto   # auto / float16 / bfloat16 / float32
  refiner_mode: auto      # auto / full_frame / tiled
  refiner_scale: 1.0      # 0.0–1.0
  flash_attention: auto   # auto / on / off

postprocess:
  fg_upsample_mode: lanczos4      # lanczos4 / bicubic / bilinear
  alpha_upsample_mode: lanczos4   # lanczos4 / bilinear
  despill_strength: 0.5           # 0.0–1.0
  auto_despeckle: true
  despeckle_size: 400
  despeckle_dilation: 25
  despeckle_blur: 5
  source_passthrough: true
  edge_erode_px: 3
  edge_blur_px: 7
  hint_sharpen: true
  hint_sharpen_dilation: 3
  debug_dump: false

writer:
  alpha_enabled: true
  alpha_format: png       # png / exr
  fg_enabled: true
  fg_format: png
  processed_enabled: true
  processed_format: png
  comp_enabled: true
  exr_compression: dwaa   # none / rle / zips / zip / piz / pxr24 / dwaa / dwab
```

## Environment variables

Every config field can be overridden with a `CK_` prefixed environment variable. Nested fields use double underscores.

```shell
CK_DEVICE=cuda
CK_INFERENCE__MODEL_PRECISION=float16
CK_WRITER__PROCESSED_FORMAT=exr
```

## Key settings explained

**`device`** — which GPU (or CPU) to use. `auto` picks the best available device at runtime: ROCm > CUDA > MPS > CPU. Use `cuda:1` to target a specific GPU by index. Use `all` to run across all CUDA GPUs in parallel.

**`preprocess.img_size`** — the square resolution the model runs at. `0` (default) auto-selects based on available VRAM: under 6 GB gives 1024, 6-12 GB gives 1536, 12+ GB gives 2048. 2048 is the native training resolution and produces the best output.

**`inference.refiner_mode`** — controls how the CNN refiner executes. `auto` probes VRAM and picks `full_frame` (12+ GB) or `tiled` (under 12 GB). Output quality is identical for both modes.

**`writer.processed_format`** — format for the premultiplied RGBA output. Use `exr` for compositing workflows that require linear light and full float precision. Use `png` for 16-bit output.

**`postprocess.despill_strength`** — how aggressively green spill is suppressed. `0.0` disables despill entirely. `1.0` applies full suppression. Start at `0.5` and adjust to taste.

**`postprocess.debug_dump`** — when `true`, saves intermediate PNG snapshots after each postprocessing step to a `debug/` subfolder. Use this to diagnose whether quality issues come from the model or postprocessing.
