# Configuration

CorridorKey is configured through `CorridorKeyConfig`, a Pydantic model that loads settings from multiple sources in priority order.

## Resolution Order

Settings are resolved from lowest to highest priority:

1. Model field defaults (defined in `CorridorKeyConfig`).
2. Global user config at `~/.config/corridorkey/corridorkey.yaml`.
3. Project config at `./corridorkey.yaml` in the current working directory.
4. Environment variables prefixed with `CORRIDORKEY_`.
5. Runtime overrides passed to `load_config(overrides={...})`.

A higher-priority source always wins. CLI flags are passed as runtime overrides and take precedence over everything.

## Config File Format

```yaml
checkpoint_dir: ~/studio/shared/corridorkey/models
device: cuda
despill_strength: 0.85
fg_format: exr
```

Run `corridorkey config init` to generate a config file at the default location.

## All Parameters

### Paths

| Parameter | Default | Description |
|---|---|---|
| `app_dir` | `~/.config/corridorkey` | Root directory for all tool-managed files. Created on first use. |
| `checkpoint_dir` | `~/.config/corridorkey/models` | Directory where model checkpoints are stored. Override to a shared network path in studio environments. |
| `model_download_url` | null | URL to download the inference model checkpoint. Defaults to the official release URL. |
| `model_filename` | null | Expected filename of the downloaded checkpoint. |

### Device and Engine

| Parameter | Default | Options | Description |
|---|---|---|---|
| `device` | `auto` | `auto`, `cuda`, `mps`, `cpu` | Compute device. `auto` selects the best available. |
| `optimization_mode` | `auto` | `auto`, `speed`, `lowvram` | CNN refiner tiling strategy. See [optimization modes](../improvements/optimization-modes.md). |
| `precision` | `auto` | `auto`, `fp16`, `bf16`, `fp32` | Inference float format. See [precision modes](../improvements/precision-modes.md). |

### Inference Parameters

These become the defaults for `InferenceParams` when calling `service.default_inference_params()`.

| Parameter | Default | Description |
|---|---|---|
| `input_is_linear` | false | Treat input frames as linear light (for example, EXR from a VFX pipeline). |
| `despill_strength` | 1.0 | Green spill suppression strength. 0.0 means no despill. 1.0 means full correction. |
| `auto_despeckle` | true | Remove small disconnected alpha islands from the matte. |
| `despeckle_size` | 400 | Minimum connected region area in pixels to keep. |
| `refiner_scale` | 1.0 | Scale factor for the CNN refiner's delta corrections. 1.0 is full correction. 0.0 skips the refiner output. |
| `source_passthrough` | false | Use original source pixels in fully opaque interior regions. See [source passthrough](../improvements/source-passthrough.md). |
| `edge_erode_px` | 3 | Interior mask erosion radius for source passthrough. |
| `edge_blur_px` | 7 | Transition seam blur radius for source passthrough. |

### Output Formats

These become the defaults for `OutputConfig` when calling `service.default_output_config()`.

| Parameter | Default | Options | Description |
|---|---|---|---|
| `fg_format` | `exr` | `exr`, `png` | Foreground output format. |
| `matte_format` | `exr` | `exr`, `png` | Alpha matte output format. |
| `comp_format` | `png` | `exr`, `png` | Composite preview output format. |
| `processed_format` | `png` | `exr` | Processed RGBA output format. EXR only. |
| `exr_compression` | `dwaa` | `dwaa`, `piz`, `zip`, `none` | EXR compression codec applied to all EXR outputs. |

## CLI Flags

All parameters are also exposed as CLI flags on the `corridorkey process` command. CLI flags are the highest-priority override.

| Flag | Config field |
|---|---|
| `--device` | `device` |
| `--opt-mode` | `optimization_mode` |
| `--precision` | `precision` |
| `--despill` | `despill_strength` |
| `--despeckle / --no-despeckle` | `auto_despeckle` |
| `--despeckle-size` | `despeckle_size` |
| `--refiner` | `refiner_scale` |
| `--linear` | `input_is_linear` |
| `--source-passthrough / --no-source-passthrough` | `source_passthrough` |
| `--edge-erode` | `edge_erode_px` |
| `--edge-blur` | `edge_blur_px` |
| `--fg-format` | `fg_format` |
| `--matte-format` | `matte_format` |
| `--comp-format` | `comp_format` |
| `--exr-compression` | `exr_compression` |

## Environment Variables

Any config field can be set via an environment variable prefixed with `CORRIDORKEY_`. For example:

```shell
CORRIDORKEY_DEVICE=cuda
CORRIDORKEY_OPTIMIZATION_MODE=lowvram
CORRIDORKEY_PRECISION=bf16
```

Two additional environment variables control engine behaviour directly and bypass the config system:

- `CORRIDORKEY_OPT_MODE` - overrides `optimization_mode` at the engine level.
- `CORRIDORKEY_BACKEND` - forces the inference backend (`torch` or `mlx`).

## Source Code

- Config model: `CorridorKeyConfig` in [corridorkey/config.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/config.py)
- CLI flags: `process` command in [corridorkey-cli/commands/process.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-cli/src/corridorkey_cli/commands/process.py)

## Related Documents

- [Optimization modes](../improvements/optimization-modes.md)
- [Precision modes](../improvements/precision-modes.md)
- [Source passthrough](../improvements/source-passthrough.md)
