# Engine Presets

When you run `ck` interactively, the wizard asks you to pick an engine preset. Presets are shortcuts that set `refiner_mode`, `model_precision`, and `img_size` together.

## Preset table

| Preset | refiner_mode | model_precision | img_size | Use when |
|---|---|---|---|---|
| `full_frame` | full_frame | float16 | 1024 | Fast turnaround, lower VRAM |
| `balanced` | auto | auto | 1536 | General purpose — good default |
| `quality` | full_frame | bfloat16 | 2048 | High quality, 12+ GB VRAM |
| `max_quality` | full_frame | float32 | 2048 | Maximum quality, 16+ GB VRAM |
| `tiled` | tiled | float16 | 1024 | Low VRAM GPUs (under 8 GB) |

## How to choose

**`balanced`** is the right starting point for most work. It auto-detects your hardware and picks sensible values.

**`quality`** or **`max_quality`** are worth trying when you need the sharpest possible edges and have the VRAM to support them. `img_size=2048` is the native training resolution.

**`full_frame`** is useful when you want a fast preview pass at lower resolution before committing to a full-quality run.

**`tiled`** is for GPUs with under 8 GB VRAM. The refiner runs in 512x512 overlapping tiles instead of on the full frame, keeping peak VRAM flat. Output quality is identical to `full_frame`.

## Manual settings

Select `manual` in the wizard to set `refiner_mode`, `model_precision`, and `img_size` individually. This is useful when you want `img_size=2048` with `tiled` refiner mode, for example.

## Saving a preset as your default

Run `ck config --write` after a run to save the current settings to your config file. The next time you run `ck /path --yes`, those values will be used without prompting.
