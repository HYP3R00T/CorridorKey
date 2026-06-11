# Writer

Writes a `ProcessedFrame` to disk. Creates output subdirectories on first write.

## Reference

::: corridorkey.stages.writer.orchestrator.write_frame

::: corridorkey.stages.writer.contracts.WriteConfig

## Usage

```python
from pathlib import Path
from corridorkey import load_config
from corridorkey.stages.writer.orchestrator import write_frame

config = load_config()
write_config = config.to_writer_config(manifest.output_dir)

write_frame(processed_frame, write_config)
```

## Output layout

`write_frame` writes to subdirectories under `config.output_dir`:

| Subdirectory | Contents | Controlled by |
|---|---|---|
| `alpha/` | Alpha matte | `alpha_enabled`, `alpha_format` |
| `fg/` | Straight foreground colour | `fg_enabled`, `fg_format` |
| `processed/` | Premultiplied RGBA | `processed_enabled`, `processed_format` |
| `comp/` | Checkerboard preview | `comp_enabled` (always PNG) |

## EXR writing

When writing EXR, CorridorKey uses `pyexr` (OpenEXR) when available, falling back to `cv2`. The `pyexr` path supports all compression codecs correctly. The `cv2` fallback remaps `dwaa` and `dwab` to `piz` (lossless) to avoid a known bug in cv2 4.13 where DWAA/DWAB compression produces corrupt files.

Install `pyexr` to use the full codec set:

```shell
pip install pyexr
```
