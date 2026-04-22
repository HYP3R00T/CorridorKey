# Loader

Validates a `Clip` and produces a `ClipManifest`. Extracts video frames to disk if the input is a video file.

## Reference

::: corridorkey.stages.loader.orchestrator.load

::: corridorkey.stages.loader.contracts.ClipManifest

## Usage

```python
from corridorkey.stages.loader.orchestrator import load
from corridorkey.stages.scanner.orchestrator import scan
from pathlib import Path

result = scan([Path("/path/to/clips")])
for clip in result.clips:
    manifest = load(clip)

    print(f"Clip: {manifest.clip_name}")
    print(f"  frames_dir: {manifest.frames_dir}")
    print(f"  frame_count: {manifest.frame_count}")
    print(f"  needs_alpha: {manifest.needs_alpha}")
    print(f"  output_dir: {manifest.output_dir}")
```

## needs_alpha

If `manifest.needs_alpha` is `True`, the clip has no alpha hint frames. You must either:

- Provide an `AlphaGenerator` plugin via `engine.set_alpha_generator()` (recommended when using `Engine`)
- Generate alpha frames externally and call `manifest.model_copy(update={"alpha_frames_dir": path, "needs_alpha": False})` before proceeding

## frame_range

`manifest.frame_range` is a half-open range `(start, end)`. By default it covers the full sequence `(0, frame_count)`. Narrow it for partial runs:

```python
# Process only the first 100 frames
manifest = manifest.model_copy(update={"frame_range": (0, 100)})
```
