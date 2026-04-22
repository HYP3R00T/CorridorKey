# Scanner

Discovers clips from one or more paths.

## Reference

::: corridorkey.stages.scanner.orchestrator.scan

::: corridorkey.stages.scanner.contracts.ScanResult

::: corridorkey.stages.scanner.contracts.Clip

::: corridorkey.stages.scanner.contracts.SkippedClip

## Usage

```python
from pathlib import Path
from corridorkey.stages.scanner.orchestrator import scan

result = scan([Path("/path/to/clips")])

print(f"Found {result.clip_count} clips, skipped {result.skipped_count}")

for clip in result.clips:
    print(f"  {clip.name}: input={clip.input_path}, alpha={clip.alpha_path}")

for skipped in result.skipped:
    print(f"  Skipped {skipped.path}: {skipped.reason}")
```

## What scan accepts

`scan()` accepts a list of paths. Each path may be:

- A clips root directory — all immediate subfolders are treated as clips
- A single clip folder — processed as one clip
- A single video file — processed as one clip

Paths that do not match any recognised structure are returned as `SkippedClip` entries with a reason.
