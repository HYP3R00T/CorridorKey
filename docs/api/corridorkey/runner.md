# Runner

## JobStats

Summary of a completed `Engine.run()` call. Returned by `engine.run()` and also passed to the `job_complete` event handler.

::: corridorkey.runtime.job_stats.JobStats

## PipelineEvents

Low-level event callback container for direct use of `run_clip()` or `run_clips()`. When using `Engine`, register handlers with `engine.on()` instead — the Engine wires `PipelineEvents` internally.

::: corridorkey.events.PipelineEvents

## Direct pipeline access

`run_clip()` and `run_clips()` are available for integrations that manage the pipeline directly without the Engine. This is an advanced use case — most integrators should use `Engine`.

```python
from pathlib import Path
from corridorkey import load_config
from corridorkey.events import PipelineEvents
from corridorkey.runtime.runner import run_clip
from corridorkey.stages.loader.orchestrator import load
from corridorkey.stages.scanner.orchestrator import scan

config = load_config()
pipeline_config = config.to_pipeline_config(device="cuda")

result = scan([Path("/path/to/clips")])
for clip in result.clips:
    manifest = load(clip)
    events = PipelineEvents(
        on_frame_written=lambda i, total: print(f"  {i + 1}/{total}"),
    )
    run_clip(manifest, pipeline_config, events=events)
```
