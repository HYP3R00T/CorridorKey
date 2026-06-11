# Engine

The top-level pipeline orchestrator. Construct with a loaded config, register an alpha generator and event handlers, then call `run()`.

Construction is cheap — no I/O happens until `run()`.

## Reference

::: corridorkey.engine.Engine

## Usage

### Minimal

```python
from pathlib import Path
from corridorkey import Engine, load_config

config = load_config()
engine = Engine(config)
engine.set_alpha_generator(MyAlphaGenerator())
stats = engine.run([Path("/path/to/clips")])
print(f"Processed {stats.clips_processed} clips, {stats.total_frames} frames")
```

### With event handlers

```python
from corridorkey import Engine, load_config

config = load_config()
engine = Engine(config)
engine.set_alpha_generator(MyAlphaGenerator())

engine.on("clip_found", lambda clip: print(f"Found: {clip.name}"))
engine.on("frame_done", lambda i, total: print(f"  {i + 1}/{total}"))
engine.on("clip_complete", lambda manifest: print(f"Done: {manifest.clip_name}"))
engine.on("clip_error", lambda stage, exc: print(f"Error at {stage}: {exc}"))

stats = engine.run([Path("/path/to/clips")])
```

### Cancellation

```python
import threading
from corridorkey import Engine, load_config

config = load_config()
engine = Engine(config)
engine.set_alpha_generator(MyAlphaGenerator())

# Cancel after 30 seconds from another thread
timer = threading.Timer(30.0, engine.cancel)
timer.start()

stats = engine.run([Path("/path/to/clips")])
timer.cancel()
```

## Events

Register handlers with `engine.on(event, handler)`. Handlers run on the thread that fires them — keep them fast or dispatch to a queue.

| Event | Arguments | When |
|---|---|---|
| `job_started` | — | Before the first clip is scanned |
| `job_complete` | `stats: JobStats` | After all clips are processed |
| `clip_found` | `clip: Clip` | Each valid clip discovered during scan |
| `clip_skipped` | `clip_or_skipped, reason: str` | Clip skipped (already complete or scan error) |
| `clip_loading` | `clip: Clip` | Before the loader runs |
| `clip_ready` | `manifest: ClipManifest` | After load and alpha, before inference |
| `clip_complete` | `manifest: ClipManifest` | After all frames written |
| `clip_cancelled` | `clip: Clip` | Clip interrupted by `cancel()` |
| `clip_error` | `stage: str, exc: Exception` | Clip failed at a named stage |
| `frame_done` | `index: int, total: int` | One frame written to disk |
| `frame_error` | `stage: str, index: int, exc: Exception` | One frame skipped due to error |
| `model_loading` | — | Before model verification |
| `model_ready` | — | After model is verified and ready |
| `download_progress` | `done: int, total: int` | During model download |
| `alpha_resolved` | `manifest: ClipManifest` | After alpha generator completes |
| `stage_start` | `stage: str, total: int` | A named pipeline stage begins |
| `stage_done` | `stage: str` | A named pipeline stage finishes |
| `queue_depth` | `preprocess_q: int, postwrite_q: int` | After each frame moves between stages |
