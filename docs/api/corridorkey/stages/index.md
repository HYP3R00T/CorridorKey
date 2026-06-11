# Stages

The pipeline stages are available as individual functions for integrations that need fine-grained control. When using `Engine`, all stages are called automatically — you do not need to call them directly.

| Stage | Function | Input | Output |
|---|---|---|---|
| [Scanner](scanner.md) | `scan(paths)` | `list[Path]` | `ScanResult` |
| [Loader](loader.md) | `load(clip)` | `Clip` | `ClipManifest` |
| [Preprocessor](preprocessor.md) | `preprocess_frame(manifest, i, config)` | `ClipManifest` | `PreprocessedFrame` |
| [Inference](inference.md) | `backend.run(frame)` | `PreprocessedFrame` | `InferenceResult` |
| [Postprocessor](postprocessor.md) | `postprocess_frame(result, config)` | `InferenceResult` | `ProcessedFrame` |
| [Writer](writer.md) | `write_frame(frame, config)` | `ProcessedFrame` | — |

## Using stages directly

```python
from pathlib import Path
from corridorkey import load_config
from corridorkey.stages.scanner.orchestrator import scan
from corridorkey.stages.loader.orchestrator import load
from corridorkey.stages.preprocessor.orchestrator import preprocess_frame
from corridorkey.stages.postprocessor.orchestrator import postprocess_frame
from corridorkey.stages.writer.orchestrator import write_frame
from corridorkey.stages.loader.validator import list_frames

config = load_config()
device = "cuda"

pipeline_config = config.to_pipeline_config(device=device)
preprocess_config = config.to_preprocess_config(device=device, resolved_img_size=2048)
postprocess_config = config.to_postprocess_config()

result = scan([Path("/path/to/clips")])
for clip in result.clips:
    manifest = load(clip)
    write_config = config.to_writer_config(manifest.output_dir)

    image_files = list_frames(manifest.frames_dir)
    alpha_files = list_frames(manifest.alpha_frames_dir)

    for i in range(*manifest.frame_range):
        preprocessed = preprocess_frame(manifest, i, preprocess_config,
                                        image_files=image_files, alpha_files=alpha_files)
        # inference omitted — use Engine or run_clip for the full pipeline
```
