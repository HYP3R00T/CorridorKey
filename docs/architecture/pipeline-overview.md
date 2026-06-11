# Pipeline Overview

The CorridorKey pipeline processes one clip at a time through six stages. Each stage has a single responsibility and communicates with the next through an immutable data contract.

## Stages

```
scan → load → preprocess → inference → postprocess → write
```

**Stage 0 — scan** (`corridorkey.stages.scanner`)

Discovers clips from one or more paths. Accepts a clips root directory, a single clip folder, or a single video file. Returns a `ScanResult` containing valid `Clip` objects and any `SkippedClip` entries with reasons.

**Stage 1 — load** (`corridorkey.stages.loader`)

Validates a `Clip` and produces a `ClipManifest`. Extracts video frames to disk if the input is a video file. Resolves the output directory. Sets `needs_alpha=True` if no alpha hint frames are found.

**Stage 2 — alpha (pluggable)**

If `manifest.needs_alpha` is `True`, the Engine calls the registered `AlphaGenerator` plugin. The plugin generates alpha hint frames and returns an updated manifest with `needs_alpha=False`. This stage is skipped when alpha hint frames are already present.

**Stage 3 — preprocess** (`corridorkey.stages.preprocessor`)

Reads one frame at a time, resizes it to the model's input resolution, normalises pixel values, and stacks the source frame with the alpha hint into a 4-channel tensor. Returns a `PreprocessedFrame`.

**Stage 4 — inference** (`corridorkey.stages.inference`)

Runs the neural network on a `PreprocessedFrame`. Returns an `InferenceResult` containing the predicted alpha matte and foreground colour as tensors.

**Stage 5 — postprocess** (`corridorkey.stages.postprocessor`)

Converts inference tensors to numpy arrays, upscales to the original frame resolution, applies despill, despeckle, source passthrough, and hint sharpening. Returns a `ProcessedFrame`.

**Stage 6 — write** (`corridorkey.stages.writer`)

Writes the `ProcessedFrame` outputs to disk. Writes alpha, foreground, processed RGBA, and composite preview to subdirectories under `manifest.output_dir`.

## Engine coordination

The `Engine` class coordinates all stages. It owns the alpha slot (pluggable), fires events at each stage boundary, and handles cancellation. Stages 3-6 run in a threaded assembly line — see [Runner and Concurrency](runner-and-concurrency.md).

## Config flow

`CorridorKeyConfig` is the single entry point for all configuration. It is loaded once at startup and converted to stage-specific configs via bridge methods (`to_pipeline_config`, `to_preprocess_config`, etc.). Stage configs are immutable dataclasses — they are never modified after construction.

See [Stage Contracts](stage-contracts.md) for the data contract pattern.
