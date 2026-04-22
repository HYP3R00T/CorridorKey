# Stage Contracts

Each pipeline stage communicates with the next through an immutable data contract. This pattern keeps stages decoupled — a stage only knows about its input and output types, not about the stages before or after it.

## The pattern

Every stage folder contains a `contracts.py` module that defines the stage's input and output types as frozen Pydantic models or frozen dataclasses. The orchestrator (`orchestrator.py`) contains the stage's logic and takes the input contract as its argument.

```
stages/
  scanner/
    contracts.py      # Clip, SkippedClip, ScanResult
    orchestrator.py   # scan(paths) -> ScanResult
  loader/
    contracts.py      # ClipManifest
    orchestrator.py   # load(clip) -> ClipManifest
  ...
```

## Why frozen

Contracts are frozen (immutable after construction). This means:

- A stage cannot accidentally modify data that belongs to another stage.
- Contracts can be safely shared across threads without locks.
- Bugs from mutating shared state are eliminated by construction.

When a stage needs to produce a modified version of a contract (for example, the loader attaches alpha frames to a manifest), it uses `model_copy(update={...})` to produce a new instance rather than mutating the existing one.

## Key contracts

**`Clip`** — output of the scanner, input to the loader. Holds the clip name, root path, input path, and optional alpha path. Validated on construction — all paths must exist.

**`ClipManifest`** — output of the loader, input to all downstream stages. Holds resolved frame paths, output directory, frame count, frame range, and metadata. `needs_alpha=True` signals that alpha hint frames are absent and must be generated before inference can proceed.

**`PreprocessedFrame`** — output of the preprocessor, input to inference. Holds the 4-channel tensor and a `FrameMeta` with the frame index and source resolution.

**`InferenceResult`** — output of inference, input to the postprocessor. Holds the predicted alpha and foreground tensors.

**`ProcessedFrame`** — output of the postprocessor, input to the writer. Holds numpy arrays for alpha, foreground, processed RGBA, and composite preview.

## Config contracts

Stage configs follow the same pattern. Each stage has an internal config dataclass (e.g. `PreprocessConfig`, `InferenceConfig`) that is constructed by `CorridorKeyConfig`'s bridge methods. These are separate from the user-facing settings models (`PreprocessSettings`, `InferenceSettings`) that live in the config layer and are loaded from TOML.

The separation means the config layer can resolve "auto" values (device, img_size, precision) once at startup and pass concrete values to the stages. Stages never see "auto" — they always receive a resolved value.
