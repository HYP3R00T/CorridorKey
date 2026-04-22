# Errors

All CorridorKey exceptions inherit from `CorridorKeyError`. Catch the base class when you do not need to distinguish between subtypes.

## Hierarchy

```
CorridorKeyError
├── EngineError                 engine: contract violations before processing starts
├── AlphaGeneratorError         engine: alpha slot unfilled or bad return
├── ClipScanError               scanner: path/structure problems, permission errors
├── ClipLoadError               loader: empty input, output dir creation failure
├── ExtractionError             loader: video extraction failures
├── FrameMismatchError          loader: input/alpha count mismatch
├── FrameReadError              preprocessor: frame file unreadable
├── InferenceError              inference: model forward pass failure
├── VRAMInsufficientError       inference: CUDA out of memory
├── PostprocessError            postprocessor: despill/despeckle/composite failure
├── WriteFailureError           writer: write failure (cv2 or OS)
├── DeviceError                 infra: requested device unavailable
├── ModelError                  infra: model download, load, or checksum failure
└── JobCancelledError           pipeline: job cancelled by the caller
```

## Reference

::: corridorkey.errors.CorridorKeyError

::: corridorkey.errors.EngineError

::: corridorkey.errors.AlphaGeneratorError

::: corridorkey.errors.ClipScanError

::: corridorkey.errors.ClipLoadError

::: corridorkey.errors.ExtractionError

::: corridorkey.errors.FrameMismatchError

::: corridorkey.errors.FrameReadError

::: corridorkey.errors.InferenceError

::: corridorkey.errors.VRAMInsufficientError

::: corridorkey.errors.PostprocessError

::: corridorkey.errors.WriteFailureError

::: corridorkey.errors.DeviceError

::: corridorkey.errors.ModelError

::: corridorkey.errors.JobCancelledError

## Handling errors

```python
from corridorkey import Engine, load_config
from corridorkey.errors import (
    CorridorKeyError,
    VRAMInsufficientError,
    ModelError,
    AlphaGeneratorError,
)

config = load_config()
engine = Engine(config)
engine.set_alpha_generator(MyAlphaGenerator())

engine.on("clip_error", lambda stage, exc: handle_clip_error(stage, exc))

def handle_clip_error(stage: str, exc: Exception) -> None:
    if isinstance(exc, VRAMInsufficientError):
        print("Not enough VRAM — try a smaller img_size or tiled refiner mode")
    elif isinstance(exc, AlphaGeneratorError):
        print(f"Alpha generation failed: {exc}")
    elif isinstance(exc, CorridorKeyError):
        print(f"Pipeline error at {stage}: {exc}")
    else:
        raise exc  # unexpected — re-raise

try:
    stats = engine.run([Path("/clips")])
except ModelError as e:
    print(f"Model could not be loaded: {e}")
```
