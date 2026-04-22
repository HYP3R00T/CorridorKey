# Device and Backends

## Device resolution

The `device` config field accepts a string that is resolved to a concrete PyTorch device at startup. Resolution happens once in `Engine._initialise()` before any clip is processed.

| Value | Resolves to |
|---|---|
| `auto` | Best available: ROCm > CUDA > MPS > CPU |
| `cuda` | CUDA device 0 |
| `cuda:N` | CUDA device N |
| `rocm` | ROCm device 0 |
| `rocm:N` | ROCm device N |
| `mps` | Apple Silicon GPU |
| `cpu` | CPU |
| `all` | All available CUDA devices |

`resolve_device(requested)` validates the string and returns a concrete device. `resolve_devices("all")` returns a list of all CUDA device strings.

## Multi-GPU dispatch

When `device = "all"`, the Engine calls `resolve_devices("all")` to get a list of all CUDA device strings. One `_InferenceWorker` thread is spawned per device. All workers pull from a shared preprocess queue, so faster GPUs can process frames from any clip without waiting for clip boundaries.

Model loading for multi-GPU runs happens in parallel — one thread per device, with a 300-second timeout. Models are not shared across devices.

## The AlphaGenerator protocol

The alpha slot is pluggable. Any object that implements the `AlphaGenerator` protocol can be registered:

```python
class AlphaGenerator(Protocol):
    def generate(self, manifest: ClipManifest) -> ClipManifest: ...
```

No inheritance is required — structural subtyping only. The Engine checks for the `generate` method at registration time and raises `EngineError` if it is absent.

The plugin receives a `ClipManifest` with `needs_alpha=True` and must return a manifest with `needs_alpha=False` and `alpha_frames_dir` set to the generated frames directory.

## Optional plugin config

A plugin can declare a `Config` inner class (Pydantic `BaseModel`) to receive validated settings from the `[plugins.alpha]` TOML section:

```python
class MyAlphaGenerator:
    class Config(BaseModel):
        sensitivity: float = 0.7

    def generate(self, manifest: ClipManifest) -> ClipManifest:
        threshold = self.config.sensitivity
        ...
```

The Engine wires `plugin.config` from the TOML section at registration time. If no section exists, schema defaults are used.

## VRAM probing

When `img_size = 0` (auto) or `refiner_mode = "auto"`, the config bridge method probes available VRAM once using `pynvml`. The same measurement resolves both values, avoiding two separate hardware queries at startup.

The VRAM thresholds are:

- Under 6 GB: `img_size = 1024`
- 6-12 GB: `img_size = 1536`
- 12+ GB: `img_size = 2048`
- Under 12 GB: `refiner_mode = tiled`
- 12+ GB: `refiner_mode = full_frame`

On MPS (Apple Silicon), `refiner_mode` is always `tiled` regardless of available memory.
