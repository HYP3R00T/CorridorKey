# Optimization Modes

`optimization_mode` controls exactly two things: whether the CNN refiner tiles its computation, and whether `torch.compile` is active. The encoder and decoders are unaffected by this setting.

## The Three Modes

| Mode | Refiner tiling | torch.compile | When to use |
|---|---|---|---|
| `auto` (default) | Probes VRAM, picks speed or lowvram | On | Right choice for most users |
| `speed` | Off - full 2048x2048 in one pass | On | GPUs with enough VRAM to hold the full refiner pass |
| `lowvram` | On - 512x512 tiles, 128px overlap | On | GPUs with limited VRAM (8 GB class) |

## Why the Refiner Needs Tiling

The CNN refiner processes the full 2048x2048 tensor in one pass in `speed` mode. On GPUs with limited VRAM, this tensor does not fit alongside the encoder and decoder outputs already in memory. `lowvram` mode splits the refiner pass into 512x512 tiles with 128px overlap and blends them back using cosine weights. The output is identical to the full-frame pass - the tiling is invisible to the caller.

## Auto-Detection

`auto` mode probes VRAM using `pynvml` at the driver level before the CUDA context is initialised. This matters because a mask generator (for example, GVM) may have just released the GPU before the inference engine initialises. `torch.cuda.get_device_properties()` was found to stall in that handoff window. `pynvml` does not have that problem.

The threshold is 12 GB. Below 12 GB free VRAM, `auto` selects `lowvram`. At or above 12 GB, it selects `speed`.

## MPS Special Case

MPS (Apple Silicon running via PyTorch rather than MLX) always forces `lowvram` and also disables `torch.compile`. Triton's code generation does not support Metal. The MLX backend is the recommended path for Apple Silicon.

## Environment Variable Override

The mode can be overridden via the `CORRIDORKEY_OPT_MODE` environment variable without changing code or config files. Valid values are `auto`, `speed`, and `lowvram`.

```shell
CORRIDORKEY_OPT_MODE=lowvram corridorkey process /path/to/clips
```

## torch.compile

`torch.compile` uses Triton to JIT-compile the model's compute graph into optimised GPU kernels. The first run pays a compilation cost (typically 30-60 seconds). Subsequent runs use the cached compiled kernels and are faster.

The compiled kernel cache is stored at `~/.cache/corridorkey/torch_compile`. If compilation fails for any reason, the engine falls back to eager mode automatically.

## Source Code

- Mode resolution: `CorridorKeyEngine.__init__` in [corridorkey-core/inference_engine.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-core/src/corridorkey_core/inference_engine.py)
- Tiled refiner: `CorridorKeyEngine._run_refiner_tiled` in the same file.
- VRAM probe: `_probe_vram_gb` in the same file.

## Related Documents

- [Configuration](../configuration/index.md) - The `optimization_mode` config parameter.
- [Infer](../pipeline/infer.md) - Stage 4, where optimization mode takes effect.
