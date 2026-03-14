# Backend Selection

`corridorkey-core` supports two inference backends: Torch and MLX. The backend is resolved once at engine creation time and is transparent to callers - `process_frame` behaves identically regardless of which backend is active.

## Resolution Order

The backend is resolved in this priority order:

1. Explicit `backend` argument passed to `create_engine`
2. `CORRIDORKEY_BACKEND` environment variable
3. Auto-detect

## Auto-detect Logic

Auto-detect selects MLX when all three conditions are true:

1. The platform is `darwin` (macOS)
2. The machine architecture is `arm64` (Apple Silicon)
3. The `corridorkey-mlx` package is importable

Otherwise Torch is used.

## Selecting a Backend Explicitly

```python
from corridorkey_core import create_engine

# Force Torch
engine = create_engine("/path/to/checkpoints", backend="torch")

# Force MLX (Apple Silicon only - raises RuntimeError on other platforms)
engine = create_engine("/path/to/checkpoints", backend="mlx")
```

## Selecting via Environment Variable

```shell
CORRIDORKEY_BACKEND=mlx uv run your_script.py
CORRIDORKEY_BACKEND=torch uv run your_script.py
```

The environment variable is read at engine creation time, not at import time.

## Checkpoint Layout

Each backend expects a different checkpoint file format in the same directory:

| Backend | Extension | Example |
|---|---|---|
| Torch | `.pth` | `greenformer_v2.pth` |
| MLX | `.safetensors` | `greenformer_v2.safetensors` |

`create_engine` scans `checkpoint_dir` for exactly one file with the matching extension. It raises `FileNotFoundError` if none are found and `ValueError` if more than one are found. If you have the wrong extension for your backend, the error message will suggest the correct `--backend` flag.

Place only one checkpoint file per backend in the directory:

```text
checkpoints/
    greenformer_v2.pth           # used by Torch
    greenformer_v2.safetensors   # used by MLX
```

## MLX Adapter

The MLX backend (`corridorkey-mlx`) returns uint8 arrays and does not apply despill or despeckle natively. `corridorkey-core` wraps the MLX engine in an adapter that:

1. Converts float32 inputs to uint8 before calling the MLX engine
2. Applies despill and despeckle in Python after receiving the uint8 output
3. Converts uint8 outputs back to float32

The adapter is transparent. Callers always receive the same float32 output contract.

## Device Selection (Torch Only)

The `device` argument to `create_engine` is passed directly to PyTorch. It has no effect on the MLX backend.

```python
# CUDA
engine = create_engine("/path/to/checkpoints", device="cuda")

# Specific GPU
engine = create_engine("/path/to/checkpoints", device="cuda:1")

# CPU (default)
engine = create_engine("/path/to/checkpoints", device="cpu")
```

## Image Size

`img_size` controls the square resolution the model runs at internally. Inputs are resized to this resolution before inference and outputs are resized back to the original frame size.

The default is `2048`. Reducing it speeds up inference at the cost of detail in fine edges.

```python
engine = create_engine("/path/to/checkpoints", img_size=1024)
```

## Related

- [create_engine reference](../../api/corridorkey-core/index.md)
- [Output contract](output-contract.md)
