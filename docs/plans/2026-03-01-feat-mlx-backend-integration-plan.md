---
title: Integrate MLX inference backend alongside Torch
type: feat
date: 2026-03-01
base: PR #33 (feature/mps-support — centralized device selection)
---

# Integrate MLX inference backend alongside Torch

## Overview

Add MLX as an alternative inference backend so CorridorKey runs natively on Apple Silicon without Torch. Build on PR #33's existing `device_utils.py` and `--device` CLI infrastructure. Keep the integration thin — import the engine from the sibling `corridorkey-mlx` package rather than copying model code.

## Current Inference Flow

```
clip_manager.py::run_inference()
  -> device = resolve_device(args.device)          # PR #33
  -> engine = get_corridor_key_engine(device)
     -> glob CorridorKeyModule/checkpoints/*.pth    # exactly 1
     -> CorridorKeyEngine(ckpt_path, device, img_size=2048)
  -> for frame in clip:
       res = engine.process_frame(image, mask, ...)
       # res keys: alpha [H,W,1] float32 0-1
       #           fg    [H,W,3] float32 0-1 sRGB
       #           comp  [H,W,3] float32 0-1 sRGB
       #           processed [H,W,4] float32 linear premul RGBA
       -> cv2.imwrite EXR (float half)
       -> cv2.imwrite PNG (uint8)
```

The narrowest integration seam is `get_corridor_key_engine()` — replace it with a backend-aware factory.

## Critical: Output Contract Mismatch

The MLX engine (`corridorkey_mlx.CorridorKeyMLXEngine`) currently returns **uint8** arrays with different shapes than Torch:

| Key | Torch | MLX (current) |
|---|---|---|
| `alpha` | `[H,W,1]` float32 0-1 | `[H,W]` uint8 0-255 |
| `fg` | `[H,W,3]` float32 0-1 | `[H,W,3]` uint8 0-255 |
| `comp` | `[H,W,3]` float32 0-1 | `[H,W,3]` uint8 0-255 |
| `processed` | `[H,W,4]` float32 linear premul RGBA | `[H,W,3]` uint8 (placeholder = fg) |

`clip_manager.py` writes EXR with float32 and calls `cv2.COLOR_RGBA2BGRA` on `processed`. MLX's uint8/3ch output will:
- **Silently corrupt** EXR files (integer 0-255 instead of float 0-1)
- **Hard crash** on `processed` (3ch passed to RGBA2BGRA conversion)

**Resolution:** Add an adapter layer in the factory that normalizes MLX output to match Torch's contract before returning. This keeps `clip_manager.py` backend-agnostic.

## Proposed Solution

### Backend selection mechanism

Add `CORRIDORKEY_BACKEND` env var and `--backend` CLI flag (mirrors PR #33's `--device` pattern):

```
--backend flag  >  CORRIDORKEY_BACKEND env var  >  auto-detect
```

Values: `auto` (default), `torch`, `mlx`

**Auto mode logic:**
1. `sys.platform == "darwin" and platform.machine() == "arm64"` → try MLX
2. MLX importable (`try: import corridorkey_mlx`) → try MLX checkpoint
3. `.safetensors` checkpoint found → use MLX
4. Any step fails → fall back to Torch with `logger.info` message

**Explicit `mlx` requested but unavailable:** raise `RuntimeError` with actionable message (don't silently fall back).

### Files changed

| File | Change |
|---|---|
| `CorridorKeyModule/backend.py` **(new)** | Backend factory: `create_engine(backend, device, img_size)`, checkpoint discovery, output adapter |
| `clip_manager.py` | Replace `get_corridor_key_engine()` call with `create_engine()`, add `--backend` arg |
| `pyproject.toml` | Add `corridorkey-mlx` as optional dep group |
| `tests/test_backend.py` **(new)** | Backend selection, checkpoint discovery, output adapter tests |
| `tests/test_mlx_smoke.py` **(new)** | 2048 integration smoke test (marked `@mlx`) |
| `tests/conftest.py` | Add `@mlx` marker |
| `README.md` | Backend selection docs, MLX setup, troubleshooting |

### Phase 1: Backend factory (`CorridorKeyModule/backend.py`)

```python
# CorridorKeyModule/backend.py

CHECKPOINT_DIR = "CorridorKeyModule/checkpoints"
TORCH_EXT = ".pth"
MLX_EXT = ".safetensors"
DEFAULT_IMG_SIZE = 2048

def resolve_backend(requested: str | None = None) -> str:
    """Resolve backend: CLI > env var > auto-detect."""
    # auto: Apple Silicon + mlx importable + .safetensors found → "mlx"
    # otherwise → "torch"

def _discover_checkpoint(ext: str) -> Path:
    """Glob CHECKPOINT_DIR for ext, enforce exactly-one rule."""
    # Cross-reference: if user asked for torch but only .safetensors exists,
    # include that in error message (and vice versa)

def _wrap_mlx_output(raw: dict) -> dict:
    """Normalize MLX uint8 output to match Torch float32 contract.

    - alpha: uint8 [H,W] -> float32 [H,W,1] / 255
    - fg: uint8 [H,W,3] -> float32 [H,W,3] / 255
    - comp: uint8 [H,W,3] -> float32 [H,W,3] / 255
    - processed: construct [H,W,4] RGBA from fg_linear + alpha
      (apply despill + despeckle via color_utils since MLX stubs them)
    """

def create_engine(backend: str | None = None,
                  device: str | None = None,
                  img_size: int = DEFAULT_IMG_SIZE):
    """Factory: returns an engine with a compatible process_frame()."""
    backend = resolve_backend(backend)
    if backend == "mlx":
        ckpt = _discover_checkpoint(MLX_EXT)
        from corridorkey_mlx import CorridorKeyMLXEngine
        raw_engine = CorridorKeyMLXEngine(ckpt, img_size=img_size)
        return _MLXEngineAdapter(raw_engine)
    else:
        ckpt = _discover_checkpoint(TORCH_EXT)
        from CorridorKeyModule.inference_engine import CorridorKeyEngine
        return CorridorKeyEngine(ckpt, device=device, img_size=img_size)
```

The `_MLXEngineAdapter` wraps `CorridorKeyMLXEngine` and:
1. Delegates `process_frame()` calls to the real engine
2. Normalizes output dict to match Torch contract (float32, correct shapes)
3. Applies despill/despeckle from `color_utils.py` (numpy, already available) since MLX stubs them
4. Constructs the `processed` RGBA output (linear premul) from fg + alpha

### Phase 2: Wire into `clip_manager.py` ✅

Minimal change — replace:
```python
engine = get_corridor_key_engine(device=device)
```
with:
```python
from CorridorKeyModule.backend import create_engine
engine = create_engine(backend=args.backend, device=device)
```

Add `--backend` to argparse (same pattern as `--device` from PR #33).

### Phase 3: Local dev install ✅

Add optional dependency group in `pyproject.toml`:
```toml
[dependency-groups]
mlx = ["corridorkey-mlx"]
```

For local dev with sibling checkout:
```bash
uv pip install -e ../corridorkey-mlx
```

Python version compat: both repos require `>=3.11`. MLX CLAUDE.md says `3.12+` but `pyproject.toml` says `>=3.11`. No conflict.

### Phase 4: Checkpoint discovery ✅

Both checkpoint types live in `CorridorKeyModule/checkpoints/`:
- Torch: `*.pth` (existing)
- MLX: `*.safetensors` (new)

Same exactly-one rule per extension. Error messages cross-reference:
```
No .safetensors checkpoint found in CorridorKeyModule/checkpoints/.
(Found .pth files — did you mean --backend=torch?)
```

### Phase 5: Tests ✅

**`tests/conftest.py`** — add `mlx` marker:
```python
markers = [
    "gpu: requires CUDA GPU",
    "slow: long-running test",
    "mlx: requires Apple Silicon with MLX installed",
]
```

**`tests/test_backend.py`** — unit tests (no GPU/MLX needed):
- `resolve_backend()` with env var, explicit, auto
- `_discover_checkpoint()` with 0, 1, 2 files
- `_wrap_mlx_output()` dtype/shape normalization
- Error messages include cross-references

**`tests/test_mlx_smoke.py`** — marked `@mlx @slow`:
- Load MLX engine via `create_engine(backend="mlx")` at 2048
- Process one synthetic frame (solid green + white mask)
- Assert output keys, shapes, dtypes (float32, [H,W,1], [H,W,3], [H,W,4])
- Assert value ranges (0-1 float)

### Phase 6: README ✅

Add section after "Device Selection" (from PR #33):

```markdown
### Backend Selection

CorridorKey supports two inference backends:
- **Torch** (default on Linux/Windows) — CUDA, MPS, or CPU
- **MLX** (Apple Silicon) — native Metal acceleration, no Torch overhead

Resolution: `--backend` flag > `CORRIDORKEY_BACKEND` env var > auto-detect.
Auto mode prefers MLX on Apple Silicon when available.

#### MLX Setup (Apple Silicon)

1. Install the MLX backend from sibling checkout:
   ```bash
   uv pip install -e ../corridorkey-mlx
   ```
2. Place converted weights in `CorridorKeyModule/checkpoints/`:
   ```
   CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors
   ```
3. Run with auto-detection or explicit backend:
   ```bash
   CORRIDORKEY_BACKEND=mlx uv run python clip_manager.py --action run_inference
   ```

MLX uses img_size=2048 by default (same as Torch).

#### Troubleshooting
- **"No .safetensors checkpoint found"** — place MLX weights in CorridorKeyModule/checkpoints/
- **"corridorkey_mlx not installed"** — run `uv pip install -e ../corridorkey-mlx`
- **"MLX requires Apple Silicon"** — MLX only works on M1+ Macs
- **Auto picked Torch unexpectedly** — check `CORRIDORKEY_BACKEND=mlx` explicitly
- **Debug at smaller resolution** — set `CORRIDORKEY_IMG_SIZE=512` (dev override only)
```

## Acceptance Criteria

- [x] `create_engine(backend="torch")` returns working Torch engine (existing behavior)
- [x] `create_engine(backend="mlx")` returns adapted MLX engine with float32 output contract
- [x] `create_engine(backend="auto")` prefers MLX on Apple Silicon, falls back to Torch
- [x] `--backend` CLI flag works in `clip_manager.py`
- [x] `CORRIDORKEY_BACKEND` env var works
- [x] Existing Torch/CUDA path unchanged on Linux/Windows
- [x] MLX checkpoint discovery finds `.safetensors` in `CorridorKeyModule/checkpoints/`
- [x] Error messages are actionable (missing package, missing checkpoint, wrong platform)
- [x] Output adapter normalizes MLX uint8 → float32 with correct shapes
- [x] Adapter applies despill/despeckle from `color_utils.py`
- [x] Adapter constructs `processed` as `[H,W,4]` linear premul RGBA
- [x] `uv run pytest` passes (existing + new tests)
- [x] MLX smoke test at 2048 passes on Apple Silicon
- [x] README documents backend selection and MLX setup
- [x] `uv pip install -e ../corridorkey-mlx` works for local dev

## Unresolved Questions

1. Should `create_engine` live in `CorridorKeyModule/backend.py` or replace `get_corridor_key_engine` inline in `clip_manager.py`? (Plan assumes new file for testability)
2. MLX `input_is_linear` is currently a no-op — acceptable for v1 or must match Torch behavior?
3. Adapter applies numpy despill/despeckle — any perf concern on 4K frames? (Torch does same numpy ops post-inference)
4. Should `CORRIDORKEY_IMG_SIZE` env var be added for debug override, or leave that for a follow-up?
5. PR #33 not yet merged — rebase `feature/mlx-port` on it, or stack as PR #34?
