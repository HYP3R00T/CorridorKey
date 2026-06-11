# Installation

## Requirements

- Python 3.13 or later
- A supported compute device (see table below)
- FFmpeg — required for video input (image sequences work without it)

## Compute device support

| Device | Requirement |
|---|---|
| NVIDIA GPU (CUDA) | CUDA 12.8 driver |
| AMD GPU (ROCm) | ROCm 7.1, Linux only |
| Apple Silicon (MPS) | macOS 13+, M1 or later |
| CPU | No GPU required — slow |

## Install

Install `corridorkey-cli` with the extra that matches your hardware.

```shell
# NVIDIA GPU
pip install "corridorkey-cli[cuda]"

# Apple Silicon
pip install "corridorkey-cli[mlx]"

# AMD GPU (Linux)
pip install "corridorkey-cli[rocm]"

# CPU only
pip install corridorkey-cli
```

If you manage your environment with `uv`:

```shell
uv sync --extra cuda
```

## First-time setup

After installing, run `ck init` once. It checks your environment, creates the config file, and offers to download the inference model.

```shell
ck init
```

The init command checks:

- Python version (3.13+)
- Compute device and available VRAM
- Config file presence (`~/.config/corridorkey/corridorkey.yaml`)
- Inference model presence (`~/.config/corridorkey/models/CorridorKey_v1.0.pth`)

If the model is not found, `ck init` will offer to download it automatically.

## Next steps

- [Quick Start](quick-start.md) — process your first clip
- [Clips Layout](clips-layout.md) — how to organise your footage
