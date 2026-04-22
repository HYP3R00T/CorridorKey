# Setup

## Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/) — Python package manager
- Git

## Clone and install

```shell
git clone https://github.com/nikopueringer/CorridorKey.git
cd CorridorKey
uv sync
```

`uv sync` installs all packages in the workspace including dev dependencies. The virtual environment is created at `.venv/`.

## Platform extras

Install the extra that matches your hardware for GPU-accelerated development:

```shell
uv sync --extra cuda    # NVIDIA GPU
uv sync --extra mlx     # Apple Silicon
uv sync --extra rocm    # AMD GPU (Linux)
```

## Verify the install

```shell
uv run pytest --co -q
```

This lists all collected tests without running them. If it completes without errors, the install is working.

## Project structure

```text
packages/
  corridorkey/          core pipeline library
  corridorkey-cli/      ck command-line interface
docs/                   documentation source
scripts/                development utilities
```

Both packages are installed as editable installs in the workspace. Changes to source files take effect immediately without reinstalling.

## Running the CLI locally

```shell
uv run ck --help
uv run ck init
```
