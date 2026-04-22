# corridorkey

Core pipeline library for CorridorKey. Import this package to build your own interface, plugin, or automation on top of the AI keying pipeline.

## Install

```shell
# NVIDIA GPU
pip install "corridorkey[cuda]"

# Apple Silicon
pip install "corridorkey[mlx]"

# AMD GPU (Linux)
pip install "corridorkey[rocm]"

# CPU only
pip install corridorkey
```

## The two-line entry point

```python
from corridorkey import Engine, load_config

config = load_config()
engine = Engine(config)
engine.set_alpha_generator(MyAlphaGenerator())
stats = engine.run([Path("/path/to/clips")])
```

`Engine` is the only class most integrators need. Everything else — device resolution, model loading, threading — happens inside `run()`.

## Public API surface

| Section | What it covers |
|---|---|
| [Engine](engine.md) | `Engine` class — the top-level orchestrator |
| [Runner](runner.md) | `JobStats`, `PipelineEvents` — run results and event callbacks |
| [Config](config.md) | `CorridorKeyConfig`, settings models, `load_config()` |
| [Errors](errors.md) | Full error hierarchy |
| [Stages](stages/index.md) | Per-stage public functions and contracts |

## What is not public

The `corridorkey.runtime` and `corridorkey.infra` internals are not part of the public API. Import from `corridorkey` directly, not from submodules, unless you are building a contributor-level integration.
