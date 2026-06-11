# Testing

## Running tests

```shell
uv run pytest
```

Tests are in `packages/corridorkey/tests/`. The test root is configured in `packages/corridorkey/pyproject.toml`.

## Test markers

Some tests are skipped by default because they require hardware or take a long time.

| Marker | Meaning | How to run |
|---|---|---|
| `gpu` | Requires a CUDA GPU | `uv run pytest -m gpu` |
| `slow` | Long-running test | `uv run pytest -m slow` |
| `mlx` | Requires Apple Silicon with MLX | `uv run pytest -m mlx` |

To run all tests including hardware-dependent ones:

```shell
uv run pytest -m "gpu or slow or mlx"
```

## Coverage

```shell
uv run pytest --cov
```

The coverage threshold is 75%. The CI gate fails if coverage drops below this. Some modules are excluded from coverage because they require GPU hardware or model weights:

- `infra/device_utils.py`
- `stages/loader/extractor.py`
- `stages/inference/model.py`
- `stages/inference/loader.py`
- `infra/model_hub.py`
- `infra/logging.py`

## Test layout

```text
tests/
  unit/
    stages/
      scanner/
      loader/
      preprocessor/
      postprocessor/
      writer/
  property/
```

Unit tests are in `tests/unit/`. Property-based tests (Hypothesis) are in `tests/property/`.

## Writing tests

- Unit tests go in `tests/unit/stages/<stage>/`.
- Use `pytest` fixtures for shared setup.
- Use `hypothesis` for property-based tests on pure functions.
- Do not write tests that require GPU hardware unless marked with `@pytest.mark.gpu`.
- Do not write tests that require the inference model.
