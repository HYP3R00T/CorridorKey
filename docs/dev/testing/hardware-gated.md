# Hardware-Gated Tests

Some tests require hardware or files that are not available in a standard CI environment. These tests are gated behind markers so they are skipped by default and only run when explicitly opted into.

## Markers

| Marker | Requirement | How to enable |
|---|---|---|
| `slow` | No hardware needed, but takes several seconds | `mise run test-slow` or `--run-slow` |
| `gpu` | CUDA GPU | `mise run test-gpu` or `--run-gpu` |
| `mlx` | Apple Silicon with `corridorkey-mlx` installed | `--run-mlx` |

Unmarked tests must always pass with no special hardware or files.

## How Markers Work

The `conftest.py` in each `tests/` folder registers custom CLI flags and automatically skips marked tests unless the flag is passed.

```python
# conftest.py
def pytest_collection_modifyitems(config, items):
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-slow to run"))
```

This means you never need to pass `-m` on the command line. The default `mise run test` just works.

## Applying a Marker

Apply markers at the class or function level.

```python
# Entire class is slow
@pytest.mark.slow
class TestGreenFormer:
    def test_forward_pass(self): ...

# Single test requires GPU
@pytest.mark.gpu
def test_process_frame_on_cuda(): ...

# Combined: slow and requires GPU
@pytest.mark.slow
@pytest.mark.gpu
def test_full_pipeline_4k(): ...
```

## Coverage and `# pragma: no cover`

Code that cannot run without hardware is excluded from coverage measurement using `# pragma: no cover`. This prevents the coverage threshold from being dragged down by code that is legitimately untestable in CI.

Current exclusions:

| Code | Reason |
|---|---|
| `CorridorKeyEngine` class | Requires a real checkpoint file |
| `_MLXEngineAdapter` class | Requires Apple Silicon and MLX |
| `_wrap_mlx_output` function | Only called by the MLX adapter |
| MLX and Torch branches in `create_engine` | Require checkpoint or MLX hardware |
| `inference_engine.py` (entire file) | Omitted via `[tool.coverage.run] omit` |

Do not add `# pragma: no cover` to avoid writing a test. It is only for code that genuinely cannot run in CI.

## Coverage Threshold

The fast suite enforces a minimum of 75% coverage. Run `mise run test-cov` to check. The threshold is defined in `packages/corridorkey-core/pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 75
```

When you add new testable code, add tests for it. If coverage drops below 75%, the `test-cov` task will fail.

## Related

- [Testing Overview](index.md)
- [Integration Tests](integration-tests.md)
- [Unit Tests](unit-tests.md)
