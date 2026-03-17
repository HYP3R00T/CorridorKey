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

The root `conftest.py` registers the custom CLI flags and automatically skips marked tests unless the corresponding flag is passed. This applies to all packages in a single run.

```python
# conftest.py (repo root)
def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)
    parser.addoption("--run-gpu", action="store_true", default=False)
    parser.addoption("--run-mlx", action="store_true", default=False)

def pytest_collection_modifyitems(config, items):
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-slow to run"))
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-gpu to run"))
        if "mlx" in item.keywords and not config.getoption("--run-mlx"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-mlx to run"))
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

## Coverage and Omitted Modules

Hardware-dependent modules are excluded from coverage measurement entirely via `omit` in each package's `pyproject.toml`. This prevents the coverage threshold from being dragged down by code that is legitimately untestable in CI.

Current exclusions:

| Package | Module | Reason |
|---|---|---|
| `corridorkey-core` | `engine.py` | Requires a real checkpoint file |
| `corridorkey` | `service.py` | Requires GPU, real checkpoint, and filesystem |
| `corridorkey` | `ffmpeg_tools.py` | Requires FFmpeg installed as a system dependency |
| `corridorkey` | `frame_io.py` | Requires real video/image files on disk |
| `corridorkey` | `device_utils.py` | Requires CUDA or MPS hardware |

Additionally, `# pragma: no cover` is used on individual branches within covered files (e.g. the MLX and Torch branches in `create_engine`) where the branch requires hardware but the surrounding code is otherwise testable.

Do not add `# pragma: no cover` or `omit` entries to avoid writing a test. They are only for code that genuinely cannot run in CI.

## Coverage Threshold

The fast suite enforces a minimum of 75% combined coverage across all packages. Run `mise run test-cov` to check. The threshold is defined in the root `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 75
```

When you add new testable code, add tests for it. If coverage drops below 75%, the `test-cov` task will fail.

## Related

- [Testing Overview](index.md)
- [Integration Tests](integration-tests.md)
- [Unit Tests](unit-tests.md)
