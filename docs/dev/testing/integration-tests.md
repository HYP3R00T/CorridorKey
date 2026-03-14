# Integration Tests

Integration tests verify that multiple components work correctly together with real inputs. In this project, that means running `CorridorKeyEngine.process_frame` with a real checkpoint file and checking the full output contract.

## What Integration Tests Cover

- `CorridorKeyEngine` loaded from a real `.pth` checkpoint
- `process_frame` output shapes, dtypes, and value ranges on a real frame
- The full pipeline: preprocessing, inference, despill, despeckle, compositing
- The MLX adapter with a real `.safetensors` checkpoint on Apple Silicon

## When to Write an Integration Test

Write an integration test when:

- You change `process_frame` and need to verify the output contract end-to-end
- You add a new post-processing step that affects the final output
- You change checkpoint loading logic in `_load_model`

Do not write integration tests for logic that can be covered by unit tests. If you can test it with a synthetic array, do that instead.

## Markers

Integration tests that require a GPU must be marked `@pytest.mark.gpu`. Tests that require Apple Silicon and MLX must be marked `@pytest.mark.mlx`. Both are skipped by default and must be opted into explicitly.

```python
@pytest.fixture(scope="module")
def engine():
    """Load CorridorKeyEngine once for the entire module."""
    from corridorkey_core.inference_engine import CorridorKeyEngine

    ckpt = Path(os.environ.get("CK_CHECKPOINT_PATH", ""))
    if not ckpt.is_file():
        pytest.skip("Set CK_CHECKPOINT_PATH to a valid .pth file to run integration tests")

    return CorridorKeyEngine(checkpoint_path=ckpt, device="cuda", img_size=2048)


@pytest.mark.gpu
class TestProcessFrameContract:
    def test_output_keys(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert set(result.keys()) == {"alpha", "fg", "comp", "processed"}

    def test_alpha_shape(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert result["alpha"].shape == (1080, 1920, 1)
```

The engine fixture is `scope="module"` so the checkpoint is loaded once and reused across all tests in the file. This avoids paying the model load cost for every test.

## Checkpoint Path

Integration tests need a checkpoint file. Pass the path via the `CK_CHECKPOINT_PATH` environment variable. The fixture calls `pytest.skip` automatically if the variable is not set or points to a missing file, so the test is safe to collect without a checkpoint present.

```shell
CK_CHECKPOINT_PATH=/path/to/model.pth mise run test-gpu
```

## Output Contract

Every integration test against `process_frame` should verify the full output contract:

| Key | Shape | Dtype | Range |
|---|---|---|---|
| `alpha` | `[H, W, 1]` | float32 | 0.0-1.0 |
| `fg` | `[H, W, 3]` | float32 | 0.0-1.0 |
| `comp` | `[H, W, 3]` | float32 | 0.0-1.0 |
| `processed` | `[H, W, 4]` | float32 | unbounded (linear premul) |

## Related

- [Testing Overview](index.md)
- [Hardware-Gated Tests](hardware-gated.md)
- [Unit Tests](unit-tests.md)
