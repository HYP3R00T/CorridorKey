# Unit Tests

Unit tests verify a single function or class in isolation. They use synthetic inputs and have no external dependencies - no checkpoint files, no GPU, no filesystem state beyond `tmp_path`.

## What to Unit Test

Unit tests are appropriate for:

- Pure functions with deterministic outputs (color math, compositing formulas)
- Error handling and validation logic (checkpoint discovery, backend resolution)
- Output shapes and dtypes
- Edge cases (zero alpha, negative inputs, empty directories)

Unit tests are not appropriate for:

- Code that requires a real model checkpoint
- Code that requires a GPU or Apple Silicon
- End-to-end pipeline behavior

## File Naming

One test file per source module, prefixed with `test_`.

`corridorkey-core`:

| Source module | Test file |
|---|---|
| `compositing.py` | `test_compositing.py` |
| `engine_factory.py` | `test_engine_factory.py` |
| `model_transformer.py` | `test_model_transformer.py` |

`corridorkey`:

| Source module | Test file |
|---|---|
| `clip_state.py` | `test_clip_state.py` |
| `config.py` | `test_config.py` |
| `errors.py` | `test_errors.py` |
| `job_queue.py` | `test_job_queue.py` |
| `models.py` | `test_models.py` |
| `natural_sort.py` | `test_natural_sort.py` |
| `pipeline.py` | `test_pipeline.py` |
| `project.py` | `test_project.py` |
| `protocols.py` | `test_protocols.py` |
| `validators.py` | `test_validators.py` |

Hardware-dependent modules (`service.py`, `ffmpeg_tools.py`, `frame_io.py`, `device_utils.py`) are not unit tested - see [Hardware-Gated Tests](hardware-gated.md).

## Class Grouping

Group tests by the function or class they cover using a `Test` class. This keeps related tests together and makes failures easier to locate.

```python
class TestDespill:
    def test_green_channel_reduced(self): ...
    def test_zero_strength_unchanged(self): ...
    def test_output_shape_preserved(self): ...
```

## Synthetic Inputs

Never use real images or model outputs in unit tests. Create minimal synthetic arrays instead.

```python
def _solid(h: int, w: int, r: float, g: float, b: float) -> np.ndarray:
    """Create a solid-color float32 [H, W, 3] array."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img
```

Keep spatial dimensions small (4x4 or 8x8). The math is the same at any resolution.

## Numpy and Tensor Dual Paths

Several functions in `compositing.py` accept both `np.ndarray` and `torch.Tensor`. Test both paths when the function dispatches differently based on input type.

```python
def test_roundtrip_numpy(self):
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-5)

def test_roundtrip_tensor(self):
    x = torch.linspace(0.0, 1.0, 256)
    result = srgb_to_linear(linear_to_srgb(x))
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, x, atol=1e-5)
```

## Mocking Platform and Environment

Use `unittest.mock.patch` to test platform-dependent logic without needing the actual hardware. Use `monkeypatch` for environment variables.

```python
def test_mlx_on_non_apple_raises(self):
    with (
        patch("corridorkey_core.engine_factory.sys") as mock_sys,
        patch("corridorkey_core.engine_factory.platform") as mock_platform,
    ):
        mock_sys.platform = "linux"
        mock_platform.machine.return_value = "x86_64"
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            resolve_backend("mlx")

def test_env_var_torch(self, monkeypatch):
    monkeypatch.setenv(BACKEND_ENV_VAR, "torch")
    result = resolve_backend(None)
    assert result == "torch"
```

## Error Handling Tests

Test that functions raise the right exception with a useful message when given invalid input. Use `pytest.raises` with a `match` pattern to assert on the message.

```python
def test_invalid_mode_raises(self):
    img = _solid(4, 4, 0.2, 0.9, 0.2)
    with pytest.raises(ValueError, match="green_limit_mode"):
        despill(img, green_limit_mode="median")
```

## Filesystem Tests

Use pytest's built-in `tmp_path` fixture for tests that need real files on disk. It creates a temporary directory that is cleaned up after the test.

```python
def test_finds_single_pth(self, tmp_path: Path):
    ckpt = tmp_path / "model.pth"
    ckpt.touch()
    result = discover_checkpoint(tmp_path, TORCH_EXT)
    assert result == ckpt
```

## Tolerance in Floating Point Assertions

Color math involves floating point. Use `np.allclose` or `np.isclose` with an explicit `atol` rather than exact equality.

```python
assert np.allclose(result, expected, atol=1e-5)
assert np.isclose(result[0], 0.0, atol=1e-6)
```

## Related

- [Testing Overview](index.md)
- [Hardware-Gated Tests](hardware-gated.md)
- [Property-Based Tests](property-tests.md)
