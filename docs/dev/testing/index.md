# Testing

This section documents how tests are structured, written, and run in this repository. It covers all test types used in the project and what contributors need to know before writing new tests.

## Test Types

| Type | Purpose | Runs in CI |
|---|---|---|
| [Unit](unit-tests.md) | Test a single function or class in isolation | Yes |
| [Integration](integration-tests.md) | Test multiple components working together with real inputs | With GPU only |
| [Property-based](property-tests.md) | Verify mathematical invariants across many random inputs | Yes |
| [Hardware-gated](hardware-gated.md) | Tests requiring a GPU, Apple Silicon, or a real checkpoint | No |

## Running Tests

All test commands are available as mise tasks from the repository root.

| Task | What it runs |
|---|---|
| `mise run test` | Fast suite - slow, GPU, and MLX tests skipped automatically |
| `mise run test-slow` | Fast suite plus slow model forward pass tests |
| `mise run test-gpu` | GPU-only tests (requires CUDA) |
| `mise run test-cov` | Fast suite with coverage report (fails under 75%) |

## File Layout

Tests live inside each package under a `tests/` folder, organized by type.

```text
packages/corridorkey-core/
  tests/
    conftest.py                        # marker auto-skip logic and CLI flags
    unit/
      test_compositing.py              # unit tests for compositing.py
      test_engine_factory.py           # unit tests for engine_factory.py
      test_model_transformer.py        # unit tests for model_transformer.py
    property/
      test_compositing_properties.py   # Hypothesis property tests for compositing.py
    integration/
      test_engine_integration.py       # GPU integration tests for CorridorKeyEngine
```

## Related

- [Unit Tests](unit-tests.md)
- [Integration Tests](integration-tests.md)
- [Property-Based Tests](property-tests.md)
- [Hardware-Gated Tests](hardware-gated.md)
- [Developer Setup](../developer-setup.md)
