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

All test commands are available as mise tasks from the repository root. Each task runs both packages in a single pytest invocation.

| Task | What it runs |
|---|---|
| `mise run test` | Fast suite for all packages - slow, GPU, and MLX tests skipped automatically |
| `mise run test-slow` | Fast suite plus slow model forward pass tests |
| `mise run test-gpu` | GPU-only tests (requires CUDA) |
| `mise run test-cov` | Fast suite with combined coverage report (fails under 75%) |

## File Layout

Tests live inside each package under a `tests/` folder, organized by type. A root-level `conftest.py` provides shared CLI flags and marker auto-skip logic for all packages.

```text
conftest.py                            # root: --run-slow/--run-gpu/--run-mlx flags, marker auto-skip

packages/corridorkey-core/
  tests/
    conftest.py                        # package-specific fixtures (if any)
    unit/
      test_compositing.py
      test_engine_factory.py
      test_model_transformer.py
    property/
      test_compositing_properties.py
    integration/
      test_engine_integration.py       # GPU integration tests for CorridorKeyEngine

packages/corridorkey/
  tests/
    conftest.py                        # package-specific fixtures (if any)
    unit/
      test_clip_state.py
      test_config.py
      test_errors.py
      test_job_queue.py
      test_models.py
      test_natural_sort.py
      test_pipeline.py
      test_project.py
      test_protocols.py
      test_validators.py
    property/
      test_models_properties.py
      test_natural_sort_properties.py
      test_validators_properties.py
```

## Related

- [Unit Tests](unit-tests.md)
- [Integration Tests](integration-tests.md)
- [Property-Based Tests](property-tests.md)
- [Hardware-Gated Tests](hardware-gated.md)
- [Developer Setup](../developer-setup.md)
