"""Pytest configuration for corridorkey-core tests.

Slow, GPU, and MLX tests are skipped automatically unless the corresponding
flag is passed:
    --run-slow   enable @pytest.mark.slow tests
    --run-gpu    enable @pytest.mark.gpu tests
    --run-mlx    enable @pytest.mark.mlx tests
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption("--run-gpu", action="store_true", default=False, help="Run GPU tests")
    parser.addoption("--run-mlx", action="store_true", default=False, help="Run MLX tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_slow = pytest.mark.skip(reason="Pass --run-slow to run")
    skip_gpu = pytest.mark.skip(reason="Pass --run-gpu to run")
    skip_mlx = pytest.mark.skip(reason="Pass --run-mlx to run")

    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(skip_gpu)
        if "mlx" in item.keywords and not config.getoption("--run-mlx"):
            item.add_marker(skip_mlx)
