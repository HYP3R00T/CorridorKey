"""Centralised configuration for CorridorKey.

Settings can be overridden via:

- A config file at ``~/.config/corridorkey/corridorkey.toml``  (global)
- A ``corridorkey.toml`` in the current working directory  (project)
- Environment variables prefixed with ``CK_``
- Runtime overrides passed to ``load_config()``

Precedence (lowest to highest):
    defaults < global config < project config < env vars < overrides
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from utilityhub_config import load_settings
from utilityhub_config.utils import expand_path

logger = logging.getLogger(__name__)

_APP_NAME = "corridorkey"


class CorridorKeyConfig(BaseModel):
    """Validated configuration for the CorridorKey pipeline.

    All Path fields support tilde (``~``) and environment variable
    expansion (e.g. ``$STUDIO_ROOT/corridorkey``).

    Load with :func:`load_config`.
    """

    log_dir: Annotated[
        Path,
        Field(
            default=Path("~/.config/corridorkey/logs"),
            description=(
                "Directory where rotating log files are written. Share the latest log file when reporting bugs."
            ),
        ),
    ] = Path("~/.config/corridorkey/logs")

    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        Field(
            default="INFO",
            description=(
                "'DEBUG' adds verbose internal details. "
                "'INFO' captures all normal processing events (recommended). "
                "'WARNING' logs only problems."
            ),
        ),
    ] = "INFO"

    device: Annotated[
        Literal["auto", "cuda", "rocm", "mps", "cpu"],
        Field(
            default="auto",
            description=(
                "Compute device for inference. "
                "'auto' detects the best available device at runtime (ROCm > CUDA > MPS > CPU). "
                "'cuda' forces NVIDIA GPU. 'rocm' forces AMD GPU. "
                "'mps' forces Apple Silicon. 'cpu' forces CPU."
            ),
        ),
    ] = "auto"

    @classmethod
    def _expand_paths(cls, v: Path | str) -> Path:
        return expand_path(str(v) if isinstance(v, Path) else v)


def load_config(overrides: dict | None = None) -> CorridorKeyConfig:
    """Load and validate CorridorKey configuration from all sources.

    Resolution order (lowest to highest priority):
        1. Model field defaults
        2. ``~/.config/corridorkey/corridorkey.toml`` (global user config)
        3. ``./corridorkey.toml`` in the current working directory
        4. Environment variables prefixed with ``CK_``
        5. ``overrides`` dict passed to this function

    Args:
        overrides: Optional dict of field values to apply at highest priority.

    Returns:
        Validated ``CorridorKeyConfig`` instance.
    """
    config, _ = load_settings(
        CorridorKeyConfig,
        app_name=_APP_NAME,
        env_prefix="CK",
        overrides=overrides,
    )

    config.log_dir.expanduser().mkdir(parents=True, exist_ok=True)

    logger.debug("Config loaded: %s", config.model_dump())
    return config
