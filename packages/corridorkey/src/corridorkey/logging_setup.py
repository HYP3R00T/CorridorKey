"""Readable file logging for CorridorKey.

Every CLI session calls ``setup_logging()`` once at startup. It wires two
handlers onto the root logger:

- ``RichHandler``  - console, WARNING+ by default (DEBUG when verbose=True).
- ``RotatingFileHandler`` - session log file, INFO+ by default (or config.log_level).
    Writes timestamped text logs with level and logger name for easy debugging.

Log files live in ``config.log_dir`` (default ``~/.config/corridorkey/logs``).
Each session creates a new file named ``YYMMDD_HHMMSS_corridorkey.log``, so
runs never overwrite each other. Up to 5 rotations of 5 MB each are kept per
session file before the oldest rotation is discarded.

Sharing a bug report:
    Share the session file printed at startup, e.g.
    ``~/.config/corridorkey/logs/260317_142301_corridorkey.log``.
"""

from __future__ import annotations

import datetime
import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corridorkey.config import CorridorKeyConfig

# Sentinel so setup_logging is idempotent within a process.
_LOGGING_CONFIGURED = False

# Rotation policy.
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 5  # keep 5 rotations (~25 MB total)


class _TextFormatter(logging.Formatter):
    """Format each log record as readable text with timestamp and metadata."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(
    verbose: bool = False,
    config: CorridorKeyConfig | None = None,
) -> Path | None:
    """Configure root logging for a CLI session.

    Safe to call multiple times - only the first call has any effect.

    Args:
        verbose: When True, the console handler drops to DEBUG level.
            The file handler always uses config.log_level (default INFO).
        config: Loaded CorridorKeyConfig. When None, falls back to
            ``~/.config/corridorkey/logs`` and INFO level.

    Returns:
        Path to the session log file, or None if file logging could not
        be initialised (e.g. permission error on the log directory).
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return None

    from rich.console import Console
    from rich.logging import RichHandler

    err_console = Console(stderr=True)

    # ------------------------------------------------------------------ #
    # Console handler - WARNING by default, DEBUG when verbose            #
    # ------------------------------------------------------------------ #
    console_level = logging.DEBUG if verbose else logging.WARNING
    console_handler = RichHandler(
        console=err_console,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(console_level)

    # ------------------------------------------------------------------ #
    # File handler - session-named readable log                           #
    # ------------------------------------------------------------------ #
    log_path: Path | None = None
    file_handler: logging.Handler | None = None

    log_dir = Path(config.log_dir) if config else Path("~/.config/corridorkey/logs").expanduser()
    file_level_name = config.log_level if config else "INFO"
    file_level = getattr(logging, file_level_name, logging.INFO)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        session_ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        log_path = (log_dir / f"{session_ts}_corridorkey.log").resolve()
        rotating = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        rotating.setLevel(file_level)
        rotating.setFormatter(_TextFormatter())
        file_handler = rotating
    except OSError as exc:
        # Non-fatal - log to console only.
        logging.getLogger(__name__).warning("Could not open log file at %s: %s. Logging to console only.", log_dir, exc)

    # ------------------------------------------------------------------ #
    # Root logger                                                          #
    # ------------------------------------------------------------------ #
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter; root must pass everything
    root.handlers.clear()
    root.addHandler(console_handler)
    if file_handler:
        root.addHandler(file_handler)

    _LOGGING_CONFIGURED = True

    # ------------------------------------------------------------------ #
    # Session header - written once at startup                            #
    # ------------------------------------------------------------------ #
    _write_session_header(config)

    return log_path


def _write_session_header(config: CorridorKeyConfig | None) -> None:
    """Emit an INFO record with session metadata.

    This is the first entry in every log file for a new session, giving
    enough context to reproduce the environment when debugging.
    """
    logger = logging.getLogger(__name__)

    cuda_info: dict = {}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cuda_info = {
                "device": torch.cuda.get_device_name(0),
                "vram_total_gb": round(props.total_mem / (1024**3), 2),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass

    logger.info(
        "CorridorKey session started | event=session_start python=%s platform=%s pid=%s cuda=%s config={device=%s optimization_mode=%s precision=%s checkpoint_dir=%s log_dir=%s log_level=%s}",
        sys.version,
        platform.platform(),
        os.getpid(),
        cuda_info or None,
        config.device if config else "unknown",
        config.optimization_mode if config else "unknown",
        config.precision if config else "unknown",
        str(config.checkpoint_dir) if config else "unknown",
        str(config.log_dir) if config else "unknown",
        config.log_level if config else "INFO",
    )


def reset_logging() -> None:
    """Reset the logging configuration sentinel.

    Intended for use in tests only - allows setup_logging to be called
    multiple times within the same process without the idempotency guard
    blocking subsequent calls.
    """
    global _LOGGING_CONFIGURED
    _LOGGING_CONFIGURED = False
