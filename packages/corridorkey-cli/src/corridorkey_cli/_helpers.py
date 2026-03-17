"""Shared Rich helpers and progress wiring for the CorridorKey CLI."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

console = Console()
err_console = Console(stderr=True)


def setup_logging(verbose: bool = False) -> Path | None:
    """Configure logging for a CLI session.

    Wires a Rich console handler (WARNING+, or DEBUG when verbose) and a
    rotating JSON file handler (INFO+ by default) onto the root logger.
    The file handler writes to ``~/.config/corridorkey/logs/corridorkey.log``.

    Args:
        verbose: Drop the console handler to DEBUG level.

    Returns:
        Path to the active log file, or None if file logging failed.
    """
    from corridorkey.config import load_config
    from corridorkey.logging_setup import setup_logging as _setup

    try:
        config = load_config()
    except Exception:
        config = None

    return _setup(verbose=verbose, config=config)


def make_progress() -> Progress:
    """Return a pre-configured Rich Progress instance."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


class ProgressContext:
    """Bridge between Application Layer callbacks and a Rich progress bar.

    Owns the Progress instance and exposes bound methods as callbacks so
    the Application Layer never imports Rich directly.

    Usage::

        with ProgressContext() as p:
            process_directory(..., on_progress=p.on_progress, on_clip_start=p.on_clip_start)
    """

    def __init__(self) -> None:
        self._progress = make_progress()
        self._task: TaskID | None = None

    def __enter__(self) -> ProgressContext:
        self._progress.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._progress.__exit__(exc_type, exc_val, exc_tb)

    def on_clip_start(self, clip_name: str, state: str) -> None:
        """Reset the progress bar for a new clip."""
        if self._task is not None:
            self._progress.remove_task(self._task)
        self._task = self._progress.add_task(
            f"[cyan]{clip_name}[/cyan] [dim]({state})[/dim]",
            total=None,
        )

    def on_progress(self, clip_name: str, current: int, total: int) -> None:
        """Advance the progress bar."""
        if self._task is None:
            self._task = self._progress.add_task(f"[cyan]{clip_name}[/cyan]", total=total)
        else:
            # On the first real frame, torch.compile fires - show a hint.
            description = (
                f"[cyan]{clip_name}[/cyan] [dim](compiling kernels...)[/dim]"
                if current == 0
                else f"[cyan]{clip_name}[/cyan]"
            )
            self._progress.update(self._task, completed=current, total=total, description=description)

    def on_warning(self, message: str) -> None:
        """Print a warning without breaking the progress bar."""
        self._progress.console.print(f"[yellow]Warning:[/yellow] {message}")
