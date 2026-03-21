"""Development entry point — runs the full pipeline on a local sample clip.

Replace CLIPS_DIR with your own path, or wire this up to the CLI/GUI.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from corridorkey_new import (
    detect_gpu,
    load,
    load_config,
    resolve_alpha,
    resolve_device,
    scan,
    setup_logging,
)
from corridorkey_new.inference import load_model
from corridorkey_new.infra.model_hub import ensure_model
from corridorkey_new.pipeline import PipelineConfig, PipelineEvents, PipelineRunner

CLIPS_DIR = Path(r"C:\Users\Rajes\Downloads\Samples\sample_inputs_mod")

console = Console()

# ---------------------------------------------------------------------------
# Rich progress printer
# ---------------------------------------------------------------------------


class RichPrinter:
    """Renders a live assembly-line progress panel using Rich.

    Shows one progress bar per active stage plus a live status line that
    reports the current inference frame and inter-stage queue depths.

    Thread-safe — all PipelineEvents callbacks fire from worker threads.
    """

    _STAGE_LABELS = {
        "extract": "Extract    ",
        "preprocess": "Preprocess ",
        "inference": "Inference  ",
        "postwrite": "Write      ",
    }
    _STAGE_ORDER = ["extract", "preprocess", "inference", "postwrite"]

    def __init__(self, total_frames: int) -> None:
        self._total = total_frames
        self._lock = threading.Lock()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
            transient=False,
        )

        # task IDs keyed by stage name — created lazily on stage_start
        self._tasks: dict[str, TaskID] = {}

        # live status fields
        self._inferring: int | None = None
        self._q_pre = 0
        self._q_post = 0
        self._fps_start: dict[str, float] = {}
        self._written = 0

        self._live = Live(self._build_renderable(), console=console, refresh_per_second=10)

    # ------------------------------------------------------------------
    # Renderable
    # ------------------------------------------------------------------

    def _build_renderable(self) -> Panel:
        """Compose the progress bars + status line into a single Panel."""
        table = Table.grid(padding=(0, 1))
        table.add_column()

        # Progress bars
        table.add_row(self._progress)

        # Status line
        parts: list[str] = []
        if self._inferring is not None:
            parts.append(f"[yellow]GPU:[/yellow] frame_{self._inferring:06d}")
        if self._q_pre > 0:
            parts.append(f"[cyan]→ inference queue:[/cyan] {self._q_pre}")
        if self._q_post > 0:
            parts.append(f"[cyan]→ write queue:[/cyan] {self._q_post}")

        status_text = Text.from_markup("  ".join(parts) if parts else " ")
        table.add_row(status_text)

        return Panel(table, title="[bold]Pipeline[/bold]", border_style="bright_black")

    def _refresh(self) -> None:
        self._live.update(self._build_renderable())

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> RichPrinter:
        self._live.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self._live.__exit__(exc_type, exc_val, exc_tb)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Event handlers (called from worker threads — always hold _lock)
    # ------------------------------------------------------------------

    def _on_stage_start(self, stage: str, total: int) -> None:
        with self._lock:
            label = self._STAGE_LABELS.get(stage, stage)
            task_total = total if total > 0 else self._total
            tid = self._progress.add_task(label, total=task_total, status="")
            self._tasks[stage] = tid
            self._fps_start[stage] = time.monotonic()
            self._refresh()

    def _on_stage_done(self, stage: str) -> None:
        with self._lock:
            if stage in self._tasks:
                self._progress.update(self._tasks[stage], status="[green]✓ done[/green]")
            self._refresh()

    def _on_extract_frame(self, idx: int, total: int) -> None:
        with self._lock:
            if "extract" in self._tasks:
                self._progress.advance(self._tasks["extract"])
            self._refresh()

    def _on_preprocess_queued(self, idx: int) -> None:
        with self._lock:
            if "preprocess" in self._tasks:
                self._progress.advance(self._tasks["preprocess"])
            self._refresh()

    def _on_inference_start(self, idx: int) -> None:
        with self._lock:
            self._inferring = idx
            self._refresh()

    def _on_inference_queued(self, idx: int) -> None:
        with self._lock:
            if "inference" in self._tasks:
                self._progress.advance(self._tasks["inference"])
            self._refresh()

    def _on_frame_written(self, idx: int, total: int) -> None:
        with self._lock:
            self._written += 1
            if "postwrite" in self._tasks:
                elapsed = time.monotonic() - self._fps_start.get("postwrite", time.monotonic())
                fps = self._written / elapsed if elapsed > 0 else 0.0
                self._progress.advance(
                    self._tasks["postwrite"],
                    advance=1,
                )
                self._progress.update(
                    self._tasks["postwrite"],
                    status=f"[dim]{fps:.2f} fps[/dim]",
                )
            self._refresh()

    def _on_queue_depth(self, pq: int, wq: int) -> None:
        with self._lock:
            self._q_pre = pq
            self._q_post = wq
            self._refresh()

    def _on_frame_error(self, stage: str, idx: int, err: Exception) -> None:
        with self._lock:
            self._live.console.print(f"[red]  ERROR[/red] {stage} frame_{idx:06d}: {err}")

    # ------------------------------------------------------------------
    # Public event factory
    # ------------------------------------------------------------------

    def as_events(self) -> PipelineEvents:
        return PipelineEvents(
            on_stage_start=self._on_stage_start,
            on_stage_done=self._on_stage_done,
            on_extract_frame=self._on_extract_frame,
            on_preprocess_queued=self._on_preprocess_queued,
            on_inference_start=self._on_inference_start,
            on_inference_queued=self._on_inference_queued,
            on_frame_written=self._on_frame_written,
            on_queue_depth=self._on_queue_depth,
            on_frame_error=self._on_frame_error,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _generate_alpha_externally(manifest) -> Path:
    console.print(f"  Alpha required for '[cyan]{manifest.clip_name}[/cyan]'.")
    console.print(f"  Run your alpha generator on: {manifest.frames_dir}")
    raw = input("  Enter path to generated alpha frames directory: ").strip()
    return Path(raw)


def main() -> None:
    config = load_config()
    setup_logging(config)

    gpu = detect_gpu()
    console.print_json(gpu.model_dump_json(indent=2))

    clips = scan(CLIPS_DIR)
    manifest = load(clips[0])
    console.print_json(manifest.model_dump_json(indent=2))

    if manifest.needs_alpha:
        alpha_dir = _generate_alpha_externally(manifest)
        manifest = resolve_alpha(manifest, alpha_dir)
        console.print(f"alpha resolved: {manifest.alpha_frames_dir}")

    device = resolve_device(config.device)
    inference_config = config.to_inference_config(device=device)

    ensure_model(dest_dir=inference_config.checkpoint_path.parent)

    console.print(f"\nLoading model from [cyan]{inference_config.checkpoint_path}[/cyan] ...")
    console.print(
        f"  img_size=[cyan]{inference_config.img_size}[/cyan]  "
        f"precision=[cyan]{inference_config.model_precision}[/cyan]  "
        f"optimization=[cyan]{inference_config.optimization_mode}[/cyan]"
    )
    model = load_model(inference_config)
    console.print("[green]Model loaded.[/green]")

    printer = RichPrinter(manifest.frame_count)

    pipeline_config = PipelineConfig(
        # Pass resolved img_size so preprocess and inference use the same resolution.
        preprocess=config.to_preprocess_config(device=device, resolved_img_size=inference_config.img_size),
        inference=inference_config,
        model=model,
        events=printer.as_events(),
    )

    console.print(f"\nRunning pipeline for '[bold]{manifest.clip_name}[/bold]' ({manifest.frame_count} frames)...\n")

    with printer:
        PipelineRunner(manifest, pipeline_config).run()

    console.print(f"\n[green]Done.[/green] Output written to: [cyan]{manifest.output_dir}[/cyan]")
