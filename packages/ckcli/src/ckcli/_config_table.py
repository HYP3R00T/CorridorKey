"""Render a Rich table showing resolved config values with source attribution."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.table import Table
from rich.text import Text

from ckcli._console import console

# Source → colour mapping
_SOURCE_STYLE: dict[str, str] = {
    "defaults": "dim",
    "global": "cyan",
    "project": "cyan",
    "env": "yellow",
    "overrides": "green",
}


def _source_text(source: str, source_path: str | None) -> Text:
    style = _SOURCE_STYLE.get(source, "")
    label = source
    if source_path and source != "defaults":
        # Show just the filename, not the full path — keeps the table narrow.
        label = f"{source} ({Path(source_path).name})"
    return Text(label, style=style)


def print_config_table(config, metadata) -> None:
    """Print a Rich table of all config fields with their resolved source.

    Args:
        config: A ``CorridorKeyConfig`` instance.
        metadata: The ``SettingsMetadata`` returned by ``load_config_with_metadata``.
    """
    table = Table(
        title="Active Configuration",
        show_header=True,
        header_style="bold",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    table.add_column("Section", style="dim")
    table.add_column("Field")
    table.add_column("Value", style="cyan")
    table.add_column("Source")

    def _add(section: str, field: str, value: object, meta_key: str) -> None:
        fs = metadata.get_source(meta_key)
        src_text = _source_text(fs.source, fs.source_path) if fs else Text("?", style="dim")
        table.add_row(section, field, str(value), src_text)

    # Top-level fields
    _add("", "device", config.device, "device")
    _add("", "log_level", config.log_level, "log_level")
    _add("", "log_dir", config.log_dir, "log_dir")

    # [preprocess]
    _add("preprocess", "img_size", config.preprocess.img_size or "auto", "preprocess.img_size")
    _add("preprocess", "resize_strategy", config.preprocess.resize_strategy, "preprocess.resize_strategy")

    # [inference]
    _add("inference", "checkpoint_path", config.inference.checkpoint_path or "auto", "inference.checkpoint_path")
    _add("inference", "use_refiner", config.inference.use_refiner, "inference.use_refiner")
    _add("inference", "mixed_precision", config.inference.mixed_precision, "inference.mixed_precision")
    _add("inference", "model_precision", config.inference.model_precision, "inference.model_precision")
    _add("inference", "optimization_mode", config.inference.optimization_mode, "inference.optimization_mode")
    _add("inference", "refiner_scale", config.inference.refiner_scale, "inference.refiner_scale")

    console.print(table)
