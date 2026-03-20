"""Stage 0 - entrypoint.

Accepts a path from the external interface (CLI, GUI, or API) and produces
a list of Clip objects ready for stage 1 to consume.

This is the only place that touches the filesystem for discovery purposes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator


def scan(path: str | Path) -> list[Clip]:
    """Scan a directory for processable clips.

    Accepts either:
    - A clips directory containing multiple clip subfolders
    - A single clip folder directly (must contain Input/Frames and AlphaHint)

    Args:
        path: Path to a clips directory or a single clip folder.

    Returns:
        List of Clip objects in READY state.

    Raises:
        ValueError: If the path does not exist or is not a directory.
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # If the path itself looks like a clip, treat it as a single clip.
    clip = _try_build_clip(path)
    if clip is not None:
        return [clip]

    # Otherwise scan subdirectories for clips.
    clips = []
    for item in sorted(path.iterdir()):
        if not item.is_dir():
            continue
        clip = _try_build_clip(item)
        if clip is not None:
            clips.append(clip)

    return clips


def _try_build_clip(clip_dir: Path) -> Clip | None:
    """Attempt to build a Clip from a directory. Returns None if not a valid clip."""
    input_path = _find_input(clip_dir)
    alpha_path = _find_alpha(clip_dir)

    if input_path is None or alpha_path is None:
        return None

    return Clip(
        name=clip_dir.name,
        root=clip_dir,
        input_path=input_path,
        alpha_path=alpha_path,
    )


def _find_input(clip_dir: Path) -> Path | None:
    """Locate the input asset (Frames/ sequence or Source/ video) inside a clip folder."""
    for name in ("Frames", "Source", "Input"):
        candidate = _find_icase(clip_dir, name)
        if candidate is not None:
            return candidate
    return None


def _find_alpha(clip_dir: Path) -> Path | None:
    """Locate the alpha hint asset (AlphaHint/ sequence or video) inside a clip folder."""
    candidate = _find_icase(clip_dir, "AlphaHint")
    return candidate


def _find_icase(parent: Path, name: str) -> Path | None:
    """Case-insensitive lookup of a child entry inside parent."""
    for child in parent.iterdir():
        if child.name.lower() == name.lower():
            return child
    return None


class Clip(BaseModel):
    """A clip ready for processing. Output contract of stage 0.

    Attributes:
        name: Human-readable clip name derived from the folder name.
        root: Absolute path to the clip folder.
        input_path: Path to the input asset (image sequence dir or video file).
        alpha_path: Path to the alpha hint asset (image sequence dir or video file).
    """

    name: str
    root: Path
    input_path: Path
    alpha_path: Path

    @field_validator("root", "input_path", "alpha_path")
    @classmethod
    def must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    @model_validator(mode="after")
    def root_must_be_directory(self) -> Clip:
        if not self.root.is_dir():
            raise ValueError(f"Clip root is not a directory: {self.root}")
        return self

    def __repr__(self) -> str:
        return f"Clip(name={self.name!r}, input={self.input_path}, alpha={self.alpha_path})"
