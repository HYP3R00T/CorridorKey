"""Stage 1 contracts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, model_validator


class ClipLayout(BaseModel):
    """Filesystem layout contract for a clip after stage 1.

    User-owned directories are never modified by the program. For image
    sequences the program reads directly from the user's folders. For video
    inputs the program extracts frames into a separate folder so the original
    video is untouched.

    Image sequence input:

        clip/
          Input/          <- user's frames, read directly
          AlphaHint/      <- user's alpha frames, read directly (optional)

    Video input:

        clip/
          Input/          <- user's video, never touched
          AlphaHint/      <- user's alpha video, never touched (optional)
          Frames/         <- program-extracted frames (safe to delete and regenerate)
          AlphaFrames/    <- program-extracted alpha frames (safe to delete and regenerate)

    Stage 2 writes generated alpha hints directly into AlphaHint/ when it is
    absent — no copy or intermediate folder is created.

    Attributes:
        root: Absolute path to the clip folder.
        frames_dir: Directory the pipeline reads input frames from.
            ``Input/`` for image sequences, ``Frames/`` for extracted video.
        alpha_frames_dir: Directory the pipeline reads alpha frames from.
            ``AlphaHint/`` for image sequences, ``AlphaFrames/`` for extracted
            video. None if no alpha hint is available (stage 2 will generate it).
    """

    root: Path
    frames_dir: Path
    alpha_frames_dir: Path | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> ClipLayout:
        if not self.root.is_dir():
            raise ValueError(f"Clip root is not a directory: {self.root}")
        if not self.frames_dir.is_dir():
            raise ValueError(f"Frames directory does not exist: {self.frames_dir}")
        if self.alpha_frames_dir is not None and not self.alpha_frames_dir.is_dir():
            raise ValueError(f"Alpha frames directory does not exist: {self.alpha_frames_dir}")
        return self


class ClipManifest(BaseModel):
    """Output contract of stage 1. Input to stage 2 or stage 3.

    Attributes:
        clip_name: Name of the clip, carried through for logging and output naming.
        layout: Filesystem layout — where to read frames from.
        needs_alpha: True if alpha is missing and stage 2 must run before stage 3.
        frame_count: Number of input frames detected.
        is_linear: True if input frames are in linear light (e.g. .exr extension).
    """

    clip_name: str
    layout: ClipLayout
    needs_alpha: bool
    frame_count: int
    is_linear: bool
