"""Stage 1 - clip manifest builder.

Validates a Clip and returns a ClipManifest with paths and metadata.
No pixel data is read here. Stage 3 handles frame iteration.
"""

from __future__ import annotations

from corridorkey_new.entrypoint import Clip
from corridorkey_new.stage1.contracts import ClipManifest
from corridorkey_new.stage1.validator import count_frames, detect_is_linear, validate


def load(clip: Clip) -> ClipManifest:
    """Validate a clip and return its manifest.

    Args:
        clip: A Clip from stage 0.

    Returns:
        ClipManifest with validated paths, needs_alpha flag, frame count, and is_linear.

    Raises:
        ValueError: If validation fails.
    """
    validate(clip)

    return ClipManifest(
        clip_name=clip.name,
        input_path=clip.input_path,
        alpha_path=clip.alpha_path,
        needs_alpha=clip.alpha_path is None,
        frame_count=count_frames(clip.input_path),
        is_linear=detect_is_linear(clip.input_path),
    )
