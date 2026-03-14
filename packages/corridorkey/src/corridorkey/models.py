"""Shared data models used across the corridorkey package.

Keeping these in a separate module breaks the circular import between
clip_state and project (both previously needed each other's types).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InOutRange:
    """In/out frame range for sub-clip processing.

    Both indices are inclusive and zero-based.

    Attributes:
        in_point: First frame index to process (inclusive).
        out_point: Last frame index to process (inclusive).
    """

    in_point: int
    out_point: int

    @property
    def frame_count(self) -> int:
        """Number of frames in the range (inclusive on both ends)."""
        return self.out_point - self.in_point + 1

    def contains(self, index: int) -> bool:
        """Check whether a frame index falls within this range.

        Args:
            index: Zero-based frame index to test.

        Returns:
            True if in_point <= index <= out_point.
        """
        return self.in_point <= index <= self.out_point

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON storage.

        Returns:
            Dict with 'in_point' and 'out_point' keys.
        """
        return {"in_point": self.in_point, "out_point": self.out_point}

    @classmethod
    def from_dict(cls, d: dict) -> InOutRange:
        """Deserialise from a plain dict.

        Args:
            d: Dict with 'in_point' and 'out_point' keys.

        Returns:
            New InOutRange instance.
        """
        return cls(in_point=d["in_point"], out_point=d["out_point"])
