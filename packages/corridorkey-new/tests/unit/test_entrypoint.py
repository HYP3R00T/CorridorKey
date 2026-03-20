"""Unit tests for corridorkey_new.entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey_new.entrypoint import Clip, scan


def _make_clip_dir(root: Path, with_alpha: bool = True) -> Path:
    """Create a minimal clip folder structure under root."""
    clip = root / "my_clip"
    (clip / "Input").mkdir(parents=True)
    if with_alpha:
        (clip / "AlphaHint").mkdir()
    return clip


class TestScan:
    def test_returns_empty_for_empty_dir(self, tmp_path: Path):
        assert scan(tmp_path) == []

    def test_finds_single_clip(self, tmp_path: Path):
        _make_clip_dir(tmp_path)
        clips = scan(tmp_path)
        assert len(clips) == 1
        assert clips[0].name == "my_clip"

    def test_finds_multiple_clips(self, tmp_path: Path):
        for name in ("clip_a", "clip_b", "clip_c"):
            clip = tmp_path / name
            (clip / "Input").mkdir(parents=True)
        clips = scan(tmp_path)
        assert len(clips) == 3

    def test_clip_without_alpha_has_none_alpha_path(self, tmp_path: Path):
        _make_clip_dir(tmp_path, with_alpha=False)
        clips = scan(tmp_path)
        assert clips[0].alpha_path is None

    def test_clip_with_alpha_has_alpha_path(self, tmp_path: Path):
        _make_clip_dir(tmp_path, with_alpha=True)
        clips = scan(tmp_path)
        assert clips[0].alpha_path is not None

    def test_nonexistent_path_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            scan(tmp_path / "ghost")

    def test_unrecognised_file_extension_raises(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(ValueError, match="not a recognised video format"):
            scan(f)

    def test_skips_non_clip_subdirs(self, tmp_path: Path):
        # A dir with no Input/ should be skipped
        (tmp_path / "not_a_clip").mkdir()
        assert scan(tmp_path) == []

    def test_directory_is_itself_a_clip(self, tmp_path: Path):
        (tmp_path / "Input").mkdir()
        clips = scan(tmp_path)
        assert len(clips) == 1

    def test_case_insensitive_input_folder(self, tmp_path: Path):
        clip = tmp_path / "my_clip"
        (clip / "input").mkdir(parents=True)  # lowercase
        clips = scan(tmp_path)
        assert len(clips) == 1

    def test_case_insensitive_alphahint_folder(self, tmp_path: Path):
        clip = tmp_path / "my_clip"
        (clip / "Input").mkdir(parents=True)
        (clip / "alphahint").mkdir()  # lowercase
        clips = scan(tmp_path)
        assert clips[0].alpha_path is not None

    def test_video_file_reorganised_when_reorganise_true(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.touch()
        clips = scan(video, reorganise=True)
        assert len(clips) == 1
        assert (tmp_path / "Input" / "clip.mp4").exists()

    def test_video_file_skipped_when_reorganise_false(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.touch()
        clips = scan(video, reorganise=False)
        assert clips == []

    def test_loose_video_in_dir_reorganised(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.touch()
        clips = scan(tmp_path, reorganise=True)
        assert len(clips) == 1
        assert (tmp_path / "Input" / "clip.mp4").exists()

    def test_loose_video_in_dir_skipped_when_reorganise_false(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.touch()
        clips = scan(tmp_path, reorganise=False)
        assert clips == []

    def test_video_inside_input_dir_used_as_input_path(self, tmp_path: Path):
        """A video file inside Input/ should be set as input_path, not the dir."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clips = scan(tmp_path)
        assert len(clips) == 1
        assert clips[0].input_path == video

    def test_video_inside_alphahint_dir_used_as_alpha_path(self, tmp_path: Path):
        """A video file inside AlphaHint/ should be set as alpha_path, not the dir."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        alpha_dir = clip_dir / "AlphaHint"
        input_dir.mkdir(parents=True)
        alpha_dir.mkdir(parents=True)
        (input_dir / "frame.png").touch()
        video = alpha_dir / "alpha.mp4"
        video.touch()
        clips = scan(tmp_path)
        assert len(clips) == 1
        assert clips[0].alpha_path == video


class TestClip:
    def test_valid_clip_without_alpha(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        assert clip.alpha_path is None

    def test_valid_clip_with_alpha(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        input_dir.mkdir()
        alpha_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)
        assert clip.alpha_path == alpha_dir

    def test_nonexistent_root_raises(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        with pytest.raises(Exception, match="does not exist"):
            Clip(name="test", root=tmp_path / "ghost", input_path=input_dir, alpha_path=None)

    def test_nonexistent_input_raises(self, tmp_path: Path):
        with pytest.raises(Exception, match="does not exist"):
            Clip(name="test", root=tmp_path, input_path=tmp_path / "Input", alpha_path=None)

    def test_root_must_be_directory(self, tmp_path: Path):
        from pydantic import ValidationError

        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(ValidationError):
            Clip(name="test", root=f, input_path=f, alpha_path=None)

    def test_repr_contains_name(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="my_clip", root=tmp_path, input_path=input_dir, alpha_path=None)
        assert "my_clip" in repr(clip)
