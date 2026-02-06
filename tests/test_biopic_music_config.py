"""
Unit tests for biopic_music_config.py.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import biopic_music_config


def _temp_dir_in_workspace():
    """Create temp dir inside workspace so sandbox allows writes."""
    workspace = Path(__file__).parent.parent
    return tempfile.mkdtemp(dir=str(workspace))


class TestBiopicMusicConfig(unittest.TestCase):
    """Test cases for biopic music configuration."""

    def test_default_moods_constant(self):
        """Test BIOPIC_MUSIC_DEFAULT_MOODS has expected values."""
        self.assertEqual(
            biopic_music_config.BIOPIC_MUSIC_DEFAULT_MOODS,
            ["relaxing", "passionate", "happy"],
        )

    def test_get_available_moods_when_dir_missing(self):
        """When biopic_music/ does not exist, returns default moods."""
        with patch.object(biopic_music_config, "BIOPIC_MUSIC_DIR", Path("/nonexistent/path/12345")):
            result = biopic_music_config.get_available_moods()
        self.assertEqual(result, ["relaxing", "passionate", "happy"])

    def test_get_available_moods_when_dir_empty(self):
        """When biopic_music/ exists but has no subdirs with MP3s, returns default moods."""
        tmp = _temp_dir_in_workspace()
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            with patch.object(biopic_music_config, "BIOPIC_MUSIC_DIR", music_dir):
                result = biopic_music_config.get_available_moods()
            self.assertEqual(result, ["relaxing", "passionate", "happy"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_get_available_moods_with_empty_subdirs(self):
        """Subdirs without MP3s are ignored."""
        tmp = _temp_dir_in_workspace()
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            (music_dir / "empty_mood").mkdir()
            (music_dir / "no_mp3s").mkdir()
            with patch.object(biopic_music_config, "BIOPIC_MUSIC_DIR", music_dir):
                result = biopic_music_config.get_available_moods()
            self.assertEqual(result, ["relaxing", "passionate", "happy"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_get_available_moods_discovers_mood_folders(self):
        """Subdirs with at least one .mp3 are returned as moods."""
        tmp = _temp_dir_in_workspace()
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            (music_dir / "relaxing").mkdir()
            (music_dir / "relaxing" / "song.mp3").touch()
            (music_dir / "passionate").mkdir()
            (music_dir / "passionate" / "track.mp3").touch()
            (music_dir / "happy").mkdir()
            (music_dir / "happy" / "tune.mp3").touch()
            with patch.object(biopic_music_config, "BIOPIC_MUSIC_DIR", music_dir):
                result = biopic_music_config.get_available_moods()
            self.assertEqual(sorted(result), ["happy", "passionate", "relaxing"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_get_available_moods_ignores_non_mp3_files(self):
        """Subdir with only non-MP3 files is ignored."""
        tmp = _temp_dir_in_workspace()
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            (music_dir / "wav_only").mkdir()
            (music_dir / "wav_only" / "song.wav").touch()
            (music_dir / "relaxing").mkdir()
            (music_dir / "relaxing" / "song.mp3").touch()
            with patch.object(biopic_music_config, "BIOPIC_MUSIC_DIR", music_dir):
                result = biopic_music_config.get_available_moods()
            self.assertEqual(result, ["relaxing"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_get_available_moods_returns_copy_not_reference(self):
        """Returned list is a copy, so mutating it doesn't affect defaults."""
        with patch.object(biopic_music_config, "BIOPIC_MUSIC_DIR", Path("/nonexistent")):
            result = biopic_music_config.get_available_moods()
        result.append("mutated")
        self.assertEqual(
            biopic_music_config.BIOPIC_MUSIC_DEFAULT_MOODS,
            ["relaxing", "passionate", "happy"],
        )

    def test_get_available_moods_with_real_biopic_music_dir(self):
        """If biopic_music/ exists in project with mood subdirs, discovers them."""
        music_dir = Path(__file__).parent.parent / "biopic_music"
        if not music_dir.exists():
            self.skipTest("biopic_music/ not present in project")
        with patch.object(biopic_music_config, "BIOPIC_MUSIC_DIR", music_dir):
            result = biopic_music_config.get_available_moods()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "Should find at least one mood folder")
        for mood in result:
            self.assertIsInstance(mood, str)
            subdir = music_dir / mood
            self.assertTrue(subdir.is_dir(), f"Mood {mood} should be a directory")
            mp3s = list(subdir.glob("*.mp3"))
            self.assertGreater(len(mp3s), 0, f"Mood {mood} should contain at least one .mp3")


if __name__ == "__main__":
    unittest.main()
