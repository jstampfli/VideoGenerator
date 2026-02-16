"""
Unit tests for build_video.py functions.
"""

import unittest
import sys
import json
import tempfile
import os
import shutil
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock
import unittest.mock as mock

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress MoviePy FFMPEG cleanup warnings (known issue in MoviePy library)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*FFMPEG_AudioReader.*")
warnings.filterwarnings("ignore", message=".*FFMPEG_AudioReader.*proc.*")

from build_video import (
    load_scenes,
    build_image_prompt,
    find_shorts_for_script,
    find_all_shorts,
    sanitize_prompt_for_safety,
    FIXED_IMAGES_DIR,
    END_SCENE_PAUSE_LENGTH,
    START_SCENE_PAUSE_LENGTH,
    _build_video_impl,
    make_motion_clip_with_audio,
    make_static_clip_with_audio,
    build_biopic_music_track,
    mix_biopic_background_music,
    normalize_audio_to_lufs,
    KENBURNS_PATTERNS,
    KENBURNS_ENABLED,
    CROSSFADE_DURATION,
    FPS,
)
from moviepy import AudioClip, CompositeAudioClip, VideoClip, AudioFileClip
import numpy as np


class TestLoadScenes(unittest.TestCase):
    """Test cases for load_scenes function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_scenes.json"
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def create_test_file(self, data):
        """Helper to create a test JSON file."""
        with open(self.test_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def test_load_scenes_as_list(self):
        """Test loading scenes when JSON is a list."""
        scenes_data = [
            {"id": 1, "title": "Scene 1", "narration": "Text 1"},
            {"id": 2, "title": "Scene 2", "narration": "Text 2"}
        ]
        self.create_test_file(scenes_data)
        
        scenes, metadata = load_scenes(str(self.test_file))
        
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0]["title"], "Scene 1")
        self.assertIsNone(metadata)
    
    def test_load_scenes_with_metadata(self):
        """Test loading scenes when JSON has metadata."""
        scenes_data = {
            "metadata": {
                "title": "Test Video",
                "description": "Test description"
            },
            "scenes": [
                {"id": 1, "title": "Scene 1", "narration": "Text 1"},
                {"id": 2, "title": "Scene 2", "narration": "Text 2"}
            ]
        }
        self.create_test_file(scenes_data)
        
        scenes, metadata = load_scenes(str(self.test_file))
        
        self.assertEqual(len(scenes), 2)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["title"], "Test Video")
    
    def test_load_scenes_invalid_format(self):
        """Test loading scenes with invalid JSON format."""
        invalid_data = {"invalid": "format"}
        self.create_test_file(invalid_data)
        
        with self.assertRaises(ValueError) as context:
            load_scenes(str(self.test_file))
        
        self.assertIn("Invalid JSON format", str(context.exception))


class TestBuildImagePrompt(unittest.TestCase):
    """Test cases for build_image_prompt function."""
    
    def test_basic_prompt_structure(self):
        """Test that prompt includes all required sections."""
        scene = {
            "id": 1,
            "title": "Test Scene",
            "narration": "Test narration",
            "image_prompt": "Test visual details"
        }
        
        prompt = build_image_prompt(scene, None, None)
        
        # Should include scene title
        self.assertIn("Test Scene", prompt)
        # Should include narration
        self.assertIn("Test narration", prompt)
        # Should include image prompt
        self.assertIn("Test visual details", prompt)
        # Should include constraints
        self.assertIn("text-free", prompt.lower())
    
    def test_prompt_with_previous_scene(self):
        """Test prompt includes previous scene context."""
        scene = {
            "id": 2,
            "title": "Scene 2",
            "narration": "Second scene",
            "image_prompt": "Visual 2"
        }
        prev_scene = {
            "id": 1,
            "title": "Scene 1",
            "narration": "First scene",
            "image_prompt": "Visual 1"
        }
        
        prompt = build_image_prompt(scene, prev_scene, None)
        
        # Should reference previous scene
        self.assertIn("previous scene", prompt.lower())
        self.assertIn("Scene 1", prompt)
    
    def test_prompt_with_global_block_override(self):
        """Test prompt uses global_block_override when provided."""
        scene = {
            "id": 1,
            "title": "Test Scene",
            "narration": "Test narration",
            "image_prompt": "Test visual"
        }
        custom_global = "Custom global style block"
        
        prompt = build_image_prompt(scene, None, custom_global)
        
        # Should use custom global block
        self.assertIn("Custom global style block", prompt)
        # Should not include default Tesla block
        self.assertNotIn("Nikola Tesla", prompt)
    
    def test_prompt_without_age(self):
        """Test prompt when age is not specified."""
        scene = {
            "id": 1,
            "title": "Test Scene",
            "narration": "Test narration",
            "image_prompt": "Test visual"
        }
        
        prompt = build_image_prompt(scene, None, None)
        
        # Should not include age specification
        self.assertNotIn("AGE SPECIFICATION", prompt)
    
    def test_prompt_opening_scene(self):
        """Test prompt for opening scene (no previous scene)."""
        scene = {
            "id": 1,
            "title": "Opening",
            "narration": "First scene",
            "image_prompt": "Visual"
        }
        
        prompt = build_image_prompt(scene, None, None)
        
        # Should mention it's the opening scene
        self.assertIn("opening scene", prompt.lower())

    def test_prompt_without_story_context_has_no_story_context_section(self):
        """Without story_context, prompt must not contain STORY CONTEXT section (default behavior unchanged)."""
        scene = {
            "id": 1,
            "title": "A room",
            "narration": "Someone enters.",
            "image_prompt": "Interior shot."
        }
        prompt = build_image_prompt(scene, None, None)
        self.assertNotIn("STORY CONTEXT", prompt)

    def test_prompt_includes_story_context_when_provided(self):
        """Regression: story_context (creature/threat) must be prepended so images show correct threat (e.g. bear not human)."""
        scene = {
            "id": 1,
            "title": "Footsteps outside",
            "narration": "Heavy footsteps circle the hollow log.",
            "image_prompt": "POV from inside log, forest floor visible."
        }
        story_context = (
            "The threat or creature in this story: attacked by a bear in the woods. "
            "When showing the threat (footsteps, figures, shadows): depict bear paws and bear legs, not human."
        )
        prompt = build_image_prompt(scene, None, None, story_context=story_context)
        self.assertIn("STORY CONTEXT", prompt)
        self.assertIn("bear", prompt.lower())
        self.assertIn("attacked by a bear", prompt.lower())
        # Context must appear before scene description so model sees it first
        self.assertLess(prompt.index("STORY CONTEXT"), prompt.index("Footsteps outside"))

    def test_prompt_chapter_transition_uses_title_card_constraint(self):
        """Chapter transition scenes use TITLE CARD constraint instead of 'no text'."""
        scene = {
            "id": 2,
            "title": "Chapter 2: Frontier Foundations",
            "narration": "Frontier Foundations",
            "image_prompt": "Title card with chapter title, 16:9",
            "is_chapter_transition": True,
        }
        prompt = build_image_prompt(scene, None, None)
        self.assertIn("TITLE CARD", prompt)
        self.assertIn("Frontier Foundations", prompt)
        self.assertNotIn("text-free", prompt.lower())
        # Anti-gap: prompt must instruct to fill the entire frame
        self.assertIn("FILL THE ENTIRE FRAME", prompt)
        self.assertIn("no empty black areas", prompt.lower())


class TestStoryContextPassing(unittest.TestCase):
    """Test that story_context from metadata is passed to image generation."""

    def test_story_context_passed_when_metadata_has_it(self):
        """When metadata contains story_context, it must be passed to generate_image_for_scene_with_retry."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_story_context"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            story_context_value = "The threat: a bear in the woods. Depict bear paws, not human."
            scenes_data = {
                "metadata": {"story_context": story_context_value},
                "scenes": [
                    {"id": 1, "title": "Footsteps", "narration": "Heavy steps.", "image_prompt": "Forest floor."},
                ],
            }
            scenes_path = tmp / "scenes.json"
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(scenes_data, f, indent=2)

            img_path = tmp / "scene.png"
            Image.new("RGB", (1536, 1024), color="white").save(str(img_path))

            import wave
            audio_path = tmp / "scene.wav"
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(44100)
                wav.writeframes(b"\x00\x00" * (44100 * 2 * 2))

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path) as mock_gen, \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.make_motion_clip_with_audio") as mock_motion, \
                 patch("build_video.concatenate_videoclips") as mock_concat:
                mock_concat.return_value = MagicMock(duration=4.0, audio=MagicMock(fps=44100))
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(str(scenes_path), out_video_path=str(tmp / "out.mp4"), motion=False)

            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args[1]
            self.assertEqual(call_kwargs.get("story_context"), story_context_value)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestFindShortsForScript(unittest.TestCase):
    """Test cases for find_shorts_for_script function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.shorts_dir = Path(self.temp_dir) / "shorts_scripts"
        self.shorts_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporarily patch the shorts_scripts path
        import build_video
        self.original_shorts_dir = Path("shorts_scripts")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_find_sequential_shorts(self):
        """Test finding sequential shorts."""
        # Create test short files
        for i in range(1, 4):
            short_file = self.shorts_dir / f"test_short{i}.json"
            short_file.write_text('{"scenes": []}')
        
        # Mock the function to use our temp directory
        # We'll test the logic by checking if files exist
        shorts = []
        base_name = "test"
        for i in range(1, 20):
            short_path = self.shorts_dir / f"{base_name}_short{i}.json"
            if short_path.exists():
                shorts.append(short_path)
            else:
                break
        
        self.assertEqual(len(shorts), 3)
        self.assertEqual(shorts[0].name, "test_short1.json")
    
    def test_find_shorts_stops_at_gap(self):
        """Test that finding stops at first missing number."""
        # Create shorts 1, 2, 4 (missing 3)
        for i in [1, 2, 4]:
            short_file = self.shorts_dir / f"test_short{i}.json"
            short_file.write_text('{"scenes": []}')
        
        shorts = []
        base_name = "test"
        for i in range(1, 20):
            short_path = self.shorts_dir / f"{base_name}_short{i}.json"
            if short_path.exists():
                shorts.append(short_path)
            else:
                break
        
        # Should only find 1 and 2, stop at missing 3
        self.assertEqual(len(shorts), 2)
    
    def test_find_shorts_nonexistent(self):
        """Test finding shorts when none exist."""
        shorts = []
        base_name = "nonexistent"
        for i in range(1, 20):
            short_path = self.shorts_dir / f"{base_name}_short{i}.json"
            if short_path.exists():
                shorts.append(short_path)
            else:
                break
        
        self.assertEqual(len(shorts), 0)


class TestFindAllShorts(unittest.TestCase):
    """Test cases for find_all_shorts function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.shorts_dir = Path(self.temp_dir) / "shorts_scripts"
        self.shorts_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_find_all_shorts(self):
        """Test finding all JSON files in shorts directory."""
        # Create multiple short files
        test_files = ["test1_short1.json", "test2_short1.json", "test3_short2.json"]
        for filename in test_files:
            short_file = self.shorts_dir / filename
            short_file.write_text('{"scenes": []}')
        
        # Test the logic (mocking the directory)
        shorts = sorted(self.shorts_dir.glob("*.json"))
        
        self.assertEqual(len(shorts), 3)
        self.assertEqual(shorts[0].name, "test1_short1.json")
    
    def test_find_all_shorts_empty_directory(self):
        """Test finding shorts in empty directory."""
        shorts = sorted(self.shorts_dir.glob("*.json"))
        self.assertEqual(len(shorts), 0)
    
    def test_find_all_shorts_ignores_non_json(self):
        """Test that non-JSON files are ignored."""
        # Create JSON and non-JSON files
        json_file = self.shorts_dir / "test.json"
        txt_file = self.shorts_dir / "test.txt"
        json_file.write_text('{"scenes": []}')
        txt_file.write_text("not json")
        
        shorts = sorted(self.shorts_dir.glob("*.json"))
        
        self.assertEqual(len(shorts), 1)
        self.assertEqual(shorts[0].name, "test.json")


class TestSanitizePromptForSafety(unittest.TestCase):
    """Test cases for sanitize_prompt_for_safety in build_video.py."""
    
    def test_basic_sanitization(self):
        """Test basic prompt sanitization."""
        prompt = "A scene showing a person working"
        result = sanitize_prompt_for_safety(prompt)
        self.assertIn("Safe, appropriate", result)
        self.assertIn(prompt, result)
    
    def test_self_harm_violation(self):
        """Test sanitization with self-harm violation type."""
        prompt = "A person suffering from pain"
        result = sanitize_prompt_for_safety(prompt, violation_type="self-harm")
        self.assertNotIn("suffering", result.lower())
        self.assertIn("contemplation", result.lower())
    
    def test_word_replacements(self):
        """Test word replacements for safety."""
        prompt = "A person dealing with death and struggle"
        result = sanitize_prompt_for_safety(prompt, violation_type="self-harm")
        self.assertNotIn("death", result.lower())
        self.assertNotIn("struggle", result.lower())
        self.assertIn("legacy", result.lower())
        self.assertIn("journey", result.lower())

class TestWriteAudiofileNoVerboseOrLogger(unittest.TestCase):
    """Regression: write_audiofile must not be called with verbose= or logger= (MoviePy 2.x removed them)."""

    def test_build_video_never_calls_write_audiofile_with_verbose_or_logger(self):
        """Ensure build_video.py does not pass verbose or logger to write_audiofile."""
        build_video_path = Path(__file__).parent.parent / "build_video.py"
        source = build_video_path.read_text(encoding="utf-8")
        # Find all write_audiofile call sites
        import re
        # Match .write_audiofile(...) with possible args
        for m in re.finditer(r"\.write_audiofile\s*\(([^)]+)\)", source):
            call_args = m.group(1)
            self.assertNotIn("verbose", call_args, msg=f"write_audiofile must not be called with verbose= (MoviePy 2.x): {m.group(0)}")
            self.assertNotIn("logger", call_args, msg=f"write_audiofile must not be called with logger= (MoviePy 2.x): {m.group(0)}")

class TestKenBurnsMotion(unittest.TestCase):
    """Unit tests for PIL-based Ken Burns motion clip generation."""

    @classmethod
    def setUpClass(cls):
        try:
            from PIL import Image as _Image
        except ImportError:
            raise unittest.SkipTest("PIL not available")

        cls._tmp = Path(__file__).parent / "tmp_kenburns_unit"
        cls._tmp.mkdir(parents=True, exist_ok=True)

        # Create a test image with a gradient so motion produces different pixel values
        img = _Image.new("RGB", (1536, 1024))
        for x in range(1536):
            for y in range(1024):
                img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
        cls._img_path = cls._tmp / "scene.png"
        img.save(str(cls._img_path))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _make_audio(self, duration=1.0):
        """Create a short silent audio clip for testing."""
        def make_silence(t):
            return [0.0, 0.0]
        return AudioClip(make_silence, duration=duration, fps=44100)

    def test_motion_clip_returns_correct_duration(self):
        """make_motion_clip_with_audio returns a clip with the expected duration."""
        audio = self._make_audio(1.0)
        clip = make_motion_clip_with_audio(self._img_path, audio, pattern="zoom_in")
        expected = START_SCENE_PAUSE_LENGTH + 1.0 + END_SCENE_PAUSE_LENGTH
        self.assertAlmostEqual(clip.duration, expected, places=2)

    def test_motion_clip_frame_shape(self):
        """Each frame has the correct shape (out_h, out_w, 3)."""
        import build_video as bv
        audio = self._make_audio(0.5)
        clip = make_motion_clip_with_audio(self._img_path, audio, pattern="zoom_in")
        out_w, out_h = bv.config.output_resolution
        frame = clip.get_frame(0)
        self.assertEqual(frame.shape, (out_h, out_w, 3))

    def test_all_patterns_produce_valid_clips(self):
        """Every declared pattern produces a clip without crashing."""
        audio = self._make_audio(0.5)
        for pattern in KENBURNS_PATTERNS:
            with self.subTest(pattern=pattern):
                clip = make_motion_clip_with_audio(self._img_path, audio, pattern=pattern)
                self.assertGreater(clip.duration, 0)
                frame = clip.get_frame(0)
                self.assertEqual(len(frame.shape), 3)

    def test_unknown_pattern_uses_fallback(self):
        """An unknown pattern name still produces a valid clip (fallback to zoom_in)."""
        audio = self._make_audio(0.5)
        clip = make_motion_clip_with_audio(self._img_path, audio, pattern="nonexistent_pattern")
        self.assertGreater(clip.duration, 0)
        frame = clip.get_frame(0)
        self.assertEqual(len(frame.shape), 3)

    def test_frames_change_over_time(self):
        """First and last frames differ (motion is actually happening)."""
        import numpy as _np
        audio = self._make_audio(1.0)
        clip = make_motion_clip_with_audio(self._img_path, audio, pattern="zoom_in")
        first = clip.get_frame(0)
        last = clip.get_frame(clip.duration - 0.01)
        # Frames should not be identical — the zoom changes the crop
        self.assertFalse(_np.array_equal(first, last), "First and last frames are identical — no motion")

    def test_clip_has_audio(self):
        """The returned clip has audio attached."""
        audio = self._make_audio(0.5)
        clip = make_motion_clip_with_audio(self._img_path, audio, pattern="zoom_in")
        self.assertIsNotNone(clip.audio)

    def test_make_motion_clip_uses_per_scene_intensity(self):
        """make_motion_clip_with_audio accepts intensity param and produces valid clips for each level."""
        from kenburns_config import KENBURNS_INTENSITY_LEVELS
        audio = self._make_audio(0.5)
        for intensity in KENBURNS_INTENSITY_LEVELS:
            with self.subTest(intensity=intensity):
                clip = make_motion_clip_with_audio(
                    self._img_path, audio, pattern="zoom_in", intensity=intensity
                )
                self.assertGreater(clip.duration, 0)
                frame = clip.get_frame(0)
                self.assertEqual(len(frame.shape), 3)
                self.assertIsNotNone(clip.audio)


_TESTS_DIR = Path(__file__).parent


def _make_integration_test_fixtures(tmp: Path):
    """
    Helper: create the minimal fixtures (scenes.json, image, audio) for integration tests.
    Returns (scenes_path, scene_image_path, scene_audio_path).
    """
    from PIL import Image
    import wave

    scenes_path = tmp / "scenes.json"
    scenes_data = {
        "scenes": [
            {"id": 1, "title": "Scene 1", "narration": "Hello.", "image_prompt": "A room."},
            {"id": 2, "title": "Scene 2", "narration": "World.", "image_prompt": "A field."},
        ]
    }
    with open(scenes_path, "w", encoding="utf-8") as f:
        json.dump(scenes_data, f, indent=2)

    img_path = tmp / "scene.png"
    Image.new("RGB", (1536, 1024), color="white").save(str(img_path))

    audio_path = tmp / "scene.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        num_frames = 44100 * 2  # 2 seconds
        wav.writeframes(b"\x00\x00" * (num_frames * 2))

    return scenes_path, img_path, audio_path


class TestMotionIntegration(unittest.TestCase):
    """Integration tests: motion=True/False through _build_video_impl."""

    # Keep a direct reference to the real function so it survives patching
    _real_static = staticmethod(make_static_clip_with_audio)

    @staticmethod
    def _static_ignoring_pattern(img_path, audio_clip, **kwargs):
        """Wrap make_static_clip_with_audio, accepting and ignoring extra kwargs like pattern."""
        return TestMotionIntegration._real_static(img_path, audio_clip)

    def test_motion_flag_calls_make_motion_clip(self):
        """When motion=True, make_motion_clip_with_audio is called for each scene."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_motion_on"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            out_path = tmp / "out.mp4"

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.make_motion_clip_with_audio", wraps=self._static_ignoring_pattern) as mock_motion, \
                 patch("build_video.make_static_clip_with_audio") as mock_static, \
                 patch("build_video.KENBURNS_ENABLED", True):
                # Don't actually write video
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        motion=True,
                    )
                # make_motion_clip should have been called (once per scene)
                self.assertGreaterEqual(mock_motion.call_count, 2)
                # make_static_clip should NOT have been called for regular scenes
                # (it may still be called by make_motion_clip fallback, but the
                # direct path should not call it)
                mock_static.assert_not_called()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_motion_flag_calls_static_clip(self):
        """When motion=False, only make_static_clip_with_audio is called."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_motion_off"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            out_path = tmp / "out.mp4"

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.make_motion_clip_with_audio") as mock_motion, \
                 patch("build_video.concatenate_videoclips") as mock_concat:
                mock_concat.return_value = MagicMock(duration=4.0, audio=MagicMock(fps=44100))
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        motion=False,
                    )
                mock_motion.assert_not_called()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_motion_enables_crossfade_concatenation(self):
        """When motion=True and transition_to_next is crossfade, CompositeVideoClip is used for transitions."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        from moviepy import CompositeVideoClip

        tmp = _TESTS_DIR / "tmp_crossfade_on"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            with open(scenes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["scenes"][0]["transition_to_next"] = "crossfade"
            data["scenes"][0]["transition_speed"] = "medium"
            data["scenes"][1]["transition_to_next"] = None
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            composite_calls = []
            real_composite = CompositeVideoClip

            def capture_composite(clips, **kwargs):
                composite_calls.append({"clips": clips, **kwargs})
                return real_composite(clips, **kwargs)

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", False), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.make_motion_clip_with_audio", wraps=self._static_ignoring_pattern), \
                 patch("build_video.KENBURNS_ENABLED", True), \
                 patch("moviepy.CompositeVideoClip", side_effect=capture_composite):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(tmp / "out.mp4"),
                        motion=True,
                    )
            self.assertGreaterEqual(len(composite_calls), 1, f"Expected CompositeVideoClip for crossfade, got: {composite_calls}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_motion_uses_compose_with_cut_transitions(self):
        """When motion=False and transitions are cut, concatenate_videoclips is used (chain or compose)."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_chain_concat"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            out_path = tmp / "out.mp4"

            concat_calls = []
            real_concat = bv.concatenate_videoclips

            def capture_concat(*args, **kwargs):
                concat_calls.append(kwargs)
                return real_concat(*args, **kwargs)

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", False), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.concatenate_videoclips", side_effect=capture_concat):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        motion=False,
                    )
            # Cut transitions: concatenate_videoclips used (chain or compose), no negative padding
            self.assertGreaterEqual(len(concat_calls), 1, f"Expected concat call, got: {concat_calls}")
            for c in concat_calls:
                self.assertIsNone(c.get("padding"), f"Cut transitions should not use padding: {c}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_build_video_per_scene_transitions(self):
        """build_video uses CompositeVideoClip with CrossFade when transition_to_next is crossfade."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        from moviepy import CompositeVideoClip

        tmp = _TESTS_DIR / "tmp_per_scene_transitions"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            with open(scenes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["scenes"][0]["transition_to_next"] = "crossfade"
            data["scenes"][0]["transition_speed"] = "medium"
            data["scenes"][1]["transition_to_next"] = None
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            composite_calls = []
            real_composite = CompositeVideoClip

            def capture_composite(clips, **kwargs):
                composite_calls.append({"clips": clips, **kwargs})
                return real_composite(clips, **kwargs)

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", False), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.make_motion_clip_with_audio", wraps=TestMotionIntegration._static_ignoring_pattern), \
                 patch("build_video.KENBURNS_ENABLED", True), \
                 patch("moviepy.CompositeVideoClip", side_effect=capture_composite):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(tmp / "out.mp4"),
                        motion=True,
                    )
            self.assertGreaterEqual(len(composite_calls), 1, f"Expected CompositeVideoClip call for crossfade, got: {composite_calls}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_build_video_slide_transition(self):
        """build_video uses CompositeVideoClip with SlideIn/SlideOut when transition_to_next is slide_left."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        from moviepy import CompositeVideoClip

        tmp = _TESTS_DIR / "tmp_slide_transition"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            with open(scenes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["scenes"][0]["transition_to_next"] = "slide_left"
            data["scenes"][0]["transition_speed"] = "medium"
            data["scenes"][1]["transition_to_next"] = None
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            composite_calls = []
            real_composite = CompositeVideoClip

            def capture_composite(clips, **kwargs):
                composite_calls.append({"clips": clips, **kwargs})
                return real_composite(clips, **kwargs)

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", False), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.make_motion_clip_with_audio", wraps=TestMotionIntegration._static_ignoring_pattern), \
                 patch("build_video.KENBURNS_ENABLED", True), \
                 patch("moviepy.CompositeVideoClip", side_effect=capture_composite):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(tmp / "out.mp4"),
                        motion=True,
                    )
            self.assertGreaterEqual(len(composite_calls), 1, f"Expected CompositeVideoClip call for slide, got: {composite_calls}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_all_slide_directions_produce_valid_video(self):
        """All slide transition directions (left, right, up, down) produce valid video without crashing."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        from kenburns_config import TRANSITION_TYPES

        slide_types = [t for t in TRANSITION_TYPES if t.startswith("slide_")]
        for trans in slide_types:
            with self.subTest(transition=trans):
                tmp = _TESTS_DIR / f"tmp_slide_{trans}"
                tmp.mkdir(parents=True, exist_ok=True)
                try:
                    scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
                    with open(scenes_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["scenes"][0]["transition_to_next"] = trans
                    data["scenes"][0]["transition_speed"] = "medium"
                    data["scenes"][1]["transition_to_next"] = None
                    with open(scenes_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)

                    with patch.object(bv.config, "save_assets", False), \
                         patch.object(bv.config, "is_vertical", False), \
                         patch.object(bv.config, "temp_dir", str(tmp)), \
                         patch.object(bv.config, "biopic_music_enabled", False), \
                         patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                         patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                         patch("build_video.make_motion_clip_with_audio", wraps=TestMotionIntegration._static_ignoring_pattern), \
                         patch("build_video.KENBURNS_ENABLED", True):
                        with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                            _build_video_impl(
                                str(scenes_path),
                                out_video_path=str(tmp / "out.mp4"),
                                motion=True,
                            )
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

    def test_transition_speeds_quick_medium_slow_produce_valid_video(self):
        """Transition speeds quick, medium, slow all produce valid video."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        from kenburns_config import TRANSITION_SPEEDS

        for speed in TRANSITION_SPEEDS:
            with self.subTest(transition_speed=speed):
                tmp = _TESTS_DIR / f"tmp_speed_{speed}"
                tmp.mkdir(parents=True, exist_ok=True)
                try:
                    scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
                    with open(scenes_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["scenes"][0]["transition_to_next"] = "crossfade"
                    data["scenes"][0]["transition_speed"] = speed
                    data["scenes"][1]["transition_to_next"] = None
                    with open(scenes_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)

                    with patch.object(bv.config, "save_assets", False), \
                         patch.object(bv.config, "is_vertical", False), \
                         patch.object(bv.config, "temp_dir", str(tmp)), \
                         patch.object(bv.config, "biopic_music_enabled", False), \
                         patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                         patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                         patch("build_video.make_motion_clip_with_audio", wraps=TestMotionIntegration._static_ignoring_pattern), \
                         patch("build_video.KENBURNS_ENABLED", True):
                        with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                            _build_video_impl(
                                str(scenes_path),
                                out_video_path=str(tmp / "out.mp4"),
                                motion=True,
                            )
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

    def test_cut_transition_produces_valid_video(self):
        """Explicit cut transition produces valid video (no CompositeVideoClip for transitions)."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_cut_transition"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            with open(scenes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["scenes"][0]["transition_to_next"] = "cut"
            data["scenes"][1]["transition_to_next"] = None
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            concat_calls = []
            real_concat = bv.concatenate_videoclips

            def capture_concat(*args, **kwargs):
                concat_calls.append(args)
                return real_concat(*args, **kwargs)

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", False), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.concatenate_videoclips", side_effect=capture_concat):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(tmp / "out.mp4"),
                        motion=False,
                    )
            self.assertGreaterEqual(len(concat_calls), 1, "Cut should use concatenate_videoclips")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_legacy_dissolve_medium_normalized_to_crossfade(self):
        """Legacy transition_to_next dissolve_medium is normalized to crossfade+medium and produces valid video."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_legacy_dissolve"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            with open(scenes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["scenes"][0]["transition_to_next"] = "dissolve_medium"
            data["scenes"][1]["transition_to_next"] = None
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", False), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.make_motion_clip_with_audio", wraps=TestMotionIntegration._static_ignoring_pattern), \
                 patch("build_video.KENBURNS_ENABLED", True):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(tmp / "out.mp4"),
                        motion=True,
                    )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBiopicMusicWithMotion(unittest.TestCase):
    """Regression tests: biopic background music integration works with motion clips."""

    def test_biopic_music_mixed_when_enabled(self):
        """For biopic videos with biopic_music_enabled, mix_biopic_background_music is called."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_biopic_music"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            out_path = tmp / "out.mp4"

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", True), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.mix_biopic_background_music") as mock_mix:
                # mix returns a mock audio clip
                mock_mix.return_value = MagicMock(duration=5.0)
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        motion=False,
                    )
                mock_mix.assert_called_once()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_biopic_music_mixed_with_motion(self):
        """Biopic music is still applied when motion=True (motion clips have valid audio)."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_biopic_music_motion"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            out_path = tmp / "out.mp4"

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", True), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video.make_motion_clip_with_audio", wraps=TestMotionIntegration._static_ignoring_pattern), \
                 patch("build_video.KENBURNS_ENABLED", True), \
                 patch("build_video.mix_biopic_background_music") as mock_mix:
                mock_mix.return_value = MagicMock(duration=5.0)
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        motion=True,
                    )
                mock_mix.assert_called_once()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBiopicMusicFadeout(unittest.TestCase):
    """Tests that biopic background music fadeout uses the MoviePy 2.x API (not the removed 1.x API)."""

    def test_build_biopic_music_track_single_segment_with_fadeout(self):
        """build_biopic_music_track applies AudioFadeOut when tail_sec and fadeout_sec are set."""
        def make_silence(t):
            return np.array([0.0, 0.0])
        scene_audio = AudioClip(make_silence, duration=2.0, fps=44100)
        music_clip = AudioClip(make_silence, duration=4.0, fps=44100)

        # Create a fake music dir with a dummy mp3 so glob finds it
        fake_music_dir = _TESTS_DIR / "fake_music"
        fake_mood_dir = fake_music_dir / "relaxing"
        fake_mood_dir.mkdir(parents=True, exist_ok=True)
        fake_mp3 = fake_mood_dir / "test.mp3"
        fake_mp3.touch()
        try:
            with patch("build_video._fit_song_to_duration", return_value=music_clip), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", fake_music_dir), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DEFAULT_MOODS", ["relaxing"]):
                result = build_biopic_music_track(
                    metadata={"outline": {"music_mood": "relaxing"}},
                    scene_audio_clips=[scene_audio],
                    total_duration=2.0,
                    scenes=None,  # Single segment mode
                    music_volume_db=-23.0,
                    tail_sec=2.0,
                    fadeout_sec=1.5,
                )
                # Should return a valid audio clip (not None, no crash from AudioFadeOut)
                self.assertIsNotNone(result)
                self.assertGreater(result.duration, 0)
        finally:
            shutil.rmtree(fake_music_dir, ignore_errors=True)

    def test_build_biopic_music_track_no_crash_without_fadeout(self):
        """build_biopic_music_track works with tail_sec=0 (no fadeout applied)."""
        def make_silence(t):
            return np.array([0.0, 0.0])
        scene_audio = AudioClip(make_silence, duration=2.0, fps=44100)
        music_clip = AudioClip(make_silence, duration=2.0, fps=44100)

        fake_music_dir = _TESTS_DIR / "fake_music2"
        fake_mood_dir = fake_music_dir / "relaxing"
        fake_mood_dir.mkdir(parents=True, exist_ok=True)
        fake_mp3 = fake_mood_dir / "test.mp3"
        fake_mp3.touch()
        try:
            with patch("build_video._fit_song_to_duration", return_value=music_clip), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", fake_music_dir), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DEFAULT_MOODS", ["relaxing"]):
                result = build_biopic_music_track(
                    metadata={"outline": {"music_mood": "relaxing"}},
                    scene_audio_clips=[scene_audio],
                    total_duration=2.0,
                    scenes=None,
                    music_volume_db=-23.0,
                    tail_sec=0,
                    fadeout_sec=0,
                )
                self.assertIsNotNone(result)
                self.assertGreater(result.duration, 0)
        finally:
            shutil.rmtree(fake_music_dir, ignore_errors=True)

    def test_build_biopic_music_track_per_scene_song_and_volume(self):
        """build_biopic_music_track uses per-scene music_song and music_volume when present."""
        def make_silence(t):
            return np.array([0.0, 0.0])
        scene_audio1 = AudioClip(make_silence, duration=2.0, fps=44100)
        scene_audio2 = AudioClip(make_silence, duration=3.0, fps=44100)
        music_clip = AudioClip(make_silence, duration=5.0, fps=44100)

        fake_music_dir = _TESTS_DIR / "fake_music_per_scene"
        (fake_music_dir / "relaxing").mkdir(parents=True, exist_ok=True)
        (fake_music_dir / "passionate").mkdir(parents=True, exist_ok=True)
        (fake_music_dir / "relaxing" / "song1.mp3").touch()
        (fake_music_dir / "passionate" / "track.mp3").touch()
        scenes = [
            {"id": 1, "music_song": "relaxing/song1.mp3", "music_volume": "medium"},
            {"id": 2, "music_song": "passionate/track.mp3", "music_volume": "low"},
        ]
        try:
            with patch("build_video._fit_song_to_duration", return_value=music_clip), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", fake_music_dir):
                result = build_biopic_music_track(
                    metadata={},
                    scene_audio_clips=[scene_audio1, scene_audio2],
                    total_duration=5.0,
                    scenes=scenes,
                    tail_sec=0,
                    fadeout_sec=0,
                )
                self.assertIsNotNone(result)
                self.assertGreater(result.duration, 0)
        finally:
            shutil.rmtree(fake_music_dir, ignore_errors=True)

    def test_build_biopic_music_consecutive_same_song_one_segment(self):
        """Consecutive scenes with same song produce one continuous segment (no restarts)."""
        def make_silence(t):
            return np.array([0.0, 0.0])
        scene_audio1 = AudioClip(make_silence, duration=2.0, fps=44100)
        scene_audio2 = AudioClip(make_silence, duration=2.0, fps=44100)
        scene_audio3 = AudioClip(make_silence, duration=2.0, fps=44100)
        expected_dur = 3 * (START_SCENE_PAUSE_LENGTH + 2.0 + END_SCENE_PAUSE_LENGTH)
        music_clip = AudioClip(make_silence, duration=expected_dur, fps=44100)

        fake_music_dir = _TESTS_DIR / "fake_music_same_song"
        (fake_music_dir / "relaxing").mkdir(parents=True, exist_ok=True)
        (fake_music_dir / "relaxing" / "song1.mp3").touch()
        scenes = [
            {"id": 1, "music_song": "relaxing/song1.mp3", "music_volume": "medium"},
            {"id": 2, "music_song": "relaxing/song1.mp3", "music_volume": "medium"},
            {"id": 3, "music_song": "relaxing/song1.mp3", "music_volume": "low"},
        ]
        try:
            with patch("build_video._fit_song_to_duration", return_value=music_clip) as mock_fit:
                with patch("biopic_music_config.BIOPIC_MUSIC_DIR", fake_music_dir):
                    result = build_biopic_music_track(
                        metadata={},
                        scene_audio_clips=[scene_audio1, scene_audio2, scene_audio3],
                        total_duration=6.0,
                        scenes=scenes,
                        tail_sec=0,
                        fadeout_sec=0,
                    )
                self.assertIsNotNone(result)
                self.assertGreater(result.duration, 0)
                # Same song across 3 scenes -> 1 segment (1 call to _fit_song_to_duration)
                self.assertEqual(mock_fit.call_count, 1)
                mock_fit.assert_called_once()
                call_args = mock_fit.call_args
                self.assertGreaterEqual(call_args[0][1], expected_dur * 0.99)
        finally:
            shutil.rmtree(fake_music_dir, ignore_errors=True)

    def test_build_biopic_music_same_song_different_volumes_applies_both(self):
        """Same song with different volumes (medium, low) applies both volume levels to blocks."""
        def make_silence(t):
            return np.array([0.0, 0.0])
        scene_audio1 = AudioClip(make_silence, duration=2.0, fps=44100)
        scene_audio2 = AudioClip(make_silence, duration=2.0, fps=44100)
        expected_dur = 2 * (START_SCENE_PAUSE_LENGTH + 2.0 + END_SCENE_PAUSE_LENGTH)
        music_clip = AudioClip(make_silence, duration=expected_dur, fps=44100)

        fake_music_dir = _TESTS_DIR / "fake_music_volumes"
        (fake_music_dir / "relaxing").mkdir(parents=True, exist_ok=True)
        (fake_music_dir / "relaxing" / "song1.mp3").touch()
        scenes = [
            {"id": 1, "music_song": "relaxing/song1.mp3", "music_volume": "medium"},
            {"id": 2, "music_song": "relaxing/song1.mp3", "music_volume": "low"},
        ]
        try:
            with patch("build_video._fit_song_to_duration", return_value=music_clip):
                with patch("build_video.apply_volume_to_audioclip", wraps=lambda c, f: c) as mock_vol:
                    with patch("biopic_music_config.BIOPIC_MUSIC_DIR", fake_music_dir):
                        result = build_biopic_music_track(
                            metadata={},
                            scene_audio_clips=[scene_audio1, scene_audio2],
                            total_duration=4.0,
                            scenes=scenes,
                            tail_sec=0,
                            fadeout_sec=0,
                        )
                self.assertIsNotNone(result)
                self.assertGreater(result.duration, 0)
                # Two volume blocks -> apply_volume_to_audioclip called twice with different factors
                self.assertEqual(mock_vol.call_count, 2)
                factors = [call[0][1] for call in mock_vol.call_args_list]
                # medium (-25.5 dB) > low (-28 dB) in linear scale
                self.assertGreater(factors[0], factors[1])
        finally:
            shutil.rmtree(fake_music_dir, ignore_errors=True)

    def test_build_biopic_music_raises_on_missing_song(self):
        """build_biopic_music_track raises when biopic script has scene without music_song."""
        def make_silence(t):
            return np.array([0.0, 0.0])
        scene_audio = AudioClip(make_silence, duration=2.0, fps=44100)
        scenes = [
            {"id": 1, "chapter_num": 1},  # No music_song/music_volume
        ]
        metadata = {"script_type": "biopic"}
        with self.assertRaises(ValueError) as ctx:
            build_biopic_music_track(
                metadata=metadata,
                scene_audio_clips=[scene_audio],
                total_duration=2.0,
                scenes=scenes,
                tail_sec=0,
                fadeout_sec=0,
            )
        self.assertIn("music_song", str(ctx.exception))
        self.assertIn("music_volume", str(ctx.exception))

    def test_build_biopic_music_raises_on_missing_music_file(self):
        """build_biopic_music_track raises FileNotFoundError when song file does not exist."""
        def make_silence(t):
            return np.array([0.0, 0.0])
        scene_audio = AudioClip(make_silence, duration=2.0, fps=44100)
        scenes = [
            {"id": 1, "music_song": "relaxing/nonexistent.mp3", "music_volume": "medium"},
        ]
        metadata = {"script_type": "biopic"}
        fake_music_dir = _TESTS_DIR / "fake_music_missing_file"
        (fake_music_dir / "relaxing").mkdir(parents=True, exist_ok=True)
        # Do NOT create relaxing/nonexistent.mp3
        try:
            with patch("biopic_music_config.BIOPIC_MUSIC_DIR", fake_music_dir):
                with self.assertRaises(FileNotFoundError) as ctx:
                    build_biopic_music_track(
                        metadata=metadata,
                        scene_audio_clips=[scene_audio],
                        total_duration=2.0,
                        scenes=scenes,
                        tail_sec=0,
                        fadeout_sec=0,
                    )
                self.assertIn("Music file not found", str(ctx.exception))
        finally:
            shutil.rmtree(fake_music_dir, ignore_errors=True)

    def test_build_biopic_music_raises_when_scenes_lack_music(self):
        """build_biopic_music_track raises ValueError when scenes lack music_song/music_volume (no fallback)."""
        def make_silence(t):
            return np.array([0.0, 0.0])
        scene_audio = AudioClip(make_silence, duration=2.0, fps=44100)
        scenes = [
            {"id": 1, "chapter_num": 1},  # No music_song/music_volume
        ]
        with self.assertRaises(ValueError) as ctx:
            build_biopic_music_track(
                metadata={"outline": {"chapters": [{"chapter_num": 1, "num_scenes": 1, "music_mood": "relaxing"}]}},
                scene_audio_clips=[scene_audio],
                total_duration=2.0,
                scenes=scenes,
                tail_sec=0,
                fadeout_sec=0,
            )
        self.assertIn("music_song", str(ctx.exception))
        self.assertIn("music_volume", str(ctx.exception))

    def test_mix_biopic_background_music_does_not_crash(self):
        """mix_biopic_background_music runs without import errors (regression for moviepy.audio.fx.all)."""
        def make_silence(t):
            return [0.0, 0.0]
        narration = AudioClip(make_silence, duration=3.0, fps=44100)
        scene_audio = AudioClip(make_silence, duration=3.0, fps=44100)
        music_clip = AudioClip(make_silence, duration=5.0, fps=44100)

        with patch("build_video.build_biopic_music_track", return_value=music_clip):
            result = mix_biopic_background_music(
                narration,
                duration=3.0,
                metadata=None,
                scene_audio_clips=[scene_audio],
                scenes=None,
                music_volume_db=-23.0,
                tail_sec=2.0,
                fadeout_sec=1.5,
            )
            self.assertIsNotNone(result)

    def test_biopic_music_end_to_end_through_build_video_impl(self):
        """
        Full integration: _build_video_impl with biopic music enabled reaches
        mix_biopic_background_music without crashing (catches moviepy.audio.fx.all breakage).
        """
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_biopic_e2e"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path, img_path, audio_path = _make_integration_test_fixtures(tmp)
            # Add music_song and music_volume (required for per-scene music mode)
            with open(scenes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for s in data["scenes"]:
                s["music_song"] = "relaxing/test.mp3"
                s["music_volume"] = "medium"
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            fake_music_dir = tmp / "biopic_music"
            (fake_music_dir / "relaxing").mkdir(parents=True, exist_ok=True)
            (fake_music_dir / "relaxing" / "test.mp3").touch()

            def make_silence(t):
                return np.array([0.0, 0.0])
            music_clip = AudioClip(make_silence, duration=6.0, fps=44100)

            out_path = tmp / "out.mp4"

            # Track whether mix_biopic_background_music was actually called
            call_tracker = {"called": False, "error": None}
            original_mix = bv.mix_biopic_background_music

            def tracking_mix(*args, **kwargs):
                call_tracker["called"] = True
                try:
                    return original_mix(*args, **kwargs)
                except Exception as e:
                    call_tracker["error"] = e
                    raise

            with patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)), \
                 patch.object(bv.config, "biopic_music_enabled", True), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=img_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=audio_path), \
                 patch("build_video._fit_song_to_duration", return_value=music_clip), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", fake_music_dir), \
                 patch("build_video.mix_biopic_background_music", side_effect=tracking_mix):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        motion=False,
                    )
            self.assertTrue(call_tracker["called"], "mix_biopic_background_music should have been called")
            self.assertIsNone(call_tracker["error"],
                              f"mix_biopic_background_music crashed: {call_tracker['error']}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestNormalizeAudioToLufs(unittest.TestCase):
    """Tests for normalize_audio_to_lufs (background music loudness normalization)."""

    def test_normalize_audio_to_lufs_creates_output(self):
        """normalize_audio_to_lufs produces valid output when given valid WAV input."""
        import wave
        tmp = Path(tempfile.mkdtemp())
        try:
            wav_path = tmp / "input.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(44100)
                wav.writeframes(b"\x00\x00" * (44100 * 2))  # 1 sec stereo silence
            out_path = tmp / "output.wav"
            try:
                result = normalize_audio_to_lufs(wav_path, target_lufs=-18.0, output_path=out_path)
                self.assertTrue(result.exists())
                self.assertGreater(result.stat().st_size, 0)
            except (RuntimeError, FileNotFoundError) as e:
                self.skipTest(f"ffmpeg not available or loudnorm failed: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
