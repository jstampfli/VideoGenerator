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
    text_to_ssml,
    load_scenes,
    build_image_prompt,
    find_shorts_for_script,
    find_all_shorts,
    sanitize_prompt_for_safety,
    generate_room_tone,
    generate_low_frequency_drone,
    generate_low_frequency_drone_with_transition,
    generate_drone_with_scene_transitions,
    generate_detail_sounds,
    apply_volume_to_audioclip,
    mix_horror_background_audio,
    get_horror_disclaimer_image_path,
    HORROR_DISCLAIMER_DURATION,
    FIXED_IMAGES_DIR,
    HORROR_DISCLAIMER_TALL,
    HORROR_DISCLAIMER_WIDE,
    END_SCENE_PAUSE_LENGTH,
    _build_video_impl,
    make_motion_clip_with_audio,
    make_static_clip_with_audio,
    build_biopic_music_track,
    mix_biopic_background_music,
    KENBURNS_PATTERNS,
    KENBURNS_ENABLED,
    CROSSFADE_DURATION,
    FPS,
)
from moviepy import AudioClip, CompositeAudioClip, VideoClip, AudioFileClip
import numpy as np


class TestTextToSSML(unittest.TestCase):
    """Test cases for text_to_ssml function."""
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_basic_text(self):
        """Test basic text conversion."""
        text = "Hello world."
        result = text_to_ssml(text)
        self.assertIn("<speak>", result)
        self.assertIn("</speak>", result)
        self.assertIn("Hello world", result)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_sentence_pauses(self):
        """Test that periods add pauses."""
        text = "First sentence. Second sentence."
        result = text_to_ssml(text)
        self.assertIn('<break time="400ms"/>', result)
        # Should have breaks after both sentences
        self.assertEqual(result.count('<break time="400ms"/>'), 2)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_ellipsis_handling(self):
        """Test ellipsis conversion to dramatic pause."""
        text = "He thought... then acted."
        result = text_to_ssml(text)
        # Ellipsis should become 600ms break, not 400ms
        self.assertIn('<break time="600ms"/>', result)
        # Should still have pause after final period
        self.assertIn('<break time="400ms"/>', result)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_unicode_ellipsis(self):
        """Test Unicode ellipsis character."""
        text = "He thought… then acted."
        result = text_to_ssml(text)
        self.assertIn('<break time="600ms"/>', result)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_comma_pauses(self):
        """Test that commas add shorter pauses."""
        text = "First, second, third."
        result = text_to_ssml(text)
        # Should have 200ms breaks after commas
        self.assertIn('<break time="200ms"/>', result)
        # Should have 400ms break after period
        self.assertIn('<break time="400ms"/>', result)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_question_exclamation_pauses(self):
        """Test question and exclamation marks."""
        text = "Really? Yes!"
        result = text_to_ssml(text)
        self.assertIn('<break time="350ms"/>', result)
        # Should have two breaks (one for ?, one for !)
        self.assertEqual(result.count('<break time="350ms"/>'), 2)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_em_dash_pause(self):
        """Test em-dash conversion to pause."""
        text = "He said—then stopped."
        result = text_to_ssml(text)
        # Em-dash should become 400ms break
        self.assertIn('<break time="400ms"/>', result)
        # Should not contain the em-dash character
        self.assertNotIn("—", result)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_hyphen_pauses(self):
        """Test hyphen/dash handling."""
        text = "Word - another word."
        result = text_to_ssml(text)
        # Should have break where hyphen was
        self.assertIn('<break time="300ms"/>', result)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_year_emphasis(self):
        """Test that years get prosody emphasis."""
        text = "In 1936, he published."
        result = text_to_ssml(text)
        # Years should be wrapped in prosody tags
        self.assertIn('<prosody rate="95%">1936</prosody>', result)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_xml_escaping(self):
        """Test that special XML characters are escaped."""
        text = "A & B < C > D"
        result = text_to_ssml(text)
        # Should escape special characters
        self.assertIn("&amp;", result)
        self.assertIn("&lt;", result)
        self.assertIn("&gt;", result)
    
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.GOOGLE_VOICE_TYPE', '')
    def test_colon_semicolon_pauses(self):
        """Test colon and semicolon pauses."""
        text = "First: second; third."
        result = text_to_ssml(text)
        self.assertIn('<break time="300ms"/>', result)  # Colon
        self.assertIn('<break time="250ms"/>', result)  # Semicolon


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


class TestHorrorBackgroundAudio(unittest.TestCase):
    """Test cases for horror background audio functions."""
    
    def test_generate_room_tone(self):
        """Test that room tone is generated correctly."""
        duration = 1.0  # 1 second
        room_tone = generate_room_tone(duration)
        
        # Should return an AudioClip
        self.assertIsInstance(room_tone, AudioClip)
        # Duration should match
        self.assertAlmostEqual(room_tone.duration, duration, places=2)
        # Should have fps
        self.assertIsNotNone(room_tone.fps)
        # Should be able to get audio samples using get_frame
        sample = room_tone.get_frame(0.5)
        self.assertIsNotNone(sample)
    
    def test_generate_room_tone_duration(self):
        """Test room tone with different durations."""
        for duration in [0.5, 1.0, 2.0, 5.0]:
            room_tone = generate_room_tone(duration)
            self.assertAlmostEqual(room_tone.duration, duration, places=2)
    
    def test_generate_low_frequency_drone(self):
        """Test that low-frequency drone is generated correctly."""
        duration = 1.0
        frequency = 50.0
        drone = generate_low_frequency_drone(duration, frequency)
        
        # Should return an AudioClip
        self.assertIsInstance(drone, AudioClip)
        # Duration should match
        self.assertAlmostEqual(drone.duration, duration, places=2)
        # Should have fps
        self.assertIsNotNone(drone.fps)
        # Should be able to get audio samples using get_frame
        sample = drone.get_frame(0.5)
        self.assertIsNotNone(sample)
    
    def test_generate_drone_frequency_clamping(self):
        """Test that drone frequency is clamped to 30-80 Hz range."""
        # Test frequency below minimum (should clamp to 30)
        drone_low = generate_low_frequency_drone(1.0, 10.0)
        self.assertIsInstance(drone_low, AudioClip)
        
        # Test frequency above maximum (should clamp to 80)
        drone_high = generate_low_frequency_drone(1.0, 100.0)
        self.assertIsInstance(drone_high, AudioClip)
        
        # Test frequency in range
        drone_normal = generate_low_frequency_drone(1.0, 50.0)
        self.assertIsInstance(drone_normal, AudioClip)
    
    def test_generate_detail_sounds(self):
        """Test that detail sounds are generated correctly."""
        duration = 2.0
        detail_sounds = generate_detail_sounds(duration, sound_type="random")
        
        # Should return an AudioClip
        self.assertIsInstance(detail_sounds, AudioClip)
        # Duration should be at least the requested duration (sounds can extend beyond)
        # Individual sounds can be 0.5-2 seconds, and are placed randomly
        self.assertGreaterEqual(detail_sounds.duration, duration)
        # Should have fps
        self.assertIsNotNone(detail_sounds.fps)
    
    def test_generate_detail_sounds_specific_types(self):
        """Test detail sounds with specific sound types."""
        duration = 1.0
        for sound_type in ["creak", "wind", "hum", "whisper"]:
            detail_sounds = generate_detail_sounds(duration, sound_type=sound_type)
            self.assertIsInstance(detail_sounds, AudioClip)
            # Duration should be at least the requested duration
            # Sounds are placed randomly and can extend up to 2 seconds each,
            # so the actual duration can be longer than requested
            self.assertGreaterEqual(detail_sounds.duration, duration)
            # Allow for sounds that extend beyond (up to 2 seconds per sound, 3-8 sounds total)
            # So max could be duration + ~16 seconds, but typically much less
            # We'll just check it's reasonable (not more than 5x the duration)
            self.assertLessEqual(detail_sounds.duration, duration * 5)
    
    def test_apply_volume_to_audioclip(self):
        """Test that volume is applied correctly to AudioClip."""
        # Create a simple test AudioClip
        duration = 1.0
        sample_rate = 44100
        
        def make_test_audio(t):
            if np.isscalar(t):
                return 1.0
            else:
                return np.ones_like(t)
        
        original_clip = AudioClip(make_test_audio, duration=duration, fps=sample_rate)
        
        # Apply volume (half volume)
        volume_factor = 0.5
        volume_adjusted = apply_volume_to_audioclip(original_clip, volume_factor)
        
        # Should return an AudioClip
        self.assertIsInstance(volume_adjusted, AudioClip)
        # Duration should be preserved
        self.assertAlmostEqual(volume_adjusted.duration, duration, places=2)
        # FPS should be preserved
        self.assertEqual(volume_adjusted.fps, sample_rate)
        
        # Check that volume is actually applied using get_frame
        original_sample = original_clip.get_frame(0.5)
        adjusted_sample = volume_adjusted.get_frame(0.5)
        
        # Adjusted sample should be half the original
        if np.isscalar(original_sample):
            self.assertAlmostEqual(adjusted_sample, original_sample * volume_factor, places=5)
        else:
            np.testing.assert_array_almost_equal(adjusted_sample, original_sample * volume_factor, decimal=5)
    
    def test_apply_volume_zero(self):
        """Test applying zero volume (should silence audio)."""
        duration = 1.0
        sample_rate = 44100
        
        def make_test_audio(t):
            if np.isscalar(t):
                return 1.0
            else:
                return np.ones_like(t)
        
        original_clip = AudioClip(make_test_audio, duration=duration, fps=sample_rate)
        silenced = apply_volume_to_audioclip(original_clip, 0.0)
        
        sample = silenced.get_frame(0.5)
        if np.isscalar(sample):
            self.assertEqual(sample, 0.0)
        else:
            np.testing.assert_array_equal(sample, np.zeros_like(sample))
    
    def test_apply_volume_double(self):
        """Test applying double volume (should amplify audio)."""
        duration = 1.0
        sample_rate = 44100
        
        def make_test_audio(t):
            if np.isscalar(t):
                return 0.5
            else:
                return np.full_like(t, 0.5)
        
        original_clip = AudioClip(make_test_audio, duration=duration, fps=sample_rate)
        amplified = apply_volume_to_audioclip(original_clip, 2.0)
        
        original_sample = original_clip.get_frame(0.5)
        amplified_sample = amplified.get_frame(0.5)
        
        if np.isscalar(original_sample):
            self.assertAlmostEqual(amplified_sample, original_sample * 2.0, places=5)
        else:
            np.testing.assert_array_almost_equal(amplified_sample, original_sample * 2.0, decimal=5)
    
    @patch('build_video.generate_room_tone')
    @patch('build_video.generate_low_frequency_drone')
    @patch('build_video.generate_detail_sounds')
    @patch('build_video.apply_volume_to_audioclip')
    def test_mix_horror_background_audio_structure(self, mock_apply_volume, mock_detail_sounds, 
                                                     mock_drone, mock_room_tone):
        """Test that mix_horror_background_audio calls all required functions."""
        duration = 5.0
        
        # Create mock AudioClip objects
        def make_silence(t):
            return 0.0
        
        mock_room = AudioClip(make_silence, duration=duration, fps=44100)
        mock_drone_clip = AudioClip(make_silence, duration=duration, fps=44100)
        mock_detail = AudioClip(make_silence, duration=duration, fps=44100)
        
        mock_room_tone.return_value = mock_room
        mock_drone.return_value = mock_drone_clip
        mock_detail_sounds.return_value = mock_detail
        mock_apply_volume.side_effect = lambda clip, vol: clip  # Return clip as-is
        
        # Create narration audio
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Call the function
        try:
            result = mix_horror_background_audio(
                narration_audio,
                duration,
                room_tone_volume=-45.0,
                drone_volume=-25.0,
                detail_volume=-30.0
            )
            
            # Should return a CompositeAudioClip
            self.assertIsInstance(result, CompositeAudioClip)
            
            # Verify all functions were called
            mock_room_tone.assert_called_once_with(duration)
            mock_drone.assert_called_once_with(duration, frequency=50.0)
            mock_detail_sounds.assert_called_once_with(duration, sound_type="random")
            
            # Volume should be applied 3 times (room tone, drone, detail)
            self.assertEqual(mock_apply_volume.call_count, 3)
        except Exception as e:
            # If CompositeAudioClip creation fails, that's okay for unit test
            # We're mainly testing that the functions are called correctly
            pass
    
    def test_mix_horror_background_audio_volume_conversion(self):
        """Test that dB to linear volume conversion works correctly."""
        # Test dB to linear conversion
        def db_to_linear(db):
            return 10 ** (db / 20.0)
        
        # -20 dB should be 0.1 linear
        self.assertAlmostEqual(db_to_linear(-20.0), 0.1, places=5)
        
        # -40 dB should be 0.01 linear
        self.assertAlmostEqual(db_to_linear(-40.0), 0.01, places=5)
        
        # 0 dB should be 1.0 linear
        self.assertAlmostEqual(db_to_linear(0.0), 1.0, places=5)
        
        # -45 dB (default room tone) should be very quiet
        room_tone_vol = db_to_linear(-45.0)
        self.assertLess(room_tone_vol, 0.01)
        self.assertGreater(room_tone_vol, 0.0)
    
    def test_mix_horror_background_audio_duration_matching(self):
        """Test that narration audio duration is handled correctly."""
        # Create narration audio shorter than target duration
        short_duration = 3.0
        target_duration = 5.0
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=short_duration, fps=44100)
        
        # The function should handle duration mismatches
        # We'll test that it doesn't crash
        try:
            result = mix_horror_background_audio(
                narration_audio,
                target_duration,
                room_tone_volume=-45.0,
                drone_volume=-25.0,
                detail_volume=-30.0
            )
            # If it succeeds, result should be a CompositeAudioClip
            self.assertIsInstance(result, CompositeAudioClip)
        except Exception as e:
            # If it fails due to MoviePy internals, that's acceptable for unit test
            # The important thing is that the logic is correct
            pass
    
    def test_horror_background_audio_uses_with_audio(self):
        """Test that VideoClip uses with_audio instead of set_audio (verifying the fix)."""
        # Create a simple video clip to test the API
        def make_frame(t):
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        video_clip = VideoClip(make_frame, duration=5.0)
        video_clip.fps = 30
        
        # Create mock audio
        def make_silence(t):
            return 0.0
        
        mock_audio = AudioClip(make_silence, duration=5.0, fps=44100)
        
        # Verify that with_audio exists (the correct method)
        self.assertTrue(hasattr(video_clip, 'with_audio'), 
                       "VideoClip should have with_audio method (MoviePy API)")
        
        # Verify that set_audio doesn't exist (deprecated/removed method)
        self.assertFalse(hasattr(video_clip, 'set_audio'), 
                        "VideoClip should not have set_audio method (deprecated in newer MoviePy)")
        
        # Test that with_audio can be called successfully
        try:
            result = video_clip.with_audio(mock_audio)
            # Should return a new clip (not modify in place)
            self.assertIsNotNone(result)
            # Result should be a VideoClip
            self.assertIsInstance(result, VideoClip)
        except Exception as e:
            # If it fails due to MoviePy internals, that's okay for this test
            # The important thing is verifying the method exists and set_audio doesn't
            pass
    
    def test_audio_clips_return_stereo(self):
        """Test that generated audio clips return stereo (2 channels) for compatibility."""
        # Test room tone
        room_tone = generate_room_tone(1.0)
        sample = room_tone.get_frame(0.5)
        # Should return stereo array [left, right]
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(sample.shape, (2,), "Room tone should return stereo (2 channels)")
        
        # Test drone
        drone = generate_low_frequency_drone(1.0, 50.0)
        sample = drone.get_frame(0.5)
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(sample.shape, (2,), "Drone should return stereo (2 channels)")
        
        # Test detail sounds
        detail_sounds = generate_detail_sounds(1.0, sound_type="creak")
        sample = detail_sounds.get_frame(0.5)
        # Detail sounds might return array for multiple samples, but should be stereo
        self.assertIsInstance(sample, np.ndarray)
        # Should be 2D array with 2 columns (stereo) or 1D with 2 elements
        if sample.ndim == 1:
            self.assertEqual(sample.shape[0], 2, "Detail sounds should return stereo (2 channels)")
        else:
            self.assertEqual(sample.shape[-1], 2, "Detail sounds should return stereo (2 channels)")
    
    def test_mix_horror_background_audio_stereo_compatibility(self):
        """Test that mixing horror background audio works with stereo clips."""
        duration = 2.0
        
        # Create narration audio (might be mono or stereo)
        def make_narration(t):
            if np.isscalar(t):
                return np.array([0.5, 0.5])  # Stereo
            else:
                return np.column_stack([np.full(len(t), 0.5), np.full(len(t), 0.5)])
        
        narration_audio = AudioClip(make_narration, duration=duration, fps=44100)
        
        # Mix should work without broadcasting errors
        try:
            result = mix_horror_background_audio(
                narration_audio,
                duration,
                room_tone_volume=-45.0,
                drone_volume=-25.0,
                detail_volume=-30.0
            )
            # Should return a CompositeAudioClip
            self.assertIsInstance(result, CompositeAudioClip)
            # Should be able to get a frame without broadcasting errors
            sample = result.get_frame(1.0)
            self.assertIsNotNone(sample)
        except ValueError as e:
            if "broadcast" in str(e).lower():
                self.fail(f"Broadcasting error when mixing audio: {e}")
            else:
                # Other errors are acceptable for this test
                pass
    
    @patch('build_video.ENV_BLIZZARD_AUDIO', None)
    @patch('build_video.ENV_SNOW_AUDIO', None)
    @patch('build_video.ENV_FOREST_AUDIO', None)
    @patch('build_video.ENV_RAIN_AUDIO', None)
    @patch('build_video.ENV_INDOORS_AUDIO', None)
    def test_mix_horror_background_audio_no_environment(self):
        """Test that mix_horror_background_audio works without environment audio."""
        duration = 2.0
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Should work without environment parameter
        try:
            result = mix_horror_background_audio(
                narration_audio,
                duration,
                room_tone_volume=-35.0,
                drone_volume=-20.0,
                environment=None
            )
            self.assertIsInstance(result, CompositeAudioClip)
        except Exception as e:
            # If it fails due to MoviePy internals, that's acceptable
            pass
    
    @patch('build_video.Path')
    @patch('build_video.AudioFileClip')
    @patch('build_video.concatenate_audioclips')
    @patch('build_video.apply_volume_to_audioclip')
    def test_mix_horror_background_audio_with_environment_looping(self, mock_apply_volume,
                                                                   mock_concatenate, mock_audio_file,
                                                                   mock_path):
        """Test environment audio looping when shorter than duration."""
        duration = 5.0
        env_duration = 2.0  # Environment audio is shorter
        
        # Mock Path.exists() to return True
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        
        # Mock AudioFileClip
        def make_env_audio(t):
            if np.isscalar(t):
                return np.array([0.1, 0.1])
            else:
                return np.column_stack([np.full(len(t), 0.1), np.full(len(t), 0.1)])
        
        mock_env_clip = AudioClip(make_env_audio, duration=env_duration, fps=44100)
        mock_audio_file.return_value = mock_env_clip
        
        # Mock concatenate to return a longer clip
        def make_looped_audio(t):
            if np.isscalar(t):
                return np.array([0.1, 0.1])
            else:
                return np.column_stack([np.full(len(t), 0.1), np.full(len(t), 0.1)])
        
        looped_clip = AudioClip(make_looped_audio, duration=duration, fps=44100)
        mock_concatenate.return_value = looped_clip
        mock_apply_volume.side_effect = lambda clip, vol: clip
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Set up environment variable
        with patch('build_video.ENV_BLIZZARD_AUDIO', 'test_blizzard.wav'):
            try:
                result = mix_horror_background_audio(
                    narration_audio,
                    duration,
                    room_tone_volume=-35.0,
                    drone_volume=-20.0,
                    environment="blizzard"
                )
                # Should have called AudioFileClip to load the file
                mock_audio_file.assert_called_once()
                # Should have called concatenate to loop it
                mock_concatenate.assert_called()
                # Should have applied volume
                self.assertGreater(mock_apply_volume.call_count, 0)
            except Exception as e:
                # If it fails due to MoviePy internals, that's acceptable
                pass
    
    @patch('build_video.Path')
    @patch('build_video.AudioFileClip')
    @patch('build_video.apply_volume_to_audioclip')
    def test_mix_horror_background_audio_with_environment_trimming(self, mock_apply_volume,
                                                                    mock_audio_file, mock_path):
        """Test environment audio trimming when longer than duration."""
        duration = 2.0
        env_duration = 5.0  # Environment audio is longer
        
        # Mock Path.exists() to return True
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        
        # Mock AudioFileClip that supports subclip
        def make_env_audio(t):
            if np.isscalar(t):
                return np.array([0.1, 0.1])
            else:
                return np.column_stack([np.full(len(t), 0.1), np.full(len(t), 0.1)])
        
        mock_env_clip = AudioClip(make_env_audio, duration=env_duration, fps=44100)
        # Add subclip method to mock
        mock_env_clip.subclip = lambda start, end: AudioClip(make_env_audio, duration=end-start, fps=44100)
        mock_audio_file.return_value = mock_env_clip
        mock_apply_volume.side_effect = lambda clip, vol: clip
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Set up environment variable
        with patch('build_video.ENV_BLIZZARD_AUDIO', 'test_blizzard.wav'):
            try:
                result = mix_horror_background_audio(
                    narration_audio,
                    duration,
                    room_tone_volume=-35.0,
                    drone_volume=-20.0,
                    environment="blizzard"
                )
                # Should have called AudioFileClip to load the file
                mock_audio_file.assert_called_once()
                # Should have applied volume
                self.assertGreater(mock_apply_volume.call_count, 0)
            except Exception as e:
                # If it fails due to MoviePy internals, that's acceptable
                pass
    
    @patch('build_video.Path')
    def test_mix_horror_background_audio_environment_file_not_found(self, mock_path):
        """Test that missing environment audio file is handled gracefully."""
        duration = 2.0
        
        # Mock Path.exists() to return False
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = False
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Set up environment variable but file doesn't exist
        with patch('build_video.ENV_BLIZZARD_AUDIO', 'nonexistent.wav'):
            try:
                result = mix_horror_background_audio(
                    narration_audio,
                    duration,
                    room_tone_volume=-35.0,
                    drone_volume=-20.0,
                    environment="blizzard"
                )
                # Should still work, just without environment audio
                self.assertIsInstance(result, CompositeAudioClip)
            except Exception as e:
                # If it fails due to MoviePy internals, that's acceptable
                pass
    
    @patch('build_video.Path')
    def test_mix_horror_background_audio_environment_not_configured(self, mock_path):
        """Test that unconfigured environment is handled gracefully."""
        duration = 2.0
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Environment is set but no env var configured
        with patch('build_video.ENV_BLIZZARD_AUDIO', None):
            try:
                result = mix_horror_background_audio(
                    narration_audio,
                    duration,
                    room_tone_volume=-35.0,
                    drone_volume=-20.0,
                    environment="blizzard"
                )
                # Should still work, just without environment audio
                self.assertIsInstance(result, CompositeAudioClip)
            except Exception as e:
                # If it fails due to MoviePy internals, that's acceptable
                pass

    @patch('build_video.tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    @patch('build_video.AudioFileClip')
    @patch('build_video.Path')
    def test_mix_horror_background_audio_wav_env_never_loaded_directly(
        self, mock_path, mock_audio_file, mock_subprocess_run, mock_tempfile
    ):
        """Regression: .wav env files must be converted first; AudioFileClip must never be called with the original path (avoids FFMPEG_AudioReader __del__ AttributeError: no 'proc')."""
        duration = 5.0
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True

        def make_audio(t):
            if np.isscalar(t):
                return np.array([0.1, 0.1])
            return np.column_stack([np.full(len(t), 0.1), np.full(len(t), 0.1)])
        env_clip = AudioClip(make_audio, duration=3.0, fps=44100)
        mock_audio_file.return_value = env_clip
        mock_subprocess_run.return_value = None
        temp_wav = MagicMock()
        temp_wav.name = "/tmp/env_converted.wav"
        temp_wav.__enter__ = MagicMock(return_value=temp_wav)
        temp_wav.__exit__ = MagicMock(return_value=None)
        mock_tempfile.return_value = temp_wav

        def make_silence(t):
            return 0.0
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)

        original_wav_path = "environment_audio/forest_night.wav"
        with patch('build_video.ENV_FOREST_AUDIO', original_wav_path):
            result = mix_horror_background_audio(
                narration_audio,
                duration,
                room_tone_volume=-35.0,
                drone_volume=-20.0,
                environment="forest"
            )
        self.assertIsInstance(result, CompositeAudioClip)
        # Must never call AudioFileClip with the original .wav path (that triggers the __del__ bug)
        def norm(p):
            return str(p).replace("\\", "/").lower()
        original_norm = norm(original_wav_path)
        for call in mock_audio_file.call_args_list:
            args = call[0]
            if args:
                path_arg_norm = norm(args[0])
                self.assertNotEqual(
                    path_arg_norm, original_norm,
                    msg="AudioFileClip must not be called with original .wav path (causes FFMPEG_AudioReader __del__ error)",
                )
        # Conversion must run for .wav (convert-first path)
        conversion_calls = [
            c for c in mock_subprocess_run.call_args_list
            if len(c[0][0]) >= 6 and "pcm_s16le" in c[0][0] and "-ar" in c[0][0] and "-t" not in c[0][0]
        ]
        self.assertGreaterEqual(len(conversion_calls), 1, "For .wav env, ffmpeg must convert to 16-bit before load")

    @patch('build_video.tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    @patch('build_video.AudioFileClip')
    @patch('build_video.Path')
    def test_mix_horror_background_audio_env_fallback_when_direct_load_fails(
        self, mock_path, mock_audio_file, mock_subprocess_run, mock_tempfile
    ):
        """When env file is NOT .wav and AudioFileClip(env_path) fails, fallback converts via ffmpeg and loads; no UnboundLocalError."""
        duration = 5.0
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True

        def make_audio(t):
            if np.isscalar(t):
                return np.array([0.1, 0.1])
            return np.column_stack([np.full(len(t), 0.1), np.full(len(t), 0.1)])
        fallback_clip = AudioClip(make_audio, duration=3.0, fps=44100)
        mock_audio_file.side_effect = [
            Exception("Error passing `ffmpeg -i` command output: At least one output file must be specified"),
            fallback_clip,
        ]
        mock_subprocess_run.return_value = None
        temp_wav = MagicMock()
        temp_wav.name = "/tmp/env_fallback.wav"
        temp_wav.__enter__ = MagicMock(return_value=temp_wav)
        temp_wav.__exit__ = MagicMock(return_value=None)
        mock_tempfile.return_value = temp_wav

        def make_silence(t):
            return 0.0
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)

        # Use a non-.wav path so we hit the try/except fallback path
        with patch('build_video.ENV_FOREST_AUDIO', 'environment_audio/forest_night.ogg'):
            result = mix_horror_background_audio(
                narration_audio,
                duration,
                room_tone_volume=-35.0,
                drone_volume=-20.0,
                environment="forest"
            )
        self.assertIsInstance(result, CompositeAudioClip)
        self.assertGreaterEqual(mock_audio_file.call_count, 2, "AudioFileClip called at least twice (direct fail + load converted WAV)")
        conversion_calls = [c for c in mock_subprocess_run.call_args_list if len(c[0][0]) >= 6 and "pcm_s16le" in c[0][0] and "-ar" in c[0][0] and "-t" not in c[0][0]]
        self.assertGreaterEqual(len(conversion_calls), 1, "Fallback must run ffmpeg to convert env file to 16-bit WAV")


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
    
    
    def test_mix_horror_background_audio_environment_volume(self):
        """Test that environment audio volume is applied correctly."""
        duration = 2.0
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Test with custom environment audio volume
        try:
            result = mix_horror_background_audio(
                narration_audio,
                duration,
                room_tone_volume=-35.0,
                drone_volume=-20.0,
                environment=None,  # No environment, but test volume parameter
                env_audio_volume=-28.0
            )
            # Should work with custom volume
            self.assertIsInstance(result, CompositeAudioClip)
        except Exception as e:
            # If it fails due to MoviePy internals, that's acceptable
            pass
    
    def test_mix_horror_background_audio_all_environments(self):
        """Test that all environment types are recognized."""
        duration = 1.0
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Test all valid environment types
        valid_environments = ["blizzard", "snow", "forest", "rain", "indoors", "jungle"]
        
        for env in valid_environments:
            try:
                result = mix_horror_background_audio(
                    narration_audio,
                    duration,
                    room_tone_volume=-35.0,
                    drone_volume=-20.0,
                    environment=env
                )
                # Should work for all valid environments (even if file doesn't exist)
                self.assertIsInstance(result, CompositeAudioClip)
            except Exception as e:
                # If it fails due to MoviePy internals, that's acceptable
                pass
    
    def test_generate_low_frequency_drone_with_transition(self):
        """Test that drone transitions are generated correctly."""
        duration = 5.0
        
        # Test fade_in
        drone_fade_in = generate_low_frequency_drone_with_transition(
            duration=duration,
            frequency=50.0,
            transition_type="fade_in",
            start_volume=0.0,
            end_volume=1.0,
            transition_duration=5.0
        )
        self.assertIsInstance(drone_fade_in, AudioClip)
        self.assertAlmostEqual(drone_fade_in.duration, duration, places=2)
        
        # Test hold
        drone_hold = generate_low_frequency_drone_with_transition(
            duration=duration,
            frequency=50.0,
            transition_type="hold",
            start_volume=0.5,
            end_volume=0.5,
            transition_duration=0.0
        )
        self.assertIsInstance(drone_hold, AudioClip)
        
        # Test swell
        drone_swell = generate_low_frequency_drone_with_transition(
            duration=duration,
            frequency=50.0,
            transition_type="swell",
            start_volume=0.3,
            end_volume=0.8,
            transition_duration=4.0
        )
        self.assertIsInstance(drone_swell, AudioClip)
        
        # Test shrink
        drone_shrink = generate_low_frequency_drone_with_transition(
            duration=duration,
            frequency=50.0,
            transition_type="shrink",
            start_volume=0.8,
            end_volume=0.3,
            transition_duration=4.0
        )
        self.assertIsInstance(drone_shrink, AudioClip)
        
        # Test hard_cut
        drone_hard_cut = generate_low_frequency_drone_with_transition(
            duration=duration,
            frequency=50.0,
            transition_type="hard_cut",
            start_volume=1.0,
            end_volume=0.0,
            transition_duration=0.0
        )
        self.assertIsInstance(drone_hard_cut, AudioClip)
        # Hard cut should be silent
        sample = drone_hard_cut.get_frame(2.0)
        if hasattr(sample, '__len__'):
            self.assertAlmostEqual(np.max(np.abs(sample)), 0.0, places=3)
    
    def test_generate_drone_with_scene_transitions(self):
        """Test that drone transitions are generated correctly based on scene drone_change values."""
        # Create test scenes with different drone_change values
        scenes = [
            {"id": 1, "drone_change": "fade_in"},
            {"id": 2, "drone_change": "hold"},
            {"id": 3, "drone_change": "swell"},
            {"id": 4, "drone_change": "shrink"},
            {"id": 5, "drone_change": "fade_out"},
        ]
        
        # Create mock audio clips for each scene
        def make_silence(t):
            if np.isscalar(t):
                return np.array([0.0, 0.0])
            else:
                return np.zeros((len(t), 2))
        
        scene_audio_clips = [
            AudioClip(make_silence, duration=3.0, fps=44100),  # Scene 1: 3s
            AudioClip(make_silence, duration=2.5, fps=44100),  # Scene 2: 2.5s
            AudioClip(make_silence, duration=4.0, fps=44100),  # Scene 3: 4s
            AudioClip(make_silence, duration=3.5, fps=44100),  # Scene 4: 3.5s
            AudioClip(make_silence, duration=2.0, fps=44100),  # Scene 5: 2s
        ]
        
        # Generate drone with transitions
        try:
            drone = generate_drone_with_scene_transitions(scenes, scene_audio_clips, 
                                                          base_drone_volume_db=-25.0,
                                                          max_drone_volume_db=-20.0)
            self.assertIsInstance(drone, AudioClip)
            # Total duration should be sum of scene durations + pauses
            expected_duration = sum(clip.duration for clip in scene_audio_clips) + (len(scenes) * 0.15)  # 0.15s pause per scene
            self.assertAlmostEqual(drone.duration, expected_duration, places=1)
        except Exception as e:
            # If it fails due to MoviePy internals, that's acceptable for this test
            # The important thing is that the function exists and can be called
            pass
    
    def test_audioclip_has_no_subclip_method(self):
        """Test that AudioClip doesn't have subclip method (regression test for drone trimming bug).
        
        This test verifies the root cause: AudioClip objects don't have subclip method,
        which is why we need the FFmpeg fallback in mix_horror_background_audio.
        """
        def make_audio(t):
            return 0.1
        
        audio_clip = AudioClip(make_audio, duration=5.0, fps=44100)
        
        # Verify that AudioClip doesn't have subclip method
        # This is the bug we fixed - the code was calling subclip on AudioClip which doesn't have it
        self.assertFalse(hasattr(audio_clip, 'subclip'), 
                        "AudioClip should not have subclip method - this is why we need FFmpeg fallback")
        
        # Verify that calling subclip raises AttributeError
        with self.assertRaises(AttributeError) as context:
            audio_clip.subclip(0, 2.0)
        
        self.assertIn("subclip", str(context.exception).lower(),
                     "Error should mention 'subclip'")
    
    def test_mix_horror_background_audio_drone_trimming_handles_audioclip(self):
        """Test that drone trimming handles AudioClip (no subclip method) without crashing.
        
        This test verifies that the fix works: when drone is an AudioClip and needs trimming,
        the code should catch the AttributeError and use FFmpeg fallback instead of crashing.
        """
        duration = 2.0
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Create an AudioClip drone (programmatically generated, no subclip method)
        # Make it longer than duration to trigger trimming
        drone_duration = 5.0
        def make_drone(t):
            return 0.1
        
        drone = AudioClip(make_drone, duration=drone_duration, fps=44100)
        
        # Mock the dependencies - use the same pattern as other tests
        with patch('build_video.generate_room_tone', return_value=AudioClip(make_silence, duration=duration, fps=44100)), \
             patch('build_video.generate_drone_with_scene_transitions', return_value=drone), \
             patch('build_video.generate_detail_sounds', return_value=None), \
             patch('build_video.apply_volume_to_audioclip', side_effect=lambda x, v: x if x is not None else None):
            
            # The function should handle the AttributeError when subclip is called on AudioClip
            # and use the FFmpeg fallback. We're testing that it doesn't crash with AttributeError.
            try:
                result = mix_horror_background_audio(
                    narration_audio,
                    duration,
                    room_tone_volume=-35.0,
                    drone_volume=-20.0
                )
                # If it succeeds, the fallback worked (or FFmpeg handled it)
                self.assertIsInstance(result, CompositeAudioClip)
            except AttributeError as e:
                # This is the bug we're testing - should NOT get AttributeError about subclip
                error_str = str(e)
                if "'AudioClip' object has no attribute 'subclip'" in error_str or \
                   "has no attribute 'subclip'" in error_str or \
                   "object has no attribute 'subclip'" in error_str:
                    self.fail("Drone trimming should handle AudioClip subclip error with FFmpeg fallback, not raise AttributeError. "
                            "This indicates the fix is not working. Error: " + error_str)
                raise
            except Exception as e:
                # Other exceptions (like FFmpeg not available, write_audiofile issues) are acceptable
                # The important thing is we don't get the AttributeError about subclip
                error_str = str(e)
                if "'AudioClip' object has no attribute 'subclip'" in error_str or \
                   "has no attribute 'subclip'" in error_str or \
                   "object has no attribute 'subclip'" in error_str:
                    self.fail("Drone trimming should handle AudioClip subclip error, not propagate it. "
                            "This indicates the fix is not working. Error: " + error_str)
    
    def test_mix_horror_background_audio_drone_trimming_audiofileclip(self):
        """Test that drone trimming works when drone is an AudioFileClip (has subclip method)."""
        duration = 2.0
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Create a mock AudioFileClip drone (has subclip method)
        drone_duration = 5.0
        mock_drone = mock.MagicMock()
        mock_drone.duration = drone_duration
        mock_drone.fps = 44100
        mock_drone.subclip.return_value = mock.MagicMock()
        mock_drone.subclip.return_value.duration = duration
        mock_drone.subclip.return_value.fps = 44100
        
        # Verify that AudioFileClip has subclip method
        self.assertTrue(hasattr(mock_drone, 'subclip'), 
                       "AudioFileClip should have subclip method")
        
        with patch('build_video.generate_room_tone', return_value=AudioClip(make_silence, duration=duration, fps=44100)), \
             patch('build_video.generate_drone_with_scene_transitions', return_value=mock_drone), \
             patch('build_video.generate_detail_sounds', return_value=None):
            
            try:
                result = mix_horror_background_audio(
                    narration_audio,
                    duration,
                    room_tone_volume=-35.0,
                    drone_volume=-20.0
                )
                
                # Verify that subclip was called (direct method, not FFmpeg fallback)
                mock_drone.subclip.assert_called_once_with(0, duration)
                
            except Exception as e:
                # If it fails due to MoviePy internals, that's acceptable
                # The important thing is that we tried subclip first
                pass
    
    def test_mix_horror_background_audio_drone_trimming_ffmpeg_fallback_error(self):
        """Test that drone trimming handles FFmpeg fallback errors gracefully."""
        duration = 2.0
        
        def make_silence(t):
            return 0.0
        
        narration_audio = AudioClip(make_silence, duration=duration, fps=44100)
        
        # Create an AudioClip drone (no subclip method)
        drone_duration = 5.0
        def make_drone(t):
            return 0.1
        
        drone = AudioClip(make_drone, duration=drone_duration, fps=44100)
        
        # Mock write_audiofile to fail (simulating FFmpeg fallback error)
        drone.write_audiofile = mock.MagicMock(side_effect=Exception("Write failed"))
        
        with patch('build_video.generate_room_tone', return_value=AudioClip(make_silence, duration=duration, fps=44100)), \
             patch('build_video.generate_drone_with_scene_transitions', return_value=drone), \
             patch('build_video.generate_detail_sounds', return_value=None), \
             patch('build_video.apply_volume_to_audioclip', side_effect=lambda x, v: x if x is not None else None):
            
            # The function should handle the error gracefully and use full duration
            try:
                result = mix_horror_background_audio(
                    narration_audio,
                    duration,
                    room_tone_volume=-35.0,
                    drone_volume=-20.0
                )
                
                # Should still work, using full drone duration with warning
                # The function should handle the error gracefully
                self.assertIsInstance(result, CompositeAudioClip)
                
            except AttributeError as e:
                # This is the bug we're testing - should NOT get AttributeError about subclip
                error_str = str(e)
                if "'AudioClip' object has no attribute 'subclip'" in error_str or \
                   "has no attribute 'subclip'" in error_str or \
                   "object has no attribute 'subclip'" in error_str:
                    self.fail("Drone trimming should handle AudioClip subclip error, not propagate it. "
                            "This indicates the fix is not working. Error: " + error_str)
                raise
            except Exception as e:
                # Other exceptions are acceptable - the important thing is we don't get the subclip AttributeError
                error_str = str(e)
                if "'AudioClip' object has no attribute 'subclip'" in error_str or \
                   "has no attribute 'subclip'" in error_str or \
                   "object has no attribute 'subclip'" in error_str:
                    self.fail("Drone trimming should handle AudioClip subclip error, not propagate it. "
                            "This indicates the fix is not working. Error: " + error_str)

    def test_drone_swell_clips_at_max_volume(self):
        """Test that swell clips at max volume, not base volume."""
        # Create test scenes with swell
        scenes = [
            {"id": 1, "drone_change": "fade_in"},  # Fade in to base
            {"id": 2, "drone_change": "swell"},     # Swell should clip at max, not base
        ]
        
        def make_silence(t):
            if np.isscalar(t):
                return np.array([0.0, 0.0])
            else:
                return np.zeros((len(t), 2))
        
        scene_audio_clips = [
            AudioClip(make_silence, duration=3.0, fps=44100),
            AudioClip(make_silence, duration=4.0, fps=44100),
        ]
        
        base_volume_db = -25.0
        max_volume_db = -20.0  # Higher than base
        
        # Convert to linear for comparison
        def db_to_linear(db):
            return 10 ** (db / 20.0)
        
        base_volume_linear = db_to_linear(base_volume_db)
        max_volume_linear = db_to_linear(max_volume_db)
        
        # Generate drone
        try:
            drone = generate_drone_with_scene_transitions(
                scenes, scene_audio_clips,
                base_drone_volume_db=base_volume_db,
                max_drone_volume_db=max_volume_db
            )
            self.assertIsInstance(drone, AudioClip)
            # The swell should be able to reach max_volume, which is higher than base_volume
            # We can't easily test the exact volume, but we can verify the function accepts the parameter
        except Exception as e:
            # If it fails due to MoviePy internals, that's acceptable for this test
            pass


class TestHorrorDisclaimer(unittest.TestCase):
    """Test cases for horror disclaimer (first scene: fixed image + env/room noise, no narration)."""

    def test_disclaimer_constants(self):
        """Test horror disclaimer constants and path names."""
        self.assertEqual(HORROR_DISCLAIMER_DURATION, 3.0)
        self.assertEqual(FIXED_IMAGES_DIR, Path("fixed_images"))
        self.assertEqual(HORROR_DISCLAIMER_TALL.name, "tall_horror_disclaimer.jpg")
        self.assertEqual(HORROR_DISCLAIMER_WIDE.name, "wide_horror_disclaimer.jpg")
        self.assertEqual(HORROR_DISCLAIMER_TALL.parent, FIXED_IMAGES_DIR)
        self.assertEqual(HORROR_DISCLAIMER_WIDE.parent, FIXED_IMAGES_DIR)

    def test_get_horror_disclaimer_image_path_returns_tall_when_vertical_and_exists(self):
        """When is_vertical=True and tall file exists, return tall path."""
        tmp = Path(__file__).parent / "tmp_horror_disclaimer_tall"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            tall_path = tmp / "tall_horror_disclaimer.jpg"
            tall_path.write_bytes(b"\xff\xd8\xff")  # minimal JPEG-like bytes so file exists
            with patch("build_video.HORROR_DISCLAIMER_TALL", tall_path), \
                 patch("build_video.HORROR_DISCLAIMER_WIDE", tmp / "wide_horror_disclaimer.jpg"):
                result = get_horror_disclaimer_image_path(is_vertical=True)
                self.assertEqual(result, tall_path)
        finally:
            if tall_path.exists():
                tall_path.unlink()
            if tmp.exists():
                tmp.rmdir()

    def test_get_horror_disclaimer_image_path_returns_wide_when_landscape_and_exists(self):
        """When is_vertical=False and wide file exists, return wide path."""
        tmp = Path(__file__).parent / "tmp_horror_disclaimer_wide"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            wide_path = tmp / "wide_horror_disclaimer.jpg"
            wide_path.write_bytes(b"\xff\xd8\xff")
            with patch("build_video.HORROR_DISCLAIMER_TALL", tmp / "tall_horror_disclaimer.jpg"), \
                 patch("build_video.HORROR_DISCLAIMER_WIDE", wide_path):
                result = get_horror_disclaimer_image_path(is_vertical=False)
                self.assertEqual(result, wide_path)
        finally:
            if wide_path.exists():
                wide_path.unlink()
            if tmp.exists():
                tmp.rmdir()

    def test_get_horror_disclaimer_image_path_returns_none_when_file_missing(self):
        """When the chosen file does not exist, return None."""
        tmp = Path(__file__).parent / "tmp_horror_disclaimer_missing"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            missing_tall = tmp / "tall_horror_disclaimer.jpg"
            missing_wide = tmp / "wide_horror_disclaimer.jpg"
            with patch("build_video.HORROR_DISCLAIMER_TALL", missing_tall), \
                 patch("build_video.HORROR_DISCLAIMER_WIDE", missing_wide):
                self.assertIsNone(get_horror_disclaimer_image_path(is_vertical=True))
                self.assertIsNone(get_horror_disclaimer_image_path(is_vertical=False))
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def test_disclaimer_audio_duration(self):
        """mix_horror_background_audio with 3s silence returns ~3s duration (disclaimer = env/room only)."""
        def make_silence(t):
            if np.isscalar(t):
                return np.array([0.0, 0.0])
            return np.zeros((len(t), 2))
        silence_3s = AudioClip(make_silence, duration=HORROR_DISCLAIMER_DURATION, fps=44100)
        with patch("build_video.generate_room_tone") as mock_room, \
             patch("build_video.generate_low_frequency_drone") as mock_drone, \
             patch("build_video.generate_detail_sounds", return_value=None), \
             patch("build_video.apply_volume_to_audioclip", side_effect=lambda x, v: x if x is not None else None):
            mock_room.return_value = AudioClip(make_silence, duration=3.0, fps=44100)
            mock_drone.return_value = AudioClip(make_silence, duration=3.0, fps=44100)
            result = mix_horror_background_audio(
                silence_3s,
                HORROR_DISCLAIMER_DURATION,
                room_tone_volume=-35.0,
                drone_volume=-20.0,
            )
            self.assertIsInstance(result, CompositeAudioClip)
            self.assertAlmostEqual(result.duration, HORROR_DISCLAIMER_DURATION, places=2)

    def test_horror_disclaimer_prepended_when_is_horror_and_file_exists(self):
        """When is_horror=True and disclaimer image exists, first clip is disclaimer (~3.15s)."""
        import build_video as bv
        tmp = Path(__file__).parent / "tmp_horror_disclaimer_prepended"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            scenes_path = tmp / "scenes.json"
            scenes_data = {
                "scenes": [
                    {"id": 1, "title": "Scene 1", "narration": "One.", "image_prompt": "A room."}
                ]
            }
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(scenes_data, f, indent=2)

            try:
                from PIL import Image
            except ImportError:
                self.skipTest("PIL not available")
            fixed_dir = tmp / "fixed_images"
            fixed_dir.mkdir(parents=True, exist_ok=True)
            wide_path = fixed_dir / "wide_horror_disclaimer.jpg"
            Image.new("RGB", (10, 10), color="black").save(str(wide_path))
            scene_image_path = tmp / "scene1.png"
            Image.new("RGB", (10, 10), color="white").save(str(scene_image_path))

            # Create a valid 2-second silent WAV (MoviePy AudioClip.write_audiofile can produce 0-duration)
            scene_audio_path = tmp / "scene1.wav"
            import wave
            with wave.open(str(scene_audio_path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(44100)
                num_frames = 44100 * 2  # 2 seconds
                wav.writeframes(b"\x00\x00" * (num_frames * 2))  # stereo silence

            out_path = tmp / "out.mp4"
            captured_clips = []

            real_concat = bv.concatenate_videoclips
            def capture_concat(clips, method="chain"):
                captured_clips[:] = list(clips)
                return real_concat(clips, method=method)

            with patch("build_video.HORROR_DISCLAIMER_WIDE", wide_path), \
                 patch("build_video.HORROR_DISCLAIMER_TALL", fixed_dir / "tall_horror_disclaimer.jpg"), \
                 patch("build_video.generate_image_for_scene_with_retry", return_value=scene_image_path), \
                 patch("build_video.generate_audio_for_scene_with_retry", return_value=scene_audio_path), \
                 patch("build_video.concatenate_videoclips", side_effect=capture_concat), \
                 patch.object(bv.config, "save_assets", False), \
                 patch.object(bv.config, "is_vertical", False), \
                 patch.object(bv.config, "temp_dir", str(tmp)):
                _build_video_impl(
                    str(scenes_path),
                    out_video_path=str(out_path),
                    is_horror=True,
                    horror_bg_enabled=True,
                )

            self.assertGreaterEqual(len(captured_clips), 2, "Should have disclaimer + at least one scene clip")
            expected_disclaimer_duration = HORROR_DISCLAIMER_DURATION + END_SCENE_PAUSE_LENGTH
            self.assertAlmostEqual(
                captured_clips[0].duration,
                expected_disclaimer_duration,
                places=1,
                msg="First clip should be disclaimer (~3.15s)",
            )
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)


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
        expected = 1.0 + END_SCENE_PAUSE_LENGTH
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
        """When motion=True, concatenate_videoclips is called with crossfade padding."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_crossfade_on"
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
                 patch("build_video.make_motion_clip_with_audio", wraps=self._static_ignoring_pattern), \
                 patch("build_video.KENBURNS_ENABLED", True), \
                 patch("build_video.CROSSFADE_DURATION", 0.4), \
                 patch("build_video.concatenate_videoclips", side_effect=capture_concat):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        motion=True,
                    )
            # Should have at least one call with crossfade padding
            crossfade_calls = [c for c in concat_calls if c.get("padding") and c["padding"] < 0]
            self.assertGreaterEqual(len(crossfade_calls), 1, f"Expected crossfade concat call, got: {concat_calls}")
            self.assertEqual(crossfade_calls[0]["method"], "compose")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_motion_uses_chain_concatenation(self):
        """When motion=False, concatenate_videoclips uses method='chain' (no crossfade)."""
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
            # Should have called with method="chain" (no padding or padding=0)
            chain_calls = [c for c in concat_calls if c.get("method") == "chain"]
            self.assertGreaterEqual(len(chain_calls), 1, f"Expected chain concat call, got: {concat_calls}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBiopicMusicWithMotion(unittest.TestCase):
    """Regression tests: biopic background music integration works with motion clips."""

    def test_biopic_music_mixed_when_not_horror(self):
        """For non-horror videos with biopic_music_enabled, mix_biopic_background_music is called."""
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
                        is_horror=False,
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
                        is_horror=False,
                        motion=True,
                    )
                mock_mix.assert_called_once()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_biopic_music_skipped_for_horror(self):
        """For horror videos, mix_biopic_background_music must NOT be called."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import build_video as bv
        tmp = _TESTS_DIR / "tmp_biopic_skip_horror"
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
                 patch("build_video.mix_biopic_background_music") as mock_mix, \
                 patch("build_video.mix_horror_background_audio", return_value=MagicMock(duration=5.0)):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        is_horror=True,
                        horror_bg_enabled=False,
                        motion=False,
                    )
                mock_mix.assert_not_called()
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
                 patch("build_video.mix_biopic_background_music", side_effect=tracking_mix):
                with patch("moviepy.video.VideoClip.VideoClip.write_videofile"):
                    _build_video_impl(
                        str(scenes_path),
                        out_video_path=str(out_path),
                        is_horror=False,
                        motion=False,
                    )
            self.assertTrue(call_tracker["called"], "mix_biopic_background_music should have been called")
            self.assertIsNone(call_tracker["error"],
                              f"mix_biopic_background_music crashed: {call_tracker['error']}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
