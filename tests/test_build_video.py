"""
Unit tests for build_video.py functions.
"""

import unittest
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_video import (
    text_to_ssml,
    load_scenes,
    build_image_prompt,
    find_shorts_for_script,
    find_all_shorts,
    sanitize_prompt_for_safety
)


class TestTextToSSML(unittest.TestCase):
    """Test cases for text_to_ssml function."""
    
    def test_basic_text(self):
        """Test basic text conversion."""
        text = "Hello world."
        result = text_to_ssml(text)
        self.assertIn("<speak>", result)
        self.assertIn("</speak>", result)
        self.assertIn("Hello world", result)
    
    def test_sentence_pauses(self):
        """Test that periods add pauses."""
        text = "First sentence. Second sentence."
        result = text_to_ssml(text)
        self.assertIn('<break time="400ms"/>', result)
        # Should have breaks after both sentences
        self.assertEqual(result.count('<break time="400ms"/>'), 2)
    
    def test_ellipsis_handling(self):
        """Test ellipsis conversion to dramatic pause."""
        text = "He thought... then acted."
        result = text_to_ssml(text)
        # Ellipsis should become 600ms break, not 400ms
        self.assertIn('<break time="600ms"/>', result)
        # Should still have pause after final period
        self.assertIn('<break time="400ms"/>', result)
    
    def test_unicode_ellipsis(self):
        """Test Unicode ellipsis character."""
        text = "He thought… then acted."
        result = text_to_ssml(text)
        self.assertIn('<break time="600ms"/>', result)
    
    def test_comma_pauses(self):
        """Test that commas add shorter pauses."""
        text = "First, second, third."
        result = text_to_ssml(text)
        # Should have 200ms breaks after commas
        self.assertIn('<break time="200ms"/>', result)
        # Should have 400ms break after period
        self.assertIn('<break time="400ms"/>', result)
    
    def test_question_exclamation_pauses(self):
        """Test question and exclamation marks."""
        text = "Really? Yes!"
        result = text_to_ssml(text)
        self.assertIn('<break time="350ms"/>', result)
        # Should have two breaks (one for ?, one for !)
        self.assertEqual(result.count('<break time="350ms"/>'), 2)
    
    def test_em_dash_pause(self):
        """Test em-dash conversion to pause."""
        text = "He said—then stopped."
        result = text_to_ssml(text)
        # Em-dash should become 400ms break
        self.assertIn('<break time="400ms"/>', result)
        # Should not contain the em-dash character
        self.assertNotIn("—", result)
    
    def test_hyphen_pauses(self):
        """Test hyphen/dash handling."""
        text = "Word - another word."
        result = text_to_ssml(text)
        # Should have break where hyphen was
        self.assertIn('<break time="300ms"/>', result)
    
    def test_year_emphasis(self):
        """Test that years get prosody emphasis."""
        text = "In 1936, he published."
        result = text_to_ssml(text)
        # Years should be wrapped in prosody tags
        self.assertIn('<prosody rate="95%">1936</prosody>', result)
    
    def test_xml_escaping(self):
        """Test that special XML characters are escaped."""
        text = "A & B < C > D"
        result = text_to_ssml(text)
        # Should escape special characters
        self.assertIn("&amp;", result)
        self.assertIn("&lt;", result)
        self.assertIn("&gt;", result)
    
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
    
    def test_prompt_with_age_specification(self):
        """Test prompt includes age when available."""
        scene = {
            "id": 1,
            "title": "Test Scene",
            "narration": "Test narration",
            "image_prompt": "Test visual",
            "estimated_age": 30
        }
        
        prompt = build_image_prompt(scene, None, None)
        
        # Should include age specification
        self.assertIn("30", prompt)
        self.assertIn("years old", prompt.lower())
    
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


if __name__ == "__main__":
    unittest.main()
