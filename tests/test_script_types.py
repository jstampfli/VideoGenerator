"""
Unit tests for script_types.py.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import script_types


class TestScriptTypeBase(unittest.TestCase):
    """Test cases for ScriptType base class."""
    
    def test_script_type_is_abstract(self):
        """Test that ScriptType cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            script_types.ScriptType()
    
    def test_script_type_has_required_methods(self):
        """Test that ScriptType defines all required abstract methods."""
        # Check that all abstract methods exist
        self.assertTrue(hasattr(script_types.ScriptType, 'get_outline_prompt'))
        self.assertTrue(hasattr(script_types.ScriptType, 'get_scene_generation_prompt'))
        self.assertTrue(hasattr(script_types.ScriptType, 'get_refinement_passes'))
        self.assertTrue(hasattr(script_types.ScriptType, 'get_metadata_prompt'))


class TestMainVideoScript(unittest.TestCase):
    """Test cases for MainVideoScript class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.script_type = script_types.MainVideoScript()
    
    def test_get_outline_prompt(self):
        """Test get_outline_prompt returns valid prompt."""
        result = self.script_type.get_outline_prompt("Einstein", chapters=6, total_scenes=24)
        self.assertIsInstance(result, str)
        self.assertIn("Einstein", result)
        self.assertIn("6", result)
        self.assertIn("24", result)
    
    def test_get_scene_generation_prompt_not_implemented(self):
        """Test that get_scene_generation_prompt raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.script_type.get_scene_generation_prompt({})
    
    def test_get_refinement_passes(self):
        """Test get_refinement_passes returns all passes."""
        result = self.script_type.get_refinement_passes()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIn('storyline', result)
        self.assertIn('pivotal', result)
        self.assertIn('final', result)
    
    def test_get_metadata_prompt(self):
        """Test get_metadata_prompt returns valid prompt."""
        result = self.script_type.get_metadata_prompt("Einstein", "the man who changed the world", 24)
        self.assertIsInstance(result, str)
        self.assertIn("Einstein", result)
        self.assertIn("the man who changed the world", result)
        self.assertIn("24", result)


class TestShortScript(unittest.TestCase):
    """Test cases for ShortScript class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.script_type = script_types.ShortScript()
    
    def test_get_outline_prompt_not_implemented(self):
        """Test that get_outline_prompt raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.script_type.get_outline_prompt("Einstein", chapters=1, total_scenes=3)
    
    def test_get_scene_generation_prompt_not_implemented(self):
        """Test that get_scene_generation_prompt raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.script_type.get_scene_generation_prompt({})
    
    def test_get_refinement_passes(self):
        """Test get_refinement_passes returns only final pass."""
        result = self.script_type.get_refinement_passes()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn('final', result)
        self.assertNotIn('storyline', result)
        self.assertNotIn('pivotal', result)
    
    def test_get_metadata_prompt_not_implemented(self):
        """Test that get_metadata_prompt raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.script_type.get_metadata_prompt("Einstein", "tagline", 3)


if __name__ == "__main__":
    unittest.main()
