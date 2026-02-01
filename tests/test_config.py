"""
Unit tests for config.py.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def test_config_attributes(self):
        """Test that Config has all expected attributes."""
        cfg = config.Config()
        
        # Main video settings
        self.assertIsInstance(cfg.chapters, int)
        self.assertIsInstance(cfg.scenes_per_chapter, int)
        self.assertIsInstance(cfg.generate_main, bool)
        
        # Shorts settings
        self.assertIsInstance(cfg.num_shorts, int)
        self.assertIsInstance(cfg.short_chapters, int)
        self.assertIsInstance(cfg.short_scenes_per_chapter, int)
        
        # Generation flags
        self.assertIsInstance(cfg.generate_thumbnails, bool)
        self.assertIsInstance(cfg.generate_short_thumbnails, bool)
        self.assertIsInstance(cfg.generate_refinement_diffs, bool)
    
    def test_total_scenes_property(self):
        """Test total_scenes property calculation."""
        cfg = config.Config()
        expected = cfg.chapters * cfg.scenes_per_chapter
        self.assertEqual(cfg.total_scenes, expected)
    
    def test_total_short_scenes_property(self):
        """Test total_short_scenes property calculation."""
        cfg = config.Config()
        expected = cfg.short_chapters * cfg.short_scenes_per_chapter
        self.assertEqual(cfg.total_short_scenes, expected)
    
    def test_config_default_values(self):
        """Test that Config has expected default values."""
        cfg = config.Config()
        
        # Check main video defaults
        self.assertEqual(cfg.chapters, 6)
        self.assertEqual(cfg.scenes_per_chapter, 4)
        self.assertTrue(cfg.generate_main)
        
        # Check shorts defaults
        self.assertEqual(cfg.num_shorts, 3)
        self.assertEqual(cfg.short_chapters, 1)
        self.assertEqual(cfg.short_scenes_per_chapter, 4)
        
        # Check generation flags defaults
        self.assertTrue(cfg.generate_thumbnails)
        self.assertFalse(cfg.generate_short_thumbnails)
        self.assertFalse(cfg.generate_refinement_diffs)
    
    def test_config_modification(self):
        """Test that Config instances can be modified."""
        cfg = config.Config()
        original_chapters = cfg.chapters
        
        cfg.chapters = 5
        self.assertEqual(cfg.chapters, 5)
        self.assertNotEqual(cfg.chapters, original_chapters)
        
        # Verify total_scenes updates
        expected_total = cfg.chapters * cfg.scenes_per_chapter
        self.assertEqual(cfg.total_scenes, expected_total)
    
    def test_multiple_instances(self):
        """Test that multiple Config instances are independent."""
        cfg1 = config.Config()
        cfg2 = config.Config()
        
        cfg1.chapters = 5
        cfg2.chapters = 7
        
        self.assertEqual(cfg1.chapters, 5)
        self.assertEqual(cfg2.chapters, 7)
        self.assertNotEqual(cfg1.chapters, cfg2.chapters)


if __name__ == "__main__":
    unittest.main()
