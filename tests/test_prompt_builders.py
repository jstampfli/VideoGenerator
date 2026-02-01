"""
Unit tests for prompt_builders.py functions.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import prompt_builders


class TestWhyWhatParadigmPrompt(unittest.TestCase):
    """Test cases for get_why_what_paradigm_prompt function."""
    
    def test_base_prompt(self):
        """Test base prompt without flags."""
        result = prompt_builders.get_why_what_paradigm_prompt()
        self.assertIn("WHY/WHAT PARADIGM", result)
        self.assertIn("WHY sections", result)
        self.assertIn("WHAT sections", result)
        self.assertIn("INTERLEAVING STRATEGY", result)
    
    def test_trailer_prompt(self):
        """Test trailer-specific prompt."""
        result = prompt_builders.get_why_what_paradigm_prompt(is_trailer=True)
        self.assertIn("TRAILER FORMAT", result)
        self.assertIn("MOSTLY WHY SCENES", result)
        self.assertIn("trailer", result.lower())
        self.assertNotIn("INTERLEAVING STRATEGY", result)  # Trailers have different structure
    
    def test_hook_chapter_prompt(self):
        """Test hook chapter prompt."""
        result = prompt_builders.get_why_what_paradigm_prompt(is_hook_chapter=True)
        self.assertIn("WHY/WHAT PARADIGM", result)
        self.assertIn("hook chapters", result.lower())
        self.assertIn("mostly WHY sections", result)
    
    def test_trailer_overrides_hook(self):
        """Test that trailer flag takes precedence."""
        result = prompt_builders.get_why_what_paradigm_prompt(is_trailer=True, is_hook_chapter=True)
        self.assertIn("TRAILER FORMAT", result)
        self.assertNotIn("hook chapters", result.lower())


class TestEmotionGenerationPrompt(unittest.TestCase):
    """Test cases for get_emotion_generation_prompt function."""
    
    def test_base_prompt(self):
        """Test base prompt without chapter tone."""
        result = prompt_builders.get_emotion_generation_prompt()
        self.assertIn("EMOTION GENERATION", result)
        self.assertIn("emotion", result.lower())
        self.assertIn("desperate", result.lower())
        self.assertIn("triumphant", result.lower())
        self.assertIn("contemplative", result.lower())
    
    def test_with_chapter_tone(self):
        """Test prompt with chapter emotional tone."""
        result = prompt_builders.get_emotion_generation_prompt("somber")
        self.assertIn("EMOTION GENERATION", result)
        self.assertIn("somber", result)
        self.assertIn("chapter's emotional tone", result.lower())


class TestImagePromptGuidelines(unittest.TestCase):
    """Test cases for get_image_prompt_guidelines function."""
    
    def test_main_video_prompt(self):
        """Test prompt for main video (not trailer)."""
        result = prompt_builders.get_image_prompt_guidelines(
            "Einstein", 
            birth_year=1879, 
            death_year=1955,
            aspect_ratio="16:9 cinematic",
            is_trailer=False
        )
        self.assertIn("IMAGE PROMPT STYLE", result)
        self.assertIn("Einstein", result)
        self.assertIn("1879", result)
        self.assertIn("1955", result)
        self.assertIn("16:9 cinematic", result)
        self.assertIn("Cinematic", result)
    
    def test_trailer_prompt(self):
        """Test prompt for trailer/shorts."""
        result = prompt_builders.get_image_prompt_guidelines(
            "Einstein",
            birth_year=1879,
            death_year=1955,
            aspect_ratio="9:16 vertical",
            is_trailer=True
        )
        self.assertIn("IMAGE PROMPTS", result)
        self.assertIn("9:16 vertical", result)
        self.assertIn("mobile-optimized", result.lower())
        self.assertIn("Einstein", result)
    
    def test_with_recurring_themes(self):
        """Test prompt with recurring themes."""
        result = prompt_builders.get_image_prompt_guidelines(
            "Einstein",
            recurring_themes="isolation, ambition"
        )
        self.assertIn("VISUAL THEME REINFORCEMENT", result)
        self.assertIn("isolation", result)
        self.assertIn("ambition", result)
    
    def test_without_birth_death_years(self):
        """Test prompt without birth/death years."""
        result = prompt_builders.get_image_prompt_guidelines("Einstein")
        self.assertIn("Einstein", result)
        self.assertIn("unknown", result)  # Should show unknown for missing years


class TestTrailerNarrationStyle(unittest.TestCase):
    """Test cases for get_trailer_narration_style function."""
    
    def test_trailer_narration_style(self):
        """Test trailer narration style prompt."""
        result = prompt_builders.get_trailer_narration_style()
        self.assertIn("NARRATION STYLE", result)
        self.assertIn("YouTuber", result)
        self.assertIn("third person", result.lower())
        self.assertIn("HIGH ENERGY", result)
        self.assertIn("curiosity gaps", result.lower())


class TestTrailerStructurePrompt(unittest.TestCase):
    """Test cases for get_trailer_structure_prompt function."""
    
    def test_trailer_structure(self):
        """Test trailer structure prompt (4 scenes: 1-3 WHY, 4 WHAT)."""
        result = prompt_builders.get_trailer_structure_prompt()
        self.assertIn("TRAILER STRUCTURE", result)
        self.assertIn("exactly 4 scenes", result.lower())
        self.assertIn("SCENE 1", result)
        self.assertIn("SCENE 2", result)
        self.assertIn("SCENE 3", result)
        self.assertIn("SCENE 4", result)
        self.assertIn("WHY", result)
        self.assertIn("WHAT", result)


class TestBuildOutlinePrompt(unittest.TestCase):
    """Test cases for build_outline_prompt function."""
    
    def test_outline_prompt_structure(self):
        """Test outline prompt includes all required sections."""
        result = prompt_builders.build_outline_prompt("Einstein", chapters=6, total_scenes=24)
        self.assertIn("Einstein", result)
        self.assertIn("6", result)
        self.assertIn("24", result)
        self.assertIn("NARRATIVE STRUCTURE", result)
        self.assertIn("STORY ARC REQUIREMENTS", result)
        self.assertIn("Chapter 1", result)
        self.assertIn("Chapter 6", result)
        self.assertIn("JSON", result)
    
    def test_outline_prompt_chapters(self):
        """Test outline prompt with different chapter counts."""
        result = prompt_builders.build_outline_prompt("Darwin", chapters=5, total_scenes=20)
        self.assertIn("Darwin", result)
        self.assertIn("5", result)
        self.assertIn("20", result)
        # Should mention chapter 5, not 6
        self.assertIn("Chapter 5", result)
        self.assertNotIn("Chapter 6", result)


class TestBuildMetadataPrompt(unittest.TestCase):
    """Test cases for build_metadata_prompt function."""
    
    def test_metadata_prompt_structure(self):
        """Test metadata prompt includes all required fields."""
        result = prompt_builders.build_metadata_prompt("Einstein", "the man who changed the world", 24)
        self.assertIn("Einstein", result)
        self.assertIn("the man who changed the world", result)
        self.assertIn("24", result)
        self.assertIn("title", result.lower())
        self.assertIn("tag_line", result.lower())
        self.assertIn("thumbnail_description", result.lower())
        self.assertIn("global_block", result.lower())
        self.assertIn("WHY SCENE THUMBNAIL", result)
        self.assertIn("MAXIMIZE CTR", result)
    
    def test_metadata_prompt_with_different_values(self):
        """Test metadata prompt with different person and scene count."""
        result = prompt_builders.build_metadata_prompt("Darwin", "the naturalist", 30)
        self.assertIn("Darwin", result)
        self.assertIn("the naturalist", result)
        self.assertIn("30", result)


if __name__ == "__main__":
    unittest.main()
