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

    def test_trailer_scene4_no_cta(self):
        """Scene 4 must not instruct a CTA to watch the full documentary (hurts view duration)."""
        result = prompt_builders.get_trailer_structure_prompt()
        self.assertIn("Do NOT end with a CTA to watch the full documentary", result)
        self.assertNotIn("End with a soft CTA to watch the full documentary", result)


class TestThumbnailPromptWhyScene(unittest.TestCase):
    """Test cases for get_thumbnail_prompt_why_scene (thumbnail focus: emotion, simplicity, rule of thirds, text, accuracy)."""

    def test_thumbnail_prompt_rule_of_thirds(self):
        """Thumbnail prompt must require rule of thirds composition."""
        result = prompt_builders.get_thumbnail_prompt_why_scene("16:9")
        self.assertIn("rule of thirds", result.lower())

    def test_thumbnail_prompt_high_contrast(self):
        """Thumbnail prompt must address contrast (subject vs background)."""
        result = prompt_builders.get_thumbnail_prompt_why_scene("16:9")
        self.assertIn("contrast", result.lower())

    def test_thumbnail_prompt_emotion_or_peak_moment(self):
        """Thumbnail prompt must emphasize emotion / peak conflict / engaging moment."""
        result = prompt_builders.get_thumbnail_prompt_why_scene("16:9")
        result_lower = result.lower()
        self.assertTrue(
            "peak" in result_lower or "conflict" in result_lower or "engaging" in result_lower or "emotion" in result_lower,
            msg="Prompt should mention peak, conflict, engaging, or emotion"
        )

    def test_thumbnail_prompt_three_why_types(self):
        """Thumbnail prompt must reference the three WHY types (counterintuitive, secret/mystery, known but misunderstood)."""
        result = prompt_builders.get_thumbnail_prompt_why_scene("16:9")
        result_lower = result.lower()
        self.assertIn("counterintuitive", result_lower)
        self.assertTrue("secret" in result_lower or "mystery" in result_lower)
        self.assertIn("misunderstood", result_lower)

    def test_thumbnail_prompt_requires_text_overlay(self):
        """Thumbnail prompt must require short bold text for WHY type."""
        result = prompt_builders.get_thumbnail_prompt_why_scene("16:9")
        self.assertIn("text", result.lower())
        self.assertIn("bold", result.lower())

    def test_thumbnail_prompt_accuracy_whole_video(self):
        """Thumbnail prompt must require representing the whole video and being accurate."""
        result = prompt_builders.get_thumbnail_prompt_why_scene("16:9")
        result_lower = result.lower()
        self.assertTrue(
            "accurate" in result_lower or "represent" in result_lower or "whole video" in result_lower,
            msg="Prompt should require accuracy or representing the whole video"
        )

    def test_thumbnail_prompt_aspect_ratio_16_9(self):
        """16:9 prompt should not include vertical note."""
        result = prompt_builders.get_thumbnail_prompt_why_scene("16:9")
        self.assertNotIn("9:16", result)
        self.assertIn("optimized for small sizes", result.lower())

    def test_thumbnail_prompt_aspect_ratio_9_16(self):
        """9:16 prompt should include vertical and mobile note."""
        result = prompt_builders.get_thumbnail_prompt_why_scene("9:16")
        self.assertIn("9:16", result)
        self.assertIn("mobile", result.lower())


class TestBuildOutlinePrompt(unittest.TestCase):
    """Test cases for build_outline_prompt function."""
    
    def test_outline_prompt_structure(self):
        """Test outline prompt includes all required sections."""
        result = prompt_builders.build_outline_prompt("Einstein", chapters=6, target_total_scenes=24)
        self.assertIn("Einstein", result)
        self.assertIn("6", result)
        self.assertIn("24", result)
        self.assertIn("NARRATIVE STRUCTURE", result)
        self.assertIn("STORY ARC REQUIREMENTS", result)
        self.assertIn("Chapter 1", result)
        self.assertIn("1-6", result)  # chapter_num range
        self.assertIn("6 chapters", result)
        self.assertIn("person", result)  # structure example
        self.assertIn("65-year-old", result)  # audience targeting
        self.assertIn("TARGET AUDIENCE", result)
    
    def test_outline_prompt_chapters(self):
        """Test outline prompt with different chapter counts."""
        result = prompt_builders.build_outline_prompt("Darwin", chapters=5, target_total_scenes=20)
        self.assertIn("Darwin", result)
        self.assertIn("5", result)
        self.assertIn("20", result)
        # Should mention 5 chapters
        self.assertIn("5 chapters", result)
        self.assertIn("1-5", result)  # chapter_num range

    def test_outline_prompt_music_mood(self):
        """Test outline prompt includes music_mood when available_moods provided."""
        result = prompt_builders.build_outline_prompt(
            "Einstein", chapters=3, target_total_scenes=12,
            available_moods=["relaxing", "passionate", "happy"]
        )
        self.assertIn("music_mood", result)
        self.assertIn("relaxing", result)
        self.assertIn("passionate", result)
        self.assertIn("happy", result)


class TestBuildShortOutlinePrompt(unittest.TestCase):
    """Test cases for build_short_outline_prompt function."""

    def test_short_outline_includes_music_mood(self):
        """Test short outline prompt includes music_mood when available_moods provided."""
        outline = {
            "chapters": [
                {"chapter_num": 1, "title": "Hook", "year_range": "1879-1955", "summary": "Preview", "key_events": ["Birth", "Annus mirabilis"]},
                {"chapter_num": 2, "title": "Early Years", "year_range": "1879-1900", "summary": "Youth", "key_events": ["School", "Patent office"]},
            ]
        }
        result = prompt_builders.build_short_outline_prompt(
            "Einstein", outline, short_num=1, total_shorts=3,
            available_moods=["relaxing", "passionate", "happy"]
        )
        self.assertIn("music_mood", result)
        self.assertIn("relaxing", result)
        self.assertIn("passionate", result)
        self.assertIn("happy", result)
        self.assertIn("high-energy trailer", result.lower())
        self.assertIn("FULL DOCUMENTARY OUTLINE", result)


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
        self.assertIn("WHY SCENE", result)  # e.g. WHY SCENE TITLE in title field
        self.assertIn("CTR", result)  # title/thumbnail synergy for CTR
    
    def test_metadata_prompt_with_different_values(self):
        """Test metadata prompt with different person and scene count."""
        result = prompt_builders.build_metadata_prompt("Darwin", "the naturalist", 30)
        self.assertIn("Darwin", result)
        self.assertIn("the naturalist", result)
        self.assertIn("30", result)

    def test_metadata_prompt_thumbnail_why_type_and_text(self):
        """Metadata prompt must ask for thumbnail_why_type and thumbnail_text for thumbnail text overlay."""
        result = prompt_builders.build_metadata_prompt("Einstein", "the man who changed the world", 24)
        self.assertIn("thumbnail_why_type", result)
        self.assertIn("thumbnail_text", result)
        self.assertIn("counterintuitive", result.lower())
        self.assertIn("secret_mystery", result.lower())
        self.assertIn("known_but_misunderstood", result.lower())


class TestResearchContextInjection(unittest.TestCase):
    """Test that prompts include RESEARCH CONTEXT when research_context is provided."""

    def test_video_questions_prompt_includes_research(self):
        """build_video_questions_prompt includes RESEARCH CONTEXT when provided."""
        from research_utils import ResearchContext
        ctx = ResearchContext(summary="Lincoln was the 16th president.")
        result = prompt_builders.build_video_questions_prompt("Lincoln", research_context=ctx)
        self.assertIn("RESEARCH CONTEXT", result)
        self.assertIn("Lincoln was the 16th president", result)
        self.assertIn("prefer them over your general knowledge", result)

    def test_video_questions_prompt_empty_without_research(self):
        """build_video_questions_prompt has no RESEARCH CONTEXT when research_context is None."""
        result = prompt_builders.build_video_questions_prompt("Lincoln", research_context=None)
        self.assertNotIn("RESEARCH CONTEXT", result)

    def test_landmark_events_prompt_includes_research(self):
        """build_landmark_events_prompt includes RESEARCH CONTEXT when provided."""
        from research_utils import ResearchContext
        ctx = ResearchContext(summary="Einstein published four papers in 1905.")
        result = prompt_builders.build_landmark_events_prompt(
            "Einstein", num_landmarks=3, research_context=ctx
        )
        self.assertIn("RESEARCH CONTEXT", result)
        self.assertIn("Einstein published four papers in 1905", result)

    def test_outline_prompt_includes_research(self):
        """build_outline_prompt includes RESEARCH CONTEXT when provided."""
        from research_utils import ResearchContext
        ctx = ResearchContext(summary="Darwin wrote On the Origin of Species.")
        result = prompt_builders.build_outline_prompt(
            "Darwin", chapters=4, target_total_scenes=12, research_context=ctx
        )
        self.assertIn("RESEARCH CONTEXT", result)
        self.assertIn("Darwin wrote On the Origin of Species", result)

    def test_scene_outline_prompt_includes_research(self):
        """build_scene_outline_prompt includes RESEARCH CONTEXT when provided."""
        from research_utils import ResearchContext
        ctx = ResearchContext(summary="Key facts about the person.")
        chapter = {"title": "Early Years", "summary": "Childhood", "key_events": ["Birth"], "chapter_type": "chronological"}
        result = prompt_builders.build_scene_outline_prompt(
            chapter, "Person", 4, research_context=ctx
        )
        self.assertIn("RESEARCH CONTEXT", result)
        self.assertIn("Key facts about the person", result)


class TestBuildMetadataPrompt3Options(unittest.TestCase):
    """Test cases for build_metadata_prompt_3_options (3 title+thumbnail pairs)."""

    def test_3_options_prompt_structure(self):
        """3-options prompt must request thumbnail_options array and 3 pairs."""
        result = prompt_builders.build_metadata_prompt_3_options(
            "Einstein", "the man who changed the world", 24
        )
        self.assertIn("thumbnail_options", result)
        self.assertIn("3", result)
        self.assertIn("tag_line", result.lower())
        self.assertIn("global_block", result.lower())
        self.assertIn("symbiotic", result.lower())
        self.assertIn("rule of thirds", result.lower())

    def test_3_options_prompt_why_types(self):
        """3-options prompt must reference the three WHY types."""
        result = prompt_builders.build_metadata_prompt_3_options(
            "Lincoln", "the president who saved the union", 30
        )
        self.assertIn("counterintuitive", result.lower())
        self.assertIn("secret_mystery", result.lower())
        self.assertIn("known_but_misunderstood", result.lower())


if __name__ == "__main__":
    unittest.main()
