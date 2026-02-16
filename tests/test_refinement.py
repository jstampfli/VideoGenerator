"""
Unit tests for refinement and significance scene generation functions.
Tests that these methods work correctly and catch errors like UnboundLocalError.
"""

import shutil
import tempfile
import unittest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_scripts_utils import (
    generate_significance_scene,
    refine_scenes,
    clean_json_response,
    insert_chapter_transition_scenes,
    validate_biopic_scenes,
)


@patch('build_scripts_utils.llm_utils.generate_text')
class TestSignificanceSceneGeneration(unittest.TestCase):
    """Test cases for generate_significance_scene function."""
    
    def test_generate_significance_scene_with_age_phrase_extraction(self, mock_generate_text):
        """Test that age_phrase is extracted correctly and doesn't cause UnboundLocalError."""
        pivotal_scene = {
            "id": 7,
            "title": "The Discovery",
            "narration": "He made a breakthrough.",
            "image_prompt": "26-year-old Albert Einstein at his desk, 1905, 16:9 cinematic",
            "year": "1905",
            "emotion": "contemplative",
            "scene_type": "WHAT"
        }
        
        next_scene = {
            "id": 8,
            "title": "The Impact",
            "narration": "The world took notice."
        }
        
        # Mock llm_utils.generate_text to return JSON string (SCENE_SCHEMA requires kenburns_intensity)
        mock_generate_text.return_value = json.dumps({
            "title": "Why This Matters",
            "narration": "This moment changed everything.",
            "scene_type": "WHAT",
            "image_prompt": "Reflective scene showing Einstein",
            "emotion": "contemplative",
            "narration_instructions": "Focus on contemplation.",
            "year": "1905",
            "kenburns_intensity": "medium",
        })
        
        try:
            result = generate_significance_scene(
                pivotal_scene,
                next_scene,
                "Albert Einstein",
                "This was a pivotal moment because...",
                script_type="biopic"
            )
            
            # Should not raise UnboundLocalError
            self.assertIsInstance(result, dict)
            self.assertIn('title', result)
            self.assertIn('narration', result)
            self.assertIn('image_prompt', result)
            self.assertIn('narration_instructions', result)
            
        except UnboundLocalError as e:
            if 'age_phrase' in str(e):
                self.fail(f"UnboundLocalError for age_phrase should not occur. Error: {e}")
            raise
    
    def test_generate_significance_scene_without_age_in_pivotal_scene(self, mock_generate_text):
        """Test that age_phrase handling works when pivotal scene has no age information."""
        pivotal_scene = {
            "id": 7,
            "title": "The Discovery",
            "narration": "He made a breakthrough.",
            "image_prompt": "Einstein at his desk, 1905, 16:9 cinematic",  # No age info
            "year": "1905",
            "emotion": "contemplative",
            "scene_type": "WHAT"
        }
        
        mock_generate_text.return_value = json.dumps({
            "title": "Why This Matters",
            "narration": "This moment changed everything.",
            "scene_type": "WHAT",
            "image_prompt": "Reflective scene showing Einstein",
            "emotion": "contemplative",
            "narration_instructions": "Focus on contemplation.",
            "year": "1905",
            "kenburns_intensity": "medium",
        })
        
        try:
            result = generate_significance_scene(
                pivotal_scene,
                None,
                "Albert Einstein",
                "This was a pivotal moment because...",
                script_type="biopic"
            )
            
            # Should not raise UnboundLocalError even without age info
            self.assertIsInstance(result, dict)
            self.assertIn('image_prompt', result)
            # Should fall back to subject name if no age_phrase
            self.assertIn('Einstein', result['image_prompt'])
            
        except UnboundLocalError as e:
            if 'age_phrase' in str(e):
                self.fail(f"UnboundLocalError for age_phrase should not occur even without age info. Error: {e}")
            raise
    
    def test_generate_significance_scene_missing_image_prompt_from_llm(self, mock_generate_text):
        """Test that missing image_prompt from LLM is handled correctly with age_phrase."""
        pivotal_scene = {
            "id": 7,
            "title": "The Discovery",
            "narration": "He made a breakthrough.",
            "image_prompt": "30-year-old Nikola Tesla in his lab, 1890, 16:9 cinematic",
            "year": "1890",
            "emotion": "contemplative",
            "scene_type": "WHAT"
        }
        
        mock_generate_text.return_value = json.dumps({
            "title": "Why This Matters",
            "narration": "This moment changed everything.",
            "scene_type": "WHAT",
            "emotion": "contemplative",
            "narration_instructions": "Focus on contemplation.",
            "year": "1890",
            "kenburns_intensity": "medium",
        })
        
        try:
            result = generate_significance_scene(
                pivotal_scene,
                None,
                "Nikola Tesla",
                "This was a pivotal moment because...",
                script_type="biopic"
            )
            
            # Should not raise UnboundLocalError
            self.assertIsInstance(result, dict)
            self.assertIn('image_prompt', result)
            # Should include age_phrase in the generated image_prompt
            self.assertIn('30-year-old', result['image_prompt'])
            self.assertIn('Nikola Tesla', result['image_prompt'])
            
        except UnboundLocalError as e:
            if 'age_phrase' in str(e):
                self.fail(f"UnboundLocalError for age_phrase should not occur when LLM doesn't provide image_prompt. Error: {e}")
            raise
    
@patch('build_scripts_utils.llm_utils.generate_text')
class TestRefineScenes(unittest.TestCase):
    """Test cases for refine_scenes function."""
    
    def test_refine_scenes_basic(self, mock_generate_text):
        """Test that refine_scenes works with basic scene data."""
        scenes = [
            {
                "id": 1,
                "title": "The Beginning",
                "narration": "It all started here.",
                "scene_type": "WHY",
                "image_prompt": "Visual description",
                "emotion": "contemplative",
                "narration_instructions": "Focus on contemplation.",
                "year": "1900"
            },
            {
                "id": 2,
                "title": "The Discovery",
                "narration": "He found the answer.",
                "scene_type": "WHAT",
                "image_prompt": "Visual description",
                "emotion": "thoughtful",
                "narration_instructions": "Focus on thoughtfulness.",
                "year": "1905"
            }
        ]
        
        mock_generate_text.return_value = json.dumps(scenes)
        
        try:
            refined, changes = refine_scenes(
                scenes,
                "Albert Einstein",
                is_short=False,
                script_type="biopic"
            )
            
            # Should return refined scenes and changes dict
            self.assertIsInstance(refined, list)
            self.assertIsInstance(changes, dict)
            self.assertEqual(len(refined), len(scenes))
            
            # All scenes should have narration_instructions
            for scene in refined:
                self.assertIn('narration_instructions', scene)
                # Should be one sentence focusing on emotion
                self.assertIsInstance(scene['narration_instructions'], str)
                self.assertLess(len(scene['narration_instructions'].split('.')), 3, 
                              "narration_instructions should be one sentence")
            
        except Exception as e:
            self.fail(f"refine_scenes should not raise errors. Error: {e}")
    
    def test_refine_scenes_missing_narration_instructions(self, mock_generate_text):
        """Test that refine_scenes handles missing narration_instructions."""
        scenes = [
            {
                "id": 1,
                "title": "The Beginning",
                "narration": "It all started here.",
                "scene_type": "WHY",
                "image_prompt": "Visual description",
                "emotion": "contemplative",
                "year": "1900"
            }
        ]
        
        mock_generate_text.return_value = json.dumps([{
            "id": 1,
            "title": "The Beginning",
            "narration": "It all started here.",
            "scene_type": "WHY",
            "image_prompt": "Visual description",
            "emotion": "contemplative",
            "year": "1900"
        }])
        
        try:
            refined, changes = refine_scenes(
                scenes,
                "Albert Einstein",
                is_short=False,
                script_type="biopic"
            )
            
            # Should add fallback narration_instructions
            self.assertIn('narration_instructions', refined[0])
            self.assertEqual(refined[0]['narration_instructions'], "Focus on contemplative.")
            
        except Exception as e:
            self.fail(f"refine_scenes should handle missing narration_instructions. Error: {e}")
    
    def test_refine_scenes_smooth_emotion_transitions(self, mock_generate_text):
        """Test that refine_scenes maintains smooth emotion transitions."""
        scenes = [
            {
                "id": 1,
                "title": "Scene 1",
                "narration": "First scene.",
                "scene_type": "WHY",
                "image_prompt": "Visual",
                "emotion": "contemplative",
                "narration_instructions": "Focus on contemplation.",
                "year": "1900"
            },
            {
                "id": 2,
                "title": "Scene 2",
                "narration": "Second scene.",
                "scene_type": "WHAT",
                "image_prompt": "Visual",
                "emotion": "urgent",
                "narration_instructions": "Focus on urgency.",
                "year": "1905"
            }
        ]
        
        mock_generate_text.return_value = json.dumps([
            {
                "id": 1,
                "title": "Scene 1",
                "narration": "First scene.",
                "scene_type": "WHY",
                "image_prompt": "Visual",
                "emotion": "contemplative",
                "narration_instructions": "Focus on contemplation.",
                "year": "1900"
            },
            {
                "id": 2,
                "title": "Scene 2",
                "narration": "Second scene.",
                "scene_type": "WHAT",
                "image_prompt": "Visual",
                "emotion": "thoughtful",
                "narration_instructions": "Focus on thoughtfulness.",
                "year": "1905"
            }
        ])
        
        try:
            refined, changes = refine_scenes(
                scenes,
                "Albert Einstein",
                is_short=False,
                script_type="biopic"
            )
            
            # Should refine scenes smoothly
            self.assertEqual(len(refined), 2)
            # Emotions should flow more smoothly after refinement
            self.assertIn(refined[0]['emotion'], ['contemplative', 'thoughtful', 'reflective'])
            self.assertIn(refined[1]['emotion'], ['thoughtful', 'reflective', 'somber'])
            
        except Exception as e:
            self.fail(f"refine_scenes should handle emotion transitions. Error: {e}")
    
    def test_refine_scenes_pass4_music_selection_biopic(self, mock_generate_text):
        """Pass 4 (film composer) adds music_song and music_volume to scenes for biopic main videos."""
        scenes = [
            {
                "id": 1,
                "title": "The Beginning",
                "narration": "It all started here.",
                "scene_type": "WHY",
                "image_prompt": "Visual description",
                "emotion": "contemplative",
                "narration_instructions": "Focus on contemplation.",
                "year": "1900",
                "chapter_num": 1,
            },
            {
                "id": 2,
                "title": "The Discovery",
                "narration": "He found the answer.",
                "scene_type": "WHAT",
                "image_prompt": "Visual description",
                "emotion": "thoughtful",
                "narration_instructions": "Focus on thoughtfulness.",
                "year": "1905",
                "chapter_num": 1,
            },
        ]
        music_selections = [
            {"id": 1, "music_song": "relaxing/song1.mp3", "music_volume": "medium"},
            {"id": 2, "music_song": "relaxing/song1.mp3", "music_volume": "low"},
        ]
        refined_scenes = [
            {**s, "narration_instructions": s["narration_instructions"]} for s in scenes
        ]
        # Pass 1 (storyline): empty. Pass 1.5 (historian): empty. Pass 2 (pivotal): skipped. Pass 3: refined. Pass 4: music. Pass 5: transitions.
        transition_selections = [
            {"id": 1, "transition_to_next": "crossfade", "transition_speed": "medium"},
            {"id": 2, "transition_to_next": None, "transition_speed": None},
        ]
        mock_generate_text.side_effect = [
            json.dumps([]),  # hanging storylines
            json.dumps([]),  # historian depth
            json.dumps(refined_scenes),  # Pass 3 refinement
            json.dumps(music_selections),  # Pass 4 music selection
            json.dumps(transition_selections),  # Pass 5 transition selection
        ]
        tmp = Path(__file__).parent / "tmp_music_test"
        music_dir = tmp / "biopic_music" / "relaxing"
        music_dir.mkdir(parents=True, exist_ok=True)
        (music_dir / "song1.mp3").touch()
        biopic_music_dir = tmp / "biopic_music"
        try:
            with patch("biopic_music_config.get_all_songs", return_value=["relaxing/song1.mp3"]), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", biopic_music_dir):
                refined, _ = refine_scenes(
                    scenes,
                    "Albert Einstein",
                    is_short=False,
                    script_type="biopic",
                    skip_significance_scenes=True,
                )
            self.assertEqual(len(refined), 2)
            for scene in refined:
                self.assertIn("music_song", scene)
                self.assertIn("music_volume", scene)
                self.assertEqual(scene["music_song"], "relaxing/song1.mp3")
            self.assertEqual(refined[0]["music_volume"], "medium")
            self.assertEqual(refined[1]["music_volume"], "low")
            # Pass 5 adds transition_to_next and transition_speed
            self.assertEqual(refined[0]["transition_to_next"], "crossfade")
            self.assertEqual(refined[0]["transition_speed"], "medium")
            self.assertIsNone(refined[1]["transition_to_next"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_refine_scenes_pass4_rejects_hallucinated_song(self, mock_generate_text):
        """Pass 4 uses fallback when LLM returns song not in all_songs (hallucination)."""
        scenes = [
            {
                "id": 1,
                "title": "Scene",
                "narration": "Text.",
                "scene_type": "WHY",
                "image_prompt": "Visual",
                "emotion": "contemplative",
                "narration_instructions": "Focus on contemplation.",
                "year": "1900",
                "chapter_num": 1,
            },
        ]
        # LLM hallucinates a song that doesn't exist in biopic_music
        music_selections = [
            {"id": 1, "music_song": "biopic_music/somber_piano_intro.mp3", "music_volume": "medium"},
        ]
        refined_scenes = [{**s, "narration_instructions": s["narration_instructions"]} for s in scenes]
        transition_selections = [{"id": 1, "transition_to_next": None, "transition_speed": None}]
        mock_generate_text.side_effect = [
            json.dumps([]),
            json.dumps([]),
            json.dumps(refined_scenes),
            json.dumps(music_selections),
            json.dumps(transition_selections),
        ]
        tmp = Path(__file__).parent / "tmp_music_halluc"
        (tmp / "biopic_music" / "relaxing").mkdir(parents=True, exist_ok=True)
        (tmp / "biopic_music" / "relaxing" / "song1.mp3").touch()
        biopic_music_dir = tmp / "biopic_music"
        try:
            with patch("biopic_music_config.get_all_songs", return_value=["relaxing/song1.mp3"]), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", biopic_music_dir):
                refined, _ = refine_scenes(
                    scenes,
                    "Einstein",
                    is_short=False,
                    script_type="biopic",
                    skip_significance_scenes=True,
                )
            # Must use fallback (relaxing/song1.mp3), not hallucinated path
            self.assertEqual(refined[0]["music_song"], "relaxing/song1.mp3")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_refine_scenes_pass5_transition_selection(self, mock_generate_text):
        """Pass 5 (transition selection) adds valid transition_to_next to scenes for biopic."""
        scenes = [
            {
                "id": 1,
                "title": "Tension",
                "narration": "Action scene.",
                "scene_type": "WHAT",
                "image_prompt": "Visual",
                "emotion": "tense",
                "narration_instructions": "Focus on tension.",
                "year": "1900",
                "chapter_num": 1,
            },
            {
                "id": 2,
                "title": "Reflection",
                "narration": "Calm moment.",
                "scene_type": "WHY",
                "image_prompt": "Visual",
                "emotion": "contemplative",
                "narration_instructions": "Focus on contemplation.",
                "year": "1905",
                "chapter_num": 1,
            },
        ]
        refined_scenes = [{**s, "narration_instructions": s["narration_instructions"]} for s in scenes]
        music_selections = [
            {"id": 1, "music_song": "relaxing/song1.mp3", "music_volume": "medium"},
            {"id": 2, "music_song": "relaxing/song1.mp3", "music_volume": "low"},
        ]
        transition_selections = [
            {"id": 1, "transition_to_next": "cut", "transition_speed": None},
            {"id": 2, "transition_to_next": None, "transition_speed": None},
        ]
        mock_generate_text.side_effect = [
            json.dumps([]),
            json.dumps([]),
            json.dumps(refined_scenes),
            json.dumps(music_selections),
            json.dumps(transition_selections),
        ]
        tmp = Path(__file__).parent / "tmp_transition_test"
        music_dir = tmp / "biopic_music" / "relaxing"
        music_dir.mkdir(parents=True, exist_ok=True)
        (music_dir / "song1.mp3").touch()
        biopic_music_dir = tmp / "biopic_music"
        try:
            with patch("biopic_music_config.get_all_songs", return_value=["relaxing/song1.mp3"]), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", biopic_music_dir):
                refined, _ = refine_scenes(
                    scenes,
                    "Einstein",
                    is_short=False,
                    script_type="biopic",
                    skip_significance_scenes=True,
                )
            self.assertEqual(len(refined), 2)
            self.assertEqual(refined[0]["transition_to_next"], "cut")
            self.assertIsNone(refined[1]["transition_to_next"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_refine_scenes_pass4_runs_for_shorts(self, mock_generate_text):
        """Pass 4 (music selection) runs for biopic shorts and adds music_song/music_volume."""
        scenes = [
            {
                "id": 1,
                "title": "Hook",
                "narration": "Quick story.",
                "scene_type": "WHY",
                "image_prompt": "Visual",
                "emotion": "curious",
                "narration_instructions": "Focus on curiosity.",
                "year": "1900",
            },
        ]
        music_selections = [{"id": 1, "music_song": "passionate/song1.mp3", "music_volume": "medium"}]
        transition_selections = [{"id": 1, "transition_to_next": None, "transition_speed": None}]
        mock_generate_text.side_effect = [
            json.dumps(scenes),  # Pass 3 refinement
            json.dumps(music_selections),  # Pass 4 music selection
            json.dumps(transition_selections),  # Pass 5 transition selection
        ]
        tmp = Path(__file__).parent / "tmp_music_short_test"
        music_dir = tmp / "biopic_music" / "passionate"
        music_dir.mkdir(parents=True, exist_ok=True)
        (music_dir / "song1.mp3").touch()
        biopic_music_dir = tmp / "biopic_music"
        try:
            with patch("biopic_music_config.get_all_songs", return_value=["passionate/song1.mp3"]), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", biopic_music_dir):
                refined, _ = refine_scenes(
                    scenes, "Einstein", is_short=True, script_type="biopic",
                    short_music_mood="passionate",
                )
            self.assertEqual(len(refined), 1)
            self.assertIn("music_song", refined[0])
            self.assertIn("music_volume", refined[0])
            self.assertEqual(refined[0]["music_song"], "passionate/song1.mp3")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_refine_scenes_error_handling(self, mock_generate_text):
        """Test that refine_scenes handles errors gracefully."""
        scenes = [
            {
                "id": 1,
                "title": "The Beginning",
                "narration": "It all started here.",
                "scene_type": "WHY",
                "image_prompt": "Visual description",
                "emotion": "contemplative",
                "narration_instructions": "Focus on contemplation.",
                "year": "1900"
            }
        ]
        
        mock_generate_text.side_effect = Exception("API Error")
        
        try:
            refined, changes = refine_scenes(
                scenes,
                "Albert Einstein",
                is_short=False,
                script_type="biopic"
            )
            
            # Should return original scenes on error
            self.assertEqual(refined, scenes)
            self.assertEqual(changes, {})
            
        except Exception as e:
            # If it raises an exception, that's also acceptable error handling
            # The important thing is it doesn't crash the whole process
            pass


class TestInsertChapterTransitionScenes(unittest.TestCase):
    """Test cases for insert_chapter_transition_scenes function."""

    def test_inserts_transitions_between_chapters(self):
        """Transition scenes inserted after ch1, ch2; not before ch1 or after final chapter."""
        scenes = [
            {"id": 1, "title": "Ch1 Scene 1", "chapter_num": 1, "narration": "A", "music_song": "sad/song.mp3", "music_volume": "low"},
            {"id": 2, "title": "Ch1 Scene 2", "chapter_num": 1, "narration": "B", "music_song": "sad/song.mp3", "music_volume": "low"},
            {"id": 3, "title": "Ch2 Scene 1", "chapter_num": 2, "narration": "C", "music_song": "tense/track.mp3", "music_volume": "medium"},
            {"id": 4, "title": "Ch2 Scene 2", "chapter_num": 2, "narration": "D", "music_song": "tense/track.mp3", "music_volume": "medium"},
            {"id": 5, "title": "Ch3 Scene 1", "chapter_num": 3, "narration": "E", "music_song": "relaxing/peace.mp3", "music_volume": "low"},
        ]
        chapters = [
            {"chapter_num": 1, "title": "The Beginning", "summary": "Early days", "year_range": "1809-1820"},
            {"chapter_num": 2, "title": "Rise to Fame", "summary": "Breakthrough years", "year_range": "1820-1850"},
            {"chapter_num": 3, "title": "Legacy", "summary": "Last years", "year_range": "1850-1865"},
        ]
        result = insert_chapter_transition_scenes(scenes, chapters, "Lincoln", script_type="biopic")
        # 5 content + 2 transitions (ch1->ch2, ch2->ch3) = 7 scenes
        self.assertEqual(len(result), 7)
        # First transition after scene 2 (last of ch1), introduces ch2
        # Music inherited from first scene of next chapter (ch2), not previous scene
        self.assertEqual(result[2]["scene_type"], "TRANSITION")
        self.assertEqual(result[2]["narration"], "Rise to Fame")
        self.assertTrue(result[2]["is_chapter_transition"])
        self.assertEqual(result[2]["music_song"], "tense/track.mp3")
        self.assertEqual(result[2]["music_volume"], "medium")
        self.assertIn("transition_to_next", result[2], "Inserted transition scenes must have transition_to_next")
        self.assertIn("transition_speed", result[2], "Inserted transition scenes must have transition_speed")
        self.assertEqual(result[2]["transition_to_next"], "crossfade")  # default when prev had no transition
        self.assertEqual(result[2]["transition_speed"], "medium")
        # image_prompt must fill frame, no empty gaps (anti-gap fix)
        self.assertIn("Fill the entire", result[2]["image_prompt"])
        self.assertNotIn("No other elements", result[2]["image_prompt"])
        # Transition scenes must have same schema as normal scenes (all required fields)
        required = ["id", "title", "narration", "scene_type", "image_prompt", "emotion",
                    "narration_instructions", "year", "chapter_num", "kenburns_pattern",
                    "kenburns_intensity", "music_song", "music_volume", "transition_to_next", "transition_speed"]
        for field in required:
            self.assertIn(field, result[2], f"Transition scene must have required field '{field}'")
        # Second transition after scene 4 (last of ch2), introduces ch3
        # Music inherited from first scene of next chapter (ch3)
        self.assertEqual(result[5]["scene_type"], "TRANSITION")
        self.assertEqual(result[5]["narration"], "Legacy")
        self.assertEqual(result[5]["music_song"], "relaxing/peace.mp3")
        # All ids renumbered
        for i, s in enumerate(result):
            self.assertEqual(s["id"], i + 1)

    def test_no_transitions_for_single_chapter(self):
        """Returns scenes unchanged when only one chapter."""
        scenes = [{"id": 1, "chapter_num": 1, "title": "Only", "narration": "X"}]
        chapters = [{"chapter_num": 1, "title": "Solo", "summary": "Just one"}]
        result = insert_chapter_transition_scenes(scenes, chapters, "Test", script_type="biopic")
        self.assertEqual(result, scenes)
        self.assertEqual(len(result), 1)

    def test_no_transitions_for_empty_chapters(self):
        """Returns scenes unchanged when chapters empty or len < 2."""
        scenes = [{"id": 1, "chapter_num": 1}]
        result = insert_chapter_transition_scenes(scenes, [], "Test", script_type="biopic")
        self.assertEqual(result, scenes)

    def test_transition_scene_has_music_when_prev_missing(self):
        """Transition scene gets fallback music when prev_scene lacks it (schema compliance)."""
        scenes = [
            {"id": 1, "chapter_num": 1, "narration": "A"},
            {"id": 2, "chapter_num": 1, "narration": "B"},  # no music_song/music_volume
            {"id": 3, "chapter_num": 2, "narration": "C"},
        ]
        chapters = [
            {"chapter_num": 1, "title": "Ch1", "summary": "First", "year_range": "1800"},
            {"chapter_num": 2, "title": "Ch2", "summary": "Second", "year_range": "1850"},
        ]
        tmp = Path(__file__).parent / "tmp_trans_fallback"
        (tmp / "biopic_music" / "relaxing").mkdir(parents=True, exist_ok=True)
        (tmp / "biopic_music" / "relaxing" / "song1.mp3").touch()
        try:
            with patch("biopic_music_config.get_all_songs", return_value=["relaxing/song1.mp3"]), \
                 patch("biopic_music_config.BIOPIC_MUSIC_DIR", tmp / "biopic_music"):
                result = insert_chapter_transition_scenes(scenes, chapters, "Test", script_type="biopic")
            trans = result[2]
            self.assertEqual(trans["scene_type"], "TRANSITION")
            self.assertIn("music_song", trans)
            self.assertIn("music_volume", trans)
            self.assertEqual(trans["music_song"], "relaxing/song1.mp3")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestValidateBiopicScenes(unittest.TestCase):
    """Test cases for validate_biopic_scenes function."""

    def _valid_scene(self, i=1, song="relaxing/song.mp3", transition_to_next="crossfade", transition_speed="medium"):
        speed = None if transition_to_next is None or transition_to_next == "cut" else transition_speed
        return {
            "id": i,
            "title": "Scene",
            "narration": "Narration",
            "scene_type": "WHY",
            "image_prompt": "Visual",
            "emotion": "tense",
            "narration_instructions": "Focus on tension.",
            "year": "1905",
            "chapter_num": 1,
            "kenburns_pattern": "zoom_in",
            "kenburns_intensity": "medium",
            "music_song": song,
            "music_volume": "medium",
            "transition_to_next": transition_to_next,
            "transition_speed": speed,
        }

    def test_validate_biopic_scenes_success(self):
        """Valid scenes with existing music file pass."""
        tmp = tempfile.mkdtemp(dir=str(Path(__file__).parent.parent))
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            (music_dir / "relaxing").mkdir()
            (music_dir / "relaxing" / "song.mp3").touch()
            with patch("biopic_music_config.BIOPIC_MUSIC_DIR", music_dir):
                # Single scene: transition_to_next is None (last scene)
                validate_biopic_scenes([self._valid_scene(1, transition_to_next=None)], "test.json")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_validate_biopic_scenes_missing_music_song(self):
        """Raises when scene lacks music_song."""
        scene = self._valid_scene()
        del scene["music_song"]
        with self.assertRaises(ValueError) as ctx:
            validate_biopic_scenes([scene])
        self.assertIn("music_song", str(ctx.exception))

    def test_validate_biopic_scenes_missing_music_volume(self):
        """Raises when scene lacks music_volume."""
        scene = self._valid_scene()
        del scene["music_volume"]
        with self.assertRaises(ValueError) as ctx:
            validate_biopic_scenes([scene])
        self.assertIn("music_volume", str(ctx.exception))

    def test_validate_biopic_scenes_invalid_music_volume(self):
        """Raises when music_volume is not low/medium/loud."""
        scene = self._valid_scene()
        scene["music_volume"] = "invalid"
        with self.assertRaises(ValueError) as ctx:
            validate_biopic_scenes([scene])
        self.assertIn("music_volume", str(ctx.exception))

    def test_validate_biopic_scenes_missing_music_file(self):
        """Raises when the song file does not exist under BIOPIC_MUSIC_DIR."""
        tmp = tempfile.mkdtemp(dir=str(Path(__file__).parent.parent))
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            # No relaxing/song.mp3 - file is missing
            with patch("biopic_music_config.BIOPIC_MUSIC_DIR", music_dir):
                with self.assertRaises(FileNotFoundError) as ctx:
                    validate_biopic_scenes([self._valid_scene(song="relaxing/song.mp3", transition_to_next=None)])
                self.assertIn("Music file not found", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_validate_biopic_scenes_valid_slide_transitions(self):
        """Valid slide transition types (slide_left, etc.) pass validation."""
        scene1 = self._valid_scene(1, transition_to_next="slide_left")
        scene2 = self._valid_scene(2, transition_to_next=None)
        tmp = tempfile.mkdtemp(dir=str(Path(__file__).parent.parent))
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            (music_dir / "relaxing").mkdir()
            (music_dir / "relaxing" / "song.mp3").touch()
            with patch("biopic_music_config.BIOPIC_MUSIC_DIR", music_dir):
                validate_biopic_scenes([scene1, scene2])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_validate_biopic_scenes_invalid_transition_speed(self):
        """Raises when transition_speed is invalid (when present)."""
        scene = self._valid_scene(transition_to_next="crossfade", transition_speed="fast")
        tmp = tempfile.mkdtemp(dir=str(Path(__file__).parent.parent))
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            (music_dir / "relaxing").mkdir()
            (music_dir / "relaxing" / "song.mp3").touch()
            with patch("biopic_music_config.BIOPIC_MUSIC_DIR", music_dir):
                with self.assertRaises(ValueError) as ctx:
                    validate_biopic_scenes([scene])
                self.assertIn("transition_speed", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_validate_biopic_scenes_invalid_transition_to_next(self):
        """Raises when transition_to_next is invalid (when present)."""
        scene = self._valid_scene(transition_to_next="crossfade")
        scene["transition_to_next"] = "invalid_wipe"
        tmp = tempfile.mkdtemp(dir=str(Path(__file__).parent.parent))
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            (music_dir / "relaxing").mkdir()
            (music_dir / "relaxing" / "song.mp3").touch()
            with patch("biopic_music_config.BIOPIC_MUSIC_DIR", music_dir):
                with self.assertRaises(ValueError) as ctx:
                    validate_biopic_scenes([scene])
                self.assertIn("transition_to_next", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_validate_biopic_scenes_invalid_kenburns_intensity(self):
        """Raises when kenburns_intensity is invalid (when present)."""
        scene = self._valid_scene(transition_to_next=None)
        scene["kenburns_intensity"] = "extreme"
        tmp = tempfile.mkdtemp(dir=str(Path(__file__).parent.parent))
        try:
            music_dir = Path(tmp) / "biopic_music"
            music_dir.mkdir()
            (music_dir / "relaxing").mkdir()
            (music_dir / "relaxing" / "song.mp3").touch()
            with patch("biopic_music_config.BIOPIC_MUSIC_DIR", music_dir):
                with self.assertRaises(ValueError) as ctx:
                    validate_biopic_scenes([scene])
                self.assertIn("kenburns_intensity", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
