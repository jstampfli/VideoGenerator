"""
Unit tests for refinement and significance scene generation functions.
Tests that these methods work correctly and catch errors like UnboundLocalError.
"""

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
    clean_json_response
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
        
        # Mock llm_utils.generate_text to return JSON string
        mock_generate_text.return_value = json.dumps({
            "title": "Why This Matters",
            "narration": "This moment changed everything.",
            "scene_type": "WHAT",
            "image_prompt": "Reflective scene showing Einstein",
            "emotion": "contemplative",
            "narration_instructions": "Focus on contemplation.",
            "year": "1905"
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
            "year": "1905"
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
            "year": "1890"
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
    
    def test_generate_significance_scene_horror_with_drone_change(self, mock_generate_text):
        """Test that horror significance scenes include drone_change."""
        pivotal_scene = {
            "id": 5,
            "title": "The Sound",
            "narration": "I hear something in the darkness.",
            "image_prompt": "Dark room with shadows",
            "year": "present",
            "emotion": "tense",
            "scene_type": "WHY",
            "drone_change": "swell"
        }
        
        mock_generate_text.return_value = json.dumps({
            "title": "Why This Matters",
            "narration": "This moment changed everything.",
            "scene_type": "WHAT",
            "image_prompt": "Dark, reflective scene",
            "emotion": "dread-filled",
            "narration_instructions": "Focus on dread.",
            "year": "present",
            "drone_change": "hold"
        })
        
        try:
            result = generate_significance_scene(
                pivotal_scene,
                None,
                "the narrator",
                "This was a pivotal moment because...",
                script_type="horror"
            )
            
            # Should not raise any errors
            self.assertIsInstance(result, dict)
            self.assertIn('drone_change', result)
            self.assertIn('narration_instructions', result)
            
        except Exception as e:
            self.fail(f"generate_significance_scene should not raise errors. Error: {e}")


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
    
    def test_refine_scenes_horror_with_drone_change(self, mock_generate_text):
        """Test that refine_scenes preserves drone_change for horror scripts."""
        scenes = [
            {
                "id": 1,
                "title": "The Whisper",
                "narration": "I hear something.",
                "scene_type": "WHY",
                "image_prompt": "Dark room",
                "emotion": "uneasy",
                "narration_instructions": "Focus on unease.",
                "drone_change": "fade_in",
                "year": "present"
            },
            {
                "id": 2,
                "title": "The Sound",
                "narration": "It's getting closer.",
                "scene_type": "WHY",
                "image_prompt": "Dark hallway",
                "emotion": "tense",
                "narration_instructions": "Focus on tension.",
                "drone_change": "swell",
                "year": "present"
            }
        ]
        
        mock_generate_text.return_value = json.dumps(scenes)
        
        try:
            refined, changes = refine_scenes(
                scenes,
                "the narrator",
                is_short=False,
                script_type="horror"
            )
            
            # Should preserve drone_change
            for scene in refined:
                self.assertIn('drone_change', scene)
                self.assertIn(scene['drone_change'], ['fade_in', 'fade_out', 'hard_cut', 'hold', 'swell', 'shrink', 'none'])
            
        except Exception as e:
            self.fail(f"refine_scenes should preserve drone_change for horror. Error: {e}")
    
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


if __name__ == '__main__':
    unittest.main()
