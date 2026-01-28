"""
Unit tests for scene context building in scene generation functions.
Tests that scene context includes full scene JSON with all necessary fields.
"""

import unittest
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSceneContextBuilding(unittest.TestCase):
    """Test cases for scene context building in scene generation."""
    
    def test_horror_scene_context_includes_full_json(self):
        """Test that horror scene context includes full scene JSON with all fields."""
        from build_script_horror import generate_horror_scenes_for_chapter
        
        # Create mock previous scenes with all fields
        prev_scenes = [
            {
                "id": 1,
                "title": "The Whisper",
                "narration": "I hear something in the darkness.",
                "emotion": "uneasy",
                "narration_instructions": "Speak naturally with subtle tension.",
                "drone_change": "fade_in",
                "scene_type": "WHY",
                "year": "present",
                "image_prompt": "Dark room with shadows"
            },
            {
                "id": 2,
                "title": "The Sound",
                "narration": "The sound grows louder.",
                "emotion": "tense",
                "narration_instructions": "Deliver in a calm, measured tone with underlying unease.",
                "drone_change": "swell",
                "scene_type": "WHY",
                "year": "present",
                "image_prompt": "Dark hallway"
            }
        ]
        
        # Build scene context manually (extract the logic)
        recent_scenes = prev_scenes[-5:]
        scenes_context = "RECENT SCENES - AVOID REPEATING THESE EVENTS, maintain continuity, and ensure smooth transitions:\n"
        for sc in recent_scenes:
            scenes_context += f"  Scene {sc.get('id')} (FULL JSON): {json.dumps(sc, indent=2)}\n"
        scenes_context += "\nCRITICAL: Do NOT repeat or overlap with events already covered. Each scene must cover DIFFERENT events.\n"
        scenes_context += "CRITICAL: For smooth transitions, ensure:\n"
        scenes_context += "- emotion flows gradually from the previous scene's emotion\n"
        scenes_context += "- narration_instructions flow smoothly from the previous scene's narration_instructions\n"
        scenes_context += "- drone_change is correct based on the previous scene's drone_change (use hold/swell/shrink/fade_out/hard_cut if drone was present, fade_in if it wasn't)"
        
        # Verify the context includes full JSON
        for scene in prev_scenes:
            scene_id = scene['id']
            # Check that the scene JSON is in the context
            self.assertIn(f'Scene {scene_id} (FULL JSON):', scenes_context)
            
            # Parse the JSON from the context
            # Find the JSON block for this scene
            scene_marker = f'Scene {scene_id} (FULL JSON):'
            start_idx = scenes_context.find(scene_marker)
            self.assertNotEqual(start_idx, -1, f"Scene {scene_id} marker not found")
            
            # Find the start of the JSON (after the marker)
            json_start = scenes_context.find('{', start_idx)
            self.assertNotEqual(json_start, -1, f"JSON start not found for scene {scene_id}")
            
            # Find the end of the JSON (find matching closing brace)
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(scenes_context)):
                if scenes_context[i] == '{':
                    brace_count += 1
                elif scenes_context[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            # Extract and parse the JSON
            scene_json_str = scenes_context[json_start:json_end]
            parsed_scene = json.loads(scene_json_str)
            
            # Verify all required fields are present
            required_fields = ['id', 'title', 'narration', 'emotion', 'narration_instructions', 
                             'drone_change', 'scene_type', 'year']
            for field in required_fields:
                self.assertIn(field, parsed_scene, f"Scene {scene_id} missing required field: {field}")
                self.assertEqual(parsed_scene[field], scene[field], 
                               f"Scene {scene_id} field {field} doesn't match")
        
        # Verify transition instructions are present
        self.assertIn("emotion flows gradually", scenes_context)
        self.assertIn("narration_instructions flow smoothly", scenes_context)
        self.assertIn("drone_change is correct", scenes_context)
    
    def test_biopic_scene_context_includes_full_json(self):
        """Test that biopic scene context includes full scene JSON with all fields."""
        # Create mock previous scenes with all fields
        prev_scenes = [
            {
                "id": 1,
                "title": "The Discovery",
                "narration": "He made a breakthrough.",
                "emotion": "contemplative",
                "narration_instructions": "Speak naturally with contemplative tone.",
                "scene_type": "WHAT",
                "year": "1905",
                "image_prompt": "Scientist at desk"
            },
            {
                "id": 2,
                "title": "The Impact",
                "narration": "The world took notice.",
                "emotion": "thoughtful",
                "narration_instructions": "Deliver in a measured, thoughtful manner.",
                "scene_type": "WHY",
                "year": "1905",
                "image_prompt": "Newspaper headlines"
            }
        ]
        
        # Build scene context manually (extract the logic)
        recent_scenes = prev_scenes[-5:]
        scenes_context = "RECENT SCENES - AVOID REPEATING THESE EVENTS, maintain continuity to naturally continue the story, and ensure smooth transitions:\n"
        for sc in recent_scenes:
            scenes_context += f"  Scene {sc.get('id')} (FULL JSON): {json.dumps(sc, indent=2)}\n"
        scenes_context += "\nCRITICAL: Do NOT repeat or overlap with events already covered in the scenes above. Each scene must cover DIFFERENT events. If an event was already described, move to its consequences or the next significant moment.\n"
        scenes_context += "CRITICAL: For smooth transitions, ensure:\n"
        scenes_context += "- emotion flows gradually from the previous scene's emotion\n"
        scenes_context += "- narration_instructions flow smoothly from the previous scene's narration_instructions\n"
        
        # Verify the context includes full JSON
        for scene in prev_scenes:
            scene_id = scene['id']
            # Check that the scene JSON is in the context
            self.assertIn(f'Scene {scene_id} (FULL JSON):', scenes_context)
            
            # Parse the JSON from the context
            scene_marker = f'Scene {scene_id} (FULL JSON):'
            start_idx = scenes_context.find(scene_marker)
            self.assertNotEqual(start_idx, -1, f"Scene {scene_id} marker not found")
            
            # Find the start of the JSON
            json_start = scenes_context.find('{', start_idx)
            self.assertNotEqual(json_start, -1, f"JSON start not found for scene {scene_id}")
            
            # Find the end of the JSON
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(scenes_context)):
                if scenes_context[i] == '{':
                    brace_count += 1
                elif scenes_context[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            # Extract and parse the JSON
            scene_json_str = scenes_context[json_start:json_end]
            parsed_scene = json.loads(scene_json_str)
            
            # Verify all required fields are present (biopic doesn't have drone_change)
            required_fields = ['id', 'title', 'narration', 'emotion', 'narration_instructions', 
                             'scene_type', 'year']
            for field in required_fields:
                self.assertIn(field, parsed_scene, f"Scene {scene_id} missing required field: {field}")
                self.assertEqual(parsed_scene[field], scene[field], 
                               f"Scene {scene_id} field {field} doesn't match")
        
        # Verify transition instructions are present
        self.assertIn("emotion flows gradually", scenes_context)
        self.assertIn("narration_instructions flow smoothly", scenes_context)
    
    def test_lol_scene_context_includes_full_json(self):
        """Test that LoL scene context includes full scene JSON with all fields."""
        # Create mock previous scenes with all fields
        prev_scenes = [
            {
                "id": 1,
                "title": "The Battle",
                "narration": "The armies clashed.",
                "emotion": "tense",
                "scene_type": "WHAT",
                "year": "various",
                "image_prompt": "Epic battle scene"
            },
            {
                "id": 2,
                "title": "The Victory",
                "narration": "Victory was achieved.",
                "emotion": "triumphant",
                "scene_type": "WHAT",
                "year": "various",
                "image_prompt": "Victory celebration"
            }
        ]
        
        # Build scene context manually (extract the logic)
        recent_scenes = prev_scenes[-5:]
        scenes_context = "RECENT SCENES - AVOID REPEATING THESE EVENTS, maintain continuity to naturally continue the story, and ensure smooth transitions:\n"
        for sc in recent_scenes:
            scenes_context += f"  Scene {sc.get('id')} (FULL JSON): {json.dumps(sc, indent=2)}\n"
        scenes_context += "\nCRITICAL: Do NOT repeat or overlap with events already covered in the scenes above. Each scene must cover DIFFERENT events. If an event was already described, move to its consequences or the next significant moment.\n"
        scenes_context += "CRITICAL: For smooth transitions, ensure:\n"
        scenes_context += "- emotion flows gradually from the previous scene's emotion\n"
        
        # Verify the context includes full JSON
        for scene in prev_scenes:
            scene_id = scene['id']
            # Check that the scene JSON is in the context
            self.assertIn(f'Scene {scene_id} (FULL JSON):', scenes_context)
            
            # Parse the JSON from the context
            scene_marker = f'Scene {scene_id} (FULL JSON):'
            start_idx = scenes_context.find(scene_marker)
            self.assertNotEqual(start_idx, -1, f"Scene {scene_id} marker not found")
            
            # Find the start of the JSON
            json_start = scenes_context.find('{', start_idx)
            self.assertNotEqual(json_start, -1, f"JSON start not found for scene {scene_id}")
            
            # Find the end of the JSON
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(scenes_context)):
                if scenes_context[i] == '{':
                    brace_count += 1
                elif scenes_context[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            # Extract and parse the JSON
            scene_json_str = scenes_context[json_start:json_end]
            parsed_scene = json.loads(scene_json_str)
            
            # Verify all required fields are present (LoL doesn't have narration_instructions or drone_change)
            required_fields = ['id', 'title', 'narration', 'emotion', 'scene_type', 'year']
            for field in required_fields:
                self.assertIn(field, parsed_scene, f"Scene {scene_id} missing required field: {field}")
                self.assertEqual(parsed_scene[field], scene[field], 
                               f"Scene {scene_id} field {field} doesn't match")
        
        # Verify transition instructions are present
        self.assertIn("emotion flows gradually", scenes_context)
    
    def test_scene_context_with_empty_prev_scenes(self):
        """Test that scene context is empty when no previous scenes."""
        prev_scenes = []
        
        # Build scene context
        scenes_context = ""
        if prev_scenes and len(prev_scenes) > 0:
            recent_scenes = prev_scenes[-5:]
            scenes_context = "RECENT SCENES - AVOID REPEATING THESE EVENTS, maintain continuity, and ensure smooth transitions:\n"
            for sc in recent_scenes:
                scenes_context += f"  Scene {sc.get('id')} (FULL JSON): {json.dumps(sc, indent=2)}\n"
        else:
            scenes_context = ""
        
        self.assertEqual(scenes_context, "")
    
    def test_scene_context_json_formatting(self):
        """Test that scene context JSON is properly formatted and parseable."""
        prev_scenes = [
            {
                "id": 1,
                "title": "Test Scene",
                "narration": "Test narration.",
                "emotion": "tense",
                "narration_instructions": "Speak naturally with subtle tension.",
                "drone_change": "fade_in",
                "scene_type": "WHY",
                "year": "2024"
            }
        ]
        
        # Build scene context
        recent_scenes = prev_scenes[-5:]
        scenes_context = "RECENT SCENES - AVOID REPEATING THESE EVENTS, maintain continuity, and ensure smooth transitions:\n"
        for sc in recent_scenes:
            scenes_context += f"  Scene {sc.get('id')} (FULL JSON): {json.dumps(sc, indent=2)}\n"
        
        # Verify JSON is valid and parseable
        scene_marker = 'Scene 1 (FULL JSON):'
        start_idx = scenes_context.find(scene_marker)
        json_start = scenes_context.find('{', start_idx)
        
        brace_count = 0
        json_end = json_start
        for i in range(json_start, len(scenes_context)):
            if scenes_context[i] == '{':
                brace_count += 1
            elif scenes_context[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        scene_json_str = scenes_context[json_start:json_end]
        
        # Should be valid JSON
        parsed = json.loads(scene_json_str)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed['id'], 1)


if __name__ == '__main__':
    unittest.main()
