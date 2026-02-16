"""
Unit tests for biopic_schemas.py.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from biopic_schemas import (
    VIDEO_QUESTIONS_SCHEMA,
    SHORT_OUTLINE_SCHEMA,
    SCENE_SCHEMA,
    SCENES_ARRAY_SCHEMA,
    build_music_selection_schema,
)
from kenburns_config import KENBURNS_INTENSITY_LEVELS


class TestVideoQuestionsSchema(unittest.TestCase):
    """Test VIDEO_QUESTIONS_SCHEMA enforces exactly 1 question."""

    def test_video_questions_has_max_items_one(self):
        """video_questions array must have maxItems=1 (exactly one question)."""
        props = VIDEO_QUESTIONS_SCHEMA.get("properties", {})
        vq = props.get("video_questions", {})
        self.assertEqual(vq.get("maxItems"), 1, "video_questions must have maxItems=1")
        self.assertEqual(vq.get("minItems"), 1, "video_questions must have minItems=1")

    def test_video_questions_description(self):
        """Schema description should indicate exactly one question."""
        props = VIDEO_QUESTIONS_SCHEMA.get("properties", {})
        vq = props.get("video_questions", {})
        desc = vq.get("description", "")
        self.assertIn("1", desc)
        self.assertIn("question", desc.lower())


class TestShortOutlineSchema(unittest.TestCase):
    """Test SHORT_OUTLINE_SCHEMA includes narrative_structure."""

    def test_short_outline_has_narrative_structure(self):
        """SHORT_OUTLINE_SCHEMA must include narrative_structure with valid enum."""
        props = SHORT_OUTLINE_SCHEMA.get("properties", {})
        ns = props.get("narrative_structure", {})
        self.assertIn("enum", ns)
        self.assertIn("question_first", ns["enum"])
        self.assertIn("in_medias_res", ns["enum"])
        self.assertIn("outcome_first", ns["enum"])
        self.assertIn("twist_structure", ns["enum"])
        self.assertIn("chronological_story", ns["enum"])

    def test_narrative_structure_is_required(self):
        """narrative_structure must be in required list."""
        required = SHORT_OUTLINE_SCHEMA.get("required", [])
        self.assertIn("narrative_structure", required)


class TestSceneSchemaKenburnsIntensity(unittest.TestCase):
    """Test SCENE_SCHEMA includes kenburns_intensity for scene generation."""

    def test_scene_schema_has_kenburns_intensity(self):
        """SCENE_SCHEMA must include kenburns_intensity with valid enum."""
        props = SCENE_SCHEMA.get("properties", {})
        self.assertIn("kenburns_intensity", props)
        ki = props["kenburns_intensity"]
        self.assertIn("enum", ki)
        for level in KENBURNS_INTENSITY_LEVELS:
            self.assertIn(level, ki["enum"], f"kenburns_intensity enum must include {level}")


class TestBuildMusicSelectionSchema(unittest.TestCase):
    """Test build_music_selection_schema produces enum from song list."""

    def test_music_song_enum_from_songs(self):
        """When songs provided, music_song has enum of exact paths."""
        songs = ["relaxing/song1.mp3", "passionate/track2.mp3"]
        schema = build_music_selection_schema(songs)
        music_song = schema["items"]["properties"]["music_song"]
        self.assertIn("enum", music_song)
        self.assertEqual(music_song["enum"], songs)

    def test_music_song_string_when_empty(self):
        """When songs empty, music_song is string (no enum)."""
        schema = build_music_selection_schema([])
        music_song = schema["items"]["properties"]["music_song"]
        self.assertNotIn("enum", music_song)
        self.assertEqual(music_song["type"], "string")


class TestScenesArraySchemaNoMusic(unittest.TestCase):
    """SCENES_ARRAY_SCHEMA must not include music - added by Pass 4 with enum."""

    def test_scenes_array_schema_has_no_music_song(self):
        """music_song must not be in schema to prevent LLM hallucination."""
        props = SCENES_ARRAY_SCHEMA["items"]["properties"]
        self.assertNotIn("music_song", props)

    def test_scenes_array_schema_has_no_music_volume(self):
        """music_volume must not be in schema to prevent LLM hallucination."""
        props = SCENES_ARRAY_SCHEMA["items"]["properties"]
        self.assertNotIn("music_volume", props)
