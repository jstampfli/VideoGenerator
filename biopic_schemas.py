"""
JSON schemas for biopic script generation.
Pass these to llm_utils.generate_text(response_json_schema=...) for structured output.
"""

from kenburns_config import KENBURNS_PATTERNS, KENBURNS_INTENSITY_LEVELS, TRANSITION_TYPES, TRANSITION_SPEEDS

# --- Video questions (FIRST step - frames everything else) ---
VIDEO_QUESTIONS_SCHEMA = {
    "type": "object",
    "title": "video_questions",
    "properties": {
        "video_questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 1,
            "description": "Exactly 1 question this documentary will answer. The question frames the entire documentary.",
        },
    },
    "required": ["video_questions"],
}

# --- Landmark events (pre-outline: most important moments to give deep focus) ---
LANDMARK_EVENTS_SCHEMA = {
    "type": "object",
    "title": "landmark_events",
    "properties": {
        "landmark_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_name": {"type": "string"},
                    "year_or_period": {"type": "string"},
                    "significance": {"type": "string"},
                    "key_details_to_cover": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["event_name", "year_or_period", "significance", "key_details_to_cover"],
            },
        },
    },
    "required": ["landmark_events"],
}

# --- Outline (full documentary outline) ---
OUTLINE_SCHEMA = {
    "type": "object",
    "title": "biopic_outline",
    "properties": {
        "person": {"type": "string"},
        "birth_year": {"type": ["integer", "null"]},
        "death_year": {"type": ["integer", "null"]},
        "tagline": {"type": "string"},
        "central_theme": {"type": "string"},
        "narrative_arc": {"type": "string"},
        "overarching_plots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "plot_name": {"type": "string"},
                    "description": {"type": "string"},
                    "starts_chapter": {"type": "integer"},
                    "peaks_chapter": {"type": "integer"},
                    "resolves_chapter": {"type": "integer"},
                    "key_moments": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "sub_plots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subplot_name": {"type": "string"},
                    "description": {"type": "string"},
                    "chapters_span": {"type": "array", "items": {"type": "integer"}},
                    "key_moments": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter_num": {"type": "integer"},
                    "chapter_type": {"type": "string"},
                    "title": {"type": "string"},
                    "year_range": {"type": "string"},
                    "num_scenes": {"type": "integer"},
                    "summary": {"type": "string"},
                    "key_events": {"type": "array", "items": {"type": "string"}},
                    "emotional_tone": {"type": "string"},
                    "music_mood": {"type": "string"},
                    "dramatic_tension": {"type": "string"},
                    "connects_to_next": {"type": "string"},
                    "recurring_threads": {"type": "array", "items": {"type": "string"}},
                    "plots_active": {"type": "array", "items": {"type": "string"}},
                    "plot_developments": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    "required": ["person", "chapters"],
}

# --- Scene outline (array of blocks: event, num_scenes, rationale) ---
SCENE_OUTLINE_SCHEMA = {
    "type": "array",
    "title": "scene_outline",
    "items": {
        "type": "object",
        "properties": {
            "event": {"type": "string"},
            "num_scenes": {"type": "integer"},
            "rationale": {"type": "string"},
        },
        "required": ["event", "num_scenes", "rationale"],
    },
}

# --- Single scene (for significance, storyline completion) ---
SCENE_SCHEMA = {
    "type": "object",
    "title": "biopic_scene",
    "properties": {
        "title": {"type": "string"},
        "narration": {"type": "string"},
        "scene_type": {"type": "string"},
        "image_prompt": {"type": "string"},
        "emotion": {"type": "string"},
        "narration_instructions": {"type": "string"},
        "year": {"type": ["string", "integer"]},
        "kenburns_intensity": {"type": "string", "enum": KENBURNS_INTENSITY_LEVELS},
    },
        "required": ["title", "narration", "scene_type", "image_prompt", "emotion", "narration_instructions", "year", "kenburns_intensity"],
}

# --- Array of scenes (chapter scenes, short scenes, refinement) ---
# chapter_num is optional (used by biopic chapter scenes)
SCENES_ARRAY_SCHEMA = {
    "type": "array",
    "title": "biopic_scenes",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "title": {"type": "string"},
            "narration": {"type": "string"},
            "scene_type": {"type": "string"},
            "image_prompt": {"type": "string"},
            "emotion": {"type": "string"},
            "narration_instructions": {"type": "string"},
            "year": {"type": ["string", "integer"]},
            "chapter_num": {"type": "integer"},  # for biopic chapter scenes
            "kenburns_pattern": {"type": "string", "enum": KENBURNS_PATTERNS},  # camera motion pattern
            "kenburns_intensity": {"type": "string", "enum": KENBURNS_INTENSITY_LEVELS},
            "transition_to_next": {"type": ["string", "null"], "enum": [*TRANSITION_TYPES, None]},
            "transition_speed": {"type": ["string", "null"], "enum": [*TRANSITION_SPEEDS, None], "description": "quick, medium, or slow; used when transition_to_next is not cut"},
            # music_song and music_volume are NOT in schema - added by Pass 4 (film composer) with enum from biopic_music/
        },
        "required": ["id", "title", "narration", "scene_type", "image_prompt", "emotion", "narration_instructions", "year", "chapter_num", "kenburns_pattern", "kenburns_intensity"],
    },
}

# --- Initial metadata (title, tag_line, thumbnail_description, global_block) ---
INITIAL_METADATA_SCHEMA = {
    "type": "object",
    "title": "initial_metadata",
    "properties": {
        "title": {"type": "string"},
        "tag_line": {"type": "string"},
        "thumbnail_description": {"type": "string"},
        "thumbnail_why_type": {"type": "string", "description": "One of: counterintuitive, secret_mystery, known_but_misunderstood"},
        "thumbnail_text": {"type": "string", "description": "Optional short phrase (2-5 words) for thumbnail text overlay"},
        "global_block": {"type": "string"},
    },
    "required": ["title", "tag_line", "thumbnail_description", "global_block"],
}

# --- Initial metadata with 3 thumbnail/title options (user picks best when uploading) ---
INITIAL_METADATA_3_OPTIONS_SCHEMA = {
    "type": "object",
    "title": "initial_metadata_3_options",
    "properties": {
        "tag_line": {"type": "string"},
        "global_block": {"type": "string"},
        "thumbnail_options": {
            "type": "array",
            "description": "Exactly 3 title+thumbnail pairs; user picks best when uploading",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "thumbnail_description": {"type": "string"},
                    "thumbnail_why_type": {"type": "string"},
                    "thumbnail_text": {"type": "string"},
                },
                "required": ["title", "thumbnail_description", "thumbnail_why_type", "thumbnail_text"],
            },
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["tag_line", "global_block", "thumbnail_options"],
}

# --- Initial metadata without global_block (global_block generated separately by Google) ---
INITIAL_METADATA_3_OPTIONS_NO_GLOBAL_SCHEMA = {
    "type": "object",
    "title": "initial_metadata_3_options_no_global",
    "properties": {
        "tag_line": {"type": "string"},
        "thumbnail_options": {
            "type": "array",
            "description": "Exactly 3 title+thumbnail pairs; user picks best when uploading",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "thumbnail_description": {"type": "string"},
                    "thumbnail_why_type": {"type": "string"},
                    "thumbnail_text": {"type": "string"},
                },
                "required": ["title", "thumbnail_description", "thumbnail_why_type", "thumbnail_text"],
            },
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["tag_line", "thumbnail_options"],
}

# --- Global block only (visual style guide, generated by Google) ---
GLOBAL_BLOCK_SCHEMA = {
    "type": "object",
    "title": "global_block",
    "properties": {
        "global_block": {"type": "string", "description": "Visual style guide (300-400 words)"},
    },
    "required": ["global_block"],
}

# --- Short outline ---
SHORT_OUTLINE_SCHEMA = {
    "type": "object",
    "title": "short_outline",
    "properties": {
        "short_title": {"type": "string"},
        "short_description": {"type": "string"},
        "tags": {"type": "string"},
        "music_mood": {"type": "string"},
        "thumbnail_prompt": {"type": "string"},
        "thumbnail_why_type": {"type": "string", "description": "One of: counterintuitive, secret_mystery, known_but_misunderstood"},
        "thumbnail_text": {"type": "string", "description": "Optional short phrase (2-5 words) for thumbnail text overlay"},
        "narrative_structure": {
            "type": "string",
            "enum": ["question_first", "in_medias_res", "outcome_first", "twist_structure", "chronological_story"],
            "description": "Narrative structure for this short. question_first: open with question; in_medias_res: drop into pivotal moment; outcome_first: start with result; twist_structure: common belief then reveal; chronological_story: linear narrative.",
        },
        "video_question": {"type": "string", "description": "The central question or curiosity this short answers. For question_first, state it in scene 1; for other structures, it can be implied or revealed later."},
        "hook_expansion": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["short_title", "short_description", "tags", "music_mood", "thumbnail_prompt", "narrative_structure", "video_question", "hook_expansion", "key_facts"],
}

# --- Final metadata (video_description, tags, pinned_comment) ---
FINAL_METADATA_SCHEMA = {
    "type": "object",
    "title": "final_metadata",
    "properties": {
        "video_description": {"type": "string"},
        "tags": {"type": "string"},
        "pinned_comment": {"type": "string"},
    },
    "required": ["video_description", "tags", "pinned_comment"],
}

# --- Pivotal moments (array of scene_id, justification) ---
PIVOTAL_MOMENTS_SCHEMA = {
    "type": "array",
    "title": "pivotal_moments",
    "items": {
        "type": "object",
        "properties": {
            "scene_id": {"type": "integer"},
            "justification": {"type": "string"},
        },
        "required": ["scene_id", "justification"],
    },
}

# --- Historian depth additions (gaps a historian would fill for depth) ---
HISTORIAN_DEPTH_ADDITIONS_SCHEMA = {
    "type": "array",
    "title": "historian_depth_additions",
    "items": {
        "type": "object",
        "properties": {
            "gap_description": {"type": "string"},
            "what_to_add": {"type": "string"},
            "insert_after_scene_id": {"type": "integer"},
            "year": {"type": ["string", "integer"]},
            "scene_type": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["gap_description", "what_to_add", "insert_after_scene_id", "year", "scene_type", "rationale"],
    },
}

# --- Music selection (film composer pass: song + volume per scene) ---
# Base schema; use build_music_selection_schema(songs) for enum-constrained music_song
MUSIC_SELECTION_SCHEMA = {
    "type": "array",
    "title": "music_selection",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Scene ID"},
            "music_song": {"type": "string", "description": "Relative path like relaxing/song1.mp3"},
            "music_volume": {"type": "string", "enum": ["low", "medium", "loud"]},
        },
        "required": ["id", "music_song", "music_volume"],
    },
}


def build_music_selection_schema(songs: list[str]) -> dict:
    """
    Build music selection schema with music_song enum from actual files in biopic_music.
    Ensures the LLM can only output valid song paths. Falls back to string if songs empty.
    """
    schema = {
        "type": "array",
        "title": "music_selection",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Scene ID"},
                "music_song": (
                    {"type": "string", "enum": songs, "description": "Exact path from available songs"}
                    if songs
                    else {"type": "string", "description": "Relative path like relaxing/song1.mp3"}
                ),
                "music_volume": {"type": "string", "enum": ["low", "medium", "loud"]},
            },
            "required": ["id", "music_song", "music_volume"],
        },
    }
    return schema

# --- Transition selection (PASS 5: transition_to_next and transition_speed per scene) ---
TRANSITION_SELECTION_SCHEMA = {
    "type": "array",
    "title": "transition_selection",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Scene ID"},
            "transition_to_next": {"type": ["string", "null"], "enum": [*TRANSITION_TYPES, None]},
            "transition_speed": {"type": ["string", "null"], "enum": [*TRANSITION_SPEEDS, None], "description": "quick, medium, or slow; use when transition_to_next is not cut; null for last scene"},
        },
        "required": ["id", "transition_to_next"],
    },
}

# --- Hanging storylines (array of storyline objects) ---
HANGING_STORYLINES_SCHEMA = {
    "type": "array",
    "title": "hanging_storylines",
    "items": {
        "type": "object",
        "properties": {
            "storyline_description": {"type": "string"},
            "introduced_in_scene_id": {"type": "integer"},
            "missing_completion": {"type": "string"},
            "completion_year": {"type": ["string", "integer"]},
            "insert_after_scene_id": {"type": "integer"},
            "scene_type": {"type": "string"},
        },
        "required": [
            "storyline_description",
            "introduced_in_scene_id",
            "missing_completion",
            "completion_year",
            "insert_after_scene_id",
            "scene_type",
        ],
    },
}
