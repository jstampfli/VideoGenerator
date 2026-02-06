"""
JSON schemas for biopic script generation.
Pass these to llm_utils.generate_text(response_json_schema=...) for structured output.
"""

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
    },
    "required": ["title", "narration", "scene_type", "image_prompt", "emotion", "narration_instructions", "year"],
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
            "chapter_num": {"type": "integer"},  # optional, for biopic chapter scenes
        },
        "required": ["id", "title", "narration", "scene_type", "image_prompt", "emotion", "narration_instructions", "year"],
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
        "global_block": {"type": "string"},
    },
    "required": ["title", "tag_line", "thumbnail_description", "global_block"],
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
        "hook_expansion": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["short_title", "short_description", "tags", "music_mood", "thumbnail_prompt", "hook_expansion", "key_facts"],
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
