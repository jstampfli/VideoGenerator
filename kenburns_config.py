"""
Ken Burns motion pattern constants shared across script generation and video building.
"""

# Available camera motion patterns
KENBURNS_PATTERNS = [
    "zoom_in",          # Slow zoom into center
    "zoom_out",         # Slow zoom out from center
    "zoom_in_up",       # Zoom in while drifting upward
    "zoom_in_down",     # Zoom in while drifting downward
    "drift_up",         # Gentle upward drift at fixed zoom
    "drift_down",       # Gentle downward drift at fixed zoom
]

# Emotional guidance for each pattern (used in LLM prompts for script generation)
KENBURNS_PATTERN_DESCRIPTIONS = {
    "zoom_in":      "intensity/tension/focus",
    "zoom_out":     "reveals/grandeur/establishing shots",
    "zoom_in_up":   "triumph/hope/rising action",
    "zoom_in_down": "loss/decline/weight",
    "drift_up":     "calm reflection/uplift",
    "drift_down":   "melancholy/settling",
}


def get_pattern_prompt_str() -> str:
    """Build a comma-separated list of 'pattern (description)' for use in LLM prompts."""
    return ", ".join(
        f"{p} ({KENBURNS_PATTERN_DESCRIPTIONS[p]})" for p in KENBURNS_PATTERNS
    )
