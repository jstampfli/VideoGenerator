"""
Ken Burns motion pattern constants shared across script generation and video building.
"""

# Per-scene Ken Burns intensity levels (zoom/pan amount beyond cover-fit)
KENBURNS_INTENSITY_LEVELS = ["subtle", "medium", "pronounced"]
KENBURNS_INTENSITY_VALUES = {
    "subtle": 0.05,      # 5% - calm, reflective, establishing shots
    "medium": 0.1,      # 10% - default, balanced
    "pronounced": 0.2,  # 20% - tension, drama, climactic moments
}

KENBURNS_INTENSITY_DESCRIPTIONS = {
    "subtle": "calm, reflective, establishing shots",
    "medium": "default, balanced",
    "pronounced": "tension, drama, climactic moments",
}


def get_intensity_prompt_str() -> str:
    """Build a comma-separated list of 'level (description)' for use in LLM prompts."""
    return ", ".join(
        f"{p} ({KENBURNS_INTENSITY_DESCRIPTIONS[p]})" for p in KENBURNS_INTENSITY_LEVELS
    )


# Scene-to-scene video transition types
# cut = no effect; crossfade = blend; slide_* = one slides out, next slides in from opposite side
TRANSITION_TYPES = [
    "cut",
    "crossfade",
    "slide_left",   # current slides out left, next slides in from right
    "slide_right",  # current slides out right, next slides in from left
    "slide_up",     # current slides out up, next slides in from bottom
    "slide_down",   # current slides out down, next slides in from top
]
TRANSITION_SPEEDS = ["quick", "medium", "slow"]
TRANSITION_SPEED_DURATIONS = {"quick": 0.2, "medium": 0.4, "slow": 0.8}


def get_transition_duration(transition_to_next: str | None, transition_speed: str | None) -> float:
    """Returns duration in seconds. 0 for cut or no transition."""
    if not transition_to_next or transition_to_next == "cut":
        return 0.0
    speed = transition_speed or "medium"
    return TRANSITION_SPEED_DURATIONS.get(speed, 0.4)


# Legacy transition values (for backward compatibility with existing scripts)
LEGACY_TRANSITION_TYPES = ["dissolve_quick", "dissolve_medium", "dissolve_slow", "fade_to_black"]
_LEGACY_TRANSITION_MAP = {
    "dissolve_quick": ("crossfade", "quick"),
    "dissolve_medium": ("crossfade", "medium"),
    "dissolve_slow": ("crossfade", "slow"),
    "fade_to_black": ("crossfade", "medium"),
}


def normalize_transition(transition_to_next: str | None, transition_speed: str | None) -> tuple[str | None, str | None]:
    """Normalize transition. Maps legacy values (dissolve_*, fade_to_black) to (type, speed)."""
    if not transition_to_next:
        return None, None
    trans = str(transition_to_next).strip().lower()
    speed = (transition_speed or "").strip().lower() or None
    if trans in _LEGACY_TRANSITION_MAP:
        return _LEGACY_TRANSITION_MAP[trans]
    if trans in TRANSITION_TYPES:
        return trans, speed if speed in TRANSITION_SPEEDS else "medium"
    return "cut", None


# Rich context for LLM transition selection (Pass 5)
TRANSITION_GUIDANCE = """
CUT - Hard cut, no blend.
  Signifies: Abrupt change, tension, action, urgency, shock.
  Use when: Scene shifts abruptly; conflict, chase, or confrontation; sudden revelation; dramatic pivot; quick cuts between locations.

CROSSFADE - Blend between scenes.
  Signifies: Continuity, flow, same location or time, emotional connection, gentle progression.
  Use when: Scenes flow naturally; same setting or time period; emotional moments; reflection; chronological flow without jarring shift.

SLIDE_LEFT - Current scene slides out left, next slides in from right.
  Signifies: Movement, progression, timeline advancing, modern/documentary feel.
  Use when: Moving forward in time; narrative progression; chapter or act change; dynamic energy.

SLIDE_RIGHT - Current slides out right, next slides in from left.
  Signifies: Similar to slide_left; can imply reversal or contrast.
  Use when: Contrasting scenes; flashback/return; alternative perspective.

SLIDE_UP - Current slides out top, next slides in from bottom.
  Signifies: Uplift, ascension, hope, triumph.
  Use when: Rising action; triumph; hope; achievement; emotional uplift.

SLIDE_DOWN - Current slides out bottom, next slides in from top.
  Signifies: Decline, weight, settling, melancholy.
  Use when: Loss; decline; settling; melancholy; aftermath.
"""


def get_transition_guidance_prompt_str() -> str:
    """Return TRANSITION_GUIDANCE for use in Pass 5 transition selection prompt."""
    return TRANSITION_GUIDANCE.strip()


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
