"""
Unit tests for kenburns_config transition types and helpers.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kenburns_config import (
    TRANSITION_TYPES,
    TRANSITION_SPEEDS,
    TRANSITION_SPEED_DURATIONS,
    get_transition_duration,
    get_transition_guidance_prompt_str,
    normalize_transition,
)


class TestTransitionTypes(unittest.TestCase):
    """Test transition type constants."""

    def test_transition_types_include_cut_crossfade_and_slides(self):
        """TRANSITION_TYPES includes cut, crossfade, and all slide directions."""
        expected = ["cut", "crossfade", "slide_left", "slide_right", "slide_up", "slide_down"]
        self.assertEqual(TRANSITION_TYPES, expected)

    def test_transition_speeds_are_quick_medium_slow(self):
        """TRANSITION_SPEEDS includes quick, medium, slow."""
        self.assertEqual(TRANSITION_SPEEDS, ["quick", "medium", "slow"])

    def test_transition_speed_durations(self):
        """TRANSITION_SPEED_DURATIONS maps speeds to correct seconds."""
        self.assertEqual(TRANSITION_SPEED_DURATIONS["quick"], 0.2)
        self.assertEqual(TRANSITION_SPEED_DURATIONS["medium"], 0.4)
        self.assertEqual(TRANSITION_SPEED_DURATIONS["slow"], 0.8)


class TestGetTransitionDuration(unittest.TestCase):
    """Test get_transition_duration helper."""

    def test_cut_returns_zero(self):
        """Cut and None return 0 duration."""
        self.assertEqual(get_transition_duration("cut", None), 0.0)
        self.assertEqual(get_transition_duration(None, "medium"), 0.0)
        self.assertEqual(get_transition_duration("", "medium"), 0.0)

    def test_crossfade_uses_speed(self):
        """Crossfade uses transition_speed for duration."""
        self.assertEqual(get_transition_duration("crossfade", "quick"), 0.2)
        self.assertEqual(get_transition_duration("crossfade", "medium"), 0.4)
        self.assertEqual(get_transition_duration("crossfade", "slow"), 0.8)

    def test_slide_uses_speed(self):
        """Slide transitions use transition_speed for duration."""
        self.assertEqual(get_transition_duration("slide_left", "quick"), 0.2)
        self.assertEqual(get_transition_duration("slide_right", "medium"), 0.4)
        self.assertEqual(get_transition_duration("slide_up", "slow"), 0.8)

    def test_defaults_to_medium_when_speed_missing(self):
        """Missing or invalid speed defaults to 0.4 (medium)."""
        self.assertEqual(get_transition_duration("crossfade", None), 0.4)
        self.assertEqual(get_transition_duration("crossfade", ""), 0.4)
        self.assertEqual(get_transition_duration("crossfade", "invalid"), 0.4)


class TestNormalizeTransition(unittest.TestCase):
    """Test normalize_transition for legacy and new values."""

    def test_legacy_dissolve_maps_to_crossfade(self):
        """Legacy dissolve_* maps to (crossfade, speed)."""
        self.assertEqual(normalize_transition("dissolve_quick", None), ("crossfade", "quick"))
        self.assertEqual(normalize_transition("dissolve_medium", None), ("crossfade", "medium"))
        self.assertEqual(normalize_transition("dissolve_slow", None), ("crossfade", "slow"))

    def test_legacy_fade_to_black_maps_to_crossfade_medium(self):
        """Legacy fade_to_black maps to (crossfade, medium)."""
        self.assertEqual(normalize_transition("fade_to_black", None), ("crossfade", "medium"))

    def test_new_types_passthrough(self):
        """New transition types pass through with speed."""
        self.assertEqual(normalize_transition("cut", None), ("cut", "medium"))  # speed unused for cut
        self.assertEqual(normalize_transition("crossfade", "slow"), ("crossfade", "slow"))
        self.assertEqual(normalize_transition("slide_left", "quick"), ("slide_left", "quick"))

    def test_new_types_default_speed_to_medium(self):
        """New types with invalid/missing speed get medium."""
        self.assertEqual(normalize_transition("crossfade", None), ("crossfade", "medium"))
        self.assertEqual(normalize_transition("crossfade", "invalid"), ("crossfade", "medium"))

    def test_none_returns_none_tuple(self):
        """None transition returns (None, None)."""
        self.assertEqual(normalize_transition(None, None), (None, None))

    def test_invalid_returns_cut(self):
        """Invalid transition type returns (cut, None)."""
        self.assertEqual(normalize_transition("invalid_wipe", "medium"), ("cut", None))


class TestTransitionGuidance(unittest.TestCase):
    """Test transition guidance for Pass 5 prompt."""

    def test_get_transition_guidance_prompt_str_returns_non_empty(self):
        """get_transition_guidance_prompt_str returns non-empty string with transition context."""
        result = get_transition_guidance_prompt_str()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)
        self.assertIn("CUT", result)
        self.assertIn("CROSSFADE", result)
        self.assertIn("SLIDE_LEFT", result)
        self.assertIn("Signifies:", result)
        self.assertIn("Use when:", result)


if __name__ == "__main__":
    unittest.main()
