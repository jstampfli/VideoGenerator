"""
Unit tests for utility functions.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import calculate_age_from_year_range


class TestCalculateAgeFromYearRange(unittest.TestCase):
    """Test cases for calculate_age_from_year_range function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.birth_year = 1912  # Alan Turing's birth year
    
    def test_int_year_input(self):
        """Test with integer year input."""
        self.assertEqual(calculate_age_from_year_range(1912, 1941), 29)
        self.assertEqual(calculate_age_from_year_range(1912, 1952), 40)
        self.assertEqual(calculate_age_from_year_range(1912, 1926), 14)
        self.assertEqual(calculate_age_from_year_range(1912, 1912), 0)
        self.assertEqual(calculate_age_from_year_range(1912, 1900), 0)  # Before birth
    
    def test_string_single_year(self):
        """Test with string containing single year."""
        self.assertEqual(calculate_age_from_year_range(1912, "1941"), 29)
        self.assertEqual(calculate_age_from_year_range(1912, "1952"), 40)
        self.assertEqual(calculate_age_from_year_range(1912, "1926"), 14)
        self.assertEqual(calculate_age_from_year_range(1912, "1936"), 24)
    
    def test_string_year_range(self):
        """Test with string containing year range."""
        # This is the problematic case - year ranges like "1936-1950"
        self.assertEqual(calculate_age_from_year_range(1912, "1936-1950"), 31)  # Average: (1936+1950)/2 = 1943, age = 1943-1912 = 31
        self.assertEqual(calculate_age_from_year_range(1912, "1912-1926"), 7)  # Average: (1912+1926)/2 = 1919, age = 1919-1912 = 7
        self.assertEqual(calculate_age_from_year_range(1912, "1952-2013"), 70)  # Average: (1952+2013)/2 = 1982.5 -> 1982, age = 1982-1912 = 70
    
    def test_string_with_around(self):
        """Test with string containing 'around' prefix."""
        self.assertEqual(calculate_age_from_year_range(1912, "around 1936"), 24)
        self.assertEqual(calculate_age_from_year_range(1912, "around 1950"), 38)
    
    def test_string_with_em_dash(self):
        """Test with em dash (—) instead of hyphen."""
        self.assertEqual(calculate_age_from_year_range(1912, "1936—1950"), 31)  # Em dash
        self.assertEqual(calculate_age_from_year_range(1912, "1912—1926"), 7)
    
    def test_none_inputs(self):
        """Test with None inputs."""
        self.assertIsNone(calculate_age_from_year_range(None, "1941"))
        self.assertIsNone(calculate_age_from_year_range(1912, None))
        self.assertIsNone(calculate_age_from_year_range(None, None))
    
    def test_invalid_year_range(self):
        """Test with invalid year range strings."""
        self.assertIsNone(calculate_age_from_year_range(1912, "invalid"))
        self.assertIsNone(calculate_age_from_year_range(1912, "abc123"))
        self.assertIsNone(calculate_age_from_year_range(1912, ""))
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Year before birth
        self.assertEqual(calculate_age_from_year_range(1912, 1911), 0)
        self.assertEqual(calculate_age_from_year_range(1912, "1910"), 0)
        
        # Birth year itself
        self.assertEqual(calculate_age_from_year_range(1912, 1912), 0)
        self.assertEqual(calculate_age_from_year_range(1912, "1912"), 0)
        
        # Large age
        self.assertEqual(calculate_age_from_year_range(1912, 2000), 88)
    
    def test_multiple_years_in_string(self):
        """Test with multiple year mentions in string."""
        # Should extract all years and use the range
        self.assertEqual(calculate_age_from_year_range(1912, "From 1936 to 1950"), 31)
        self.assertEqual(calculate_age_from_year_range(1912, "In 1912 he was born, and by 1926 he had..."), 7)
    
    def test_actual_scene_data(self):
        """Test with actual data from alan_turing_script.json."""
        # These are the actual problematic cases from the JSON
        self.assertNotEqual(calculate_age_from_year_range(1912, "1936-1950"), 0)  # Should not be 0!
        self.assertNotEqual(calculate_age_from_year_range(1912, "1912-1926"), 0)  # Should not be 0!
        self.assertNotEqual(calculate_age_from_year_range(1912, "1952-2013"), 0)  # Should not be 0!
        self.assertNotEqual(calculate_age_from_year_range(1912, "1926"), 0)  # Should not be 0!


if __name__ == "__main__":
    unittest.main()
