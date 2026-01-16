"""
Utility functions shared across the video generator scripts.
"""

import re


def calculate_age_from_year_range(birth_year: int | None, year_range: str | int | None) -> int | None:
    """
    Calculate the approximate age of a person based on their birth year and a year range.
    
    Args:
        birth_year: Birth year of the person (e.g., 1879)
        year_range: Year range string (e.g., "1905-1910", "1831-1834", "around 1859") or int (e.g., 1905)
    
    Returns:
        Estimated age (int) or None if birth_year is None or year_range cannot be parsed
    """
    if birth_year is None or year_range is None:
        return None
    
    # Handle int input directly (e.g., year is just 1905)
    if isinstance(year_range, int):
        scene_year = year_range
        age = scene_year - birth_year
        return max(0, age)
    
    # Handle string input - parse it
    # Try to extract years from various formats:
    # "1905-1910" -> (1905, 1910)
    # "1905" -> (1905, 1905)
    # "around 1859" -> (1859, 1859)
    # "1831–1834" -> (1831, 1834) (em dash)
    # "1900s" -> (1900, 1909)
    
    # Convert to string if not already
    year_range_str = str(year_range)
    
    # Match year patterns (1800s, 1900s, 2000s)
    # Use non-capturing group so findall returns full matches, not just the group
    year_pattern = r'\b(?:18|19|20)\d{2}\b'
    years = re.findall(year_pattern, year_range_str)
    
    if not years:
        return None
    
    # Convert to integers (years are now full 4-digit strings like "1936")
    year_ints = [int(y) for y in years]
    
    if len(year_ints) == 1:
        # Single year
        scene_year = year_ints[0]
    elif len(year_ints) >= 2:
        # Range - use the middle or average
        scene_year = (min(year_ints) + max(year_ints)) // 2
    else:
        return None
    
    # Calculate age
    age = scene_year - birth_year
    
    # Ensure age is non-negative (cap at 0 if before birth)
    return max(0, age)
