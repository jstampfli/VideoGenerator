"""
Configuration for biopic background music.
Music is stored in biopic_music/ with subfolders by mood (e.g., relaxing/, passionate/, happy/).
"""

import os
from pathlib import Path

# Default moods when biopic_music/ cannot be scanned (e.g., folder missing)
BIOPIC_MUSIC_DEFAULT_MOODS = ["relaxing", "passionate", "happy"]

# Music volume in dB (under narration)
BIOPIC_MUSIC_VOLUME_DB = float(os.getenv("BIOPIC_MUSIC_VOLUME", "-24.0"))

# Crossfade duration in seconds between chapter segments
BIOPIC_MUSIC_CROSSFADE_SEC = float(os.getenv("BIOPIC_MUSIC_CROSSFADE", "1.5"))

# Base directory for mood-categorized MP3 files
BIOPIC_MUSIC_DIR = Path(os.getenv("BIOPIC_MUSIC_DIR", "biopic_music"))


def get_available_moods() -> list[str]:
    """
    Scan biopic_music/ for subdirectories that contain at least one .mp3.
    Returns list of mood folder names. Falls back to BIOPIC_MUSIC_DEFAULT_MOODS if folder missing or empty.
    """
    if not BIOPIC_MUSIC_DIR.exists() or not BIOPIC_MUSIC_DIR.is_dir():
        return BIOPIC_MUSIC_DEFAULT_MOODS.copy()

    moods = []
    for subdir in sorted(BIOPIC_MUSIC_DIR.iterdir()):
        if subdir.is_dir() and any(subdir.glob("*.mp3")):
            moods.append(subdir.name)

    return moods if moods else BIOPIC_MUSIC_DEFAULT_MOODS.copy()
