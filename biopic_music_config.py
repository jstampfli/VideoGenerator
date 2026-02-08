"""
Configuration for biopic background music.
Music is stored in biopic_music/ with subfolders by mood (e.g., relaxing/, passionate/, happy/).
"""

import os
from pathlib import Path

# Default moods when biopic_music/ cannot be scanned (e.g., folder missing)
BIOPIC_MUSIC_DEFAULT_MOODS = ["relaxing", "passionate", "happy"]

# Music volume in dB (under narration)
BIOPIC_MUSIC_VOLUME_DB = float(os.getenv("BIOPIC_MUSIC_VOLUME", "-23.5"))

# Crossfade duration in seconds between chapter segments
BIOPIC_MUSIC_CROSSFADE_SEC = float(os.getenv("BIOPIC_MUSIC_CROSSFADE", "1.5"))

# Extra seconds at end of biopic videos with only background music (no narration)
BIOPIC_END_TAIL_SEC = float(os.getenv("BIOPIC_END_TAIL", "2.0"))

# Fade-out duration for the end tail (music transition as if moving to next scene)
BIOPIC_END_TAIL_FADEOUT_SEC = float(os.getenv("BIOPIC_END_TAIL_FADEOUT", "1.5"))

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
