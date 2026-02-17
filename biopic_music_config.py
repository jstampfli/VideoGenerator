"""
Configuration for biopic background music.
Music is stored in biopic_music/ with subfolders by mood (e.g., relaxing/, passionate/, happy/).
"""

import os
from pathlib import Path

# Default moods when biopic_music/ cannot be scanned (e.g., folder missing)
BIOPIC_MUSIC_DEFAULT_MOODS = ["relaxing", "passionate", "happy"]

# Music volume in dB (under narration)
BIOPIC_MUSIC_VOLUME_DB = float(os.getenv("BIOPIC_MUSIC_VOLUME", "-24.5"))

# Per-scene volume levels (used by film composer pass)
BIOPIC_MUSIC_VOLUME_LOW_DB = -28.0
BIOPIC_MUSIC_VOLUME_MEDIUM_DB = -25.0
BIOPIC_MUSIC_VOLUME_LOUD_DB = -22.0

# Crossfade duration in seconds between chapter segments
BIOPIC_MUSIC_CROSSFADE_SEC = float(os.getenv("BIOPIC_MUSIC_CROSSFADE", "1.5"))

# Seconds to shift the crossfade forward—prepend to new segments so the new song fades in more
# during the next scene rather than disrupting the end of the previous. 0 = no shift.
BIOPIC_MUSIC_CROSSFADE_OFFSET_SEC = float(os.getenv("BIOPIC_MUSIC_CROSSFADE_OFFSET", "0.4"))

# Extra seconds at end of biopic videos with only background music (no narration)
BIOPIC_END_TAIL_SEC = float(os.getenv("BIOPIC_END_TAIL", "2.0"))

# Fade-out duration for the end tail (music transition as if moving to next scene)
BIOPIC_END_TAIL_FADEOUT_SEC = float(os.getenv("BIOPIC_END_TAIL_FADEOUT", "1.5"))

# Base directory for mood-categorized MP3 files
BIOPIC_MUSIC_DIR = Path(os.getenv("BIOPIC_MUSIC_DIR", "biopic_music"))

# Normalize music to this LUFS before applying per-scene volume. Set to 0 to disable.
# When enabled, all songs are brought to the same baseline loudness so low/medium/loud sound consistent.
BIOPIC_MUSIC_NORMALIZE_LUFS = float(os.getenv("BIOPIC_MUSIC_NORMALIZE_LUFS", "-18"))

# Use two-pass loudnorm for more accurate normalization (recommended for music). Set to 0 to use single-pass.
BIOPIC_MUSIC_NORMALIZE_TWO_PASS = os.getenv("BIOPIC_MUSIC_NORMALIZE_TWO_PASS", "1").lower() in ("1", "true", "yes")


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


def get_all_songs() -> list[str]:
    """
    Return list of relative paths like 'relaxing/song1.mp3', 'passionate/track2.mp3'
    for all MP3s under BIOPIC_MUSIC_DIR. Returns empty list if directory missing or no MP3s.
    """
    if not BIOPIC_MUSIC_DIR.exists() or not BIOPIC_MUSIC_DIR.is_dir():
        return []

    songs = []
    for subdir in sorted(BIOPIC_MUSIC_DIR.iterdir()):
        if subdir.is_dir():
            for mp3 in sorted(subdir.glob("*.mp3")):
                songs.append(f"{subdir.name}/{mp3.name}")

    return sorted(songs)


def volume_label_to_db(label: str) -> float:
    """Map volume label (low/medium/loud) to dB. Defaults to medium for unknown labels."""
    label = (label or "").strip().lower()
    if label == "low":
        return BIOPIC_MUSIC_VOLUME_LOW_DB
    if label == "loud":
        return BIOPIC_MUSIC_VOLUME_LOUD_DB
    return BIOPIC_MUSIC_VOLUME_MEDIUM_DB
