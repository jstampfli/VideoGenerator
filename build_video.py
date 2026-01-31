import os
import sys
import json
import base64
import time
import random
import argparse
import tempfile
import shutil
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, AudioClip, CompositeAudioClip, concatenate_audioclips
from concurrent.futures import ThreadPoolExecutor, as_completed


# ------------- CONFIG -------------

load_dotenv()  # Load API keys from .env file
client = OpenAI()

# OpenAI models
IMG_MODEL = "gpt-image-1.5"

# TTS Provider: "openai", "elevenlabs", or "google"
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "google").lower()

# OpenAI TTS settings
OPENAI_TTS_MODEL = "gpt-4o-mini-tts-2025-12-15"
OPENAI_TTS_VOICE = "marin"

# ElevenLabs TTS settings
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # Default: George (narrative)
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# Google Cloud TTS settings
# Voice options: en-US-Wavenet-D (male), en-US-Wavenet-F (female), en-US-Neural2-D (male), en-US-Neural2-F (female)
# See full list: https://cloud.google.com/text-to-speech/docs/voices
GOOGLE_TTS_VOICE = os.getenv("GOOGLE_TTS_VOICE", "en-US-Studio-Q")  # High quality studio voice
GOOGLE_TTS_LANGUAGE = os.getenv("GOOGLE_TTS_LANGUAGE", "en-US")
GOOGLE_TTS_SPEAKING_RATE = float(os.getenv("GOOGLE_TTS_SPEAKING_RATE", "0.92"))  # 0.25 to 4.0 (slightly slower for clarity)
GOOGLE_TTS_PITCH = float(os.getenv("GOOGLE_TTS_PITCH", "-1.0"))  # -20.0 to 20.0 semitones (slightly deeper)
GOOGLE_VOICE_TYPE = os.getenv("GOOGLE_VOICE_TYPE", "").lower()  # "chirp", "gemini", or empty - chirp voices don't support SSML/prosody, gemini uses new API with prompt parameter
GOOGLE_GEMINI_MALE_SPEAKER = os.getenv("GOOGLE_GEMINI_MALE_SPEAKER", "Charon")  # Default Gemini male voice
GOOGLE_GEMINI_FEMALE_SPEAKER = os.getenv("GOOGLE_GEMINI_FEMALE_SPEAKER", "")  # Gemini female voice

# Narration instructions for Gemini/OpenAI TTS (from .env): biopic vs horror mode
BIOPIC_NARRATION_INSTRUCTIONS = os.getenv("BIOPIC_NARRATION_INSTRUCTIONS", "")
HORROR_NARRATION_INSTRUCTIONS = os.getenv("HORROR_NARRATION_INSTRUCTIONS", "")
# Set at start of build_video based on --horror: which instructions to use for this run
NARRATION_INSTRUCTIONS_FOR_TTS = ""

# Horror background audio volume settings (from .env or defaults)
# More impactful defaults: room tone louder, drone more prominent
DRONE_BASE_VOLUME_DB = float(os.getenv("DRONE_BASE", "-20.0"))  # Base drone volume (default: -20 dB, was -25)
DRONE_MAX_VOLUME_DB = float(os.getenv("DRONE_MAX", "-15.0"))  # Max drone volume for swell (default: -15 dB, was -20)
ROOM_SOUND_BASE_VOLUME_DB = float(os.getenv("ROOM_SOUND_BASE", "-35.0"))  # Room tone volume (default: -35 dB, was -45)
ENV_AUDIO_VOLUME_DB = float(os.getenv("ENV_AUDIO_VOLUME", "-28.0"))  # Environment audio volume (default: -28 dB)

# Environment audio file paths (from .env)
ENV_BLIZZARD_AUDIO = os.getenv("ENV_BLIZZARD_AUDIO", None)
ENV_SNOW_AUDIO = os.getenv("ENV_SNOW_AUDIO", None)
ENV_FOREST_AUDIO = os.getenv("ENV_FOREST_AUDIO", None)
ENV_RAIN_AUDIO = os.getenv("ENV_RAIN_AUDIO", None)
ENV_INDOORS_AUDIO = os.getenv("ENV_INDOORS_AUDIO", None)
ENV_JUNGLE_AUDIO = os.getenv("ENV_JUNGLE_AUDIO", None)

# Global flag to track if female voice should be used (set by --female flag)
USE_FEMALE_VOICE = False

# Scene pause settings
END_SCENE_PAUSE_LENGTH = float(os.getenv("END_SCENE_PAUSE_LENGTH", "0.15"))  # Pause length in seconds at end of each scene (default: 150ms)

# Horror disclaimer: first scene for horror videos (fixed image + env/room noise, no narration)
HORROR_DISCLAIMER_DURATION = 3.0  # seconds
FIXED_IMAGES_DIR = Path("fixed_images")
HORROR_DISCLAIMER_TALL = FIXED_IMAGES_DIR / "tall_horror_disclaimer.jpg"   # 9:16 shorts
HORROR_DISCLAIMER_WIDE = FIXED_IMAGES_DIR / "wide_horror_disclaimer.jpg"   # 16:9 main


def get_horror_disclaimer_image_path(is_vertical: bool) -> Path | None:
    """
    Return the path to the horror disclaimer image for the given format, or None if not found.
    - is_vertical True (shorts): tall_horror_disclaimer.jpg
    - is_vertical False (main): wide_horror_disclaimer.jpg
    """
    path = HORROR_DISCLAIMER_TALL if is_vertical else HORROR_DISCLAIMER_WIDE
    return path if path.exists() else None


def get_emotion_prosody(emotion: str | None) -> dict:
    """
    Map emotion strings to subtle SSML prosody settings (rate, pitch, volume).
    Returns a dict with 'rate', 'pitch', and 'volume' strings for use in SSML prosody tags.
    Uses subtle adjustments (5-10% rate, 1-2 semitones pitch, 2-3dB volume) to maintain documentary style.
    """
    if not emotion:
        # Default: no adjustment
        return {"rate": "100%", "pitch": "0st", "volume": "0dB"}
    
    emotion_lower = emotion.lower().strip()
    
    # Check for compound emotions first (e.g., "triumphant, foreboding" or "urgent dread")
    # For compound emotions, use the primary emotion
    
    # Urgent/desperate emotions - slightly faster, slightly higher, slightly louder
    if any(word in emotion_lower for word in ["urgent", "desperate"]):
        if "dread" in emotion_lower:
            return {"rate": "103%", "pitch": "+0.5st", "volume": "+1dB"}  # urgent dread: tense but darker
        return {"rate": "105%", "pitch": "+1st", "volume": "+2dB"}
    
    # Tense/anxious emotions
    if emotion_lower in ["tense", "claustrophobic", "uneasy"]:
        if emotion_lower == "claustrophobic":
            return {"rate": "104%", "pitch": "+0.5st", "volume": "+1dB"}  # tighter, more compressed
        return {"rate": "105%", "pitch": "+1st", "volume": "+2dB"}
    
    # Dread/fear emotions - slower, lower, quieter (but still tense)
    if emotion_lower in ["dread", "menacing"]:
        return {"rate": "95%", "pitch": "-1st", "volume": "-1dB"}
    
    # Triumphant/positive emotions - faster, higher, louder
    if any(word in emotion_lower for word in ["triumphant", "exhilarating"]):
        if "foreboding" in emotion_lower:
            return {"rate": "102%", "pitch": "+1st", "volume": "+2dB"}  # triumphant but with tension
        return {"rate": "105%", "pitch": "+2st", "volume": "+3dB"}
    
    # Defiant/determined emotions - normal to slightly faster, higher, louder
    if any(word in emotion_lower for word in ["defiant", "daring", "ominous resolve"]):
        if "quietly" in emotion_lower:
            return {"rate": "98%", "pitch": "+0.5st", "volume": "+1dB"}  # quietly defiant: subdued strength
        return {"rate": "100%", "pitch": "+1st", "volume": "+2dB"}
    
    # Sad/somber emotions - slower, lower, softer
    if emotion_lower in ["somber", "elegiac"]:
        return {"rate": "90%", "pitch": "-2st", "volume": "-3dB"}
    
    # Shattered/devastated emotions - slower, lower, much quieter
    if emotion_lower in ["shattered", "devastated"]:
        return {"rate": "88%", "pitch": "-2.5st", "volume": "-4dB"}
    
    # Contemplative/reflective - slower, lower, softer (subtle)
    if emotion_lower in ["contemplative", "watchful", "controlled"]:
        if emotion_lower == "controlled":
            return {"rate": "95%", "pitch": "0st", "volume": "0dB"}  # controlled: measured and neutral
        return {"rate": "92%", "pitch": "-1st", "volume": "-2dB"}
    
    # Calculating/strategic - slightly slower, neutral pitch, slightly softer
    if emotion_lower in ["calculating"]:
        return {"rate": "96%", "pitch": "0st", "volume": "-1dB"}
    
    # Humiliated - slower, lower, quieter
    if emotion_lower in ["humiliated"]:
        return {"rate": "93%", "pitch": "-1.5st", "volume": "-2dB"}
    
    # Sensuous/romantic - slightly slower, slightly higher, softer
    if "sensuous" in emotion_lower or "intimacy" in emotion_lower:
        if "ambitious" in emotion_lower:
            return {"rate": "98%", "pitch": "+0.5st", "volume": "+1dB"}  # ambitious intimacy: subtle intensity
        return {"rate": "94%", "pitch": "+0.5st", "volume": "-1dB"}
    
    # Relief (ruthless relief) - normal speed, slight variation
    if "relief" in emotion_lower:
        return {"rate": "97%", "pitch": "+0.5st", "volume": "+1dB"}
    
    # Default: no adjustment for unknown emotions
    return {"rate": "100%", "pitch": "0st", "volume": "0dB"}


def text_to_ssml(text: str, emotion: str | None = None) -> str:
    """
    Convert plain text to SSML with natural pauses and pacing.
    This makes Google Cloud TTS sound more natural and documentary-like.
    
    Args:
        text: The text to convert to SSML
        emotion: Optional emotion string (e.g., "tense", "triumphant", "contemplative")
                 to apply subtle prosody adjustments matching the scene's emotional tone
    """
    import re
    
    # If using Chirp or Gemini voice, skip SSML and return plain text
    # Chirp doesn't support SSML, Gemini uses prompt parameter instead
    if TTS_PROVIDER == "google" and GOOGLE_VOICE_TYPE in ["chirp", "gemini"]:
        return text
    
    # Get emotion-based prosody settings
    prosody = get_emotion_prosody(emotion)
    
    # Escape special XML characters
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    
    # IMPORTANT: Handle ellipsis FIRST, before processing individual periods
    # Replace ellipsis with dramatic pause (don't keep the dots - TTS will read them otherwise)
    # Handle both three-dot ellipsis (...) and Unicode ellipsis character (…)
    text = re.sub(r'\.\.\.(\s+)', r'<break time="600ms"/>\1', text)  # Mid-text with space after
    text = re.sub(r'\.\.\.$', r'<break time="600ms"/>', text)  # End of text
    text = re.sub(r'…(\s+)', r'<break time="600ms"/>\1', text)  # Unicode ellipsis mid-text
    text = re.sub(r'…$', r'<break time="600ms"/>', text)  # Unicode ellipsis end of text
    
    # Add pauses after sentences (periods, exclamation, question marks)
    # Handle both mid-text (followed by space) and end-of-text cases
    # Note: This won't match periods that are part of ellipsis since we handled those above
    text = re.sub(r'\.(\s+)', r'.<break time="400ms"/>\1', text)
    text = re.sub(r'\.$', r'.<break time="400ms"/>', text)  # End of text
    text = re.sub(r'\!(\s+)', r'!<break time="350ms"/>\1', text)
    text = re.sub(r'\!$', r'!<break time="350ms"/>', text)  # End of text
    text = re.sub(r'\?(\s+)', r'?<break time="350ms"/>\1', text)
    text = re.sub(r'\?$', r'?<break time="350ms"/>', text)  # End of text
    
    # Add shorter pauses after commas
    text = re.sub(r',(\s+)', r',<break time="200ms"/>\1', text)
    text = re.sub(r',$', r',<break time="200ms"/>', text)  # End of text (rare but possible)
    
    # Add pauses after colons and semicolons
    text = re.sub(r':(\s+)', r':<break time="300ms"/>\1', text)
    text = re.sub(r';(\s+)', r';<break time="250ms"/>\1', text)
    
    # Handle dashes/hyphens as pauses (remove the dash, add pause)
    # Em-dash (—) - dramatic pause, remove the dash character completely
    text = re.sub(r'—', r'<break time="400ms"/>', text)
    
    # Regular hyphen/dash - replace with pause when used as separator (with spaces)
    # This handles: "word - word" or "word -word" or "word- word" patterns
    text = re.sub(r'(\s+)-(\s+)', r'\1<break time="300ms"/>\2', text)  # Space-dash-space
    text = re.sub(r'(\s+)-(\w)', r'\1<break time="300ms"/>\2', text)  # Space-dash-word
    text = re.sub(r'(\w)-(\s+)', r'\1<break time="300ms"/>\2', text)  # Word-dash-space
    text = re.sub(r'(\s+)-$', r'\1<break time="300ms"/>', text)  # Space-dash at end
    text = re.sub(r'^-(\s+)', r'<break time="300ms"/>\1', text)  # Dash-space at start
    
    # Emphasize numbers/years (slightly slower)
    text = re.sub(r'\b(1[89]\d{2}|20\d{2})\b', r'<prosody rate="95%">\1</prosody>', text)
    
    # Wrap in prosody tag if emotion is provided and prosody has adjustments
    if emotion and (prosody["rate"] != "100%" or prosody["pitch"] != "0st" or prosody["volume"] != "0dB"):
        # Studio voices don't support pitch attribute - exclude it if using Studio voice
        is_studio_voice = "Studio" in GOOGLE_TTS_VOICE if TTS_PROVIDER == "google" else False
        
        # Build prosody attributes - exclude pitch for Studio voices
        prosody_attrs_parts = [f'rate="{prosody["rate"]}"']
        if not is_studio_voice:
            prosody_attrs_parts.append(f'pitch="{prosody["pitch"]}"')
        prosody_attrs_parts.append(f'volume="{prosody["volume"]}"')
        
        prosody_attrs = " ".join(prosody_attrs_parts)
        text = f'<prosody {prosody_attrs}>{text}</prosody>'
    
    # Wrap in speak tags
    ssml = f'<speak>{text}</speak>'
    
    return ssml

# Initialize TTS clients
elevenlabs_client = None
google_tts_client = None

if TTS_PROVIDER == "elevenlabs":
    try:
        from elevenlabs import ElevenLabs
        elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        print(f"[TTS] Using ElevenLabs (voice: {ELEVENLABS_VOICE_ID})")
    except ImportError:
        print("[WARNING] elevenlabs not installed. Run: py -m pip install elevenlabs")
        print("[WARNING] Falling back to OpenAI TTS")
        TTS_PROVIDER = "openai"
    except Exception as e:
        print(f"[WARNING] ElevenLabs init failed: {e}")
        print("[WARNING] Falling back to OpenAI TTS")
        TTS_PROVIDER = "openai"

elif TTS_PROVIDER == "google":
    try:
        from google.cloud import texttospeech
        google_tts_client = texttospeech.TextToSpeechClient()
        print(f"[TTS] Using Google Cloud TTS (voice: {GOOGLE_TTS_VOICE})")
    except ImportError:
        print("[WARNING] google-cloud-texttospeech not installed. Run: py -m pip install google-cloud-texttospeech")
        print("[WARNING] Falling back to OpenAI TTS")
        TTS_PROVIDER = "openai"
    except Exception as e:
        print(f"[WARNING] Google Cloud TTS init failed: {e}")
        print("[WARNING] Make sure GOOGLE_APPLICATION_CREDENTIALS is set to your service account key JSON file")
        print("[WARNING] Falling back to OpenAI TTS")
        TTS_PROVIDER = "openai"

if TTS_PROVIDER == "openai":
    print(f"[TTS] Using OpenAI ({OPENAI_TTS_MODEL}, voice: {OPENAI_TTS_VOICE})")

# Final video resolution (we'll crop/scale images to this)
OUTPUT_RESOLUTION_LANDSCAPE = (1920, 1080)  # 16:9 for main videos
OUTPUT_RESOLUTION_VERTICAL = (1080, 1920)   # 9:16 for shorts
FPS = 30

IMAGES_DIR = Path("generated_images")
AUDIO_DIR = Path("generated_audio")

MAX_WORKERS = 5  # 2–4 is usually safe; higher risks hitting rate limits

# Runtime config (set by build_video)
class Config:
    save_assets = False
    temp_dir = None
    is_vertical = False  # True for shorts (9:16), False for main videos (16:9)
    
    @property
    def output_resolution(self):
        return OUTPUT_RESOLUTION_VERTICAL if self.is_vertical else OUTPUT_RESOLUTION_LANDSCAPE
    
    @property
    def image_size(self):
        # OpenAI image generation sizes
        return "1024x1536" if self.is_vertical else "1536x1024"

config = Config()

MAX_PREV_SUMMARY_CHARS = 300  # keep it short so prompts don't blow up

OPENAI_TTS_STYLE_PROMPT = (
    "Read the following text in a calm, neutral, and consistent tone. "
    "Maintain stable volume and pitch throughout. Do not add emotional "
    "inflection, dramatic emphasis, or noticeable changes in speaking speed. "
)


def build_image_prompt(scene: dict, prev_scene: dict | None, global_block_override: str | None = None, exclude_title_narration: bool = False, include_safety_instructions: bool = False) -> str:
    """
    Build a rich image prompt for the current scene, including:
    - Current scene description (title + narration + image_prompt, unless exclude_title_narration is True)
    - Brief memory of the previous scene for continuity
    - Global visual style and constraints (no text, documentary tone)
    - Age specification when the person appears
    
    Args:
        exclude_title_narration: If True, only use image_prompt (for attempts 4+ to avoid problematic text)
        include_safety_instructions: If True, include safety constraints block (only after first attempt fails)
    """
    title = scene.get("title", "").strip()
    narration = scene.get("narration", "").strip()
    scene_img_prompt = scene.get("image_prompt", "").strip()

    # --- Previous scene memory ---
    if prev_scene is None:
        prev_block = (
            "This is the opening scene of a long-form documentary. "
            "Establish the core visual style that will stay consistent across the entire video."
        )
    else:
        prev_title = prev_scene.get("title", "").strip()
        prev_narr = prev_scene.get("narration", "").strip()
        prev_img_prompt = prev_scene.get("image_prompt", "").strip()

        # Build a short textual summary of the previous scene
        prev_text = " ".join(
            x for x in [prev_title, prev_narr, prev_img_prompt] if x
        ).strip()
        if len(prev_text) > MAX_PREV_SUMMARY_CHARS:
            prev_text = prev_text[:MAX_PREV_SUMMARY_CHARS].rsplit(" ", 1)[0] + "..."

        prev_block = (
            "Maintain visual continuity with the previous scene, which showed: "
            f"{prev_text} "
            "Use similar color grading, lighting mood, and overall artistic treatment."
        )

    # --- Global documentary style block ---
    if global_block_override:
        global_block = global_block_override
    else:
        # Default global block (for backward compatibility)
        global_block = (
            "This image is part of a single, cohesive cinematic documentary about Nikola Tesla "
            "for a YouTube channel called Human Footprints. Every scene must look like a frame "
            "from the same film.\n\n"
            "Use one consistent visual style across ALL scenes: a semi-realistic digital painting "
            "that resembles a high-end historical documentary still. Avoid cartoony, anime, comic, "
            "3D render, or flat graphic styles.\n\n"
            "Color palette should remain unified throughout the video: deep navy blues, desaturated teals, "
            "muted warm golds and ambers for light sources, and soft sepia shadows. Highlights should feel "
            "like warm lamplight or early electrical glow contrasting against cool, dark backgrounds.\n\n"
            "Lighting should be dramatic, almost chiaroscuro: strong key light on the subject, soft falloff "
            "into darkness, and subtle atmospheric haze in the background. Overall mood: serious, thoughtful, "
            "slightly melancholic but with a sense of awe and intellect.\n\n"
            "Framing and composition should also be consistent: cinematic 16:9 framing, mostly medium shots "
            "and medium-wide shots, with simple, uncluttered backgrounds. Use a shallow depth-of-field feel "
            "so the main subject is clearly separated from the background. Camera angles should be mostly eye-level "
            "or slightly above, avoiding extreme wide-angle distortion or exaggerated perspectives.\n\n"
            "Whenever Nikola Tesla is shown, keep his appearance consistent across scenes: tall, thin face, sharp features, "
            "dark neatly combed hair, distinct mustache, and dark formal suit appropriate to the late 19th and early 20th century. "
            "Do not change his ethnicity, facial structure, or general look between scenes. Adjust only age and clothing "
            "when appropriate to the period of his life."
        )

    # --- No text / cleanliness constraints ---
    constraints_block = (
        "Do NOT include any text, titles, captions, subtitles, numbers, letters, labels, watermarks, logos, "
        "user interface elements, or graphic overlays in the image. The frame must be completely text-free.\n\n"
        "Keep the visual style, color palette, and overall mood strictly consistent with the rest of the video. "
        "This should look like one continuous documentary, not a mix of different art styles."
    )

    # --- Safety constraints (only include after first attempt fails) ---
    safety_block = ""
    if include_safety_instructions:
        safety_block = (
            "\n\nSAFETY: Safe, appropriate, educational documentary content only. Avoid graphic imagery - for sensitive historical content (death, injury, conflict), focus on emotional impact and aftermath rather than graphic physical details."
        )

    # --- Current scene description block ---
    current_block_parts = []
    # On attempts 4+, exclude title and narration (which may contain problematic words)
    # and only use the image_prompt (which should be cleaner)
    if not exclude_title_narration:
        if title:
            current_block_parts.append(f"Scene title: {title}.")
        if narration:
            current_block_parts.append("Narration for this scene:\n" + narration)
    if scene_img_prompt:
        current_block_parts.append("Visual details to emphasize:\n" + scene_img_prompt)
    current_block = "\n".join(current_block_parts)

    # Final prompt
    prompt = (
        current_block
        + "\n\n"
        + prev_block
        + "\n\n"
        + global_block
        + "\n\n"
        + constraints_block
        + safety_block
    )

    return prompt


def load_scenes(path: str):
    """
    Load scenes from a JSON file.

    Expected structure: either
    1. A list of scene dicts directly, or
    2. A dict with a "scenes" key containing the list.

    Each scene dict should have:
    {
      "id": 1,
      "title": "Opening",
      "narration": "Text to be spoken",
      "image_prompt": "Prompt for image (optional)"
    }
    
    Returns: tuple of (scenes_list, metadata_dict_or_none)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    metadata = None
    if isinstance(data, list):
        scenes = data
    elif isinstance(data, dict) and "scenes" in data:
        scenes = data["scenes"]
        metadata = data.get("metadata")
    else:
        raise ValueError(f"Invalid JSON format in {path}. Expected array or dict with 'scenes' key.")
    
    return scenes, metadata


class SafetyViolationError(Exception):
    """Custom exception for safety violations with metadata."""
    def __init__(self, message, violation_type=None, original_prompt=None):
        super().__init__(message)
        self.violation_type = violation_type
        self.original_prompt = original_prompt


def sanitize_prompt_for_safety(prompt: str, violation_type: str = None, sanitize_attempt: int = 0, exclude_title_narration: bool = False) -> str:
    """
    Sanitize an image prompt to avoid safety violations with progressive sanitization.
    When a safety violation occurs, we modify the prompt to be more abstract,
    focus on achievements rather than struggles, and add explicit safety instructions.
    
    Args:
        prompt: The original prompt to sanitize
        violation_type: Type of violation detected (self-harm, violence, death, harmful)
        sanitize_attempt: Retry attempt number (0=first, 1=light, 2=medium, 3+=heavy)
    """
    import re
    original_prompt = prompt
    
    # Progressive sanitization based on attempt number - apply all levels up to sanitize_attempt
    # Light sanitization (attempt 1+): Replace specific problematic words
    if sanitize_attempt >= 1:
        # Blood-related
        prompt = re.sub(r'\bbleeding\b', 'injury', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bsmeared with blood\b', 'damaged armor', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bbody\s+slack\b', 'figure', prompt, flags=re.IGNORECASE)
        
        # Death-related
        prompt = re.sub(r'\bDECEASED\b', 'no longer present', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bdies\b', 'passes', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bdeath\b', 'passing', prompt, flags=re.IGNORECASE)
        
        # Violence
        prompt = re.sub(r'\bchains\b', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bkill\b', 'end', prompt, flags=re.IGNORECASE)
        
        # Suicide
        prompt = re.sub(r'\bkill\s+himself\b', 'end his life', prompt, flags=re.IGNORECASE)
        
        # General problematic words
        prompt = re.sub(r'\b(harm|hurt|pain|suffering|suicide|cut|violence)\b', 'challenge', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\b(depression|despair|hopeless)\b', 'contemplation', prompt, flags=re.IGNORECASE)
    
    # Medium sanitization (attempt 2+): Replace phrases and abstract physical details
    if sanitize_attempt >= 2:
        # Additional phrase replacements
        prompt = re.sub(r'\bblood\b', 'stain', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bsuffering\b', 'contemplation', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bstruggle\b', 'journey', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\billness\b', 'health challenges', prompt, flags=re.IGNORECASE)
        
        # Abstract physical details
        prompt = re.sub(r'\bgraphic\b', 'symbolic', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bwound\b', 'injury', prompt, flags=re.IGNORECASE)
    
    # Heavy sanitization (attempt 3): Focus on aftermath/emotions, use symbolic representation
    if sanitize_attempt >= 3:
        # Replace sensitive scenes with aftermath/emotional focus
        # If scene contains death/suicide references, refocus on aftermath
        if re.search(r'\b(passing|end|death|suicide|kill|bleeding|blood)\b', prompt, re.IGNORECASE):
            # Replace with aftermath focus
            prompt = re.sub(r'.*?(moment of|scene of|showing|inside).*?(death|passing|suicide|end|bleeding|blood|mausoleum|monument).*?', 
                           'solemn scene of grief and loss, aftermath of significant event', prompt, flags=re.IGNORECASE)
        
        # For violence/conflict, focus on emotional impact
        if re.search(r'\b(violence|conflict|battle|war)\b', prompt, re.IGNORECASE):
            prompt = re.sub(r'.*?(showing|scene of|depicting).*?(violence|conflict|battle).*?',
                           'conquered halls, somber mood, aftermath of conflict', prompt, flags=re.IGNORECASE)
    
    # Extreme sanitization (attempt 4+): Drastically replace the entire scene description with safe alternative
    # BUT: If title/narration are already excluded, we should preserve the image_prompt details
    # and only remove problematic phrases, not replace the entire visual section
    if sanitize_attempt >= 4:
        # If title/narration were excluded, the image_prompt is all we have - preserve it but clean problematic phrases
        if exclude_title_narration:
            # Only remove problematic phrases from the image_prompt, don't replace it entirely
            # Be careful to preserve age information, locations, and context
            # Remove problematic multi-word phrases first (more specific, preserves context better)
            problematic_phrases_multiword = [
                r'\bbody\s+slack\b',  # "body slack" as a complete phrase
                r'\bbleeding\s+and\s+barely\s+conscious\b',  # full phrase
                r'\barmor[^.]*\s+blood\b',  # armor ... blood (flexible)
                r'\btunic[^.]*\s+blood\b',  # tunic ... blood
                r'\bsmeared\s+with\s+blood\b',  # smeared with blood
                r'\bkill\s+himself\b',  # kill himself
            ]
            for phrase in problematic_phrases_multiword:
                prompt = re.sub(phrase, '', prompt, flags=re.IGNORECASE)
            
            # Then remove single problematic words (but preserve "mausoleum" and "monument" as they're just locations)
            # Only remove words that are clearly problematic in safety context
            # NOTE: We preserve location words like "mausoleum", "monument", "body" (when not with "slack")
            # to maintain visual context and age information
            problematic_words = [
                r'\b(bleeding|blood)\b',  # blood-related (most problematic)
                r'\b(kill|suicide)\b',  # violence/suicide related
            ]
            for phrase in problematic_words:
                prompt = re.sub(phrase, '', prompt, flags=re.IGNORECASE)
        else:
            # Title/narration are still included, so we can replace the visual section if needed
            # Extract scene title and emotion if present for context
            title_match = re.search(r'Scene title:\s*([^.]+)', prompt, re.IGNORECASE)
            scene_title = title_match.group(1) if title_match else ""
            
            # Find the main visual description block (usually after "Visual details to emphasize:")
            visual_start = prompt.find("Visual details to emphasize:")
            if visual_start != -1:
                # Get everything from "Visual details to emphasize:" to end or next section
                visual_section = prompt[visual_start:]
                # If there's a scene that contains problematic content, replace it completely
                if re.search(r'\b(bleeding|blood|kill|suicide|death|mausoleum|monument|slack|body|armor.*?blood)\b', visual_section, re.IGNORECASE):
                    # Completely replace the visual section with a safe, emotional aftermath scene
                    safe_visual = (
                        "Visual details to emphasize:\n"
                        "Cinematic scene showing the aftermath of a significant historical moment: "
                        "interior of an ancient chamber with soft torchlight casting long shadows, "
                        "emotional weight conveyed through composition and lighting rather than explicit details, "
                        "figures shown in silhouette or from behind, focusing on architecture, atmosphere, and symbolic elements "
                        "that convey grief, loss, or historical significance without showing graphic content. "
                        "Somber, contemplative mood with dramatic chiaroscuro lighting. "
                        "Emphasize the emotional impact and historical weight of the moment through visual poetry, not explicit imagery. "
                        "16:9 cinematic"
                    )
                    prompt = prompt[:visual_start] + safe_visual
            
            # Also aggressively remove any remaining problematic phrases from the entire prompt
            problematic_phrases = [
                r'armor.*?blood',
                r'tunic.*?blood',
                r'smeared.*?blood',
                r'bleeding.*?conscious',
                r'body.*?slack',
                r'kill.*?himself',
                r'wound.*?finish',
            ]
            for phrase in problematic_phrases:
                prompt = re.sub(phrase, '', prompt, flags=re.IGNORECASE)
    
    # If we know the violation type, we can be more specific
    if violation_type == "self-harm":
        prompt = prompt.replace("suffering", "contemplation")
        prompt = prompt.replace("pain", "challenge")
        prompt = prompt.replace("struggle", "journey")
        prompt = prompt.replace("death", "legacy")
        prompt = prompt.replace("illness", "health challenges")
    
    # Add safety instruction based on sanitization level
    if sanitize_attempt >= 4:
        safety_instruction = " CRITICAL SAFETY: Safe, educational documentary content only. Use symbolic, abstract, or aftermath-focused representation. NO graphic imagery, NO explicit violence, NO blood, NO death scenes. Focus on architecture, atmosphere, emotions, and historical significance through visual poetry and composition. Emphasize emotional weight through lighting and composition, not explicit physical details."
    elif sanitize_attempt >= 3:
        safety_instruction = " Safe, appropriate, educational content. Focus on aftermath, emotional impact, and symbolic representation. Use abstract representation if needed. Avoid any graphic or disturbing imagery."
    elif sanitize_attempt == 2:
        safety_instruction = " Safe, appropriate, educational content. Focus on achievements, intellectual work, contemplation, and positive moments. Use symbolic or abstract representation if needed. Avoid any graphic or disturbing imagery."
    else:
        safety_instruction = " Safe, appropriate, educational content only. Focus on achievements, intellectual work, and positive moments. Avoid graphic or disturbing imagery."
    
    # Add explicit violation type avoidance if we know what violated
    violation_avoidance = ""
    if violation_type:
        # Map violation types to clearer descriptions
        violation_descriptions = {
            "self-harm": "self-harm, suicide, or self-injury",
            "violence": "violence, physical harm, or aggressive actions",
            "death": "death, dying, or explicit death scenes",
            "harmful": "harmful, dangerous, or potentially harmful content"
        }
        violation_desc = violation_descriptions.get(violation_type, violation_type)
        violation_avoidance = f" CRITICAL: TO AVOID {violation_desc.upper()}. Do not depict any {violation_desc}, themes, imagery, or references related to this content."
    
    # Append safety instruction and violation avoidance
    sanitized = prompt + safety_instruction + violation_avoidance
    
    return sanitized


def generate_image_for_scene(scene: dict, prev_scene: dict | None, global_block_override: str | None = None, sanitize_attempt: int = 0, violation_type: str = None) -> Path:
    """
    Generate an image for the given scene using OpenAI Images API.
    If config.save_assets is True, caches to generated_images/. Otherwise uses temp directory.
    
    Args:
        sanitize_attempt: Number of times we've tried to sanitize the prompt (0 = original, 1+ = sanitized)
    """
    if config.save_assets:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        img_path = IMAGES_DIR / f"scene_{scene['id']:02d}.png"
        if img_path.exists():
            print(f"[IMAGE] Scene {scene['id']}: using cached {img_path.name}")
            return img_path
    else:
        # Use temp directory
        img_path = Path(config.temp_dir) / f"scene_{scene['id']:02d}.png"

    # On attempts 4+, exclude title and narration (may contain problematic words)
    exclude_title_narration = (sanitize_attempt >= 4)
    if exclude_title_narration:
        print(f"[IMAGE] Scene {scene['id']}: excluding title and narration (using only image_prompt)")
    
    # Only include safety instructions after first attempt fails
    include_safety_instructions = (sanitize_attempt > 0)
    
    prompt = build_image_prompt(scene, prev_scene, global_block_override, exclude_title_narration=exclude_title_narration, include_safety_instructions=include_safety_instructions)
    
    # Sanitize prompt if this is a retry after safety violation
    if sanitize_attempt > 0:
        print(f"[IMAGE] Scene {scene['id']}: sanitizing prompt (attempt {sanitize_attempt})...")
        prompt = sanitize_prompt_for_safety(prompt, violation_type=violation_type, sanitize_attempt=sanitize_attempt, exclude_title_narration=exclude_title_narration)
        # Log the full sanitized prompt for debugging
        print(f"[IMAGE] Scene {scene['id']}: SANITIZED PROMPT (length={len(prompt)} chars):")
        print("-" * 80)
        print(prompt[:2000])  # Print first 2000 chars
        if len(prompt) > 2000:
            print(f"... (truncated, full length: {len(prompt)} chars)")
        print("-" * 80)
    
    print(f"[IMAGE] Scene {scene['id']}: generating image...")

    try:
        resp = client.images.generate(
            model=IMG_MODEL,
            prompt=prompt,
            size=config.image_size,  # vertical for shorts, landscape for main
            n=1,
            moderation="low",
        )

        b64_data = resp.data[0].b64_json
        img_bytes = base64.b64decode(b64_data)
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        if config.save_assets:
            print(f"[IMAGE] Scene {scene['id']}: saved {img_path.name}")
        else:
            print(f"[IMAGE] Scene {scene['id']}: generated (temp)")
        return img_path
    
    except Exception as e:
        # Check if this is a safety violation error
        error_str = str(e).lower()
        if 'safety' in error_str or 'moderation' in error_str or 'rejected' in error_str:
            # Extract violation type if available - check for common violation types
            violation_type = None
            if 'self-harm' in error_str or 'suicide' in error_str:
                violation_type = "self-harm"
            elif 'violence' in error_str or 'violent' in error_str:
                violation_type = "violence"
            elif 'death' in error_str or 'dead' in error_str or 'kill' in error_str:
                violation_type = "death"
            elif 'harmful' in error_str or 'harm' in error_str:
                violation_type = "harmful"
            
            # Re-raise with violation info so retry mechanism can handle it
            raise SafetyViolationError(f"Safety violation detected: {e}", violation_type=violation_type, original_prompt=prompt)
        else:
            # Re-raise other errors as-is
            raise


def generate_audio_for_scene(scene: dict) -> Path:
    """
    Generate TTS audio for the given scene's narration.
    Uses ElevenLabs or OpenAI based on TTS_PROVIDER setting.
    If config.save_assets is True, caches to generated_audio/. Otherwise uses temp directory.
    """
    if config.save_assets:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = AUDIO_DIR / f"scene_{scene['id']:02d}.mp3"
        if audio_path.exists():
            print(f"[AUDIO] Scene {scene['id']}: using cached {audio_path.name}")
            return audio_path
    else:
        # Use temp directory
        audio_path = Path(config.temp_dir) / f"scene_{scene['id']:02d}.mp3"

    text = scene["narration"]
    emotion = scene.get("emotion")
    if emotion:
        print(f"[AUDIO] Scene {scene['id']}: generating audio ({TTS_PROVIDER}) with emotion={emotion}...")
    else:
        print(f"[AUDIO] Scene {scene['id']}: generating audio ({TTS_PROVIDER})...")

    if TTS_PROVIDER == "elevenlabs" and elevenlabs_client:
        # ElevenLabs TTS
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id=ELEVENLABS_MODEL,
        )
        # Write the audio chunks to file
        with open(audio_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
    elif TTS_PROVIDER == "google" and google_tts_client:
        # Google Cloud TTS with SSML for natural pacing and emotion-based prosody
        from google.cloud import texttospeech
        
        # Check voice type
        is_chirp_voice = GOOGLE_VOICE_TYPE == "chirp"
        is_gemini_voice = GOOGLE_VOICE_TYPE == "gemini"
        
        if is_gemini_voice:
            # Gemini TTS uses new API with prompt parameter and model_name
            # Use BIOPIC_NARRATION_INSTRUCTIONS or HORROR_NARRATION_INSTRUCTIONS from env (set by --horror)
            if NARRATION_INSTRUCTIONS_FOR_TTS:
                prompt_text = NARRATION_INSTRUCTIONS_FOR_TTS
                print(f"[AUDIO] Scene {scene['id']}: using narration instructions from env for Gemini TTS")
            else:
                # Fall back to emotion-based prompt
                emotion_desc = f"with {emotion} emotion" if emotion else ""
                prompt_text = f"Read this text {emotion_desc}, matching the scene's emotional tone."
            
            # Gemini uses text input (not SSML) with prompt parameter
            synthesis_input = texttospeech.SynthesisInput(text=text, prompt=prompt_text)
            # synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Select Gemini voice based on female flag
            if USE_FEMALE_VOICE:
                gemini_voice_name = GOOGLE_GEMINI_FEMALE_SPEAKER if GOOGLE_GEMINI_FEMALE_SPEAKER else GOOGLE_TTS_VOICE
            else:
                gemini_voice_name = GOOGLE_GEMINI_MALE_SPEAKER
            
            # Gemini voices use model_name instead of just name
            voice = texttospeech.VoiceSelectionParams(
                language_code=GOOGLE_TTS_LANGUAGE,
                name=gemini_voice_name,  # Voice name from GOOGLE_GEMINI_MALE_SPEAKER or GOOGLE_GEMINI_FEMALE_SPEAKER
                model_name="gemini-2.5-pro-tts"
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
        elif is_chirp_voice:
            # Chirp voices don't support SSML or prosody - use plain text
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=GOOGLE_TTS_LANGUAGE,
                name=GOOGLE_TTS_VOICE,
            )
            
            # Chirp voices don't support pitch parameter
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
            )
        else:
            # Standard Google TTS with SSML for natural pauses and emotion-based prosody adjustments
            ssml_text = text_to_ssml(text, emotion=emotion)
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=GOOGLE_TTS_LANGUAGE,
                name=GOOGLE_TTS_VOICE,
            )
            
            audio_config_params = {
                "audio_encoding": texttospeech.AudioEncoding.MP3,
                "pitch": GOOGLE_TTS_PITCH,
            }
            audio_config = texttospeech.AudioConfig(**audio_config_params)
        
        response = google_tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        
        with open(audio_path, "wb") as f:
            f.write(response.audio_content)
    else:
        # OpenAI TTS (fallback)
        # Use BIOPIC_NARRATION_INSTRUCTIONS or HORROR_NARRATION_INSTRUCTIONS from env (set by --horror)
        if NARRATION_INSTRUCTIONS_FOR_TTS:
            tts_instructions = NARRATION_INSTRUCTIONS_FOR_TTS
            print(f"[AUDIO] Scene {scene['id']}: using narration instructions from env")
        else:
            # Fall back to default style prompt
            tts_instructions = OPENAI_TTS_STYLE_PROMPT
        
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            instructions=tts_instructions,
        ) as response:
            response.stream_to_file(str(audio_path))

    if config.save_assets:
        print(f"[AUDIO] Scene {scene['id']}: saved {audio_path.name}")
    else:
        print(f"[AUDIO] Scene {scene['id']}: generated (temp)")
    return audio_path


def make_static_clip_with_audio(image_path: Path, audio_clip: AudioFileClip):
    """
    Make a static full-frame clip for one scene.
    - Scale image to cover output resolution.
    - Center & crop.
    - Match duration to audio.
    - Add pause at the end (no audio, image continues).
    """
    audio_duration = audio_clip.duration
    pause_length = END_SCENE_PAUSE_LENGTH
    total_duration = audio_duration + pause_length
    output_res = config.output_resolution

    # Create image clip with total duration (audio + pause)
    clip = ImageClip(str(image_path)).with_duration(total_duration)
    base_w, base_h = clip.size

    # Scale so image fully covers the frame
    scale = max(
        output_res[0] / base_w,
        output_res[1] / base_h,
    )
    clip = clip.resized(scale)
    w, h = clip.size

    x_center, y_center = w / 2, h / 2

    clip = clip.cropped(
        x_center=x_center,
        y_center=y_center,
        width=output_res[0],
        height=output_res[1],
    )

    # Create silent audio clip for the pause
    # Use the same fps as the original audio
    fps = audio_clip.fps
    nchannels = audio_clip.nchannels if hasattr(audio_clip, 'nchannels') else 2
    
    # Create a function that returns silence (zeros)
    def make_silence(t):
        # Return array of zeros matching the number of channels
        if nchannels == 1:
            return [0.0]
        else:
            return [0.0, 0.0]
    
    silent_audio = AudioClip(make_silence, duration=pause_length, fps=fps)
    
    # Concatenate original audio with silent pause
    # Use concatenate_audioclips for audio clips (not concatenate_videoclips)
    extended_audio = concatenate_audioclips([audio_clip, silent_audio])

    return clip.with_audio(extended_audio).with_duration(total_duration)


def generate_room_tone(
    duration: float,
    sample_rate: int = 44100,
    gain: float = 0.008,          # start low; room tone should be subtle
    fade_in: float = 0.05,        # 50 ms
    fade_out: float = 0.05,       # 50 ms
    seed: int | None = None
) -> AudioClip:
    """
    Generate barely audible room tone (pink-ish noise).
    """

    num_samples = int(duration * sample_rate)
    if num_samples <= 0:
        return AudioClip(lambda t: np.array([0.0, 0.0], dtype=np.float32), duration=duration, fps=sample_rate)

    rng = np.random.default_rng(seed)

    # --- FFT-based pink noise approximation ---
    # Generate random complex spectrum and scale by 1/sqrt(f) (pink ~ 1/f power)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    spectrum = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))

    # Avoid division by zero at DC; keep DC very small (we'll remove mean anyway)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])  # ~1/sqrt(f) amplitude => ~1/f power

    spectrum *= scale
    noise = np.fft.irfft(spectrum, n=num_samples)

    # Remove DC / center
    noise = noise - np.mean(noise)

    # Normalize to a reasonable RMS (not peak=1), then apply gain
    rms = np.sqrt(np.mean(noise**2)) + 1e-9
    noise = (noise / rms) * gain

    # Tiny fades to prevent clicks when spliced
    env = np.ones(num_samples, dtype=np.float32)

    fi = int(min(fade_in, duration) * sample_rate)
    fo = int(min(fade_out, duration) * sample_rate)

    if fi > 0:
        x = np.linspace(0, 1, fi, endpoint=False)
        env[:fi] *= 0.5 - 0.5 * np.cos(np.pi * x)

    if fo > 0:
        x = np.linspace(0, 1, fo, endpoint=False)
        env[-fo:] *= 0.5 + 0.5 * np.cos(np.pi * x)

    tone = (noise * env).astype(np.float32)

    def make_audio(tt):
        if np.isscalar(tt):
            idx = int(tt * sample_rate)
            if 0 <= idx < len(tone):
                v = float(tone[idx])
                return np.array([v, v], dtype=np.float32)
            return np.array([0.0, 0.0], dtype=np.float32)
        else:
            indices = (tt * sample_rate).astype(int)
            indices = np.clip(indices, 0, len(tone) - 1)
            mono = tone[indices]
            return np.column_stack([mono, mono]).astype(np.float32)

    return AudioClip(make_audio, duration=duration, fps=sample_rate)


def generate_low_frequency_drone_with_transition(duration: float, frequency: float = 50.0, 
                                                  transition_type: str = "none", 
                                                  start_volume: float = 1.0,
                                                  end_volume: float = 1.0,
                                                  transition_duration: float = 5.0,
                                                  sample_rate: int = 44100) -> AudioClip:
    """
    Generate low-frequency drone with volume transitions.
    
    Args:
        duration: Total duration in seconds
        frequency: Frequency in Hz (default: 50, range: 30-80)
        transition_type: Type of transition - "fade_in", "fade_out", "swell", "shrink", "hold", "hard_cut", or "none"
        start_volume: Starting volume (0.0 to 1.0) - used for fade_in, swell, shrink, hold
        end_volume: Ending volume (0.0 to 1.0) - used for fade_out, swell, shrink
        transition_duration: Duration of transition in seconds (default: 5.0 for fade_in/fade_out, 3-8 for swell/shrink)
        sample_rate: Sample rate in Hz (default: 44100)
    
    Returns:
        AudioClip with low-frequency sine wave and volume transition
    """
    # Clamp frequency to 30-80 Hz range
    frequency = max(30.0, min(80.0, frequency))
    
    # Generate sine wave
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    
    # Apply volume envelope based on transition type
    if transition_type == "fade_in":
        # Fade in from 0 to end_volume over transition_duration (3-10 seconds)
        transition_samples = int(transition_duration * sample_rate)
        fade_envelope = np.linspace(0.0, end_volume, min(transition_samples, num_samples))
        if len(fade_envelope) < num_samples:
            fade_envelope = np.pad(fade_envelope, (0, num_samples - len(fade_envelope)), mode='constant', constant_values=end_volume)
        sine_wave = sine_wave * fade_envelope
    elif transition_type == "fade_out":
        # Fade out from start_volume to minimum (not zero - room tone remains)
        min_volume = 0.0
        transition_samples = int(transition_duration * sample_rate)
        fade_start = max(0, num_samples - transition_samples)
        fade_envelope = np.ones(num_samples) * start_volume
        fade_portion = np.linspace(start_volume, min_volume, num_samples - fade_start)
        fade_envelope[fade_start:] = fade_portion
        sine_wave = sine_wave * fade_envelope
    elif transition_type == "swell":
        # Increase volume from start_volume to end_volume over transition_duration (3-8 seconds)
        transition_samples = int(transition_duration * sample_rate)
        swell_envelope = np.ones(num_samples) * start_volume
        swell_portion = np.linspace(start_volume, end_volume, min(transition_samples, num_samples))
        if len(swell_portion) <= num_samples:
            swell_envelope[:len(swell_portion)] = swell_portion
            if len(swell_portion) < num_samples:
                swell_envelope[len(swell_portion):] = end_volume
        sine_wave = sine_wave * swell_envelope
    elif transition_type == "shrink":
        # Decrease volume from start_volume to end_volume over transition_duration (3-8 seconds)
        min_volume = 0.1  # Never shrink completely
        end_volume = max(min_volume, end_volume)
        transition_samples = int(transition_duration * sample_rate)
        shrink_envelope = np.ones(num_samples) * start_volume
        shrink_portion = np.linspace(start_volume, end_volume, min(transition_samples, num_samples))
        if len(shrink_portion) <= num_samples:
            shrink_envelope[:len(shrink_portion)] = shrink_portion
            if len(shrink_portion) < num_samples:
                shrink_envelope[len(shrink_portion):] = end_volume
        sine_wave = sine_wave * shrink_envelope
    elif transition_type == "hard_cut":
        # Instant cut to silence (or very low volume)
        sine_wave = np.zeros_like(sine_wave)
    elif transition_type == "hold":
        # Hold at start_volume throughout
        sine_wave = sine_wave * start_volume
    else:  # "none" or default
        # Constant volume
        sine_wave = sine_wave * start_volume
    
    # Create audio function for MoviePy
    # Return stereo (2 channels) for compatibility
    def make_audio(t):
        # t can be a scalar or array
        if np.isscalar(t):
            idx = int(t * sample_rate)
            if 0 <= idx < len(sine_wave):
                # Return stereo: [left, right] with same value
                return np.array([sine_wave[idx], sine_wave[idx]])
            return np.array([0.0, 0.0])
        else:
            # Array of times - return stereo array
            indices = (t * sample_rate).astype(int)
            indices = np.clip(indices, 0, len(sine_wave) - 1)
            mono_samples = sine_wave[indices]
            # Convert to stereo by duplicating the channel
            if mono_samples.ndim == 0:
                return np.array([mono_samples, mono_samples])
            else:
                return np.column_stack([mono_samples, mono_samples])
    
    return AudioClip(make_audio, duration=duration, fps=sample_rate)


def generate_low_frequency_drone(duration: float, frequency: float = 90.0, sample_rate: int = 44100) -> AudioClip:
    """
    Generate low-frequency drone (30-80 Hz).
    Feels like pressure, not sound - you should feel it more than hear it.
    
    Args:
        duration: Duration in seconds
        frequency: Frequency in Hz (default: 50, range: 30-80)
        sample_rate: Sample rate in Hz (default: 44100)
    
    Returns:
        AudioClip with low-frequency sine wave
    """
    # Clamp frequency to 30-80 Hz range
    frequency = max(60.0, min(120.0, frequency))
    
    # Generate sine wave
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    
    # Create audio function for MoviePy
    # MoviePy's AudioClip expects a function that takes a time array and returns audio samples
    # Return stereo (2 channels) for compatibility with CompositeAudioClip
    def make_audio(t):
        # t can be a scalar or array
        if np.isscalar(t):
            idx = int(t * sample_rate)
            if 0 <= idx < len(sine_wave):
                # Return stereo: [left, right] with same value
                return np.array([sine_wave[idx], sine_wave[idx]])
            return np.array([0.0, 0.0])
        else:
            # Array of times - return stereo array
            indices = (t * sample_rate).astype(int)
            indices = np.clip(indices, 0, len(sine_wave) - 1)
            mono_samples = sine_wave[indices]
            # Convert to stereo by duplicating the channel
            if mono_samples.ndim == 0:
                return np.array([mono_samples, mono_samples])
            else:
                return np.column_stack([mono_samples, mono_samples])
    
    return AudioClip(make_audio, duration=duration, fps=sample_rate)


def generate_detail_sounds(duration: float, sound_type: str = "random", sample_rate: int = 44100) -> AudioClip:
    """
    Generate occasional detail sounds (creaks, wind, hum, whispers).
    Randomly places 3-8 sounds throughout the duration.
    
    Args:
        duration: Duration in seconds
        sound_type: Type of sound ("creak", "wind", "hum", "whisper", "random")
        sample_rate: Sample rate in Hz (default: 44100)
    
    Returns:
        AudioClip with occasional detail sounds
    """
    HORROR_AUDIO_DIR = Path("horror_audio")
    
    # Try to load audio files first
    sound_files = {
        "creak": list((HORROR_AUDIO_DIR / "creaks").glob("*.mp3")) + list((HORROR_AUDIO_DIR / "creaks").glob("*.wav")),
        "wind": list((HORROR_AUDIO_DIR / "wind").glob("*.mp3")) + list((HORROR_AUDIO_DIR / "wind").glob("*.wav")),
        "hum": list((HORROR_AUDIO_DIR / "hum").glob("*.mp3")) + list((HORROR_AUDIO_DIR / "hum").glob("*.wav")),
        "whisper": list((HORROR_AUDIO_DIR / "whispers").glob("*.mp3")) + list((HORROR_AUDIO_DIR / "whispers").glob("*.wav")),
    }
    
    # Create silent audio clip (stereo)
    def make_silence(t):
        # Handle both scalar and array inputs
        # Return stereo (2 channels) for compatibility
        if np.isscalar(t):
            return np.array([0.0, 0.0])
        else:
            # Return stereo array
            return np.zeros((len(t), 2))
    
    result_audio = AudioClip(make_silence, duration=duration, fps=sample_rate)
    
    # Determine number of sounds (3-8)
    num_sounds = random.randint(3, 8)
    
    # Generate sound positions (avoid overlapping)
    min_gap = duration / (num_sounds + 1)
    positions = []
    for _ in range(num_sounds):
        pos = random.uniform(min_gap, duration - min_gap)
        # Ensure minimum gap between sounds
        if not positions or all(abs(pos - p) >= min_gap for p in positions):
            positions.append(pos)
    
    # Sort positions
    positions.sort()
    
    # Add sounds at each position
    for pos in positions:
        # Select sound type
        if sound_type == "random":
            selected_type = random.choice(["creak", "wind", "hum", "whisper"])
        else:
            selected_type = sound_type
        
        # Try to use audio file if available
        if sound_files[selected_type]:
            try:
                sound_file = random.choice(sound_files[selected_type])
                sound_clip = AudioFileClip(str(sound_file))
                # Place sound at position
                sound_clip = sound_clip.with_start(pos)
                # Composite with existing audio
                result_audio = CompositeAudioClip([result_audio, sound_clip])
                continue
            except Exception as e:
                print(f"[HORROR AUDIO] Failed to load {sound_file}: {e}, using programmatic fallback")
        
        # Fallback: generate programmatic sound
        sound_duration = random.uniform(0.5, 2.0)  # 0.5-2 seconds
        
        if selected_type == "creak":
            # Creak: short burst of noise with frequency sweep
            num_samples = int(sound_duration * sample_rate)
            t = np.linspace(0, sound_duration, num_samples, False)
            # Frequency sweep from high to low
            freq_sweep = np.linspace(800, 200, num_samples)
            creak = np.sin(2 * np.pi * freq_sweep * t) * np.exp(-t * 2)  # Decay envelope
            creak = creak / np.max(np.abs(creak)) if np.max(np.abs(creak)) > 0 else creak
            sound_data = creak
        elif selected_type == "wind":
            # Wind: low-frequency noise with modulation
            num_samples = int(sound_duration * sample_rate)
            t = np.linspace(0, sound_duration, num_samples, False)
            wind = np.random.randn(num_samples) * 0.3
            # Low-pass filter effect (simple)
            for i in range(1, len(wind)):
                wind[i] = 0.7 * wind[i-1] + 0.3 * wind[i]
            wind = wind / np.max(np.abs(wind)) if np.max(np.abs(wind)) > 0 else wind
            sound_data = wind
        elif selected_type == "hum":
            # Hum: low-frequency sine wave with slight variation
            num_samples = int(sound_duration * sample_rate)
            t = np.linspace(0, sound_duration, num_samples, False)
            base_freq = 60.0 + random.uniform(-10, 10)
            hum = np.sin(2 * np.pi * base_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
            hum = hum / np.max(np.abs(hum)) if np.max(np.abs(hum)) > 0 else hum
            sound_data = hum
        else:  # whisper
            # Whisper: high-frequency filtered noise
            num_samples = int(sound_duration * sample_rate)
            t = np.linspace(0, sound_duration, num_samples, False)
            whisper = np.random.randn(num_samples) * 0.2
            # High-pass filter effect (simple)
            for i in range(1, len(whisper)):
                whisper[i] = 0.3 * whisper[i-1] + 0.7 * whisper[i]
            whisper = whisper / np.max(np.abs(whisper)) if np.max(np.abs(whisper)) > 0 else whisper
            sound_data = whisper
        
        # Create audio clip for this sound (stereo)
        def make_sound_audio(t_local):
            # t_local is relative to the clip start (0 to sound_duration)
            # t can be a scalar or array
            # Return stereo (2 channels) for compatibility
            if np.isscalar(t_local):
                idx = int(t_local * sample_rate)
                if 0 <= idx < len(sound_data):
                    # Return stereo: [left, right] with same value
                    return np.array([sound_data[idx], sound_data[idx]])
                return np.array([0.0, 0.0])
            else:
                # Array of times - return stereo array
                indices = (t_local * sample_rate).astype(int)
                indices = np.clip(indices, 0, len(sound_data) - 1)
                mono_samples = sound_data[indices]
                # Convert to stereo by duplicating the channel
                if mono_samples.ndim == 0:
                    return np.array([mono_samples, mono_samples])
                else:
                    return np.column_stack([mono_samples, mono_samples])
        
        sound_clip = AudioClip(make_sound_audio, duration=sound_duration, fps=sample_rate)
        sound_clip = sound_clip.with_start(pos)
        result_audio = CompositeAudioClip([result_audio, sound_clip])
    
    return result_audio


def generate_drone_with_scene_transitions(scenes: list[dict], scene_audio_clips: list[AudioFileClip], 
                                         base_drone_volume_db: float = None,
                                         max_drone_volume_db: float = None) -> AudioClip:
    """
    Generate low-frequency drone with transitions based on scene drone_change values.
    Tracks drone state across scenes to ensure smooth transitions.
    
    Args:
        scenes: List of scene dictionaries with drone_change field
        scene_audio_clips: List of audio clips for each scene (used to get scene durations)
        base_drone_volume_db: Base drone volume in dB (default: from DRONE_BASE env var or -20)
        max_drone_volume_db: Maximum drone volume in dB (default: from DRONE_MAX env var or -15, used to clip swell)
    
    Returns:
        AudioClip with drone that transitions based on scene instructions
    """
    # Use environment variable defaults if not provided
    if base_drone_volume_db is None:
        base_drone_volume_db = DRONE_BASE_VOLUME_DB
    if max_drone_volume_db is None:
        max_drone_volume_db = DRONE_MAX_VOLUME_DB
    
    # Convert dB to linear volume multiplier
    def db_to_linear(db):
        return 10 ** (db / 20.0)
    
    base_volume_linear = db_to_linear(base_drone_volume_db)
    max_volume_linear = db_to_linear(max_drone_volume_db)
    
    # Calculate total duration for continuous generation (prevents pops)
    total_duration = sum(clip.duration + END_SCENE_PAUSE_LENGTH for clip in scene_audio_clips)
    sample_rate = 44100
    num_samples = int(total_duration * sample_rate)
    
    # Generate continuous sine wave for entire duration
    t = np.arange(num_samples) / sample_rate
    sine_wave = np.sin(2 * np.pi * 50.0 * t)
    
    # Build volume envelope for entire duration
    volume_envelope = np.zeros(num_samples)
    current_time = 0.0
    current_drone_volume = 0.0  # Start with no drone
    
    # Process each scene to build the volume envelope
    for i, (scene, audio_clip) in enumerate(zip(scenes, scene_audio_clips)):
        scene_id = scene.get('id', i + 1)
        scene_duration = audio_clip.duration + END_SCENE_PAUSE_LENGTH  # Include pause
        drone_change = scene.get('drone_change', 'none')

        if current_drone_volume == 0.0 and drone_change == "swell":
            drone_change = "fade_in"
        
        # Determine transition parameters based on drone_change
        if drone_change == "fade_in":
            # Fade in from 0 to base volume (3-10 seconds)
            start_volume = 0.0
            end_volume = min(max_volume_linear, base_volume_linear)
            transition_duration = min(10.0, max(3.0, scene_duration * 0.3))  # 3-10 seconds, adaptive to scene length
            transition_type = "fade_in"
            current_drone_volume = end_volume  # Update state
        elif drone_change == "hold":
            # Hold at current volume (no transition)
            start_volume = current_drone_volume
            end_volume = start_volume
            transition_duration = 0.0  # No transition
            transition_type = "hold"
            current_drone_volume = start_volume  # Maintain state
        elif drone_change == "swell":
            # Increase from current to higher volume (3-8 seconds)
            # Clip by max volume, not base volume
            start_volume = current_drone_volume
            end_volume = min(max_volume_linear, start_volume * 1.4)  # Increase by up to 40%, but clip at max
            transition_duration = min(8.0, max(3.0, scene_duration * 0.3))  # 3-8 seconds
            transition_type = "swell"
            current_drone_volume = end_volume  # Update state
        elif drone_change == "shrink":
            # Decrease from current volume (but not to zero) (3-8 seconds)
            min_volume = base_volume_linear * 0.1  # Minimum 10% of base
            start_volume = current_drone_volume
            end_volume = max(min_volume, start_volume * 0.6)  # Decrease to 60% of start
            transition_duration = min(8.0, max(3.0, scene_duration * 0.3))  # 3-8 seconds
            transition_type = "shrink"
            current_drone_volume = end_volume  # Update state
        elif drone_change == "fade_out":
            # Fade out from current to minimum (3-10 seconds, but never completely)
            min_volume = 0.0 
            start_volume = current_drone_volume
            end_volume = min_volume
            transition_duration = min(10.0, max(3.0, scene_duration * 0.3))  # 3-10 seconds
            transition_type = "fade_out"
            current_drone_volume = end_volume  # Update state
        elif drone_change == "hard_cut":
            # Instant cut to silence
            start_volume = current_drone_volume
            end_volume = 0.0
            transition_duration = 0.0  # Instant
            transition_type = "hard_cut"
            current_drone_volume = 0.0  # Update state
        else:  # "none" or unknown
            # No change - hold at current volume (or default to base if no previous state)
            start_volume = current_drone_volume
            end_volume = start_volume
            transition_duration = 0.0
            transition_type = "hold"
            current_drone_volume = start_volume  # Maintain state
        
        transition_duration = min(transition_duration, scene_duration)
        
        # Calculate sample indices for this scene
        start_sample = int(current_time * sample_rate)
        end_sample = int((current_time + scene_duration) * sample_rate)
        transition_samples = int(transition_duration * sample_rate)
        
        # Apply volume envelope for this scene (continuous, no cuts = no pops)
        if transition_duration > 0 and transition_samples > 0:
            # Transition portion
            transition_end_sample = min(start_sample + transition_samples, end_sample)
            transition_indices = np.arange(start_sample, transition_end_sample)
            if len(transition_indices) > 0:
                transition_volumes = np.linspace(start_volume, end_volume, len(transition_indices))
                volume_envelope[transition_indices] = transition_volumes
            
            # Rest of scene at end_volume
            if transition_end_sample < end_sample:
                volume_envelope[transition_end_sample:end_sample] = end_volume
        else:
            # No transition - constant volume
            volume_envelope[start_sample:end_sample] = end_volume
        
        print(f"[HORROR AUDIO] Scene {scene_id}: drone_change={drone_change}, volume={start_volume:.3f}->{end_volume:.3f}")
        
        current_time += scene_duration
    
    # Apply volume envelope to sine wave (ensure it's a contiguous array)
    sine_wave = (sine_wave * volume_envelope).copy()
    
    # Create audio function for MoviePy (stereo)
    def make_audio(t):
        if np.isscalar(t):
            idx = int(t * sample_rate)
            if 0 <= idx < len(sine_wave):
                return np.array([sine_wave[idx], sine_wave[idx]])
            else:
                return np.array([0.0, 0.0])
        else:
            indices = (t * sample_rate).astype(int)
            indices = np.clip(indices, 0, len(sine_wave) - 1)
            samples = sine_wave[indices]
            # Return stereo: [left, right] with same values
            return np.column_stack([samples, samples])
    
    # Create AudioClip (continuous, no concatenation = no pops)
    drone = AudioClip(make_audio, duration=total_duration, fps=sample_rate)
    
    return drone


def apply_volume_to_audioclip(clip: AudioClip, volume_factor: float) -> AudioClip:
    """
    Apply volume to an AudioClip by wrapping its audio function.
    
    Args:
        clip: The AudioClip to modify
        volume_factor: Volume multiplier (1.0 = no change, 0.5 = half volume, etc.)
    
    Returns:
        New AudioClip with volume applied
    """
    # Access the original audio function
    # AudioClip stores the function - we'll use get_frame which should work
    original_get_frame = clip.get_frame
    
    # Create a new function that wraps the original and applies volume
    def volume_adjusted_audio(t):
        audio = original_get_frame(t)
        # Handle both scalar and array returns
        return audio * volume_factor
    
    # Create new AudioClip with volume-adjusted function
    return AudioClip(volume_adjusted_audio, duration=clip.duration, fps=clip.fps)


def mix_horror_background_audio(narration_audio: AudioFileClip, duration: float, 
                                scenes: list[dict] = None,
                                scene_audio_clips: list[AudioFileClip] = None,
                                room_tone_volume: float = None, 
                                drone_volume: float = None,
                                max_drone_volume: float = None,
                                detail_volume: float = -30.0,
                                environment: str = None,
                                env_audio_volume: float = None) -> CompositeAudioClip:
    """
    Mix horror background audio layers with narration.
    
    Args:
        narration_audio: The main narration audio clip
        duration: Total duration in seconds
        scenes: List of scene dictionaries (optional, for drone transitions)
        scene_audio_clips: List of audio clips for each scene (optional, for calculating scene durations)
        room_tone_volume: Room tone volume in dB (default: from ROOM_SOUND_BASE env var or -35)
        drone_volume: Base drone volume in dB (default: from DRONE_BASE env var or -20)
        max_drone_volume: Maximum drone volume in dB (default: from DRONE_MAX env var or -15, used to clip swell)
        detail_volume: Detail sounds volume in dB (default: -30)
        environment: Environment type for ambient audio: "blizzard", "snow", "forest", "rain", or "indoors" (optional)
        env_audio_volume: Environment audio volume in dB (default: from ENV_AUDIO_VOLUME env var or -28)
    
    Returns:
        CompositeAudioClip with all layers mixed
    """
    # Use environment variable defaults if not provided
    if room_tone_volume is None:
        room_tone_volume = ROOM_SOUND_BASE_VOLUME_DB
    if drone_volume is None:
        drone_volume = DRONE_BASE_VOLUME_DB
    if max_drone_volume is None:
        max_drone_volume = DRONE_MAX_VOLUME_DB
    if env_audio_volume is None:
        env_audio_volume = ENV_AUDIO_VOLUME_DB
    
    print(f"[HORROR AUDIO] Generating background layers (duration: {duration:.2f}s)...")
    
    # Generate all background layers
    print("[HORROR AUDIO] Generating room tone...")
    room_tone = generate_room_tone(duration)
    
    # Generate drone with transitions if scenes are provided
    if scenes and scene_audio_clips and len(scenes) == len(scene_audio_clips):
        print("[HORROR AUDIO] Generating low-frequency drone with scene-based transitions...")
        drone = generate_drone_with_scene_transitions(scenes, scene_audio_clips, 
                                                     base_drone_volume_db=drone_volume,
                                                     max_drone_volume_db=max_drone_volume)
    else:
        print("[HORROR AUDIO] Generating low-frequency drone (constant volume)...")
        drone = generate_low_frequency_drone(duration, frequency=50.0)
    
    print("[HORROR AUDIO] Generating detail sounds...")
    detail_sounds = generate_detail_sounds(duration, sound_type="random")
    
    # Load environment audio if specified
    env_audio = None
    env_audio_original_path = None  # Store original path for FFmpeg trimming if needed later
    if environment:
        env_audio_path = None
        env_map = {
            "blizzard": ENV_BLIZZARD_AUDIO,
            "snow": ENV_SNOW_AUDIO,
            "forest": ENV_FOREST_AUDIO,
            "rain": ENV_RAIN_AUDIO,
            "indoors": ENV_INDOORS_AUDIO,
            "jungle": ENV_JUNGLE_AUDIO
        }
        env_audio_path = env_map.get(environment.lower())
        env_audio_original_path = env_audio_path  # Store for later use
        
        if env_audio_path and Path(env_audio_path).exists():
            try:
                print(f"[HORROR AUDIO] Loading environment audio: {environment} from {env_audio_path}")
                env_audio = AudioFileClip(str(env_audio_path))
                
                # Loop the environment audio to match duration
                # NOTE: This initial loop may not work reliably with MoviePy concatenation
                # We'll check again after volume/stereo processing and use FFmpeg if needed
                if env_audio.duration < duration:
                    print(f"[HORROR AUDIO] Initial loop attempt: {env_audio.duration:.3f}s < {duration:.3f}s")
                    loops_needed = int(np.ceil(duration / env_audio.duration))
                    print(f"[HORROR AUDIO] Attempting to loop {loops_needed} times using MoviePy concatenation")
                    env_audio_clips = [env_audio] * loops_needed
                    env_audio = concatenate_audioclips(env_audio_clips)
                    print(f"[HORROR AUDIO] After initial loop: {env_audio.duration:.3f}s (target: {duration:.3f}s)")
                    # After concatenation, try subclip (may not work on CompositeAudioClip)
                    if env_audio.duration > duration:
                        try:
                            env_audio = env_audio.subclip(0, duration)
                        except (AttributeError, TypeError):
                            # If subclip doesn't work on CompositeAudioClip, write to file and trim with FFmpeg
                            try:
                                import subprocess
                                temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                temp_audio_path = temp_audio_file.name
                                temp_audio_file.close()
                                
                                # Write the concatenated audio to temp file
                                try:
                                    env_audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                                    # Now trim the written file to exact duration
                                    temp_trimmed_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                    temp_trimmed_path = temp_trimmed_file.name
                                    temp_trimmed_file.close()
                                    
                                    result = subprocess.run([
                                        'ffmpeg', '-y', '-i', temp_audio_path,
                                        '-t', str(duration),
                                        '-acodec', 'copy',
                                        temp_trimmed_path
                                    ], check=True, capture_output=True, text=True)
                                    
                                    try:
                                        env_audio.close()
                                    except:
                                        pass
                                    env_audio = AudioFileClip(temp_trimmed_path)
                                except Exception as write_error:
                                    # If write_audiofile fails, close the concatenated clip and use original file approach
                                    try:
                                        env_audio.close()
                                    except:
                                        pass
                                    # Reload original and use FFmpeg to create properly looped and trimmed version
                                    env_audio = AudioFileClip(str(env_audio_path))
                                    # Calculate how many loops we need
                                    loops_needed = int(np.ceil(duration / env_audio.duration))
                                    # Use FFmpeg aloop filter to loop and trim in one go (more reliable than -stream_loop)
                                    result = subprocess.run([
                                        'ffmpeg', '-y',
                                        '-i', str(env_audio_path),
                                        '-filter_complex', f'aloop=loop={loops_needed - 1}:size=2e+09',
                                        '-t', str(duration),
                                        '-acodec', 'pcm_s16le',  # Re-encode needed for aloop filter
                                        temp_audio_path
                                    ], check=True, capture_output=True, text=True)
                                    env_audio = AudioFileClip(temp_audio_path)
                            except Exception as e:
                                print(f"[HORROR AUDIO] Warning: Could not trim concatenated environment audio: {e}")
                                print(f"[HORROR AUDIO] Using full audio duration ({env_audio.duration:.2f}s) instead of requested {duration:.2f}s")
                elif env_audio.duration > duration:
                    # For AudioFileClip, try subclip first, but if it doesn't work, use FFmpeg directly
                    try:
                        env_audio = env_audio.subclip(0, duration)
                    except (AttributeError, TypeError):
                        # If subclip doesn't work, use FFmpeg directly to trim the file
                        # This avoids recursion issues with get_frame wrappers
                        try:
                            import subprocess
                            temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            temp_audio_path = temp_audio_file.name
                            temp_audio_file.close()
                            
                            # Use FFmpeg to trim the audio file (decode to PCM for .wav; -acodec copy fails for OGG/MP3 etc.)
                            result = subprocess.run([
                                'ffmpeg', '-y', '-i', str(env_audio_path),
                                '-t', str(duration),
                                '-acodec', 'pcm_s16le',
                                '-ar', '44100', '-ac', '2',
                                temp_audio_path
                            ], check=True, capture_output=True, text=True)
                            
                            # Close the original clip and reload the trimmed version
                            try:
                                env_audio.close()
                            except:
                                pass
                            env_audio = AudioFileClip(temp_audio_path)
                            
                            # Note: Temp file will be cleaned up when clip is closed or process ends
                        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
                            print(f"[HORROR AUDIO] Warning: Could not trim environment audio using FFmpeg: {e}")
                            print(f"[HORROR AUDIO] Using full audio duration ({env_audio.duration:.2f}s) instead of requested {duration:.2f}s")
                            # Will be handled later in the duration matching section
                
                print(f"[HORROR AUDIO] Environment audio loaded: {env_audio.duration:.2f}s")
            except Exception as e:
                print(f"[HORROR AUDIO] Warning: Failed to load environment audio '{env_audio_path}': {e}")
                env_audio = None
        elif env_audio_path:
            print(f"[HORROR AUDIO] Warning: Environment audio file not found: {env_audio_path}")
        else:
            print(f"[HORROR AUDIO] No audio file path configured for environment: {environment}")
    
    # Convert dB to linear volume multiplier
    def db_to_linear(db):
        return 10 ** (db / 20.0)
    
    # Apply volume adjustments
    room_tone = apply_volume_to_audioclip(room_tone, db_to_linear(room_tone_volume))
    # Drone volume is already applied in generate_drone_with_scene_transitions if scenes are provided
    if not (scenes and scene_audio_clips and len(scenes) == len(scene_audio_clips)):
        drone = apply_volume_to_audioclip(drone, db_to_linear(drone_volume))
    detail_sounds = apply_volume_to_audioclip(detail_sounds, db_to_linear(detail_volume))
    
    # Apply volume to environment audio if loaded
    if env_audio:
        env_audio = apply_volume_to_audioclip(env_audio, db_to_linear(env_audio_volume))
    
    # Ensure narration audio matches duration - NEVER trim narration, it's the source of truth
    # If narration is longer than expected duration, use narration duration as the actual duration
    if narration_audio.duration < duration:
        # Extend narration with silence if needed
        silence_duration = duration - narration_audio.duration
        def make_silence(t):
            # Handle both scalar and array inputs
            if np.isscalar(t):
                return 0.0
            else:
                return np.zeros_like(t)
        silence = AudioClip(make_silence, duration=silence_duration, fps=narration_audio.fps)
        narration_audio = concatenate_audioclips([narration_audio, silence])
    elif narration_audio.duration > duration:
        # Narration is longer than expected - use narration duration as the actual duration
        # This means the video duration calculation was slightly off, but narration is correct
        # We must NEVER trim narration - it's the source of truth
        narration_duration = narration_audio.duration
        print(f"[HORROR AUDIO] Narration audio ({narration_duration:.3f}s) is longer than expected duration ({duration:.3f}s)")
        print(f"[HORROR AUDIO] Using narration duration as actual duration - extending other audio layers to match")
        
        # Extend all other audio layers to match narration duration
        extension_needed = narration_duration - duration
        
        # Extend room_tone
        if room_tone.duration < narration_duration:
            silence_duration = narration_duration - room_tone.duration
            def make_silence(t):
                if np.isscalar(t):
                    return np.array([0.0, 0.0])
                else:
                    return np.zeros((len(t), 2))
            silence = AudioClip(make_silence, duration=silence_duration, fps=room_tone.fps)
            room_tone = concatenate_audioclips([room_tone, silence])
            print(f"[HORROR AUDIO] Extended room_tone from {duration:.3f}s to {narration_duration:.3f}s")
        
        # Extend drone
        if drone.duration < narration_duration:
            silence_duration = narration_duration - drone.duration
            def make_silence(t):
                if np.isscalar(t):
                    return np.array([0.0, 0.0])
                else:
                    return np.zeros((len(t), 2))
            silence = AudioClip(make_silence, duration=silence_duration, fps=drone.fps)
            drone = concatenate_audioclips([drone, silence])
            print(f"[HORROR AUDIO] Extended drone from {duration:.3f}s to {narration_duration:.3f}s")
        
        # Extend detail_sounds
        if detail_sounds and detail_sounds.duration < narration_duration:
            silence_duration = narration_duration - detail_sounds.duration
            def make_silence(t):
                if np.isscalar(t):
                    return np.array([0.0, 0.0])
                else:
                    return np.zeros((len(t), 2))
            silence = AudioClip(make_silence, duration=silence_duration, fps=detail_sounds.fps)
            detail_sounds = concatenate_audioclips([detail_sounds, silence])
            print(f"[HORROR AUDIO] Extended detail_sounds from {duration:.3f}s to {narration_duration:.3f}s")
        
        # Update duration to narration duration (will be used for env_audio check below)
        duration = narration_duration
    
    # All generated clips (room_tone, drone, detail_sounds) are now stereo (2 channels)
    # But narration_audio might be mono, so ensure it's stereo if needed
    def ensure_stereo(clip):
        """Ensure audio clip is stereo (2 channels) - safety check for narration_audio."""
        try:
            sample = clip.get_frame(0.0)
            # Check if it's mono (scalar or single channel)
            is_mono = (np.isscalar(sample) or 
                      (hasattr(sample, 'shape') and (len(sample.shape) == 0 or 
                       (len(sample.shape) == 1 and sample.shape[0] != 2))))
            if is_mono:
                # Convert mono to stereo
                def stereo_func(t):
                    mono = clip.get_frame(t)
                    if np.isscalar(mono):
                        return np.array([mono, mono])
                    elif mono.ndim == 0:
                        return np.array([mono, mono])
                    elif mono.ndim == 1:
                        if len(mono) == 1:
                            return np.array([mono[0], mono[0]])
                        else:
                            # Already stereo or multi-channel
                            return mono
                    else:
                        # Multi-dimensional - assume compatible
                        return mono
                return AudioClip(stereo_func, duration=clip.duration, fps=clip.fps)
        except:
            # If we can't determine, assume it's already compatible
            pass
        return clip
    
    # Ensure narration_audio is stereo (generated clips are already stereo)
    narration_audio = ensure_stereo(narration_audio)
    
    # Ensure environment audio is stereo and matches duration
    if env_audio:
        env_audio = ensure_stereo(env_audio)
        # Check duration after stereo conversion (duration might have changed slightly)
        # CRITICAL: Always check if we need to loop, even if it was looped earlier
        # The initial loop might not have worked, or duration might have changed
        print(f"[HORROR AUDIO] Environment audio duration check: {env_audio.duration:.3f}s vs target {duration:.3f}s")
        if env_audio.duration < duration:  # No tolerance - if shorter, we MUST loop
            print(f"[HORROR AUDIO] Environment audio is shorter than target duration - looping required")
            # Loop the environment audio to match duration (don't extend with silence)
            # Use FFmpeg to loop from original file for better reliability
            if env_audio_original_path and Path(env_audio_original_path).exists():
                try:
                    import subprocess
                    # Get original file duration to calculate loops needed
                    original_clip = AudioFileClip(str(env_audio_original_path))
                    original_duration = original_clip.duration
                    original_clip.close()
                    
                    loops_needed = int(np.ceil(duration / original_duration))
                    print(f"[HORROR AUDIO] Looping environment audio: {loops_needed} loops needed (original: {original_duration:.2f}s, target: {duration:.2f}s)")
                    
                    # Use FFmpeg aloop filter to loop audio (more reliable than -stream_loop for audio files)
                    temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_audio_path = temp_audio_file.name
                    temp_audio_file.close()
                    
                    # aloop filter: loop=LOOPS means the input will play (LOOPS+1) times total
                    # So if we need 5 loops total, we use loop=4
                    # size=2e+09 sets a large buffer size to ensure the entire file fits
                    result = subprocess.run([
                        'ffmpeg', '-y',
                        '-i', str(env_audio_original_path),
                        '-filter_complex', f'aloop=loop={loops_needed - 1}:size=2e+09',
                        '-t', str(duration),
                        '-acodec', 'pcm_s16le',  # Use PCM for better compatibility
                        temp_audio_path
                    ], check=True, capture_output=True, text=True)
                    
                    # Close old clip and load looped version
                    try:
                        env_audio.close()
                    except:
                        pass
                    
                    env_audio = AudioFileClip(temp_audio_path)
                    env_audio = ensure_stereo(env_audio)
                    env_audio = apply_volume_to_audioclip(env_audio, db_to_linear(env_audio_volume))
                    # Verify the looped audio has the correct duration (within 0.1s tolerance)
                    if abs(env_audio.duration - duration) > 0.1:
                        print(f"[HORROR AUDIO] Warning: Looped audio duration ({env_audio.duration:.2f}s) doesn't match target ({duration:.2f}s)")
                        # If duration is still too short, we need to loop again or extend
                        if env_audio.duration < duration:
                            print(f"[HORROR AUDIO] Audio is still too short, attempting additional loop...")
                            # This shouldn't happen with aloop filter, but handle it gracefully
                    print(f"[HORROR AUDIO] Environment audio looped successfully: {env_audio.duration:.2f}s (target: {duration:.2f}s)")
                except Exception as e:
                    print(f"[HORROR AUDIO] Warning: Could not loop environment audio with FFmpeg: {e}")
                    print(f"[HORROR AUDIO] Attempting MoviePy concatenation fallback...")
                    # Fallback to MoviePy concatenation
                    try:
                        loops_needed = int(np.ceil(duration / env_audio.duration))
                        env_audio_clips = [env_audio] * loops_needed
                        env_audio = concatenate_audioclips(env_audio_clips)
                        # Trim to exact duration if needed
                        if env_audio.duration > duration + 0.1:
                            try:
                                env_audio = env_audio.subclip(0, duration)
                            except (AttributeError, TypeError):
                                # If subclip fails, write and trim with FFmpeg
                                temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                temp_audio_path = temp_audio_file.name
                                temp_audio_file.close()
                                
                                try:
                                    env_audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                                    temp_trimmed_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                    temp_trimmed_path = temp_trimmed_file.name
                                    temp_trimmed_file.close()
                                    
                                    result = subprocess.run([
                                        'ffmpeg', '-y', '-i', temp_audio_path,
                                        '-t', str(duration),
                                        '-acodec', 'copy',
                                        temp_trimmed_path
                                    ], check=True, capture_output=True, text=True)
                                    
                                    try:
                                        env_audio.close()
                                    except:
                                        pass
                                    env_audio = AudioFileClip(temp_trimmed_path)
                                    env_audio = ensure_stereo(env_audio)
                                    env_audio = apply_volume_to_audioclip(env_audio, db_to_linear(env_audio_volume))
                                except Exception as write_error:
                                    print(f"[HORROR AUDIO] Warning: Could not trim looped audio: {write_error}")
                    except Exception as fallback_error:
                        print(f"[HORROR AUDIO] Warning: MoviePy concatenation fallback also failed: {fallback_error}")
            else:
                print(f"[HORROR AUDIO] Warning: Could not loop environment audio (original path not available), using current duration")
        elif env_audio.duration > duration:
            # Try subclip first (should work on processed clips from concatenation/ensure_stereo)
            try:
                env_audio = env_audio.subclip(0, duration)
            except (AttributeError, TypeError):
                # If subclip doesn't work, try FFmpeg using the original file path we stored
                try:
                    import subprocess
                    if env_audio_original_path and Path(env_audio_original_path).exists():
                        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        temp_audio_path = temp_audio_file.name
                        temp_audio_file.close()
                        
                        result = subprocess.run([
                            'ffmpeg', '-y', '-i', str(env_audio_original_path),
                            '-t', str(duration),
                            '-acodec', 'pcm_s16le',
                            '-ar', '44100', '-ac', '2',
                            temp_audio_path
                        ], check=True, capture_output=True, text=True)
                        
                        try:
                            env_audio.close()
                        except:
                            pass
                        env_audio = AudioFileClip(temp_audio_path)
                        # Re-apply stereo conversion and volume after reloading
                        env_audio = ensure_stereo(env_audio)
                        env_audio = apply_volume_to_audioclip(env_audio, db_to_linear(env_audio_volume))
                    else:
                        print(f"[HORROR AUDIO] Warning: Could not trim environment audio (original path not available), using full duration")
                except Exception as e:
                    print(f"[HORROR AUDIO] Warning: Could not trim environment audio: {e}, using full duration")
    
    # Ensure drone duration matches total duration (use narration duration as source of truth)
    if drone.duration < duration:
        # Extend drone with silence (maintain last volume level)
        silence_duration = duration - drone.duration
        last_volume = drone.get_frame(drone.duration - 0.1) if drone.duration > 0.1 else np.array([0.0, 0.0])
        # Get the average volume from the last sample
        if hasattr(last_volume, '__len__') and len(last_volume) > 0:
            avg_volume = np.mean(np.abs(last_volume))
        else:
            avg_volume = 0.0
        
        def make_silence(t):
            # Return silence (or maintain last volume if needed)
            if np.isscalar(t):
                return np.array([0.0, 0.0])
            else:
                return np.zeros((len(t), 2))
        silence = AudioClip(make_silence, duration=silence_duration, fps=drone.fps)
        drone = concatenate_audioclips([drone, silence])
    elif drone.duration > duration:
        # Trim drone to match duration (narration is source of truth, so drone should match it)
        try:
            drone = drone.subclip(0, duration)
        except (AttributeError, TypeError):
            # If subclip doesn't work on AudioClip, write to temp file and trim with FFmpeg
            try:
                import subprocess
                temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_audio_path = temp_audio_file.name
                temp_audio_file.close()
                
                # Write the AudioClip to a temporary file
                try:
                    drone.write_audiofile(temp_audio_path, verbose=False, logger=None)
                    # Now trim the written file to exact duration
                    temp_trimmed_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_trimmed_path = temp_trimmed_file.name
                    temp_trimmed_file.close()
                    
                    result = subprocess.run([
                        'ffmpeg', '-y', '-i', temp_audio_path,
                        '-t', str(duration),
                        '-acodec', 'copy',
                        temp_trimmed_path
                    ], check=True, capture_output=True, text=True)
                    
                    try:
                        drone.close()
                    except:
                        pass
                    drone = AudioFileClip(temp_trimmed_path)
                except Exception as write_error:
                    print(f"[HORROR AUDIO] Warning: Could not write drone audio to file for trimming: {write_error}")
                    print(f"[HORROR AUDIO] Using full drone duration ({drone.duration:.2f}s) instead of requested {duration:.2f}s")
            except Exception as e:
                print(f"[HORROR AUDIO] Warning: Could not trim drone audio: {e}")
                print(f"[HORROR AUDIO] Using full drone duration ({drone.duration:.2f}s) instead of requested {duration:.2f}s")
    
    # Mix all layers
    print("[HORROR AUDIO] Mixing all audio layers...")
    
    # Debug: Check audio levels before mixing (sample multiple points to avoid zero crossings)
    try:
        # Sample multiple points across a short window for better RMS estimation
        # This avoids issues with sine waves where single-point sampling can hit zero crossings
        sample_window = 0.1  # Sample 0.1 seconds of audio
        mid_start = duration * 0.3
        mid_end = duration * 0.7
        num_samples = max(10, int(sample_window * 44100))  # At least 10 samples
        sample_times = np.linspace(mid_start, mid_end, num_samples)
        
        def calc_rms_from_samples(clip, times):
            """Calculate RMS from multiple sample points."""
            samples = []
            for t in times:
                if 0 <= t < clip.duration:
                    try:
                        sample = clip.get_frame(t)
                        if np.isscalar(sample):
                            samples.append(abs(sample))
                        else:
                            sample_array = np.asarray(sample)
                            samples.extend(np.abs(sample_array).flatten())
                    except:
                        pass
            if len(samples) == 0:
                return 0.0
            return np.sqrt(np.mean(np.array(samples) ** 2))
        
        room_rms = calc_rms_from_samples(room_tone, sample_times)
        drone_rms = calc_rms_from_samples(drone, sample_times)
        narration_rms = calc_rms_from_samples(narration_audio, sample_times)
        
        rms_msg = f"[HORROR AUDIO] Audio levels (RMS from {len(sample_times)} samples): room_tone={room_rms:.6f}, drone={drone_rms:.6f}, narration={narration_rms:.6f}"
        if env_audio:
            env_rms = calc_rms_from_samples(env_audio, sample_times)
            rms_msg += f", environment={env_rms:.6f}"
        print(rms_msg)
    except Exception as e:
        print(f"[HORROR AUDIO] Warning: Could not check audio levels: {e}")
    
    # Build list of audio clips to mix
    audio_clips = [
        narration_audio,
        room_tone,
        drone,
        # detail_sounds
    ]
    
    # Add environment audio if loaded
    if env_audio:
        audio_clips.append(env_audio)
        print(f"[HORROR AUDIO] Environment audio added to mix: {environment}")
    
    mixed_audio = CompositeAudioClip(audio_clips)
    
    return mixed_audio


def retry_call(name: str, func, *args, max_attempts: int = 3, base_delay: float = 2.0, **kwargs):
    """
    Generic retry wrapper for network/API calls.
    Retries on any Exception up to max_attempts, with exponential backoff.
    """
    attempt = 1
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt >= max_attempts:
                print(f"[RETRY][{name}] Failed after {attempt} attempts: {e}")
                raise

            delay = base_delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
            print(f"[RETRY][{name}] Attempt {attempt} failed: {e}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            attempt += 1


def generate_image_for_scene_with_retry(scene: dict, prev_scene: dict | None, global_block_override: str | None = None) -> Path:
    """
    Generate image with retry logic that handles safety violations by sanitizing prompts.
    """
    attempt = 1
    max_attempts = 5  # More attempts for safety violations - increased to 5
    base_delay = 2.0
    violation_type = None  # Store violation type from first SafetyViolationError
    
    while True:
        try:
            sanitize_attempt = attempt - 1  # First attempt is 0 (no sanitization), subsequent attempts sanitize
            return generate_image_for_scene(scene, prev_scene, global_block_override, sanitize_attempt=sanitize_attempt, violation_type=violation_type)
        except SafetyViolationError as e:
            if attempt >= max_attempts:
                print(f"[RETRY][image] Scene {scene['id']}: Failed after {attempt} attempts with safety violations")
                print(f"[RETRY][image] Original prompt: {e.original_prompt[:200]}...")
                raise Exception(f"Image generation failed due to safety violations after {attempt} attempts. Scene {scene['id']} may need manual prompt adjustment.")
            
            # Capture violation_type from first safety violation
            if e.violation_type and violation_type is None:
                violation_type = e.violation_type
            
            print(f"[RETRY][image] Scene {scene['id']}: Safety violation detected (attempt {attempt}/{max_attempts})")
            if violation_type:
                print(f"[RETRY][image] Violation type: {violation_type}")
            print(f"[RETRY][image] Sanitizing prompt and retrying...")
            
            delay = base_delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
            time.sleep(delay)
            attempt += 1
        except Exception as e:
            if attempt >= max_attempts:
                print(f"[RETRY][image] Scene {scene['id']}: Failed after {attempt} attempts: {e}")
                raise
            
            delay = base_delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
            print(f"[RETRY][image] Scene {scene['id']}: Attempt {attempt} failed: {e}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            attempt += 1


def generate_audio_for_scene_with_retry(scene: dict) -> Path:
    return retry_call(
        "audio",
        generate_audio_for_scene,
        scene,
        max_attempts=3,
        base_delay=2.0,
    )


def build_video(scenes_path: str, out_video_path: str | None = None, save_assets: bool = False, is_short: bool = False, scene_id: int | None = None, audio_only: bool = False,
                is_horror: bool = False, horror_bg_enabled: bool = True, room_tone_volume: float = None, 
                drone_volume: float = None, detail_volume: float = -30.0):
    """
    Build a video from scenes JSON file.
    
    Uses individual scene audio files for PERFECT synchronization:
    - Each image is displayed for exactly the duration of its audio
    - Audio clips are concatenated directly (no crossfades to avoid sync issues)
    
    Args:
        save_assets: If True, save images/audio to permanent directories. 
                     If False (default), use temp files that are cleaned up.
        is_short: If True, use vertical 9:16 format for YouTube Shorts.
                  If False (default), use landscape 16:9 format.
        horror_bg_enabled: If True, add horror background audio for horror videos (default: True)
        room_tone_volume: Room tone volume in dB (default: from ROOM_SOUND_BASE env var or -35)
        drone_volume: Drone volume in dB (default: from DRONE_BASE env var or -20)
        detail_volume: Detail sounds volume in dB (default: -30)
    """
    config.save_assets = save_assets
    config.is_vertical = is_short

    # Set which narration instructions to use for TTS based on horror mode (--horror)
    global NARRATION_INSTRUCTIONS_FOR_TTS
    NARRATION_INSTRUCTIONS_FOR_TTS = (HORROR_NARRATION_INSTRUCTIONS if is_horror else BIOPIC_NARRATION_INSTRUCTIONS) or ""
    if NARRATION_INSTRUCTIONS_FOR_TTS and (GOOGLE_VOICE_TYPE == "gemini" or TTS_PROVIDER == "openai"):
        print(f"[TTS] Using {'horror' if is_horror else 'biopic'} narration instructions from env")

    if config.is_vertical:
        print(f"[FORMAT] Vertical 9:16 (YouTube Short)")
    else:
        print(f"[FORMAT] Landscape 16:9 (Main video)")
    
    # Set output directory based on video type
    from pathlib import Path
    if is_short:
        output_dir = Path("finished_shorts")
    else:
        output_dir = Path("finished_videos")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If output path is just a filename, put it in the appropriate directory
    # Skip this in audio-only mode since we don't need a video output path
    if out_video_path is not None:
        out_path = Path(out_video_path)
        if not out_path.is_absolute() and str(out_path.parent) == ".":
            # Just a filename, put it in the appropriate directory
            out_video_path = str(output_dir / out_path.name)
        elif not out_path.is_absolute() and str(out_path.parent) not in ["finished_shorts", "finished_videos"]:
            # Relative path but not in the right directory, move it
            out_video_path = str(output_dir / out_path.name)
    
    # Use environment variable defaults if not provided
    if room_tone_volume is None:
        room_tone_volume = ROOM_SOUND_BASE_VOLUME_DB
    if drone_volume is None:
        drone_volume = DRONE_BASE_VOLUME_DB
    
    # Set up temp directory if not saving assets
    if not config.save_assets:
        config.temp_dir = tempfile.mkdtemp(prefix="video_build_")
        print(f"[TEMP] Using temporary directory: {config.temp_dir}")
    
    try:
        _build_video_impl(scenes_path, out_video_path, scene_id=scene_id, audio_only=audio_only,
                          is_horror=is_horror, horror_bg_enabled=horror_bg_enabled, room_tone_volume=room_tone_volume,
                          drone_volume=drone_volume, detail_volume=detail_volume)
    finally:
        # Clean up temp directory
        if not config.save_assets and config.temp_dir and os.path.exists(config.temp_dir):
            print(f"[TEMP] Cleaning up temporary files...")
            try:
                # Try to remove the directory
                shutil.rmtree(config.temp_dir)
            except PermissionError as e:
                # Files might still be in use - try again after a short delay
                import time
                time.sleep(0.5)
                try:
                    shutil.rmtree(config.temp_dir)
                except PermissionError:
                    # If still failing, log warning but don't crash
                    print(f"[WARNING] Could not clean up temp directory {config.temp_dir}: files may still be in use")
            except Exception as e:
                # Other errors - log but don't crash
                print(f"[WARNING] Error cleaning up temp directory {config.temp_dir}: {e}")


def _build_video_impl(scenes_path: str, out_video_path: str | None = None, scene_id: int | None = None, audio_only: bool = False,
                     is_horror: bool = False, horror_bg_enabled: bool = True, room_tone_volume: float = None, 
                     drone_volume: float = None, detail_volume: float = -30.0):
    """Internal implementation of build_video."""
    # Use environment variable defaults if not provided
    if room_tone_volume is None:
        room_tone_volume = ROOM_SOUND_BASE_VOLUME_DB
    if drone_volume is None:
        drone_volume = DRONE_BASE_VOLUME_DB
    
    scenes, metadata = load_scenes(scenes_path)
    
    # Use the is_horror flag passed from command line
    if is_horror:
        print("[HORROR] Horror video mode enabled")
    
    # Horror disclaimer: first scene (fixed image + env/room noise, no narration)
    disclaimer_image_path = None
    disclaimer_duration = None
    if is_horror:
        disclaimer_image_path = get_horror_disclaimer_image_path(config.is_vertical)
        if disclaimer_image_path is not None:
            print(f"[HORROR] Disclaimer: using {disclaimer_image_path.name} as first scene ({HORROR_DISCLAIMER_DURATION}s)")
    
    # If scene_id is specified, filter to only that scene
    test_mode_prev_scene = None
    if scene_id is not None:
        original_scenes = scenes
        scenes = [s for s in scenes if s.get('id') == scene_id]
        if not scenes:
            available_ids = [s.get('id') for s in original_scenes]
            raise ValueError(f"Scene ID {scene_id} not found. Available scene IDs: {available_ids}")
        print(f"[TEST MODE] Generating only scene {scene_id} out of {len(original_scenes)} total scenes")
        # Find the previous scene for image continuity context (but don't generate it)
        scene_idx = next((i for i, s in enumerate(original_scenes) if s.get('id') == scene_id), None)
        if scene_idx is not None and scene_idx > 0:
            test_mode_prev_scene = original_scenes[scene_idx - 1]
            print(f"[TEST MODE] Using previous scene {test_mode_prev_scene.get('id')} for image continuity context (will not generate audio/video for it)")
    
    num_scenes = len(scenes)
    
    # Extract global_block from metadata if available
    global_block = None
    if metadata and "global_block" in metadata:
        global_block = metadata["global_block"]
        print(f"[METADATA] Using global_block from script metadata")
    else:
        print(f"[METADATA] Using default global_block")
    
    # Previous-scene references for image continuity
    if test_mode_prev_scene is not None:
        # In test mode with a specific scene_id, use the actual previous scene for context
        prev_scenes: list[dict | None] = [test_mode_prev_scene]
    else:
        prev_scenes: list[dict | None] = [None] + scenes[:-1]

    image_paths: list[Path | None] = [None] * num_scenes
    scene_audio_clips: list[AudioFileClip | None] = [None] * num_scenes

    # In audio-only mode, use stock image for all scenes
    if audio_only:
        stock_image_path = Path(FIXED_IMAGES_DIR, "stock_image.png")
        if not stock_image_path.exists():
            raise FileNotFoundError(f"Stock image not found: {stock_image_path}")
        print(f"[AUDIO ONLY MODE] Using stock image for all {num_scenes} scenes...")
        image_paths = [stock_image_path] * num_scenes
    else:
        print(f"[SCENES] Starting asset generation for {num_scenes} scenes...")

    def image_job(idx: int):
        scene = scenes[idx]
        prev_scene = prev_scenes[idx]
        print(f"[IMAGE] Scene {scene['id']}: generating...")
        img_path = generate_image_for_scene_with_retry(scene, prev_scene, global_block)
        return idx, img_path

    def audio_job(idx: int):
        scene = scenes[idx]
        audio_path = generate_audio_for_scene_with_retry(scene)
        audio_clip = AudioFileClip(str(audio_path))
        print(f"[AUDIO] Scene {scene['id']}: duration={audio_clip.duration:.3f}s")
        return idx, audio_clip

    # --- PARALLEL PHASE: generate images and audio ---
    if not audio_only:
        print("\n[PARALLEL] Generating images...")
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(image_job, i) for i in range(num_scenes)]
                for fut in as_completed(futures):
                    idx, img_path = fut.result()
                    image_paths[idx] = img_path
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Canceling... (press Ctrl+C again to force quit)")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    print("\n[PARALLEL] Generating audio...")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(audio_job, i) for i in range(num_scenes)]
            for fut in as_completed(futures):
                idx, audio_clip = fut.result()
                scene_audio_clips[idx] = audio_clip
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Canceling... (press Ctrl+C again to force quit)")
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    # Verify all assets were generated
    missing_imgs = [i for i, p in enumerate(image_paths) if p is None]
    if missing_imgs:
        raise RuntimeError(f"Missing image paths for indices: {missing_imgs}")
    missing_audio = [i for i, c in enumerate(scene_audio_clips) if c is None]
    if missing_audio:
        raise RuntimeError(f"Missing audio clips for indices: {missing_audio}")

    total_audio_duration = sum(clip.duration for clip in scene_audio_clips)
    print(f"\n[AUDIO] Total duration: {total_audio_duration:.3f}s ({total_audio_duration/60:.1f} min)")

    # --- Build video clips (each image synced perfectly to its audio) ---
    print("\n[VIDEO] Assembling clips...")
    
    # Parallelize clip creation for better performance
    def create_clip(i):
        scene = scenes[i]
        img_path = image_paths[i]
        scene_audio = scene_audio_clips[i]
        print(f"[CLIP {i}] Scene {scene['id']}: {scene_audio.duration:.3f}s")
        return i, make_static_clip_with_audio(img_path, scene_audio)
    
    clips = [None] * num_scenes
    # Use ThreadPoolExecutor to create clips in parallel (this helps with image processing)
    with ThreadPoolExecutor(max_workers=min(4, num_scenes)) as executor:
        futures = {executor.submit(create_clip, i): i for i in range(num_scenes)}
        for future in as_completed(futures):
            i, clip = future.result()
            clips[i] = clip

    # Prepend horror disclaimer clip (fixed image + 3s env/room noise, no narration)
    if disclaimer_image_path is not None:
        def make_silence(t):
            if np.isscalar(t):
                return np.array([0.0, 0.0])
            return np.zeros((len(t), 2))
        silence_3s = AudioClip(make_silence, duration=HORROR_DISCLAIMER_DURATION, fps=44100)
        environment = metadata.get("environment") if metadata else None
        disclaimer_audio = mix_horror_background_audio(
            silence_3s,
            HORROR_DISCLAIMER_DURATION,
            scenes=None,
            scene_audio_clips=None,
            room_tone_volume=room_tone_volume,
            drone_volume=drone_volume,
            max_drone_volume=None,
            detail_volume=detail_volume,
            environment=environment,
            env_audio_volume=None,
        )
        disclaimer_clip = make_static_clip_with_audio(disclaimer_image_path, disclaimer_audio)
        disclaimer_duration = disclaimer_clip.duration
        clips = [disclaimer_clip] + list(clips)
        print(f"[HORROR] Disclaimer clip added: {disclaimer_duration:.2f}s")

    # Concatenate all clips using method="chain" for better performance with static clips
    # "chain" is faster than "compose" when clips have the same size/dimensions
    print("\n[VIDEO] Concatenating clips...")
    try:
        # Try "chain" method first (faster for clips with same dimensions)
        final = concatenate_videoclips(clips, method="chain")
    except Exception:
        # Fall back to "compose" if "chain" fails
        print("[VIDEO] Using compose method instead of chain...")
        final = concatenate_videoclips(clips, method="compose")

    print(f"[VIDEO] Final duration: {final.duration:.3f}s ({final.duration/60:.1f} min)")

    # Apply horror background audio if this is a horror video
    if is_horror and horror_bg_enabled:
        print(f"\n[HORROR AUDIO] Adding background audio layers...")
        try:
            # Get environment from metadata if available
            environment = None
            if metadata:
                environment = metadata.get("environment")
            
            # If disclaimer was prepended, mix horror only on the story part (disclaimer already has env/room noise)
            if disclaimer_duration is not None:
                # MoviePy 2.x uses subclipped(); older versions use subclip()
                subclip_fn = getattr(final.audio, "subclipped", getattr(final.audio, "subclip", None))
                if subclip_fn is None:
                    raise RuntimeError("Audio clip has no subclip/subclipped method")
                disclaimer_part = subclip_fn(0, disclaimer_duration)
                story_part = subclip_fn(disclaimer_duration, final.duration)
                story_duration = final.duration - disclaimer_duration
                mixed_story = mix_horror_background_audio(
                    story_part,
                    story_duration,
                    scenes=scenes,
                    scene_audio_clips=scene_audio_clips,
                    room_tone_volume=room_tone_volume,
                    drone_volume=drone_volume,
                    max_drone_volume=None,
                    detail_volume=detail_volume,
                    environment=environment,
                    env_audio_volume=None,
                )
                mixed_audio = concatenate_audioclips([disclaimer_part, mixed_story])
            else:
                narration_audio = final.audio
                mixed_audio = mix_horror_background_audio(
                    narration_audio,
                    final.duration,
                    scenes=scenes,
                    scene_audio_clips=scene_audio_clips,
                    room_tone_volume=room_tone_volume,
                    drone_volume=drone_volume,
                    max_drone_volume=None,
                    detail_volume=detail_volume,
                    environment=environment,
                    env_audio_volume=None,
                )
            
            # Replace video audio with mixed audio
            final = final.with_audio(mixed_audio)
            print("[HORROR AUDIO] Background audio layers added successfully")
        except Exception as e:
            print(f"[WARNING] Failed to add horror background audio: {e}")
            print("[WARNING] Continuing with narration audio only...")
            import traceback
            traceback.print_exc()

    # Write final video with optimized encoding settings for speed
    if out_video_path is None:
        print(f"\n[WARNING] No output path provided, skipping video encoding")
    else:
        if audio_only:
            print(f"\n[AUDIO ONLY] Encoding video with stock images (audio files generated)...")
        else:
            print(f"\n[VIDEO] Encoding video (this may take a while)...")
        
        # Use faster preset and more threads for better performance
        # "fast" preset is ~2-3x faster than "medium" with minimal quality loss
        # Increase threads based on available CPU cores
        max_threads = os.cpu_count() or 8  # Use all available CPU cores
        threads = min(max_threads, 16)  # Cap at 16 to avoid overhead
        
        final.write_videofile(
            out_video_path,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="fast",  # Changed from "medium" to "fast" (~2-3x faster with minimal quality loss)
            threads=threads,  # Use more CPU cores for faster encoding
            bitrate=None,  # Let codec choose optimal bitrate
            ffmpeg_params=[
                "-movflags", "+faststart",  # Optimize for web streaming
                "-pix_fmt", "yuv420p",  # Ensure compatibility
            ],
        )
        print("[VIDEO] Done!")


# ------------- ENTRY POINT -------------

def find_shorts_for_script(script_path: str) -> list[Path]:
    """Find all short JSON files associated with a main script."""
    script_path = Path(script_path)
    base_name = script_path.stem.replace("_script", "")
    
    shorts_dir = Path("shorts_scripts")
    if not shorts_dir.exists():
        return []
    
    # Look for shorts matching pattern: {base_name}_short{N}.json
    shorts = []
    for i in range(1, 20):  # Support up to 20 shorts
        short_path = shorts_dir / f"{base_name}_short{i}.json"
        if short_path.exists():
            shorts.append(short_path)
        else:
            break  # Stop at first missing number
    
    return shorts


def find_all_shorts() -> list[Path]:
    """Find all JSON files in the shorts directory."""
    shorts_dir = Path("shorts_scripts")
    if not shorts_dir.exists():
        return []
    
    # Get all JSON files, sorted by name
    shorts = sorted(shorts_dir.glob("*.json"))
    return shorts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build video from scenes JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build video (temp files cleaned up after)
  python build_video.py scenes.json output.mp4

  # Build video and keep generated images/audio
  python build_video.py scenes.json output.mp4 --save-assets

  # Build main video + all associated shorts
  python build_video.py einstein_script.json einstein.mp4 --with-shorts

  # Build all shorts in the shorts_scripts/ directory
  python build_video.py --all-shorts

  # Generate only a specific scene for testing (e.g., scene 25)
  python build_video.py scenes.json test_scene_25.mp4 --scene-id 25 --save-assets

  # Generate only audio files (skip images and video) - useful for testing TTS models
  python build_video.py scenes.json --audio-only --save-assets
        """
    )
    
    parser.add_argument("scenes_file", nargs="?", help="Path to scenes JSON file")
    parser.add_argument("output_file", nargs="?", help="Output video file path (e.g., output.mp4)")
    parser.add_argument("--save-assets", action="store_true",
                        help="Save generated images and audio files (default: use temp files)")
    parser.add_argument("--with-shorts", action="store_true",
                        help="Also build all associated YouTube Shorts after main video")
    parser.add_argument("--all-shorts", action="store_true",
                        help="Build all JSON files in the shorts_scripts/ directory")
    parser.add_argument("--scene-id", type=int, metavar="ID",
                        help="Generate only a specific scene by ID (for testing). Creates a short video with just that scene.")
    parser.add_argument("--audio-only", action="store_true",
                        help="Generate only audio files (skip images and video assembly). Useful for testing TTS models.")
    parser.add_argument("--female", action="store_true",
                        help="Use female voice (GOOGLE_TTS_FEMALE_VOICE from .env) instead of default GOOGLE_TTS_VOICE")
    parser.add_argument("--horror-bg-room-tone-volume", type=float, default=None, metavar="DB",
                        help=f"Room tone volume in dB for horror videos (default: {ROOM_SOUND_BASE_VOLUME_DB} from ROOM_SOUND_BASE env var)")
    parser.add_argument("--horror-bg-drone-volume", type=float, default=None, metavar="DB",
                        help=f"Low-frequency drone volume in dB for horror videos (default: {DRONE_BASE_VOLUME_DB} from DRONE_BASE env var)")
    parser.add_argument("--horror-bg-detail-volume", type=float, default=-30.0, metavar="DB",
                        help="Detail sounds volume in dB for horror videos (default: -30)")
    parser.add_argument("--horror", action="store_true",
                        help="Enable horror video mode (adds horror background audio)")
    parser.add_argument("--horror-bg-disable", action="store_true",
                        help="Disable horror background audio even for horror videos")
    
    args = parser.parse_args()
    
    # Set default values from environment variables if not provided via command line
    if args.horror_bg_room_tone_volume is None:
        args.horror_bg_room_tone_volume = ROOM_SOUND_BASE_VOLUME_DB
    if args.horror_bg_drone_volume is None:
        args.horror_bg_drone_volume = DRONE_BASE_VOLUME_DB
    
    # Validate: need either positional args or --all-shorts
    # For --audio-only, output_file is optional (will use default if not provided)
    if not args.all_shorts and not args.scenes_file:
        parser.error("scenes_file is required unless using --all-shorts")
    
    # Override voice if --female is specified
    global USE_FEMALE_VOICE, GOOGLE_TTS_VOICE
    if args.female and TTS_PROVIDER == "google":
        USE_FEMALE_VOICE = True
        if GOOGLE_VOICE_TYPE == "gemini":
            # For Gemini, use the female speaker env var
            female_voice = GOOGLE_GEMINI_FEMALE_SPEAKER
            if female_voice:
                GOOGLE_TTS_VOICE = female_voice
                print(f"[TTS] Using Gemini female voice: {GOOGLE_TTS_VOICE}")
            else:
                print("[WARNING] --female specified for Gemini but GOOGLE_GEMINI_FEMALE_SPEAKER not set in .env. Using default voice.")
        else:
            # For standard Google TTS, use GOOGLE_TTS_FEMALE_VOICE
            female_voice = os.getenv("GOOGLE_TTS_FEMALE_VOICE")
            if female_voice:
                GOOGLE_TTS_VOICE = female_voice
                print(f"[TTS] Using female voice: {GOOGLE_TTS_VOICE}")
            else:
                print("[WARNING] --female specified but GOOGLE_TTS_FEMALE_VOICE not set in .env. Using default voice.")
    else:
        USE_FEMALE_VOICE = False
    
    return args


if __name__ == "__main__":
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set the OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    if args.save_assets:
        print("[MODE] Saving assets to generated_images/ and generated_audio/")
    else:
        print("[MODE] Using temporary files (will be cleaned up after)")

    # Handle --all-shorts mode
    if args.all_shorts:
        shorts = find_all_shorts()
        
        if not shorts:
            print("\n[ERROR] No JSON files found in shorts_scripts/ directory")
            sys.exit(1)
        
        print(f"\n[ALL SHORTS] Found {len(shorts)} short(s) to build")
        
        for i, short_path in enumerate(shorts, 1):
            # Generate output name from JSON filename: einstein_short1.json → einstein_short1.mp4
            # Output goes to finished_shorts/ directory
            short_output = f"finished_shorts/{short_path.stem}.mp4"
            
            print(f"\n{'='*60}")
            print(f"[SHORT {i}/{len(shorts)}] Building: {short_output}")
            print(f"{'='*60}")
            
            build_video(str(short_path), short_output, save_assets=args.save_assets, is_short=True,
                       is_horror=args.horror, horror_bg_enabled=not args.horror_bg_disable,
                       room_tone_volume=args.horror_bg_room_tone_volume,
                       drone_volume=args.horror_bg_drone_volume,
                       detail_volume=args.horror_bg_detail_volume)
        
        print(f"\n{'='*60}")
        print("[COMPLETE] All shorts built!")
        print(f"{'='*60}")
        sys.exit(0)

    # Detect if input is a short based on path
    # Check if file is in shorts_scripts directory or has "short" in the name
    scenes_path = Path(args.scenes_file)
    is_short = ("short" in args.scenes_file.lower() or 
                "shorts_scripts" in str(scenes_path.parent).lower())
    
    # Generate default output path if not provided (for audio-only mode)
    if args.audio_only and not args.output_file:
        # Generate output name from JSON filename: einstein_script.json → einstein_audio_only.mp4
        base_name = scenes_path.stem.replace("_script", "")
        if is_short:
            output_file = f"finished_shorts/{base_name}_audio_only.mp4"
        else:
            output_file = f"finished_videos/{base_name}_audio_only.mp4"
        args.output_file = output_file
    
    # Build main video
    print(f"\n{'='*60}")
    if args.audio_only:
        print(f"[AUDIO ONLY MODE] Generating audio only: {args.scenes_file}")
        print(f"[AUDIO ONLY MODE] Output: {args.output_file}")
    elif args.scene_id is not None:
        print(f"[TEST MODE] Building only scene {args.scene_id}: {args.output_file}")
    elif is_short:
        print(f"[SHORT] Building: {args.output_file}")
    else:
        print(f"[MAIN VIDEO] Building: {args.output_file}")
    print(f"{'='*60}")
    build_video(args.scenes_file, args.output_file, save_assets=args.save_assets, is_short=is_short, 
               scene_id=args.scene_id, audio_only=args.audio_only,
               is_horror=args.horror, horror_bg_enabled=not args.horror_bg_disable,
               room_tone_volume=args.horror_bg_room_tone_volume,
               drone_volume=args.horror_bg_drone_volume,
               detail_volume=args.horror_bg_detail_volume)
    
    # Build shorts if requested
    if args.with_shorts:
        shorts = find_shorts_for_script(args.scenes_file)
        
        if not shorts:
            print("\n[SHORTS] No short files found")
        else:
            print(f"\n[SHORTS] Found {len(shorts)} short(s) to build")
            
            for i, short_path in enumerate(shorts, 1):
                # Generate output name: einstein.mp4 → einstein_short1.mp4
                # Output goes to finished_shorts/ directory
                base_name = Path(args.output_file).stem
                short_output = f"finished_shorts/{base_name}_short{i}.mp4"
                
                print(f"\n{'='*60}")
                print(f"[SHORT {i}/{len(shorts)}] Building: {short_output}")
                print(f"{'='*60}")
                
                # Shorts are always vertical
                build_video(str(short_path), short_output, save_assets=args.save_assets, is_short=True,
                           is_horror=args.horror, horror_bg_enabled=not args.horror_bg_disable,
                           room_tone_volume=args.horror_bg_room_tone_volume,
                           drone_volume=args.horror_bg_drone_volume,
                           detail_volume=args.horror_bg_detail_volume)
        
        print(f"\n{'='*60}")
        print("[COMPLETE] All videos built!")
        print(f"{'='*60}")