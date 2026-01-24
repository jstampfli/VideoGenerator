import os
import sys
import json
import base64
import time
import random
import argparse
import tempfile
import shutil
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
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "cedar"

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
GOOGLE_VOICE_TYPE = os.getenv("GOOGLE_VOICE_TYPE", "").lower()  # "chirp" or empty - chirp voices don't support SSML/prosody

# Scene pause settings
END_SCENE_PAUSE_LENGTH = float(os.getenv("END_SCENE_PAUSE_LENGTH", "0.15"))  # Pause length in seconds at end of each scene (default: 150ms)


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
    
    # If using Chirp voice, skip SSML and return plain text
    if TTS_PROVIDER == "google" and GOOGLE_VOICE_TYPE == "chirp":
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
            n=1
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
        
        # Check if using Chirp voice (doesn't support SSML/prosody)
        is_chirp_voice = GOOGLE_VOICE_TYPE == "chirp"
        
        if is_chirp_voice:
            # Chirp voices don't support SSML or prosody - use plain text
            synthesis_input = texttospeech.SynthesisInput(text=text)
        else:
            # Convert text to SSML for natural pauses and emphasis, with emotion-based prosody adjustments
            ssml_text = text_to_ssml(text, emotion=emotion)
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=GOOGLE_TTS_LANGUAGE,
            name=GOOGLE_TTS_VOICE,
        )
        
        # Chirp voices don't support pitch parameter
        audio_config_params = {
            "audio_encoding": texttospeech.AudioEncoding.MP3,
            # "speaking_rate": GOOGLE_TTS_SPEAKING_RATE,
            # "effects_profile_id": ["headphone-class-device"],
        }
        
        # Only add pitch if not using Chirp voice
        if not is_chirp_voice:
            audio_config_params["pitch"] = GOOGLE_TTS_PITCH
        
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
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            instructions=OPENAI_TTS_STYLE_PROMPT,
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


def build_video(scenes_path: str, out_video_path: str, save_assets: bool = False, is_short: bool = False, scene_id: int | None = None, audio_only: bool = False):
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
    """
    config.save_assets = save_assets
    config.is_vertical = is_short
    
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
    out_path = Path(out_video_path)
    if not out_path.is_absolute() and str(out_path.parent) == ".":
        # Just a filename, put it in the appropriate directory
        out_video_path = str(output_dir / out_path.name)
    elif not out_path.is_absolute() and str(out_path.parent) not in ["finished_shorts", "finished_videos"]:
        # Relative path but not in the right directory, move it
        out_video_path = str(output_dir / out_path.name)
    
    # Set up temp directory if not saving assets
    if not config.save_assets:
        config.temp_dir = tempfile.mkdtemp(prefix="video_build_")
        print(f"[TEMP] Using temporary directory: {config.temp_dir}")
    
    try:
        _build_video_impl(scenes_path, out_video_path, scene_id=scene_id, audio_only=audio_only)
    finally:
        # Clean up temp directory
        if not config.save_assets and config.temp_dir and os.path.exists(config.temp_dir):
            print(f"[TEMP] Cleaning up temporary files...")
            shutil.rmtree(config.temp_dir)


def _build_video_impl(scenes_path: str, out_video_path: str, scene_id: int | None = None, audio_only: bool = False):
    """Internal implementation of build_video."""
    scenes, metadata = load_scenes(scenes_path)
    
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
        stock_image_path = Path("stock_image.png")
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

    # Write final video with optimized encoding settings for speed
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
    
    args = parser.parse_args()
    
    # Validate: need either positional args, --all-shorts, or --audio-only
    if not args.all_shorts and not args.audio_only and (not args.scenes_file or not args.output_file):
        parser.error("scenes_file and output_file are required unless using --all-shorts or --audio-only")
    
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
            
            build_video(str(short_path), short_output, save_assets=args.save_assets, is_short=True)
        
        print(f"\n{'='*60}")
        print("[COMPLETE] All shorts built!")
        print(f"{'='*60}")
        sys.exit(0)

    # Detect if input is a short based on path
    # Check if file is in shorts_scripts directory or has "short" in the name
    scenes_path = Path(args.scenes_file)
    is_short = ("short" in args.scenes_file.lower() or 
                "shorts_scripts" in str(scenes_path.parent).lower())
    
    # Build main video
    print(f"\n{'='*60}")
    if args.audio_only:
        print(f"[AUDIO ONLY MODE] Generating audio only: {args.scenes_file}")
    elif args.scene_id is not None:
        print(f"[TEST MODE] Building only scene {args.scene_id}: {args.output_file}")
    elif is_short:
        print(f"[SHORT] Building: {args.output_file}")
    else:
        print(f"[MAIN VIDEO] Building: {args.output_file}")
    print(f"{'='*60}")
    build_video(args.scenes_file, args.output_file, save_assets=args.save_assets, is_short=is_short, scene_id=args.scene_id, audio_only=args.audio_only)
    
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
                build_video(str(short_path), short_output, save_assets=args.save_assets, is_short=True)
        
        print(f"\n{'='*60}")
        print("[COMPLETE] All videos built!")
        print(f"{'='*60}")