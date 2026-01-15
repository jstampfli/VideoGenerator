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
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
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


def text_to_ssml(text: str) -> str:
    """
    Convert plain text to SSML with natural pauses and pacing.
    This makes Google Cloud TTS sound more natural and documentary-like.
    """
    import re
    
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


def build_image_prompt(scene: dict, prev_scene: dict | None, global_block_override: str | None = None) -> str:
    """
    Build a rich image prompt for the current scene, including:
    - Current scene description (title + narration + image_prompt)
    - Brief memory of the previous scene for continuity
    - Global visual style and constraints (no text, documentary tone)
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

    # --- Current scene description block ---
    current_block_parts = []
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


def sanitize_prompt_for_safety(prompt: str, violation_type: str = None) -> str:
    """
    Sanitize an image prompt to avoid safety violations.
    When a safety violation occurs, we modify the prompt to be more abstract,
    focus on achievements rather than struggles, and add explicit safety instructions.
    """
    # Add explicit safety constraints
    safety_instruction = " Safe, appropriate, educational content only. Focus on achievements, intellectual work, and positive moments. Avoid graphic or disturbing imagery."
    
    # If we know the violation type, we can be more specific
    if violation_type == "self-harm":
        # For self-harm violations, focus on positive aspects, achievements, and abstract representations
        prompt = prompt.replace("suffering", "contemplation")
        prompt = prompt.replace("pain", "challenge")
        prompt = prompt.replace("struggle", "journey")
        prompt = prompt.replace("death", "legacy")
        prompt = prompt.replace("illness", "health challenges")
        # Add instruction to focus on positive aspects
        safety_instruction = " Safe, appropriate, educational content. Focus on achievements, intellectual work, contemplation, and positive moments. Use symbolic or abstract representation if needed. Avoid any graphic or disturbing imagery."
    
    # General sanitization: remove or soften potentially problematic words
    problematic_patterns = [
        (r'\b(harm|hurt|pain|suffering|death|suicide|cut|bleed|violence)\b', 'challenge'),
        (r'\b(depression|despair|hopeless)\b', 'contemplation'),
    ]
    
    import re
    for pattern, replacement in problematic_patterns:
        prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
    
    # Append safety instruction
    sanitized = prompt + safety_instruction
    
    return sanitized


def generate_image_for_scene(scene: dict, prev_scene: dict | None, global_block_override: str | None = None, sanitize_attempt: int = 0) -> Path:
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

    prompt = build_image_prompt(scene, prev_scene, global_block_override)
    
    # Sanitize prompt if this is a retry after safety violation
    if sanitize_attempt > 0:
        print(f"[IMAGE] Scene {scene['id']}: sanitizing prompt (attempt {sanitize_attempt})...")
        prompt = sanitize_prompt_for_safety(prompt)
    
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
        error_str = str(e)
        if 'safety' in error_str.lower() or 'moderation' in error_str.lower() or 'rejected' in error_str.lower():
            # Extract violation type if available
            violation_type = None
            if 'self-harm' in error_str.lower():
                violation_type = "self-harm"
            
            # Re-raise with violation info so retry mechanism can handle it
            raise SafetyViolationError(f"Safety violation detected: {error_str}", violation_type=violation_type, original_prompt=prompt)
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
        # Google Cloud TTS with SSML for natural pacing
        from google.cloud import texttospeech
        
        # Convert text to SSML for natural pauses and emphasis
        ssml_text = text_to_ssml(text)
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=GOOGLE_TTS_LANGUAGE,
            name=GOOGLE_TTS_VOICE,
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=GOOGLE_TTS_SPEAKING_RATE,
            pitch=GOOGLE_TTS_PITCH,
            # Add effects for richer sound
            effects_profile_id=["headphone-class-device"],
        )
        
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
    """
    duration = audio_clip.duration
    output_res = config.output_resolution

    clip = ImageClip(str(image_path)).with_duration(duration)
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

    return clip.with_audio(audio_clip).with_duration(duration)


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
    max_attempts = 3  # More attempts for safety violations
    base_delay = 2.0
    
    while True:
        try:
            sanitize_attempt = attempt - 1  # First attempt is 0 (no sanitization), subsequent attempts sanitize
            return generate_image_for_scene(scene, prev_scene, global_block_override, sanitize_attempt=sanitize_attempt)
        except SafetyViolationError as e:
            if attempt >= max_attempts:
                print(f"[RETRY][image] Scene {scene['id']}: Failed after {attempt} attempts with safety violations")
                print(f"[RETRY][image] Original prompt: {e.original_prompt[:200]}...")
                raise Exception(f"Image generation failed due to safety violations after {attempt} attempts. Scene {scene['id']} may need manual prompt adjustment.")
            
            print(f"[RETRY][image] Scene {scene['id']}: Safety violation detected (attempt {attempt}/{max_attempts})")
            if e.violation_type:
                print(f"[RETRY][image] Violation type: {e.violation_type}")
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


def build_video(scenes_path: str, out_video_path: str, save_assets: bool = False, is_short: bool = False):
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
        _build_video_impl(scenes_path, out_video_path)
    finally:
        # Clean up temp directory
        if not config.save_assets and config.temp_dir and os.path.exists(config.temp_dir):
            print(f"[TEMP] Cleaning up temporary files...")
            shutil.rmtree(config.temp_dir)


def _build_video_impl(scenes_path: str, out_video_path: str):
    """Internal implementation of build_video."""
    scenes, metadata = load_scenes(scenes_path)
    num_scenes = len(scenes)
    
    # Extract global_block from metadata if available
    global_block = None
    if metadata and "global_block" in metadata:
        global_block = metadata["global_block"]
        print(f"[METADATA] Using global_block from script metadata")
    else:
        print(f"[METADATA] Using default global_block")

    # Previous-scene references for image continuity
    prev_scenes: list[dict | None] = [None] + scenes[:-1]

    image_paths: list[Path | None] = [None] * num_scenes
    scene_audio_clips: list[AudioFileClip | None] = [None] * num_scenes

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
    missing_audio = [i for i, c in enumerate(scene_audio_clips) if c is None]
    if missing_imgs:
        raise RuntimeError(f"Missing image paths for indices: {missing_imgs}")
    if missing_audio:
        raise RuntimeError(f"Missing audio clips for indices: {missing_audio}")

    total_audio_duration = sum(clip.duration for clip in scene_audio_clips)
    print(f"\n[AUDIO] Total duration: {total_audio_duration:.3f}s ({total_audio_duration/60:.1f} min)")

    # --- Build video clips (each image synced perfectly to its audio) ---
    print("\n[VIDEO] Assembling clips...")
    clips = []
    for i, scene in enumerate(scenes):
        img_path = image_paths[i]
        scene_audio = scene_audio_clips[i]
        
        print(f"[CLIP {i}] Scene {scene['id']}: {scene_audio.duration:.3f}s")
        clip = make_static_clip_with_audio(img_path, scene_audio)
        clips.append(clip)

    # Concatenate all clips
    print("\n[VIDEO] Concatenating clips...")
    final = concatenate_videoclips(clips, method="compose")

    print(f"[VIDEO] Final duration: {final.duration:.3f}s ({final.duration/60:.1f} min)")

    # Write final video
    print(f"\n[VIDEO] Writing to {out_video_path}...")
    final.write_videofile(
        out_video_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
    )
    print("[VIDEO] Done!")


# ------------- ENTRY POINT -------------

def find_shorts_for_script(script_path: str) -> list[Path]:
    """Find all short JSON files associated with a main script."""
    script_path = Path(script_path)
    base_name = script_path.stem.replace("_script", "")
    
    shorts_dir = Path("shorts")
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

  # Build all shorts in the shorts/ directory
  python build_video.py --all-shorts
        """
    )
    
    parser.add_argument("scenes_file", nargs="?", help="Path to scenes JSON file")
    parser.add_argument("output_file", nargs="?", help="Output video file path (e.g., output.mp4)")
    parser.add_argument("--save-assets", action="store_true",
                        help="Save generated images and audio files (default: use temp files)")
    parser.add_argument("--with-shorts", action="store_true",
                        help="Also build all associated YouTube Shorts after main video")
    parser.add_argument("--all-shorts", action="store_true",
                        help="Build all JSON files in the shorts/ directory")
    
    args = parser.parse_args()
    
    # Validate: need either positional args or --all-shorts
    if not args.all_shorts and (not args.scenes_file or not args.output_file):
        parser.error("scenes_file and output_file are required unless using --all-shorts")
    
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
            print("\n[ERROR] No JSON files found in shorts/ directory")
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
    is_short = "short" in args.scenes_file.lower()
    
    # Build main video
    print(f"\n{'='*60}")
    if is_short:
        print(f"[SHORT] Building: {args.output_file}")
    else:
        print(f"[MAIN VIDEO] Building: {args.output_file}")
    print(f"{'='*60}")
    build_video(args.scenes_file, args.output_file, save_assets=args.save_assets, is_short=is_short)
    
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