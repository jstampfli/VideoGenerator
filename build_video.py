import os
import sys
import json
import base64
import time
import random
import argparse
import tempfile
import shutil
import subprocess
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, AudioClip, CompositeAudioClip, concatenate_audioclips
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, SlideIn, SlideOut
from concurrent.futures import ThreadPoolExecutor, as_completed

import llm_utils


# ------------- CONFIG -------------

load_dotenv()  # Load API keys from .env file

# TTS provider and Google voice type (used for --female logic and narration instructions print)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "google").lower()
GOOGLE_VOICE_TYPE = os.getenv("GOOGLE_VOICE_TYPE", "").lower()

# Google Cloud TTS settings (used for --female voice override; llm_utils handles actual TTS)
GOOGLE_TTS_VOICE = os.getenv("GOOGLE_TTS_VOICE", "en-US-Studio-Q")
GOOGLE_GEMINI_MALE_SPEAKER = os.getenv("GOOGLE_GEMINI_MALE_SPEAKER", "Charon")
GOOGLE_GEMINI_FEMALE_SPEAKER = os.getenv("GOOGLE_GEMINI_FEMALE_SPEAKER", "")

# Narration instructions for Gemini/OpenAI TTS (from .env)
# BIOPIC_NARRATION_INSTRUCTIONS = os.getenv("BIOPIC_NARRATION_INSTRUCTIONS", "")
BIOPIC_NARRATION_INSTRUCTIONS = ""
NARRATION_INSTRUCTIONS_FOR_TTS = ""

# Global flag to track if female voice should be used (set by --female flag)
USE_FEMALE_VOICE = False

# Scene pause settings
END_SCENE_PAUSE_LENGTH = float(os.getenv("END_SCENE_PAUSE_LENGTH", "0.15"))  # Pause length in seconds at end of each scene (default: 150ms)
# Pause before narration starts in each scene (gives breathing room after scene change; should be >= crossfade to avoid overlapping narration)
START_SCENE_PAUSE_LENGTH = float(os.getenv("START_SCENE_PAUSE_LENGTH", "0.4"))  # Default 400ms

FIXED_IMAGES_DIR = Path("fixed_images")

# Ken Burns motion settings
KENBURNS_ENABLED = os.getenv("KENBURNS_ENABLED", "true").lower() in ("1", "true", "yes")
KENBURNS_INTENSITY = float(os.getenv("KENBURNS_INTENSITY", "0.04"))  # Max zoom beyond cover-fit (4% default)
KENBURNS_FPS = int(os.getenv("KENBURNS_FPS", "24"))  # Motion clip FPS (24 = cinema standard, smooth & fast)

# Crossfade transition settings
CROSSFADE_DURATION = float(os.getenv("CROSSFADE_DURATION", "0.4"))  # Dissolve between scenes in seconds (0 = disabled)

# Ken Burns motion patterns and transition types (from kenburns_config)
from kenburns_config import (
    KENBURNS_PATTERNS,
    KENBURNS_PATTERN_DESCRIPTIONS,
    KENBURNS_INTENSITY_VALUES,
    TRANSITION_TYPES,
    get_transition_duration,
    normalize_transition,
)


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
    biopic_music_enabled = True
    biopic_music_volume = None  # dB; None = use default from biopic_music_config
    log_image_prompts = False  # If True, print full image prompts (set by --log-image-prompts)
    
    @property
    def output_resolution(self):
        return OUTPUT_RESOLUTION_VERTICAL if self.is_vertical else OUTPUT_RESOLUTION_LANDSCAPE
    
    @property
    def image_size(self):
        # OpenAI image generation sizes
        return "1024x1536" if self.is_vertical else "1536x1024"

config = Config()

# No truncation of prev-scene context for length/cost. Include full context for visual continuity.


def build_image_prompt(scene: dict, prev_scene: dict | None, global_block_override: str | None = None, exclude_title_narration: bool = False, include_safety_instructions: bool = False, story_context: str | None = None) -> str:
    """
    Build a rich image prompt for the current scene, including:
    - Optional story context (e.g. creature/threat so footsteps/figures are drawn correctly)
    - Current scene description (title + narration + image_prompt, unless exclude_title_narration is True)
    - Brief memory of the previous scene for continuity
    - Global visual style and constraints (no text, documentary tone)
    - Age specification when the person appears
    
    Args:
        exclude_title_narration: If True, only use image_prompt (for attempts 4+ to avoid problematic text)
        include_safety_instructions: If True, include safety constraints block (only after first attempt fails)
        story_context: Optional context (e.g. "bear attack") so threat/creature is depicted correctly, not as human.
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

    # --- Constraints block: text-free for normal scenes, TITLE CARD for chapter transitions ---
    if scene.get("scene_type") == "TRANSITION" or scene.get("is_chapter_transition"):
        constraints_block = (
            "TITLE CARD: Display the chapter title text prominently as the main visual element. "
            "Use clean, elegant typography consistent with documentary style. "
            "The frame should be a title card with the chapter title as the focal point.\n\n"
            "CRITICAL - FILL THE ENTIRE FRAME: No empty black areas, gaps, or unrendered space. "
            "The background must extend to all edges with continuous color, gradient, or texture "
            "consistent with the documentary's palette (e.g. warm sepia, muted golds, deep browns). "
            "Every part of the image must be intentionally rendered.\n\n"
            "Keep the visual style, color palette, and overall mood strictly consistent with the rest of the video. "
            "This should look like one continuous documentary, not a mix of different art styles."
        )
    else:
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

    # Story context (creature/threat) so footsteps, figures, shadows match the concept (e.g. bear not human)
    context_block = ""
    if story_context:
        context_block = (
            "STORY CONTEXT (CRITICAL for correct visuals): "
            + story_context
            + "\n\n"
        )

    # Final prompt
    prompt = (
        context_block
        + current_block
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


def generate_image_for_scene(scene: dict, prev_scene: dict | None, global_block_override: str | None = None, sanitize_attempt: int = 0, violation_type: str = None, story_context: str | None = None) -> Path:
    """
    Generate an image for the given scene using OpenAI Images API.
    If config.save_assets is True, caches to generated_images/. Otherwise uses temp directory.
    
    Args:
        sanitize_attempt: Number of times we've tried to sanitize the prompt (0 = original, 1+ = sanitized)
        story_context: Optional context (e.g. creature/threat) so footsteps/figures are drawn correctly.
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
    
    prompt = build_image_prompt(scene, prev_scene, global_block_override, exclude_title_narration=exclude_title_narration, include_safety_instructions=include_safety_instructions, story_context=story_context)
    
    # Log the image prompt for debugging (only when --log-image-prompts is set)
    if getattr(config, "log_image_prompts", False):
        print(f"[IMAGE] Scene {scene['id']}: PROMPT (length={len(prompt)} chars):")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
    
    # Sanitize prompt if this is a retry after safety violation
    if sanitize_attempt > 0:
        print(f"[IMAGE] Scene {scene['id']}: sanitizing prompt (attempt {sanitize_attempt})...")
        prompt = sanitize_prompt_for_safety(prompt, violation_type=violation_type, sanitize_attempt=sanitize_attempt, exclude_title_narration=exclude_title_narration)
        # Log the full sanitized prompt for debugging (only when --log-image-prompts is set)
        if getattr(config, "log_image_prompts", False):
            print(f"[IMAGE] Scene {scene['id']}: SANITIZED PROMPT (length={len(prompt)} chars):")
            print("-" * 80)
            print(prompt[:2000])  # Print first 2000 chars
            if len(prompt) > 2000:
                print(f"... (truncated, full length: {len(prompt)} chars)")
            print("-" * 80)
    
    print(f"[IMAGE] Scene {scene['id']}: generating image...")

    try:
        llm_utils.generate_image(
            prompt=prompt,
            output_path=img_path,
            size=config.image_size,  # vertical for shorts, landscape for main
            output_format="png",
            moderation="low",
        )

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


def generate_audio_for_scene(
    scene: dict,
    previous_narration: str | None = None,
    next_narration: str | None = None,
) -> Path:
    """
    Generate TTS audio for the given scene's narration via llm_utils.generate_speech.
    If config.save_assets is True, caches to generated_audio/. Otherwise uses temp directory.
    previous_narration and next_narration improve continuity for ElevenLabs TTS.
    """
    if config.save_assets:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = AUDIO_DIR / f"scene_{scene['id']:02d}.mp3"
        if audio_path.exists():
            print(f"[AUDIO] Scene {scene['id']}: using cached {audio_path.name}")
            return audio_path
    else:
        audio_path = Path(config.temp_dir) / f"scene_{scene['id']:02d}.mp3"

    text = scene["narration"]
    emotion = scene.get("emotion")
    tts_provider = os.getenv("TTS_PROVIDER", "google").lower()
    if emotion:
        print(f"[AUDIO] Scene {scene['id']}: generating audio ({tts_provider}) with emotion={emotion}...")
    else:
        print(f"[AUDIO] Scene {scene['id']}: generating audio ({tts_provider})...")

    narration_instructions = NARRATION_INSTRUCTIONS_FOR_TTS or scene.get("narration_instructions") or None
    llm_utils.generate_speech(
        text=text,
        output_path=audio_path,
        emotion=emotion,
        narration_instructions=narration_instructions,
        use_female_voice=USE_FEMALE_VOICE,
        previous_text=previous_narration,
        next_text=next_narration,
    )

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
    - Add start pause (breathing room before narration) and end pause.
    """
    audio_duration = audio_clip.duration
    start_pause = START_SCENE_PAUSE_LENGTH
    end_pause = END_SCENE_PAUSE_LENGTH
    total_duration = start_pause + audio_duration + end_pause
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

    # Create silent audio clips for start and end pauses
    fps = audio_clip.fps
    nchannels = audio_clip.nchannels if hasattr(audio_clip, 'nchannels') else 2

    def make_silence(t):
        if nchannels == 1:
            return [0.0]
        return [0.0, 0.0]

    silent_start = AudioClip(make_silence, duration=start_pause, fps=fps)
    silent_end = AudioClip(make_silence, duration=end_pause, fps=fps)
    extended_audio = concatenate_audioclips([silent_start, audio_clip, silent_end])

    return clip.with_audio(extended_audio).with_duration(total_duration)



def make_motion_clip_with_audio(image_path: Path, audio_clip: AudioFileClip,
                                pattern: str | None = None,
                                intensity: str | None = None):
    """
    Create a scene clip with smooth Ken Burns motion using OpenCV warpAffine.

    Each frame is generated lazily via a ``VideoClip(make_frame)`` callback so
    only one frame is in memory at a time.  ``cv2.warpAffine`` performs the
    combined crop + scale in a single SIMD-optimized C operation on numpy arrays,
    avoiding PIL Image object creation/destruction overhead per frame.

    Sub-pixel precision is native to OpenCV's interpolation, completely avoiding
    the integer-pixel snapping that plagued FFmpeg's zoompan filter.

    Parameters
    ----------
    image_path : Path
        Path to the scene image.
    audio_clip : AudioFileClip
        Narration audio for this scene.
    pattern : str or None
        Motion pattern name (from KENBURNS_PATTERNS). If None, one is chosen at random.
    intensity : str or None
        Per-scene intensity (subtle/medium/pronounced). Uses KENBURNS_INTENSITY_VALUES.
        If None, uses global KENBURNS_INTENSITY.
    """
    import math
    import cv2
    from PIL import Image
    from moviepy import VideoClip

    audio_duration = audio_clip.duration
    start_pause = START_SCENE_PAUSE_LENGTH
    end_pause = END_SCENE_PAUSE_LENGTH
    total_duration = start_pause + audio_duration + end_pause
    output_res = config.output_resolution
    out_w, out_h = output_res

    # Load source image once as a numpy array (RGB).
    # PIL handles format decoding; we convert to numpy immediately and close PIL.
    pil_img = Image.open(image_path).convert("RGB")
    in_w, in_h = pil_img.size
    src = np.asarray(pil_img)   # H x W x 3 uint8, RGB order
    pil_img.close()

    # Zoom range: min_zoom makes the image exactly cover the output frame;
    # max_zoom adds intensity-based extra zoom (per-scene or global).
    min_zoom = max(out_w / in_w, out_h / in_h)
    intensity_val = KENBURNS_INTENSITY_VALUES.get((intensity or "").strip().lower()) if intensity else None
    if intensity_val is None:
        intensity_val = KENBURNS_INTENSITY
    max_zoom = min_zoom * (1 + intensity_val)

    # Pick motion pattern
    if pattern is None:
        pattern = random.choice(KENBURNS_PATTERNS)

    # Maximum drift (in source-image pixels) available at the midpoint zoom.
    # Used by drift / combined patterns to stay within the image bounds.
    mid_zoom = (min_zoom + max_zoom) / 2
    mid_crop_w = out_w / mid_zoom
    mid_crop_h = out_h / mid_zoom
    max_drift_x = max((in_w - mid_crop_w) / 2, 0)
    max_drift_y = max((in_h - mid_crop_h) / 2, 0)

    # Use KENBURNS_FPS (default 24) for motion clips — fewer frames to generate
    kb_fps = KENBURNS_FPS

    # Pre-allocate the 2x3 affine matrix (reused every frame, mutated in place)
    M = np.zeros((2, 3), dtype=np.float64)

    # ---- per-frame generator (lazy — only one frame in memory at a time) -----
    def make_frame(t):
        # Cosine ease-in-out: 0 → 1 over total_duration
        progress = min(t / total_duration, 1.0) if total_duration > 0 else 0.0
        ease = (1 - math.cos(progress * math.pi)) / 2

        # --- compute zoom and centre offsets for the chosen pattern ----------
        if pattern == "zoom_in":
            zoom = min_zoom + (max_zoom - min_zoom) * ease
            cx_off, cy_off = 0.0, 0.0

        elif pattern == "zoom_out":
            zoom = max_zoom - (max_zoom - min_zoom) * ease
            cx_off, cy_off = 0.0, 0.0

        elif pattern == "zoom_in_up":
            zoom = min_zoom + (max_zoom - min_zoom) * ease
            cx_off = 0.0
            cy_off = max_drift_y * 0.4 * (1 - ease)   # drift upward

        elif pattern == "zoom_in_down":
            zoom = min_zoom + (max_zoom - min_zoom) * ease
            cx_off = 0.0
            cy_off = -max_drift_y * 0.4 * (1 - ease)  # drift downward

        elif pattern == "drift_up":
            zoom = mid_zoom
            cx_off = 0.0
            cy_off = max_drift_y * (1 - 2 * ease)      # top → bottom in source = upward in frame

        elif pattern == "drift_down":
            zoom = mid_zoom
            cx_off = 0.0
            cy_off = -max_drift_y * (1 - 2 * ease)     # bottom → top in source = downward in frame

        else:
            # Fallback: gentle zoom in
            zoom = min_zoom + (max_zoom - min_zoom) * ease
            cx_off, cy_off = 0.0, 0.0

        # --- derive the floating-point crop box in source-image coords -------
        crop_w = out_w / zoom
        crop_h = out_h / zoom

        # Centre the crop, then apply pattern offset
        left = (in_w - crop_w) / 2 + cx_off
        top  = (in_h - crop_h) / 2 + cy_off

        # Clamp so the box never exceeds image bounds
        left = max(0.0, min(left, in_w - crop_w))
        top  = max(0.0, min(top,  in_h - crop_h))

        # --- build affine matrix: maps output pixel (x,y) → source pixel -----
        # scale = out_w / crop_w = zoom  (same for height by aspect ratio)
        # For output pixel (ox, oy) the source pixel is:
        #   sx = ox / zoom + left
        #   sy = oy / zoom + top
        # warpAffine uses WARP_INVERSE_MAP: dst(x,y) = src(M * [x,y,1]^T)
        inv_zoom = 1.0 / zoom
        M[0, 0] = inv_zoom          # sx scale
        M[0, 1] = 0.0
        M[0, 2] = left               # sx offset
        M[1, 0] = 0.0
        M[1, 1] = inv_zoom          # sy scale
        M[1, 2] = top                # sy offset

        return cv2.warpAffine(
            src, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    clip = VideoClip(make_frame, duration=total_duration).with_fps(kb_fps)

    # Build extended audio (start pause + narration + end pause) — same logic as static clips
    fps = audio_clip.fps
    nchannels = audio_clip.nchannels if hasattr(audio_clip, 'nchannels') else 2

    def make_silence(t):
        if nchannels == 1:
            return [0.0]
        return [0.0, 0.0]

    silent_start = AudioClip(make_silence, duration=start_pause, fps=fps)
    silent_end = AudioClip(make_silence, duration=end_pause, fps=fps)
    extended_audio = concatenate_audioclips([silent_start, audio_clip, silent_end])

    total_frames = int(total_duration * kb_fps)
    print(f"[KENBURNS] Scene {image_path.stem}: {pattern} ({total_duration:.1f}s, {total_frames} frames @ {kb_fps}fps)")
    return clip.with_audio(extended_audio).with_duration(total_duration)


def load_audio_with_accurate_duration(audio_path: Path) -> AudioFileClip:
    """
    Load scene audio so the clip has accurate duration (avoids tiny cut-off at the end).

    FFmpeg/MoviePy can report MP3 duration slightly short (VBR/probe limits), so we use
    the reported duration and the last fraction of a second of narration is never played.
    For MP3 we convert to WAV first (ffmpeg decodes the full file), then load the WAV so
    the clip's duration matches the actual audio. END_SCENE_PAUSE_LENGTH is added after
    the full narration, so the pause is not the cause of cut-off.
    """
    path = Path(audio_path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        temp_dir = config.temp_dir if config.temp_dir else tempfile.gettempdir()
        wav_path = Path(temp_dir) / f"{path.stem}_acc.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-acodec", "pcm_s16le", "-ar", "44100",
                    "-ac", "2",  # stereo so duration/samples are exact
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Fallback: load MP3 directly (may have slight duration under-report)
            return AudioFileClip(str(path))
        return AudioFileClip(str(wav_path))
    return AudioFileClip(str(path))


def generate_image_for_scene_with_retry(scene: dict, prev_scene: dict | None, global_block_override: str | None = None, story_context: str | None = None) -> Path:
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
            return generate_image_for_scene(scene, prev_scene, global_block_override, sanitize_attempt=sanitize_attempt, violation_type=violation_type, story_context=story_context)
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


def generate_audio_for_scene_with_retry(
    scene: dict,
    previous_narration: str | None = None,
    next_narration: str | None = None,
) -> Path:
    """Generate audio with retry logic for network/API failures."""
    attempt = 1
    max_attempts = 3
    base_delay = 2.0
    while True:
        try:
            return generate_audio_for_scene(
                scene,
                previous_narration=previous_narration,
                next_narration=next_narration,
            )
        except Exception as e:
            if attempt >= max_attempts:
                print(f"[RETRY][audio] Scene {scene.get('id')}: Failed after {attempt} attempts: {e}")
                raise
            delay = base_delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
            print(f"[RETRY][audio] Scene {scene.get('id')}: Attempt {attempt} failed: {e}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            attempt += 1


def build_video(scenes_path: str, out_video_path: str | None = None, save_assets: bool = False, is_short: bool = False, scene_id: int | None = None, audio_only: bool = False,
                biopic_music_enabled: bool = True, biopic_music_volume: float = None,
                motion: bool = False):
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
    config.biopic_music_enabled = biopic_music_enabled
    config.biopic_music_volume = biopic_music_volume

    global NARRATION_INSTRUCTIONS_FOR_TTS
    NARRATION_INSTRUCTIONS_FOR_TTS = BIOPIC_NARRATION_INSTRUCTIONS or ""
    if NARRATION_INSTRUCTIONS_FOR_TTS and (GOOGLE_VOICE_TYPE == "gemini" or TTS_PROVIDER == "openai"):
        print(f"[TTS] Using biopic narration instructions from env")

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
    
    # Set up temp directory if not saving assets
    if not config.save_assets:
        config.temp_dir = tempfile.mkdtemp(prefix="video_build_")
        print(f"[TEMP] Using temporary directory: {config.temp_dir}")
    
    try:
        _build_video_impl(scenes_path, out_video_path, scene_id=scene_id, audio_only=audio_only, motion=motion)
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


def _build_video_impl(scenes_path: str, out_video_path: str | None = None, scene_id: int | None = None, audio_only: bool = False, motion: bool = False):
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
    
    # Extract global_block and story_context from metadata if available
    global_block = None
    if metadata and "global_block" in metadata:
        global_block = metadata["global_block"]
        print(f"[METADATA] Using global_block from script metadata")
    else:
        print(f"[METADATA] Using default global_block")
    story_context = metadata.get("story_context") if metadata else None
    
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
        img_path = generate_image_for_scene_with_retry(scene, prev_scene, global_block, story_context=story_context)
        return idx, img_path

    def audio_job(idx: int):
        scene = scenes[idx]
        prev_narration = scenes[idx - 1]["narration"] if idx > 0 else None
        next_narration = scenes[idx + 1]["narration"] if idx < num_scenes - 1 else None
        audio_path = generate_audio_for_scene_with_retry(
            scene,
            previous_narration=prev_narration,
            next_narration=next_narration,
        )
        audio_clip = load_audio_with_accurate_duration(audio_path)
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
    use_kenburns = motion and KENBURNS_ENABLED
    if use_kenburns:
        print(f"[KENBURNS] Motion enabled (intensity={KENBURNS_INTENSITY})")

    # Pre-compute Ken Burns patterns for all scenes so we can enforce the
    # no-consecutive-repeat constraint even though clips are built in parallel.
    scene_patterns = [None] * num_scenes
    if use_kenburns:
        for i in range(num_scenes):
            requested = (scenes[i].get("kenburns_pattern") or "").strip().lower()
            prev_pattern = scene_patterns[i - 1] if i > 0 else None
            next_requested = (
                (scenes[i + 1].get("kenburns_pattern") or "").strip().lower()
                if i + 1 < num_scenes else None
            )

            if requested in KENBURNS_PATTERNS and requested != prev_pattern:
                # LLM-picked pattern is valid and doesn't repeat — use it
                scene_patterns[i] = requested
            else:
                # No pattern, invalid pattern, or would repeat the previous scene.
                # Also avoid the next scene's requested pattern when possible so
                # the next scene doesn't have to override its own choice.
                exclude = {prev_pattern}
                if next_requested in KENBURNS_PATTERNS:
                    exclude.add(next_requested)
                candidates = [p for p in KENBURNS_PATTERNS if p not in exclude]
                if not candidates:
                    # Edge case: exclude set covers all patterns — just avoid prev
                    candidates = [p for p in KENBURNS_PATTERNS if p != prev_pattern]
                scene_patterns[i] = random.choice(candidates) if candidates else random.choice(KENBURNS_PATTERNS)

        for i, p in enumerate(scene_patterns):
            print(f"[KENBURNS] Scene {scenes[i]['id']}: {p}")

    def create_clip(i):
        scene = scenes[i]
        img_path = image_paths[i]
        scene_audio = scene_audio_clips[i]
        print(f"[CLIP {i}] Scene {scene['id']}: {scene_audio.duration:.3f}s")
        if use_kenburns:
            intensity = (scene.get("kenburns_intensity") or "").strip().lower()
            return i, make_motion_clip_with_audio(
                img_path, scene_audio,
                pattern=scene_patterns[i],
                intensity=intensity if intensity in KENBURNS_INTENSITY_VALUES else None,
            )
        return i, make_static_clip_with_audio(img_path, scene_audio)
    
    clips = [None] * num_scenes
    # Use ThreadPoolExecutor to create clips in parallel (this helps with image processing)
    with ThreadPoolExecutor(max_workers=min(4, num_scenes)) as executor:
        futures = {executor.submit(create_clip, i): i for i in range(num_scenes)}
        for future in as_completed(futures):
            i, clip = future.result()
            clips[i] = clip

    # Concatenate clips with per-scene transitions from transition_to_next and transition_speed
    # Build (trans_type, duration) for each pair: clip i -> clip i+1
    has_disclaimer = False
    num_story_clips = len(scenes)
    transition_info = []  # list of (trans_type, duration) for each pair (i, i+1)
    for pair_idx in range(len(clips) - 1):
        if has_disclaimer and pair_idx == 0:
            trans, speed = "crossfade", "medium"
        else:
            scene_idx = pair_idx - 1 if has_disclaimer else pair_idx
            scene = scenes[scene_idx]
            trans, speed = normalize_transition(
                scene.get("transition_to_next"),
                scene.get("transition_speed"),
            )
        dur = get_transition_duration(trans, speed)
        transition_info.append((trans, dur))

    transition_durations = [d for _, d in transition_info]
    any_transition = any(d > 0 for d in transition_durations)
    # Average transition duration so music total matches video; music is trimmed to narration if still long
    if transition_durations:
        crossfade = sum(transition_durations) / len(transition_durations)
    else:
        crossfade = CROSSFADE_DURATION
    print(f"\n[VIDEO] Concatenating clips...{f' (per-scene transitions from script)' if any_transition else ' (cut)'}")

    if any_transition and len(clips) > 1:
        from moviepy import CompositeVideoClip
        original_clips = list(clips)
        try:
            # Apply effects per clip based on transition type (crossfade vs slide_*)
            processed = []
            for i, clip in enumerate(clips):
                effects = []
                if i > 0:
                    trans_in, dur_in = transition_info[i - 1]
                    if dur_in > 0:
                        if trans_in == "crossfade":
                            effects.append(CrossFadeIn(dur_in))
                        elif trans_in in ("slide_left", "slide_right", "slide_up", "slide_down"):
                            # Incoming: slide in from opposite side
                            slide_in_map = {"slide_left": "right", "slide_right": "left", "slide_up": "bottom", "slide_down": "top"}
                            effects.append(SlideIn(dur_in, slide_in_map[trans_in]))
                if i < len(clips) - 1:
                    trans_out, dur_out = transition_info[i]
                    if dur_out > 0:
                        if trans_out == "crossfade":
                            effects.append(CrossFadeOut(dur_out))
                        elif trans_out in ("slide_left", "slide_right", "slide_up", "slide_down"):
                            # Outgoing: slide out to this side
                            slide_out_map = {"slide_left": "left", "slide_right": "right", "slide_up": "top", "slide_down": "bottom"}
                            effects.append(SlideOut(dur_out, slide_out_map[trans_out]))
                processed.append(clip.with_effects(effects) if effects else clip)

            # Build composite with variable overlap via set_start
            t = 0.0
            composited = [processed[0]]
            for i in range(1, len(processed)):
                overlap = transition_durations[i - 1]
                t += processed[i - 1].duration - overlap
                composited.append(processed[i].with_start(t))
            final = CompositeVideoClip(composited)
        except Exception as e:
            print(f"[VIDEO] Per-scene transitions failed ({e}), falling back to simple concatenation...")
            final = concatenate_videoclips(clips, method="compose")
    else:
        try:
            final = concatenate_videoclips(clips, method="chain")
        except Exception:
            print("[VIDEO] Using compose method instead of chain...")
            final = concatenate_videoclips(clips, method="compose")

    print(f"[VIDEO] Final duration: {final.duration:.3f}s ({final.duration/60:.1f} min)")

    # Apply biopic background music if enabled
    if getattr(config, "biopic_music_enabled", True):
        biopic_vol = getattr(config, "biopic_music_volume", None)
        try:
            from biopic_music_config import BIOPIC_END_TAIL_SEC, BIOPIC_END_TAIL_FADEOUT_SEC
            # Shorts: 1s music-only tail. Main videos: longer tail from config.
            if getattr(config, "is_vertical", False):
                tail_sec = 1.0
                fadeout_sec = 0.5
            else:
                tail_sec = BIOPIC_END_TAIL_SEC
                fadeout_sec = BIOPIC_END_TAIL_FADEOUT_SEC

            if tail_sec > 0:
                # Extend video with last frame for tail (music-only outro)
                original_duration = final.duration
                last_frame = final.get_frame(original_duration - 0.01)
                tail_clip = ImageClip(last_frame).with_duration(tail_sec)
                tail_clip = tail_clip.resized(final.size)
                # Extend narration with silence so tail has only music (before concat)
                fps = getattr(final.audio, "fps", 44100) or 44100

                def make_silence(t):
                    if np.isscalar(t):
                        return np.array([0.0, 0.0])
                    return np.zeros((len(t), 2))

                silence_tail = AudioClip(make_silence, duration=tail_sec, fps=fps)
                narration_audio = concatenate_audioclips([final.audio, silence_tail])
                # Extend video with tail
                final = concatenate_videoclips([final, tail_clip], method="chain")
            else:
                narration_audio = final.audio

            print("\n[BIOPIC MUSIC] Adding background music" + (f" (with {tail_sec}s music-only tail)" if tail_sec > 0 else "") + "...")
            mixed_audio = mix_biopic_background_music(
                narration_audio,
                original_duration if tail_sec > 0 else final.duration,
                metadata=metadata,
                scene_audio_clips=scene_audio_clips,
                scenes=scenes,
                music_volume_db=biopic_vol,
                tail_sec=tail_sec,
                fadeout_sec=fadeout_sec,
                crossfade_overlap=crossfade,
                transition_durations=transition_durations,
            )
            final = final.with_audio(mixed_audio)
            print("[BIOPIC MUSIC] Background music added successfully")
        except Exception as e:
            print(f"[WARNING] Failed to add biopic background music: {e}")
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
        # Motion videos have actual inter-frame differences so "ultrafast" avoids
        # the encoder spending time on quality the subtle motion doesn't need.
        # Static videos benefit from "fast" since identical frames compress trivially.
        max_threads = os.cpu_count() or 8  # Use all available CPU cores
        threads = min(max_threads, 16)  # Cap at 16 to avoid overhead
        encode_preset = "ultrafast" if motion else "fast"
        
        # Use KENBURNS_FPS when motion is enabled, otherwise standard FPS.
        # Motion clips are already built at KENBURNS_FPS; encoding at the same
        # rate avoids MoviePy having to duplicate/interpolate frames.
        output_fps = KENBURNS_FPS if motion else FPS
        final.write_videofile(
            out_video_path,
            fps=output_fps,
            codec="libx264",
            audio_codec="aac",
            preset=encode_preset,
            threads=threads,  # Use more CPU cores for faster encoding
            bitrate=None,  # Let codec choose optimal bitrate
            ffmpeg_params=[
                "-movflags", "+faststart",  # Optimize for web streaming
                "-pix_fmt", "yuv420p",  # Ensure compatibility
            ],
        )
        print("[VIDEO] Done!")


def apply_volume_to_audioclip(clip: AudioClip, volume_factor: float) -> AudioClip:
    """Apply volume to an AudioClip by wrapping its audio function."""
    original_get_frame = clip.get_frame

    def volume_adjusted_audio(t):
        audio = original_get_frame(t)
        return audio * volume_factor

    return AudioClip(volume_adjusted_audio, duration=clip.duration, fps=clip.fps)


def normalize_audio_to_lufs(input_path: Path, target_lufs: float = -18.0, output_path: Path | None = None,
                            two_pass: bool = True) -> Path:
    """
    Normalize audio file to target LUFS using ffmpeg loudnorm (EBU R128).
    Uses two-pass mode by default for more accurate results with music.
    Returns path to normalized WAV. Uses temp file if output_path not provided.
    Raises on ffmpeg failure.
    """
    path = Path(input_path)
    if output_path is None:
        temp_dir = config.temp_dir if config.temp_dir else tempfile.gettempdir()
        output_path = Path(temp_dir) / f"{path.stem}_norm.wav"
    out = Path(output_path)
    target_lra = 11.0  # Music typically has more dynamics than speech
    target_tp = -2.0

    if two_pass:
        # Pass 1: measure loudness (outputs JSON to stderr)
        result1 = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-af", f"loudnorm=I={target_lufs}:LRA={target_lra}:tp={target_tp}:print_format=json",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
        )
        if result1.returncode != 0:
            raise RuntimeError(f"ffmpeg loudnorm pass 1 failed: {result1.stderr or result1.stdout}")

        # Parse JSON from stderr (loudnorm prints it in the last lines)
        stderr_text = (result1.stderr or "").strip()
        stderr_lines = stderr_text.split("\n")
        stats = None
        # Try last 15 lines joined (JSON may be multiline)
        for n in range(15, 0, -1):
            if len(stderr_lines) < n:
                continue
            block = "\n".join(stderr_lines[-n:])
            # Find JSON object
            start = block.find("{")
            if start >= 0:
                depth = 0
                end = -1
                for i, c in enumerate(block[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end >= 0:
                    try:
                        stats = json.loads(block[start:end + 1])
                        if "input_i" in stats and "target_offset" in stats:
                            break
                    except json.JSONDecodeError:
                        pass
        if not stats or "input_i" not in stats or "target_offset" not in stats:
            # Fall back to single-pass if we can't parse
            two_pass = False

    if two_pass and stats:
        # Pass 2: apply measured values for accurate normalization
        af_filter = (
            f"loudnorm=I={target_lufs}:LRA={target_lra}:tp={target_tp}:"
            f"linear=true:"
            f"measured_I={stats['input_i']}:"
            f"measured_LRA={stats['input_lra']}:"
            f"measured_tp={stats['input_tp']}:"
            f"measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}"
        )
        result2 = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-af", af_filter,
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        if result2.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg loudnorm pass 2 failed: {result2.stderr or result2.stdout}")
    else:
        # Single-pass fallback
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-af", f"loudnorm=I={target_lufs}:LRA={target_lra}:tp={target_tp}",
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg loudnorm failed: {result.stderr or result.stdout}")
    return out


def _fit_song_to_duration(song_path: Path, required_duration: float) -> AudioFileClip | None:
    """Load an MP3 and fit it to required_duration by looping (if shorter) or trimming (if longer)."""
    load_path = song_path
    try:
        from biopic_music_config import BIOPIC_MUSIC_NORMALIZE_LUFS, BIOPIC_MUSIC_NORMALIZE_TWO_PASS
        normalize_lufs = BIOPIC_MUSIC_NORMALIZE_LUFS
        two_pass = BIOPIC_MUSIC_NORMALIZE_TWO_PASS
    except ImportError:
        normalize_lufs = None
        two_pass = True
    if normalize_lufs is not None and normalize_lufs != 0:
        temp_dir = config.temp_dir if config.temp_dir else tempfile.gettempdir()
        norm_path = Path(temp_dir) / f"{song_path.stem}_norm.wav"
        try:
            load_path = normalize_audio_to_lufs(
                song_path, target_lufs=normalize_lufs, output_path=norm_path, two_pass=two_pass
            )
        except (RuntimeError, FileNotFoundError) as e:
            print(f"[BIOPIC MUSIC] Loudness normalization failed for {song_path.name}: {e}, using original")
    try:
        song = AudioFileClip(str(load_path))
    except Exception as e:
        print(f"[BIOPIC MUSIC] Warning: Could not load {song_path}: {e}")
        return None

    subclip_fn = getattr(song, "subclipped", getattr(song, "subclip", None))
    if subclip_fn is None:
        try:
            song.close()
        except Exception:
            pass
        print("[BIOPIC MUSIC] Warning: Could not fit song to duration: clip has no subclip/subclipped method")
        return None

    try:
        if song.duration < required_duration:
            loops_needed = int(np.ceil(required_duration / song.duration))
            clips = [song] * loops_needed
            combined = concatenate_audioclips(clips)
            if combined.duration > required_duration:
                combined_subclip = getattr(combined, "subclipped", getattr(combined, "subclip", None))
                if combined_subclip:
                    combined = combined_subclip(0, required_duration)
            return combined
        elif song.duration > required_duration:
            return subclip_fn(0, required_duration)
        return song
    except Exception as e:
        print(f"[BIOPIC MUSIC] Warning: Could not fit song to duration: {e}")
        try:
            song.close()
        except Exception:
            pass
        return None


def build_biopic_music_track(metadata: dict | None, scene_audio_clips: list, total_duration: float,
                            scenes: list[dict] | None = None, music_volume_db: float = None,
                            tail_sec: float = 0.0, fadeout_sec: float = 0.0,
                            crossfade_overlap: float = 0.0,
                            transition_durations: list[float] | None = None) -> AudioClip | None:
    """
    Build a continuous music track from biopic_music/.
    When scenes have music_song and music_volume (per-scene mode), uses them.
    Otherwise uses chapter-based or single-segment mode.
    """
    try:
        from collections import defaultdict
        from biopic_music_config import (
            BIOPIC_MUSIC_DIR,
            BIOPIC_MUSIC_VOLUME_DB,
            BIOPIC_MUSIC_CROSSFADE_SEC,
            BIOPIC_MUSIC_CROSSFADE_OFFSET_SEC,
            BIOPIC_MUSIC_DEFAULT_MOODS,
            volume_label_to_db,
        )
    except ImportError:
        print("[BIOPIC MUSIC] biopic_music_config not found, skipping")
        return None

    if not BIOPIC_MUSIC_DIR.exists() or not BIOPIC_MUSIC_DIR.is_dir():
        print(f"[BIOPIC MUSIC] Directory {BIOPIC_MUSIC_DIR} not found, skipping")
        return None

    crossfade = BIOPIC_MUSIC_CROSSFADE_SEC
    crossfade_offset = BIOPIC_MUSIC_CROSSFADE_OFFSET_SEC
    num_clips = len(scene_audio_clips)
    outline = (metadata or {}).get("outline") or {}
    chapters = outline.get("chapters") or (metadata or {}).get("chapters") or []
    script_type = (metadata or {}).get("script_type", "")

    # Per-scene mode: all scenes have music_song and music_volume
    has_per_scene = (
        scenes is not None
        and len(scenes) == num_clips
        and all(s.get("music_song") and s.get("music_volume") for s in scenes)
    )

    # Require music_song/music_volume when biopic with scenes
    if scenes is not None and len(scenes) == num_clips and not has_per_scene:
        if script_type == "biopic" or (chapters and any(ch.get("num_scenes") for ch in chapters)):
            missing = [i for i, s in enumerate(scenes) if not s.get("music_song") or not s.get("music_volume")]
            if missing:
                raise ValueError(
                    f"Scenes {[scenes[i].get('id') for i in missing]} lack music_song or music_volume. "
                    "Biopic videos require per-scene music selection."
                )

    if has_per_scene:
        # Compute scene start times in the video timeline (accounts for transition overlaps).
        # t[i] = when scene i starts; scene i duration = d[i] = start_pause + audio + end_pause.
        scene_durations = []
        for j in range(num_clips):
            d = START_SCENE_PAUSE_LENGTH + (scene_audio_clips[j].duration if j < len(scene_audio_clips) else 0.0) + END_SCENE_PAUSE_LENGTH
            scene_durations.append(d)
        scene_starts = [0.0]
        for j in range(1, num_clips):
            o = transition_durations[j - 1] if transition_durations and j - 1 < len(transition_durations) else crossfade_overlap
            scene_starts.append(scene_starts[-1] + scene_durations[j - 1] - o)
        # Total video duration = sum of scene durations minus overlaps
        n_trans = len(scene_durations) - 1 if scene_durations else 0
        overlaps = (
            [transition_durations[j] for j in range(min(n_trans, len(transition_durations or [])))]
            if transition_durations
            else [crossfade_overlap] * n_trans
        )
        total_video_dur = sum(scene_durations) - sum(overlaps) if scene_durations else 0.0

        # Per-scene: build segments, merging consecutive scenes with same song (by music_song only).
        # Segment boundaries align with video scene starts so song changes sync with scene changes.
        segments = []
        i = 0
        while i < len(scenes):
            song_rel = (scenes[i].get("music_song") or "").strip()
            if not song_rel:
                i += 1
                continue
            start_i = i
            volume_blocks = []
            cum_dur = 0.0
            block_start_cum = 0.0
            prev_vol = None
            prev_cum = 0.0
            while i < len(scenes) and (scenes[i].get("music_song") or "").strip() == song_rel:
                vol_label = (scenes[i].get("music_volume") or "medium").strip().lower()
                scene_dur = scene_durations[i] if i < len(scene_durations) else 0.0
                cum_dur += scene_dur
                n_scenes_so_far = i - start_i + 1
                overlap_so_far = crossfade_overlap * (n_scenes_so_far - 1) if n_scenes_so_far > 1 else 0.0
                cum_after_overlap = cum_dur - overlap_so_far
                if vol_label != prev_vol:
                    if prev_vol is not None:
                        block_dur = prev_cum - block_start_cum
                        if block_dur > 0:
                            volume_blocks.append((block_dur, prev_vol))
                    block_start_cum = prev_cum
                    prev_vol = vol_label
                prev_cum = cum_after_overlap
                i += 1
            if prev_vol is not None:
                block_dur = prev_cum - block_start_cum
                if block_dur > 0:
                    volume_blocks.append((block_dur, prev_vol))
            # Segment ends when next scene starts. Add crossfade to the segment that STARTS (not ends)
            # so the crossfade happens during the previous scene—new song fades in before the cut.
            t_start = scene_starts[start_i] if start_i < len(scene_starts) else 0.0
            t_end = scene_starts[i] if i < len(scene_starts) else total_video_dur
            is_last_segment = i >= len(scenes)
            # Add crossfade to segments that start at a boundary (start_i > 0), not to the first segment.
            # This keeps total duration correct after acrossfade and ensures crossfade happens during the previous scene.
            total_dur = t_end - t_start + (crossfade if start_i > 0 else 0.0)
            # Scale volume_blocks to match target segment duration
            block_sum = sum(d for d, _ in volume_blocks)
            if block_sum > 0 and total_dur > 0 and abs(block_sum - total_dur) > 0.001:
                scale = total_dur / block_sum
                volume_blocks = [(d * scale, v) for d, v in volume_blocks]
            if total_dur > 0 and volume_blocks:
                song_path = BIOPIC_MUSIC_DIR / song_rel
                if not song_path.exists():
                    raise FileNotFoundError(f"Music file not found for scene (song={song_rel}): {song_rel}")
                segments.append((song_rel, volume_blocks))

        if not segments:
            return None

        # Extend last segment with tail (music-only outro) when requested.
        if tail_sec > 0 and segments:
            last_vol = segments[-1][1][-1][1]
            segments[-1][1].append((tail_sec, last_vol))

        segment_clips = []
        for song_rel, volume_blocks in segments:
            total_dur = sum(d for d, _ in volume_blocks)
            song_path = BIOPIC_MUSIC_DIR / song_rel
            clip = _fit_song_to_duration(song_path, total_dur)
            if clip is None:
                continue
            subclip_fn = getattr(clip, "subclipped", getattr(clip, "subclip", None))
            if subclip_fn is None:
                vol_db = volume_label_to_db(volume_blocks[0][1])
                segment_clips.append(apply_volume_to_audioclip(clip, 10 ** (vol_db / 20.0)))
                continue
            block_clips = []
            start_t = 0.0
            clip_dur = getattr(clip, "duration", None) or total_dur
            for block_dur, vol_label in volume_blocks:
                end_t = min(start_t + block_dur, clip_dur)
                if end_t <= start_t:
                    break
                subclip = subclip_fn(start_t, end_t)
                vol_db = volume_label_to_db(vol_label)
                block_clips.append(apply_volume_to_audioclip(subclip, 10 ** (vol_db / 20.0)))
                start_t = end_t
            if block_clips:
                segment_clips.append(concatenate_audioclips(block_clips))

        if not segment_clips:
            return None

        if len(segment_clips) == 1:
            music = segment_clips[0]
        else:
            # Prepend a small offset to new segments so the crossfade shifts forward—new song fades in
            # more during the next scene, less disruptive at the end of the previous.
            if crossfade_offset > 0:
                fps = getattr(segment_clips[0], "fps", 44100) or 44100
                def _make_silence(t):
                    if np.isscalar(t):
                        return np.array([0.0, 0.0], dtype=np.float32)
                    return np.zeros((len(t), 2), dtype=np.float32)
                for i in range(1, len(segment_clips)):
                    silence = AudioClip(_make_silence, duration=crossfade_offset, fps=fps)
                    segment_clips[i] = concatenate_audioclips([silence, segment_clips[i]])
            temp_dir = tempfile.mkdtemp(prefix="biopic_music_")
            try:
                wav_paths = []
                for i, seg in enumerate(segment_clips):
                    p = Path(temp_dir) / f"seg_{i}.wav"
                    seg.write_audiofile(str(p), logger=None)
                    wav_paths.append(p)
                    seg.close()

                if len(wav_paths) == 2:
                    filter_complex = f"[0][1]acrossfade=d={crossfade}:c1=tri:c2=tri[out]"
                else:
                    chain = []
                    for i in range(1, len(wav_paths)):
                        prev = "[0]" if i == 1 else f"[a{i-1}]"
                        out_label = "[out]" if i == len(wav_paths) - 1 else f"[a{i}]"
                        chain.append(f"{prev}[{i}]acrossfade=d={crossfade}:c1=tri:c2=tri{out_label}")
                    filter_complex = ";".join(chain)

                out_wav = Path(temp_dir) / "mixed.wav"
                cmd = ["ffmpeg", "-y"]
                for p in wav_paths:
                    cmd.extend(["-i", str(p)])
                cmd.extend(["-filter_complex", filter_complex, "-map", "[out]", str(out_wav)])
                subprocess.run(cmd, check=True, capture_output=True)
                music = AudioFileClip(str(out_wav))
            finally:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

        if tail_sec > 0 and fadeout_sec > 0:
            from moviepy.audio.fx import AudioFadeOut
            music = music.with_effects([AudioFadeOut(min(fadeout_sec, music.duration))])
        return music

    # Fallback: chapter-based or single-segment (from HEAD)
    vol_db = music_volume_db if music_volume_db is not None else BIOPIC_MUSIC_VOLUME_DB
    total_scenes_from_chapters = sum(ch.get("num_scenes", 0) for ch in chapters)
    use_scene_based = (
        scenes is not None
        and len(scenes) == num_clips
        and any(s.get("chapter_num") for s in scenes)
    )

    if not use_scene_based and (not chapters or num_clips != total_scenes_from_chapters):
        mood = (outline.get("music_mood") or (metadata or {}).get("music_mood") or "").strip().lower()
        default_mood = BIOPIC_MUSIC_DEFAULT_MOODS[0] if BIOPIC_MUSIC_DEFAULT_MOODS else "relaxing"
        mood = mood or default_mood
        mood_dir = BIOPIC_MUSIC_DIR / mood
        songs = list(mood_dir.glob("*.mp3")) if mood_dir.exists() else []
        if not songs:
            for fallback in BIOPIC_MUSIC_DEFAULT_MOODS:
                if fallback != mood:
                    fallback_dir = BIOPIC_MUSIC_DIR / fallback
                    if fallback_dir.exists():
                        songs = list(fallback_dir.glob("*.mp3"))
                        if songs:
                            mood_dir = fallback_dir
                            break
        if not songs:
            print(f"[BIOPIC MUSIC] No MP3s for mood '{mood}', skipping")
            return None
        song_path = random.choice(songs)
        seg_duration = total_duration + tail_sec if tail_sec > 0 else total_duration
        clip = _fit_song_to_duration(song_path, seg_duration)
        if clip is None:
            return None
        music = apply_volume_to_audioclip(clip, 10 ** (vol_db / 20.0))
        if tail_sec > 0 and fadeout_sec > 0:
            from moviepy.audio.fx import AudioFadeOut
            music = music.with_effects([AudioFadeOut(min(fadeout_sec, music.duration))])
        return music

    # Chapter-based mode
    segment_clips = []
    chapter_to_mood = {ch.get("chapter_num"): (ch.get("music_mood") or "relaxing").strip().lower() for ch in chapters if ch.get("chapter_num")}
    outline_mood = (outline.get("music_mood") or "").strip().lower() or None

    if use_scene_based:
        chapter_durations = defaultdict(float)
        for i, scene in enumerate(scenes):
            ch_num = scene.get("chapter_num") or 1
            if i < len(scene_audio_clips):
                chapter_durations[ch_num] += START_SCENE_PAUSE_LENGTH + scene_audio_clips[i].duration + END_SCENE_PAUSE_LENGTH
        if crossfade_overlap > 0:
            for ch_num in chapter_durations:
                chapter_durations[ch_num] = max(chapter_durations[ch_num] - crossfade_overlap, 0.1)
        sorted_chapters = sorted(chapter_durations.keys())
        last_ch_num = sorted_chapters[-1] if sorted_chapters else None
        for idx, ch_num in enumerate(sorted_chapters):
            chapter_duration = chapter_durations[ch_num]
            if chapter_duration <= 0:
                continue
            if ch_num == last_ch_num and tail_sec > 0:
                chapter_duration += tail_sec
            # Add crossfade to chapters that start (idx > 0) so crossfade happens during previous chapter
            if idx > 0:
                chapter_duration += crossfade
            mood = chapter_to_mood.get(ch_num) or outline_mood or "relaxing"
            mood_dir = BIOPIC_MUSIC_DIR / mood
            if not mood_dir.exists():
                mood_dir = BIOPIC_MUSIC_DIR / (BIOPIC_MUSIC_DEFAULT_MOODS[0] if BIOPIC_MUSIC_DEFAULT_MOODS else "relaxing")
            mp3s = list(mood_dir.glob("*.mp3"))
            if not mp3s:
                for fallback in BIOPIC_MUSIC_DEFAULT_MOODS:
                    if fallback != mood:
                        fallback_dir = BIOPIC_MUSIC_DIR / fallback
                        if fallback_dir.exists():
                            mp3s = list(fallback_dir.glob("*.mp3"))
                            if mp3s:
                                mood_dir = fallback_dir
                                break
            if not mp3s:
                continue
            song_path = random.choice(mp3s)
            clip = _fit_song_to_duration(song_path, chapter_duration)
            if clip is not None:
                segment_clips.append(clip)
    else:
        scene_idx = 0
        chapter_list = [ch for ch in chapters if ch.get("num_scenes", 0) > 0]
        for ch_idx, ch in enumerate(chapter_list):
            num_scenes = ch.get("num_scenes", 0)
            if num_scenes <= 0:
                continue
            chapter_duration = 0.0
            for i in range(num_scenes):
                if scene_idx + i < len(scene_audio_clips):
                    chapter_duration += START_SCENE_PAUSE_LENGTH + scene_audio_clips[scene_idx + i].duration + END_SCENE_PAUSE_LENGTH
            if crossfade_overlap > 0:
                intra = num_scenes - 1
                inter = 1 if ch_idx > 0 else 0
                chapter_duration -= (intra + inter) * crossfade_overlap
                chapter_duration = max(chapter_duration, 0.1)
            if ch_idx == len(chapter_list) - 1 and tail_sec > 0:
                chapter_duration += tail_sec
            # Add crossfade to chapters that start (ch_idx > 0) so crossfade happens during previous chapter
            if ch_idx > 0:
                chapter_duration += crossfade
            mood = (ch.get("music_mood") or "relaxing").strip().lower()
            mood_dir = BIOPIC_MUSIC_DIR / mood
            if not mood_dir.exists():
                mood_dir = BIOPIC_MUSIC_DIR / BIOPIC_MUSIC_DEFAULT_MOODS[0]
            mp3s = list(mood_dir.glob("*.mp3"))
            if not mp3s:
                scene_idx += num_scenes
                continue
            song_path = random.choice(mp3s)
            clip = _fit_song_to_duration(song_path, chapter_duration)
            if clip is not None:
                segment_clips.append(clip)
            scene_idx += num_scenes

    if not segment_clips:
        return None

    # Prepend crossfade_offset to new segments so the crossfade shifts forward—new song fades in
    # more during the next scene, less disruptive at the end of the previous.
    if len(segment_clips) > 1 and crossfade_offset > 0:
        fps = getattr(segment_clips[0], "fps", 44100) or 44100
        def _make_silence_ch(t):
            if np.isscalar(t):
                return np.array([0.0, 0.0], dtype=np.float32)
            return np.zeros((len(t), 2), dtype=np.float32)
        for i in range(1, len(segment_clips)):
            silence = AudioClip(_make_silence_ch, duration=crossfade_offset, fps=fps)
            segment_clips[i] = concatenate_audioclips([silence, segment_clips[i]])
    if len(segment_clips) == 1:
        music = segment_clips[0]
    else:
        temp_dir = tempfile.mkdtemp(prefix="biopic_music_")
        try:
            wav_paths = []
            for i, seg in enumerate(segment_clips):
                p = Path(temp_dir) / f"seg_{i}.wav"
                seg.write_audiofile(str(p), logger=None)
                wav_paths.append(p)
                seg.close()
            if len(wav_paths) == 2:
                filter_complex = f"[0][1]acrossfade=d={crossfade}:c1=tri:c2=tri[out]"
            else:
                chain = []
                for i in range(1, len(wav_paths)):
                    prev = "[0]" if i == 1 else f"[a{i-1}]"
                    out_label = "[out]" if i == len(wav_paths) - 1 else f"[a{i}]"
                    chain.append(f"{prev}[{i}]acrossfade=d={crossfade}:c1=tri:c2=tri{out_label}")
                filter_complex = ";".join(chain)
            out_wav = Path(temp_dir) / "mixed.wav"
            cmd = ["ffmpeg", "-y"]
            for p in wav_paths:
                cmd.extend(["-i", str(p)])
            cmd.extend(["-filter_complex", filter_complex, "-map", "[out]", str(out_wav)])
            subprocess.run(cmd, check=True, capture_output=True)
            music = AudioFileClip(str(out_wav))
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    music = apply_volume_to_audioclip(music, 10 ** (vol_db / 20.0))
    if tail_sec > 0 and fadeout_sec > 0:
        from moviepy.audio.fx import AudioFadeOut
        music = music.with_effects([AudioFadeOut(min(fadeout_sec, music.duration))])
    return music


def mix_biopic_background_music(narration_audio: AudioClip, duration: float, metadata: dict | None,
                                scene_audio_clips: list, scenes: list[dict] | None = None,
                                music_volume_db: float = None, tail_sec: float = 0.0,
                                fadeout_sec: float = 0.0,
                                crossfade_overlap: float = 0.0,
                                transition_durations: list[float] | None = None) -> CompositeAudioClip:
    """Mix biopic background music under narration."""
    music_track = build_biopic_music_track(
        metadata, scene_audio_clips, duration,
        scenes=scenes, music_volume_db=music_volume_db,
        tail_sec=tail_sec, fadeout_sec=fadeout_sec,
        crossfade_overlap=crossfade_overlap,
        transition_durations=transition_durations,
    )
    if music_track is None:
        return CompositeAudioClip([narration_audio])
    # Trim music to narration duration so it never extends past the video (avoids long fadeout)
    target_dur = narration_audio.duration
    if music_track.duration > target_dur:
        subclip_fn = getattr(music_track, "subclipped", getattr(music_track, "subclip", None))
        if subclip_fn:
            music_track = subclip_fn(0, target_dur)
    return CompositeAudioClip([narration_audio, music_track])


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

  # Build video with Ken Burns motion and crossfade transitions
  python build_video.py scenes.json output.mp4 --motion

  # Print full image prompts to stdout (for debugging)
  python build_video.py scenes.json output.mp4 --log-image-prompts
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
    parser.add_argument("--no-biopic-music", action="store_true",
                        help="Disable biopic background music (enabled by default)")
    parser.add_argument("--biopic-music-volume", type=float, default=None, metavar="DB",
                        help="Biopic music volume in dB (default: -22)")
    parser.add_argument("--motion", action="store_true",
                        help="Enable Ken Burns motion and crossfade transitions (smooth pan/zoom on images)")
    parser.add_argument("--log-image-prompts", action="store_true",
                        help="Print full image prompts to stdout (for debugging)")
    
    args = parser.parse_args()
    
    # Validate: need either positional args or --all-shorts
    # For --audio-only, output_file is optional (will use default if not provided)
    if not args.all_shorts and not args.scenes_file:
        parser.error("scenes_file is required unless using --all-shorts")
    
    # Override voice if --female is specified (llm_utils reads from env / use_female_voice)
    global USE_FEMALE_VOICE
    if args.female and TTS_PROVIDER == "google":
        USE_FEMALE_VOICE = True
        if GOOGLE_VOICE_TYPE == "gemini":
            if GOOGLE_GEMINI_FEMALE_SPEAKER:
                print(f"[TTS] Using Gemini female voice: {GOOGLE_GEMINI_FEMALE_SPEAKER}")
            else:
                print("[WARNING] --female specified for Gemini but GOOGLE_GEMINI_FEMALE_SPEAKER not set in .env. Using default voice.")
        else:
            female_voice = os.getenv("GOOGLE_TTS_FEMALE_VOICE")
            if female_voice:
                os.environ["GOOGLE_TTS_VOICE"] = female_voice
                print(f"[TTS] Using female voice: {female_voice}")
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

    config.log_image_prompts = args.log_image_prompts
    if args.log_image_prompts:
        print("[MODE] Logging image prompts to stdout")

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
                       biopic_music_enabled=not args.no_biopic_music,
                       biopic_music_volume=args.biopic_music_volume,
                       motion=args.motion)
        
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
               biopic_music_enabled=not args.no_biopic_music,
               biopic_music_volume=args.biopic_music_volume,
               motion=args.motion)
    
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
                           biopic_music_enabled=not args.no_biopic_music,
                           biopic_music_volume=args.biopic_music_volume,
                           motion=args.motion)
        
        print(f"\n{'='*60}")
        print("[COMPLETE] All videos built!")
        print(f"{'='*60}")