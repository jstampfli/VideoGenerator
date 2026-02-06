"""
Kids Shorts Story Generator

Generates kid-friendly animated stories for YouTube Shorts (~2 minutes, ~20 scenes).
Each story is designed for ages 4-10 with positive themes and engaging characters.
"""
import os
import sys
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv

import llm_utils

load_dotenv()

# Scene configuration (text/image provider and model from .env via llm_utils)
DEFAULT_NUM_SCENES = int(os.getenv("KIDS_SCENES_COUNT", "20"))
SCENE_DURATION = int(os.getenv("KIDS_SCENE_DURATION", "8")) - 2

SCRIPTS_DIR = Path("kids_scripts")
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_json_response(text: str) -> str:
    """Clean JSON response from LLM, removing markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def generate_kids_story(prompt: str, num_scenes: int = DEFAULT_NUM_SCENES) -> dict:
    """
    Generate a complete kid-friendly story structure from a user prompt.
    
    Returns:
        {
            "title": str,
            "summary": str,
            "characters": list[str],
            "setting": str,
            "theme": str,
            "age_range": str
        }
    """
    print(f"\n[STORY] Generating kid-friendly story from prompt: {prompt}")
    
    story_prompt = f"""Generate a kid-friendly animated story suitable for ages 4-10 based on this prompt: {prompt}

REQUIREMENTS:
- Age-appropriate content (suitable for ages 4-10)
- Positive themes (friendship, adventure, learning, kindness, courage, helping others)
- Clear beginning, middle, and end
- Simple, engaging characters (2-4 main characters)
- Bright, colorful, animated style
- Educational or moral lesson woven naturally into the story
- Approximately {num_scenes} scenes when broken down (each ~{SCENE_DURATION} seconds)

STORY STRUCTURE:
- Title: Catchy, kid-friendly title (3-6 words)
- Summary: Brief 2-3 sentence summary of the story
- Characters: List of 2-4 main characters with simple, clear visual descriptions. For each character, include:
  * Appearance description: Simple, clear description focusing ONLY on what the character looks like visually (size, shape, colors, key physical features). Do NOT include personality traits - only visual appearance that can be seen.
  * Voice description: Simple description of the character's voice. Start with a celebrity/character reference (just the name) and what movie they are from, then briefly describe the voice. Format: "Voice reference: [Celebrity/Character Name]. [Brief voice description: age, pitch, tone]"
    - Example: "Voice reference: Wall-E from Wall-E. Young robot, mid-range pitch, friendly and curious tone with slight metallic echo."
    - Example: "Voice reference: Dug from Up. Young male, energetic and cheerful, mid-high pitch."
- Setting: Where the story takes place
- Theme: Main lesson/message (e.g., "friendship", "bravery", "helping others", "learning from mistakes")
- Age range: Target age (e.g., "4-10")
- Music description: Detailed description of the background music style that fits this story. Include:
  * Musical style and mood (e.g., "cheerful and magical", "adventurous and uplifting", "gentle and warm")
  * Instrumentation (e.g., "playful piano, gentle strings, soft bells", "light orchestral with woodwinds", "whimsical melodies with chimes")
  * Tempo and energy level (e.g., "moderate tempo, steady and calming", "upbeat but not overwhelming")
  * Overall feel that matches the story's theme and characters
- Style prompt: Detailed visual style description in Pixar animation style. Include:
  * Animation style: Pixar-style 3D computer animation, polished and refined
  * Character design: Expressive, rounded, friendly character designs with distinct personalities
  * Color palette: Vibrant, saturated colors with excellent lighting and depth
  * Visual quality: High-quality rendering with smooth textures, realistic materials, and cinematic lighting
  * Atmosphere: Warm, inviting, magical atmosphere typical of Pixar films
  * Camera work: Cinematic camera angles and smooth camera movements
  * Overall feel: Professional, polished, heartwarming animated film quality

{{
  "title": "Story title",
  "summary": "Brief story summary (2-3 sentences)",
  "characters": [
    {{
      "name": "Character name",
      "description": "Simple, clear visual appearance description only (what they look like: size, shape, colors, key features). No personality traits.",
      "voice_description": "Start with celebrity/character reference (just the name), then brief voice description. Format: 'Voice reference: [Name]. [Brief description: age, pitch, tone].' Example: 'Voice reference: Wall-E. Young robot, mid-range pitch, friendly and curious tone with slight metallic echo.'"
    }}
  ],
  "setting": "Where the story takes place",
  "theme": "Main lesson/message",
  "age_range": "4-10",
  "music_description": "Detailed description of background music style, instrumentation, tempo, and mood that fits this story",
  "style_prompt": "Detailed Pixar-style visual description for animation"
}}"""

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You are a children's story writer who creates engaging, age-appropriate animated stories with positive themes. "},
            {"role": "user", "content": story_prompt}
        ],
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    
    story = json.loads(clean_json_response(content))
    
    # Ensure style_prompt exists (fallback if not generated)
    if not story.get('style_prompt'):
        story['style_prompt'] = "Pixar-style 3D computer animation with vibrant colors, expressive character designs, polished rendering, cinematic lighting, and warm, magical atmosphere. High-quality animation with smooth textures, realistic materials, and professional film quality."
    
    # Ensure music_description exists (fallback if not generated)
    if not story.get('music_description'):
        story['music_description'] = "Kid-friendly, cheerful background music with a magical, uplifting feel. Instrumental music with playful piano, gentle strings, soft bells, and airy melodies. Moderate tempo, steady and calming. The music should be warm, inviting, and enhance the story's mood without overwhelming dialogue or visuals."
    
    print(f"[STORY] Generated: {story.get('title', 'Untitled')}")
    print(f"[STORY] Theme: {story.get('theme', 'N/A')}")
    print(f"[STORY] Characters: {', '.join([c.get('name', '?') if isinstance(c, dict) else c for c in story.get('characters', [])])}")
    print(f"[STORY] Style: {story.get('style_prompt', 'N/A')[:100]}...")
    print(f"[STORY] Music: {story.get('music_description', 'N/A')[:100]}...")
    
    return story


def generate_kids_scenes(story: dict, num_scenes: int = DEFAULT_NUM_SCENES) -> list[dict]:
    """
    Generate scenes from the story structure.
    Each scene should be ~{SCENE_DURATION} seconds when animated.
    
    Args:
        story: Story dict with title, summary, characters, etc.
        num_scenes: Number of scenes to generate (default: 20)
    
    Returns:
        List of scene dicts with: id, title, description, image_prompt, video_prompt, duration_estimate
    """
    print(f"\n[SCENES] Generating {num_scenes} scenes for story...")
    
    characters_str = "\n".join([
        f"- {c.get('name', '?')}: {c.get('description', '')}" if isinstance(c, dict) else f"- {c}"
        for c in story.get('characters', [])
    ])
    
    # Special handling for single-scene videos
    if num_scenes == 1:
        scenes_prompt = f"""Generate exactly 1 scene that contains the COMPLETE STORY for this kid-friendly animated video. This is a single 10-second animation that tells the entire story from beginning to end.

STORY:
Title: {story.get('title', 'Untitled')}
Summary: {story.get('summary', '')}
Characters:
{characters_str}
Setting: {story.get('setting', '')}
Theme: {story.get('theme', '')}

CRITICAL REQUIREMENTS FOR SINGLE-SCENE VIDEO:
- This is ONE complete 10-second animation that tells the FULL STORY
- The scene must have a clear beginning, middle, and end
- All characters should be involved in the action
- Make it interesting, engaging, and visually appealing
- The story should feel complete and satisfying in just 10 seconds
- Bright, colorful, animated style for kids (ages 4-10)
- Positive, engaging tone
- Keep it simple but complete - tell the whole story visually

For the single scene, provide:
- "id": 1
- "title": Short scene title (2-4 words) - can be the story title or a summary title
- "description": Brief description of what happens in this complete story (1-2 sentences)
- "image_prompt": Simple visual description of the first frame. Match the first moment of video_prompt. Include only characters and elements present at the very start. Style: animated, bright and colorful. Format: 9:16 vertical.
- "video_prompt": Detailed description of the COMPLETE STORY happening in this 10-second animation. This must tell the entire story from beginning to end. Use clear, straightforward language. Describe actions in sequence. Guidelines:
  * CRITICAL - SIMPLE ACTIONS ONLY: All actions must be extremely simple and large-scale. No intricate movements, small details, or subtle gestures. Everything should happen in open space with clear, obvious movements. Examples: walking, running, jumping, waving, reaching out, turning around. Avoid: small hand movements, subtle expressions, detailed interactions with tiny objects.
  * CRITICAL - MINIMAL DIALOGUE: Keep dialogue to an absolute minimum. Prefer visual storytelling over dialogue. Only include dialogue when absolutely essential to convey the story. The scene should work without dialogue.
  * CRITICAL - COMPLETE STORY: This single scene must tell the complete story - include the beginning (characters meet or problem starts), middle (action or interaction), and end (resolution or conclusion). All characters should be involved.
  * CRITICAL - DIALOGUE CLARITY (if dialogue is included): When characters speak, ALWAYS explicitly state which character is speaking. Format: "[Character Name] says, \"[dialogue]\"" or "[Character Name] speaks: \"[dialogue]\"". Never use ambiguous dialogue.
  * Include 5-8 sentences with simple, large-scale actions that tell the complete story. Example: "The robot walks across the open meadow. A bird flies down and lands in front of the robot. The robot waves its arm. The bird hops closer. The robot reaches out its hand. The bird hops onto the robot's hand. They both smile and walk together across the meadow."
  * Keep all movements and actions simple, obvious, and happening in open space. Avoid crowded scenes or complex interactions.
  * At the end, all characters should be clearly visible together, showing the story's resolution.
  * Use character names consistently. Reference their appearance from the character descriptions when first mentioned.

CRITICAL - NO TEXT OR CREDITS:
- The video must contain NO text, NO credits, NO logos, NO end screens, NO titles, NO watermarks, NO written words
- Simply end with the characters in their final positions showing the story's resolution

{{
  "scenes": [
    {{
      "id": 1,
      "title": "Scene title (can be story title)",
      "description": "Complete story description",
      "image_prompt": "Visual description of the first frame. Match the first moment of video_prompt. Include only characters/elements present at the very start.",
      "video_prompt": "Complete 10-second story animation. The robot walks across the open meadow. A bird flies down and lands in front of the robot. The robot waves its arm. The bird hops closer. The robot reaches out its hand. The bird hops onto the robot's hand. They both smile and walk together across the meadow. At the end, both characters are clearly visible together, showing friendship.",
      "duration_estimate": {SCENE_DURATION}
    }}
  ]
}}"""
    else:
        scenes_prompt = f"""Generate exactly {num_scenes} scenes for this kid-friendly animated story.

STORY:
Title: {story.get('title', 'Untitled')}
Summary: {story.get('summary', '')}
Characters:
{characters_str}
Setting: {story.get('setting', '')}
Theme: {story.get('theme', '')}

REQUIREMENTS:
- Exactly {num_scenes} scenes total
- Each scene is ~{SCENE_DURATION} seconds
- Scenes flow smoothly from one to the next
- Character appearances stay consistent across all scenes
- Bright, colorful, animated style for kids (ages 4-10)
- Positive, engaging tone

VIDEO CONTINUITY:
- Scene 2+ automatically start from the previous scene's final frame
- For scenes 2+, only describe NEW actions - don't repeat the previous scene's ending

For each scene, provide:
- "id": Scene number (1 to {num_scenes})
- "title": Short scene title (2-4 words)
- "description": What happens in this scene (1-2 sentences)
- "image_prompt": (ONLY for scene 1) Simple visual description of the first frame. Match the first moment of video_prompt. Include only characters and elements present at the very start. Style: animated, bright and colorful. Format: 9:16 vertical.
- "video_prompt": Detailed description of what happens in the video. Each scene is ~{SCENE_DURATION} seconds, so include enough actions to fill that time. Use clear, straightforward language. Describe actions in sequence. Guidelines:
  * CRITICAL - SIMPLE ACTIONS ONLY: All actions must be extremely simple and large-scale. No intricate movements, small details, or subtle gestures. Everything should happen in open space with clear, obvious movements. Examples: walking, running, jumping, waving, reaching out, turning around. Avoid: small hand movements, subtle expressions, detailed interactions with tiny objects.
  * CRITICAL - MINIMAL DIALOGUE: Keep dialogue to an absolute minimum. Prefer visual storytelling over dialogue. Only include dialogue when absolutely essential to convey the story. Most scenes should have NO dialogue at all.
  * CRITICAL - ONE SCENE PER CHARACTER SPEAKING: Each character can only speak in ONE scene throughout the entire story. Once a character has spoken in a scene, they should NOT speak in any other scenes. This ensures voice consistency is not an issue. Plan dialogue carefully so each character speaks only once.
  * CRITICAL - DIALOGUE CLARITY (when dialogue is included): When characters speak, ALWAYS explicitly state which character is speaking. Format: "[Character Name] says, \"[dialogue]\"" or "[Character Name] speaks: \"[dialogue]\"". Never use ambiguous dialogue without clearly identifying the speaker.
  * CRITICAL - END OF SCENE VISIBILITY: At the end of each scene, ALL characters that have been introduced in the story so far must be clearly visible in the frame. Describe their positions clearly so they are all in view.
  * For scene 1: Describe what happens from start to finish with multiple simple actions and interactions.
  * For scenes 2+: Start with the new actions that happen in this scene. Do NOT describe how the scene starts - the video automatically begins from the previous scene's final frame.
  * Include 3-5 sentences with simple, large-scale actions. Example: "The robot walks across the open meadow. A bird flies down and lands in front of the robot. The robot waves its arm. The bird hops closer. The robot reaches out its hand."
  * Keep all movements and actions simple, obvious, and happening in open space. Avoid crowded scenes or complex interactions.
  * Use character names consistently. Reference their appearance from the character descriptions when first mentioned in a scene.

CONTINUITY:
- Keep character appearances consistent (same colors, clothing, features)
- Scenes should flow naturally from one to the next
- All introduced characters must be clearly visible at the end of each scene

{{
  "scenes": [
    {{
      "id": 1,
      "title": "Scene title",
      "description": "What happens",
      "image_prompt": "Visual description of the first frame of scene 1 (ONLY for scene 1). Match the first moment of video_prompt. Include only characters/elements present at the very start.",
      "video_prompt": "The young girl walks across an open meadow. A glowing sprite flies down and lands in front of her. The girl stops walking. The girl reaches out her hand. The sprite floats up and hovers near the girl's hand. At the end, both the girl and the sprite are clearly visible standing together in the open meadow.",
      "duration_estimate": {SCENE_DURATION}
    }},
    {{
      "id": 2,
      "title": "Scene title",
      "description": "What happens",
      "video_prompt": "The girl turns around in the open meadow. A bunny hops out into the open space and stops. The girl walks toward the bunny. The bunny hops closer. The girl reaches out her hand. At the end, the girl, the sprite, and the bunny are all clearly visible together in the open meadow.",
      "duration_estimate": {SCENE_DURATION}
    }},
    ...
    (exactly {num_scenes} scenes, only scene 1 has image_prompt. Keep dialogue to an absolute minimum - most scenes should have NO dialogue. When dialogue is absolutely necessary, each character can only speak in ONE scene total throughout the entire story.)
  ]
}}"""
    
    # Use different system message for single scene vs multiple scenes
    if num_scenes == 1:
        system_message = "You are a children's story animator who creates a single complete 10-second animated story. CRITICAL REQUIREMENTS: 1) This is ONE complete 10-second animation that tells the FULL STORY from beginning to end. 2) ALL actions must be extremely simple and large-scale - no intricate movements, small details, or subtle gestures. Everything happens in open space with clear, obvious movements (walking, running, jumping, waving, reaching out). 3) MINIMAL DIALOGUE: Keep dialogue to an absolute minimum - prefer visual storytelling. The scene should work without dialogue. 4) COMPLETE STORY: Include beginning (characters meet or problem starts), middle (action or interaction), and end (resolution or conclusion). All characters should be involved. 5) When characters do speak, ALWAYS explicitly state which character is speaking using format: '[Character Name] says, \"[dialogue]\"'. 6) video_prompt must be detailed enough to fill 10 seconds - include 5-8 sentences with simple, large-scale actions that tell the complete story. 7) At the end, all characters should be clearly visible together, showing the story's resolution. 8) image_prompt should match the first moment of video_prompt. 9) CRITICAL - NO TEXT OR CREDITS: The video must contain NO text, NO credits, NO logos, NO end screens, NO titles, NO watermarks, NO written words. Simply end with the characters in their final positions. Use clear, straightforward language. "
    else:
        system_message = "You are a children's story animator who creates detailed scene descriptions for animated videos. CRITICAL REQUIREMENTS: 1) ALL actions must be extremely simple and large-scale - no intricate movements, small details, or subtle gestures. Everything happens in open space with clear, obvious movements (walking, running, jumping, waving, reaching out). 2) MINIMAL DIALOGUE: Keep dialogue to an absolute minimum - prefer visual storytelling. Most scenes should have NO dialogue. Only include dialogue when absolutely essential. 3) ONE SCENE PER CHARACTER SPEAKING: Each character can only speak in ONE scene throughout the entire story. Once a character has spoken, they should NOT speak in any other scenes. Plan dialogue carefully. 4) When characters do speak, ALWAYS explicitly state which character is speaking using format: '[Character Name] says, \"[dialogue]\"'. 5) At the end of each scene, ALL characters introduced so far must be clearly visible in the frame - describe their positions. 6) video_prompt must be detailed enough to fill the scene duration (~{SCENE_DURATION} seconds) - include 3-5 sentences with simple, large-scale actions. 7) For scenes 2+: Start with new actions only - the video automatically begins from the previous scene's final frame. 8) Keep character appearances consistent across all scenes. 9) image_prompt (scene 1 only) should match the first moment of video_prompt. 10) CRITICAL - NO TEXT OR CREDITS: The video must contain NO text, NO credits, NO logos, NO end screens, NO titles, NO watermarks, NO written words. The final scene should end with story action only - no closing scenes, credits, or text overlays. Use clear, straightforward language. "

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": scenes_prompt}
        ],
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(clean_json_response(content))
    scenes = result.get("scenes", [])
    
    if len(scenes) != num_scenes:
        print(f"[WARNING] Generated {len(scenes)} scenes, expected {num_scenes}")
    
    print(f"[SCENES] Generated {len(scenes)} scenes")
    for scene in scenes:
        print(f"  Scene {scene.get('id', '?')}: {scene.get('title', 'Untitled')}")
    
    return scenes


def generate_video_metadata(story: dict, scenes: list[dict]) -> dict:
    """
    Generate YouTube video description and tags for the story.
    
    Args:
        story: Story dict with title, summary, characters, theme, etc.
        scenes: List of scene dicts
    
    Returns:
        Dict with "description" and "tags" keys
    """
    print(f"\n[METADATA] Generating video description and tags...")
    
    characters_str = ", ".join([
        c.get('name', '?') if isinstance(c, dict) else str(c)
        for c in story.get('characters', [])
    ])
    
    metadata_prompt = f"""Generate YouTube Shorts metadata for this kid-friendly animated story.

STORY:
Title: {story.get('title', 'Untitled')}
Summary: {story.get('summary', '')}
Characters: {characters_str}
Setting: {story.get('setting', '')}
Theme: {story.get('theme', '')}
Age Range: {story.get('age_range', '4-10')}

REQUIREMENTS:
- Video Description: Engaging, kid-friendly description (150-300 words) that:
  * Introduces the story and main characters
  * Highlights the theme/lesson
  * Mentions it's an animated story suitable for kids
  * Includes a friendly call-to-action (like, subscribe, comment)
  * Uses emojis sparingly (2-3 max) to make it fun but not overwhelming
  * Is appropriate for parents and kids
  * Encourages engagement (comments about favorite characters, etc.)

- Tags: Comma-separated string of 10-15 relevant tags for YouTube discoverability:
  * Include story-specific tags (character names, theme, setting)
  * Include generic kids content tags (kids stories, animated stories, children's content)
  * Include educational/theme tags if applicable
  * Keep tags relevant and not spammy
  * Use common YouTube search terms for kids content
  * Format: "tag1,tag2,tag3,..." (single string with commas)

{{
  "description": "Full YouTube video description (150-300 words)",
  "tags": "tag1,tag2,tag3,tag4,tag5,..."
}}"""

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You are a YouTube content creator who writes engaging, kid-friendly video descriptions and tags for children's animated content. "},
            {"role": "user", "content": metadata_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    metadata = json.loads(clean_json_response(content))
    
    # Ensure tags is a comma-separated string (convert from list if needed)
    tags = metadata.get('tags', '')
    if isinstance(tags, list):
        tags = ','.join(tags)
        metadata['tags'] = tags
    
    print(f"[METADATA] Generated description ({len(metadata.get('description', ''))} chars)")
    print(f"[METADATA] Generated tags: {tags[:100]}..." if len(tags) > 100 else f"[METADATA] Generated tags: {tags}")
    
    return metadata


def build_scene_context(scene: dict, prev_scene: dict | None, next_scene: dict | None, story: dict) -> str:
    """
    Build context string for image/video generation to ensure continuity.
    
    Args:
        scene: Current scene dict
        prev_scene: Previous scene dict or None
        next_scene: Next scene dict or None
        story: Story dict with title, summary, characters, etc.
    
    Returns:
        Context string to include in prompts
    """
    context_parts = [
        f"STORY: {story.get('title', 'Untitled')}",
        f"SUMMARY: {story.get('summary', '')}",
        f"THEME: {story.get('theme', '')}",
        f"SETTING: {story.get('setting', '')}"
    ]
    
    # Add character descriptions
    characters = story.get('characters', [])
    if characters:
        char_descriptions = []
        for char in characters:
            if isinstance(char, dict):
                char_descriptions.append(f"{char.get('name', '?')}: {char.get('description', '')}")
            else:
                char_descriptions.append(str(char))
        context_parts.append(f"CHARACTERS: {', '.join(char_descriptions)}")
    
    # Add previous scene context
    if prev_scene:
        context_parts.append(f"\nPREVIOUS SCENE:")
        context_parts.append(f"  Title: {prev_scene.get('title', 'N/A')}")
        context_parts.append(f"  Description: {prev_scene.get('description', 'N/A')}")
        context_parts.append(f"  Visual: {prev_scene.get('image_prompt', 'N/A')[:200]}...")
    
    # Add next scene context
    if next_scene:
        context_parts.append(f"\nNEXT SCENE:")
        context_parts.append(f"  Title: {next_scene.get('title', 'N/A')}")
        context_parts.append(f"  Description: {next_scene.get('description', 'N/A')}")
    
    return "\n".join(context_parts)


def _save_kids_script(
    output_path: Path,
    story: dict,
    scenes: list,
    *,
    video_description: str = "",
    tags: list | None = None,
) -> None:
    """Write current script state to JSON so progress is saved if a later step fails."""
    if tags is None:
        tags = []
    output_data = {
        "metadata": {
            "title": story.get("title", "Untitled Story"),
            "summary": story.get("summary", ""),
            "characters": story.get("characters", []),
            "setting": story.get("setting", ""),
            "theme": story.get("theme", ""),
            "age_range": story.get("age_range", "4-10"),
            "style_prompt": story.get("style_prompt", ""),
            "music_description": story.get("music_description", ""),
            "num_scenes": len(scenes),
            "total_duration_estimate": len(scenes) * SCENE_DURATION,
            "video_description": video_description,
            "tags": tags,
        },
        "scenes": scenes,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def generate_kids_script(prompt: str, output_path: str | None = None, num_scenes: int = DEFAULT_NUM_SCENES) -> dict:
    """
    Main orchestrator: Generate complete story and scenes, save to JSON.
    
    Args:
        prompt: User prompt for the story
        output_path: Optional output path (defaults to kids_scripts/{sanitized_prompt}_story.json)
        num_scenes: Number of scenes to generate (default: 20)
    
    Returns:
        Complete story dict with scenes
    """
    print(f"\n{'='*60}")
    print(f"[KIDS SCRIPT] Generating story from prompt: {prompt}")
    print(f"[CONFIG] Scenes: {num_scenes} (~{num_scenes * SCENE_DURATION} seconds / ~{num_scenes * SCENE_DURATION / 60:.1f} minutes)")
    print(f"{'='*60}")
    
    # Determine output path early so we can save progress if a later step fails
    if output_path is None:
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in prompt)
        safe_name = safe_name.replace(' ', '_').lower()[:50]  # Limit length
        output_path = SCRIPTS_DIR / f"{safe_name}_story.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = SCRIPTS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate story structure
    story = generate_kids_story(prompt, num_scenes=num_scenes)
    # Save progress (story only) in case scene generation or metadata fails
    _save_kids_script(output_path, story, [], video_description="", tags=[])
    print(f"[SCRIPT] Saved progress: {output_path} (story only)")
    
    # Step 2: Generate scenes
    scenes = generate_kids_scenes(story, num_scenes=num_scenes)
    # Save progress (story + scenes) in case metadata generation fails
    _save_kids_script(output_path, story, scenes, video_description="", tags=[])
    print(f"[SCRIPT] Saved progress: {output_path} ({len(scenes)} scenes)")
    
    # Step 3: Generate video metadata (description and tags)
    video_metadata = generate_video_metadata(story, scenes)
    
    # Combine story and scenes
    output_data = {
        "metadata": {
            "title": story.get("title", "Untitled Story"),
            "summary": story.get("summary", ""),
            "characters": story.get("characters", []),
            "setting": story.get("setting", ""),
            "theme": story.get("theme", ""),
            "age_range": story.get("age_range", "4-10"),
            "style_prompt": story.get("style_prompt", ""),
            "music_description": story.get("music_description", ""),
            "num_scenes": len(scenes),
            "total_duration_estimate": len(scenes) * SCENE_DURATION,
            "video_description": video_metadata.get("description", ""),
            "tags": video_metadata.get("tags", [])
        },
        "scenes": scenes
    }
    
    # Save final
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Store output path in metadata for reference
    output_data['_output_path'] = str(output_path)
    
    print(f"\n[SCRIPT] Saved: {output_path}")
    print(f"[SCRIPT] Total scenes: {len(scenes)}")
    print(f"[SCRIPT] Estimated duration: {len(scenes) * SCENE_DURATION} seconds (~{len(scenes) * SCENE_DURATION / 60:.1f} minutes)")
    
    return output_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate kid-friendly animated story scripts for YouTube Shorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate story from prompt (default: 20 scenes)
  python build_script_kids.py "A brave little robot explores a magical forest"
  
  # Generate story with custom number of scenes
  python build_script_kids.py "A friendly dragon learns to share" --scenes 15
  
  # Generate story with custom output path
  python build_script_kids.py "A friendly dragon learns to share" -o robot_story.json --scenes 25
        """
    )
    
    parser.add_argument("prompt", help="Story prompt (e.g., 'A brave little robot explores a magical forest')")
    parser.add_argument("-o", "--output", help="Output JSON file path (default: kids_scripts/{prompt}_story.json)")
    parser.add_argument("--scenes", type=int, default=None,
                        help=f"Number of scenes to generate (default: {DEFAULT_NUM_SCENES}, or KIDS_SCENES_COUNT env var)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)
    
    # Determine number of scenes (command line > env var > default)
    num_scenes = args.scenes if args.scenes is not None else DEFAULT_NUM_SCENES
    
    try:
        script_data = generate_kids_script(args.prompt, args.output, num_scenes=num_scenes)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        output_file = args.output or script_data.get('_output_path', 'kids_scripts/story.json')
        
        print(f"\n📖 STORY:")
        print(f"   Script: {output_file}")
        print(f"   Scenes: {script_data['metadata']['num_scenes']}")
        print(f"   Duration: ~{script_data['metadata']['total_duration_estimate']} seconds")
        
        print(f"\n🎬 Next step - Generate videos:")
        print(f"   python build_video_kids.py {output_file} output.mp4")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
