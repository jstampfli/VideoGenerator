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
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Model configuration
SCRIPT_MODEL = os.getenv("KIDS_SCRIPT_MODEL", "gpt-5.2")
IMG_MODEL = os.getenv("KIDS_IMAGE_MODEL", "gpt-image-1.5")
DEFAULT_NUM_SCENES = int(os.getenv("KIDS_SCENES_COUNT", "20"))
SCENE_DURATION = int(os.getenv("KIDS_SCENE_DURATION", "8")) - 4

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
  * Voice description: Very specific, detailed description of the character's voice to ensure consistency across all scenes. Include:
    - Age and gender (e.g., "young boy, 6 years old", "wise old woman, 70s")
    - Vocal pitch/register (e.g., "high-pitched", "mid-range", "deep", "squeaky")
    - Voice quality/timbre (e.g., "smooth and warm", "raspy", "crisp and clear", "soft and airy", "robotic with mechanical undertones")
    - Energy level and pace (e.g., "energetic and fast-paced", "calm and slow", "nervous and quick", "confident and steady")
    - Emotional tone (e.g., "friendly and cheerful", "serious and wise", "anxious and worried", "playful and upbeat")
    - Any distinctive characteristics (e.g., "slight accent", "breathy quality", "metallic resonance", "childlike lisp")
    - Example: "young robot, 7 years old, mid-range pitch with mechanical resonance, smooth and steady pace, friendly and curious tone, slight metallic echo"
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

Respond with JSON:
{{
  "title": "Story title",
  "summary": "Brief story summary (2-3 sentences)",
  "characters": [
    {{
      "name": "Character name",
      "description": "Simple, clear visual appearance description only (what they look like: size, shape, colors, key features). No personality traits.",
      "voice_description": "Very specific, detailed voice description including age, gender, vocal pitch/register, voice quality/timbre, energy level, pace, emotional tone, and any distinctive characteristics. Example: 'young robot, 7 years old, mid-range pitch with mechanical resonance, smooth and steady pace, friendly and curious tone, slight metallic echo'"
    }}
  ],
  "setting": "Where the story takes place",
  "theme": "Main lesson/message",
  "age_range": "4-10",
  "music_description": "Detailed description of background music style, instrumentation, tempo, and mood that fits this story",
  "style_prompt": "Detailed Pixar-style visual description for animation"
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a children's story writer who creates engaging, age-appropriate animated stories with positive themes. Respond with valid JSON only."},
            {"role": "user", "content": story_prompt}
        ],
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    
    story = json.loads(clean_json_response(response.choices[0].message.content))
    
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
- Each scene should be ~{SCENE_DURATION} seconds when animated
- Scenes must flow smoothly from one to the next (continuity is critical)
- Each scene should advance the story
- Include clear visual descriptions for both static images and animated video
- Character appearances must be CONSISTENT across all scenes
- Bright, colorful, animated cartoon style suitable for kids
- Age-appropriate content (ages 4-10)
- Positive, engaging tone

CRITICAL FOR VIDEO CONTINUITY:
- Each video scene MUST start exactly where the previous scene ended
- If scene 1 ends with "the girl walks to the clearing", then scene 2 MUST start with "the girl in the clearing"
- If scene 2 ends with "the girl looks in the bush beside the clearing", then scene 3 MUST start with "the girl looking in the bush"
- Think of it like a continuous film: the last frame of scene N becomes the first frame of scene N+1
- Character positions, poses, and the visual state at the end of one scene = the starting state of the next scene

For each scene, provide:
- "id": Scene number (1 to {num_scenes})
- "title": Short scene title (2-4 words)
- "description": What happens in this scene (1-2 sentences)
- "image_prompt": (ONLY for scene 1) Detailed visual description for generating the INITIAL static image - this should be a snapshot of the VERY FIRST FRAME/INSTANT of scene 1, matching ONLY the first moment described in the video_prompt. Include ONLY characters and elements that are present at the absolute beginning, before any action or dialogue occurs. Include:
  * Character appearances and initial positions (ONLY characters present at the very start, before any events happen)
  * Setting/location at the beginning
  * Composition and framing of the opening frame
  * Colors and mood of the starting moment
  * Style: animated, cartoon-like, bright and colorful
  * Format: 9:16 vertical (1080x1920) for YouTube Shorts
  * CRITICAL: Match EXACTLY the first moment described in video_prompt. If video_prompt says "Bolt rolls forward, then Luma darts in", the image should show ONLY Bolt at the start, NOT Luma. Do NOT include characters, objects, or elements that appear later in the scene. Only describe what exists in the absolute first frame before any action begins.
- "video_prompt": Simple, direct description of what happens in the video. Use clear, straightforward language. Tell the story with the video. Keep it concise and focused on what viewers will see. IMPORTANT: Each scene is ~{SCENE_DURATION} seconds long, so limit the number of events to what can realistically happen in that time. Include:
  * For scene 1: What happens from start to finish. Describe the action clearly.
  * For scenes 2+: How the scene STARTS (matching the end of previous scene) + what happens + how it ENDS
  * Use direct, simple language. Limit to 2-4 key actions per scene. Example: "The girl walks into the clearing. She looks around and sees a bunny rabbit." (Simple, fits in ~{SCENE_DURATION} seconds)
  * CRITICAL: Don't overstuff scenes with too many actions - pick only the most important 2-4 events. Too many events will cause the video to feel rushed or get cut off.
  * IMPORTANT: Clearly describe the FINAL STATE/FRAME of each scene so the next scene can start there
  * CRITICAL: When referencing characters by name, include a shortened character description after the first mention in each sentence. Example: Instead of "Beep-Bop puts his hands on the log", write "Beep-Bop, the small robot, puts his hands on the log". Instead of "Luma flies over", write "Luma, the glowing sprite, flies over". This helps the video generation understand who each character is visually.
  * CRITICAL: If characters speak during the scene, include the dialogue directly in the video_prompt. Integrate it naturally into the action description, specifying when and what each character says. Example: "Luma, the tiny forest sprite, appears and says: 'I'm worried I'm losing my light!' The girl responds: 'Don't worry, we'll help you!'" (Keep dialogue concise - 1-2 exchanges max per scene)

CRITICAL FOR CONTINUITY:
- Characters must look the same across all scenes (same appearance, clothing, colors)
- Settings should transition naturally between scenes
- Each scene should logically flow from the previous one
- The final scene should provide a satisfying conclusion

Respond with JSON:
{{
  "scenes": [
    {{
      "id": 1,
      "title": "Scene title",
      "description": "What happens",
      "image_prompt": "Detailed visual description of the VERY FIRST FRAME of scene 1 (ONLY for scene 1). Match EXACTLY the first moment of video_prompt. Include ONLY characters/elements present at the absolute start. Example: If video_prompt says 'Bolt rolls forward, then Luma darts in', show ONLY Bolt at the start, NOT Luma. If video_prompt says 'The girl walks into the forest, then a bunny appears', show ONLY the girl entering, NOT the bunny.",
      "video_prompt": "Simple, direct description with 2-4 key actions that fit in ~{SCENE_DURATION} seconds. Use shortened character descriptions after character names. The girl enters the forest. She walks along the path. Luma, the tiny glowing sprite, appears and says: 'Welcome to the magical forest!'",
      "duration_estimate": {SCENE_DURATION}
    }},
    {{
      "id": 2,
      "title": "Scene title",
      "description": "What happens",
      "video_prompt": "Simple, direct description with 2-4 key actions that fit in ~{SCENE_DURATION} seconds. Starting from where scene 1 ended. Use shortened character descriptions after character names. The girl in the clearing looks around. A bunny rabbit, small and fluffy, appears from behind a bush and says: 'Hello there!'",
      "duration_estimate": {SCENE_DURATION}
    }},
    ...
    (exactly {num_scenes} scenes, only scene 1 has image_prompt. Dialogue should be integrated directly into video_prompt when characters speak.)
  ]
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a children's story animator who creates simple, clear scene descriptions for animated videos. CRITICAL: Use direct, straightforward language in video_prompt - tell the story clearly with simple sentences. Each scene's video must start exactly where the previous scene ended - like a continuous film where the last frame becomes the first frame. When characters speak, include their dialogue directly in the video_prompt, integrated naturally into the action description. Character descriptions should focus ONLY on visual appearance (what they look like: size, shape, colors, features), not personality traits. CRITICAL for image_prompt: It must match EXACTLY the very first moment of video_prompt - include ONLY characters/elements present at the absolute beginning before any actions or events occur. If video_prompt introduces characters later (e.g., 'Bolt rolls, then Luma darts in'), the image_prompt should show ONLY the initial characters (just Bolt), NOT characters that appear later (NOT Luma). Only scene 1 needs an image_prompt (for initial image generation). All other scenes flow directly from the previous video's ending. Ensure character consistency and smooth, seamless transitions. Respond with valid JSON only."},
            {"role": "user", "content": scenes_prompt}
        ],
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(clean_json_response(response.choices[0].message.content))
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

Respond with JSON:
{{
  "description": "Full YouTube video description (150-300 words)",
  "tags": "tag1,tag2,tag3,tag4,tag5,..."
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTube content creator who writes engaging, kid-friendly video descriptions and tags for children's animated content. Respond with valid JSON only."},
            {"role": "user", "content": metadata_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    metadata = json.loads(clean_json_response(response.choices[0].message.content))
    
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
    
    # Step 1: Generate story structure
    story = generate_kids_story(prompt, num_scenes=num_scenes)
    
    # Step 2: Generate scenes
    scenes = generate_kids_scenes(story, num_scenes=num_scenes)
    
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
    
    # Determine output path
    if output_path is None:
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in prompt)
        safe_name = safe_name.replace(' ', '_').lower()[:50]  # Limit length
        output_path = SCRIPTS_DIR / f"{safe_name}_story.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = SCRIPTS_DIR / output_path
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
