import os
import sys
import json
import base64
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ------------- CONFIG -------------

load_dotenv()  # Load API key from .env file
client = OpenAI()

SCRIPT_MODEL = "gpt-5-mini"  # Using gpt-4o for better quality and longer context
IMG_MODEL = "gpt-image-1-mini"

THUMBNAILS_DIR = Path("thumbnails")
SHORTS_DIR = Path("shorts")

NO_TEXT_CONSTRAINT = """
CRITICAL: Do NOT include any text, words, letters, numbers, titles, labels, watermarks, or any written content in the image. The image must be completely text-free."""


# Generation settings (can be overridden via command line)
class Config:
    # Main video settings
    chapters = 10           # Number of outline chapters
    scenes_per_chapter = 4  # Scenes per chapter (total = chapters * scenes_per_chapter)
    generate_main = True    # Whether to generate main video
    
    # Shorts settings
    num_shorts = 3              # Number of YouTube Shorts
    short_chapters = 1          # Chapters per short (keep at 1 for now: hook → build → build → cliffhanger)
    short_scenes_per_chapter = 4  # Scenes per chapter in shorts
    
    generate_thumbnails = True  # Whether to generate thumbnail images
    
    @property
    def total_scenes(self):
        return self.chapters * self.scenes_per_chapter
    
    @property
    def total_short_scenes(self):
        return self.short_chapters * self.short_scenes_per_chapter

config = Config()


def clean_json_response(content: str) -> str:
    """Remove markdown code blocks from JSON response."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def generate_thumbnail(prompt: str, output_path: Path, size: str = "1024x1024") -> Path | None:
    """Generate a thumbnail image and save it."""
    if not config.generate_thumbnails:
        print(f"[THUMBNAIL] Skipped (--no-thumbnails)")
        return None
    
    full_prompt = prompt + NO_TEXT_CONSTRAINT
    
    try:
        resp = client.images.generate(
            model=IMG_MODEL,
            prompt=full_prompt,
            size=size,
            n=1
        )
        
        b64_data = resp.data[0].b64_json
        img_bytes = base64.b64decode(b64_data)
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        
        print(f"[THUMBNAIL] Saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"[WARNING] Failed to generate thumbnail: {e}")
        return None


def generate_outline(person_of_interest: str) -> dict:
    """Generate a detailed chronological outline of the person's life."""
    print(f"\n[OUTLINE] Generating {config.chapters}-chapter outline...")
    
    outline_prompt = f"""You are a master documentary filmmaker. Create a compelling narrative outline for a ~20 minute documentary about: {person_of_interest}

This will be a {config.total_scenes}-scene documentary with EXACTLY {config.chapters} chapters. Think of this as a FEATURE FILM with continuous story arcs, not disconnected episodes.

NARRATIVE STRUCTURE:
- The documentary should feel like ONE CONTINUOUS STORY, not a list of facts
- Each chapter should FLOW into the next with clear cause-and-effect
- Plant SEEDS in early chapters that PAY OFF later (foreshadowing)
- Build RECURRING THEMES that echo throughout (e.g., isolation, ambition, sacrifice)
- Create emotional MOMENTUM that builds to a climax around chapter 7-8, then resolves

For each of the {config.chapters} chapters, provide:
- "chapter_num": 1-{config.chapters}
- "title": A compelling chapter title
- "year_range": The years covered
- "summary": 2-3 sentences about what happens
- "key_events": 4-6 specific dramatic moments to show
- "emotional_tone": The mood of this chapter
- "dramatic_tension": What conflict drives this chapter
- "connects_to_next": How this chapter sets up or flows into the next one
- "recurring_threads": Which themes/motifs from earlier chapters appear here

STORY ARC REQUIREMENTS:
1. Chapter 1 - THE HOOK: A rapid-fire "trailer" of their life. Tease the MOST interesting facts, achievements, controversies, and dramatic moments that will be covered. Make viewers think "I NEED to know more." This is NOT chronological - it's a highlight reel that hooks the audience. End with something like "But how did they get here? Let's start from the beginning..."
2. Chapters 2-4: Early life and rising action - origins, struggles, first successes, growing stakes
3. Chapters 5-7: Peak conflict - major breakthroughs AND major crises
4. Chapters 8-9: Resolution and consequences
5. Chapter 10: Legacy and emotional conclusion that echoes the hook

CRITICAL:
- Every chapter must CONNECT to what came before and set up what comes after
- Include SPECIFIC details - dates, names, places, quotes, sensory details
- Focus on HUMAN drama - relationships, emotions, internal conflicts
- Make it feel like watching a movie, not reading a textbook

Respond with JSON:
{{
  "person": "{person_of_interest}",
  "birth_year": YYYY,
  "death_year": YYYY or null,
  "tagline": "One compelling sentence that captures their story",
  "central_theme": "The overarching theme that ties the whole documentary together",
  "narrative_arc": "Brief description of the emotional journey from start to finish",
  "chapters": [
    {{
      "chapter_num": 1,
      "title": "...",
      "year_range": "...",
      "summary": "...",
      "key_events": ["...", ...],
      "emotional_tone": "...",
      "dramatic_tension": "...",
      "connects_to_next": "...",
      "recurring_threads": ["...", ...]
    }},
    ... ({config.chapters} chapters total)
  ]
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a historian who finds the drama and humanity in every life story. Respond with valid JSON only."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    outline_data = json.loads(clean_json_response(response.choices[0].message.content))
    chapters = outline_data.get("chapters", [])
    
    print(f"[OUTLINE] Generated {len(chapters)} chapters")
    for ch in chapters:
        print(f"  Ch {ch['chapter_num']}: {ch['title']} ({ch['year_range']})")
    
    return outline_data


def generate_scenes_for_chapter(person: str, chapter: dict, scenes_per_chapter: int, start_id: int, 
                                 global_style: str, prev_chapter: dict = None, prev_scenes: list = None) -> list[dict]:
    """Generate scenes for a single chapter of the outline with continuity context."""
    
    # Build context from previous chapter for smooth transitions
    chapter_num = chapter.get('chapter_num', 1)
    is_hook_chapter = (chapter_num == 1)
    
    if prev_chapter:
        prev_context = f"""PREVIOUS CHAPTER (for continuity):
Chapter {prev_chapter['chapter_num']}: "{prev_chapter['title']}" ({prev_chapter['year_range']})
Summary: {prev_chapter['summary']}
Emotional Tone: {prev_chapter['emotional_tone']}
"""
    elif is_hook_chapter:
        prev_context = """THIS IS THE HOOK CHAPTER - a rapid-fire "trailer" for the documentary.
Tease the most shocking, interesting, and dramatic moments from their ENTIRE life.
This is NOT chronological - jump around to the highlights that will make viewers stay.
End with a transition like "But how did it all begin?" to set up Chapter 2."""
    else:
        prev_context = "This is the OPENING chapter - establish the story with impact!"
    
    # Include last few scenes for continuity
    if prev_scenes and len(prev_scenes) > 0:
        # Get last 3 scenes for context
        recent_scenes = prev_scenes[-3:]
        scenes_context = "LAST FEW SCENES (maintain continuity, don't repeat):\n"
        for sc in recent_scenes:
            scenes_context += f"  Scene {sc.get('id')}: {sc.get('title')} - \"{sc.get('narration', '')[:100]}...\"\n"
    else:
        scenes_context = ""
    
    # Get narrative context
    connects_to_next = chapter.get('connects_to_next', '')
    recurring_threads = chapter.get('recurring_threads', [])
    threads_str = ', '.join(recurring_threads) if recurring_threads else 'none specified'
    
    scene_prompt = f"""You are writing scenes {start_id}-{start_id + scenes_per_chapter - 1} of a {config.total_scenes}-scene documentary about {person}.

This is a CONTINUOUS NARRATIVE - scenes should flow like a movie, not feel like separate segments.

{prev_context}
{scenes_context}
NOW WRITING CHAPTER {chapter['chapter_num']} of {config.chapters}: "{chapter['title']}"
Time Period: {chapter['year_range']}
Emotional Tone: {chapter['emotional_tone']}
Dramatic Tension: {chapter['dramatic_tension']}
Recurring Themes: {threads_str}
Sets Up Next Chapter: {connects_to_next}

Chapter Summary: {chapter['summary']}

Key Events to Dramatize:
{chr(10).join(f"• {event}" for event in chapter.get('key_events', []))}

Generate EXACTLY {scenes_per_chapter} scenes that FLOW CONTINUOUSLY.

{"HOOK CHAPTER SPECIAL RULES:" if is_hook_chapter else "CONTINUOUS FLOW REQUIREMENTS:"}
{'''- This is the "trailer" - rapid-fire highlights from their ENTIRE life
- Jump between their most interesting achievements, controversies, and moments
- NOT chronological - pick the most jaw-dropping facts regardless of when they happened
- Each scene should make the viewer think "Wait, WHAT? I need to know more"
- Final scene should transition to "Let's go back to the beginning..."
- Pacing: fast, punchy, exciting - like a movie trailer''' if is_hook_chapter else '''- Scene 1 should TRANSITION smoothly from where the last chapter ended
- Each scene should CONNECT to the next - end with forward momentum
- Reference recurring themes/motifs from earlier in the documentary
- The final scene should SET UP the next chapter's content
- Think of scenes as continuous shots in a film, not separate segments'''}

SCENE-TO-SCENE TRANSITIONS:
- Use temporal connectors: "Days later...", "That same evening...", "Meanwhile..."
- Use emotional connectors: build on feelings from the previous scene
- Use narrative connectors: cause → effect, action → consequence
- Avoid abrupt topic changes - each scene should feel inevitable after the last

SCENE REQUIREMENTS:
1. SPECIFIC FACTS - names, dates, places, numbers, amounts
2. CONCRETE details - what exactly happened, who was there, what was said
3. INTERESTING information the viewer likely doesn't know
4. CAUSE and EFFECT - why did this happen, what resulted
5. NO filler, NO fluff, NO vague statements
6. Every sentence must contain NEW information

NARRATION STYLE - CRITICAL:
- SIMPLE, CLEAR language. Write for a general audience.
- AVOID flowery, artistic, or poetic language
- AVOID vague phrases like "little did he know", "destiny awaited", "the world would never be the same"
- NO dramatic pauses or buildup - just deliver the facts engagingly
- Use present tense: "Einstein submits his paper to the journal..."
- 2-3 sentences per scene (~12-18 seconds of narration)
- Pack MAXIMUM information into minimum words
- CRITICAL: This is SPOKEN narration for text-to-speech. Do NOT include:
  * Film directions like "Smash cut—", "Hard cut to", "Cut to:", "Fade in:"
  * Camera directions like "Close-up of", "Wide shot:", "Pan to"
  * Any production/editing terminology
  Write ONLY words that should be spoken by a narrator's voice.

BAD example: "In the quiet of his study, a revolution was brewing in Einstein's mind."
GOOD example: "In 1905, Einstein publishes four papers that redefine physics - including E=mc², proving mass and energy are the same thing."

IMAGE PROMPT STYLE:
- Cinematic, dramatic lighting
- Specific composition (close-up, wide shot, over-shoulder, etc.)
- Mood and atmosphere matching the scene
- Period-accurate details
- End with ", 16:9 cinematic"

Respond with JSON array:
[
  {{
    "id": {start_id},
    "title": "Evocative 2-5 word title",
    "narration": "Vivid, dramatic narration...",
    "image_prompt": "Detailed visual description, 16:9 cinematic"
  }},
  ...
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a master documentary storyteller. Create vivid, emotionally engaging scenes. Respond with valid JSON array only."},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.85,
    )
    
    scenes = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    return scenes


def generate_short_outline(person: str, main_outline: dict, short_num: int, total_shorts: int) -> dict:
    """Generate a focused outline for a single YouTube Short."""
    
    chapters_context = "\n".join([
        f"• {ch['title']} ({ch['year_range']}): {ch['summary']}"
        for ch in main_outline.get("chapters", [])
    ])
    
    # Different focus for each short
    focus_options = [
        "the most SHOCKING or little-known fact",
        "the most DRAMATIC conflict or rivalry",
        "their most MIND-BLOWING achievement",
        "a TRAGIC or heartbreaking moment",
        "a MYSTERIOUS or unexplained aspect",
    ]
    focus = focus_options[(short_num - 1) % len(focus_options)]
    
    outline_prompt = f"""Create a VIRAL YouTube Short about {person}.

Short #{short_num} of {total_shorts}. Focus on: {focus}

Their life story (for context):
{chapters_context}

This Short has EXACTLY 4 scenes with this structure:
1. HOOK: Start with the most surprising/shocking fact. Grab attention in 3 seconds.
2. BUILD 1: Add context that makes the hook more interesting. Specific details.
3. BUILD 2: Escalate - another surprising fact or consequence. Keep stacking interesting info.
4. CLIFFHANGER: End with unresolved tension or "but that's not even the craziest part..."

CRITICAL RULES:
- Every scene must contain a SPECIFIC, INTERESTING FACT
- NO vague statements or filler
- NO artistic/poetic language
- Simple, clear sentences packed with information
- The short should feel like rapid-fire interesting facts that connect together

Provide JSON:
{{
  "short_title": "VIRAL title (max 50 chars) - shocking, specific",
  "short_description": "YouTube description (100 words) with hashtags",
  "tags": "10-15 SEO tags comma-separated",
  "thumbnail_prompt": "Vertical 9:16, dramatic, mobile-optimized",
  "hook_fact": "The ONE shocking fact that opens the short",
  "story_angle": "What specific story/incident are we telling",
  "key_facts": ["4-6 specific facts to include across the 4 scenes"]
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You create viral content. Every word must deliver value. No fluff. Respond with valid JSON only."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.9,
        response_format={"type": "json_object"}
    )
    
    return json.loads(clean_json_response(response.choices[0].message.content))


def generate_short_scenes(person: str, short_outline: dict) -> list[dict]:
    """Generate all 4 scenes for a YouTube Short (hook, build, build, cliffhanger)."""
    
    key_facts = short_outline.get('key_facts', [])
    facts_str = "\n".join(f"• {fact}" for fact in key_facts)
    
    scene_prompt = f"""Write 4 scenes for a YouTube Short about {person}.

TITLE: "{short_outline.get('short_title', '')}"
OPENING HOOK: {short_outline.get('hook_fact', '')}
STORY ANGLE: {short_outline.get('story_angle', '')}

KEY FACTS TO USE:
{facts_str}

STRUCTURE (exactly 4 scenes):
1. HOOK: Open with the most shocking/surprising fact. Grab attention immediately.
2. BUILD 1: Add context - why is this interesting? Specific details.
3. BUILD 2: Escalate - another surprising consequence or related fact.
4. CLIFFHANGER: End with "But that's not even the craziest part..." or similar hook to full video.

NARRATION RULES - CRITICAL:
- 1-2 SHORT sentences per scene (~8-10 seconds when spoken)
- SIMPLE language - no fancy words, no poetry
- SPECIFIC facts - names, numbers, dates, places
- NO filler phrases like "little did he know" or "destiny awaited"
- NO vague statements - every sentence must have concrete information
- Present tense: "Einstein walks into the patent office..."
- CRITICAL: Do NOT include film directions (Cut to, Fade, Smash cut) or camera directions.

BAD: "The genius stood on the precipice of destiny, unaware that fate had other plans."
GOOD: "Einstein is 26 years old, working at a patent office, and about to change physics forever."

IMAGE PROMPTS:
- Vertical 9:16, dramatic, mobile-optimized
- High contrast, single clear subject
- End with ", 9:16 vertical"

Respond with JSON array of exactly 4 scenes:
[
  {{"id": 1, "title": "2-4 words", "narration": "...", "image_prompt": "..."}},
  {{"id": 2, "title": "...", "narration": "...", "image_prompt": "..."}},
  {{"id": 3, "title": "...", "narration": "...", "image_prompt": "..."}},
  {{"id": 4, "title": "...", "narration": "...", "image_prompt": "..."}}
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You write viral content. Simple words, specific facts, no fluff. Respond with valid JSON array only."},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.85,
    )
    
    scenes = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    return scenes


def generate_shorts(person_of_interest: str, main_title: str, global_block: str, outline: dict, base_output_path: str):
    """Generate YouTube Shorts (4 scenes each: hook, build, build, cliffhanger)."""
    if config.num_shorts == 0:
        print("\n[SHORTS] Skipped (--shorts 0)")
        return []
    
    print(f"\n{'='*60}")
    print(f"[SHORTS] Generating {config.num_shorts} YouTube Short(s)")
    print(f"[SHORTS] Structure: {config.total_short_scenes} scenes each (hook → build → build → cliffhanger)")
    print(f"{'='*60}")
    
    SHORTS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    
    generated_shorts = []
    base_name = Path(base_output_path).stem.replace("_script", "")
    
    for short_num in range(1, config.num_shorts + 1):
        print(f"\n[SHORT {short_num}/{config.num_shorts}] Creating outline...")
        
        # Step 1: Generate short outline
        short_outline = generate_short_outline(
            person=person_of_interest,
            main_outline=outline,
            short_num=short_num,
            total_shorts=config.num_shorts
        )
        
        short_title = short_outline.get("short_title", f"Short {short_num}")
        print(f"[SHORT {short_num}] Title: {short_title}")
        print(f"[SHORT {short_num}] Hook: {short_outline.get('hook_fact', '')[:60]}...")
        
        # Step 2: Generate all scenes
        print(f"[SHORT {short_num}] Generating {config.total_short_scenes} scenes...")
        
        try:
            all_scenes = generate_short_scenes(
                person=person_of_interest,
                short_outline=short_outline
            )
            print(f"[SHORT {short_num}] → {len(all_scenes)} scenes generated")
            
        except Exception as e:
            print(f"[SHORT {short_num}] ERROR generating scenes: {e}")
            raise
        
        # Fix scene IDs
        for i, scene in enumerate(all_scenes):
            scene["id"] = i + 1
        
        # Step 3: Generate thumbnail
        thumbnail_prompt = short_outline.get("thumbnail_prompt", "")
        thumbnail_path = None
        if thumbnail_prompt:
            thumbnail_prompt += "\n\nYouTube Shorts thumbnail. Vertical 9:16, bold, dramatic, mobile-optimized, single powerful image."
            thumb_file = THUMBNAILS_DIR / f"{base_name}_short{short_num}_thumbnail.png"
            thumbnail_path = generate_thumbnail(thumbnail_prompt, thumb_file, size="1024x1536")
        
        # Step 4: Save short
        short_output = {
            "metadata": {
                "short_id": short_num,
                "title": short_title,
                "description": short_outline.get("short_description", ""),
                "tags": short_outline.get("tags", ""),
                "hook_fact": short_outline.get("hook_fact", ""),
                "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
                "story_angle": short_outline.get("story_angle", ""),
                "person_of_interest": person_of_interest,
                "main_video_title": main_title,
                "global_block": global_block,
                "num_scenes": len(all_scenes),
                "outline": short_outline
            },
            "scenes": all_scenes
        }
        
        short_file = SHORTS_DIR / f"{base_name}_short{short_num}.json"
        with open(short_file, "w", encoding="utf-8") as f:
            json.dump(short_output, f, indent=2, ensure_ascii=False)
        
        print(f"[SHORT {short_num}] ✓ Saved: {short_file} ({len(all_scenes)} scenes)")
        
        generated_shorts.append({
            "file": str(short_file),
            "title": short_title,
            "scenes": len(all_scenes)
        })
    
    return generated_shorts


def generate_script(person_of_interest: str, output_path: str):
    """Generate a complete documentary script using outline-guided generation."""
    
    print(f"\n{'='*60}")
    print(f"[SCRIPT] Generating script for: {person_of_interest}")
    print(f"{'='*60}")
    if config.generate_main:
        print(f"[CONFIG] Main video: {config.chapters} chapters × {config.scenes_per_chapter} scenes = {config.total_scenes} scenes")
    else:
        print(f"[CONFIG] Main video: SKIPPED")
    if config.num_shorts > 0:
        print(f"[CONFIG] Shorts: {config.num_shorts} × {config.total_short_scenes} scenes = {config.num_shorts * config.total_short_scenes} scenes")
    else:
        print(f"[CONFIG] Shorts: SKIPPED")
    print(f"[CONFIG] Thumbnails: {'Yes' if config.generate_thumbnails else 'No'}")
    print(f"[CONFIG] Model: {SCRIPT_MODEL}")
    
    # Step 1: Generate detailed outline (needed for both main and shorts)
    print("\n[STEP 1] Creating life outline...")
    outline = generate_outline(person_of_interest)
    chapters = outline.get("chapters", [])
    
    if config.generate_main and len(chapters) < config.chapters:
        print(f"[WARNING] Got {len(chapters)} chapters, expected {config.chapters}")
    
    # Step 2: Generate metadata
    print("\n[STEP 2] Generating video metadata...")
    
    metadata_prompt = f"""Create metadata for a documentary about: {person_of_interest}

Their story in one line: {outline.get('tagline', '')}

Generate JSON:
{{
  "title": "Compelling YouTube title (60-80 chars)",
  "video_description": "YouTube description (500-800 words) - hook, highlights, key facts, call to action. SEO optimized.",
  "tags": "15-20 SEO tags separated by commas. Mix of: person's name, key topics, related figures, time periods, achievements, fields (e.g., 'Albert Einstein, physics, relativity, Nobel Prize, Germany, Princeton, E=mc2, quantum mechanics, genius, scientist, 20th century, biography, documentary')",
  "thumbnail_description": "Thumbnail visual: composition, colors, mood, subject appearance. NO TEXT in image.",
  "global_block": "Visual style guide (300-400 words): semi-realistic digital painting style, color palette, dramatic lighting, how {person_of_interest} should appear consistently across {config.total_scenes} scenes."
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "Documentary producer. Respond with valid JSON only."},
            {"role": "user", "content": metadata_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    metadata = json.loads(clean_json_response(response.choices[0].message.content))
    
    title = metadata["title"]
    video_description = metadata.get("video_description", "")
    tags = metadata.get("tags", "")
    thumbnail_description = metadata["thumbnail_description"]
    global_block = metadata["global_block"]
    
    print(f"[METADATA] Title: {title}")
    print(f"[METADATA] Tags: {tags[:80]}..." if len(tags) > 80 else f"[METADATA] Tags: {tags}")
    
    # Generate main video thumbnail (only if generating main video)
    generated_thumb = None
    if config.generate_main:
        print("\n[THUMBNAIL] Main video thumbnail...")
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        thumbnail_path = THUMBNAILS_DIR / f"{Path(output_path).stem}_thumbnail.png"
        
        thumbnail_prompt = f"""{thumbnail_description}

YouTube thumbnail. Cinematic, dramatic, high contrast. Optimized for small sizes."""
        
        generated_thumb = generate_thumbnail(thumbnail_prompt, thumbnail_path)
    
    # Step 3: Generate scenes chapter by chapter (only if generating main video)
    all_scenes = []
    
    if config.generate_main:
        print(f"\n[STEP 3] Generating {config.total_scenes} scenes from {len(chapters)} chapters...")
        
        for i, chapter in enumerate(chapters):
            start_id = len(all_scenes) + 1
            
            # Get previous chapter and scenes for continuity
            prev_chapter = chapters[i - 1] if i > 0 else None
            
            print(f"\n[CHAPTER {chapter['chapter_num']}/{len(chapters)}] {chapter['title']}")
            print(f"  Generating {config.scenes_per_chapter} scenes...")
            
            try:
                scenes = generate_scenes_for_chapter(
                    person=person_of_interest,
                    chapter=chapter,
                    scenes_per_chapter=config.scenes_per_chapter,
                    start_id=start_id,
                    global_style=global_block,
                    prev_chapter=prev_chapter,
                    prev_scenes=list(all_scenes)  # Copy of scenes so far
                )
                
                if len(scenes) != config.scenes_per_chapter:
                    print(f"  [WARNING] Got {len(scenes)} scenes, expected {config.scenes_per_chapter}")
                
                all_scenes.extend(scenes)
                print(f"  ✓ {len(scenes)} scenes (total: {len(all_scenes)})")
                
            except Exception as e:
                print(f"  [ERROR] Failed: {e}")
                raise
        
        print(f"\n[SCRIPT] Total scenes: {len(all_scenes)}")
        
        # Validate and fix scene IDs
        for i, scene in enumerate(all_scenes):
            scene["id"] = i + 1
            for field in ["title", "narration", "image_prompt"]:
                if field not in scene:
                    raise ValueError(f"Scene {i+1} missing: {field}")
    else:
        print("\n[STEP 3] Skipping main video scene generation...")
    
    # Step 4: Generate Shorts
    print("\n[STEP 4] Generating YouTube Shorts...")
    shorts_info = generate_shorts(person_of_interest, title, global_block, outline, output_path)
    
    # Step 5: Save
    print("\n[STEP 5] Saving script...")
    
    output_data = {
        "metadata": {
            "title": title,
            "video_description": video_description,
            "tags": tags,
            "thumbnail_description": thumbnail_description,
            "thumbnail_path": str(generated_thumb) if generated_thumb else None,
            "global_block": global_block,
            "person_of_interest": person_of_interest,
            "num_scenes": len(all_scenes),
            "outline": outline,
            "shorts": shorts_info
        },
        "scenes": all_scenes
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SCRIPT] Saved: {output_path}")
    if config.generate_main:
        total_narration = sum(len(s['narration']) for s in all_scenes)
        print(f"[SCRIPT] Main video scenes: {len(all_scenes)}")
        print(f"[SCRIPT] Narration: {total_narration} chars (~{total_narration // 15} seconds)")
    if shorts_info:
        print(f"[SCRIPT] Shorts: {len(shorts_info)}")
    
    return output_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate documentary scripts with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full production run (40 scenes main video, 3 shorts with 4 scenes each)
  python build_script.py "Albert Einstein"

  # Quick test (4 scenes main, 1 short with 4 scenes, no thumbnails)
  python build_script.py "Albert Einstein" --test

  # Custom main video settings
  python build_script.py "Albert Einstein" --chapters 5 --scenes 3

  # Custom shorts settings  
  python build_script.py "Albert Einstein" --shorts 2 --short-scenes 4

  # No thumbnails (faster iteration)
  python build_script.py "Albert Einstein" --no-thumbnails

  # Only generate shorts (skip main video)
  python build_script.py "Albert Einstein" --shorts-only

  # Only generate main video (skip shorts)
  python build_script.py "Albert Einstein" --main-only
        """
    )
    
    parser.add_argument("person", help="Person to create documentary about")
    parser.add_argument("output", nargs="?", help="Output JSON file (default: <person>_script.json)")
    
    # Quick test mode
    parser.add_argument("--test", action="store_true", 
                        help="Quick test: 2 chapters × 2 scenes, 1 short with 2 chapters × 2 scenes, no thumbnails")
    
    # What to generate
    parser.add_argument("--main-only", action="store_true",
                        help="Only generate main video (skip shorts)")
    parser.add_argument("--shorts-only", action="store_true",
                        help="Only generate shorts (skip main video)")
    
    # Main video settings (defaults from Config class)
    parser.add_argument("--chapters", type=int, default=Config.chapters,
                        help=f"Main video outline chapters (default: {Config.chapters})")
    parser.add_argument("--scenes", type=int, default=Config.scenes_per_chapter,
                        help=f"Scenes per main chapter (default: {Config.scenes_per_chapter}, total = chapters × scenes)")
    
    # Shorts settings (defaults from Config class)
    parser.add_argument("--shorts", type=int, default=Config.num_shorts,
                        help=f"Number of YouTube Shorts (default: {Config.num_shorts}, use 0 to skip)")
    parser.add_argument("--short-scenes", type=int, default=Config.short_scenes_per_chapter,
                        help=f"Scenes per short (default: {Config.short_scenes_per_chapter}: hook, build, build, cliffhanger)")
    
    parser.add_argument("--no-thumbnails", action="store_true",
                        help="Skip thumbnail generation")
    
    return parser.parse_args()


# ------------- ENTRY POINT -------------

if __name__ == "__main__":
    args = parse_args()
    
    # Apply test mode settings
    if args.test:
        config.chapters = 2
        config.scenes_per_chapter = 2
        config.num_shorts = 1
        config.short_chapters = 1
        config.short_scenes_per_chapter = 4
        config.generate_thumbnails = False
        print("[MODE] Test mode enabled")
    else:
        config.chapters = args.chapters
        config.scenes_per_chapter = args.scenes
        config.num_shorts = args.shorts
        config.short_scenes_per_chapter = args.short_scenes
        config.generate_thumbnails = not args.no_thumbnails
    
    # Handle --main-only and --shorts-only flags
    if args.main_only and args.shorts_only:
        print("ERROR: Cannot use both --main-only and --shorts-only")
        sys.exit(1)
    
    if args.main_only:
        config.num_shorts = 0
        print("[MODE] Main video only (skipping shorts)")
    elif args.shorts_only:
        config.generate_main = False
        print("[MODE] Shorts only (skipping main video)")
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in args.person)
        safe_name = safe_name.replace(' ', '_').lower()
        output_file = f"{safe_name}_script.json"
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)
    
    try:
        script_data = generate_script(args.person, output_file)
        shorts_info = script_data.get("metadata", {}).get("shorts", [])
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        
        if config.generate_main:
            print(f"\n📺 MAIN VIDEO:")
            print(f"   Script: {output_file}")
            print(f"   Scenes: {script_data['metadata']['num_scenes']}")
            if script_data['metadata'].get('thumbnail_path'):
                print(f"   Thumbnail: {script_data['metadata']['thumbnail_path']}")
        
        if shorts_info:
            print(f"\n📱 YOUTUBE SHORTS ({len(shorts_info)}):")
            for short in shorts_info:
                sc = short.get('scenes', '?')
                print(f"   • {short.get('title', 'Untitled')}")
                print(f"     {sc} scenes → {short.get('file', '')}")
        
        print(f"\n🎬 Build video:")
        if config.generate_main:
            print(f"   python build_video.py {output_file} output.mp4")
        if shorts_info:
            print(f"   python build_video.py shorts/<name>_short1.json short1.mp4")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
