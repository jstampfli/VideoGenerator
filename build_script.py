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

SCRIPT_MODEL = "gpt-5.2"  # Using gpt-4o for better quality and longer context
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
    
    # Shorts settings (chapter-based like main video)
    num_shorts = 3              # Number of YouTube Shorts
    short_chapters = 2          # Chapters per short (build, cliffhanger)
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
1. Chapter 1: Establish the WORLD and plant seeds of future conflict
2. Chapters 2-4: Rising action - struggles, first successes, growing stakes
3. Chapters 5-7: Peak conflict - major breakthroughs AND major crises
4. Chapters 8-9: Resolution and consequences
5. Chapter 10: Legacy and emotional conclusion that echoes the beginning

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
    if prev_chapter:
        prev_context = f"""PREVIOUS CHAPTER (for continuity):
Chapter {prev_chapter['chapter_num']}: "{prev_chapter['title']}" ({prev_chapter['year_range']})
Summary: {prev_chapter['summary']}
Emotional Tone: {prev_chapter['emotional_tone']}
"""
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

CONTINUOUS FLOW REQUIREMENTS:
- Scene 1 should TRANSITION smoothly from where the last chapter ended
- Each scene should CONNECT to the next - end with forward momentum
- Reference recurring themes/motifs from earlier in the documentary
- The final scene should SET UP the next chapter's content
- Think of scenes as continuous shots in a film, not separate segments

SCENE-TO-SCENE TRANSITIONS:
- Use temporal connectors: "Days later...", "That same evening...", "Meanwhile..."
- Use emotional connectors: build on feelings from the previous scene
- Use narrative connectors: cause → effect, action → consequence
- Avoid abrupt topic changes - each scene should feel inevitable after the last

SCENE REQUIREMENTS:
1. VIVID and SPECIFIC - not generic summaries
2. SENSORY details - what do we see, hear, feel?
3. CONCRETE specifics - dates, names, places, numbers
4. EMOTION - what was the person feeling in this moment?
5. TENSION - what was at stake? What could go wrong?
6. DIALOGUE or QUOTES when impactful (paraphrased naturally)
7. Vary PACING - mix intimate moments with bigger events

NARRATION STYLE:
- Write like a master storyteller, not a textbook
- Use present tense for immediacy: "Einstein stands alone in his study..."
- Create vivid mental images
- Build emotional connection to the subject
- 2-4 sentences per scene (~15-25 seconds of narration)
- CRITICAL: This is SPOKEN narration for text-to-speech. Do NOT include:
  * Film directions like "Smash cut—", "Hard cut to", "Cut to:", "Fade in:"
  * Camera directions like "Close-up of", "Wide shot:", "Pan to"
  * Any production/editing terminology
  Write ONLY words that should be spoken by a narrator's voice.

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
    """Generate a focused outline for a single YouTube Short with chapter structure."""
    
    chapters_context = "\n".join([
        f"• {ch['title']} ({ch['year_range']}): {ch['summary']}"
        for ch in main_outline.get("chapters", [])
    ])
    
    # Different focus for each short
    focus_options = [
        "the most SHOCKING or little-known fact that will make viewers gasp",
        "the most DRAMATIC conflict, rivalry, or betrayal in their life",
        "their most MIND-BLOWING achievement or genius moment",
        "a TRAGIC or heartbreaking moment that humanizes them",
        "a MYSTERIOUS or unexplained aspect of their life",
    ]
    focus = focus_options[(short_num - 1) % len(focus_options)]
    
    outline_prompt = f"""You're creating a VIRAL YouTube Short about {person}.

This is Short #{short_num} of {total_shorts}. Focus on: {focus}

Their life story (for context):
{chapters_context}

Create a {config.short_chapters}-chapter outline for a ~60 second viral Short that tells ONE CONTINUOUS MINI-STORY:
- HOOK viewers instantly with something jaw-dropping
- Build TENSION through cause-and-effect storytelling
- End on a CLIFFHANGER that drives them to the full documentary

THE SHORT MUST FEEL LIKE A CONTINUOUS NARRATIVE, not disconnected facts.

NARRATIVE STRUCTURE:
1. THE BUILD (Chapter 1): Hook + rapidly escalating stakes. Each moment causes the next.
2. THE CLIFFHANGER (Chapter 2): Peak drama + unresolved tension. "But what happened next..."

For each chapter provide:
{{
  "chapter_num": 1-{config.short_chapters},
  "title": "Punchy chapter title",
  "purpose": "build" | "cliffhanger",
  "key_moments": ["specific dramatic moment 1", "moment 2", ...],
  "emotional_arc": "what emotion should viewers feel",
  "pacing": "explosive" | "building" | "suspenseful",
  "connects_to_next": "How this flows into the next chapter"
}}

Also provide:
- "short_title": VIRAL title (max 50 chars) - curiosity gaps, shocking claims, mysteries
- "short_description": YouTube description (100-150 words) with hashtags, ending with "🎬 Full story: [LINK]"
- "tags": 10-15 SEO tags separated by commas (person name, topic, related keywords, trending terms for Shorts)
- "thumbnail_prompt": Vertical 9:16, dramatic single image, high contrast, mobile-optimized
- "hook_type": "shocking_fact" | "mystery" | "untold_story" | "what_if" | "betrayal" | "genius_moment"
- "teaser_line": One sentence that makes people NEED to watch
- "central_tension": The ONE driving question/conflict that runs through the entire short
- "narrative_thread": How the story connects from first scene to last

VIRAL TITLE FORMULAS:
- "The [X] Nobody Talks About"
- "[Person] Did WHAT?!"
- "This Changed Everything..."
- "The $X Mistake"
- "Why [Person] [Shocking Action]"

Respond with JSON:
{{
  "short_title": "...",
  "short_description": "...",
  "tags": "...",
  "thumbnail_prompt": "...",
  "hook_type": "...",
  "teaser_line": "...",
  "central_tension": "...",
  "narrative_thread": "...",
  "chapters": [...]
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You create viral content that gets millions of views. Every word must earn its place. Respond with valid JSON only."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.9,
        response_format={"type": "json_object"}
    )
    
    return json.loads(clean_json_response(response.choices[0].message.content))


def generate_short_scenes(person: str, short_outline: dict, chapter: dict, start_id: int, 
                          all_chapters: list, previous_scenes: list) -> list[dict]:
    """Generate scenes for one chapter of a YouTube Short with full context."""
    
    # Build context of all chapters so the model understands the full arc
    chapters_overview = "\n".join([
        f"  {i+1}. {ch.get('title', 'Untitled')} ({ch.get('purpose', 'build')}) - {ch.get('summary', '')[:100] if ch.get('summary') else 'Key moments: ' + ', '.join(ch.get('key_moments', [])[:2])}"
        for i, ch in enumerate(all_chapters)
    ])
    
    current_chapter_num = chapter.get('chapter_num', 1)
    
    # Build context of what's already been covered
    if previous_scenes:
        prev_context = "SCENES ALREADY WRITTEN (don't repeat this content):\n"
        for sc in previous_scenes:
            prev_context += f"  - Scene {sc.get('id')}: {sc.get('title')} - \"{sc.get('narration', '')[:80]}...\"\n"
    else:
        prev_context = "This is the OPENING of the Short - start with maximum impact!"
    
    # Get narrative context from outline
    central_tension = short_outline.get('central_tension', '')
    narrative_thread = short_outline.get('narrative_thread', '')
    connects_to_next = chapter.get('connects_to_next', '')
    
    scene_prompt = f"""You're writing scenes for a VIRAL YouTube Short about {person}.

This Short tells ONE CONTINUOUS MINI-STORY. Every scene must connect to the next like a chain.

SHORT TITLE: "{short_outline.get('short_title', '')}"
TEASER: {short_outline.get('teaser_line', '')}
CENTRAL TENSION: {central_tension}
NARRATIVE THREAD: {narrative_thread}

FULL SHORT STRUCTURE (2-act: build → cliffhanger):
{chapters_overview}

{prev_context}

NOW WRITING CHAPTER {current_chapter_num}: "{chapter['title']}"
Purpose: {chapter.get('purpose', 'build')}
Emotional Arc: {chapter.get('emotional_tone', chapter.get('emotional_arc', 'engaging'))}
Pacing: {chapter.get('pacing', 'dynamic')}
Flows Into: {connects_to_next}

Key Moments for This Chapter:
{chr(10).join(f"• {m}" for m in chapter.get('key_moments', []))}

Generate EXACTLY {config.short_scenes_per_chapter} scenes that FLOW AS ONE CONTINUOUS STORY.

CONTINUOUS NARRATIVE REQUIREMENTS:
- Each scene must CAUSE the next scene (action → consequence)
- Use transitions: "But then...", "What he didn't know was...", "That's when..."
- The CENTRAL TENSION should thread through every scene
- Build emotional momentum - each scene more intense than the last
- If chapter 1: Hook → escalate → set up the crisis
- If chapter 2: Crisis peaks → twist/revelation → cliffhanger to full video

SCENE-TO-SCENE FLOW:
- End each scene with forward momentum (question, tension, or setup)
- Start each scene by building on the previous one
- NO scene should feel disconnected from the story
- Think: "Because of Scene 1, Scene 2 happens. Because of Scene 2, Scene 3 happens..."

VIRAL SHORT SCENE RULES:
1. EVERY second counts - no filler, no fluff
2. SPECIFIC details that create vivid mental images
3. Each scene creates CURIOSITY for the next
4. Conversational but dramatic - like telling a friend an incredible story
5. Build TENSION - what's at stake? What could go wrong?

NARRATION STYLE:
- 1-2 punchy sentences per scene (~8-10 seconds spoken)
- Present tense for immediacy: "He stares at the letter. His hands shake."
- Conversational but dramatic
- Include sensory details
- CRITICAL: This is SPOKEN narration for text-to-speech. Do NOT include:
  * Film directions like "Smash cut—", "Hard cut to", "Cut to:", "Fade in:"
  * Camera directions like "Close-up of", "Wide shot:", "Pan to"
  * Any production/editing terminology
  Write ONLY words that should be spoken by a narrator's voice.

IMAGE PROMPTS:
- Vertical 9:16 composition
- Dramatic, mobile-optimized
- High contrast, bold composition
- Single clear focal point
- End with ", 9:16 vertical dramatic"

Respond with JSON array:
[
  {{
    "id": {start_id},
    "title": "2-4 word punchy title",
    "narration": "Gripping narration...",
    "image_prompt": "Vertical dramatic composition, 9:16 vertical dramatic"
  }},
  ...
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You write viral content. Every word must hook, engage, or create curiosity. Respond with valid JSON array only."},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.85,
    )
    
    scenes = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    return scenes


def generate_shorts(person_of_interest: str, main_title: str, global_block: str, outline: dict, base_output_path: str):
    """Generate YouTube Shorts using chapter-based structure like main video."""
    if config.num_shorts == 0:
        print("\n[SHORTS] Skipped (--shorts 0)")
        return []
    
    print(f"\n{'='*60}")
    print(f"[SHORTS] Generating {config.num_shorts} YouTube Short(s)")
    print(f"[SHORTS] Structure: {config.short_chapters} chapters × {config.short_scenes_per_chapter} scenes = {config.total_short_scenes} scenes each")
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
        print(f"[SHORT {short_num}] Hook: {short_outline.get('hook_type', 'unknown')}")
        
        chapters = short_outline.get("chapters", [])
        print(f"[SHORT {short_num}] Chapters: {len(chapters)}")
        
        # Step 2: Generate scenes for each chapter (with full context)
        all_scenes = []
        for chapter in chapters:
            start_id = len(all_scenes) + 1
            ch_num = chapter.get("chapter_num", "?")
            ch_title = chapter.get("title", "Untitled")
            
            print(f"[SHORT {short_num}] Chapter {ch_num}: {ch_title} ({chapter.get('purpose', 'build')})")
            
            try:
                scenes = generate_short_scenes(
                    person=person_of_interest,
                    short_outline=short_outline,
                    chapter=chapter,
                    start_id=start_id,
                    all_chapters=chapters,           # Full chapter structure for context
                    previous_scenes=list(all_scenes) # What's been written so far
                )
                
                all_scenes.extend(scenes)
                print(f"[SHORT {short_num}]   → {len(scenes)} scenes (total: {len(all_scenes)})")
                
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
                "teaser_line": short_outline.get("teaser_line", ""),
                "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
                "hook_type": short_outline.get("hook_type", ""),
                "person_of_interest": person_of_interest,
                "main_video_title": main_title,
                "global_block": global_block,
                "num_scenes": len(all_scenes),
                "chapters": len(chapters),
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
            "scenes": len(all_scenes),
            "chapters": len(chapters)
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
        print(f"[CONFIG] Shorts: {config.num_shorts} × ({config.short_chapters} chapters × {config.short_scenes_per_chapter} scenes) = {config.num_shorts * config.total_short_scenes} scenes")
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
  # Full production run (100 scenes, 3 shorts with 12 scenes each)
  python build_script.py "Albert Einstein"

  # Quick test (4 scenes, 1 short with 4 scenes, no thumbnails)
  python build_script.py "Albert Einstein" --test

  # Custom main video settings
  python build_script.py "Albert Einstein" --chapters 5 --scenes 3

  # Custom shorts settings  
  python build_script.py "Albert Einstein" --shorts 1 --short-chapters 2 --short-scenes 3

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
    parser.add_argument("--short-chapters", type=int, default=Config.short_chapters,
                        help=f"Chapters per short (default: {Config.short_chapters}: build, cliffhanger)")
    parser.add_argument("--short-scenes", type=int, default=Config.short_scenes_per_chapter,
                        help=f"Scenes per short chapter (default: {Config.short_scenes_per_chapter}, total per short = short-chapters × short-scenes)")
    
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
        config.short_chapters = 2
        config.short_scenes_per_chapter = 2
        config.generate_thumbnails = False
        print("[MODE] Test mode enabled")
    else:
        config.chapters = args.chapters
        config.scenes_per_chapter = args.scenes
        config.num_shorts = args.shorts
        config.short_chapters = args.short_chapters
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
                ch = short.get('chapters', '?')
                sc = short.get('scenes', '?')
                print(f"   • {short.get('title', 'Untitled')}")
                print(f"     {ch} chapters, {sc} scenes → {short.get('file', '')}")
        
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
