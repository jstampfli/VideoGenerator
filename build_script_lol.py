"""
League of Legends Lore Script Generator

Generates documentary-style scripts for League of Legends lore videos.
Supports both character-specific lore (champions) and high-level lore (regions, groups, entire Runeterra).
"""
import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

# Import shared utilities
import build_scripts_utils

# Initialize utils module with client and config
load_dotenv()  # Load API key from .env file
build_scripts_utils.client = OpenAI()
build_scripts_utils.SCRIPT_MODEL = "gpt-5.2"
build_scripts_utils.IMG_MODEL = "gpt-image-1.5"
build_scripts_utils.NO_TEXT_CONSTRAINT = """
CRITICAL: Do NOT include any text, words, letters, numbers, titles, labels, watermarks, or any written content in the image. The image must be completely text-free."""

# Export for backward compatibility
client = build_scripts_utils.client
SCRIPT_MODEL = build_scripts_utils.SCRIPT_MODEL
IMG_MODEL = build_scripts_utils.IMG_MODEL
NO_TEXT_CONSTRAINT = build_scripts_utils.NO_TEXT_CONSTRAINT

THUMBNAILS_DIR = Path("thumbnails")
SHORTS_DIR = Path("shorts_scripts")
SCRIPTS_DIR = Path("scripts")


# Generation settings (can be overridden via command line)
class Config:
    # Main video settings
    chapters = 10           # Number of outline chapters
    scenes_per_chapter = 4  # Scenes per chapter (total = chapters * scenes_per_chapter)
    generate_main = True    # Whether to generate main video
    
    # Shorts settings
    num_shorts = 3              # Number of YouTube Shorts
    short_chapters = 1          # Chapters per short (1 chapter: 5 scenes telling one complete story)
    short_scenes_per_chapter = 5  # Scenes per chapter in shorts (complete story with natural conclusion)
    
    generate_thumbnails = True  # Whether to generate thumbnail images (main video)
    generate_short_thumbnails = False  # Whether to generate thumbnails for shorts (usually not needed)
    generate_refinement_diffs = False  # Whether to generate refinement diff JSON files
    
    # Top N video settings
    top_n_mode = False  # Whether to generate a top N list-style video
    top_n_entries = 10  # Number of entries in the top N list (configurable via --top-n flag)
    top_n_intro_scenes = 2  # Number of introduction scenes for top N videos
    top_n_entry_min_scenes = 3  # Minimum scenes per entry
    top_n_entry_max_scenes = 5  # Maximum scenes per entry
    
    @property
    def total_scenes(self):
        return self.chapters * self.scenes_per_chapter
    
    @property
    def total_short_scenes(self):
        return self.short_chapters * self.short_scenes_per_chapter

config = Config()

# Import shared functions from utils (using "lore" content_type)
clean_json_response = build_scripts_utils.clean_json_response
get_shared_scene_requirements = lambda: build_scripts_utils.get_shared_scene_requirements("lore")
get_shared_narration_style = build_scripts_utils.get_shared_narration_style
get_shared_scene_flow_instructions = build_scripts_utils.get_shared_scene_flow_instructions
get_shared_examples = lambda: build_scripts_utils.get_shared_examples("lore")
generate_refinement_diff = build_scripts_utils.generate_refinement_diff
identify_pivotal_moments = build_scripts_utils.identify_pivotal_moments
generate_significance_scene = build_scripts_utils.generate_significance_scene

# Wrapper for refine_scenes to determine subject_type based on subject
def refine_scenes_lol(scenes: list[dict], subject: str, is_short: bool = False, chapter_context: str = None, diff_output_path: Path | None = None, skip_significance_scenes: bool = False) -> tuple[list[dict], dict]:
    """Wrapper for refine_scenes from utils, determining subject_type automatically."""
    # Determine subject type based on common LoL patterns
    subject_lower = subject.lower()
    if any(region in subject_lower for region in ['demacia', 'noxus', 'ionia', 'piltover', 'zaun', 'bilgewater', 'freljord', 'shurima', 'targon', 'bandle', 'ixtal', 'runeterra']):
        subject_type = "region"
    elif any(group in subject_lower for group in ['yordle', 'vastaya', 'darkin', 'aspect', 'void']):
        subject_type = "topic"  # Groups/species
    elif subject_lower == "runeterra" or "all of runeterra" in subject_lower:
        subject_type = "topic"  # Entire world
    else:
        subject_type = "character"  # Champion
    
    return build_scripts_utils.refine_scenes(scenes, subject, is_short, chapter_context, diff_output_path, subject_type, skip_significance_scenes)

# generate_thumbnail wrapper
def generate_thumbnail(prompt: str, output_path: Path, size: str = "1024x1024") -> Path | None:
    """Generate a thumbnail image and save it with safety violation handling."""
    return build_scripts_utils.generate_thumbnail(prompt, output_path, size, config.generate_thumbnails)


def detect_subject_type(subject: str) -> str:
    """
    Detect the type of LoL lore subject based on the input string.
    
    Returns:
        "character" for champions, "region" for regions, "topic" for groups/entire world
    """
    subject_lower = subject.lower()
    
    # Common regions
    regions = ['demacia', 'noxus', 'ionia', 'piltover', 'zaun', 'bilgewater', 'freljord', 
               'shurima', 'targon', 'bandle', 'ixtal', 'shadow isles']
    if any(region in subject_lower for region in regions):
        return "region"
    
    # Common groups/species
    groups = ['yordle', 'yordles', 'vastaya', 'darkin', 'aspect', 'void', 'voidborn']
    if any(group in subject_lower for group in groups):
        return "topic"
    
    # Entire world
    if "runeterra" in subject_lower and ("all" in subject_lower or "entire" in subject_lower or "complete" in subject_lower):
        return "topic"
    
    # Default to character (champion)
    return "character"


def generate_lol_outline(subject: str) -> dict:
    """
    Generate a detailed outline for LoL lore content.
    Handles both character-specific lore (champions) and high-level lore (regions, groups, Runeterra).
    """
    subject_type = detect_subject_type(subject)
    
    print(f"\n[OUTLINE] Generating {config.chapters}-chapter outline for {subject_type} lore...")
    
    # Adapt prompt based on subject type
    if subject_type == "character":
        subject_desc = f"the champion {subject}"
        timeline_note = "Include birth/creation era if known, or timeline periods"
        structure_note = "life story, origins, key battles, relationships, major events"
    elif subject_type == "region":
        subject_desc = f"the lore and history of {subject}"
        timeline_note = "Use timeline periods/eras (e.g., 'Ancient Era', 'Rune Wars', 'Present Day')"
        structure_note = "formation, major conflicts, key figures, political structure, significant events"
    elif subject_type == "topic":
        if "runeterra" in subject.lower():
            subject_desc = "the complete lore and history of Runeterra"
            timeline_note = "Use timeline periods spanning all of history (Ancient Era through Present Day)"
            structure_note = "world formation, major eras, conflicts that shaped the world, key civilizations, present state"
        else:
            subject_desc = f"the lore of the {subject}"
            timeline_note = "Use timeline periods when this group/species was most active"
            structure_note = "origins, evolution, key members, significant events, current status"
    else:
        subject_desc = f"the lore of {subject}"
        timeline_note = "Use appropriate timeline periods"
        structure_note = "key events, significant moments, major conflicts, relationships"
    
    outline_prompt = f"""You are a master lore writer for League of Legends. Create a compelling narrative outline for a ~20 minute documentary about: {subject_desc}

This will be a {config.total_scenes}-scene documentary with EXACTLY {config.chapters} chapters. Think of this as a FEATURE FILM with continuous story arcs, not disconnected episodes.

NARRATIVE STRUCTURE:
- The documentary should feel like ONE CONTINUOUS STORY, not a list of facts
- Each chapter should FLOW into the next with clear cause-and-effect
- Plant SEEDS in early chapters that PAY OFF later (foreshadowing)
- Build RECURRING THEMES that echo throughout (e.g., conflict, sacrifice, magic, honor, corruption)
- Create emotional MOMENTUM that builds to a climax around chapter 7-8, then resolves

For each of the {config.chapters} chapters, provide:
- "chapter_num": 1-{config.chapters}
- "title": A compelling chapter title
- "year_range" or "timeline": The time period covered ({timeline_note})
- "summary": 2-3 sentences about what happens
- "key_events": 4-6 specific dramatic moments to show
- "emotional_tone": The mood of this chapter
- "dramatic_tension": What conflict drives this chapter
- "connects_to_next": How this chapter sets up or flows into the next one
- "recurring_threads": Which themes/motifs from earlier chapters appear here

STORY ARC REQUIREMENTS:
1. Chapter 1 - THE HOOK: Start by introducing {subject_desc} with context. Then present a rapid-fire "trailer" of the MOST interesting facts, conflicts, dramatic moments, and mysteries that will be covered. Give viewers context for what we're exploring. Make viewers think "I NEED to know more." This is NOT chronological - it's a highlight reel that hooks the audience. End with something like "But how did it all begin? Let's start from the beginning..."
2. Chapters 2-6: {structure_note}
3. Chapter 7: Current state and legacy - what is the situation now, what mysteries remain, how has this shaped Runeterra

CRITICAL:
- Every chapter must CONNECT to what came before and set up what comes after
- Include SPECIFIC details - names, places, events, magic types, conflicts
- Focus on DRAMA and CONFLICT - relationships, wars, betrayals, magical discoveries
- Make it feel like watching an epic fantasy story, not reading a wiki
- Emphasize MAGICAL elements, MYSTICAL powers, REGIONAL conflicts, FACTION relationships
- Include WORLD-BUILDING details that make Runeterra feel alive

Respond with JSON:
{{
  "subject": "{subject}",
  "subject_type": "{subject_type}",
  "timeline_start": "Era or year when story begins",
  "timeline_end": "Era or year when story ends (or 'Present Day')",
  "tagline": "One compelling sentence that captures the story",
  "central_theme": "The overarching theme that ties the whole documentary together",
  "narrative_arc": "Brief description of the emotional journey from start to finish",
  "overarching_plots": [
    {{
      "plot_name": "The main plot thread (e.g., 'The Rune Wars', 'The Darkin Threat', 'The Ionian Invasion')",
      "description": "What this plot is about and why it matters",
      "starts_chapter": 1-{config.chapters},
      "peaks_chapter": 1-{config.chapters},
      "resolves_chapter": 1-{config.chapters},
      "key_moments": ["specific plot points that develop this story"]
    }}
  ],
  "sub_plots": [
    {{
      "subplot_name": "A sub-plot that spans 2-4 chapters",
      "description": "What this sub-plot is about",
      "chapters_span": [1-3],
      "key_moments": ["specific moments that advance this sub-plot"]
    }}
  ],
  "chapters": [
    {{
      "chapter_num": 1,
      "title": "...",
      "year_range": "..." or "timeline": "...",
      "summary": "...",
      "key_events": ["...", ...],
      "emotional_tone": "...",
      "dramatic_tension": "...",
      "connects_to_next": "...",
      "recurring_threads": ["...", ...],
      "plots_active": ["plot names or subplot names that are active/developing in this chapter"],
      "plot_developments": ["How overarching plots and sub-plots develop in this chapter"]
    }},
    ... ({config.chapters} chapters total)
  ]
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a League of Legends lore expert who finds the drama and epic scope in every story. You understand Runeterra's history, magic systems, and conflicts. Respond with valid JSON only."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    outline_data = json.loads(clean_json_response(response.choices[0].message.content))
    chapters = outline_data.get("chapters", [])
    
    print(f"[OUTLINE] Generated {len(chapters)} chapters")
    for ch in chapters:
        timeline = ch.get('year_range') or ch.get('timeline', '?')
        print(f"  Ch {ch['chapter_num']}: {ch['title']} ({timeline})")
    
    return outline_data


def generate_cta_top_n(topic: str) -> dict:
    """
    Generate a simple CTA scene for top N ranking videos.
    No story references - just a clean call to action before the list.
    """
    cta_prompt = f"""Generate a single CTA scene for a top N ranking video about: {topic}

CONTEXT:
- This is a ranking/countdown video, NOT a story
- The scene appears after the intro and before the list begins
- It's a simple call-to-action asking viewers to like, subscribe, and comment

REQUIREMENTS:
1. NARRATION (MUST BE ONE SENTENCE):
   - Simple, direct CTA - no story references, no "going back to the beginning", no narrative elements
   - MUST include: request to like, subscribe, and comment
   - Use phrases like: "Before we jump into the list, if you like this type of content please like, subscribe, and comment"
   - Keep it short and natural - one sentence (~5-8 seconds when spoken)
   - Examples:
     * "Before we jump into the list, if you like this type of content please like, subscribe, and comment below."
     * "If you enjoy ranking videos like this, make sure to like, subscribe, and leave a comment before we dive into the countdown."

2. NARRATION STYLE:
   - Write from the YouTuber's perspective (YOUR script)
   - Simple, clear, direct language
   - Friendly and engaging tone
   - High energy - this is a ranking video

3. IMAGE PROMPT:
   - Bright, happy, upbeat mood - positive and engaging
   - Related to {topic} but in a cheerful context
   - Fantasy/magical elements appropriate for League of Legends lore
   - 16:9 cinematic format
   - The image should be inviting and make viewers want to continue watching

4. EMOTION: "upbeat" or "engaging"

5. YEAR/TIMELINE: "introduction" or "transition"

Respond with JSON:
{{
  "id": 0,
  "title": "3-5 word CTA title",
  "narration": "ONE sentence with CTA to like, subscribe, and comment. No story references. ~5-8 seconds when spoken.",
  "image_prompt": "Bright, happy, upbeat scene related to {topic} - positive and engaging, makes viewers want to continue. Fantasy/magical elements. 16:9 cinematic",
  "emotion": "upbeat",
  "year": "introduction"
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating ranking list content. Write naturally from YOUR perspective. Keep the CTA simple and direct - no story references. Respond with valid JSON only."},
            {"role": "user", "content": cta_prompt}
        ],
        temperature=0.85,
    )
    
    scene = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scene, dict):
        raise ValueError(f"Expected dict, got {type(scene)}")
    
    # Ensure all required fields are present
    if 'title' not in scene:
        scene['title'] = "Support the Channel"
    if 'narration' not in scene:
        scene['narration'] = "Before we jump into the list, if you like this type of content please like, subscribe, and comment below."
    else:
        # Validate that narration includes CTA (like, subscribe, comment) and is ONE sentence
        narration_lower = scene['narration'].lower()
        has_like = 'like' in narration_lower
        has_subscribe = 'subscribe' in narration_lower
        has_comment = 'comment' in narration_lower
        
        # If CTA is missing, add it to the narration
        if not (has_like and has_subscribe and has_comment):
            base_narration = scene['narration'].rstrip('.!?')
            scene['narration'] = f"{base_narration} If you like this type of content, please like, subscribe, and comment below."
    if 'image_prompt' not in scene:
        scene['image_prompt'] = f"Bright, happy, upbeat scene related to {topic} - positive and engaging. Fantasy elements. 16:9 cinematic"
    if 'emotion' not in scene:
        scene['emotion'] = "upbeat"
    if 'year' not in scene:
        scene['year'] = "introduction"
    
    return scene


def generate_cta_transition_scene_lol(subject: str, tag_line: str | None = None) -> dict:
    """
    Generate a CTA transition scene between chapter 1 (hook) and chapter 2 (story begins).
    Adapted for LoL lore content (used for regular documentaries, not top N videos).
    """
    tag_line_text = f" - {tag_line}" if tag_line else ""
    
    cta_prompt = f"""Generate a single transition scene for a lore documentary about {subject}.

CONTEXT:
- This scene appears BETWEEN the hook chapter (preview/trailer) and the chronological story (chapter 2)
- It transitions from the preview back to where the story begins chronologically
- It includes a call-to-action (like, subscribe, comment) naturally woven into the narration
- The tone should be engaging, friendly, and encourage viewer engagement

REQUIREMENTS:
1. TRANSITION TEXT AND CTA (MUST BE ONE SENTENCE):
   - Combine transition from hook/preview to chronological beginning WITH call-to-action in a SINGLE sentence
   - MUST include: request to like, subscribe, and comment
   - Use phrases like "But how did it all begin?", "Let's start from the beginning", or similar natural transitions
   - CRITICAL: Must mention "like", "subscribe", and "comment" in this one sentence
   - Examples: 
     * "But how did it all begin? Before we dive into the story, if you're enjoying this, make sure to like, subscribe, and comment below."
     * "Let's start from the very beginning - and don't forget to like, subscribe, and leave a comment if you want to see more lore documentaries like this."
   - Keep it short, natural, and conversational - one sentence total (~8-12 seconds when spoken)

2. NARRATION STYLE:
   - Write from the YouTuber's perspective (YOUR script, YOU telling the story)
   - Simple, clear language
   - Friendly and engaging tone
   - MUST be exactly ONE sentence that combines transition + CTA

3. IMAGE PROMPT:
   - Bright, happy, upbeat mood - this should feel positive and engaging
   - Related to {subject} and the lore topic but in a cheerful context
   - Include visual elements that suggest engagement: perhaps a moment of triumph, discovery, or positive milestone
   - Fantasy/magical elements appropriate for League of Legends lore
   - 16:9 cinematic format
   - The image should be inviting and make viewers want to continue watching

4. EMOTION: "upbeat" or "engaging"

5. YEAR/TIMELINE: "transition" or relevant timeline from early in the story

Respond with JSON:
{{
  "id": 0,
  "title": "3-5 word transition title",
  "narration": "ONE sentence that transitions from the hook to the story beginning and includes CTA to like, subscribe, and comment. ~8-12 seconds when spoken.",
  "image_prompt": "Bright, happy, upbeat scene related to {subject} - an inspiring moment, discovery, or positive milestone from the lore that's visually engaging and makes viewers want to continue. Fantasy/magical elements. 16:9 cinematic",
  "emotion": "upbeat",
  "year": "transition" or a relevant timeline from early in the story
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating engaging lore content. Write naturally from YOUR perspective. Make the transition smooth and the CTA feel authentic, not forced. Respond with valid JSON only."},
            {"role": "user", "content": cta_prompt}
        ],
        temperature=0.85,
    )
    
    scene = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scene, dict):
        raise ValueError(f"Expected dict, got {type(scene)}")
    
    # Ensure all required fields are present
    if 'title' not in scene:
        scene['title'] = "Back to the Beginning"
    if 'narration' not in scene:
        scene['narration'] = "But how did it all begin? Before we dive into the story, make sure to like, subscribe, and comment below if you're enjoying this lore documentary."
    else:
        # Validate that narration includes CTA (like, subscribe, comment) and is ONE sentence
        narration_lower = scene['narration'].lower()
        has_like = 'like' in narration_lower
        has_subscribe = 'subscribe' in narration_lower
        has_comment = 'comment' in narration_lower
        
        # If CTA is missing, add it to the narration
        if not (has_like and has_subscribe and has_comment):
            base_narration = scene['narration'].rstrip('.!?')
            scene['narration'] = f"{base_narration} Make sure to like, subscribe, and comment below."
    if 'image_prompt' not in scene:
        scene['image_prompt'] = f"Bright, happy, upbeat scene related to {subject} - an inspiring moment or positive milestone. Fantasy elements. 16:9 cinematic"
    if 'emotion' not in scene:
        scene['emotion'] = "upbeat"
    if 'year' not in scene:
        scene['year'] = "transition"
    
    return scene


def generate_top10_outline(topic: str, num_entries: int = 10) -> dict:
    """
    Generate a top N list outline for LoL lore content.
    LLM automatically determines and ranks entries based on the provided topic.
    """
    print(f"\n[OUTLINE] Generating top {num_entries} list outline for: {topic}")
    
    outline_prompt = f"""You are a League of Legends lore expert creating a definitive ranking. Generate a list of the top {num_entries} entries for: {topic}

The topic is: {topic}

REQUIREMENTS:
- Rank them from {num_entries} (lowest on this list) to 1 (highest/top on this list)
- The ranking criteria should match the topic - if it's about "powerful beings", rank by power; if it's about "important events", rank by importance/impact; if it's about "best champions", rank by that criteria, etc.
- Be specific about WHY each ranks at their position based on the topic's criteria
- Include actual lore details and facts, not speculation
- Adapt the entry structure to match the topic (e.g., events have different fields than characters)

For each entry, provide:
- "rank": Number from {num_entries} to 1 ({num_entries} is lowest on this list, 1 is highest/top)
- "name": The entry's name (could be a character, event, location, concept, etc. depending on topic)
- "type": "character" for champions/individuals, "entity" for gods/cosmic beings, "event" for historical events, "location" for places, "concept" for abstract forces, etc.
- "description": A few sentences explaining what this entry is and why it's notable/relevant to the ranking criteria
- "key_feats": (CRITICAL) Array of 3-5 specific lore facts, achievements, or demonstrations relevant to the ranking criteria
- "why_this_rank": Explanation of why they rank at this position (how they compare to others based on the topic's criteria)
- "suggested_scenes": Number between {config.top_n_entry_min_scenes} and {config.top_n_entry_max_scenes} - how many scenes needed to properly cover this entry (more complex = more scenes)

CRITICAL: Rankings must be justified and make sense based on actual LoL lore and the specific topic/criteria. Higher-ranked entries should have more impressive/relevant facts.

Respond with JSON:
{{
  "topic": "{topic}",
  "num_entries": {num_entries},
  "entries": [
    {{
      "rank": {num_entries},
      "name": "Entry Name",
      "type": "character|entity|event|location|concept",
      "description": "What this entry is and why it's notable/relevant",
      "key_feats": ["fact1", "fact2", "fact3"],
      "why_this_rank": "Why they rank at position {num_entries}",
      "suggested_scenes": {config.top_n_entry_min_scenes}
    }},
    ...
    {{
      "rank": 1,
      "name": "Top Entry",
      "type": "character|entity|event|location|concept",
      "description": "What this entry is and why it ranks #1",
      "key_feats": ["fact1", "fact2", "fact3"],
      "why_this_rank": "Why they rank #1",
      "suggested_scenes": {config.top_n_entry_max_scenes}
    }}
  ]
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a League of Legends lore expert who understands power scaling, cosmic hierarchies, and demonstrated feats. You can accurately rank beings based on actual lore events and achievements. Respond with valid JSON only."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    outline_data = json.loads(clean_json_response(response.choices[0].message.content))
    entries = outline_data.get("entries", [])
    
    # Sort entries by rank (descending, so 10 comes first, 1 comes last)
    entries.sort(key=lambda x: x.get("rank", 0), reverse=True)
    
    print(f"[OUTLINE] Generated {len(entries)} entries")
    for entry in entries:
        rank = entry.get('rank', '?')
        name = entry.get('name', '?')
        scenes = entry.get('suggested_scenes', '?')
        print(f"  #{rank}: {name} ({scenes} scenes)")
    
    return outline_data


def generate_top10_intro(topic: str, entries: list[dict]) -> list[dict]:
    """
    Generate introduction scenes for a top N list video.
    Returns introduction scenes that set up the countdown format.
    """
    entries_summary = "\n".join([
        f"#{e.get('rank')}: {e.get('name')} - {e.get('description', e.get('power_description', ''))[:80]}..."
        for e in entries[:3]  # Preview first few for context
    ])
    
    num_entries = len(entries)
    intro_prompt = f"""Generate {config.top_n_intro_scenes} introduction scenes for a top {num_entries} list video about: {topic}

CONTEXT:
- This is a countdown-style video (ranking from {num_entries} to 1, building to the top/highest)
- The video will explore the top {num_entries} entries for: {topic}
- We're counting down: starting at #{num_entries} (lowest on this list) and building to #1 (highest/top)
- The intro should create anticipation and explain what viewers will discover

PREVIEW OF ENTRIES (for context):
{entries_summary}

STRUCTURE (exactly {config.top_n_intro_scenes} scenes):
- SCENE 1: Introduce the concept - what criteria are we using for this ranking? What makes an entry rank higher? Set up the countdown format. Create anticipation.
- SCENE 2: Tease what viewers will discover - mention that we're counting down from {num_entries} to 1, each entry ranking higher than the last. Build excitement. Transition naturally into entry #{num_entries}.

REQUIREMENTS:
- Use natural, engaging narration (YouTuber perspective)
- ULTRA-SHORT and FACT-DENSE: Target ~3-5 seconds per scene (one very short sentence or half a sentence)
- Keep it concise - pack anticipation and concept setup into minimal words
- DO NOT list all {num_entries} beings in the intro - just tease the concept
- Maximum impact with minimal words - think TikTok/TikTok Reels style
- High energy, fast pace - this is a ranking video, not a story

Each scene must include:
- "title": 2-5 word title
- "narration": Ultra-short engaging narration (~3-5 seconds - one very short sentence)
- "image_prompt": Visual that matches the scene (fantasy/magical LoL elements, 16:9 cinematic)
- "emotion": Scene emotion (e.g., "exciting", "anticipatory", "epic")
- "year": "introduction" or relevant timeline

Respond with JSON array:
[
  {{"id": 1, "title": "...", "narration": "...", "image_prompt": "...", "emotion": "...", "year": "introduction"}},
  {{"id": 2, "title": "...", "narration": "...", "image_prompt": "...", "emotion": "...", "year": "introduction"}}
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating engaging top 10 list content. Write naturally from YOUR perspective. Create anticipation and excitement. Respond with valid JSON array only."},
            {"role": "user", "content": intro_prompt}
        ],
        temperature=0.85,
    )
    
    scenes = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    if len(scenes) != config.top_n_intro_scenes:
        print(f"[WARNING] Got {len(scenes)} intro scenes, expected {config.top_n_intro_scenes}")
    
    return scenes


def generate_top10_entry_scenes(entry: dict, entry_num: int, total_entries: int, global_style: str, prev_entry: dict | None = None, subject_type: str = "character") -> list[dict]:
    """
    Generate scenes for a single top N entry (fact-based ranking format).
    Returns variable number of scenes (3-5) based on entry complexity.
    Focuses on high-energy, fast-paced presentation of interesting LoL lore facts.
    """
    rank = entry.get("rank", entry_num)
    name = entry.get("name", "Unknown")
    description = entry.get("description", entry.get("power_description", ""))  # Support both old and new field names
    key_feats = entry.get("key_feats", [])
    why_rank = entry.get("why_this_rank", "")
    num_scenes = min(max(entry.get("suggested_scenes", config.top_n_entry_min_scenes), config.top_n_entry_min_scenes), config.top_n_entry_max_scenes)
    
    # Build transition context from previous entry (minimal - just to flow between entries)
    transition_context = ""
    if prev_entry:
        prev_rank = prev_entry.get("rank", 0)
        prev_name = prev_entry.get("name", "")
        transition_context = f"""PREVIOUS ENTRY: #{prev_rank}: {prev_name}

TRANSITION: Keep it SHORT. Just naturally move to the next entry. Use phrases like "Next up...", "But ranking higher is...", "But even more [relevant adjective] is...". ONE short phrase max, then dive into facts.
"""
    else:
        transition_context = f"""FIRST ENTRY AFTER INTRO: This is entry #{rank}. Keep transition minimal - just move from intro to first entry. ONE short phrase, then facts.
"""
    
    feats_str = "\n".join(f"• {feat}" for feat in key_feats)
    
    scene_prompt = f"""Generate {num_scenes} scenes for entry #{rank} in a top {total_entries} countdown: {name}

ENTRY DETAILS:
- Rank: #{rank} of {total_entries}
- Type: {subject_type}
- Description: {description}
- Why this rank: {why_rank}

KEY FEATS TO COVER:
{feats_str}

{transition_context}

STRUCTURE (exactly {num_scenes} scenes - NO STORY, JUST FACTS):
- This is NOT a story or mini-documentary - it's a RANKING PRESENTATION
- NO narrative arcs, NO story structure, NO plot development
- Just present INTERESTING FACTS about their power, one per scene
- Each scene = ONE fact about their power/feats/abilities - rapid-fire information
- SCENE 1: One interesting fact about who they are or their power (transition from previous if needed, but keep it SHORT)
- SCENE 2+: More facts from KEY FEATS - their most impressive achievements, abilities, or lore demonstrations
- Final scene: One more fact or why they rank here (comparison context if relevant)
- NO need for conclusions or transitions between scenes within an entry - just string facts together

EMOTION GENERATION (CRITICAL):
- Each scene MUST include an "emotion" field - single word (e.g., "epic", "awe-inspiring", "powerful", "intense", "formidable")
- Base emotion on: the scale of power and the rank position
- Higher ranks (#1, #2, #3) should feel more epic/powerful than lower ranks
- Keep emotions high-energy and intense - this is a ranking video, not a contemplative documentary

CRITICAL REQUIREMENTS:
- HIGH ENERGY, FAST PACE: This is a ranking countdown - viewers want EPIC, QUICK, HYPER content
- ULTRA-FAST FACT-DENSE STYLE: Target ~3-5 seconds per scene (half a sentence to one very short sentence)
- Each scene = ONE interesting LoL lore fact about their power - NO filler, NO fluff, pure information
- Rely on INTERESTING PARTS OF LEAGUE OF LEGENDS LORE to engage - use compelling feats, shocking abilities, epic moments
- Maximum fact density - pack the most interesting, shocking, or epic power demonstration into minimal words
- Think TikTok/TikTok Reels style - rapid-fire facts, instant hooks, maximum engagement
- Every word must deliver value - if it doesn't immediately show their power or importance, cut it
- CRITICAL: Use KEY FEATS from the entry - these are the most interesting lore facts about their power
- Show specific, concrete feats and achievements - measurable power demonstrations that make the ranking make sense
- NO story structure needed - just present facts in a way that shows why they rank at this position
- Build momentum toward #1 through escalating power intensity in facts (more powerful = more epic facts)

BASIC SCENE REQUIREMENTS:
- Each scene needs: title, narration, image_prompt, emotion, year
- Keep narration focused on facts, not story

NARRATION STYLE:
- Write from YOUR perspective as the YouTuber
- Simple, direct language
- High-energy, enthusiastic tone
- Focus on interesting facts and power demonstrations

IMAGE PROMPTS:
- 16:9 cinematic format
- CRITICAL - EMOTION AND MOOD: Reflect the scene's emotion in lighting, composition, and mood
- Fantasy/magical elements appropriate for LoL lore
- Show the being's power visually - epic scale, magical effects, dramatic moments
- Include visual elements that demonstrate their power level
- For character type: include appearance details
- For entity/concept type: focus on visual representation of their power/essence
- End with ", 16:9 cinematic, highly detailed, fantasy art"

Respond with JSON array of exactly {num_scenes} scenes:
[
  {{"id": 1, "title": "2-5 words", "narration": "...", "image_prompt": "...", "emotion": "A single word describing scene emotion", "year": "timeline or 'various'"}},
  ...
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating high-energy top N ranking content. Write narration from YOUR perspective. Focus on interesting League of Legends lore facts about power and feats. Keep it fast-paced, fact-dense, and high-energy. NO story structure - just rank and present facts. Respond with valid JSON array only."},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.85,
    )
    
    scenes = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    # Validate required fields
    for i, scene in enumerate(scenes):
        if 'year' not in scene:
            scene['year'] = "various"
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
    
    return scenes


def generate_lol_scenes_for_chapter(subject: str, chapter: dict, scenes_per_chapter: int, start_id: int, 
                                     global_style: str, prev_chapter: dict = None, prev_scenes: list = None,
                                     central_theme: str = None, narrative_arc: str = None, 
                                     planted_seeds: list[str] = None, is_retention_hook_point: bool = False,
                                     timeline_start: str | None = None, timeline_end: str | None = None,
                                     subject_type: str = "character", tag_line: str | None = None, 
                                     overarching_plots: list[dict] = None, sub_plots: list[dict] = None) -> list[dict]:
    """Generate scenes for a single chapter of the LoL lore outline with continuity context."""
    
    if planted_seeds is None:
        planted_seeds = []
    
    # Build context from previous chapter for smooth transitions
    chapter_num = chapter.get('chapter_num', 1)
    is_hook_chapter = (chapter_num == 1)
    
    # LoL lore videos should be fast-paced, highlight-reel style (fantasy history, not real history)
    # Always use fast pacing for epic, quick, hyper cut-together videos
    pacing_instruction = """ULTRA-FAST FACT-DENSE STYLE (CRITICAL):
- This is FANTASY LORE - viewers want EPIC, QUICK, HYPER, CUT-TOGETHER videos like a highlight reel
- Keep narration ULTRA-SHORT and FACT-DENSE: Target ~3-5 seconds per scene (half a sentence to one very short sentence)
- Each scene = ONE interesting fact or moment - NO filler, NO fluff, NO explanations
- Think TikTok/TikTok Reels style - rapid-fire facts, instant hooks, pure information
- Every word must deliver value - cut anything that doesn't immediately grab attention
- Maximum fact density - pack the most interesting, shocking, or epic information into minimal words
- No transitions between scenes needed - just pure facts strung together
- Each scene should be a "wait, what?" moment that makes viewers want the next one
- If a sentence can be cut in half and still convey the fact, do it - shorter is always better"""
    
    if prev_chapter:
        timeline = prev_chapter.get('year_range') or prev_chapter.get('timeline', '?')
        prev_context = f"""PREVIOUS CHAPTER (for continuity):
Chapter {prev_chapter['chapter_num']}: "{prev_chapter['title']}" ({timeline})
Summary: {prev_chapter['summary']}
Emotional Tone: {prev_chapter['emotional_tone']}
"""
    elif is_hook_chapter:
        prev_context = """THIS IS THE HOOK CHAPTER - a rapid-fire "trailer" for the documentary.
ULTRA-FAST FACT-DENSE: Each scene = ONE shocking/interesting fact (~3-5 seconds).
Tease the most shocking, interesting, and dramatic moments from the ENTIRE lore story.
This is NOT chronological - jump around to the highlights that will make viewers stay.
NO filler - just pure interesting facts, one per scene.
End with a transition like "But how did it all begin?" to set up Chapter 2."""
    else:
        prev_context = "This is the OPENING chapter - establish the story with impact!"
    
    # Include last few scenes for continuity and to avoid overlapping events
    if prev_scenes and len(prev_scenes) > 0:
        recent_scenes = prev_scenes[-5:]
        scenes_context = "RECENT SCENES - AVOID REPEATING THESE EVENTS, maintain continuity to naturally continue the story:\n"
        for sc in recent_scenes:
            scenes_context += f"  Scene {sc.get('id')}: \"{sc.get('title')}\" - {sc.get('narration', '')[:150]}...\n"
        scenes_context += "\nCRITICAL: Do NOT repeat or overlap with events already covered in the scenes above. Each scene must cover DIFFERENT events. If an event was already described, move to its consequences or the next significant moment."
    else:
        scenes_context = ""
    
    # Get narrative context
    connects_to_next = chapter.get('connects_to_next', '')
    recurring_threads = chapter.get('recurring_threads', [])
    threads_str = ', '.join(recurring_threads) if recurring_threads else 'none specified'
    
    # Build central theme and narrative arc context
    theme_context = ""
    if central_theme:
        theme_context += f"\nCENTRAL THEME (connect all scenes to this): {central_theme}\n"
    if narrative_arc:
        theme_context += f"NARRATIVE ARC (we are at this point in the journey): {narrative_arc}\n"
    
    # Build callback context for planted seeds
    callback_context = ""
    if planted_seeds:
        callback_context = f"""
PLANTED SEEDS TO REFERENCE (create satisfying callbacks):
{chr(10).join(f"• {seed}" for seed in planted_seeds)}

These details were mentioned in earlier chapters. Reference them naturally in your scenes to create "aha moments" when earlier details suddenly matter. Don't force it - weave them in organically."""
    
    # Build plot context - this is CRITICAL for making the video feel like a story
    plot_context = ""
    if overarching_plots is None:
        overarching_plots = []
    if sub_plots is None:
        sub_plots = []
    
    # Get active plots for this chapter
    active_plots = chapter.get('plots_active', [])
    plot_developments = chapter.get('plot_developments', [])
    
    if active_plots or plot_developments or overarching_plots or sub_plots:
        plot_context = "\nOVERARCHING PLOTS & SUB-PLOTS - CRITICAL FOR STORY COHESION:\n"
        
        # Show all overarching plots
        if overarching_plots:
            plot_context += "OVERARCHING PLOTS (spanning multiple chapters):\n"
            for plot in overarching_plots:
                plot_name = plot.get('plot_name', '')
                plot_desc = plot.get('description', '')
                starts_ch = plot.get('starts_chapter', '?')
                peaks_ch = plot.get('peaks_chapter', '?')
                resolves_ch = plot.get('resolves_chapter', '?')
                current_chapter_num = chapter.get('chapter_num', 0)
                
                # Determine plot stage
                if current_chapter_num < starts_ch:
                    stage = "NOT YET STARTED"
                elif current_chapter_num == starts_ch:
                    stage = "BEGINS HERE - introduce this plot thread"
                elif starts_ch < current_chapter_num < peaks_ch:
                    stage = "DEVELOPING - advance this plot"
                elif current_chapter_num == peaks_ch:
                    stage = "PEAKS HERE - this is the climax of this plot"
                elif peaks_ch < current_chapter_num < resolves_ch:
                    stage = "RESOLVING - consequences unfolding"
                elif current_chapter_num == resolves_ch:
                    stage = "RESOLVES HERE - conclude this plot thread"
                else:
                    stage = "COMPLETED"
                
                plot_context += f"• \"{plot_name}\": {plot_desc} (Starts: Ch {starts_ch}, Peaks: Ch {peaks_ch}, Resolves: Ch {resolves_ch}) [{stage}]\n"
        
        # Show active sub-plots for this chapter
        relevant_subplots = []
        current_chapter_num = chapter.get('chapter_num', 0)
        if sub_plots:
            for subplot in sub_plots:
                subplot_chapters = subplot.get('chapters_span', [])
                if current_chapter_num in subplot_chapters:
                    relevant_subplots.append(subplot)
        
        if relevant_subplots:
            plot_context += "\nACTIVE SUB-PLOTS (developing in this chapter):\n"
            for subplot in relevant_subplots:
                subplot_name = subplot.get('subplot_name', '')
                subplot_desc = subplot.get('description', '')
                key_moments = subplot.get('key_moments', [])
                plot_context += f"• \"{subplot_name}\": {subplot_desc}\n"
                if key_moments:
                    plot_context += f"  Key moments: {', '.join(key_moments[:3])}\n"
        
        # Show plot developments for this chapter
        if plot_developments:
            plot_context += "\nPLOT DEVELOPMENTS IN THIS CHAPTER:\n"
            for dev in plot_developments:
                plot_context += f"• {dev}\n"
        
        plot_context += """
STORYTELLING REQUIREMENTS:
- Weave plot elements NATURALLY throughout your scenes - don't make it obvious
- Show plot threads developing through actual events, not exposition
- Connect scenes through plot progression, not just chronology
- Reference plot threads from earlier chapters when relevant
- Build tension by showing how plots are developing or converging
- Make viewers feel like they're following a story, not a timeline
- Each scene should advance at least one plot thread, even if subtly
- Show how different plots intersect and influence each other"""
    
    # Build retention hook instruction
    retention_hook_instruction = ""
    if is_retention_hook_point:
        retention_hook_instruction = """
CRITICAL RETENTION HOOK: This scene falls at a key retention point (~30s, 60s, or 90s mark). 
The FINAL scene in this chapter MUST end with a compelling hook:
- A surprising reveal or twist
- An unresolved question that makes viewers NEED to see what happens next
- A moment of high tension or conflict
- A "wait, what?" moment that creates curiosity
This hook is essential for YouTube algorithm performance - viewers must stay past this point."""
    
    # Get timeline info from chapter
    chapter_timeline = chapter.get('year_range') or chapter.get('timeline', '')
    
    # Build image prompt guidance based on subject type
    image_guidance = ""
    if subject_type == "character":
        # For champions, include age guidance similar to historical
        image_guidance = f"""- CRITICAL - CHARACTER APPEARANCE:
  * ALWAYS include {subject}'s appearance details relevant to the time period in the image_prompt
  * Include age-appropriate physical details, magical abilities visible, iconic weapons/items if relevant
  * Include period-accurate clothing, armor, or magical artifacts that reflect the timeline
  * Timeline context: {timeline_start if timeline_start else 'various eras'} to {timeline_end if timeline_end else 'Present Day'}"""
    else:
        # For regions/groups, emphasize fantasy/magical elements and world-building
        image_guidance = f"""- CRITICAL - LORE VISUALS:
  * Emphasize FANTASY and MAGICAL elements - mystical powers, magical settings, otherworldly landscapes
  * Include REGIONAL characteristics - architecture, geography, cultural elements specific to {subject}
  * Show FACTION relationships, conflicts, or alliances visually
  * Timeline context: {chapter_timeline}"""
    
    scene_prompt = f"""You are writing scenes {start_id}-{start_id + scenes_per_chapter - 1} of a {config.total_scenes}-scene lore documentary about {subject}.

{prev_context}
{scenes_context}
NOW WRITING CHAPTER {chapter['chapter_num']} of {config.chapters}: "{chapter['title']}"
Time Period: {chapter_timeline}
Emotional Tone: {chapter['emotional_tone']}
Dramatic Tension: {chapter['dramatic_tension']}
Recurring Themes: {threads_str}
Sets Up What Comes Next: {connects_to_next}

EMOTION GENERATION (CRITICAL):
- Each scene MUST include an "emotion" field - a single word or short phrase (e.g., "tense", "triumphant", "desperate", "contemplative", "exhilarating", "somber", "urgent", "defiant")
- Base the emotion on: what the character/subject is feeling at this moment, the dramatic tension, and the significance of the event
- The emotion should align with the chapter's emotional tone ({chapter['emotional_tone']}) but be scene-specific
- Use the emotion to guide both narration tone/style and image mood:
  * If emotion is "desperate" - narration should feel urgent/anxious with short, sharp sentences; image should show tense atmosphere, frantic expressions
  * If emotion is "triumphant" - narration should feel uplifting with elevated language; image should show celebration, confident expressions
  * If emotion is "contemplative" - narration should be slower, reflective; image should show quiet mood, thoughtful expressions
- The emotion field will be used to ensure narration tone and image mood match the emotional reality of the moment
{theme_context}
{plot_context}
{callback_context}
{pacing_instruction}
{retention_hook_instruction}

Chapter Summary: {chapter['summary']}

Key Events to Dramatize:
{chr(10).join(f"• {event}" for event in chapter.get('key_events', []))}

Generate EXACTLY {scenes_per_chapter} scenes that FLOW CONTINUOUSLY.

{"HOOK CHAPTER - INTRODUCTION/PREVIEW (NOT A STORY):" if is_hook_chapter else "STORYTELLING - this is a STORY, not a timeline:"}
{f'''CRITICAL: This is an INTRODUCTION/PREVIEW, not a story. It should quickly answer "Why should I watch this?"

STRUCTURE:
- SCENE 1 (COLD OPEN - ~3-5 SECONDS): Start with ONE shocking fact or moment about {subject}. Hook immediately - no introduction, just pure interesting fact. The first 3-5 seconds are CRITICAL for YouTube - this is what viewers see in search/preview. Make them NEED to know more with one epic, dramatic, or mysterious fact.
- SCENES 2-4: Rapid-fire preview of ONE shocking/interesting fact per scene (~3-5 seconds each) from the ENTIRE lore - epic battles, magical discoveries, betrayals, dramatic conflicts, surprising revelations
- NOT chronological - pick jaw-dropping highlights from any point in the timeline
- Each scene = ONE fact only - NO filler, NO explanations, just pure interesting information
- FINAL SCENE (~3-5 seconds): Natural bridge to the chronological story. DO NOT say "now the story rewinds" or "let's go back" - that's awkward. Instead, use something like: "But how did it all begin?", "It all started when...", or simply start the chronological narrative: "[Timeline/era context]. This is where our story begins."
- Tone: Exciting, intriguing preview. Ultra-fast like TikTok - pure facts, instant hooks.

NARRATION STYLE FOR INTRO:
- Ultra-short (~3-5 seconds per scene) - ONE fact per scene, maximum fact density
- Speak directly to the viewer about what they'll discover, but KEEP IT SHORT
- Use phrases like "You'll discover...", "Wait until you learn...", "This happened..." - but cut to the fact immediately
- Present facts as previews, not as events happening in real-time
- Every word must deliver value - if it doesn't hook or inform immediately, cut it
- If a sentence can be cut in half and still convey the fact, do it - shorter is always better''' if is_hook_chapter else '''CRITICAL STORYTELLING REQUIREMENTS:
- This MUST feel like a STORY being told, not a chronological list of events
- Each scene should ADVANCE a plot thread - show how overarching plots and sub-plots are developing
- Connect scenes through PLOT PROGRESSION, not just "this happened, then that happened"
- Show CAUSE AND EFFECT - why this event matters, what it leads to, how it changes things
- Build NARRATIVE TENSION through plot development - show conflicts developing, stakes rising, problems emerging or resolving
- Reference plot threads from earlier in the story naturally - make viewers feel like they're following a continuous story
- Each scene should answer "what's happening with the story?" not just "what happened next?"
- Think like a novelist: every scene should advance character development, plot, or theme
- EMOTIONAL ENGAGEMENT (CRITICAL): Make viewers feel emotionally invested by showing:
  * How events FEEL - the internal experience, fears, hopes, reactions of key characters/subjects
  * The EMOTIONAL SIGNIFICANCE - why this moment matters, not just what happened
  * What's at stake EMOTIONALLY - what failure would feel like, what success means
  * Details that create empathy - physical sensations, internal thoughts, personal costs
  * Make events feel REAL and SIGNIFICANT by connecting them to feelings and experiences
- PIVOTAL MOMENTS - SIGNIFICANCE (CRITICAL): For the most PIVOTAL moments in this chapter (major breakthroughs, turning points, critical decisions, moments that change the story trajectory), you MUST explicitly explain WHY this moment matters - its significance to the overall story, its impact on the broader narrative, and what it changes. Make viewers FEEL the weight of these moments. Don't just describe what happened - explain WHY it's pivotal and HOW it reshapes what comes after.
- Make viewers feel emotionally invested in HOW the story unfolds, not just WHAT happened - pull them into the emotional reality
- CRITICAL: Each scene must cover DIFFERENT, NON-OVERLAPPING events. Do NOT repeat events already covered in previous scenes. If Scene X describes Event A in detail, Scene Y should NOT re-describe Event A - instead, move to Event A's consequences, what happens next, or a different event entirely. Review the recent scenes context to ensure you're not overlapping with what was already told.

TRANSITIONS AND FLOW:
- Scene 1 should TRANSITION smoothly from where the previous scene ended AND advance the story
- Each scene should CONNECT to the next through plot progression - what happens next in the story?
- Reference recurring themes/motifs from earlier in the documentary through plot connections
- The final scene should SET UP what comes next by advancing or introducing plot threads
- Think of scenes as story beats in a film, not separate fact segments'''}

{"" if is_hook_chapter else get_shared_scene_flow_instructions()}

{get_shared_scene_requirements()}

9. PLANT SEEDS (Early in the story): If this is early in the documentary, include specific details, objects, relationships, magical items, or concepts that could pay off later. Examples: a specific magical artifact mentioned, a relationship that will matter later, a prophecy or warning that will be relevant, a small detail that seems unimportant now but will become significant. These create satisfying "aha moments" when referenced later.

{get_shared_narration_style(is_short=False)}

LOL LORE PACING (CRITICAL - Overrides default pacing):
- This is FANTASY LORE - viewers want EPIC, QUICK, HYPER, CUT-TOGETHER videos like a highlight reel
- Keep narration ULTRA-SHORT and FACT-DENSE: Target ~3-5 seconds per scene (half a sentence to one very short sentence)
- Each scene = ONE interesting fact or moment - NO filler, NO fluff, pure information
- Think TikTok/TikTok Reels style - rapid-fire facts, instant hooks, maximum fact density
- Every word must deliver value - cut anything that doesn't immediately grab attention
- No long explanations or drawn-out descriptions - just pure interesting facts
- Each scene should be a "wait, what?" moment - epic moments, powerful feats, shocking revelations
- Keep it moving - ultra-fast pacing, even for emotional moments
- If a sentence can be cut in half and still convey the fact, do it - shorter is always better

{("- HOOK/INTRO: Speak directly to the viewer about what they'll discover. Use preview language: \"You'll discover...\", \"This is the story of...\", \"Wait until you learn...\", \"Here's why this matters...\" Present facts as teasers, not as events happening in real-time.\n" +
"- TRANSITION TO STORY: The final scene should naturally bridge to the chronological narrative without saying \"rewind\" or \"go back\". Simply start with the beginning context: \"[Timeline/era context]. This is where our story begins.\" or \"It all started when...\"") if is_hook_chapter else ""}

IMAGE PROMPT STYLE:
- Cinematic, dramatic lighting
- Specific composition (close-up, wide shot, over-shoulder, etc.)
- CRITICAL - EMOTION AND MOOD (MUST MATCH SCENE'S EMOTION FIELD):
  * The scene's "emotion" field MUST be reflected in the image_prompt - this is critical for visual consistency
  * Use the emotion to guide lighting, composition, facial expressions, and overall mood
  * Examples: "desperate" → tense atmosphere, frantic expressions, harsh shadows, chaotic composition; "triumphant" → bright lighting, confident expressions, celebratory mood; "contemplative" → soft lighting, thoughtful expressions, quiet atmosphere; "tense" → sharp contrasts, wary expressions, claustrophobic framing
  * Explicitly include the emotion in image description: "tense atmosphere", "triumphant expression", "contemplative mood", "desperate urgency"
  * The visual mood must match the emotional reality of the moment
{image_guidance}
- VISUAL THEME REINFORCEMENT: If recurring themes include concepts like "isolation", "ambition", "conflict", "magic", "corruption", etc., incorporate visual motifs that reinforce these themes through composition, lighting, or symbolism
- Recurring Themes to Consider Visually: {threads_str}
- FANTASY/MAGICAL ELEMENTS: Emphasize mystical powers, magical settings, otherworldly landscapes, regional architecture, faction symbols - make it feel like League of Legends lore
- End with ", 16:9 cinematic"

Respond with JSON array:
[
  {{
    "id": {start_id},
    "title": "Evocative 2-5 word title",
    "narration": "Vivid, dramatic narration...",
    "image_prompt": "Detailed visual description with fantasy/magical elements, {subject_type}-appropriate details, 16:9 cinematic",
    "emotion": "A single word or short phrase describing the scene's emotional tone (e.g., 'tense', 'triumphant', 'desperate', 'contemplative', 'exhilarating', 'somber', 'urgent', 'defiant'). Should match the chapter's emotional tone but be scene-specific.",
    "year": YYYY or "YYYY-YYYY" or timeline era (the specific year/timeline when this scene takes place)
  }},
  ...
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating lore documentary content. Write narration from YOUR perspective - this is YOUR script that YOU wrote. Tell the story naturally, directly to the viewer. Avoid any meta references to chapters, production elements, or the script structure. Focus on what actually happened in the lore, why it mattered, and how it felt. Respond with valid JSON array only."},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.85,
    )
    
    scenes = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    # Validate that each scene has required fields
    for i, scene in enumerate(scenes):
        if 'year' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'year' field")
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
    
    return scenes


def generate_lol_short_outline(subject: str, main_outline: dict, short_num: int, total_shorts: int, previously_covered_stories: list[str] = None, scene_highlights: list = None) -> dict:
    """Generate a focused outline for a single LoL lore YouTube Short."""
    
    if previously_covered_stories is None:
        previously_covered_stories = []
    if scene_highlights is None:
        scene_highlights = []
    
    chapters_context = "\n".join([
        f"• {ch['title']} ({ch.get('year_range') or ch.get('timeline', '?')}): {ch['summary']}"
        for ch in main_outline.get("chapters", [])
    ])
    
    # Different focus for each short
    focus_options = [
        "the most SHOCKING or little-known fact",
        "the most DRAMATIC conflict or battle",
        "the most MIND-BLOWING magical event or discovery",
        "a TRAGIC or heartbreaking moment",
        "a MYSTERIOUS or unexplained aspect of the lore",
    ]
    focus = focus_options[(short_num - 1) % len(focus_options)]
    
    # Build context about previously covered stories
    previously_covered_text = ""
    if previously_covered_stories:
        previously_covered_text = f"""

CRITICAL: You must choose a DIFFERENT story than what was already covered in previous shorts.
Previously covered stories (DO NOT repeat these):
{chr(10).join(f"• {story}" for story in previously_covered_stories)}

Choose a COMPLETELY DIFFERENT story/incident/event from the {subject} lore. Make sure it's distinct and doesn't overlap with the above."""
    
    # Build scene highlights context for shorts
    scene_highlights_text = ""
    if scene_highlights:
        scene_highlights_text = f"""
KEY MOMENTS FROM MAIN VIDEO (reference these to create connections):
{chr(10).join(f"• Scene {s.get('id')}: {s.get('title')} - {s.get('narration_preview', '')}" for s in scene_highlights[:15])}

IMPORTANT: This short should feel like a TEASER that drives viewers to the main video. Reference specific moments from the main video and create intrigue: "Wait until you see how this plays out in the full documentary" or "This moment from the main story gets explored in depth..." Make viewers want to watch the full video to see how this story connects to the bigger picture."""
    
    outline_prompt = f"""Create a VIRAL YouTube Short about {subject} lore.

Short #{short_num} of {total_shorts}. Focus on: {focus}
{previously_covered_text}

The lore story (for context):
{chapters_context}
{scene_highlights_text}

This Short tells ONE CONTINUOUS STORY from the {subject} lore - not disconnected facts, but a single narrative arc.

IMPORTANT: This must be a COMPLETELY DIFFERENT story from any previous shorts. Choose a distinct incident, event, moment, or period from the lore that hasn't been covered yet. For example, if a previous short covered an early battle, choose a different story like a later conflict, a different magical discovery from another era, a relationship story, or a different challenge. Each short should feel like a standalone story from a different part of the lore timeline.

This Short has EXACTLY 5 scenes that tell ONE COMPLETE STORY with depth and detail. The final scene should be a NATURAL CONCLUSION, not a hook/cliffhanger.

STRUCTURE (5 scenes):
1. SCENE 1: Start the story. Set up a specific incident, moment, or event from the lore. Give context and specific details. Hook the viewer with the opening of the story.
2. SCENE 2: Build the story. Show what happened next, the development, or how the situation evolved. Add more depth and specific details.
3. SCENE 3: Escalate the story. Show the consequences, conflicts, challenges, or complications. This is the rising action - make it engaging and detailed.
4. SCENE 4: Continue building. Show how the story develops further, what happened, and the stakes involved. Add emotional depth.
5. SCENE 5: NATURAL CONCLUSION. This should provide a satisfying ending to THIS story. Show what happened as a result, how it resolved, or what the outcome was. Do NOT end with a cliffhanger. Instead, provide a natural conclusion that makes the viewer feel the story is complete.

CRITICAL RULES:
- Tell ONE continuous story - scenes must flow like a mini-narrative, not separate facts
- Every scene must contain SPECIFIC, INTERESTING FACTS about the lore
- NO vague statements or filler
- NO artistic/poetic language
- Simple, clear sentences packed with information
- The short should feel like watching a mini-story, not a list of facts
- The ending should be a NATURAL conclusion that makes sense for the story being told
- Emphasize FANTASY/MAGICAL elements, epic battles, conflicts, and dramatic lore moments

Provide JSON:
{{
  "short_title": "VIRAL CLICKBAIT title (max 50 chars). Use power words: SHOCKING, SECRET, EXPOSED, INSANE, UNBELIEVABLE, MIND-BLOWING, CRAZY, FORBIDDEN. Create curiosity gaps. Ask questions. Use numbers. Make bold claims. Examples: 'The SHOCKING Secret Nobody Knows', 'How {subject} Did the IMPOSSIBLE', 'This Changed EVERYTHING', 'The Dark Truth Exposed', 'You WON'T Believe This'. Must make viewers NEED to watch immediately while staying accurate to LoL lore.",
  "short_description": "YouTube description (100 words) with hashtags related to League of Legends lore",
  "tags": "10-15 SEO tags comma-separated (mix of {subject}, League of Legends, LoL lore, Runeterra, etc.)",
  "thumbnail_prompt": "CLICKBAIT thumbnail for mobile (9:16 vertical): EXTREME close-up or dramatic composition, intense emotional expression (not neutral - show passion/conflict/shock), HIGH CONTRAST dramatic lighting (chiaroscuro), bold eye-catching colors (reds/yellows for urgency), subject in MOMENT OF IMPACT or dramatic pose, fantasy/magical elements, symbolic elements representing the story's peak moment, movie-poster energy, optimized for small mobile screens - must grab attention instantly when scrolling",
  "hook_fact": "The opening fact that starts the story (for context, but we start with the story, not just the fact)",
  "story_angle": "What specific continuous story/incident are we telling (ONE narrative arc)",
  "key_facts": ["5-8 specific facts to include across the 5 scenes that tell ONE complete story with depth"]
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You create viral LoL lore content. Every word must deliver value. No fluff. When generating multiple shorts about the same subject, ensure each tells a COMPLETELY DIFFERENT story from the lore - different events, different moments, different incidents. Respond with valid JSON only."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.9,
        response_format={"type": "json_object"}
    )
    
    return json.loads(clean_json_response(response.choices[0].message.content))


def generate_lol_short_scenes(subject: str, short_outline: dict, subject_type: str = "character", timeline_start: str | None = None, timeline_end: str | None = None) -> list[dict]:
    """Generate all 5 scenes for a LoL lore YouTube Short (complete story with natural conclusion).
    
    Each scene must include a "year" field indicating when it takes place (timeline/era).
    """
    
    key_facts = short_outline.get('key_facts', [])
    facts_str = "\n".join(f"• {fact}" for fact in key_facts)
    
    # Build image prompt guidance based on subject type
    image_guidance = ""
    if subject_type == "character":
        image_guidance = f"""- CRITICAL - CHARACTER APPEARANCE:
  * Include {subject}'s appearance details relevant to the time period
  * Include age-appropriate physical details, magical abilities visible, iconic weapons/items if relevant
  * Timeline context: {timeline_start if timeline_start else 'various eras'} to {timeline_end if timeline_end else 'Present Day'}"""
    else:
        image_guidance = f"""- CRITICAL - LORE VISUALS:
  * Emphasize FANTASY and MAGICAL elements - mystical powers, magical settings, otherworldly landscapes
  * Include REGIONAL characteristics - architecture, geography, cultural elements specific to {subject}
  * Show FACTION relationships, conflicts, or alliances visually
  * Timeline context: {timeline_start if timeline_start else 'various eras'} to {timeline_end if timeline_end else 'Present Day'}"""
    
    scene_prompt = f"""Write 5 scenes for a YouTube Short about {subject} lore.

TITLE: "{short_outline.get('short_title', '')}"
STORY ANGLE: {short_outline.get('story_angle', '')}

KEY FACTS TO USE:
{facts_str}

CRITICAL: This short tells ONE COMPLETE, CONTINUOUS STORY from the {subject} lore with DEPTH and DETAIL. The scenes must flow like a mini-narrative, not disconnected facts. This should be high-quality content that is enjoyable on its own.

EMOTION GENERATION (CRITICAL):
- Each scene MUST include an "emotion" field - a single word or short phrase (e.g., "tense", "triumphant", "desperate", "contemplative", "exhilarating", "somber", "urgent", "defiant")
- Base the emotion on: what characters/subjects are feeling at this moment, the dramatic tension, and the significance of the event
- Use the emotion to guide both narration tone/style and image mood:
  * If emotion is "desperate" - narration should feel urgent/anxious with short, sharp sentences; image should show tense atmosphere, frantic expressions
  * If emotion is "triumphant" - narration should feel uplifting with elevated language; image should show celebration, confident expressions
  * If emotion is "contemplative" - narration should be slower, reflective; image should show quiet mood, thoughtful expressions
- The emotion field will be used to ensure narration tone and image mood match the emotional reality of the moment

STRUCTURE (exactly 5 scenes):
1. SCENE 1: Start the story. Set up a specific incident, moment, or event. Give context and specific details. Hook the viewer by showing how this moment FELT - what characters were thinking, fearing, or hoping. Make them feel the significance.
2. SCENE 2: Build the story. Show what happened next, the development, or how the situation evolved. Add more depth and specific details about who was involved, what happened, what the stakes were. Include emotional details - how does this feel? What's at stake emotionally?
3. SCENE 3: Escalate the story. Show the consequences, conflicts, challenges, or complications. This is the rising action - make it engaging and detailed. Show what made this moment difficult or significant EMOTIONALLY - what fears, hopes, or pressures do characters face?
4. SCENE 4: Continue building. Show how the story develops further, what happens, and the stakes involved. Add emotional depth - show relationships, motivations, and the personal impact. What does this mean? How do characters FEEL?
5. SCENE 5: NATURAL CONCLUSION. This should provide a satisfying ending to THIS story. Show what happened as a result, how it resolved, or what the outcome was. Include the emotional significance - what did this mean? Examples of good conclusions:
   - "The battle ends in victory. The region is secured. But the cost of war echoes for generations."
   - "The magical discovery changes everything. Within months, it reshapes how mages understand the world. But with power comes danger."
   
   DO NOT end with a cliffhanger or hook like:
   - "But what happened next would change everything..." (NO - this is a hook, not a conclusion)
   - "The consequences of this decision would haunt them for years..." (NO - unresolved)
   - "Little did they know, this was just the beginning..." (NO - this is a hook)
   
   Instead, provide a natural, satisfying conclusion that makes the viewer feel the story is complete.

{get_shared_scene_flow_instructions()}

{get_shared_scene_requirements()}

{get_shared_narration_style(is_short=True)}

{get_shared_examples()}

IMAGE PROMPTS:
- Vertical 9:16, dramatic, mobile-optimized
- High contrast, single clear subject
- CRITICAL - EMOTION AND MOOD (MUST MATCH SCENE'S EMOTION FIELD):
  * The scene's "emotion" field MUST be reflected in the image_prompt - this is critical for visual consistency
  * Use the emotion to guide lighting, composition, facial expressions, and overall mood
  * Examples: "desperate" → tense atmosphere, frantic expressions, harsh shadows; "triumphant" → bright lighting, confident expressions; "contemplative" → soft lighting, thoughtful expressions
  * Explicitly include the emotion in image description: "tense atmosphere", "triumphant expression", "contemplative mood"
{image_guidance}
- FANTASY/MAGICAL ELEMENTS: Emphasize mystical powers, magical settings, otherworldly landscapes, regional architecture, faction symbols - make it feel like League of Legends lore
- End with ", 9:16 vertical"

Respond with JSON array of exactly 5 scenes:
[
  {{"id": 1, "title": "2-4 words", "narration": "...", "image_prompt": "... (include fantasy/magical elements and {subject_type}-appropriate details)", "emotion": "A single word or short phrase (e.g., 'tense', 'triumphant', 'desperate', 'contemplative')", "year": "YYYY or YYYY-YYYY or timeline era"}},
  {{"id": 2, "title": "...", "narration": "...", "image_prompt": "... (include fantasy/magical elements and {subject_type}-appropriate details)", "emotion": "A single word or short phrase describing the scene's emotional tone", "year": "YYYY or YYYY-YYYY or timeline era"}},
  {{"id": 3, "title": "...", "narration": "...", "image_prompt": "... (include fantasy/magical elements and {subject_type}-appropriate details)", "emotion": "A single word or short phrase describing the scene's emotional tone", "year": "YYYY or YYYY-YYYY or timeline era"}},
  {{"id": 4, "title": "...", "narration": "...", "image_prompt": "... (include fantasy/magical elements and {subject_type}-appropriate details)", "emotion": "A single word or short phrase describing the scene's emotional tone", "year": "YYYY or YYYY-YYYY or timeline era"}},
  {{"id": 5, "title": "...", "narration": "...", "image_prompt": "... (include fantasy/magical elements and {subject_type}-appropriate details)", "emotion": "A single word or short phrase describing the scene's emotional tone", "year": "YYYY or YYYY-YYYY or timeline era"}}
]

IMPORTANT: Each scene must include:
- "year" field indicating when the scene takes place (timeline/era)
- "emotion" field - a single word or short phrase (e.g., "tense", "triumphant", "desperate", "contemplative") that describes the scene's emotional tone. Base this on what characters/subjects are feeling, the dramatic tension, and the significance of the moment. The narration tone and image mood should match this emotion.
- The image_prompt MUST include fantasy/magical elements, {subject_type}-appropriate visual details, and reflect the scene's emotion in lighting, composition, and mood.
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating viral LoL lore content. Write narration from YOUR perspective - this is YOUR script. Simple words, specific facts, deep storytelling with details. No fluff, no made-up transitions. Tell continuous stories with actual lore events. Avoid any meta references to chapters, production elements, or script structure. Respond with valid JSON array only."},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.85,
    )
    
    scenes = json.loads(clean_json_response(response.choices[0].message.content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    # Validate that each scene has required fields
    for i, scene in enumerate(scenes):
        if 'year' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'year' field")
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
    
    return scenes


def generate_lol_shorts(subject_of_interest: str, main_title: str, global_block: str, outline: dict, base_output_path: str, scene_highlights: list = None):
    """Generate LoL lore YouTube Shorts (5 scenes each: complete story with natural conclusion)."""
    if scene_highlights is None:
        scene_highlights = []
    if config.num_shorts == 0:
        print("\n[SHORTS] Skipped (--shorts 0)")
        return []
    
    print(f"\n{'='*60}")
    print(f"[SHORTS] Generating {config.num_shorts} YouTube Short(s)")
    print(f"[SHORTS] Structure: {config.total_short_scenes} scenes each (complete story with natural conclusion)")
    print(f"{'='*60}")
    
    SHORTS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    
    generated_shorts = []
    base_name = Path(base_output_path).stem.replace("_script", "")
    previously_covered_stories = []  # Track story angles to ensure diversity
    
    # Get subject type from outline
    subject_type = outline.get('subject_type', 'character')
    timeline_start = outline.get('timeline_start')
    timeline_end = outline.get('timeline_end')
    
    # STEP 1: Generate all outlines sequentially (they depend on previously_covered_stories)
    print(f"\n[OPTIMIZATION] Step 1: Generating all short outlines sequentially...")
    short_outlines = []
    for short_num in range(1, config.num_shorts + 1):
        print(f"\n[SHORT {short_num}/{config.num_shorts}] Creating outline...")
        
        short_outline = generate_lol_short_outline(
            subject=subject_of_interest,
            main_outline=outline,
            short_num=short_num,
            total_shorts=config.num_shorts,
            previously_covered_stories=previously_covered_stories,
            scene_highlights=scene_highlights
        )
        
        story_angle = short_outline.get("story_angle", "")
        if story_angle:
            previously_covered_stories.append(story_angle)
        
        short_outlines.append((short_num, short_outline))
        print(f"[SHORT {short_num}] Title: {short_outline.get('short_title', f'Short {short_num}')}")
    
    # STEP 2: Generate all scenes in parallel (independent once outlines are done)
    print(f"\n[OPTIMIZATION] Step 2: Generating all short scenes in parallel...")
    
    def generate_single_short(short_num, short_outline):
        """Helper function to generate a complete short (scenes + save)."""
        try:
            print(f"[SHORT {short_num}] Generating {config.total_short_scenes} scenes...")
            all_scenes = generate_lol_short_scenes(
                subject=subject_of_interest,
                short_outline=short_outline,
                subject_type=subject_type,
                timeline_start=timeline_start,
                timeline_end=timeline_end
            )
            print(f"[SHORT {short_num}] → {len(all_scenes)} scenes generated")
            
            # Fix scene IDs
            for i, scene in enumerate(all_scenes):
                scene["id"] = i + 1
            
            # Refine short scenes
            short_context = f"Story: {short_outline.get('story_angle', '')}. Title: {short_outline.get('short_title', '')}"
            # Determine short file path first for diff path
            short_file = SHORTS_DIR / f"{base_name}_short{short_num}.json"
            diff_path = short_file.parent / f"{short_file.stem}_refinement_diff.json" if config.generate_refinement_diffs else None
            all_scenes, refinement_diff = refine_scenes_lol(all_scenes, subject_of_interest, is_short=True, chapter_context=short_context, diff_output_path=diff_path)
            
            # Generate thumbnail (if enabled)
            thumbnail_path = None
            if config.generate_short_thumbnails:
                thumbnail_prompt = short_outline.get("thumbnail_prompt", "")
                if thumbnail_prompt:
                    thumbnail_prompt += "\n\nCLICKBAIT YouTube Shorts thumbnail - MAXIMIZE SCROLL-STOPPING POWER: Vertical 9:16, EXTREME close-up or dramatic composition, intense emotional expression (shock/passion/conflict), HIGH CONTRAST dramatic lighting (chiaroscuro), bold eye-catching colors (reds/yellows/oranges for urgency), subject in MOMENT OF IMPACT, fantasy/magical elements, movie-poster energy, optimized for mobile scrolling - must instantly grab attention when tiny in feed."
                    thumb_file = THUMBNAILS_DIR / f"{base_name}_short{short_num}_thumbnail.png"
                    thumbnail_path = generate_thumbnail(thumbnail_prompt, thumb_file, size="1024x1536")
            
            # Build short output
            short_title = short_outline.get("short_title", f"Short {short_num}")
            short_output = {
                "metadata": {
                    "short_id": short_num,
                    "title": short_title,
                    "description": short_outline.get("short_description", ""),
                    "tags": short_outline.get("tags", ""),
                    "hook_fact": short_outline.get("hook_fact", ""),
                    "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
                    "story_angle": short_outline.get("story_angle", ""),
                    "subject_of_interest": subject_of_interest,
                    "main_video_title": main_title,
                    "global_block": global_block,
                    "num_scenes": len(all_scenes),
                    "outline": short_outline
                },
                "scenes": all_scenes
            }
            
            # Save short
            with open(short_file, "w", encoding="utf-8") as f:
                json.dump(short_output, f, indent=2, ensure_ascii=False)
            
            print(f"[SHORT {short_num}] ✓ Saved: {short_file} ({len(all_scenes)} scenes)")
            
            return {
                "file": str(short_file),
                "title": short_title,
                "scenes": len(all_scenes)
            }
        except Exception as e:
            print(f"[SHORT {short_num}] ERROR: {e}")
            raise
    
    # Generate all shorts in parallel (up to 3 concurrent API calls)
    max_workers = min(3, config.num_shorts)  # Limit concurrent API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_single_short, short_num, short_outline): short_num 
            for short_num, short_outline in short_outlines
        }
        
        for future in as_completed(futures):
            short_num = futures[future]
            try:
                result = future.result()
                generated_shorts.append(result)
            except Exception as e:
                print(f"[SHORT {short_num}] Failed: {e}")
                raise
    
    # Sort by short_id to maintain order
    generated_shorts.sort(key=lambda x: int(Path(x['file']).stem.split('_short')[1]))
    
    return generated_shorts


def generate_lol_script(subject_of_interest: str, output_path: str):
    """Generate a complete LoL lore documentary script using outline-guided generation."""
    
    print(f"\n{'='*60}")
    if config.top_n_mode:
        print(f"[SCRIPT] Generating TOP {config.top_n_entries} list script for: {subject_of_interest}")
    else:
        print(f"[SCRIPT] Generating lore script for: {subject_of_interest}")
    print(f"{'='*60}")
    if config.generate_main:
        if config.top_n_mode:
            print(f"[CONFIG] Top {config.top_n_entries} video mode: intro + {config.top_n_entries} entries (variable scenes per entry)")
        else:
            print(f"[CONFIG] Main video: {config.chapters} chapters × {config.scenes_per_chapter} scenes = {config.total_scenes} scenes")
    else:
        print(f"[CONFIG] Main video: SKIPPED")
    if config.num_shorts > 0:
        print(f"[CONFIG] Shorts: {config.num_shorts} × {config.total_short_scenes} scenes = {config.num_shorts * config.total_short_scenes} scenes")
    else:
        print(f"[CONFIG] Shorts: SKIPPED")
    print(f"[CONFIG] Thumbnails: {'Yes' if config.generate_thumbnails else 'No'}")
    print(f"[CONFIG] Model: {SCRIPT_MODEL}")
    
    # Handle Top N mode
    if config.top_n_mode:
        return generate_lol_top10_script(subject_of_interest, output_path)
    
    # Step 1: Generate detailed outline (needed for both main and shorts)
    print("\n[STEP 1] Creating lore outline...")
    outline = generate_lol_outline(subject_of_interest)
    chapters = outline.get("chapters", [])
    subject_type = outline.get("subject_type", "character")
    timeline_start = outline.get("timeline_start")
    timeline_end = outline.get("timeline_end")
    
    if config.generate_main and len(chapters) < config.chapters:
        print(f"[WARNING] Got {len(chapters)} chapters, expected {config.chapters}")
    
    # Step 2: Generate initial metadata (title, thumbnail, global_block) - we'll regenerate description after scenes
    print("\n[STEP 2] Generating initial metadata...")
    
    initial_metadata_prompt = f"""Create initial metadata for a League of Legends lore documentary about: {subject_of_interest}

The story in one line: {outline.get('tagline', '')}

Subject type: {subject_type} (character/champion, region, or topic/group)

Generate JSON:
{{
  "title": "CLICKBAIT YouTube title (60-80 chars). Use power words like: SHOCKING, SECRET, REVEALED, EXPOSED, DARK, UNTOLD, UNBELIEVABLE, INCREDIBLE, INSANE, CRAZY, MIND-BLOWING, FORBIDDEN, HIDDEN. Create curiosity gaps, ask questions, use numbers, make bold claims. Examples: 'The SHOCKING Secret That Changed Everything', 'How {subject_of_interest} Did the IMPOSSIBLE', 'The Dark Truth They DON'T Want You to Know', '{subject_of_interest}: The Forbidden Discovery', 'This ONE Decision Changed Runeterra FOREVER'. Must be engaging and make viewers NEED to click while staying accurate to LoL lore.",
  "tag_line": "Short, succinct, catchy tagline (5-10 words) that captures the lore subject. Examples: 'the champion who changed everything', 'the region torn by war', 'the mystical beings of legend', 'the forgotten empire'. Should be memorable and accurate.",
  "thumbnail_description": "CLICKBAIT thumbnail visual: Maximize visual impact! Use intense close-ups, dramatic expressions, extreme lighting (chiaroscuro), bold colors (red/yellow for danger/urgency), powerful symbols, emotional moments. Show conflict, tension, or peak dramatic moment. Composition should be bold and arresting - eyes staring directly, hands in dramatic pose, symbolic objects. Use high contrast, dramatic shadows, cinematic lighting. Subject should appear in a MOMENT OF IMPACT - not calm portrait. Think action movie poster, not museum painting. Include fantasy/magical elements appropriate for League of Legends lore. NO TEXT in image, but visually SCREAM importance and drama.",
  "global_block": "Visual style guide (300-400 words): semi-realistic digital painting style with fantasy elements, color palette, dramatic lighting, how {subject_of_interest} should appear consistently across {config.total_scenes} scenes. Emphasize magical/mystical elements, regional characteristics, and League of Legends lore aesthetics."
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "League of Legends lore documentary producer. Respond with valid JSON only."},
            {"role": "user", "content": initial_metadata_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    initial_metadata = json.loads(clean_json_response(response.choices[0].message.content))
    
    title = initial_metadata["title"]
    tag_line = initial_metadata.get("tag_line", f"the lore of {subject_of_interest}")  # Fallback if not generated
    thumbnail_description = initial_metadata["thumbnail_description"]
    global_block = initial_metadata["global_block"]
    
    print(f"[METADATA] Title: {title}")
    print(f"[METADATA] Tag line: {tag_line}")
    
    # Generate main video thumbnail (only if generating main video)
    generated_thumb = None
    if config.generate_main:
        print("\n[THUMBNAIL] Main video thumbnail...")
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        thumbnail_path = THUMBNAILS_DIR / f"{Path(output_path).stem}_thumbnail.png"
        
        thumbnail_prompt = f"""{thumbnail_description}

YouTube CLICKBAIT thumbnail - MUST MAXIMIZE CLICKS:
- EXTREME close-up or dramatic wide shot with strong composition
- Intense emotional expression - not neutral, show passion/conflict/determination
- Dramatic lighting with HIGH CONTRAST (chiaroscuro) - bright highlights, deep shadows
- Bold, eye-catching colors (reds, yellows, oranges for urgency/importance) against darker backgrounds
- Subject in ACTION or MOMENT OF IMPACT - not passive pose
- Symbolic elements that represent their greatest achievement or conflict
- Fantasy/magical elements appropriate for League of Legends lore
- Cinematic, movie-poster quality - think Marvel movie poster energy
- Optimized for small sizes - subject must be CLEARLY visible even when tiny
- Background should be dramatic but not distract from subject
- Overall feeling: URGENT, IMPORTANT, UNMISSABLE"""
        
        generated_thumb = generate_thumbnail(thumbnail_prompt, thumbnail_path)
    
    # Step 3: Generate scenes chapter by chapter (only if generating main video)
    all_scenes = []
    planted_seeds = []  # Track details from early chapters for callbacks
    
    if config.generate_main:
        print(f"\n[STEP 3] Generating {config.total_scenes} scenes from {len(chapters)} chapters...")
        
        # Get central theme, narrative arc, and plots from outline
        central_theme = outline.get('central_theme', '')
        narrative_arc = outline.get('narrative_arc', '')
        overarching_plots = outline.get('overarching_plots', [])
        sub_plots = outline.get('sub_plots', [])
        
        for i, chapter in enumerate(chapters):
            start_id = len(all_scenes) + 1
            
            # Get previous chapter and scenes for continuity
            prev_chapter = chapters[i - 1] if i > 0 else None
            
            # Determine if this is a retention hook point (~30s, 60s, 90s marks)
            # Assuming ~15 seconds per scene on average
            estimated_time = start_id * 15
            is_retention_hook = (estimated_time >= 25 and estimated_time <= 35) or \
                               (estimated_time >= 55 and estimated_time <= 65) or \
                               (estimated_time >= 85 and estimated_time <= 95)
            
            print(f"\n[CHAPTER {chapter['chapter_num']}/{len(chapters)}] {chapter['title']}")
            if is_retention_hook:
                print(f"  ⚠ RETENTION HOOK POINT (~{estimated_time}s mark)")
            print(f"  Generating {config.scenes_per_chapter} scenes...")
            
            try:
                scenes = generate_lol_scenes_for_chapter(
                    subject=subject_of_interest,
                    chapter=chapter,
                    scenes_per_chapter=config.scenes_per_chapter,
                    start_id=start_id,
                    global_style=global_block,
                    prev_chapter=prev_chapter,
                    prev_scenes=list(all_scenes),  # Copy of scenes so far
                    central_theme=central_theme,
                    narrative_arc=narrative_arc,
                    planted_seeds=planted_seeds if i > 0 else [],  # Only pass seeds after first chapter
                    is_retention_hook_point=is_retention_hook,
                    timeline_start=timeline_start,
                    timeline_end=timeline_end,
                    subject_type=subject_type,
                    tag_line=tag_line if i == 0 else None,  # Only pass tag_line for first chapter (hook chapter)
                    overarching_plots=overarching_plots,
                    sub_plots=sub_plots
                )
                
                if len(scenes) != config.scenes_per_chapter:
                    print(f"  [WARNING] Got {len(scenes)} scenes, expected {config.scenes_per_chapter}")
                
                all_scenes.extend(scenes)
                print(f"  ✓ {len(scenes)} scenes (total: {len(all_scenes)})")
                
                # Insert CTA transition scene after chapter 1 (hook chapter) and before chapter 2 (story begins)
                if i == 0:  # After chapter 1
                    print(f"\n[CTA] Generating transition/CTA scene between chapters 1 and 2...")
                    try:
                        cta_scene = generate_cta_transition_scene_lol(subject_of_interest, tag_line)
                        all_scenes.append(cta_scene)
                        print(f"  ✓ CTA transition scene added (total: {len(all_scenes)})")
                    except Exception as e:
                        print(f"  [WARNING] Failed to generate CTA scene: {e}")
                        # Continue without CTA scene if generation fails
                
                # Extract "planted seeds" from early chapters (first 3 chapters) for callback mechanism
                if i < 3:  # First 3 chapters plant seeds
                    for scene in scenes:
                        narration = scene.get('narration', '')
                        scene_title = scene.get('title', '')
                        if narration:
                            planted_seeds.append(f"{scene_title}: {narration[:80]}...")
                    # Also add chapter's key events as potential seeds
                    for event in chapter.get('key_events', []):
                        # Extract specific details that could be callbacks (LoL-appropriate: artifacts, magical items, prophecies, conflicts, etc.)
                        if any(word in event.lower() for word in ['artifact', 'magic', 'prophecy', 'relationship', 'conflict', 'discovery', 'secret', 'promise', 'warning', 'fear', 'rune', 'champion']):
                            planted_seeds.append(event[:120])  # Store as seed
                
            except Exception as e:
                print(f"  [ERROR] Failed: {e}")
                raise
        
        print(f"\n[SCRIPT] Total scenes: {len(all_scenes)}")
        
        # Validate and fix scene IDs
        for i, scene in enumerate(all_scenes):
            scene["id"] = i + 1
            for field in ["title", "narration", "image_prompt", "emotion", "year"]:
                if field not in scene:
                    raise ValueError(f"Scene {i+1} missing required field: {field}")
        
        # Step 3.4: Refine scenes for awkward transitions and improvements
        print("\n[STEP 3.4] Refining main video scenes...")
        chapter_summaries = "\n".join([
            f"Chapter {ch['chapter_num']}: {ch['title']} ({ch.get('year_range') or ch.get('timeline', '?')}) - {ch['summary']}"
            for ch in chapters
        ])
        diff_path = Path(output_path).parent / f"{Path(output_path).stem}_refinement_diff.json" if config.generate_refinement_diffs else None
        all_scenes, refinement_diff = refine_scenes_lol(all_scenes, subject_of_interest, is_short=False, chapter_context=chapter_summaries, diff_output_path=diff_path)
        
        # Step 3.5: Generate final metadata (description and tags) AFTER scenes are generated
        print("\n[STEP 3.5] Generating final metadata from actual scenes...")
        
        # Extract memorable moments from scenes for metadata
        scene_highlights = []
        for scene in all_scenes[:10]:  # First 10 scenes for highlights
            scene_highlights.append(f"Scene {scene['id']}: {scene.get('title', '')} - {scene.get('narration', '')[:80]}...")
        
        final_metadata_prompt = f"""Create final metadata for a League of Legends lore documentary about: {subject_of_interest}

The story in one line: {outline.get('tagline', '')}

Actual memorable moments from the documentary (use these to make description more accurate):
{chr(10).join(scene_highlights[:10])}

Generate JSON:
{{
  "video_description": "YouTube description (500-800 words) - hook, highlights, key facts, call to action. SEO optimized for League of Legends lore. Reference specific compelling moments from the actual scenes above to make it accurate and engaging. Include relevant LoL lore terminology.",
  "tags": "15-20 SEO tags separated by commas. Mix of: {subject_of_interest}, League of Legends, LoL lore, Runeterra, {subject_type}, key topics, related champions/regions, time periods, achievements, fields (e.g., '{subject_of_interest}, League of Legends, LoL lore, Runeterra, champions, magic, battles, documentary, League of Legends lore')",
  "pinned_comment": "An engaging question or comment to pin below the video (1-2 sentences max). Should: spark discussion, ask a thought-provoking question about the lore/subject, create curiosity, encourage viewers to share their thoughts/opinions, be conversational and inviting. Examples: 'Which moment from this lore story surprised you most? Drop a comment below!', 'What do you think was the most significant event in this story? Share your thoughts!', 'This lore shows how epic the world of Runeterra is. What other lore stories would you like to see?'. Should feel authentic, not clickbaity - genuine curiosity about viewer perspectives."
}}"""

        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": "League of Legends lore documentary producer. Create compelling metadata that accurately reflects the actual content. Respond with valid JSON only."},
                {"role": "user", "content": final_metadata_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        final_metadata = json.loads(clean_json_response(response.choices[0].message.content))
        video_description = final_metadata.get("video_description", "")
        tags = final_metadata.get("tags", "")
        pinned_comment = final_metadata.get("pinned_comment", "")
        
        print(f"[METADATA] Description generated from {len(all_scenes)} scenes")
        print(f"[METADATA] Tags: {tags[:80]}..." if len(tags) > 80 else f"[METADATA] Tags: {tags}")
        if pinned_comment:
            print(f"[METADATA] Pinned comment: {pinned_comment}")
    else:
        print("\n[STEP 3] Skipping main video scene generation...")
        # Generate basic metadata if not generating main video
        video_description = ""
        tags = ""
        pinned_comment = ""
    
    # Step 4: Generate Shorts (pass scene highlights if available)
    print("\n[STEP 4] Generating YouTube Shorts...")
    scene_highlights_for_shorts = []
    if config.generate_main and all_scenes:
        # Extract key scene moments for shorts to reference
        for scene in all_scenes:
            scene_highlights_for_shorts.append({
                "id": scene.get("id"),
                "title": scene.get("title"),
                "narration_preview": scene.get("narration", "")[:100]
            })
    shorts_info = generate_lol_shorts(subject_of_interest, title, global_block, outline, output_path, scene_highlights=scene_highlights_for_shorts)
    
    # Step 5: Save
    print("\n[STEP 5] Saving script...")
    
    output_data = {
        "metadata": {
            "title": title,
            "tag_line": tag_line,
            "video_description": video_description,
            "tags": tags,
            "pinned_comment": pinned_comment if config.generate_main else "",
            "thumbnail_description": thumbnail_description,
            "thumbnail_path": str(generated_thumb) if generated_thumb else None,
            "global_block": global_block,
            "subject_of_interest": subject_of_interest,
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


def generate_lol_top10_script(topic: str, output_path: str):
    """Generate a top N list-style LoL lore video script."""
    
    num_entries = config.top_n_entries
    
    # Step 1: Generate top N outline
    print(f"\n[STEP 1] Creating top {num_entries} outline...")
    outline = generate_top10_outline(topic, num_entries=num_entries)
    entries = outline.get("entries", [])
    
    if len(entries) != num_entries:
        print(f"[WARNING] Got {len(entries)} entries, expected {num_entries}")
    
    # Step 2: Generate initial metadata
    print("\n[STEP 2] Generating initial metadata...")
    
    entries_preview = ", ".join([e.get("name", "?") for e in entries[:5]])
    
    num_entries = config.top_n_entries
    
    initial_metadata_prompt = f"""Create initial metadata for a top {num_entries} list video about: {topic}

Entries in this list (for context): {entries_preview}...

Generate JSON:
{{
  "title": "CLICKBAIT YouTube title (60-80 chars) for a top {num_entries} list about: {topic}. Use power words like: TOP {num_entries}, RANKED, SHOCKING, REVEALED, UNBELIEVABLE. Adapt examples to the topic - if it's about events use 'MOST IMPORTANT EVENTS', if it's about power use 'MOST POWERFUL', if it's about champions use 'BEST CHAMPIONS', etc. Must be engaging and make viewers NEED to see the full ranking.",
  "tag_line": "Short tagline for the top {num_entries} list (5-10 words). Adapt to the topic - examples: 'the definitive ranking', 'ranked by importance', 'the most impactful events', etc.",
  "thumbnail_description": "CLICKBAIT thumbnail visual for a top {num_entries} list about: {topic}. Maximize visual impact! Use dramatic composition showing relevant elements (characters, events, locations, etc. depending on topic), cosmic/magical scale, extreme lighting (chiaroscuro), bold colors (reds/yellows for urgency), symbolic elements representing the ranking. Show epic scale. Think epic fantasy movie poster. Include fantasy/magical elements appropriate for League of Legends lore. NO TEXT in image, but visually SCREAM importance and epic scale.",
  "global_block": "Visual style guide (300-400 words): semi-realistic digital painting style with fantasy elements, color palette, dramatic lighting, how beings should appear consistently across all scenes. Emphasize magical/mystical elements, cosmic scale, power hierarchies, and League of Legends lore aesthetics."
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "League of Legends lore documentary producer. Create compelling top 10 list metadata. Respond with valid JSON only."},
            {"role": "user", "content": initial_metadata_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    initial_metadata = json.loads(clean_json_response(response.choices[0].message.content))
    
    title = initial_metadata["title"]
    tag_line = initial_metadata.get("tag_line", "the definitive power ranking")
    thumbnail_description = initial_metadata["thumbnail_description"]
    global_block = initial_metadata["global_block"]
    
    print(f"[METADATA] Title: {title}")
    print(f"[METADATA] Tag line: {tag_line}")
    
    # Generate main video thumbnail
    generated_thumb = None
    if config.generate_main:
        print("\n[THUMBNAIL] Main video thumbnail...")
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        thumbnail_path = THUMBNAILS_DIR / f"{Path(output_path).stem}_thumbnail.png"
        
        thumbnail_prompt = f"""{thumbnail_description}

YouTube CLICKBAIT thumbnail - MUST MAXIMIZE CLICKS:
- Dramatic composition showing epic scale and relevant elements (characters, events, locations, etc. depending on topic)
- Extreme lighting with HIGH CONTRAST (chiaroscuro)
- Bold, eye-catching colors (reds, yellows, oranges)
- Fantasy/magical elements appropriate for League of Legends lore
- Cinematic, movie-poster quality
- Optimized for small sizes
- Overall feeling: EPIC, POWERFUL, UNMISSABLE"""
        
        generated_thumb = generate_thumbnail(thumbnail_prompt, thumbnail_path)
    
    # Step 3: Generate scenes
    all_scenes = []
    
    if config.generate_main:
        num_entries = config.top_n_entries
        print(f"\n[STEP 3] Generating top {num_entries} video scenes...")
        
        # Generate introduction scenes
        print(f"\n[INTRO] Generating {config.top_n_intro_scenes} introduction scenes...")
        intro_scenes = generate_top10_intro(topic, entries)
        all_scenes.extend(intro_scenes)
        print(f"  ✓ {len(intro_scenes)} intro scenes (total: {len(all_scenes)})")
        
        # Generate entry scenes in countdown order (10 to 1)
        # Entries are already sorted by rank (descending: 10, 9, ..., 1)
        for i, entry in enumerate(entries):
            rank = entry.get("rank", 0)
            name = entry.get("name", "Unknown")
            entry_type = entry.get("type", "character")
            prev_entry = entries[i - 1] if i > 0 else None
            
            print(f"\n[ENTRY #{rank}/{len(entries)}] {name}")
            print(f"  Generating entry scenes...")
            
            try:
                entry_scenes = generate_top10_entry_scenes(
                    entry=entry,
                    entry_num=rank,
                    total_entries=len(entries),
                    global_style=global_block,
                    prev_entry=prev_entry,
                    subject_type=entry_type
                )
                
                all_scenes.extend(entry_scenes)
                print(f"  ✓ {len(entry_scenes)} scenes (total: {len(all_scenes)})")
                
            except Exception as e:
                print(f"  [ERROR] Failed: {e}")
                raise
        
        print(f"\n[SCRIPT] Total scenes: {len(all_scenes)}")
        
        # Validate and fix scene IDs
        for i, scene in enumerate(all_scenes):
            scene["id"] = i + 1
            for field in ["title", "narration", "image_prompt", "emotion", "year"]:
                if field not in scene:
                    raise ValueError(f"Scene {i+1} missing required field: {field}")
        
        # Insert CTA scene after intro and before first entry
        print(f"\n[CTA] Generating CTA scene after intro...")
        try:
            cta_scene = generate_cta_top_n(topic)
            all_scenes.insert(len(intro_scenes), cta_scene)  # Insert after intro scenes
            print(f"  ✓ CTA scene added (total: {len(all_scenes)})")
        except Exception as e:
            print(f"  [WARNING] Failed to generate CTA scene: {e}")
            # Continue without CTA scene if generation fails
        
        # Refine scenes (skip significance scenes for top 10 - not applicable)
        print("\n[STEP 3.4] Refining main video scenes...")
        entries_summary = "\n".join([
            f"#{e.get('rank')}: {e.get('name')} - {e.get('description', e.get('power_description', ''))[:80]}..."
            for e in entries
        ])
        diff_path = Path(output_path).parent / f"{Path(output_path).stem}_refinement_diff.json" if config.generate_refinement_diffs else None
        # Skip significance scenes for top 10 videos - they don't make sense in this context
        all_scenes, refinement_diff = refine_scenes_lol(all_scenes, topic, is_short=False, chapter_context=entries_summary, diff_output_path=diff_path, skip_significance_scenes=True)
        
        # Generate final metadata
        print("\n[STEP 3.5] Generating final metadata from actual scenes...")
        
        scene_highlights = []
        for scene in all_scenes[:10]:
            scene_highlights.append(f"Scene {scene['id']}: {scene.get('title', '')} - {scene.get('narration', '')[:80]}...")
        
        num_entries = config.top_n_entries
        final_metadata_prompt = f"""Create final metadata for a top {num_entries} list video about: {topic}

Actual memorable moments from the video (use these to make description more accurate):
{chr(10).join(scene_highlights[:10])}

Generate JSON:
{{
  "video_description": "YouTube description (500-800 words) - hook, overview of the top {num_entries} list about: {topic}, mention it's a countdown format ({num_entries} to 1), SEO optimized for League of Legends lore. Reference specific compelling moments from the actual scenes above to make it accurate and engaging. Mention that it's a ranked countdown building to the top/highest entry.",
  "tags": "15-20 SEO tags separated by commas. Mix of: top {num_entries}, {topic}, League of Legends, LoL lore, Runeterra, rankings, lore (e.g., 'top {num_entries}, {topic}, League of Legends, LoL lore, Runeterra, rankings, lore documentary'). Adapt tags to match the topic.",
  "pinned_comment": "An engaging question or comment to pin below the video (1-2 sentences max). Should: spark discussion, ask viewers about the rankings, create curiosity, encourage viewers to share their thoughts/opinions. Examples: 'Do you agree with this ranking? Who would you put at #1? Drop a comment below!', 'Which entry surprised you most? Share your thoughts!'. Should feel authentic, not clickbaity - genuine curiosity about viewer perspectives."
}}"""

        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": "League of Legends lore documentary producer. Create compelling metadata that accurately reflects the actual content. Respond with valid JSON only."},
                {"role": "user", "content": final_metadata_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        final_metadata = json.loads(clean_json_response(response.choices[0].message.content))
        video_description = final_metadata.get("video_description", "")
        tags = final_metadata.get("tags", "")
        pinned_comment = final_metadata.get("pinned_comment", "")
        
        print(f"[METADATA] Description generated from {len(all_scenes)} scenes")
        print(f"[METADATA] Tags: {tags[:80]}..." if len(tags) > 80 else f"[METADATA] Tags: {tags}")
        if pinned_comment:
            print(f"[METADATA] Pinned comment: {pinned_comment}")
    else:
        print("\n[STEP 3] Skipping main video scene generation...")
        video_description = ""
        tags = ""
        pinned_comment = ""
    
    # Step 4: Generate Shorts (configurable number from random spots, excluding #1, #2, #3)
    num_shorts = config.num_shorts
    print(f"\n[STEP 4] Generating YouTube Shorts ({num_shorts} random spots, excluding #1, #2, #3)...")
    scene_highlights_for_shorts = []
    if config.generate_main and all_scenes:
        for scene in all_scenes:
            scene_highlights_for_shorts.append({
                "id": scene.get("id"),
                "title": scene.get("title"),
                "narration_preview": scene.get("narration", "")[:100]
            })
    
    # Generate shorts for top 10 - 3 random spots (excluding #1, #2, #3)
    shorts_info = []
    if config.num_shorts > 0 and entries:
        # Generate shorts (will randomly select from ranks 4-10)
        shorts_info = generate_top10_shorts(topic, title, global_block, outline, output_path, entries)
    
    # Step 5: Save
    print("\n[STEP 5] Saving script...")
    
    output_data = {
        "metadata": {
            "title": title,
            "tag_line": tag_line,
            "video_description": video_description,
            "tags": tags,
            "pinned_comment": pinned_comment if config.generate_main else "",
            "thumbnail_description": thumbnail_description,
            "thumbnail_path": str(generated_thumb) if generated_thumb else None,
            "global_block": global_block,
            "subject_of_interest": topic,
            "num_scenes": len(all_scenes),
            "outline": outline,
            "shorts": shorts_info,
            "top_n_mode": True,
            "top_n_entries": config.top_n_entries
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


def generate_top10_shorts(topic: str, main_title: str, global_block: str, outline: dict, base_output_path: str, entries: list[dict]):
    """Generate YouTube Shorts for top N entries (configurable number from random spots, excluding #1, #2, #3)."""
    num_shorts = config.num_shorts
    if num_shorts == 0:
        print("\n[SHORTS] Skipped (--shorts 0)")
        return []
    
    # Filter entries to exclude ranks 1, 2, 3 (only consider ranks 4 and above)
    eligible_entries = [e for e in entries if e.get("rank", 0) >= 4]
    
    if len(eligible_entries) == 0:
        print(f"\n[SHORTS] No eligible entries for shorts (need ranks 4-{len(entries)})")
        return []
    
    # Select random entries (up to num_shorts, but not more than available)
    import random
    num_shorts_to_generate = min(num_shorts, len(eligible_entries))
    selected_entries = random.sample(eligible_entries, num_shorts_to_generate)
    
    # Sort selected entries by rank for consistent ordering
    selected_entries.sort(key=lambda x: x.get("rank", 0))
    
    print(f"\n{'='*60}")
    print(f"[SHORTS] Generating {num_shorts_to_generate} YouTube Short(s) from random spots (excluding #1, #2, #3)")
    print(f"[SHORTS] Selected ranks: {[e.get('rank') for e in selected_entries]}")
    print(f"{'='*60}")
    
    SHORTS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    
    generated_shorts = []
    base_name = Path(base_output_path).stem.replace("_script", "")
    
    # Generate shorts for selected entries
    for entry in selected_entries:
        rank = entry.get("rank", 0)
        name = entry.get("name", "Unknown")
        entry_type = entry.get("type", "character")
        
        print(f"\n[SHORT #{rank}] Creating short for: {name}")
        
        # Create a simplified outline for this entry (for short generation)
        total_entries = len(entries)  # Get total from entries list
        entry_outline = {
            "short_title": f"#{rank}: {name}",
            "short_description": f"Exploring why {name} ranks at #{rank} in our top {total_entries} list.",
            "tags": f"{name}, top {total_entries}, League of Legends, LoL lore, Runeterra, rankings, #{rank}",
            "thumbnail_prompt": f"CLICKBAIT thumbnail for mobile (9:16 vertical): EXTREME close-up or dramatic composition, {name} in MOMENT OF IMPACT, intense emotional expression, HIGH CONTRAST dramatic lighting (chiaroscuro), bold eye-catching colors (reds/yellows for urgency), fantasy/magical elements, movie-poster energy, optimized for small mobile screens",
            "hook_fact": f"{name} ranks at #{rank} on the list",
            "story_angle": f"Why {name} ranks at #{rank} - exploring their significance and facts",
            "key_facts": entry.get("key_feats", [])
        }
        
        try:
            print(f"[SHORT #{rank}] Generating {config.total_short_scenes} scenes...")
            all_scenes = generate_lol_short_scenes(
                subject=name,
                short_outline=entry_outline,
                subject_type=entry_type,
                timeline_start=None,
                timeline_end=None
            )
            print(f"[SHORT #{rank}] → {len(all_scenes)} scenes generated")
            
            # Fix scene IDs
            for i, scene in enumerate(all_scenes):
                scene["id"] = i + 1
            
            # Refine short scenes
            short_context = f"Top 10 Entry #{rank}: {name}. Story: {entry_outline.get('story_angle', '')}"
            short_file = SHORTS_DIR / f"{base_name}_entry{rank}_short.json"
            diff_path = short_file.parent / f"{short_file.stem}_refinement_diff.json" if config.generate_refinement_diffs else None
            all_scenes, refinement_diff = refine_scenes_lol(all_scenes, name, is_short=True, chapter_context=short_context, diff_output_path=diff_path)
            
            # Generate thumbnail (if enabled)
            thumbnail_path = None
            if config.generate_short_thumbnails:
                thumbnail_prompt = entry_outline.get("thumbnail_prompt", "")
                if thumbnail_prompt:
                    thumbnail_prompt += "\n\nCLICKBAIT YouTube Shorts thumbnail - MAXIMIZE SCROLL-STOPPING POWER: Vertical 9:16, EXTREME close-up or dramatic composition, intense emotional expression, HIGH CONTRAST dramatic lighting, bold eye-catching colors, fantasy/magical elements, movie-poster energy, optimized for mobile scrolling."
                    thumb_file = THUMBNAILS_DIR / f"{base_name}_entry{rank}_short_thumbnail.png"
                    thumbnail_path = generate_thumbnail(thumbnail_prompt, thumb_file, size="1024x1536")
            
            # Build short output
            short_title = entry_outline.get("short_title", f"Entry #{rank}: {name}")
            short_output = {
                "metadata": {
                    "short_id": rank,
                    "title": short_title,
                    "description": entry_outline.get("short_description", ""),
                    "tags": entry_outline.get("tags", ""),
                    "hook_fact": entry_outline.get("hook_fact", ""),
                    "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
                    "story_angle": entry_outline.get("story_angle", ""),
                    "subject_of_interest": name,
                    "main_video_title": main_title,
                    "global_block": global_block,
                    "num_scenes": len(all_scenes),
                    "entry_rank": rank,
                    "outline": entry_outline
                },
                "scenes": all_scenes
            }
            
            # Save short
            with open(short_file, "w", encoding="utf-8") as f:
                json.dump(short_output, f, indent=2, ensure_ascii=False)
            
            print(f"[SHORT #{rank}] ✓ Saved: {short_file} ({len(all_scenes)} scenes)")
            
            generated_shorts.append({
                "file": str(short_file),
                "title": short_title,
                "scenes": len(all_scenes)
            })
            
        except Exception as e:
            print(f"[SHORT #{rank}] ERROR: {e}")
            raise
    
    # Sort by rank
    generated_shorts.sort(key=lambda x: int(Path(x['file']).stem.split('_entry')[1].split('_')[0]))
    
    return generated_shorts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate League of Legends lore documentary scripts with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full production run (28 scenes main video, 3 shorts with 5 scenes each)
  python build_script_lol.py "Jinx"

  # Quick test (4 scenes main, 1 short with 5 scenes, no thumbnails)
  python build_script_lol.py "Jinx" --test

  # Custom main video settings
  python build_script_lol.py "Demacia" --chapters 5 --scenes 3

  # Custom shorts settings  
  python build_script_lol.py "Yasuo" --shorts 2 --short-scenes 5

  # No thumbnails (faster iteration)
  python build_script_lol.py "Noxus" --no-thumbnails

  # Only generate shorts (skip main video)
  python build_script_lol.py "Yordles" --shorts-only

  # Only generate main video (skip shorts)
  python build_script_lol.py "Runeterra" --main-only

  # Top N list format (countdown style)
  python build_script_lol.py "The 10 Most Powerful Beings" --top-n 10
  python build_script_lol.py "The 5 Most Powerful Beings" --top-n 5 --shorts 2
        """
    )
    
    parser.add_argument("subject", help="LoL lore subject to create documentary about (champion, region, or group)")
    parser.add_argument("output", nargs="?", help="Output JSON file (default: <subject>_script.json)")
    
    # Quick test mode
    parser.add_argument("--test", action="store_true", 
                        help="Quick test: 2 chapters × 2 scenes, 1 short with 5 scenes, no thumbnails")
    
    # What to generate
    parser.add_argument("--top-n", type=int, metavar="N", default=None,
                        help="Generate a top N list-style video (countdown format: N to 1, building to most powerful). Example: --top-n 10")
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
                        help=f"Scenes per short (default: {Config.short_scenes_per_chapter})")
    
    parser.add_argument("--no-thumbnails", action="store_true",
                        help="Skip main video thumbnail generation")
    parser.add_argument("--short-thumbnails", action="store_true",
                        help="Generate thumbnails for shorts (disabled by default)")
    parser.add_argument("--refinement-diffs", action="store_true",
                        help="Generate refinement diff JSON files showing what changed during scene refinement")
    
    return parser.parse_args()


# ------------- ENTRY POINT -------------

if __name__ == "__main__":
    args = parse_args()
    
    # Apply test mode settings first (these take precedence)
    if args.test:
        config.chapters = 2
        config.scenes_per_chapter = 2
        config.num_shorts = 1
        config.short_chapters = 1
        config.short_scenes_per_chapter = 5
        config.generate_thumbnails = False
        config.generate_short_thumbnails = False
        config.generate_refinement_diffs = False
        print("[MODE] Test mode enabled")
    else:
        config.chapters = args.chapters
        config.scenes_per_chapter = args.scenes
        config.num_shorts = args.shorts
        config.short_scenes_per_chapter = args.short_scenes
        config.generate_thumbnails = not args.no_thumbnails
        config.generate_short_thumbnails = args.short_thumbnails
        config.generate_refinement_diffs = args.refinement_diffs
    
    # Handle top N mode (after applying args, so we can override num_shorts if needed)
    if args.top_n is not None:
        if args.top_n < 3:
            print("ERROR: --top-n must be at least 3 (need at least 3 entries for shorts exclusion)")
            sys.exit(1)
        config.top_n_mode = True
        config.top_n_entries = args.top_n
        # num_shorts is already set above from args.shorts (defaults to 3)
        # If user explicitly set --shorts to something else, that value is already applied
        print(f"[MODE] Top {config.top_n_entries} list mode enabled (countdown format: {config.top_n_entries} to 1)")
    
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
    
    # Determine output file (goes to scripts/ directory)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_file = args.output
        # If output doesn't include scripts/ directory, add it
        if not output_file.startswith("scripts/") and not Path(output_file).is_absolute():
            output_file = str(SCRIPTS_DIR / output_file)
    else:
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in args.subject)
        safe_name = safe_name.replace(' ', '_').lower()
        output_file = str(SCRIPTS_DIR / f"{safe_name}_script.json")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)
    
    try:
        script_data = generate_lol_script(args.subject, output_file)
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
