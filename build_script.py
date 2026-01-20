import os
import sys
import json
import base64
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
build_scripts_utils.SCRIPT_MODEL = "gpt-5.2"  # Using gpt-5.2 for better quality and longer context
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
    chapters = 7           # Number of outline chapters
    scenes_per_chapter = 4  # Scenes per chapter (total = chapters * scenes_per_chapter)
    generate_main = True    # Whether to generate main video
    
    # Shorts settings
    num_shorts = 3              # Number of YouTube Shorts
    short_chapters = 1          # Chapters per short (1 chapter: 5 scenes telling one complete story)
    short_scenes_per_chapter = 5  # Scenes per chapter in shorts (complete story with natural conclusion)
    
    generate_thumbnails = True  # Whether to generate thumbnail images (main video)
    generate_short_thumbnails = False  # Whether to generate thumbnails for shorts (usually not needed)
    generate_refinement_diffs = False  # Whether to generate refinement diff JSON files
    
    @property
    def total_scenes(self):
        return self.chapters * self.scenes_per_chapter
    
    @property
    def total_short_scenes(self):
        return self.short_chapters * self.short_scenes_per_chapter

config = Config()

# Import shared functions from utils
clean_json_response = build_scripts_utils.clean_json_response
get_shared_scene_requirements = lambda: build_scripts_utils.get_shared_scene_requirements("historical")
get_shared_narration_style = build_scripts_utils.get_shared_narration_style
get_shared_scene_flow_instructions = build_scripts_utils.get_shared_scene_flow_instructions
get_shared_examples = lambda: build_scripts_utils.get_shared_examples("historical")
generate_refinement_diff = build_scripts_utils.generate_refinement_diff
identify_pivotal_moments = build_scripts_utils.identify_pivotal_moments
generate_significance_scene = build_scripts_utils.generate_significance_scene

# Wrapper for refine_scenes to maintain backward compatibility (person parameter)
def refine_scenes(scenes: list[dict], person: str, is_short: bool = False, chapter_context: str = None, diff_output_path: Path | None = None) -> tuple[list[dict], dict]:
    """Wrapper for refine_scenes from utils, maintaining backward compatibility with 'person' parameter."""
    return build_scripts_utils.refine_scenes(scenes, person, is_short, chapter_context, diff_output_path, "person")


# Shared prompt functions are now imported from build_scripts_utils
# Historical-specific prompt wrappers use "historical" content_type


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
    import re
    problematic_patterns = [
        (r'\b(harm|hurt|pain|suffering|death|suicide|cut|bleed|violence)\b', 'challenge'),
        (r'\b(depression|despair|hopeless)\b', 'contemplation'),
    ]
    
    for pattern, replacement in problematic_patterns:
        prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
    
    # Append safety instruction
    sanitized = prompt + safety_instruction
    
    return sanitized


class SafetyViolationError(Exception):
    """Custom exception for safety violations with metadata."""
    def __init__(self, message, violation_type=None, original_prompt=None):
        super().__init__(message)
        self.violation_type = violation_type
        self.original_prompt = original_prompt


# generate_thumbnail is imported from build_scripts_utils
# Create a wrapper that uses config.generate_thumbnails
def generate_thumbnail(prompt: str, output_path: Path, size: str = "1024x1024") -> Path | None:
    """Generate a thumbnail image and save it with safety violation handling."""
    return build_scripts_utils.generate_thumbnail(prompt, output_path, size, config.generate_thumbnails)


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
1. Chapter 1 - THE HOOK: Start by introducing the person with context: "This is the story of [Name]..." or "Meet [Name]..." Then present a rapid-fire "trailer" of their MOST interesting facts, achievements, controversies, and dramatic moments that will be covered. Give viewers context for who we're talking about before diving into the highlights. Make viewers think "I NEED to know more." This is NOT chronological - it's a highlight reel that hooks the audience. End with something like "But how did they get here? Let's start from the beginning..."
2. Chapters 2: Early life and rising action - origins, struggles, first successes, growing stakes
3. Chapters 3-5: Peak conflict - major breakthroughs AND major crises
4. Chapters 6: Resolution and consequences
5. Chapter 7: Legacy and emotional conclusion that echoes the hook

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
  "overarching_plots": [
    {{
      "plot_name": "The main plot thread (e.g., 'The Quest for Recognition', 'The Rivalry with X', 'The Secret That Changed Everything')",
      "description": "What this plot is about and why it matters",
      "starts_chapter": 1-{config.chapters},
      "peaks_chapter": 1-{config.chapters},
      "resolves_chapter": 1-{config.chapters},
      "key_moments": ["specific plot points that develop this story"]
    }}
  ],
  "sub_plots": [
    {{
      "subplot_name": "A sub-plot that spans 2-4 chapters (e.g., 'The Relationship with Y', 'The Internal Conflict', 'The Personal Cost')",
      "description": "What this sub-plot is about",
      "chapters_span": [1-3],  // which chapters this sub-plot appears in
      "key_moments": ["specific moments that advance this sub-plot"]
    }}
  ],
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
      "recurring_threads": ["...", ...],
      "plots_active": ["plot names or subplot names that are active/developing in this chapter"],
      "plot_developments": ["How overarching plots and sub-plots develop in this chapter - what happens to them"]
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


def generate_cta_transition_scene(person: str, tag_line: str | None = None) -> dict:
    """
    Generate a CTA transition scene between chapter 1 (hook) and chapter 2 (story begins).
    This scene transitions from the hook/intro to the chronological story and includes a call-to-action.
    Only used for main videos, not shorts.
    """
    tag_line_text = f" - {tag_line}" if tag_line else ""
    
    cta_prompt = f"""Generate a single transition scene for a documentary about {person}.

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
     * "Let's start from the very beginning - and don't forget to like, subscribe, and leave a comment if you want to see more documentaries like this."
   - Keep it short, natural, and conversational - one sentence total (~8-12 seconds when spoken)

2. NARRATION STYLE:
   - Write from the YouTuber's perspective (YOUR script, YOU telling the story)
   - Simple, clear language
   - Friendly and engaging tone
   - MUST be exactly ONE sentence that combines transition + CTA

4. IMAGE PROMPT:
   - Bright, happy, upbeat mood - this should feel positive and engaging
   - Related to {person} and the documentary topic but in a cheerful context
   - Include visual elements that suggest engagement: perhaps a celebratory moment, an achievement, a positive milestone from their life
   - 16:9 cinematic format
   - The image should be inviting and make viewers want to continue watching

5. EMOTION: "upbeat" or "engaging"

Respond with JSON:
{{
  "id": 0,
  "title": "3-5 word transition title",
  "narration": "ONE sentence that transitions from the hook to the story beginning and includes CTA to like, subscribe, and comment. ~8-12 seconds when spoken.",
  "image_prompt": "Bright, happy, upbeat scene related to {person} - an inspiring moment, achievement, or positive milestone from their life that's visually engaging and makes viewers want to continue. 16:9 cinematic",
  "emotion": "upbeat",
  "year": "transition" or a relevant year from early in their story
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating engaging content. Write naturally from YOUR perspective. Make the transition smooth and the CTA feel authentic, not forced. Respond with valid JSON only."},
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
        scene['narration'] = "But how did it all begin? Before we dive into the story, make sure to like, subscribe, and comment below if you're enjoying this documentary."
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
        scene['image_prompt'] = f"Bright, happy, upbeat scene related to {person} - an inspiring moment or positive milestone. 16:9 cinematic"
    if 'emotion' not in scene:
        scene['emotion'] = "upbeat"
    if 'year' not in scene:
        scene['year'] = "transition"
    
    return scene


def generate_scenes_for_chapter(person: str, chapter: dict, scenes_per_chapter: int, start_id: int, 
                                 global_style: str, prev_chapter: dict = None, prev_scenes: list = None,
                                 central_theme: str = None, narrative_arc: str = None, 
                                 planted_seeds: list[str] = None, is_retention_hook_point: bool = False,
                                 birth_year: int | None = None, death_year: int | None = None,
                                 tag_line: str | None = None, overarching_plots: list[dict] = None,
                                 sub_plots: list[dict] = None) -> list[dict]:
    """Generate scenes for a single chapter of the outline with continuity context."""
    
    if planted_seeds is None:
        planted_seeds = []
    
    # Build context from previous chapter for smooth transitions
    chapter_num = chapter.get('chapter_num', 1)
    is_hook_chapter = (chapter_num == 1)
    
    # Determine pacing based on emotional tone and dramatic tension
    emotional_tone = chapter.get('emotional_tone', '').lower()
    dramatic_tension = chapter.get('dramatic_tension', '').lower()
    
    # Fast pacing for action/excitement, slower for emotional weight
    if any(word in emotional_tone for word in ['propulsive', 'energetic', 'intense', 'thrilling', 'action']):
        pacing_instruction = "FAST PACING: Scenes should be punchy and energetic. Use 1-2 sentences per scene (~8-12 seconds). Quick cuts, high energy."
    elif any(word in emotional_tone for word in ['contemplative', 'somber', 'reflective', 'melancholic', 'weighty']):
        pacing_instruction = "SLOWER PACING: Scenes should have emotional weight. Use 3-4 sentences per scene (~18-24 seconds). Allow moments to breathe."
    else:
        pacing_instruction = "MODERATE PACING: Use 2-3 sentences per scene (~12-18 seconds). Match pacing to the emotional beats of the story."
    
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
    
    # Include last few scenes for continuity and to avoid overlapping events
    if prev_scenes and len(prev_scenes) > 0:
        # Get last 5 scenes for context (increased to better track what was covered)
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
    
    scene_prompt = f"""You are writing scenes {start_id}-{start_id + scenes_per_chapter - 1} of a {config.total_scenes}-scene documentary about {person}.

{prev_context}
{scenes_context}
NOW WRITING CHAPTER {chapter['chapter_num']} of {config.chapters}: "{chapter['title']}"
Time Period: {chapter['year_range']}
Emotional Tone: {chapter['emotional_tone']}
Dramatic Tension: {chapter['dramatic_tension']}
Recurring Themes: {threads_str}
Sets Up What Comes Next: {connects_to_next}

EMOTION GENERATION (CRITICAL):
- Each scene MUST include an "emotion" field - a single word or short phrase (e.g., "tense", "triumphant", "desperate", "contemplative", "exhilarating", "somber", "urgent", "defiant")
- Base the emotion on: what the character is feeling at this moment, the dramatic tension, and the significance of the event
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
- SCENE 1 (COLD OPEN - FIRST 15 SECONDS): The first sentence MUST begin with: "Welcome to Human Footprints. Today we'll talk about {person} - {tag_line if tag_line else ''}." After this introduction, immediately transition to the MOST shocking, intriguing, or compelling moment or question. The tag_line should be short, catchy, and accurate (e.g., "the man who changed the world", "the codebreaker who saved millions", "the mind that rewrote physics"). For example: "Welcome to Human Footprints. Today we'll talk about Albert Einstein - the man who changed the world. In 1905, a 26-year-old patent clerk publishes a paper that will change everything. But first, he must survive the criticism of his own father." OR "Welcome to Human Footprints. Today we'll talk about Charles Darwin - the man who changed the way we understand life. The letter arrives in June 1858. Darwin's hands tremble as he reads his own theory in another man's words." The first 15 seconds are CRITICAL for YouTube - this is what viewers see in search/preview. Hook them immediately after the introduction.
- SCENES 2-4: Rapid-fire preview of the most shocking, interesting, or impactful moments from their entire life - achievements, controversies, dramatic moments, surprising facts
- NOT chronological - pick jaw-dropping highlights from any point in their life
- Each scene should hook the viewer: "You'll discover how...", "You'll see the moment when...", "Wait until you hear about..."
- FINAL SCENE: Natural bridge to the chronological story. DO NOT say "now the story rewinds" or "let's go back" or "rewind to the beginning" - that's awkward. Instead, use something like: "But it all started when...", "It begins with...", or simply start the chronological narrative: "[Birth year/early life context]. This is where our story begins."
- Tone: Exciting, intriguing preview. Fast-paced like a trailer. NOT a narrative story.

NARRATION STYLE FOR INTRO:
- Speak directly to the viewer about what they'll discover
- Use phrases like "You'll discover...", "This is the story of...", "Wait until you learn...", "Here's why this matters..."
- Present facts as previews, not as events happening in real-time
- Make it clear this is a preview/introduction, not the actual story''' if is_hook_chapter else '''CRITICAL STORYTELLING REQUIREMENTS:
- This MUST feel like a STORY being told, not a chronological list of events
- Each scene should ADVANCE a plot thread - show how overarching plots and sub-plots are developing
- Connect scenes through PLOT PROGRESSION, not just "this happened, then that happened"
- Show CAUSE AND EFFECT - why this event matters, what it leads to, how it changes things
- Build NARRATIVE TENSION through plot development - show conflicts developing, stakes rising, problems emerging or resolving
- Reference plot threads from earlier in the story naturally - make viewers feel like they're following a continuous story
- Each scene should answer "what's happening with the story?" not just "what happened next?"
- Think like a novelist: every scene should advance character development, plot, or theme
- EMOTIONAL ENGAGEMENT (CRITICAL): Make viewers feel emotionally invested by showing:
  * How events FEEL to the character - their internal experience, fears, hopes, reactions
  * The EMOTIONAL SIGNIFICANCE - why this moment matters personally, not just historically
  * What's at stake EMOTIONALLY - what failure would feel like, what success means
  * Human details that create empathy - physical sensations, internal thoughts, personal costs
  * Make events feel REAL and SIGNIFICANT by connecting them to human feelings and experiences
- PIVOTAL MOMENTS - SIGNIFICANCE (CRITICAL): For the most PIVOTAL moments in this chapter (major breakthroughs, turning points, critical decisions, moments that change the story trajectory), you MUST explicitly explain WHY this moment matters - its significance to the overall story, its impact on the broader narrative, and what it changes. Make viewers FEEL the weight of these moments. Don't just describe what happened - explain WHY it's pivotal and HOW it reshapes what comes after.
- Make viewers feel emotionally invested in HOW the story unfolds, not just WHAT happened - pull them into the character's emotional reality
- CRITICAL: Each scene must cover DIFFERENT, NON-OVERLAPPING events. Do NOT repeat events already covered in previous scenes. If Scene X describes Event A in detail, Scene Y should NOT re-describe Event A - instead, move to Event A's consequences, what happens next, or a different event entirely. Review the recent scenes context to ensure you're not overlapping with what was already told.

TRANSITIONS AND FLOW:
- Scene 1 should TRANSITION smoothly from where the previous scene ended AND advance the story
- Each scene should CONNECT to the next through plot progression - what happens next in the story?
- Reference recurring themes/motifs from earlier in the documentary through plot connections
- The final scene should SET UP what comes next by advancing or introducing plot threads
- Think of scenes as story beats in a film, not separate fact segments'''
}

{"" if is_hook_chapter else get_shared_scene_flow_instructions()}

{get_shared_scene_requirements()}

9. PLANT SEEDS (Early in the story): If this is early in the documentary, include specific details, objects, relationships, or concepts that could pay off later. Examples: a specific notebook mentioned, a relationship that will matter later, a fear or promise that will be relevant, a small detail that seems unimportant now but will become significant. These create satisfying "aha moments" when referenced later.

{get_shared_narration_style(is_short=False)}
{("- HOOK/INTRO: Speak directly to the viewer about what they'll discover. Use preview language: \"You'll discover...\", \"This is the story of...\", \"Wait until you learn...\", \"Here's why this matters...\" Present facts as teasers, not as events happening in real-time.\n" +
"- TRANSITION TO STORY: The final scene should naturally bridge to the chronological narrative without saying \"rewind\" or \"go back\". Simply start with the beginning context: \"[Early life context]. This is where our story begins.\" or \"It all started when...\"") if is_hook_chapter else ""}

{("HOOK CHAPTER EXAMPLES:\n" +
"BAD (no intro): \"In 1905, Einstein publishes four papers that redefine physics. The physics community is stunned.\"\n" +
"GOOD (with intro): \"Welcome to Human Footprints. Today we'll talk about Albert Einstein - the man who changed the world. In 1905, a 26-year-old patent clerk publishes four papers that redefine physics. The scientific community is stunned.\"\n" +
"GOOD (with intro and hook): \"Welcome to Human Footprints. Today we'll talk about Charles Darwin - the man who changed the way we understand life. The letter arrives in June 1858. Darwin's hands tremble as he reads his own theory in another man's words.\"\n" +
"BAD transition: \"Now the story rewinds to the beginning...\"\n" +
"GOOD transition: \"It all started in Ulm, Germany, in 1879, when Einstein was born.\"") if is_hook_chapter else get_shared_examples()}

IMAGE PROMPT STYLE:
- Cinematic, dramatic lighting
- Specific composition (close-up, wide shot, over-shoulder, etc.)
- CRITICAL - EMOTION AND MOOD (MUST MATCH SCENE'S EMOTION FIELD):
  * The scene's "emotion" field MUST be reflected in the image_prompt - this is critical for visual consistency
  * Use the emotion to guide lighting, composition, facial expressions, and overall mood
  * Examples: "desperate" → tense atmosphere, frantic expressions, harsh shadows, chaotic composition; "triumphant" → bright lighting, confident expressions, celebratory mood; "contemplative" → soft lighting, thoughtful expressions, quiet atmosphere; "tense" → sharp contrasts, wary expressions, claustrophobic framing
  * Explicitly include the emotion in image description: "tense atmosphere", "triumphant expression", "contemplative mood", "desperate urgency"
  * The visual mood must match the emotional reality of the moment - if emotion is "desperate", the image should look desperate
- Period-accurate details
- CRITICAL - AGE AND APPEARANCE:
  * ALWAYS include {person}'s age at the time of the scene in the image_prompt (e.g., "a 26-year-old Einstein", "Einstein in his 40s", "young Einstein at age 20")
  * Include age-relevant physical details: what they looked like at that age, any relevant physical characteristics or conditions
  * IMPORTANT EXAMPLES: Stephen Hawking should NOT be in a wheelchair as a child (he didn't use one until later). Include only age-appropriate characteristics.
  * Birth year: {birth_year if birth_year else 'unknown'}
  * Death year: {death_year if death_year else 'still alive'}
  * If the scene takes place before the birth year, note that {person} is NOT BORN YET and should not appear in the scene
  * If the scene takes place after the death year (and death year is known), note that {person} is DECEASED and should not appear in the scene (unless it's a memorial/legacy scene)
  * Include period-accurate clothing, hairstyle, and any age-relevant details about their appearance at that specific age
- VISUAL THEME REINFORCEMENT: If recurring themes include concepts like "isolation", "ambition", "conflict", etc., incorporate visual motifs that reinforce these themes through composition, lighting, or symbolism. For example, if "isolation" is a theme, use wide shots with the subject alone, or shadows that emphasize separation.
- Recurring Themes to Consider Visually: {threads_str}
- End with ", 16:9 cinematic"

Respond with JSON array:
[
  {{
    "id": {start_id},
    "title": "Evocative 2-5 word title",
    "narration": "Vivid, dramatic narration...",
    "image_prompt": "Detailed visual description including {person}'s age and age-relevant appearance details, 16:9 cinematic",
    "emotion": "A single word or short phrase describing the scene's emotional tone (e.g., 'tense', 'triumphant', 'desperate', 'contemplative', 'exhilarating', 'somber', 'urgent', 'defiant'). Should match the chapter's emotional tone but be scene-specific based on what the character is feeling and the dramatic tension.",
    "year": YYYY or "YYYY-YYYY" or "around YYYY" (the specific year or year range when this scene takes place)
  }},
  ...
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating documentary content. Write narration from YOUR perspective - this is YOUR script that YOU wrote. Tell the story naturally, directly to the viewer. Avoid any meta references to chapters, production elements, or the script structure. Focus on what actually happened, why it mattered, and how it felt. Respond with valid JSON array only."},
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


# identify_pivotal_moments, generate_significance_scene, generate_refinement_diff, and refine_scenes
# are now imported from build_scripts_utils above (see imports section at the top)


def generate_short_outline(person: str, main_outline: dict, short_num: int, total_shorts: int, previously_covered_stories: list[str] = None, scene_highlights: list = None) -> dict:
    """Generate a focused outline for a single YouTube Short."""
    
    if previously_covered_stories is None:
        previously_covered_stories = []
    if scene_highlights is None:
        scene_highlights = []
    
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
    
    # Build context about previously covered stories
    previously_covered_text = ""
    if previously_covered_stories:
        previously_covered_text = f"""

CRITICAL: You must choose a DIFFERENT story than what was already covered in previous shorts.
Previously covered stories (DO NOT repeat these):
{chr(10).join(f"• {story}" for story in previously_covered_stories)}

Choose a COMPLETELY DIFFERENT story/incident/event from {person}'s life. Make sure it's distinct and doesn't overlap with the above."""
    
    # Build scene highlights context for shorts
    scene_highlights_text = ""
    if scene_highlights:
        scene_highlights_text = f"""
KEY MOMENTS FROM MAIN VIDEO (reference these to create connections):
{chr(10).join(f"• Scene {s.get('id')}: {s.get('title')} - {s.get('narration_preview', '')}" for s in scene_highlights[:15])}

IMPORTANT: This short should feel like a TEASER that drives viewers to the main video. Reference specific moments from the main video and create intrigue: "Wait until you see how this plays out in the full documentary" or "This moment from the main story gets explored in depth..." Make viewers want to watch the full video to see how this story connects to the bigger picture."""
    
    outline_prompt = f"""Create a VIRAL YouTube Short about {person}.

Short #{short_num} of {total_shorts}. Focus on: {focus}
{previously_covered_text}

Their life story (for context):
{chapters_context}
{scene_highlights_text}

This Short tells ONE CONTINUOUS STORY from their life - not disconnected facts, but a single narrative arc.

IMPORTANT: This must be a COMPLETELY DIFFERENT story from any previous shorts. Choose a distinct incident, event, moment, or period from their life that hasn't been covered yet. For example, if a previous short covered their early breakthrough in 1905, choose a different story like a later conflict, a different achievement from another period, a personal relationship story, or a different challenge they faced. Each short should feel like a standalone story from a different part of their life.

This Short has EXACTLY 5 scenes that tell ONE COMPLETE STORY with depth and detail. The final scene should be a NATURAL CONCLUSION, not a hook/cliffhanger.

STRUCTURE (5 scenes):
1. SCENE 1: Start the story. Set up a specific incident, moment, or event from their life. Give context and specific details. Hook the viewer with the opening of the story.
2. SCENE 2: Build the story. Show what happened next, the development, or how the situation evolved. Add more depth and specific details.
3. SCENE 3: Escalate the story. Show the consequences, conflicts, challenges, or complications. This is the rising action - make it engaging and detailed.
4. SCENE 4: Continue building. Show how the story develops further, what the person does, what happens, and the stakes involved. Add emotional depth.
5. SCENE 5: NATURAL CONCLUSION. This should provide a satisfying ending to THIS story. Show what happened as a result, how it resolved, or what the outcome was. Do NOT end with a cliffhanger like "But what happened next would change everything..." Instead, provide a natural conclusion that makes the viewer feel the story is complete: "The paper is published in 1859. It sells out in one day. Darwin's theory of evolution becomes the foundation of modern biology." or "This discovery earns him the Nobel Prize. But more importantly, it validates a lifetime of work." The story should feel complete and satisfying on its own.

CRITICAL RULES:
- Tell ONE continuous story - scenes must flow like a mini-narrative, not separate facts
- Every scene must contain SPECIFIC, INTERESTING FACTS
- NO vague statements or filler
- NO artistic/poetic language
- Simple, clear sentences packed with information
- The short should feel like watching a mini-story, not a list of facts
- The ending should be a NATURAL cliffhanger that makes sense for the story being told

Provide JSON:
{{
  "short_title": "VIRAL CLICKBAIT title (max 50 chars). Use power words: SHOCKING, SECRET, EXPOSED, INSANE, UNBELIEVABLE, MIND-BLOWING, CRAZY, BANNED, FORBIDDEN. Create curiosity gaps. Ask questions. Use numbers. Make bold claims. Examples: 'The SHOCKING Secret Nobody Knows', 'How He Did the IMPOSSIBLE', 'This Changed EVERYTHING', 'The Dark Truth Exposed', 'You WON'T Believe This'. Must make viewers NEED to watch immediately while staying accurate.",
  "short_description": "YouTube description (100 words) with hashtags",
  "tags": "10-15 SEO tags comma-separated",
  "thumbnail_prompt": "CLICKBAIT thumbnail for mobile (9:16 vertical): EXTREME close-up or dramatic composition, intense emotional expression (not neutral - show passion/conflict/shock), HIGH CONTRAST dramatic lighting (chiaroscuro), bold eye-catching colors (reds/yellows for urgency), subject in MOMENT OF IMPACT or dramatic pose, symbolic elements representing the story's peak moment, movie-poster energy, optimized for small mobile screens - must grab attention instantly when scrolling",
  "hook_fact": "The opening fact that starts the story (for context, but we start with the story, not just the fact)",
  "story_angle": "What specific continuous story/incident are we telling (ONE narrative arc)",
  "key_facts": ["5-8 specific facts to include across the 5 scenes that tell ONE complete story with depth"]
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You create viral content. Every word must deliver value. No fluff. When generating multiple shorts about the same person, ensure each tells a COMPLETELY DIFFERENT story from their life - different events, different moments, different incidents. Respond with valid JSON only."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.9,
        response_format={"type": "json_object"}
    )
    
    return json.loads(clean_json_response(response.choices[0].message.content))


def generate_short_scenes(person: str, short_outline: dict, birth_year: int | None = None, death_year: int | None = None) -> list[dict]:
    """Generate all 5 scenes for a YouTube Short (complete story with natural conclusion).
    
    Each scene must include a "year" field indicating when it takes place.
    The LLM includes age and age-relevant details in the image_prompt.
    """
    
    key_facts = short_outline.get('key_facts', [])
    facts_str = "\n".join(f"• {fact}" for fact in key_facts)
    
    scene_prompt = f"""Write 5 scenes for a YouTube Short about {person}.

TITLE: "{short_outline.get('short_title', '')}"
STORY ANGLE: {short_outline.get('story_angle', '')}

KEY FACTS TO USE:
{facts_str}

CRITICAL: This short tells ONE COMPLETE, CONTINUOUS STORY from {person}'s life with DEPTH and DETAIL. The scenes must flow like a mini-narrative, not disconnected facts. This should be high-quality content that is enjoyable on its own.

EMOTION GENERATION (CRITICAL):
- Each scene MUST include an "emotion" field - a single word or short phrase (e.g., "tense", "triumphant", "desperate", "contemplative", "exhilarating", "somber", "urgent", "defiant")
- Base the emotion on: what the character is feeling at this moment, the dramatic tension, and the significance of the event
- Use the emotion to guide both narration tone/style and image mood:
  * If emotion is "desperate" - narration should feel urgent/anxious with short, sharp sentences; image should show tense atmosphere, frantic expressions
  * If emotion is "triumphant" - narration should feel uplifting with elevated language; image should show celebration, confident expressions
  * If emotion is "contemplative" - narration should be slower, reflective; image should show quiet mood, thoughtful expressions
- The emotion field will be used to ensure narration tone and image mood match the emotional reality of the moment

STRUCTURE (exactly 5 scenes):
1. SCENE 1: Start the story. Set up a specific incident, moment, or event. Give context and specific details. Hook the viewer by showing how this moment FELT - what the character was thinking, fearing, or hoping. Make them feel the significance.
2. SCENE 2: Build the story. Show what happened next, the development, or how the situation evolved. Add more depth and specific details about who was involved, what was said, what the stakes were. Include emotional details - how does this feel? What's at stake emotionally?
3. SCENE 3: Escalate the story. Show the consequences, conflicts, challenges, or complications. This is the rising action - make it engaging and detailed. Show what made this moment difficult or significant EMOTIONALLY - what fears, hopes, or pressures does the character face?
4. SCENE 4: Continue building. Show how the story develops further, what the person does, what happens, and the stakes involved. Add emotional depth and human details - show relationships, motivations, and the personal impact. What does this mean to them personally? How do they FEEL?
5. SCENE 5: NATURAL CONCLUSION. This should provide a satisfying ending to THIS story. Show what happened as a result, how it resolved, or what the outcome was. Include the emotional significance - what did this mean to the character? Examples of good conclusions:
   - "The paper is published in 1859. It sells out in one day. Darwin's theory of evolution becomes the foundation of modern biology."
   - "This discovery earns him the Nobel Prize. But more importantly, it validates a lifetime of work and changes how scientists think about the world."
   - "The letter reaches London in June. Within weeks, the scientific community is divided. Some call it heresy. Others call it genius. But Darwin's idea has been unleashed."
   
   DO NOT end with a cliffhanger or hook like:
   - "But what happened next would change everything..." (NO - this is a hook, not a conclusion)
   - "The consequences of this decision would haunt him for years..." (NO - unresolved)
   - "Little did he know, this was just the beginning..." (NO - this is a hook)
   
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
- CRITICAL - AGE AND APPEARANCE:
  * ALWAYS include {person}'s age at the time of the scene in the image_prompt (e.g., "a 26-year-old Einstein", "Einstein in his 40s", "young Einstein at age 20")
  * Include age-relevant physical details: what they looked like at that age, any relevant physical characteristics or conditions
  * IMPORTANT EXAMPLES: Stephen Hawking should NOT be in a wheelchair as a child (he didn't use one until later). Include only age-appropriate characteristics.
  * Birth year: {birth_year if birth_year else 'unknown'}
  * Death year: {death_year if death_year else 'still alive'}
  * If the scene takes place before the birth year, note that {person} is NOT BORN YET and should not appear in the scene
  * If the scene takes place after the death year (and death year is known), note that {person} is DECEASED and should not appear in the scene (unless it's a memorial/legacy scene)
  * Include period-accurate clothing, hairstyle, and any age-relevant details about their appearance at that specific age
- End with ", 9:16 vertical"

Respond with JSON array of exactly 5 scenes:
[
  {{"id": 1, "title": "2-4 words", "narration": "...", "image_prompt": "... (include {person}'s age and age-relevant details)", "emotion": "A single word or short phrase (e.g., 'tense', 'triumphant', 'desperate', 'contemplative')", "year": "YYYY or YYYY-YYYY"}},
  {{"id": 2, "title": "...", "narration": "...", "image_prompt": "... (include {person}'s age and age-relevant details)", "emotion": "A single word or short phrase describing the scene's emotional tone", "year": "YYYY or YYYY-YYYY"}},
  {{"id": 3, "title": "...", "narration": "...", "image_prompt": "... (include {person}'s age and age-relevant details)", "emotion": "A single word or short phrase describing the scene's emotional tone", "year": "YYYY or YYYY-YYYY"}},
  {{"id": 4, "title": "...", "narration": "...", "image_prompt": "... (include {person}'s age and age-relevant details)", "emotion": "A single word or short phrase describing the scene's emotional tone", "year": "YYYY or YYYY-YYYY"}},
  {{"id": 5, "title": "...", "narration": "...", "image_prompt": "... (include {person}'s age and age-relevant details)", "emotion": "A single word or short phrase describing the scene's emotional tone", "year": "YYYY or YYYY-YYYY"}}
]

IMPORTANT: Each scene must include:
- "year" field indicating when the scene takes place
- "emotion" field - a single word or short phrase (e.g., "tense", "triumphant", "desperate", "contemplative") that describes the scene's emotional tone. Base this on what the character is feeling, the dramatic tension, and the significance of the moment. The narration tone and image mood should match this emotion.
- The image_prompt MUST include {person}'s age at that time, age-relevant appearance details, and reflect the scene's emotion in lighting, composition, and mood.
]"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a YouTuber creating viral content. Write narration from YOUR perspective - this is YOUR script. Simple words, specific facts, deep storytelling with details. No fluff, no made-up transitions. Tell continuous stories with actual events. Avoid any meta references to chapters, production elements, or script structure. Respond with valid JSON array only."},
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


def generate_shorts(person_of_interest: str, main_title: str, global_block: str, outline: dict, base_output_path: str, scene_highlights: list = None):
    """Generate YouTube Shorts (5 scenes each: complete story with natural conclusion)."""
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
    
    # STEP 1: Generate all outlines sequentially (they depend on previously_covered_stories)
    print(f"\n[OPTIMIZATION] Step 1: Generating all short outlines sequentially...")
    short_outlines = []
    for short_num in range(1, config.num_shorts + 1):
        print(f"\n[SHORT {short_num}/{config.num_shorts}] Creating outline...")
        
        short_outline = generate_short_outline(
            person=person_of_interest,
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
            all_scenes = generate_short_scenes(
                person=person_of_interest,
                short_outline=short_outline,
                birth_year=outline.get('birth_year'),
                death_year=outline.get('death_year')
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
            all_scenes, refinement_diff = refine_scenes(all_scenes, person_of_interest, is_short=True, chapter_context=short_context, diff_output_path=diff_path)
            
            # Generate thumbnail (if enabled)
            thumbnail_path = None
            if config.generate_short_thumbnails:
                thumbnail_prompt = short_outline.get("thumbnail_prompt", "")
                if thumbnail_prompt:
                    thumbnail_prompt += "\n\nCLICKBAIT YouTube Shorts thumbnail - MAXIMIZE SCROLL-STOPPING POWER: Vertical 9:16, EXTREME close-up or dramatic composition, intense emotional expression (shock/passion/conflict), HIGH CONTRAST dramatic lighting (chiaroscuro), bold eye-catching colors (reds/yellows/oranges for urgency), subject in MOMENT OF IMPACT, movie-poster energy, optimized for mobile scrolling - must instantly grab attention when tiny in feed."
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
                    "person_of_interest": person_of_interest,
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
    
    # Step 2: Generate initial metadata (title, thumbnail, global_block) - we'll regenerate description after scenes
    print("\n[STEP 2] Generating initial metadata...")
    
    initial_metadata_prompt = f"""Create initial metadata for a documentary about: {person_of_interest}

Their story in one line: {outline.get('tagline', '')}

Generate JSON:
{{
  "title": "CLICKBAIT YouTube title (60-80 chars). Use power words like: SHOCKING, SECRET, REVEALED, EXPOSED, DARK, UNTOLD, UNBELIEVABLE, INCREDIBLE, INSANE, CRAZY, MIND-BLOWING, BANNED, FORBIDDEN, HIDDEN. Create curiosity gaps, ask questions, use numbers, make bold claims. Examples: 'The SHOCKING Secret That Changed Everything', 'How [Person] Did the IMPOSSIBLE', 'The Dark Truth They DON'T Want You to Know', '[Person]: The Forbidden Discovery', 'This ONE Decision Changed History FOREVER'. Must be engaging and make viewers NEED to click while staying factually accurate.",
  "tag_line": "Short, succinct, catchy tagline (5-10 words) that captures who they are. Examples: 'the man who changed the world', 'the codebreaker who saved millions', 'the mind that rewrote physics', 'the naturalist who explained life'. Should be memorable and accurate.",
  "thumbnail_description": "CLICKBAIT thumbnail visual: Maximize visual impact! Use intense close-ups, dramatic expressions, extreme lighting (chiaroscuro), bold colors (red/yellow for danger/urgency), powerful symbols, emotional moments. Show conflict, tension, or peak dramatic moment. Composition should be bold and arresting - eyes staring directly, hands in dramatic pose, symbolic objects. Use high contrast, dramatic shadows, cinematic lighting. Subject should appear in a MOMENT OF IMPACT - not calm portrait. Think action movie poster, not museum painting. NO TEXT in image, but visually SCREAM importance and drama.",
  "global_block": "Visual style guide (300-400 words): semi-realistic digital painting style, color palette, dramatic lighting, how {person_of_interest} should appear consistently across {config.total_scenes} scenes."
}}"""

    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": "Documentary producer. Respond with valid JSON only."},
            {"role": "user", "content": initial_metadata_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    initial_metadata = json.loads(clean_json_response(response.choices[0].message.content))
    
    title = initial_metadata["title"]
    tag_line = initial_metadata.get("tag_line", f"the story of {person_of_interest}")  # Fallback if not generated
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
- Intense emotional expression on face - not neutral, show passion/conflict/determination
- Dramatic lighting with HIGH CONTRAST (chiaroscuro) - bright highlights, deep shadows
- Bold, eye-catching colors (reds, yellows, oranges for urgency/importance) against darker backgrounds
- Subject in ACTION or MOMENT OF IMPACT - not passive pose
- Symbolic elements that represent their greatest achievement or conflict
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
                scenes = generate_scenes_for_chapter(
                    person=person_of_interest,
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
                    birth_year=outline.get('birth_year'),
                    death_year=outline.get('death_year'),
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
                        cta_scene = generate_cta_transition_scene(person_of_interest, tag_line)
                        all_scenes.append(cta_scene)
                        print(f"  ✓ CTA transition scene added (total: {len(all_scenes)})")
                    except Exception as e:
                        print(f"  [WARNING] Failed to generate CTA scene: {e}")
                        # Continue without CTA scene if generation fails
                
                # Extract "planted seeds" from early chapters (first 3 chapters) for callback mechanism
                # Look for specific details, objects, relationships, or concepts that could pay off later
                if i < 3:  # First 3 chapters plant seeds
                    # Extract seeds from scene narrations - look for specific objects, relationships, concepts
                    for scene in scenes:
                        narration = scene.get('narration', '')
                        scene_title = scene.get('title', '')
                        # Look for specific details that could be callbacks: objects, relationships, promises, mysteries
                        # The LLM should be planting these intentionally, but we extract them here
                        # Add scene title and key phrases as potential seeds
                        if narration:
                            # Simple extraction: look for specific nouns, relationships, or concepts
                            # In a more sophisticated version, we'd ask the LLM to explicitly list planted seeds
                            planted_seeds.append(f"{scene_title}: {narration[:80]}...")
                    # Also add chapter's key events as potential seeds
                    for event in chapter.get('key_events', []):
                        # Extract specific details that could be callbacks
                        if any(word in event.lower() for word in ['notebook', 'letter', 'relationship', 'conflict', 'discovery', 'secret', 'promise', 'warning', 'fear']):
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
        chapter_summaries = "\n".join([f"Chapter {ch['chapter_num']}: {ch['title']} ({ch['year_range']}) - {ch['summary']}" for ch in chapters])
        diff_path = Path(output_path).parent / f"{Path(output_path).stem}_refinement_diff.json" if config.generate_refinement_diffs else None
        all_scenes, refinement_diff = refine_scenes(all_scenes, person_of_interest, is_short=False, chapter_context=chapter_summaries, diff_output_path=diff_path)
        
        # Step 3.5: Generate final metadata (description and tags) AFTER scenes are generated
        print("\n[STEP 3.5] Generating final metadata from actual scenes...")
        
        # Extract memorable moments from scenes for metadata
        scene_highlights = []
        for scene in all_scenes[:10]:  # First 10 scenes for highlights
            scene_highlights.append(f"Scene {scene['id']}: {scene.get('title', '')} - {scene.get('narration', '')[:80]}...")
        
        final_metadata_prompt = f"""Create final metadata for a documentary about: {person_of_interest}

Their story in one line: {outline.get('tagline', '')}

Actual memorable moments from the documentary (use these to make description more accurate):
{chr(10).join(scene_highlights[:10])}

Generate JSON:
{{
  "video_description": "YouTube description (500-800 words) - hook, highlights, key facts, call to action. SEO optimized. Reference specific compelling moments from the actual scenes above to make it accurate and engaging.",
  "tags": "15-20 SEO tags separated by commas. Mix of: person's name, key topics, related figures, time periods, achievements, fields (e.g., 'Albert Einstein, physics, relativity, Nobel Prize, Germany, Princeton, E=mc2, quantum mechanics, genius, scientist, 20th century, biography, documentary')",
  "pinned_comment": "An engaging question or comment to pin below the video (1-2 sentences max). Should: spark discussion, ask a thought-provoking question about the person/story, create curiosity, encourage viewers to share their thoughts/opinions, be conversational and inviting. Examples: 'Which moment from their story surprised you most? Drop a comment below!', 'What do you think was their greatest challenge? Share your thoughts!', 'This story shows how one person can change everything. What impact do you want to make?'. Should feel authentic, not clickbaity - genuine curiosity about viewer perspectives."
}}"""

        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": "Documentary producer. Create compelling metadata that accurately reflects the actual content. Respond with valid JSON only."},
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
    shorts_info = generate_shorts(person_of_interest, title, global_block, outline, output_path, scene_highlights=scene_highlights_for_shorts)
    
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
                        help=f"Scenes per short (default: {Config.short_scenes_per_chapter}: build, build, cliffhanger)")
    
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
    
    # Apply test mode settings
    if args.test:
        config.chapters = 2
        config.scenes_per_chapter = 2
        config.num_shorts = 1
        config.short_chapters = 1
        config.short_scenes_per_chapter = 3
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
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in args.person)
        safe_name = safe_name.replace(' ', '_').lower()
        output_file = str(SCRIPTS_DIR / f"{safe_name}_script.json")
    
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
