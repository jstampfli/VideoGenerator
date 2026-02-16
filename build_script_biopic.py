import os
import sys
import json
import base64
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Import shared utilities and LLM utils (provider/model from .env)
import build_scripts_utils
import llm_utils
from kenburns_config import KENBURNS_PATTERNS, get_pattern_prompt_str, get_intensity_prompt_str
from biopic_schemas import (
    VIDEO_QUESTIONS_SCHEMA,
    LANDMARK_EVENTS_SCHEMA,
    OUTLINE_SCHEMA,
    SCENE_OUTLINE_SCHEMA,
    SCENES_ARRAY_SCHEMA,
    INITIAL_METADATA_SCHEMA,
    INITIAL_METADATA_3_OPTIONS_NO_GLOBAL_SCHEMA,
    GLOBAL_BLOCK_SCHEMA,
    SHORT_OUTLINE_SCHEMA,
    FINAL_METADATA_SCHEMA,
)

load_dotenv()

THUMBNAILS_DIR = Path("thumbnails")
SHORTS_DIR = Path("shorts_scripts")
SCRIPTS_DIR = Path("scripts")

# Default thumbnail text phrases per WHY type (used when LLM does not provide thumbnail_text)
THUMBNAIL_WHY_TYPE_DEFAULTS = {
    "counterintuitive": "NOT WHAT YOU THINK",
    "secret_mystery": "THE SECRET",
    "known_but_misunderstood": "EVERYONE GETS THIS WRONG",
}


def _thumbnail_text_instruction(why_type: str | None, custom_text: str | None) -> str:
    """Return a prompt block for thumbnail text overlay. Always returns an instruction so text is never omitted."""
    if why_type:
        text = (custom_text or "").strip() or THUMBNAIL_WHY_TYPE_DEFAULTS.get(why_type, "")
        if text:
            return f'''

CRITICAL - TEXT ON IMAGE (MANDATORY): The thumbnail MUST display bold, legible text that says exactly: "{text}". Do not omit the text. TEXT PLACEMENT: Put the text in the LOWER third or bottom area of the image—NEVER at the top or near the top edge. Keep the text fully INSIDE the frame with clear margin from all edges so it is never cut off or clipped. The image is incomplete without this text.'''
    # Fallback when why_type is missing (e.g. older scripts): still require text
    return '''

CRITICAL - TEXT ON IMAGE (MANDATORY): The thumbnail MUST display bold, legible text. Use one of these exact phrases: "THE SECRET", "NOT WHAT YOU THINK", or "EVERYONE GETS THIS WRONG". Place text in the LOWER third only—never at the top. Keep text fully inside the frame with margin so it is never cut off. The image is incomplete without this text.'''


# Import config from separate module
import config
config = config.Config()

# Import shared functions from utils - use directly
clean_json_response = build_scripts_utils.clean_json_response
generate_refinement_diff = build_scripts_utils.generate_refinement_diff
identify_pivotal_moments = build_scripts_utils.identify_pivotal_moments
generate_significance_scene = build_scripts_utils.generate_significance_scene


# Shared prompt functions are now imported from build_scripts_utils
# Historical-specific prompt wrappers use "historical" content_type


# generate_thumbnail is now imported directly from build_scripts_utils
# Use build_scripts_utils.generate_thumbnail(prompt, output_path, size, config.generate_thumbnails) directly


def generate_video_questions(person_of_interest: str, research_context=None) -> list[str]:
    """
    Generate the question(s) this documentary will answer.
    This is the FIRST step - the questions frame the outline, chapters, and scenes.
    """
    import prompt_builders

    print(f"\n[QUESTIONS] Generating video questions (frames everything else)...")
    prompt = prompt_builders.build_video_questions_prompt(person_of_interest, research_context=research_context)

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "Documentary producer. Your job is to craft the most engaging, compelling questions that will make viewers stop scrolling and click. These questions are the primary hook—they appear at the very start and drive the entire video. Frame them for an audience that values substance, legacy, and leadership. Make them impossible to ignore."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        response_json_schema=VIDEO_QUESTIONS_SCHEMA,
        provider=llm_utils.get_provider_for_step("VIDEO_QUESTIONS"),
    )
    data = json.loads(clean_json_response(content))
    questions = data.get("video_questions", [])
    if not isinstance(questions, list):
        questions = [questions] if questions else []
    questions = [str(q).strip() for q in questions if q]

    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q}")
    return questions


def generate_landmark_events(person_of_interest: str, research_context=None) -> list[dict]:
    """
    Generate the most important landmark events in the person's life.
    These will become their own chapters with deep, in-depth coverage.
    Returns list of dicts with event_name, year_or_period, significance, key_details_to_cover.
    """
    num_landmarks = min(4, max(1, config.chapters - 2))  # Room for hook + legacy
    print(f"\n[LANDMARKS] Identifying {num_landmarks} pivotal moments...")

    import prompt_builders
    prompt = prompt_builders.build_landmark_events_prompt(person_of_interest, num_landmarks=num_landmarks, research_context=research_context)
    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": f"You are an expert biographer who identifies the defining moments in a person's life. Focus on works, breakthroughs, and decisions that deserve deep documentary treatment with technical and historical detail. {prompt_builders.get_biopic_audience_profile()}"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        response_json_schema=LANDMARK_EVENTS_SCHEMA,
        provider=llm_utils.get_provider_for_step("LANDMARKS"),
    )
    data = json.loads(clean_json_response(content))
    landmarks = data.get("landmark_events", [])
    if not isinstance(landmarks, list):
        landmarks = []
    for lm in landmarks:
        print(f"  • {lm.get('event_name', '?')} ({lm.get('year_or_period', '?')})")
    return landmarks


def generate_outline(person_of_interest: str, landmark_events: list[dict] | None = None,
                    video_questions: list[str] | None = None, research_context=None) -> dict:
    """Generate a detailed chronological outline of the person's life, framed by video_questions."""
    print(f"\n[OUTLINE] Generating {config.chapters}-chapter outline...")

    # Discover available music moods from biopic_music/ for LLM to pick per chapter
    try:
        from biopic_music_config import get_available_moods
        available_moods = get_available_moods()
        print(f"[OUTLINE] Music moods available: {available_moods}")
    except ImportError:
        available_moods = ["relaxing", "passionate", "happy"]

    # Use prompt builder for outline - video_questions frame the narrative
    import prompt_builders
    outline_prompt = prompt_builders.build_outline_prompt(
        person_of_interest, config.chapters, config.target_total_scenes,
        config.min_scenes_per_chapter, config.max_scenes_per_chapter,
        available_moods=available_moods,
        landmark_events=landmark_events,
        video_questions=video_questions,
        research_context=research_context,
    )

    import prompt_builders
    system_msg = f"You are a historian who finds the drama and humanity in every life story. {prompt_builders.get_biopic_audience_profile()}"
    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.7,
        response_json_schema=OUTLINE_SCHEMA,
        provider=llm_utils.get_provider_for_step("OUTLINE"),
    )
    outline_data = json.loads(clean_json_response(content))
    chapters = outline_data.get("chapters", [])
    
    # Ensure each chapter has num_scenes (fallback for legacy or incomplete LLM output)
    fallback_per_chapter = max(config.min_scenes_per_chapter, config.target_total_scenes // max(1, len(chapters)))
    for ch in chapters:
        if "num_scenes" not in ch or not isinstance(ch.get("num_scenes"), (int, float)):
            ch["num_scenes"] = fallback_per_chapter
            print(f"[OUTLINE] Added num_scenes={fallback_per_chapter} to Ch {ch.get('chapter_num')} (missing from outline)")
        ch["num_scenes"] = max(config.min_scenes_per_chapter, min(config.max_scenes_per_chapter, int(ch["num_scenes"])))
        # Fallback music_mood for legacy scripts or incomplete LLM output
        if not ch.get("music_mood") or not isinstance(ch.get("music_mood"), str):
            ch["music_mood"] = available_moods[0] if available_moods else "relaxing"
            print(f"[OUTLINE] Added music_mood={ch['music_mood']} to Ch {ch.get('chapter_num')} (missing from outline)")
    
    print(f"[OUTLINE] Generated {len(chapters)} chapters")
    for ch in chapters:
        year_range = ch.get('year_range', ch.get('time_setting', 'Unknown'))
        num_scenes = ch.get('num_scenes', '?')
        ch_type = ch.get('chapter_type', '')
        type_str = f" [{ch_type}]" if ch_type else ""
        print(f"  Ch {ch['chapter_num']}: {ch['title']} ({year_range}) - {num_scenes} scenes{type_str}")

    # Inject video_questions into outline so downstream steps have them
    if video_questions:
        outline_data["video_questions"] = video_questions

    return outline_data


# CTA scene generation removed - no longer needed


def generate_scene_outline(chapter: dict, person: str, scene_budget: int, landmarks: list[dict] | None = None, research_context=None) -> list[dict]:
    """
    Generate scene blocks for a chapter - allocates scene_budget across events with variable depth.
    
    For hook chapter (ch 1), may return a single block or simple allocation.
    For story chapters, pivotal moments get 2-4 scenes; minor events get 1.
    
    Returns list of dicts: [{"event": str, "num_scenes": int, "rationale": str}, ...]
    Sum of num_scenes must equal scene_budget.
    """
    import prompt_builders
    
    scene_outline_prompt = prompt_builders.build_scene_outline_prompt(chapter, person, scene_budget, landmarks=landmarks, research_context=research_context)
    
    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": f"You allocate documentary scenes across events. Pivotal moments get 2-4 scenes; minor events get 1. Sum of num_scenes must equal the budget exactly. {prompt_builders.get_biopic_audience_profile()}"},
            {"role": "user", "content": scene_outline_prompt}
        ],
        temperature=0.5,
        response_json_schema=SCENE_OUTLINE_SCHEMA,
        provider=llm_utils.get_provider_for_step("SCENES"),
    )
    blocks = json.loads(clean_json_response(content))
    
    if not isinstance(blocks, list):
        raise ValueError(f"Scene outline expected array, got {type(blocks)}")
    
    total = sum(b.get("num_scenes", 0) for b in blocks)
    if total != scene_budget:
        # Normalize: scale or pad to match budget
        if total == 0:
            blocks = [{"event": chapter.get("summary", "Chapter content"), "num_scenes": scene_budget, "rationale": "Fallback"}]
        else:
            # Adjust the largest block to hit the target
            blocks = sorted(blocks, key=lambda b: b.get("num_scenes", 0), reverse=True)
            diff = scene_budget - total
            blocks[0]["num_scenes"] = blocks[0].get("num_scenes", 1) + diff
    
    return blocks


def generate_scenes_for_chapter(person: str, chapter: dict, scene_blocks: list[dict], chapter_scene_budget: int, start_id: int, 
                                 global_style: str, prev_chapter: dict = None, prev_scenes: list = None,
                                 central_theme: str = None, narrative_arc: str = None, 
                                 planted_seeds: list[str] = None, is_retention_hook_point: bool = False,
                                 birth_year: int | None = None, death_year: int | None = None,
                                 tag_line: str | None = None, overarching_plots: list[dict] = None,
                                 sub_plots: list[dict] = None, landmarks: list[dict] | None = None,
                                 video_questions: list[str] | None = None, research_context=None) -> list[dict]:
    """Generate scenes for a single chapter. Uses scene_blocks for variable depth per event."""
    import prompt_builders
    
    if planted_seeds is None:
        planted_seeds = []
    
    # Build scene block context for multi-scene events
    blocks_instruction = ""
    if any(b.get("num_scenes", 1) > 1 for b in scene_blocks):
        blocks_instruction = """
SCENE BLOCKS - VARIABLE DEPTH (CRITICAL for blocks with multiple scenes):
""" + "\n".join(
            f"- Block: \"{b.get('event', '')}\" → {b.get('num_scenes', 1)} scene(s). {b.get('rationale', '')}"
            for b in scene_blocks
        ) + """

For events with multiple scenes, create a SEQUENCE that explores the moment in depth:
- Scene 1: Setup/context - who, where, why this moment is approaching
- Scene 2: The event itself - what happens, step by step
- Scene 3+: Reactions, consequences, or significance - how the world responds, what changes
Do NOT repeat the same information across scenes; each scene adds NEW detail.
"""
    
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
        pacing_instruction = "SLOWER PACING: Scenes should have emotional weight and sufficient context. Use 3-4 sentences per scene (~18-24 seconds). Establish the situation, who's involved, and why it matters before the event. Allow moments to breathe."
    else:
        pacing_instruction = "MODERATE PACING: Use 2-4 sentences per scene (~12-24 seconds). When a scene needs more context—new location, new characters, a pivotal moment—use 3-4 sentences to establish the situation before the event. Prioritize depth and clarity over brevity."
    
    if prev_chapter:
        prev_context = f"""PREVIOUS CHAPTER (for continuity):
Chapter {prev_chapter['chapter_num']}: "{prev_chapter['title']}" ({prev_chapter.get('year_range') or prev_chapter.get('time_setting', 'Unknown')})
Summary: {prev_chapter['summary']}
Emotional Tone: {prev_chapter['emotional_tone']}
"""
    elif is_hook_chapter:
        prev_context = """THIS IS THE HOOK CHAPTER - a rapid-fire "trailer" for the documentary.
Tease the most shocking, interesting, and dramatic moments from their ENTIRE life.
CRITICAL: Scenes should be in CHRONOLOGICAL ORDER by year - start with earlier moments and progress to later ones.
Even though this is a preview, the scenes should flow chronologically to create a sense of progression.
End with a transition like "But how did it all begin?" to set up Chapter 2."""
    else:
        prev_context = "This is the OPENING chapter - establish the story with impact!"
    
    # Include last few scenes for continuity and to avoid overlapping events - include FULL scene JSON
    if prev_scenes and len(prev_scenes) > 0:
        # Get last 5 scenes for context (increased to better track what was covered)
        recent_scenes = prev_scenes[-5:]
        scenes_context = "RECENT SCENES - AVOID REPEATING THESE EVENTS, maintain continuity to naturally continue the story, and ensure smooth transitions:\n"
        for sc in recent_scenes:
            # Include full scene JSON for proper emotion and narration_instructions transitions
            scenes_context += f"  Scene {sc.get('id')} (FULL JSON): {json.dumps(sc, indent=2)}\n"
        scenes_context += "\nCRITICAL: Do NOT repeat or overlap with events already covered in the scenes above. Each scene must cover DIFFERENT events. If an event was already described, move to its consequences or the next significant moment.\n"
        scenes_context += "CRITICAL: For smooth transitions, ensure:\n"
        scenes_context += "- emotion flows gradually from the previous scene's emotion\n"
        scenes_context += "- narration_instructions flow smoothly from the previous scene's narration_instructions\n"
    else:
        scenes_context = ""
    
    # Get narrative context
    connects_to_next = chapter.get('connects_to_next', '')
    recurring_threads = chapter.get('recurring_threads', [])
    threads_str = ', '.join(recurring_threads) if recurring_threads else 'none specified'
    
    # Build central theme and narrative arc context
    theme_context = ""
    if is_hook_chapter and video_questions:
        questions_str = "\n".join(f"• {q}" for q in video_questions)
        theme_context += f"""
VIDEO QUESTION (must be woven into chapter 1 naturally—as the central thread):
{questions_str}

CRITICAL - QUESTION BEFORE ITS FACTS: The question must come BEFORE any facts that help answer it. The question creates curiosity; the facts satisfy it. So: [Question] → [facts that answer it]. The question can unfold across scenes rather than being forced into one beat. NEVER put facts that answer the question before the question—that inverts the paradigm.
- CORRECT: "How did Patton nearly destroy his career before his greatest triumph? By 1943, his tactical genius was undeniable, yet a single moment of rage in a Sicilian medical tent nearly erased his legacy."
- WRONG: "By 1943, his tactical genius was undeniable, yet a single moment of rage... How did Patton nearly destroy his career?" (facts before the question they answer = backwards)
"""
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
    
    # First story chapter (Ch 2): must start from birth/early life - seamless transition from hook
    first_story_chapter_instruction = ""
    if not is_hook_chapter and prev_chapter and prev_chapter.get("chapter_num") == 1:
        first_story_chapter_instruction = """
CRITICAL - FIRST STORY CHAPTER (transition from hook): The hook's final scene promised "Let's start from the beginning" or "It all started when...". You MUST deliver on that. The FIRST scene of this chapter MUST establish birth and early life—where they came from, their childhood, upbringing. Do NOT jump to adulthood. Viewers need to see the beginning of the chronological story. Brief is fine (1-2 scenes for sparse childhood) but never skip entirely."""

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

    # Build landmark context - LLM infers when this chapter covers a landmark
    landmark_instruction = ""
    if landmarks and not is_hook_chapter:
        lines = []
        for lm in landmarks:
            details = lm.get("key_details_to_cover", [])
            details_str = "\n    ".join(f"• {d}" for d in details)
            lines.append(f"- \"{lm.get('event_name', '')}\" ({lm.get('year_or_period', '')}): {lm.get('significance', '')}\n  Key details to cover:\n    {details_str}")
        landmark_list = "\n".join(lines)
        landmark_instruction = f"""

LANDMARK EVENTS - If this chapter covers one of these, you MUST include its key_details_to_cover across your scenes:
{landmark_list}

Decide based on chapter title, summary, key_events, and time period. Structure: setup → event with key details → reactions/significance. Weave details naturally into the narration."""

    # Build research context block for scene generation
    research_block = ""
    if research_context and not research_context.is_empty():
        import research_utils
        research_block = research_utils.get_research_context_block(research_context)
    
    # Calculate total scenes for prompt (approximate - actual is sum of chapter num_scenes)
    total_scenes_for_prompt = config.target_total_scenes
    scene_prompt = f"""You are writing scenes {start_id}-{start_id + chapter_scene_budget - 1} of a ~{total_scenes_for_prompt}-scene documentary about {person}.
{research_block}
{prev_context}
{scenes_context}
{blocks_instruction}
NOW WRITING CHAPTER {chapter['chapter_num']} of {config.chapters}: "{chapter['title']}"
Time Period: {chapter.get('year_range') or chapter.get('time_setting', 'Unknown')}
Emotional Tone: {chapter['emotional_tone']}
Dramatic Tension: {chapter['dramatic_tension']}
Recurring Themes: {threads_str}
Sets Up What Comes Next: {connects_to_next}

{prompt_builders.get_why_what_paradigm_prompt(is_hook_chapter=is_hook_chapter)}

{prompt_builders.get_emotion_generation_prompt(chapter.get('emotional_tone', ''))}
{theme_context}
{plot_context}
{callback_context}
{pacing_instruction}
{first_story_chapter_instruction}
{retention_hook_instruction}
{landmark_instruction}

Chapter Summary: {chapter['summary']}

Key Events to Dramatize:
{chr(10).join(f"• {event}" for event in chapter.get('key_events', []))}

Generate EXACTLY {chapter_scene_budget} scenes that FLOW CONTINUOUSLY.

{"HOOK CHAPTER - ONE THREAD, DEPTH OVER BREADTH:" if is_hook_chapter else "STORYTELLING - this is a STORY, not a timeline:"}
{f'''{prompt_builders.get_hook_content_guidance()}

ONE NARRATIVE THREAD: All scenes in this chapter should build toward the same central question or tension. Do not hop between unrelated topics.

DEPTH OVER BREADTH: Give each moment room to breathe. 2-4 sentences with context—who, where, why it matters. Avoid one-sentence fact bullets.

STORYTELLING, NOT PREVIEW: Write as if the events are happening. Avoid meta phrases like "You'll discover" or "Wait until you hear." Draw viewers in by showing the moment, not by promising it.

CONNECTED SCENES: Each scene should flow from the previous. Reference what came before. Build toward the question, don't scatter random facts.

STRUCTURE:
- SCENE 1 (COLD OPEN): CRITICAL - QUESTION BEFORE ITS FACTS. The video_question must come BEFORE any facts that help answer it. Format: [Question] → [facts that answer it]. The question can unfold across scenes. Example: "How did Theodore Roosevelt come to be the way he was? The answer begins in Milwaukee, 1912, when an assassin fires point-blank into his chest." Never put facts that answer the question before the question.
- SCENES 2-4: Build the SAME thread. Pick 3-5 moments that all relate to the SAME question/tension. Give each moment context—who, where, why it matters. Chronological order if it serves the thread, or stay in one period for tighter focus.
- FINAL SCENE: Natural bridge to the chronological story. DO NOT say "now the story rewinds" or "let's go back" or "rewind to the beginning" - that's awkward. Instead, use something like: "But it all started when...", "It begins with...", or simply start the chronological narrative: "[Birth year/early life context]. This is where our story begins."
- Tone: Exciting, intriguing. Substantive storytelling—show the moment, don't tease it.

WHY/WHAT PARADIGM FOR HOOK CHAPTER:
- This chapter should be MOSTLY WHY sections since it creates interest
- WHY sections: pose the question, create curiosity, build anticipation
- Use WHAT sections sparingly - only to provide brief context
- The goal is to hook viewers and make them want to watch the full story''' if is_hook_chapter else '''THREE PRINCIPLES (CRITICAL - every scene must satisfy all three):
1. The viewer must know WHAT IS HAPPENING - establish the situation, who is involved, where we are. Never leave viewers confused.
2. The viewer must know WHY IT IS IMPORTANT - why this moment matters, its significance, impact. Make the stakes clear.
3. The viewer must know WHAT COULD GO WRONG - what's at risk, what failure would mean. Show what can go wrong (or right) so the moment has weight.

CRITICAL STORYTELLING REQUIREMENTS:
- This MUST feel like a STORY being told, not a chronological list of events
- CONTEXT AND DEPTH (CRITICAL): Viewers must never be lost. Every scene should give enough context that someone who didn't see the previous scene still understands what's happening. For each scene:
  * ESTABLISH THE SITUATION first: Who is involved? Where are we? What's the moment (e.g. "By 1754, the French and British are fighting over the Ohio Valley—and a young colonial officer is about to make a fatal mistake.")?
  * GROUND EVENTS in cause and consequence: What led to this? What's at stake? What does this change?
  * WHEN INTRODUCING a place, person, or term (Fort Necessity, General Braddock, the Newburgh Conspiracy, Valley Forge, etc.), add one phrase of context so a general audience gets it—e.g. "at the muddy outpost where the French had him surrounded" or "his officers, unpaid and furious with Congress, are now plotting to make him King."
  * AVOID name-dropping without context. If you mention a battle, a document, or a person, briefly establish why they matter.
  * CONSEQUENCES: After major events, briefly note what this changes or what would have happened otherwise—give the moment weight.
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

TRANSITIONS AND FLOW - SEAMLESS JOURNEY (CRITICAL):
- Create a SEAMLESS JOURNEY through the video - scenes should feel CONNECTED, not like consecutive pieces of disjoint information (A and B and C)
- Scene 1 should TRANSITION smoothly from where the previous scene ended AND advance the story
- Each scene should BUILD ON the previous scene - reference what came before naturally, show how events connect, demonstrate cause-and-effect
- Each scene should CONNECT to the next through plot progression - what happens next in the story?
- Reference recurring themes/motifs from earlier in the documentary through plot connections
- Avoid feeling like a list of facts - instead, create a flowing narrative where each scene grows from the last
- Use WHY/WHAT interleaving to create natural connections - WHY scenes set up questions that WHAT scenes answer, creating seamless flow
- The final scene should SET UP what comes next by advancing or introducing plot threads
- Think of scenes as story beats in a film, not separate fact segments
- The goal: When scenes are strung together, they should feel like one continuous, connected story, not separate disconnected pieces'''
}

{"" if is_hook_chapter else build_scripts_utils.get_shared_scene_flow_instructions()}

{build_scripts_utils.get_shared_scene_requirements("historical")}

9. PLANT SEEDS (Early in the story): If this is early in the documentary, include specific details, objects, relationships, or concepts that could pay off later. Examples: a specific notebook mentioned, a relationship that will matter later, a fear or promise that will be relevant, a small detail that seems unimportant now but will become significant. These create satisfying "aha moments" when referenced later.

{build_scripts_utils.get_shared_narration_style(is_short=False, script_type="biopic")}
{("- HOOK/INTRO: Write as if events are happening. Avoid meta phrases like \"You'll discover\" or \"Wait until you hear.\" Show the moment, don't promise it. Substantive storytelling.\n" +
"- TRANSITION TO STORY: The final scene should naturally bridge to the chronological narrative without saying \"rewind\" or \"go back\". Simply start with the beginning context: \"[Early life context]. This is where our story begins.\" or \"It all started when...\"") if is_hook_chapter else ""}

{("HOOK CHAPTER EXAMPLES:\n" +
"BAD (meta/preview): \"You'll discover how Einstein changed physics forever. Wait until you hear about 1905.\"\n" +
"GOOD (storytelling): \"Welcome to Human Footprints. Today we'll talk about Albert Einstein. In 1905, a 26-year-old patent clerk in Bern sits at his desk. What he's about to publish will redefine physics. Four papers. One miraculous year.\"\n" +
"GOOD (one thread, depth): \"The letter arrives in June 1858. Darwin's hands tremble as he reads his own theory in another man's words. How did it come to this? The answer stretches back twenty years.\"\n" +
"BAD transition: \"Now the story rewinds to the beginning...\"\n" +
"GOOD transition: \"It all started in Ulm, Germany, in 1879, when Einstein was born.\"") if is_hook_chapter else build_scripts_utils.get_shared_examples("historical")}

{prompt_builders.get_image_prompt_guidelines(person, birth_year, death_year, "16:9 cinematic", is_trailer=False, recurring_themes=threads_str)}

[
  {{
    "id": {start_id},
    "title": "Evocative 2-5 word title",
    "narration": "Vivid, dramatic narration. MUST satisfy the three principles: (1) viewer knows WHAT IS HAPPENING, (2) viewer knows WHY IT IS IMPORTANT, (3) viewer knows WHAT COULD GO WRONG.",
    "scene_type": "WHY" or "WHAT" - MUST be one of these two values. WHY sections frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered or don't understand the significance of. CRITICAL: Every scene (WHY or WHAT) must satisfy the three principles: what is happening, why it's important, what could go wrong. WHY sections should set up what will happen next, why it matters, and what the stakes are for upcoming WHAT sections. WHAT sections deliver core content/solutions/information and must clearly communicate what is happening, why it's important, and what the stakes are (what can go wrong, what can go right, what's at risk).",
    "image_prompt": "Detailed visual description including {person}'s age and age-relevant appearance details, 16:9 cinematic. REQUIRED.",
    "emotion": "A single word or short phrase describing the scene's emotional tone (e.g., 'tense', 'triumphant', 'desperate', 'contemplative', 'exhilarating', 'somber', 'urgent', 'defiant'). Should match the chapter's emotional tone but be scene-specific based on what the character is feeling and the dramatic tension. CRITICAL: Emotions must flow SMOOTHLY between scenes - only change gradually from the previous scene's emotion. Build intensity gradually: 'contemplative' → 'thoughtful' → 'somber' → 'serious' → 'tense'. Avoid dramatic jumps like 'calm' → 'urgent' or 'contemplative' → 'exhilarating'.",
    "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field. Example: 'Focus on tension.' or 'Focus on contemplation.' Keep it simple - just the emotion to emphasize.",
    "year": YYYY or "YYYY-YYYY" or "around YYYY" (the specific year or year range when this scene takes place),
    "kenburns_pattern": "One of: {', '.join(KENBURNS_PATTERNS)}. Choose based on emotional tone: {get_pattern_prompt_str()}. CRITICAL: NEVER repeat the same pattern in consecutive scenes.",
    "kenburns_intensity": "One of: {get_intensity_prompt_str()}. subtle for contemplative/calm; medium for balanced; pronounced for tense/dramatic/climactic."
  }},
  ...
]"""

    scene_system = f"""You are a YouTuber creating documentary content. Write narration from YOUR perspective - this is YOUR script that YOU wrote. Tell the story naturally, directly to the viewer. CRITICAL: Use third person narration - you are speaking ABOUT the main character (he/she/they), not AS the main character. The narrator tells the story of the person, not in their voice.

{prompt_builders.get_biopic_audience_profile()}

THREE PRINCIPLES (every scene must satisfy all three): (1) The viewer must know WHAT IS HAPPENING - establish the situation, who is involved, where we are. (2) The viewer must know WHY IT IS IMPORTANT - why this moment matters, its significance. (3) The viewer must know WHAT COULD GO WRONG - what's at risk, what failure would mean. Check every scene against these three. CRITICAL - SIMPLE, STRAIGHTFORWARD NARRATION: Let facts speak for themselves. Avoid sensationalist language, gimmicky rhetorical hooks, or repeated dramatic phrases (no 'shocking', 'incredible', 'what nobody expected', 'the secret that would change everything'). Prefer direct, clear statements. The significance of events is in the facts—don't oversell with hype. CRITICAL - DEPTH AND CONTEXT: Viewers need enough context to understand every scene. Establish the situation before the event: who is involved, where we are, why this moment is happening. When you introduce a place, person, or concept (e.g. a battle, a document, a political crisis), give one phrase of context so a general viewer isn't lost. Avoid name-dropping without explanation. After major events, briefly note what it changes or what would have happened otherwise. Use 2-4 sentences per scene when context demands it—prioritize clarity and depth over brevity. CRITICAL FOR RETENTION - AVOID VIEWER CONFUSION: The biggest issue for retention is viewer confusion. WHY sections MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions. Don't create confusion by being vague about what's happening. CRITICAL: Create a SEAMLESS JOURNEY through the video - scenes should feel CONNECTED, not like consecutive pieces of disjoint information (A and B and C). Each scene should build on the previous scene, reference what came before naturally, and show how events connect. CRITICAL: Every scene must be classified as WHY or WHAT. WHY sections frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered or don't understand the significance of. WHY sections MUST clearly establish what is happening (MOST IMPORTANT for retention) before introducing questions or mysteries. WHY sections should set up what will happen next, why it matters, and what the stakes are for upcoming WHAT sections. WHAT sections deliver core content, solutions, and information. Every WHAT scene must clearly communicate: what is happening, why it's important, and what the stakes are (what can go wrong, what can go right, what's at risk). Interleave WHY and WHAT sections strategically - WHY sections should create anticipation for upcoming WHAT sections by establishing what/why/stakes, but NEVER at the expense of clarity about what's happening. Use WHY/WHAT interleaving to create natural connections. For hook chapters, use mostly WHY sections. CRITICAL: For each scene, provide narration_instructions as ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on contemplation.' The narration_instructions should flow smoothly between scenes - if previous scene was 'Focus on tension', next might be 'Focus on concern' (gradual progression). Avoid overly dramatic language. Avoid any meta references to chapters, production elements, or the script structure. Focus on what actually happened, why it mattered, and how it felt. CAMERA MOTION: Each scene must include kenburns_pattern and kenburns_intensity. kenburns_pattern: Choose from {get_pattern_prompt_str()}. Match the pattern to the scene's emotional tone. CRITICAL: NEVER use the same kenburns_pattern in two consecutive scenes. kenburns_intensity: Choose from {get_intensity_prompt_str()}. Use subtle for contemplative/calm; medium for balanced; pronounced for tense/dramatic/climactic moments."""
    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": scene_system},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.85,
        response_json_schema=SCENES_ARRAY_SCHEMA,
        provider=llm_utils.get_provider_for_step("SCENES"),
    )
    scenes = json.loads(clean_json_response(content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    # Validate that each scene has required fields and add chapter_num for music track alignment
    for i, scene in enumerate(scenes):
        if 'year' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'year' field")
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
        if 'scene_type' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'scene_type' field")
        if scene.get('scene_type') not in ['WHY', 'WHAT']:
            raise ValueError(f"Scene {i+1} has invalid 'scene_type' value: {scene.get('scene_type')}. Must be 'WHY' or 'WHAT'")
        scene['chapter_num'] = chapter_num
    return scenes


# identify_pivotal_moments, generate_significance_scene, generate_refinement_diff, and refine_scenes
# are now imported from build_scripts_utils above (see imports section at the top)


def generate_short_outline(person: str, outline: dict, short_num: int = 1, total_shorts: int = 3, birth_year: int = None, death_year: int = None, previously_used_topics: list[str] | None = None, previously_used_structures: list[str] | None = None, research_context=None) -> dict:
    """Generate a trailer outline for a YouTube Short. LLM picks the best topic from the full documentary outline."""
    import prompt_builders

    # Discover available music moods for LLM to pick (shorts get 1 song, LLM picks mood)
    try:
        from biopic_music_config import get_available_moods
        available_moods = get_available_moods()
    except ImportError:
        available_moods = ["relaxing", "passionate", "happy"]

    outline_prompt = prompt_builders.build_short_outline_prompt(
        person, outline, short_num, total_shorts,
        available_moods=available_moods,
        previously_used_topics=previously_used_topics,
        previously_used_structures=previously_used_structures,
        research_context=research_context,
    )

    short_system = f"You create high-energy viral trailers. Every word must grab attention and create curiosity. Focus on making viewers NEED to watch the full video. {prompt_builders.get_biopic_audience_profile()} CRITICAL: For each scene, provide narration_instructions as ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on urgency.' The narration_instructions should flow smoothly between scenes - if previous scene was 'Focus on tension', next might be 'Focus on intensity' (gradual progression). Avoid overly dramatic language."
    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": short_system},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.9,
        response_json_schema=SHORT_OUTLINE_SCHEMA,
        provider=llm_utils.get_provider_for_step("OUTLINE"),
    )
    data = json.loads(clean_json_response(content))
    # LLM sometimes returns a list instead of an object; normalize to dict
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        data = data[0]
    elif not isinstance(data, dict):
        raise ValueError(f"Short outline must be a JSON object, got {type(data).__name__}")
    # Fallback music_mood for legacy or incomplete LLM output
    if not data.get("music_mood") or not isinstance(data.get("music_mood"), str):
        data["music_mood"] = available_moods[0] if available_moods else "passionate"
    # Fallback narrative_structure for legacy outlines
    valid_structures = ("question_first", "in_medias_res", "outcome_first", "twist_structure", "chronological_story")
    if data.get("narrative_structure") not in valid_structures:
        data["narrative_structure"] = "question_first"
    return data


def generate_short_scenes(person: str, short_outline: dict, birth_year: int | None = None, death_year: int | None = None, research_context=None) -> list[dict]:
    """Generate 4 scenes for a YouTube Short. Structure varies by narrative_structure (question_first, in_medias_res, etc.).
    
    Each scene must include a "year" field indicating when it takes place.
    """
    import prompt_builders
    
    key_facts = short_outline.get('key_facts', [])
    facts_str = "\n".join(f"• {fact}" for fact in key_facts) if key_facts else "Use the hook expansion to create curiosity"
    hook_expansion = short_outline.get('hook_expansion', '')
    video_question = short_outline.get('video_question', '')
    narrative_structure = short_outline.get('narrative_structure', 'question_first')
    valid_structures = ("question_first", "in_medias_res", "outcome_first", "twist_structure", "chronological_story")
    if narrative_structure not in valid_structures:
        narrative_structure = "question_first"
    
    from research_utils import get_research_context_block
    research_block = get_research_context_block(research_context) if research_context else ""
    
    # Structure-specific blocks: only enforce "Scene 1 MUST start with video_question" for question_first
    if narrative_structure == "question_first":
        question_block = f"""VIDEO QUESTION (CRITICAL - Scene 1 MUST start with this): "{video_question}"

CRITICAL - SCENE 1 MUST START WITH THE VIDEO QUESTION: Scene 1's narration MUST begin with the video_question above. Format: "[The question]? [Then expand the hook with context.]" NEVER open with facts or context first—the question comes FIRST."""
        structure_block = prompt_builders.get_trailer_structure_prompt()
        scene1_hint = 'CRITICAL: The FIRST words of scene 1 MUST be the video_question. Format: \\"[video_question]? [Then 2-3 sentences of context.]\\"'
        scene2_hint = "Build the SAME mystery - deepen the same thread. 2-3 sentences with context. Escalate curiosity."
        scene3_hint = "Deepen the SAME mystery. Build to a climax and END WITH A CLEAR QUESTION that scene 4 will answer."
        scene4_hint = "ANSWER the question from scene 3. WHAT scene - deliver the payoff. No CTA to watch full documentary."
    else:
        question_block = f"""CENTRAL QUESTION (answered by scene 4): "{video_question}"

STRUCTURE: {narrative_structure}. Follow the scene-by-scene flow below. Do NOT force the question into scene 1—each structure has a different opening."""
        structure_block = prompt_builders.get_short_structure_prompt(narrative_structure, video_question)
        scene1_hint = "Follow the structure's Scene 1 instructions above. High energy, immersive."
        scene2_hint = "Follow the structure's Scene 2 instructions. Build on scene 1."
        scene3_hint = "Follow the structure's Scene 3 instructions. Set up the payoff."
        scene4_hint = "Follow the structure's Scene 4 instructions. Payoff/resolution. No CTA to watch full documentary."
    
    scene_prompt = f"""Write 4 scenes for a YouTube Short about {person}.
{research_block}
TITLE: "{short_outline.get('short_title', '')}"
{question_block}
HOOK EXPANSION: {hook_expansion}

KEY FACTS TO USE:
{facts_str}

{prompt_builders.get_biopic_audience_profile()}

ONE THREAD: Scenes should advance the same story. Do not introduce new unrelated facts each scene.

DEPTH: Give key facts room to breathe. 2-3 sentences per scene with context. Avoid one-sentence bullets.

Be substantive and clear—viewers should understand what's happening, not just feel teased. Do NOT end scene 4 with a CTA to watch the full documentary.

{prompt_builders.get_hook_content_guidance()}

{prompt_builders.get_why_what_paradigm_prompt(is_trailer=True)}

{prompt_builders.get_emotion_generation_prompt()}

{structure_block}

{prompt_builders.get_trailer_narration_style()}

{prompt_builders.get_image_prompt_guidelines(person, birth_year, death_year, "9:16 vertical", is_trailer=True)}

[
  {{"id": 1, "title": "2-4 words", "narration": "{scene1_hint}", "scene_type": "WHY", "image_prompt": "Dramatic, attention-grabbing visual including {person}'s age at this time ({birth_year if birth_year else 'unknown'}), 9:16 vertical", "emotion": "High energy emotion. CRITICAL: Emotions must flow SMOOTHLY between scenes - only change gradually.", "narration_instructions": "ONE SENTENCE: Focus on a single emotion. Example: 'Focus on tension.'", "year": "YYYY or YYYY-YYYY", "kenburns_pattern": "{', '.join(KENBURNS_PATTERNS)}", "kenburns_intensity": "{get_intensity_prompt_str()}"}},
  {{"id": 2, "title": "...", "narration": "{scene2_hint}", "scene_type": "WHY", "image_prompt": "Dramatic visual including {person}'s age at this time, 9:16 vertical", "emotion": "High energy - GRADUAL progression from scene 1.", "narration_instructions": "ONE SENTENCE: Focus on emotion, flowing from scene 1.", "year": "YYYY or YYYY-YYYY", "kenburns_pattern": "MUST differ from scene 1", "kenburns_intensity": "subtle/medium/pronounced"}},
  {{"id": 3, "title": "...", "narration": "{scene3_hint}", "scene_type": "WHY", "image_prompt": "Dramatic visual including {person}'s age at this time, 9:16 vertical", "emotion": "High energy - GRADUAL progression from scene 2.", "narration_instructions": "ONE SENTENCE: Focus on emotion, flowing from scene 2.", "year": "YYYY or YYYY-YYYY", "kenburns_pattern": "MUST differ from scene 2", "kenburns_intensity": "subtle/medium/pronounced"}},
  {{"id": 4, "title": "...", "narration": "{scene4_hint}", "scene_type": "WHAT", "image_prompt": "Dramatic visual showing resolution including {person}'s age, 9:16 vertical", "emotion": "Satisfying resolution - flow from scene 3.", "narration_instructions": "ONE SENTENCE: Focus on payoff/revelation.", "year": "YYYY or YYYY-YYYY", "kenburns_pattern": "MUST differ from scene 3", "kenburns_intensity": "subtle/medium/pronounced"}}
]

IMPORTANT: 
- Scene 4: scene_type MUST be "WHAT"; it delivers the payoff/resolution.
- "emotion" should flow smoothly across all 4 scenes.
- "image_prompt" MUST include {person}'s age at that time.
- "year" field indicating when the scene takes place.
- "kenburns_pattern" MUST differ between consecutive scenes. Choose from: {', '.join(KENBURNS_PATTERNS)}.
- "kenburns_intensity": One of subtle, medium, pronounced. Use pronounced for high-energy scenes; medium for balanced; subtle for reflective moments.
]"""

    system_base = f"You create high-energy viral shorts. Write narration from YOUR perspective - this is YOUR script. Use third person narration - you are speaking ABOUT the main character (he/she/they), not AS the main character. {prompt_builders.get_biopic_audience_profile()}"
    if narrative_structure == "question_first":
        system_base += " CRITICAL: Scenes 1-3 are WHY scenes (build the hook, end scene 3 with a clear QUESTION). Scene 4 is a WHAT scene that ANSWERS that question (payoff)."
    else:
        system_base += f" Structure: {narrative_structure}. Follow the structure instructions in the prompt—each structure has a different flow. Scene 4 delivers the payoff/resolution."
    system_base += " NEVER end scene 4 with a CTA to watch the full documentary—it hurts view duration; end with a satisfying resolution only. Ensure the viewer knows WHAT IS HAPPENING - provide clear context. High energy; scene 4 delivers satisfying resolution. Simple, clear, punchy language. CRITICAL: For each scene, provide narration_instructions as ONE SENTENCE focusing on a single emotion. Keep it simple: 'Focus on [emotion].' Emotions should flow smoothly between scenes. Avoid overly dramatic language. CAMERA MOTION: Each scene must include kenburns_pattern and kenburns_intensity. kenburns_pattern: NEVER repeat the same pattern in consecutive scenes. kenburns_intensity: subtle (calm/reflective), medium (balanced), pronounced (tense/dramatic)."
    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": system_base},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.9,
        response_json_schema=SCENES_ARRAY_SCHEMA,
        provider=llm_utils.get_provider_for_step("SCENES"),
    )
    scenes = json.loads(clean_json_response(content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    # Validate that each scene has required fields and add chapter_num for music track alignment
    # Shorts are single-topic, so all scenes get chapter_num=1
    for i, scene in enumerate(scenes):
        if 'year' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'year' field")
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
        if 'scene_type' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'scene_type' field")
        if scene.get('scene_type') not in ['WHY', 'WHAT']:
            raise ValueError(f"Scene {i+1} has invalid 'scene_type' value: {scene.get('scene_type')}. Must be 'WHY' or 'WHAT'")
        scene['chapter_num'] = 1
    return scenes


def generate_shorts(person_of_interest: str, main_title: str, global_block: str, outline: dict, base_output_path: str, research_context=None):
    """Generate YouTube Shorts (4 scenes each). LLM picks the best topic from the full documentary outline."""
    if config.num_shorts == 0:
        print("\n[SHORTS] Skipped (--shorts 0)")
        return []
    
    print(f"\n{'='*60}")
    print(f"[SHORTS] Generating {config.num_shorts} YouTube Short(s)")
    print(f"[SHORTS] Structure: {config.total_short_scenes} scenes each (LLM picks best topic from full outline)")
    print(f"{'='*60}")
    
    SHORTS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    
    generated_shorts = []
    base_name = Path(base_output_path).stem.replace("_script", "")
    
    # STEP 1: Generate all outlines sequentially (LLM picks from full outline, ensuring no topic repeat)
    print(f"\n[OPTIMIZATION] Step 1: Generating all short outlines from full documentary outline...")
    short_outlines = []
    previously_used_topics = []  # Track topics to prevent repeats across shorts
    previously_used_structures = []  # Track structures for variety
    for short_num in range(1, config.num_shorts + 1):
        print(f"\n[SHORT {short_num}/{config.num_shorts}] Picking best topic from full outline...")
        if previously_used_topics:
            print(f"[SHORTS]   • Avoiding topics already used: {', '.join(previously_used_topics[:3])}{'...' if len(previously_used_topics) > 3 else ''}")
        
        short_outline = generate_short_outline(
            person=person_of_interest,
            outline=outline,
            short_num=short_num,
            total_shorts=config.num_shorts,
            birth_year=outline.get('birth_year'),
            death_year=outline.get('death_year'),
            previously_used_topics=previously_used_topics,
            previously_used_structures=previously_used_structures,
            research_context=research_context,
        )
        
        # Add this short's topic to the exclusion list so next shorts pick different topics
        short_title = short_outline.get('short_title', '')
        hook_expansion = short_outline.get('hook_expansion', '')
        if short_title:
            previously_used_topics.append(short_title)
        # Track narrative_structure for variety in subsequent shorts
        ns = short_outline.get('narrative_structure', 'question_first')
        if ns and ns not in previously_used_structures:
            previously_used_structures.append(ns)
        # Also add key topic from hook_expansion (e.g. "Verrocchio/Baptism of Christ") to catch semantic overlap
        if hook_expansion:
            # Extract a concise topic: first 60 chars often captures the main subject
            topic_snippet = hook_expansion[:60].strip() + ("..." if len(hook_expansion) > 60 else "")
            if topic_snippet and topic_snippet not in previously_used_topics:
                previously_used_topics.append(topic_snippet)
        
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
                death_year=outline.get('death_year'),
                research_context=research_context,
            )
            print(f"[SHORT {short_num}] → {len(all_scenes)} scenes generated")
            
            # Fix scene IDs
            for i, scene in enumerate(all_scenes):
                scene["id"] = i + 1
            
            # Refine short scenes (PASS 3 + PASS 4 music selection for shorts)
            short_video_question = short_outline.get('video_question', '')
            short_narrative_structure = short_outline.get('narrative_structure', 'question_first')
            short_music_mood = short_outline.get('music_mood', '')
            short_context = f"Trailer: {short_outline.get('short_title', '')}"
            if short_narrative_structure == "question_first" and short_video_question:
                short_context += f". CRITICAL: Scene 1's narration MUST begin with the video_question—the question comes FIRST, before any facts. Video question: \"{short_video_question}\""
            # Determine short file path first for diff path
            short_file = SHORTS_DIR / f"{base_name}_short{short_num}.json"
            diff_path = short_file.parent / f"{short_file.stem}_refinement_diff.json" if config.generate_refinement_diffs else None
            all_scenes, refinement_diff = build_scripts_utils.refine_scenes(all_scenes, person_of_interest, is_short=True, chapter_context=short_context, diff_output_path=diff_path, subject_type="person", skip_significance_scenes=True, scenes_per_chapter=None, short_video_question=short_video_question if short_narrative_structure == "question_first" else None, short_narrative_structure=short_narrative_structure, short_music_mood=short_music_mood)
            
            # Generate thumbnail (if enabled) - use short-specific prompt + audience targeting
            thumbnail_path = None
            if config.generate_short_thumbnails:
                import prompt_builders
                short_thumb = short_outline.get("thumbnail_prompt", "")
                short_why_type = short_outline.get("thumbnail_why_type")
                short_thumb_text = short_outline.get("thumbnail_text")
                text_instr = _thumbnail_text_instruction(short_why_type, short_thumb_text)
                thumbnail_prompt = f"""{short_thumb}{text_instr}

{prompt_builders.get_thumbnail_prompt_why_scene("9:16")}

{prompt_builders.get_thumbnail_audience_targeting()}"""
                thumb_file = THUMBNAILS_DIR / f"{base_name}_short{short_num}_thumbnail.jpg"
                thumbnail_path = build_scripts_utils.generate_thumbnail(thumbnail_prompt, thumb_file, "1024x1536", config.generate_short_thumbnails)
            
            # Build short output
            short_title = short_outline.get("short_title", f"Short {short_num}")
            short_output = {
                "metadata": {
                    "short_id": short_num,
                    "title": short_title,
                    "description": short_outline.get("short_description", ""),
                    "tags": short_outline.get("tags", ""),
                    "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
                    "video_question": short_outline.get("video_question", ""),
                    "hook_expansion": short_outline.get("hook_expansion", ""),
                    "person_of_interest": person_of_interest,
                    "main_video_title": main_title,
                    "global_block": global_block,
                    "num_scenes": len(all_scenes),
                    "outline": short_outline
                },
                "scenes": all_scenes
            }

            build_scripts_utils.validate_biopic_scenes(all_scenes, script_path_hint=str(short_file))

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


def _save_biopic_script(
    output_path: str,
    *,
    title: str,
    tag_line: str,
    video_description: str,
    tags: str,
    pinned_comment: str,
    thumbnail_description: str,
    generated_thumb: Path | None,
    global_block: str,
    person_of_interest: str,
    outline: dict,
    all_scenes: list,
    shorts_info: list,
    thumbnail_why_type: str | None = None,
    thumbnail_text: str | None = None,
    thumbnail_options: list | None = None,
) -> None:
    """Write current script state to JSON so progress is saved if a later step fails."""
    metadata = {
        "video_questions": outline.get("video_questions", []),
        "title": title,
        "tag_line": tag_line,
        "video_description": video_description,
        "tags": tags,
        "pinned_comment": pinned_comment,
        "thumbnail_description": thumbnail_description,
        "thumbnail_path": str(generated_thumb) if generated_thumb else None,
        "global_block": global_block,
        "person_of_interest": person_of_interest,
        "script_type": "biopic",
        "num_scenes": len(all_scenes),
        "outline": outline,
        "shorts": shorts_info,
    }
    if thumbnail_why_type is not None:
        metadata["thumbnail_why_type"] = thumbnail_why_type
    if thumbnail_text is not None:
        metadata["thumbnail_text"] = thumbnail_text
    if thumbnail_options is not None:
        metadata["thumbnail_options"] = thumbnail_options
    output_data = {
        "metadata": metadata,
        "scenes": all_scenes,
    }
    # Validate when scenes have music (after refinement); skip for intermediate saves
    if all(s.get("music_song") and s.get("music_volume") for s in all_scenes):
        build_scripts_utils.validate_biopic_scenes(all_scenes, script_path_hint=output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"[SCRIPT] Saved progress: {output_path} ({len(all_scenes)} scenes)")


def generate_script(person_of_interest: str, output_path: str):
    """Generate a complete documentary script using outline-guided generation."""
    import script_types
    
    script_type_instance = script_types.MainVideoScript()
    
    print(f"\n{'='*60}")
    print(f"[SCRIPT] Generating script for: {person_of_interest}")
    print(f"{'='*60}")
    
    if config.generate_main:
        print(f"[CONFIG] Main video: {config.chapters} chapters, target ~{config.target_total_scenes} scenes (variable per chapter)")
    else:
        print(f"[CONFIG] Main video: SKIPPED")
    if config.num_shorts > 0:
        print(f"[CONFIG] Shorts: {config.num_shorts} × {config.total_short_scenes} scenes = {config.num_shorts * config.total_short_scenes} scenes")
    else:
        print(f"[CONFIG] Shorts: SKIPPED")
    print(f"[CONFIG] Thumbnails: {'Yes' if config.generate_thumbnails else 'No'}")
    print(f"[CONFIG] Model: {llm_utils.get_text_model_display()}")
    # Log per-step providers when any override is set
    steps = ["VIDEO_QUESTIONS", "LANDMARKS", "OUTLINE", "INITIAL_METADATA", "GLOBAL_BLOCK", "SCENES",
             "REFINEMENT", "REFINEMENT_HANGING_STORYLINES", "REFINEMENT_HISTORIAN_DEPTH",
             "REFINEMENT_FILM_COMPOSER", "REFINEMENT_TRANSITIONS"]
    overrides = {s: llm_utils.get_provider_for_step(s) for s in steps}
    if any(overrides[s] != llm_utils.TEXT_PROVIDER.lower() for s in steps):
        print(f"[CONFIG] LLM per-step: {', '.join(f'{s.lower()}={overrides[s]}' for s in steps)}")
    print(f"[CONFIG] Research: {'Yes' if config.use_research else 'No (--no-research)'}")
    print(f"[CONFIG] Chapter transitions: {'Yes' if config.chapter_transitions else 'No (use --chapter-transitions to enable)'}")
    
    # Research step: fetch Wikipedia (and optionally other sources) for prompt injection
    research_context = None
    if config.use_research:
        import research_utils
        print("\n[RESEARCH] Fetching Wikipedia research...")
        research_context = research_utils.fetch_research(person_of_interest)
        if research_context.is_empty():
            print("[RESEARCH] No research found, continuing with LLM-only")
        else:
            print(f"[RESEARCH] Loaded {len(research_context.summary)} chars from Wikipedia")
    
    # Step 0: Generate video questions FIRST - these frame everything else
    video_questions = generate_video_questions(person_of_interest, research_context=research_context)

    # Step 1a: Generate landmark events (for adding detail when we cover these events—does NOT change outline structure)
    landmarks = generate_landmark_events(person_of_interest, research_context=research_context)

    # Step 1b: Generate detailed outline (framed by video_questions, needed for both main and shorts)
    print("\n[STEP 1] Creating life outline...")
    outline = generate_outline(person_of_interest, landmark_events=landmarks, video_questions=video_questions, research_context=research_context)
    chapters = outline.get("chapters", [])
    
    if config.generate_main and len(chapters) < config.chapters:
        print(f"[WARNING] Got {len(chapters)} chapters, expected {config.chapters}")
    
    # Step 2a: Generate initial metadata (3 title+thumbnail pairs; no global_block)
    print("\n[STEP 2] Generating initial metadata (3 title+thumbnail options)...")
    
    import prompt_builders
    initial_metadata_prompt = prompt_builders.build_metadata_prompt_3_options_no_global(
        person_of_interest, outline.get('tagline', ''), config.total_scenes,
        video_questions=outline.get("video_questions"),
        research_context=research_context,
    )

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": f"Documentary producer. Create 3 title+thumbnail pairs tailored for our target audience. {prompt_builders.get_biopic_audience_profile()}"},
            {"role": "user", "content": initial_metadata_prompt}
        ],
        temperature=0.7,
        response_json_schema=INITIAL_METADATA_3_OPTIONS_NO_GLOBAL_SCHEMA,
        provider=llm_utils.get_provider_for_step("INITIAL_METADATA"),
    )
    initial_metadata = json.loads(clean_json_response(content))
    
    tag_line = initial_metadata.get("tag_line", f"the story of {person_of_interest}")
    thumbnail_options_raw = initial_metadata.get("thumbnail_options", [])
    
    # Step 2b: Generate global_block separately (visual style guide)
    # Provider: TEXT_PROVIDER_GLOBAL_BLOCK if set, else TEXT_PROVIDER
    title = thumbnail_options_raw[0]["title"] if thumbnail_options_raw else "Untitled"
    global_block_prompt = prompt_builders.build_global_block_prompt(
        person_of_interest, config.total_scenes, outline, tag_line, title
    )
    global_content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "Documentary visual director. Create a detailed visual style guide for consistent imagery across all scenes."},
            {"role": "user", "content": global_block_prompt}
        ],
        temperature=0.6,
        response_json_schema=GLOBAL_BLOCK_SCHEMA,
        provider=llm_utils.get_provider_for_step("GLOBAL_BLOCK"),
    )
    global_block = json.loads(clean_json_response(global_content))["global_block"]
    
    # Use first option's title for rest of script (shorts, etc.); user picks final pair when uploading
    thumbnail_description = thumbnail_options_raw[0].get("thumbnail_description", "") if thumbnail_options_raw else ""
    thumbnail_why_type = thumbnail_options_raw[0].get("thumbnail_why_type") if thumbnail_options_raw else None
    thumbnail_text = thumbnail_options_raw[0].get("thumbnail_text") if thumbnail_options_raw else None
    
    print(f"[METADATA] Title (option 1): {title}")
    print(f"[METADATA] Tag line: {tag_line}")
    
    # Generate 3 thumbnails (one per option) - user picks best when uploading
    thumbnail_options = []
    stem = Path(output_path).stem
    for i, opt in enumerate(thumbnail_options_raw[:3]):
        desc = opt.get("thumbnail_description", "")
        why = opt.get("thumbnail_why_type")
        txt = opt.get("thumbnail_text")
        thumb_path = None
        if config.generate_main and config.generate_thumbnails:
            THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"\n[THUMBNAIL] Generating option {i + 1}/3: {opt.get('title', '')[:50]}...")
            text_instr = _thumbnail_text_instruction(why, txt)
            thumbnail_prompt = f"""{desc}{text_instr}

{prompt_builders.get_thumbnail_prompt_why_scene("16:9")}

{prompt_builders.get_thumbnail_audience_targeting()}"""
            thumb_file = THUMBNAILS_DIR / f"{stem}_thumbnail_{i + 1}.jpg"
            generated = build_scripts_utils.generate_thumbnail(thumbnail_prompt, thumb_file, "1024x1024", config.generate_thumbnails)
            thumb_path = str(thumb_file) if generated else None
        thumbnail_options.append({
            "title": opt.get("title", ""),
            "thumbnail_description": desc,
            "thumbnail_why_type": why,
            "thumbnail_text": txt,
            "thumbnail_path": thumb_path,
        })
    generated_thumb = Path(thumbnail_options[0]["thumbnail_path"]) if thumbnail_options and thumbnail_options[0].get("thumbnail_path") else None
    
    # Step 3: Generate scenes chapter by chapter (only if generating main video)
    all_scenes = []
    planted_seeds = []  # Track details from early chapters for callbacks
    
    if config.generate_main:
        total_planned = sum(ch.get('num_scenes', config.scenes_per_chapter_fallback) for ch in chapters)
        print(f"\n[STEP 3] Generating ~{total_planned} scenes from {len(chapters)} chapters...")
        
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
            
            chapter_scene_budget = chapter.get('num_scenes', config.scenes_per_chapter_fallback)
            print(f"\n[CHAPTER {chapter['chapter_num']}/{len(chapters)}] {chapter['title']}")
            if is_retention_hook:
                print(f"  ⚠ RETENTION HOOK POINT (~{estimated_time}s mark)")
            print(f"  Generating scene outline for {chapter_scene_budget} scenes...")
            scene_blocks = generate_scene_outline(chapter, person_of_interest, chapter_scene_budget, landmarks=landmarks, research_context=research_context)
            print(f"  Generating {chapter_scene_budget} scenes ({len(scene_blocks)} blocks)...")

            try:
                scenes = generate_scenes_for_chapter(
                    person=person_of_interest,
                    chapter=chapter,
                    scene_blocks=scene_blocks,
                    chapter_scene_budget=chapter_scene_budget,
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
                    sub_plots=sub_plots,
                    landmarks=landmarks,
                    video_questions=video_questions if i == 0 else None,  # Only for hook chapter
                    research_context=research_context,
                )
                
                if len(scenes) != chapter_scene_budget:
                    print(f"  [WARNING] Got {len(scenes)} scenes, expected {chapter_scene_budget}")
                
                all_scenes.extend(scenes)
                print(f"  ✓ {len(scenes)} scenes (total: {len(all_scenes)})")
                
                # Fix scene IDs so partial save is valid; save progress in case a later step fails
                for idx, scene in enumerate(all_scenes):
                    scene["id"] = idx + 1
                _save_biopic_script(
                    output_path,
                    title=title,
                    tag_line=tag_line,
                    video_description="",
                    tags="",
                    pinned_comment="",
                    thumbnail_description=thumbnail_description,
                    generated_thumb=generated_thumb,
                    global_block=global_block,
                    person_of_interest=person_of_interest,
                    outline=outline,
                    all_scenes=all_scenes,
                    shorts_info=[],
                    thumbnail_why_type=thumbnail_why_type,
                    thumbnail_text=thumbnail_text,
                    thumbnail_options=thumbnail_options,
                )
                
                # CTA scene generation removed
                
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
                        # Extract specific details that could be callbacks: objects, relationships, discoveries
                        if any(word in event.lower() for word in ['notebook', 'letter', 'relationship', 'conflict', 'discovery', 'secret', 'promise', 'warning', 'fear']):
                            planted_seeds.append(event[:120])
                
            except Exception as e:
                print(f"  [ERROR] Failed: {e}")
                raise
        
        print(f"\n[SCRIPT] Total scenes: {len(all_scenes)}")
        
        # Validate and fix scene IDs
        for i, scene in enumerate(all_scenes):
            scene["id"] = i + 1
            for field in ["title", "narration", "image_prompt", "emotion", "narration_instructions", "year"]:
                if field not in scene:
                    raise ValueError(f"Scene {i+1} missing required field: {field}")
        
        # Step 3.4: Refine scenes for awkward transitions and improvements
        print("\n[STEP 3.4] Refining main video scenes...")
        chapter_summaries = "\n".join([f"Chapter {ch['chapter_num']}: {ch['title']} ({ch.get('year_range', 'Unknown')}) - {ch['summary']}" for ch in chapters])
        diff_path = Path(output_path).parent / f"{Path(output_path).stem}_refinement_diff.json" if config.generate_refinement_diffs else None
        # Build chapter boundaries for variable scene counts (exclude ch1 + transition from pivotal moments)
        chapter_boundaries = []
        cumulative = 0
        for ch in chapters:
            n = ch.get('num_scenes', config.scenes_per_chapter_fallback)
            start = cumulative + 1
            end = cumulative + n
            chapter_boundaries.append((start, end))
            cumulative += n
        all_scenes, refinement_diff = build_scripts_utils.refine_scenes(all_scenes, person_of_interest, is_short=False, chapter_context=chapter_summaries, diff_output_path=diff_path, subject_type="person", skip_significance_scenes=True, chapter_boundaries=chapter_boundaries, script_type="biopic", video_questions=outline.get("video_questions"))

        # Insert chapter transition scenes (after refinement; no transition before ch1 or after final chapter)
        if config.chapter_transitions and len(chapters) >= 2:
            all_scenes = build_scripts_utils.insert_chapter_transition_scenes(
                all_scenes, chapters, person_of_interest, script_type="biopic"
            )
            print(f"[CHAPTER TRANSITIONS] Inserted transitions between {len(chapters)} chapters")

        # Save progress after refinement (metadata and shorts not yet generated)
        _save_biopic_script(
            output_path,
            title=title,
            tag_line=tag_line,
            video_description="",
            tags="",
            pinned_comment="",
            thumbnail_description=thumbnail_description,
            generated_thumb=generated_thumb,
            global_block=global_block,
            person_of_interest=person_of_interest,
            outline=outline,
            all_scenes=all_scenes,
            shorts_info=[],
            thumbnail_options=thumbnail_options,
        )
        
        # Step 3.5: Generate final metadata (description and tags) AFTER scenes are generated
        print("\n[STEP 3.5] Generating final metadata from actual scenes...")
        
        # Extract memorable moments from scenes for metadata
        scene_highlights = []
        for scene in all_scenes[:10]:  # First 10 scenes for highlights
            scene_highlights.append(f"Scene {scene['id']}: {scene.get('title', '')} - {scene.get('narration', '')[:80]}...")
        
        from research_utils import get_research_context_block
        final_research_block = get_research_context_block(research_context) if research_context else ""
        final_metadata_prompt = f"""Create final metadata for a documentary about: {person_of_interest}

Their story in one line: {outline.get('tagline', '')}
{final_research_block}
Actual memorable moments from the documentary (use these to make description more accurate):
{chr(10).join(scene_highlights[:10])}

Generate JSON:
{{
  "video_description": "Brief YouTube description (100-200 words max) - concise summary optimized for SEO. Start with a compelling hook that includes key SEO keywords. Provide a brief overview of the person's story and why it matters. Keep it concise - don't include excessive detail about specific scenes or events. End with: 'If you enjoyed this video, please like and subscribe for more stories like this!'",
  "tags": "15-20 SEO tags separated by commas. Mix of: person's name, key topics, related figures, time periods, achievements, fields (e.g., 'Albert Einstein, physics, relativity, Nobel Prize, Germany, Princeton, E=mc2, quantum mechanics, genius, scientist, 20th century, biography, documentary')",
  "pinned_comment": "An engaging question or comment to pin below the video (1-2 sentences max). Should: spark discussion, ask a thought-provoking question about the person/story, create curiosity, encourage viewers to share their thoughts/opinions, be conversational and inviting. Examples: 'Which moment from their story surprised you most? Drop a comment below!', 'What do you think was their greatest challenge? Share your thoughts!', 'This story shows how one person can change everything. What impact do you want to make?'. Should feel authentic, not clickbaity - genuine curiosity about viewer perspectives."
}}"""

        system_role = f"Documentary producer. Create compelling metadata (description, tags, pinned comment) that accurately reflects the actual content and is tailored for our target audience. {prompt_builders.get_biopic_audience_profile()}"
        content = llm_utils.generate_text(
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": final_metadata_prompt}
            ],
            temperature=0.7,
            response_json_schema=FINAL_METADATA_SCHEMA,
            provider=llm_utils.get_provider_for_step("REFINEMENT"),
        )
        final_metadata = json.loads(clean_json_response(content))
        video_description = final_metadata.get("video_description", "")
        tags = final_metadata.get("tags", "")
        pinned_comment = final_metadata.get("pinned_comment", "")
        
        print(f"[METADATA] Description generated from {len(all_scenes)} scenes")
        print(f"[METADATA] Tags: {tags[:80]}..." if len(tags) > 80 else f"[METADATA] Tags: {tags}")
        if pinned_comment:
            print(f"[METADATA] Pinned comment: {pinned_comment}")
        # Save progress with full metadata before shorts (in case shorts step fails)
        _save_biopic_script(
            output_path,
            title=title,
            tag_line=tag_line,
            video_description=video_description,
            tags=tags,
            pinned_comment=pinned_comment,
            thumbnail_description=thumbnail_description,
            generated_thumb=generated_thumb,
            global_block=global_block,
            person_of_interest=person_of_interest,
            outline=outline,
            all_scenes=all_scenes,
            shorts_info=[],
            thumbnail_why_type=thumbnail_why_type,
            thumbnail_text=thumbnail_text,
            thumbnail_options=thumbnail_options,
        )
    else:
        print("\n[STEP 3] Skipping main video scene generation...")
        # Generate basic metadata if not generating main video
        video_description = ""
        tags = ""
        pinned_comment = ""
    
    # Step 4: Generate Shorts (LLM picks best topic from full outline)
    print("\n[STEP 4] Generating YouTube Shorts...")
    shorts_info = generate_shorts(person_of_interest, title, global_block, outline, output_path, research_context=research_context)
    
    # Step 5: Save
    print("\n[STEP 5] Saving script...")
    
    output_metadata = {
        "video_questions": outline.get("video_questions", []),
        "title": title,
        "tag_line": tag_line,
        "video_description": video_description,
        "tags": tags,
        "pinned_comment": pinned_comment if config.generate_main else "",
        "thumbnail_description": thumbnail_description,
        "thumbnail_path": str(generated_thumb) if generated_thumb else None,
        "thumbnail_options": thumbnail_options,
        "global_block": global_block,
        "person_of_interest": person_of_interest,
        "script_type": "biopic",
        "num_scenes": len(all_scenes),
        "outline": outline,
        "shorts": shorts_info
    }
    if thumbnail_why_type is not None:
        output_metadata["thumbnail_why_type"] = thumbnail_why_type
    if thumbnail_text is not None:
        output_metadata["thumbnail_text"] = thumbnail_text
    output_data = {
        "metadata": output_metadata,
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


def regenerate_thumbnail(script_path: str, include_shorts: bool = False) -> dict:
    """
    Regenerate thumbnail(s) for an existing biopic script.
    Loads the script, generates thumbnail from thumbnail_description, updates and saves.
    
    Args:
        script_path: Path to the script JSON file
        include_shorts: If True, also regenerate thumbnails for shorts listed in metadata.shorts
    
    Returns:
        Updated script data (with new thumbnail_path)
    """
    import prompt_builders
    
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    if not metadata:
        raise ValueError("Script has no metadata")
    
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    
    # Main video: regenerate 3 options if thumbnail_options exists, else single thumbnail
    thumbnail_options = metadata.get("thumbnail_options", [])
    if thumbnail_options:
        print("\n[THUMBNAIL] Regenerating 3 thumbnail options...")
        for i, opt in enumerate(thumbnail_options[:3]):
            desc = opt.get("thumbnail_description", "")
            if not desc:
                print(f"[THUMBNAIL] Option {i + 1} has no description, skipping")
                continue
            print(f"[THUMBNAIL] Regenerating option {i + 1}/3...")
            why = opt.get("thumbnail_why_type")
            txt = opt.get("thumbnail_text")
            text_instr = _thumbnail_text_instruction(why, txt)
            thumbnail_prompt = f"""{desc}{text_instr}

{prompt_builders.get_thumbnail_prompt_why_scene("16:9")}

{prompt_builders.get_thumbnail_audience_targeting()}"""
            thumb_file = THUMBNAILS_DIR / f"{stem}_thumbnail_{i + 1}.jpg"
            generated = build_scripts_utils.generate_thumbnail(
                thumbnail_prompt, thumb_file, "1024x1024", generate_thumbnails=True
            )
            opt["thumbnail_path"] = str(thumb_file) if generated else None
        metadata["thumbnail_path"] = thumbnail_options[0].get("thumbnail_path") if thumbnail_options else None
    else:
        thumbnail_description = metadata.get("thumbnail_description")
        if not thumbnail_description:
            raise ValueError("Script metadata has no thumbnail_description or thumbnail_options - cannot regenerate thumbnail")
        main_why_type = metadata.get("thumbnail_why_type")
        main_thumb_text = metadata.get("thumbnail_text")
        text_instr = _thumbnail_text_instruction(main_why_type, main_thumb_text)
        print("\n[THUMBNAIL] Regenerating main video thumbnail...")
        thumb_file = THUMBNAILS_DIR / f"{stem}_thumbnail.jpg"
        thumbnail_prompt = f"""{thumbnail_description}{text_instr}

{prompt_builders.get_thumbnail_prompt_why_scene("16:9")}

{prompt_builders.get_thumbnail_audience_targeting()}"""
        generated_thumb = build_scripts_utils.generate_thumbnail(
            thumbnail_prompt, thumb_file, "1024x1024", generate_thumbnails=True
        )
        metadata["thumbnail_path"] = str(generated_thumb) if generated_thumb else None
    
    # Short thumbnails (optional)
    if include_shorts:
        shorts_info = metadata.get("shorts", [])
        for short_entry in shorts_info:
            short_file = short_entry.get("file")
            if not short_file:
                continue
            short_path = Path(short_file)
            if not short_path.is_absolute():
                # Try cwd, then project root (parent of scripts dir)
                for base in [Path.cwd(), path.resolve().parent.parent]:
                    candidate = base / short_file
                    if candidate.exists():
                        short_path = candidate
                        break
            if not short_path.exists():
                print(f"[THUMBNAIL] Short not found, skipping: {short_file}")
                continue
            with open(short_path, "r", encoding="utf-8") as f:
                short_data = json.load(f)
            short_metadata = short_data.get("metadata", {})
            short_outline = short_metadata.get("outline", short_data.get("outline", {}))
            short_thumb = short_outline.get("thumbnail_prompt", "")
            if not short_thumb:
                print(f"[THUMBNAIL] Short has no thumbnail_prompt, skipping: {short_file}")
                continue
            short_why_type = short_outline.get("thumbnail_why_type")
            short_thumb_text = short_outline.get("thumbnail_text")
            short_text_instr = _thumbnail_text_instruction(short_why_type, short_thumb_text)
            short_id = short_metadata.get("short_id", short_path.stem.split("_short")[-1].split("_")[0])
            short_stem = short_path.stem
            thumb_prompt = f"""{short_thumb}{short_text_instr}

{prompt_builders.get_thumbnail_prompt_why_scene("9:16")}

{prompt_builders.get_thumbnail_audience_targeting()}"""
            short_thumb_file = THUMBNAILS_DIR / f"{short_stem}_thumbnail.jpg"
            print(f"[THUMBNAIL] Regenerating short {short_id} thumbnail...")
            short_generated = build_scripts_utils.generate_thumbnail(
                thumb_prompt, short_thumb_file, "1024x1536", generate_thumbnails=True
            )
            short_metadata["thumbnail_path"] = str(short_generated) if short_generated else None
            with open(short_path, "w", encoding="utf-8") as f:
                json.dump(short_data, f, indent=2, ensure_ascii=False)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[THUMBNAIL] Saved script: {path}")
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate documentary scripts with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full production run (24 scenes main video, 3 shorts with 3 scenes each)
  python build_script_biopic.py "Albert Einstein"

  # Quick test (4 scenes main, 1 short with 4 scenes, no thumbnails)
  python build_script_biopic.py "Albert Einstein" --test

  # Regenerate thumbnail for existing script
  python build_script_biopic.py --thumbnail-only scripts/leonardo_vinci.json

  # Regenerate main + short thumbnails for existing script
  python build_script_biopic.py --thumbnail-only scripts/leonardo_vinci.json --short-thumbnails

  # Custom main video settings
  python build_script_biopic.py "Albert Einstein" --chapters 5 --scenes 3

  # Custom shorts settings  
  python build_script_biopic.py "Albert Einstein" --shorts 2 --short-scenes 4

  # No thumbnails (faster iteration)
  python build_script_biopic.py "Albert Einstein" --no-thumbnails

  # Include chapter transition scenes (title cards between chapters)
  python build_script_biopic.py "Albert Einstein" --chapter-transitions

  # Only generate shorts (skip main video)
  python build_script_biopic.py "Albert Einstein" --shorts-only

  # Only generate main video (skip shorts)
  python build_script_biopic.py "Albert Einstein" --main-only
        """
    )
    
    parser.add_argument("person", nargs="?", help="Person to create documentary about")
    parser.add_argument("output", nargs="?", help="Output JSON file (default: <person>_script.json)")
    
    # Quick test mode
    parser.add_argument("--test", action="store_true", 
                        help="Quick test: 2 chapters × 2 scenes, 1 short with 2 chapters × 2 scenes, no thumbnails")
    
    # What to generate
    parser.add_argument("--main-only", action="store_true",
                        help="Only generate main video (skip shorts)")
    parser.add_argument("--shorts-only", action="store_true",
                        help="Only generate shorts (skip main video)")
    
    # Main video settings (defaults from config)
    parser.add_argument("--chapters", type=int, default=config.chapters,
                        help=f"Main video outline chapters (default: {config.chapters})")
    parser.add_argument("--target-scenes", "--scenes", dest="target_scenes", type=int, default=config.target_total_scenes,
                        help=f"Target total scenes - outline distributes across chapters (default: {config.target_total_scenes})")
    
    # Shorts settings (defaults from config)
    parser.add_argument("--shorts", type=int, default=config.num_shorts,
                        help=f"Number of YouTube Shorts (default: {config.num_shorts}, use 0 to skip)")
    parser.add_argument("--short-scenes", type=int, default=config.short_scenes_per_chapter,
                        help=f"Scenes per short (default: {config.short_scenes_per_chapter}: build, build, cliffhanger)")
    
    parser.add_argument("--thumbnail-only", metavar="SCRIPT", dest="thumbnail_only_script",
                        help="Regenerate thumbnail for existing script (path to JSON); skips full generation")
    parser.add_argument("--no-thumbnails", action="store_true",
                        help="Skip main video thumbnail generation")
    parser.add_argument("--short-thumbnails", action="store_true",
                        help="Generate thumbnails for shorts (disabled by default); with --thumbnail-only, also regen short thumbnails")
    parser.add_argument("--refinement-diffs", action="store_true",
                        help="Generate refinement diff JSON files showing what changed during scene refinement")
    parser.add_argument("--no-research", action="store_true",
                        help="Skip Wikipedia research fetching (faster, LLM only)")
    parser.add_argument("--research-debug", action="store_true",
                        help="Enable verbose research/Wikipedia API logging (DEBUG=1)")
    parser.add_argument("--chapter-transitions", action="store_true",
                        help="Insert chapter transition scenes (title cards) between chapters")
    
    return parser.parse_args()


# ------------- ENTRY POINT -------------

if __name__ == "__main__":
    args = parse_args()
    
    # Thumbnail-only mode: regenerate thumbnail for existing script
    if args.thumbnail_only_script:
        if not args.person:
            pass  # person not needed
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: Set OPENAI_API_KEY environment variable first.")
            sys.exit(1)
        try:
            script_data = regenerate_thumbnail(
                args.thumbnail_only_script,
                include_shorts=args.short_thumbnails
            )
            print("\n" + "="*60)
            print("THUMBNAIL REGENERATED!")
            print("="*60)
            print(f"   Script: {args.thumbnail_only_script}")
            if script_data.get("metadata", {}).get("thumbnail_path"):
                print(f"   Thumbnail: {script_data['metadata']['thumbnail_path']}")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Normal mode: require person
    if not args.person:
        print("ERROR: Person required (e.g. 'Albert Einstein') or use --thumbnail-only <script.json>")
        sys.exit(1)
    
    # Apply test mode settings
    if args.test:
        config.chapters = 2
        config.target_total_scenes = 4
        config.num_shorts = 1
        config.short_chapters = 1
        config.short_scenes_per_chapter = 4
        config.generate_thumbnails = False
        config.generate_short_thumbnails = False
        config.generate_refinement_diffs = False
        config.use_research = not args.no_research
        config.chapter_transitions = args.chapter_transitions
        print("[MODE] Test mode enabled")
    else:
        config.chapters = args.chapters
        config.target_total_scenes = args.target_scenes
        config.num_shorts = args.shorts
        config.short_scenes_per_chapter = args.short_scenes
        config.generate_thumbnails = not args.no_thumbnails
        config.generate_short_thumbnails = args.short_thumbnails
        config.generate_refinement_diffs = args.refinement_diffs
        config.use_research = not args.no_research
        config.chapter_transitions = args.chapter_transitions
    
    if args.research_debug:
        os.environ["DEBUG"] = "1"
        print("[MODE] Research debug logging enabled")
    
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
