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

load_dotenv()

THUMBNAILS_DIR = Path("thumbnails")
SHORTS_DIR = Path("shorts_scripts")
SCRIPTS_DIR = Path("scripts")


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


def generate_outline(person_of_interest: str) -> dict:
    """Generate a detailed chronological outline of the person's life."""
    print(f"\n[OUTLINE] Generating {config.chapters}-chapter outline...")
    
    # Use prompt builder for outline
    import prompt_builders
    outline_prompt = prompt_builders.build_outline_prompt(person_of_interest, config.chapters, config.total_scenes)

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You are a historian who finds the drama and humanity in every life story. Respond with valid JSON only."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    outline_data = json.loads(clean_json_response(content))
    chapters = outline_data.get("chapters", [])
    
    print(f"[OUTLINE] Generated {len(chapters)} chapters")
    for ch in chapters:
        print(f"  Ch {ch['chapter_num']}: {ch['title']} ({ch['year_range']})")
    
    return outline_data


# CTA scene generation removed - no longer needed

def generate_scenes_for_chapter(person: str, chapter: dict, scenes_per_chapter: int, start_id: int, 
                                 global_style: str, prev_chapter: dict = None, prev_scenes: list = None,
                                 central_theme: str = None, narrative_arc: str = None, 
                                 planted_seeds: list[str] = None, is_retention_hook_point: bool = False,
                                 birth_year: int | None = None, death_year: int | None = None,
                                 tag_line: str | None = None, overarching_plots: list[dict] = None,
                                 sub_plots: list[dict] = None) -> list[dict]:
    """Generate scenes for a single chapter of the outline with continuity context."""
    import prompt_builders
    
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
    
    # Calculate total scenes for prompt
    total_scenes_for_prompt = config.chapters * scenes_per_chapter
    scene_prompt = f"""You are writing scenes {start_id}-{start_id + scenes_per_chapter - 1} of a {total_scenes_for_prompt}-scene documentary about {person}.

{prev_context}
{scenes_context}
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
{retention_hook_instruction}

Chapter Summary: {chapter['summary']}

Key Events to Dramatize:
{chr(10).join(f"• {event}" for event in chapter.get('key_events', []))}

Generate EXACTLY {scenes_per_chapter} scenes that FLOW CONTINUOUSLY.

{"HOOK CHAPTER - INTRODUCTION/PREVIEW (NOT A STORY):" if is_hook_chapter else "STORYTELLING - this is a STORY, not a timeline:"}
{f'''CRITICAL: This is an INTRODUCTION/PREVIEW, not a story. It should quickly answer "Why should I watch this?"

CHRONOLOGICAL ORDER (CRITICAL): Even though this is a preview, scenes MUST be in CHRONOLOGICAL ORDER by year - start with earlier moments and progress chronologically to later ones. This creates a sense of progression and makes the preview flow naturally.

WHY/WHAT PARADIGM FOR HOOK CHAPTER:
- This chapter should be MOSTLY WHY sections since it's a preview that creates interest
- WHY sections should dominate: pose questions, create curiosity, build anticipation
- Use WHAT sections sparingly - only to provide brief context or teasers
- The goal is to hook viewers and make them want to watch the full story

STRUCTURE:
- SCENE 1 (COLD OPEN - FIRST 15 SECONDS): The first sentence MUST begin with: "Welcome to Human Footprints. Today we'll talk about {person} - {tag_line if tag_line else ''}." After this introduction, immediately transition to the MOST shocking, intriguing, or compelling moment or question. The tag_line should be short, catchy, and accurate (e.g., "the man who changed the world", "the codebreaker who saved millions", "the mind that rewrote physics"). For example: "Welcome to Human Footprints. Today we'll talk about Albert Einstein - the man who changed the world. In 1905, a 26-year-old patent clerk publishes a paper that will change everything. But first, he must survive the criticism of his own father." OR "Welcome to Human Footprints. Today we'll talk about Charles Darwin - the man who changed the way we understand life. The letter arrives in June 1858. Darwin's hands tremble as he reads his own theory in another man's words." The first 15 seconds are CRITICAL for YouTube - this is what viewers see in search/preview. Hook them immediately after the introduction.
- SCENES 2-4: Rapid-fire preview of the most shocking, interesting, or impactful moments from their entire life - achievements, controversies, dramatic moments, surprising facts
- Pick the most important and impactful moments to include in the preview
- CRITICAL: Scenes MUST be in CHRONOLOGICAL ORDER by year - start with earlier moments and progress chronologically to later ones, even though this is a preview
- Each scene should hook the viewer: "You'll discover how...", "You'll see the moment when...", "Wait until you hear about..."
- FINAL SCENE: Natural bridge to the chronological story. DO NOT say "now the story rewinds" or "let's go back" or "rewind to the beginning" - that's awkward. Instead, use something like: "But it all started when...", "It begins with...", or simply start the chronological narrative: "[Birth year/early life context]. This is where our story begins."
- Tone: Exciting, intriguing preview. Fast-paced like a trailer. NOT a narrative story.

NARRATION STYLE FOR INTRO:
- Speak directly to the viewer about what they'll discover
- Use phrases like "You'll discover...", "This is the story of...", "Wait until you learn...", "Here's why this matters..."
- Present facts as previews, not as events happening in real-time
- Make it clear this is a preview/introduction, not the actual story''' if is_hook_chapter else '''THREE PRINCIPLES (CRITICAL - every scene must satisfy all three):
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

{build_scripts_utils.get_shared_narration_style(is_short=False)}
{("- HOOK/INTRO: Speak directly to the viewer about what they'll discover. Use preview language: \"You'll discover...\", \"This is the story of...\", \"Wait until you learn...\", \"Here's why this matters...\" Present facts as teasers, not as events happening in real-time.\n" +
"- TRANSITION TO STORY: The final scene should naturally bridge to the chronological narrative without saying \"rewind\" or \"go back\". Simply start with the beginning context: \"[Early life context]. This is where our story begins.\" or \"It all started when...\"") if is_hook_chapter else ""}

{("HOOK CHAPTER EXAMPLES:\n" +
"BAD (no intro): \"In 1905, Einstein publishes four papers that redefine physics. The physics community is stunned.\"\n" +
"GOOD (with intro): \"Welcome to Human Footprints. Today we'll talk about Albert Einstein - the man who changed the world. In 1905, a 26-year-old patent clerk publishes four papers that redefine physics. The scientific community is stunned.\"\n" +
"GOOD (with intro and hook): \"Welcome to Human Footprints. Today we'll talk about Charles Darwin - the man who changed the way we understand life. The letter arrives in June 1858. Darwin's hands tremble as he reads his own theory in another man's words.\"\n" +
"BAD transition: \"Now the story rewinds to the beginning...\"\n" +
"GOOD transition: \"It all started in Ulm, Germany, in 1879, when Einstein was born.\"") if is_hook_chapter else build_scripts_utils.get_shared_examples("historical")}

{prompt_builders.get_image_prompt_guidelines(person, birth_year, death_year, "16:9 cinematic", is_trailer=False, recurring_themes=threads_str)}

Respond with JSON array:
[
  {{
    "id": {start_id},
    "title": "Evocative 2-5 word title",
    "narration": "Vivid, dramatic narration. MUST satisfy the three principles: (1) viewer knows WHAT IS HAPPENING, (2) viewer knows WHY IT IS IMPORTANT, (3) viewer knows WHAT COULD GO WRONG.",
    "scene_type": "WHY" or "WHAT" - MUST be one of these two values. WHY sections frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered or don't understand the significance of. CRITICAL: Every scene (WHY or WHAT) must satisfy the three principles: what is happening, why it's important, what could go wrong. WHY sections should set up what will happen next, why it matters, and what the stakes are for upcoming WHAT sections. WHAT sections deliver core content/solutions/information and must clearly communicate what is happening, why it's important, and what the stakes are (what can go wrong, what can go right, what's at risk).",
    "image_prompt": "Detailed visual description including {person}'s age and age-relevant appearance details, 16:9 cinematic",
    "emotion": "A single word or short phrase describing the scene's emotional tone (e.g., 'tense', 'triumphant', 'desperate', 'contemplative', 'exhilarating', 'somber', 'urgent', 'defiant'). Should match the chapter's emotional tone but be scene-specific based on what the character is feeling and the dramatic tension. CRITICAL: Emotions must flow SMOOTHLY between scenes - only change gradually from the previous scene's emotion. Build intensity gradually: 'contemplative' → 'thoughtful' → 'somber' → 'serious' → 'tense'. Avoid dramatic jumps like 'calm' → 'urgent' or 'contemplative' → 'exhilarating'.",
    "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field. Example: 'Focus on tension.' or 'Focus on contemplation.' Keep it simple - just the emotion to emphasize.",
    "year": YYYY or "YYYY-YYYY" or "around YYYY" (the specific year or year range when this scene takes place)
  }},
  ...
]"""

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You are a YouTuber creating documentary content. Write narration from YOUR perspective - this is YOUR script that YOU wrote. Tell the story naturally, directly to the viewer. CRITICAL: Use third person narration - you are speaking ABOUT the main character (he/she/they), not AS the main character. The narrator tells the story of the person, not in their voice. THREE PRINCIPLES (every scene must satisfy all three): (1) The viewer must know WHAT IS HAPPENING - establish the situation, who is involved, where we are. (2) The viewer must know WHY IT IS IMPORTANT - why this moment matters, its significance. (3) The viewer must know WHAT COULD GO WRONG - what's at risk, what failure would mean. Check every scene against these three. CRITICAL - DEPTH AND CONTEXT: Viewers need enough context to understand every scene. Establish the situation before the event: who is involved, where we are, why this moment is happening. When you introduce a place, person, or concept (e.g. a battle, a document, a political crisis), give one phrase of context so a general viewer isn't lost. Avoid name-dropping without explanation. After major events, briefly note what it changes or what would have happened otherwise. Use 2-4 sentences per scene when context demands it—prioritize clarity and depth over brevity. CRITICAL FOR RETENTION - AVOID VIEWER CONFUSION: The biggest issue for retention is viewer confusion. WHY sections MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions. Don't create confusion by being vague about what's happening. CRITICAL: Create a SEAMLESS JOURNEY through the video - scenes should feel CONNECTED, not like consecutive pieces of disjoint information (A and B and C). Each scene should build on the previous scene, reference what came before naturally, and show how events connect. CRITICAL: Every scene must be classified as WHY or WHAT. WHY sections frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered or don't understand the significance of. WHY sections MUST clearly establish what is happening (MOST IMPORTANT for retention) before introducing questions or mysteries. WHY sections should set up what will happen next, why it matters, and what the stakes are for upcoming WHAT sections. WHAT sections deliver core content, solutions, and information. Every WHAT scene must clearly communicate: what is happening, why it's important, and what the stakes are (what can go wrong, what can go right, what's at risk). Interleave WHY and WHAT sections strategically - WHY sections should create anticipation for upcoming WHAT sections by establishing what/why/stakes, but NEVER at the expense of clarity about what's happening. Use WHY/WHAT interleaving to create natural connections. For hook chapters, use mostly WHY sections. CRITICAL: For each scene, provide narration_instructions as ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on contemplation.' The narration_instructions should flow smoothly between scenes - if previous scene was 'Focus on tension', next might be 'Focus on concern' (gradual progression). Avoid overly dramatic language. Avoid any meta references to chapters, production elements, or the script structure. Focus on what actually happened, why it mattered, and how it felt. Respond with valid JSON array only."},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.85,
    )
    scenes = json.loads(clean_json_response(content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    # Validate that each scene has required fields
    for i, scene in enumerate(scenes):
        if 'year' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'year' field")
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
        if 'scene_type' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'scene_type' field")
        if scene.get('scene_type') not in ['WHY', 'WHAT']:
            raise ValueError(f"Scene {i+1} has invalid 'scene_type' value: {scene.get('scene_type')}. Must be 'WHY' or 'WHAT'")
    
    return scenes


# Horror scene generation moved to build_script_horror.py


# identify_pivotal_moments, generate_significance_scene, generate_refinement_diff, and refine_scenes
# are now imported from build_scripts_utils above (see imports section at the top)


def generate_short_outline(person: str, selected_hook: dict = None, short_num: int = 1, total_shorts: int = 3, birth_year: int = None, death_year: int = None) -> dict:
    """Generate a trailer outline for a YouTube Short based on a hook from the intro chapter.
    All shorts use the same shared trailer-style prompt.
    """
    import prompt_builders
    
    outline_prompt = prompt_builders.build_short_outline_prompt(person, selected_hook, short_num, total_shorts)

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You create high-energy viral trailers. Every word must grab attention and create curiosity. Focus on making viewers NEED to watch the full video. CRITICAL: For each scene, provide narration_instructions as ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on urgency.' The narration_instructions should flow smoothly between scenes - if previous scene was 'Focus on tension', next might be 'Focus on intensity' (gradual progression). Avoid overly dramatic language. Respond with valid JSON only. You MUST respond with a single JSON object (not an array) containing short_title, short_description, tags, thumbnail_prompt, hook_expansion, key_facts."},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    data = json.loads(clean_json_response(content))
    # LLM sometimes returns a list instead of an object; normalize to dict
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        data = data[0]
    elif not isinstance(data, dict):
        raise ValueError(f"Short outline must be a JSON object, got {type(data).__name__}")
    return data


def generate_short_scenes(person: str, short_outline: dict, birth_year: int | None = None, death_year: int | None = None) -> list[dict]:
    """Generate 4 scenes for a YouTube Short: scenes 1-3 build the hook and end with a question; scene 4 answers that question.
    
    Each scene must include a "year" field indicating when it takes place.
    Scenes 1-3 are WHY scenes (trailer format); scene 4 is a WHAT scene (payoff/answer).
    """
    import prompt_builders
    
    key_facts = short_outline.get('key_facts', [])
    facts_str = "\n".join(f"• {fact}" for fact in key_facts) if key_facts else "Use the hook expansion to create curiosity"
    hook_expansion = short_outline.get('hook_expansion', '')
    
    scene_prompt = f"""Write 4 scenes for a YouTube Short about {person}.

TITLE: "{short_outline.get('short_title', '')}"
HOOK EXPANSION: {hook_expansion}

KEY FACTS TO USE:
{facts_str}

CRITICAL: Scenes 1-3 are HIGH-ENERGY TRAILER (WHY) that build the hook and end with a clear QUESTION. Scene 4 ANSWERS that question (WHAT scene - payoff), then invites viewers to the full documentary.

{prompt_builders.get_why_what_paradigm_prompt(is_trailer=True)}

{prompt_builders.get_emotion_generation_prompt()}

{prompt_builders.get_trailer_structure_prompt()}

{prompt_builders.get_trailer_narration_style()}

{prompt_builders.get_image_prompt_guidelines(person, birth_year, death_year, "9:16 vertical", is_trailer=True)}

Respond with JSON array of exactly 4 scenes (scenes 1-3 WHY, scene 4 WHAT - answers the question from scene 3):
[
  {{"id": 1, "title": "2-4 words", "narration": "HIGH ENERGY - expand the hook, create curiosity, make viewers NEED to know more", "scene_type": "WHY", "image_prompt": "Dramatic, attention-grabbing visual including {person}'s age at this time ({birth_year if birth_year else 'unknown'}), 9:16 vertical", "emotion": "High energy emotion (e.g., 'urgent', 'shocking', 'tense', 'intense'). CRITICAL: Emotions must flow SMOOTHLY between scenes - only change gradually.", "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field. Example: 'Focus on tension.' or 'Focus on urgency.'", "year": "YYYY or YYYY-YYYY"}},
  {{"id": 2, "title": "...", "narration": "HIGH ENERGY - build the hook, escalate curiosity, deepen the mystery", "scene_type": "WHY", "image_prompt": "Dramatic, attention-grabbing visual including {person}'s age at this time, 9:16 vertical", "emotion": "High energy emotion - must be a GRADUAL progression from scene 1's emotion (e.g., if scene 1 was 'tense', this might be 'intense' or 'urgent', not 'calm').", "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field, flowing smoothly from scene 1. Example: if scene 1 was 'Focus on tension.', this might be 'Focus on intensity.'", "year": "YYYY or YYYY-YYYY"}},
  {{"id": 3, "title": "...", "narration": "HIGH ENERGY - build to a climax and END WITH A CLEAR QUESTION that scene 4 will answer (e.g. 'How did he do it?' 'What happened next?' 'Why did this work?'). Do not answer it here.", "scene_type": "WHY", "image_prompt": "Dramatic, attention-grabbing visual including {person}'s age at this time, 9:16 vertical", "emotion": "High energy emotion - must be a GRADUAL progression from scene 2's emotion.", "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field, flowing smoothly from scene 2. End narration with the question.", "year": "YYYY or YYYY-YYYY"}},
  {{"id": 4, "title": "...", "narration": "ANSWER the question from scene 3. WHAT scene - deliver the payoff with clear, punchy facts. End with a soft CTA to watch the full documentary (e.g. 'Watch the full documentary for the complete story.').", "scene_type": "WHAT", "image_prompt": "Dramatic visual showing the resolution/moment of answer including {person}'s age at this time, 9:16 vertical", "emotion": "Satisfying resolution - can be 'triumphant', 'revelatory', 'relieved', or 'impactful' - must flow from scene 3.", "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field (e.g. 'Focus on the payoff.' or 'Focus on revelation.').", "year": "YYYY or YYYY-YYYY"}}
]

IMPORTANT: 
- Scenes 1-3: scene_type MUST be "WHY"; scene 3 MUST end with a clear question.
- Scene 4: scene_type MUST be "WHAT"; it MUST answer the question from scene 3.
- "emotion" should flow smoothly across all 4 scenes.
- "image_prompt" MUST include {person}'s age at that time and be dramatic/attention-grabbing.
- "year" field indicating when the scene takes place.
]"""

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You create high-energy viral shorts. Write narration from YOUR perspective - this is YOUR script. Use third person narration - you are speaking ABOUT the main character (he/she/they), not AS the main character. CRITICAL: Scenes 1-3 are WHY scenes (build the hook, end scene 3 with a clear QUESTION). Scene 4 is a WHAT scene that ANSWERS that question (payoff), then invites viewers to the full documentary. WHY scenes MUST ensure the viewer knows WHAT IS HAPPENING - provide clear context before introducing mysteries. High energy in 1-3; scene 4 delivers satisfying resolution. Simple, clear, punchy language. CRITICAL: For each scene, provide narration_instructions as ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' The narration_instructions should flow smoothly between scenes. Avoid overly dramatic language. Respond with valid JSON array only."},
            {"role": "user", "content": scene_prompt}
        ],
        temperature=0.9,
    )
    scenes = json.loads(clean_json_response(content))
    
    if not isinstance(scenes, list):
        raise ValueError(f"Expected array, got {type(scenes)}")
    
    # Validate that each scene has required fields
    for i, scene in enumerate(scenes):
        if 'year' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'year' field")
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
        if 'scene_type' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'scene_type' field")
        if scene.get('scene_type') not in ['WHY', 'WHAT']:
            raise ValueError(f"Scene {i+1} has invalid 'scene_type' value: {scene.get('scene_type')}. Must be 'WHY' or 'WHAT'")
    
    return scenes


def generate_shorts(person_of_interest: str, main_title: str, global_block: str, outline: dict, base_output_path: str, hook_chapter_scenes: list = None):
    """Generate YouTube Shorts (4 scenes each: scenes 1-3 build hook and end with a question, scene 4 answers it)."""
    if hook_chapter_scenes is None:
        hook_chapter_scenes = []
    if config.num_shorts == 0:
        print("\n[SHORTS] Skipped (--shorts 0)")
        return []
    
    print(f"\n{'='*60}")
    print(f"[SHORTS] Generating {config.num_shorts} YouTube Short(s)")
    print(f"[SHORTS] Structure: {config.total_short_scenes} scenes each (high-energy trailers from intro hooks)")
    print(f"{'='*60}")
    
    SHORTS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    
    generated_shorts = []
    base_name = Path(base_output_path).stem.replace("_script", "")
    previously_used_hooks = []  # Track which hooks have been used
    
    # Extract hooks from intro chapter scenes
    # Hook chapter scenes are scenes 1 to scenes_per_chapter
    # We want scenes 2 to scenes_per_chapter (skip scene 1 which is the intro)
    available_hooks = []
    if hook_chapter_scenes:
        # hook_chapter_scenes should already be filtered to scenes 2 to scenes_per_chapter
        # But let's be safe and filter by scene ID
        for scene in hook_chapter_scenes:
            scene_id = scene.get("id", 0)
            # Only include scenes from chapter 1 (scenes 1 to scenes_per_chapter), but skip scene 1 (intro)
            if 2 <= scene_id <= config.scenes_per_chapter:
                available_hooks.append({
                    "scene_id": scene_id,
                    "title": scene.get("title", ""),
                    "narration": scene.get("narration", ""),
                    "year": scene.get("year", "")
                })
        print(f"[SHORTS] Extracted {len(available_hooks)} hooks from intro chapter (scenes 2-{config.scenes_per_chapter})")
    
    if not available_hooks:
        print(f"[SHORTS] WARNING: No hooks available from intro chapter. Shorts may not be generated properly.")
    
    # STEP 1: Generate all outlines sequentially (selecting different hooks)
    print(f"\n[OPTIMIZATION] Step 1: Generating all short outlines from intro hooks...")
    short_outlines = []
    for short_num in range(1, config.num_shorts + 1):
        print(f"\n[SHORT {short_num}/{config.num_shorts}] Creating trailer outline from intro hook...")
        
        # Select a hook that hasn't been used yet
        selected_hook = None
        if available_hooks:
            # Try to find an unused hook
            for hook in available_hooks:
                hook_id = hook.get("scene_id")
                if hook_id not in previously_used_hooks:
                    selected_hook = hook
                    previously_used_hooks.append(hook_id)
                    break
            
            # If all hooks used, cycle back (shouldn't happen with 3 shorts and 4 scenes, but handle it)
            if not selected_hook and available_hooks:
                selected_hook = available_hooks[(short_num - 1) % len(available_hooks)]
                print(f"[SHORTS]   • Reusing hook from Scene {selected_hook.get('scene_id')} (all hooks used)")
        else:
            print(f"[SHORTS]   • WARNING: No hooks available, generating without hook")
        
        short_outline = generate_short_outline(
            person=person_of_interest,
            selected_hook=selected_hook,
            short_num=short_num,
            total_shorts=config.num_shorts,
            birth_year=outline.get('birth_year'),
            death_year=outline.get('death_year')
        )
        
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
            
            # Refine short scenes (only PASS 3 - skip storyline completion and pivotal moments for trailers)
            short_context = f"Trailer: {short_outline.get('short_title', '')}"
            # Determine short file path first for diff path
            short_file = SHORTS_DIR / f"{base_name}_short{short_num}.json"
            diff_path = short_file.parent / f"{short_file.stem}_refinement_diff.json" if config.generate_refinement_diffs else None
            all_scenes, refinement_diff = build_scripts_utils.refine_scenes(all_scenes, person_of_interest, is_short=True, chapter_context=short_context, diff_output_path=diff_path, subject_type="person", skip_significance_scenes=False, scenes_per_chapter=None)
            
            # Generate thumbnail (if enabled) - using shared WHY scene prompt
            thumbnail_path = None
            if config.generate_short_thumbnails:
                import prompt_builders
                thumbnail_prompt = prompt_builders.get_thumbnail_prompt_why_scene("9:16")
                thumb_file = THUMBNAILS_DIR / f"{base_name}_short{short_num}_thumbnail.jpeg"
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
                    "hook_expansion": short_outline.get("hook_expansion", ""),
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
) -> None:
    """Write current script state to JSON so progress is saved if a later step fails."""
    output_data = {
        "metadata": {
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
        },
        "scenes": all_scenes,
    }
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
        print(f"[CONFIG] Main video: {config.chapters} chapters × {config.scenes_per_chapter} scenes = {config.total_scenes} scenes")
    else:
        print(f"[CONFIG] Main video: SKIPPED")
    if config.num_shorts > 0:
        print(f"[CONFIG] Shorts: {config.num_shorts} × {config.total_short_scenes} scenes = {config.num_shorts * config.total_short_scenes} scenes")
    else:
        print(f"[CONFIG] Shorts: SKIPPED")
    print(f"[CONFIG] Thumbnails: {'Yes' if config.generate_thumbnails else 'No'}")
    print(f"[CONFIG] Model: {llm_utils.get_text_model_display()}")
    
    # Step 1: Generate detailed outline (needed for both main and shorts)
    print("\n[STEP 1] Creating life outline...")
    outline = generate_outline(person_of_interest)
    chapters = outline.get("chapters", [])
    
    if config.generate_main and len(chapters) < config.chapters:
        print(f"[WARNING] Got {len(chapters)} chapters, expected {config.chapters}")
    
    # Step 2: Generate initial metadata (title, thumbnail, global_block) - we'll regenerate description after scenes
    print("\n[STEP 2] Generating initial metadata...")
    
    import prompt_builders
    initial_metadata_prompt = script_type_instance.get_metadata_prompt(person_of_interest, outline.get('tagline', ''), config.total_scenes)

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "Documentary producer. Respond with valid JSON only."},
            {"role": "user", "content": initial_metadata_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    initial_metadata = json.loads(clean_json_response(content))
    
    title = initial_metadata["title"]
    tag_line = initial_metadata.get("tag_line", f"the story of {person_of_interest}")
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
        
        # Use shared WHY scene thumbnail prompt
        thumbnail_prompt = f"""{thumbnail_description}

{prompt_builders.get_thumbnail_prompt_why_scene("16:9")}"""
        
        generated_thumb = build_scripts_utils.generate_thumbnail(thumbnail_prompt, thumbnail_path, "1024x1024", config.generate_thumbnails)
    
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
        all_scenes, refinement_diff = build_scripts_utils.refine_scenes(all_scenes, person_of_interest, is_short=False, chapter_context=chapter_summaries, diff_output_path=diff_path, subject_type="person", skip_significance_scenes=False, scenes_per_chapter=config.scenes_per_chapter, script_type="biopic")
        
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
        )
        
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
  "video_description": "Brief YouTube description (100-200 words max) - concise summary optimized for SEO. Start with a compelling hook that includes key SEO keywords. Provide a brief overview of the person's story and why it matters. Keep it concise - don't include excessive detail about specific scenes or events. End with: 'If you enjoyed this video, please like and subscribe for more stories like this!'",
  "tags": "15-20 SEO tags separated by commas. Mix of: person's name, key topics, related figures, time periods, achievements, fields (e.g., 'Albert Einstein, physics, relativity, Nobel Prize, Germany, Princeton, E=mc2, quantum mechanics, genius, scientist, 20th century, biography, documentary')",
  "pinned_comment": "An engaging question or comment to pin below the video (1-2 sentences max). Should: spark discussion, ask a thought-provoking question about the person/story, create curiosity, encourage viewers to share their thoughts/opinions, be conversational and inviting. Examples: 'Which moment from their story surprised you most? Drop a comment below!', 'What do you think was their greatest challenge? Share your thoughts!', 'This story shows how one person can change everything. What impact do you want to make?'. Should feel authentic, not clickbaity - genuine curiosity about viewer perspectives."
}}"""

        system_role = "Documentary producer. Create compelling metadata that accurately reflects the actual content."
        content = llm_utils.generate_text(
            messages=[
                {"role": "system", "content": f"{system_role} Respond with valid JSON only."},
                {"role": "user", "content": final_metadata_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
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
        )
    else:
        print("\n[STEP 3] Skipping main video scene generation...")
        # Generate basic metadata if not generating main video
        video_description = ""
        tags = ""
        pinned_comment = ""
    
    # Step 4: Generate Shorts (extract hooks from intro chapter)
    print("\n[STEP 4] Generating YouTube Shorts...")
    hook_chapter_scenes = []
    if config.generate_main and all_scenes:
        # Extract hook chapter scenes (chapter 1 scenes, excluding intro scene)
        # Hook chapter = scenes 1 to scenes_per_chapter
        # We want scenes 2 to scenes_per_chapter (skip intro)
        hook_start_idx = 1  # Skip scene 1 (intro)
        hook_end_idx = config.scenes_per_chapter
        for i in range(hook_start_idx, min(hook_end_idx, len(all_scenes))):
            if i < config.scenes_per_chapter:  # Only include hook chapter scenes
                hook_chapter_scenes.append(all_scenes[i])
        print(f"[SHORTS] Extracted {len(hook_chapter_scenes)} hooks from intro chapter (scenes 2-{config.scenes_per_chapter})")
    
    # Generate shorts
    shorts_info = generate_shorts(person_of_interest, title, global_block, outline, output_path, hook_chapter_scenes=hook_chapter_scenes)
    
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
            "script_type": "biopic",
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
  # Full production run (24 scenes main video, 3 shorts with 3 scenes each)
  python build_script_biopic.py "Albert Einstein"

  # Quick test (4 scenes main, 1 short with 4 scenes, no thumbnails)
  python build_script_biopic.py "Albert Einstein" --test

  # Custom main video settings
  python build_script_biopic.py "Albert Einstein" --chapters 5 --scenes 3

  # Custom shorts settings  
  python build_script_biopic.py "Albert Einstein" --shorts 2 --short-scenes 4

  # No thumbnails (faster iteration)
  python build_script_biopic.py "Albert Einstein" --no-thumbnails

  # Only generate shorts (skip main video)
  python build_script_biopic.py "Albert Einstein" --shorts-only

  # Only generate main video (skip shorts)
  python build_script_biopic.py "Albert Einstein" --main-only
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
    
    # Main video settings (defaults from config)
    parser.add_argument("--chapters", type=int, default=config.chapters,
                        help=f"Main video outline chapters (default: {config.chapters})")
    parser.add_argument("--scenes", type=int, default=config.scenes_per_chapter,
                        help=f"Scenes per main chapter (default: {config.scenes_per_chapter}, total = chapters × scenes)")
    
    # Shorts settings (defaults from config)
    parser.add_argument("--shorts", type=int, default=config.num_shorts,
                        help=f"Number of YouTube Shorts (default: {config.num_shorts}, use 0 to skip)")
    parser.add_argument("--short-scenes", type=int, default=config.short_scenes_per_chapter,
                        help=f"Scenes per short (default: {config.short_scenes_per_chapter}: build, build, cliffhanger)")
    
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
        config.short_scenes_per_chapter = 4
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
