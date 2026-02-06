import os
import sys
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv

# Import shared utilities and LLM utils (provider/model from .env)
import build_scripts_utils
import llm_utils

load_dotenv()

THUMBNAILS_DIR = Path("thumbnails")
SCRIPTS_DIR = Path("scripts")
SHORTS_DIR = Path("shorts_scripts")

# Import config from separate module
import config
config = config.Config()

# Import shared functions from utils - use directly
clean_json_response = build_scripts_utils.clean_json_response

# Import script types
import script_types

# Valid drone_change values for horror scenes
VALID_DRONE_CHANGES = ['fade_in', 'fade_out', 'hard_cut', 'hold', 'swell', 'shrink', 'none']

# Valid environment values for horror scenes
VALID_ENVIRONMENTS = ['blizzard', 'snow', 'forest', 'rain', 'indoors', 'jungle']


def generate_horror_outline(story_concept: str, chapters: int, total_scenes: int) -> dict:
    """Generate a detailed outline for a horror story."""
    print(f"\n[OUTLINE] Generating {chapters}-chapter horror story outline...")
    
    import prompt_builders
    outline_prompt = prompt_builders.build_horror_outline_prompt(story_concept, chapters, total_scenes)

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You are a master horror storyteller who creates terrifying, atmospheric stories. "},
            {"role": "user", "content": outline_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    outline_data = json.loads(clean_json_response(content))
    chapters_list = outline_data.get("chapters", [])
    
    print(f"[OUTLINE] Generated {len(chapters_list)} chapters")
    for ch in chapters_list:
        time_setting = ch.get('time_setting', 'Unknown')
        print(f"  Ch {ch['chapter_num']}: {ch['title']} ({time_setting})")
    
    return outline_data


def generate_horror_scenes_for_chapter(story_concept: str, chapter: dict, scenes_per_chapter: int, start_id: int,
                                      global_style: str, prev_chapter: dict = None, prev_scenes: list = None,
                                      central_theme: str = None, narrative_arc: str = None,
                                      planted_seeds: list[str] = None, is_retention_hook_point: bool = False,
                                      tag_line: str | None = None, overarching_plots: list[dict] = None,
                                      sub_plots: list[dict] = None) -> list[dict]:
    """Generate scenes for a horror story chapter with first-person narration."""
    import prompt_builders
    
    if planted_seeds is None:
        planted_seeds = []
    
    is_hook_chapter = (chapter.get('chapter_num', 0) == 1)
    
    # Build previous chapter context
    prev_context = ""
    if prev_chapter:
        prev_context = f"""PREVIOUS CHAPTER: "{prev_chapter.get('title', '')}"
Summary: {prev_chapter.get('summary', '')}
Ended with: {prev_chapter.get('connects_to_next', '')}

Your scenes should naturally continue from where the previous chapter ended."""
    else:
        prev_context = "This is the FIRST chapter - establish the horror story setup."
    
    # Build recent scenes context - include FULL scene JSON for proper transitions
    scenes_context = ""
    if prev_scenes and len(prev_scenes) > 0:
        recent_scenes = prev_scenes[-5:]
        scenes_context = "RECENT SCENES - AVOID REPEATING THESE EVENTS, maintain continuity, and ensure smooth transitions:\n"
        for sc in recent_scenes:
            # Include full scene JSON for proper emotion, narration_instructions, and drone_change transitions
            scenes_context += f"  Scene {sc.get('id')} (FULL JSON): {json.dumps(sc, indent=2)}\n"
        scenes_context += "\nCRITICAL: Do NOT repeat or overlap with events already covered. Each scene must cover DIFFERENT events.\n"
        scenes_context += "CRITICAL: For smooth transitions, ensure:\n"
        scenes_context += "- emotion flows gradually from the previous scene's emotion\n"
        scenes_context += "- narration_instructions flow smoothly from the previous scene's narration_instructions\n"
        scenes_context += "- drone_change is correct based on the previous scene's drone_change (use hold/swell/shrink/fade_out/hard_cut if drone was present, fade_in if it wasn't)"
    else:
        scenes_context = ""
    
    # Get narrative context
    connects_to_next = chapter.get('connects_to_next', '')
    horror_elements = chapter.get('horror_elements', [])
    horror_elements_str = ', '.join(horror_elements) if horror_elements else 'none specified'
    time_setting = chapter.get('time_setting', 'Unknown')
    
    # Build central theme and narrative arc context
    theme_context = ""
    if central_theme:
        theme_context += f"\nCENTRAL HORROR THEME (connect all scenes to this): {central_theme}\n"
    if narrative_arc:
        theme_context += f"NARRATIVE ARC (we are at this point in the horror journey): {narrative_arc}\n"
    
    # Build callback context for planted seeds
    callback_context = ""
    if planted_seeds:
        callback_context = f"""
PLANTED SEEDS TO REFERENCE (create satisfying horror callbacks):
{chr(10).join(f"• {seed}" for seed in planted_seeds)}

These details were mentioned earlier. Reference them naturally to create "aha moments" when earlier details suddenly matter. Don't force it - weave them in organically."""
    
    # Build plot context for horror
    plot_context = ""
    if overarching_plots is None:
        overarching_plots = []
    if sub_plots is None:
        sub_plots = []
    
    active_plots = chapter.get('plots_active', [])
    plot_developments = chapter.get('plot_developments', [])
    
    if active_plots or plot_developments or overarching_plots or sub_plots:
        plot_context = "\nHORROR PLOTS & SUB-PLOTS - CRITICAL FOR STORY COHESION:\n"
        
        if overarching_plots:
            plot_context += "OVERARCHING HORROR PLOTS (spanning multiple chapters):\n"
            for plot in overarching_plots:
                plot_name = plot.get('plot_name', '')
                plot_desc = plot.get('description', '')
                starts_ch = plot.get('starts_chapter', '?')
                peaks_ch = plot.get('peaks_chapter', '?')
                resolves_ch = plot.get('resolves_chapter', '?')
                current_chapter_num = chapter.get('chapter_num', 0)
                
                if current_chapter_num < starts_ch:
                    stage = "NOT YET STARTED"
                elif current_chapter_num == starts_ch:
                    stage = "BEGINS HERE - introduce this horror thread"
                elif starts_ch < current_chapter_num < peaks_ch:
                    stage = "DEVELOPING - escalate this horror"
                elif current_chapter_num == peaks_ch:
                    stage = "PEAKS HERE - this is the climax of this horror"
                elif peaks_ch < current_chapter_num:
                    stage = "RESOLVING - but should remain OPEN/UNRESOLVED"
                else:
                    stage = "COMPLETED"
                
                plot_context += f"• \"{plot_name}\": {plot_desc} (Starts: Ch {starts_ch}, Peaks: Ch {peaks_ch}) [{stage}]\n"
        
        if relevant_subplots := [sp for sp in (sub_plots or []) if chapter.get('chapter_num', 0) in sp.get('chapters_span', [])]:
            plot_context += "\nACTIVE SUB-PLOTS (developing in this chapter):\n"
            for subplot in relevant_subplots:
                plot_context += f"• \"{subplot.get('subplot_name', '')}\": {subplot.get('description', '')}\n"
        
        if plot_developments:
            plot_context += "\nPLOT DEVELOPMENTS IN THIS CHAPTER:\n"
            for dev in plot_developments:
                plot_context += f"• {dev}\n"
        
        plot_context += """
HORROR STORYTELLING REQUIREMENTS:
- Weave horror plot elements NATURALLY throughout your scenes
- Show horror threads developing through actual events, not exposition
- Build TENSION by showing how horror plots are escalating
- Reference horror threads from earlier chapters when relevant
- Each scene should advance at least one horror plot thread
- Focus on ATMOSPHERE and FEAR, not just events"""
    
    # Build retention hook instruction
    retention_hook_instruction = ""
    if is_retention_hook_point:
        retention_hook_instruction = """
CRITICAL RETENTION HOOK: This scene falls at a key retention point (~30s, 60s, or 90s mark).
The FINAL scene in this chapter MUST end with a compelling horror hook:
- A scary reveal or moment
- An unresolved question that makes viewers NEED to see what happens next
- A moment of high tension or fear
- A "wait, what?" moment that creates fear and curiosity
This hook is essential for YouTube algorithm performance."""
    
    # Determine chapter-specific instructions
    chapter_num = chapter.get('chapter_num', 0)
    if chapter_num == 1:
        chapter_instructions = """CHAPTER 1 - SETUP & TENSION BUILDING:
- Introduce protagonist in FIRST PERSON (I/me/my)
- Establish normal world before horror begins
- Introduce mystery/threat/unease
- Build initial tension and atmosphere
- Mostly WHY scenes (mystery, questions, unease)
- End with something unsettling that makes viewers want to continue
- Use first person throughout: "I walk into...", "I hear...", "I feel..."
- Present tense for immediacy"""
    elif chapter_num == 2:
        chapter_instructions = """CHAPTER 2 - ESCALATION & SCARES:
- Tension escalates significantly
- Scare moments and reveals
- Threat becomes clearer (but not fully explained)
- Mix of WHY (mystery) and WHAT (scares/reveals)
- Build to a major scare or revelation
- Increase fear and paranoia
- Use first person: "I see...", "I realize...", "I'm terrified because..."
- Present tense for immediacy"""
    else:  # Chapter 3
        chapter_instructions = """CHAPTER 3 - CLIMAX & OPEN ENDING:
- Final confrontation/climax with the horror
- Open ending (unresolved, keeps viewer scared)
- Lingering questions and unease
- Mostly WHY scenes (unresolved tension)
- End with something that makes viewers still feel scared/unsettled
- DO NOT fully resolve - leave mystery and fear lingering
- Use first person: "I face...", "I realize...", "I'm still scared because..."
- Present tense for immediacy
- CRITICAL: Open ending - don't fully explain or resolve everything"""
    
    scene_prompt = f"""You are writing scenes {start_id}-{start_id + scenes_per_chapter - 1} of a horror story about: {story_concept}

{prev_context}
{scenes_context}
NOW WRITING CHAPTER {chapter['chapter_num']} of 3: "{chapter['title']}"
Time Setting: {time_setting}
Emotional Tone: {chapter['emotional_tone']}
Dramatic Tension: {chapter['dramatic_tension']}
Horror Elements: {horror_elements_str}
Sets Up What Comes Next: {connects_to_next}

{prompt_builders.get_why_what_paradigm_prompt(is_hook_chapter=is_hook_chapter)}

{prompt_builders.get_emotion_generation_prompt(chapter.get('emotional_tone', ''))}

{prompt_builders.get_horror_narration_style()}

{theme_context}
{plot_context}
{callback_context}
{retention_hook_instruction}

Chapter Summary: {chapter['summary']}

Key Horror Events to Dramatize:
{chr(10).join(f"• {event}" for event in chapter.get('key_events', []))}

Generate EXACTLY {scenes_per_chapter} scenes that FLOW CONTINUOUSLY.

{chapter_instructions}

CRITICAL HORROR REQUIREMENTS:
- FIRST PERSON throughout (I/me/my) - the protagonist tells their own story
- Build TENSION progressively - each scene should increase fear/unease
- Create ATMOSPHERE through description: shadows, sounds, feelings, unease
- Use WHY/WHAT paradigm at scene level:
  * WHY scenes: Frame mysteries, questions, unease, "what's happening?" moments
  * WHAT scenes: Reveal scares, show threats, deliver horror moments
- Focus on FEAR and ATMOSPHERE, not just jump scares
- Make viewers feel UNSETTLED and SCARED
- Use sensory details: what I see, hear, feel, smell
- Internal thoughts and reactions: "I think...", "I wonder...", "I'm terrified because..."
- Physical sensations: "My hands shake...", "My heart races...", "I feel cold..."

LOW FREQUENCY DRONE CHANGES - CRITICAL FOR ATMOSPHERE:
The low frequency drone (background audio) should change at specific moments to enhance horror atmosphere. 
For each scene, indicate if the drone should change and how:

1. FADE IN:
   - What the audience feels: "Something has entered the situation."
   - When to use:
     * Early unease scenes
     * After a normal or calm scene
     * When suspicion begins
     * When tension starts building
   - How: Very slow fade (3-10 seconds), no noticeable start point
   - Ideally begins before the narration acknowledges fear
   - Use when drone is not currently present or at low volume

2. HOLD:
   - What the audience feels: "It's still here."
   - When to use:
     * When drone is already present from previous scene
     * Maintaining the same level of tension
     * Sustaining unease without escalation
   - How: Keep drone at current volume level
   - Use when the threat/presence remains constant

3. SWELL:
   - What the audience feels: "It's getting closer / worse."
   - When to use:
     * When tension needs to escalate
     * When threat is approaching
     * Building to a scare or reveal
     * When fear is intensifying
   - How: Gradually increase drone volume (3-8 seconds)
   - Use when drone is already present and needs to intensify

4. SHRINK:
   - What the audience feels: "It's backing off (for now)."
   - When to use:
     * Brief moments of relief (but not safety)
     * When threat temporarily recedes
     * Before a bigger escalation
     * Creating false sense of security
   - How: Gradually decrease drone volume (3-8 seconds)
   - Important: Never shrink completely—room tone must remain
   - Use when drone is already present and needs to decrease

5. FADE OUT:
   - What the audience feels: "We might be safe."
   - When to use:
     * After a scare (brief relief)
     * When the character convinces themselves everything is fine
     * Before a bigger escalation (false sense of security)
   - Important: Never fade out completely—room tone must remain.
   - Creates contrast that makes the next scare more effective.

6. HARD CUT (Instant Silence of Drone):
   - What the audience feels: "Reality just snapped."
   - When to use:
     * Realization moments
     * Final lines of a scene or chapter
     * Right before or after a reveal
     * End of the video
   - This is extremely effective when used ONCE or sparingly.
   - Creates maximum impact through sudden silence.

For each scene, add a "drone_change" field with one of these values:
- "fade_in" - Drone should fade in (use when drone is not present or at low volume, early unease, after calm scenes)
- "hold" - Drone should hold at current level (use when drone is already present and tension remains constant)
- "swell" - Drone should increase in volume (use when drone is already present and tension is escalating)
- "shrink" - Drone should decrease in volume (use when drone is already present and threat temporarily recedes)
- "fade_out" - Drone should fade out (use after scares, false safety moments, before bigger escalations)
- "hard_cut" - Drone should cut instantly to silence (use sparingly for realization moments, reveals, final lines)
- "none" - No change, maintain current drone level (legacy option, prefer "hold" when drone is already present)

CRITICAL: You can only use hold, swell, shrink, fade_out, and hard_cut once the drone is present. You make the drone present by fade_in. You cannot use fade_in or none when the drone is already present.

{build_scripts_utils.get_shared_scene_flow_instructions()}

{build_scripts_utils.get_shared_scene_requirements("historical")}

{prompt_builders.get_image_prompt_guidelines(story_concept, None, None, "16:9 cinematic", is_trailer=False, recurring_themes=horror_elements_str, is_horror=True)}

[
  {{
    "id": {start_id},
    "title": "Evocative 2-5 word horror title",
    "narration": "First-person, present tense horror narration (I/me/my)...",
    "scene_type": "WHY" or "WHAT" - MUST be one of these two values. WHY sections frame mysteries, questions, unease, "what's happening?" moments. WHAT sections reveal scares, show threats, deliver horror moments.",
    "image_prompt": "Horror visual description with dark, shadowy, eerie atmosphere, 16:9 cinematic",
    "emotion": "Horror emotion (e.g., 'terrified', 'tense', 'atmospheric', 'dread-filled', 'fearful', 'paranoid', 'uneasy'). Should match the chapter's emotional tone but be scene-specific. CRITICAL: Emotions must flow SMOOTHLY between scenes - only change gradually from the previous scene's emotion. Build intensity gradually: 'uneasy' → 'tense' → 'anxious' → 'fearful' → 'terrified'. Avoid dramatic jumps like 'calm' → 'terrified'.",
    "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field. Example: 'Focus on tension.' or 'Focus on unease.' Keep it simple - just the emotion to emphasize.",
    "drone_change": "fade_in" or "hold" or "swell" or "shrink" or "fade_out" or "hard_cut" or "none" - Indicate how the low frequency drone should change for this scene. Use fade_in when drone is not present, hold/swell/shrink when drone is already present, fade_out for temporary relief, hard_cut sparingly for realization moments/reveals.",
    "year": "time setting or 'present' or relevant time reference"
  }},
  ...
]
"""
    
    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You are a horror storyteller creating a first-person scary story. Write narration in FIRST PERSON (I/me/my), present tense. CRITICAL: Use first person throughout - the protagonist is telling their own story. Create tension, atmosphere, and fear. Use WHY/WHAT paradigm at scene level. Build horror progressively. Focus on atmosphere and unease. CRITICAL: For each scene, provide narration_instructions as ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on unease.' The narration_instructions should flow smoothly between scenes - if previous scene was 'Focus on tension', next might be 'Focus on anxiety' (gradual progression). Avoid overly dramatic language. "},
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
            scene['year'] = time_setting  # Default to chapter time setting
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
        if 'scene_type' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'scene_type' field")
        if scene.get('scene_type') not in ['WHY', 'WHAT']:
            raise ValueError(f"Scene {i+1} has invalid 'scene_type' value: {scene.get('scene_type')}. Must be 'WHY' or 'WHAT'")
        if 'narration_instructions' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'narration_instructions' field")
        if 'drone_change' not in scene:
            # Default to "none" if not specified
            scene['drone_change'] = "none"
        elif scene.get('drone_change') not in VALID_DRONE_CHANGES:
            valid_values_str = ', '.join(f"'{v}'" for v in VALID_DRONE_CHANGES)
            raise ValueError(f"Scene {i+1} has invalid 'drone_change' value: {scene.get('drone_change')}. Must be one of: {valid_values_str}")
    
    return scenes


def generate_horror_short_outline(story_concept: str) -> dict:
    """Generate a short outline for a horror short story (quick scary story, open-ended, trailer-like)."""
    print(f"\n[SHORT OUTLINE] Generating horror short outline...")
    
    short_outline_prompt = f"""Create a quick, scary horror story outline for a YouTube Short.

STORY CONCEPT: {story_concept}

CRITICAL: STAY FAITHFUL TO THE CONCEPT - The story must honor the user's concept. If the concept is a bear attack, the horror is a bear attack (survival, nature, threat)—do NOT turn it into a mimic, doppelganger, or "thing that looks like X but is really human-like." If the concept is a haunted house, the horror is the haunted house. Only use mimicry/copy/uncanny-human themes when the concept itself clearly asks for them (e.g. "something that imitates people").

This is a SHORT horror story (3 scenes) that should:
- Be a quick, scary story that can stand alone
- Feel like a trailer but doesn't need to be a trailer for anything specific
- Be very open-ended to create curiosity and drive traffic to the channel
- Use first-person narration (I/me/my), present tense
- Focus on building tension and fear quickly
- End with an open-ended, unsettling conclusion that leaves viewers wanting more
- Be designed to drive traffic and attention to the channel and full-length videos

The story should be:
- High energy and attention-grabbing
- Atmospheric and scary
- Quick-paced (3 scenes total)
- Open-ended (doesn't need to fully resolve)
- Designed to make viewers want to watch more content

CRITICAL: TITLE AND THUMBNAIL MUST BE COHESIVE AND SYNCED - They must create the SAME curiosity gap and work together to maximize CTR. The thumbnail should visually represent the same horror/mystery/threat that the title frames.

CRITICAL: ENVIRONMENT SELECTION - You must select ONE environment for the entire short story. This will determine the background ambient sound. Choose from: "blizzard", "snow", "forest", "rain", or "indoors". Base your choice on the story's setting and atmosphere. The environment should match where the horror takes place.

Generate JSON:
{{
  "short_title": "MYSTERIOUS HORROR TITLE - MAXIMIZE CTR (max 40 chars): Stay true to the story concept. Use simple, evocative words that create curiosity. Examples (vary by concept): creature/nature—'Something in the Woods', 'The Clearing', 'The Hollow'; supernatural—'The Shining', 'The Echo', 'The Shadow'; ambiguous—'Blizzard Whistle', 'Something in the Snow', 'The Voice', 'The Presence'. Do NOT default to mimic/copy/reflection themes unless the concept is about imitation. The title should hint at horror without revealing it. Max 40 chars. Must create a curiosity gap while staying appropriate.",
  "short_description": "Brief description of what this short horror story is about (1-2 sentences)",
  "environment": "ONE of: {VALID_ENVIRONMENTS} - the ambient environment for the entire short story",
  "hook_expansion": "The core hook/idea that will be expanded into 3 scenes. This should be a compelling, scary premise that can build tension quickly.",
  "key_facts": ["Key fact 1 about the horror", "Key fact 2", "Key fact 3"],
  "emotional_tone": "The overall emotional tone (e.g., 'terrified', 'paranoid', 'dread-filled', 'atmospheric fear')",
  "horror_elements": ["horror element 1", "horror element 2", "horror element 3"],
  "tags": "horror,scary,creepy,thriller,suspense,terrifying,horror story,scary story,horror narration,first person horror,short horror story,quick horror,scary short"
}}"""

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You are a master horror storyteller creating quick, scary short stories for YouTube Shorts. These shorts are designed to drive traffic and attention to the channel. Create compelling, open-ended horror that leaves viewers wanting more. "},
            {"role": "user", "content": short_outline_prompt}
        ],
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    
    return json.loads(clean_json_response(content))


def generate_horror_short_scenes(story_concept: str, short_outline: dict) -> list[dict]:
    """Generate 3 scenes for a horror short story.
    
    Each scene must include narration_instructions and should be mostly WHY scenes.
    """
    import prompt_builders
    
    key_facts = short_outline.get('key_facts', [])
    facts_str = "\n".join(f"• {fact}" for fact in key_facts) if key_facts else "Use the hook expansion to create fear and tension"
    hook_expansion = short_outline.get('hook_expansion', '')
    emotional_tone = short_outline.get('emotional_tone', 'terrified')
    horror_elements = short_outline.get('horror_elements', [])
    horror_elements_str = ", ".join(horror_elements) if horror_elements else "atmospheric fear, tension, unease"
    
    scene_prompt = f"""Write 3 HIGH-ENERGY HORROR scenes for a YouTube Short horror story.

STORY CONCEPT: {story_concept}
TITLE: "{short_outline.get('short_title', '')}"
HOOK EXPANSION: {hook_expansion}
EMOTIONAL TONE: {emotional_tone}

KEY FACTS TO USE:
{facts_str}

HORROR ELEMENTS: {horror_elements_str}

CRITICAL: This is a QUICK SCARY STORY (3 scenes) that should:
- Be high energy and attention-grabbing
- Build tension and fear quickly
- Use FIRST PERSON narration (I/me/my), present tense
- Feel like a trailer but doesn't need to be a trailer for anything specific
- End with an open-ended, unsettling conclusion
- Be designed to drive traffic to the channel

LOW FREQUENCY DRONE CHANGES - CRITICAL FOR ATMOSPHERE:
The low frequency drone (background audio) should change at specific moments to enhance horror atmosphere. 
For each scene, indicate if the drone should change and how:

1. FADE IN:
   - What the audience feels: "Something has entered the situation."
   - When to use: Early unease scenes, after calm scenes, when suspicion begins, when tension starts building
   - How: Very slow fade (3-10 seconds), no noticeable start point
   - Use when drone is not currently present or at low volume

2. HOLD:
   - What the audience feels: "It's still here."
   - When to use: When drone is already present from previous scene, maintaining same tension level
   - How: Keep drone at current volume level

3. SWELL:
   - What the audience feels: "It's getting closer / worse."
   - When to use: When tension needs to escalate, threat is approaching, building to a scare
   - How: Gradually increase drone volume (3-8 seconds)
   - Use when drone is already present and needs to intensify

4. SHRINK:
   - What the audience feels: "It's backing off (for now)."
   - When to use: Brief moments of relief, when threat temporarily recedes, before bigger escalation
   - How: Gradually decrease drone volume (3-8 seconds)
   - Important: Never shrink completely—room tone must remain

5. FADE OUT:
   - What the audience feels: "We might be safe."
   - When to use: After a scare (brief relief), when character convinces themselves everything is fine
   - Important: Never fade out completely—room tone must remain.

6. HARD CUT (Instant Silence of Drone):
   - What the audience feels: "Reality just snapped."
   - When to use: Realization moments, final lines, right before or after a reveal, end of video
   - This is extremely effective when used ONCE or sparingly.

For each scene, add a "drone_change" field with one of these values:
- "fade_in" - Drone should fade in (use when drone is not present or at low volume)
- "hold" - Drone should hold at current level (use when drone is already present)
- "swell" - Drone should increase in volume (use when drone is already present and tension escalates)
- "shrink" - Drone should decrease in volume (use when drone is already present and threat recedes)
- "fade_out" - Drone should fade out (use after scares, false safety moments)
- "hard_cut" - Drone should cut instantly to silence (use sparingly for realization moments, reveals, final lines)
- "none" - No change, maintain current drone level (legacy option, prefer "hold" when drone is already present)

{prompt_builders.get_why_what_paradigm_prompt(is_trailer=True)}

{prompt_builders.get_emotion_generation_prompt(emotional_tone)}

{prompt_builders.get_horror_narration_style()}

{prompt_builders.get_image_prompt_guidelines(story_concept, None, None, "9:16 vertical", is_trailer=True, recurring_themes=horror_elements_str, is_horror=True)}

[
  {{"id": 1, "title": "2-4 words", "narration": "HIGH ENERGY - first-person horror narration that establishes the scary premise, creates immediate tension and fear", "scene_type": "WHY", "image_prompt": "Horror visual with dark, shadowy, eerie atmosphere, 9:16 vertical", "emotion": "Horror emotion (e.g., 'uneasy', 'tense', 'atmospheric', 'dread-filled', 'fearful', 'paranoid'). CRITICAL: Emotions must flow SMOOTHLY between scenes - only change gradually. Build intensity gradually: 'uneasy' → 'tense' → 'anxious' → 'fearful' → 'terrified'.", "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field. Example: 'Focus on tension.' or 'Focus on unease.'", "drone_change": "fade_in" or "hold" or "swell" or "shrink" or "fade_out" or "hard_cut" or "none" - Indicate how the low frequency drone should change for this scene.", "year": "present or relevant time reference"}},
  {{"id": 2, "title": "...", "narration": "HIGH ENERGY - escalate the horror, deepen the fear, intensify the threat", "scene_type": "WHY", "image_prompt": "Horror visual with dark, shadowy, eerie atmosphere, 9:16 vertical", "emotion": "Horror emotion - must be a GRADUAL progression from scene 1's emotion (e.g., if scene 1 was 'uneasy', this might be 'tense' or 'anxious', not 'terrified').", "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field, flowing smoothly from scene 1. Example: if scene 1 was 'Focus on tension.', this might be 'Focus on anxiety.'", "drone_change": "fade_in" or "hold" or "swell" or "shrink" or "fade_out" or "hard_cut" or "none" - Indicate how the low frequency drone should change for this scene.", "year": "present or relevant time reference"}},
  {{"id": 3, "title": "...", "narration": "HIGH ENERGY - peak horror moment, open-ended conclusion that leaves viewers scared and wanting more", "scene_type": "WHY", "image_prompt": "Horror visual with dark, shadowy, eerie atmosphere, 9:16 vertical", "emotion": "Horror emotion - must be a GRADUAL progression from scene 2's emotion (e.g., if scene 2 was 'tense', this might be 'anxious' or 'fearful', building naturally).", "narration_instructions": "ONE SENTENCE: Focus on a single emotion from the emotion field, flowing smoothly from scene 2. Example: if scene 2 was 'Focus on anxiety.', this might be 'Focus on fear.'", "drone_change": "fade_in" or "hold" or "swell" or "shrink" or "fade_out" or "hard_cut" or "none" - Indicate how the low frequency drone should change for this scene. For final scene, consider "hard_cut" for maximum impact.", "year": "present or relevant time reference"}}
]

IMPORTANT: 
- ALL scenes should be WHY scenes (this is a quick horror story, not a complete resolution)
- "scene_type" MUST be "WHY" for all 3 scenes
- "emotion" should be horror-focused (terrified, tense, atmospheric, dread-filled, fearful, paranoid, uneasy, etc.)
- "image_prompt" should be dark, shadowy, eerie, atmospheric horror visuals
- "narration" must be FIRST PERSON (I/me/my), present tense
- "narration_instructions" must specify first-person emotional delivery matching the horror
- "year" field indicating time setting
]"""

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "You are a horror storyteller creating a quick, scary first-person horror story for YouTube Shorts. Write narration in FIRST PERSON (I/me/my), present tense. CRITICAL: Use first person throughout - the protagonist is telling their own story. Create tension, atmosphere, and fear quickly. This is a short horror story designed to drive traffic - make it compelling and open-ended. All scenes should be WHY scenes that build fear. CRITICAL: For each scene, provide narration_instructions as ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on unease.' The narration_instructions should flow smoothly between scenes - if previous scene was 'Focus on tension', next might be 'Focus on anxiety' (gradual progression). Avoid overly dramatic language. "},
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
            scene['year'] = "present"  # Default for horror shorts
        if 'emotion' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'emotion' field")
        if 'scene_type' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'scene_type' field")
        if scene.get('scene_type') not in ['WHY', 'WHAT']:
            raise ValueError(f"Scene {i+1} has invalid 'scene_type' value: {scene.get('scene_type')}. Must be 'WHY' or 'WHAT'")
        if 'narration_instructions' not in scene:
            raise ValueError(f"Scene {i+1} missing required 'narration_instructions' field")
        if 'drone_change' not in scene:
            # Default to "none" if not specified
            scene['drone_change'] = "none"
        elif scene.get('drone_change') not in VALID_DRONE_CHANGES:
            valid_values_str = ', '.join(f"'{v}'" for v in VALID_DRONE_CHANGES)
            raise ValueError(f"Scene {i+1} has invalid 'drone_change' value: {scene.get('drone_change')}. Must be one of: {valid_values_str}")
    
    return scenes


def _save_horror_script(
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
    story_concept: str,
    environment: str,
    outline: dict,
    all_scenes: list,
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
            "story_concept": story_concept,
            "script_type": "horror",
            "num_scenes": len(all_scenes),
            "environment": environment,
            "outline": outline,
            "shorts": [],
        },
        "scenes": all_scenes,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"[SCRIPT] Saved progress: {output_path} ({len(all_scenes)} scenes)")


def generate_script(story_concept: str, output_path: str, is_short: bool = False):
    """Generate a complete horror story script using outline-guided generation.
    
    If is_short is True, generates a short horror story (3 scenes) instead of a full video.
    """
    import script_types
    
    if is_short:
        # SHORT MODE: Generate a quick scary story (3 scenes)
        print(f"\n{'='*60}")
        print(f"[SCRIPT] Generating HORROR SHORT for: {story_concept}")
        print(f"{'='*60}")
        print(f"[CONFIG] Short: 3 scenes (quick scary story)")
        print(f"[CONFIG] Thumbnails: {'Yes' if config.generate_thumbnails else 'No'}")
        print(f"[CONFIG] Model: {llm_utils.get_text_model_display()}")
        
        # Step 1: Generate short outline
        print("\n[STEP 1] Creating horror short outline...")
        short_outline = generate_horror_short_outline(story_concept)
        
        # Step 2: Generate 3 scenes
        print("\n[STEP 2] Generating 3 scenes for horror short...")
        all_scenes = generate_horror_short_scenes(story_concept, short_outline)
        
        # Step 3: Refine scenes (only final pass for shorts)
        print("\n[STEP 3] Refining short scenes (final pass only)...")
        import prompt_builders
        diff_path = Path(output_path).parent / f"{Path(output_path).stem}_refinement_diff.json" if config.generate_refinement_diffs else None
        all_scenes, refinement_diff = build_scripts_utils.refine_scenes(
            all_scenes, 
            story_concept, 
            is_short=True, 
            chapter_context=None, 
            diff_output_path=diff_path, 
            subject_type="character", 
            skip_significance_scenes=True, 
            scenes_per_chapter=None, 
            script_type="horror"
        )
        
        # Step 4: Generate metadata
        print("\n[STEP 4] Generating metadata...")
        short_title = short_outline.get('short_title', '')
        short_description = short_outline.get('short_description', '')
        tags = short_outline.get('tags', '')
        
        # Get environment from outline
        environment = short_outline.get("environment", "indoors")
        if environment not in VALID_ENVIRONMENTS:
            print(f"[WARNING] Invalid environment '{environment}', defaulting to 'indoors'")
            environment = "indoors"
        
        print(f"[METADATA] Environment: {environment}")
        
        # Build short script
        short_script = {
            "metadata": {
                "title": short_title,
                "description": short_description,
                "tags": tags,
                "thumbnail_path": None,  # Shorts don't need thumbnails
                "story_concept": story_concept,
                "script_type": "horror_short",
                "num_scenes": len(all_scenes),
                "environment": environment,  # Store environment in metadata for audio mixing
                "outline": short_outline
            },
            "scenes": all_scenes
        }
        
        # Save short script
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(short_script, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SHORT] ✓ Saved: {output_path} ({len(all_scenes)} scenes)")
        
        return {
            "metadata": short_script["metadata"],
            "scenes": all_scenes
        }
    
    # MAIN VIDEO MODE: Generate full horror story
    script_type_instance = script_types.HorrorStoryScript()
    actual_chapters = 3  # Horror uses 3 chapters
    
    print(f"\n{'='*60}")
    print(f"[SCRIPT] Generating HORROR STORY script for: {story_concept}")
    print(f"{'='*60}")
    
    if config.generate_main:
        scenes_per_chapter = config.scenes_per_chapter
        total_scenes = actual_chapters * scenes_per_chapter
        print(f"[CONFIG] Main video: {actual_chapters} chapters × {scenes_per_chapter} scenes = {total_scenes} scenes")
    else:
        print(f"[CONFIG] Main video: SKIPPED")
    print(f"[CONFIG] Thumbnails: {'Yes' if config.generate_thumbnails else 'No'}")
    print(f"[CONFIG] Model: {llm_utils.get_text_model_display()}")
    
    # Step 1: Generate detailed outline
    print("\n[STEP 1] Creating horror story outline...")
    outline = generate_horror_outline(story_concept, actual_chapters, total_scenes)
    chapters = outline.get("chapters", [])
    
    if config.generate_main and len(chapters) < actual_chapters:
        print(f"[WARNING] Got {len(chapters)} chapters, expected {actual_chapters}")
    
    # Step 2: Generate initial metadata (title, thumbnail, global_block)
    print("\n[STEP 2] Generating initial metadata...")
    
    import prompt_builders
    initial_metadata_prompt = script_type_instance.get_metadata_prompt(story_concept, outline.get('tagline', ''), total_scenes)

    content = llm_utils.generate_text(
        messages=[
            {"role": "system", "content": "Horror story producer. "},
            {"role": "user", "content": initial_metadata_prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    initial_metadata = json.loads(clean_json_response(content))
    
    title = initial_metadata["title"]
    tag_line = initial_metadata.get("tag_line", f"a story that will haunt you")
    thumbnail_description = initial_metadata["thumbnail_description"]
    global_block = initial_metadata["global_block"]
    
    # Get environment from outline (will be moved to metadata later)
    environment = outline.get("environment", "indoors")
    if environment not in VALID_ENVIRONMENTS:
        print(f"[WARNING] Invalid environment '{environment}', defaulting to 'indoors'")
        environment = "indoors"
    
    print(f"[METADATA] Title: {title}")
    print(f"[METADATA] Tag line: {tag_line}")
    print(f"[METADATA] Environment: {environment}")
    
    # Generate main video thumbnail
    generated_thumb = None
    if config.generate_main:
        print("\n[THUMBNAIL] Main video thumbnail...")
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        thumbnail_path = THUMBNAILS_DIR / f"{Path(output_path).stem}_thumbnail.png"
        
        # Use shared WHY scene thumbnail prompt
        thumbnail_prompt = f"""{thumbnail_description}

{prompt_builders.get_thumbnail_prompt_why_scene("16:9")}"""
        
        generated_thumb = build_scripts_utils.generate_thumbnail(thumbnail_prompt, thumbnail_path, "1024x1024", config.generate_thumbnails)
    
    # Step 3: Generate scenes chapter by chapter
    all_scenes = []
    planted_seeds = []  # Track details from early chapters for callbacks
    
    if config.generate_main:
        print(f"\n[STEP 3] Generating {total_scenes} scenes from {len(chapters)} chapters...")
        
        # Get central theme, narrative arc, and plots from outline
        central_theme = outline.get('central_theme', '') or outline.get('central_horror', '')
        narrative_arc = outline.get('narrative_arc', '')
        overarching_plots = outline.get('overarching_plots', [])
        sub_plots = outline.get('sub_plots', [])
        
        for i, chapter in enumerate(chapters):
            start_id = len(all_scenes) + 1
            
            # Get previous chapter and scenes for continuity
            prev_chapter = chapters[i - 1] if i > 0 else None
            
            # Determine if this is a retention hook point (~30s, 60s, 90s marks)
            estimated_time = start_id * 15
            is_retention_hook = (estimated_time >= 25 and estimated_time <= 35) or \
                               (estimated_time >= 55 and estimated_time <= 65) or \
                               (estimated_time >= 85 and estimated_time <= 95)
            
            print(f"\n[CHAPTER {chapter['chapter_num']}/{len(chapters)}] {chapter['title']}")
            if is_retention_hook:
                print(f"  ⚠ RETENTION HOOK POINT (~{estimated_time}s mark)")
            print(f"  Generating {config.scenes_per_chapter} scenes...")
            
            try:
                scenes = generate_horror_scenes_for_chapter(
                    story_concept=story_concept,
                    chapter=chapter,
                    scenes_per_chapter=config.scenes_per_chapter,
                    start_id=start_id,
                    global_style=global_block,
                    prev_chapter=prev_chapter,
                    prev_scenes=list(all_scenes),
                    central_theme=central_theme,
                    narrative_arc=narrative_arc,
                    planted_seeds=planted_seeds if i > 0 else [],
                    is_retention_hook_point=is_retention_hook,
                    tag_line=tag_line if i == 0 else None,
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
                _save_horror_script(
                    output_path,
                    title=title,
                    tag_line=tag_line,
                    video_description="",
                    tags="",
                    pinned_comment="",
                    thumbnail_description=thumbnail_description,
                    generated_thumb=generated_thumb,
                    global_block=global_block,
                    story_concept=story_concept,
                    environment=environment,
                    outline=outline,
                    all_scenes=all_scenes,
                )
                
                # Extract "planted seeds" from early chapters (first 3 chapters) for callback mechanism
                if i < 3:  # First 3 chapters plant seeds
                    for scene in scenes:
                        narration = scene.get('narration', '')
                        scene_title = scene.get('title', '')
                        if narration:
                            planted_seeds.append(f"{scene_title}: {narration[:80]}...")
                    # Also add chapter's key events as potential seeds
                    for event in chapter.get('key_events', []):
                        # Horror seeds: objects, sounds, feelings, threats, mysteries
                        if any(word in event.lower() for word in ['sound', 'shadow', 'feeling', 'threat', 'mystery', 'object', 'door', 'window', 'voice', 'presence', 'fear', 'warning']):
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
        
        # Step 3.4: Refine scenes
        print("\n[STEP 3.4] Refining main video scenes...")
        chapter_summaries = "\n".join([f"Chapter {ch['chapter_num']}: {ch['title']} ({ch.get('time_setting', 'Unknown')}) - {ch['summary']}" for ch in chapters])
        diff_path = Path(output_path).parent / f"{Path(output_path).stem}_refinement_diff.json" if config.generate_refinement_diffs else None
        all_scenes, refinement_diff = build_scripts_utils.refine_scenes(all_scenes, story_concept, is_short=False, chapter_context=chapter_summaries, diff_output_path=diff_path, subject_type="character", skip_significance_scenes=False, scenes_per_chapter=config.scenes_per_chapter, script_type="horror")
        
        # Save progress after refinement (metadata not yet generated)
        _save_horror_script(
            output_path,
            title=title,
            tag_line=tag_line,
            video_description="",
            tags="",
            pinned_comment="",
            thumbnail_description=thumbnail_description,
            generated_thumb=generated_thumb,
            global_block=global_block,
            story_concept=story_concept,
            environment=environment,
            outline=outline,
            all_scenes=all_scenes,
        )
        
        # Step 3.5: Generate final metadata (description and tags) AFTER scenes are generated
        print("\n[STEP 3.5] Generating final metadata from actual scenes...")
        
        # Extract memorable moments from scenes for metadata
        scene_highlights = []
        for scene in all_scenes[:10]:  # First 10 scenes for highlights
            scene_highlights.append(f"Scene {scene['id']}: {scene.get('title', '')} - {scene.get('narration', '')[:80]}...")
        
        final_metadata_prompt = f"""Create final metadata for a horror story video about: {story_concept}

Story tagline: {outline.get('tagline', '')}

Actual memorable moments from the horror story (use these to make description more accurate):
{chr(10).join(scene_highlights[:10])}

Generate JSON:
{{
  "video_description": "Brief YouTube description (100-200 words max) - concise summary optimized for SEO. Start with a compelling hook that includes key SEO keywords (horror, scary, thriller, etc.). Provide a brief overview of the horror story without revealing too much detail. Keep it concise - don't include excessive detail about specific scenes or scares. End with: 'If you enjoyed this video, please like and subscribe for more stories like this!'",
  "tags": "15-20 SEO tags separated by commas. Mix of: horror, scary, creepy, thriller, suspense, paranormal, haunted, terrifying, horror story, scary story, first person horror, horror narration, etc.",
  "pinned_comment": "An engaging horror question or comment to pin below the video (1-2 sentences max). Should: spark discussion about the horror, ask a thought-provoking question about what viewers think happened, create curiosity, encourage viewers to share their theories. Examples: 'What do you think was really happening? Drop your theory below!', 'Which moment scared you the most? Share in the comments!', 'Do you think the ending means what I think it means? Let me know!'. Should feel authentic and engaging."
}}"""

        content = llm_utils.generate_text(
            messages=[
                {"role": "system", "content": "Horror story producer. Create compelling metadata that accurately reflects the actual horror content. "},
                {"role": "user", "content": final_metadata_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        final_metadata = json.loads(clean_json_response(content))
        video_description = final_metadata.get("video_description", "")
        tags = final_metadata.get("tags", "")
        pinned_comment = final_metadata.get("pinned_comment", "")
        
        print(f"[METADATA] Description generated from {len(all_scenes)} scenes")
        print(f"[METADATA] Tags: {tags[:80]}..." if len(tags) > 80 else f"[METADATA] Tags: {tags}")
        if pinned_comment:
            print(f"[METADATA] Pinned comment: {pinned_comment}")
        # Save progress with full metadata (in case a later step fails)
        _save_horror_script(
            output_path,
            title=title,
            tag_line=tag_line,
            video_description=video_description,
            tags=tags,
            pinned_comment=pinned_comment,
            thumbnail_description=thumbnail_description,
            generated_thumb=generated_thumb,
            global_block=global_block,
            story_concept=story_concept,
            environment=environment,
            outline=outline,
            all_scenes=all_scenes,
        )
    else:
        print("\n[STEP 3] Skipping main video scene generation...")
        video_description = ""
        tags = ""
        pinned_comment = ""
    
    # Step 4: Save
    print("\n[STEP 4] Saving script...")
    
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
            "story_concept": story_concept,
            "script_type": "horror",
            "num_scenes": len(all_scenes),
            "environment": environment,  # Store environment in metadata for audio mixing
            "outline": outline,  # Outline without environment (moved to metadata)
            "shorts": []  # Horror shorts not implemented yet
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
    
    return output_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate horror story scripts with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full production run (3 chapters × 4 scenes = 12 scenes)
  python build_script_horror.py "A haunted house story"

  # Generate a horror short (3 scenes, quick scary story)
  python build_script_horror.py "A haunted house story" --shorts

  # Quick test (2 chapters × 2 scenes, no thumbnails)
  python build_script_horror.py "A haunted house story" --test

  # Custom settings
  python build_script_horror.py "A haunted house story" --chapters 3 --scenes 5

  # No thumbnails (faster iteration)
  python build_script_horror.py "A haunted house story" --no-thumbnails
        """
    )
    
    parser.add_argument("story_concept", help="Story concept (e.g., 'A haunted house story')")
    parser.add_argument("output", nargs="?", help="Output JSON file (default: <story_concept>_script.json)")
    
    # Quick test mode
    parser.add_argument("--test", action="store_true", 
                        help="Quick test: 2 chapters × 2 scenes, no thumbnails")
    
    # Main video settings (defaults from config)
    parser.add_argument("--chapters", type=int, default=3,
                        help=f"Main video outline chapters (default: 3 for horror)")
    parser.add_argument("--scenes", type=int, default=config.scenes_per_chapter,
                        help=f"Scenes per main chapter (default: {config.scenes_per_chapter}, total = chapters × scenes)")
    
    parser.add_argument("--no-thumbnails", action="store_true",
                        help="Skip main video thumbnail generation")
    parser.add_argument("--refinement-diffs", action="store_true",
                        help="Generate refinement diff JSON files showing what changed during scene refinement")
    
    # Shorts mode
    parser.add_argument("--shorts", action="store_true",
                        help="Generate a horror short (3 scenes) instead of a full video. Creates a quick scary story that's open-ended and designed to drive traffic.")
    
    return parser.parse_args()


# ------------- ENTRY POINT -------------

if __name__ == "__main__":
    args = parse_args()
    
    # Apply test mode settings
    if args.test:
        config.chapters = 2
        config.scenes_per_chapter = 2
        config.generate_thumbnails = False
        config.generate_refinement_diffs = False
        print("[MODE] Test mode enabled")
    else:
        config.chapters = args.chapters
        config.scenes_per_chapter = args.scenes
        config.generate_thumbnails = not args.no_thumbnails
        config.generate_refinement_diffs = args.refinement_diffs
    
    # Determine output file (goes to scripts/ or shorts_scripts/ directory)
    if args.shorts:
        SHORTS_DIR.mkdir(parents=True, exist_ok=True)
        if args.output:
            output_file = args.output
            if not output_file.startswith("shorts_scripts/") and not Path(output_file).is_absolute():
                output_file = str(SHORTS_DIR / output_file)
        else:
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in args.story_concept)
            safe_name = safe_name.replace(' ', '_').lower()
            output_file = str(SHORTS_DIR / f"{safe_name}_short.json")
    else:
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        if args.output:
            output_file = args.output
            if not output_file.startswith("scripts/") and not Path(output_file).is_absolute():
                output_file = str(SCRIPTS_DIR / output_file)
        else:
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in args.story_concept)
            safe_name = safe_name.replace(' ', '_').lower()
            output_file = str(SCRIPTS_DIR / f"{safe_name}_script.json")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)
    
    try:
        script_data = generate_script(args.story_concept, output_file, is_short=args.shorts)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        
        if args.shorts:
            print(f"\n🎬 HORROR SHORT:")
            print(f"   Script: {output_file}")
            print(f"   Scenes: {script_data['metadata']['num_scenes']}")
            print(f"\n🎬 Build video:")
            print(f"   python build_video.py {output_file} output.mp4")
        else:
            if config.generate_main:
                print(f"\n📺 MAIN VIDEO:")
                print(f"   Script: {output_file}")
                print(f"   Scenes: {script_data['metadata']['num_scenes']}")
                if script_data['metadata'].get('thumbnail_path'):
                    print(f"   Thumbnail: {script_data['metadata']['thumbnail_path']}")
            
            print(f"\n🎬 Build video:")
            if config.generate_main:
                print(f"   python build_video.py {output_file} output.mp4")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
