"""
Shared utilities for script generation.
Used by build_script_biopic.py for historical documentary scripts.
"""
import json
import time
import re
from pathlib import Path

from PIL import Image

import llm_utils
import prompt_builders
from biopic_schemas import (
    PIVOTAL_MOMENTS_SCHEMA,
    SCENE_SCHEMA,
    HANGING_STORYLINES_SCHEMA,
    HISTORIAN_DEPTH_ADDITIONS_SCHEMA,
    SCENES_ARRAY_SCHEMA,
    MUSIC_SELECTION_SCHEMA,
    TRANSITION_SELECTION_SCHEMA,
)

NO_TEXT_CONSTRAINT: str = """
CRITICAL: Do NOT include any text, words, letters, numbers, titles, labels, watermarks, or any written content in the image. The image must be completely text-free."""

THUMBNAIL_MAX_BYTES = 2 * 1024 * 1024  # 2 MB


def _compress_thumbnail_if_needed(path: Path, max_bytes: int = THUMBNAIL_MAX_BYTES) -> None:
    """Re-save thumbnail with lower JPEG quality if file size exceeds max_bytes."""
    if not path.exists():
        return
    size = path.stat().st_size
    if size <= max_bytes:
        return
    try:
        img = Image.open(path).convert("RGB")
        for quality in (85, 75, 65, 55, 45, 35):
            img.save(path, "JPEG", quality=quality, optimize=True)
            if path.stat().st_size <= max_bytes:
                print(f"[THUMBNAIL] Compressed to {path.stat().st_size / 1024:.0f} KB (was {size / 1024:.0f} KB, quality={quality})")
                return
        print(f"[WARNING] Thumbnail still {path.stat().st_size / 1024:.0f} KB after compression (target < 2 MB)")
    except Exception as e:
        print(f"[WARNING] Could not compress thumbnail: {e}")


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


def validate_biopic_scenes(scenes: list[dict], script_path_hint: str = "") -> None:
    """
    Validate biopic scenes: required fields, valid values, music file existence.
    Raises ValueError or FileNotFoundError with descriptive message on first validation failure.
    """
    from kenburns_config import KENBURNS_PATTERNS, KENBURNS_INTENSITY_LEVELS, TRANSITION_TYPES, TRANSITION_SPEEDS, LEGACY_TRANSITION_TYPES
    try:
        from biopic_music_config import BIOPIC_MUSIC_DIR
    except ImportError:
        raise ImportError("biopic_music_config not found; cannot validate music file existence")

    required_fields = [
        "id", "title", "narration", "scene_type", "image_prompt", "emotion",
        "narration_instructions", "year", "chapter_num", "kenburns_pattern",
        "kenburns_intensity", "music_song", "music_volume", "transition_to_next",
    ]
    # transition_to_next can be None for the last scene; transition_speed is optional (defaults to medium)
    fields_allowing_none = frozenset(["transition_to_next"])
    valid_scene_types = ["WHY", "WHAT", "TRANSITION"]
    valid_music_volumes = ["low", "medium", "loud"]

    hint = f" ({script_path_hint})" if script_path_hint else ""

    for i, scene in enumerate(scenes):
        sid = scene.get("id", i + 1)
        for field in required_fields:
            if field not in scene:
                raise ValueError(
                    f"Scene {i + 1} (id={sid}) missing required field '{field}'{hint}"
                )
            if scene[field] is None and field not in fields_allowing_none:
                raise ValueError(
                    f"Scene {i + 1} (id={sid}) has null required field '{field}'{hint}"
                )
            if isinstance(scene[field], str) and not scene[field].strip():
                raise ValueError(
                    f"Scene {i + 1} (id={sid}) has empty required field '{field}'{hint}"
                )

        if scene.get("scene_type") not in valid_scene_types:
            raise ValueError(
                f"Scene {i + 1} (id={sid}) has invalid scene_type '{scene.get('scene_type')}'. "
                f"Must be one of: {valid_scene_types}{hint}"
            )
        if scene.get("kenburns_pattern") not in KENBURNS_PATTERNS:
            raise ValueError(
                f"Scene {i + 1} (id={sid}) has invalid kenburns_pattern '{scene.get('kenburns_pattern')}'. "
                f"Must be one of: {KENBURNS_PATTERNS}{hint}"
            )
        intensity = (scene.get("kenburns_intensity") or "").strip().lower()
        if intensity and intensity not in KENBURNS_INTENSITY_LEVELS:
            raise ValueError(
                f"Scene {i + 1} (id={sid}) has invalid kenburns_intensity '{scene.get('kenburns_intensity')}'. "
                f"Must be one of: {KENBURNS_INTENSITY_LEVELS}{hint}"
            )
        trans = scene.get("transition_to_next")
        valid_transitions = [t.lower() for t in TRANSITION_TYPES + LEGACY_TRANSITION_TYPES]
        if trans is not None and trans != "" and str(trans).strip().lower() not in valid_transitions:
            raise ValueError(
                f"Scene {i + 1} (id={sid}) has invalid transition_to_next '{trans}'. "
                f"Must be one of: {TRANSITION_TYPES}{hint}"
            )
        speed = scene.get("transition_speed")
        if speed is not None and speed != "" and str(speed).strip().lower() not in [s.lower() for s in TRANSITION_SPEEDS]:
            raise ValueError(
                f"Scene {i + 1} (id={sid}) has invalid transition_speed '{speed}'. "
                f"Must be one of: {TRANSITION_SPEEDS}{hint}"
            )
        vol = (scene.get("music_volume") or "").strip().lower()
        if vol not in valid_music_volumes:
            raise ValueError(
                f"Scene {i + 1} (id={sid}) has invalid music_volume '{scene.get('music_volume')}'. "
                f"Must be one of: {valid_music_volumes}{hint}"
            )

        song_rel = (scene.get("music_song") or "").strip()
        song_path = BIOPIC_MUSIC_DIR / song_rel if song_rel else None
        if not song_path or not song_path.is_file():
            raise FileNotFoundError(
                f"Music file not found for scene {i + 1} (id={sid}): "
                f"{BIOPIC_MUSIC_DIR / song_rel if song_rel else 'None'}{hint}"
            )


def get_shared_scene_requirements(content_type: str = "historical") -> str:
    """
    Shared scene requirements for both main video and shorts.
    
    Args:
        content_type: "historical" for historical documentaries, "lore" for fantasy/game lore
    """
    base_requirements = """SCENE REQUIREMENTS - DOCUMENTARY DEPTH:
1. SPECIFIC FACTS - names, dates, places, numbers, amounts, exact details
2. CONCRETE EVENTS - what exactly happened, step-by-step, who was involved, what was said, how it unfolded
3. CONTEXT and BACKGROUND - why this matters, what led to this moment, what the stakes were
4. HUMAN EMOTION AND EXPERIENCE (CRITICAL FOR ENGAGEMENT):
   - How does this event FEEL to the main character? What are they thinking, fearing, hoping?
   - What's the EMOTIONAL WEIGHT of this moment? Is it terrifying? Exhilarating? Crushing? Triumphant?
   - Show INTERNAL EXPERIENCE - not just what happens, but what it MEANS to them personally
   - What's at stake EMOTIONALLY? What would failure feel like? What does success mean?
   - Use specific emotional details: "Her hands shake as she writes...", "He can barely breathe from the pressure...", "The weight of what's at stake makes every word matter..."
   - Connect events to human feelings: isolation, fear, determination, joy, desperation, pride, grief
   - Make the viewer FEEL what the character feels - pull them into the emotional reality of the moment
5. SIGNIFICANCE AND IMPACT (CRITICAL FOR PIVOTAL MOMENTS):
   - For EVERY scene, explain why THIS moment matters - how significant it is, what it changes, its impact on the overall story
   - For PIVOTAL MOMENTS (major breakthroughs, turning points, critical decisions, moments that change the story trajectory), you MUST explicitly explain WHY this moment is pivotal - its significance to the broader story, how it shifts the narrative, what it changes about the character's journey, and what makes it a turning point
   - Make pivotal moments FEEL significant - don't just mention them, EXPLAIN their weight and importance
   - Show how pivotal moments reshape what comes after - what changes as a result of this moment?
6. CONSEQUENCES - what happened as a result, how it changed things, who was affected
7. INTERESTING information the viewer likely doesn't know - surprising details, behind-the-scenes facts
8. NO filler, NO fluff, NO vague statements, NO bullet-point style
9. Every sentence must contain NEW information that moves the story forward AND connects to the human experience"""
    
    return base_requirements


def get_shared_narration_style(is_short: bool = False, script_type: str | None = None) -> str:
    """Shared narration style instructions for both main video and shorts.
    
    Args:
        is_short: If True, use shorter scene length guidance for shorts.
        script_type: "biopic" adds audience-specific guidance for 65-year-old American men.
    """
    base_style = """NARRATION STYLE - CRITICAL:

THREE PRINCIPLES (every scene must satisfy all three):
1. The viewer must know WHAT IS HAPPENING - establish the situation, who is involved, where we are. Never leave viewers confused.
2. The viewer must know WHY IT IS IMPORTANT - why this moment matters, its significance, impact. Make the stakes clear.
3. The viewer must know WHAT COULD GO WRONG - what's at risk, what failure would mean. Show what can go wrong (or right) so the moment has weight.

- PERSPECTIVE: Write from the YouTuber's perspective - this is YOUR script, YOU are telling the story to YOUR audience
- THIRD PERSON NARRATION (CRITICAL): The narrator is speaking ABOUT the main character, not AS the main character. Always use third person (he/she/they) when referring to the main character. The narrator is telling the story of the person, not speaking in their voice. Examples: "Einstein publishes his paper..." NOT "I publish my paper..." or "We publish our paper..."
- SIMPLE, CLEAR language. Write for a general audience.
- AVOID flowery, artistic, or poetic language
- AVOID vague phrases like "little did he know", "destiny awaited", "the world would never be the same"
- AVOID sensationalist or gimmicky language: no "shocking", "incredible", "unbelievable", "mind-blowing", "what nobody expected", "the secret that would change everything", or similar rhetorical hooks repeated across scenes
- LET FACTS SPEAK: Present what happened clearly. The significance of events is in the facts themselves—don't oversell with hype or repeated dramatic framing
- SIMPLE AND STRAIGHTFORWARD: Prefer direct, clear statements over theatrical buildup. "In 1905, Einstein published four papers" over "The year 1905 would witness something that would reshape the very fabric of reality"
- NO dramatic pauses or buildup - just deliver the facts engagingly
- NO made-up temporal transitions - stick to what actually happened
- Use present tense: "Einstein submits his paper to the journal..."
- Tell a DEEP story with DETAILS - not just "he did X" but "he did X because Y, and here's how it happened, and here's what it meant"
- CONTEXT AND DEPTH (CRITICAL for main video): Viewers must always understand WHAT is happening, WHERE we are, and WHY it matters. For each scene:
  * ESTABLISH THE SITUATION before the event: Who are the key players? What's the setting or moment? Why is this happening now?
  * GROUND EVENTS in the bigger picture: What led to this? What's at stake for the person, the country, or the world? One sentence of setup prevents confusion.
  * WHEN INTRODUCING a place, person, or concept (e.g. Fort Necessity, the Royal Commission, Valley Forge), give one phrase of context so a general viewer isn't lost—e.g. "at a muddy frontier outpost the French had surrounded" or "the officer rank that would grant him respect in London."
  * AVOID name-dropping without context: Don't assume viewers know who General Braddock is, what the Newburgh Conspiracy was, or why a given moment is pivotal—briefly establish it.
  * CONSEQUENCES: When something major happens, briefly note what it changes or what would have happened otherwise, so the event has weight.
- EMOTIONAL ENGAGEMENT - Pull the viewer into the story by making them FEEL what the character feels:
  * Don't just state events - describe how they FEEL. What's the emotional weight? What's the internal experience?
  * Show what's at stake emotionally, not just factually. What does this moment mean to them personally?
  * Use specific emotional details: physical sensations (hands shaking, heart racing), internal thoughts, fears, hopes
  * Make events feel SIGNIFICANT by connecting them to human feelings - isolation, fear, determination, triumph, despair
  * Help the viewer understand not just WHAT happened, but how it FELT to experience it
- NARRATION TONE MUST MATCH SCENE EMOTION (CRITICAL):
  * The scene's "emotion" field determines narration tone/style - word choice, sentence structure, and pacing
  * Examples: "desperate" → urgent, short sentences, anxious language; "triumphant" → elevated language, confident rhythm; "contemplative" → slower pace, reflective word choice; "tense" → clipped sentences, heightened awareness; "somber" → measured pace, weighty words
  * Match the narration's emotional tone to the scene's emotion - if emotion is "desperate", the narration itself should feel urgent and anxious
- Think like a YouTuber telling a compelling story: show the viewer what happened, why it mattered, and how it FELT - make them experience the emotions alongside the character, and match your narration tone to reflect the scene's emotion
- ABSOLUTELY NO META REFERENCES - Do NOT mention:
  * "Chapter" or "chapters" - viewers don't know about chapters
  * "In this video" or "In this documentary" 
  * "As we'll see" or "Later in this story" - just tell the story
  * "Let me tell you" or "I want to show you" - just narrate directly
  * Any production elements, prompts, outlines, or behind-the-scenes information
  * References to "the script" or "this episode" or "this part"
- Write as if you're naturally telling a story, not referencing the structure behind it"""
    
    if script_type == "biopic":
        import prompt_builders
        base_style += f"\n\n{prompt_builders.get_biopic_audience_profile()}"
    
    if is_short:
        base_style += "\n- 1-2 sentences per scene (~8-12 seconds when spoken) - shorts need to be concise but detailed"
    else:
        base_style += "\n- 2-4 sentences per scene (~12-24 seconds of narration). Use 3-4 sentences when a scene needs more context: new locations, new characters, pivotal moments, or complex situations. Prioritize clarity and depth over brevity—viewers should never be lost."
    
    base_style += """
- Pack MAXIMUM information into minimum words
- CRITICAL: This is SPOKEN narration for text-to-speech. Do NOT include:
  * Film directions like "Smash cut—", "Hard cut to", "Cut to:", "Fade in:"
  * Camera directions like "Close-up of", "Wide shot:", "Pan to"
  * Any production/editing terminology
  Write ONLY words that should be spoken by a narrator's voice.
- QUOTES (CRITICAL FOR TTS): Use quotes ONLY around proper nouns you want to emphasize (e.g. titles of works, key terms). Otherwise do NOT use any quotes—they will mess up the text-to-speech. No quoted dialogue, no quoted phrases for emphasis, no scare quotes."""
    
    return base_style


def get_shared_scene_flow_instructions() -> str:
    """Shared scene-to-scene flow instructions."""
    return """SCENE-TO-SCENE FLOW - SEAMLESS JOURNEY (CRITICAL):
- Create a SEAMLESS JOURNEY through the video - scenes should feel CONNECTED, not like consecutive pieces of disjoint information (A and B and C)
- Each scene should BUILD ON the previous scene - reference what came before naturally, show how events connect, demonstrate cause-and-effect
- Avoid feeling like a list of facts - instead, create a flowing narrative where each scene grows from the last
- Connect scenes through:
  * Natural cause-and-effect: Show how one event leads to the next
  * Continuation of themes: Reference recurring threads, motifs, or plot elements from earlier scenes
  * Emotional progression: Build on the emotional journey established in previous scenes
  * Logical progression: Each scene should feel like the natural next step in the story
- Reference what came before when relevant - don't treat each scene as if it exists in isolation
- Use WHY/WHAT interleaving to create natural connections - WHY scenes set up questions that WHAT scenes answer, creating a seamless flow
- DO NOT use made-up temporal transitions like "Days later...", "Later that week...", "That afternoon...", "The next morning...", "Weeks passed...", "Meanwhile..."
- Instead, let the story flow through what actually happened and how events connect: "The paper is published. Physicists worldwide take notice."
- Each scene should feel inevitable because of what came before, not because of a transition phrase
- Build narrative momentum through actual events and their connections, not filler words
- CRITICAL: SMOOTH EMOTIONAL TRANSITIONS - The emotion field and narration_instructions must flow smoothly between scenes:
  * Emotions should only change GRADUALLY from one scene to the next - avoid dramatic jumps (e.g., don't go from "calm" to "terrified" in one scene)
  * Build emotional intensity gradually: "uneasy" → "tense" → "anxious" → "fearful" → "terrified" (not "calm" → "terrified")
  * If the previous scene was "contemplative", the next scene might be "thoughtful" or "reflective" or "somber" - not "urgent" or "excited"
  * If the previous scene was "tense", the next scene might be "anxious" or "uneasy" or "worried" - not "triumphant" or "calm"
  * Consider the emotional arc: each scene's emotion should be a natural progression from the previous scene's emotion
  * narration_instructions should also transition smoothly - if previous scene was "Focus on tension", next might be "Focus on anxiety" or "Focus on concern" - not "Focus on panic" (gradual progression)
  * Keep narration_instructions to ONE SENTENCE focusing on a single emotion: "Focus on [emotion]."
  * The narration should not sound completely different from one scene to the next - maintain consistency while allowing subtle emotional shifts
  * Think of emotions as a spectrum: move gradually along the spectrum rather than jumping to opposite ends
- CRITICAL: Each scene must cover DIFFERENT, NON-OVERLAPPING events. Do NOT repeat the same event, moment, or action that was already covered in a previous scene. Each scene should advance the story with NEW information, not re-tell what happened before. If an event was already described in detail, move to its consequences or the next significant event instead of describing it again.
- The goal: When scenes are strung together, they should feel like one continuous, connected story, not separate disconnected pieces."""


def get_shared_examples(content_type: str = "historical") -> str:
    """
    Shared good/bad examples for narration.
    
    Args:
        content_type: "historical" for historical documentaries
    """
    return """EXAMPLES:
BAD: "In the quiet of his study, a revolution was brewing in Einstein's mind."
BAD: "Days later, Einstein would submit his groundbreaking paper."
GOOD: "In 1905, Einstein publishes four papers that redefine physics - including E=mc², proving mass and energy are the same thing. The physics community is stunned. Max Planck, the leading physicist of the era, immediately recognizes the significance and invites Einstein to Berlin." """


def generate_thumbnail(prompt: str, output_path: Path, size: str = "1024x1024", generate_thumbnails: bool = True) -> Path | None:
    """
    Generate a thumbnail image and save it with safety violation handling.
    
    Args:
        prompt: The image prompt
        output_path: Where to save the thumbnail
        size: Image size (default "1024x1024")
        generate_thumbnails: Whether thumbnails should be generated (default True)
    """
    if not generate_thumbnails:
        print(f"[THUMBNAIL] Skipped (--no-thumbnails)")
        return None
    
    full_prompt = prompt + NO_TEXT_CONSTRAINT
    max_attempts = 5
    attempt = 1
    
    def sanitize_prompt_for_safety_simple(prompt: str, violation_type: str = None) -> str:
        """Simple sanitization for thumbnails."""
        safety_instruction = " Safe, appropriate, educational content only. Focus on achievements, intellectual work, and positive moments. Avoid graphic or disturbing imagery."
        
        if violation_type == "self-harm":
            prompt = prompt.replace("suffering", "contemplation")
            prompt = prompt.replace("pain", "challenge")
            prompt = prompt.replace("struggle", "journey")
            prompt = prompt.replace("death", "legacy")
            prompt = prompt.replace("illness", "health challenges")
            safety_instruction = " Safe, appropriate, educational content. Focus on achievements, intellectual work, contemplation, and positive moments. Use symbolic or abstract representation if needed. Avoid any graphic or disturbing imagery."
        
        return prompt + safety_instruction
    
    while attempt <= max_attempts:
        try:
            # Sanitize prompt if this is a retry after safety violation
            current_prompt = full_prompt
            if attempt > 1:
                print(f"[THUMBNAIL] Sanitizing prompt (attempt {attempt}/{max_attempts})...")
                current_prompt = sanitize_prompt_for_safety_simple(full_prompt)
            
            result = llm_utils.generate_image(
                prompt=current_prompt,
                output_path=output_path,
                size=size,
                output_format="jpeg",
                moderation="low",
            )
            
            _compress_thumbnail_if_needed(output_path)
            print(f"[THUMBNAIL] Saved: {output_path}")
            return output_path
            
        except Exception as e:
            error_str = str(e)
            # Check if this is a safety violation error
            if 'safety' in error_str.lower() or 'moderation' in error_str.lower() or 'rejected' in error_str.lower():
                violation_type = None
                if 'self-harm' in error_str.lower():
                    violation_type = "self-harm"
                
                if attempt >= max_attempts:
                    print(f"[WARNING] Failed to generate thumbnail after {max_attempts} attempts due to safety violations: {e}")
                    return None
                
                print(f"[THUMBNAIL] Safety violation detected (attempt {attempt}/{max_attempts}). Sanitizing and retrying...")
                if violation_type:
                    print(f"[THUMBNAIL] Violation type: {violation_type}")
                attempt += 1
                time.sleep(2.0 * attempt)  # Exponential backoff
            else:
                # Non-safety errors: just fail
                print(f"[WARNING] Failed to generate thumbnail: {e}")
                return None
    
    return None


def generate_refinement_diff(original_scenes: list[dict], refined_scenes: list[dict]) -> dict:
    """Generate a detailed diff JSON showing what changed between original and refined scenes."""
    diff_data = {
        "summary": {
            "total_scenes": len(original_scenes),
            "scenes_changed": 0,
            "scenes_unchanged": 0
        },
        "scene_diffs": []
    }
    
    for i, (original, refined) in enumerate(zip(original_scenes, refined_scenes)):
        scene_id = original.get('id', i + 1)
        scene_diff = {
            "scene_id": scene_id,
            "changed": False,
            "fields_changed": [],
            "changes": {}
        }
        
        # Compare each field
        fields_to_compare = ["title", "narration", "image_prompt", "emotion", "year", "scene_type"]
        
        for field in fields_to_compare:
            original_value = original.get(field, '').strip() if isinstance(original.get(field), str) else original.get(field)
            refined_value = refined.get(field, '').strip() if isinstance(refined.get(field), str) else refined.get(field)
            
            if original_value != refined_value:
                scene_diff["changed"] = True
                scene_diff["fields_changed"].append(field)
                scene_diff["changes"][field] = {
                    "original": original_value,
                    "refined": refined_value
                }
        
        if scene_diff["changed"]:
            diff_data["summary"]["scenes_changed"] += 1
        else:
            diff_data["summary"]["scenes_unchanged"] += 1
        
        diff_data["scene_diffs"].append(scene_diff)
    
    return diff_data


def identify_pivotal_moments(scenes: list[dict], subject_name: str, chapter_context: str = None, excluded_scene_ids: set = None, script_type: str = "biopic") -> list[dict]:
    """
    Identify 4-7 most pivotal moments from the scenes.
    
    Pivotal moments are: major breakthroughs, turning points, critical decisions,
    moments that change the story trajectory.
    
    Args:
        scenes: List of scene dictionaries
        subject_name: Name of the subject (person, character, region, etc.)
        chapter_context: Optional chapter context string
        excluded_scene_ids: Set of scene IDs to exclude from pivotal moment identification (e.g., chapter 1 scenes, CTA scene)
        
    Returns:
        list of dicts with 'scene_id' and 'justification' (why it's pivotal)
    """
    if not scenes:
        return []
    
    scenes_json = json.dumps(scenes, indent=2, ensure_ascii=False)
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    # Build exclusion note if there are excluded scenes
    exclusion_note = ""
    if excluded_scene_ids:
        excluded_list = sorted(excluded_scene_ids)
        exclusion_note = f"""
CRITICAL EXCLUSION: Do NOT select scenes from chapter 1 or the CTA transition scene as pivotal moments.
Excluded scene IDs: {excluded_list}
These scenes are excluded because:
- Chapter 1 is the hook/preview chapter and should not have significance scenes
- The CTA scene is a transition scene and should not have significance scenes
Only select pivotal moments from scenes AFTER chapter 1 and the CTA scene.
"""
    
    identification_prompt = f"""Analyze these scenes from a documentary about {subject_name} and identify the 4-7 MOST PIVOTAL MOMENTS.

CURRENT SCENES (JSON):
{scenes_json}
{context_info}
{exclusion_note}
PIVOTAL MOMENTS are:
- Major breakthroughs or discoveries that change everything
- Turning points where the story shifts direction
- Critical decisions that reshape the narrative
- Moments that fundamentally change the character's journey or the story's trajectory
- Moments where the consequences are so significant they reshape what comes after

Your task: Identify 4-7 scenes (by scene ID) that represent the most pivotal moments in this story.

For each pivotal moment, explain:
- Why this moment is pivotal (what makes it a turning point?)
- How it changes the story trajectory
- What makes it significant to the overall narrative

[
  {{
    "scene_id": 12,
    "justification": "This is the moment where the discovery changes everything - it's not just an achievement, it rewrites the rules of physics and shifts the entire scientific paradigm."
  }},
  ...
]

Return 4-7 pivotal moments only. Be selective - only the moments that are truly game-changers."""

    try:
        audience_note = f" {prompt_builders.get_biopic_audience_profile()}" if script_type == "biopic" else ""
        content = llm_utils.generate_text(
            messages=[
                {"role": "system", "content": f"You are an expert story analyst who identifies pivotal moments in narratives. You understand what makes a moment truly significant and game-changing. CRITICAL: Do NOT select scenes from chapter 1 (the hook/preview chapter) or the CTA transition scene as pivotal moments - these scenes should not have significance scenes inserted after them. Only select pivotal moments from scenes after chapter 1 and the CTA scene.{audience_note}"},
                {"role": "user", "content": identification_prompt}
            ],
            temperature=0.5,
            response_json_schema=PIVOTAL_MOMENTS_SCHEMA,
            provider=llm_utils.get_provider_for_step("REFINEMENT"),
        )
        pivotal_moments = json.loads(clean_json_response(content))
        
        if not isinstance(pivotal_moments, list):
            print(f"[PIVOTAL MOMENTS] WARNING: Expected array, got {type(pivotal_moments)}. Returning empty list.")
            return []
        
        # Validate that scene IDs exist and are not excluded
        scene_ids = {scene.get('id') for scene in scenes}
        if excluded_scene_ids is None:
            excluded_scene_ids = set()
        valid_moments = []
        for moment in pivotal_moments:
            scene_id = moment.get('scene_id')
            if scene_id not in scene_ids:
                print(f"[PIVOTAL MOMENTS] WARNING: Scene ID {scene_id} not found in scenes, skipping.")
                continue
            if scene_id in excluded_scene_ids:
                print(f"[PIVOTAL MOMENTS] WARNING: Scene ID {scene_id} is excluded (chapter 1 or CTA scene), skipping.")
                continue
            valid_moments.append(moment)
        
        return valid_moments[:4]  # Limit to 4 maximum
        
    except Exception as e:
        print(f"[PIVOTAL MOMENTS] WARNING: Failed to identify pivotal moments ({e}). Continuing without significance scenes.")
        return []


def generate_significance_scene(pivotal_scene: dict, next_scene: dict | None, subject_name: str, justification: str, chapter_context: str = None, script_type: str = "biopic") -> dict:
    """
    Generate a significance scene that explains why a pivotal moment matters.
    
    Args:
        pivotal_scene: The scene that was identified as pivotal
        next_scene: The scene that comes after the pivotal scene (for context)
        subject_name: Name of the subject (person, character, region, etc.)
        justification: Why this moment is pivotal (from identify_pivotal_moments)
        chapter_context: Optional chapter context
        
    Returns:
        New significance scene dict
    """
    pivotal_id = pivotal_scene.get('id', 0)
    pivotal_title = pivotal_scene.get('title', '')
    pivotal_narration = pivotal_scene.get('narration', '')
    pivotal_image_prompt = pivotal_scene.get('image_prompt', '')
    
    # Extract age_phrase from pivotal scene's image_prompt early (before LLM call)
    age_phrase = None
    if pivotal_image_prompt:
        age_match = re.search(rf'(\d+)[-\s]year[-\s]old\s+{re.escape(subject_name)}|{re.escape(subject_name)}[,\s]+(\d+)[-\s]year[-\s]old|{re.escape(subject_name)}\s+in\s+(?:his|her|their)\s+(\d+)s|{re.escape(subject_name)}\s+at\s+age\s+(\d+)', pivotal_image_prompt, re.IGNORECASE)
        if age_match:
            age_num = age_match.group(1) or age_match.group(2) or age_match.group(3) or age_match.group(4)
            if 'year-old' in age_match.group(0):
                age_phrase = f"{age_num}-year-old {subject_name}"
            elif 'in his' in age_match.group(0) or 'in her' in age_match.group(0) or 'in their' in age_match.group(0):
                age_phrase = f"{subject_name} in his {age_num}s"
            else:
                age_phrase = f"{subject_name} at age {age_num}"
    
    age_instruction = f"\nCRITICAL - AGE: The pivotal scene's image_prompt is: \"{pivotal_image_prompt[:300]}...\". Extract the age information from this (look for phrases like \"26-year-old\", \"in his 40s\", \"at age 20\") and include the SAME age description in your image_prompt. The significance scene should show {subject_name} at the SAME age as in the pivotal moment."
    
    next_context = ""
    if next_scene:
        next_title = next_scene.get('title', '')
        next_context = f"\nThe next scene after this will be: \"{next_title}\" - keep this in mind for flow."
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    narration_instructions_desc = "ONE SENTENCE: Focus on a single emotion from the emotion field. Example: 'Focus on contemplation.' or 'Focus on thoughtfulness.' Keep it simple - just the emotion to emphasize."
    
    significance_prompt = f"""Generate a SIGNIFICANCE SCENE that comes immediately after Scene {pivotal_id}: "{pivotal_title}"

PIVOTAL MOMENT (Scene {pivotal_id}):
Title: "{pivotal_title}"
Narration: "{pivotal_narration}"
Image Prompt: "{pivotal_image_prompt}"

WHY THIS MOMENT IS PIVOTAL:
{justification}
{age_instruction}
{next_context}
{context_info}

YOUR TASK: Create a scene that explains WHY this pivotal moment matters - its significance to the overall story.

This scene should:
- Explain WHY this moment is pivotal - its significance to the broader story
- Show HOW this moment changes or reshapes what comes after
- Make viewers FEEL the weight and importance of this moment
- Connect this moment to the larger narrative - how it fits into the overall story
- Use 1-3 sentences (let the moment's importance determine length - very important moments can be longer)
- Flow naturally from the pivotal moment and into what comes next
- Write from the YouTuber's perspective - natural storytelling

This is NOT a repeat of the pivotal moment - it's an EXPLANATION of WHY that moment matters.

CRITICAL: Make this scene feel significant and weighty. The viewer should understand: "This is why this moment changed everything."

{{
  "title": "2-5 word title about the significance",
  "narration": "1-3 sentences explaining why this pivotal moment matters - its significance, impact, and what it changed. Make viewers FEEL the weight.",
  "scene_type": "WHAT" - This scene explains the significance and delivers information about why the moment matters,
  "image_prompt": "Visual that reflects the significance and weight of this moment - contemplative, monumental, or reflective mood. Use the same year/time period AND THE SAME AGE as the pivotal moment (extract age from pivotal scene's image_prompt above). Include {subject_name} at the exact same age as in the pivotal scene. Match the visual style of the documentary. 16:9 cinematic",
  "emotion": "contemplative" or "reflective" or "weighty" or "monumental" - the emotion should reflect the significance being explained,
  "narration_instructions": "{narration_instructions_desc} - REQUIRED FIELD",
  "year": Same as pivotal moment (use scene {pivotal_id}'s year)
}}

CRITICAL: The "narration_instructions" field is REQUIRED and must be included in your JSON response. Do not omit it."""

    narration_instructions_note = "CRITICAL: The 'narration_instructions' field is REQUIRED in your JSON response. It must be ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on contemplation.' or 'Focus on thoughtfulness.' The narration_instructions should flow smoothly from the previous scene's narration_instructions (gradual emotion progression). DO NOT omit this field."
    
    try:
        audience_note = f" {prompt_builders.get_biopic_audience_profile()}" if script_type == "biopic" else ""
        content = llm_utils.generate_text(
            messages=[
                {"role": "system", "content": f"You create powerful scenes that explain why pivotal moments matter. You help viewers understand the significance and weight of game-changing moments. {narration_instructions_note}{audience_note}"},
                {"role": "user", "content": significance_prompt}
            ],
            temperature=0.7,
            response_json_schema=SCENE_SCHEMA,
            provider=llm_utils.get_provider_for_step("REFINEMENT"),
        )
        scene = json.loads(clean_json_response(content))
        
        if not isinstance(scene, dict):
            raise ValueError(f"Expected dict, got {type(scene)}")
        
        # Ensure all required fields are present, with fallbacks if missing
        if 'title' not in scene:
            scene['title'] = "Why This Moment Matters"
        if 'scene_type' not in scene:
            scene['scene_type'] = "WHAT"  # Significance scenes explain why something matters - they deliver information
        if scene.get('scene_type') not in ['WHY', 'WHAT']:
            scene['scene_type'] = "WHAT"  # Default to WHAT if invalid
        if 'narration' not in scene:
            scene['narration'] = f"This moment changes everything - its significance reshapes the entire story."
        if 'emotion' not in scene:
            scene['emotion'] = "contemplative"  # Default emotion for significance scenes
        # CRITICAL: Ensure narration_instructions is always present
        if 'narration_instructions' not in scene or not scene.get('narration_instructions'):
            # Generate fallback narration_instructions based on emotion and script type
            emotion = scene.get('emotion', 'contemplative')
            scene['narration_instructions'] = f"Focus on {emotion}."
            print(f"[SIGNIFICANCE SCENE] WARNING: LLM did not provide narration_instructions, generated fallback: {scene['narration_instructions']}")
        if 'image_prompt' not in scene:
            # Use the extracted age_phrase if available, otherwise use subject name
            if not age_phrase:
                age_phrase = subject_name
            
            scene['image_prompt'] = f"Reflective, contemplative scene showing the weight of this pivotal moment, {age_phrase}, {pivotal_scene.get('year', 'same period')}, 16:9 cinematic"
        
        # Post-process: Ensure the generated image_prompt includes the age if it was extracted
        if age_phrase and age_phrase != subject_name:
            # Check if the generated image_prompt already includes age information
            generated_prompt = scene.get('image_prompt', '')
            if not re.search(rf'{re.escape(age_phrase.split()[0])}|{re.escape(subject_name)}.*\d+.*year|{re.escape(subject_name)}.*age\s+\d+', generated_prompt, re.IGNORECASE):
                # Age not found in generated prompt, prepend or insert it
                # Try to insert before the subject name or at the beginning
                if subject_name.lower() in generated_prompt.lower():
                    # Replace subject name with age_phrase
                    generated_prompt = re.sub(rf'\b{re.escape(subject_name)}\b', age_phrase, generated_prompt, count=1, flags=re.IGNORECASE)
                    scene['image_prompt'] = generated_prompt
                else:
                    # Insert age_phrase at the beginning
                    scene['image_prompt'] = f"{age_phrase}, {generated_prompt}"
        if 'emotion' not in scene:
            scene['emotion'] = "contemplative"
        if 'year' not in scene:
            scene['year'] = pivotal_scene.get('year', 'same period')
        if 'kenburns_pattern' not in scene:
            scene['kenburns_pattern'] = "zoom_in"
        # kenburns_intensity is required in SCENE_SCHEMA

        return scene
        
    except Exception as e:
        print(f"[SIGNIFICANCE SCENE] WARNING: Failed to generate significance scene for Scene {pivotal_id} ({e}).")
        # Return a simple fallback scene
        fallback_scene = {
            "title": "Why This Moment Matters",
            "narration": f"This moment is pivotal because {justification[:100]}... It changes the trajectory of the entire story.",
            "scene_type": "WHAT",  # Significance scenes explain why something matters - they deliver information
            "image_prompt": f"Reflective, contemplative scene showing the significance of this moment, {pivotal_scene.get('year', 'same period')}, 16:9 cinematic",
            "emotion": "contemplative",
            "year": pivotal_scene.get('year', 'same period'),
            "kenburns_pattern": "zoom_in",
            "kenburns_intensity": "medium",
        }
        return fallback_scene


def check_and_add_missing_storyline_scenes(scenes: list[dict], subject_name: str, chapter_context: str = None, max_scenes: int = None, script_type: str = "biopic") -> list[dict]:
    """
    PASS 1: Check for hanging storylines and add scenes to wrap them up.
    Uses LLM to identify storylines that were introduced but not resolved, then generates scenes to complete them.
    
    Args:
        scenes: List of scene dictionaries
        subject_name: Name of the subject
        chapter_context: Optional chapter context string
        max_scenes: Maximum number of scenes allowed (e.g., 5 for shorts). If None, no limit.
        
    Returns:
        Updated list of scenes with missing storyline scenes added in correct chronological positions
    """
    
    if not scenes:
        return scenes
    
    # If we're already at or over the max, don't add more scenes
    if max_scenes is not None and len(scenes) >= max_scenes:
        return scenes
    
    scenes_json = json.dumps(scenes, indent=2, ensure_ascii=False)
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    doc_type = "documentary"
    hanging_focus = "storylines, plot threads, or events that were introduced but NOT resolved or completed"
    example = "engagement mentioned but marriage never shown, conflict introduced but resolution never shown"
    
    hanging_storylines_prompt = f"""Analyze these scenes from a {doc_type} about {subject_name} and identify HANGING STORYLINES - {hanging_focus}.

CURRENT SCENES (JSON):
{scenes_json}
{context_info}

HANGING STORYLINES are:
- Events or situations that were introduced but never shown to completion (e.g., {example})
- Plot threads that were set up but left hanging (e.g., relationship started but never developed, problem introduced but never solved)
- Storylines from the chapter's key_events or plot_developments that were mentioned but not fully covered

Your task: Identify ALL hanging storylines and determine:
1. What storyline/event was introduced but not completed
2. What scene(s) need to be added to complete it
3. Where chronologically these scenes should be inserted (after which scene ID, based on year/timeline)

For each hanging storyline, provide:
- The scene ID where it was introduced
- A description of what's missing (what needs to happen to complete the storyline)
- The year when the completion should occur
- Where to insert the new scene(s) (after which scene ID)

[
  {{
    "storyline_description": "Brief description of the hanging storyline (e.g., 'Engagement to Martha mentioned but marriage never shown')",
    "introduced_in_scene_id": 5,
    "missing_completion": "What needs to happen to complete this storyline (e.g., 'Show the marriage ceremony in 1886')",
    "completion_year": 1886,
    "insert_after_scene_id": 7,
    "scene_type": "WHAT" or "WHY" - typically "WHAT" since these are completing storylines
  }},
  ...
]

If there are NO hanging storylines, return an empty array []."""

    try:
        audience_note = f" {prompt_builders.get_biopic_audience_profile()}" if script_type == "biopic" else ""
        content = llm_utils.generate_text(
            messages=[
                {"role": "system", "content": f"You are an expert story analyst who identifies incomplete storylines in narratives. You understand narrative structure and ensure all introduced plot threads are resolved.{audience_note}"},
                {"role": "user", "content": hanging_storylines_prompt}
            ],
            temperature=0.5,
            response_json_schema=HANGING_STORYLINES_SCHEMA,
            provider=llm_utils.get_provider_for_step("REFINEMENT_HANGING_STORYLINES"),
        )
        hanging_storylines = json.loads(clean_json_response(content))
        
        if not isinstance(hanging_storylines, list):
            print(f"[STORYLINE CHECK] WARNING: Expected array, got {type(hanging_storylines)}. Skipping storyline completion.")
            return scenes
        
        if not hanging_storylines:
            print(f"[STORYLINE CHECK] ✓ No hanging storylines found")
            return scenes
        
        print(f"[STORYLINE CHECK] Found {len(hanging_storylines)} hanging storyline(s) to complete")
        
        # Sort by insert_after_scene_id (descending) so we can insert from end to start
        hanging_storylines.sort(key=lambda x: x.get('insert_after_scene_id', 0), reverse=True)
        
        updated_scenes = [scene.copy() for scene in scenes]
        inserted_count = 0
        
        for storyline in hanging_storylines:
            insert_after_id = storyline.get('insert_after_scene_id')
            missing_completion = storyline.get('missing_completion', '')
            completion_year = storyline.get('completion_year', '')
            scene_type = storyline.get('scene_type', 'WHAT')
            
            # Find the scene to insert after
            scene_id_to_index = {scene.get('id'): i for i, scene in enumerate(updated_scenes)}
            if insert_after_id not in scene_id_to_index:
                print(f"[STORYLINE CHECK]   WARNING: Scene ID {insert_after_id} not found, skipping storyline: {storyline.get('storyline_description', '')}")
                continue
            
            insert_index = scene_id_to_index[insert_after_id]
            scene_after = updated_scenes[insert_index]
            scene_before = updated_scenes[insert_index - 1] if insert_index > 0 else None
            
            narration_instructions_desc = "ONE SENTENCE: Focus on a single emotion from the emotion field. Example: 'Focus on contemplation.' or 'Focus on thoughtfulness.'"
            
            # Generate scene to complete the storyline
            completion_prompt = f"""Generate a scene to complete a hanging storyline in a documentary about {subject_name}.

HANGING STORYLINE:
{storyline.get('storyline_description', '')}

WHAT'S MISSING:
{missing_completion}

CONTEXT:
- This scene should be inserted after Scene {insert_after_id}: "{scene_after.get('title', '')}"
- Year: {completion_year}
- Previous scene: {scene_before.get('title', '') if scene_before else 'N/A'}
- Next scene: {scene_after.get('title', '')}

YOUR TASK: Create a scene that completes this storyline naturally and chronologically.

This scene should:
- Complete the hanging storyline in a satisfying way
- Flow naturally from the previous scene
- Connect smoothly to the next scene
- Be chronologically accurate (year: {completion_year})
- Write from the YouTuber's perspective - natural storytelling
- Use 1-2 sentences (brief but complete)

{{
  "title": "2-5 word title",
  "narration": "1-2 sentences that complete the hanging storyline naturally",
  "scene_type": "{scene_type}",
  "image_prompt": "Visual description appropriate for this moment, including {subject_name}'s age at this time ({completion_year}), 16:9 cinematic",
  "emotion": "Appropriate emotion for this moment",
  "narration_instructions": "{narration_instructions_desc}",
  "year": {completion_year}
}}"""

            audience_note = f" {prompt_builders.get_biopic_audience_profile()}" if script_type == "biopic" else ""
            system_content = f"You create scenes that complete hanging storylines in documentaries. You ensure narrative completeness and chronological accuracy. CRITICAL: narration_instructions should match the scene's emotion field with brief delivery guidance. Keep it simple - just follow the emotion with some context.{audience_note}"
            
            try:
                completion_content = llm_utils.generate_text(
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": completion_prompt}
                    ],
                    temperature=0.7,
                    response_json_schema=SCENE_SCHEMA,
                    provider=llm_utils.get_provider_for_step("REFINEMENT_HANGING_STORYLINES"),
                )
                new_scene = json.loads(clean_json_response(completion_content))
                
                if not isinstance(new_scene, dict):
                    print(f"[STORYLINE CHECK]   WARNING: Expected dict for completion scene, got {type(new_scene)}. Skipping.")
                    continue
                
                # Ensure required fields
                if 'scene_type' not in new_scene:
                    new_scene['scene_type'] = scene_type
                if new_scene.get('scene_type') not in ['WHY', 'WHAT']:
                    new_scene['scene_type'] = 'WHAT'
                if 'year' not in new_scene:
                    new_scene['year'] = completion_year
                if 'emotion' not in new_scene:
                    new_scene['emotion'] = "contemplative"  # Default emotion
                
                # CRITICAL: Ensure narration_instructions is always present (fix if missing)
                if 'narration_instructions' not in new_scene or not new_scene.get('narration_instructions'):
                    emotion = new_scene.get('emotion', 'contemplative')
                    new_scene['narration_instructions'] = f"Focus on {emotion}."
                    print(f"[STORYLINE CHECK] WARNING: Generated scene missing narration_instructions, generated fallback: {new_scene['narration_instructions']}")
                
                # Mark as storyline completion scene and inherit chapter_num for music track alignment
                new_scene['is_storyline_completion'] = True
                new_scene['chapter_num'] = scene_after.get('chapter_num', 1)
                if 'kenburns_pattern' not in new_scene:
                    new_scene['kenburns_pattern'] = "zoom_in"
                # kenburns_intensity is required in SCENE_SCHEMA
                
                # Set temporary ID (will be renumbered later)
                new_scene['id'] = insert_after_id + 0.5
                
                # Check if we've reached the max scenes limit
                if max_scenes is not None and len(updated_scenes) >= max_scenes:
                    print(f"[STORYLINE CHECK]   • Reached max scenes limit ({max_scenes}), skipping remaining storylines")
                    break
                
                # Insert the scene
                updated_scenes.insert(insert_index + 1, new_scene)
                inserted_count += 1
                print(f"[STORYLINE CHECK]   • Added scene to complete: {storyline.get('storyline_description', '')[:60]}...")
                
            except Exception as e:
                print(f"[STORYLINE CHECK]   WARNING: Failed to generate completion scene for storyline ({e}). Skipping.")
                continue
        
        # Renumber all scene IDs
        for i, scene in enumerate(updated_scenes):
            scene['id'] = i + 1
        
        # Cap at max_scenes if specified
        if max_scenes is not None and len(updated_scenes) > max_scenes:
            print(f"[STORYLINE CHECK]   • Capping scenes at {max_scenes} (was {len(updated_scenes)})")
            updated_scenes = updated_scenes[:max_scenes]
            # Renumber again after capping
            for i, scene in enumerate(updated_scenes):
                scene['id'] = i + 1
        
        print(f"[STORYLINE CHECK] ✓ Added {inserted_count} scene(s) to complete hanging storylines (total: {len(updated_scenes)} scenes)")
        return updated_scenes
        
    except Exception as e:
        print(f"[STORYLINE CHECK] WARNING: Failed to check for hanging storylines ({e}). Continuing without storyline completion.")
        return scenes


def check_and_add_historian_depth_scenes(scenes: list[dict], subject_name: str, chapter_context: str = None,
                                        max_scenes: int = None, script_type: str = "biopic") -> list[dict]:
    """
    From the perspective of a well-studied professional historian, identify gaps in the documentary
    and add scenes to fill in missing details, context, significance, or additional meaning.
    Only runs for biopic scripts.
    """
    if script_type != "biopic":
        return scenes

    if max_scenes is not None and len(scenes) >= max_scenes:
        return scenes

    scenes_json = json.dumps(scenes, indent=2, ensure_ascii=False)

    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"

    historian_prompt = f"""You are a well-studied professional historian who has dedicated years to the life and era of {subject_name}. You are reviewing a documentary script and identifying where it could use MORE DEPTH.

CURRENT SCENES (JSON):
{scenes_json}
{context_info}

Your task: Identify gaps where the documentary would benefit from additional scenes. Consider:
- Important missing details that historians would expect to see (key facts, dates, figures, decisions)
- Missing context (background events, political climate, cultural factors that shaped the moment)
- Unexplored significance (why a moment mattered more than is shown, long-term impact, historiographical debate)
- Additional meaning (human dimension, emotional weight, what it meant for those involved)
- Connections or parallels that would deepen understanding
- Moments that feel rushed or glossed over when they deserve more attention

Only suggest additions that:
1. Naturally fit the narrative flow
2. Can be answered/satisfied by the documentary with real historical content
3. Add genuine depth—not filler or repetition
4. Would make a fellow historian nod in approval

For each gap you identify, provide:
- "gap_description": Brief description of what's missing
- "what_to_add": What the new scene should cover (the specific content to add)
- "insert_after_scene_id": After which scene ID to insert (chronologically appropriate)
- "year": When this addition takes place
- "scene_type": "WHY" or "WHAT" - typically "WHAT" for factual depth, "WHY" for significance/meaning
- "rationale": Why a historian would consider this addition valuable

Return a JSON array. If there are no meaningful gaps to fill, return an empty array [].

[
  {{
    "gap_description": "Brief description of the gap",
    "what_to_add": "Specific content the new scene should cover",
    "insert_after_scene_id": 12,
    "year": 1905,
    "scene_type": "WHAT",
    "rationale": "Why this adds depth from a historian's perspective"
  }},
  ...
]"""

    try:
        audience_note = f" {prompt_builders.get_biopic_audience_profile()}" if script_type == "biopic" else ""
        content = llm_utils.generate_text(
            messages=[
                {"role": "system", "content": f"You are a well-studied professional historian with deep expertise in your subject. You identify where narratives need more depth, context, and significance. You add value through substance—never filler. Your additions would make a documentary feel more authoritative and satisfying to a knowledgeable viewer.{audience_note}"},
                {"role": "user", "content": historian_prompt}
            ],
            temperature=0.5,
            response_json_schema=HISTORIAN_DEPTH_ADDITIONS_SCHEMA,
            provider=llm_utils.get_provider_for_step("REFINEMENT_HISTORIAN_DEPTH"),
        )
        historian_additions = json.loads(clean_json_response(content))

        if not isinstance(historian_additions, list):
            print(f"[HISTORIAN DEPTH] WARNING: Expected array, got {type(historian_additions)}. Skipping.")
            return scenes

        if not historian_additions:
            print(f"[HISTORIAN DEPTH] ✓ No depth additions suggested")
            for scene in scenes:
                if 'is_historian_scene' not in scene:
                    scene['is_historian_scene'] = False
            return scenes

        print(f"[HISTORIAN DEPTH] Found {len(historian_additions)} addition(s) to add depth")

        # Sort by insert_after_scene_id (descending) so we insert from end to start
        historian_additions.sort(key=lambda x: x.get('insert_after_scene_id', 0), reverse=True)

        updated_scenes = [scene.copy() for scene in scenes]
        inserted_count = 0

        for addition in historian_additions:
            insert_after_id = addition.get('insert_after_scene_id')
            what_to_add = addition.get('what_to_add', '')
            year = addition.get('year', '')
            scene_type = addition.get('scene_type', 'WHAT')

            scene_id_to_index = {scene.get('id'): i for i, scene in enumerate(updated_scenes)}
            if insert_after_id not in scene_id_to_index:
                print(f"[HISTORIAN DEPTH]   WARNING: Scene ID {insert_after_id} not found, skipping: {addition.get('gap_description', '')[:50]}...")
                continue

            insert_index = scene_id_to_index[insert_after_id]
            scene_after = updated_scenes[insert_index]
            scene_before = updated_scenes[insert_index - 1] if insert_index > 0 else None

            # Build full script context (compact: id, title, narration, year)
            script_lines = []
            for s in updated_scenes:
                script_lines.append(f"Scene {s.get('id')}: \"{s.get('title', '')}\" ({s.get('year', '')}) — {s.get('narration', '')}")

            # Full content of adjacent scenes for flow
            prev_block = ""
            if scene_before:
                prev_block = f"""
PREVIOUS SCENE (your new scene must flow FROM this—your first sentence must connect to its ending):
  ID: {scene_before.get('id')}
  Title: {scene_before.get('title', '')}
  Year: {scene_before.get('year', '')}
  Emotion: {scene_before.get('emotion', '')}
  Narration: {scene_before.get('narration', '')}
"""
            next_block = f"""
NEXT SCENE (your new scene must flow INTO this—set up or connect to what follows):
  ID: {scene_after.get('id')}
  Title: {scene_after.get('title', '')}
  Year: {scene_after.get('year', '')}
  Emotion: {scene_after.get('emotion', '')}
  Narration: {scene_after.get('narration', '')}
"""

            prev_emotion = scene_before.get('emotion', '') if scene_before else ''
            next_emotion = scene_after.get('emotion', '')
            if prev_emotion and next_emotion:
                emotion_note = f"Your emotion should fall between previous ({prev_emotion}) and next ({next_emotion}) for gradual progression."
            elif next_emotion:
                emotion_note = f"Your emotion should transition smoothly toward the next scene's emotion ({next_emotion})."
            else:
                emotion_note = ""

            depth_prompt = f"""Generate a scene to add depth to a documentary about {subject_name}.

HISTORIAN'S ADDITION:
{addition.get('gap_description', '')}

WHAT TO ADD:
{what_to_add}

RATIONALE: {addition.get('rationale', '')}

BRIDGE REQUIREMENTS (CRITICAL):
- Your first sentence MUST connect to or reference the last sentence/idea of the previous scene. Do not start with a standalone topic—bridge from what came before.
- Your last sentence MUST set up or lead into the first sentence of the next scene. Tee up the topic, theme, or transition that the next scene will pick up.
- {emotion_note}

{get_shared_scene_flow_instructions()}

FULL SCRIPT CONTEXT (for narrative flow):
{chr(10).join(script_lines)}
{prev_block}
{next_block}
---

Your new scene will be inserted BETWEEN the previous and next scenes above. Create a scene that adds this depth naturally—flowing from the previous scene and into the next. Be substantive and authoritative. 1-3 sentences.

{{
  "title": "2-5 word title",
  "narration": "1-3 sentences adding the missing depth",
  "scene_type": "{scene_type}",
  "image_prompt": "Visual description, including {subject_name}'s age at this time ({year}), 16:9 cinematic",
  "emotion": "Appropriate emotion between previous and next for gradual progression",
  "narration_instructions": "Focus on [emotion from above].",
  "year": {json.dumps(year)}
}}"""

            try:
                completion_content = llm_utils.generate_text(
                    messages=[
                        {"role": "system", "content": f"You create scenes that add historical depth to documentaries. Substantive, authoritative, and flowing naturally. {prompt_builders.get_biopic_audience_profile()}"},
                        {"role": "user", "content": depth_prompt}
                    ],
                    temperature=0.6,
                    response_json_schema=SCENE_SCHEMA,
                    provider=llm_utils.get_provider_for_step("REFINEMENT_HISTORIAN_DEPTH"),
                )
                new_scene = json.loads(clean_json_response(completion_content))

                if not isinstance(new_scene, dict):
                    continue

                if 'scene_type' not in new_scene:
                    new_scene['scene_type'] = scene_type
                if new_scene.get('scene_type') not in ['WHY', 'WHAT']:
                    new_scene['scene_type'] = 'WHAT'
                if 'year' not in new_scene:
                    new_scene['year'] = year
                if 'emotion' not in new_scene:
                    new_scene['emotion'] = "contemplative"
                if 'narration_instructions' not in new_scene or not new_scene.get('narration_instructions'):
                    new_scene['narration_instructions'] = f"Focus on {new_scene.get('emotion', 'contemplative')}."

                new_scene['is_historian_scene'] = True
                new_scene['chapter_num'] = scene_after.get('chapter_num', 1)
                new_scene['id'] = insert_after_id + 0.5
                if 'kenburns_pattern' not in new_scene:
                    new_scene['kenburns_pattern'] = "zoom_in"
                # kenburns_intensity is required in SCENE_SCHEMA

                if max_scenes is not None and len(updated_scenes) >= max_scenes:
                    break

                updated_scenes.insert(insert_index + 1, new_scene)
                inserted_count += 1
                print(f"[HISTORIAN DEPTH]   • Added: {addition.get('gap_description', '')[:55]}...")

            except Exception as e:
                print(f"[HISTORIAN DEPTH]   WARNING: Failed to generate scene ({e}). Skipping.")
                continue

        for i, scene in enumerate(updated_scenes):
            scene['id'] = i + 1
            if 'is_historian_scene' not in scene:
                scene['is_historian_scene'] = False

        if max_scenes is not None and len(updated_scenes) > max_scenes:
            updated_scenes = updated_scenes[:max_scenes]
            for i, scene in enumerate(updated_scenes):
                scene['id'] = i + 1

        print(f"[HISTORIAN DEPTH] ✓ Added {inserted_count} scene(s) for depth (total: {len(updated_scenes)} scenes)")
        return updated_scenes

    except Exception as e:
        print(f"[HISTORIAN DEPTH] WARNING: Failed ({e}). Continuing without historian additions.")
        for scene in scenes:
            if 'is_historian_scene' not in scene:
                scene['is_historian_scene'] = False
        return scenes


def refine_scenes(scenes: list[dict], subject_name: str, is_short: bool = False, chapter_context: str = None, 
                  diff_output_path: Path | None = None, subject_type: str = "person", skip_significance_scenes: bool = False,
                  scenes_per_chapter: int = None, chapter_boundaries: list[tuple[int, int]] = None,
                  script_type: str = "biopic", video_questions: list[str] | None = None,
                  short_video_question: str | None = None, short_narrative_structure: str | None = None,
                  short_music_mood: str | None = None) -> tuple[list[dict], dict]:
    """
    Refine generated scenes through THREE PASSES:
    1. Check for hanging storylines and add scenes to complete them
    2. Identify pivotal moments and add significance scenes
    3. Final refinement for transitions, storytelling, and polish
    
    Args:
        scenes: List of scene dictionaries to refine
        subject_name: Name of the subject (person, character, region, etc.)
        is_short: Whether these are short video scenes
        chapter_context: Optional chapter context string
        diff_output_path: Optional path to save refinement diff JSON
        subject_type: Type of subject - "person", "character", "region", "topic", etc. (for prompt customization)
        skip_significance_scenes: If True, skip identifying pivotal moments and generating significance scenes (for top 10 lists, etc.)
        scenes_per_chapter: Legacy - fixed scenes per chapter (used when chapter_boundaries not provided)
        chapter_boundaries: List of (start_id, end_id) per chapter for variable scene counts. Takes precedence over scenes_per_chapter.
    
    Returns:
        tuple: (refined_scenes, diff_data) - The refined scenes and a diff dict showing what changed
    """
    if not scenes:
        return scenes, {}
    
    # Save original scenes for comparison
    original_scenes = [scene.copy() for scene in scenes]
    
    # PASS 1: Check for hanging storylines and add completion scenes
    # SKIP for shorts (shorts are trailers with loose storylines by design)
    if is_short:
        print(f"\n[REFINEMENT PASS 1] Skipped for shorts (trailers have loose storylines by design)")
        scenes_after_storyline_check = [scene.copy() for scene in scenes]
    else:
        print(f"\n[REFINEMENT PASS 1] Checking for hanging storylines...")
        scenes_after_storyline_check = check_and_add_missing_storyline_scenes(scenes, subject_name, chapter_context, max_scenes=None, script_type=script_type)
    
    # PASS 1.5: For biopic main videos, historian depth pass - add scenes for missing details, context, significance
    scenes_after_historian = [scene.copy() for scene in scenes_after_storyline_check]
    if is_short:
        pass  # No historian pass for shorts
    elif script_type == "biopic":
        print(f"\n[REFINEMENT PASS 1.5] Historian depth: identifying gaps for missing details, context, significance...")
        scenes_after_historian = check_and_add_historian_depth_scenes(
            scenes_after_storyline_check, subject_name, chapter_context, max_scenes=None, script_type=script_type
        )
    else:
        pass  # Only biopic gets historian pass

    # PASS 2: For main videos (not shorts), identify pivotal moments and insert significance scenes
    # SKIP for shorts (trailers don't need significance scenes)
    # UNLESS skip_significance_scenes is True (e.g., for top 10 list videos)
    scenes_after_pivotal = [scene.copy() for scene in scenes_after_historian]
    if is_short:
        print(f"\n[REFINEMENT PASS 2] Skipped for shorts (trailers don't need significance scenes)")
    elif not skip_significance_scenes:
        print(f"\n[REFINEMENT PASS 2] Identifying pivotal moments and adding significance scenes...")
        print(f"[REFINEMENT] Identifying pivotal moments and adding significance scenes (before refinement)...")
        
        # Exclude chapter 1 scenes and first scene of chapter 2 (transition) from pivotal moment identification
        excluded_scene_ids = set()
        if chapter_boundaries and len(chapter_boundaries) >= 2:
            # Use chapter boundaries: exclude all of chapter 1 + first scene of chapter 2
            ch1_start, ch1_end = chapter_boundaries[0]
            ch2_start, _ = chapter_boundaries[1]
            chapter_1_scene_ids = set(range(ch1_start, ch1_end + 1))
            excluded_scene_ids.update(chapter_1_scene_ids)
            excluded_scene_ids.add(ch2_start)  # Transition scene from hook to story
            print(f"[REFINEMENT]   • Excluding chapter 1 scenes (IDs {ch1_start}-{ch1_end}) and transition scene (ID {ch2_start}) from pivotal moment identification")
        elif scenes_per_chapter is not None:
            # Legacy: fixed scenes per chapter
            chapter_1_scene_ids = set(range(1, scenes_per_chapter + 1))
            excluded_scene_ids.update(chapter_1_scene_ids)
            cta_scene_id = scenes_per_chapter + 1
            excluded_scene_ids.add(cta_scene_id)
            print(f"[REFINEMENT]   • Excluding chapter 1 scenes (IDs 1-{scenes_per_chapter}) and CTA scene (ID {cta_scene_id}) from pivotal moment identification")
        
        pivotal_moments = identify_pivotal_moments(scenes_after_pivotal, subject_name, chapter_context, excluded_scene_ids=excluded_scene_ids, script_type=script_type)
        
        if pivotal_moments:
            print(f"[REFINEMENT PASS 2]   • Identified {len(pivotal_moments)} pivotal moment(s)")
            
            # Sort pivotal moments by scene_id (descending) so we can insert from end to start
            # This way, scene IDs remain valid as we insert
            pivotal_moments.sort(key=lambda x: x.get('scene_id', 0), reverse=True)
            
            inserted_scenes_count = 0
            
            # Insert significance scenes after each pivotal moment
            for pivotal_moment in pivotal_moments:
                pivotal_scene_id = pivotal_moment.get('scene_id')
                justification = pivotal_moment.get('justification', '')
                
                # Rebuild scene_id_to_index map (indices may have shifted from previous insertions)
                scene_id_to_index = {scene.get('id'): i for i, scene in enumerate(scenes_after_pivotal)}
                
                if pivotal_scene_id not in scene_id_to_index:
                    print(f"[REFINEMENT PASS 2]   WARNING: Pivotal scene ID {pivotal_scene_id} not found, skipping")
                    continue
                
                pivotal_index = scene_id_to_index[pivotal_scene_id]
                pivotal_scene = scenes_after_pivotal[pivotal_index]
                
                # Get next scene for context
                next_scene = scenes_after_pivotal[pivotal_index + 1] if pivotal_index + 1 < len(scenes_after_pivotal) else None
                
                # Generate significance scene
                significance_scene = generate_significance_scene(
                    pivotal_scene=pivotal_scene,
                    next_scene=next_scene,
                    subject_name=subject_name,
                    justification=justification,
                    chapter_context=chapter_context,
                    script_type=script_type
                )
                
                # CRITICAL: Ensure narration_instructions is present (fix if missing)
                if 'narration_instructions' not in significance_scene or not significance_scene.get('narration_instructions', '').strip():
                    emotion = significance_scene.get('emotion', 'contemplative')
                    significance_scene['narration_instructions'] = f"Focus on {emotion}."
                    print(f"[REFINEMENT PASS 2] WARNING: Generated significance scene missing narration_instructions, generated fallback: {significance_scene['narration_instructions']}")
                
                # Mark this as a significance scene and inherit chapter_num for music track alignment
                significance_scene['is_significance_scene'] = True
                significance_scene['chapter_num'] = pivotal_scene.get('chapter_num', 1)
                
                # Set temporary ID (will be renumbered later)
                significance_scene['id'] = pivotal_scene_id + 0.5  # Temporary ID between pivotal and next
                
                # Insert significance scene after pivotal moment
                insert_index = pivotal_index + 1
                scenes_after_pivotal.insert(insert_index, significance_scene)
                
                inserted_scenes_count += 1
                print(f"[REFINEMENT PASS 2]   • Added significance scene after Scene {pivotal_scene_id}")
            
            # Renumber all scene IDs to be sequential integers
            # Also ensure all scenes have is_significance_scene set (False for non-significance scenes)
            for i, scene in enumerate(scenes_after_pivotal):
                scene['id'] = i + 1
                # Only set to False if not already set to True
                if 'is_significance_scene' not in scene:
                    scene['is_significance_scene'] = False
            
            print(f"[REFINEMENT PASS 2]   • Inserted {inserted_scenes_count} significance scene(s)")
            print(f"[REFINEMENT PASS 2]   • Scene count after pivotal scenes: {len(scenes_after_pivotal)} (was {len(original_scenes)})")
    else:
        scenes_after_pivotal = scenes_after_historian
    
    # PASS 3: Final refinement for transitions, storytelling, and polish
    # For shorts, cap at 5 scenes maximum before final refinement
    if is_short and len(scenes_after_pivotal) > 5:
        print(f"[REFINEMENT PASS 3]   • Shorts mode: capping at 5 scenes before final refinement (was {len(scenes_after_pivotal)})")
        scenes_after_pivotal = scenes_after_pivotal[:5]
        # Renumber scenes
        for i, scene in enumerate(scenes_after_pivotal):
            scene['id'] = i + 1
    
    print(f"\n[REFINEMENT PASS 3] Final refinement: transitions, storytelling, and polish...")
    scenes_before_refinement = [scene.copy() for scene in scenes_after_pivotal]
    print(f"[REFINEMENT PASS 3] Refining {len(scenes_before_refinement)} scenes (including any storyline completion and significance scenes)...")
    
    # Identify significance scenes and historian scenes for transition checking
    significance_scene_info = ""
    historian_scene_info = ""
    if not is_short:
        # Find scenes that are significance scenes (marked with is_significance_scene: true)
        # Also check for scenes that follow significance scenes
        significance_context = ""
        for i, scene in enumerate(scenes_before_refinement):
            # Check if this scene is marked as a significance scene
            is_significance = scene.get('is_significance_scene', False)
            
            # If this is a significance scene, and there's a scene after it, note it
            if is_significance and i + 1 < len(scenes_before_refinement):
                scene_id = scene.get('id', i + 1)
                next_scene_id = scenes_before_refinement[i + 1].get('id', i + 2)
                next_scene_title = scenes_before_refinement[i + 1].get('title', '')
                significance_context += f"\n  - Scene {scene_id} is a significance scene (explaining why a moment matters). Scene {next_scene_id} (\"{next_scene_title}\") immediately follows it - ensure Scene {next_scene_id}'s beginning transitions FROM Scene {scene_id}, not from the scene before Scene {scene_id}.\n"
        
        if significance_context:
            significance_scene_info = f"""
CRITICAL TRANSITION CHECK - SIGNIFICANCE SCENES:
The following scenes are significance scenes (explaining why pivotal moments matter) that were inserted after pivotal moments:
{significance_context}

IMPORTANT: Scenes that come IMMEDIATELY AFTER significance scenes must transition FROM the significance scene, not from the original pivotal scene. The significance scene creates a bridge - scenes after it should flow from that bridge, not skip over it."""
        
        # Build historian scene info (parallel to significance scene)
        historian_context = ""
        for i, scene in enumerate(scenes_before_refinement):
            is_historian = scene.get('is_historian_scene', False)
            if is_historian and i + 1 < len(scenes_before_refinement):
                scene_id = scene.get('id', i + 1)
                next_scene_id = scenes_before_refinement[i + 1].get('id', i + 2)
                next_scene_title = scenes_before_refinement[i + 1].get('title', '')
                historian_context += f"\n  - Scene {scene_id} is a historian scene (added for depth). Scene {next_scene_id} (\"{next_scene_title}\") immediately follows it - ensure Scene {next_scene_id}'s beginning transitions FROM Scene {scene_id}, not from the scene before Scene {scene_id}.\n"
        
        historian_scene_info = ""
        if historian_context:
            historian_scene_info = f"""
CRITICAL TRANSITION CHECK - HISTORIAN SCENES:
The following scenes are historian scenes (added for depth) that were inserted between existing scenes:
{historian_context}

IMPORTANT: Scenes that come IMMEDIATELY AFTER historian scenes must transition FROM the historian scene, not from the original scene that preceded it. Historian scenes must flow FROM the previous scene (their opening should connect) and INTO the next scene (their closing should set up what follows). If a historian scene reads like a standalone insertion, refine it to bridge."""
    
    # Prepare scene context for the LLM
    scenes_json = json.dumps(scenes_before_refinement, indent=2, ensure_ascii=False)
    
    # Build WHY/WHAT scene type summary for easy reference
    scene_type_summary = ""
    for scene in scenes_before_refinement:
        scene_id = scene.get('id', '?')
        scene_type = scene.get('scene_type', 'UNKNOWN')
        scene_title = scene.get('title', 'Untitled')
        if scene_type in ['WHY', 'WHAT']:
            scene_type_summary += f"  - Scene {scene_id} (\"{scene_title}\"): {scene_type}\n"
    
    if scene_type_summary:
        scene_type_summary = f"""
WHY/WHAT SCENE TYPES (for reference):
{scene_type_summary}
Each scene's type is also indicated in the JSON below. Use this information to understand each scene's purpose when refining.
"""
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    pacing_note = "For shorts: maintain the concise 1-2 sentence per scene format" if is_short else "Maintain 2-4 sentences per scene for main video; use 3-4 when a scene needs more context (new locations, characters, pivotal moments). Prioritize depth and clarity over brevity."
    
    subject_descriptor = {
        "person": "a documentary about",
        "character": "a documentary about the character",
    }.get(subject_type, "a documentary about")
    narration_style_note = "Write from the YouTuber's perspective - natural storytelling without referencing how it's organized."
    
    biopic_audience_note = ""
    why_before_what_note = ""
    biopic_language_note = ""
    short_video_question_note = ""
    # Only enforce "Scene 1 MUST start with video_question" for question_first structure
    if is_short and short_video_question and (short_narrative_structure is None or short_narrative_structure == "question_first"):
        short_video_question_note = f"""
CRITICAL - SHORT SCENE 1 MUST START WITH THE VIDEO QUESTION: Scene 1's narration MUST begin with the video_question. The question comes FIRST, before any facts or context. If Scene 1 does not start with the question, rewrite it so that it does. Video question: "{short_video_question}"
"""
    if script_type == "biopic":
        import prompt_builders
        biopic_audience_note = f"\n\n{prompt_builders.get_biopic_audience_profile()}\n"
        if video_questions:
            if chapter_boundaries:
                ch1_start, ch1_end = chapter_boundaries[0]
            elif scenes_per_chapter:
                ch1_start, ch1_end = 1, scenes_per_chapter
            else:
                ch1_start, ch1_end = 1, 10  # fallback
            why_before_what_note = f"""
CRITICAL - QUESTION BEFORE ITS FACTS (chapter 1 scenes, IDs {ch1_start}-{ch1_end}): Each question must come BEFORE any facts that help answer that particular question. When refining, NEVER reorder so that facts that answer a question appear before that question. Example of WRONG: "By 1943, his genius was undeniable... How did Patton nearly destroy his career?" Example of CORRECT: "How did Patton nearly destroy his career before his greatest triumph? By 1943, his genius was undeniable..."
"""
        biopic_language_note = """
2.5. LIBERAL/PC TERMINOLOGY (CRITICAL for this audience) - Replace terms that alienate conservative 65-year-old American men:
   * AVOID: "privilege", "privileged", "born into privilege", "privileged upbringing", "marginalized", "oppressed", "colonization", "toxic", "problematic"
   * REPLACE WITH: "wealthy family", "family of means", "from a prominent family", "born to wealth", "family fortune", "advantage"—factual alternatives that describe the same reality without liberal political framing
   * Example: "Born into privilege" → "Born to a wealthy California family" or "From a family of means"
"""
    
    refinement_prompt = f"""You are reviewing and refining scenes for {subject_descriptor} {subject_name}.
{biopic_audience_note}
{why_before_what_note}
{short_video_question_note}

CURRENT SCENES (JSON):
{scenes_json}
{scene_type_summary}
{context_info}
{significance_scene_info}
{historian_scene_info}

CRITICAL: For scenes that contain requests to "like", "subscribe", and "comment" in the narration, you can refine and improve them (clarity, flow, naturalness) BUT you MUST preserve the like/subscribe/comment call-to-action. The CTA is essential and must remain in the narration - refine around it, don't remove it.

THREE PRINCIPLES (every scene must satisfy all three): (1) The viewer must know WHAT IS HAPPENING. (2) The viewer must know WHY IT IS IMPORTANT. (3) The viewer must know WHAT COULD GO WRONG. If a scene misses any of these, add the missing element.

YOUR TASK: Review these scenes and improve them. Look for:
1. WHY/WHAT SCENE PURPOSE (CRITICAL) - Each scene has a scene_type of either "WHY" or "WHAT". Understand and refine based on their purpose:
   * WHY scenes should: frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered or don't understand the significance of. They create anticipation and make viewers want to see what comes next. CRITICAL FOR RETENTION - AVOID VIEWER CONFUSION: The biggest issue for retention is viewer confusion. WHY scenes MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions. Don't create confusion by being vague about what's happening.
     - MOST IMPORTANT: Make sure viewers understand WHAT IS HAPPENING - provide clear context and establish the situation before introducing questions or mysteries
     - If a WHY scene doesn't create curiosity or anticipation, strengthen it by adding: a mystery to solve, a problem to overcome, a counterintuitive fact, a secret revealed, or something unexpected
     - If a WHY scene is confusing or vague about what's happening, strengthen it by adding: clear context about the situation, specific facts about what's happening, and clear establishment of the story state
     - WHY scenes should hook viewers and make them anticipate the upcoming WHAT section, but NEVER at the expense of clarity about what's happening
     - Examples of good WHY scenes (straightforward, not sensationalist): "In 1905, Einstein faces a major challenge. How did he manage to...?", "A discovery was about to change his work. What he found would surprise him.", "This moment would prove significant. The stakes were high because..."
   * WHAT scenes should: deliver core content, solutions, actual information, and details. They satisfy the anticipation created by WHY sections. CRITICAL: Every WHAT scene must clearly communicate:
     - WHAT is happening: The specific events, actions, or information
     - WHY it's important: The significance, impact, or meaning of what's happening
     - WHAT THE STAKES ARE: What can go wrong, what can go right, what's at risk, what success/failure means
     - If a WHAT scene doesn't clearly communicate what/why/stakes, strengthen it by adding: specific facts about what's happening, explanation of why it matters, and clear stakes (risks, potential outcomes, consequences)
     - WHAT scenes should answer questions, solve mysteries, or provide information that WHY sections set up
     - Examples of good WHAT scenes: "He solved it by...", "The breakthrough came when...", "In 1905, Einstein published...", "Here's what actually happened...", "If this failed, he would lose everything...", "Success meant..."
   * WHY scenes should: set up the upcoming WHAT section by establishing what is happening (MOST IMPORTANT - avoid confusion), what will happen next, why it matters, and what the stakes are (what can go wrong, what can go right, what's at risk)
     - MOST IMPORTANT: Make sure viewers understand WHAT IS HAPPENING - provide clear context and establish the situation before introducing questions or mysteries
     - If a WHY scene doesn't clearly establish what is happening (causing potential confusion), strengthen it by adding: clear context about the situation, specific facts about what's happening, and clear establishment of the story state
     - If a WHY scene doesn't set up what/why/stakes for the next WHAT scene, strengthen it by adding: what will happen (or what question/problem needs addressing), why it matters, and what the stakes are
     - WHY scenes should create anticipation by establishing the context that the WHAT scene will address, but NEVER at the expense of clarity about what's happening
     - Examples of good WHY scenes (straightforward, not sensationalist): "In 1905, Einstein faces a major challenge. How did he manage to...?", "A discovery was about to change his work.", "This moment would prove significant. If he failed, the consequences would be severe.", "The stakes were high because..."
   * Ensure WHY sections set up what/why/stakes for upcoming WHAT sections - if a WHY scene doesn't establish what will happen, why it matters, and what the stakes are, refine it to do so
   * Ensure WHAT sections clearly communicate what/why/stakes - if a WHAT scene doesn't clearly explain what's happening, why it's important, and what the stakes are, refine it to do so
2. SENSATIONALIST LANGUAGE - Simplify and tone down:
   * Remove or replace gimmicky phrases: "shocking", "incredible", "unbelievable", "what nobody expected", "the secret that would change everything", "mind-blowing"
   * Let facts speak for themselves—don't oversell with hype or repeated dramatic framing
   * Prefer direct, straightforward statements over theatrical buildup
{biopic_language_note}
3. META REFERENCES (CRITICAL) - Remove ANY references to:
   * "Chapter" or "chapters" - viewers don't know about chapters
   * "In this video" or "In this documentary" or "In this story"
   * "As we'll see" or "Later in this video" or "As we continue"
   * "Let me tell you" or "I want to show you" - just narrate directly
   * Any production elements, scripts, outlines, prompts, or behind-the-scenes info
   * References like "this part" or "this section" or "here we see"
4. OVERLAPPING/DUPLICATE EVENTS (CRITICAL) - If multiple scenes describe the SAME event, moment, or action in detail, consolidate or remove the duplicate. Each scene should cover DIFFERENT events. For example, if Scene 24 describes Antony's suicide attempt and being brought to the mausoleum, Scene 27 should NOT repeat this same event - instead it should focus on what happens NEXT (his death, Cleopatra's response, or the consequences). Remove overlapping content and ensure each scene advances the story with NEW information.
5. MISSING EMOTIONAL ENGAGEMENT - If scenes read too factually without emotional weight, add:
   * How events feel to the character (fear, determination, despair, triumph)
   * The emotional significance and personal stakes
   * Internal experience details (what they're thinking, feeling, fearing)
   * Physical sensations and reactions that create empathy
   * Make events feel significant by connecting them to human emotions
6. EMOTION CONSISTENCY AND SMOOTH TRANSITIONS - Ensure the scene's "emotion" field matches the narration tone and image mood, AND flows smoothly from the previous scene:
   * The emotion field should accurately reflect how the scene FEELS
   * Narration tone should match the emotion (e.g., "desperate" → urgent/anxious narration)
   * Image prompt mood should match the emotion (e.g., "desperate" → tense atmosphere in image)
   * If narration or image don't match the emotion field, refine them to be consistent
   * CRITICAL: Emotions must flow SMOOTHLY between scenes - only change gradually from the previous scene's emotion
   * Build intensity gradually: 'contemplative' → 'thoughtful' → 'somber' → 'serious' → 'tense' (not 'calm' → 'urgent')
   * Avoid dramatic emotional jumps - each scene's emotion should be a natural progression from the previous scene
   * narration_instructions should also transition smoothly - keep to ONE SENTENCE focusing on a single emotion: "Focus on [emotion]." If previous scene was "Focus on tension", next might be "Focus on anxiety" (gradual progression)
7. CONTEXT AND DEPTH (documentary/biopic) - Viewers must never be lost. If a scene name-drops a place, person, or event (e.g. Fort Necessity, General Braddock, the Newburgh Conspiracy) without context, add one phrase so a general viewer understands. If a scene jumps into an event without establishing the situation (who is involved, where we are, why this moment), add a sentence of setup. If a major event has no consequence stated, briefly note what it changes or what would have happened otherwise. Prioritize clarity and depth—use 3-4 sentences when a scene needs more context.
8. VIEWER CONFUSION (CRITICAL FOR RETENTION) - The biggest issue for retention is viewer confusion. Ensure WHY scenes make it clear what is happening:
   * WHY scenes MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions
   * If a WHY scene is confusing or vague about what's happening, strengthen it by adding: clear context about the situation, specific facts about what's happening, and clear establishment of the story state
   * Don't create confusion by being vague - viewers should always understand what is happening in the story, even when mysteries or questions are being introduced
   * Examples of clear WHY scenes: "In 1905, Einstein faces an impossible challenge. But how did he manage to...?" (clear context first, then question) vs. "But how did he manage to...?" (confusing - no context)
9. SEAMLESS JOURNEY AND CONNECTIONS (CRITICAL) - Ensure scenes feel connected, not like consecutive pieces of disjoint information:
   * Each scene should build on the previous scene - reference what came before naturally, show how events connect
   * Scenes should feel like a flowing narrative where each scene grows from the last, not like separate disconnected facts
   * If scenes feel disconnected or like they're just listing information (A and B and C), strengthen connections by:
     - Referencing events, themes, or emotions from previous scenes
     - Showing cause-and-effect relationships between scenes
     - Continuing threads or plot elements established earlier
     - Creating logical progression where each scene feels like the natural next step
   * Use WHY/WHAT interleaving to create natural connections - WHY scenes should set up questions that the following WHAT scenes answer
   * The goal: When scenes are strung together, they should feel like one continuous, connected story
10. AWKWARD TRANSITIONS - scene endings that don't flow smoothly into the next scene
   * CRITICAL: If significance scenes were inserted after pivotal moments, ensure scenes immediately AFTER significance scenes transition FROM the significance scene, not from the original pivotal scene
   * Example: If Scene 17 is pivotal, Scene 18 is a significance scene (inserted), and Scene 19 follows - Scene 19's beginning should reference/flow from Scene 18, not Scene 17
   * Significance scenes bridge pivotal moments and their consequences - scenes after them should acknowledge this bridge
   * CRITICAL: Similarly, scenes immediately AFTER historian scenes must transition FROM the historian scene, not from the original scene that preceded it
   * Historian scenes must flow FROM the previous scene (their opening should connect) and INTO the next scene (their closing should set up what follows). If a historian scene reads like a standalone insertion, refine it to bridge.
11. WEIRD OR UNNATURAL SENTENCES - phrases that sound odd when spoken, overly flowery language, vague statements
12. REPETITIVE LANGUAGE - same words or phrases used too frequently
13. CLARITY ISSUES - sentences that are confusing or hard to understand when spoken aloud
14. NARRATION STYLE VIOLATIONS - film directions ("Cut to:", "Smash cut—"), camera directions ("Close-up of", "Wide shot"), production terminology, or unnecessary quotes (use quotes ONLY for proper nouns to emphasize—e.g. titles of works; otherwise no quotes, they mess up TTS)
15. MISSING CONNECTIONS - scenes that don't reference what came before when they should
16. PACING ISSUES - scenes that feel rushed or too slow for the story beat
17. FACTUAL INCONSISTENCIES - any contradictions or inaccuracies

IMPORTANT GUIDELINES:
- Keep ALL factual information accurate
- Maintain the same scene structure and IDs
- Preserve all fields (id, title, narration, image_prompt, emotion, year, scene_type, narration_instructions, etc.)
- CRITICAL: Preserve the WHY/WHAT structure - maintain each scene's scene_type field (WHY or WHAT). WHY sections frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered or don't understand the significance of. WHY sections should set up what will happen, why it matters, and what the stakes are for upcoming WHAT sections. WHAT sections deliver content/solutions and must clearly communicate what is happening, why it's important, and what the stakes are. Ensure WHY sections create anticipation for upcoming WHAT sections by establishing what/why/stakes.
- CRITICAL: Preserve and refine narration_instructions - Each scene MUST have narration_instructions as ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on unease.' CRITICAL: SMOOTH EMOTIONAL TRANSITIONS - Ensure emotions and narration_instructions flow smoothly between scenes. Emotions should only change GRADUALLY from one scene to the next. narration_instructions should also transition smoothly - if previous scene was 'Focus on tension', next might be 'Focus on anxiety' (gradual progression). The narration should not sound completely different from one scene to the next. ABSOLUTELY DO NOT remove narration_instructions - every scene must have this field. If a scene is missing narration_instructions, add it based on the scene's emotion field: 'Focus on [emotion].'
- Keep the same tone and style
- {pacing_note}
- Only make changes that IMPROVE clarity, flow, or naturalness
- DO NOT add new information or change facts
- DO NOT add film/camera directions - narration should be pure spoken words
- ABSOLUTELY REMOVE any meta references to chapters, production elements, or the script structure itself
- {narration_style_note}
- CRITICAL: For scenes mentioning "like", "subscribe", and "comment", you can improve them but MUST keep all three CTA elements in the narration

Return the SAME JSON structure with refined scenes. Only change what needs improvement - don't rewrite everything.

CRITICAL JSON STRUCTURE REQUIREMENTS:
- Every scene MUST include ALL required fields: id, title, narration, image_prompt, emotion, year, scene_type, and narration_instructions
- The "narration_instructions" field is REQUIRED for every scene - it must be ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on contemplation.'
- DO NOT omit any fields from the JSON - every scene must have the complete structure
- If a scene is missing narration_instructions, add it based on the scene's emotion field

Each scene object must include: id, title, narration, image_prompt, emotion, year, scene_type, narration_instructions."""
    
    try:
        audience_note = f" {prompt_builders.get_biopic_audience_profile()}" if script_type == "biopic" else ""
        content = llm_utils.generate_text(
            messages=[
                {"role": "system", "content": f"You are an expert editor who refines historical biopics' narration for clarity, flow, and naturalness. THREE PRINCIPLES (every scene must satisfy all three): (1) The viewer must know WHAT IS HAPPENING. (2) The viewer must know WHY IT IS IMPORTANT. (3) The viewer must know WHAT COULD GO WRONG. If a scene misses any of these, add the missing element. CRITICAL FOR RETENTION - AVOID VIEWER CONFUSION: The biggest issue for retention is viewer confusion. WHY scenes MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions. Don't create confusion by being vague about what's happening. CRITICAL: Create a SEAMLESS JOURNEY through the video - scenes should feel CONNECTED, not like consecutive pieces of disjoint information (A and B and C). Each scene should build on the previous scene, reference what came before naturally, and show how events connect. You understand the WHY/WHAT paradigm: WHY scenes frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered - they create anticipation by setting up what will happen, why it matters, and what the stakes are for upcoming WHAT sections. WHY scenes MUST clearly establish what is happening (MOST IMPORTANT for retention) before introducing questions or mysteries. WHAT scenes deliver core content, solutions, and information - they satisfy anticipation by clearly communicating what is happening, why it's important, and what the stakes are (what can go wrong, what can go right, what's at risk).  You catch awkward transitions, weird sentences, style violations, sensationalist or gimmicky phrasing, and especially meta references (chapters, production elements, etc.). Simplify sensationalist language—let facts speak for themselves. You ensure scenes feel connected and build on each other, WHY scenes clearly establish what is happening (avoiding confusion) and set up what/why/stakes for upcoming WHAT sections, and WHAT scenes clearly communicate what/why/stakes. {narration_style_note}{audience_note} CRITICAL JSON REQUIREMENT: Every scene in your response MUST include the 'narration_instructions' field. This field must be ONE SENTENCE focusing on a single emotion from the emotion field. Keep it simple: 'Focus on [emotion].' Examples: 'Focus on tension.' or 'Focus on contemplation.' The narration_instructions should flow smoothly from the previous scene's narration_instructions (gradual emotion progression). DO NOT omit this field from any scene. Same structure as input, including narration_instructions for every scene. "},
                {"role": "user", "content": refinement_prompt}
            ],
            temperature=0.3,  # Lower temperature for refinement - more focused changes
            response_json_schema=SCENES_ARRAY_SCHEMA,
            provider=llm_utils.get_provider_for_step("REFINEMENT"),
        )
        refined_scenes = json.loads(clean_json_response(content))
        
        if not isinstance(refined_scenes, list):
            print(f"[REFINEMENT] WARNING: Expected array, got {type(refined_scenes)}. Using original scenes.")
            return scenes, {}
         
        if len(refined_scenes) != len(scenes_before_refinement):
            print(f"[REFINEMENT] WARNING: Scene count changed ({len(scenes_before_refinement)} → {len(refined_scenes)}). Using original scenes.")
            return scenes_before_refinement, {}
        
        # For shorts, cap at 5 scenes maximum after refinement
        if is_short and len(refined_scenes) > 5:
            print(f"[REFINEMENT PASS 3]   • Shorts mode: capping at 5 scenes after refinement (was {len(refined_scenes)})")
            refined_scenes = refined_scenes[:5]
            # Renumber scenes
            for i, scene in enumerate(refined_scenes):
                scene['id'] = i + 1
         
        # Validate all required fields are present and fix missing narration_instructions
        # Preserve is_significance_scene, is_historian_scene from original (LLM may omit them)
        for i, scene in enumerate(refined_scenes):
            original = scenes_before_refinement[i] if i < len(scenes_before_refinement) else {}
            if 'is_significance_scene' not in scene:
                scene['is_significance_scene'] = original.get('is_significance_scene', False)
            if 'is_historian_scene' not in scene:
                scene['is_historian_scene'] = original.get('is_historian_scene', False)

            # Check for missing required fields
            missing_fields = []
            for field in ["id", "title", "narration", "image_prompt", "emotion", "year", "scene_type"]:
                if field not in scene:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"[REFINEMENT] WARNING: Scene {i+1} missing required fields: {missing_fields}. Using original scenes.")
                return scenes, {}
            
            # CRITICAL: Ensure narration_instructions is always present (fix if missing)
            if 'narration_instructions' not in scene or not scene.get('narration_instructions'):
                emotion = scene.get('emotion', 'contemplative')
                scene['narration_instructions'] = f"Focus on {emotion}."
                print(f"[REFINEMENT] WARNING: Scene {i+1} missing narration_instructions, generated fallback: {scene['narration_instructions']}")
            
            # Validate scene_type is valid
            if scene.get('scene_type') not in ['WHY', 'WHAT']:
                print(f"[REFINEMENT] WARNING: Scene {i+1} has invalid 'scene_type' value: {scene.get('scene_type')}. Using original scenes.")
                return scenes, {}
        
        # Track changes made during refinement
        changes_stats = {
            "scenes_changed": 0,
            "title_changes": 0,
            "narration_changes": 0,
            "image_prompt_changes": 0,
            "total_changes": 0
        }
        
        for i, (original_scene, refined_scene) in enumerate(zip(scenes_before_refinement, refined_scenes)):
            scene_changed = False
            
            # Check title changes
            if original_scene.get('title', '').strip() != refined_scene.get('title', '').strip():
                changes_stats["title_changes"] += 1
                changes_stats["total_changes"] += 1
                scene_changed = True
            
            # Check narration changes
            original_narration = original_scene.get('narration', '').strip()
            refined_narration = refined_scene.get('narration', '').strip()
            if original_narration != refined_narration:
                changes_stats["narration_changes"] += 1
                changes_stats["total_changes"] += 1
                scene_changed = True
            
            # Check image_prompt changes
            if original_scene.get('image_prompt', '').strip() != refined_scene.get('image_prompt', '').strip():
                changes_stats["image_prompt_changes"] += 1
                changes_stats["total_changes"] += 1
                scene_changed = True
            
            if scene_changed:
                changes_stats["scenes_changed"] += 1
        
        # Log refinement statistics
        total_scenes = len(refined_scenes)
        unchanged_scenes = total_scenes - changes_stats["scenes_changed"]
        print(f"[REFINEMENT] ✓ Refined {total_scenes} scenes")
        print(f"[REFINEMENT]   Statistics:")
        print(f"[REFINEMENT]   • Scenes changed: {changes_stats['scenes_changed']}/{total_scenes} ({changes_stats['scenes_changed']*100//total_scenes if total_scenes > 0 else 0}%)")
        print(f"[REFINEMENT]   • Unchanged scenes: {unchanged_scenes}/{total_scenes} ({unchanged_scenes*100//total_scenes if total_scenes > 0 else 0}%)")
        print(f"[REFINEMENT]   • Title changes: {changes_stats['title_changes']}")
        print(f"[REFINEMENT]   • Narration changes: {changes_stats['narration_changes']}")
        print(f"[REFINEMENT]   • Image prompt changes: {changes_stats['image_prompt_changes']}")
        print(f"[REFINEMENT]   • Total field changes: {changes_stats['total_changes']}")
        
        # Final validation: Ensure ALL scenes have narration_instructions (fix any that are missing)
        # This is a critical safeguard to prevent missing narration_instructions after refinement
        for i, scene in enumerate(refined_scenes):
            if 'narration_instructions' not in scene or not scene.get('narration_instructions', '').strip():
                emotion = scene.get('emotion', 'contemplative')
                scene['narration_instructions'] = f"Focus on {emotion}."
                print(f"[REFINEMENT] CRITICAL FIX: Scene {i+1} (ID: {scene.get('id', i+1)}) missing narration_instructions after refinement, generated fallback: {scene['narration_instructions']}")
        
        # PASS 4: Film composer music selection (biopic main videos and shorts)
        if script_type == "biopic":
            try:
                from biopic_music_config import get_all_songs, BIOPIC_MUSIC_DIR
                from biopic_schemas import build_music_selection_schema
                all_songs = get_all_songs()
                # For shorts, filter to short_music_mood folder when provided (keeps short cohesive)
                if is_short and short_music_mood and all_songs:
                    mood_lower = (short_music_mood or "").strip().lower()
                    all_songs = [s for s in all_songs if s.lower().startswith(mood_lower + "/")]
                    if not all_songs:
                        all_songs = get_all_songs()  # fallback to full list
                if all_songs:
                    pass_label = "shorts" if is_short else "main video"
                    print(f"\n[REFINEMENT PASS 4] Film composer: selecting song and volume for each scene ({pass_label})...")
                    # Strip any existing music (LLM may have hallucinated it before Pass 4)
                    for s in refined_scenes:
                        s.pop("music_song", None)
                        s.pop("music_volume", None)
                    scene_summaries = []
                    for s in refined_scenes:
                        scene_summaries.append({
                            "id": s.get("id", 0),
                            "chapter_num": s.get("chapter_num"),
                            "title": s.get("title", ""),
                            "narration": (s.get("narration", "") or ""),
                            "emotion": s.get("emotion", ""),
                            "scene_type": s.get("scene_type", ""),
                        })
                    songs_list = "\n".join(f"  - {s}" for s in all_songs)
                    continuity_guidance = """SONG CONTINUITY (STRICT PREFERENCE):
- DEFAULT: Keep the SAME song as the previous scene. Do NOT change unless there is a strong reason.
- Change songs ONLY when: (1) a new chapter begins (chapter_num changes), (2) a major mood shift (e.g., tense → triumphant, contemplative → urgent), or (3) scene_type changes dramatically.
- Do NOT change songs for minor emotional variations—use volume (low/medium/loud) to reflect subtle shifts instead.
- Prefer fewer song changes overall; continuity is more important than matching every nuance."""
                    music_prompt = f"""You are a film composer selecting background music for a documentary. For each scene, pick exactly one song from the available list and a volume level (low, medium, or loud).

{continuity_guidance}

AVAILABLE SONGS (use the exact path):
{songs_list}

SCENES:
{json.dumps(scene_summaries, indent=2, ensure_ascii=False)}

Return a JSON array with one object per scene: {{"id": <scene_id>, "music_song": "<exact path from list>", "music_volume": "low"|"medium"|"loud"}}."""
                    system_content = "You are a film composer selecting background music for documentaries. STRONGLY prefer keeping the same song across consecutive scenes; change only for chapter boundaries or major mood shifts. Use volume (low/medium/loud) for subtle mood variations. Pick ONLY from the provided song list."
                    music_schema = build_music_selection_schema(all_songs)
                    content = llm_utils.generate_text(
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": music_prompt}
                        ],
                        temperature=0.3,
                        response_json_schema=music_schema,
                        provider=llm_utils.get_provider_for_step("REFINEMENT_FILM_COMPOSER"),
                    )
                    selections = json.loads(clean_json_response(content))
                    if isinstance(selections, list) and len(selections) == len(refined_scenes):
                        id_to_selection = {s["id"]: s for s in selections}
                        fallback_song = all_songs[0]
                        for i, scene in enumerate(refined_scenes):
                            sid = scene.get("id", i + 1)
                            sel = id_to_selection.get(sid, {})
                            song_path = (sel.get("music_song") or "").strip()
                            volume = (sel.get("music_volume") or "medium").strip().lower()
                            if volume not in ("low", "medium", "loud"):
                                volume = "medium"
                            # Defensive: LLM may ignore enum and hallucinate - only accept paths from all_songs
                            if song_path not in all_songs:
                                song_path = fallback_song
                            full_path = BIOPIC_MUSIC_DIR / song_path if song_path else None
                            if not full_path or not full_path.exists():
                                song_path = fallback_song
                                full_path = BIOPIC_MUSIC_DIR / song_path
                            if full_path.exists():
                                scene["music_song"] = song_path
                                scene["music_volume"] = volume
                                print(f"[REFINEMENT PASS 4]   Scene {sid}: {song_path} ({volume})")
                            else:
                                scene["music_song"] = fallback_song
                                scene["music_volume"] = "medium"
                                print(f"[REFINEMENT PASS 4]   Scene {sid}: fallback {fallback_song} (medium)")
                    else:
                        print(f"[REFINEMENT PASS 4] WARNING: Invalid response (expected {len(refined_scenes)} items), skipping music selection")
                else:
                    print(f"\n[REFINEMENT PASS 4] Skipped (no songs in biopic_music/)")
            except Exception as e:
                print(f"[REFINEMENT PASS 4] WARNING: Music selection failed ({e}), skipping")

        # PASS 5: Transition selection
        if script_type == "biopic":
            try:
                from kenburns_config import TRANSITION_TYPES, TRANSITION_SPEEDS, get_transition_guidance_prompt_str
                pass_label = "shorts" if is_short else "main video"
                print(f"\n[REFINEMENT PASS 5] Transition selection: choosing transition_to_next and transition_speed for each scene ({pass_label})...")
                scene_summaries = []
                for s in refined_scenes:
                    scene_summaries.append({
                        "id": s.get("id", 0),
                        "title": s.get("title", ""),
                        "emotion": s.get("emotion", ""),
                        "scene_type": s.get("scene_type", ""),
                        "is_chapter_transition": s.get("is_chapter_transition", False),
                    })
                transitions_str = ", ".join(TRANSITION_TYPES)
                speeds_str = ", ".join(TRANSITION_SPEEDS)
                transition_guidance = get_transition_guidance_prompt_str()
                transition_prompt = f"""You are a film editor selecting video transitions between scenes for a documentary.

For each scene, pick TWO choices for the transition TO THE NEXT scene:
1. transition_to_next: the type of transition
2. transition_speed: how fast (only when transition is not cut). The last scene can have transition_to_next: null.

WHAT EACH TRANSITION SIGNIFIES AND WHEN TO USE IT:
{transition_guidance}

TRANSITION SPEED (for crossfade and slide_* only):
- quick: 0.2s. Subtle, snappy.
- medium: 0.4s. Default, balanced.
- slow: 0.8s. Emotional moments, reflection, chapter boundaries.

Use the guidance above to match each transition to the emotional and narrative context of the scene pair.
Chapter transition scenes (is_chapter_transition: true): Prefer crossfade or slide with slow speed.
Last scene: use null for transition_to_next (transition_speed can be null).

SCENES:
{json.dumps(scene_summaries, indent=2, ensure_ascii=False)}

Return a JSON array with one object per scene: {{"id": <scene_id>, "transition_to_next": "<type>" or null, "transition_speed": "<speed>" or null}}.
Use transition_to_next: one of {transitions_str}, or null for the last scene.
Use transition_speed: one of {speeds_str} when transition_to_next is not cut; null for last scene or when cut."""
                system_content = "You are a film editor selecting transitions for documentaries. Match each transition to the emotional and narrative context of the scene pair. Pick both transition type and speed."
                content = llm_utils.generate_text(
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": transition_prompt}
                    ],
                    temperature=0.3,
                    response_json_schema=TRANSITION_SELECTION_SCHEMA,
                    provider=llm_utils.get_provider_for_step("REFINEMENT_TRANSITIONS"),
                )
                selections = json.loads(clean_json_response(content))
                if isinstance(selections, list) and len(selections) == len(refined_scenes):
                    id_to_sel = {s["id"]: s for s in selections}
                    default_transition = "crossfade"
                    default_speed = "medium"
                    for i, scene in enumerate(refined_scenes):
                        sid = scene.get("id", i + 1)
                        sel = id_to_sel.get(sid, {})
                        trans = sel.get("transition_to_next")
                        speed = sel.get("transition_speed")
                        if i == len(refined_scenes) - 1:
                            scene["transition_to_next"] = None
                            scene["transition_speed"] = None
                        elif trans and str(trans).strip().lower() in [t.lower() for t in TRANSITION_TYPES]:
                            scene["transition_to_next"] = trans
                            if trans == "cut":
                                scene["transition_speed"] = None
                            else:
                                scene["transition_speed"] = speed if speed and str(speed).strip().lower() in [s.lower() for s in TRANSITION_SPEEDS] else default_speed
                            print(f"[REFINEMENT PASS 5]   Scene {sid}: transition_to_next={trans}, transition_speed={scene.get('transition_speed')}")
                        else:
                            scene["transition_to_next"] = default_transition
                            scene["transition_speed"] = default_speed
                            print(f"[REFINEMENT PASS 5]   Scene {sid}: default {default_transition}, {default_speed}")
                else:
                    print(f"[REFINEMENT PASS 5] WARNING: Invalid response (expected {len(refined_scenes)} items), skipping transition selection")
            except Exception as e:
                print(f"[REFINEMENT PASS 5] WARNING: Transition selection failed ({e}), skipping")

        # Generate diff comparing original scenes to final refined scenes (which may include significance scenes)
        diff_data = generate_refinement_diff(original_scenes, refined_scenes)
        # Save diff to file if output path provided
        if diff_output_path:
            diff_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diff_output_path, "w", encoding="utf-8") as f:
                json.dump(diff_data, f, indent=2, ensure_ascii=False)
            print(f"[REFINEMENT]   • Diff saved: {diff_output_path}")
        
        return refined_scenes, diff_data
        
    except ValueError:
        raise
    except Exception as e:
        print(f"[REFINEMENT] WARNING: Refinement failed ({e}). Using original scenes.")
        return scenes, {}


def insert_chapter_transition_scenes(
    scenes: list[dict],
    chapters: list[dict],
    subject_name: str,
    script_type: str = "biopic",
) -> list[dict]:
    """
    Insert chapter transition scenes between chapters (after ch1, ch2, ... up to but not after the last chapter).
    No transition before chapter 1 or after the final chapter.
    Transition scenes have the same schema as normal scenes (all required fields) so validation never fails.
    Music: inherited from the scene immediately before (last scene of the chapter we're leaving).
    """
    if not scenes or not chapters or len(chapters) < 2:
        return scenes

    # Fallback music when prev_scene lacks it (should not happen after Pass 4)
    def _fallback_music() -> tuple[str, str]:
        try:
            from biopic_music_config import get_all_songs
            songs = get_all_songs()
            return (songs[0], "medium") if songs else ("", "medium")
        except Exception:
            return ("", "medium")

    # Build map from chapter_num to (first_scene_index, last_scene_index) from refined scenes
    chapter_to_indices: dict[int, tuple[int, int]] = {}
    for i, scene in enumerate(scenes):
        ch_num = scene.get("chapter_num")
        if ch_num is None:
            continue
        if ch_num not in chapter_to_indices:
            chapter_to_indices[ch_num] = (i, i)
        else:
            first, _ = chapter_to_indices[ch_num]
            chapter_to_indices[ch_num] = (first, i)

    chapters_by_num = {ch.get("chapter_num"): ch for ch in chapters if ch.get("chapter_num") is not None}
    max_chapter = max(chapters_by_num.keys()) if chapters_by_num else 0

    # Collect (insert_after_index, next_chapter) for each transition
    # Transition from ch N to ch N+1: insert after last scene of ch N, introduce ch N+1
    inserts: list[tuple[int, dict]] = []
    for next_ch_num in range(2, max_chapter + 1):
        prev_ch_num = next_ch_num - 1
        if prev_ch_num not in chapter_to_indices or next_ch_num not in chapters_by_num:
            continue
        _, last_idx = chapter_to_indices[prev_ch_num]
        next_chapter = chapters_by_num[next_ch_num]
        inserts.append((last_idx, next_chapter))

    if not inserts:
        return scenes

    # Insert in reverse order so indices stay valid
    inserts.sort(key=lambda x: x[0], reverse=True)

    for insert_after_idx, next_chapter in inserts:
        prev_scene = scenes[insert_after_idx]
        ch_title = next_chapter.get("title", "Chapter")
        ch_summary = (next_chapter.get("summary", "") or "")[:60].strip()
        year_range = next_chapter.get("year_range") or next_chapter.get("time_setting") or "various"

        summary_snippet = f" Subtle background that conveys: {ch_summary}." if ch_summary else ""

        transition_scene = {
            "title": f"Chapter {next_chapter.get('chapter_num', '?')}: {ch_title}",
            "narration": ch_title,
            "scene_type": "TRANSITION",
            "image_prompt": f"Title card: documentary style. The text '{ch_title}' displayed prominently in elegant serif font.{summary_snippet} Fill the entire 16:9 frame—no empty black areas or gaps. Use a continuous background (gradient, texture, or subtle imagery) extending to all edges, consistent with the documentary's color palette.",
            "emotion": "anticipatory",
            "narration_instructions": "Clear, calm announcement.",
            "year": year_range,
            "chapter_num": next_chapter.get("chapter_num"),
            "kenburns_pattern": "zoom_out",
            "kenburns_intensity": "medium",
            "is_chapter_transition": True,
            "is_significance_scene": False,
            "is_historian_scene": False,
        }
        # Music: inherit from first scene of next chapter (the chapter we're transitioning into)
        fallback_song, fallback_vol = _fallback_music()
        next_ch_num = next_chapter.get("chapter_num")
        first_idx, _ = chapter_to_indices.get(next_ch_num, (None, None))
        if first_idx is not None:
            next_chapter_first_scene = scenes[first_idx]
            transition_scene["music_song"] = next_chapter_first_scene.get("music_song") or fallback_song
            transition_scene["music_volume"] = next_chapter_first_scene.get("music_volume") or fallback_vol
        else:
            transition_scene["music_song"] = prev_scene.get("music_song") or fallback_song
            transition_scene["music_volume"] = prev_scene.get("music_volume") or fallback_vol
        # transition_to_next and transition_speed: prev_scene was transitioning to first of next chapter; now prev->transition->next
        prev_transition = prev_scene.get("transition_to_next")
        prev_scene["transition_to_next"] = "crossfade"  # to chapter title card
        prev_scene["transition_speed"] = "slow"
        transition_scene["transition_to_next"] = prev_transition if prev_transition is not None else "crossfade"
        transition_scene["transition_speed"] = "medium"

        scenes.insert(insert_after_idx + 1, transition_scene)

    # Renumber all scene ids
    for i, scene in enumerate(scenes):
        scene["id"] = i + 1

    return scenes
