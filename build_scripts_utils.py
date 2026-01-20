"""
Shared utilities for script generation.
Used by both build_script.py (historical documentaries) and build_script_lol.py (LoL lore).
"""
import json
import base64
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

# These will be set by the importing module
client: Optional[OpenAI] = None
SCRIPT_MODEL: str = "gpt-5.2"
IMG_MODEL: str = "gpt-image-1.5"
NO_TEXT_CONSTRAINT: str = """
CRITICAL: Do NOT include any text, words, letters, numbers, titles, labels, watermarks, or any written content in the image. The image must be completely text-free."""


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
    
    if content_type == "lore":
        # Add lore-specific emphasis
        base_requirements += "\n10. LORE-SPECIFIC: Emphasize magical elements, mystical powers, regional conflicts, faction relationships, world-building details, and the fantasy/mythical aspects of the story"
    
    return base_requirements


def get_shared_narration_style(is_short: bool = False) -> str:
    """Shared narration style instructions for both main video and shorts."""
    base_style = """NARRATION STYLE - CRITICAL:
- PERSPECTIVE: Write from the YouTuber's perspective - this is YOUR script, YOU are telling the story to YOUR audience
- SIMPLE, CLEAR language. Write for a general audience.
- AVOID flowery, artistic, or poetic language
- AVOID vague phrases like "little did he know", "destiny awaited", "the world would never be the same"
- NO dramatic pauses or buildup - just deliver the facts engagingly
- NO made-up temporal transitions - stick to what actually happened
- Use present tense: "Einstein submits his paper to the journal..."
- Tell a DEEP story with DETAILS - not just "he did X" but "he did X because Y, and here's how it happened, and here's what it meant"
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
    
    if is_short:
        base_style += "\n- 1-2 sentences per scene (~8-12 seconds when spoken) - shorts need to be concise but detailed"
    else:
        base_style += "\n- 2-3 sentences per scene (~12-18 seconds of narration)"
    
    base_style += """
- Pack MAXIMUM information into minimum words
- CRITICAL: This is SPOKEN narration for text-to-speech. Do NOT include:
  * Film directions like "Smash cut—", "Hard cut to", "Cut to:", "Fade in:"
  * Camera directions like "Close-up of", "Wide shot:", "Pan to"
  * Any production/editing terminology
  Write ONLY words that should be spoken by a narrator's voice."""
    
    return base_style


def get_shared_scene_flow_instructions() -> str:
    """Shared scene-to-scene flow instructions."""
    return """SCENE-TO-SCENE FLOW:
- Flow naturally through the story using ACTUAL events and their consequences
- Connect scenes through cause-and-effect, not artificial transitions
- DO NOT use made-up temporal transitions like "Days later...", "Later that week...", "That afternoon...", "The next morning...", "Weeks passed...", "Meanwhile..."
- Instead, let the story flow through what actually happened: "The paper is published. Physicists worldwide take notice."
- Each scene should feel inevitable because of what came before, not because of a transition phrase
- Build narrative momentum through actual events, not filler words
- CRITICAL: Each scene must cover DIFFERENT, NON-OVERLAPPING events. Do NOT repeat the same event, moment, or action that was already covered in a previous scene. Each scene should advance the story with NEW information, not re-tell what happened before. If an event was already described in detail, move to its consequences or the next significant event instead of describing it again."""


def get_shared_examples(content_type: str = "historical") -> str:
    """
    Shared good/bad examples for narration.
    
    Args:
        content_type: "historical" for historical examples, "lore" for fantasy/game lore examples
    """
    if content_type == "lore":
        return """EXAMPLES:
BAD: "In the ancient realm, destiny awaited the champion."
BAD: "Days later, the battle would begin."
GOOD: "In 996 AN, Demacia's armies march on Noxus. The battle lasts three days. Garen's blade cuts through enemy ranks, but Noxian magic turns the tide. Thousands die. The war changes everything." """
    
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
    
    if client is None:
        raise ValueError("OpenAI client not initialized. Set build_scripts_utils.client before calling generate_thumbnail.")
    
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
            
            resp = client.images.generate(
                model=IMG_MODEL,
                prompt=current_prompt,
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
        fields_to_compare = ["title", "narration", "image_prompt", "emotion", "year"]
        
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


def identify_pivotal_moments(scenes: list[dict], subject_name: str, chapter_context: str = None) -> list[dict]:
    """
    Identify 4-7 most pivotal moments from the scenes.
    
    Pivotal moments are: major breakthroughs, turning points, critical decisions,
    moments that change the story trajectory.
    
    Args:
        scenes: List of scene dictionaries
        subject_name: Name of the subject (person, character, region, etc.)
        chapter_context: Optional chapter context string
        
    Returns:
        list of dicts with 'scene_id' and 'justification' (why it's pivotal)
    """
    if not scenes:
        return []
    
    if client is None:
        raise ValueError("OpenAI client not initialized. Set build_scripts_utils.client before calling identify_pivotal_moments.")
    
    scenes_json = json.dumps(scenes, indent=2, ensure_ascii=False)
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    identification_prompt = f"""Analyze these scenes from a documentary about {subject_name} and identify the 4-7 MOST PIVOTAL MOMENTS.

CURRENT SCENES (JSON):
{scenes_json}
{context_info}

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

Respond with JSON array:
[
  {{
    "scene_id": 12,
    "justification": "This is the moment where the discovery changes everything - it's not just an achievement, it rewrites the rules of physics and shifts the entire scientific paradigm."
  }},
  ...
]

Return 4-7 pivotal moments only. Be selective - only the moments that are truly game-changers."""

    try:
        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert story analyst who identifies pivotal moments in narratives. You understand what makes a moment truly significant and game-changing. Respond with valid JSON array only."},
                {"role": "user", "content": identification_prompt}
            ],
            temperature=0.5,
        )
        
        pivotal_moments = json.loads(clean_json_response(response.choices[0].message.content))
        
        if not isinstance(pivotal_moments, list):
            print(f"[PIVOTAL MOMENTS] WARNING: Expected array, got {type(pivotal_moments)}. Returning empty list.")
            return []
        
        # Validate that scene IDs exist
        scene_ids = {scene.get('id') for scene in scenes}
        valid_moments = []
        for moment in pivotal_moments:
            scene_id = moment.get('scene_id')
            if scene_id in scene_ids:
                valid_moments.append(moment)
            else:
                print(f"[PIVOTAL MOMENTS] WARNING: Scene ID {scene_id} not found in scenes, skipping.")
        
        return valid_moments[:4]  # Limit to 4 maximum
        
    except Exception as e:
        print(f"[PIVOTAL MOMENTS] WARNING: Failed to identify pivotal moments ({e}). Continuing without significance scenes.")
        return []


def generate_significance_scene(pivotal_scene: dict, next_scene: dict | None, subject_name: str, justification: str, chapter_context: str = None) -> dict:
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
    if client is None:
        raise ValueError("OpenAI client not initialized. Set build_scripts_utils.client before calling generate_significance_scene.")
    
    pivotal_id = pivotal_scene.get('id', 0)
    pivotal_title = pivotal_scene.get('title', '')
    pivotal_narration = pivotal_scene.get('narration', '')
    
    next_context = ""
    if next_scene:
        next_title = next_scene.get('title', '')
        next_context = f"\nThe next scene after this will be: \"{next_title}\" - keep this in mind for flow."
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    significance_prompt = f"""Generate a SIGNIFICANCE SCENE that comes immediately after Scene {pivotal_id}: "{pivotal_title}"

PIVOTAL MOMENT (Scene {pivotal_id}):
Title: "{pivotal_title}"
Narration: "{pivotal_narration}"

WHY THIS MOMENT IS PIVOTAL:
{justification}
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

Respond with JSON:
{{
  "title": "2-5 word title about the significance",
  "narration": "1-3 sentences explaining why this pivotal moment matters - its significance, impact, and what it changed. Make viewers FEEL the weight.",
  "image_prompt": "Visual that reflects the significance and weight of this moment - contemplative, monumental, or reflective mood. Use the same year/time period as the pivotal moment. Match the visual style of the documentary. 16:9 cinematic",
  "emotion": "contemplative" or "reflective" or "weighty" or "monumental" - the emotion should reflect the significance being explained,
  "year": Same as pivotal moment (use scene {pivotal_id}'s year)
}}"""

    try:
        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": "You create powerful scenes that explain why pivotal moments matter. You help viewers understand the significance and weight of game-changing moments. Respond with valid JSON only."},
                {"role": "user", "content": significance_prompt}
            ],
            temperature=0.7,
        )
        
        scene = json.loads(clean_json_response(response.choices[0].message.content))
        
        if not isinstance(scene, dict):
            raise ValueError(f"Expected dict, got {type(scene)}")
        
        # Ensure all required fields are present
        if 'title' not in scene:
            scene['title'] = "Why This Moment Matters"
        if 'narration' not in scene:
            scene['narration'] = f"This moment changes everything - its significance reshapes the entire story."
        if 'image_prompt' not in scene:
            scene['image_prompt'] = f"Reflective, contemplative scene showing the weight of this pivotal moment, {pivotal_scene.get('year', 'same period')}, 16:9 cinematic"
        if 'emotion' not in scene:
            scene['emotion'] = "contemplative"
        if 'year' not in scene:
            scene['year'] = pivotal_scene.get('year', 'same period')
        
        return scene
        
    except Exception as e:
        print(f"[SIGNIFICANCE SCENE] WARNING: Failed to generate significance scene for Scene {pivotal_id} ({e}).")
        # Return a simple fallback scene
        return {
            "title": "Why This Moment Matters",
            "narration": f"This moment is pivotal because {justification[:100]}... It changes the trajectory of the entire story.",
            "image_prompt": f"Reflective, contemplative scene showing the significance of this moment, {pivotal_scene.get('year', 'same period')}, 16:9 cinematic",
            "emotion": "contemplative",
            "year": pivotal_scene.get('year', 'same period')
        }


def refine_scenes(scenes: list[dict], subject_name: str, is_short: bool = False, chapter_context: str = None, 
                  diff_output_path: Path | None = None, subject_type: str = "person", skip_significance_scenes: bool = False) -> tuple[list[dict], dict]:
    """
    Refine generated scenes by checking for awkward transitions, weird sentences, and improvements.
    
    Args:
        scenes: List of scene dictionaries to refine
        subject_name: Name of the subject (person, character, region, etc.)
        is_short: Whether these are short video scenes
        chapter_context: Optional chapter context string
        diff_output_path: Optional path to save refinement diff JSON
        subject_type: Type of subject - "person", "character", "region", "topic", etc. (for prompt customization)
        skip_significance_scenes: If True, skip identifying pivotal moments and generating significance scenes (for top 10 lists, etc.)
    
    Returns:
        tuple: (refined_scenes, diff_data) - The refined scenes and a diff dict showing what changed
    """
    if not scenes:
        return scenes, {}
    
    if client is None:
        raise ValueError("OpenAI client not initialized. Set build_scripts_utils.client before calling refine_scenes.")
    
    # Save original scenes for comparison
    original_scenes = [scene.copy() for scene in scenes]
    
    # For main videos (not shorts), identify pivotal moments and insert significance scenes BEFORE refinement
    # UNLESS skip_significance_scenes is True (e.g., for top 10 list videos)
    scenes_before_refinement = [scene.copy() for scene in scenes]
    if not is_short and not skip_significance_scenes:
        print(f"[REFINEMENT] Identifying pivotal moments and adding significance scenes (before refinement)...")
        pivotal_moments = identify_pivotal_moments(scenes_before_refinement, subject_name, chapter_context)
        
        if pivotal_moments:
            print(f"[REFINEMENT]   • Identified {len(pivotal_moments)} pivotal moment(s)")
            
            # Sort pivotal moments by scene_id (descending) so we can insert from end to start
            # This way, scene IDs remain valid as we insert
            pivotal_moments.sort(key=lambda x: x.get('scene_id', 0), reverse=True)
            
            inserted_scenes_count = 0
            
            # Insert significance scenes after each pivotal moment
            for pivotal_moment in pivotal_moments:
                pivotal_scene_id = pivotal_moment.get('scene_id')
                justification = pivotal_moment.get('justification', '')
                
                # Rebuild scene_id_to_index map (indices may have shifted from previous insertions)
                scene_id_to_index = {scene.get('id'): i for i, scene in enumerate(scenes_before_refinement)}
                
                if pivotal_scene_id not in scene_id_to_index:
                    print(f"[REFINEMENT]   WARNING: Pivotal scene ID {pivotal_scene_id} not found, skipping")
                    continue
                
                pivotal_index = scene_id_to_index[pivotal_scene_id]
                pivotal_scene = scenes_before_refinement[pivotal_index]
                
                # Get next scene for context
                next_scene = scenes_before_refinement[pivotal_index + 1] if pivotal_index + 1 < len(scenes_before_refinement) else None
                
                # Generate significance scene
                significance_scene = generate_significance_scene(
                    pivotal_scene=pivotal_scene,
                    next_scene=next_scene,
                    subject_name=subject_name,
                    justification=justification,
                    chapter_context=chapter_context
                )
                
                # Mark this as a significance scene
                significance_scene['is_significance_scene'] = True
                
                # Set temporary ID (will be renumbered later)
                significance_scene['id'] = pivotal_scene_id + 0.5  # Temporary ID between pivotal and next
                
                # Insert significance scene after pivotal moment
                insert_index = pivotal_index + 1
                scenes_before_refinement.insert(insert_index, significance_scene)
                
                inserted_scenes_count += 1
                print(f"[REFINEMENT]   • Added significance scene after Scene {pivotal_scene_id}")
            
            # Renumber all scene IDs to be sequential integers
            # Also ensure all scenes have is_significance_scene set (False for non-significance scenes)
            for i, scene in enumerate(scenes_before_refinement):
                scene['id'] = i + 1
                # Only set to False if not already set to True
                if 'is_significance_scene' not in scene:
                    scene['is_significance_scene'] = False
            
            print(f"[REFINEMENT]   • Inserted {inserted_scenes_count} significance scene(s)")
            print(f"[REFINEMENT]   • Scene count before refinement: {len(scenes_before_refinement)} (was {len(original_scenes)})")
    
    # Now refine ALL scenes (original + significance scenes if any were added)
    print(f"[REFINEMENT] Refining {len(scenes_before_refinement)} scenes (including any significance scenes)...")
    
    # Identify significance scenes and scenes that follow them for transition checking
    significance_scene_info = ""
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
    
    # Prepare scene context for the LLM
    scenes_json = json.dumps(scenes_before_refinement, indent=2, ensure_ascii=False)
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    pacing_note = "For shorts: maintain the concise 1-2 sentence per scene format" if is_short else "Maintain the 2-3 sentence per scene format for main video"
    
    # Adjust prompt based on subject_type
    subject_descriptor = {
        "person": "a documentary about",
        "character": "a documentary about the character",
        "region": "a documentary about the lore of",
        "topic": "a documentary about"
    }.get(subject_type, "a documentary about")
    
    refinement_prompt = f"""You are reviewing and refining scenes for {subject_descriptor} {subject_name}.

CURRENT SCENES (JSON):
{scenes_json}
{context_info}
{significance_scene_info}

CRITICAL: For scenes that contain requests to "like", "subscribe", and "comment" in the narration, you can refine and improve them (clarity, flow, naturalness) BUT you MUST preserve the like/subscribe/comment call-to-action. The CTA is essential and must remain in the narration - refine around it, don't remove it.

YOUR TASK: Review these scenes and improve them. Look for:
1. META REFERENCES (CRITICAL) - Remove ANY references to:
   * "Chapter" or "chapters" - viewers don't know about chapters
   * "In this video" or "In this documentary" or "In this story"
   * "As we'll see" or "Later in this video" or "As we continue"
   * "Let me tell you" or "I want to show you" - just narrate directly
   * Any production elements, scripts, outlines, prompts, or behind-the-scenes info
   * References like "this part" or "this section" or "here we see"
2. OVERLAPPING/DUPLICATE EVENTS (CRITICAL) - If multiple scenes describe the SAME event, moment, or action in detail, consolidate or remove the duplicate. Each scene should cover DIFFERENT events. For example, if Scene 24 describes Antony's suicide attempt and being brought to the mausoleum, Scene 27 should NOT repeat this same event - instead it should focus on what happens NEXT (his death, Cleopatra's response, or the consequences). Remove overlapping content and ensure each scene advances the story with NEW information.
3. MISSING EMOTIONAL ENGAGEMENT - If scenes read too factually without emotional weight, add:
   * How events feel to the character (fear, determination, despair, triumph)
   * The emotional significance and personal stakes
   * Internal experience details (what they're thinking, feeling, fearing)
   * Physical sensations and reactions that create empathy
   * Make events feel significant by connecting them to human emotions
4. EMOTION CONSISTENCY - Ensure the scene's "emotion" field matches the narration tone and image mood:
   * The emotion field should accurately reflect how the scene FEELS
   * Narration tone should match the emotion (e.g., "desperate" → urgent/anxious narration)
   * Image prompt mood should match the emotion (e.g., "desperate" → tense atmosphere in image)
   * If narration or image don't match the emotion field, refine them to be consistent
5. AWKWARD TRANSITIONS - scene endings that don't flow smoothly into the next scene
   * CRITICAL: If significance scenes were inserted after pivotal moments, ensure scenes immediately AFTER significance scenes transition FROM the significance scene, not from the original pivotal scene
   * Example: If Scene 17 is pivotal, Scene 18 is a significance scene (inserted), and Scene 19 follows - Scene 19's beginning should reference/flow from Scene 18, not Scene 17
   * Significance scenes bridge pivotal moments and their consequences - scenes after them should acknowledge this bridge
6. WEIRD OR UNNATURAL SENTENCES - phrases that sound odd when spoken, overly flowery language, vague statements
7. REPETITIVE LANGUAGE - same words or phrases used too frequently
8. CLARITY ISSUES - sentences that are confusing or hard to understand when spoken aloud
9. NARRATION STYLE VIOLATIONS - film directions ("Cut to:", "Smash cut—"), camera directions ("Close-up of", "Wide shot"), or production terminology
10. MISSING CONNECTIONS - scenes that don't reference what came before when they should
11. PACING ISSUES - scenes that feel rushed or too slow for the story beat
12. FACTUAL INCONSISTENCIES - any contradictions or inaccuracies

IMPORTANT GUIDELINES:
- Keep ALL factual information accurate
- Maintain the same scene structure and IDs
- Preserve all fields (id, title, narration, image_prompt, emotion, year, etc.)
- Keep the same tone and style
- {pacing_note}
- Only make changes that IMPROVE clarity, flow, or naturalness
- DO NOT add new information or change facts
- DO NOT add film/camera directions - narration should be pure spoken words
- ABSOLUTELY REMOVE any meta references to chapters, production elements, or the script structure itself
- Write from the YouTuber's perspective - natural storytelling without referencing how it's organized
- CRITICAL: For scenes mentioning "like", "subscribe", and "comment", you can improve them but MUST keep all three CTA elements in the narration

Return the SAME JSON structure with refined scenes. Only change what needs improvement - don't rewrite everything.

Respond with JSON array only (no markdown, no explanation):"""
    
    try:
        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert editor who refines documentary narration for clarity, flow, and naturalness. You catch awkward transitions, weird sentences, style violations, and especially meta references (chapters, production elements, etc.). The narration should feel like natural storytelling from the YouTuber's perspective. Respond with valid JSON array only - same structure as input."},
                {"role": "user", "content": refinement_prompt}
            ],
            temperature=0.3,  # Lower temperature for refinement - more focused changes
        )
        
        refined_scenes = json.loads(clean_json_response(response.choices[0].message.content))
        
        if not isinstance(refined_scenes, list):
            print(f"[REFINEMENT] WARNING: Expected array, got {type(refined_scenes)}. Using original scenes.")
            return scenes, {}
         
        if len(refined_scenes) != len(scenes_before_refinement):
            print(f"[REFINEMENT] WARNING: Scene count changed ({len(scenes_before_refinement)} → {len(refined_scenes)}). Using original scenes.")
            return scenes_before_refinement, {}
         
        # Validate all required fields are present
        for i, scene in enumerate(refined_scenes):
            for field in ["id", "title", "narration", "image_prompt", "emotion", "year"]:
                if field not in scene:
                    print(f"[REFINEMENT] WARNING: Scene {i+1} missing field '{field}'. Using original scenes.")
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
        
        # Generate diff comparing original scenes to final refined scenes (which may include significance scenes)
        diff_data = generate_refinement_diff(original_scenes, refined_scenes)
        # Save diff to file if output path provided
        if diff_output_path:
            diff_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diff_output_path, "w", encoding="utf-8") as f:
                json.dump(diff_data, f, indent=2, ensure_ascii=False)
            print(f"[REFINEMENT]   • Diff saved: {diff_output_path}")
        
        return refined_scenes, diff_data
        
    except Exception as e:
        print(f"[REFINEMENT] WARNING: Refinement failed ({e}). Using original scenes.")
        return scenes, {}
