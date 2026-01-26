"""
Shared utilities for script generation.
Used by both build_script.py (historical documentaries) and build_script_lol.py (LoL lore).
"""
import json
import base64
import time
import re
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
- THIRD PERSON NARRATION (CRITICAL): The narrator is speaking ABOUT the main character, not AS the main character. Always use third person (he/she/they) when referring to the main character. The narrator is telling the story of the person, not speaking in their voice. Examples: "Einstein publishes his paper..." NOT "I publish my paper..." or "We publish our paper..."
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
- CRITICAL: Each scene must cover DIFFERENT, NON-OVERLAPPING events. Do NOT repeat the same event, moment, or action that was already covered in a previous scene. Each scene should advance the story with NEW information, not re-tell what happened before. If an event was already described in detail, move to its consequences or the next significant event instead of describing it again.
- The goal: When scenes are strung together, they should feel like one continuous, connected story, not separate disconnected pieces."""


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
                n=1,
                output_format="jpeg",
                moderation="low",
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


def identify_pivotal_moments(scenes: list[dict], subject_name: str, chapter_context: str = None, excluded_scene_ids: set = None) -> list[dict]:
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
    
    if client is None:
        raise ValueError("OpenAI client not initialized. Set build_scripts_utils.client before calling identify_pivotal_moments.")
    
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
                {"role": "system", "content": "You are an expert story analyst who identifies pivotal moments in narratives. You understand what makes a moment truly significant and game-changing. CRITICAL: Do NOT select scenes from chapter 1 (the hook/preview chapter) or the CTA transition scene as pivotal moments - these scenes should not have significance scenes inserted after them. Only select pivotal moments from scenes after chapter 1 and the CTA scene. Respond with valid JSON array only."},
                {"role": "user", "content": identification_prompt}
            ],
            temperature=0.5,
        )
        
        pivotal_moments = json.loads(clean_json_response(response.choices[0].message.content))
        
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
    if client is None:
        raise ValueError("OpenAI client not initialized. Set build_scripts_utils.client before calling generate_significance_scene.")
    
    pivotal_id = pivotal_scene.get('id', 0)
    pivotal_title = pivotal_scene.get('title', '')
    pivotal_narration = pivotal_scene.get('narration', '')
    pivotal_image_prompt = pivotal_scene.get('image_prompt', '')
    age_instruction = f"\nCRITICAL - AGE: The pivotal scene's image_prompt is: \"{pivotal_image_prompt[:300]}...\". Extract the age information from this (look for phrases like \"26-year-old\", \"in his 40s\", \"at age 20\") and include the SAME age description in your image_prompt. The significance scene should show {subject_name} at the SAME age as in the pivotal moment."
    
    next_context = ""
    if next_scene:
        next_title = next_scene.get('title', '')
        next_context = f"\nThe next scene after this will be: \"{next_title}\" - keep this in mind for flow."
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    # Build narration_instructions based on script type
    if script_type == "horror":
        narration_instructions_desc = "Match the emotion field with brief delivery guidance. Example: 'Speak with terrified urgency, voice trembling.'"
    else:
        narration_instructions_desc = "Match the emotion field with brief delivery guidance. Example: 'Deliver with contemplative weight, emphasizing significance.'"
    
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

Respond with JSON:
{{
  "title": "2-5 word title about the significance",
  "narration": "1-3 sentences explaining why this pivotal moment matters - its significance, impact, and what it changed. Make viewers FEEL the weight.",
  "scene_type": "WHAT" - This scene explains the significance and delivers information about why the moment matters,
  "image_prompt": "Visual that reflects the significance and weight of this moment - contemplative, monumental, or reflective mood. Use the same year/time period AND THE SAME AGE as the pivotal moment (extract age from pivotal scene's image_prompt above). Include {subject_name} at the exact same age as in the pivotal scene. Match the visual style of the documentary. 16:9 cinematic",
  "emotion": "contemplative" or "reflective" or "weighty" or "monumental" - the emotion should reflect the significance being explained,
  "narration_instructions": "{narration_instructions_desc}",
  "year": Same as pivotal moment (use scene {pivotal_id}'s year)
}}"""

    # Adjust narration instructions based on script type
    if script_type == "horror":
        narration_instructions_note = "CRITICAL: narration_instructions should match the scene's emotion field with brief delivery guidance. Keep it simple - just follow the emotion with some context."
    else:
        narration_instructions_note = "CRITICAL: narration_instructions should match the scene's emotion field with brief delivery guidance. Keep it simple - just follow the emotion with some context."
    
    try:
        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": f"You create powerful scenes that explain why pivotal moments matter. You help viewers understand the significance and weight of game-changing moments. {narration_instructions_note} Respond with valid JSON only."},
                {"role": "user", "content": significance_prompt}
            ],
            temperature=0.7,
        )
        
        scene = json.loads(clean_json_response(response.choices[0].message.content))
        
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
            if script_type == "horror":
                scene['narration_instructions'] = f"Speak with {emotion}."
            else:
                scene['narration_instructions'] = f"Deliver with {emotion}."
            print(f"[SIGNIFICANCE SCENE] WARNING: LLM did not provide narration_instructions, generated fallback: {scene['narration_instructions']}")
        if 'image_prompt' not in scene:
            # Use the extracted age_phrase if available, otherwise construct from pivotal scene
            if not age_phrase:
                # Fallback: try to extract age from pivotal scene's image_prompt
                pivotal_img_prompt = pivotal_scene.get('image_prompt', '')
                if pivotal_img_prompt:
                    age_match = re.search(rf'(\d+)[-\s]year[-\s]old\s+{re.escape(subject_name)}|{re.escape(subject_name)}[,\s]+(\d+)[-\s]year[-\s]old|{re.escape(subject_name)}\s+in\s+(?:his|her|their)\s+(\d+)s|{re.escape(subject_name)}\s+at\s+age\s+(\d+)', pivotal_img_prompt, re.IGNORECASE)
                    if age_match:
                        age_num = age_match.group(1) or age_match.group(2) or age_match.group(3) or age_match.group(4)
                        if 'year-old' in age_match.group(0):
                            age_phrase = f"{age_num}-year-old {subject_name}"
                        elif 'in his' in age_match.group(0) or 'in her' in age_match.group(0):
                            age_phrase = f"{subject_name} in his {age_num}s"
                        else:
                            age_phrase = f"{subject_name} at age {age_num}"
            
            # If still no age_phrase, just use subject name
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
        
        return scene
        
    except Exception as e:
        print(f"[SIGNIFICANCE SCENE] WARNING: Failed to generate significance scene for Scene {pivotal_id} ({e}).")
        # Return a simple fallback scene
        return {
            "title": "Why This Moment Matters",
            "narration": f"This moment is pivotal because {justification[:100]}... It changes the trajectory of the entire story.",
            "scene_type": "WHAT",  # Significance scenes explain why something matters - they deliver information
            "image_prompt": f"Reflective, contemplative scene showing the significance of this moment, {pivotal_scene.get('year', 'same period')}, 16:9 cinematic",
            "emotion": "contemplative",
            "year": pivotal_scene.get('year', 'same period')
        }


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
    
    if not scenes or client is None:
        return scenes
    
    # If we're already at or over the max, don't add more scenes
    if max_scenes is not None and len(scenes) >= max_scenes:
        return scenes
    
    scenes_json = json.dumps(scenes, indent=2, ensure_ascii=False)
    
    context_info = ""
    if chapter_context:
        context_info = f"\nCHAPTER CONTEXT: {chapter_context}\n"
    
    if script_type == "horror":
        doc_type = "horror story"
        hanging_focus = "horror threads, threats, or mysteries that were introduced but NOT resolved or left hanging"
        example = "mystery introduced but never explained, threat mentioned but never confronted, object found but never explained"
    else:
        doc_type = "documentary"
        hanging_focus = "storylines, plot threads, or events that were introduced but NOT resolved or completed"
        example = "engagement mentioned but marriage never shown, conflict introduced but resolution never shown"
    
    hanging_storylines_prompt = f"""Analyze these scenes from a {doc_type} about {subject_name} and identify HANGING STORYLINES - {hanging_focus}.

CURRENT SCENES (JSON):
{scenes_json}
{context_info}

HANGING STORYLINES are:
- Events or situations that were introduced but never shown to completion (e.g., {example})
- Plot threads that were set up but left hanging (e.g., {'mystery introduced but never explained, threat mentioned but never confronted' if script_type == 'horror' else 'relationship started but never developed, problem introduced but never solved'})
- Storylines from the chapter's key_events or plot_developments that were mentioned but not fully covered
{'- For horror: Focus on ensuring tension builds properly and no threats are dropped without escalation' if script_type == 'horror' else ''}

Your task: Identify ALL hanging storylines and determine:
1. What storyline/event was introduced but not completed
2. What scene(s) need to be added to complete it
3. Where chronologically these scenes should be inserted (after which scene ID, based on year/timeline)

For each hanging storyline, provide:
- The scene ID where it was introduced
- A description of what's missing (what needs to happen to complete the storyline)
- The year when the completion should occur
- Where to insert the new scene(s) (after which scene ID)

Respond with JSON array:
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
        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": f"You are an expert story analyst who identifies incomplete storylines in {'horror stories' if script_type == 'horror' else 'narratives'}. You understand narrative structure and ensure all introduced plot threads are resolved. {'For horror: Focus on tension building and ensuring threats/mysteries are properly developed.' if script_type == 'horror' else ''} Respond with valid JSON array only."},
                {"role": "user", "content": hanging_storylines_prompt}
            ],
            temperature=0.5,
        )
        
        hanging_storylines = json.loads(clean_json_response(response.choices[0].message.content))
        
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
            
            # Build narration_instructions based on script type
            if script_type == "horror":
                narration_instructions_desc = "Match the emotion field with brief delivery guidance. Example: 'Speak with terrified urgency, voice trembling.'"
            else:
                narration_instructions_desc = "Match the emotion field with brief delivery guidance. Example: 'Deliver with measured authority, conveying completion.'"
            
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

Respond with JSON:
{{
  "title": "2-5 word title",
  "narration": "1-2 sentences that complete the hanging storyline naturally",
  "scene_type": "{scene_type}",
  "image_prompt": "Visual description appropriate for this moment, including {subject_name}'s age at this time ({completion_year}), 16:9 cinematic",
  "emotion": "Appropriate emotion for this moment",
  "narration_instructions": "{narration_instructions_desc}",
  "year": {completion_year}
}}"""

            # Adjust system prompt based on script type
            if script_type == "horror":
                system_content = "You create scenes that complete hanging horror threads in horror stories. You ensure narrative completeness and maintain horror atmosphere. CRITICAL: narration_instructions should match the scene's emotion field with brief delivery guidance. Keep it simple - just follow the emotion with some context. Respond with valid JSON only."
            else:
                system_content = "You create scenes that complete hanging storylines in documentaries. You ensure narrative completeness and chronological accuracy. CRITICAL: narration_instructions should match the scene's emotion field with brief delivery guidance. Keep it simple - just follow the emotion with some context. Respond with valid JSON only."
            
            try:
                completion_response = client.chat.completions.create(
                    model=SCRIPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": completion_prompt}
                    ],
                    temperature=0.7,
                )
                
                new_scene = json.loads(clean_json_response(completion_response.choices[0].message.content))
                
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
                    # Generate fallback narration_instructions based on emotion and script type
                    emotion = new_scene.get('emotion', 'contemplative')
                    if script_type == "horror":
                        new_scene['narration_instructions'] = f"Speak with {emotion} urgency, voice trembling."
                    else:
                        new_scene['narration_instructions'] = f"Deliver with {emotion} weight, emphasizing completion."
                    print(f"[STORYLINE CHECK] WARNING: Generated scene missing narration_instructions, generated fallback: {new_scene['narration_instructions']}")
                
                # Mark as storyline completion scene
                new_scene['is_storyline_completion'] = True
                
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


def refine_scenes(scenes: list[dict], subject_name: str, is_short: bool = False, chapter_context: str = None, 
                  diff_output_path: Path | None = None, subject_type: str = "person", skip_significance_scenes: bool = False,
                  scenes_per_chapter: int = None, script_type: str = "biopic") -> tuple[list[dict], dict]:
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
        scenes_per_chapter: Number of scenes per chapter (used to identify chapter 1 scenes and CTA scene to exclude from pivotal moments)
    
    Returns:
        tuple: (refined_scenes, diff_data) - The refined scenes and a diff dict showing what changed
    """
    if not scenes:
        return scenes, {}
    
    if client is None:
        raise ValueError("OpenAI client not initialized. Set build_scripts_utils.client before calling refine_scenes.")
    
    # Save original scenes for comparison
    original_scenes = [scene.copy() for scene in scenes]
    
    # PASS 1: Check for hanging storylines and add completion scenes
    # SKIP for shorts (shorts are trailers with loose storylines by design)
    if is_short:
        print(f"\n[REFINEMENT PASS 1] Skipped for shorts (trailers have loose storylines by design)")
        scenes_after_storyline_check = [scene.copy() for scene in scenes]
    else:
        if script_type == "horror":
            print(f"\n[REFINEMENT PASS 1] Checking for hanging horror threads and tension building...")
        else:
            print(f"\n[REFINEMENT PASS 1] Checking for hanging storylines...")
        scenes_after_storyline_check = check_and_add_missing_storyline_scenes(scenes, subject_name, chapter_context, max_scenes=None, script_type=script_type)
    
    # PASS 2: For main videos (not shorts), identify pivotal moments and insert significance scenes
    # SKIP for shorts (trailers don't need significance scenes)
    # UNLESS skip_significance_scenes is True (e.g., for top 10 list videos)
    scenes_after_pivotal = [scene.copy() for scene in scenes_after_storyline_check]
    if is_short:
        print(f"\n[REFINEMENT PASS 2] Skipped for shorts (trailers don't need significance scenes)")
    elif not skip_significance_scenes:
        print(f"\n[REFINEMENT PASS 2] Identifying pivotal moments and adding significance scenes...")
        print(f"[REFINEMENT] Identifying pivotal moments and adding significance scenes (before refinement)...")
        
        # Exclude chapter 1 scenes and CTA scene from pivotal moment identification
        excluded_scene_ids = set()
        if scenes_per_chapter is not None:
            # Chapter 1 scenes: scenes 1 to scenes_per_chapter
            chapter_1_scene_ids = set(range(1, scenes_per_chapter + 1))
            excluded_scene_ids.update(chapter_1_scene_ids)
            
            # CTA scene: scene right after chapter 1 (scenes_per_chapter + 1)
            cta_scene_id = scenes_per_chapter + 1
            excluded_scene_ids.add(cta_scene_id)
            
            print(f"[REFINEMENT]   • Excluding chapter 1 scenes (IDs {min(chapter_1_scene_ids)}-{max(chapter_1_scene_ids)}) and CTA scene (ID {cta_scene_id}) from pivotal moment identification")
        
        pivotal_moments = identify_pivotal_moments(scenes_after_pivotal, subject_name, chapter_context, excluded_scene_ids=excluded_scene_ids)
        
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
                    if script_type == "horror":
                        significance_scene['narration_instructions'] = f"Speak with {emotion} urgency, voice trembling."
                    else:
                        significance_scene['narration_instructions'] = f"Deliver with {emotion} weight, emphasizing significance."
                    print(f"[REFINEMENT PASS 2] WARNING: Generated significance scene missing narration_instructions, generated fallback: {significance_scene['narration_instructions']}")
                
                # Mark this as a significance scene
                significance_scene['is_significance_scene'] = True
                
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
        scenes_after_pivotal = scenes_after_storyline_check
    
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
    
    pacing_note = "For shorts: maintain the concise 1-2 sentence per scene format" if is_short else "Maintain the 2-3 sentence per scene format for main video"
    
    # Adjust prompt based on subject_type and script_type
    if script_type == "horror":
        subject_descriptor = "a horror story about"
        narration_style_note = "CRITICAL: This is a FIRST PERSON horror story. All narration must be in first person (I/me/my), present tense. The protagonist is telling their own story. Maintain first person throughout."
        horror_focus = """
HORROR-SPECIFIC REFINEMENT FOCUS:
- TENSION BUILDING: Ensure tension builds progressively - each scene should increase fear/unease
- ATMOSPHERE: Strengthen horror atmosphere through description: shadows, sounds, feelings, unease
- SCARE MOMENTS: Ensure key scare moments are impactful and well-timed
- HORROR PACING: Maintain horror pacing - build tension, deliver scares, create unease
- OPEN ENDING: For final chapter scenes, ensure ending remains open/unresolved to keep viewers scared
- FIRST PERSON: All narration must be in first person (I/me/my) - never use third person
- PRESENT TENSE: Use present tense for immediacy: "I walk...", "I hear...", "I feel..."
- SENSORY DETAILS: Include what I see, hear, feel, smell - make it visceral
- INTERNAL THOUGHTS: Include protagonist's thoughts and reactions: "I think...", "I wonder...", "I'm terrified because..."
- PHYSICAL SENSATIONS: Include physical reactions: "My hands shake...", "My heart races...", "I feel cold..."
"""
    else:
        subject_descriptor = {
            "person": "a documentary about",
            "character": "a documentary about the character",
            "region": "a documentary about the lore of",
            "topic": "a documentary about"
        }.get(subject_type, "a documentary about")
        narration_style_note = "Write from the YouTuber's perspective - natural storytelling without referencing how it's organized."
        horror_focus = ""
    
    refinement_prompt = f"""You are reviewing and refining scenes for {subject_descriptor} {subject_name}.

CURRENT SCENES (JSON):
{scenes_json}
{scene_type_summary}
{context_info}
{significance_scene_info}

CRITICAL: For scenes that contain requests to "like", "subscribe", and "comment" in the narration, you can refine and improve them (clarity, flow, naturalness) BUT you MUST preserve the like/subscribe/comment call-to-action. The CTA is essential and must remain in the narration - refine around it, don't remove it.

{horror_focus}

YOUR TASK: Review these scenes and improve them. Look for:
1. WHY/WHAT SCENE PURPOSE (CRITICAL) - Each scene has a scene_type of either "WHY" or "WHAT". Understand and refine based on their purpose:
   * WHY scenes should: frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered or don't understand the significance of. They create anticipation and make viewers want to see what comes next. CRITICAL FOR RETENTION - AVOID VIEWER CONFUSION: The biggest issue for retention is viewer confusion. WHY scenes MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions. Don't create confusion by being vague about what's happening.
     - MOST IMPORTANT: Make sure viewers understand WHAT IS HAPPENING - provide clear context and establish the situation before introducing questions or mysteries
     - If a WHY scene doesn't create curiosity or anticipation, strengthen it by adding: a mystery to solve, a problem to overcome, a counterintuitive fact, a secret revealed, or something unexpected
     - If a WHY scene is confusing or vague about what's happening, strengthen it by adding: clear context about the situation, specific facts about what's happening, and clear establishment of the story state
     - WHY scenes should hook viewers and make them anticipate the upcoming WHAT section, but NEVER at the expense of clarity about what's happening
     - Examples of good WHY scenes that maintain clarity: "In 1905, Einstein faces an impossible challenge. But how did he manage to...?", "The secret that would change everything was hidden in plain sight. What he didn't know was...", "What nobody realized was that this moment would define everything. The risk was enormous because..."
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
     - Examples of good WHY scenes that maintain clarity: "In 1905, Einstein faces an impossible challenge. But how did he manage to...?", "The secret that would change everything was hidden in plain sight. What he didn't know was...", "What nobody realized was that this moment would define everything. If he failed, everything would be lost...", "The risk was enormous because..."
   * Ensure WHY sections set up what/why/stakes for upcoming WHAT sections - if a WHY scene doesn't establish what will happen, why it matters, and what the stakes are, refine it to do so
   * Ensure WHAT sections clearly communicate what/why/stakes - if a WHAT scene doesn't clearly explain what's happening, why it's important, and what the stakes are, refine it to do so
2. META REFERENCES (CRITICAL) - Remove ANY references to:
   * "Chapter" or "chapters" - viewers don't know about chapters
   * "In this video" or "In this documentary" or "In this story"
   * "As we'll see" or "Later in this video" or "As we continue"
   * "Let me tell you" or "I want to show you" - just narrate directly
   * Any production elements, scripts, outlines, prompts, or behind-the-scenes info
   * References like "this part" or "this section" or "here we see"
3. OVERLAPPING/DUPLICATE EVENTS (CRITICAL) - If multiple scenes describe the SAME event, moment, or action in detail, consolidate or remove the duplicate. Each scene should cover DIFFERENT events. For example, if Scene 24 describes Antony's suicide attempt and being brought to the mausoleum, Scene 27 should NOT repeat this same event - instead it should focus on what happens NEXT (his death, Cleopatra's response, or the consequences). Remove overlapping content and ensure each scene advances the story with NEW information.
4. MISSING EMOTIONAL ENGAGEMENT - If scenes read too factually without emotional weight, add:
   * How events feel to the character (fear, determination, despair, triumph)
   * The emotional significance and personal stakes
   * Internal experience details (what they're thinking, feeling, fearing)
   * Physical sensations and reactions that create empathy
   * Make events feel significant by connecting them to human emotions
5. EMOTION CONSISTENCY - Ensure the scene's "emotion" field matches the narration tone and image mood:
   * The emotion field should accurately reflect how the scene FEELS
   * Narration tone should match the emotion (e.g., "desperate" → urgent/anxious narration)
   * Image prompt mood should match the emotion (e.g., "desperate" → tense atmosphere in image)
   * If narration or image don't match the emotion field, refine them to be consistent
6. VIEWER CONFUSION (CRITICAL FOR RETENTION) - The biggest issue for retention is viewer confusion. Ensure WHY scenes make it clear what is happening:
   * WHY scenes MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions
   * If a WHY scene is confusing or vague about what's happening, strengthen it by adding: clear context about the situation, specific facts about what's happening, and clear establishment of the story state
   * Don't create confusion by being vague - viewers should always understand what is happening in the story, even when mysteries or questions are being introduced
   * Examples of clear WHY scenes: "In 1905, Einstein faces an impossible challenge. But how did he manage to...?" (clear context first, then question) vs. "But how did he manage to...?" (confusing - no context)
7. SEAMLESS JOURNEY AND CONNECTIONS (CRITICAL) - Ensure scenes feel connected, not like consecutive pieces of disjoint information:
   * Each scene should build on the previous scene - reference what came before naturally, show how events connect
   * Scenes should feel like a flowing narrative where each scene grows from the last, not like separate disconnected facts
   * If scenes feel disconnected or like they're just listing information (A and B and C), strengthen connections by:
     - Referencing events, themes, or emotions from previous scenes
     - Showing cause-and-effect relationships between scenes
     - Continuing threads or plot elements established earlier
     - Creating logical progression where each scene feels like the natural next step
   * Use WHY/WHAT interleaving to create natural connections - WHY scenes should set up questions that the following WHAT scenes answer
   * The goal: When scenes are strung together, they should feel like one continuous, connected story
8. AWKWARD TRANSITIONS - scene endings that don't flow smoothly into the next scene
   * CRITICAL: If significance scenes were inserted after pivotal moments, ensure scenes immediately AFTER significance scenes transition FROM the significance scene, not from the original pivotal scene
   * Example: If Scene 17 is pivotal, Scene 18 is a significance scene (inserted), and Scene 19 follows - Scene 19's beginning should reference/flow from Scene 18, not Scene 17
   * Significance scenes bridge pivotal moments and their consequences - scenes after them should acknowledge this bridge
9. WEIRD OR UNNATURAL SENTENCES - phrases that sound odd when spoken, overly flowery language, vague statements
10. REPETITIVE LANGUAGE - same words or phrases used too frequently
11. CLARITY ISSUES - sentences that are confusing or hard to understand when spoken aloud
12. NARRATION STYLE VIOLATIONS - film directions ("Cut to:", "Smash cut—"), camera directions ("Close-up of", "Wide shot"), or production terminology
13. MISSING CONNECTIONS - scenes that don't reference what came before when they should
14. PACING ISSUES - scenes that feel rushed or too slow for the story beat
15. FACTUAL INCONSISTENCIES - any contradictions or inaccuracies

IMPORTANT GUIDELINES:
- Keep ALL factual information accurate
- Maintain the same scene structure and IDs
- Preserve all fields (id, title, narration, image_prompt, emotion, year, scene_type, narration_instructions, etc.)
- CRITICAL: Preserve the WHY/WHAT structure - maintain each scene's scene_type field (WHY or WHAT). WHY sections frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered or don't understand the significance of. WHY sections should set up what will happen, why it matters, and what the stakes are for upcoming WHAT sections. WHAT sections deliver content/solutions and must clearly communicate what is happening, why it's important, and what the stakes are. Ensure WHY sections create anticipation for upcoming WHAT sections by establishing what/why/stakes.
- CRITICAL: Preserve and refine narration_instructions - Each scene MUST have narration_instructions that match the scene's emotion field with brief delivery guidance. When refining, ensure narration_instructions accurately reflect the scene's emotion. If narration_instructions don't match the refined narration or emotion, update them to be consistent. Keep it simple - just follow the emotion with some context. ABSOLUTELY DO NOT remove narration_instructions - every scene must have this field. If a scene is missing narration_instructions, add it based on the scene's emotion field.
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
- The "narration_instructions" field is REQUIRED for every scene - it must match the scene's emotion field with brief delivery guidance
- DO NOT omit any fields from the JSON - every scene must have the complete structure
- If a scene is missing narration_instructions, add it based on the scene's emotion field

Respond with JSON array only (no markdown, no explanation). Each scene object must include: id, title, narration, image_prompt, emotion, year, scene_type, narration_instructions."""
    
    try:
        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": f"You are an expert editor who refines {'horror story' if script_type == 'horror' else 'documentary'} narration for clarity, flow, and naturalness. CRITICAL FOR RETENTION - AVOID VIEWER CONFUSION: The biggest issue for retention is viewer confusion. WHY scenes MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions. Don't create confusion by being vague about what's happening. CRITICAL: Create a SEAMLESS JOURNEY through the video - scenes should feel CONNECTED, not like consecutive pieces of disjoint information (A and B and C). Each scene should build on the previous scene, reference what came before naturally, and show how events connect. You understand the WHY/WHAT paradigm: WHY scenes frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered - they create anticipation by setting up what will happen, why it matters, and what the stakes are for upcoming WHAT sections. WHY scenes MUST clearly establish what is happening (MOST IMPORTANT for retention) before introducing questions or mysteries. WHAT scenes deliver core content, solutions, and information - they satisfy anticipation by clearly communicating what is happening, why it's important, and what the stakes are (what can go wrong, what can go right, what's at risk). {'CRITICAL FOR HORROR: All narration must be in FIRST PERSON (I/me/my), present tense. The protagonist is telling their own story. Focus on tension building, atmosphere, and horror pacing. For final chapter, ensure open ending that keeps viewers scared.' if script_type == 'horror' else ''} You catch awkward transitions, weird sentences, style violations, and especially meta references (chapters, production elements, etc.). You ensure scenes feel connected and build on each other, WHY scenes clearly establish what is happening (avoiding confusion) and set up what/why/stakes for upcoming WHAT sections, and WHAT scenes clearly communicate what/why/stakes. {narration_style_note} CRITICAL JSON REQUIREMENT: Every scene in your response MUST include the 'narration_instructions' field. This field must match the scene's emotion field with brief delivery guidance (e.g., 'Speak with tense urgency, voice trembling.' or 'Deliver with contemplative weight, emphasizing significance.'). DO NOT omit this field from any scene. Respond with valid JSON array only - same structure as input, including narration_instructions for every scene."},
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
        
        # For shorts, cap at 5 scenes maximum after refinement
        if is_short and len(refined_scenes) > 5:
            print(f"[REFINEMENT PASS 3]   • Shorts mode: capping at 5 scenes after refinement (was {len(refined_scenes)})")
            refined_scenes = refined_scenes[:5]
            # Renumber scenes
            for i, scene in enumerate(refined_scenes):
                scene['id'] = i + 1
         
        # Validate all required fields are present and fix missing narration_instructions
        for i, scene in enumerate(refined_scenes):
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
                # Generate fallback narration_instructions based on emotion and script type
                emotion = scene.get('emotion', 'contemplative')
                if script_type == "horror":
                    scene['narration_instructions'] = f"Speak with {emotion} urgency, voice trembling."
                else:
                    scene['narration_instructions'] = f"Deliver with {emotion} weight, emphasizing significance."
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
                # Generate fallback narration_instructions based on emotion and script type
                emotion = scene.get('emotion', 'contemplative')
                if script_type == "horror":
                    scene['narration_instructions'] = f"Speak with {emotion} urgency, voice trembling."
                else:
                    scene['narration_instructions'] = f"Deliver with {emotion} weight, emphasizing significance."
                print(f"[REFINEMENT] CRITICAL FIX: Scene {i+1} (ID: {scene.get('id', i+1)}) missing narration_instructions after refinement, generated fallback: {scene['narration_instructions']}")
        
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
