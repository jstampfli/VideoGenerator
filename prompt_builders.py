"""
Modular prompt builders for script generation.
Extracts shared prompt components to avoid duplication and enable easy customization.
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from research_utils import ResearchContext


def get_biopic_audience_profile() -> str:
    """
    Audience profile for biopic documentaries.
    Primary audience: 65-year-old American men. Tilts toward conservative, traditional American rhetoric.
    """
    return """TARGET AUDIENCE - CRITICAL (65-year-old American men):
- TONE: Authoritative, substantive, and respectful - like a PBS documentary or Ken Burns. They appreciate depth over flash.
- PACING: Deliberate and clear. Give events room to breathe. Avoid rapid-fire cuts or TikTok-style urgency - they prefer stories that unfold.
- THEMES THAT RESONATE: Legacy, perseverance, American values, leadership, what it means to build something lasting. Military history, political leadership, inventors, business titans - figures who shaped the world they grew up in.
- LANGUAGE: Clear, straightforward, no slang or Gen Z references. Let facts speak for themselves—avoid sensationalist phrasing, gimmicky rhetorical hooks, or repeated dramatic framing in the narration. No "shocking", "incredible", "what nobody expected", or similar hype.
- VOICE & RHETORIC: Use conservative, traditional American rhetoric—the language of their generation. Avoid politically correct or liberal academic terminology. Use factual, neutral descriptors instead of loaded terms.
- AVOID (liberal/PC terms that alienate this audience): "privilege", "privileged", "born into privilege", "marginalized", "oppressed", "colonization", "toxic", "problematic", "privileged upbringing". Instead use: "wealthy family", "family of means", "from a prominent family", "born to wealth", "advantage", "family fortune"—factual alternatives that describe the same reality without the political framing.
- AVOID: Overly casual tone, flashy editing language, references that would confuse someone who didn't grow up with social media."""


def get_three_principles_prompt() -> str:
    """The three retention principles every scene must satisfy. Use in all script generation."""
    return """THREE PRINCIPLES (CRITICAL - every scene must satisfy all three):
1. The viewer must know WHAT IS HAPPENING - establish the situation, who is involved, where we are, what is going on. Never leave viewers confused about what's happening.
2. The viewer must know WHY IT IS IMPORTANT - why this moment matters, its significance, impact, or meaning. Make the stakes of the story clear.
3. The viewer must know WHAT COULD GO WRONG - what's at risk, what failure would mean, what the consequences are. Show what can go wrong (or right) so the moment has weight."""


def get_why_what_paradigm_prompt(is_trailer: bool = False, is_hook_chapter: bool = False) -> str:
    """
    Get WHY/WHAT paradigm instructions.
    
    Args:
        is_trailer: If True, emphasizes WHY scenes for trailers
        is_hook_chapter: If True, emphasizes WHY scenes for hook chapters
    """
    base_prompt = f"""WHY/WHAT PARADIGM (CRITICAL):

{get_three_principles_prompt()}

Apply these three principles to EVERY scene. WHY and WHAT sections below are how you deliver them.

- Every scene must be classified as either "WHY" or "WHAT"
- WHY sections: Pull audience in by framing mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggesting there's something we haven't considered or don't understand the significance of. CRITICAL FOR RETENTION - AVOID VIEWER CONFUSION: The biggest issue for retention is viewer confusion. WHY sections MUST ensure the viewer knows WHAT IS HAPPENING in the story - provide clear context, establish the situation, and make sure viewers understand the basic facts before introducing mysteries or questions. Don't create confusion by being vague about what's happening. CRITICAL: WHY sections should set up the upcoming WHAT section by establishing:
  * WHAT IS HAPPENING: Clearly establish what's happening in the story/situation so viewers aren't confused (this is the MOST IMPORTANT for retention)
  * WHAT will happen next (or what question/problem needs to be addressed)
  * WHY it matters (the importance or significance)
  * WHAT THE STAKES ARE (what can go wrong, what can go right, what's at risk)
  Examples (NOTE: Always establish what is happening FIRST, then introduce the mystery/question. Keep it straightforward—avoid sensationalist phrasing.):
  * Frame a mystery, problem, question, or obstacle: "In 1905, Einstein faces a major challenge. How did he manage to...?" (clear context first) OR "The challenge was significant. What obstacle would block his path?" (clear situation first)
  * Hook with something counterintuitive: "In June 1858, Darwin receives a letter that changes his plans. What he finds inside surprises him." (clear context first) OR "What happened next was unexpected." (establish what happened first)
  * Hook with a secret: "Something important was about to be revealed." (establish context first) OR "The discovery would reshape his work." (clear that something exists)
  * Suggest something we haven't considered or don't understand the significance of: "But there was something else at play that nobody realized..." (establish that something exists) OR "What nobody realized was that this moment would define everything. The true significance wouldn't be clear until..." (clear context about the moment first)
  * Set up stakes: "If he failed, everything would be lost. Success would mean...", "The risk was enormous because..." (establish the situation first)
  * Set up what comes next: Make viewers anticipate the upcoming WHAT section by establishing what is happening, what will happen next, why it matters, and what's at stake
- WHAT sections: Deliver core content, solutions, actual information and details. CRITICAL: Every WHAT scene must clearly communicate:
  * WHAT is happening: The specific events, actions, or information
  * WHY it's important: The significance, impact, or meaning of what's happening
  * WHAT THE STAKES ARE: What can go wrong, what can go right, what's at risk, what success/failure means
  Examples:
  * Provide solutions or answers: "He solved it by...", "The breakthrough came when...", "Here's what actually happened..."
  * Give specific facts and details: "In 1905, Einstein published...", "The paper contained four revolutionary ideas..."
  * Explain what happened: "The experiment proved...", "The result was...", "This led to..."
  * Show stakes: "If this failed, he would lose everything...", "Success meant...", "The risk was enormous because..."
  * Satisfy the anticipation created by WHY sections
- INTERLEAVING STRATEGY:
  * WHY sections should create anticipation for upcoming WHAT sections
  * Can have multiple WHAT sections between WHY sections (flexible ratio)
  * WHY sections should always set up what comes next
  * Start with WHY to hook viewers, then deliver WHAT content
  * Insert WHY sections strategically throughout to maintain interest"""
    
    if is_trailer:
        return """TRAILER FORMAT - MOSTLY WHY SCENES:
- This is a TRAILER, so scenes should be MOSTLY or ONLY WHY scenes
- WHY scenes: Create curiosity, frame mysteries, problems, questions, obstacles, counterintuitive information, secrets, or suggest there's something we haven't considered
- CRITICAL: WHY scenes MUST ensure the viewer knows WHAT IS HAPPENING - provide clear context before introducing mysteries
- Examples of good WHY trailer scenes (straightforward, not sensationalist):
  * "In 1905, Einstein faces a major challenge. How did he manage to...?" (clear context, then mystery)
  * "A discovery was about to change his work. What he found would surprise him." (establish context, then question)
  * "This moment would prove significant. The stakes were high because..." (clear context, then stakes)
- The goal: Make viewers NEED to watch the full documentary to get answers
- Use WHAT scenes SPARINGLY (only if absolutely necessary for context) - trailers should tease, not fully reveal"""
    
    if is_hook_chapter:
        return base_prompt + "\n- For hook chapters: Should be mostly WHY sections since they're previews that create interest"
    
    return base_prompt


def get_emotion_generation_prompt(chapter_emotional_tone: Optional[str] = None) -> str:
    """
    Get emotion generation instructions.
    
    Args:
        chapter_emotional_tone: Optional emotional tone from chapter to align with
    """
    base_prompt = """EMOTION GENERATION (CRITICAL):
- Each scene MUST include an "emotion" field - a single word or short phrase (e.g., "tense", "triumphant", "desperate", "contemplative", "exhilarating", "somber", "urgent", "defiant")
- Base the emotion on: what the character is feeling at this moment, the dramatic tension, and the significance of the event"""
    
    if chapter_emotional_tone:
        base_prompt += f"\n- The emotion should align with the chapter's emotional tone ({chapter_emotional_tone}) but be scene-specific"
    
    base_prompt += """
- Use the emotion to guide both narration tone/style and image mood:
  * If emotion is "desperate" - narration should feel urgent/anxious with short, sharp sentences; image should show tense atmosphere, frantic expressions
  * If emotion is "triumphant" - narration should feel uplifting with elevated language; image should show celebration, confident expressions
  * If emotion is "contemplative" - narration should be slower, reflective; image should show quiet mood, thoughtful expressions
- The emotion field will be used to ensure narration tone and image mood match the emotional reality of the moment"""
    
    return base_prompt


def get_image_prompt_guidelines(
    person: str,
    birth_year: Optional[int] = None,
    death_year: Optional[int] = None,
    aspect_ratio: str = "16:9 cinematic",
    is_trailer: bool = False,
    recurring_themes: Optional[str] = None,
    is_horror: bool = False
) -> str:
    """
    Get image prompt guidelines.
    
    Args:
        person: Name of the person (or protagonist name for horror)
        birth_year: Birth year for age validation (not used for horror)
        death_year: Death year for age validation (not used for horror)
        aspect_ratio: Image aspect ratio (default "16:9 cinematic" for main video, "9:16 vertical" for shorts)
        is_trailer: If True, uses trailer-specific image guidelines
        recurring_themes: Optional recurring themes to reinforce visually
        is_horror: If True, uses horror-specific image guidelines
    """
    # Evaluate expressions before formatting
    birth_year_str = str(birth_year) if birth_year else 'unknown'
    death_year_str = str(death_year) if death_year else 'still alive'
    
    if is_horror:
        # Horror-specific image guidelines
        base = f"""IMAGE PROMPT STYLE - HORROR:
- Dark, shadowy, eerie atmosphere
- Horror mood: tense, fearful, mysterious, unsettling
- Dramatic lighting with HIGH CONTRAST (chiaroscuro) - deep shadows, harsh highlights
- Color palette: blues, grays, deep shadows, muted tones
- Specific composition (close-up for fear, wide shot for isolation, etc.)
- CRITICAL - EMOTION AND MOOD (MUST MATCH SCENE'S EMOTION FIELD):
  * The scene's "emotion" field MUST be reflected in the image_prompt - this is critical for visual consistency
  * Use the emotion to guide lighting, composition, facial expressions, and overall mood
  * Examples: "terrified" → wide eyes, tense body, dark shadows; "tense" → sharp contrasts, wary expressions, claustrophobic framing; "atmospheric" → soft shadows, mysterious mood, eerie lighting; "dread-filled" → heavy shadows, oppressive atmosphere
  * Explicitly include the emotion in image description: "terrified expression", "tense atmosphere", "dread-filled mood"
  * The visual mood must match the emotional reality of the moment - if emotion is "terrified", the image should look terrifying
- Horror elements: shadows, darkness, eerie lighting, unsettling atmosphere
- Focus on MOOD and ATMOSPHERE rather than explicit gore
- Show fear, tension, unease through composition and lighting
- Symbolic horror elements: shadows suggesting presence, objects that create unease, environments that feel wrong
- No historical/period accuracy needed - focus on horror atmosphere
- Include {person} (the protagonist) in first-person perspective scenes when appropriate
- End with ", {aspect_ratio}\""""
        
        if recurring_themes:
            base += f"""
- VISUAL THEME REINFORCEMENT: If recurring themes include concepts like "isolation", "paranoia", "unseen threat", etc., incorporate visual motifs that reinforce these themes through composition, lighting, or symbolism. For example, if "isolation" is a theme, use wide shots with the subject alone, or shadows that emphasize separation.
- Recurring Horror Themes to Consider Visually: {recurring_themes}"""
        
        return base
    
    if is_trailer:
        base = f"""IMAGE PROMPTS:
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
  * Birth year: {birth_year_str}
  * Death year: {death_year_str}
  * If the scene takes place before the birth year, note that {person} is NOT BORN YET and should not appear in the scene
  * If the scene takes place after the death year (and death year is known), note that {person} is DECEASED and should not appear in the scene (unless it's a memorial/legacy scene)
  * Include period-accurate clothing, hairstyle, and any age-relevant details about their appearance at that specific age
- End with ", 9:16 vertical\""""
    else:
        base = f"""IMAGE PROMPT STYLE:
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
  * Birth year: {birth_year_str}
  * Death year: {death_year_str}
  * If the scene takes place before the birth year, note that {person} is NOT BORN YET and should not appear in the scene
  * If the scene takes place after the death year (and death year is known), note that {person} is DECEASED and should not appear in the scene (unless it's a memorial/legacy scene)
  * Include period-accurate clothing, hairstyle, and any age-relevant details about their appearance at that specific age"""
        
        if recurring_themes:
            base += f"""
- VISUAL THEME REINFORCEMENT: If recurring themes include concepts like "isolation", "ambition", "conflict", etc., incorporate visual motifs that reinforce these themes through composition, lighting, or symbolism. For example, if "isolation" is a theme, use wide shots with the subject alone, or shadows that emphasize separation.
- Recurring Themes to Consider Visually: {recurring_themes}"""
        
        base += f"\n- End with \", {aspect_ratio}\""
    
    return base


def get_trailer_narration_style() -> str:
    """Get narration style specifically for trailers/shorts."""
    return """NARRATION STYLE:
- Write from the YouTuber's perspective - this is YOUR script, YOU are telling the story
- Use third person narration - you are speaking ABOUT the main character (he/she/they), not AS the main character
- HIGH ENERGY - every sentence should grab attention
- Simple, clear, punchy language
- Create curiosity gaps - tease information but don't fully reveal
- Make viewers NEED to watch the full documentary
- QUOTES (CRITICAL FOR TTS): Use quotes ONLY around proper nouns you want to emphasize (e.g. titles of works, key terms). Otherwise do NOT use any quotes—they will mess up the text-to-speech."""


def get_trailer_structure_prompt() -> str:
    """Get trailer structure instructions for 4-scene biopic shorts (scene 4 answers the question from scene 3)."""
    return """TRAILER STRUCTURE (exactly 4 scenes - scenes 1-3 WHY, scene 4 WHAT):
1. SCENE 1: Expand the hook with HIGH ENERGY. Grab attention immediately. Set up the mystery, problem, or question from the hook. Make viewers curious: "What is this about?" "Why does this matter?" "What happens next?"
2. SCENE 2: Build the hook - escalate curiosity, deepen the mystery, add more intrigue. Create more questions. Make viewers NEED to know: "But wait, there's more..." "What makes this shocking is..." "The twist nobody expected..."
3. SCENE 3: Create anticipation - end with a compelling QUESTION that demands an answer. Pose the central mystery explicitly so scene 4 can answer it. Examples:
   - "But how did this one moment change everything?"
   - "So how did he actually stop them?"
   - "What did he do next that saved the day?"
   - "How did something so simple resolve the crisis?"
   Scene 3 must END with a clear question that scene 4 will answer.
4. SCENE 4: ANSWER the question posed at the end of scene 3. This is a WHAT scene - deliver the payoff. Give the viewer the resolution so they feel satisfied: what actually happened, how it worked, or why it mattered. Use clear, punchy facts. Do NOT end with a CTA to watch the full documentary—it hurts view duration; end with a satisfying resolution only."""


def build_video_questions_prompt(person_of_interest: str, research_context: "ResearchContext | None" = None) -> str:
    """
    Build the prompt for generating the question(s) this documentary will answer.
    This is the FIRST step - the questions frame everything else.
    These questions are the PRIMARY HOOK that gets users to click and watch.
    """
    from research_utils import get_research_context_block
    research_block = get_research_context_block(research_context) if research_context else ""
    return f"""Identify 1-3 questions that this documentary about {person_of_interest} will answer.
{research_block}
{get_biopic_audience_profile()}

CRITICAL - THESE QUESTIONS ARE YOUR #1 HOOK:
These questions appear at the very start of the video. They are what make viewers stop scrolling and click. They must be as engaging and interesting as possible—framed specifically for this audience. A great question makes someone think "I need to know the answer."

MAKE THEM COMPELLING:
- Tap into themes that resonate: legacy, perseverance, leadership, American values, what it means to build something lasting, how individuals shaped the world
- Create genuine curiosity—not clickbait, but substantive questions that promise a satisfying answer
- Each question should feel like the opening of a story you can't ignore
- Prefer questions that hint at struggle, triumph, sacrifice, or unexpected turns
- Avoid dry or academic phrasing; make them human and gripping

QUESTION PARADIGMS (use when they naturally fit):
The most engaging questions often follow one of these paradigms—but ONLY if it naturally fits the person's story AND the documentary can deliver a satisfying answer. Do not force a paradigm if it doesn't fit or if we can't answer it well.
- Counterintuitive: Something that defies expectations (e.g., "Why did the man who preached peace ultimately embrace war?")
- Somewhat secret or unknown: A lesser-known angle, detail, or truth (e.g., "What secret did he take to his grave that shaped his decisions?")
- Well known but misunderstood: A familiar moment or fact that most people get wrong (e.g., "What did everyone get wrong about his greatest speech?")
If none of these fit naturally, a strong substantive question is better than forcing a paradigm.

CRITICAL RULES:
- If you provide multiple questions, each MUST ask something FUNCTIONALLY DIFFERENT. Do NOT ask variations of the same question.
  - BAD: "How did X become who they were?" and "How did a sickly child become successful?" (same question, rephrased)
  - GOOD: "How did X overcome their physical limitations?" and "What drove X to break from convention?" and "Why did X's greatest achievement almost fail?"
- Each question should be substantive—something the documentary will spend real time answering.
- Tone: Authoritative and substantive (PBS/Ken Burns), not sensationalist. No "shocking" or "incredible"—create interest through substance and stakes.

Return JSON:
{{
  "video_questions": ["Question 1", "Question 2 (optional)", "Question 3 (optional)"]
}}"""


def build_landmark_events_prompt(person_of_interest: str, num_landmarks: int = 4,
                                  research_context: "ResearchContext | None" = None) -> str:
    """
    Build the prompt for identifying the most important landmark events in a person's life.
    These will become their own chapters with deep, in-depth coverage.

    Args:
        person_of_interest: Name of the person
        num_landmarks: Number of landmark events to identify (3-6)
    """
    from research_utils import get_research_context_block
    research_block = get_research_context_block(research_context) if research_context else ""
    return f"""Identify the {num_landmarks} most iconic, pivotal moments in the life of {person_of_interest}.
{research_block}
These are the moments that define their legacy—major works, breakthroughs, critical decisions, or events that changed history. They deserve DEEP, DETAILED treatment in a documentary—not a single scene, but multiple scenes exploring technical details, significance, and impact.

For each landmark event, provide:
- "event_name": The name of the event, work, or moment (e.g., "The Last Supper", "The Annus Mirabilis", "The Nobel Prize")
- "year_or_period": When it occurred (e.g., "1495-1498", "1905", "1921")
- "significance": Why this moment matters—its impact on history, the field, or the person's legacy
- "key_details_to_cover": 3-5 SPECIFIC technical, historical, or artistic details that MUST be included when we cover this event. These are the facts viewers should learn. Examples:
  - For a painting: "All perspective lines converge to a single vanishing point on Christ's forehead", "Oil-and-tempera on dry plaster experiment (technical gamble)"
  - For a scientific breakthrough: "The four papers published in 1905", "How the photoelectric effect proved light is quantized"
  - For a decision: "The specific terms of the deal", "What he risked by accepting"

Be specific. "Key details" should be concrete facts, techniques, or historical specifics—not vague themes.

Return JSON:
{{
  "landmark_events": [
    {{
      "event_name": "...",
      "year_or_period": "...",
      "significance": "...",
      "key_details_to_cover": ["specific detail 1", "specific detail 2", "specific detail 3", ...]
    }},
    ...
  ]
}}"""


def build_outline_prompt(person_of_interest: str, chapters: int, target_total_scenes: int, min_scenes: int = 2, max_scenes: int = 10,
                        available_moods: list[str] | None = None, landmark_events: list[dict] | None = None,
                        video_questions: list[str] | None = None, research_context: "ResearchContext | None" = None) -> str:
    """
    Build the outline generation prompt with flexible chapter structure.

    Args:
        person_of_interest: Name of the person
        chapters: Number of chapters
        target_total_scenes: Target total scenes (outline distributes across chapters)
        min_scenes: Minimum scenes per chapter
        max_scenes: Maximum scenes per chapter
        available_moods: List of music mood folder names (e.g., ["relaxing", "passionate", "happy"]) for LLM to pick per chapter
        landmark_events: Optional list of landmark event dicts; when provided, each becomes its own chapter with 6-10 scenes
        video_questions: The question(s) this documentary will answer - the outline MUST be structured to answer each one
    """
    moods_str = ", ".join(available_moods) if available_moods else "relaxing, passionate, happy"

    questions_section = ""
    if video_questions:
        q_list = "\n".join(f"- {q}" for q in video_questions)
        questions_section = f"""
THE QUESTIONS THIS DOCUMENTARY WILL ANSWER (CRITICAL - FRAME EVERYTHING):
These questions were generated first. Your outline MUST be structured so the documentary answers each one. Every chapter should advance toward answering at least one of these. Chapter 1 must explicitly ask all of them.

{q_list}

"""
    landmark_section = ""
    if landmark_events:
        lines = []
        for lm in landmark_events:
            details = lm.get('key_details_to_cover', [])
            details_str = "; ".join(details[:4]) + ("..." if len(details) > 4 else "")
            lines.append(f"- \"{lm.get('event_name', '')}\" ({lm.get('year_or_period', '')}): {lm.get('significance', '')}\n  Key details to cover: {details_str}")
        landmark_list = "\n".join(lines)
        landmark_section = f"""
LANDMARK EVENTS - When your story reaches these events, assign 6-8 num_scenes to that chapter so we can cover them in depth:
{landmark_list}

Landmarks can share a chapter with context (e.g. "Milan and the Last Supper") or be their own chapter—choose what serves the narrative flow. Total chapters = {chapters}.
"""
    from research_utils import get_research_context_block
    research_block = get_research_context_block(research_context) if research_context else ""
    return f"""You are a master documentary filmmaker. Create a compelling narrative outline for a ~20 minute documentary about: {person_of_interest}
{research_block}
{get_biopic_audience_profile()}

This will be a documentary with EXACTLY {chapters} chapters and approximately {target_total_scenes} total scenes. Think of this as a FEATURE FILM with continuous story arcs, not disconnected episodes.
{questions_section}{landmark_section}
FLEXIBLE CHAPTER STRUCTURE (CRITICAL):
Chapters can be different types - not just chronological time chunks:
- "chronological": Covers a time period (e.g., "Early Years 1879-1900", "The Berlin Years 1914-1933")
- "event": Centers on one major event (e.g., "The Annus Mirabilis 1905", "The Nobel Prize 1921")
- "work": Centers on a specific creation (e.g., "The Theory of Relativity", "The Origin of Species")
- "theme": Centers on a relationship or conflict (e.g., "The Rivalry with Hilbert", "Exile and Legacy")

Assign "num_scenes" (between {min_scenes} and {max_scenes}) per chapter based on significance. Dense periods or major works deserve more scenes (6-10); sparse transitions can be brief (2-3). The sum of all chapter num_scenes should approximate {target_total_scenes} (e.g., {target_total_scenes - 4}-{target_total_scenes + 6}).

NARRATIVE STRUCTURE:
- The documentary should feel like ONE CONTINUOUS STORY, not a list of facts
- Each chapter should FLOW into the next with clear cause-and-effect
- Plant SEEDS in early chapters that PAY OFF later (foreshadowing)
- Build RECURRING THEMES that echo throughout (e.g., isolation, ambition, sacrifice)
- Create emotional MOMENTUM that builds to a climax, then resolves

For each of the {chapters} chapters, provide:
- "chapter_num": 1-{chapters}
- "chapter_type": "chronological" | "event" | "work" | "theme"
- "title": A compelling chapter title
- "year_range": The years covered (required for chronological/event/work; optional for theme - use "various" if spanning many years)
- "num_scenes": Integer {min_scenes}-{max_scenes} - how many scenes this chapter needs based on significance
- "summary": 2-3 sentences about what happens
- "key_events": 4-6 specific dramatic moments to show (for event/work chapters with one focus, key_events can break down that focus)
- "emotional_tone": The mood of this chapter
- "music_mood": Exactly one of: {moods_str}. Pick the music mood that best fits this chapter's emotional tone and content.
- "dramatic_tension": What conflict drives this chapter
- "connects_to_next": How this chapter sets up or flows into the next one
- "recurring_threads": Which themes/motifs from earlier chapters appear here

FULL LIFE SPAN - CRITICAL:
- The documentary MUST cover the person's ENTIRE life from birth to death. No time periods may be skipped.
- Chapter 2 (the first story chapter after the hook) MUST begin with birth and early life. The hook ends with "But how did they get here? Let's start from the beginning"—so the chronological story MUST start there. Do NOT jump to adulthood; viewers need to see where they came from.
- Sparse periods (e.g. childhood with little documented detail) can be brief (1-2 scenes)—brief is fine, but never skip entirely. Dense periods get more scenes.
- Chapter year_ranges should flow continuously with no gaps (e.g. Ch2: 1452-1469, Ch3: 1469-1482). Overlap at boundaries is fine; skipping years is not.

STORY ARC REQUIREMENTS:
1. Chapter 1 - THE HOOK: Always the hook chapter. Start by introducing the person with context. Present a rapid-fire "trailer" of their MOST interesting facts, achievements, controversies, and dramatic moments. Make viewers think "I NEED to know more." This is NOT chronological - it's a highlight reel. End with "But how did they get here? Let's start from the beginning..." Assign num_scenes (typically 4-5).
2. Chapter 2 - MUST START AT BIRTH: The first story chapter MUST cover birth and early years (childhood, upbringing, formative experiences). This creates a seamless transition from the hook. Can be brief (2-3 scenes) if little is known.
3. Chapters 3+: Use ANY chapter type. Continue chronologically. Dense periods or major works deserve more scenes. Sparse transitions get fewer scenes.
4. Final chapter: Legacy and emotional conclusion that echoes the hook.

CRITICAL:
- Every chapter must CONNECT to what came before and set up what comes after
- Include SPECIFIC details - dates, names, places, quotes, sensory details
- Focus on HUMAN drama - relationships, emotions, internal conflicts
- Make it feel like watching a movie, not reading a textbook
- Sum of num_scenes across all chapters must be approximately {target_total_scenes}

{{
  "person": "{person_of_interest}",
  "birth_year": YYYY,
  "death_year": YYYY or null,
  "tagline": "One compelling sentence that captures their story",
  "central_theme": "The overarching theme that ties the whole documentary together",
  "narrative_arc": "Brief description of the emotional journey from start to finish",
  "overarching_plots": [
    {{
      "plot_name": "The main plot thread",
      "description": "What this plot is about and why it matters",
      "starts_chapter": 1-{chapters},
      "peaks_chapter": 1-{chapters},
      "resolves_chapter": 1-{chapters},
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
      "chapter_type": "chronological",
      "title": "...",
      "year_range": "...",
      "num_scenes": 4,
      "summary": "...",
      "key_events": ["...", ...],
      "emotional_tone": "...",
      "music_mood": "one of {moods_str}",
      "dramatic_tension": "...",
      "connects_to_next": "...",
      "recurring_threads": ["...", ...],
      "plots_active": ["plot names or subplot names active in this chapter"],
      "plot_developments": ["How plots develop in this chapter"]
    }},
    ... ({chapters} chapters total, each with num_scenes and music_mood)
  ]
}}"""


def build_scene_outline_prompt(chapter: dict, person: str, scene_budget: int, landmarks: list[dict] | None = None,
                               research_context: "ResearchContext | None" = None) -> str:
    """
    Build the scene outline prompt for allocating scenes within a chapter.
    
    Args:
        chapter: Chapter dict with key_events, summary, chapter_type, etc.
        person: Name of the person
        scene_budget: Number of scenes to allocate (chapter's num_scenes)
        landmarks: Optional list of landmark events; when provided, LLM allocates 4-6 scenes to landmarks if this chapter covers them
    """
    chapter_type = chapter.get("chapter_type", "chronological")
    key_events = chapter.get("key_events", [])
    key_events_str = "\n".join(f"• {e}" for e in key_events) if key_events else "Use the chapter summary"

    landmark_section = ""
    if landmarks:
        lines = []
        for lm in landmarks:
            details = lm.get("key_details_to_cover", [])
            details_str = "; ".join(details[:3]) + ("..." if len(details) > 3 else "")
            lines.append(f"- \"{lm.get('event_name', '')}\" ({lm.get('year_or_period', '')}): {details_str}")
        landmark_list = "\n".join(lines)
        landmark_section = f"""
LANDMARK EVENTS (if this chapter covers one or more of these, allocate at least 4-6 scenes to that landmark):
{landmark_list}

Decide based on chapter title, summary, and key_events. Structure landmark blocks: setup, event with key details, reactions, significance.
"""
    from research_utils import get_research_context_block
    research_block = get_research_context_block(research_context) if research_context else ""
    return f"""Allocate {scene_budget} scenes for this chapter of a documentary about {person}.
{research_block}
CHAPTER: "{chapter.get('title', '')}"
Type: {chapter_type}
Summary: {chapter.get('summary', '')}

Key events to cover:
{key_events_str}
{landmark_section}
YOUR TASK: Break this chapter into "scene blocks" - each block is an event or sub-topic that gets 1 or more scenes.
- Pivotal moments (major works, breakthroughs, critical decisions) should get 2-4 scenes to cover: setup/context, the event itself, reactions/consequences, significance
- Minor events or transitions get 1 scene
- The sum of num_scenes across all blocks MUST equal {scene_budget} exactly

For event/work chapters with a single focus, you may have fewer blocks (e.g., one block of 6 scenes for "The Annus Mirabilis").
For chronological chapters, blocks typically map to key_events with variable allocation.

[
  {{"event": "Brief description of what this block covers", "num_scenes": N, "rationale": "Why this needs N scenes"}},
  ...
]

Sum of num_scenes must equal {scene_budget}."""


def build_metadata_prompt(person_of_interest: str, tagline: str, total_scenes: int,
                         video_questions: list[str] | None = None,
                         research_context: "ResearchContext | None" = None) -> str:
    """
    Build the initial metadata generation prompt.
    
    Args:
        person_of_interest: Name of the person
        tagline: One-line tagline
        total_scenes: Total number of scenes
        video_questions: Optional list of questions the documentary will answer (drives title and thumbnail)
    """
    video_questions_section = ""
    if video_questions:
        q_list = "\n".join(f"- {q}" for q in video_questions)
        video_questions_section = f"""
VIDEO QUESTIONS (the documentary will answer these):
{q_list}

CRITICAL: The title and thumbnail must BOTH be driven by these video questions. Either reflect all questions (if feasible) or anchor on the most important/compelling one. The thumbnail MUST visually reflect the questions the video answers—incorporate visual elements that hint at all questions, or focus on the most important one. The thumbnail should make viewers ask the SAME question(s) the video will answer.
"""
    from research_utils import get_research_context_block
    research_block = get_research_context_block(research_context) if research_context else ""
    return f"""Create initial metadata for a documentary about: {person_of_interest}

Their story in one line: {tagline}
{research_block}
{get_biopic_audience_profile()}
{video_questions_section}
TITLE AND THUMBNAIL - SAME WHY TYPE, MAXIMUM SYNERGY:
1. Choose ONE primary WHY type for this documentary (thumbnail_why_type). Only three options: counterintuitive, secret_mystery, known_but_misunderstood.
2. The TITLE must use that SAME WHY type—same paradigm, same curiosity. The thumbnail (description + thumbnail_text) must also use that SAME type. Title and thumbnail are one package; they must match and reinforce each other.
3. SYNERGY: Title and thumbnail should play off each other to maximize CTR. If the title asks a question, the thumbnail visual should hint at that question or its stakes. If the thumbnail shows a peak moment, the title should name the mystery or promise the revelation. Use complementary angles—not redundant, not conflicting. For this audience, prefer substantive power words (UNTOLD, REVEALED, LEGACY, SECRET) over flashy ones (INSANE, CRAZY).

THE THREE WHY TYPES (pick one; then align title + thumbnail to it):
- counterintuitive: Something that defies expectations. Title frames the twist or the unexpected; thumbnail suggests the same. E.g. "Why [Person] Did the UNTHINKABLE", "The Decision Nobody Saw Coming".
- secret_mystery: Hidden or little-known truth. Title hints at the secret; thumbnail hints at revelation or hidden thing. E.g. "The Secret [Person] Took to the Grave", "What [Person] Never Wanted You to Know".
- known_but_misunderstood: Familiar story most people get wrong. Title promises the real story; thumbnail can show the moment everyone misremembers. E.g. "What Everyone Gets WRONG About [Person]", "The Real [Person] (Not the Myth)".

THUMBNAIL RULES: ONE focal subject only. Clean solid dark background—no figures, letters, or half-visible elements. Subject lit; natural contrast (not extreme). RULE OF THIRDS. Bold text in LOWER third only (never at top). REPRESENT FULL VIDEO, ACCURATE.

Generate JSON (choose thumbnail_why_type FIRST; then write title and thumbnail_description + thumbnail_text to match that type):
{{
  "thumbnail_why_type": "Exactly one of: counterintuitive, secret_mystery, known_but_misunderstood. Choose the primary WHY paradigm that best fits this documentary. Title and thumbnail MUST both use this type.",
  "title": "WHY SCENE TITLE (60-80 chars) - MUST match thumbnail_why_type. Create one curiosity gap that synergizes with the thumbnail. counterintuitive: frame the twist/unexpected (e.g. Why [Person] Did the UNTHINKABLE, The Decision That Defied Everyone); secret_mystery: frame the hidden truth (e.g. The Secret [Person] Kept Hidden, What [Person] Never Told Anyone); known_but_misunderstood: frame the real story (e.g. What Everyone Gets WRONG About [Person], The Real Story of [Person]'s Greatest Moment). Use power words: UNTOLD, REVEALED, SECRET, HIDDEN, REAL, WRONG, UNTHINKABLE. Title and thumbnail together = one package for CTR.",
  "tag_line": "Short, succinct, catchy tagline (5-10 words) that captures who they are. Examples: 'the man who changed the world', 'the codebreaker who saved millions'. Memorable and accurate.",
  "thumbnail_description": "Describe the ONE PEAK MOMENT. MUST match thumbnail_why_type and synergize with the title—same curiosity, same WHY. ONE subject only. Clean solid dark background (no figures, no letters, no half-visible elements). Subject lit so they stand out; natural contrast, not extreme or weird colors. Rule of thirds. Bold text will be placed in lower third. REPRESENT FULL VIDEO, ACCURATE.",
  "thumbnail_text": "Short phrase (2-5 words) for thumbnail overlay. MUST match thumbnail_why_type: counterintuitive -> NOT WHAT YOU THINK, DEFIED EXPECTATIONS; secret_mystery -> THE SECRET, HIDDEN TRUTH; known_but_misunderstood -> EVERYONE GETS THIS WRONG, THE REAL STORY. Omit for default.",
  "global_block": "Visual style guide (300-400 words): semi-realistic digital painting style, color palette, dramatic lighting, how {person_of_interest} should appear consistently across {total_scenes} scenes."
}}"""


def build_metadata_prompt_3_options(person_of_interest: str, tagline: str, total_scenes: int,
                                   video_questions: list[str] | None = None,
                                   research_context: "ResearchContext | None" = None) -> str:
    """
    Build the initial metadata prompt that returns 3 title+thumbnail pairs.
    User picks the best pair when uploading.
    """
    video_questions_section = ""
    if video_questions:
        q_list = "\n".join(f"- {q}" for q in video_questions)
        video_questions_section = f"""
VIDEO QUESTIONS (the documentary will answer these):
{q_list}

CRITICAL: Each title and thumbnail pair must be driven by these video questions.
"""
    from research_utils import get_research_context_block
    research_block = get_research_context_block(research_context) if research_context else ""
    return f"""Create initial metadata for a documentary about: {person_of_interest}

Their story in one line: {tagline}
{research_block}
{get_biopic_audience_profile()}
{video_questions_section}
Generate EXACTLY 3 different title+thumbnail pairs. The user will pick the best when uploading. Each pair must be self-contained and maximally symbiotic.

FOR EACH PAIR - TITLE AND THUMBNAIL MUST BE SYMBIOTIC:
1. Choose ONE WHY type for that pair (counterintuitive, secret_mystery, or known_but_misunderstood). Use a different WHY type for at least 2 of the 3 options to give variety.
2. TITLE: Frame the curiosity using that WHY type. Power words: UNTOLD, REVEALED, SECRET, HIDDEN, REAL, WRONG, UNTHINKABLE.
3. THUMBNAIL_TEXT: Must DIRECTLY reinforce the title. This is the short phrase that appears on the thumbnail. If title says "The Secret X Kept", thumbnail_text = "THE SECRET". If title says "What Everyone Gets WRONG", thumbnail_text = "EVERYONE GETS THIS WRONG" or "THE REAL STORY". If title says "Why X Did the UNTHINKABLE", thumbnail_text = "NOT WHAT YOU THINK" or "DEFIED EXPECTATIONS". Title and thumbnail_text are ONE symbiotic pair—they must say the same thing in different words.
4. THUMBNAIL_DESCRIPTION: Describe the ONE peak moment. Same WHY type. Clean solid dark background. RULE OF THIRDS: Subject on intersection of thirds (left or right third), text in lower third band. REPRESENT FULL VIDEO, ACCURATE.

THE THREE WHY TYPES (use different ones across the 3 options):
- counterintuitive: Defies expectations. Title: "Why [Person] Did the UNTHINKABLE", "The Decision Nobody Saw Coming". thumbnail_text: "NOT WHAT YOU THINK", "DEFIED EXPECTATIONS".
- secret_mystery: Hidden truth. Title: "The Secret [Person] Took to the Grave", "What [Person] Never Wanted You to Know". thumbnail_text: "THE SECRET", "HIDDEN TRUTH".
- known_but_misunderstood: Familiar story most get wrong. Title: "What Everyone Gets WRONG About [Person]", "The Real [Person] (Not the Myth)". thumbnail_text: "EVERYONE GETS THIS WRONG", "THE REAL STORY".

RULE OF THIRDS (CRITICAL for thumbnail_description): Describe composition using the 9-part grid. Subject must be on a THIRD (left third line or right third line, or at an intersection). NOT centered. Text goes in the LOWER third band only. The composition must follow this grid.

Generate JSON with exactly 3 thumbnail_options (each has title, thumbnail_description, thumbnail_why_type, thumbnail_text):
{{
  "tag_line": "Short tagline (5-10 words) for this documentary.",
  "global_block": "Visual style guide (300-400 words): semi-realistic digital painting style, color palette, dramatic lighting, how {person_of_interest} should appear consistently across {total_scenes} scenes.",
  "thumbnail_options": [
    {{"title": "Option 1 title (60-80 chars)", "thumbnail_description": "Peak moment. Subject on left or right third. Text in lower third. Clean background.", "thumbnail_why_type": "counterintuitive|secret_mystery|known_but_misunderstood", "thumbnail_text": "2-5 words that DIRECTLY reinforce the title"}},
    {{"title": "Option 2 title (different angle)", "thumbnail_description": "...", "thumbnail_why_type": "...", "thumbnail_text": "..."}},
    {{"title": "Option 3 title (different angle)", "thumbnail_description": "...", "thumbnail_why_type": "...", "thumbnail_text": "..."}}
  ]
}}"""


def build_short_outline_prompt(person: str, outline: dict, short_num: int = 1, total_shorts: int = 3,
                              available_moods: list[str] | None = None, previously_used_topics: list[str] | None = None,
                              research_context: "ResearchContext | None" = None) -> str:
    """
    Build a shared prompt for generating short outlines (trailer style).
    LLM picks the best topic from the full documentary outline.
    
    Args:
        person: Name of the person
        outline: Full documentary outline with chapters, key_events, etc.
        short_num: Current short number
        total_shorts: Total number of shorts
        available_moods: List of music mood folder names for LLM to pick (e.g., ["relaxing", "passionate", "happy"])
        previously_used_topics: Titles or topics already used in other shorts—MUST pick a different topic
    """
    moods_str = ", ".join(available_moods) if available_moods else "relaxing, passionate, happy"
    no_repeat_section = ""
    if previously_used_topics:
        topics_list = "\n".join(f"- {t}" for t in previously_used_topics)
        no_repeat_section = f"""
CRITICAL - NO REPEAT TOPICS: The following topics have ALREADY been used in other shorts. You MUST pick a COMPLETELY DIFFERENT topic for this short. Do NOT cover the same story, event, or angle.
{topics_list}
Choose a different moment, work, or aspect of {person}'s life that is NOT listed above.
"""
    # Build full outline context from all chapters
    chapters = outline.get("chapters", [])
    outline_lines = []
    for ch in chapters:
        title = ch.get("title", "")
        year_range = ch.get("year_range", ch.get("time_setting", ""))
        summary = ch.get("summary", "")
        key_events = ch.get("key_events", [])
        events_str = "\n    ".join(f"• {e}" for e in key_events) if key_events else "  (use summary)"
        outline_lines.append(f"Chapter {ch.get('chapter_num')}: \"{title}\" ({year_range})\n  Summary: {summary}\n  Key events:\n    {events_str}")
    outline_context = "\n\n".join(outline_lines) if outline_lines else f"Full documentary about {person} - pick any compelling moment from their life."
    if not outline_lines:
        outline_context = f"Documentary about {person}. Pick whatever moment, work, or story would make the best short."
    from research_utils import get_research_context_block
    research_block = get_research_context_block(research_context) if research_context else ""
    return f"""Create a HIGH-ENERGY TRAILER for a YouTube Short about {person}.
{research_block}
Short #{short_num} of {total_shorts}
{no_repeat_section}

FULL DOCUMENTARY OUTLINE - Pick whatever you think will make the BEST short form video from these events:
{outline_context}

YOUR TASK: Choose ONE moment, work, or story from the outline above that would make a compelling 4-scene short. Expand it into a high-energy trailer: scenes 1-3 build the hook and end with a clear question; scene 4 answers that question (payoff). Do not instruct the short to end with a CTA to watch the full documentary in the narration—that hurts view duration.

{get_biopic_audience_profile()}

CRITICAL: This is a TRAILER, not a complete story. The short should:
- Be HIGH ENERGY and attention-grabbing
- Create CURIOSITY and make viewers NEED to watch the full documentary
- Expand the hook into 4 scenes: scenes 1-3 build the hook and end with a clear question; scene 4 answers that question (payoff)
- Scene 4 gives viewers a satisfying answer so they feel the short is complete. Do not add a spoken CTA to watch the full documentary at the end of scene 4—it hurts view duration and causes users to leave before the short finishes.
- Drive viewers to the main video via title/description only; never via end-of-narration CTA.

This Short has EXACTLY 4 scenes (scenes 1-3 WHY, scene 4 WHAT):
1. SCENE 1: Expand the hook - create high energy, grab attention immediately. Set up the mystery, problem, or question from the hook.
2. SCENE 2: Build the hook - escalate the curiosity, add more intrigue, deepen the mystery or stakes.
3. SCENE 3: Create anticipation - end with a compelling QUESTION that scene 4 will answer (e.g. "How did he do it?" "What happened next?" "Why did this work?").
4. SCENE 4: ANSWER the question from scene 3. This is a WHAT scene - deliver the payoff with clear facts. Do NOT end with a CTA to watch the full documentary; end with a satisfying resolution only.

CRITICAL RULES:
- HIGH ENERGY throughout - every scene should be attention-grabbing
- Create CURIOSITY GAPS - tease information but don't fully reveal
- Make viewers NEED to watch the main video to get answers
- Scene 4 is the only resolution - it answers the question from scene 3; the full story remains in the main video
- Simple, clear, punchy sentences
- Every scene must contain SPECIFIC, INTERESTING FACTS
- NO vague statements or filler
- Scenes 1-3 must be WHY scenes (trailer format); scene 4 must be a WHAT scene (payoff/answer)

TITLE AND THUMBNAIL - SAME WHY TYPE, SYNERGY: Choose ONE WHY type (thumbnail_why_type) for this short. short_title and thumbnail (thumbnail_prompt + thumbnail_text) MUST both use that SAME type and play off each other—one curiosity gap, complementary angles. short_title must also reflect the video_question.

THUMBNAIL RULES: ONE subject, two colors, subject lit brighter than background. Peak moment, rule of thirds, high contrast. Set thumbnail_why_type and thumbnail_text to match the title.

THE THREE WHY TYPES (pick one; align short_title + thumbnail): counterintuitive (twist/unexpected), secret_mystery (hidden truth), known_but_misunderstood (what everyone gets wrong).

Provide JSON (choose thumbnail_why_type FIRST; then short_title and thumbnail_prompt + thumbnail_text to match):
{{
  "thumbnail_why_type": "Exactly one of: counterintuitive, secret_mystery, known_but_misunderstood. Primary WHY type for this short. short_title and thumbnail MUST match this type.",
  "short_title": "WHY SCENE TITLE (max 50 chars) - MUST match thumbnail_why_type AND reflect video_question. Same curiosity as thumbnail. counterintuitive: frame the twist (e.g. How He Survived the IMPOSSIBLE); secret_mystery: frame the secret (e.g. The Secret That Saved His Life); known_but_misunderstood: frame the real story (e.g. What Everyone Gets WRONG About That Day). Title + thumbnail = one package for CTR.",
  "short_description": "YouTube description (100 words) with hashtags. Should drive viewers to watch the full documentary.",
  "tags": "10-15 SEO tags comma-separated",
  "music_mood": "Exactly one of: {moods_str}. Pick the music mood that best fits this high-energy trailer.",
  "thumbnail_prompt": "ONE subject, ONE peak moment. MUST match thumbnail_why_type and synergize with short_title. Clean solid dark background—no figures, no letters, no half-visible elements. Subject lit; natural contrast (not extreme). Rule of thirds. Bold text in lower third only. ACCURATE to short's story. Mobile-friendly.",
  "thumbnail_text": "Short phrase (2-5 words) for thumbnail. MUST match thumbnail_why_type: counterintuitive -> NOT WHAT YOU THINK; secret_mystery -> THE SECRET; known_but_misunderstood -> EVERYONE GETS THIS WRONG. Omit for default.",
  "video_question": "The question this short will answer - MUST be asked at the very start of scene 1. Must be specific to this short's chosen topic. Example: 'How did Roosevelt survive a point-blank shot to the chest?'",
  "hook_expansion": "How to expand the chosen topic into a 4-scene short - what story/mystery to tease, and what question scene 3 should pose for scene 4 to answer",
  "key_facts": ["3-5 specific facts to include across the 4 scenes: use in scenes 1-3 for curiosity, and in scene 4 for the payoff answer"]
}}"""


def get_thumbnail_audience_targeting() -> str:
    """
    Brief audience-targeting suffix for thumbnail image generation.
    Targets 65-year-old American men - authoritative, substantive, PBS-style.
    """
    return """TARGET AUDIENCE (65-year-old American men): Authoritative, substantive visual style - like PBS documentary or Ken Burns. Avoid flashy, TikTok-style, or Gen Z imagery. Cinematic, documentary-quality composition. Prefer depth and gravitas over sensationalism."""


def get_thumbnail_prompt_why_scene(aspect_ratio: str = "16:9") -> str:
    """
    Get the shared WHY scene thumbnail prompt to maximize CTR.
    Used for both main video and shorts thumbnails.
    
    Args:
        aspect_ratio: Image aspect ratio ("16:9" for main video, "9:16" for shorts)
    """
    size_note = "optimized for small sizes - subject and key mystery elements must be CLEARLY visible even when tiny" if aspect_ratio == "16:9" else "optimized for mobile scrolling - must instantly create curiosity when tiny in feed"
    vertical_note = "Vertical 9:16," if aspect_ratio == "9:16" else ""
    
    return f"""WHY SCENE THUMBNAIL - MAXIMIZE CTR:

CRITICAL - TEXT PLACEMENT: The thumbnail MUST include bold text (exact phrase given in instruction above—counterintuitive, secret/mystery, or known but misunderstood). Place the text in the LOWER third or bottom of the image only—NEVER at the top or near the top edge. Keep text fully INSIDE the frame with clear margin from all edges so it is never cut off or clipped. Text must be readable at small sizes.

MAXIMUM SIMPLICITY - ONE FOCUS:
- ONE focal subject only. No extra figures, objects, or background detail. If the scene needs one prop or symbol, that is the absolute maximum—otherwise subject only.
- NO clutter, NO busy backgrounds. Think: one person, one moment, one idea.

CLEAN BACKGROUND (CRITICAL): The background must be CLEAN and simple. Use only: solid dark (black or dark gray), or a very soft simple gradient. NO figures, NO faces, NO characters, NO letters, NO symbols, and NO half-visible or partially visible elements in the background. Nothing creeping in from the edges. The background should be empty of any detail—no ghostly shapes, no faint text, no extra people.

COLOR - NATURAL, NOT EXTREME:
- Use a limited palette: one accent (e.g. gold, warm red, or soft warm light) and a dark background. Keep contrast strong enough that the subject is clearly separate from the background, but avoid EXTREME contrast that makes colors look weird, oversaturated, or harsh. Natural skin tones where visible. No harsh color grading or unnatural color casts.

SUBJECT AND BACKGROUND: Light the subject so they stand out from the background, but keep the look natural. Background stays dark and empty. Do not let the subject blend in; do not use such extreme contrast that colors look strange.

RULE OF THIRDS (MANDATORY): Use the 9-part grid. (1) Place the main subject on a THIRD—either the left third line or the right third line, or at one of the four intersection points. Do NOT center the subject. (2) Place the text in the LOWER third band only, fully inside the frame with margin. The composition must follow this grid exactly.

EMOTION: Show the PEAK CONFLICT, EXCITEMENT, or MOST ENGAGING MOMENT from the video. One moment only. Represent the WHOLE VIDEO accurately—no misleading or generic imagery.

VISUAL COMPOSITION:
- {vertical_note} Strong composition, rule of thirds
- Intense emotional expression—conflict, shock, determination, or revelation
- Cinematic, documentary-quality—PBS/Ken Burns gravitas
- {size_note}"""


def get_horror_narration_style() -> str:
    """Get first-person narration style for horror stories."""
    return """NARRATION STYLE - FIRST PERSON HORROR (CRITICAL):
- PERSPECTIVE: Write in FIRST PERSON - the protagonist is telling their own story (I/me/my)
- CRITICAL: Use first person throughout - "I walk into the room...", "My heart pounds as I...", "I realize that..."
- Present tense for immediacy and immersion: "I hear a sound...", "I turn around...", "I see something..."
- IMMERSIVE and PERSONAL - make the viewer feel like they ARE the protagonist experiencing the horror
- Horror tone: suspenseful, atmospheric, tense, fearful
- Use sensory details: what I see, hear, feel, smell - make it visceral and immediate
- Internal thoughts and reactions: "I think...", "I wonder...", "I'm terrified because..."
- Physical sensations: "My hands shake...", "My heart races...", "I feel cold..."
- Build tension through pacing: short, sharp sentences for scares; longer, atmospheric sentences for buildup
- Create atmosphere through description: shadows, sounds, feelings of being watched
- NO third person - never say "he/she/they" when referring to the protagonist
- NO meta references - don't mention chapters, videos, or production elements
- 2-3 sentences per scene (~12-18 seconds of narration)
- Pack maximum atmosphere and tension into minimum words
- CRITICAL: This is SPOKEN narration for text-to-speech. Do NOT include film directions or camera directions.
- Write ONLY words that should be spoken by the protagonist's voice.
- QUOTES (CRITICAL FOR TTS): Use quotes ONLY around proper nouns you want to emphasize (e.g. titles of works, key terms). Otherwise do NOT use any quotes—they will mess up the text-to-speech."""


def build_horror_outline_prompt(story_concept: str, chapters: int, total_scenes: int) -> str:
    """
    Build the horror story outline generation prompt.
    
    Args:
        story_concept: The story concept/prompt (e.g., "A haunted house story")
        chapters: Number of chapters (should be 3 for horror)
        total_scenes: Total number of scenes
    """
    return f"""You are a master horror storyteller. Create a compelling narrative outline for a ~10 minute horror story about: {story_concept}

This will be a {total_scenes}-scene horror story with EXACTLY {chapters} chapters. Think of this as a SHORT HORROR FILM with building tension, scares, and an open ending.

HORROR NARRATIVE STRUCTURE:
- The story should feel like ONE CONTINUOUS HORROR EXPERIENCE, not disconnected scenes
- Each chapter should ESCALATE tension from the previous one
- Build ATMOSPHERE and FEAR throughout
- Create MYSTERY and UNEASE that keeps viewers scared
- End with an OPEN ENDING that leaves viewers unsettled and scared

For each of the {chapters} chapters, provide:
- "chapter_num": 1-{chapters}
- "title": A compelling, scary chapter title
- "time_setting": When/where this takes place (e.g., "Late night in an abandoned house", "Present day, suburban neighborhood")
- "summary": 2-3 sentences about what happens (from first-person perspective)
- "key_events": 4-6 specific scary moments or tension-building events to show
- "emotional_tone": The horror mood of this chapter (e.g., "tense", "terrifying", "atmospheric", "dread-filled")
- "dramatic_tension": What horror/threat drives this chapter
- "connects_to_next": How this chapter escalates into the next
- "horror_elements": Specific horror elements (e.g., "unexplained sounds", "shadowy figures", "growing paranoia")

HORROR STORY ARC REQUIREMENTS:
1. Chapter 1 - SETUP & TENSION BUILDING:
   - Introduce protagonist in first person (I/me/my)
   - Establish normal world before horror begins
   - Introduce mystery/threat/unease
   - Build initial tension and atmosphere
   - Mostly WHY scenes (mystery, questions, unease)
   - End with something unsettling that makes viewers want to continue

2. Chapter 2 - ESCALATION & SCARES:
   - Tension escalates significantly
   - Scare moments and reveals
   - Threat becomes clearer (but not fully explained)
   - Mix of WHY (mystery) and WHAT (scares/reveals)
   - Build to a major scare or revelation
   - Increase fear and paranoia

3. Chapter 3 - CLIMAX & OPEN ENDING:
   - Final confrontation/climax with the horror
   - Open ending (unresolved, keeps viewer scared)
   - Lingering questions and unease
   - Mostly WHY scenes (unresolved tension)
   - End with something that makes viewers still feel scared/unsettled
   - DO NOT fully resolve - leave mystery and fear lingering

CRITICAL HORROR REQUIREMENTS:
- First person perspective throughout (I/me/my)
- Build TENSION progressively - each chapter should be scarier than the last
- Create ATMOSPHERE through description: shadows, sounds, feelings, unease
- Use WHY/WHAT paradigm at scene level:
  * WHY scenes: Frame mysteries, questions, unease, "what's happening?" moments
  * WHAT scenes: Reveal scares, show threats, deliver horror moments
- Focus on FEAR and ATMOSPHERE, not just jump scares
- Open ending is CRITICAL - don't fully explain or resolve everything
- Make viewers feel UNSETTLED and SCARED even after watching

CRITICAL: ENVIRONMENT SELECTION - You must select ONE environment for the entire story. This will determine the background ambient sound. Choose from: "blizzard", "snow", "forest", "rain", "indoors", or "jungle". Base your choice on the story's setting and atmosphere. The environment should match where the horror takes place.

{{
  "story_concept": "{story_concept}",
  "protagonist_name": "Name of the protagonist (optional, can be generic like 'the narrator')",
  "setting": "Where and when the story takes place",
  "environment": "ONE of: 'blizzard', 'snow', 'forest', 'rain', 'indoors', or 'jungle' - the ambient environment for the entire story",
  "central_horror": "The main horror element/threat (e.g., 'haunted house', 'unexplained entity', 'paranormal activity')",
  "narrative_arc": "Brief description of the horror journey from normal to terrified",
  "overarching_plots": [
    {{
      "plot_name": "The main horror thread (e.g., 'The Haunting', 'The Growing Threat', 'The Unseen Presence')",
      "description": "What this horror plot is about",
      "starts_chapter": 1-{chapters},
      "peaks_chapter": 1-{chapters},
      "resolves_chapter": null or {chapters} (but should remain open/unresolved),
      "key_moments": ["specific horror moments that develop this story"]
    }}
  ],
  "sub_plots": [
    {{
      "subplot_name": "A sub-plot that spans chapters (e.g., 'The Discovery', 'The Escalation', 'The Realization')",
      "description": "What this sub-plot is about",
      "chapters_span": [1-3],
      "key_moments": ["specific moments that advance this sub-plot"]
    }}
  ],
  "chapters": [
    {{
      "chapter_num": 1,
      "title": "...",
      "time_setting": "...",
      "summary": "...",
      "key_events": ["...", ...],
      "emotional_tone": "...",
      "dramatic_tension": "...",
      "connects_to_next": "...",
      "horror_elements": ["...", ...],
      "plots_active": ["plot names or subplot names that are active/developing in this chapter"],
      "plot_developments": ["How horror plots develop in this chapter - what happens to them"]
    }},
    ... ({chapters} chapters total)
  ]
}}"""


def build_horror_metadata_prompt(story_concept: str, tagline: str, total_scenes: int) -> str:
    """
    Build the horror story metadata generation prompt.
    
    Args:
        story_concept: The story concept
        tagline: One-line tagline
        total_scenes: Total number of scenes
    """
    return f"""Create initial metadata for a horror story video about: {story_concept}

Story tagline: {tagline}

CRITICAL: TITLE AND THUMBNAIL MUST BE COHESIVE AND SYNCED - They must create the SAME curiosity gap and work together to maximize CTR. The thumbnail should visually represent the same horror/mystery/threat that the title frames.

Generate JSON:
{{
  "title": "MYSTERIOUS HORROR TITLE - MAXIMIZE CTR (30-50 chars): Inspired by famous horror book titles like 'The Shining', 'It', 'The Exorcist', 'Pet Sematary', 'The Thing'. Keep it SHORT, MYSTERIOUS, and AMBIGUOUS. Use simple, evocative words that create curiosity through ambiguity, not explicit questions. Avoid verbose explanations or parentheticals. The title should hint at horror without revealing it. Examples: 'The Mimic', 'Blizzard Whistle', 'Something in the Snow', 'The Echo', 'The Shadow', 'The Voice', 'The Presence', 'The Thing in the Basement', 'The Watcher', 'The Copy', 'The Reflection'. Use power words SPARINGLY - only when they add mystery (e.g., 'The Cursed', 'The Haunted', 'The Forbidden'). The title should make viewers think: 'What is this about?' through mystery, not explicit questions. Must create a curiosity gap that makes viewers NEED to click to discover the horror, while staying appropriate.",
  "tag_line": "Short, scary tagline (5-10 words) that captures the horror. Examples: 'a story that will haunt you', 'the night that changed everything', 'when the horror found me', 'the thing that still watches'. Should be memorable and scary.",
  "thumbnail_description": "WHY SCENE THUMBNAIL - MAXIMIZE CTR THROUGH HORROR (MUST BE COHESIVE WITH TITLE): The thumbnail must function as a WHY scene that creates the SAME curiosity gap and fear as the title. Visually frame the SAME MYSTERY, THREAT, or HORROR ELEMENT that the title frames. If the title asks 'What horror is this?', the thumbnail should visually hint at that horror. If the title mentions 'I Found Something EVIL', the thumbnail should show the moment of discovery or the evil element. If the title mentions 'The CURSED Object', the thumbnail should show visual elements suggesting the cursed object or its effects. Show scary, unsettling, or creepy elements that match the title's horror theme. Use visual storytelling to ask the SAME questions the title asks: 'What horror is this?', 'What threat is here?', 'What will happen?', 'What is watching?', 'What did they find?'. Composition: intense close-ups, fearful expressions, dramatic shadows, eerie lighting, horror atmosphere, symbolic elements suggesting danger or horror matching the title's theme, visual curiosity gaps that make viewers ask the SAME questions the title asks. Subject in MOMENT OF FEAR, DISCOVERY, or CONFRONTATION with horror - not passive. Show visual hints of the horror/threat without fully revealing it. Think: 'What question does this image make me ask?' It should be the SAME question the title asks. The viewer should look at this and think 'I NEED to know what this horror is about' - create visual fear and curiosity gaps that sync with the title. NO TEXT in image, but visually SCREAM horror, fear, and the promise of scares that matches the title's promise.",
  "global_block": "Visual style guide (300-400 words): horror atmosphere, dark and shadowy, eerie lighting, color palette (blues, grays, deep shadows), how the horror should appear consistently across {total_scenes} scenes. Focus on mood, atmosphere, and fear rather than explicit gore."
}}"""
