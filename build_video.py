import os
import sys
import json
import base64
import time
import random
from pathlib import Path

from openai import OpenAI
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment, silence

# ------------- CONFIG -------------

# Make sure OPENAI_API_KEY is set in your environment before running:
#   set OPENAI_API_KEY=sk-...
client = OpenAI()

# OpenAI models
IMG_MODEL = "gpt-image-1-mini"        # or "gpt-image-1-mini"
TTS_MODEL = "gpt-4o-mini-tts"    # or another TTS model you have access to

# Final video resolution (we’ll crop/scale images to this)
OUTPUT_RESOLUTION = (1920, 1080)
FPS = 30

IMAGES_DIR = Path("generated_images")
AUDIO_DIR = Path("generated_audio")

MAX_WORKERS = 5  # 2–4 is usually safe; higher risks hitting rate limits

MAX_PREV_SUMMARY_CHARS = 300  # keep it short so prompts don't blow up

TTS_SCENE_STYPE_PROMPT = (
    "Read the following text in a calm, neutral, and consistent tone. "
    "Maintain stable volume and pitch throughout. Do not add emotional "
    "inflection, dramatic emphasis, or noticeable changes in speaking speed. "
)
TTS_FULL_AUDIO_STYLE_PROMPT = (
    "Read the following text in a calm, neutral, and consistent tone. "
    "Maintain stable volume and pitch throughout. Do not add emotional "
    "inflection, dramatic emphasis, or noticeable changes in speaking speed. "
    "Take a brief pause when there are gaps in the text."
)
TTS_VOICE = "cedar"


def build_image_prompt(scene: dict, prev_scene: dict | None) -> str:
    """
    Build a rich image prompt for the current scene, including:
    - Current scene description (title + narration + image_prompt)
    - Brief memory of the previous scene for continuity
    - Global visual style and constraints (no text, documentary tone)
    """

    title = scene.get("title", "").strip()
    narration = scene.get("narration", "").strip()
    scene_img_prompt = scene.get("image_prompt", "").strip()

    # --- Previous scene memory ---
    if prev_scene is None:
        prev_block = (
            "This is the opening scene of a long-form documentary. "
            "Establish the core visual style that will stay consistent across the entire video."
        )
    else:
        prev_title = prev_scene.get("title", "").strip()
        prev_narr = prev_scene.get("narration", "").strip()
        prev_img_prompt = prev_scene.get("image_prompt", "").strip()

        # Build a short textual summary of the previous scene
        prev_text = " ".join(
            x for x in [prev_title, prev_narr, prev_img_prompt] if x
        ).strip()
        if len(prev_text) > MAX_PREV_SUMMARY_CHARS:
            prev_text = prev_text[:MAX_PREV_SUMMARY_CHARS].rsplit(" ", 1)[0] + "..."

        prev_block = (
            "Maintain visual continuity with the previous scene, which showed: "
            f"{prev_text} "
            "Use similar color grading, lighting mood, and overall artistic treatment."
        )

    # --- Global documentary style block ---
    global_block = (
        "This image is part of a single, cohesive cinematic documentary about Nikola Tesla "
        "for a YouTube channel called Human Footprints. Every scene must look like a frame "
        "from the same film.\n\n"
        "Use one consistent visual style across ALL scenes: a semi-realistic digital painting "
        "that resembles a high-end historical documentary still. Avoid cartoony, anime, comic, "
        "3D render, or flat graphic styles.\n\n"
        "Color palette should remain unified throughout the video: deep navy blues, desaturated teals, "
        "muted warm golds and ambers for light sources, and soft sepia shadows. Highlights should feel "
        "like warm lamplight or early electrical glow contrasting against cool, dark backgrounds.\n\n"
        "Lighting should be dramatic, almost chiaroscuro: strong key light on the subject, soft falloff "
        "into darkness, and subtle atmospheric haze in the background. Overall mood: serious, thoughtful, "
        "slightly melancholic but with a sense of awe and intellect.\n\n"
        "Framing and composition should also be consistent: cinematic 16:9 framing, mostly medium shots "
        "and medium-wide shots, with simple, uncluttered backgrounds. Use a shallow depth-of-field feel "
        "so the main subject is clearly separated from the background. Camera angles should be mostly eye-level "
        "or slightly above, avoiding extreme wide-angle distortion or exaggerated perspectives.\n\n"
        "Whenever Nikola Tesla is shown, keep his appearance consistent across scenes: tall, thin face, sharp features, "
        "dark neatly combed hair, distinct mustache, and dark formal suit appropriate to the late 19th and early 20th century. "
        "Do not change his ethnicity, facial structure, or general look between scenes. Adjust only age and clothing "
        "when appropriate to the period of his life."
    )

    # --- No text / cleanliness constraints ---
    constraints_block = (
        "Do NOT include any text, titles, captions, subtitles, numbers, letters, labels, watermarks, logos, "
        "user interface elements, or graphic overlays in the image. The frame must be completely text-free.\n\n"
        "Keep the visual style, color palette, and overall mood strictly consistent with the rest of the video. "
        "This should look like one continuous documentary, not a mix of different art styles."
    )

    # --- Current scene description block ---
    current_block_parts = []
    if title:
        current_block_parts.append(f"Scene title: {title}.")
    if narration:
        current_block_parts.append(
            "Narration for this scene:\n" + narration
        )
    if scene_img_prompt:
        current_block_parts.append(
            "Visual details to emphasize:\n" + scene_img_prompt
        )
    current_block = "\n".join(current_block_parts)

    # Final prompt
    prompt = (
        current_block
        + "\n\n"
        + prev_block
        + "\n\n"
        + global_block
        + "\n\n"
        + constraints_block
    )

    return prompt



def load_scenes(path: str):
    """
    Load scenes from a JSON file.

    Expected structure: list of dicts, each like:
    {
      "id": 1,
      "title": "Opening",
      "narration": "Text to be spoken",
      "image_prompt": "Prompt for image (optional)"
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_image_for_scene(scene: dict, prev_scene: dict | None) -> Path:
    """
    Generate an image for the given scene using OpenAI Images API.
    Caches result on disk; if file already exists, it is reused.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMAGES_DIR / f"scene_{scene['id']:02d}.png"
    if img_path.exists():
        print(f"[IMAGE] Scene {scene['id']}: using cached {img_path.name}")
        return img_path

    prompt = build_image_prompt(scene, prev_scene)
    print(f"[IMAGE] Scene {scene['id']}: generating image...")

    # NOTE: size must be one of: "1024x1024", "1024x1536", "1536x1024", or "auto"
    resp = client.images.generate(
        model=IMG_MODEL,
        prompt=prompt,
        size="1536x1024",  # landscape, good for video
        n=1
    )

    b64_data = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64_data)
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    print(f"[IMAGE] Scene {scene['id']}: saved {img_path.name}")
    return img_path


def generate_audio_for_scene(scene: dict) -> Path:
    """
    Generate TTS audio for the given scene's narration.
    Caches result on disk; if file already exists, it is reused.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = AUDIO_DIR / f"scene_{scene['id']:02d}.mp3"
    if audio_path.exists():
        print(f"[AUDIO] Scene {scene['id']}: using cached {audio_path.name}")
        return audio_path

    text = scene["narration"]
    print(f"[AUDIO] Scene {scene['id']}: generating audio...")

    # Correct TTS streaming call for the new OpenAI API
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions=TTS_SCENE_STYPE_PROMPT,
    ) as response:
        response.stream_to_file(str(audio_path))

    print(f"[AUDIO] Scene {scene['id']}: saved {audio_path.name}")
    return audio_path


def generate_full_audio_for_all_scenes(scenes: list[dict]) -> Path:
    """
    Generate a single 'one take' TTS audio file for all scenes combined.
    The text is just all narrations concatenated with paragraph breaks.
    Result is cached on disk.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    full_audio_path = AUDIO_DIR / "full_narration_all_scenes.mp3"

    if full_audio_path.exists():
        print(f"[AUDIO] Full narration: using cached {full_audio_path.name}")
        return full_audio_path

    parts: list[str] = []
    for scene in scenes:
        narr = (scene.get("narration") or "").strip()
        if narr:
            parts.append(narr)

    if not parts:
        raise ValueError("No narration text found in scenes for full audio.")

    full_script = "\n\n\n".join(parts)

    print("[AUDIO] Full narration: generating one-take audio...")

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=full_script,
        instructions=TTS_FULL_AUDIO_STYLE_PROMPT,
    ) as response:
        response.stream_to_file(str(full_audio_path))

    print(f"[AUDIO] Full narration: saved {full_audio_path.name}")
    return full_audio_path



def make_static_clip(image_path: Path, audio_path: Path):
    """
    Make a static full-frame clip for one scene:
    - Scale image to cover 1920x1080.
    - Center & crop.
    - Attach the scene's audio and match duration.
    """
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration

    clip = ImageClip(str(image_path)).with_duration(duration)
    base_w, base_h = clip.size

    # Scale so image fully covers the frame
    scale = max(
        OUTPUT_RESOLUTION[0] / base_w,
        OUTPUT_RESOLUTION[1] / base_h,
    )
    clip = clip.resized(scale)
    w, h = clip.size

    x_center, y_center = w / 2, h / 2

    clip = clip.cropped(
        x_center=x_center,
        y_center=y_center,
        width=OUTPUT_RESOLUTION[0],
        height=OUTPUT_RESOLUTION[1],
    )

    return clip.with_audio(audio).with_duration(duration)


def make_static_clip_with_audio_clip(image_path: Path, audio_clip: AudioFileClip):
    """
    Make a static full-frame clip for one scene, using an existing AudioFileClip.
    - Scale image to cover 1920x1080.
    - Center & crop.
    - Attach the provided audio and match duration.
    """
    duration = audio_clip.duration

    clip = ImageClip(str(image_path)).with_duration(duration)
    base_w, base_h = clip.size

    # Scale so image fully covers the frame
    scale = max(
        OUTPUT_RESOLUTION[0] / base_w,
        OUTPUT_RESOLUTION[1] / base_h,
    )
    clip = clip.resized(scale)
    w, h = clip.size

    x_center, y_center = w / 2, h / 2

    clip = clip.cropped(
        x_center=x_center,
        y_center=y_center,
        width=OUTPUT_RESOLUTION[0],
        height=OUTPUT_RESOLUTION[1],
    )

    return clip.with_audio(audio_clip).with_duration(duration)



def retry_call(name: str, func, *args, max_attempts: int = 3, base_delay: float = 2.0, **kwargs):
    """
    Generic retry wrapper for network/API calls.
    Retries on any Exception up to max_attempts, with exponential backoff.

    name: a label for logging (e.g. "image", "audio").
    func: the function to call.
    *args, **kwargs: passed to func.
    """
    attempt = 1
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt >= max_attempts:
                print(f"[RETRY][{name}] Failed after {attempt} attempts: {e}")
                raise

            delay = base_delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
            print(f"[RETRY][{name}] Attempt {attempt} failed: {e}. "
                  f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            attempt += 1


def generate_image_for_scene_with_retry(scene: dict, prev_scene: dict | None) -> Path:
    return retry_call(
        "image",
        generate_image_for_scene,
        scene,
        prev_scene,
        max_attempts=3,
        base_delay=2.0,
    )


def generate_audio_for_scene_with_retry(scene: dict) -> Path:
    return retry_call(
        "audio",
        generate_audio_for_scene,
        scene,
        max_attempts=3,
        base_delay=2.0,
    )


def generate_full_audio_for_scenes_with_retry(scenes: list[dict]) -> Path:
    return retry_call(
        "full_audio",
        generate_full_audio_for_all_scenes,
        scenes,
        max_attempts=3,
        base_delay=2.0,
    )


# def build_video(scenes_path: str, out_video_path: str):
#     scenes = load_scenes(scenes_path)
#     num_scenes = len(scenes)
#
#     # Previous-scene references for image continuity
#     prev_scenes: list[dict | None] = [None] + scenes[:-1]
#
#     image_paths: list[Path | None] = [None] * num_scenes
#     audio_paths: list[Path | None] = [None] * num_scenes
#
#     print(f"[SCENES] Starting parallel generation for {num_scenes} scenes...")
#
#     def scene_job(idx: int):
#         scene = scenes[idx]
#         prev_scene = prev_scenes[idx]
#
#         print(f"\n[SCENE {idx}] Generating assets for scene {scene['id']}: {scene.get('title', '')}")
#
#         img_path = generate_image_for_scene_with_retry(scene, prev_scene)
#         audio_path = generate_audio_for_scene_with_retry(scene)
#
#         print(f"[SCENE {idx}] Done -> image={img_path}, audio={audio_path}")
#         return idx, img_path, audio_path
#
#     # --- PARALLEL PHASE: image + audio per scene ---
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = [executor.submit(scene_job, i) for i in range(num_scenes)]
#         for fut in as_completed(futures):
#             idx, img_path, audio_path = fut.result()
#             image_paths[idx] = img_path
#             audio_paths[idx] = audio_path
#
#     # Safety checks
#     missing_imgs = [i for i, p in enumerate(image_paths) if p is None]
#     missing_auds = [i for i, p in enumerate(audio_paths) if p is None]
#     if missing_imgs:
#         raise RuntimeError(f"Missing image paths for indices: {missing_imgs}")
#     if missing_auds:
#         raise RuntimeError(f"Missing audio paths for indices: {missing_auds}")
#
#     print("\n[VIDEO] Assembling clips sequentially...")
#
#     clips = []
#     for i, scene in enumerate(scenes):
#         print(f"[CLIP {i}] Building clip for scene {scene['id']}: {scene.get('title', '')}")
#
#         img_path = image_paths[i]
#         audio_path = audio_paths[i]
#
#         clip = make_static_clip(img_path, audio_path)
#         clips.append(clip)
#
#     print("\n[VIDEO] Concatenating clips...")
#     final = concatenate_videoclips(clips, method="compose")
#
#     # # --- FULL-TAKE AUDIO: generate once for all scenes ---
#     # full_audio_path = generate_full_audio_for_all_scenes(scenes)
#     # full_audio = AudioFileClip(str(full_audio_path))
#     #
#     # video_duration = final.duration
#     # audio_duration = full_audio.duration
#     #
#     # if video_duration <= 0 or audio_duration <= 0:
#     #     raise RuntimeError(
#     #         f"Non-positive duration(s): video={video_duration}, audio={audio_duration}"
#     #     )
#     #
#     # # We want: new_video_duration == audio_duration
#     # # MultiplySpeed(factor) makes duration ~= old_duration / factor
#     # # So choose factor so that: video_duration / factor = audio_duration
#     # factor = video_duration / audio_duration
#     #
#     # print(
#     #     f"[SYNC] Video duration={video_duration:.3f}s, "
#     #     f"audio duration={audio_duration:.3f}s. "
#     #     f"Adjusting video speed by factor={factor:.6f} to match audio."
#     # )
#     #
#     # # Time-stretch the *video* (static images) to match the full audio length.
#     # # Audio is left untouched.
#     # speed_effect = MultiplySpeed(factor=factor)
#     # video_adjusted = final.with_effects([speed_effect])
#     #
#     # # At this point, video_adjusted.duration should closely match audio_duration.
#     # # No clipping needed; just attach the full-take audio.
#     # final_with_full_audio = video_adjusted.with_audio(full_audio)
#
#     print(f"[VIDEO] Writing final video to {out_video_path}...")
#     final.write_videofile(
#         out_video_path,
#         fps=FPS,
#         codec="libx264",
#         audio_codec="aac",
#         preset="medium",
#         threads=4,
#     )
#     print("[VIDEO] Done.")


def split_full_audio_by_silence(full_audio_path: Path, num_scenes: int):
    audio = AudioSegment.from_file(full_audio_path)

    # Tweak these thresholds as needed
    min_silence_len = 400  # ms
    silence_thresh = audio.dBFS - 16

    silences = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    # silences is a list of [start_ms, end_ms] intervals

    # Choose the N-1 longest silences as scene boundaries
    silences_sorted = sorted(silences, key=lambda x: x[1] - x[0], reverse=True)
    boundaries = sorted(silences_sorted[: num_scenes - 1], key=lambda x: x[0])

    # Build start/end times in seconds
    starts = [0.0]
    for s, e in boundaries:
        starts.append(e / 1000.0)   # end of silence = start of next scene
    durations = []
    for i, st in enumerate(starts):
        if i < len(starts) - 1:
            durations.append(starts[i+1] - st)
        else:
            durations.append(len(audio) / 1000.0 - st)

    return starts, durations


def build_video(scenes_path: str, out_video_path: str):
    scenes = load_scenes(scenes_path)
    num_scenes = len(scenes)

    # Previous-scene references for image continuity
    prev_scenes: list[dict | None] = [None] + scenes[:-1]

    image_paths: list[Path | None] = [None] * num_scenes

    print(f"[SCENES] Starting image generation for {num_scenes} scenes...")

    def image_job(idx: int):
        scene = scenes[idx]
        prev_scene = prev_scenes[idx]
        print(f"\n[SCENE {idx}] Generating image for scene {scene['id']}: {scene.get('title', '')}")
        img_path = generate_image_for_scene_with_retry(scene, prev_scene)
        print(f"[SCENE {idx}] Done -> image={img_path}")
        return idx, img_path

    # --- PARALLEL PHASE: images only ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(image_job, i) for i in range(num_scenes)]
        for fut in as_completed(futures):
            idx, img_path = fut.result()
            image_paths[idx] = img_path

    missing_imgs = [i for i, p in enumerate(image_paths) if p is None]
    if missing_imgs:
        raise RuntimeError(f"Missing image paths for indices: {missing_imgs}")

    # --- FULL-TAKE AUDIO ---
    full_audio_path = generate_full_audio_for_scenes_with_retry(scenes)
    full_audio = AudioFileClip(str(full_audio_path))
    total_audio_duration = full_audio.duration

    print(f"\n[AUDIO-FULL] Duration of full narration: {total_audio_duration:.3f}s")

    # --- Derive scene timing from silences in the full audio ---
    print("[AUDIO-FULL] Detecting scene boundaries using silence...")
    cumulative_starts, scene_durations = split_full_audio_by_silence(full_audio_path, num_scenes)

    if len(cumulative_starts) != num_scenes or len(scene_durations) != num_scenes:
        raise RuntimeError(
            f"Silence-based split returned {len(cumulative_starts)} starts / "
            f"{len(scene_durations)} durations for {num_scenes} scenes."
        )

    print("\n[AUDIO-FULL] Scene timing based on detected silences:")
    for i, (start, dur) in enumerate(zip(cumulative_starts, scene_durations)):
        print(
            f"  Scene {scenes[i]['id']}: start={start:.3f}s, "
            f"duration={dur:.3f}s, end={start+dur:.3f}s"
        )

    # --- Build clips using audio segments from the one-take audio ---
    print("\n[VIDEO] Assembling clips with audio segments from full narration...")
    clips = []
    for i, scene in enumerate(scenes):
        img_path = image_paths[i]
        start_t = cumulative_starts[i]
        dur = scene_durations[i]

        if dur <= 0:
            # No audible duration detected for this scene; give it a small placeholder
            dur = 0.5

        end_t = start_t + dur

        # MoviePy 2.2.1: use subclipped(start, end)
        scene_audio = full_audio.subclipped(start_t, end_t)

        print(
            f"[CLIP {i}] Scene {scene['id']}: "
            f"using audio [{start_t:.3f}, {end_t:.3f}] ({dur:.3f}s)"
        )

        clip = make_static_clip_with_audio_clip(img_path, scene_audio)
        clips.append(clip)

    print("\n[VIDEO] Concatenating clips...")
    final = concatenate_videoclips(clips, method="compose")

    print(
        f"[VIDEO] Final video duration={final.duration:.3f}s, "
        f"full audio duration={total_audio_duration:.3f}s"
    )

    print(f"[VIDEO] Writing final video to {out_video_path}...")
    final.write_videofile(
        out_video_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
    )
    print("[VIDEO] Done.")




# ------------- ENTRY POINT -------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python build_video.py scenes.json output_video.mp4")
        sys.exit(1)

    scenes_file = sys.argv[1]
    out_file = sys.argv[2]

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set the OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    build_video(scenes_file, out_file)
