"""
Kids Shorts Video Generator

Generates animated videos for each scene using OpenAI Sora API,
then concatenates them into a final YouTube Short.

NOTE: Sora API may not be publicly available yet. This code includes
multiple fallback attempts to call the API, but you may need to update
the API call structure based on the actual Sora API documentation when available.
"""
import os
import sys
import json
import base64
import time
import tempfile
import argparse
import shutil
import requests
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import imageio_ffmpeg

import llm_utils

load_dotenv()
client = OpenAI()  # Used for video (Sora) API; images use llm_utils

# Configuration
VIDEO_MODEL = os.getenv("KIDS_VIDEO_MODEL", "sora-2-pro")  # Sora model name
SCENE_DURATION = int(os.getenv("KIDS_SCENE_DURATION", "12"))

# Directories
IMAGES_DIR = Path("generated_images_kids")
VIDEOS_DIR = Path("kids_videos")
OUTPUT_DIR = Path("finished_kids_shorts")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_RESOLUTION = "1024x1792"

# Video settings
OUTPUT_RESOLUTION = (1024, 1792)  # 9:16 vertical for YouTube Shorts (1080, 1920)
FPS = 30
MAX_WORKERS = 3  # Lower for video generation (more resource intensive)


class Config:
    save_assets = False
    temp_dir = None

config = Config()


def build_scene_context(scene: dict, prev_scene: dict | None, next_scene: dict | None, story: dict) -> str:
    """
    Build context string for image/video generation to ensure continuity.
    Same function as in build_script_kids.py - kept here for video generation.
    """
    context_parts = [
        f"STORY: {story.get('title', 'Untitled')}",
        f"SUMMARY: {story.get('summary', '')}",
        f"THEME: {story.get('theme', '')}",
        f"SETTING: {story.get('setting', '')}"
    ]
    
    # Add style prompt if available
    style_prompt = story.get('style_prompt', '')
    if style_prompt:
        context_parts.append(f"STYLE: {style_prompt}")
    
    # Add music description if available
    music_description = story.get('music_description', '')
    if music_description:
        context_parts.append(f"MUSIC: {music_description}")
    
    # Add character descriptions and voice descriptions
    characters = story.get('characters', [])
    if characters:
        char_descriptions = []
        char_voices = []
        for char in characters:
            if isinstance(char, dict):
                char_descriptions.append(f"{char.get('name', '?')}: {char.get('description', '')}")
                voice_desc = char.get('voice_description', '')
                if voice_desc:
                    char_voices.append(f"{char.get('name', '?')}: {voice_desc}")
            else:
                char_descriptions.append(str(char))
        context_parts.append(f"CHARACTERS: {', '.join(char_descriptions)}")
        if char_voices:
            context_parts.append(f"CHARACTER_VOICES: {', '.join(char_voices)}")
    
    # Add previous scene context
    if prev_scene:
        context_parts.append(f"\nPREVIOUS SCENE:")
        context_parts.append(f"  Title: {prev_scene.get('title', 'N/A')}")
        context_parts.append(f"  Description: {prev_scene.get('description', 'N/A')}")
        context_parts.append(f"  Visual: {prev_scene.get('image_prompt', 'N/A')[:200]}...")
    
    # Add next scene context
    if next_scene:
        context_parts.append(f"\nNEXT SCENE:")
        context_parts.append(f"  Title: {next_scene.get('title', 'N/A')}")
        context_parts.append(f"  Description: {next_scene.get('description', 'N/A')}")
    
    return "\n".join(context_parts)


def generate_image_for_scene(scene: dict, story: dict, save_assets: bool = False) -> Path:
    """
    Generate an image for the scene using OpenAI Images API.
    
    Args:
        scene: Scene dict with image_prompt
        story: Story metadata dict with title, style_prompt, characters, etc.
        save_assets: If True, save to IMAGES_DIR, else use temp
    
    Returns:
        Path to generated image
    """
    scene_id = scene.get('id', 0)
    
    # Only scene 1 should have an image_prompt
    if not scene.get('image_prompt'):
        raise ValueError(f"Scene {scene_id} missing image_prompt. Only scene 1 should have an image_prompt.")
    
    if save_assets:
        img_path = IMAGES_DIR / f"scene_{scene_id:02d}.png"
        if img_path.exists():
            print(f"[IMAGE] Scene {scene_id}: using cached {img_path.name}")
            return img_path
    else:
        img_path = Path(config.temp_dir) / f"scene_{scene_id:02d}.png"
    
    # Extract values directly from story dict
    story_title = story.get('title', 'Untitled')
    style_prompt = story.get('style_prompt', '')
    
    # Build character descriptions from story dict
    character_descriptions = ""
    characters = story.get('characters', [])
    if characters:
        char_desc_parts = []
        for char in characters:
            if isinstance(char, dict):
                char_desc_parts.append(f"{char.get('name', '?')}: {char.get('description', '')}")
            else:
                char_desc_parts.append(str(char))
        character_descriptions = ', '.join(char_desc_parts)
    
    # Build concise style section - just a paragraph about Pixar style
    style_section = f"\n\nPixar-style animation: {style_prompt}" if style_prompt else "\n\nPixar-style 3D computer animation with vibrant colors, expressive character designs, polished rendering, and warm, magical atmosphere."
    
    # Include character descriptions if available
    character_section = f"\n\nCharacters: {character_descriptions}" if character_descriptions else ""
    
    # Simplified image prompt - scene prompt, story title, characters, and concise style
    image_prompt = f"""{scene.get('image_prompt', '')}
    
Story: {story_title}{character_section}{style_section}"""

    print(f"[IMAGE] Image prompt: {image_prompt}")
    print(f"[IMAGE] Scene {scene_id}: generating image...")
    
    try:
        llm_utils.generate_image(
            prompt=image_prompt,
            output_path=img_path,
            size=IMAGE_RESOLUTION,  # Vertical format for Shorts
            output_format="png",
        )
        
        if save_assets:
            print(f"[IMAGE] Scene {scene_id}: saved {img_path.name}")
        else:
            print(f"[IMAGE] Scene {scene_id}: generated (temp)")
        
        return img_path
    
    except Exception as e:
        print(f"[IMAGE] Scene {scene_id}: ERROR - {e}")
        raise


def extract_last_frame(video_path: Path, output_path: Path | None = None) -> Path:
    """
    Extract the last frame from a video file using ffmpeg subprocess (most reliable method).
    
    Args:
        video_path: Path to the video file
        output_path: Optional output path for the frame image (defaults to video_path with .png extension)
    
    Returns:
        Path to the extracted frame image
    """
    if output_path is None:
        output_path = video_path.with_suffix('.png')
    
    print(f"[FRAME] Extracting last frame from {video_path.name} using ffmpeg...")
    
    try:
        # Get ffmpeg executable path - prefer system ffmpeg if in PATH, otherwise use bundled
        try:
            # Try system ffmpeg first (if in PATH)
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                ffmpeg_path = 'ffmpeg'
                print(f"[FRAME] Using system ffmpeg from PATH")
            else:
                raise FileNotFoundError("System ffmpeg not found")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback to bundled ffmpeg
            try:
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"[FRAME] Using bundled ffmpeg from: {ffmpeg_path}")
            except Exception:
                raise RuntimeError("ffmpeg not found. Please install ffmpeg and add it to PATH, or ensure imageio-ffmpeg is installed.")
        
        # Get total frame count using ffprobe for frame-accurate extraction
        # Determine ffprobe path based on ffmpeg path
        if ffmpeg_path == 'ffmpeg':
            ffprobe_path = 'ffprobe'  # System ffprobe from PATH
        else:
            # Bundled ffmpeg - ffprobe should be in the same directory
            ffprobe_path = str(Path(ffmpeg_path).parent / 'ffprobe.exe') if Path(ffmpeg_path).suffix == '.exe' else str(Path(ffmpeg_path).parent / 'ffprobe')
        
        probe_cmd = [
            ffprobe_path,
            '-v', 'error',
            '-count_frames',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_read_frames',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=30)
            frame_count_str = probe_result.stdout.strip()
            if not frame_count_str:
                raise ValueError("Empty frame count output")
            frame_count = int(frame_count_str)
            if frame_count <= 0:
                raise ValueError(f"Invalid frame count: {frame_count}")
            print(f"[FRAME] Total frames: {frame_count}, extracting last frame (index {frame_count - 1})")
        except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[FRAME] Warning: Could not get frame count (ffprobe may not be available), using -sseof method: {e}")
            frame_count = None
        
        # Use ffmpeg to extract the last frame
        # Method 1: If we have frame count, use frame-number selection (most reliable)
        # Method 2: Otherwise, use -sseof (less reliable but works as fallback)
        if frame_count is not None:
            # Extract the last frame by frame number using video filter
            # select=eq(n\,N-1) selects frame at index N-1 (the last frame)
            cmd = [
                ffmpeg_path,
                '-y',  # Overwrite output
                '-i', str(video_path),
                '-vf', f'select=eq(n\\,{frame_count - 1})',  # Select last frame by index
                '-update', '1',  # Update single file (required for image2 muxer with single output)
                '-q:v', '2',  # High quality (for PNG, lower is better, but 2 is good)
                str(output_path)
            ]
        else:
            # Fallback to -sseof method with more buffer
            cmd = [
                ffmpeg_path,
                '-y',  # Overwrite output
                '-sseof', '-0.2',  # Seek to 0.2 seconds before end (more buffer)
                '-i', str(video_path),
                '-t', '0.1',  # Limit to 0.1 second segment
                '-frames:v', '1',  # Extract exactly 1 frame
                '-q:v', '2',  # High quality (for PNG, lower is better, but 2 is good)
                str(output_path)
            ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check ffmpeg output for any warnings/errors even if return code is 0
        if result.stderr:
            print(f"[FRAME] ffmpeg stderr: {result.stderr}")
        
        # Verify the output file was created
        if not output_path.exists():
            error_details = f"stderr: {result.stderr}" if result.stderr else "no error output"
            raise RuntimeError(f"ffmpeg did not create output file: {output_path}. {error_details}")
        
        print(f"[FRAME] Successfully extracted last frame: {output_path.name}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else str(e))
        print(f"[FRAME] ERROR: ffmpeg command failed")
        print(f"[FRAME] Command: {' '.join(cmd)}")
        print(f"[FRAME] Return code: {e.returncode}")
        if e.stderr:
            print(f"[FRAME] stderr: {e.stderr}")
        if e.stdout:
            print(f"[FRAME] stdout: {e.stdout}")
        raise RuntimeError(f"Failed to extract last frame from {video_path}: {error_msg}")
    except FileNotFoundError as e:
        print(f"[FRAME] ERROR: ffmpeg executable not found")
        print(f"[FRAME] Tried path: {ffmpeg_path}")
        print(f"[FRAME] Error: {e}")
        raise RuntimeError(f"ffmpeg not found at {ffmpeg_path}. Make sure ffmpeg is installed and in PATH.")
    except Exception as e:
        print(f"[FRAME] ERROR extracting frame: {e}")
        raise


def generate_video_for_scene(scene: dict, image_path: Path, story: dict, save_assets: bool = False, max_attempts: int = 3) -> Path:
    """
    Generate a video for the scene using OpenAI Sora API.
    Uses the generated image as a reference frame.
    
    Args:
        scene: Scene dict with video_prompt
        image_path: Path to the generated image (reference frame)
        story: Story metadata dict with style_prompt, music_description, characters, etc.
        save_assets: If True, save to VIDEOS_DIR, else use temp
        max_attempts: Maximum retry attempts for API calls
    
    Returns:
        Path to generated video
    """
    scene_id = scene.get('id', 0)
    
    if save_assets:
        video_path = VIDEOS_DIR / f"scene_{scene_id:02d}.mp4"
        if video_path.exists():
            print(f"[VIDEO] Scene {scene_id}: using cached {video_path.name}")
            return video_path
    else:
        video_path = Path(config.temp_dir) / f"scene_{scene_id:02d}.mp4"
    
    # Extract values directly from story dict
    style_prompt = story.get('style_prompt', '')
    music_description = story.get('music_description', '')
    
    # Build character voices from story dict - format as individual character cards
    voice_section = ""
    characters = story.get('characters', [])
    if characters:
        voice_cards = []
        for char in characters:
            if isinstance(char, dict):
                voice_desc = char.get('voice_description', '')
                if voice_desc:
                    char_name = char.get('name', '?')
                    voice_cards.append(
                        f"{char_name}'s VOICE (use this EXACT voice every time {char_name} speaks): {voice_desc}"
                    )
        
        if voice_cards:
            voice_section = "CHARACTER VOICE SPECIFICATIONS (CRITICAL - use these EXACT voices consistently throughout the entire video):\n" + "\n\n".join(voice_cards) + "\n\nIMPORTANT: When characters speak, they MUST use their EXACT voices as specified above. Maintain consistent pitch, tone, pace, timbre, and all vocal characteristics for each character. Do NOT change voices between scenes."
    
    # Concise style section - just a paragraph about Pixar style
    style_section = f"Pixar-style animation: {style_prompt}" if style_prompt else "Pixar-style 3D computer animation with vibrant colors, expressive character designs, polished rendering, and warm, magical atmosphere."
    
    # Camera motion instructions to avoid nauseating movements
    camera_instruction = "Camera motion: Keep camera movement simple, smooth, and steady. Avoid rapid movements, shaky camera, or nauseating rotations. Use slow, gentle pans or fixed camera positions when possible."
    
    # Constraints on dialogue and actions
    constraints_instruction = "CRITICAL CONSTRAINTS: The video must contain ONLY the actions and dialogue specified in the Video Content above. Do NOT add any extra dialogue beyond what is explicitly described. Do NOT add any additional major actions by the characters beyond what is specified. If the scene needs to fill time to reach the full duration, use subtle background activity or minor environmental details (like leaves rustling, birds flying in the distance, gentle character movements like breathing or slight head turns) that do not affect the story or advance the plot. Keep all filler activity minimal and non-intrusive."
    
    # CRITICAL - No audio that isn't explicitly written
    no_extra_audio_instruction = "CRITICAL - NO EXTRA AUDIO: The video must contain ONLY the audio that is explicitly written in the Video Content above. Do NOT add any sound effects, ambient sounds, voice-over, narration, or any other audio that is not specifically mentioned in the Video Content. Only include dialogue that is explicitly written (e.g., '[Character Name] says, \"...\"'). Do NOT add footsteps, rustling, chirping, or any other sounds unless they are explicitly described in the Video Content. The only audio should be: 1) Background music (as specified), 2) Dialogue that is explicitly written in the Video Content. Nothing else."
    
    # CRITICAL - No text, credits, or logos
    no_text_instruction = "CRITICAL - NO TEXT OR CREDITS: The video must contain NO text, NO credits, NO logos, NO end screens, NO titles, NO watermarks, NO written words of any kind, and NO closing scenes. The video should end with the story action only - simply end with the characters in their final positions. Do NOT add any text overlays, credits, company logos, or end screens at any point in the video."
    
    # Use the story's music description, or fallback to generic description
    # Include fade-out instruction at the end of the scene
    if music_description:
        music_instruction = f"Background music: {music_description}. The music should fade out gradually at the end of the scene."
    else:
        # Fallback if music_description wasn't generated
        music_instruction = "Background music: Use kid-friendly, cheerful background music with a magical, uplifting feel. The music should be instrumental, light, and whimsical - think playful piano, gentle strings, soft bells, and airy melodies. The tempo should be moderate and steady. The style should be consistent throughout all scenes - warm, inviting, and suitable for a children's animated story. The music should enhance the story's mood without overwhelming the dialogue or visuals. The music should fade out gradually at the end of the scene."
    
    # Put voice descriptions FIRST for maximum prominence, then scene description
    video_prompt = f"""{voice_section}

Video Content: {scene.get('video_prompt', '')}

{style_section}

{camera_instruction}

{music_instruction}

{constraints_instruction}

{no_extra_audio_instruction}

{no_text_instruction}"""
    
    print(f"[VIDEO] Video prompt: {video_prompt}")
    print(f"[VIDEO] Scene {scene_id}: generating video from image...")
    
    # Retry logic for API calls
    attempt = 1
    base_delay = 3.0
    
    while attempt <= max_attempts:
        try:
            # Pass image path directly - API will detect MIME type from file extension
            # Path objects are PathLike and SDK can determine MIME type (.png -> image/png)
            video_job = client.videos.create(
                model=VIDEO_MODEL,
                prompt=video_prompt,
                input_reference=image_path,  # Pass PathLike - SDK detects MIME type
                size=IMAGE_RESOLUTION,
                seconds=SCENE_DURATION,
            )
            
            if video_job is None:
                raise ValueError("Sora API call returned None")

            # Wait for video generation to complete (polling)
            video_id = video_job.id
            max_wait_time = 1200  # 20 minutes max
            poll_interval = 30  # Check every 30 seconds
            waited = 0
            
            print(f"[VIDEO] Scene {scene_id}: waiting for video generation (job: {video_id})...")
            while waited < max_wait_time:
                status = client.videos.retrieve(video_id=video_id)
                
                # Status might be "pending", "processing", "completed", "failed", etc.
                # Adjust based on actual API response structure
                if hasattr(status, 'status'):
                    job_status = status.status
                elif isinstance(status, dict):
                    job_status = status.get('status', 'unknown')
                else:
                    # If status not available, assume ready after first poll
                    job_status = 'completed'
                
                if job_status in ['completed', 'succeeded', 'ready']:
                    print(f"[VIDEO] Scene {scene_id}: video generation completed")
                    break
                elif job_status in ['failed', 'error', 'cancelled']:
                    raise RuntimeError(f"Video generation failed with status: {job_status}")
                
                print(f"[VIDEO] Scene {scene_id}: status={job_status}, waiting... ({waited}s)")
                time.sleep(poll_interval)
                waited += poll_interval
            
            if waited >= max_wait_time:
                raise TimeoutError(f"Video generation timed out after {max_wait_time} seconds")

            # Download the completed video
            response = client.videos.download_content(video_id=video_id)
            content = response.read()
            
            # Save video content to file
            with open(video_path, "wb") as f:
                f.write(content)
            
            if save_assets:
                print(f"[VIDEO] Scene {scene_id}: saved {video_path.name}")
            else:
                print(f"[VIDEO] Scene {scene_id}: generated (temp)")
            
            return video_path
        
        except NotImplementedError:
            # Don't retry - this is a configuration issue
            raise
        except Exception as e:
            if attempt >= max_attempts:
                print(f"[VIDEO] Scene {scene_id}: Failed after {attempt} attempts: {e}")
                raise
            
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[VIDEO] Scene {scene_id}: Attempt {attempt} failed: {e}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            attempt += 1


def build_kids_video(story_path: str, output_path: str, save_assets: bool = False):
    """
    Main orchestrator: Generate videos for all scenes and concatenate them.
    
    Args:
        story_path: Path to story JSON file
        output_path: Output video file path
        save_assets: If True, save images/videos to permanent directories
    """
    config.save_assets = save_assets
    
    # Set up temp directory if not saving assets
    if not config.save_assets:
        config.temp_dir = tempfile.mkdtemp(prefix="kids_video_")
        print(f"[TEMP] Using temporary directory: {config.temp_dir}")
    
    try:
        # Load story
        print(f"\n[LOAD] Loading story from: {story_path}")
        with open(story_path, "r", encoding="utf-8") as f:
            story_data = json.load(f)
        
        story = story_data.get("metadata", {})
        scenes = story_data.get("scenes", [])
        
        print(f"[STORY] {story.get('title', 'Untitled')}")
        print(f"[STORY] {len(scenes)} scenes")
        
        if not scenes:
            raise ValueError("No scenes found in story file")
        
        # Generate images and videos for each scene
        print(f"\n[GENERATION] Generating images and videos for {len(scenes)} scenes...")
        
        image_paths = [None] * len(scenes)
        video_paths = [None] * len(scenes)
        
        def process_scene(idx: int):
            """Generate image then video for a single scene."""
            scene = scenes[idx]
            scene_id = scene.get('id', idx + 1)
            
            # Only generate image from scratch for scene 1 (if it has image_prompt)
            # For subsequent scenes, extract the last frame from the previous video
            if idx == 0 and scene.get('image_prompt'):
                # Scene 1: Generate image from scratch
                print(f"[SCENE {scene_id}] Generating initial image from scratch...")
                img_path = generate_image_for_scene(scene, story, save_assets=save_assets)
                image_paths[idx] = img_path
            elif idx > 0 and video_paths[idx - 1] is not None:
                # Scene 2+: Extract last frame from previous video
                print(f"[SCENE {scene_id}] Extracting last frame from previous scene...")
                prev_video_path = video_paths[idx - 1]
                
                # Verify the previous video exists and is valid
                if not prev_video_path.exists():
                    raise FileNotFoundError(f"Previous scene video not found: {prev_video_path}")
                
                print(f"[SCENE {scene_id}] Using previous video: {prev_video_path.name}")
                
                if save_assets:
                    img_path = IMAGES_DIR / f"scene_{scene_id:02d}_from_prev.png"
                else:
                    img_path = Path(config.temp_dir) / f"scene_{scene_id:02d}_from_prev.png"
                img_path = extract_last_frame(prev_video_path, img_path)
                image_paths[idx] = img_path
            else:
                raise ValueError(f"Scene {scene_id} cannot generate image: scene 1 missing image_prompt or previous scene missing video")
            
            # Generate video from image (or previous video's last frame)
            video_path = generate_video_for_scene(scene, img_path, story, save_assets=save_assets)
            video_paths[idx] = video_path
            
            return idx, img_path, video_path
        
        # Process scenes (can parallelize, but video generation is resource-intensive)
        # Process sequentially for now to avoid overwhelming API
        print("\n[PROCESSING] Generating scenes sequentially...")
        for idx in range(len(scenes)):
            try:
                process_scene(idx)
            except Exception as e:
                print(f"[ERROR] Scene {scenes[idx].get('id', idx)}: {e}")
                raise
        
        # Verify all videos were generated
        missing_videos = [i for i, p in enumerate(video_paths) if p is None or not p.exists()]
        if missing_videos:
            raise RuntimeError(f"Missing video files for scenes: {missing_videos}")
        
        # Concatenate videos
        print(f"\n[CONCATENATE] Stitching {len(video_paths)} video clips together...")
        
        video_clips = []
        for i, video_path in enumerate(video_paths):
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            try:
                clip = VideoFileClip(str(video_path))
                # Ensure all clips are same resolution (resize if needed)
                if clip.size != OUTPUT_RESOLUTION:
                    print(f"[CLIP {i+1}] Resizing from {clip.size} to {OUTPUT_RESOLUTION}")
                    clip = clip.resized(OUTPUT_RESOLUTION)
                
                print(f"[CLIP {i+1}] Scene {scenes[i].get('id', i+1)}: {clip.duration:.2f}s, {clip.size}")
                video_clips.append(clip)
            except Exception as e:
                print(f"[ERROR] Failed to load video clip {i+1} from {video_path}: {e}")
                raise
        
        if not video_clips:
            raise RuntimeError("No valid video clips to concatenate")
        
        # Concatenate all clips
        print(f"[CONCATENATE] Concatenating {len(video_clips)} clips...")
        try:
            final = concatenate_videoclips(video_clips, method="compose")
        except Exception as e:
            print(f"[ERROR] Concatenation failed: {e}")
            print("[INFO] Trying 'chain' method instead...")
            final = concatenate_videoclips(video_clips, method="chain")
        
        print(f"[VIDEO] Final duration: {final.duration:.2f}s ({final.duration/60:.1f} minutes)")
        
        # Ensure output is in correct directory
        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute() and str(output_path_obj.parent) == ".":
            output_path = str(OUTPUT_DIR / output_path_obj.name)
        
        # Write final video
        print(f"\n[ENCODE] Encoding final video: {output_path}")
        print("[ENCODE] This may take a while...")
        
        max_threads = os.cpu_count() or 8
        threads = min(max_threads, 16)
        
        # Write final video (preserve audio if present)
        final.write_videofile(
            output_path,
            fps=FPS,
            codec="libx264",
            audio=True,  # Preserve audio from source videos if present
            preset="medium",  # fast
            threads=threads,
            bitrate=None,
            ffmpeg_params=[
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
            ],
        )
        
        # Close clips to free resources
        for clip in video_clips:
            clip.close()
        final.close()
        
        print("[VIDEO] Done!")
        
    finally:
        # Clean up temp directory
        if not config.save_assets and config.temp_dir and os.path.exists(config.temp_dir):
            print(f"[TEMP] Cleaning up temporary files...")
            shutil.rmtree(config.temp_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate animated videos for kids shorts from story JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate videos from story
  python build_video_kids.py kids_scripts/robot_story.json output.mp4
  
  # Generate videos and save assets
  python build_video_kids.py kids_scripts/robot_story.json output.mp4 --save-assets
        """
    )
    
    parser.add_argument("story_file", help="Path to story JSON file")
    parser.add_argument("output_file", help="Output video file path")
    parser.add_argument("--save-assets", action="store_true",
                        help="Save generated images and videos to permanent directories")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"[KIDS VIDEO] Building video from: {args.story_file}")
    print(f"{'='*60}")
    
    if args.save_assets:
        print("[MODE] Saving assets to generated_images_kids/ and kids_videos/")
    else:
        print("[MODE] Using temporary files (will be cleaned up after)")
    
    try:
        build_kids_video(args.story_file, args.output_file, save_assets=args.save_assets)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\n🎬 Video saved: {args.output_file}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
