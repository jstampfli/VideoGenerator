"""
Configuration settings for script generation.
Can be overridden via command line arguments.
"""

import os

# LLM provider and model selection (from .env); used by llm_utils
# TEXT_PROVIDER / IMAGE_PROVIDER: "google" or "openai"
# Default to openai when unset so existing behavior is unchanged
TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "openai").lower()
TEXT_MODEL_OPENAI = os.getenv("TEXT_MODEL_OPENAI", "gpt-5.2")
TEXT_MODEL_GOOGLE = os.getenv("TEXT_MODEL_GOOGLE", "gemini-2.0-flash")
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "openai").lower()
IMAGE_MODEL_OPENAI = os.getenv("IMAGE_MODEL_OPENAI", "gpt-image-1.5")
IMAGE_MODEL_GOOGLE = os.getenv("IMAGE_MODEL_GOOGLE", "gemini-2.0-flash-exp")


class Config:
    # Main video settings
    chapters = 6           # Number of outline chapters
    target_total_scenes = 24  # Target total scenes; outline distributes across chapters via num_scenes
    min_scenes_per_chapter = 2  # Minimum scenes per chapter (for outline guidance)
    max_scenes_per_chapter = 10  # Maximum scenes per chapter (for outline guidance)
    # Legacy fallback: when outline lacks num_scenes, use target_total_scenes / chapters
    generate_main = True    # Whether to generate main video
    
    # Shorts settings
    num_shorts = 3              # Number of YouTube Shorts
    short_chapters = 1          # Chapters per short (1 chapter: 3 scenes telling one complete story)
    short_scenes_per_chapter = 4  # Scenes per chapter in shorts (3 build + scene 4 answers question from scene 3)
    
    generate_thumbnails = True  # Whether to generate thumbnail images (main video)
    generate_short_thumbnails = False  # Whether to generate thumbnails for shorts (usually not needed)
    generate_refinement_diffs = False  # Whether to generate refinement diff JSON files
    use_research = True  # Whether to fetch Wikipedia research for script depth (--no-research to disable)
    
    @property
    def total_scenes(self):
        """Target total scenes (used before outline is generated). Actual count = sum of chapter num_scenes."""
        return self.target_total_scenes

    @property
    def scenes_per_chapter_fallback(self):
        """Fallback when outline lacks num_scenes per chapter (e.g., legacy outlines)."""
        return max(self.min_scenes_per_chapter, self.target_total_scenes // self.chapters)

    @property
    def total_short_scenes(self):
        return self.short_chapters * self.short_scenes_per_chapter
