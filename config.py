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
    scenes_per_chapter = 4  # Scenes per chapter (total = chapters * scenes_per_chapter)
    generate_main = True    # Whether to generate main video
    
    # Shorts settings
    num_shorts = 3              # Number of YouTube Shorts
    short_chapters = 1          # Chapters per short (1 chapter: 3 scenes telling one complete story)
    short_scenes_per_chapter = 4  # Scenes per chapter in shorts (3 build + scene 4 answers question from scene 3)
    
    generate_thumbnails = True  # Whether to generate thumbnail images (main video)
    generate_short_thumbnails = False  # Whether to generate thumbnails for shorts (usually not needed)
    generate_refinement_diffs = False  # Whether to generate refinement diff JSON files
    
    @property
    def total_scenes(self):
        return self.chapters * self.scenes_per_chapter
    
    @property
    def total_short_scenes(self):
        return self.short_chapters * self.short_scenes_per_chapter
