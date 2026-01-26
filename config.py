"""
Configuration settings for script generation.
Can be overridden via command line arguments.
"""


class Config:
    # Main video settings
    chapters = 6           # Number of outline chapters
    scenes_per_chapter = 4  # Scenes per chapter (total = chapters * scenes_per_chapter)
    generate_main = True    # Whether to generate main video
    
    # Shorts settings
    num_shorts = 3              # Number of YouTube Shorts
    short_chapters = 1          # Chapters per short (1 chapter: 3 scenes telling one complete story)
    short_scenes_per_chapter = 3  # Scenes per chapter in shorts (complete story with natural conclusion)
    
    generate_thumbnails = True  # Whether to generate thumbnail images (main video)
    generate_short_thumbnails = False  # Whether to generate thumbnails for shorts (usually not needed)
    generate_refinement_diffs = False  # Whether to generate refinement diff JSON files
    
    @property
    def total_scenes(self):
        return self.chapters * self.scenes_per_chapter
    
    @property
    def total_short_scenes(self):
        return self.short_chapters * self.short_scenes_per_chapter
