"""
Base abstraction for different script types.
Makes it easy to add new script types by creating a new class that inherits from ScriptType.
"""
from abc import ABC, abstractmethod
from typing import Optional
import prompt_builders


class ScriptType(ABC):
    """Base class for different script types (main video, shorts, etc.)."""
    
    @abstractmethod
    def get_outline_prompt(self, subject: str, chapters: int, total_scenes: int) -> str:
        """Generate the outline prompt for this script type.
        
        Args:
            subject: Generic subject (person name for biopics, story concept for horror, etc.)
            chapters: Number of chapters
            total_scenes: Total number of scenes
        """
        pass
    
    @abstractmethod
    def get_scene_generation_prompt(self, context: dict) -> str:
        """Generate the scene generation prompt for this script type."""
        pass
    
    @abstractmethod
    def get_refinement_passes(self) -> list[str]:
        """Return which refinement passes to run (e.g., ['storyline', 'pivotal', 'final'])."""
        pass
    
    @abstractmethod
    def get_metadata_prompt(self, subject: str, tagline: str, total_scenes: int) -> str:
        """Generate the metadata prompt for this script type.
        
        Args:
            subject: Generic subject (person name for biopics, story concept for horror, etc.)
            tagline: One-line tagline
            total_scenes: Total number of scenes
        """
        pass


class MainVideoScript(ScriptType):
    """Main video script type - full documentary with chapters."""
    
    def get_outline_prompt(self, subject: str, chapters: int, total_scenes: int) -> str:
        """Use the standard outline prompt builder. Subject is person name for biopics."""
        return prompt_builders.build_outline_prompt(subject, chapters, total_scenes)
    
    def get_scene_generation_prompt(self, context: dict) -> str:
        """
        Build scene generation prompt for main video.
        
        Context should include:
        - person, chapter, scenes_per_chapter, start_id, global_style
        - prev_chapter, prev_scenes, central_theme, narrative_arc
        - planted_seeds, is_retention_hook_point, birth_year, death_year
        - tag_line, overarching_plots, sub_plots
        """
        # This is complex - for now, keep the existing function
        # In the future, this could be refactored to use prompt builders more extensively
        raise NotImplementedError("Main video scene generation is still handled by generate_scenes_for_chapter()")
    
    def get_refinement_passes(self) -> list[str]:
        """Main videos go through all refinement passes."""
        return ['storyline', 'pivotal', 'final']
    
    def get_metadata_prompt(self, subject: str, tagline: str, total_scenes: int) -> str:
        """Use the standard metadata prompt builder. Subject is person name for biopics."""
        return prompt_builders.build_metadata_prompt(subject, tagline, total_scenes)


class ShortScript(ScriptType):
    """Short script type - high-energy trailers."""
    
    def get_outline_prompt(self, subject: str, chapters: int, total_scenes: int) -> str:
        """Shorts don't use outline prompts - they use hook-based outlines."""
        raise NotImplementedError("Shorts use generate_short_outline() instead")
    
    def get_scene_generation_prompt(self, context: dict) -> str:
        """
        Build scene generation prompt for shorts.
        
        Context should include:
        - person, short_outline, birth_year, death_year
        """
        # This is handled by generate_short_scenes() which uses prompt builders
        raise NotImplementedError("Short scene generation is handled by generate_short_scenes()")
    
    def get_refinement_passes(self) -> list[str]:
        """Shorts only go through final refinement (skip storyline and pivotal)."""
        return ['final']
    
    def get_metadata_prompt(self, subject: str, tagline: str, total_scenes: int) -> str:
        """Shorts don't use this - they have their own metadata in the outline."""
        raise NotImplementedError("Shorts generate metadata in generate_short_outline()")


class HorrorStoryScript(ScriptType):
    """Horror story script type - fictional scary stories with first-person narration."""
    
    def get_outline_prompt(self, subject: str, chapters: int, total_scenes: int) -> str:
        """Use horror-specific outline prompt builder. Subject is story concept."""
        return prompt_builders.build_horror_outline_prompt(subject, chapters, total_scenes)
    
    def get_scene_generation_prompt(self, context: dict) -> str:
        """
        Build scene generation prompt for horror stories.
        
        Context should include:
        - story_concept, chapter, scenes_per_chapter, start_id, global_style
        - prev_chapter, prev_scenes, central_theme, narrative_arc
        - planted_seeds, is_retention_hook_point
        - tag_line, overarching_plots, sub_plots
        """
        # This is handled by generate_horror_scenes_for_chapter() which uses prompt builders
        raise NotImplementedError("Horror scene generation is handled by generate_horror_scenes_for_chapter()")
    
    def get_refinement_passes(self) -> list[str]:
        """Horror stories go through all refinement passes with horror focus."""
        return ['storyline', 'pivotal', 'final']
    
    def get_metadata_prompt(self, subject: str, tagline: str, total_scenes: int) -> str:
        """Use horror-specific metadata prompt builder. Subject is story concept."""
        return prompt_builders.build_horror_metadata_prompt(subject, tagline, total_scenes)
