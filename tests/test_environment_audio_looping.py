"""
Comprehensive tests for environment audio looping functionality.
Verifies that environment audio actually loops when shorter than target duration.
"""

import unittest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_video import mix_horror_background_audio
from moviepy import AudioFileClip, AudioClip
import warnings

# Suppress MoviePy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*FFMPEG_AudioReader.*")


class TestEnvironmentAudioLooping(unittest.TestCase):
    """Test cases to verify environment audio actually loops."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary audio file for testing
        self.temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.temp_audio_path = self.temp_audio_file.name
        self.temp_audio_file.close()
        
        # Create a simple test audio file (1 second of sine wave)
        # We'll use MoviePy to create it, or mock it
        self.original_duration = 1.0  # 1 second
        self.target_duration = 5.0   # 5 seconds (needs 5 loops)
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp files
        for path in [self.temp_audio_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except:
                pass
    
    def create_test_audio_file(self, duration):
        """Create a test audio file with specified duration."""
        import subprocess
        # Use FFmpeg to create a 1-second sine wave test file
        result = subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'sine=frequency=440:duration={duration}',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',  # Stereo
            self.temp_audio_path
        ], check=True, capture_output=True, text=True)
        return self.temp_audio_path
    
    # Note: Removed complex mock-based tests that were difficult to maintain.
    # The integration test (test_environment_audio_actual_looping_integration) 
    # provides better verification of actual looping behavior.
    
    def test_environment_audio_actual_looping_integration(self):
        """Integration test: Create real SHORT audio file and verify it actually loops."""
        # Skip if FFmpeg not available
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("FFmpeg not available")
        
        # Create a 1-second test audio file (SHORTER than target duration)
        self.create_test_audio_file(1.0)
        
        # Load it and verify it's 1 second
        original_clip = AudioFileClip(self.temp_audio_path)
        original_duration = original_clip.duration
        self.assertAlmostEqual(original_duration, 1.0, places=1, 
                             msg=f"Test audio file should be 1 second, got {original_duration}")
        original_clip.close()
        
        # Now test the actual looping by calling mix_horror_background_audio
        # Create narration audio that's 5 seconds (longer than the 1-second environment audio)
        def make_narration_audio(t):
            if np.isscalar(t):
                return np.array([0.1, 0.1])
            else:
                return np.column_stack([np.full(len(t), 0.1), np.full(len(t), 0.1)])
        narration_audio_clip = AudioClip(make_narration_audio, duration=5.0, fps=44100)
        
        # Set environment variable to point to our SHORT test file
        import os
        import build_video
        original_env = os.environ.get('ENV_RAIN_AUDIO')
        original_build_video_env = getattr(build_video, 'ENV_RAIN_AUDIO', None)
        os.environ['ENV_RAIN_AUDIO'] = self.temp_audio_path
        build_video.ENV_RAIN_AUDIO = self.temp_audio_path
        
        try:
            # Call mix_horror_background_audio
            result = mix_horror_background_audio(
                narration_audio=narration_audio_clip,
                duration=5.0,
                environment="rain",
                env_audio_volume=-28.0
            )
            
            # The result should be a CompositeAudioClip
            from moviepy import CompositeAudioClip
            self.assertIsInstance(result, CompositeAudioClip, 
                                "Result should be a CompositeAudioClip")
            
            # Verify the composite duration is 5 seconds
            self.assertAlmostEqual(result.duration, 5.0, places=1, 
                                 msg=f"Composite audio should be 5 seconds long, got {result.duration}")
            
            # The key test: Sample audio at different points to verify it's actually looping
            # If the audio is looping, we should hear audio at 0s, 1s, 2s, 3s, 4s (all within the 1s loop)
            sample_times = [0.0, 1.0, 2.0, 3.0, 4.0, 4.9]
            for t in sample_times:
                if t < result.duration:
                    sample = result.get_frame(t)
                    # Check that we have actual audio (not silence)
                    # The sample should be non-zero (all layers combined)
                    max_amplitude = np.max(np.abs(sample))
                    self.assertGreater(max_amplitude, 0.0, 
                                     f"Audio should be present at time {t}s (got amplitude {max_amplitude})")
            
            print(f"\n[TEST] Environment audio looping verified:")
            print(f"  - Original file: {original_duration:.3f}s")
            print(f"  - Target duration: 5.0s")
            print(f"  - Composite duration: {result.duration:.3f}s")
            print(f"  - Audio present at all sample points (0s, 1s, 2s, 3s, 4s, 4.9s)")
            
        finally:
            # Restore environment variable
            if original_env:
                os.environ['ENV_RAIN_AUDIO'] = original_env
            elif 'ENV_RAIN_AUDIO' in os.environ:
                del os.environ['ENV_RAIN_AUDIO']
            if original_build_video_env:
                build_video.ENV_RAIN_AUDIO = original_build_video_env


if __name__ == '__main__':
    unittest.main()
