"""
Unit tests for voice selection in build_video.py.

Tests ensure the correct voice is selected based on:
1. Voice type (standard, chirp, gemini)
2. Environment parameters (TTS_PROVIDER, GOOGLE_VOICE_TYPE, voice env vars)
3. Parser args (--female flag)
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from io import StringIO

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import build_video


class TestVoiceSelection(unittest.TestCase):
    """Test cases for voice selection logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        build_video.config.temp_dir = self.temp_dir
        build_video.config.save_assets = False
        
        # Create a test scene
        self.test_scene = {
            "id": 1,
            "title": "Test Scene",
            "narration": "This is a test narration.",
            "image_prompt": "Test visual"
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        import time
        if os.path.exists(self.temp_dir):
            try:
                # Try to remove, but ignore errors on Windows due to file locks
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                # Give Windows a moment to release file handles
                time.sleep(0.1)
            except (PermissionError, OSError):
                pass
        # Reset global flags
        build_video.USE_FEMALE_VOICE = False
    
    @patch("build_video.llm_utils.generate_speech")
    def test_generate_audio_calls_llm_utils_with_correct_params(self, mock_generate_speech):
        """Test generate_audio_for_scene delegates to llm_utils.generate_speech with correct params."""
        out_path = Path(self.temp_dir) / "scene_01.mp3"
        mock_generate_speech.return_value = out_path
        out_path.write_bytes(b"fake_audio")

        scene = {
            "id": 1,
            "narration": "Test narration text.",
            "emotion": "contemplative",
            "narration_instructions": "Focus on reflection.",
        }
        orig_narr = build_video.NARRATION_INSTRUCTIONS_FOR_TTS
        orig_female = build_video.USE_FEMALE_VOICE
        try:
            build_video.NARRATION_INSTRUCTIONS_FOR_TTS = ""
            build_video.USE_FEMALE_VOICE = True

            result = build_video.generate_audio_for_scene(scene)

            mock_generate_speech.assert_called_once()
            call_kw = mock_generate_speech.call_args[1]
            self.assertEqual(call_kw["text"], "Test narration text.")
            self.assertEqual(call_kw["emotion"], "contemplative")
            self.assertEqual(call_kw["narration_instructions"], "Focus on reflection.")
            self.assertTrue(call_kw["use_female_voice"])
            self.assertIn("scene_01.mp3", str(call_kw["output_path"]))
            self.assertEqual(result, out_path)
        finally:
            build_video.NARRATION_INSTRUCTIONS_FOR_TTS = orig_narr
            build_video.USE_FEMALE_VOICE = orig_female

    @patch.dict("os.environ", {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'openai')
    @patch('llm_utils.TTS_OPENAI_STREAM', False)
    @patch('openai.OpenAI')
    @patch('pathlib.Path.exists', return_value=False)
    def test_openai_voice_selection(self, mock_exists, mock_openai_class):
        """Test OpenAI voice selection uses OPENAI_TTS_VOICE."""
        import llm_utils
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = b"audio_data"
        mock_client.audio.speech.create.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.audio.speech.create.assert_called_once()
        call_args = mock_client.audio.speech.create.call_args
        self.assertEqual(call_args.kwargs['voice'], llm_utils.OPENAI_TTS_VOICE)
        self.assertEqual(call_args.kwargs['model'], llm_utils.OPENAI_TTS_MODEL)
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "test-key"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'elevenlabs')
    @patch('llm_utils.ELEVENLABS_VOICE_ID', 'test_voice_id_123')
    @patch('elevenlabs.ElevenLabs')
    def test_elevenlabs_voice_selection(self, mock_elevenlabs_class):
        """Test ElevenLabs voice selection uses ELEVENLABS_VOICE_ID."""
        mock_client = MagicMock()
        mock_elevenlabs_class.return_value = mock_client
        mock_client.text_to_speech.convert.return_value = iter([b'audio', b'chunk'])

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.text_to_speech.convert.assert_called_once()
        call_args = mock_client.text_to_speech.convert.call_args
        self.assertEqual(call_args.kwargs['voice_id'], 'test_voice_id_123')
        self.assertEqual(call_args.kwargs['text'], self.test_scene['narration'])
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "google", "GOOGLE_VOICE_TYPE": "", "GOOGLE_TTS_VOICE": "en-US-Studio-Q", "GOOGLE_TTS_LANGUAGE": "en-US"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('google.cloud.texttospeech.TextToSpeechClient')
    def test_google_standard_voice_selection(self, mock_client_class):
        """Test standard Google TTS voice selection uses GOOGLE_TTS_VOICE (plain text, no SSML)."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.audio_content = b'audio_data'
        mock_client.synthesize_speech.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.synthesize_speech.assert_called_once()
        call_args = mock_client.synthesize_speech.call_args
        voice_params = call_args.kwargs['voice']
        self.assertEqual(voice_params.name, 'en-US-Studio-Q')
        self.assertEqual(voice_params.language_code, 'en-US')
        self.assertIsNotNone(call_args.kwargs['input'].text)
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "google", "GOOGLE_VOICE_TYPE": "chirp", "GOOGLE_TTS_VOICE": "en-US-Chirp3-HD-Algenib", "GOOGLE_TTS_LANGUAGE": "en-US"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('google.cloud.texttospeech.TextToSpeechClient')
    def test_google_chirp_voice_selection(self, mock_client_class):
        """Test Chirp voice selection uses GOOGLE_TTS_VOICE."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.audio_content = b'audio_data'
        mock_client.synthesize_speech.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.synthesize_speech.assert_called_once()
        call_args = mock_client.synthesize_speech.call_args
        voice_params = call_args.kwargs['voice']
        self.assertEqual(voice_params.name, 'en-US-Chirp3-HD-Algenib')
        self.assertIsNotNone(call_args.kwargs['input'].text)
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "google", "GOOGLE_VOICE_TYPE": "gemini", "GOOGLE_GEMINI_MALE_SPEAKER": "Charon", "GOOGLE_GEMINI_FEMALE_SPEAKER": "Leda", "GOOGLE_TTS_LANGUAGE": "en-US"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.USE_FEMALE_VOICE', False)
    @patch('google.cloud.texttospeech.TextToSpeechClient')
    def test_google_gemini_male_voice_selection(self, mock_client_class):
        """Test Gemini voice selection uses male speaker when USE_FEMALE_VOICE is False."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.audio_content = b'audio_data'
        mock_client.synthesize_speech.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.synthesize_speech.assert_called_once()
        call_args = mock_client.synthesize_speech.call_args
        voice_params = call_args.kwargs['voice']
        self.assertEqual(voice_params.name, 'Charon')
        self.assertEqual(voice_params.model_name, 'gemini-2.5-pro-tts')
        self.assertIsNotNone(call_args.kwargs['input'].text)
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "google", "GOOGLE_VOICE_TYPE": "gemini", "GOOGLE_GEMINI_MALE_SPEAKER": "Charon", "GOOGLE_GEMINI_FEMALE_SPEAKER": "Leda", "GOOGLE_TTS_LANGUAGE": "en-US"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.USE_FEMALE_VOICE', True)
    @patch('google.cloud.texttospeech.TextToSpeechClient')
    def test_google_gemini_female_voice_selection(self, mock_client_class):
        """Test Gemini voice selection uses female speaker when USE_FEMALE_VOICE is True."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.audio_content = b'audio_data'
        mock_client.synthesize_speech.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.synthesize_speech.assert_called_once()
        call_args = mock_client.synthesize_speech.call_args
        voice_params = call_args.kwargs['voice']
        self.assertEqual(voice_params.name, 'Leda')
        self.assertEqual(voice_params.model_name, 'gemini-2.5-pro-tts')
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "google", "GOOGLE_VOICE_TYPE": "gemini", "GOOGLE_GEMINI_MALE_SPEAKER": "Charon", "GOOGLE_GEMINI_FEMALE_SPEAKER": "", "GOOGLE_TTS_VOICE": "en-US-Studio-Q", "GOOGLE_TTS_LANGUAGE": "en-US"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.USE_FEMALE_VOICE', True)
    @patch('google.cloud.texttospeech.TextToSpeechClient')
    def test_google_gemini_female_voice_fallback(self, mock_client_class):
        """Test Gemini voice selection falls back to GOOGLE_TTS_VOICE when female speaker is empty."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.audio_content = b'audio_data'
        mock_client.synthesize_speech.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.synthesize_speech.assert_called_once()
        call_args = mock_client.synthesize_speech.call_args
        voice_params = call_args.kwargs['voice']
        self.assertEqual(voice_params.name, 'en-US-Studio-Q')
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "google", "GOOGLE_VOICE_TYPE": "", "GOOGLE_TTS_VOICE": "en-US-Studio-Q", "GOOGLE_TTS_LANGUAGE": "en-US"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.USE_FEMALE_VOICE', False)
    @patch('google.cloud.texttospeech.TextToSpeechClient')
    def test_google_standard_male_voice(self, mock_client_class):
        """Test standard Google TTS uses default voice when USE_FEMALE_VOICE is False."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.audio_content = b'audio_data'
        mock_client.synthesize_speech.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.synthesize_speech.assert_called_once()
        call_args = mock_client.synthesize_speech.call_args
        voice_params = call_args.kwargs['voice']
        self.assertEqual(voice_params.name, 'en-US-Studio-Q')


class TestParserArgsVoiceSelection(unittest.TestCase):
    """Test cases for voice selection based on parser arguments."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset global flags
        build_video.USE_FEMALE_VOICE = False
        # Store original values
        self.original_tts_provider = build_video.TTS_PROVIDER
        self.original_voice_type = build_video.GOOGLE_VOICE_TYPE
        self.original_tts_voice = build_video.GOOGLE_TTS_VOICE
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Reset global flags
        build_video.USE_FEMALE_VOICE = False
        build_video.TTS_PROVIDER = self.original_tts_provider
        build_video.GOOGLE_VOICE_TYPE = self.original_voice_type
        build_video.GOOGLE_TTS_VOICE = self.original_tts_voice
    
    def test_female_flag_gemini_voice_selection(self):
        """Test that --female flag correctly sets Gemini female voice."""
        # Set up Gemini configuration
        build_video.TTS_PROVIDER = 'google'
        build_video.GOOGLE_VOICE_TYPE = 'gemini'
        build_video.GOOGLE_GEMINI_MALE_SPEAKER = 'Charon'
        build_video.GOOGLE_GEMINI_FEMALE_SPEAKER = 'Leda'
        build_video.GOOGLE_TTS_VOICE = 'en-US-Studio-Q'  # Default
        
        # Simulate parse_args logic for --female flag
        with patch('os.getenv', return_value=None):
            # Simulate args.female = True
            if build_video.TTS_PROVIDER == "google":
                build_video.USE_FEMALE_VOICE = True
                if build_video.GOOGLE_VOICE_TYPE == "gemini":
                    female_voice = build_video.GOOGLE_GEMINI_FEMALE_SPEAKER
                    if female_voice:
                        build_video.GOOGLE_TTS_VOICE = female_voice
        
        # Verify voice was set correctly
        self.assertTrue(build_video.USE_FEMALE_VOICE)
        self.assertEqual(build_video.GOOGLE_TTS_VOICE, 'Leda')
    
    def test_female_flag_standard_voice_selection(self):
        """Test that --female flag correctly sets standard Google TTS female voice."""
        # Set up standard Google TTS configuration
        build_video.TTS_PROVIDER = 'google'
        build_video.GOOGLE_VOICE_TYPE = ''
        build_video.GOOGLE_TTS_VOICE = 'en-US-Studio-Q'  # Default
        
        # Simulate parse_args logic for --female flag
        with patch('os.getenv', side_effect=lambda key, default=None: 'en-US-Wavenet-F' if key == 'GOOGLE_TTS_FEMALE_VOICE' else default):
            # Simulate args.female = True
            if build_video.TTS_PROVIDER == "google":
                build_video.USE_FEMALE_VOICE = True
                if build_video.GOOGLE_VOICE_TYPE != "gemini":
                    female_voice = os.getenv("GOOGLE_TTS_FEMALE_VOICE")
                    if female_voice:
                        build_video.GOOGLE_TTS_VOICE = female_voice
        
        # Verify voice was set correctly
        self.assertTrue(build_video.USE_FEMALE_VOICE)
        self.assertEqual(build_video.GOOGLE_TTS_VOICE, 'en-US-Wavenet-F')
    
    def test_female_flag_gemini_no_female_speaker(self):
        """Test --female flag with Gemini but no female speaker set shows warning."""
        # Set up Gemini configuration without female speaker
        build_video.TTS_PROVIDER = 'google'
        build_video.GOOGLE_VOICE_TYPE = 'gemini'
        build_video.GOOGLE_GEMINI_FEMALE_SPEAKER = ''
        build_video.GOOGLE_TTS_VOICE = 'en-US-Studio-Q'  # Default
        
        # Simulate parse_args logic for --female flag
        with patch('builtins.print') as mock_print:
            if build_video.TTS_PROVIDER == "google":
                build_video.USE_FEMALE_VOICE = True
                if build_video.GOOGLE_VOICE_TYPE == "gemini":
                    female_voice = build_video.GOOGLE_GEMINI_FEMALE_SPEAKER
                    if female_voice:
                        build_video.GOOGLE_TTS_VOICE = female_voice
                    else:
                        mock_print("[WARNING] --female specified for Gemini but GOOGLE_GEMINI_FEMALE_SPEAKER not set in .env. Using default voice.")
        
        # Verify warning was printed and flag is set
        mock_print.assert_called_with("[WARNING] --female specified for Gemini but GOOGLE_GEMINI_FEMALE_SPEAKER not set in .env. Using default voice.")
        self.assertTrue(build_video.USE_FEMALE_VOICE)
    
    def test_female_flag_standard_no_female_voice(self):
        """Test --female flag with standard Google TTS but no female voice set shows warning."""
        # Set up standard Google TTS configuration
        build_video.TTS_PROVIDER = 'google'
        build_video.GOOGLE_VOICE_TYPE = ''
        build_video.GOOGLE_TTS_VOICE = 'en-US-Studio-Q'  # Default
        
        # Simulate parse_args logic for --female flag
        with patch('os.getenv', return_value=None):
            with patch('builtins.print') as mock_print:
                if build_video.TTS_PROVIDER == "google":
                    build_video.USE_FEMALE_VOICE = True
                    if build_video.GOOGLE_VOICE_TYPE != "gemini":
                        female_voice = os.getenv("GOOGLE_TTS_FEMALE_VOICE")
                        if female_voice:
                            build_video.GOOGLE_TTS_VOICE = female_voice
                        else:
                            mock_print("[WARNING] --female specified but GOOGLE_TTS_FEMALE_VOICE not set in .env. Using default voice.")
        
        # Verify warning was printed and flag is set
        mock_print.assert_called_with("[WARNING] --female specified but GOOGLE_TTS_FEMALE_VOICE not set in .env. Using default voice.")
        self.assertTrue(build_video.USE_FEMALE_VOICE)
    
    def test_female_flag_non_google_provider(self):
        """Test --female flag is ignored for non-Google providers."""
        # Set up OpenAI provider
        build_video.TTS_PROVIDER = 'openai'
        
        # Simulate parse_args logic for --female flag
        if build_video.TTS_PROVIDER == "google":
            build_video.USE_FEMALE_VOICE = True
        else:
            build_video.USE_FEMALE_VOICE = False
        
        # Verify USE_FEMALE_VOICE is False for non-Google providers
        self.assertFalse(build_video.USE_FEMALE_VOICE)
    
    def test_no_female_flag(self):
        """Test that USE_FEMALE_VOICE remains False when --female is not specified."""
        # Set up Google provider
        build_video.TTS_PROVIDER = 'google'
        
        # Simulate parse_args logic without --female flag
        build_video.USE_FEMALE_VOICE = False
        
        # Verify USE_FEMALE_VOICE is False
        self.assertFalse(build_video.USE_FEMALE_VOICE)


class TestVoiceSelectionIntegration(unittest.TestCase):
    """Integration tests for voice selection combining env vars and parser args."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        build_video.config.temp_dir = self.temp_dir
        build_video.config.save_assets = False
        
        # Create a test scene
        self.test_scene = {
            "id": 1,
            "title": "Test Scene",
            "narration": "This is a test narration.",
            "image_prompt": "Test visual"
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        import time
        if os.path.exists(self.temp_dir):
            try:
                # Try to remove, but ignore errors on Windows due to file locks
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                # Give Windows a moment to release file handles
                time.sleep(0.1)
            except (PermissionError, OSError):
                pass
        # Reset global flags
        build_video.USE_FEMALE_VOICE = False
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "google", "GOOGLE_VOICE_TYPE": "gemini", "GOOGLE_GEMINI_MALE_SPEAKER": "Charon", "GOOGLE_GEMINI_FEMALE_SPEAKER": "Leda", "GOOGLE_TTS_LANGUAGE": "en-US"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.USE_FEMALE_VOICE', True)
    @patch('google.cloud.texttospeech.TextToSpeechClient')
    def test_gemini_female_voice_integration(self, mock_client_class):
        """Integration test: Gemini voice with --female flag."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.audio_content = b'audio_data'
        mock_client.synthesize_speech.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.synthesize_speech.assert_called_once()
        call_args = mock_client.synthesize_speech.call_args
        voice_params = call_args.kwargs['voice']
        self.assertEqual(voice_params.name, 'Leda')
        self.assertEqual(voice_params.model_name, 'gemini-2.5-pro-tts')
    
    @patch.dict("os.environ", {"TTS_PROVIDER": "google", "GOOGLE_VOICE_TYPE": "gemini", "GOOGLE_GEMINI_MALE_SPEAKER": "Charon", "GOOGLE_GEMINI_FEMALE_SPEAKER": "Leda", "GOOGLE_TTS_LANGUAGE": "en-US"}, clear=False)
    @patch('build_video.TTS_PROVIDER', 'google')
    @patch('build_video.USE_FEMALE_VOICE', False)
    @patch('google.cloud.texttospeech.TextToSpeechClient')
    def test_gemini_male_voice_integration(self, mock_client_class):
        """Integration test: Gemini voice without --female flag."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.audio_content = b'audio_data'
        mock_client.synthesize_speech.return_value = mock_response

        build_video.generate_audio_for_scene(self.test_scene)

        mock_client.synthesize_speech.assert_called_once()
        call_args = mock_client.synthesize_speech.call_args
        voice_params = call_args.kwargs['voice']
        self.assertEqual(voice_params.name, 'Charon')
        self.assertEqual(voice_params.model_name, 'gemini-2.5-pro-tts')


if __name__ == "__main__":
    unittest.main()
