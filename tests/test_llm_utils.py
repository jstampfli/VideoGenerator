"""
Minimal tests for llm_utils: generate_text and generate_image return expected shape.
Mocks OpenAI/Google clients so tests do not hit real APIs.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import llm_utils


class TestGetProviderForStep(unittest.TestCase):
    """Test get_provider_for_step returns env override when set, else TEXT_PROVIDER."""

    @patch("llm_utils.os.getenv")
    def test_returns_env_override_when_set(self, mock_getenv):
        mock_getenv.side_effect = lambda k, d=None: "xai" if k == "TEXT_PROVIDER_VIDEO_QUESTIONS" else ("openai" if k == "TEXT_PROVIDER" else d)
        with patch.object(llm_utils, "TEXT_PROVIDER", "openai"):
            self.assertEqual(llm_utils.get_provider_for_step("VIDEO_QUESTIONS"), "xai")

    @patch("llm_utils.os.getenv")
    def test_returns_text_provider_when_override_unset(self, mock_getenv):
        def getenv(k, d=None):
            if k == "TEXT_PROVIDER_VIDEO_QUESTIONS":
                return None  # No override
            if k == "TEXT_PROVIDER":
                return "google"
            return d
        mock_getenv.side_effect = getenv
        with patch.object(llm_utils, "TEXT_PROVIDER", "google"):
            self.assertEqual(llm_utils.get_provider_for_step("VIDEO_QUESTIONS"), "google")

    @patch("llm_utils.os.getenv")
    def test_step_name_uppercased(self, mock_getenv):
        mock_getenv.side_effect = lambda k, d=None: "google" if k == "TEXT_PROVIDER_SCENES" else ("openai" if k == "TEXT_PROVIDER" else d)
        with patch.object(llm_utils, "TEXT_PROVIDER", "openai"):
            self.assertEqual(llm_utils.get_provider_for_step("scenes"), "google")


class TestGetTextModelDisplay(unittest.TestCase):
    """Test get_text_model_display returns a string with provider and model."""

    def test_returns_string(self):
        out = llm_utils.get_text_model_display()
        self.assertIsInstance(out, str)
        self.assertIn("/", out)
        parts = out.split("/", 1)
        self.assertEqual(len(parts), 2)
        self.assertIn(parts[0].strip().lower(), ("openai", "xai", "google"))
        self.assertTrue(len(parts[1].strip()) > 0)


class TestGenerateText(unittest.TestCase):
    """Test generate_text returns a string; mock underlying API."""

    @patch.dict("os.environ", {"TEXT_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_openai_returns_string(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hello, world."
        mock_client.chat.completions.create.return_value = mock_resp

        result = llm_utils.generate_text(
            messages=[{"role": "user", "content": "Hi"}],
            provider="openai",
        )
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hello, world.")

    @patch.dict("os.environ", {"TEXT_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_openai_with_response_json_schema(self, mock_openai_class):
        """When response_json_schema is provided, OpenAI receives json_schema in response_format."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "test"}'
        mock_client.chat.completions.create.return_value = mock_resp

        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        llm_utils.generate_text(
            messages=[{"role": "user", "content": "Hi"}],
            provider="openai",
            response_json_schema=schema,
        )
        call_kw = mock_client.chat.completions.create.call_args[1]
        rf = call_kw.get("response_format")
        self.assertIsNotNone(rf)
        self.assertEqual(rf["type"], "json_schema")
        self.assertIn("json_schema", rf)
        self.assertTrue(rf["json_schema"]["strict"])
        self.assertIn("additionalProperties", rf["json_schema"]["schema"])

    @patch.dict("os.environ", {"TEXT_PROVIDER": "google", "GOOGLE_API_KEY": "test-key"}, clear=False)
    @patch("google.genai.Client")
    def test_google_with_response_json_schema(self, mock_client_class):
        """When response_json_schema is provided, Google receives it in config."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.text = '{"name": "test"}'
        mock_resp.candidates = []
        mock_client.models.generate_content.return_value = mock_resp

        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        llm_utils.generate_text(
            messages=[{"role": "user", "content": "Hi"}],
            provider="google",
            response_json_schema=schema,
        )
        call_kw = mock_client.models.generate_content.call_args[1]
        config = call_kw.get("config")
        self.assertIsNotNone(config)
        self.assertEqual(getattr(config, "response_mime_type", None) or config.get("response_mime_type"), "application/json")
        schema_in_config = getattr(config, "response_json_schema", None) or config.get("response_json_schema")
        self.assertIsNotNone(schema_in_config)

    @patch.dict("os.environ", {"TEXT_PROVIDER": "xai", "XAI_API_KEY": "xai-test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_xai_returns_string(self, mock_openai_class):
        """Test xAI (Grok) text generation via OpenAI-compatible API."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Grok says hello."
        mock_client.chat.completions.create.return_value = mock_resp

        result = llm_utils.generate_text(
            messages=[{"role": "user", "content": "Hi"}],
            provider="xai",
        )
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Grok says hello.")
        mock_openai_class.assert_called_once_with(
            api_key="xai-test-key",
            base_url="https://api.x.ai/v1",
        )
        call_kw = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kw["model"], llm_utils.TEXT_MODEL_XAI)

    def test_invalid_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            llm_utils.generate_text(
                messages=[{"role": "user", "content": "Hi"}],
                provider="invalid",
            )
        self.assertIn("openai", str(ctx.exception).lower())
        self.assertIn("xai", str(ctx.exception).lower())
        self.assertIn("google", str(ctx.exception).lower())


class TestGenerateImage(unittest.TestCase):
    """Test generate_image returns bytes or Path; mock underlying API."""

    @patch.dict("os.environ", {"IMAGE_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_openai_returns_bytes_when_no_output_path(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_resp = MagicMock()
        import base64
        mock_resp.data = [MagicMock()]
        mock_resp.data[0].b64_json = base64.b64encode(b"\x89PNG\r\n\x1a\n")
        mock_resp.data[0].model_dump.return_value = {}
        mock_client.images.generate.return_value = mock_resp

        result = llm_utils.generate_image(
            prompt="A test image",
            provider="openai",
        )
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    @patch.dict("os.environ", {"IMAGE_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_openai_returns_path_when_output_path_set(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_resp = MagicMock()
        import base64
        mock_resp.data = [MagicMock()]
        mock_resp.data[0].b64_json = base64.b64encode(b"\x89PNG\r\n\x1a\n")
        mock_resp.data[0].model_dump.return_value = {}
        mock_client.images.generate.return_value = mock_resp

        out = Path(__file__).parent / "tmp_llm_utils_test_image.png"
        try:
            result = llm_utils.generate_image(
                prompt="A test image",
                output_path=out,
                provider="openai",
            )
            self.assertIsInstance(result, Path)
            self.assertEqual(result, out)
            self.assertTrue(out.is_file())
        finally:
            if out.exists():
                out.unlink(missing_ok=True)

    @patch.dict("os.environ", {"IMAGE_PROVIDER": "xai", "XAI_API_KEY": "xai-test-key"}, clear=False)
    @patch("xai_sdk.Client")
    def test_xai_returns_path(self, mock_client_class):
        """Test xAI (Grok) image generation via xai_sdk."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.image = b"\x89PNG\r\n\x1a\n"
        mock_client.image.sample.return_value = mock_response

        out = Path(__file__).parent / "tmp_llm_utils_test_xai_image.png"
        try:
            result = llm_utils.generate_image(
                prompt="A test image",
                output_path=out,
                provider="xai",
                size="1536x1024",
            )
            self.assertIsInstance(result, Path)
            self.assertEqual(result, out)
            self.assertTrue(out.is_file())
            mock_client.image.sample.assert_called_once()
            call_kw = mock_client.image.sample.call_args[1]
            self.assertEqual(call_kw["model"], llm_utils.IMAGE_MODEL_XAI)
            self.assertEqual(call_kw["aspect_ratio"], "16:9")
            self.assertEqual(call_kw["image_format"], "base64")
        finally:
            if out.exists():
                out.unlink(missing_ok=True)

    def test_invalid_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            llm_utils.generate_image("A test", provider="invalid")
        self.assertIn("openai", str(ctx.exception).lower())
        self.assertIn("xai", str(ctx.exception).lower())
        self.assertIn("google", str(ctx.exception).lower())


class TestGenerateSpeech(unittest.TestCase):
    """Test generate_speech returns Path; mock underlying TTS APIs."""

    def setUp(self):
        self.temp_dir = Path(__file__).parent / "tmp_tts_test"
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.temp_dir.exists():
            for f in self.temp_dir.glob("*.mp3"):
                f.unlink(missing_ok=True)
            try:
                self.temp_dir.rmdir()
            except OSError:
                pass

    @patch.dict("os.environ", {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_openai_returns_path(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_stream = MagicMock()
        mock_client.audio.speech.with_streaming_response.create.return_value.__enter__.return_value = mock_stream

        out = self.temp_dir / "test_openai.mp3"
        result = llm_utils.generate_speech(
            text="Hello world",
            output_path=out,
            provider="openai",
        )
        self.assertIsInstance(result, Path)
        self.assertEqual(result, out)
        mock_client.audio.speech.with_streaming_response.create.assert_called_once()
        call_kw = mock_client.audio.speech.with_streaming_response.create.call_args[1]
        self.assertEqual(call_kw["input"], "Hello world")
        self.assertEqual(call_kw["voice"], llm_utils.OPENAI_TTS_VOICE)

    @patch.dict("os.environ", {"TTS_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "test-key"}, clear=False)
    @patch("elevenlabs.ElevenLabs")
    def test_elevenlabs_returns_path(self, mock_elevenlabs_class):
        mock_client = MagicMock()
        mock_elevenlabs_class.return_value = mock_client
        mock_client.text_to_speech.convert.return_value = iter([b"chunk1", b"chunk2"])

        out = self.temp_dir / "test_elevenlabs.mp3"
        result = llm_utils.generate_speech(
            text="Hello world",
            output_path=out,
            provider="elevenlabs",
        )
        self.assertIsInstance(result, Path)
        self.assertEqual(result, out)
        mock_client.text_to_speech.convert.assert_called_once()
        call_kw = mock_client.text_to_speech.convert.call_args[1]
        self.assertEqual(call_kw["text"], "Hello world")
        self.assertEqual(call_kw["voice_id"], llm_utils.ELEVENLABS_VOICE_ID)

    @patch.dict(
        "os.environ",
        {
            "TTS_PROVIDER": "google",
            "GOOGLE_VOICE_TYPE": "chirp",
            "GOOGLE_TTS_VOICE": "en-US-Studio-Q",
            "GOOGLE_TTS_LANGUAGE": "en-US",
        },
        clear=False,
    )
    @patch("google.cloud.texttospeech.TextToSpeechClient")
    def test_google_chirp_returns_path(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.audio_content = b"audio_data"
        mock_client.synthesize_speech.return_value = mock_resp

        out = self.temp_dir / "test_google.mp3"
        result = llm_utils.generate_speech(
            text="Hello world",
            output_path=out,
            provider="google",
        )
        self.assertIsInstance(result, Path)
        self.assertEqual(result, out)
        self.assertTrue(out.is_file())
        mock_client.synthesize_speech.assert_called_once()
        call_kw = mock_client.synthesize_speech.call_args[1]
        self.assertEqual(call_kw["input"].text, "Hello world")

    @patch.dict(
        "os.environ",
        {
            "TTS_PROVIDER": "google",
            "GOOGLE_VOICE_TYPE": "gemini",
            "GOOGLE_GEMINI_MALE_SPEAKER": "Charon",
            "GOOGLE_TTS_LANGUAGE": "en-US",
        },
        clear=False,
    )
    @patch("google.cloud.texttospeech.TextToSpeechClient")
    def test_google_gemini_returns_path(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.audio_content = b"audio_data"
        mock_client.synthesize_speech.return_value = mock_resp

        out = self.temp_dir / "test_gemini.mp3"
        result = llm_utils.generate_speech(
            text="Hello world",
            output_path=out,
            provider="google",
            use_female_voice=False,
        )
        self.assertIsInstance(result, Path)
        self.assertEqual(result, out)
        call_kw = mock_client.synthesize_speech.call_args[1]
        voice = call_kw["voice"]
        self.assertEqual(voice.name, "Charon")
        self.assertEqual(voice.model_name, "gemini-2.5-pro-tts")

    def test_invalid_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            llm_utils.generate_speech("Hello", "/tmp/out.mp3", provider="invalid")
        self.assertIn("openai", str(ctx.exception).lower())
        self.assertIn("elevenlabs", str(ctx.exception).lower())
        self.assertIn("google", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
