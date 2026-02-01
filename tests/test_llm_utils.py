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


class TestGetTextModelDisplay(unittest.TestCase):
    """Test get_text_model_display returns a string with provider and model."""

    def test_returns_string(self):
        out = llm_utils.get_text_model_display()
        self.assertIsInstance(out, str)
        self.assertIn("/", out)
        parts = out.split("/", 1)
        self.assertEqual(len(parts), 2)
        self.assertIn(parts[0].strip().lower(), ("openai", "google"))
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

    def test_invalid_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            llm_utils.generate_text(
                messages=[{"role": "user", "content": "Hi"}],
                provider="invalid",
            )
        self.assertIn("openai", str(ctx.exception).lower())
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

    def test_invalid_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            llm_utils.generate_image("A test", provider="invalid")
        self.assertIn("openai", str(ctx.exception).lower())
        self.assertIn("google", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
