"""
Unified LLM utilities for text, image, and speech generation.
Dispatches to OpenAI, xAI (Grok), or Google (Gemini) based on .env TEXT_PROVIDER / IMAGE_PROVIDER.
TTS dispatches to OpenAI, ElevenLabs, or Google based on TTS_PROVIDER.

.env variables (defaults preserve OpenAI-only behavior):
  TEXT_PROVIDER      - "openai", "xai", or "google" (default: openai)
  TEXT_MODEL_OPENAI  - OpenAI chat model (default: gpt-5.2)
  TEXT_MODEL_XAI     - xAI Grok model (default: grok-4-1-fast-reasoning)
  TEXT_MODEL_GOOGLE  - Gemini model (default: gemini-2.0-flash)
  IMAGE_PROVIDER     - "openai", "xai", or "google" (default: openai)
  IMAGE_MODEL_OPENAI - OpenAI image model (default: gpt-image-1.5)
  IMAGE_MODEL_XAI    - xAI image model (default: grok-imagine-image)
  IMAGE_MODEL_GOOGLE - Google image-capable model (default: gemini-2.0-flash-exp)
  TTS_PROVIDER       - "openai", "elevenlabs", or "google" (default: google)
  OPENAI_API_KEY     - Required for OpenAI text/images/TTS
  XAI_API_KEY        - Required for xAI (Grok) text/images
  GOOGLE_API_KEY     - Required for Google (Gemini) text/images (GEMINI_API_KEY also supported)
  GOOGLE_APPLICATION_CREDENTIALS - Required for Google Cloud TTS (service account JSON)
  ELEVENLABS_API_KEY - Required for ElevenLabs TTS
"""

import os
import base64
import time
from pathlib import Path
from typing import Any

# Retry config for transient API errors (500, 502, 503, 429, etc.)
_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
_LLM_BASE_DELAY = float(os.getenv("LLM_BASE_DELAY", "2.0"))

from dotenv import load_dotenv

load_dotenv()

# Provider and model from env (defaults: openai for backward compatibility)
TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "openai").lower()
TEXT_MODEL_OPENAI = os.getenv("TEXT_MODEL_OPENAI", "gpt-5.2")
TEXT_MODEL_XAI = os.getenv("TEXT_MODEL_XAI", "grok-4-1-fast-reasoning")
TEXT_MODEL_GOOGLE = os.getenv("TEXT_MODEL_GOOGLE", "gemini-3.1-pro-preview")
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "openai").lower()
IMAGE_MODEL_OPENAI = os.getenv("IMAGE_MODEL_OPENAI", "gpt-image-1.5")
IMAGE_MODEL_XAI = os.getenv("IMAGE_MODEL_XAI", "grok-imagine-image")
IMAGE_MODEL_GOOGLE = os.getenv("IMAGE_MODEL_GOOGLE", "gemini-2.0-flash-exp")

# TTS provider and settings
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "google").lower()
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts-2025-12-15")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "marin")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
GOOGLE_TTS_VOICE = os.getenv("GOOGLE_TTS_VOICE", "en-US-Studio-Q")
GOOGLE_TTS_LANGUAGE = os.getenv("GOOGLE_TTS_LANGUAGE", "en-US")
GOOGLE_TTS_PITCH = float(os.getenv("GOOGLE_TTS_PITCH", "-1.0"))
GOOGLE_VOICE_TYPE = os.getenv("GOOGLE_VOICE_TYPE", "").lower()
GOOGLE_GEMINI_MALE_SPEAKER = os.getenv("GOOGLE_GEMINI_MALE_SPEAKER", "Charon")
GOOGLE_GEMINI_FEMALE_SPEAKER = os.getenv("GOOGLE_GEMINI_FEMALE_SPEAKER", "")
GEMINI_TTS_TEXT_PADDING = "   "
OPENAI_TTS_STYLE_PROMPT = (
    "Read the following text in a calm, neutral, and consistent tone. "
    "Maintain stable volume and pitch throughout. Do not add emotional "
    "inflection, dramatic emphasis, or noticeable changes in speaking speed. "
)

# TTS text preprocessing: normalize chars that cause cutouts/artifacts (default on)
TTS_PREPROCESS_TEXT = os.getenv("TTS_PREPROCESS_TEXT", "1").lower() in ("1", "true", "yes")
TTS_OPENAI_STREAM = os.getenv("TTS_OPENAI_STREAM", "0").lower() in ("1", "true", "yes")


def _sanitize_text_for_tts(text: str) -> str:
    """
    Normalize problematic characters before TTS to reduce cutouts and artifacts.
    Em dashes and curly quotes can cause cutouts or mispronunciation.
    """
    if not text or not isinstance(text, str):
        return text or ""
    s = text
    # Em dash (U+2014) and en dash (U+2013) -> hyphen with spaces
    s = s.replace("\u2014", " - ")
    s = s.replace("\u2013", " - ")
    # Curly/smart quotes -> straight quotes
    s = s.replace("\u2018", "'")  # left single
    s = s.replace("\u2019", "'")  # right single
    s = s.replace("\u201c", '"')  # left double
    s = s.replace("\u201d", '"')  # right double
    return s.strip()


def get_text_model_display() -> str:
    """Return a short string for logging: provider / model (e.g. 'openai / gpt-5.2')."""
    prov = TEXT_PROVIDER.lower()
    if prov == "openai":
        model = TEXT_MODEL_OPENAI
    elif prov == "xai":
        model = TEXT_MODEL_XAI
    else:
        model = TEXT_MODEL_GOOGLE
    return f"{prov} / {model}"


def get_provider_for_step(step: str) -> str:
    """Return provider for pipeline step. Uses TEXT_PROVIDER_<STEP> if set, else TEXT_PROVIDER."""
    val = os.getenv(f"TEXT_PROVIDER_{step.upper()}")
    return (val or TEXT_PROVIDER).lower()


def _resolve_json_schema(schema: dict | type) -> dict:
    """Resolve a JSON schema from a Pydantic model or dict."""
    if isinstance(schema, dict):
        return schema
    # Pydantic BaseModel
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    raise TypeError("response_json_schema must be a dict or Pydantic BaseModel")


def _ensure_openai_schema(schema: dict) -> dict:
    """Ensure schema has additionalProperties: false for OpenAI Structured Outputs."""
    if schema.get("type") != "object":
        return schema
    result = dict(schema)
    if "additionalProperties" not in result:
        result["additionalProperties"] = False
    if "properties" in result:
        result["properties"] = {
            k: _ensure_openai_schema(v) if isinstance(v, dict) else v
            for k, v in result["properties"].items()
        }
    if "items" in result and isinstance(result["items"], dict):
        result["items"] = _ensure_openai_schema(result["items"])
    if "$defs" in result:
        result["$defs"] = {
            k: _ensure_openai_schema(v) if isinstance(v, dict) else v
            for k, v in result["$defs"].items()
        }
    return result


def _xai_text(
    messages: list[dict[str, str]],
    model: str | None,
    temperature: float,
    schema_dict: dict | None,
    response_format: dict | None,
    **kwargs: Any,
) -> str:
    """Generate text using xAI (Grok) via OpenAI-compatible API."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY is not set. Set it in .env for xAI (Grok) text.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    model_name = model or TEXT_MODEL_XAI
    req: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        **kwargs,
    }
    if schema_dict is not None:
        openai_schema = _ensure_openai_schema(schema_dict)
        schema_name = openai_schema.get("title", "response")
        req["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": openai_schema,
            },
        }
    elif response_format is not None:
        req["response_format"] = response_format
    last_error = None
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**req)
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            retryable = status in (429, 500, 502, 503) or "connection" in str(type(e).__name__).lower()
            if not retryable or attempt >= _LLM_MAX_RETRIES:
                raise
            delay = _LLM_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[LLM] Attempt {attempt}/{_LLM_MAX_RETRIES} failed ({e}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("xAI text generation failed.")


def generate_text(
    messages: list[dict[str, str]],
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.7,
    response_format: dict | None = None,
    response_json_schema: dict | type | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate text from messages using OpenAI, xAI (Grok), or Google Gemini.

    Args:
        messages: List of {"role": "user"|"system"|"assistant", "content": str} (OpenAI shape).
        model: Model name; if None, use env TEXT_MODEL_OPENAI, TEXT_MODEL_XAI, or TEXT_MODEL_GOOGLE.
        provider: "openai", "xai", or "google"; if None, use env TEXT_PROVIDER.
        temperature: Sampling temperature.
        response_format: Optional e.g. {"type": "json_object"} for JSON mode.
        response_json_schema: Optional JSON schema (dict or Pydantic BaseModel) for structured
            output. When provided, the schema is passed in the API config (not the prompt).
            For Google: uses response_json_schema in config. For OpenAI: uses response_format
            with type json_schema. Takes precedence over response_format for structured output.
        **kwargs: Passed through to the underlying API.

    Returns:
        The assistant reply as a single string.
    """
    prov = (provider or TEXT_PROVIDER).lower()
    if prov not in ("openai", "xai", "google"):
        raise ValueError(
            f"TEXT_PROVIDER must be 'openai', 'xai', or 'google'. Got: {prov}. "
            "Set TEXT_PROVIDER in .env or pass provider=."
        )

    # Resolve schema if provided
    schema_dict: dict | None = None
    if response_json_schema is not None:
        schema_dict = _resolve_json_schema(response_json_schema)

    if prov == "xai":
        return _xai_text(
            messages=messages,
            model=model,
            temperature=temperature,
            schema_dict=schema_dict,
            response_format=response_format,
            **kwargs,
        )

    if prov == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Set it in .env for OpenAI text.")
        from openai import OpenAI
        client = OpenAI()
        model_name = model or TEXT_MODEL_OPENAI
        req: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if schema_dict is not None:
            # Structured Outputs: pass schema in response_format
            openai_schema = _ensure_openai_schema(schema_dict)
            schema_name = openai_schema.get("title", "response")
            req["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": openai_schema,
                },
            }
        elif response_format is not None:
            req["response_format"] = response_format
        last_error = None
        for attempt in range(1, _LLM_MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(**req)
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                retryable = status in (429, 500, 502, 503) or "connection" in str(type(e).__name__).lower()
                if not retryable or attempt >= _LLM_MAX_RETRIES:
                    raise
                delay = _LLM_BASE_DELAY * (2 ** (attempt - 1))
                print(f"[LLM] Attempt {attempt}/{_LLM_MAX_RETRIES} failed ({e}). Retrying in {delay:.1f}s...")
                time.sleep(delay)
        raise last_error

    # Google Gemini (google.genai SDK)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is not set. Set one in .env for Google (Gemini) text. "
            "You can create an API key in Google AI Studio."
        )
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    model_name = model or TEXT_MODEL_GOOGLE
    system_parts: list[str] = []
    chat_parts: list[tuple[str, str]] = []  # (role, content)
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        else:
            chat_parts.append(("user" if role == "user" else "model", content))
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    # Only add "Respond with valid JSON only" when using json_object mode (no schema)
    if schema_dict is None and response_format == {"type": "json_object"} and system_instruction:
        system_instruction = system_instruction.rstrip() + "\n\nRespond with valid JSON only."
    elif schema_dict is None and response_format == {"type": "json_object"}:
        system_instruction = "Respond with valid JSON only."
    config_kw: dict[str, Any] = {"temperature": temperature}
    if schema_dict is not None:
        config_kw["response_mime_type"] = "application/json"
        config_kw["response_json_schema"] = schema_dict
    elif response_format == {"type": "json_object"}:
        config_kw["response_mime_type"] = "application/json"
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            **config_kw,
        )
    except Exception:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
    # Single user turn: one contents string; multi-turn: fold history into one prompt
    if len(chat_parts) <= 1 and (not chat_parts or chat_parts[0][0] == "user"):
        contents = chat_parts[0][1] if chat_parts else ""
    else:
        parts_str = []
        for role, content in chat_parts:
            label = "User" if role == "user" else "Assistant"
            parts_str.append(f"{label}: {content}")
        contents = "\n\n".join(parts_str)
    from google.genai import errors as genai_errors
    response = None
    last_error = None
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            break
        except genai_errors.ServerError as e:
            last_error = e
            if attempt >= _LLM_MAX_RETRIES:
                raise
            delay = _LLM_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[LLM] Attempt {attempt}/{_LLM_MAX_RETRIES} failed (500/server error). Retrying in {delay:.1f}s...")
            time.sleep(delay)
        except Exception as e:
            # Retry on other transient-looking errors (e.g. 429, 503 from different wrapper)
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status in (429, 502, 503):
                last_error = e
                if attempt >= _LLM_MAX_RETRIES:
                    raise
                delay = _LLM_BASE_DELAY * (2 ** (attempt - 1))
                print(f"[LLM] Attempt {attempt}/{_LLM_MAX_RETRIES} failed ({e}). Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
    else:
        if last_error is not None:
            raise last_error
    if not response:
        raise RuntimeError("Google Gemini returned no response.")
    text = getattr(response, "text", None) or ""
    if not text and getattr(response, "candidates", None) and response.candidates:
        c0 = response.candidates[0]
        if getattr(c0, "content", None) and getattr(c0.content, "parts", None) and c0.content.parts:
            part = c0.content.parts[0]
            text = getattr(part, "text", None) or ""
    if not text:
        raise RuntimeError("Google Gemini returned empty text. The model may have blocked the response.")
    return text


def _size_to_aspect_ratio(size: str) -> str:
    """Map pixel size to xAI aspect_ratio."""
    mapping = {
        "1536x1024": "16:9",
        "1024x1536": "9:16",
        "1024x1024": "1:1",
    }
    return mapping.get(size, "16:9")


def _xai_image(
    prompt: str,
    output_path: Path | None,
    size: str,
    model_name: str,
    output_format: str,
    **kwargs: Any,
) -> bytes | Path:
    """Generate image using xAI (Grok) via xai_sdk."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY is not set. Set it in .env for xAI (Grok) images.")
    from xai_sdk import Client
    client = Client(api_key=api_key)
    aspect_ratio = _size_to_aspect_ratio(size)
    response = client.image.sample(
        prompt=prompt,
        model=model_name,
        aspect_ratio=aspect_ratio,
        image_format="base64",
    )
    img_bytes = getattr(response, "image", None)
    if not img_bytes:
        raise RuntimeError("xAI image response had no image data")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(img_bytes)
        return output_path
    return img_bytes


def _openai_image(
    prompt: str,
    output_path: Path | None,
    size: str,
    model_name: str,
    output_format: str,
    **kwargs: Any,
) -> bytes | Path:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Set it in .env for OpenAI images.")
    client = OpenAI()
    # response_format is only supported for dall-e-2 and dall-e-3; GPT image models
    # (gpt-image-1, gpt-image-1-mini, gpt-image-1.5) always return base64 and reject it.
    is_dalle = model_name and model_name.lower().startswith("dall-e-")
    req_kwargs: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "size": size,
        "n": 1,
        **kwargs,
    }
    if is_dalle:
        req_kwargs["response_format"] = "b64_json"
    resp = client.images.generate(**req_kwargs)
    b64_data = getattr(resp.data[0], "b64_json", None) or (resp.data[0].model_dump().get("b64_json"))
    if not b64_data:
        raise RuntimeError("OpenAI image response had no b64_json")
    img_bytes = base64.b64decode(b64_data)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(img_bytes)
        return output_path
    return img_bytes


def _google_image(
    prompt: str,
    output_path: Path | None,
    size: str,
    model_name: str,
    output_format: str,
    **kwargs: Any,
) -> bytes | Path:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is not set. Set one in .env for Google (Gemini) images. "
            "You can create an API key in Google AI Studio."
        )
    from google import genai
    client = genai.Client(api_key=api_key)
    # Image-capable Gemini model (e.g. gemini-2.0-flash-exp) returns image in response.
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    img_bytes = None
    if getattr(response, "candidates", None) and response.candidates:
        c0 = response.candidates[0]
        if getattr(c0, "content", None) and getattr(c0.content, "parts", None):
            for part in c0.content.parts:
                if getattr(part, "inline_data", None) and part.inline_data:
                    img_bytes = getattr(part.inline_data, "data", None) or getattr(
                        part.inline_data, "image_data", None
                    )
                    if img_bytes is None and hasattr(part.inline_data, "data"):
                        img_bytes = part.inline_data.data
                    break
    if not img_bytes:
        raise RuntimeError(
            "Google image model did not return image data. "
            "Set IMAGE_MODEL_GOOGLE to an image-capable model (e.g. gemini-2.0-flash-exp) or use IMAGE_PROVIDER=openai."
        )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(img_bytes)
        return output_path
    return img_bytes


def generate_image(
    prompt: str,
    output_path: Path | str | None = None,
    size: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    output_format: str = "png",
    **kwargs: Any,
) -> bytes | Path:
    """
    Generate an image from a text prompt using OpenAI, xAI (Grok), or Google.

    Args:
        prompt: The image description.
        output_path: If set, write image here and return Path; else return bytes.
        size: e.g. "1536x1024", "1024x1024"; provider-specific (xAI maps to aspect_ratio).
        model: Model name; if None, use env IMAGE_MODEL_OPENAI, IMAGE_MODEL_XAI, or IMAGE_MODEL_GOOGLE.
        provider: "openai", "xai", or "google"; if None, use env IMAGE_PROVIDER.
        output_format: "png" or "jpeg"; used for file extension / OpenAI response_format when applicable.
        **kwargs: Passed through (e.g. moderation="low" for OpenAI).

    Returns:
        Path if output_path was set, else raw image bytes.
    """
    kwargs.setdefault("moderation", "low")
    prov = (provider or IMAGE_PROVIDER).lower()
    if prov not in ("openai", "xai", "google"):
        raise ValueError(
            f"IMAGE_PROVIDER must be 'openai', 'xai', or 'google'. Got: {prov}. "
            "Set IMAGE_PROVIDER in .env or pass provider=."
        )
    size = size or "1024x1024"
    if prov == "openai":
        model_name = model or IMAGE_MODEL_OPENAI
        return _openai_image(
            prompt=prompt,
            output_path=Path(output_path) if output_path else None,
            size=size,
            model_name=model_name,
            output_format=output_format,
            **kwargs,
        )
    if prov == "xai":
        model_name = model or IMAGE_MODEL_XAI
        return _xai_image(
            prompt=prompt,
            output_path=Path(output_path) if output_path else None,
            size=size,
            model_name=model_name,
            output_format=output_format,
            **kwargs,
        )
    model_name = model or IMAGE_MODEL_GOOGLE
    return _google_image(
        prompt=prompt,
        output_path=Path(output_path) if output_path else None,
        size=size,
        model_name=model_name,
        output_format=output_format,
        **kwargs,
    )


def generate_speech(
    text: str,
    output_path: Path | str,
    emotion: str | None = None,
    narration_instructions: str | None = None,
    use_female_voice: bool = False,
    provider: str | None = None,
    previous_text: str | None = None,
    next_text: str | None = None,
) -> Path:
    """
    Generate speech from text using OpenAI, ElevenLabs, or Google TTS.

    Args:
        text: The text to synthesize.
        output_path: Path to write the audio file (MP3).
        emotion: Optional emotion string for Gemini/OpenAI style hints.
        narration_instructions: Optional instructions (e.g. biopic narration style) for Gemini/OpenAI.
        use_female_voice: If True, use female voice for Gemini.
        provider: "openai", "elevenlabs", or "google"; if None, use env TTS_PROVIDER.
        previous_text: Optional text before this segment (ElevenLabs only, improves continuity).
        next_text: Optional text after this segment (ElevenLabs only, improves continuity).

    Returns:
        Path to the generated audio file.
    """
    prov = (provider or os.getenv("TTS_PROVIDER", "google")).lower()
    if prov not in ("openai", "elevenlabs", "google"):
        raise ValueError(
            f"TTS_PROVIDER must be 'openai', 'elevenlabs', or 'google'. Got: {prov}. "
            "Set TTS_PROVIDER in .env or pass provider=."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if TTS_PREPROCESS_TEXT:
        text = _sanitize_text_for_tts(text)
        if previous_text is not None:
            previous_text = _sanitize_text_for_tts(previous_text)
        if next_text is not None:
            next_text = _sanitize_text_for_tts(next_text)

    last_error = None
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            if prov == "openai":
                return _openai_tts(
                    text=text,
                    output_path=output_path,
                    narration_instructions=narration_instructions,
                )
            if prov == "elevenlabs":
                return _elevenlabs_tts(
                    text=text,
                    output_path=output_path,
                    previous_text=previous_text,
                    next_text=next_text,
                )
            return _google_tts(
                text=text,
                output_path=output_path,
                emotion=emotion,
                narration_instructions=narration_instructions,
                use_female_voice=use_female_voice,
            )
        except Exception as e:
            last_error = e
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            retryable = status in (429, 500, 502, 503) or "connection" in str(type(e).__name__).lower()
            if not retryable or attempt >= _LLM_MAX_RETRIES:
                raise
            delay = _LLM_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[TTS] Attempt {attempt}/{_LLM_MAX_RETRIES} failed ({e}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("TTS generation failed.")


def _openai_tts(
    text: str,
    output_path: Path,
    narration_instructions: str | None = None,
) -> Path:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Set it in .env for OpenAI TTS.")
    client = OpenAI()
    instructions = narration_instructions or OPENAI_TTS_STYLE_PROMPT
    if TTS_OPENAI_STREAM:
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            instructions=instructions,
        ) as response:
            response.stream_to_file(str(output_path))
    else:
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            instructions=instructions,
        )
        output_path.write_bytes(response.content)
    return output_path


def _elevenlabs_tts(
    text: str,
    output_path: Path,
    previous_text: str | None = None,
    next_text: str | None = None,
) -> Path:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is not set. Set it in .env for ElevenLabs TTS.")
    from elevenlabs import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    kwargs = {
        "voice_id": ELEVENLABS_VOICE_ID,
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "output_format": "mp3_44100_128",
        "optimize_streaming_latency": 0,
        "apply_text_normalization": "on",
        "language_code": "en",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75,
            "style": 0.5,
            "use_speaker_boost": False,
            "speed": 1.0,
        },
    }
    if previous_text is not None:
        kwargs["previous_text"] = previous_text
    if next_text is not None:
        kwargs["next_text"] = next_text
    result = client.text_to_speech.convert(**kwargs)
    # SDK may return bytes or an iterator; handle both for compatibility
    if isinstance(result, bytes):
        output_path.write_bytes(result)
    else:
        chunks = list(result)
        if chunks and isinstance(chunks[0], bytes):
            output_path.write_bytes(b"".join(chunks))
        else:
            output_path.write_bytes(bytes(chunks))
    return output_path


def _google_tts(
    text: str,
    output_path: Path,
    emotion: str | None = None,
    narration_instructions: str | None = None,
    use_female_voice: bool = False,
) -> Path:
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()

    voice_type = os.getenv("GOOGLE_VOICE_TYPE", "").lower()
    tts_voice = os.getenv("GOOGLE_TTS_VOICE", GOOGLE_TTS_VOICE)
    lang = os.getenv("GOOGLE_TTS_LANGUAGE", GOOGLE_TTS_LANGUAGE)
    pitch = float(os.getenv("GOOGLE_TTS_PITCH", str(GOOGLE_TTS_PITCH)))
    gemini_male = os.getenv("GOOGLE_GEMINI_MALE_SPEAKER", GOOGLE_GEMINI_MALE_SPEAKER)
    gemini_female = os.getenv("GOOGLE_GEMINI_FEMALE_SPEAKER", GOOGLE_GEMINI_FEMALE_SPEAKER)

    is_chirp = voice_type == "chirp"
    is_gemini = voice_type == "gemini"

    if is_gemini:
        text_for_gemini = (text or "").rstrip() + GEMINI_TTS_TEXT_PADDING
        synthesis_input = texttospeech.SynthesisInput(text=text_for_gemini)
        if use_female_voice and gemini_female:
            gemini_voice = gemini_female
        elif use_female_voice and not gemini_female:
            gemini_voice = tts_voice  # Fallback when female speaker not set
        else:
            gemini_voice = gemini_male
        if not gemini_voice:
            gemini_voice = tts_voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=gemini_voice,
            model_name="gemini-2.5-pro-tts",
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    elif is_chirp:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=tts_voice,
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    else:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=tts_voice,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            pitch=pitch,
        )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    output_path.write_bytes(response.audio_content)
    return output_path
