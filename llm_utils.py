"""
Unified LLM utilities for text and image generation.
Dispatches to OpenAI or Google (Gemini) based on .env TEXT_PROVIDER / IMAGE_PROVIDER.

.env variables (defaults preserve OpenAI-only behavior):
  TEXT_PROVIDER      - "openai" or "google" (default: openai)
  TEXT_MODEL_OPENAI  - OpenAI chat model (default: gpt-5.2)
  TEXT_MODEL_GOOGLE  - Gemini model (default: gemini-2.0-flash)
  IMAGE_PROVIDER     - "openai" or "google" (default: openai)
  IMAGE_MODEL_OPENAI - OpenAI image model (default: gpt-image-1.5)
  IMAGE_MODEL_GOOGLE - Google image-capable model (default: gemini-2.0-flash-exp)
  OPENAI_API_KEY     - Required for OpenAI text/images
  GOOGLE_API_KEY     - Required for Google (Gemini) text/images (GEMINI_API_KEY also supported)
"""

import os
import base64
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Provider and model from env (defaults: openai for backward compatibility)
TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "openai").lower()
TEXT_MODEL_OPENAI = os.getenv("TEXT_MODEL_OPENAI", "gpt-5.2")
TEXT_MODEL_GOOGLE = os.getenv("TEXT_MODEL_GOOGLE", "gemini-2.0-flash")
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "openai").lower()
IMAGE_MODEL_OPENAI = os.getenv("IMAGE_MODEL_OPENAI", "gpt-image-1.5")
IMAGE_MODEL_GOOGLE = os.getenv("IMAGE_MODEL_GOOGLE", "gemini-2.0-flash-exp")


def get_text_model_display() -> str:
    """Return a short string for logging: provider / model (e.g. 'openai / gpt-5.2')."""
    prov = TEXT_PROVIDER.lower()
    model = TEXT_MODEL_OPENAI if prov == "openai" else TEXT_MODEL_GOOGLE
    return f"{prov} / {model}"


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
    Generate text from messages using OpenAI or Google Gemini.

    Args:
        messages: List of {"role": "user"|"system"|"assistant", "content": str} (OpenAI shape).
        model: Model name; if None, use env TEXT_MODEL_OPENAI or TEXT_MODEL_GOOGLE.
        provider: "openai" or "google"; if None, use env TEXT_PROVIDER.
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
    if prov not in ("openai", "google"):
        raise ValueError(
            f"TEXT_PROVIDER must be 'openai' or 'google'. Got: {prov}. "
            "Set TEXT_PROVIDER in .env or pass provider=."
        )

    # Resolve schema if provided
    schema_dict: dict | None = None
    if response_json_schema is not None:
        schema_dict = _resolve_json_schema(response_json_schema)

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
        response = client.chat.completions.create(**req)
        return response.choices[0].message.content or ""

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
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
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
    Generate an image from a text prompt using OpenAI or Google.

    Args:
        prompt: The image description.
        output_path: If set, write image here and return Path; else return bytes.
        size: e.g. "1536x1024", "1024x1024"; provider-specific.
        model: Model name; if None, use env IMAGE_MODEL_OPENAI or IMAGE_MODEL_GOOGLE.
        provider: "openai" or "google"; if None, use env IMAGE_PROVIDER.
        output_format: "png" or "jpeg"; used for file extension / OpenAI response_format when applicable.
        **kwargs: Passed through (e.g. moderation="low" for OpenAI).

    Returns:
        Path if output_path was set, else raw image bytes.
    """
    kwargs.setdefault("moderation", "low")
    prov = (provider or IMAGE_PROVIDER).lower()
    if prov not in ("openai", "google"):
        raise ValueError(
            f"IMAGE_PROVIDER must be 'openai' or 'google'. Got: {prov}. "
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
    model_name = model or IMAGE_MODEL_GOOGLE
    return _google_image(
        prompt=prompt,
        output_path=Path(output_path) if output_path else None,
        size=size,
        model_name=model_name,
        output_format=output_format,
        **kwargs,
    )
