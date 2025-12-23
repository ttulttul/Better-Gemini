from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_GEMINI_INT32_MAX = 2**31 - 1
_warned_seed_mapping = False


class BetterGeminiError(RuntimeError):
    pass


class BetterGeminiConfigError(ValueError):
    pass


SUPPORTED_ASPECT_RATIOS: tuple[str, ...] = (
    "auto",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)


@dataclass(frozen=True)
class BetterGeminiRequest:
    model: str
    prompt: str
    response_modalities: tuple[str, ...]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    seed: int | None = None
    thinking_budget: int | None = None
    image_aspect_ratio: str | None = None
    image_resolution: str | None = None
    image_width: int | None = None
    image_height: int | None = None


def normalize_seed(seed: int) -> int | None:
    """
    Normalize a ComfyUI-style seed into the int32 range expected by Gemini.

    The Gemini API expects `generation_config.seed` to be a protobuf int32. ComfyUI
    commonly uses 64-bit seeds, so we deterministically fold larger values into
    the supported range.
    """

    global _warned_seed_mapping

    if not seed or seed < 0:
        return None
    if seed <= _GEMINI_INT32_MAX:
        return seed

    normalized = seed % (2**31)
    if normalized == 0:
        normalized = 1

    if not _warned_seed_mapping:
        logger.warning(
            "Seed %s is outside Gemini's int32 range; mapping to %s (seed %% 2**31).",
            seed,
            normalized,
        )
        _warned_seed_mapping = True
    else:
        logger.debug("Seed %s mapped to int32 %s.", seed, normalized)
    return normalized


def thinking_budget_from_difficulty(difficulty: str) -> int | None:
    if not difficulty or difficulty == "auto":
        return None
    mapping = {
        "low": 1024,
        "medium": 4096,
        "high": 8192,
    }
    if difficulty not in mapping:
        raise BetterGeminiConfigError(f"Unsupported thinking difficulty: {difficulty!r}")
    return mapping[difficulty]


def build_request(
    *,
    model: str,
    prompt: str,
    response_modalities: str,
    aspect_ratio: str = "auto",
    resolution: str = "auto",
    width: int = 0,
    height: int = 0,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_output_tokens: int | None = None,
    thinking_difficulty: str = "auto",
    thinking_budget: int = 0,
    seed: int = 0,
) -> BetterGeminiRequest:
    if not prompt or not isinstance(prompt, str):
        raise BetterGeminiConfigError("`prompt` must be a non-empty string.")
    if not model or not isinstance(model, str):
        raise BetterGeminiConfigError("`model` must be a non-empty string.")

    modalities = tuple(m.strip().upper() for m in response_modalities.split("+") if m.strip())
    if not modalities:
        raise BetterGeminiConfigError("`response_modalities` must include at least one modality.")

    ar = aspect_ratio or "auto"
    if ar not in SUPPORTED_ASPECT_RATIOS:
        raise BetterGeminiConfigError(f"Unsupported aspect ratio: {ar!r}")

    resolved_thinking_budget = thinking_budget if thinking_budget and thinking_budget > 0 else None
    if resolved_thinking_budget is None:
        resolved_thinking_budget = thinking_budget_from_difficulty(thinking_difficulty)

    image_width = width if width and width > 0 else None
    image_height = height if height and height > 0 else None
    if (image_width is None) != (image_height is None):
        raise BetterGeminiConfigError("`width` and `height` must be set together (both > 0), or both left as 0.")

    normalized_seed = normalize_seed(seed)

    return BetterGeminiRequest(
        model=model.strip(),
        prompt=prompt,
        response_modalities=modalities,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        seed=normalized_seed,
        thinking_budget=resolved_thinking_budget,
        image_aspect_ratio=ar if ar != "auto" else None,
        image_resolution=resolution if resolution and resolution != "auto" else None,
        image_width=image_width,
        image_height=image_height,
    )


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iter_parts(response: Any) -> Iterable[Any]:
    candidates = _get_attr(response, "candidates")
    if not candidates:
        return []
    first_candidate = candidates[0]
    content = _get_attr(first_candidate, "content")
    parts = _get_attr(content, "parts")
    return parts or []


def _extract_inline_data(part: Any) -> tuple[str | None, Any | None]:
    inline = _get_attr(part, "inline_data", None)
    if inline is None:
        inline = _get_attr(part, "inlineData", None)
    if inline is None:
        return None, None

    mime_type = _get_attr(inline, "mime_type", None)
    if mime_type is None:
        mime_type = _get_attr(inline, "mimeType", None)
    data = _get_attr(inline, "data", None)
    return mime_type, data


def _decode_maybe_base64(data: Any) -> bytes | None:
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if not isinstance(data, str):
        return None

    s = data.strip()
    if s.startswith("data:") and ";base64," in s:
        s = s.split(";base64,", 1)[1]
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        logger.debug("Failed to base64-decode inline data payload.", exc_info=True)
        return None


def extract_text_and_images(response: Any) -> tuple[str, list[bytes]]:
    """
    Extract text and image bytes from a google-genai GenerateContent response.

    This is intentionally dependency-free (no PIL/torch) so it can be unit-tested
    without a full ComfyUI runtime.
    """
    texts: list[str] = []
    images: list[bytes] = []

    for part in _iter_parts(response):
        text = _get_attr(part, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)

        mime_type, data = _extract_inline_data(part)
        if mime_type and mime_type.startswith("image/"):
            decoded = _decode_maybe_base64(data)
            if decoded:
                images.append(decoded)

    return "\n".join(texts).strip(), images
