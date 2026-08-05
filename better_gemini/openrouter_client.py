from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .openrouter_core import BetterOpenRouterError, BetterOpenRouterRequest

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
APP_URL = "https://github.com/ttulttul/Better-Gemini"
APP_TITLE = "ComfyUI Better Gemini"
USER_AGENT = f"{APP_TITLE} ({APP_URL})"
DEFAULT_MODEL = "openai/gpt-image-2"
DEFAULT_MODELS = [
    "openai/gpt-image-2",
    "openai/gpt-image-1",
    "openai/gpt-image-1-mini",
    "google/gemini-3.1-flash-image",
    "google/gemini-3-pro-image",
    "bytedance-seed/seedream-4.5",
    "black-forest-labs/flux.2-pro",
]
_MODEL_LIST_CACHE: dict[str, tuple[float, list[str]]] = {}
_MODEL_LIST_CACHE_TTL_S = 10 * 60


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _data_uri_from_bytes(image_bytes: bytes, *, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_error_message(body: str) -> str:
    if not body:
        return "empty error response"
    try:
        payload = json.loads(body)
    except Exception:
        return body.strip() or "empty error response"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("error")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if isinstance(error, str) and error.strip():
            return error.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return body.strip() or "empty error response"


def _build_headers(*, api_key: str | None, has_payload: bool) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": USER_AGENT,
        "HTTP-Referer": APP_URL,
        "X-OpenRouter-Title": APP_TITLE,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if has_payload:
        headers["Content-Type"] = "application/json"
    return headers


def _request_json(
    *,
    method: str,
    path: str,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 300.0,
) -> Any:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = _build_headers(api_key=api_key, has_payload=payload is not None)
    request = Request(url=url, data=data, headers=headers, method=method)

    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        message = _extract_error_message(body)
        logger.error("OpenRouter API request failed: %s %s -> %s", method, path, message)
        raise BetterOpenRouterError(
            f"OpenRouter API request failed ({e.code}) for {path}: {message}"
        ) from e
    except URLError as e:
        logger.error("OpenRouter API request failed: %s %s -> %s", method, path, e)
        raise BetterOpenRouterError(f"OpenRouter API request failed for {path}: {e}") from e

    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        logger.error("OpenRouter API returned invalid JSON for %s %s.", method, path)
        raise BetterOpenRouterError(f"OpenRouter API returned invalid JSON for {path}.") from e


def _parse_model_names(payload: Any) -> list[str]:
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id.strip():
            names.append(model_id.strip())
    return sorted(dict.fromkeys(names))


def _model_list_cache_key(api_key: str | None) -> str:
    if not api_key:
        return "public"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def list_models_sync(
    *,
    api_key: str | None,
    cache_ttl_s: int = _MODEL_LIST_CACHE_TTL_S,
) -> list[str]:
    resolved_api_key = api_key or _first_env("OPENROUTER_API_KEY")
    cache_key = _model_list_cache_key(resolved_api_key)
    now = time.monotonic()
    if cache_ttl_s > 0:
        cached = _MODEL_LIST_CACHE.get(cache_key)
        if cached is not None:
            cached_at, models = cached
            if now - cached_at < cache_ttl_s:
                logger.debug("Using cached OpenRouter image model list (%d models).", len(models))
                return list(models)

    payload = _request_json(
        method="GET",
        path="/images/models",
        api_key=resolved_api_key,
        timeout_s=10.0,
    )
    models = _parse_model_names(payload)
    if cache_ttl_s > 0:
        _MODEL_LIST_CACHE[cache_key] = (now, models)
    logger.info("Discovered %d OpenRouter image model(s).", len(models))
    return list(models)


def _build_image_request(request: BetterOpenRouterRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "prompt": request.prompt,
    }
    if request.n > 1:
        payload["n"] = request.n
    if request.image_width is not None and request.image_height is not None:
        payload["size"] = f"{request.image_width}x{request.image_height}"
    else:
        if request.resolution:
            payload["resolution"] = request.resolution
        if request.aspect_ratio:
            payload["aspect_ratio"] = request.aspect_ratio
    if request.quality:
        payload["quality"] = request.quality
    if request.output_format:
        payload["output_format"] = request.output_format
    if request.background:
        payload["background"] = request.background
    if request.output_compression is not None:
        payload["output_compression"] = request.output_compression
    if request.seed is not None:
        payload["seed"] = request.seed
    if request.input_images:
        payload["input_references"] = [
            {
                "type": "image_url",
                "image_url": {"url": _data_uri_from_bytes(image)},
            }
            for image in request.input_images
        ]
    return payload


def _decode_image_item(item: Any) -> bytes | None:
    if not isinstance(item, dict):
        return None
    b64_payload = item.get("b64_json") or item.get("b64Json")
    if not isinstance(b64_payload, str) or not b64_payload:
        return None
    try:
        return base64.b64decode(b64_payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise BetterOpenRouterError("OpenRouter returned an invalid base64 image payload.") from e


def _is_svg_image(item: Any, image_bytes: bytes) -> bool:
    media_type = item.get("media_type") if isinstance(item, dict) else None
    if isinstance(media_type, str) and media_type.lower() == "image/svg+xml":
        return True
    return image_bytes.lstrip().startswith((b"<svg", b"<?xml"))


def _usage_note(response: Any) -> str | None:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return None
    return f"Reported cost: ${cost:.6f}."


def generate_images_sync(
    *,
    api_key: str | None,
    request: BetterOpenRouterRequest,
) -> tuple[str, list[bytes]]:
    resolved_api_key = api_key or _first_env("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise BetterOpenRouterError(
            "No API key provided. Set `OPENROUTER_API_KEY` or pass `api_key`."
        )

    payload = _build_image_request(request)
    logger.info(
        "Calling OpenRouter image API with model=%s n=%d input_references=%d",
        request.model,
        request.n,
        len(request.input_images),
    )
    response = _request_json(
        method="POST",
        path="/images",
        api_key=resolved_api_key,
        payload=payload,
    )
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list):
        data = []

    images: list[bytes] = []
    for item in data:
        decoded = _decode_image_item(item)
        if decoded is None:
            logger.warning("Ignoring OpenRouter image response item without b64_json data.")
            continue
        if _is_svg_image(item, decoded):
            raise BetterOpenRouterError(
                "OpenRouter returned SVG output, but ComfyUI IMAGE outputs require a raster image. "
                "Select a raster model or request PNG, JPEG, or WebP output."
            )
        images.append(decoded)

    if images:
        notes = [f"OpenRouter returned {len(images)} image(s) using {request.model}."]
        usage_note = _usage_note(response)
        if usage_note:
            notes.append(usage_note)
        return " ".join(notes), images

    message = (
        f"OpenRouter returned no raster images for model {request.model}. "
        "The request may have been filtered or the API response may have changed."
    )
    logger.warning(message)
    return message, []


async def generate_images(
    *,
    api_key: str | None,
    request: BetterOpenRouterRequest,
) -> tuple[str, list[bytes]]:
    return await asyncio.to_thread(generate_images_sync, api_key=api_key, request=request)
