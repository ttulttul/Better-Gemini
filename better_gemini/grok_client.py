from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .grok_core import BetterGrokError, BetterGrokRequest

logger = logging.getLogger(__name__)

BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-imagine-image"
DEFAULT_MODELS = [
    "grok-imagine-image",
    "grok-imagine-image-pro",
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
    encoded = base64.b64encode(image_bytes).decode("utf-8")
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
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return body.strip() or "empty error response"


def _request_json(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> Any:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url=url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        message = _extract_error_message(body)
        logger.error("xAI API request failed: %s %s -> %s", method, path, message)
        raise BetterGrokError(f"xAI API request failed ({e.code}) for {path}: {message}") from e
    except URLError as e:
        logger.error("xAI API request failed: %s %s -> %s", method, path, e)
        raise BetterGrokError(f"xAI API request failed for {path}: {e}") from e

    if not body.strip():
        return {}
    return json.loads(body)


def _parse_model_names(payload: Any) -> list[str]:
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            names.append(model_id)
        aliases = model.get("aliases")
        if isinstance(aliases, list):
            names.extend(alias for alias in aliases if isinstance(alias, str) and alias)

    return sorted(dict.fromkeys(names))


def list_models_sync(*, api_key: str | None, cache_ttl_s: int = _MODEL_LIST_CACHE_TTL_S) -> list[str]:
    resolved_api_key = api_key or _first_env("XAI_API_KEY")
    if not resolved_api_key:
        raise BetterGrokError("No API key provided. Set `XAI_API_KEY` or pass `api_key`.")

    cache_key = hashlib.sha256(resolved_api_key.encode("utf-8")).hexdigest()
    now = time.monotonic()
    if cache_ttl_s > 0:
        cached = _MODEL_LIST_CACHE.get(cache_key)
        if cached is not None:
            cached_at, models = cached
            if now - cached_at < cache_ttl_s:
                logger.debug("Using cached xAI image model list (%d models).", len(models))
                return list(models)

    payload = _request_json(method="GET", path="/image-generation-models", api_key=resolved_api_key)
    models = _parse_model_names(payload)
    if cache_ttl_s > 0:
        _MODEL_LIST_CACHE[cache_key] = (now, models)
    return list(models)


def _build_image_request(request: BetterGrokRequest) -> tuple[str, dict[str, Any], str]:
    payload: dict[str, Any] = {
        "model": request.model,
        "prompt": request.prompt,
        "response_format": "b64_json",
    }
    if request.aspect_ratio:
        payload["aspect_ratio"] = request.aspect_ratio
    if request.resolution:
        payload["resolution"] = request.resolution
    if request.n > 1:
        payload["n"] = request.n

    if request.input_images:
        images = [{"type": "image_url", "url": _data_uri_from_bytes(image)} for image in request.input_images]
        if len(images) == 1:
            payload["image"] = images[0]
        else:
            payload["images"] = images
        return "/images/edits", payload, "edit"

    return "/images/generations", payload, "generation"


def _decode_b64_json(item: Any) -> bytes | None:
    if not isinstance(item, dict):
        return None
    b64_payload = item.get("b64_json") or item.get("b64Json")
    if not isinstance(b64_payload, str) or not b64_payload:
        return None
    return base64.b64decode(b64_payload)


def generate_images_sync(*, api_key: str | None, request: BetterGrokRequest) -> tuple[str, list[bytes]]:
    resolved_api_key = api_key or _first_env("XAI_API_KEY")
    if not resolved_api_key:
        raise BetterGrokError("No API key provided. Set `XAI_API_KEY` or pass `api_key`.")

    path, payload, mode = _build_image_request(request)
    logger.info(
        "Calling xAI image API with model=%s mode=%s n=%d input_images=%d",
        request.model,
        mode,
        request.n,
        len(request.input_images),
    )
    response = _request_json(method="POST", path=path, api_key=resolved_api_key, payload=payload)
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list):
        data = []

    images: list[bytes] = []
    revised_prompts: list[str] = []
    for item in data:
        decoded = _decode_b64_json(item)
        if decoded:
            images.append(decoded)
        if isinstance(item, dict):
            revised_prompt = item.get("revised_prompt") or item.get("revisedPrompt")
            if isinstance(revised_prompt, str) and revised_prompt.strip():
                revised_prompts.append(revised_prompt.strip())

    notes: list[str] = []
    model_used = response.get("model") if isinstance(response, dict) else None
    if not isinstance(model_used, str) or not model_used.strip():
        model_used = request.model

    if images:
        notes.append(f"Grok {mode} returned {len(images)} image(s) using {model_used}.")
    else:
        notes.append(
            f"xAI returned no images for model {model_used}. The request may have been filtered or the API response may have changed."
        )

    for revised_prompt in revised_prompts:
        if revised_prompt != request.prompt:
            notes.append(f"Revised prompt: {revised_prompt}")
            break

    return "\n\n".join(notes).strip(), images


async def generate_images(*, api_key: str | None, request: BetterGrokRequest) -> tuple[str, list[bytes]]:
    return await asyncio.to_thread(generate_images_sync, api_key=api_key, request=request)
