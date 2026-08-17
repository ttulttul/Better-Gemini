from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .grok_core import BetterGrokError, BetterGrokRequest
from .grok_rate_limit import GrokRateLimitError, grok_rate_limit_coordinator

logger = logging.getLogger(__name__)

BASE_URL = "https://api.x.ai/v1"
USER_AGENT = "ComfyUI-Better-Gemini/1.5.0 (https://github.com/ttulttul/Better-Gemini)"
DEFAULT_MODEL = "grok-imagine-image"
DEFAULT_IMAGE_MODELS = [
    "grok-imagine-image",
    "grok-imagine-image-2.0",
    "grok-imagine-image-pro",
    "grok-imagine-image-quality",
]
DEFAULT_TEXT_MODELS = [
    "grok-latest",
    "grok-4",
    "grok-4-fast-non-reasoning",
    "grok-3-mini",
    "grok-code-fast-1",
]
DEFAULT_MODELS = list(dict.fromkeys([*DEFAULT_IMAGE_MODELS, *DEFAULT_TEXT_MODELS]))
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
    except json.JSONDecodeError:
        return body.strip() or "empty error response"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        if isinstance(error, dict):
            message = error.get("message") or error.get("error")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return body.strip() or "empty error response"


def _extract_error_payload(body: str) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _build_headers(*, api_key: str, has_payload: bool) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": USER_AGENT,
    }
    if has_payload:
        headers["Content-Type"] = "application/json"
    return headers


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    raw_value = headers.get("Retry-After")
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    value = raw_value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _request_json_once(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> Any:
    url = f"{BASE_URL}{path}"
    data = None
    headers = _build_headers(api_key=api_key, has_payload=payload is not None)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = Request(url=url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        error_payload = _extract_error_payload(body)
        message = _extract_error_message(body)
        logger.error("xAI API request failed: %s %s -> %s", method, path, message)

        if e.code == 429:
            raise GrokRateLimitError(
                f"xAI API request failed (429) for {path}: {message}",
                retry_after_seconds=_retry_after_seconds(e.headers),
            ) from e

        if (
            e.code == 403
            and isinstance(error_payload, dict)
            and error_payload.get("error_code") == 1010
        ):
            raise BetterGrokError(
                "xAI API request failed (403) for "
                f"{path}: Cloudflare blocked the default client signature. "
                "The node now sends an explicit application User-Agent; restart ComfyUI and retry. "
                f"Original response: {message}"
            ) from e

        raise BetterGrokError(f"xAI API request failed ({e.code}) for {path}: {message}") from e
    except URLError as e:
        logger.error("xAI API request failed: %s %s -> %s", method, path, e)
        raise BetterGrokError(f"xAI API request failed for {path}: {e}") from e

    if not body.strip():
        return {}
    return json.loads(body)


def _request_json(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
    _cancel_event: threading.Event | None = None,
) -> Any:
    model = payload.get("model") if isinstance(payload, dict) else None
    bucket = model if isinstance(model, str) and model.strip() else path
    execute_kwargs: dict[str, Any] = {
        "model": bucket,
        "operation": lambda: _request_json_once(
            method=method,
            path=path,
            api_key=api_key,
            payload=payload,
            timeout_s=timeout_s,
        ),
    }
    if _cancel_event is not None:
        execute_kwargs["cancel_event"] = _cancel_event
    return grok_rate_limit_coordinator.execute(
        **execute_kwargs,
    )


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


def _model_list_cache_key(*, api_key: str, model_type: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"{model_type}:{digest}"


def list_models_sync(
    *,
    api_key: str | None,
    model_type: str = "all",
    cache_ttl_s: int = _MODEL_LIST_CACHE_TTL_S,
) -> list[str]:
    resolved_api_key = api_key or _first_env("XAI_API_KEY")
    if not resolved_api_key:
        raise BetterGrokError("No API key provided. Set `XAI_API_KEY` or pass `api_key`.")
    if model_type not in {"all", "image", "language"}:
        raise BetterGrokError(f"Unsupported Grok model listing type: {model_type!r}")

    cache_key = _model_list_cache_key(api_key=resolved_api_key, model_type=model_type)
    now = time.monotonic()
    if cache_ttl_s > 0:
        cached = _MODEL_LIST_CACHE.get(cache_key)
        if cached is not None:
            cached_at, models = cached
            if now - cached_at < cache_ttl_s:
                logger.debug("Using cached xAI %s model list (%d models).", model_type, len(models))
                return list(models)

    paths: list[str]
    if model_type == "image":
        paths = ["/image-generation-models"]
    elif model_type == "language":
        paths = ["/language-models"]
    else:
        paths = ["/image-generation-models", "/language-models"]

    models: list[str] = []
    for path in paths:
        payload = _request_json(method="GET", path=path, api_key=resolved_api_key)
        models.extend(_parse_model_names(payload))
    models = sorted(dict.fromkeys(models))
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


def _build_text_input(request: BetterGrokRequest) -> list[dict[str, Any]]:
    if not request.input_images:
        return [{"role": "user", "content": request.prompt}]

    content: list[dict[str, Any]] = [{"type": "input_text", "text": request.prompt}]
    for image in request.input_images:
        content.append(
            {
                "type": "input_image",
                "image_url": _data_uri_from_bytes(image),
            }
        )
    return [{"role": "user", "content": content}]


def _build_text_request(request: BetterGrokRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "store": False,
        "reasoning": {"effort": request.reasoning_effort},
        "input": _build_text_input(request),
    }


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n\n".join(texts).strip()


def _extract_responses_text(response: Any) -> tuple[str, str]:
    if not isinstance(response, dict):
        return "", ""

    model_used = response.get("model")
    if not isinstance(model_used, str) or not model_used.strip():
        model_used = ""

    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return model_used, output_text.strip()

    output = response.get("output")
    if not isinstance(output, list):
        return model_used, ""

    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "output_text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            continue
        if item_type not in {"message", "assistant_message"}:
            continue
        text = _extract_text_content(item.get("content"))
        if text:
            texts.append(text)
    return model_used, "\n\n".join(texts).strip()


def _extract_chat_text(response: Any) -> tuple[str, str]:
    return _extract_responses_text(response)


def _decode_b64_json(item: Any) -> bytes | None:
    if not isinstance(item, dict):
        return None
    b64_payload = item.get("b64_json") or item.get("b64Json")
    if not isinstance(b64_payload, str) or not b64_payload:
        return None
    return base64.b64decode(b64_payload)


def generate_images_sync(
    *,
    api_key: str | None,
    request: BetterGrokRequest,
    _cancel_event: threading.Event | None = None,
) -> tuple[str, list[bytes]]:
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
    request_kwargs: dict[str, Any] = {
        "method": "POST",
        "path": path,
        "api_key": resolved_api_key,
        "payload": payload,
    }
    if _cancel_event is not None:
        request_kwargs["_cancel_event"] = _cancel_event
    response = _request_json(**request_kwargs)
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


def generate_text_sync(
    *,
    api_key: str | None,
    request: BetterGrokRequest,
    _cancel_event: threading.Event | None = None,
) -> tuple[str, list[bytes]]:
    resolved_api_key = api_key or _first_env("XAI_API_KEY")
    if not resolved_api_key:
        raise BetterGrokError("No API key provided. Set `XAI_API_KEY` or pass `api_key`.")

    payload = _build_text_request(request)
    logger.info(
        "Calling xAI responses API with model=%s reasoning_effort=%s prompt_images=%d",
        request.model,
        request.reasoning_effort,
        len(request.input_images),
    )
    request_kwargs: dict[str, Any] = {
        "method": "POST",
        "path": "/responses",
        "api_key": resolved_api_key,
        "payload": payload,
    }
    if _cancel_event is not None:
        request_kwargs["_cancel_event"] = _cancel_event
    response = _request_json(**request_kwargs)
    model_used, text = _extract_responses_text(response)
    if text:
        return text, []

    resolved_model = model_used or request.model
    message = (
        f"xAI returned no text for model {resolved_model}. "
        "The request may have been filtered or the API response may have changed."
    )
    logger.warning(message)
    return message, []


def generate_sync(
    *,
    api_key: str | None,
    request: BetterGrokRequest,
    _cancel_event: threading.Event | None = None,
) -> tuple[str, list[bytes]]:
    if "IMAGE" in request.response_modalities:
        return generate_images_sync(
            api_key=api_key,
            request=request,
            _cancel_event=_cancel_event,
        )
    return generate_text_sync(
        api_key=api_key,
        request=request,
        _cancel_event=_cancel_event,
    )


async def _run_sync_with_cancellation(
    function: Callable[..., tuple[str, list[bytes]]],
    **kwargs: Any,
) -> tuple[str, list[bytes]]:
    cancel_event = threading.Event()
    try:
        return await asyncio.to_thread(
            function,
            **kwargs,
            _cancel_event=cancel_event,
        )
    except asyncio.CancelledError:
        cancel_event.set()
        raise


async def generate_images(*, api_key: str | None, request: BetterGrokRequest) -> tuple[str, list[bytes]]:
    return await _run_sync_with_cancellation(
        generate_images_sync,
        api_key=api_key,
        request=request,
    )


async def generate_content(*, api_key: str | None, request: BetterGrokRequest) -> tuple[str, list[bytes]]:
    return await _run_sync_with_cancellation(
        generate_sync,
        api_key=api_key,
        request=request,
    )
