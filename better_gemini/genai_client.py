from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any

from .core import BetterGeminiError, BetterGeminiRequest, extract_text_and_images, normalize_seed

logger = logging.getLogger(__name__)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        try:
            sig = inspect.signature(fn.__init__)  # type: ignore[attr-defined]
        except Exception:
            return kwargs
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    allowed.discard("cls")
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def _set_kwarg_candidates(target_kwargs: dict[str, Any], candidates: list[str], value: Any) -> None:
    for name in candidates:
        target_kwargs[name] = value


def _build_types_config(types_module: Any, request: BetterGeminiRequest) -> Any:
    gen_cfg_cls = getattr(types_module, "GenerateContentConfig", None)
    thinking_cfg_cls = getattr(types_module, "ThinkingConfig", None)
    image_cfg_cls = getattr(types_module, "ImageConfig", None)

    config_kwargs: dict[str, Any] = {}

    _set_kwarg_candidates(config_kwargs, ["response_modalities", "responseModalities"], list(request.response_modalities))
    _set_kwarg_candidates(config_kwargs, ["temperature"], request.temperature)
    _set_kwarg_candidates(config_kwargs, ["top_p", "topP"], request.top_p)
    _set_kwarg_candidates(config_kwargs, ["top_k", "topK"], request.top_k)
    _set_kwarg_candidates(config_kwargs, ["max_output_tokens", "maxOutputTokens"], request.max_output_tokens)
    normalized_seed = normalize_seed(request.seed) if request.seed is not None else None
    _set_kwarg_candidates(config_kwargs, ["seed"], normalized_seed)

    if request.thinking_budget and thinking_cfg_cls is not None:
        thinking_kwargs: dict[str, Any] = {}
        _set_kwarg_candidates(thinking_kwargs, ["thinking_budget", "thinkingBudget"], request.thinking_budget)
        thinking_kwargs = _filter_kwargs_for_callable(thinking_cfg_cls, thinking_kwargs)
        thinking_cfg = thinking_cfg_cls(**thinking_kwargs) if thinking_kwargs else None
        if thinking_cfg is not None:
            _set_kwarg_candidates(config_kwargs, ["thinking_config", "thinkingConfig"], thinking_cfg)

    if image_cfg_cls is not None and (
        request.image_aspect_ratio or request.image_resolution or (request.image_width and request.image_height)
    ):
        image_kwargs: dict[str, Any] = {}
        if request.image_aspect_ratio:
            _set_kwarg_candidates(image_kwargs, ["aspect_ratio", "aspectRatio"], request.image_aspect_ratio)
        if request.image_resolution:
            _set_kwarg_candidates(image_kwargs, ["image_size", "imageSize", "resolution"], request.image_resolution)
        if request.image_width and request.image_height:
            _set_kwarg_candidates(image_kwargs, ["width"], request.image_width)
            _set_kwarg_candidates(image_kwargs, ["height"], request.image_height)
        image_kwargs = _filter_kwargs_for_callable(image_cfg_cls, image_kwargs)
        image_cfg = image_cfg_cls(**image_kwargs) if image_kwargs else None
        if image_cfg is not None:
            _set_kwarg_candidates(config_kwargs, ["image_config", "imageConfig"], image_cfg)

    if gen_cfg_cls is None:
        return {k: v for k, v in config_kwargs.items() if v is not None}

    filtered = _filter_kwargs_for_callable(gen_cfg_cls, config_kwargs)
    return gen_cfg_cls(**filtered)


def _call_generate_content(client: Any, *, model: str, contents: Any, config: Any) -> Any:
    generate_fn = client.models.generate_content
    try:
        sig = inspect.signature(generate_fn)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}

    kwargs = {"model": model}
    if "contents" in params:
        kwargs["contents"] = contents
    elif "content" in params:
        kwargs["content"] = contents
    else:
        kwargs["contents"] = contents

    if "config" in params:
        kwargs["config"] = config
    elif "generation_config" in params:
        kwargs["generation_config"] = config
    else:
        kwargs["config"] = config

    return generate_fn(**kwargs)


def _build_contents(types_module: Any, *, prompt: str, input_images: tuple[bytes, ...]) -> Any:
    if not input_images:
        return prompt

    content_cls = getattr(types_module, "Content", None)
    part_cls = getattr(types_module, "Part", None)
    if (
        content_cls is not None
        and part_cls is not None
        and hasattr(part_cls, "from_text")
        and hasattr(part_cls, "from_bytes")
    ):
        parts = [part_cls.from_text(prompt)]
        parts.extend(part_cls.from_bytes(data=image, mime_type="image/png") for image in input_images)
        return [content_cls(role="user", parts=parts)]

    import base64

    parts: list[dict[str, Any]] = [{"text": prompt}]
    for image in input_images:
        parts.append(
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode("utf-8")}},
        )
    return [{"role": "user", "parts": parts}]


def generate_image_sync(
    *,
    api_key: str | None,
    request: BetterGeminiRequest,
    system_prompt: str = "",
) -> tuple[str, list[bytes]]:
    try:
        from google import genai  # type: ignore[import-not-found]
        from google.genai import types  # type: ignore[import-not-found]
    except Exception as e:
        raise BetterGeminiError(
            "Missing dependency `google-genai`. Install it with `pip install -r ComfyUI-Better-Gemini/requirements.txt`."
        ) from e

    resolved_api_key = api_key or _first_env("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not resolved_api_key:
        raise BetterGeminiError("No API key provided. Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) or pass `api_key`.")

    prompt = request.prompt
    if system_prompt:
        prompt = f"{system_prompt.strip()}\n\n{prompt}"

    client = genai.Client(api_key=resolved_api_key)
    cfg = _build_types_config(types, request)
    contents = _build_contents(types, prompt=prompt, input_images=request.input_images)
    logger.debug(
        "Calling Gemini generate_content with model=%s modalities=%s prompt_images=%d",
        request.model,
        request.response_modalities,
        len(request.input_images),
    )
    response = _call_generate_content(client, model=request.model, contents=contents, config=cfg)

    text, images = extract_text_and_images(response)
    return text, images


async def generate_image(
    *,
    api_key: str | None,
    request: BetterGeminiRequest,
    system_prompt: str = "",
) -> tuple[str, list[bytes]]:
    return await asyncio.to_thread(generate_image_sync, api_key=api_key, request=request, system_prompt=system_prompt)
