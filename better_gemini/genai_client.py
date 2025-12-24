from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlencode

from .core import BetterGeminiError, BetterGeminiRequest, describe_response_block, extract_text_and_images, normalize_seed

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "models/gemini-3-pro-image-preview"
_MODEL_LIST_CACHE: dict[str, tuple[float, list[str]]] = {}
_MODEL_LIST_CACHE_TTL_S = 10 * 60


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


def _build_image_config_patch(request: BetterGeminiRequest) -> dict[str, Any] | None:
    image_cfg: dict[str, Any] = {}
    if request.image_aspect_ratio:
        image_cfg["aspectRatio"] = request.image_aspect_ratio
    if request.image_resolution:
        image_cfg["imageSize"] = request.image_resolution
    if request.image_width and request.image_height:
        image_cfg["width"] = request.image_width
        image_cfg["height"] = request.image_height
    if not image_cfg:
        return None
    return {"imageConfig": image_cfg}


def _merge_generation_config_patch(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if (
            key == "imageConfig"
            and key in target
            and isinstance(target.get(key), dict)
            and isinstance(value, dict)
        ):
            target[key] = {**target[key], **value}
        else:
            target[key] = value


def _call_generate_content_with_generation_config_patch(
    client: Any,
    *,
    model: str,
    contents: Any,
    config: Any,
    generation_config_patch: dict[str, Any],
) -> Any:
    try:
        from google.genai import models as genai_models  # type: ignore[import-not-found]
        from google.genai import types  # type: ignore[import-not-found]
        from google.genai import _common as genai_common  # type: ignore[import-not-found]
    except Exception:
        logger.debug("Falling back to SDK generate_content (failed to import genai internals).", exc_info=True)
        return _call_generate_content(client, model=model, contents=contents, config=config)

    parameter_model = types._GenerateContentParameters(model=model, contents=contents, config=config)
    api_client = client._api_client

    if api_client.vertexai:
        request_dict = genai_models._GenerateContentParameters_to_vertex(api_client, parameter_model)
        request_url_dict = request_dict.get("_url")
        path = "{model}:generateContent".format_map(request_url_dict) if request_url_dict else "{model}:generateContent"
    else:
        request_dict = genai_models._GenerateContentParameters_to_mldev(api_client, parameter_model)
        request_url_dict = request_dict.get("_url")
        path = "{model}:generateContent".format_map(request_url_dict) if request_url_dict else "{model}:generateContent"

    query_params = request_dict.get("_query")
    if query_params:
        path = f"{path}?{urlencode(query_params)}"

    request_dict.pop("config", None)
    gen_cfg = request_dict.get("generationConfig")
    if gen_cfg is None:
        gen_cfg = {}
        request_dict["generationConfig"] = gen_cfg
    if isinstance(gen_cfg, dict):
        _merge_generation_config_patch(gen_cfg, generation_config_patch)
    else:
        request_dict["generationConfig"] = generation_config_patch

    http_options = None
    if (
        parameter_model.config is not None
        and getattr(parameter_model.config, "http_options", None) is not None
    ):
        http_options = parameter_model.config.http_options

    request_dict = genai_common.convert_to_dict(request_dict)
    request_dict = genai_common.encode_unserializable_types(request_dict)
    response = api_client.request("post", path, request_dict, http_options)

    response_dict = {} if not response.body else json.loads(response.body)
    if api_client.vertexai:
        response_dict = genai_models._GenerateContentResponse_from_vertex(response_dict)
    else:
        response_dict = genai_models._GenerateContentResponse_from_mldev(response_dict)
    return response_dict


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
        parts = [part_cls.from_text(text=prompt)]
        parts.extend(part_cls.from_bytes(data=image, mime_type="image/png") for image in input_images)
        return [content_cls(role="user", parts=parts)]

    import base64

    parts: list[dict[str, Any]] = [{"text": prompt}]
    for image in input_images:
        parts.append(
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode("utf-8")}},
        )
    return [{"role": "user", "parts": parts}]


def _model_supported_actions(model: Any) -> set[str]:
    if model is None:
        return set()
    if isinstance(model, dict):
        actions = model.get("supported_actions") or model.get("supportedActions") or []
    else:
        actions = getattr(model, "supported_actions", None) or getattr(model, "supportedActions", None) or []
    if actions is None:
        return set()
    if isinstance(actions, str):
        return {actions}
    try:
        return {str(action) for action in actions}
    except TypeError:
        return set()


def _model_name(model: Any) -> str | None:
    if model is None:
        return None
    if isinstance(model, dict):
        name = model.get("name")
    else:
        name = getattr(model, "name", None)
    if not name:
        return None
    return str(name)


def list_models_sync(
    *,
    api_key: str | None,
    filter_action: str | None = "generateContent",
    cache_ttl_s: int = _MODEL_LIST_CACHE_TTL_S,
) -> list[str]:
    """
    List available models using the official `google-genai` SDK (`client.models.list()`).

    `filter_action` defaults to "generateContent" to match the node's usage.
    """

    try:
        from google import genai  # type: ignore[import-not-found]
    except Exception as e:
        raise BetterGeminiError(
            "Missing dependency `google-genai`. Install it with `pip install -r ComfyUI-Better-Gemini/requirements.txt`."
        ) from e

    resolved_api_key = api_key or _first_env("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not resolved_api_key:
        raise BetterGeminiError("No API key provided. Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) or pass `api_key`.")

    cache_key = hashlib.sha256(resolved_api_key.encode("utf-8")).hexdigest()
    now = time.monotonic()
    if cache_ttl_s > 0:
        cached = _MODEL_LIST_CACHE.get(cache_key)
        if cached is not None:
            cached_at, models = cached
            if now - cached_at < cache_ttl_s:
                logger.debug("Using cached Gemini model list (%d models).", len(models))
                return list(models)

    client = genai.Client(api_key=resolved_api_key)
    models: list[str] = []
    for model in client.models.list():
        name = _model_name(model)
        if not name:
            continue
        if filter_action:
            actions = _model_supported_actions(model)
            if filter_action not in actions:
                continue
        models.append(name)

    models = sorted(set(models))
    if cache_ttl_s > 0:
        _MODEL_LIST_CACHE[cache_key] = (now, models)
    return list(models)


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
    generation_config_patch = _build_image_config_patch(request)
    logger.debug(
        "Calling Gemini generate_content with model=%s modalities=%s prompt_images=%d",
        request.model,
        request.response_modalities,
        len(request.input_images),
    )
    if generation_config_patch:
        logger.debug(
            "Applying generationConfig patch (keys=%s) for model=%s",
            sorted(generation_config_patch.keys()),
            request.model,
        )
        response = _call_generate_content_with_generation_config_patch(
            client,
            model=request.model,
            contents=contents,
            config=cfg,
            generation_config_patch=generation_config_patch,
        )
    else:
        response = _call_generate_content(client, model=request.model, contents=contents, config=cfg)

    text, images = extract_text_and_images(response)
    if "IMAGE" in request.response_modalities and not images:
        block_desc = describe_response_block(response)
        if block_desc:
            logger.warning("%s No images were returned.", block_desc)
            if text:
                text = f"{text}\n\n{block_desc}".strip()
            else:
                text = block_desc
        else:
            no_image_msg = (
                f"Gemini returned no images for model {request.model}. "
                "The request may have been blocked or the model may not support image output."
            )
            logger.warning(no_image_msg)
            if text:
                text = f"{text}\n\n{no_image_msg}".strip()
            else:
                text = no_image_msg
    return text, images


async def generate_image(
    *,
    api_key: str | None,
    request: BetterGeminiRequest,
    system_prompt: str = "",
) -> tuple[str, list[bytes]]:
    return await asyncio.to_thread(generate_image_sync, api_key=api_key, request=request, system_prompt=system_prompt)
