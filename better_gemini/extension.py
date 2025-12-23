from __future__ import annotations

import logging
from typing import Any

from .core import BetterGeminiConfigError, build_request, max_dim_from_resolution
from .genai_client import generate_image

logger = logging.getLogger(__name__)


def _bytes_list_to_comfy_image(
    images: list[bytes],
    *,
    target_max_dim: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
):
    if not images:
        raise ValueError("Gemini returned no images.")
    try:
        from io import BytesIO

        import numpy as np
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "Missing image deps (torch/numpy/Pillow). This node must run inside a ComfyUI environment."
        ) from e

    tensors = []
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for img_bytes in images:
        with Image.open(BytesIO(img_bytes)) as img:
            img = img.convert("RGB")
            if target_width is not None and target_height is not None:
                img = img.resize((target_width, target_height), resample=resample)
            elif target_max_dim is not None:
                w, h = img.size
                current_max = max(w, h)
                if current_max and current_max != target_max_dim:
                    scale = target_max_dim / current_max
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    img = img.resize((new_w, new_h), resample=resample)
            arr = np.array(img).astype("float32") / 255.0
            tensors.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.cat(tensors, dim=0)


def _comfy_image_to_png_bytes(prompt_images: Any) -> list[bytes]:
    if prompt_images is None:
        return []
    try:
        from io import BytesIO

        import numpy as np
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "Missing image deps (torch/numpy/Pillow). This node must run inside a ComfyUI environment."
        ) from e

    if not isinstance(prompt_images, torch.Tensor):
        raise TypeError("`prompt_images` must be a ComfyUI IMAGE (torch.Tensor).")

    images = prompt_images.detach().cpu()
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError("`prompt_images` must have shape [N,H,W,C] (or [H,W,C]).")

    channels = images.shape[-1]
    if channels not in (1, 3, 4):
        raise ValueError(f"`prompt_images` must have 1, 3, or 4 channels; got {channels}.")

    images = images.float().clamp(0.0, 1.0)
    pngs: list[bytes] = []
    for image in images:
        arr = image.numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]

        rgb = (arr * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(rgb, mode="RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        pngs.append(buf.getvalue())

    return pngs


try:
    from comfy_api.latest import IO, ComfyExtension  # type: ignore
    from typing_extensions import override
except Exception as e:  # pragma: no cover
    IO = None  # type: ignore
    ComfyExtension = object  # type: ignore

    def override(fn):  # type: ignore
        return fn

    logger.debug("ComfyUI runtime not available: %s", e)


if IO is not None:

    class BetterGemini(IO.ComfyNode):
        @classmethod
        def define_schema(cls):
            return IO.Schema(
                node_id="BetterGemini",
                display_name="Better Gemini",
                category="api node/image/BetterGemini",
                description="Generate images with Google Gemini using the official `google-genai` Python SDK.",
                not_idempotent=True,
                inputs=[
                    IO.String.Input(
                        "prompt",
                        multiline=True,
                        default="",
                        tooltip="Text prompt for image generation.",
                    ),
                    IO.String.Input(
                        "model",
                        default="gemini-2.5-flash-image-preview",
                        tooltip="Gemini model name (must support image output).",
                    ),
                    IO.String.Input(
                        "api_key",
                        optional=True,
                        default="",
                        tooltip="Optional. If empty, uses env var GOOGLE_API_KEY (or GEMINI_API_KEY).",
                    ),
                    IO.Combo.Input(
                        "response_modalities",
                        options=["IMAGE", "IMAGE+TEXT"],
                        default="IMAGE+TEXT",
                        tooltip="Choose IMAGE-only output, or IMAGE+TEXT to also return text.",
                    ),
                    IO.Image.Input(
                        "prompt_images",
                        optional=True,
                        tooltip="Optional images to include with the prompt (reference / edit). Batched IMAGE tensors send multiple images.",
                    ),
                    IO.Combo.Input(
                        "aspect_ratio",
                        options=[
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
                        ],
                        default="auto",
                        tooltip="If 'auto', the model chooses. Otherwise requests a specific aspect ratio.",
                        optional=True,
                    ),
                    IO.Combo.Input(
                        "resolution",
                        options=["auto", "1K", "2K", "4K"],
                        default="auto",
                        tooltip="Target output resolution (best-effort). Gemini often returns ~1K; this node will resize to match the selection unless width+height are set.",
                        optional=True,
                    ),
                    IO.Int.Input(
                        "width",
                        default=0,
                        min=0,
                        max=8192,
                        step=64,
                        tooltip="Optional override. Set > 0 to request a specific width (must set height too).",
                        optional=True,
                    ),
                    IO.Int.Input(
                        "height",
                        default=0,
                        min=0,
                        max=8192,
                        step=64,
                        tooltip="Optional override. Set > 0 to request a specific height (must set width too).",
                        optional=True,
                    ),
                    IO.Float.Input(
                        "temperature",
                        default=0.9,
                        min=0.0,
                        max=2.0,
                        step=0.01,
                        tooltip="Sampling temperature (model-dependent).",
                        optional=True,
                    ),
                    IO.Float.Input(
                        "top_p",
                        default=0.95,
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        tooltip="Nucleus sampling (model-dependent).",
                        optional=True,
                    ),
                    IO.Int.Input(
                        "top_k",
                        default=40,
                        min=0,
                        max=1000,
                        step=1,
                        tooltip="Top-K sampling (model-dependent).",
                        optional=True,
                    ),
                    IO.Int.Input(
                        "max_output_tokens",
                        default=8192,
                        min=1,
                        max=131072,
                        step=1,
                        tooltip="Max tokens for any text the model returns (model-dependent).",
                        optional=True,
                    ),
                    IO.Combo.Input(
                        "thinking_difficulty",
                        options=["auto", "low", "medium", "high"],
                        default="auto",
                        tooltip="Hint for how much 'thinking' budget to allow (model-dependent).",
                        optional=True,
                    ),
                    IO.Int.Input(
                        "thinking_budget",
                        default=0,
                        min=0,
                        max=131072,
                        step=256,
                        tooltip="Optional override. Set > 0 to request an explicit thinking budget.",
                        optional=True,
                    ),
                    IO.Int.Input(
                        "seed",
                        default=0,
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        step=1,
                        control_after_generate=True,
                        tooltip="Best-effort seed for determinism (not guaranteed). Set 0 for 'unset'. Gemini requires int32; larger values are folded.",
                        optional=True,
                    ),
                    IO.String.Input(
                        "system_prompt",
                        multiline=True,
                        default="",
                        optional=True,
                        tooltip="Optional system prompt. If set, it will be prepended to your prompt.",
                    ),
                ],
                outputs=[
                    IO.Image.Output(),
                    IO.String.Output(),
                ],
            )

        @classmethod
        async def execute(
            cls,
            prompt: str,
            model: str,
            api_key: str = "",
            response_modalities: str = "IMAGE+TEXT",
            prompt_images: Any = None,
            aspect_ratio: str = "auto",
            resolution: str = "auto",
            width: int = 0,
            height: int = 0,
            temperature: float = 0.9,
            top_p: float = 0.95,
            top_k: int = 40,
            max_output_tokens: int = 8192,
            thinking_difficulty: str = "auto",
            thinking_budget: int = 0,
            seed: int = 0,
            system_prompt: str = "",
        ) -> Any:
            try:
                prompt_image_bytes = _comfy_image_to_png_bytes(prompt_images)
                request = build_request(
                    model=model,
                    prompt=prompt,
                    response_modalities=response_modalities,
                    input_images=prompt_image_bytes,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    width=width,
                    height=height,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=max_output_tokens,
                    thinking_difficulty=thinking_difficulty,
                    thinking_budget=thinking_budget,
                    seed=seed,
                )
            except BetterGeminiConfigError as e:
                raise ValueError(str(e)) from e

            text, images = await generate_image(
                api_key=(api_key.strip() or None),
                request=request,
                system_prompt=system_prompt or "",
            )
            requested_width = request.image_width
            requested_height = request.image_height
            target_max_dim = None
            if requested_width is None or requested_height is None:
                target_max_dim = max_dim_from_resolution(request.image_resolution)

            image_tensor = _bytes_list_to_comfy_image(
                images,
                target_max_dim=target_max_dim,
                target_width=requested_width,
                target_height=requested_height,
            )
            return IO.NodeOutput(image_tensor, text)


    class BetterGeminiExtension(ComfyExtension):
        @override
        async def get_node_list(self):
            return [BetterGemini]


async def comfy_entrypoint():  # pragma: no cover
    if IO is None:
        raise RuntimeError("BetterGemini: ComfyUI V3 runtime not available (missing `comfy_api.latest`).")
    return BetterGeminiExtension()
