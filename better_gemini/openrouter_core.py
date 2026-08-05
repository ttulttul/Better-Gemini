from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


class BetterOpenRouterError(RuntimeError):
    pass


class BetterOpenRouterConfigError(ValueError):
    pass


SUPPORTED_ASPECT_RATIOS: tuple[str, ...] = (
    "auto",
    "1:1",
    "1:2",
    "2:1",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "9:19.5",
    "19.5:9",
    "9:20",
    "20:9",
    "9:21",
    "21:9",
)

SUPPORTED_IMAGE_RESOLUTIONS: tuple[str, ...] = (
    "auto",
    "512",
    "1K",
    "2K",
    "4K",
)

SUPPORTED_QUALITIES: tuple[str, ...] = ("auto", "low", "medium", "high")
SUPPORTED_OUTPUT_FORMATS: tuple[str, ...] = ("auto", "png", "jpeg", "webp")
SUPPORTED_BACKGROUNDS: tuple[str, ...] = ("auto", "transparent", "opaque")
MAX_IMAGES_PER_REQUEST = 10


@dataclass(frozen=True)
class BetterOpenRouterRequest:
    model: str
    prompt: str
    aspect_ratio: str | None = None
    resolution: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    quality: str | None = None
    output_format: str | None = None
    background: str | None = None
    output_compression: int | None = None
    n: int = 1
    seed: int | None = None
    input_images: tuple[bytes, ...] = ()


def build_request(
    *,
    model: str,
    prompt: str,
    aspect_ratio: str = "auto",
    resolution: str = "auto",
    width: int = 0,
    height: int = 0,
    quality: str = "auto",
    output_format: str = "auto",
    background: str = "auto",
    output_compression: int = -1,
    n: int = 1,
    seed: int = 0,
    input_images: Iterable[bytes] | None = None,
) -> BetterOpenRouterRequest:
    if not prompt or not isinstance(prompt, str):
        raise BetterOpenRouterConfigError("`prompt` must be a non-empty string.")
    if not model or not isinstance(model, str):
        raise BetterOpenRouterConfigError("`model` must be a non-empty string.")

    resolved_aspect_ratio = (aspect_ratio or "auto").strip()
    if resolved_aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        raise BetterOpenRouterConfigError(f"Unsupported aspect ratio: {resolved_aspect_ratio!r}")

    resolution_value = (resolution or "auto").strip()
    resolved_resolution = "auto" if resolution_value.lower() == "auto" else resolution_value.upper()
    if resolved_resolution not in SUPPORTED_IMAGE_RESOLUTIONS:
        raise BetterOpenRouterConfigError(f"Unsupported resolution: {resolved_resolution!r}")

    resolved_quality = (quality or "auto").strip().lower()
    if resolved_quality not in SUPPORTED_QUALITIES:
        raise BetterOpenRouterConfigError(f"Unsupported quality: {resolved_quality!r}")

    resolved_output_format = (output_format or "auto").strip().lower()
    if resolved_output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise BetterOpenRouterConfigError(f"Unsupported output format: {resolved_output_format!r}")

    resolved_background = (background or "auto").strip().lower()
    if resolved_background not in SUPPORTED_BACKGROUNDS:
        raise BetterOpenRouterConfigError(f"Unsupported background: {resolved_background!r}")
    if resolved_background == "transparent" and resolved_output_format == "jpeg":
        raise BetterOpenRouterConfigError("Transparent backgrounds require PNG or WebP output, not JPEG.")

    if output_compression < -1 or output_compression > 100:
        raise BetterOpenRouterConfigError(
            f"`output_compression` must be -1 (automatic) or between 0 and 100; got {output_compression}."
        )
    if output_compression >= 0 and resolved_output_format == "png":
        raise BetterOpenRouterConfigError("`output_compression` only applies to JPEG or WebP output.")

    image_width = width if width and width > 0 else None
    image_height = height if height and height > 0 else None
    if (image_width is None) != (image_height is None):
        raise BetterOpenRouterConfigError(
            "`width` and `height` must be set together (both > 0), or both left as 0."
        )
    if image_width is not None and image_height is not None:
        if image_width > 8192 or image_height > 8192:
            raise BetterOpenRouterConfigError("`width` and `height` must not exceed 8192 pixels.")
        if resolved_aspect_ratio != "auto" or resolved_resolution != "auto":
            logger.info(
                "OpenRouter explicit size %dx%d overrides aspect_ratio=%s and resolution=%s.",
                image_width,
                image_height,
                resolved_aspect_ratio,
                resolved_resolution,
            )
        resolved_aspect_ratio = "auto"
        resolved_resolution = "auto"

    if n < 1 or n > MAX_IMAGES_PER_REQUEST:
        raise BetterOpenRouterConfigError(
            f"`n` must be between 1 and {MAX_IMAGES_PER_REQUEST}; got {n}."
        )
    if seed < 0:
        raise BetterOpenRouterConfigError(f"`seed` must be 0 (unset) or a positive integer; got {seed}.")

    resolved_input_images: list[bytes] = []
    if input_images is not None:
        for idx, image in enumerate(input_images):
            if image is None:
                continue
            if isinstance(image, memoryview):
                image = image.tobytes()
            elif isinstance(image, bytearray):
                image = bytes(image)
            if not isinstance(image, bytes):
                raise BetterOpenRouterConfigError(
                    f"`input_images[{idx}]` must be bytes; got {type(image)!r}."
                )
            if not image:
                raise BetterOpenRouterConfigError(f"`input_images[{idx}]` is empty.")
            resolved_input_images.append(image)

    return BetterOpenRouterRequest(
        model=model.strip(),
        prompt=prompt,
        aspect_ratio=resolved_aspect_ratio if resolved_aspect_ratio != "auto" else None,
        resolution=resolved_resolution if resolved_resolution != "auto" else None,
        image_width=image_width,
        image_height=image_height,
        quality=resolved_quality if resolved_quality != "auto" else None,
        output_format=resolved_output_format if resolved_output_format != "auto" else None,
        background=resolved_background if resolved_background != "auto" else None,
        output_compression=output_compression if output_compression >= 0 else None,
        n=n,
        seed=seed if seed > 0 else None,
        input_images=tuple(resolved_input_images),
    )
