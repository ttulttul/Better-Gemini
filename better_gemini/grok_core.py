from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


class BetterGrokError(RuntimeError):
    pass


class BetterGrokConfigError(ValueError):
    pass


SUPPORTED_ASPECT_RATIOS: tuple[str, ...] = (
    "auto",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "2:1",
    "1:2",
    "19.5:9",
    "9:19.5",
    "20:9",
    "9:20",
)

SUPPORTED_IMAGE_RESOLUTIONS: tuple[str, ...] = (
    "auto",
    "1k",
    "2k",
)

MAX_EDIT_IMAGES = 5
MAX_IMAGES_PER_REQUEST = 10
SUPPORTED_RESPONSE_MODALITIES: frozenset[str] = frozenset({"IMAGE", "TEXT"})


@dataclass(frozen=True)
class BetterGrokRequest:
    model: str
    prompt: str
    response_modalities: tuple[str, ...] = ("IMAGE",)
    aspect_ratio: str | None = None
    resolution: str | None = None
    n: int = 1
    input_images: tuple[bytes, ...] = ()


def build_request(
    *,
    model: str,
    prompt: str,
    response_modalities: str = "IMAGE",
    aspect_ratio: str = "auto",
    resolution: str = "auto",
    n: int = 1,
    input_images: Iterable[bytes] | None = None,
) -> BetterGrokRequest:
    if not prompt or not isinstance(prompt, str):
        raise BetterGrokConfigError("`prompt` must be a non-empty string.")
    if not model or not isinstance(model, str):
        raise BetterGrokConfigError("`model` must be a non-empty string.")

    modalities = tuple(dict.fromkeys(m.strip().upper() for m in response_modalities.split("+") if m.strip()))
    if not modalities:
        raise BetterGrokConfigError("`response_modalities` must include at least one modality.")
    unsupported_modalities = tuple(modality for modality in modalities if modality not in SUPPORTED_RESPONSE_MODALITIES)
    if unsupported_modalities:
        raise BetterGrokConfigError(
            "Unsupported response modality value(s): {}.".format(
                ", ".join(repr(modality) for modality in unsupported_modalities)
            )
        )

    resolved_aspect_ratio = (aspect_ratio or "auto").strip()
    if resolved_aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        raise BetterGrokConfigError(f"Unsupported aspect ratio: {resolved_aspect_ratio!r}")

    resolved_resolution = (resolution or "auto").strip().lower()
    if resolved_resolution not in SUPPORTED_IMAGE_RESOLUTIONS:
        raise BetterGrokConfigError(f"Unsupported resolution: {resolved_resolution!r}")

    if n < 1 or n > MAX_IMAGES_PER_REQUEST:
        raise BetterGrokConfigError(
            f"`n` must be between 1 and {MAX_IMAGES_PER_REQUEST}; got {n}."
        )

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
                raise BetterGrokConfigError(f"`input_images[{idx}]` must be bytes; got {type(image)!r}.")
            if not image:
                raise BetterGrokConfigError(f"`input_images[{idx}]` is empty.")
            resolved_input_images.append(image)

    if len(resolved_input_images) > MAX_EDIT_IMAGES:
        raise BetterGrokConfigError(
            f"xAI image edits support up to {MAX_EDIT_IMAGES} input images; got {len(resolved_input_images)}."
        )

    image_requested = "IMAGE" in modalities
    return BetterGrokRequest(
        model=model.strip(),
        prompt=prompt,
        response_modalities=modalities,
        aspect_ratio=resolved_aspect_ratio if image_requested and resolved_aspect_ratio != "auto" else None,
        resolution=resolved_resolution if image_requested and resolved_resolution != "auto" else None,
        n=n if image_requested else 1,
        input_images=tuple(resolved_input_images),
    )
