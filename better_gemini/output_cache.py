from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
CACHE_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    return value


def request_cache_key(*, provider: str, request: Any, extra: dict[str, Any] | None = None) -> str:
    payload = {
        "provider": provider,
        "request": _canonicalize(request),
        "extra": _canonicalize(extra or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(encoded)


def load_cached_output(cache_key: str, *, cache_dir: Path = CACHE_DIR) -> tuple[str, list[bytes]] | None:
    manifest_path = cache_dir / f"{cache_key}.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Ignoring unreadable model output cache manifest: %s", manifest_path, exc_info=True)
        return None

    if manifest.get("version") != CACHE_VERSION:
        logger.debug("Ignoring model output cache manifest with unsupported version: %s", manifest_path)
        return None

    try:
        text_digest = manifest["text_sha256"]
        image_digests = manifest["image_sha256s"]
        if not isinstance(text_digest, str) or not isinstance(image_digests, list):
            raise ValueError("invalid manifest shape")

        text_path = cache_dir / "strings" / f"{text_digest}.txt"
        text_bytes = text_path.read_bytes()
        if _sha256_bytes(text_bytes) != text_digest:
            raise ValueError(f"cached text digest mismatch for {text_path}")
        text = text_bytes.decode("utf-8")

        images: list[bytes] = []
        for image_digest in image_digests:
            if not isinstance(image_digest, str):
                raise ValueError("invalid image digest")
            image_path = cache_dir / "images" / f"{image_digest}.bin"
            image_bytes = image_path.read_bytes()
            if _sha256_bytes(image_bytes) != image_digest:
                raise ValueError(f"cached image digest mismatch for {image_path}")
            images.append(image_bytes)
    except Exception:
        logger.warning("Ignoring incomplete model output cache entry: %s", manifest_path, exc_info=True)
        return None

    logger.info("Using cached model output for request %s.", cache_key)
    return text, images


def store_cached_output(
    cache_key: str,
    *,
    text: str,
    images: list[bytes],
    cache_dir: Path = CACHE_DIR,
) -> None:
    strings_dir = cache_dir / "strings"
    images_dir = cache_dir / "images"
    strings_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    text_bytes = text.encode("utf-8")
    text_digest = _sha256_bytes(text_bytes)
    (strings_dir / f"{text_digest}.txt").write_bytes(text_bytes)

    image_digests: list[str] = []
    for image in images:
        image_digest = _sha256_bytes(image)
        (images_dir / f"{image_digest}.bin").write_bytes(image)
        image_digests.append(image_digest)

    manifest = {
        "version": CACHE_VERSION,
        "text_sha256": text_digest,
        "image_sha256s": image_digests,
    }
    manifest_path = cache_dir / f"{cache_key}.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    logger.info("Stored model output cache entry for request %s.", cache_key)
