from __future__ import annotations

import logging

from .better_gemini.extension import comfy_entrypoint

logger = logging.getLogger(__name__)

# Ensure ComfyUI doesn't treat this as a V1 node pack.
NODE_CLASS_MAPPINGS = None
NODE_DISPLAY_NAME_MAPPINGS = None

