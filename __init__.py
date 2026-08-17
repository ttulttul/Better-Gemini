from __future__ import annotations

import logging

from .better_gemini.extension import comfy_entrypoint as comfy_entrypoint
from .better_gemini.grok_oauth_routes import register_oauth_routes

logger = logging.getLogger(__name__)

WEB_DIRECTORY = "./web"

register_oauth_routes()

# Ensure ComfyUI doesn't treat this as a V1 node pack.
NODE_CLASS_MAPPINGS = None
NODE_DISPLAY_NAME_MAPPINGS = None
