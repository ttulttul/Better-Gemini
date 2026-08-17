from __future__ import annotations

import logging
from typing import Any

from .grok_oauth import GrokOAuthError, oauth_manager

logger = logging.getLogger(__name__)

ROUTE_PREFIX = "/better-gemini/grok/oauth"
REQUEST_HEADER = "X-Better-Gemini-Request"
_registered_server_ids: set[int] = set()


def _require_frontend_request(request: Any) -> None:
    if request.headers.get(REQUEST_HEADER) != "1":
        raise GrokOAuthError("Missing BetterGrok frontend request header.")


def register_oauth_routes(prompt_server: Any = None) -> bool:
    """Register BetterGrok OAuth routes once ComfyUI's PromptServer exists."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        logger.debug(
            "ComfyUI HTTP runtime is unavailable; BetterGrok OAuth routes were not registered."
        )
        return False

    server = prompt_server or getattr(PromptServer, "instance", None)
    if server is None:
        logger.debug(
            "PromptServer is not ready; BetterGrok OAuth routes were not registered."
        )
        return False
    server_id = id(server)
    if server_id in _registered_server_ids:
        return True

    async def status_handler(_request: Any):
        try:
            return web.json_response(oauth_manager.status())
        except GrokOAuthError as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def login_handler(request: Any):
        try:
            _require_frontend_request(request)
            login = await oauth_manager.start_login()
            return web.json_response({"authenticated": False, "login": login})
        except GrokOAuthError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def login_status_handler(request: Any):
        try:
            login = oauth_manager.login_status(request.match_info["flow_id"])
            return web.json_response(
                {"authenticated": oauth_manager.has_credentials(), "login": login}
            )
        except GrokOAuthError as exc:
            return web.json_response({"error": str(exc)}, status=404)

    async def logout_handler(request: Any):
        try:
            _require_frontend_request(request)
            await oauth_manager.logout()
            return web.json_response({"authenticated": False, "login": None})
        except GrokOAuthError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    server.routes.get(f"{ROUTE_PREFIX}/status")(status_handler)
    server.routes.post(f"{ROUTE_PREFIX}/login")(login_handler)
    server.routes.get(f"{ROUTE_PREFIX}/login/{{flow_id}}")(login_status_handler)
    server.routes.post(f"{ROUTE_PREFIX}/logout")(logout_handler)
    _registered_server_ids.add(server_id)
    logger.info("Registered BetterGrok xAI OAuth routes.")
    return True
