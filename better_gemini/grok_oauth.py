from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

logger = logging.getLogger(__name__)

OAUTH_ISSUER = "https://auth.x.ai"
OAUTH_DEVICE_URL = f"{OAUTH_ISSUER}/oauth2/device/code"
OAUTH_TOKEN_URL = f"{OAUTH_ISSUER}/oauth2/token"
OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
OAUTH_SCOPE = (
    "openid profile email offline_access grok-cli:access api:access "
    "conversations:read conversations:write"
)
OAUTH_DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
OAUTH_REFRESH_SKEW_SECONDS = 2 * 60
OAUTH_MAX_LOGIN_SECONDS = 15 * 60
OAUTH_REQUEST_TIMEOUT_SECONDS = 15
OAUTH_MAX_RESPONSE_BYTES = 64 * 1024
OAUTH_SLOW_DOWN_SECONDS = 5
USER_AGENT = "comfyui-better-gemini/1.3.0"
ALLOWED_VERIFICATION_ORIGINS = {OAUTH_ISSUER, "https://accounts.x.ai"}


class GrokOAuthError(RuntimeError):
    """A redacted xAI OAuth failure safe to display to the user."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class OAuthCredentials:
    access_token: str
    refresh_token: str
    expires_at: float
    token_type: str = "Bearer"

    @classmethod
    def from_token_payload(
        cls,
        payload: dict[str, Any],
        *,
        now: float,
        fallback_refresh_token: str = "",
    ) -> OAuthCredentials:
        access_token = _bounded_string(
            payload.get("access_token"), OAUTH_MAX_RESPONSE_BYTES
        )
        refresh_token = _bounded_string(
            payload.get("refresh_token"), OAUTH_MAX_RESPONSE_BYTES
        )
        refresh_token = refresh_token or fallback_refresh_token
        expires_in = payload.get("expires_in", 3600)
        token_type = _bounded_string(payload.get("token_type", "Bearer"), 64)
        if not access_token:
            raise GrokOAuthError("xAI token response did not include an access token.")
        if not refresh_token:
            raise GrokOAuthError("xAI token response did not include a refresh token.")
        if (
            not isinstance(expires_in, (int, float))
            or isinstance(expires_in, bool)
            or not math.isfinite(expires_in)
            or expires_in <= 0
        ):
            raise GrokOAuthError("xAI token response contained an invalid expiry.")
        if not token_type:
            raise GrokOAuthError("xAI token response contained an invalid token type.")
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=now + min(float(expires_in), 24 * 60 * 60),
            token_type=token_type,
        )


@dataclass(frozen=True)
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    interval_seconds: int
    expires_in_seconds: int


@dataclass
class LoginState:
    flow_id: str
    state: str
    user_code: str = ""
    verification_uri: str = ""
    expires_at: float = 0.0
    error: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "state": self.state,
            "user_code": self.user_code,
            "verification_uri": self.verification_uri,
            "expires_at": self.expires_at,
            "error": self.error,
        }


def _bounded_string(value: Any, maximum: int) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    return value


def _positive_integer(value: Any, maximum: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if int(value) != value or value <= 0 or value > maximum:
        return None
    return int(value)


def _validate_verification_uri(value: Any) -> str | None:
    uri = _bounded_string(value, 2048)
    if not uri:
        return None
    parsed = urlparse(uri)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if (
        parsed.scheme != "https"
        or origin not in ALLOWED_VERIFICATION_ORIGINS
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        return None
    return uri


def _decoded_value_contains_secret(value: str, secret: str) -> bool:
    current = value
    for _ in range(8):
        if secret in current:
            return True
        decoded = unquote(current)
        if decoded == current:
            return False
        current = decoded
    return True


def _oauth_headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": USER_AGENT,
        "X-Grok-Client-Version": "1.3.0",
        "X-Grok-Client-Surface": "ui",
    }


def _read_json_response(response: Any, *, label: str) -> dict[str, Any]:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type.lower():
        raise GrokOAuthError(f"{label} did not return JSON.")
    body = response.read(OAUTH_MAX_RESPONSE_BYTES + 1)
    if len(body) > OAUTH_MAX_RESPONSE_BYTES:
        raise GrokOAuthError(f"{label} was too large.")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GrokOAuthError(f"{label} returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise GrokOAuthError(f"{label} returned an invalid payload.")
    return payload


def _post_form(
    url: str, form: dict[str, str], *, label: str
) -> tuple[int, dict[str, Any]]:
    if url not in {OAUTH_DEVICE_URL, OAUTH_TOKEN_URL}:
        raise GrokOAuthError(
            "Refusing to send xAI credentials to an untrusted endpoint."
        )
    request = Request(
        url,
        data=urlencode(form).encode("utf-8"),
        headers=_oauth_headers(),
        method="POST",
    )
    opener = build_opener(_NoRedirectHandler())
    try:
        with opener.open(request, timeout=OAUTH_REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, _read_json_response(response, label=label)
    except HTTPError as exc:
        try:
            return exc.code, _read_json_response(exc, label=label)
        except GrokOAuthError:
            return exc.code, {}
    except (URLError, TimeoutError, OSError) as exc:
        raise GrokOAuthError(
            f"{label} failed; check the ComfyUI network connection."
        ) from exc


def request_device_authorization_sync() -> DeviceAuthorization:
    status, payload = _post_form(
        OAUTH_DEVICE_URL,
        {
            "client_id": OAUTH_CLIENT_ID,
            "scope": OAUTH_SCOPE,
            "referrer": "comfyui-better-gemini",
        },
        label="xAI device authorization request",
    )
    if status != 200:
        if status == 404:
            raise GrokOAuthError("xAI device authorization is not currently available.")
        raise GrokOAuthError(f"xAI device authorization failed with status {status}.")

    device_code = _bounded_string(payload.get("device_code"), 4096)
    user_code = _bounded_string(payload.get("user_code"), 128)
    verification_uri = _validate_verification_uri(payload.get("verification_uri"))
    expires_in = _positive_integer(payload.get("expires_in"), 24 * 60 * 60)
    interval = _positive_integer(payload.get("interval", 5), 24 * 60 * 60)
    if (
        not device_code
        or not user_code
        or not all(
            char.isascii() and (char.isalnum() or char == "-") for char in user_code
        )
        or not verification_uri
        or not expires_in
        or not interval
        or _decoded_value_contains_secret(verification_uri, device_code)
    ):
        raise GrokOAuthError("xAI device authorization returned an invalid response.")
    return DeviceAuthorization(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        interval_seconds=max(1, interval),
        expires_in_seconds=min(expires_in, OAUTH_MAX_LOGIN_SECONDS),
    )


async def poll_device_authorization(
    device: DeviceAuthorization,
    *,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], Any] = asyncio.sleep,
    post_form: Callable[..., tuple[int, dict[str, Any]]] = _post_form,
) -> OAuthCredentials:
    deadline = now() + min(device.expires_in_seconds, OAUTH_MAX_LOGIN_SECONDS)
    interval = max(1, device.interval_seconds)
    while now() < deadline:
        await sleep(min(interval, max(0, deadline - now())))
        if now() >= deadline:
            break
        status, payload = await asyncio.to_thread(
            post_form,
            OAUTH_TOKEN_URL,
            {
                "grant_type": OAUTH_DEVICE_GRANT_TYPE,
                "device_code": device.device_code,
                "client_id": OAUTH_CLIENT_ID,
            },
            label="xAI device token request",
        )
        if status == 200:
            return OAuthCredentials.from_token_payload(payload, now=now())
        error = _bounded_string(payload.get("error"), 128)
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            replacement = _positive_integer(payload.get("interval"), 24 * 60 * 60)
            interval = max(interval + OAUTH_SLOW_DOWN_SECONDS, replacement or 0)
            continue
        if error in {"access_denied", "authorization_denied"}:
            raise GrokOAuthError("xAI device authorization was denied.")
        if error == "expired_token":
            break
        raise GrokOAuthError(f"xAI device token request failed with status {status}.")
    raise GrokOAuthError("xAI device authorization expired; click Login to try again.")


def refresh_credentials_sync(credentials: OAuthCredentials) -> OAuthCredentials:
    status, payload = _post_form(
        OAUTH_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token,
            "client_id": OAUTH_CLIENT_ID,
        },
        label="xAI token refresh",
    )
    if status != 200:
        raise GrokOAuthError(
            f"xAI token refresh failed with status {status}; log out and log in again."
        )
    return OAuthCredentials.from_token_payload(
        payload,
        now=time.time(),
        fallback_refresh_token=credentials.refresh_token,
    )


def default_credentials_path() -> Path:
    try:
        import folder_paths  # type: ignore

        base = Path(folder_paths.get_system_user_directory("better_gemini"))
    except (ImportError, AttributeError):
        configured_user_dir = os.environ.get("COMFYUI_USER_DIRECTORY")
        base = (
            Path(configured_user_dir) / "__better_gemini"
            if configured_user_dir
            else Path.home() / ".comfyui" / "__better_gemini"
        )
    return base / "xai_oauth.json"


class CredentialStore:
    def __init__(self, path_resolver: Callable[[], Path] = default_credentials_path):
        self._path_resolver = path_resolver
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path_resolver()

    def load(self) -> OAuthCredentials | None:
        with self._lock:
            path = self.path
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, json.JSONDecodeError) as exc:
                raise GrokOAuthError(
                    "Stored xAI OAuth credentials could not be read."
                ) from exc
            if not isinstance(payload, dict):
                raise GrokOAuthError("Stored xAI OAuth credentials are invalid.")
            access_token = _bounded_string(
                payload.get("access_token"), OAUTH_MAX_RESPONSE_BYTES
            )
            refresh_token = _bounded_string(
                payload.get("refresh_token"), OAUTH_MAX_RESPONSE_BYTES
            )
            token_type = _bounded_string(payload.get("token_type", "Bearer"), 64)
            try:
                expires_at = float(payload["expires_at"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GrokOAuthError(
                    "Stored xAI OAuth credentials are invalid."
                ) from exc
            if (
                not access_token
                or not refresh_token
                or not token_type
                or not math.isfinite(expires_at)
                or expires_at <= 0
            ):
                raise GrokOAuthError("Stored xAI OAuth credentials are invalid.")
            return OAuthCredentials(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                token_type=token_type,
            )

    def save(self, credentials: OAuthCredentials) -> None:
        with self._lock:
            path = self.path
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                os.chmod(path.parent, 0o700)
            except OSError:
                logger.debug(
                    "Could not tighten xAI OAuth credential directory permissions.",
                    exc_info=True,
                )
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            data = json.dumps(asdict(credentials), separators=(",", ":"))
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                os.chmod(path, 0o600)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def clear(self) -> bool:
        with self._lock:
            try:
                self.path.unlink()
                return True
            except FileNotFoundError:
                return False
            except OSError as exc:
                raise GrokOAuthError(
                    "Stored xAI OAuth credentials could not be removed."
                ) from exc


class GrokOAuthManager:
    def __init__(self, store: CredentialStore | None = None):
        self.store = store or CredentialStore()
        self._state_lock = threading.RLock()
        self._login_state: LoginState | None = None
        self._login_task: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[OAuthCredentials] | None = None
        self._generation = 0

    def has_credentials(self) -> bool:
        return self.store.load() is not None

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            login = self._login_state.public_dict() if self._login_state else None
        return {"authenticated": self.has_credentials(), "login": login}

    async def start_login(self) -> dict[str, Any]:
        device = await asyncio.to_thread(request_device_authorization_sync)
        flow_id = uuid.uuid4().hex
        state = LoginState(
            flow_id=flow_id,
            state="pending",
            user_code=device.user_code,
            verification_uri=device.verification_uri,
            expires_at=time.time() + device.expires_in_seconds,
        )
        with self._state_lock:
            if self._login_task and not self._login_task.done():
                self._login_task.cancel()
            self._generation += 1
            generation = self._generation
            self._login_state = state
            self._login_task = asyncio.create_task(
                self._complete_login(flow_id, generation, device),
                name="better-grok-oauth-login",
            )
        logger.info("Started xAI device authorization for BetterGrok.")
        return state.public_dict()

    async def _complete_login(
        self,
        flow_id: str,
        generation: int,
        device: DeviceAuthorization,
    ) -> None:
        try:
            credentials = await poll_device_authorization(device)
            with self._state_lock:
                if (
                    generation != self._generation
                    or self._login_state is None
                    or self._login_state.flow_id != flow_id
                ):
                    return
                self.store.save(credentials)
                self._login_state.state = "authenticated"
                self._login_state.user_code = ""
                self._login_state.verification_uri = ""
            logger.info("BetterGrok xAI OAuth login completed.")
        except asyncio.CancelledError:
            logger.debug("BetterGrok xAI OAuth login was cancelled.")
            raise
        except Exception as exc:  # noqa: BLE001 - background task errors must become redacted UI state
            message = (
                str(exc)
                if isinstance(exc, GrokOAuthError)
                else "xAI OAuth login failed."
            )
            with self._state_lock:
                if (
                    generation == self._generation
                    and self._login_state is not None
                    and self._login_state.flow_id == flow_id
                ):
                    self._login_state.state = "error"
                    self._login_state.error = message
                    self._login_state.user_code = ""
                    self._login_state.verification_uri = ""
            logger.warning("BetterGrok xAI OAuth login failed: %s", message)

    def login_status(self, flow_id: str) -> dict[str, Any]:
        with self._state_lock:
            if not self._login_state or self._login_state.flow_id != flow_id:
                raise GrokOAuthError("xAI OAuth login session was not found.")
            return self._login_state.public_dict()

    async def logout(self) -> None:
        with self._state_lock:
            self._generation += 1
            if self._login_task and not self._login_task.done():
                self._login_task.cancel()
            if self._refresh_task and not self._refresh_task.done():
                self._refresh_task.cancel()
            self._login_state = None
            self._login_task = None
            self._refresh_task = None
            self.store.clear()
        logger.info("Removed BetterGrok xAI OAuth credentials.")

    async def get_access_token(self) -> str:
        credentials = self.store.load()
        if credentials is None:
            raise GrokOAuthError(
                "BetterGrok is not logged in to xAI. Click Login on the node first."
            )

        now = time.time()
        must_wait = credentials.expires_at <= now
        should_refresh = credentials.expires_at <= now + OAUTH_REFRESH_SKEW_SECONDS
        if not should_refresh:
            return credentials.access_token

        task = self._get_or_start_refresh(credentials)
        if must_wait:
            return (await task).access_token
        return credentials.access_token

    def _get_or_start_refresh(
        self, credentials: OAuthCredentials
    ) -> asyncio.Task[OAuthCredentials]:
        with self._state_lock:
            if self._refresh_task and not self._refresh_task.done():
                return self._refresh_task
            generation = self._generation
            task = asyncio.create_task(
                self._refresh_and_store(credentials, generation),
                name="better-grok-oauth-refresh",
            )
            task.add_done_callback(self._log_background_refresh_result)
            self._refresh_task = task
            return task

    async def _refresh_and_store(
        self,
        credentials: OAuthCredentials,
        generation: int,
    ) -> OAuthCredentials:
        refreshed = await asyncio.to_thread(refresh_credentials_sync, credentials)
        with self._state_lock:
            if generation != self._generation:
                raise GrokOAuthError("xAI OAuth credentials changed while refreshing.")
            self.store.save(refreshed)
        logger.info("Refreshed BetterGrok xAI OAuth credentials.")
        return refreshed

    @staticmethod
    def _log_background_refresh_result(task: asyncio.Task[OAuthCredentials]) -> None:
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:  # noqa: BLE001 - callback must consume every task failure
            logger.warning("Background BetterGrok xAI OAuth refresh failed: %s", exc)


oauth_manager = GrokOAuthManager()


async def resolve_grok_credential(
    *,
    api_key: str,
    auth_mode: str,
    allow_oauth: bool = True,
) -> str | None:
    mode = auth_mode.strip().lower().replace(" ", "_")
    explicit_key = api_key.strip()
    if mode not in {"auto", "oauth", "api_key"}:
        raise GrokOAuthError(
            f"Unsupported BetterGrok authentication mode: {auth_mode!r}."
        )
    if mode == "api_key":
        return explicit_key or None
    if mode == "oauth":
        if not allow_oauth:
            raise GrokOAuthError(
                "BetterGrok OAuth is available for xAI Imagine image requests. "
                "Use API key authentication for TEXT mode."
            )
        return await oauth_manager.get_access_token()
    if explicit_key:
        return explicit_key
    if allow_oauth and oauth_manager.has_credentials():
        return await oauth_manager.get_access_token()
    return None
