import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from better_gemini import grok_oauth, grok_oauth_routes


def credentials(*, expires_at: float, access: str = "access", refresh: str = "refresh"):
    return grok_oauth.OAuthCredentials(
        access_token=access,
        refresh_token=refresh,
        expires_at=expires_at,
    )


class OAuthPayloadTests(unittest.TestCase):
    def test_device_authorization_validates_and_normalizes_response(self):
        with mock.patch.object(
            grok_oauth,
            "_post_form",
            return_value=(
                200,
                {
                    "device_code": "opaque-device-code",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://auth.x.ai/device",
                    "expires_in": 1200,
                    "interval": 5,
                },
            ),
        ):
            device = grok_oauth.request_device_authorization_sync()

        self.assertEqual(device.user_code, "ABCD-1234")
        self.assertEqual(device.verification_uri, "https://auth.x.ai/device")
        self.assertEqual(device.expires_in_seconds, grok_oauth.OAUTH_MAX_LOGIN_SECONDS)

    def test_device_authorization_rejects_untrusted_verification_origin(self):
        with (
            mock.patch.object(
                grok_oauth,
                "_post_form",
                return_value=(
                    200,
                    {
                        "device_code": "opaque-device-code",
                        "user_code": "ABCD-1234",
                        "verification_uri": "https://example.com/device",
                        "expires_in": 600,
                        "interval": 5,
                    },
                ),
            ),
            self.assertRaisesRegex(grok_oauth.GrokOAuthError, "invalid response"),
        ):
            grok_oauth.request_device_authorization_sync()

    def test_device_authorization_does_not_reflect_encoded_device_secret(self):
        with (
            mock.patch.object(
                grok_oauth,
                "_post_form",
                return_value=(
                    200,
                    {
                        "device_code": "opaque/device",
                        "user_code": "ABCD-1234",
                        "verification_uri": "https://auth.x.ai/device?value=opaque%252Fdevice",
                        "expires_in": 600,
                        "interval": 5,
                    },
                ),
            ),
            self.assertRaisesRegex(grok_oauth.GrokOAuthError, "invalid response"),
        ):
            grok_oauth.request_device_authorization_sync()

    def test_credentials_preserve_rotating_refresh_fallback(self):
        result = grok_oauth.OAuthCredentials.from_token_payload(
            {"access_token": "new-access", "expires_in": 600},
            now=100.0,
            fallback_refresh_token="old-refresh",
        )
        self.assertEqual(result.access_token, "new-access")
        self.assertEqual(result.refresh_token, "old-refresh")
        self.assertEqual(result.expires_at, 700.0)


class CredentialStoreTests(unittest.TestCase):
    def test_store_round_trip_uses_private_permissions_and_clear(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "secrets" / "xai_oauth.json"
            store = grok_oauth.CredentialStore(lambda: path)
            original = credentials(expires_at=1234.5)

            store.save(original)

            self.assertEqual(store.load(), original)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            self.assertNotIn(
                "access_token",
                [item.name for item in path.parent.iterdir() if item != path],
            )
            self.assertTrue(store.clear())
            self.assertFalse(store.clear())

    def test_store_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "xai_oauth.json"
            path.write_text("not json", encoding="utf-8")
            store = grok_oauth.CredentialStore(lambda: path)
            with self.assertRaisesRegex(grok_oauth.GrokOAuthError, "could not be read"):
                store.load()

    def test_store_rejects_non_string_tokens(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "xai_oauth.json"
            path.write_text(
                '{"access_token":null,"refresh_token":"refresh","expires_at":1234}',
                encoding="utf-8",
            )
            store = grok_oauth.CredentialStore(lambda: path)
            with self.assertRaisesRegex(grok_oauth.GrokOAuthError, "credentials are invalid"):
                store.load()


class DevicePollingTests(unittest.IsolatedAsyncioTestCase):
    async def test_poll_waits_for_approval_then_returns_credentials(self):
        clock = [100.0]
        responses = iter(
            [
                (400, {"error": "authorization_pending"}),
                (
                    200,
                    {
                        "access_token": "approved-access",
                        "refresh_token": "approved-refresh",
                        "expires_in": 3600,
                    },
                ),
            ]
        )

        async def sleep(seconds):
            clock[0] += seconds

        result = await grok_oauth.poll_device_authorization(
            grok_oauth.DeviceAuthorization(
                device_code="opaque",
                user_code="CODE",
                verification_uri="https://auth.x.ai/device",
                interval_seconds=2,
                expires_in_seconds=30,
            ),
            now=lambda: clock[0],
            sleep=sleep,
            post_form=lambda *args, **kwargs: next(responses),
        )

        self.assertEqual(result.access_token, "approved-access")
        self.assertEqual(result.refresh_token, "approved-refresh")
        self.assertEqual(clock[0], 104.0)

    async def test_poll_honors_slow_down(self):
        clock = [0.0]
        sleeps = []
        responses = iter(
            [
                (400, {"error": "slow_down"}),
                (
                    200,
                    {
                        "access_token": "approved-access",
                        "refresh_token": "approved-refresh",
                    },
                ),
            ]
        )

        async def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        await grok_oauth.poll_device_authorization(
            grok_oauth.DeviceAuthorization(
                device_code="opaque",
                user_code="CODE",
                verification_uri="https://auth.x.ai/device",
                interval_seconds=1,
                expires_in_seconds=30,
            ),
            now=lambda: clock[0],
            sleep=sleep,
            post_form=lambda *args, **kwargs: next(responses),
        )

        self.assertEqual(sleeps, [1, 6])

    async def test_poll_reports_denial_without_exposing_payload(self):
        clock = [0.0]

        async def sleep(seconds):
            clock[0] += seconds

        with self.assertRaisesRegex(grok_oauth.GrokOAuthError, "was denied"):
            await grok_oauth.poll_device_authorization(
                grok_oauth.DeviceAuthorization(
                    device_code="opaque",
                    user_code="CODE",
                    verification_uri="https://auth.x.ai/device",
                    interval_seconds=1,
                    expires_in_seconds=30,
                ),
                now=lambda: clock[0],
                sleep=sleep,
                post_form=lambda *args, **kwargs: (400, {"error": "access_denied"}),
            )


class OAuthManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "xai_oauth.json"
        self.store = grok_oauth.CredentialStore(lambda: self.path)
        self.manager = grok_oauth.GrokOAuthManager(self.store)

    def tearDown(self):
        self.temporary_directory.cleanup()

    async def test_login_task_persists_tokens_and_updates_status(self):
        device = grok_oauth.DeviceAuthorization(
            device_code="opaque",
            user_code="ABCD-1234",
            verification_uri="https://auth.x.ai/device",
            interval_seconds=1,
            expires_in_seconds=60,
        )
        approved = credentials(expires_at=time.time() + 3600)
        with (
            mock.patch.object(
                grok_oauth, "request_device_authorization_sync", return_value=device
            ),
            mock.patch.object(
                grok_oauth,
                "poll_device_authorization",
                new=mock.AsyncMock(return_value=approved),
            ),
        ):
            login = await self.manager.start_login()
            await self.manager._login_task

        self.assertEqual(login["user_code"], "ABCD-1234")
        self.assertTrue(self.manager.status()["authenticated"])
        self.assertEqual(
            self.manager.login_status(login["flow_id"])["state"], "authenticated"
        )
        self.assertEqual(self.store.load(), approved)

    async def test_fresh_token_does_not_refresh(self):
        current = credentials(expires_at=time.time() + 3600)
        self.store.save(current)
        with mock.patch.object(grok_oauth, "refresh_credentials_sync") as refresh:
            token = await self.manager.get_access_token()
        self.assertEqual(token, "access")
        refresh.assert_not_called()

    async def test_near_expiry_token_refreshes_in_background(self):
        current = credentials(expires_at=time.time() + 30)
        refreshed = credentials(
            expires_at=time.time() + 3600, access="new-access", refresh="new-refresh"
        )
        self.store.save(current)
        with mock.patch.object(
            grok_oauth, "refresh_credentials_sync", return_value=refreshed
        ):
            token = await self.manager.get_access_token()
            await self.manager._refresh_task
        self.assertEqual(token, "access")
        self.assertEqual(self.store.load(), refreshed)

    async def test_expired_token_waits_for_refresh(self):
        current = credentials(expires_at=time.time() - 1)
        refreshed = credentials(expires_at=time.time() + 3600, access="new-access")
        self.store.save(current)
        with mock.patch.object(
            grok_oauth, "refresh_credentials_sync", return_value=refreshed
        ):
            token = await self.manager.get_access_token()
        self.assertEqual(token, "new-access")

    async def test_logout_clears_credentials(self):
        self.store.save(credentials(expires_at=time.time() + 3600))
        await self.manager.logout()
        self.assertFalse(self.manager.has_credentials())


class CredentialResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_prefers_explicit_api_key(self):
        fake_manager = mock.Mock()
        fake_manager.has_credentials.return_value = True
        fake_manager.get_access_token = mock.AsyncMock(return_value="oauth")
        with mock.patch.object(grok_oauth, "oauth_manager", fake_manager):
            token = await grok_oauth.resolve_grok_credential(
                api_key=" explicit ", auth_mode="auto"
            )
        self.assertEqual(token, "explicit")
        fake_manager.get_access_token.assert_not_awaited()

    async def test_oauth_mode_requires_oauth_token(self):
        fake_manager = mock.Mock()
        fake_manager.get_access_token = mock.AsyncMock(return_value="oauth-access")
        with mock.patch.object(grok_oauth, "oauth_manager", fake_manager):
            token = await grok_oauth.resolve_grok_credential(
                api_key="ignored", auth_mode="oauth"
            )
        self.assertEqual(token, "oauth-access")

    async def test_api_key_mode_leaves_environment_fallback_to_client(self):
        token = await grok_oauth.resolve_grok_credential(
            api_key="", auth_mode="api_key"
        )
        self.assertIsNone(token)

    async def test_oauth_mode_rejects_text_only_public_api_route(self):
        with self.assertRaisesRegex(grok_oauth.GrokOAuthError, "Imagine image requests"):
            await grok_oauth.resolve_grok_credential(
                api_key="",
                auth_mode="oauth",
                allow_oauth=False,
            )


class FrontendContractTests(unittest.TestCase):
    def test_frontend_adds_node_chrome_login_logout_and_never_handles_tokens(self):
        source = (Path(__file__).parents[1] / "web" / "grok_oauth.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("buttonLabel()", source)
        self.assertIn('return "Logout"', source)
        self.assertIn('return "Login"', source)
        self.assertIn(".lg-node-header > div", source)
        self.assertIn("X-Better-Gemini-Request", source)
        self.assertNotIn("access_token", source)
        self.assertNotIn("refresh_token", source)


class OAuthRouteContractTests(unittest.TestCase):
    def test_routes_register_once_with_prompt_server(self):
        class Routes:
            def __init__(self):
                self.registered = []

            def get(self, path):
                return self._capture("GET", path)

            def post(self, path):
                return self._capture("POST", path)

            def _capture(self, method, path):
                def decorator(handler):
                    self.registered.append((method, path, handler))
                    return handler

                return decorator

        fake_server = types.SimpleNamespace(routes=Routes())
        fake_aiohttp = types.ModuleType("aiohttp")
        fake_aiohttp.web = types.SimpleNamespace()
        fake_server_module = types.ModuleType("server")
        fake_server_module.PromptServer = types.SimpleNamespace(instance=None)
        grok_oauth_routes._registered_server_ids.clear()

        with mock.patch.dict(
            sys.modules,
            {"aiohttp": fake_aiohttp, "server": fake_server_module},
        ):
            self.assertTrue(grok_oauth_routes.register_oauth_routes(fake_server))
            self.assertTrue(grok_oauth_routes.register_oauth_routes(fake_server))

        self.assertEqual(
            [(method, path) for method, path, _handler in fake_server.routes.registered],
            [
                ("GET", "/better-gemini/grok/oauth/status"),
                ("POST", "/better-gemini/grok/oauth/login"),
                ("GET", "/better-gemini/grok/oauth/login/{flow_id}"),
                ("POST", "/better-gemini/grok/oauth/logout"),
            ],
        )

    def test_mutations_require_frontend_header(self):
        request = types.SimpleNamespace(headers={})
        with self.assertRaisesRegex(grok_oauth.GrokOAuthError, "frontend request header"):
            grok_oauth_routes._require_frontend_request(request)


if __name__ == "__main__":
    unittest.main()
