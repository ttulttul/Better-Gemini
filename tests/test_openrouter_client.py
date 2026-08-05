import asyncio
import base64
import os
import unittest
from unittest import mock

from better_gemini.openrouter_client import (
    APP_TITLE,
    APP_URL,
    DEFAULT_MODEL,
    DEFAULT_MODELS,
    USER_AGENT,
    _MODEL_LIST_CACHE,
    _build_headers,
    _build_image_request,
    generate_images,
    generate_images_sync,
    list_models_sync,
)
from better_gemini.openrouter_core import BetterOpenRouterError, BetterOpenRouterRequest


class OpenRouterClientTests(unittest.TestCase):
    def setUp(self):
        _MODEL_LIST_CACHE.clear()

    def test_default_model_is_in_bundled_defaults(self):
        self.assertIn(DEFAULT_MODEL, DEFAULT_MODELS)

    def test_build_headers_includes_auth_and_app_metadata(self):
        headers = _build_headers(api_key="k", has_payload=True)
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertEqual(headers["User-Agent"], USER_AGENT)
        self.assertEqual(headers["HTTP-Referer"], APP_URL)
        self.assertEqual(headers["X-OpenRouter-Title"], APP_TITLE)
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_build_headers_allows_public_model_discovery(self):
        headers = _build_headers(api_key=None, has_payload=False)
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Content-Type", headers)

    def test_list_models_sync_parses_public_image_model_ids(self):
        with mock.patch(
            "better_gemini.openrouter_client._request_json",
            return_value={
                "data": [
                    {"id": "openai/gpt-image-2"},
                    {"id": "google/gemini-3.1-flash-image"},
                    {"id": "openai/gpt-image-2"},
                    {"name": "missing id"},
                ]
            },
        ) as request_json:
            models = list_models_sync(api_key=None, cache_ttl_s=0)
        self.assertEqual(
            models,
            ["google/gemini-3.1-flash-image", "openai/gpt-image-2"],
        )
        request_json.assert_called_once_with(
            method="GET",
            path="/images/models",
            api_key=None,
            timeout_s=10.0,
        )

    def test_build_image_request_includes_supported_parameters(self):
        payload = _build_image_request(
            BetterOpenRouterRequest(
                model="openai/gpt-image-2",
                prompt="city skyline",
                aspect_ratio="16:9",
                resolution="2K",
                quality="high",
                output_format="webp",
                background="opaque",
                output_compression=80,
                n=3,
                seed=42,
            )
        )
        self.assertEqual(
            payload,
            {
                "model": "openai/gpt-image-2",
                "prompt": "city skyline",
                "n": 3,
                "resolution": "2K",
                "aspect_ratio": "16:9",
                "quality": "high",
                "output_format": "webp",
                "background": "opaque",
                "output_compression": 80,
                "seed": 42,
            },
        )

    def test_build_image_request_uses_explicit_size(self):
        payload = _build_image_request(
            BetterOpenRouterRequest(
                model="openai/gpt-image-2",
                prompt="city skyline",
                image_width=1536,
                image_height=864,
            )
        )
        self.assertEqual(payload["size"], "1536x864")
        self.assertNotIn("resolution", payload)
        self.assertNotIn("aspect_ratio", payload)

    def test_build_image_request_encodes_reference_images(self):
        payload = _build_image_request(
            BetterOpenRouterRequest(
                model="openai/gpt-image-2",
                prompt="watercolor version",
                input_images=(b"one", b"two"),
            )
        )
        references = payload["input_references"]
        self.assertEqual(len(references), 2)
        self.assertEqual(references[0]["type"], "image_url")
        self.assertTrue(
            references[0]["image_url"]["url"].startswith("data:image/png;base64,")
        )

    def test_generate_images_sync_requires_api_key(self):
        prior_key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with self.assertRaises(BetterOpenRouterError):
                generate_images_sync(
                    api_key=None,
                    request=BetterOpenRouterRequest(
                        model="openai/gpt-image-2",
                        prompt="garden",
                    ),
                )
        finally:
            if prior_key is not None:
                os.environ["OPENROUTER_API_KEY"] = prior_key

    def test_generate_images_sync_decodes_images_and_reports_cost(self):
        image_bytes = b"fake-image-bytes"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        request = BetterOpenRouterRequest(
            model="openai/gpt-image-2",
            prompt="garden",
        )
        with mock.patch(
            "better_gemini.openrouter_client._request_json",
            return_value={
                "data": [{"b64_json": encoded, "media_type": "image/png"}],
                "usage": {"cost": 0.04},
            },
        ) as request_json:
            text, images = generate_images_sync(api_key="k", request=request)
        self.assertEqual(images, [image_bytes])
        self.assertIn("returned 1 image(s)", text)
        self.assertIn("$0.040000", text)
        request_json.assert_called_once_with(
            method="POST",
            path="/images",
            api_key="k",
            payload={"model": "openai/gpt-image-2", "prompt": "garden"},
        )

    def test_generate_images_sync_rejects_svg_output(self):
        encoded = base64.b64encode(b"<svg></svg>").decode("ascii")
        with mock.patch(
            "better_gemini.openrouter_client._request_json",
            return_value={
                "data": [{"b64_json": encoded, "media_type": "image/svg+xml"}],
            },
        ):
            with self.assertRaisesRegex(BetterOpenRouterError, "SVG output"):
                generate_images_sync(
                    api_key="k",
                    request=BetterOpenRouterRequest(
                        model="recraft/recraft-v4-vector",
                        prompt="logo",
                    ),
                )

    def test_generate_images_sync_rejects_invalid_base64(self):
        with mock.patch(
            "better_gemini.openrouter_client._request_json",
            return_value={"data": [{"b64_json": "not valid base64"}]},
        ):
            with self.assertRaisesRegex(BetterOpenRouterError, "invalid base64"):
                generate_images_sync(
                    api_key="k",
                    request=BetterOpenRouterRequest(
                        model="openai/gpt-image-2",
                        prompt="garden",
                    ),
                )

    def test_generate_images_sync_surfaces_empty_response(self):
        with mock.patch(
            "better_gemini.openrouter_client._request_json",
            return_value={"data": []},
        ):
            text, images = generate_images_sync(
                api_key="k",
                request=BetterOpenRouterRequest(
                    model="openai/gpt-image-2",
                    prompt="garden",
                ),
            )
        self.assertEqual(images, [])
        self.assertIn("returned no raster images", text)

    def test_generate_images_runs_sync_client_in_thread(self):
        async def run_test():
            request = BetterOpenRouterRequest(
                model="openai/gpt-image-2",
                prompt="garden",
            )
            with mock.patch(
                "better_gemini.openrouter_client.generate_images_sync",
                return_value=("done", [b"image"]),
            ) as generate_sync:
                result = await generate_images(api_key="k", request=request)
            self.assertEqual(result, ("done", [b"image"]))
            generate_sync.assert_called_once_with(api_key="k", request=request)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
