import unittest
from unittest import mock

from better_gemini import extension
from better_gemini.genai_client import DEFAULT_MODELS as GEMINI_DEFAULT_MODELS
from better_gemini.grok_client import DEFAULT_MODELS as GROK_DEFAULT_MODELS


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        extension._warned_gemini_model_listing = False
        extension._warned_grok_model_listing = False

    def test_model_dropdown_options_falls_back_to_bundled_defaults_on_error(self):
        with mock.patch.object(extension, "list_gemini_models_sync", side_effect=RuntimeError("boom")):
            options = extension._model_dropdown_options()
        self.assertEqual(options, GEMINI_DEFAULT_MODELS)

    def test_model_dropdown_options_falls_back_to_bundled_defaults_on_empty_list(self):
        with mock.patch.object(extension, "list_gemini_models_sync", return_value=[]):
            options = extension._model_dropdown_options()
        self.assertEqual(options, GEMINI_DEFAULT_MODELS)

    def test_model_dropdown_options_prepends_bundled_defaults_when_listing_succeeds(self):
        with mock.patch.object(
            extension,
            "list_gemini_models_sync",
            return_value=[
                "models/custom-image-model",
                "models/gemini-3-pro-image-preview",
            ],
        ):
            options = extension._model_dropdown_options()
        self.assertEqual(
            options,
            [
                "models/gemini-3.1-flash-lite-image",
                "models/gemini-3-flash-preview",
                "models/gemini-3.1-flash-image-preview",
                "models/gemini-3.1-flash-lite-preview",
                "models/gemini-3-pro-image-preview",
                "models/gemini-3.1-pro-preview",
                "models/imagen-4.0-generate-001",
                "models/imagen-4.0-ultra-generate-001",
                "models/custom-image-model",
            ],
        )

    def test_placeholder_dimensions_returns_minimal_empty_placeholder(self):
        self.assertEqual(
            extension._placeholder_dimensions(
                requested_aspect_ratio="16:9",
                requested_resolution="4K",
                requested_width=1920,
                requested_height=1080,
                empty_placeholder=True,
            ),
            (1, 1),
        )

    def test_grok_model_dropdown_options_falls_back_to_bundled_defaults_on_error(self):
        with mock.patch.object(extension, "list_grok_models_sync", side_effect=RuntimeError("boom")):
            options = extension._grok_model_dropdown_options()
        self.assertEqual(options, GROK_DEFAULT_MODELS)

    def test_grok_model_dropdown_options_falls_back_to_bundled_defaults_on_empty_list(self):
        with mock.patch.object(extension, "list_grok_models_sync", return_value=[]):
            options = extension._grok_model_dropdown_options()
        self.assertEqual(options, GROK_DEFAULT_MODELS)

    def test_grok_model_dropdown_options_prepends_bundled_defaults_when_listing_succeeds(self):
        with mock.patch.object(
            extension,
            "list_grok_models_sync",
            return_value=[
                "grok-imagine-image-ultra",
                "grok-imagine-image",
                "grok-4",
                "grok-5-experimental",
            ],
        ):
            options = extension._grok_model_dropdown_options()
        self.assertEqual(
            options,
            [
                "grok-imagine-image",
                "grok-imagine-image-pro",
                "grok-imagine-image-quality",
                "grok-latest",
                "grok-4",
                "grok-4-fast-non-reasoning",
                "grok-3-mini",
                "grok-code-fast-1",
                "grok-imagine-image-ultra",
                "grok-5-experimental",
            ]
        )


class ExtensionCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_or_generate_output_returns_cached_output_without_calling_provider(self):
        request = object()

        async def fail_generate():
            raise AssertionError("provider should not be called")

        with mock.patch.object(extension, "request_cache_key", return_value="cache-key"), mock.patch.object(
            extension, "load_cached_output", return_value=("cached text", [b"cached image"])
        ) as load_cached, mock.patch.object(extension, "store_cached_output") as store_cached:
            text, images = await extension._get_or_generate_output(
                cache_outputs=True,
                provider="grok",
                request=request,
                extra_cache_data=None,
                generate_fn=fail_generate,
            )

        self.assertEqual((text, images), ("cached text", [b"cached image"]))
        load_cached.assert_called_once_with("cache-key")
        store_cached.assert_not_called()

    async def test_get_or_generate_output_stores_cache_miss_result(self):
        request = object()

        async def generate():
            return "fresh text", [b"fresh image"]

        with mock.patch.object(extension, "request_cache_key", return_value="cache-key"), mock.patch.object(
            extension, "load_cached_output", return_value=None
        ), mock.patch.object(extension, "store_cached_output") as store_cached:
            text, images = await extension._get_or_generate_output(
                cache_outputs=True,
                provider="gemini",
                request=request,
                extra_cache_data={"system_prompt": "sys"},
                generate_fn=generate,
            )

        self.assertEqual((text, images), ("fresh text", [b"fresh image"]))
        store_cached.assert_called_once_with("cache-key", text="fresh text", images=[b"fresh image"])

    async def test_get_or_generate_output_bypasses_cache_when_disabled(self):
        async def generate():
            return "fresh text", []

        with mock.patch.object(extension, "load_cached_output") as load_cached, mock.patch.object(
            extension, "store_cached_output"
        ) as store_cached:
            text, images = await extension._get_or_generate_output(
                cache_outputs=False,
                provider="gemini",
                request=object(),
                extra_cache_data=None,
                generate_fn=generate,
            )

        self.assertEqual((text, images), ("fresh text", []))
        load_cached.assert_not_called()
        store_cached.assert_not_called()


if __name__ == "__main__":
    unittest.main()
