import base64
import os
import unittest
from unittest import mock

from better_gemini.grok_client import (
    DEFAULT_MODEL,
    DEFAULT_MODELS,
    _MODEL_LIST_CACHE,
    _build_image_request,
    generate_images_sync,
    list_models_sync,
)
from better_gemini.grok_core import BetterGrokError, BetterGrokRequest


class GrokClientTests(unittest.TestCase):
    def setUp(self):
        _MODEL_LIST_CACHE.clear()

    def test_default_model_is_in_bundled_default_models(self):
        self.assertIn(DEFAULT_MODEL, DEFAULT_MODELS)

    def test_list_models_sync_parses_ids_and_aliases(self):
        with mock.patch(
            "better_gemini.grok_client._request_json",
            return_value={
                "models": [
                    {"id": "grok-imagine-image-pro", "aliases": ["grok-imagine-image"]},
                    {"id": "custom-image-model", "aliases": []},
                ]
            },
        ):
            models = list_models_sync(api_key="k", cache_ttl_s=0)
        self.assertEqual(models, ["custom-image-model", "grok-imagine-image", "grok-imagine-image-pro"])

    def test_list_models_sync_requires_api_key(self):
        prior_key = os.environ.pop("XAI_API_KEY", None)
        try:
            with self.assertRaises(BetterGrokError):
                list_models_sync(api_key=None, cache_ttl_s=0)
        finally:
            if prior_key is not None:
                os.environ["XAI_API_KEY"] = prior_key

    def test_build_image_request_for_generation(self):
        path, payload, mode = _build_image_request(
            BetterGrokRequest(
                model="grok-imagine-image",
                prompt="city skyline",
                aspect_ratio="16:9",
                resolution="2k",
                n=3,
            )
        )
        self.assertEqual(path, "/images/generations")
        self.assertEqual(mode, "generation")
        self.assertEqual(
            payload,
            {
                "model": "grok-imagine-image",
                "prompt": "city skyline",
                "response_format": "b64_json",
                "aspect_ratio": "16:9",
                "resolution": "2k",
                "n": 3,
            },
        )

    def test_build_image_request_for_single_image_edit(self):
        path, payload, mode = _build_image_request(
            BetterGrokRequest(
                model="grok-imagine-image",
                prompt="turn this into a sketch",
                input_images=(b"png-bytes",),
            )
        )
        self.assertEqual(path, "/images/edits")
        self.assertEqual(mode, "edit")
        self.assertIn("image", payload)
        self.assertNotIn("images", payload)
        self.assertTrue(payload["image"]["url"].startswith("data:image/png;base64,"))

    def test_build_image_request_for_multi_image_edit(self):
        path, payload, mode = _build_image_request(
            BetterGrokRequest(
                model="grok-imagine-image",
                prompt="merge these subjects",
                input_images=(b"one", b"two"),
            )
        )
        self.assertEqual(path, "/images/edits")
        self.assertEqual(mode, "edit")
        self.assertEqual(len(payload["images"]), 2)
        self.assertNotIn("image", payload)

    def test_generate_images_sync_decodes_base64_images(self):
        image_bytes = b"fake-image-bytes"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        with mock.patch(
            "better_gemini.grok_client._request_json",
            return_value={
                "model": "grok-imagine-image",
                "data": [{"b64_json": b64}],
            },
        ):
            text, images = generate_images_sync(
                api_key="k",
                request=BetterGrokRequest(model="grok-imagine-image", prompt="garden"),
            )
        self.assertEqual(images, [image_bytes])
        self.assertIn("returned 1 image(s)", text)

    def test_generate_images_sync_surfaces_empty_image_response(self):
        with mock.patch(
            "better_gemini.grok_client._request_json",
            return_value={"model": "grok-imagine-image", "data": []},
        ):
            text, images = generate_images_sync(
                api_key="k",
                request=BetterGrokRequest(model="grok-imagine-image", prompt="garden"),
            )
        self.assertEqual(images, [])
        self.assertIn("returned no images", text)


if __name__ == "__main__":
    unittest.main()
