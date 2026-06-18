import base64
import os
import unittest
from unittest import mock

from better_gemini.grok_client import (
    DEFAULT_MODEL,
    DEFAULT_MODELS,
    DEFAULT_TEXT_MODELS,
    USER_AGENT,
    _MODEL_LIST_CACHE,
    _build_headers,
    _build_image_request,
    _build_text_input,
    _build_text_request,
    _extract_chat_text,
    _extract_responses_text,
    generate_content,
    generate_sync,
    generate_images_sync,
    generate_text_sync,
    list_models_sync,
)
from better_gemini.grok_core import BetterGrokError, BetterGrokRequest


class GrokClientTests(unittest.TestCase):
    def setUp(self):
        _MODEL_LIST_CACHE.clear()

    def test_default_model_is_in_bundled_default_models(self):
        self.assertIn(DEFAULT_MODEL, DEFAULT_MODELS)
        self.assertTrue(set(DEFAULT_TEXT_MODELS).issubset(DEFAULT_MODELS))

    def test_build_headers_sets_explicit_user_agent(self):
        headers = _build_headers(api_key="k", has_payload=True)
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertEqual(headers["User-Agent"], USER_AGENT)
        self.assertEqual(headers["Accept-Encoding"], "identity")
        self.assertEqual(headers["Content-Type"], "application/json")


    def test_list_models_sync_parses_ids_and_aliases(self):
        with mock.patch(
            "better_gemini.grok_client._request_json",
            side_effect=[
                {
                    "models": [
                        {"id": "grok-imagine-image-pro", "aliases": ["grok-imagine-image"]},
                        {"id": "custom-image-model", "aliases": []},
                    ]
                },
                {
                    "models": [
                        {"id": "grok-4", "aliases": ["grok-4-fast"]},
                    ]
                },
            ],
        ):
            models = list_models_sync(api_key="k", cache_ttl_s=0)
        self.assertEqual(
            models,
            [
                "custom-image-model",
                "grok-4",
                "grok-4-fast",
                "grok-imagine-image",
                "grok-imagine-image-pro",
            ],
        )

    def test_list_models_sync_can_limit_to_language_models(self):
        with mock.patch(
            "better_gemini.grok_client._request_json",
            return_value={"models": [{"id": "grok-4"}, {"id": "grok-3-mini"}]},
        ) as request_json:
            models = list_models_sync(api_key="k", model_type="language", cache_ttl_s=0)
        self.assertEqual(models, ["grok-3-mini", "grok-4"])
        request_json.assert_called_once_with(method="GET", path="/language-models", api_key="k")

    def test_list_models_sync_requires_api_key(self):
        prior_key = os.environ.pop("XAI_API_KEY", None)
        try:
            with self.assertRaises(BetterGrokError):
                list_models_sync(api_key=None, cache_ttl_s=0)
        finally:
            if prior_key is not None:
                os.environ["XAI_API_KEY"] = prior_key

    def test_list_models_sync_rejects_unknown_model_type(self):
        with self.assertRaises(BetterGrokError):
            list_models_sync(api_key="k", model_type="audio", cache_ttl_s=0)

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

    def test_build_text_messages_without_images_uses_string_content(self):
        messages = _build_text_input(
            BetterGrokRequest(
                model="grok-4",
                prompt="Explain this.",
                response_modalities=("TEXT",),
            )
        )
        self.assertEqual(messages, [{"role": "user", "content": "Explain this."}])

    def test_build_text_messages_with_images_uses_openai_style_content_parts(self):
        messages = _build_text_input(
            BetterGrokRequest(
                model="grok-4",
                prompt="What's in this image?",
                response_modalities=("TEXT",),
                input_images=(b"png-bytes",),
            )
        )
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0], {"type": "input_text", "text": "What's in this image?"})
        self.assertEqual(messages[0]["content"][1]["type"], "input_image")
        self.assertTrue(messages[0]["content"][1]["image_url"].startswith("data:image/png;base64,"))

    def test_build_text_request_includes_reasoning_effort(self):
        payload = _build_text_request(
            BetterGrokRequest(
                model="grok-latest",
                prompt="Explain this.",
                response_modalities=("TEXT",),
                reasoning_effort="low",
            )
        )
        self.assertEqual(
            payload,
            {
                "model": "grok-latest",
                "reasoning": {"effort": "low"},
                "input": [{"role": "user", "content": "Explain this."}],
            },
        )

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

    def test_extract_responses_text_supports_output_text_field(self):
        model, text = _extract_responses_text(
            {
                "model": "grok-4",
                "output_text": "Hello world",
            }
        )
        self.assertEqual(model, "grok-4")
        self.assertEqual(text, "Hello world")

    def test_extract_responses_text_supports_structured_content(self):
        model, text = _extract_responses_text(
            {
                "model": "grok-4",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "First"},
                            {"type": "text", "text": "Second"},
                        ],
                    }
                ],
            }
        )
        self.assertEqual(model, "grok-4")
        self.assertEqual(text, "First\n\nSecond")

    def test_extract_chat_text_alias_uses_responses_extractor(self):
        model, text = _extract_chat_text({"model": "grok-4", "output_text": "Hello"})
        self.assertEqual((model, text), ("grok-4", "Hello"))

    def test_generate_text_sync_returns_responses_text(self):
        with mock.patch(
            "better_gemini.grok_client._request_json",
            return_value={
                "model": "grok-4",
                "output_text": "Answer text",
            },
        ) as request_json:
            text, images = generate_text_sync(
                api_key="k",
                request=BetterGrokRequest(
                    model="grok-4",
                    prompt="Explain",
                    response_modalities=("TEXT",),
                    reasoning_effort="medium",
                ),
            )
        self.assertEqual(text, "Answer text")
        self.assertEqual(images, [])
        request_json.assert_called_once_with(
            method="POST",
            path="/responses",
            api_key="k",
            payload={
                "model": "grok-4",
                "reasoning": {"effort": "medium"},
                "input": [{"role": "user", "content": "Explain"}],
            },
        )

    def test_generate_sync_routes_text_only_requests_to_responses(self):
        with mock.patch("better_gemini.grok_client.generate_text_sync", return_value=("Answer text", [])) as text_sync:
            text, images = generate_sync(
                api_key="k",
                request=BetterGrokRequest(model="grok-4", prompt="Explain", response_modalities=("TEXT",)),
            )
        self.assertEqual((text, images), ("Answer text", []))
        text_sync.assert_called_once()

    def test_generate_content_async_routes_text_only_requests(self):
        async def run_test():
            with mock.patch("better_gemini.grok_client.generate_sync", return_value=("Answer text", [])) as generate:
                result = await generate_content(
                    api_key="k",
                    request=BetterGrokRequest(model="grok-4", prompt="Explain", response_modalities=("TEXT",)),
                )
            self.assertEqual(result, ("Answer text", []))
            generate.assert_called_once()

        import asyncio

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
