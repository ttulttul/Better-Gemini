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

    def test_build_mustache_variable_list_with_string_value(self):
        self.assertEqual(
            extension._build_mustache_variable_list("name", "Ada"),
            [{"name": "Ada"}],
        )

    def test_build_mustache_variable_list_with_list_value(self):
        self.assertEqual(
            extension._build_mustache_variable_list("items", ["one", "two"]),
            [{"items": ["one", "two"]}],
        )

    def test_build_mustache_variable_list_accepts_tuple_of_strings(self):
        self.assertEqual(
            extension._build_mustache_variable_list("items", ("one", "two")),
            [{"items": ["one", "two"]}],
        )

    def test_build_mustache_variable_list_rejects_empty_key(self):
        with self.assertRaises(ValueError):
            extension._build_mustache_variable_list("", "Ada")

    def test_build_mustache_variable_list_rejects_non_string_values(self):
        with self.assertRaises(TypeError):
            extension._build_mustache_variable_list("items", ["one", 2])  # type: ignore[list-item]

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
            ],
        ):
            options = extension._grok_model_dropdown_options()
        self.assertEqual(
            options,
            [
                "grok-imagine-image",
                "grok-imagine-image-pro",
                "grok-imagine-image-ultra",
            ],
        )


if __name__ == "__main__":
    unittest.main()
