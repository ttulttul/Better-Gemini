import unittest
from unittest import mock

from better_gemini import extension
from better_gemini.genai_client import DEFAULT_MODELS


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        extension._warned_model_listing = False

    def test_model_dropdown_options_falls_back_to_bundled_defaults_on_error(self):
        with mock.patch.object(extension, "list_models_sync", side_effect=RuntimeError("boom")):
            options = extension._model_dropdown_options()
        self.assertEqual(options, DEFAULT_MODELS)

    def test_model_dropdown_options_falls_back_to_bundled_defaults_on_empty_list(self):
        with mock.patch.object(extension, "list_models_sync", return_value=[]):
            options = extension._model_dropdown_options()
        self.assertEqual(options, DEFAULT_MODELS)

    def test_model_dropdown_options_prepends_bundled_defaults_when_listing_succeeds(self):
        with mock.patch.object(
            extension,
            "list_models_sync",
            return_value=[
                "models/custom-image-model",
                "models/gemini-3-pro-image-preview",
            ],
        ):
            options = extension._model_dropdown_options()
        self.assertEqual(
            options,
            [
                "models/gemini-3.1-flash-image-preview",
                "models/gemini-3-pro-image-preview",
                "models/imagen-4.0-generate-001",
                "models/imagen-4.0-ultra-generate-001",
                "models/custom-image-model",
            ],
        )


if __name__ == "__main__":
    unittest.main()
