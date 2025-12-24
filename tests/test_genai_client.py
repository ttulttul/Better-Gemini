import os
import sys
import types
import unittest

from better_gemini.core import BetterGeminiError, BetterGeminiRequest
from better_gemini.genai_client import _MODEL_LIST_CACHE, _build_contents, _build_image_config_patch, list_models_sync


class _Part:
    @classmethod
    def from_text(cls, *, text: str):
        return {"kind": "text", "text": text}

    @classmethod
    def from_bytes(cls, *, data: bytes, mime_type: str):
        return {"kind": "bytes", "data": data, "mime_type": mime_type}


class _Content:
    def __init__(self, *, parts=None, role=None):
        self.parts = parts
        self.role = role


class _Types:
    Part = _Part
    Content = _Content


class GenaiClientTests(unittest.TestCase):
    def setUp(self):
        _MODEL_LIST_CACHE.clear()

    def test_build_contents_text_only_returns_str(self):
        contents = _build_contents(_Types, prompt="hello", input_images=())
        self.assertEqual(contents, "hello")

    def test_build_contents_with_images_uses_keyword_only_from_text(self):
        contents = _build_contents(_Types, prompt="hello", input_images=(b"png-bytes",))
        self.assertIsInstance(contents, list)
        self.assertEqual(len(contents), 1)
        self.assertIsInstance(contents[0], _Content)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(
            contents[0].parts,
            [
                {"kind": "text", "text": "hello"},
                {"kind": "bytes", "data": b"png-bytes", "mime_type": "image/png"},
            ],
        )

    def test_build_image_config_patch_none_when_unset(self):
        req = BetterGeminiRequest(model="m", prompt="p", response_modalities=("IMAGE",))
        self.assertIsNone(_build_image_config_patch(req))

    def test_build_image_config_patch_sets_image_size(self):
        req = BetterGeminiRequest(model="m", prompt="p", response_modalities=("IMAGE",), image_resolution="2K")
        self.assertEqual(_build_image_config_patch(req), {"imageConfig": {"imageSize": "2K"}})

    def test_build_image_config_patch_sets_aspect_ratio(self):
        req = BetterGeminiRequest(model="m", prompt="p", response_modalities=("IMAGE",), image_aspect_ratio="5:4")
        self.assertEqual(_build_image_config_patch(req), {"imageConfig": {"aspectRatio": "5:4"}})

    def test_build_image_config_patch_sets_width_height(self):
        req = BetterGeminiRequest(
            model="m",
            prompt="p",
            response_modalities=("IMAGE",),
            image_width=640,
            image_height=480,
        )
        self.assertEqual(_build_image_config_patch(req), {"imageConfig": {"width": 640, "height": 480}})

    def test_list_models_sync_filters_generate_content_and_sorts(self):
        class _Model:
            def __init__(self, name, supported_actions=None, supportedActions=None):
                self.name = name
                self.supported_actions = supported_actions
                self.supportedActions = supportedActions

        class _Models:
            def list(self):
                return [
                    _Model("models/zeta", supported_actions=["generateContent"]),
                    _Model("models/alpha", supported_actions=["generateContent"]),
                    _Model("models/ignored", supported_actions=["somethingElse"]),
                    _Model("models/alpha", supported_actions=["generateContent"]),
                    _Model("models/beta", supportedActions="generateContent"),
                    {"name": "models/delta", "supportedActions": ["generateContent"]},
                ]

        class _Client:
            def __init__(self, api_key):
                self.models = _Models()

        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = _Client  # type: ignore[attr-defined]
        google.genai = genai  # type: ignore[attr-defined]

        prior_google = sys.modules.get("google")
        prior_google_genai = sys.modules.get("google.genai")
        sys.modules["google"] = google
        sys.modules["google.genai"] = genai
        try:
            models = list_models_sync(api_key="k", cache_ttl_s=0)
        finally:
            if prior_google is None:
                sys.modules.pop("google", None)
            else:
                sys.modules["google"] = prior_google
            if prior_google_genai is None:
                sys.modules.pop("google.genai", None)
            else:
                sys.modules["google.genai"] = prior_google_genai

        self.assertEqual(models, ["models/alpha", "models/beta", "models/delta", "models/zeta"])

    def test_list_models_sync_allows_unfiltered_listing(self):
        class _Models:
            def list(self):
                return [
                    {"name": "models/a", "supportedActions": []},
                    {"name": "models/b", "supportedActions": ["somethingElse"]},
                ]

        class _Client:
            def __init__(self, api_key):
                self.models = _Models()

        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = _Client  # type: ignore[attr-defined]
        google.genai = genai  # type: ignore[attr-defined]

        prior_google = sys.modules.get("google")
        prior_google_genai = sys.modules.get("google.genai")
        sys.modules["google"] = google
        sys.modules["google.genai"] = genai
        try:
            models = list_models_sync(api_key="k", filter_action=None, cache_ttl_s=0)
        finally:
            if prior_google is None:
                sys.modules.pop("google", None)
            else:
                sys.modules["google"] = prior_google
            if prior_google_genai is None:
                sys.modules.pop("google.genai", None)
            else:
                sys.modules["google.genai"] = prior_google_genai

        self.assertEqual(models, ["models/a", "models/b"])

    def test_list_models_sync_requires_api_key(self):
        class _Models:
            def list(self):
                return []

        class _Client:
            def __init__(self, api_key):
                self.models = _Models()

        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = _Client  # type: ignore[attr-defined]
        google.genai = genai  # type: ignore[attr-defined]

        prior_google = sys.modules.get("google")
        prior_google_genai = sys.modules.get("google.genai")
        sys.modules["google"] = google
        sys.modules["google.genai"] = genai
        try:
            prior_key = os.environ.pop("GOOGLE_API_KEY", None)
            prior_gemini_key = os.environ.pop("GEMINI_API_KEY", None)
            try:
                with self.assertRaises(BetterGeminiError):
                    list_models_sync(api_key=None, cache_ttl_s=0)
            finally:
                if prior_key is not None:
                    os.environ["GOOGLE_API_KEY"] = prior_key
                if prior_gemini_key is not None:
                    os.environ["GEMINI_API_KEY"] = prior_gemini_key
        finally:
            if prior_google is None:
                sys.modules.pop("google", None)
            else:
                sys.modules["google"] = prior_google
            if prior_google_genai is None:
                sys.modules.pop("google.genai", None)
            else:
                sys.modules["google.genai"] = prior_google_genai


if __name__ == "__main__":
    unittest.main()
