import unittest

from better_gemini.core import BetterGeminiRequest
from better_gemini.genai_client import _build_contents, _build_image_config_patch


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


if __name__ == "__main__":
    unittest.main()
