import unittest

from better_gemini.genai_client import _build_contents


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


if __name__ == "__main__":
    unittest.main()

