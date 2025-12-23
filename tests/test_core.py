import base64
import unittest

from better_gemini.core import (
    BetterGeminiConfigError,
    build_request,
    extract_text_and_images,
    normalize_seed,
    thinking_budget_from_difficulty,
)


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CoreTests(unittest.TestCase):
    def test_normalize_seed_unset(self):
        self.assertIsNone(normalize_seed(0))
        self.assertIsNone(normalize_seed(-1))

    def test_normalize_seed_int32_passthrough(self):
        self.assertEqual(normalize_seed(123), 123)
        self.assertEqual(normalize_seed(2**31 - 1), 2**31 - 1)

    def test_normalize_seed_folds_large_seed(self):
        self.assertEqual(normalize_seed(363775908667919), 768631311)
        self.assertEqual(normalize_seed(2**31), 1)

    def test_thinking_budget_mapping(self):
        self.assertIsNone(thinking_budget_from_difficulty("auto"))
        self.assertEqual(thinking_budget_from_difficulty("low"), 1024)
        self.assertEqual(thinking_budget_from_difficulty("medium"), 4096)
        self.assertEqual(thinking_budget_from_difficulty("high"), 8192)
        with self.assertRaises(BetterGeminiConfigError):
            thinking_budget_from_difficulty("nope")

    def test_build_request_requires_both_dimensions(self):
        with self.assertRaises(BetterGeminiConfigError):
            build_request(
                model="m",
                prompt="p",
                response_modalities="IMAGE",
                width=512,
                height=0,
            )

    def test_build_request_normalizes_seed(self):
        req = build_request(
            model="m",
            prompt="p",
            response_modalities="IMAGE",
            seed=363775908667919,
        )
        self.assertEqual(req.seed, 768631311)

    def test_extract_text_and_images(self):
        img_bytes = b"not-a-real-png-but-bytes"
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        response = _Obj(
            candidates=[
                _Obj(
                    content=_Obj(
                        parts=[
                            _Obj(text="hello"),
                            _Obj(inline_data=_Obj(mime_type="image/png", data=b64)),
                        ]
                    )
                )
            ]
        )
        text, images = extract_text_and_images(response)
        self.assertEqual(text, "hello")
        self.assertEqual(images, [img_bytes])

    def test_extract_supports_data_uri(self):
        img_bytes = b"abc123"
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"mimeType": "image/png", "data": data_uri}},
                        ]
                    }
                }
            ]
        }
        text, images = extract_text_and_images(response)
        self.assertEqual(text, "")
        self.assertEqual(images, [img_bytes])


if __name__ == "__main__":
    unittest.main()
