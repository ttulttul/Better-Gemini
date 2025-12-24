import base64
import unittest

from better_gemini.core import (
    BetterGeminiConfigError,
    build_request,
    describe_response_block,
    extract_text_and_images,
    max_dim_from_resolution,
    normalize_seed,
    resolution_mismatch_message,
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

    def test_max_dim_from_resolution(self):
        self.assertIsNone(max_dim_from_resolution(None))
        self.assertIsNone(max_dim_from_resolution("auto"))
        self.assertEqual(max_dim_from_resolution("1K"), 1024)
        self.assertEqual(max_dim_from_resolution("2K"), 2048)
        self.assertEqual(max_dim_from_resolution("4K"), 4096)

    def test_build_request_requires_both_dimensions(self):
        with self.assertRaises(BetterGeminiConfigError):
            build_request(
                model="m",
                prompt="p",
                response_modalities="IMAGE",
                width=512,
                height=0,
            )

    def test_build_request_rejects_invalid_resolution(self):
        with self.assertRaises(BetterGeminiConfigError):
            build_request(
                model="m",
                prompt="p",
                response_modalities="IMAGE",
                resolution="not-a-real-resolution",
            )

    def test_resolution_mismatch_message_none_when_unset(self):
        self.assertIsNone(
            resolution_mismatch_message(
                actual_width=1024,
                actual_height=768,
                requested_resolution=None,
                requested_width=None,
                requested_height=None,
            )
        )

    def test_resolution_mismatch_message_respects_explicit_dimensions(self):
        self.assertIsNone(
            resolution_mismatch_message(
                actual_width=640,
                actual_height=480,
                requested_resolution="4K",
                requested_width=640,
                requested_height=480,
            )
        )
        self.assertIn(
            "Requested size 640x480",
            resolution_mismatch_message(
                actual_width=1024,
                actual_height=768,
                requested_resolution="4K",
                requested_width=640,
                requested_height=480,
            )
            or "",
        )

    def test_resolution_mismatch_message_respects_resolution(self):
        self.assertIsNone(
            resolution_mismatch_message(
                actual_width=4096,
                actual_height=2304,
                requested_resolution="4K",
                requested_width=None,
                requested_height=None,
            )
        )
        self.assertIn(
            "Requested resolution 4K",
            resolution_mismatch_message(
                actual_width=1024,
                actual_height=1024,
                requested_resolution="4K",
                requested_width=None,
                requested_height=None,
            )
            or "",
        )

    def test_build_request_normalizes_seed(self):
        req = build_request(
            model="m",
            prompt="p",
            response_modalities="IMAGE",
            seed=363775908667919,
        )
        self.assertEqual(req.seed, 768631311)

    def test_build_request_accepts_input_images(self):
        req = build_request(
            model="m",
            prompt="p",
            response_modalities="IMAGE",
            input_images=[b"fake-png-bytes"],
        )
        self.assertEqual(req.input_images, (b"fake-png-bytes",))

    def test_build_request_rejects_non_bytes_input_images(self):
        with self.assertRaises(BetterGeminiConfigError):
            build_request(
                model="m",
                prompt="p",
                response_modalities="IMAGE",
                input_images=["not-bytes"],  # type: ignore[list-item]
            )

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

    def test_describe_response_block_prompt_feedback(self):
        response = _Obj(
            prompt_feedback=_Obj(block_reason="SAFETY", block_reason_message="Nope."),
        )
        self.assertEqual(describe_response_block(response), "Gemini blocked the request (SAFETY): Nope.")

    def test_describe_response_block_prompt_feedback_dict_keys(self):
        response = {
            "promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "Nope."},
        }
        self.assertEqual(describe_response_block(response), "Gemini blocked the request (SAFETY): Nope.")

    def test_describe_response_block_candidate_finish_reason(self):
        response = _Obj(candidates=[_Obj(finish_reason="IMAGE_SAFETY", finish_message="Nope.")])
        self.assertEqual(describe_response_block(response), "Gemini blocked generation (IMAGE_SAFETY): Nope.")

    def test_describe_response_block_candidate_safety_ratings(self):
        response = _Obj(
            candidates=[
                _Obj(
                    safety_ratings=[
                        _Obj(blocked=True, category=_Obj(value="HARASSMENT"), probability=_Obj(value="HIGH"))
                    ]
                )
            ]
        )
        self.assertEqual(
            describe_response_block(response),
            "Gemini safety filters blocked generation: HARASSMENT (HIGH).",
        )

    def test_describe_response_block_none_when_unblocked(self):
        self.assertIsNone(describe_response_block(_Obj(candidates=[_Obj()])))


if __name__ == "__main__":
    unittest.main()
