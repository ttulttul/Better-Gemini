import unittest

from better_gemini.grok_core import BetterGrokConfigError, MAX_EDIT_IMAGES, build_request


class GrokCoreTests(unittest.TestCase):
    def test_build_request_normalizes_resolution(self):
        request = build_request(
            model="grok-imagine-image",
            prompt="test prompt",
            resolution="2K",
        )
        self.assertEqual(request.resolution, "2k")

    def test_build_request_rejects_invalid_aspect_ratio(self):
        with self.assertRaises(BetterGrokConfigError):
            build_request(
                model="grok-imagine-image",
                prompt="test prompt",
                aspect_ratio="21:10",
            )

    def test_build_request_rejects_out_of_range_n(self):
        with self.assertRaises(BetterGrokConfigError):
            build_request(
                model="grok-imagine-image",
                prompt="test prompt",
                n=11,
            )

    def test_build_request_accepts_byte_like_images(self):
        request = build_request(
            model="grok-imagine-image",
            prompt="test prompt",
            input_images=[bytearray(b"one"), memoryview(b"two")],
        )
        self.assertEqual(request.input_images, (b"one", b"two"))

    def test_build_request_rejects_too_many_edit_images(self):
        with self.assertRaises(BetterGrokConfigError):
            build_request(
                model="grok-imagine-image",
                prompt="test prompt",
                input_images=[b"x"] * (MAX_EDIT_IMAGES + 1),
            )


if __name__ == "__main__":
    unittest.main()
