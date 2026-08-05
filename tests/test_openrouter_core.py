import unittest

from better_gemini.openrouter_core import (
    BetterOpenRouterConfigError,
    build_request,
)


class OpenRouterCoreTests(unittest.TestCase):
    def test_build_request_normalizes_supported_options(self):
        request = build_request(
            model=" openai/gpt-image-2 ",
            prompt="city skyline",
            aspect_ratio="16:9",
            resolution="2k",
            quality="HIGH",
            output_format="WEBP",
            background="OPAQUE",
            output_compression=82,
            n=3,
            seed=42,
        )
        self.assertEqual(request.model, "openai/gpt-image-2")
        self.assertEqual(request.aspect_ratio, "16:9")
        self.assertEqual(request.resolution, "2K")
        self.assertEqual(request.quality, "high")
        self.assertEqual(request.output_format, "webp")
        self.assertEqual(request.background, "opaque")
        self.assertEqual(request.output_compression, 82)
        self.assertEqual(request.n, 3)
        self.assertEqual(request.seed, 42)

    def test_build_request_omits_automatic_options(self):
        request = build_request(
            model="openai/gpt-image-2",
            prompt="city skyline",
        )
        self.assertIsNone(request.aspect_ratio)
        self.assertIsNone(request.resolution)
        self.assertIsNone(request.quality)
        self.assertIsNone(request.output_format)
        self.assertIsNone(request.background)
        self.assertIsNone(request.output_compression)
        self.assertIsNone(request.seed)

    def test_explicit_size_overrides_resolution_and_aspect_ratio(self):
        request = build_request(
            model="openai/gpt-image-2",
            prompt="city skyline",
            aspect_ratio="16:9",
            resolution="4K",
            width=1536,
            height=864,
        )
        self.assertEqual((request.image_width, request.image_height), (1536, 864))
        self.assertIsNone(request.aspect_ratio)
        self.assertIsNone(request.resolution)

    def test_build_request_accepts_byte_like_reference_images(self):
        request = build_request(
            model="openai/gpt-image-2",
            prompt="combine these subjects",
            input_images=[bytearray(b"one"), memoryview(b"two")],
        )
        self.assertEqual(request.input_images, (b"one", b"two"))

    def test_build_request_requires_width_and_height_together(self):
        with self.assertRaises(BetterOpenRouterConfigError):
            build_request(
                model="openai/gpt-image-2",
                prompt="city skyline",
                width=1024,
            )

    def test_build_request_rejects_transparent_jpeg(self):
        with self.assertRaises(BetterOpenRouterConfigError):
            build_request(
                model="openai/gpt-image-2",
                prompt="product photo",
                output_format="jpeg",
                background="transparent",
            )

    def test_build_request_rejects_png_compression(self):
        with self.assertRaises(BetterOpenRouterConfigError):
            build_request(
                model="openai/gpt-image-2",
                prompt="product photo",
                output_format="png",
                output_compression=50,
            )

    def test_build_request_rejects_out_of_range_n(self):
        with self.assertRaises(BetterOpenRouterConfigError):
            build_request(
                model="openai/gpt-image-2",
                prompt="city skyline",
                n=11,
            )

    def test_build_request_rejects_invalid_reference_image(self):
        with self.assertRaises(BetterOpenRouterConfigError):
            build_request(
                model="openai/gpt-image-2",
                prompt="city skyline",
                input_images=["not bytes"],
            )


if __name__ == "__main__":
    unittest.main()
