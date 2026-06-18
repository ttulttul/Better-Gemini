import tempfile
import unittest
from pathlib import Path

from better_gemini.core import BetterGeminiRequest
from better_gemini.output_cache import load_cached_output, request_cache_key, store_cached_output


class OutputCacheTests(unittest.TestCase):
    def test_request_cache_key_is_stable_for_same_request_data(self):
        request = BetterGeminiRequest(
            model="models/gemini",
            prompt="draw a house",
            response_modalities=("IMAGE", "TEXT"),
            seed=123,
            input_images=(b"png-bytes",),
        )
        self.assertEqual(
            request_cache_key(provider="gemini", request=request, extra={"system_prompt": "style"}),
            request_cache_key(provider="gemini", request=request, extra={"system_prompt": "style"}),
        )

    def test_request_cache_key_changes_when_request_data_changes(self):
        base = BetterGeminiRequest(
            model="models/gemini",
            prompt="draw a house",
            response_modalities=("IMAGE",),
            input_images=(b"one",),
        )
        changed = BetterGeminiRequest(
            model="models/gemini",
            prompt="draw a house",
            response_modalities=("IMAGE",),
            input_images=(b"two",),
        )
        self.assertNotEqual(
            request_cache_key(provider="gemini", request=base),
            request_cache_key(provider="gemini", request=changed),
        )

    def test_store_and_load_cached_output_uses_content_addressed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            store_cached_output(
                "request-digest",
                text="hello",
                images=[b"image-one", b"image-two"],
                cache_dir=cache_dir,
            )

            self.assertTrue((cache_dir / "request-digest.json").exists())
            self.assertEqual(load_cached_output("request-digest", cache_dir=cache_dir), ("hello", [b"image-one", b"image-two"]))
            self.assertEqual(len(list((cache_dir / "strings").iterdir())), 1)
            self.assertEqual(len(list((cache_dir / "images").iterdir())), 2)

    def test_load_cached_output_returns_none_for_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_cached_output("missing", cache_dir=Path(tmp)))


if __name__ == "__main__":
    unittest.main()
