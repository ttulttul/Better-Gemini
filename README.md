# ComfyUI Better Gemini

Custom ComfyUI node(s) for generating images and text with Google Gemini and xAI Grok.

## Install

1. Clone into your ComfyUI `custom_nodes` directory:
   - `cd /path/to/ComfyUI/custom_nodes`
   - `git clone https://github.com/<you>/ComfyUI-Better-Gemini.git`
2. Install Python deps with `uv` (network required):
   - `uv sync`
   - `uv pip install --python /path/to/ComfyUI/python/bin/python -e ./ComfyUI-Better-Gemini`
3. Set API keys as needed:
   - `export GOOGLE_API_KEY="..."` (or `GEMINI_API_KEY`) for Gemini
   - `export XAI_API_KEY="..."` for Grok
4. Restart ComfyUI.

## Nodes

- `Better Gemini` (image / text)
  - Inputs: prompt, model (dropdown; populated via `models.list()`), `response_modalities` (`IMAGE`, `IMAGE+TEXT`, or `TEXT`), prompt_images (optional), aspect ratio, resolution / width+height, temperature, top_p/top_k, max tokens, thinking difficulty, seed.
  - Outputs: `IMAGE`, `STRING` (returned text / notes). `TEXT` mode emits a minimal blank `IMAGE` tensor and places the usable result in `STRING`.
  - Use `TEXT` with text-only Gemini models such as `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, or `gemini-3.1-pro-preview` when you want the node to behave like an LLM call instead of an image generator.
- `Better Grok` (image / text)
  - Inputs: prompt, model (dropdown; populated from xAI image + language model listings when `XAI_API_KEY` is available), `response_modalities` (`IMAGE`, `IMAGE+TEXT`, or `TEXT`), prompt_images (optional), aspect ratio, resolution, `n`.
  - Outputs: `IMAGE`, `STRING` (returned text / notes). `TEXT` mode emits a minimal blank `IMAGE` tensor and places the usable result in `STRING`.
  - Use `TEXT` with Grok language models such as `grok-4`, `grok-4-fast-non-reasoning`, `grok-3-mini`, or `grok-code-fast-1` when you want the node to behave like an LLM call instead of an image generator.

## Example Workflow

![Better Gemini example workflow](examples/better-gemini-workflow.png)

## Notes

- The Gemini node imports `google-genai` lazily so ComfyUI can still boot even if dependencies aren’t installed yet; execution will raise a clear error until installed.
- This extension uses ComfyUI’s V3 extension loader (`comfy_entrypoint`).
- The Gemini `model` dropdown is populated via `client.models.list()` without action-based filtering, so text-only and image-capable Gemini models can both appear. It requires an API key via `GOOGLE_API_KEY`/`GEMINI_API_KEY`; otherwise it falls back to bundled defaults and logs a warning: `gemini-3-flash-preview`, `gemini-3.1-flash-image-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-pro-image-preview`, `gemini-3.1-pro-preview`, `imagen-4.0-generate-001`, `imagen-4.0-ultra-generate-001`.
- When `response_modalities=TEXT`, Gemini requests omit image-specific config and the node's usable output is returned entirely through `STRING`.
- The Grok `model` dropdown merges xAI’s `/v1/image-generation-models` and `/v1/language-models`. It requires `XAI_API_KEY`; otherwise it falls back to bundled defaults and logs a warning: `grok-imagine-image`, `grok-imagine-image-pro`, `grok-4`, `grok-4-fast-non-reasoning`, `grok-3-mini`, `grok-code-fast-1`.
- Gemini requires `seed` to fit in an `int32`; larger ComfyUI seeds are deterministically folded via `seed % 2**31`.
- Grok image generation is wired against xAI’s documented image endpoints and requests `response_format="b64_json"`, so the node can return image tensors directly instead of downloading temporary URLs.
- The Grok HTTP client sends an explicit application `User-Agent` because `api.x.ai` can reject the default `Python-urllib` signature with Cloudflare 1010.
- Grok image edits use xAI’s JSON-based `/v1/images/edits` API and send ComfyUI `IMAGE` inputs as PNG data URIs. Multiple prompt images are supported for edit/merge workflows.
- Grok `TEXT` mode uses xAI’s legacy-but-documented `/v1/chat/completions` endpoint. If prompt images are attached, the node sends them as chat image inputs and returns the model’s text through `STRING`.
- `resolution`/`aspect_ratio` are best-effort (model-dependent). The node logs a warning if the returned size doesn’t match (no auto-resize).
- If Gemini or Grok returns no images, the node returns a blank placeholder image and includes a note in the `STRING` output. In Gemini or Grok `TEXT` mode, the placeholder is a minimal 1x1 tensor so text-only flows do not allocate a large dummy image.

## Dev

- Sync dev environment: `uv sync --dev`
- Run unit tests: `uv run python -m unittest discover -s tests -p 'test_*.py' -v`
