# ComfyUI Better Gemini

ComfyUI V3 nodes for generating images and text with Google Gemini and xAI Grok.

This repo provides two nodes:

- `Better Gemini` for Gemini image and text generation
- `Better Grok` for Grok image generation, image editing, and text chat

## Install

1. Clone into your ComfyUI `custom_nodes` directory:
   - `cd /path/to/ComfyUI/custom_nodes`
   - `git clone https://github.com/<you>/ComfyUI-Better-Gemini.git`
2. Install Python deps with `uv`:
   - `uv sync`
   - `uv pip install --python /path/to/ComfyUI/python/bin/python -e ./ComfyUI-Better-Gemini`
3. Set API keys as needed:
   - `export GOOGLE_API_KEY="..."` or `export GEMINI_API_KEY="..."`
   - `export XAI_API_KEY="..."`
4. Restart ComfyUI.

## Nodes

### Better Gemini

- Inputs: prompt, model, `response_modalities` (`IMAGE`, `IMAGE+TEXT`, `TEXT`), optional prompt images, aspect ratio, resolution or width+height, temperature, top_p, top_k, max tokens, thinking controls, seed, optional output caching
- Outputs: `IMAGE`, `STRING`
- Use it when you want one node that can handle both Gemini image models and Gemini text-only models

### Better Grok

- Inputs: prompt, model, `response_modalities` (`IMAGE`, `IMAGE+TEXT`, `TEXT`), reasoning effort (`none`, `low`, `medium`, `high`), optional prompt images, aspect ratio, resolution, `n`, optional output caching
- Outputs: `IMAGE`, `STRING`
- Use it when you want Grok image generation, image editing, or a text-only Grok call from the same node

## Output Behavior

- In `TEXT` mode, the usable model output is returned through `STRING`.
- In `TEXT` mode, the `IMAGE` output is a minimal blank `1x1` tensor so ComfyUI graphs can stay connected without allocating a large placeholder.
- In `IMAGE+TEXT` mode, image generation still runs and any returned notes or revised prompt text are placed in `STRING`.
- If Gemini or Grok returns no images when image output was requested, the node emits a blank placeholder image and includes a note in `STRING`.
- If `cache_outputs` is enabled, model outputs are stored under `.cache/` and identical future requests reuse the cached `IMAGE`/`STRING` outputs without calling Gemini or Grok.

## Model Dropdowns

- Gemini model options are populated via `client.models.list()` without action-based filtering, so both image-capable and text-only Gemini models can appear in the same dropdown.
- Grok model options merge xAI `/v1/image-generation-models` and `/v1/language-models`, so image and language models can appear in the same dropdown.
- If an API key is unavailable or model listing fails, each node falls back to bundled default model names.

Bundled fallback models:

- Gemini: `gemini-3-flash-preview`, `gemini-3.1-flash-image-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-pro-image-preview`, `gemini-3.1-pro-preview`, `imagen-4.0-generate-001`, `imagen-4.0-ultra-generate-001`
- Grok: `grok-imagine-image`, `grok-imagine-image-pro`, `grok-imagine-image-quality`, `grok-latest`, `grok-4`, `grok-4-fast-non-reasoning`, `grok-3-mini`, `grok-code-fast-1`

Recommended text-only examples:

- Gemini: `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro-preview`
- Grok: `grok-latest`, `grok-4`, `grok-4-fast-non-reasoning`, `grok-3-mini`, `grok-code-fast-1`

## Example Workflow

![Better Gemini example workflow](examples/better-gemini-workflow.png)

## Implementation Notes

- The Gemini node imports `google-genai` lazily so ComfyUI can still boot even if dependencies are not installed yet; execution raises a clear error until installed.
- This extension uses ComfyUI's V3 extension loader via `comfy_entrypoint`.
- Gemini requires `seed` to fit in an `int32`; larger ComfyUI seeds are deterministically folded via `seed % 2**31`.
- When `response_modalities=TEXT`, Gemini requests omit image-specific config.
- Grok image generation is wired against xAI's documented image endpoints and requests `response_format="b64_json"`, so the node can return image tensors directly instead of downloading temporary URLs.
- The Grok HTTP client sends an explicit application `User-Agent` because `api.x.ai` can reject the default `Python-urllib` signature with Cloudflare 1010.
- Grok image edits use xAI's JSON-based `/v1/images/edits` API and send ComfyUI `IMAGE` inputs as PNG data URIs. Multiple prompt images are supported for edit and merge workflows.
- Grok `TEXT` mode uses xAI's `/v1/responses` endpoint with configurable reasoning effort and `store=false`, since node calls are not reused as xAI chat sessions. If prompt images are attached, the node sends them as response image inputs and returns the model's text through `STRING`.
- Output caching uses the SHA-256 checksum of canonical Gemini or Grok request data as the manifest filename, with string and image payloads stored as separate content-addressed files.
- `resolution` and `aspect_ratio` are best-effort, model-dependent settings. The node logs a warning if the returned size does not match the request.

## Dev

- Sync dev environment: `uv sync --dev`
- Run unit tests: `uv run python -m unittest discover -s tests -p 'test_*.py' -v`
