# ComfyUI Better Gemini

Custom ComfyUI node(s) for generating images with Google Gemini via the official `google-genai` Python SDK.

## Install

1. Clone into your ComfyUI `custom_nodes` directory:
   - `cd /path/to/ComfyUI/custom_nodes`
   - `git clone https://github.com/<you>/ComfyUI-Better-Gemini.git`
2. Install Python deps with `uv` (network required):
   - `uv sync`
   - `uv pip install --python /path/to/ComfyUI/python/bin/python -e ./ComfyUI-Better-Gemini`
3. Set an API key (recommended):
   - `export GOOGLE_API_KEY="..."` (or `GEMINI_API_KEY`)
4. Restart ComfyUI.

## Nodes

- `Better Gemini` (image)
  - Inputs: prompt, model (dropdown; populated via `models.list()`), prompt_images (optional), aspect ratio, resolution / width+height, temperature, top_p/top_k, max tokens, thinking difficulty, seed.
  - Outputs: `IMAGE`, `STRING` (any returned text / notes).

## Example Workflow

![Better Gemini example workflow](examples/better-gemini-workflow.png)

## Notes

- The node imports `google-genai` lazily so ComfyUI can still boot even if dependencies aren’t installed yet; execution will raise a clear error until installed.
- This extension uses ComfyUI’s V3 extension loader (`comfy_entrypoint`).
- The `model` dropdown is populated via `client.models.list()` (filtered to models supporting `generateContent`). It requires an API key via `GOOGLE_API_KEY`/`GEMINI_API_KEY`; otherwise it falls back to bundled defaults and logs a warning: `gemini-3.1-flash-image-preview`, `gemini-3-pro-image-preview`, `imagen-4.0-generate-001`, `imagen-4.0-ultra-generate-001`.
- Gemini requires `seed` to fit in an `int32`; larger ComfyUI seeds are deterministically folded via `seed % 2**31`.
- `resolution`/`aspect_ratio` are best-effort (model-dependent). For models that support it (e.g. `gemini-3-pro-image-preview`), the node sends `generationConfig.imageConfig` with `imageSize`/`aspectRatio`; it logs a warning if the returned size doesn’t match (no auto-resize).
- If Gemini returns no images (blocked or a text-only model), the node returns a blank placeholder image and includes a note in the `STRING` output (and logs a warning).

## Dev

- Sync dev environment: `uv sync --dev`
- Run unit tests: `uv run python -m unittest discover -s tests -p 'test_*.py' -v`
