# Learnings

- ComfyUI’s custom node loader supports both V1 (`NODE_CLASS_MAPPINGS`) and the newer V3 extension entrypoint (`comfy_entrypoint` returning a `ComfyExtension`). This repo uses V3.
- Gemini’s `generation_config.seed` must fit in a protobuf `int32`; ComfyUI commonly uses 64-bit seeds, so this node folds seeds into range via `seed % 2**31` (with `0` meaning “unset”).
- Gemini supports multimodal prompts; ComfyUI `IMAGE` tensors can be encoded to PNG and sent alongside the text prompt as additional parts.
- The Gemini `generate_content` API does not expose pixel-precise output sizing; this node logs a warning when the returned size doesn’t match `resolution`/`width`+`height` (no auto-resize).
- The `google-genai` SDK may not yet map `generationConfig.imageConfig`; when `resolution`/`aspect_ratio` are set, this node injects `imageConfig` into the raw request to enable sizing on models that support it.
- If Gemini blocks generation (e.g. `prompt_feedback.block_reason` or candidate `finish_reason` like `IMAGE_SAFETY`) and returns no image parts, the node surfaces the reason in the text output and returns a blank placeholder image tensor.
- `client.models.list()` returns model objects with `name` and `supported_actions`, but the Gemini dropdown should not rely on action-based filtering now that the node supports both image and text-only models; cache the raw list, dedupe it, and merge bundled defaults for offline fallback.
- Not every model that supports `"generateContent"` can output images; if a selected model returns text-only, handle empty-image responses by returning a placeholder image and surfacing a note in the node’s text output.
- Gemini text-only models fit the same node if `response_modalities` includes `TEXT`; skip `imageConfig` for those requests and emit only a minimal 1x1 placeholder tensor on the `IMAGE` output so ComfyUI graphs stay connected without wasting memory.
- For better offline/no-key ergonomics, keep a bundled fallback list of common Gemini defaults in the model dropdown, including both image-preview and text-preview models, and merge these defaults into successful `models.list()` responses without duplicates.
- xAI’s image edit endpoint uses JSON, not multipart form data. Sending ComfyUI images as `data:image/png;base64,...` URLs matches the documented `/v1/images/edits` payload and avoids temporary file hosting.
- xAI’s image generation endpoints can return temporary URLs or base64; requesting `response_format="b64_json"` is the simplest way to turn Grok responses into ComfyUI `IMAGE` tensors without a second download step.
- `api.x.ai` may reject the default `Python-urllib` user-agent with Cloudflare error 1010; set an explicit application `User-Agent` header in direct HTTP clients instead of relying on urllib defaults.
- xAI exposes `/v1/image-generation-models`, which is a better source for the Grok model dropdown than the generic `/v1/models` endpoint because it already scopes results to image-capable models.
- xAI also exposes `/v1/language-models`; merging it with `/v1/image-generation-models` lets one Grok node support both image generation and text-only chat models from the same dropdown.
- Grok text-only mode fits the same ComfyUI node if it routes to `/v1/responses`, returns text through `STRING`, exposes `reasoning.effort`, and emits a minimal 1x1 placeholder tensor on the `IMAGE` output so text-only graphs still satisfy ComfyUI’s type expectations.
- Provider output caching should hash canonical request data, not API keys, so cached Gemini and Grok results can skip network calls for identical prompts/settings while keeping credentials out of cache filenames and manifests.
- Converting to `uv` project workflows works best by declaring dependencies in `pyproject.toml` and using `uv sync` / `uv run` as the default install+test path.
