# Learnings

- ComfyUI’s custom node loader supports both V1 (`NODE_CLASS_MAPPINGS`) and the newer V3 extension entrypoint (`comfy_entrypoint` returning a `ComfyExtension`). This repo uses V3.
- Gemini’s `generation_config.seed` must fit in a protobuf `int32`; ComfyUI commonly uses 64-bit seeds, so this node folds seeds into range via `seed % 2**31` (with `0` meaning “unset”).
- Gemini supports multimodal prompts; ComfyUI `IMAGE` tensors can be encoded to PNG and sent alongside the text prompt as additional parts.
- The Gemini `generate_content` API does not expose pixel-precise output sizing; this node logs a warning when the returned size doesn’t match `resolution`/`width`+`height` (no auto-resize).
- The `google-genai` SDK may not yet map `generationConfig.imageConfig`; when `resolution`/`aspect_ratio` are set, this node injects `imageConfig` into the raw request to enable sizing on models that support it.
