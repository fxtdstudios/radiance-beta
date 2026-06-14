# Albabit — Modifications & Fixes

Changes applied on top of upstream `fxtdstudios/radiance-beta`.
Branch: `fix/bugs` — Fork: `https://github.com/Albabit/radiance-beta`

---

## nodes/generate/resolution.py, js/radiance_resolution.js — Per-model VAE compression, frame computation, multi-output, preset auto-fill

### Spatial & temporal VAE compression fixes

**Problem 1 (spatial):** The empty latent's spatial size always used a global
`LATENT_SCALE = 8` divisor, regardless of `model_type`. This produced grossly
oversized latents for LTXV (×32 VAE) and Flux.2 (×16 VAE).

**Fix:** Added `SPATIAL_SCALE` map (`LTXV (128ch)`: 32, `Flux.2 / Flux.2 Klein
(128ch)`: 16, default: 8), applied in the empty-latent computation and in
`_render_preview_card`.

**Problem 2 (temporal — major perf bug):** For video latents, the temporal
(frame) dimension of the empty 5D latent tensor (`torch.zeros(1, latent_c,
T, H, W)`) was set directly to the raw pixel-space `video_frames` (e.g. 241
for a 10s/24fps LTX 2.3 clip), instead of the 3D-VAE-compressed latent frame
count (31). This made the sampler process ~8x more "frames" than necessary.

Measured impact on an identical LTX 2.3 / 20-step test: **65.89s** (old
`radiance` version, latent T=31) vs **443.33s** (beta before fix, latent
T=241) — a ~6.7x slowdown, closely matching the 241/31 ≈ 7.8x latent size
ratio.

**Fix:** Added `TEMPORAL_SCALE` map (`LTXV (128ch)`: 8, `WAN (16ch)`: 4,
`HunyuanVideo (16ch)`: 4, `CogVideoX (16ch)`: 4, default: 4), and compute
`lat_t = (video_frames - 1) // temporal_scale + 1` for the latent's temporal
dimension — mirroring the formula from the previous `radiance` version.
Outputs (`frame_count`, `duration_sec`, `video_frames`) remain unchanged
(raw pixel-space frame count), only the latent tensor itself is resized.

### Restored `frame_computation` (Manual/Auto Seconds) + `duration_seconds`

Restored from the previous `radiance` version: a `frame_computation` combo
(`"Manual (Frames)"` / `"Auto (Seconds)"`) plus a `duration_seconds` float
input. In "Auto (Seconds)" mode, `video_frames` is computed from
`duration_seconds × frame_rate`, aligned to the model's temporal stride
(`n*stride + 1`, stride=8 for LTX, 4 for WAN/Hunyuan/other).

`js/radiance_resolution.js`: added `_frameStride(modelType)` helper and
toggle/sync logic in `frameModeW.callback` so `video_frames` and
`duration_seconds` stay in sync when switching modes (previously the hidden
field kept a stale value).

### Restored multi-output `RETURN_TYPES`

Previously `RETURN_TYPES = ("LATENT",)` only. Restored from the previous
`radiance` version:
`("LATENT", "INT", "INT", "INT", "STRING", "FLOAT", "INT", "STRING", "FLOAT")`
→ `(latent, width, height, channels, info, frame_rate, frame_count,
latent_format, duration_sec)`, so this node can drive Sampler Pro and other
downstream nodes directly.

### JS preset auto-fill

`js/radiance_resolution.js`: selecting a `preset` now auto-fills
`width`/`height` (from the `(WxH)` in the preset name), auto-toggles
`enable_video` for video presets (WAN/LTX/Hunyuan/CogVideoX), sets
`model_type` to match the preset's family, and resets `scale_factor` to 1.0
(per-model spatial scale is now handled by `SPATIAL_SCALE`, not this widget).

---

## js/radiance_pm_launcher.js, radiance_menu.js, radiance_studio.js, radiance_workspace.js, project_manager_dashboard.mjs, assets_dashboard.mjs, workspace_dashboard.html — Dashboard URL fix

**Problem:** Opening Manager, Library, or Assets from the Radiance Project
Manager node (or the floating FAB menu) returned a 404 / blank page when the
custom node folder was named anything other than exactly `radiance`
(e.g. `radiance-beta` for beta testers).

**Root cause:** All dashboard URLs and asset paths were hardcoded as
`/extensions/radiance/<file>`. ComfyUI derives the `/extensions/<name>/`
path from the actual folder name on disk. Any mismatch causes a 404.

**Fix:** Added `const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');`
at the top of each affected ES-module JS file and replaced every hardcoded
`/extensions/radiance/<file>` with `` `${_EXT_BASE}/<file>` ``
(`radiance_pm_launcher.js`, `radiance_menu.js`, `radiance_studio.js`,
`radiance_workspace.js`, `project_manager_dashboard.mjs`,
`assets_dashboard.mjs`). In `workspace_dashboard.html` (static, no JS module
scope), replaced the absolute `/extensions/radiance/r_icon.png` with the
relative path `r_icon.png`.

---

## js/radiance_resolution.js — v2.4 (Widget visibility fix)

**Problem:** In ComfyUI Nodes 2.0 (Vue 3 frontend), conditional widgets
(`video_frames`, `frame_rate`, `batch_size`, `mp_aspect_ratio`) left empty
ghost spaces when hidden, and failed to restore on workflow reload.

**Root cause:** The original code used `widget.inputEl.style.display` and
`widget.draw` overrides, which bypass Vue's reactivity system. Also,
`toggleFields` ran before Vue's first layout pass, so
`widget.computedHeight` was `undefined` when hiding, breaking the restore.

**Fix:** `setWidgetVisible(widget, visible, node)` — three-mechanism pattern
(`widget.options.hidden`, `widget.hidden`, `widget.type="hidden"` +
`computeSize=[0,-4]` + `computedHeight=4`), `node.widgets.splice(0, 0)` after
every visibility change to trigger Vue's reactive proxy, `node.size[1] =
sz[1]` in-place mutation, initial `toggleFields` deferred 150ms, and
`onConfigure` re-applies toggle state at 150ms/600ms after deserialization.

---

## js/radiance_sampler.js — Widget visibility & preset fixes

### Bug A — Empty spaces in Custom mode after a named preset

**Root cause:** `resolveModelType` trusted the backend's `model_type="ltxav"`
default for every preset (hiding LTX-incompatible widgets even for Flux);
`applyPreset` never updated `model_type` for presets without one in their
config; Vue 3 reuses component instances on a no-op `splice(0,0)`, so a
widget restored from `type="hidden"` kept stale zero-height state.

**Fix:** `resolveModelType` now checks the **preset name first**;
`inferModelTypeForPreset` + `applyPreset` write the inferred `model_type`
when the preset config omits it; `_forceWidgetReinsert(widget, node)` does a
real `splice(remove) + splice(reinsert)` to force Vue to recreate the
component whenever a widget transitions hidden→visible; `toggleFields`
deletes synthetic `computedHeight=32` values after restore (50ms). `tile_size`/
`tile_overlap`/`tile_blend` are now also hidden in Custom mode when
`tile_mode=false`.

### Bug B — Page reload: all widgets hidden after browser refresh

**Root cause:** `onNodeCreated`'s 150ms timer could fire while
`graph.configure()` was still running, reading the still-default
`preset="None"` and hiding everything.

**Fix:** `onConfigure` sets `node._configuredByLoad = true` synchronously;
`onNodeCreated`'s timer checks this flag and bails, leaving `onConfigure`'s
own 150ms/600ms timers in control.

### Bug C — `model_type` shows "ltxav" for non-LTX presets

**Fix:** `applyPreset` calls `inferModelTypeForPreset` and writes the correct
family to the `model_type` widget whenever the preset config doesn't specify
one. Also: switching to **None or Custom** resets `model_type` to `"auto"`.

---

## model/detect.py — LTX latent channel count fix

**Problem:** `LATENT_CHANNELS`/`_FORMAT_MAP` listed `"ltx"`/`"ltxav"` as 16ch
(`"ltx_16ch"`), but the LTX-Video VAE (incl. LTX 2.3) uses 128 channels. Model
loading was unaffected (real MODEL/VAE come from the checkpoint), but
`model_meta` reported incorrect `latent_ch`/`latent_format` for downstream
QC/analytics nodes.

**Fix:** `LATENT_CHANNELS["ltx"/"ltxav"] = 128`, `_FORMAT_MAP[...] =
"ltx_128ch"`, matching `MODEL_VAE_CONFIG["ltx-video"]`.

---

## nodes_loader.py — RadianceVideoLoader: LTX 2.3 VAE size mismatch fix

**Problem:** Loading `LTX23_video_vae_bf16.safetensors` raised a
`size mismatch` error in `VideoVAE`.

**Root cause:** The VAE was loaded via `comfy.utils.load_torch_file(vae_path)`
(no metadata) + `comfy.sd.VAE(sd=sd)`. Without the safetensors metadata's
`config` key, `comfy.sd.VAE` falls back to a default internal layout that
doesn't match LTX 2.3's video VAE.

**Fix:** Added `_load_vae_sd_metadata(vae_path)`. `RadianceUnifiedLoader`
(image loader) is unchanged. `RadianceVideoLoader` overrides it to use
`comfy.utils.load_torch_file(vae_path, return_metadata=True)`, so
`comfy.sd.VAE(sd=sd, metadata=metadata)` picks the correct internal config.

---

## nodes_loader.py — RadianceVideoLoader: Baked VAE, Audio VAE & Latent Upscale Model

Ported the legacy loader's "Baked VAE", Audio VAE, and latent-upscale-model
support to `RadianceVideoLoader` (v3.x architecture:
`loader_utils.py` SSOT, `_unet_cache`/`_clip_cache`/`_vae_cache`,
`_apply_preset_override`). `RadianceUnifiedLoader` (image loader) untouched.

- New outputs: `AUDIO_VAE`, `LATENT_UPSCALE_MODEL`.
- `vae_name` offers **"Baked VAE (from UNET)"** (default): extracted via
  `comfy.sd.load_checkpoint_guess_config(..., output_vae=True)`.
- New optional `audio_vae_name`: `"None"`, `"Baked Audio VAE (from UNET)"`, or
  a standalone checkpoint/VAE file (LTX 2.3 audio). Both paths use
  `comfy.utils.state_dict_prefix_replace(..., {"audio_vae.": "autoencoder.",
  "vocoder.": "vocoder."}) + comfy.sd.VAE(sd=..., metadata=...)`.
- New optional `upscale_model_name` (from `models/latent_upscale_models/`):
  recognizes HunyuanVideo SR 720p/1080p and LTX `LatentUpsampler` via
  safetensors metadata.
- New dedicated LRU caches `_audio_vae_cache` / `_upscale_model_cache`.
- `model_meta` gains `"audio_vae"` and `"upscale_model"` fields.
- `audio_vae_name`/`upscale_model_name` placed right after `vae_name`
  (matching the legacy layout).
- CLIP-slot tooltips clarified: `t5xxl` = LTX **pre-2.3**; `llm_encoder` =
  LTX 2.3 (Gemma 3) + Kolors/HunyuanVideo (ChatGLM3); `text_projection` =
  LTX 2.3 with Gemma 3.

---

## js/radiance_loader.js, config/model_map.py, nodes_loader.py, loader_utils.py — Preset dropdown fix & preset corrections

**Problem:** The "preset" dropdown in both loaders was non-functional —
selecting a preset did nothing.

**Root cause:** `app.registerExtension({ nodeCreated, loadedGraphNode })`
wrapped `presetW.callback`/`modelTypeW.callback` *after* Vue (Nodes 2.0) had
already mounted the combo widget components.

**Fix:** Switched to `beforeRegisterNodeDef` +
`nodeType.prototype.onNodeCreated`/`onConfigure` (mirrors
`radiance_sampler.js`), wrapping callbacks before widget construction. Added
`_forceWidgetReinsert(widget, node)` (ported from `radiance_sampler.js`) for
restoring hidden widgets; "Custom" preset now correctly restores all hidden
widgets.

**Preset hint/auto-fill corrections** (real-world `.safetensors` filenames as
of June 2026):
- **Flux Dev / Flux Dev (Low VRAM)**: added `flux1-krea-dev`/`krea-dev` hints.
- **LTX Video 2.3 / (Low VRAM)**: fixed unet hint collision between fp8 and
  full-precision presets (exact-filename priority); `vae_name` defaults to
  "Baked VAE (from UNET)"; added `upscale_model_name`
  (`ltx-2.3-spatial-upscaler-x2-1.1/1.0.safetensors`); `llm_encoder`
  prioritizes `gemma_3_12B_it.safetensors` (`gemma_3_12B_it_fp4_mixed` for Low
  VRAM).
- **LTX Video / LTX Video 13B**: added "Baked VAE (from UNET)" + corrected
  `ltxvideo_vae` hints (was `ltx_vae`/`ltxv_vae`/`causal_vae`); added
  `upscale_model_name`.
- **Wan 2.1**: fixed `t5xxl` hints to `umt5_xxl_*` (underscore, not
  `umt5-xxl`); corrected `vae_name` to `wan_2.1_vae.safetensors`.
- **Wan 2.2 (new preset)**: split out from "Wan 2.1" with its own
  `unet_hints` (`wan2.2`/`wan_2.2`/`wan-2.2`) and `vae_hints` prioritizing
  `wan2.2_vae.safetensors` (fallback `wan_2.1_vae.safetensors`); same `t5xxl`
  hints as Wan 2.1. (Wan 2.2's MoE high/low-noise multi-file UNETs and t2v/i2v
  variants deferred.)
- **PixArt Sigma**: corrected `vae_name` hints to the SDXL-based VAE
  (`pixart_sigma_sdxlvae`/`sdxl_vae`/`vae-ft-mse`).
- **Lumina2**: corrected text encoder to `gemma_2_2b_fp16.safetensors`,
  routed to the **`llm_encoder`** clip slot (not `t5xxl`); `vae_name` →
  `ae.safetensors` (Flux VAE, reused by Lumina2).
- **Z-Image**: corrected text encoder to `qwen_3_4b.safetensors`, routed to
  the **`llm_encoder`** clip slot (not `t5xxl`); `vae_name` →
  `flux_vae.safetensors`.
- **"Flux Dev (Flux.2)"**: removed — it conflated `flux1-krea-dev.safetensors`
  (FLUX.1, `clip_l`+`t5xxl`+`ae.safetensors`) with true Flux.2
  (`flux2-dev.safetensors`, Mistral-3 encoder + `flux2-vae.safetensors`,
  128ch latent — not supported by the clip-slot infrastructure at the time).

**Per-loader preset filtering:** Added `VIDEO_PRESET_NAMES` to
`config/model_map.py` (HunyuanVideo, Wan 2.1, Wan 2.2, LTX Video, LTX Video
13B, LTX Video 2.3, LTX Video 2.3 (Low VRAM)). "Radiance Video Loader" only
lists `Custom` + video presets; "Radiance Loader" only lists `Custom` + image
presets, via `INPUT_TYPES` filtering in `nodes_loader.py`.

---

## js/radiance_loader.js, radiance_sampler.js, radiance_resolution.js, radiance_upscale.js — Node resize & widget remount fixes for preset folding

**Problem:** On "Radiance Video Loader", switching presets hid/showed widgets
but the node box didn't shrink/grow to match. Also, after a page reload or
repeated `Custom -> <preset> -> Custom -> <preset>` toggling, all widgets
incorrectly re-appeared as if "Custom".

**Root causes & fixes (`radiance_loader.js`):**
- **Node box not resizing**: `refreshNodeSize` mutated `node.size[i]`
  directly — Vue's resize handling never observes this. Fix: use
  `node.setSize([Math.max(node.size[0], sz[0]), sz[1]])`.
- **Stale widgets after reload/repeated toggling**: once a widget's Vue
  component has (re)mounted, it stops reacting to `type`/`hidden` mutations
  via a no-op `splice(0,0)`. Fix: `setWidgetVisible` now calls
  `_forceWidgetReinsert(widget, node)` unconditionally, in both directions,
  on every call.

**Generalized** to every other JS file with conditional widgets/resizable
nodes: `radiance_sampler.js` (made `_forceWidgetReinsert` unconditional),
`radiance_resolution.js` (added the helper, v2.4 → v2.5), `radiance_upscale.js`
(added the helper).

**Note (not fixed, dead code):** `radiance_upscale.js`'s
`beforeRegisterNodeDef` targets `"RadianceAIUpscale"`, a node type that no
longer exists (current: `RadianceUpscaleTiler`/`Image`/`Video`/`FaceRestore`).
Also, `radiance_resolution.js`'s `preset` widget doesn't drive widget
visibility (existing behavior, not a regression).

---

## nodes/generate/resolution.py, js/radiance_resolution.js — model_type-driven presets, alignment & VIDEO_MODEL_TYPES coverage

**Problem:** Model-specific behavior (pixel alignment, video-latent
detection, WAN's 4k+1 frame rule, frame-count stride, `latent_format`) was
keyed off the **preset category** (`VIDEO_PRESET_CATEGORIES`,
`WAN_FRAME_CATEGORIES`, `ALIGN32_CATEGORIES`, `VIDEO_LATENT_FORMAT_MAP`), with
~51 model-specific entries in `PRESETS`. `_align8` also rounded to the
*nearest* multiple of 8 (could round down, making exact target resolutions
unreachable).

**Fix — inversion to `model_type`-driven:**
- `PRESETS` is now a plain, model-agnostic list of Cinema (12) + Social (4)
  resolutions; all model-specific preset categories removed.
- `model_type` is now the single source of truth for: pixel alignment via
  `SPATIAL_SCALE` + `_align_up(val, scale)` (always rounds **up**),
  5D video-latent detection (`VIDEO_MODEL_TYPES`), WAN's 4k+1 frame rule,
  the "Auto (Seconds)" frame stride (`TEMPORAL_SCALE`), and `latent_format`
  (single `LATENT_FORMAT_MAP`, used for image and video).
- `generate()` returns `computed_width`/`computed_height`;
  `radiance_resolution.js`'s `onExecuted` hook writes the final aligned
  values back into `width`/`height`.
- JS: removed preset-name-based `enable_video`/`model_type` auto-detection
  (no longer meaningful with model-agnostic presets); added
  `VIDEO_MODEL_TYPES_JS` + `model_type`'s callback to auto-toggle
  `enable_video`; added `SPATIAL_SCALE_JS`/`_alignUp`/`_applyAlignment` for
  **instant** width/height re-alignment on `model_type` change, tracking an
  unaligned "base" resolution (`node._resBaseW/H`); `+`/`-` step size now
  matches the current `model_type`'s alignment (`_setWidgetStep`); `preset`
  auto-switches to/from `"Custom"` when `width`/`height` is edited away from
  / back to the preset's aligned resolution; `video_frames` instantly snaps
  to a valid `N*stride+1` for all `VIDEO_MODEL_TYPES`.
- Default `preset` changed `"Flux Square (1024×1024)"` → `"HD 1080p
  (1920×1080)"` → final default **`"Custom"`** (matching the default
  1024×1024 width/height).

**`model_type` coverage — final state** (after several incremental
follow-ups, each audited against ComfyUI core: `comfy/sd.py`,
`comfy_extras/nodes_lt.py`, `nodes_wan.py`, `nodes_hunyuan.py`,
`nodes_cosmos.py`, `nodes_mochi.py`, `comfy/ldm/cogvideo/vae.py`):
- Merged equivalent entries (identical `LATENT_CHANNELS`/`SPATIAL_SCALE`/
  `TEMPORAL_SCALE`, neither in `VIDEO_MODEL_TYPES`, `model_type` is
  input-only so renaming is safe): `"Flux / SD3 (16ch)"` + `"Lumina2 /
  Z-Image (16ch)"` → `"Flux / SD3 / Lumina2 / Z-Image (16ch)"`; `"SDXL / SD
  1.5 (4ch)"` + `"PixArt / Aura Flow / Kolors (4ch)"` → `"SDXL / SD 1.5 /
  PixArt / Aura Flow / Kolors (4ch)"`.
- Added `"Chroma (16ch)"` (16ch, 8px, `latent_format="chroma"` — distinct
  sampler defaults in `sampler_utils.py`, kept separate from the Flux group).
- `VIDEO_MODEL_TYPES` now covers 6 models with verified `TEMPORAL_SCALE`:
  WAN/HunyuanVideo/CogVideoX (4), Mochi (6), LTXV/"Cosmos World (16ch)" (8).
  "Cosmos World" was named to make explicit that only Cosmos 1.0
  (text/image-to-video, ×8 stride) is covered — Cosmos Predict2 (×4) is
  deferred.
- `"Auto (Flux 16ch)"` → **`"Manual"`** (new default for the `model_type`
  combo, repurposed as fully unconstrained: `SPATIAL_SCALE["Manual"]=1`,
  `TEMPORAL_SCALE["Manual"]=1`, `LATENT_CHANNELS["Manual"]=16`,
  `LATENT_FORMAT_MAP["Manual"]="flux"`; not in `VIDEO_MODEL_TYPES` by default,
  but `is_video_latent` still computes a 5D latent with stride=1 if the user
  manually enables video).

**Deferred / audited-not-added:**
- `ChromaRadiance` (pixel-space, 3ch, no VAE) — fundamentally different,
  not covered.
- `StepVideo` — no `latent_formats`/`CLIPType`/`supported_models` entry in
  this ComfyUI checkout; `sampler_utils.py` references are forward-looking
  stubs with no working load path.
- Cosmos Predict2 (×4 stride) — not covered by "Cosmos World (16ch)".

### Audit cleanup (dead code + VRAM estimate)

- Removed `_align8`/`_align32` (both leftovers from the pre-model_type
  scheme; `_align32` had zero callers, `_align8`'s 8px pre-rounding inside
  `_mp_target_dimensions()` was redundant with — and for "Manual" mode,
  actively wrong vs — the unconditional `_align_up(val, SPATIAL_SCALE[...])`
  that `generate()` already applies afterward). `_mp_target_dimensions()` now
  takes `align_val` directly.
- `MODEL_BASE_VRAM`: removed unreachable `"sd15": 2.5` (no `model_type`
  resolves to it post-merge); added `"chroma": 12.0` (was falling back to the
  generic 4.0GB default).
- `_estimate_vram()`: now takes `spatial_scale: int = 8` instead of
  hardcoding `w//8, h//8` for the latent tensor — was underestimating LTXV
  (×32)/Flux.2 (×16) and overestimating "Manual" (×1).

1382 tests pass (unchanged).

---

## Radiance Loader — model_type/preset sync with Resolution, clip_dtype fix, LTX 2.3 audio/text_projection

### New `model_type`s + presets: Chroma, Flux.2 Dev/Klein, Cosmos World, CogVideoX, Mochi

`model/detect.py` (`LATENT_CHANNELS`, `_FORMAT_MAP`, `CLIP_SLOT_ORDER`,
`_CLIP_TYPE_VARIANTS`, `_BASE_CLIP_VRAM`, `_BASE_VRAM`, `get_clip_type_enum`),
`config/model_map.py` (`MODEL_VAE_CONFIG` + new `CHECKPOINT_PRESETS` entries)
and `js/radiance_loader.js` (`PRESET_SLOTS` + `PRESET_CONFIGS`) now cover
these 6 architectures, bringing the Loader in line with Radiance
Resolution's `model_type` list. `weight_dtype`/`clip_dtype` use `"default"`
for all of them (no entry in `dtype_map` → ComfyUI's own VRAM-aware
auto-selection, adapts to both full-precision and pre-quantized files).

### Fix: `CLIP_SLOT_ORDER["z_image"]` / `["lumina2"]` pointed to `t5xxl`

Both presets fill `llm_encoder` (Qwen3-4B / Gemma-2 2B), not `t5xxl` — caused
"No CLIP encoders provided" with only `llm_encoder` filled. Fixed to
`["llm_encoder"]`.

### Fix: `clip_dtype: "fp16"` forced on large bf16-native LLM/T5 encoders

Root cause of Z-Image producing black/abstract images: forcing an fp16 cast
on a bf16-native encoder (Qwen3-4B) risks overflow/NaN. Changed `clip_dtype`
to `"default"` (zero VRAM cost, bf16 == fp16 size) for: HunyuanVideo, Wan
2.1/2.2, LTX Video, LTX Video 13B, LTX Video 2.3 (and Low VRAM, was
`fp8_e4m3fn`), Kolors, Lumina2, Z-Image. Z-Image's `weight_dtype` (UNet) was
also `"fp16"` → `"default"` for the same reason.

`"Flux Dev (Low VRAM)"` keeps `clip_dtype: "fp8_e4m3fn"` — that's a T5-XXL
cast, the established low-VRAM pattern, a different risk profile from the
LLM-class encoders above.

### LTX Video 2.3 — Audio VAE + dedicated VAE filenames

Added `audio_vae_name` widget (Radiance Video Loader) for the "LTX Video 2.3"
preset, auto-filling `LTX23_audio_vae_bf16.safetensors`. `vae_name` now
prioritizes `LTX23_video_vae_bf16.safetensors`.

### LTX Video 2.3 — "Baked (from UNET)" for `text_projection`

In the native LTX 2.3 txt2img workflow, "LTXV Audio Text Encoder Loader"
(`comfy_extras/nodes_lt_audio.py`) loads the Gemma 3 12B encoder *and* the
LTX 2.3 UNET checkpoint itself as a second CLIP source — the
`text_embedding_projection.*` weights are baked into the UNET file, and this
2-file load path is what selects the correct `ltxav_te`/`LTXAVTEModel`
wrapper (with projection + normalization) instead of the bare `gemma3_te`
wrapper.

- `model/detect.py`: `assemble_clip_paths()` now takes `unet_path`; when
  `text_projection == "Baked (from UNET)"`, the UNET path is appended as the
  second CLIP source instead of resolving a `text_encoders` file.
- `nodes_loader.py`: `text_projection` gets its own options list
  (`["None", "Baked (from UNET)", ...text_encoders]`), separate from the
  other CLIP slots; both loaders pass `unet_path` through to
  `assemble_clip_paths`.
- "LTX Video 2.3" preset keeps prioritizing the standalone
  `ltx-2.3_text_projection_bf16.safetensors` (Kijai); "LTX Video 2.3 (Low
  VRAM)" now defaults `text_projection` to `"Baked (from UNET)"`. Both keep
  the `text_projection` widget visible.

### `model/detect.py` — fixed/added `_ARCH_HEURISTICS` for Auto-Detect

Cross-checked against `comfy/model_detection.py`. The `"lumina2"` heuristic
(`cap_v_projection.weight`) matched no real checkpoint — Lumina2/Z-Image were
never auto-detected. `chroma` and `flux2` were entirely absent and silently
misdetected as `"flux"` (wrong `CLIP_SLOT_ORDER`: `clip_l`+`t5xxl` instead of
`t5xxl`/`llm_encoder`). `mochi`, `cosmos`, `cogvideox` were also absent.

- Fixed `lumina2` to use the real key pair `cap_embedder.1.weight` +
  `noise_refiner.0.attention.k_norm.weight`. Added `z_image`, distinguished
  from `lumina2` by the shape of `cap_embedder.1.weight` (3840 = Z-Image,
  2304 = Lumina2) via new `_tensor_dim0()` helper.
- Added `chroma` and `flux2` heuristics, checked *before* `flux` (both share
  `double_blocks`/`img_in` with Flux). Chroma Radiance (`nerf_blocks.*`) is
  excluded from the `chroma` match and falls through unchanged (unsupported).
- Added `mochi` (`t5_yproj.weight`), `cosmos`/Cosmos World
  (`blocks.block0.blocks.0.block.attn.to_q.0.weight`), `cogvideox`
  (`blocks.0.norm1.linear.weight`).
- Removed the `_SAFETENSORS_PEEK = 200` key-count limit on
  `list(f.keys())` — listing keys only reads the safetensors header (no extra
  cost), and some keys (e.g. Mochi's `t5_yproj.weight`) sort past 200 entries.
- Verified in real conditions (Albabit): Z-Image checkpoint now logs
  `Auto-detected architecture: z_image`; Flux checkpoint still logs `flux`
  (non-regression). 1421 tests pass.

22 loader smoke tests pass (unchanged).
