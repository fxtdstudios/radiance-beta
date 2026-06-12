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
path from the actual folder name on disk. Any mismatch between the hardcoded
name and the real folder name causes a 404.

**Fixes:**
- Added `const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');` at the
  top of each affected ES-module JS file. This resolves the real extension
  base URL at runtime, regardless of the install folder name.
- Replaced every hardcoded `/extensions/radiance/<file>` with
  `` `${_EXT_BASE}/<file>` `` in `radiance_pm_launcher.js`,
  `radiance_menu.js`, `radiance_studio.js`, `radiance_workspace.js`,
  `project_manager_dashboard.mjs`, and `assets_dashboard.mjs`.
- In `workspace_dashboard.html` (static file, no JS module scope), replaced
  the absolute `/extensions/radiance/r_icon.png` with the relative path
  `r_icon.png` — the browser resolves it correctly from the file's own URL.

---

## js/radiance_resolution.js — v2.4 (Widget visibility fix)

**Problem:** In ComfyUI Nodes 2.0 (Vue 3 frontend), conditional widgets
(`video_frames`, `frame_rate`, `batch_size`, `mp_aspect_ratio`) left empty
ghost spaces when hidden, and failed to restore on workflow reload.

**Root cause:** The original code used `widget.inputEl.style.display` and
`widget.draw` overrides, which bypass Vue's reactivity system. Also,
`toggleFields` was called immediately on node creation — before Vue's first
layout pass — so `widget.computedHeight` was `undefined` at the time of
hiding, causing the restore path to fall back to an incorrect height.

**Fixes:**
- `setWidgetVisible(widget, visible, node)` — three-mechanism pattern:
  1. `widget.options.hidden` — Vue 3 filter in Nodes 2.0
  2. `widget.hidden` — LiteGraph `getLayoutWidgets()` exclusion
  3. `widget.type="hidden"` + `computeSize=[0,-4]` + `computedHeight=4` — physical height collapse
- `node.widgets.splice(0, 0)` after every visibility change — triggers Vue
  reactive proxy to re-evaluate `options.hidden`.
- `node.size[1] = sz[1]` (in-place mutation) instead of array replacement —
  preserves Vue's reactive tracking of the existing array reference.
- Initial `toggleFields` deferred 150 ms via `setTimeout` — Vue must complete
  its first layout pass before any widget is hidden.
- `onConfigure` hook with `setTimeout(150ms)` and `setTimeout(600ms)` —
  re-applies toggle state after a saved workflow deserializes widget values.

---

## js/radiance_sampler.js — Widget visibility & preset fixes

### Bug A — Empty spaces in Custom mode after a named preset

**Problem:** Switching from e.g. "LTX 2.3 LowRes" to "Custom" left empty
ghost spaces where `flux_guidance`, `flux_guidance_profile`,
`sigma_blend_steps`, `preview_method`, `noise_alpha_start/end` should be.

**Root cause:**
1. `resolveModelType` returned `modelTypeVal` immediately if it was not
   `"auto"`. Since the backend initialises `model_type = "ltxav"`, every
   named preset (including Flux) was resolved as `isLTX = true`, hiding all
   `LTX_INCOMPATIBLE_WIDGETS` even for non-LTX presets.
2. `applyPreset` never updated `model_type` for presets whose config did not
   explicitly include it (all Flux / WAN / AYS presets).
3. Vue 3's virtual-DOM differ reuses component instances when the same object
   reference stays in the widgets array — a `splice(0, 0)` no-op notifies Vue
   the array changed but Vue skips re-reading widget properties (`type`,
   `options.hidden`) on the existing instance. Widgets restored from
   `type="hidden"` would keep a stale component state (zero height, no content).

**Fixes:**
- `resolveModelType` now checks the **preset name first** (authoritative),
  falling back to `modelTypeVal` only when the name gives no match. Flux,
  WAN, AYS, SD3.5, Lumina2, HunyuanVideo, z_image presets all resolve
  correctly regardless of the current `model_type` widget value.
- `inferModelTypeForPreset` + `applyPreset` — when applying a named preset
  that has no `model_type` in its config, the correct model type is inferred
  from the preset name and written to the `model_type` widget.
- `_forceWidgetReinsert(widget, node)` — performs a real
  `splice(remove) + splice(reinsert)` to force Vue to destroy and recreate
  the component instance. Called in `setWidgetVisible` whenever a widget
  transitions from `type="hidden"` back to visible.
- `toggleFields` deferred cleanup (50 ms) — deletes synthetic
  `computedHeight = 32` values set during the restore so Vue recomputes the
  real per-widget height in its next layout pass.
- `tile_size`, `tile_overlap`, `tile_blend` are now hidden in **Custom mode**
  as well when `tile_mode = false`. `tile_mode` is in `foldTriggers` so
  toggling it immediately shows/hides the sub-options.

### Bug B — Page reload: all widgets hidden after browser refresh

**Problem:** After a browser page reload, the node displayed only the preset
dropdown regardless of the saved preset value.

**Root cause:** The `onNodeCreated` 150 ms timer could fire while
`graph.configure()` was still running (large workflows), at which point the
preset widget still held its default value `"None"` → `applyFolding` hid
everything. `onConfigure`'s timers would eventually correct this, but a
second race meant both timers competed and left the node stuck.

**Fix:**
- `onConfigure` sets `node._configuredByLoad = true` synchronously (before
  any timers). The `onNodeCreated` 150 ms timer checks this flag and returns
  early — `onConfigure`'s own timers (150 ms + 600 ms safety net) take over
  with the correct, already-deserialized widget values.

### Bug C — model_type shows "ltxav" for non-LTX presets

**Problem:** Selecting "Flux Ultra Fast (8 steps)" still showed
`model_type = ltxav`.

**Fix:** `applyPreset` calls `inferModelTypeForPreset` and writes the correct
family ("flux", "wan", "ltxv", "ltxav", "hunyuan_video", etc.) to the
`model_type` widget whenever the preset config does not include it explicitly.

### model_type reset on None / Custom

When the user switches **to None or Custom** (via the preset dropdown or via
a manual widget edit that auto-switches to Custom), `model_type` is reset to
`"auto"` so the widget reflects that no specific model family is locked.

---

## model/detect.py — LTX latent channel count fix

**Problem:** `LATENT_CHANNELS` and `_FORMAT_MAP` listed `"ltx"` and `"ltxav"`
as 16-channel (`"ltx_16ch"`), but the LTX-Video VAE (including LTX 2.3) uses
128 latent channels. This did not break model loading (the real MODEL/VAE
come from the checkpoint), but the `model_meta` JSON output from
`RadianceUnifiedLoader` reported `latent_ch: 16` / `latent_format: "ltx_16ch"`
for any LTX model — incorrect info for downstream QC/analytics nodes.

**Fix:** `LATENT_CHANNELS["ltx"]` and `["ltxav"]` set to `128`;
`_FORMAT_MAP["ltx"]` and `["ltxav"]` set to `"ltx_128ch"`, matching
`config/model_map.py`'s `MODEL_VAE_CONFIG["ltx-video"]`.

---

## nodes_loader.py — RadianceVideoLoader: LTX 2.3 VAE size mismatch fix

**Problem:** Loading `LTX23_video_vae_bf16.safetensors` with "Radiance Video
Loader" raised:
```
RuntimeError: Failed to load VAE 'LTX23_video_vae_bf16.safetensors':
Error(s) in loading state_dict for VideoVAE: size mismatch for ...
```

**Root cause:** The VAE was loaded via `comfy.utils.load_torch_file(vae_path)`
(no metadata) and `comfy.sd.VAE(sd=sd)`. Without the file's metadata,
`comfy.sd.VAE` cannot read the VAE's internal architecture config (`config`
key in the safetensors metadata) and falls back to a default internal layout
that doesn't match LTX 2.3's video VAE, causing a state_dict size mismatch
when loading the encoder/decoder weights. (Note: this is unrelated to the
128 latent channels, which is correct and unchanged for LTX/LTX 2.3.)

**Fix:** Added `_load_vae_sd_metadata(vae_path)`:
- `RadianceUnifiedLoader` (image loader): unchanged behaviour —
  `comfy.utils.load_torch_file(vae_path)`, no metadata.
- `RadianceVideoLoader` (video loader): overrides it to use
  `comfy.utils.load_torch_file(vae_path, return_metadata=True)`, so
  `comfy.sd.VAE(sd=sd, metadata=metadata)` picks the correct internal VAE
  architecture for LTX 2.3.

Verified: "Radiance Video Loader" now loads `ltx-2.3-22b-dev.safetensors` +
`LTX23_video_vae_bf16.safetensors` successfully, with `model_meta` reporting
`"latent_ch": 128, "latent_format": "ltx_128ch"`.

---

## nodes_loader.py — RadianceVideoLoader: Baked VAE, Audio VAE & Latent Upscale Model

**Goal:** Port the legacy Radiance loader's "Baked VAE", Audio VAE, and
latent upscale model support to `RadianceVideoLoader`, adapted to the v3.x
loader architecture (`loader_utils.py` SSOT, `_unet_cache`/`_clip_cache`/
`_vae_cache`, `_apply_preset_override`, etc.). `RadianceUnifiedLoader`
(image loader) is untouched.

**Changes (`RadianceVideoLoader` only):**
- New outputs: `AUDIO_VAE` and `LATENT_UPSCALE_MODEL` (alongside the
  existing `MODEL`, `CLIP`, `VAE`, `lora_stack`, `model_meta`).
- `vae_name` now offers **"Baked VAE (from UNET)"** (default) in addition
  to standalone `.safetensors` files. When selected, the VAE is extracted
  natively from the checkpoint via
  `comfy.sd.load_checkpoint_guess_config(..., output_vae=True)`.
- New optional `audio_vae_name` input: `"None"`, `"Baked Audio VAE (from
  UNET)"`, or a standalone checkpoint/VAE file (LTX 2.3 audio). Both baked
  and standalone paths use `comfy.utils.state_dict_prefix_replace(...,
  {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}) +
  comfy.sd.VAE(sd=..., metadata=...)` — the ComfyUI 0.22.0+ compatible
  pattern (mirrors the built-in `LTXVAudioVAELoader`).
- New optional `upscale_model_name` input (from
  `models/latent_upscale_models/`): recognizes HunyuanVideo SR 720p/1080p
  upsamplers and the LTX `LatentUpsampler` (via safetensors metadata
  `config`).
- New dedicated LRU caches `_audio_vae_cache` / `_upscale_model_cache`
  (separate from the core unet/clip/vae caches) for the baked/standalone
  Audio VAE and upscale model.
- `model_meta` JSON gains `"audio_vae"` and `"upscale_model"` fields.
- Input layout: `audio_vae_name` and `upscale_model_name` are placed right
  after `vae_name` (matching the legacy node layout) instead of at the
  bottom of the node.

**Tooltip clarifications (CLIP slots, both loaders):**
- `t5xxl`: now notes it's used by LTX **pre-2.3** (LTX 2.3 uses Gemma 3 via
  `llm_encoder` instead).
- `llm_encoder`: now explicitly mentions LTX 2.3 (Gemma 3), in addition to
  Kolors/HunyuanVideo (ChatGLM3).
- `text_projection`: now explicitly scoped to LTX 2.3 with Gemma 3.

Confirmed: `config/model_map.py`'s LTX/LTX-AV profiles already route Gemma 3
(`gemma_3_12B_it*`) to `llm_encoder` + `text_projection`, consistent with our
earlier fix — no regression from upstream.

---

## js/radiance_loader.js, config/model_map.py, nodes_loader.py, loader_utils.py — Preset dropdown fix & preset corrections

**Problem:** The "preset" dropdown in "Radiance Video Loader" / "Radiance
Loader" was completely non-functional — selecting a preset produced no
console output and did not fold/unfold widgets or auto-fill model files.

**Root cause:** The previous registration used
`app.registerExtension({ nodeCreated, loadedGraphNode })`, which wraps
`presetW.callback` / `modelTypeW.callback` *after* Vue (Nodes 2.0) has
already mounted the combo widget components — so the wrapped callback was
never invoked on live user interaction.

**Fixes:**
- Replaced the registration block with the `beforeRegisterNodeDef` +
  `nodeType.prototype.onNodeCreated` / `onConfigure` pattern (mirroring the
  confirmed-working pattern in `radiance_sampler.js`), wrapping
  `presetW.callback` / `modelTypeW.callback` before widget construction.
- Added `_forceWidgetReinsert(widget, node)` (ported from
  `radiance_sampler.js`) to work around a Vue 3 component-reuse issue where
  restoring a hidden widget left it visually broken; `setWidgetVisible` now
  calls this instead of a bare `splice`.
- "Custom" preset now correctly restores all hidden widgets via the above.

**Preset hint/auto-fill corrections** (researched real-world `.safetensors`
filenames as of June 2026):
- **Flux Dev / Flux Dev (Low VRAM)**: added `flux1-krea-dev` / `krea-dev`
  hints so `flux1-krea-dev.safetensors` is correctly auto-filled.
- **LTX Video 2.3 / (Low VRAM)**: fixed unet hint collision where the fp8 and
  full-precision presets could both match `ltx-2.3-22b-dev*` (now uses exact
  filename priority); `vae_name` now defaults to "Baked VAE (from UNET)" for
  all LTX presets; added `extra_widgets`/`upscale_hints` mechanism so
  `upscale_model_name` is shown and auto-filled
  (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors` then `...-1.0.safetensors`);
  `llm_encoder` now prioritizes `gemma_3_12B_it.safetensors`
  (`gemma_3_12B_it_fp4_mixed.safetensors` for Low VRAM).
- **LTX Video / LTX Video 13B**: added "Baked VAE (from UNET)" + corrected
  `ltxvideo_vae` hints (was `ltx_vae`/`ltxv_vae`/`causal_vae`, none of which
  match the real `ltxvideo_vae_bf16.safetensors` / `LTX23_video_vae_bf16.safetensors`
  filenames); added `extra_widgets`/`upscale_hints` for `upscale_model_name`.
- **Wan 2.1**: fixed `t5xxl` hints — `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
  uses an underscore (`umt5_xxl`), not `umt5-xxl`/`umt5xxl`; corrected
  `vae_name` hints to `wan_2.1_vae.safetensors`.
- **Wan 2.2 (new preset)**: split out from "Wan 2.1" as a separate preset
  with its own `unet_hints` (`wan2.2`/`wan_2.2`/`wan-2.2`) and `vae_hints`
  prioritizing `wan2.2_vae.safetensors` (falling back to
  `wan_2.1_vae.safetensors`); same corrected `t5xxl` hints as Wan 2.1.
  (Wan 2.2's multi-file MoE high/low-noise expert UNETs and t2v/i2v variants
  are a larger architectural change, deferred for a future task.)
- **PixArt Sigma**: corrected `vae_name` hints to the SDXL-based VAE
  (`pixart_sigma_sdxlvae` / `sdxl_vae` / `vae-ft-mse`).
- **Lumina2**: corrected text encoder to `gemma_2_2b_fp16.safetensors`
  (Gemma-2 2B), routed to the **`llm_encoder`** clip slot (not `t5xxl`),
  consistent with how Gemma 3 is handled for LTX 2.3; corrected `vae_name`
  to `ae.safetensors` (Flux VAE, reused by Lumina2).
- **Z-Image**: corrected text encoder to `qwen_3_4b.safetensors` (Qwen3-4B),
  routed to the **`llm_encoder`** clip slot (not `t5xxl`); corrected
  `vae_name` to `flux_vae.safetensors`.
- **"Flux Dev (Flux.2)"**: this preset (added during this session) was found
  to conflate `flux1-krea-dev.safetensors` (FLUX.1, correctly using
  `clip_l`+`t5xxl`+`ae.safetensors`) with `flux2-dev.safetensors` (true
  Flux.2: Mistral-3 text encoder + `flux2-vae.safetensors`, 128-channel
  latent — incompatible with the current clip-slot infrastructure). Removed
  entirely rather than ship an incorrect preset; true Flux.2 support is
  deferred for a future task.

**Per-loader preset filtering:**
- Added `VIDEO_PRESET_NAMES` to `config/model_map.py` (HunyuanVideo, Wan 2.1,
  Wan 2.2, LTX Video, LTX Video 13B, LTX Video 2.3, LTX Video 2.3 (Low VRAM)).
- "Radiance Video Loader" now only lists `Custom` + video presets;
  "Radiance Loader" now only lists `Custom` + image presets, via filtering
  in `INPUT_TYPES` in `nodes_loader.py`.

---

## js/radiance_loader.js, radiance_sampler.js, radiance_resolution.js, radiance_upscale.js — Node resize & widget remount fixes for preset folding

**Problem:** On "Radiance Video Loader", switching presets correctly hid/showed
widgets but the node box did not shrink/grow to match (leftover empty space or
clipped widgets). Additionally, after a page reload, or after toggling
`Custom -> <preset> -> Custom -> <preset>`, all widgets would incorrectly
re-appear as if the preset were "Custom".

**Root causes & fixes (`radiance_loader.js`):**
- **Node box not resizing**: `refreshNodeSize` mutated `node.size[i]` directly.
  This updates LiteGraph's internal model but Vue's resize handling never
  observes it — the rendered box keeps its old size indefinitely.
  `node.computeSize()` is already correct synchronously after a widget
  visibility change (no DOM-timing issue). Fix: `refreshNodeSize` now calls
  `node.setSize([Math.max(node.size[0], sz[0]), sz[1]])`, the API Vue's resize
  handling actually reacts to.
- **Stale widgets after reload / repeated preset toggling**: `_forceWidgetReinsert`
  (remove+reinsert via `splice`) was only called when a widget transitioned
  hidden→visible. Empirically, once a widget's Vue component has been
  (re)mounted, it stops reacting to *any* later `type`/`hidden` mutation via a
  no-op `splice(0,0)` — it keeps rendering its previous state until reinserted
  again. Fix: `setWidgetVisible` now calls `_forceWidgetReinsert(widget, node)`
  unconditionally, in both directions (visible→hidden and hidden→visible), on
  every call.

**Generalization:** the same two fixes (`setSize` in `refreshNodeSize`,
unconditional bidirectional `_forceWidgetReinsert` in `setWidgetVisible`) were
ported to every other JS file with conditionally shown/hidden widgets and a
node size that can change:
- `radiance_sampler.js`: `_forceWidgetReinsert` already existed but was called
  conditionally; now called unconditionally for both directions.
- `radiance_resolution.js`: `_forceWidgetReinsert` helper added (didn't exist
  before); outdated header comment claiming in-place `node.size[i]` mutation
  was "required for Vue reactivity" removed/corrected (v2.4 → v2.5).
- `radiance_upscale.js`: `_forceWidgetReinsert` helper added.

All four nodes confirmed working (no resize gap, correct folding across
reload and repeated preset switches) by Albabit.

**Note (not fixed, tracked for later):** `radiance_upscale.js`'s
`beforeRegisterNodeDef` targets `nodeData.name === "RadianceAIUpscale"`, a node
type that no longer exists (current upscale nodes are `RadianceUpscaleTiler`,
`RadianceUpscaleImage`, `RadianceUpscaleVideo`, `RadianceUpscaleFaceRestore`) —
this file's widget-folding logic is currently dead code. Also,
`radiance_resolution.js`'s `preset` widget does not drive any widget
visibility (only `enable_video` and `mp_target` do) — switching resolution
presets does not fold/unfold any widgets, which is the existing behavior, not
a regression.

## nodes/generate/resolution.py, js/radiance_resolution.js — model_type-driven presets & alignment

**Problem:** Model-specific behavior (pixel alignment, video-latent
detection, WAN's 4k+1 frame rule, frame-count stride, `latent_format`) was
keyed off the **preset category** (`VIDEO_PRESET_CATEGORIES`,
`WAN_FRAME_CATEGORIES`, `ALIGN32_CATEGORIES`, `VIDEO_LATENT_FORMAT_MAP`), and
`PRESETS` contained ~51 model-specific entries (Flux/SDXL/SD1.5/WAN/LTX/
Hunyuan/CogVideoX). This coupled the preset list to model behavior and
duplicated logic between image and video paths. Additionally, `_align8`
rounded to the *nearest* multiple of 8, which could round a requested
resolution **down** — making the exact target resolution unreachable.

**Fix — inversion:**
- `PRESETS` is now a plain list of Cinema (12) + Social (4) resolutions,
  model-agnostic. All model-specific preset categories were removed.
- `model_type` is now the single source of truth for: pixel alignment (via
  `SPATIAL_SCALE`, always rounded **up** via the new `_align_up(val, scale)`
  helper — `_align8`/`_align32` are now thin wrappers around it), 5D
  video-latent detection (`VIDEO_MODEL_TYPES`), WAN's 4k+1 frame validation
  (`"wan" in model_type.lower()`), the "Auto (Seconds)" frame stride (8 for
  LTX, 4 otherwise), and `latent_format` (single `LATENT_FORMAT_MAP` lookup,
  used for both image and video latents — `VIDEO_LATENT_FORMAT_MAP` removed).
- Default `preset` changed from the removed `"Flux Square (1024×1024)"` to
  `"HD 1080p (1920×1080)"`.
- `generate()` now returns `computed_width`/`computed_height` in its `ui`
  payload; `radiance_resolution.js` adds an `onExecuted` hook that writes
  these final, model_type-aligned values back into the `width`/`height`
  widgets after each run (the preset callback still sets a raw baseline
  immediately for visual feedback).
- `radiance_resolution.js`: removed the preset → `enable_video`/`model_type`
  auto-detection (matched preset names against "WAN"/"LTX"/"Hunyuan"/
  "CogVideoX"/"Flux"/"SDXL" substrings) — no longer meaningful now that
  presets are model-agnostic.

**Deferred (not addressed here):**
- WAN previously got a 16px alignment heuristic via `WAN_FRAME_CATEGORIES`;
  it now falls back to the 8px default (no `SPATIAL_SCALE` entry for WAN).
  Revisit if WAN actually requires 16px alignment.
- `Cosmos (16ch)`, `CogVideoX (16ch)`, `Mochi (12ch)` are excluded from
  `VIDEO_MODEL_TYPES` — no preset ever exercised a 5D latent for these, so
  `TEMPORAL_SCALE`/5D-shape correctness for them is unverified.
- Flux.1 vs Flux.2 (and other version-specific) alignment distinctions are
  not further differentiated beyond the existing `SPATIAL_SCALE`/
  `LATENT_CHANNELS` entries.

### Follow-up — model_type-driven `enable_video` auto-toggle + TEMPORAL_SCALE audit

**Problem:** This refactor removed the old preset-name-based
auto-toggle of `enable_video`/`model_type` but didn't add a model_type-based
equivalent, so selecting a video `model_type` (LTXV/WAN/HunyuanVideo) no
longer auto-enabled video mode.

**Fix:** Added `VIDEO_MODEL_TYPES_JS` (mirrors `VIDEO_MODEL_TYPES` in
`resolution.py`) to `radiance_resolution.js` and wired `model_type`'s
callback to toggle `enable_video` on/off accordingly.

**Validated by Albabit** (real WAN run, preset HD 1080p, 10s/24fps):
`Video latent 5D: (1, 16, 61, 135, 240)` and
`1920×1080 (16:9) 2.07MP │ Latent: 240×135×16ch (WAN) │ Video: 241f @ 24.0fps`
— `(241-1)//4+1=61` ✓, `1920/8=240` ✓, `1080/8=135` ✓.

**`TEMPORAL_SCALE` audit against ComfyUI core** (`comfy_extras/nodes_lt.py`,
`nodes_wan.py`, `nodes_hunyuan.py`, `nodes_cosmos.py`, `nodes_mochi.py`,
`comfy/ldm/cogvideo/vae.py`):
- LTXV (8), WAN (4), HunyuanVideo (4), CogVideoX (4, default
  `AutoencoderKLCogVideoX.temporal_compression_ratio`) — all match our
  `TEMPORAL_SCALE`/`_frameStride`. No changes needed for the 4 model_types
  currently in `VIDEO_MODEL_TYPES`.
- Mochi requires stride **6** (`nodes_mochi.py`: `(length-1)//6+1`) — our
  default of 4 would be wrong if Mochi is ever added to `VIDEO_MODEL_TYPES`.
- Cosmos (`nodes_cosmos.py`) has **two different** temporal strides (8 and 4)
  depending on variant — needs clarification before enabling 5D latents for
  Cosmos.

### Follow-up — instant width/height alignment preview on model_type change

**Problem:** Changing `model_type` only updated the aligned `width`/`height`
shown in the UI after running generation (via the `onExecuted` sync added by
the refactor above). Selecting e.g. `LTXV (128ch)` after a `HD 1080p (1920×1080)` preset
still showed `1920x1080` until the next run.

**Fix (`radiance_resolution.js`):**
- Added `SPATIAL_SCALE_JS` (mirrors `SPATIAL_SCALE` in `resolution.py`: LTXV=32,
  Flux.2/Flux.2 Klein=16, default=8) and `_alignUp`/`_applyAlignment` helpers.
- The node now tracks an unaligned "base" resolution (`node._resBaseW/_resBaseH`),
  set from the selected preset, restored from a saved workflow (`onConfigure`),
  or updated live when the user edits `width`/`height` directly (Custom mode).
- `model_type`'s callback re-aligns `width`/`height` from this base instantly —
  e.g. `1920x1080` -> `1920x1088` for LTXV, then back to `1920x1080` for
  `Flux / SD3 (16ch)` (`_align_up` only rounds up, so re-aligning from the
  already-aligned value couldn't recover the smaller alignment — hence the
  separate base tracking).
- `preset`'s callback also re-aligns immediately after setting the raw preset
  values, so switching presets after picking a video `model_type` doesn't
  briefly show un-aligned values.

**Validated by Albabit**: `HD 1080p` -> `LTXV (128ch)` shows `1920x1088`
instantly; switching to `Flux / SD3 (16ch)` correctly returns to `1920x1080`.
`1280x720` (HD 720p) with `Flux.2 / Flux.2 Klein (128ch)` stays `1280x720`
(already a multiple of 16).

### Follow-up — model_type-driven step size, auto-Custom preset, and video_frames N*stride+1 snapping

**`width`/`height` +/- step now matches the current `model_type`'s alignment**
(32 for LTXV, 16 for Flux.2, 8 default) via `_setWidgetStep`/`_syncStepsToModelType`,
so +/- always lands on a valid value (e.g. LTXV: 1088 -> 1056 -> 1024, never 1080).

**Default `preset` changed from `"HD 1080p (1920×1080)"` to `"Custom"`**
(`resolution.py`), matching the default `width`/`height` (1024x1024) so a freshly
added node doesn't show a preset name inconsistent with its displayed resolution.

**Auto-switch `preset` <-> `"Custom"`:** editing `width`/`height` away from the
currently-selected preset's aligned resolution switches `preset` to `"Custom"`
(`_flagCustomIfNotPreset`); editing back to that exact aligned resolution switches
`preset` back to its original name (`_restorePresetIfMatching`). Tracked via
`node._presetRawW/H` (preset's pre-alignment resolution) and `node._lastPresetName`.

**`video_frames` now instantly snaps to a valid `N*stride+1` value** for all
`VIDEO_MODEL_TYPES` (4k+1 for WAN/HunyuanVideo, 8k+1 for LTXV) — both when typing a
value directly and when switching `model_type` to a video model. Generalizes the
existing WAN-only `4k+1` Python warning (`_alignNk1`, using `_frameStride` for the
per-model stride).

**Validated by Albabit**: HD 1080p + LTXV, `-` on height -> 1056 (preset switches to
"Custom"), `+` back to 1088 -> preset returns to "HD 1080p". Typing `100` into
`video_frames` with LTXV -> snaps to `97` (8k+1); with WAN -> snaps to `101` (4k+1).

## Follow-up — added `Chroma (16ch)` model type

Audited against ComfyUI core (`comfy/sd.py` CLIPType.CHROMA, `comfy/latent_formats.py`
Flux) and this repo's `sampler_utils.py` (already maps `"Chroma"`/"ChromaRadiance" ->
`"chroma"` -> flux sampling defaults). Standard Chroma uses the same 16ch / 8px-aligned
Flux latent shape, so it's a low-risk addition:

- `MODEL_TYPES`: added `"Chroma (16ch)"`.
- `LATENT_CHANNELS["Chroma (16ch)"] = 16`.
- `LATENT_FORMAT_MAP["Chroma (16ch)"] = "chroma"`.
- No `SPATIAL_SCALE` entry needed (8px default is correct, same as Flux/SD3).
- Not a video model — no `VIDEO_MODEL_TYPES`/`TEMPORAL_SCALE` change.

**Deferred**: `ChromaRadiance` (pixel-space variant, 3ch, no VAE) is a fundamentally
different case and is NOT covered by this addition.

**Also audited, not added**: `StepVideo` — ComfyUI core in this checkout has no
`latent_formats` class, no `CLIPType`, and no `supported_models` entry for it. The
`sampler_utils.py` "stepvideo" references appear to be forward-looking stubs with no
working load path yet. Adding it to `resolution.py` now would let users select a
model_type the loader can't actually run — deferred until core ComfyUI ships support.
