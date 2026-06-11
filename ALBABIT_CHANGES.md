# Albabit — Modifications & Fixes

Changes applied on top of upstream `fxtdstudios/radiance-beta`.
Branch: `fix/bugs` — Fork: `https://github.com/Albabit/radiance-beta`

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
