# Albabit — Modifications & Fixes

Changes applied on top of upstream `fxtdstudios/radiance-beta`.
Fork: `https://github.com/Albabit/radiance-beta`

Organized by node/area. Within each section, fixes are listed roughly in the
order they were made.

---

## Widget visibility / node resize (shared across loader, sampler, resolution, upscale)

- **v2.4 widget-visibility fix** (`radiance_resolution.js`): conditional
  widgets (`video_frames`, `frame_rate`, `batch_size`, `mp_aspect_ratio`) left
  ghost spaces / didn't restore on reload, because
  `widget.inputEl.style.display`/`widget.draw` overrides bypass Vue. Fixed
  with `setWidgetVisible()` (three-mechanism: `options.hidden`, `hidden`,
  `type="hidden"` + `computeSize`/`computedHeight`), deferred 150ms/600ms
  re-application in `onConfigure`.
- **Node resize / stale widgets after preset folding** (`radiance_loader.js`):
  node box didn't resize on preset switch (`refreshNodeSize` mutated
  `node.size[i]` directly — Vue doesn't observe it; fixed via
  `node.setSize(...)`), and repeated `Custom ↔ preset` toggling left all
  widgets visible as if "Custom" (Vue stops reacting to a no-op
  `splice(0,0)` once mounted; fixed by making `_forceWidgetReinsert()`
  unconditional in `setWidgetVisible`). Generalized to
  `radiance_sampler.js`, `radiance_resolution.js` (v2.4 → v2.5),
  `radiance_upscale.js`.
- **Note (not fixed, dead code)**: `radiance_upscale.js`'s
  `beforeRegisterNodeDef` targets `"RadianceAIUpscale"`, a node type that no
  longer exists (current: `RadianceUpscaleTiler`/`Image`/`Video`/`FaceRestore`).
  `radiance_resolution.js`'s `preset` widget doesn't drive widget visibility
  (pre-existing, not a regression).

---

## Dashboards (Project Manager, Menu, Studio, Workspace)

Files: `js/radiance_pm_launcher.js`, `radiance_menu.js`, `radiance_studio.js`,
`radiance_workspace.js`, `project_manager_dashboard.mjs`,
`assets_dashboard.mjs`, `workspace_dashboard.html`.

**Problem:** Opening Manager/Library/Assets from the Project Manager node (or
the FAB menu) returned a 404 when the custom-node folder wasn't named exactly
`radiance` (e.g. `radiance-beta`) — all URLs were hardcoded as
`/extensions/radiance/<file>`, but ComfyUI derives `/extensions/<name>/` from
the actual folder name.

**Fix:** Added `const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');`
and replaced every hardcoded `/extensions/radiance/<file>` with
`` `${_EXT_BASE}/<file>` ``. In `workspace_dashboard.html` (no JS module
scope), used the relative path `r_icon.png`.

---

## Radiance Loader (RadianceUnifiedLoader / RadianceVideoLoader)

Files: `nodes_loader.py`, `loader_utils.py`, `model/detect.py`,
`config/model_map.py`, `js/radiance_loader.js`.

### Preset dropdown was completely non-functional

Selecting a `preset` did nothing (`app.registerExtension({nodeCreated, ...})`
wrapped widget callbacks *after* Vue had already mounted them). Fixed by
switching to `beforeRegisterNodeDef` + `onNodeCreated`/`onConfigure` (mirrors
`radiance_sampler.js`), with `_forceWidgetReinsert()` to restore hidden
widgets when switching back to "Custom".

### model_type / architecture coverage

- Added 6 new `model_type`s + presets to bring the Loader in line with
  Radiance Resolution: **Chroma, Flux.2 Dev/Klein, Cosmos World, CogVideoX,
  Mochi** (`model/detect.py`: `LATENT_CHANNELS`, `_FORMAT_MAP`,
  `CLIP_SLOT_ORDER`, `_CLIP_TYPE_VARIANTS`, `_BASE_CLIP_VRAM`, `_BASE_VRAM`,
  `get_clip_type_enum`; `config/model_map.py`; `js`: `PRESET_SLOTS` +
  `PRESET_CONFIGS`).
- `LATENT_CHANNELS`/`_FORMAT_MAP` had `"ltx"`/`"ltxav"` as 16ch
  (`"ltx_16ch"`) — the LTX-Video VAE (incl. 2.3) is actually **128ch**. Fixed
  to `"ltx_128ch"`, matching `MODEL_VAE_CONFIG["ltx-video"]`. (Only affected
  `model_meta` reporting for downstream QC nodes, not actual loading.)
- `CLIP_SLOT_ORDER["z_image"]`/`["lumina2"]` pointed to `t5xxl`, but both fill
  `llm_encoder` (Qwen3-4B / Gemma-2 2B) → caused "No CLIP encoders provided".
  Fixed to `["llm_encoder"]`.
- **`"ltxav"` was unselectable** in `RadianceVideoLoader`'s Custom-mode
  `model_type` dropdown (~5.5GB VRAM-estimate understatement for manual LTX
  2.3 setups). Root cause: two module-level `MODEL_TYPES` lists in
  `nodes_loader.py` (upstream v3.1 leftover) — the second silently shadowed
  the first, and only the first (dead) one included `"ltxav"`. Removed the
  dead block, added `"ltxav"` to the active list.

### `_ARCH_HEURISTICS` (Auto-Detect) — fixed/added, cross-checked vs `comfy/model_detection.py`

- `lumina2`'s heuristic (`cap_v_projection.weight`) matched no real
  checkpoint — Lumina2/Z-Image were never auto-detected. Fixed to the real
  key pair `cap_embedder.1.weight` + `noise_refiner.0.attention.k_norm.weight`.
  Added `z_image`, distinguished from `lumina2` via `cap_embedder.1.weight`'s
  shape (3840 = Z-Image, 2304 = Lumina2, new `_tensor_dim0()` helper).
- `chroma`/`flux2` were absent → silently misdetected as `"flux"` (wrong CLIP
  slots). Added both, checked *before* `flux` (Chroma Radiance's
  `nerf_blocks.*` excluded/unsupported).
- Added `mochi` (`t5_yproj.weight`), `cosmos`/Cosmos World
  (`blocks.block0.blocks.0.block.attn.to_q.0.weight`), `cogvideox`
  (`blocks.0.norm1.linear.weight`).
- Removed the `_SAFETENSORS_PEEK = 200` key-count limit (listing keys only
  reads the header; some keys like Mochi's `t5_yproj.weight` sort past 200).
- Verified: Z-Image checkpoint now logs `Auto-detected architecture: z_image`;
  Flux still logs `flux` (non-regression).

### `clip_dtype`/`weight_dtype` forced casts on bf16-native models

- Forcing `clip_dtype: "fp8_e4m3fn"`/`"fp16"` on large bf16-native LLM/T5
  encoders risks overflow/NaN (root cause of Z-Image producing black/abstract
  images). Changed `clip_dtype` → `"default"` (bf16 == fp16 size, zero VRAM
  cost) for: HunyuanVideo, Wan 2.1/2.2, LTX Video, LTX Video 13B, LTX Video
  2.3 (+ Low VRAM), Kolors, Lumina2, Z-Image. Z-Image's `weight_dtype` (UNet)
  likewise `"fp16"` → `"default"`. (`"Flux Dev (Low VRAM)"` keeps
  `clip_dtype: "fp8_e4m3fn"` — that's a T5-XXL cast, the established
  low-VRAM pattern, different risk profile.)
- Same issue for **"LTX Video 2.3"**'s `weight_dtype: "fp16"` —
  `ltx-2.3-22b-dev.safetensors` is bf16-native, so this was a no-benefit
  bf16→fp16 cast. Changed to `"default"`.

### LTX Video 2.3 — VAE, Audio VAE & text_projection

- **VAE size-mismatch fix**: `LTX23_video_vae_bf16.safetensors` raised `size
  mismatch` in `VideoVAE` because it was loaded via `load_torch_file()` (no
  metadata) + `comfy.sd.VAE(sd=sd)`, losing the safetensors `config` key.
  Added `_load_vae_sd_metadata()` (RadianceVideoLoader only) using
  `load_torch_file(..., return_metadata=True)`.
- **Ported Baked VAE / Audio VAE / Latent Upscale Model** from the legacy
  loader to `RadianceVideoLoader`: new outputs `AUDIO_VAE`,
  `LATENT_UPSCALE_MODEL`; `vae_name` defaults to **"Baked VAE (from UNET)"**;
  new optional `audio_vae_name` (`"None"` / `"Baked Audio VAE (from UNET)"` /
  standalone file) and `upscale_model_name`; dedicated `_audio_vae_cache` /
  `_upscale_model_cache`; `model_meta` gains `"audio_vae"`/`"upscale_model"`.
- **"Baked (from UNET)" for `text_projection`**: the native LTX 2.3 workflow
  loads the Gemma 3 12B encoder *and* the UNET checkpoint as a second CLIP
  source (the `text_embedding_projection.*` weights are baked into the UNET,
  and this 2-file path selects the `ltxav_te`/`LTXAVTEModel` wrapper instead
  of the bare `gemma3_te`). `assemble_clip_paths()` now takes `unet_path` and
  appends it as a second CLIP source when `text_projection == "Baked (from
  UNET)"`. "LTX Video 2.3" still prioritizes the standalone
  `ltx-2.3_text_projection_bf16.safetensors`; "LTX Video 2.3 (Low VRAM)"
  defaults to "Baked (from UNET)".
- **"LTX Video 2.3" — no `text_projection` fallback**: this preset only
  listed the standalone `ltx-2.3_text_projection_bf16.safetensors` hints; if
  that file is absent, auto-fill left `text_projection` empty. Added
  `"Baked (from UNET)"` as a last-resort hint, mirroring the Low VRAM preset.
- **"LTX Video 2.3 (Low VRAM)" — Audio VAE not extracted**
  (`AssertionError: Audio VAE model is required` from `nodes_lt_audio.py`):
  the JS preset config had no `audio_vae_hints`, so `audio_vae_name` defaulted
  to `"None"` and `extract_audio_vae` stayed `False`. Added
  `"audio_vae_hints": ["Baked Audio VAE (from UNET)"]` (mirrors the
  `text_projection` pattern; widget stays hidden but is correctly filled).
- **Double UNET file read for audio VAE extraction**: when
  `extract_audio_vae=True`, the (often multi-GB) UNET file was read twice —
  once via `load_diffusion_model`/`load_checkpoint_guess_config`, once more
  via `comfy.utils.load_torch_file` for the audio VAE state dict.
  `RadianceVideoLoader.load_radiance_stack` now reads the state dict once and
  reuses it for both `load_diffusion_model_state_dict`/
  `load_state_dict_guess_config` (model + optional baked VAE) and the audio
  VAE extraction; `cached_patcher_init` is set manually on the resulting
  model/VAE patchers to preserve multi-GPU dynamic-delegate reload support.

### Preset hint/auto-fill corrections (real-world filenames, June 2026)

- **Flux Dev / (Low VRAM)**: added `flux1-krea-dev`/`krea-dev` hints.
- **LTX Video 2.3 / (Low VRAM)**: fixed unet hint collision between fp8 and
  full-precision presets; `vae_name` → "Baked VAE (from UNET)"; added
  `upscale_model_name` hints; `llm_encoder` prioritizes `gemma_3_12B_it`
  (`_fp4_mixed` for Low VRAM); `vae_hints`/audio VAE filenames corrected
  (`LTX23_video_vae_bf16`/`LTX23_audio_vae_bf16.safetensors`).
- **LTX Video / 13B**: added "Baked VAE (from UNET)" + corrected
  `ltxvideo_vae` hints (was `ltx_vae`/`ltxv_vae`/`causal_vae`); added
  `upscale_model_name`.
- **Wan 2.1**: fixed `t5xxl` hints to `umt5_xxl_*`; corrected `vae_name` to
  `wan_2.1_vae.safetensors`.
- **Wan 2.2** (new preset): own `unet_hints`/`vae_hints`; vae_hints prioritize
  `wan_2.1_vae` (16ch, correct for t2v/i2v 14B models); `wan2.2_vae` excluded —
  it is for WAN2.2-TI2V-5B (different 48ch architecture, incompatible with 14B
  t2v/i2v models).
- **Wan 2.2 TI2V** (new preset): single-UNET variant (`wan2.2_ti2v_5B_*`); no
  high/low_noise companion (companion auto-detect is a no-op for this filename);
  vae_hints prioritize `wan2.2_vae` (48ch, `vae2_2.py` code path, required for
  TI2V-5B); weight_dtype `default` (native fp16).
- **PixArt Sigma**: corrected `vae_name` to the SDXL-based VAE.
- **Lumina2**: text encoder → `gemma_2_2b_fp16.safetensors` on `llm_encoder`
  (not `t5xxl`); `vae_name` → `ae.safetensors`.
- **Z-Image**: text encoder → `qwen_3_4b.safetensors` on `llm_encoder`;
  `vae_name` → `flux_vae.safetensors`.
- **"Flux Dev (Flux.2)"**: removed — conflated FLUX.1 (`flux1-krea-dev`) with
  true Flux.2 (`flux2-dev` + Mistral-3 + 128ch latent, not supported by the
  clip-slot infra at the time).
- Added `VIDEO_PRESET_NAMES` (`config/model_map.py`) so "Radiance Video
  Loader" only lists video presets and "Radiance Loader" only image presets.

### T5-XXL `clip_hints` quality/correctness pass (June 2026)

- **"Cosmos World"**: Cosmos 1.0 requires the "old" T5-XXL (T5 1.0) encoder,
  not the T5 1.1 `t5xxl_*` used by Flux/SD3/etc. — distinct, non-interchangeable
  checkpoints. `t5xxl` hints now prioritize `oldt5_xxl_fp8_e4m3fn` /
  `oldt5_xxl_fp16` / `oldt5_xxl`, with `t5xxl_fp8_e4m3fn`/`t5xxl_fp16`/`t5xxl`
  kept as last-resort fallbacks.
- **All other `t5xxl` presets** (Flux, SD3, Chroma, CogVideoX, Mochi, LTX
  Video/13B/2.3, Flux.2 Dev/Klein): hint order was `t5xxl_fp8_e4m3fn` before
  `t5xxl_fp16`, defaulting auto-fill to the lower-precision fp8 variant.
  Reordered to prioritize `t5xxl_fp16` (max quality), `t5xxl_fp8_e4m3fn` as
  fallback. WAN/HunyuanVideo (`umt5_xxl_*`) and Cosmos World (`oldt5_xxl_*`,
  see above) unaffected.

### `offload_mode` ("none" / "cpu_offload" / "sequential")

- **"Low VRAM" presets now expose `offload_mode`**: previously hardcoded to
  `"cpu_offload"` (forces the CLIP/text-encoder to load *and run* entirely on
  CPU — ~87s for a 12B Gemma-3 encoder in fp8 vs ~6s on GPU; only needed on
  genuinely ~8-12GB cards). Removed `offload_mode` from
  `_apply_preset_override` (both loaders) and added it to `extra_widgets` for
  "Flux Dev (Low VRAM)" / "LTX Video 2.3 (Low VRAM)" — visible and
  user-editable, JS still sets the previous default (`"cpu_offload"`).
- **Stale-cache fix**: switching `offload_mode` between runs had no effect —
  `unet_key`/`clip_key` didn't include `offload_mode`, so a cached
  CLIP/UNET kept whatever `load_device`/lowvram-patching it was first loaded
  with. Added `offload_mode` to both cache keys.

### Dead code removed

- `loader_utils.py`: `resolve_hint()` (defined, never called).
- `js/radiance_loader.js`: `MODEL_SLOTS` dict (defined, never referenced,
  already stale vs. `PRESET_SLOTS`).
- `config/model_map.py`: removed `offload_mode`/`clip_slots`/`vram_gb`/
  `unet_hints`/`vae_hints`/`clip_hints` from all 26 `CHECKPOINT_PRESETS`
  entries — audited, none of these fields were ever read (`_apply_preset_override`
  only uses `model_type`/`weight_dtype`/`clip_dtype`); they had drifted out of
  sync with the equivalent JS `PRESET_SLOTS`/`PRESET_CONFIGS` (6 presets
  diverged, worst case "LTX Video 2.3 (Low VRAM)" `text_projection`/audio VAE
  hints). JS is now the sole source of truth for hints/slots/offload_mode.

### De-duplicated `load_radiance_stack` pipeline (SSOT refactor)

`RadianceUnifiedLoader.load_radiance_stack` (image) and
`RadianceVideoLoader.load_radiance_stack` (video) were ~80% copy-pasted
(preset application, Auto-Detect, offload setup, VRAM estimation,
UNET/CLIP/VAE/LoRA loading, HUD printing) — every fix above (`offload_mode`
cache-key, audio-VAE double-read, etc.) had to be located and applied in both
copies. Extracted the shared pipeline steps into 10 new free functions in
`loader_utils.py`, per this file's stated SSOT convention: `resolve_divider`,
`apply_checkpoint_preset`, `resolve_architecture`, `setup_offload_mode`,
`estimate_vram_for_load`, `load_unet_and_baked_vae`, `load_clip_stack`,
`load_standalone_vae`, `apply_lora_stack`, plus the private
`_apply_preset_field` helper. `_audio_vae_cache` moved from `nodes_loader.py`
to `model/cache.py` alongside `_unet_cache`/`_clip_cache`/`_vae_cache`
(`_upscale_model_cache` stays in `nodes_loader.py`, video-only). Each
`load_radiance_stack` is now a thin orchestrator; video keeps inline only its
unique sections (baked/standalone Audio VAE, Latent Upscale Model). No
behavior change — `nodes_loader.py` shrank from 1389 to 882 lines; the
extracted logic was verified byte-for-byte against the pre-refactor superset.

### WAN 2.2 — Dual-UNET (MoE) companion auto-detect

WAN 2.2 (t2v, i2v, fun_camera, fun_inhance, vace, ...) uses a two-expert UNET
architecture (Mixture of Experts): a `high_noise` expert for the early denoising
phase and a `low_noise` expert for the late phase, used in sequence via two
KSamplerAdvanced nodes.

- **Auto-detect companion file**: `_find_wan_moe_companion()` (module-level
  helper in `nodes_loader.py`) swaps the `high_noise`/`low_noise` tag in the
  selected filename (case-insensitive detection, case-preserving replacement) to
  derive the companion path. The user selects either file in `unet_name`; the
  companion is resolved automatically.
- **`model_low_noise` output slot**: `RadianceVideoLoader` gains a second
  `MODEL` output (`model_low_noise`). If the companion file is found it is
  loaded and returned; if not found, `model_low_noise` is `None` (a warning is
  logged).
- **Guaranteed semantic ordering**: regardless of which expert file the user
  picks, after companion loading the outputs are swapped if necessary so that
  `model` (first output) always carries the `high_noise` expert and
  `model_low_noise` always carries the `low_noise` expert. This matches the
  expected wire-up: first KSampler <- `model`, second KSampler <- `model_low_noise`.
- **VAE note**: WAN 2.2 t2v/i2v 14B models use `wan_2.1_vae.safetensors` (16ch).
  `wan2.2_vae.safetensors` is for WAN2.2-TI2V-5B (48ch latents, `vae2_2.py`
  code path) and is incompatible with the 14B t2v/i2v models.

### Lowercase output slot names

`RadianceVideoLoader.RETURN_NAMES` and `RadianceUnifiedLoader.RETURN_NAMES`
now use lowercase for the primary type slots (`model`, `model_low_noise`,
`clip`, `vae`, `audio_vae`) to match the convention of other ComfyUI nodes.
Technical/internal slots (`lora_stack`, `upscale_model`, `model_meta`) were
already lowercase and are unchanged.

---

## Radiance Resolution (nodes/generate/resolution.py, js/radiance_resolution.js)

### Per-model VAE compression (spatial & temporal)

- **Spatial**: empty-latent size always divided by a global `LATENT_SCALE =
  8`, producing oversized latents for LTXV (×32 VAE) and Flux.2 (×16 VAE).
  Added `SPATIAL_SCALE` map (LTXV: 32, Flux.2/Klein: 16, default: 8).
- **Temporal (major perf bug)**: the 5D empty latent's frame dimension used
  the raw pixel-space `video_frames` (e.g. 241) instead of the VAE-compressed
  latent frame count (31) — sampler processed ~8x more "frames" than needed.
  Measured: **65.89s** (correct, T=31) vs **443.33s** (bug, T=241) for an
  identical LTX 2.3/20-step run. Added `TEMPORAL_SCALE` map (LTXV: 8,
  WAN/HunyuanVideo/CogVideoX: 4, default: 4); `lat_t = (video_frames - 1) //
  temporal_scale + 1`. Outputs (`frame_count`/`duration_sec`/`video_frames`)
  unchanged — only the latent tensor is resized.

### Restored features (from previous `radiance` version)

- `frame_computation` combo (`"Manual (Frames)"` / `"Auto (Seconds)"`) +
  `duration_seconds` input; in Auto mode `video_frames = duration_seconds ×
  frame_rate`, aligned to the model's temporal stride.
- Multi-output `RETURN_TYPES`: `("LATENT", "INT", "INT", "INT", "STRING",
  "FLOAT", "INT", "STRING", "FLOAT")` (latent, width, height, channels, info,
  frame_rate, frame_count, latent_format, duration_sec) — drives Sampler Pro
  and other downstream nodes directly.

### `model_type`-driven inversion (replaced preset-category system)

Previously, pixel alignment / video-latent detection / WAN's 4k+1 rule /
frame stride / `latent_format` were keyed off **preset category**
(`VIDEO_PRESET_CATEGORIES`, `WAN_FRAME_CATEGORIES`, `ALIGN32_CATEGORIES`,
`VIDEO_LATENT_FORMAT_MAP`, ~51 model-specific `PRESETS` entries), and `_align8`
rounded to the *nearest* multiple of 8 (could round down).

- `PRESETS` is now a plain, model-agnostic list of Cinema (12) + Social (4)
  resolutions.
- `model_type` is the single source of truth: `SPATIAL_SCALE` +
  `_align_up()` (always rounds **up**), `VIDEO_MODEL_TYPES` (5D latent),
  WAN's 4k+1 rule, `TEMPORAL_SCALE` (Auto-Seconds stride), `LATENT_FORMAT_MAP`.
- `generate()` returns `computed_width`/`computed_height`; JS `onExecuted`
  writes the aligned values back into `width`/`height`.
- JS: removed preset-name-based `enable_video`/`model_type` auto-detection;
  added `model_type`'s callback to auto-toggle `enable_video`
  (`VIDEO_MODEL_TYPES_JS`); added instant width/height re-alignment on
  `model_type` change (`SPATIAL_SCALE_JS`/`_alignUp`/`_applyAlignment`,
  tracking unaligned "base" resolution `node._resBaseW/H`); `+`/`-` step size
  matches alignment (`_setWidgetStep`); `preset` auto-switches to/from
  "Custom" when width/height is edited; `video_frames` snaps to `N*stride+1`.
- Default `preset` → final default **`"Custom"`** (1024×1024).

**`model_type` coverage** (audited against `comfy/sd.py`,
`comfy_extras/nodes_lt.py`, `nodes_wan.py`, `nodes_hunyuan.py`,
`nodes_cosmos.py`, `nodes_mochi.py`, `comfy/ldm/cogvideo/vae.py`):
- Merged equivalent entries: `"Flux/SD3 (16ch)"` + `"Lumina2/Z-Image (16ch)"`
  → `"Flux / SD3 / Lumina2 / Z-Image (16ch)"`; `"SDXL/SD1.5 (4ch)"` +
  `"PixArt/AuraFlow/Kolors (4ch)"` → merged.
- Added `"Chroma (16ch)"` (distinct sampler defaults, kept separate from Flux).
- `VIDEO_MODEL_TYPES` covers 6 models: WAN/HunyuanVideo/CogVideoX (×4), Mochi
  (×6), LTXV/"Cosmos World (16ch)" (×8). "Cosmos World" = Cosmos 1.0 only
  (Predict2 ×4 deferred).
- `"Auto (Flux 16ch)"` → **`"Manual"`** (new default, fully unconstrained:
  scale/temporal ×1, 16ch, `latent_format="flux"`).

**Deferred / audited-not-added**: `ChromaRadiance` (pixel-space, no VAE —
fundamentally different), `StepVideo` (no working load path in this ComfyUI
checkout), Cosmos Predict2.

### Cleanup

- Removed `_align8`/`_align32` (dead/redundant with `_align_up`).
  `_mp_target_dimensions()` now takes `align_val` directly.
- `MODEL_BASE_VRAM`: removed unreachable `"sd15": 2.5`; added `"chroma": 12.0`.
- `_estimate_vram()` takes `spatial_scale: int = 8` instead of hardcoding
  `w//8, h//8` (was under/overestimating LTXV/Flux.2/Manual).

### JS preset auto-fill

Selecting a `preset` auto-fills `width`/`height` (from `(WxH)` in the name),
auto-toggles `enable_video`, sets `model_type`, resets `scale_factor` to 1.0.

---

## Radiance Sampler (js/radiance_sampler.js, nodes_sampler.py)

- **`sigmas_override` widget-greying regression**: the v3 JS rewrite dropped
  `checkSigmaConnection()` entirely — connecting an active `sigmas_override`
  no longer greyed out the now-inert widgets. Fixed: added `SIGMA_OVERRIDE_WIDGETS`
  constant (`steps`, `denoise`, `scheduler`, `scheduler_mode`, `flux_shift`,
  `terminal_sigma_to_zero`, `ays_schedule`, `custom_ays_anchors`,
  `force_exact_steps` — `start_step`/`end_step` intentionally excluded as they
  still slice the override to produce the v3 `sigmas_remaining` output);
  `isSigmaOverrideActive()` checks the upstream node's mode (Muted=2,
  Bypassed=4); `updateSigmaLocks()` follows the same `disabled`/`inputEl`
  pattern as `updateUILocks()`; called from `toggleFields()` (covers all
  existing event paths) and via `setInterval(250ms)` in `onNodeCreated` (needed
  for upstream mute/bypass, which don't fire `onConnectionsChange` on this
  node). Also restored the `logger.info` warning in `_prepare_sigmas()` that
  disappeared alongside the JS regression.

- **`sigma_report`/`latent_meta` dead computation removed**: both were computed
  on every `sample()` call (`nodes_sampler.py`) but never included in the v3
  return tuple — leftover from v2 `RETURN_TYPES ("LATENT","SIGMAS","STRING","STRING")`,
  replaced by `sigmas_remaining`/`sigma_plot` without removing the old
  computation. Removed both call sites and `_build_latent_meta` from the
  `nodes_sampler.py` imports (still defined in `sampler_utils.py`, still
  tested via `test_sdr_conditioning.py` which now imports it directly from
  `sys.modules["sampler_utils"]`). `build_sigma_report` import retained —
  it has a second live call site (trivial sigma schedule warning, line ~807).

- **Dead imports `MULTI_COND_MODES` / `merge_conditionings` removed**
  (`nodes_sampler.py`): multi-conditioning (`positive_2`, `cond_weight_b`,
  `multi_cond_mode`) was intentionally dropped in v3 (confirmed by
  fxtdstudios). Both imports had zero call sites in `RadianceSamplerPro`.
  `route_conditioning` retained — live call site at `sample()` line ~827.

- **Empty spaces in Custom mode after a named preset**: `resolveModelType`
  trusted the backend's `model_type="ltxav"` default for *every* preset
  (hiding LTX-only widgets even for Flux); `applyPreset` never updated
  `model_type` for presets without one. Fixed: `resolveModelType` checks the
  **preset name first**; `inferModelTypeForPreset` + `applyPreset` write the
  inferred `model_type`; `_forceWidgetReinsert()` forces Vue to remount
  hidden→visible widgets; `tile_size`/`tile_overlap`/`tile_blend` hidden in
  Custom mode when `tile_mode=false`.
- **Page reload hid all widgets**: `onNodeCreated`'s 150ms timer could fire
  mid-`graph.configure()`, reading the still-default `preset="None"`. Fixed:
  `onConfigure` sets `node._configuredByLoad = true` synchronously;
  `onNodeCreated` bails if set.
- **`model_type` showed "ltxav" for non-LTX presets**: `applyPreset` now
  calls `inferModelTypeForPreset` to write the correct family; switching to
  None/Custom resets `model_type` to `"auto"`.

---

## Radiance Video Sampler (nodes/video/t2v.py)

- **Triplicated noise-shape + model-defaults logic factored out**: the 12-line
  noise shape calculation (`sc`/`tc`/`ch`/`temp` → `(B,C,T,H,W)`) and the
  14-line `dit_config` parsing + sampling defaults resolution were copy-pasted
  across `RadianceVideoLatentNoise.generate()`, `RadianceT2VPipeline.generate()`,
  and `RadianceI2VPipeline.generate()`. Extracted into two module-level helpers
  in `t2v.py`: `_resolve_dit_config()` (T2V + I2V) and `_build_noise_shape()`
  (all three). No behaviour change — each `generate()` now calls the helpers
  instead of inlining the logic.

- **Dead `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` removed from
  `t2v.py`**: same trap as the Loader audit — the live registry is
  `nodes/video/__init__.py`; `t2v.py`'s block was never read by ComfyUI and
  had drifted (display names used "◎ Radiance Video ..." prefix instead of
  "◎ Video ...", 4 HDR nodes missing). Replaced with a one-line comment
  pointing to the authoritative registry.

- **`cfg_schedule_json` silent failure + misleading report fixed**: when the
  JSON was invalid or malformed, the `except Exception: pass` block silently
  fell back to the static CFG value — no warning, no indication of failure.
  Worse, the sampler report still showed `(from schedule)` regardless. Fixed:
  added `logger.warning` on parse failure; introduced `_cfg_from_schedule`
  boolean to gate the report label on actual parse success, not just on
  whether the string was non-empty.

- **NaN/Inf guard added to `_comfy_sample()`**: `RadianceSamplerPro` already
  sanitizes non-finite values after sampling; `RadianceVideoSampler` had no
  equivalent. CFG blowups, fp16/bf16 overflow, or a degenerate schedule can
  produce NaN/Inf that silently decode to black or corrupt frames with no error.
  Added the same guard: detects non-finite values, logs a warning with the
  count, and sanitizes via `torch.nan_to_num`.

- **`_comfy_sample()` migrated from `KSampler` to `sample_custom`** (`t2v.py`):
  fixes `'dict' has no attribute is_nested'` crash with LTX 2.3 on ComfyUI
  v0.26.0. Root cause: `_comfy_sample()` was passing a LATENT dict
  `{"samples": tensor}` as `latent_image` to the legacy `KSampler` API;
  ComfyUI v0.26.0 now calls `.is_nested` directly on that argument inside
  `CFGGuider.sample()`, which fails on plain dicts. Fix: dict unwrapped to
  raw tensor, then migrated to `comfy.samplers.sampler_object()` +
  `comfy.sample.sample_custom()` — the same modern API used by
  `RadianceSamplerPro`. Sigma computation replicates `KSampler.set_steps()`
  logic (penultimate-sigma discard, partial-denoise slicing). Affects all
  three callers: `RadianceVideoSampler`, `RadianceT2VPipeline`,
  `RadianceI2VPipeline`.

---

## Radiance HDR VAE Decode (hdr/vae.py, fast_vae.py, nodes/generate/engine.py)

### RUDRA model_type detection + graceful fallback (PR #12 — fix/rudra-model-detection)

Both `RadianceHDRVAEDecode.apply()` and `RadianceNDISender.apply()` detected
the RUDRA `model_type` with a 2-bucket heuristic (`_ch == 16 → wan/flux else →
sdxl`). Any latent with 12ch (Mochi) or 128ch (LTX-Video, Flux.2 Klein) was
misclassified as `"sdxl"` → `nn.Conv2d` channel-mismatch crash.

- Added `detect_rudra_model_type(latent_channels, is_video, vae)` in
  `fast_vae.py` (SSOT for both nodes): 128ch+5D → `ltx-video`, 128ch+4D →
  `flux2-klein`, 12ch → `mochi`, 16ch+video → `wan`/`cogvideox`/`cosmos`,
  16ch+image → `flux`, 4ch → `sdxl`.
- `_FILM_CONDITIONED_MODEL_TYPES = {"flux2"}` — FiLM-conditioned checkpoint
  skipped cleanly (returns `None`) until upstream `dr_dim` integration lands.
- `_DECODER_TYPE_FALLBACKS` table replaces hardcoded flux↔wan if/elif:
  `ltx-video → ["ltx"]` (on-disk turbo file has no "-video" suffix),
  plus RUDRA README cross-model reuse (chroma/sd3/lumina2 → flux, etc.).
- `load_radiance_decoder_weights()` now returns `None` (not a randomly-
  initialised model) on "no checkpoint found" or `strict=True` load failure,
  with explicit console messages. `hdr_mode`/`source_space` forcing gated on
  `turbo_decoder is not None`.
- `RadianceNDISender` `model_size="turbo"` naming bug fixed → `"rudra_turbo"`.
- Cache key extended to `(channels, n_upsample, is_full)` — prevents 128ch
  ltx-video (n_upsample=3) / flux2-klein (n_upsample=4) collision.

**Manual validation:** Flux/SDXL unchanged; LTX 128ch no longer crashes (loads
via `"ltx"` fallback); Mochi 12ch cleanly falls back to standard VAE.

### LTX tiled decode b-update fix (fix/ltx-rudra-video-decode)

In `_tiled_decode()` (`hdr/vae.py`), RUDRA's 5D→4D reshape runs *before* the
BHWC permute, so `tile_decoded` exits as 4D. The FIX-4 `ndim==5` branch that
updates `b` from 1 to `B*F` never fires. The lazy accumulator is allocated as
`(1, H, W, C)` while each tile produces `(F, H, W, C)` → `RuntimeError: size
mismatch` for any tiled resolution.

Fix: after the BHWC permute, `b = tile_decoded.shape[0]` when
`_tile_video_frames is not None and turbo_decoder is not None`.

**Validated:** 768×432 output (2×1 tiles at tile_size=512), 16 RHDR frames
exported, no crash.

### LTX RUDRA warning log (fix/ltx-rudra-video-decode)

`logger.warning(...)` added in `load_radiance_decoder_weights()` after a
successful `ltx-video` checkpoint load: the decoder was trained on isolated
still images (T=1, no temporal context) via `ltx_vae.safetensors` (LTX v1),
while inference receives multi-frame causal video latents from LTX 2.3. This
mismatch causes abstract noise; the decoder needs retraining on real LTX 2.3
video data.

### 4ch spurious warning removed (`detect_rudra_model_type`)

`detect_rudra_model_type()` previously tried to confirm SDXL architecture via
the VAE class name (`"xl" in _vae_cls`), then emitted a warning when the check
failed. In practice, ComfyUI wraps every VAE — SDXL and SD 1.5 alike — in the
same generic `comfy.sd.VAE` class, so the check could never succeed. Since
`"sdxl"` is the only 4ch RUDRA checkpoint that exists, the VAE-class check and
the warning were removed; the function now returns `"sdxl"` unconditionally for
4ch latents.

### Temporal chunking (`temporal_size` / `temporal_overlap`)

`RadianceVAE4KDecode` and `RadianceHDRVAEDecode` gain two new optional inputs:

- **`temporal_size`** (INT, default 0): latent frames per temporal chunk.
  `0` = disabled (all frames decoded at once, same as before).
- **`temporal_overlap`** (INT, default 0): overlap in latent frames between
  consecutive chunks.

**Architecture:** Temporal chunking is handled in `decode()`, *before* the
`pix_h <= ts_px` spatial decision. This is deliberate: for 3D-native VAEs like
Mochi (848×480), the spatial footprint fits within one tile, so `_tiled_decode()`
is never called — placing temporal chunking inside `_tiled_decode()` (initial
approach) caused it to be silently bypassed, resulting in an OOM crash during
testing. Moving it to `decode()` ensures it fires for all 5D latents regardless
of spatial size or tiling mode.

When `temporal_size > 0` and the latent is 5D with `T > temporal_size`, `decode()`
splits the T axis into chunks and calls itself recursively with `temporal_size=0`
per chunk (to prevent infinite recursion). Each chunk independently chooses tiled
or direct decode based on its spatial size. For overlapping chunks, the
overlapping pixel frames at the end of each non-last chunk are trimmed (the next
chunk's start covers those frames with more intra-chunk context). `_tiled_decode()`
has no temporal params — temporal and spatial concerns are fully orthogonal.

The `RadianceHDRVAEDecode` node inherits the new inputs automatically via
`**safe_kwargs`; no changes to `engine.py` were required.

**Use case:** Mochi OOM at 848×480 / 49 frames — the full (1, 12, 9, 53, 60)
latent exhausts 32GB VRAM when decoded in one shot alongside bf16 models.
Setting `temporal_size=2` with `temporal_overlap=1` splits the 9 latent T frames
into 8 overlapping chunks, keeping each VAE forward pass well within available
VRAM.

### Temporal chunking — post-merge polish

Three additional fixes applied after the initial PR merge:

- **`soft_empty_cache()` before each chunk** (`hdr/vae.py`): calls
  `comfy.model_management.soft_empty_cache()` at the start of each temporal
  chunk iteration. This nudges ComfyUI's DynamicVRAM scheduler to offload idle
  models (UNET, text encoder) to CPU before the VideoVAE conv activations are
  allocated, significantly reducing the risk of OOM even when large models
  remain staged.

- **Per-chunk `pix_per_lat` in overlap trimming** (`hdr/vae.py`): the overlap
  pixel count was previously computed from the first chunk's frame ratio and
  applied uniformly to all chunks. Fixed to compute `pix_per_lat` per chunk
  inside the trim loop — correct for future 3D causal VAEs where padding may
  produce asymmetric pixel-per-latent-frame ratios.

- **Per-chunk diagnostic warning suppressed** (`hdr/vae.py`): the v4.5
  `source_space='Linear'` luma heuristic warning was firing once per temporal
  chunk (8× in a typical Mochi run). Gated on `not _quiet_diag` so it fires
  at most once per top-level `decode()` call.

- **`hunyuanvideo` dead entry removed** (`fast_vae.py`): `"hunyuanvideo": ["wan"]`
  in `_DECODER_TYPE_FALLBACKS` was never reachable — `detect_rudra_model_type()`
  maps HunyuanVideo (16ch video) directly to `"wan"` before any fallback lookup.
  Entry removed.

- **`LATENT_FORMAT_MAP` extended** (`hdr/vae.py`): added `12 → "mochi_12ch"` and
  `128 → "ltx_128ch"`. Previously these channel counts fell through to the
  `"unknown_Nch"` fallback, causing misleading `fmt=sd_4ch` labels in decode logs
  for Mochi (12ch) and LTX-Video (128ch). Also annotated existing entries with
  accurate model names.

### Test suite — temporal chunking (`tests/test_temporal_chunking.py`)

13 new tests added, all passing:

- **`TileEngine.compute_tiles`** (3 tests): confirms equal-sized chunks with
  `t_ov > 0`, full T-axis coverage, and single-chunk edge case.
- **Overlap trimming — per-chunk `pix_per_lat`** (3 tests): validates the
  asymmetric-chunk case where first-chunk ratio would be wrong (confirms fix),
  equal-chunk path unchanged (no regression), minimum-1-frame guard.
- **`decode()` smoke** (3 tests): no-overlap and with-overlap frame count
  assertions using a mock VAE; `temporal_size=0` confirmed to disable chunking.
- **`LATENT_FORMAT_MAP`** (4 tests): new 12ch/128ch entries, existing entries
  unchanged, unknown-channel fallback.

---

Tests: 1395 pass (41 unrelated gsplat/splatting tests excluded — CUDA DLL not
available in this environment).
