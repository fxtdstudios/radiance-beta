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
- **Wan 2.2** (new preset): own `unet_hints`/`vae_hints` (`wan2.2_vae`,
  fallback `wan_2.1_vae`); MoE multi-file UNETs / t2v-i2v variants deferred.
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

## Radiance Sampler (js/radiance_sampler.js)

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

## RUDRA decoder (fast_vae.py, RadianceHDRVAEDecode, RadianceNDISender)

**Problem:** both `RadianceHDRVAEDecode.apply()` and `RadianceNDISender.apply()`
detected the RUDRA `model_type` with a 2-bucket heuristic
(`_ch == 16 -> wan/flux else -> sdxl`, resp. `ch >= 16 -> flux else -> sdxl`).
Any latent with 12ch (Mochi) or 128ch (LTX-Video, Flux.2, Flux.2 Klein) was
misclassified as 4ch "sdxl"; `load_radiance_decoder_weights` then built a
4-channel decoder, loaded the (shape-compatible) sdxl checkpoint successfully,
and the caller fed it a 12/128-channel latent — `nn.Conv2d` channel-mismatch
crash whenever the RUDRA decoder was enabled for these models.

**Fix:**

- Added a single shared `detect_rudra_model_type(latent_channels, is_video,
  vae)` helper in `fast_vae.py` (SSOT for both nodes), replacing both
  heuristics.
  - 128ch: `ltx-video` for 5D (video) latents, `flux2-klein` for 4D (image)
    latents — the only two 128ch entries in `MODEL_VAE_CONFIG`, distinguished
    by shape alone.
  - 16ch: `cogvideox`/`cosmos`/`wan` for video (by VAE class name), `flux` for
    image (covers Chroma/Z-Image/Qwen/SD3/Lumina2, which share Flux's 16ch
    VAE).
  - 12ch: `mochi` (only 12ch entry). 4ch: `sdxl` (with the existing
    "could not confirm SDXL architecture" warning for non-XL 4ch VAEs).
- **Flux.2 (non-Klein) FiLM architecture excluded**:
  `rudra_turbo_decoder_flux2_ema.safetensors` uses a FiLM-conditioned
  architecture (`film_in`/`film_out`/`projection`) incompatible with
  `RadianceTurboDecoder`/`RadianceFullDecoder` — confirmed against the
  upstream RUDRA README, where "Flux.2" (non-Klein) is absent from the
  supported-file table (only "Flux.2 Klein" is listed). New
  `_FILM_CONDITIONED_MODEL_TYPES` set makes `load_radiance_decoder_weights`
  skip this `model_type` cleanly (returns `None`) until upstream's `dr_dim`
  decoder integration lands.
- **`load_radiance_decoder_weights` no longer returns random weights**:
  previously, both "no checkpoint found" and a failed `strict=True` load fell
  through to returning the model with its randomly-initialised weights
  ("output will be garbage"). Both paths now log and return `None`, cached, so
  callers fall back to the standard (slower, correct) VAE decode instead of
  producing noise.
- **`RadianceHDRVAEDecode.apply()`** previously force-set `hdr_mode` to
  "Compress (Log)" and `source_space` to a log curve *unconditionally* whenever
  `rudra_decoder=Enabled` — even if the decoder load silently failed/returned
  garbage. Now this forcing only happens when `load_radiance_decoder_weights`
  actually returns a decoder; otherwise the user's original `hdr_mode`/
  `source_space` are preserved and the standard VAE decode runs normally.
- **`RadianceNDISender` `model_size="turbo"` naming bug**: on-disk checkpoints
  are named `rudra_turbo_decoder_*_ema.safetensors`, but the node passed
  `model_size="turbo"` — the generated candidates (`turbo_decoder_*_ema...`)
  never matched, so NDI turbo mode silently always fell back to the raw image
  input. Aligned to `"rudra_turbo"` (matches `RadianceHDRVAEDecode`'s
  `decoder_size` dropdown). A `None` return from
  `load_radiance_decoder_weights` now raises, caught by the existing
  `try/except` that falls back to the image input (BUG 6 FIX path).
- **Decoder cache key collision**: the cache was keyed on
  `(channels, is_full)` only. With 128ch routing fixed, `ltx-video` (8x VAE,
  `n_upsample=3`) and `flux2-klein` (16x VAE, `n_upsample=4`) would collide and
  could return a wrong-shape decoder. Added `n_upsample` to the cache key.
- **`ltx-video` turbo checkpoint never found**: the on-disk turbo file is named
  `rudra_turbo_decoder_ltx_ema.safetensors` (no "-video"), but the candidate
  list only tried the canonical `model_type` ("ltx-video"). Replaced the
  hardcoded flux↔wan fallback if/elif with a `_DECODER_TYPE_FALLBACKS` table
  that adds `"ltx"` as a fallback token for `"ltx-video"`, and also covers the
  RUDRA README's documented cross-model decoder reuse (e.g. "Z-Image: use the
  Flux decoder") for chroma/cosmos/sd3/lumina2 (→ flux) and
  cogvideox/hunyuanvideo (→ wan).

**Known limitation:** Flux.2 (non-Klein) and Flux.2 Klein are both 128ch/4D
with identical generic VAE classes — no signal distinguishes them.
`detect_rudra_model_type` defaults this bucket to `"flux2-klein"` (the
README-documented/supported checkpoint). A true non-Klein Flux.2 user gets
Flux.2 Klein's RUDRA weights applied to Flux.2 latents (shape-compatible,
won't crash) rather than a clean standard-VAE fallback — acceptable for now
since the FiLM `flux2_ema` checkpoint is never referenced by either code path
either way.

---

Tests: 1382 pass (41 unrelated gsplat/splatting tests excluded — CUDA DLL not
available in this environment).
