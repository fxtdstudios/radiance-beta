# Changelog

All notable changes to FXTD Radiance will be documented in this file.

## [3.1.2] - 2026-07-02 ("GPU-First Release Candidate")

GPU-first completion pass for HDR, RUDRA decode, denoise, motion/flow, upscale, and VFX finishing paths, plus the missing HDR tone-map node migration.

### Fixed

- **HDR Tone Map now loads from the organized HDR package.** `RadianceHDRToneMap` and `RadianceHDRExpandDynamicRange` are registered through `nodes/hdr/`, so saved workflows and the HDR menu can resolve the tone-map node again.
- **HDR Tone Map is GPU-first end to end.** The implementation no longer falls back to NumPy or forces CUDA results back to CPU; tone mapping now stays on the selected Torch device.
- **RUDRA compatibility restored on the public decoder API.** `radiance.model.vae` now reuses the maintained fast decoder implementation, including dynamic-range conditioning (`dr_dim` / `dr_proj`), predictor fallback, checkpoint inference, and hardened explicit-checkpoint loading.
- **Version metadata synchronized.** Python package metadata, runtime constants, README badge, and `package.json` now agree on `3.1.2`.

### Changed

- ACES/HDR color operations, denoise, optical-flow motion blur, multipass helpers, and upscale inference paths were tightened to prefer Torch/GPU execution and avoid unnecessary CPU transfers.
- The node-key snapshot was updated for the intentional HDR tone-map registry restoration.

### Testing

- Restored and enabled the RUDRA compatibility tests that were previously skipped.
- Release verification run performed with real Torch, OpenEXR, OpenColorIO, and colour-science dependencies.

## [3.1.1] - 2026-06-02 ("Fidelity and Trust Release")

Correctness, safety, and delivery-trust fixes for the 32-bit / HDR / EXR pipeline, plus real-AOV ingestion and hardened model loading. Closes the critical data-integrity and security findings from the pre-release engineering review.

### Fixed

- **HDR/EXR load no longer crushes scene-linear data.** The image-to-tensor path previously divided any array whose brightest pixel exceeded 2.0 by 255 (or 65535), silently darkening every real HDR/EXR plate ~255x. Normalization is now driven by the source format: integer formats normalize by their type max; float formats (EXR/HDR/float-TIFF) pass through untouched. Read -> Write of an EXR is now lossless.
- **EXR writer can no longer write a silent 0-byte or downgraded file.** It raises a clear error when no EXR backend is available instead of reporting a phantom success.
- **EXR writer preserves alpha and single-channel mattes** (1->RGB grayscale, 3->RGB, 4->RGBA), and no longer crashes on `(H,W)` matte input.
- **`RadianceWrite` surfaces failures** by re-raising on write error (node turns red) instead of swallowing the exception.
- **Sampler non-finite guard.** `RadianceSamplerPro` detects NaN/Inf in the sampled latent, warns, and sanitizes with `nan_to_num` so blown CFG/precision runs are visible instead of shipping black frames.
- **HDR VAE Decode metadata output.** The node now emits its decode-settings JSON as a second `STRING` output (previously built and discarded).
- Removed internal imports of deprecated shim paths, eliminating load-time `DeprecationWarning`s.

### Security

- **`torch.load` hardened.** All shippable checkpoint loads use `weights_only=True` (VAE, fast VAE, multipass depth/normal models, training data prep), preventing arbitrary-code-execution from malicious `.ckpt`/`.pth` files.
- **Model downloads are atomic + integrity-checked.** The loader and multipass downloaders write to a temp file and atomically move into place, with optional SHA-256 verification and `RADIANCE_LOADER_OFFLINE` / `RADIANCE_UPSCALE_OFFLINE` switches for airgapped setups.

### Added

- **Multipass: AOV Reader.** Reads a real multilayer/AOV OpenEXR (Arnold, Redshift, Karma, Cycles, V-Ray) and splits its named layers into the same outputs as the Multipass Master extractor, so ground-truth render passes flow straight into the EXR-passes writer and relight/comp chain. Scene-linear values preserved; missing layers come through black.
- **Alpha output on the EXR write node.** `RadianceWrite` gained an optional `mask` input, written as the EXR alpha channel (RGBA) for EXR formats.
- **Upscaler HDR + color handling.** Image and video upscalers gained `hdr_mode` (Reinhard tonemap round-trip) and `color_encoding` (linear<->sRGB / linear<->LogC3 OETF round-trip).

### Documentation

- Website docs now render Mermaid workflow diagrams as themed graphs (previously shown as raw code) via `scripts/build_website_docs.py`.

### Testing & CI

- Full `pytest tests/` passes against real torch + OpenEXR + OpenColorIO + colour-science (1347 passed, 34 skipped).
- CI installs real runtime dependencies and runs the whole suite; the publish gate runs the full suite. Removed a stale CI import of a non-existent module.
- Added `tests/test_io_hdr_regression.py` covering the load/write/mask fixes.

## [3.1.0] - 2026-05-25 ("Temporal, Viewer, and Registry Release")

### Added

- Viewer timeline upgrades: filmstrip thumbnails, pinned-frame A/B comparison, and OCIO display transform controls.
- Four color-science nodes: Hue Curves, RGB Curves, White Balance using Bradford adaptation, and Color Space Convert with OCIO-first behavior plus analytical fallback.
- Organized package namespace for color, HDR, I/O, monitor, pipeline, training, upscale, video, VFX, and generation nodes while preserving legacy flat module imports.
- GitHub CI and registry publish workflows for the v3 release path.

### Fixed

- Printer Lights viewer listener leak after tab switches and UI rebuilds.
- HDR VAE decode category placement and repeated engine construction.
- VAE encode/decode/roundtrip docstring placement, category labels, and mode-aware NaN/Inf sanitation.
- Fast VAE trained decoder cache now keys by latent channel count and model mode.
- Viewer grading edge cases, including `luma_mix` fast-path handling, CLog3 output clamping, ACES matrix allocation, vectorized saturation, and bounded progress tracking.

### Release

- Updated Registry metadata for `radiance` v3.1.0 under publisher `fxtdstudios`.
- Restored GitHub/Registry image assets referenced by README and `pyproject.toml`.
- Added `.comfyignore` so Registry packages exclude tests, local scratch files, build caches, and review documents.

---

## [3.0.1] - 2026-05-07 ("Color Science Precision & Demo Tooling")

### Added

- **BT.1886 EOTF** (`nodes_hdr_colorspace.py`): Added `"BT.1886 (TV γ2.4)"` to `_EOTF_MAP`
  — ITU-R BT.1886 display EOTF (γ 2.4) for Rec.709 broadcast reference monitors. Distinct
  from `"Rec.709 (OETF)"` (camera signal encoding) and `"Gamma 2.4"` (generic power curve).
- **Rec.709 / BT.1886 IDT** (`nodes_colorscience.py`): Added `"Rec.709 / BT.1886"` to
  `_COLOR_SPACES` — the correct Input Device Transform for display-referred Rec.709 material.
  OCIO mapping: `"Output - Rec.709"`. Analytical fallback: γ 2.4 linearise / γ 1/2.4 encode.
- **◎ Radiance Bit-Depth Degrade** (`nodes_colorscience.py`): Quantize float images to N-bit
  (4–16) precision. Dither modes: none (hard clip), triangular TPDF, Floyd-Steinberg error
  diffusion. Outputs: quantized image, amplified error delta, banding mask, metrics JSON
  (PSNR dB, max error, dynamic range loss in stops). Critical for demo workflows:
  ARRI LogC / Venice RAW → 8-bit → AI HDR reconstruct → EXR.

### Tests

- **`tests/test_colorscience_v3.py`** (32 tests): BT.1886 decode/encode round-trip,
  `_COLOR_SPACES` contents, `RadianceBitDepthDegrade` node registration, quantisation
  correctness (4-bit ≤16 levels, 8-bit ≤256, 16-bit near-lossless), PSNR monotonicity
  with increasing bit depth, TPDF vs hard-clip divergence, Floyd-Steinberg completion,
  banding mask binary property, alpha channel preservation.

### Changed

- `_COLOR_SPACES` in `nodes_colorscience.py` grouped into labelled sections (scene-linear,
  display-referred SDR, camera log/raw) for clarity.
- `_EOTF_MAP` in `nodes_hdr_colorspace.py` restructured with inline comments separating
  SDR EOTFs, HDR EOTFs, and camera log curves.

---

## [3.0.0] - 2026-04-30 ("The Full Pipeline & Intelligence Update")

### Added — ACES 2.0 Full Pipeline (Pillar 03)

- **◎ Radiance ACES 2.0 RRT+ODT** (`nodes_aces2.py`): Analytical ACES 2.0 Reference
  Rendering Transform with four Output Display Transforms (sRGB D65, DCI-P3 D65,
  Rec.2020 PQ HDR10, Rec.2020 HLG). No external CTL toolchain required.
- **◎ Radiance ACES Input Transform** (`nodes_aces2.py`): 25 IDT matrices covering
  ARRI ALEXA, RED, Sony Venice, Canon C-series, Nikon Z, Panasonic Varicam, Blackmagic,
  and GoPro camera systems.
- **◎ Radiance ACES LMT** (`nodes_aces2.py`): Five LMT presets — Linear, Blue Light Fix,
  Golden, Desaturate Highlights, and Kodak-2383 emulation.
- **◎ Radiance ACES CDL** (`nodes_aces2.py`): ASC CDL v1.2 in ACES working space — slope,
  offset, power, saturation with Rec.709 luma weighting and optional clamp.

### Added — Studio Integrations (Pillar 04)

- **◎ Radiance DaVinci Send** (`nodes_studio_integrations.py`): One-click image/sequence
  export to DaVinci Resolve shared folder with configurable bit depth (8/16 bit, EXR).
- **◎ Radiance Nuke Send** (`nodes_studio_integrations.py`): Write a .nk snippet that
  imports the current image into a Nuke session via Read node.
- **◎ Radiance Shot Metadata** (`nodes_studio_integrations.py`): Attach scene/shot/take
  labels, camera info, and lens data as a JSON sidecar and EXIF-compatible comment.
- **◎ Radiance ASC CDL Export** (`nodes_studio_integrations.py`): Write ASC CDL v1.2 XML
  (single `<ColorCorrection>`) or JSON override file from slope/offset/power/saturation
  values — standard interchange with Resolve, SCRATCH, and Nuke.

### Added — Real-Time Preview (Pillar 05)

- **◎ Radiance False Color Monitor** (`nodes_realtime_preview.py`): Cinema-style false
  colour overlay with 10 configurable exposure zones (adjustable luma thresholds and
  zone colours). Toggle between false-colour and source view per node.
- **◎ Radiance Focus Peaking** (`nodes_realtime_preview.py`): Sobel edge highlight overlay
  for critical focus evaluation. Configurable threshold and peak colour.
- **◎ Radiance Split View** (`nodes_realtime_preview.py`): Side-by-side or top-bottom
  wipe comparison between two images, with adjustable split position and guide line.
- **◎ Radiance Contact Sheet** (`nodes_realtime_preview.py`): Auto-tiled contact sheet
  from a batch of images — configurable grid columns and thumbnail padding.
- **◎ Radiance Flipbook GIF** (`nodes_realtime_preview.py`): Assemble batch frames into an
  animated GIF at configurable FPS and scale for quick motion previews.
- **◎ Radiance Frame Stamp** (`nodes_realtime_preview.py`): Burn timecode, frame number,
  and optional label overlay into images. SMPTE 12M timecode — both NDF (24/25/30/48/50 fps)
  and drop-frame (29.97 DF with d=2; 59.94 DF with d=4). `;` separator for DF, `:` for NDF.
- **◎ Radiance Preview Server** (`nodes_realtime_preview.py`): Ephemeral HTTP server
  (daemon thread) serving the last processed frame as JPEG — lets any browser on the LAN
  see the current result without streaming infrastructure.

### Added — AI Assist Layer (Pillar 06)

- **◎ Radiance Auto Grade** (`nodes_ai_assist.py`): Zone-based ASC CDL matching — aligns
  a source image to a reference still by independently correcting shadow offset, midtone
  slope, highlight power, and global saturation. Strength blend 0–1.
- **◎ Radiance CLIP Match** (`nodes_ai_assist.py`): Cosine similarity match between source
  and a pool of reference images using CLIP ViT-B/32 embeddings. Falls back to
  Bhattacharyya histogram similarity when `transformers` is unavailable. Returns the best
  matching reference image plus a similarity score.
- **◎ Radiance Continuity Check** (`nodes_ai_assist.py`): Shot-to-shot continuity analysis
  — flags luma drift, colour cast, saturation drift, contrast drift, and histogram
  dissimilarity between consecutive frames. Outputs a JSON report + clean/dirty flag.
- **◎ Radiance Grade Prompt** (`nodes_ai_assist.py`): Natural language CDL grading — parse
  35 intent rules ("warmer", "crushed blacks", "pull the highlights", etc.) with intensity
  modifiers (very / slightly / a touch / less). Six preset looks: film, bleach, day-for-
  night, instagram, neon, noir.

### Added — Test Infrastructure (Pillar 07)

- `tests/test_cdl.py`: CDL math, XML round-trip, and node registration tests.
- `tests/test_scene_cut.py`: Scene-cut detection — histogram diff, Sobel edge diff,
  combined method, min-shot-frame enforcement.
- `tests/test_temporal_coherence.py`: Flicker removal and temporal smoothing helpers
  (`_luminance_per_frame`, `_rolling_median`, `_gaussian_smooth`, `_smooth_track`).
- `tests/test_colorscience.py`: Bradford matrix algebra, `_xy_to_XYZ`, and illuminant
  chromaticity coverage — 18 tests (Bradford matrix tests torch-gated).
- `tests/test_curves.py`: HSL round-trip for grey, saturated primaries, and random
  images; hue/saturation range checks.
- `tests/test_optics.py`: Lens distortion, chromatic aberration, anamorphic streaks, and
  vignette — registration, input structure, and return-type validation.
- `conftest.py`: Extended torch stub with 20+ additional attributes so module-level
  `torch.*` calls in all nodes succeed at collection time without real torch.

### Added — AI & Pipeline Nodes (Pillar 08)

- **◎ Radiance LLM Driver** (`nodes_llm_driver.py`): Model-agnostic LLM backend
  configuration — Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google), Ollama (local).
  Single driver JSON wire connects to any downstream AI node.
- **◎ Radiance LLM Prompt** (`nodes_llm_driver.py`): Send text prompts to the configured
  backend; returns response string and echoes driver config.
- **◎ Radiance LLM Image Query** (`nodes_llm_driver.py`): Vision-capable multimodal query
  (base-64 PNG encoded) — falls back to text-only for non-vision backends.
- **◎ Radiance Agent Shot Analyst** (`nodes_agent_pipeline.py`): Per-shot technical analysis
  — exposure, contrast, saturation, dynamic range, and scene classification via LLM.
- **◎ Radiance Agent Grade Advisor** (`nodes_agent_pipeline.py`): AI-driven CDL suggestions
  from shot analysis; outputs slope/offset/power/saturation JSON.
- **◎ Radiance Agent QC Check** (`nodes_agent_pipeline.py`): Automated quality gate —
  runs policy rules and returns pass/fail flag + violation report.
- **◎ Radiance Agent Pipeline** (`nodes_agent_pipeline.py`): Shot Analyst → Grade Advisor →
  QC Check orchestration chain in a single node.
- **◎ Radiance Studio Knowledge Base** (`nodes_knowledge_base.py`): CLIP/HSV image vector
  store with semantic query — index reference stills and retrieve by similarity.
- **◎ Radiance MCP Bridge** (`nodes_mcp_bridge.py`): JSON-RPC MCP server exposing Radiance
  as a tool endpoint; includes client stubs for Nuke, Maya, and ShotGrid.
- **◎ Radiance Policy Preset** (`nodes_policy_guard.py`): Six delivery policy presets —
  Broadcast SDR, Cinema DCP, Streaming HDR, Web SDR, Custom. Outputs JSON policy dict.
- **◎ Radiance Policy Guard** (`nodes_policy_guard.py`): Per-frame delivery compliance gate
  — evaluates peak, clipping, black crush, saturation, and metadata presence against policy.
  Returns original image (pass-through), compliance report STRING, and pass/fail BOOLEAN.
- **◎ Radiance Render Dispatch** (`nodes_render_dispatch.py`): Submit render jobs to
  Deadline, Tractor, and OpenCue; poll status; cancel; list queued jobs.
- **◎ Radiance Audio Cut** (`nodes_audio_cut.py`): Beat/onset detection from audio waveform
  — outputs cut timecodes and Whisper speech transcription (when available).

### Added — DiT Video Model Integration (Tier 1)

- **◎ Radiance DiT Model Config** (`nodes_dit_adapter.py`): Unified architecture config for
  SD-VAE, SDXL-VAE, LTX-Video (128ch), HunyuanVideo (16ch), Wan 2.1 (16ch), CogVideoX
  (16ch), and Mochi-1 (12ch). Outputs spec JSON for downstream adapter nodes.
- **◎ Radiance DiT Latent Adapter** (`nodes_dit_adapter.py`): SD-VAE latent ↔ DiT latent
  channel projection with per-architecture mean/std normalisation. Adaptive channel project
  supports expand (tile) and contract (fold) modes.
- **◎ Radiance DiT Latent Norm** (`nodes_dit_adapter.py`): Per-architecture normalisation
  and denormalisation of DiT latents. Forward and inverse pass.
- **◎ Radiance DiT Inspect** (`nodes_dit_adapter.py`): Latent shape / stats inspector
  reporting channels, spatial dimensions, dtype, min/max/mean.
- **◎ Radiance DiT Frame Split / Merge** (`nodes_dit_adapter.py`): Split video latents
  into per-frame chunks and reassemble — enables frame-parallel DiT workflows.
- **◎ Radiance HDR Video Conditioner** (`nodes_video_hdr.py`): HDR conditioning for DiT
  video models — builds latent sequence from EXR frames with scene-linear encoding.
- **◎ Radiance HDR Video Decoder** (`nodes_video_hdr.py`): Decode DiT video latents back
  to HDR image sequence with optional PQ/HLG EOTF application.
- **◎ Radiance HDR Video Prompt Builder** (`nodes_video_hdr.py`): Construct HDR-aware CLIP
  prompt strings from scene metadata (peak nits, colour space, HDR format).
- **◎ Radiance HDR Video Assembler** (`nodes_video_hdr.py`): Assemble decoded frames into
  an EXR sequence or MP4 (ProRes 4444 / H.265 10-bit).

### Added — Character Consistency System

- **◎ Radiance Character Anchor** (`nodes_character.py`): Build a character profile from a
  reference image — CLIP embedding, HSV histogram, face embedding (facexlib), and metadata.
  Saves to JSON sidecar; returns profile STRING for downstream enforcement.
- **◎ Radiance Character Enforce** (`nodes_character.py`): Inject character embedding into
  CONDITIONING via token-append, IP-Adapter strength, or text-blend strategies.
- **◎ Radiance Character Checker** (`nodes_character.py`): Per-frame cosine similarity check
  against a stored profile. Outputs similarity score, pass/fail flag, and report JSON.
- **◎ Radiance Character Blend** (`nodes_character.py`): Weighted blend of two character
  profiles — useful for gradual style evolution across a sequence.
- **◎ Radiance Character Gallery** (`nodes_character.py`): Load all character profiles from
  a folder; display as structured JSON gallery for selection.
- **◎ Radiance Character Score Timeline** (`nodes_character.py`): Per-shot similarity scores
  serialised as JSON timeline for consistency trend analysis.

### Added — T2V & I2V Pipeline

- **◎ Radiance T2V Wrapper** (`nodes_t2v_pipeline.py`): Text-to-video inference wrapper for
  LTX-Video, HunyuanVideo, Wan 2.1, and CogVideoX. Unified prompt/resolution/steps API.
- **◎ Radiance I2V Wrapper** (`nodes_t2v_pipeline.py`): Image-to-video inference with first-
  frame conditioning; supports motion bucket / flow magnitude parameters.
- **◎ Radiance Video Sampler** (`nodes_t2v_pipeline.py`): Low-level DiT sampler with
  sigma schedule, CFG, and optional restart sampling.
- **◎ Radiance Batch Decode** (`nodes_t2v_pipeline.py`): Decode a batch of video latents to
  pixel frames using the architecture-appropriate VAE.
- **◎ Radiance Video Export** (`nodes_t2v_pipeline.py`): Write decoded frames to ProRes
  4444, DNxHR, H.265, or EXR sequence; embeds timecode metadata.

### Added — AI Upscaler (nodes_upscale.py)

- **◎ Radiance Upscale Tiler** (`nodes_upscale.py`): Anti-seam tiling engine with
  Gaussian-weighted overlap (≥20%), 4-level Laplacian pyramid blending, and per-tile
  confidence map. Handles arbitrary resolution with no visible tile boundaries.
- **◎ Radiance Upscale Image** (`nodes_upscale.py`): Three-tier upscaler for images.
  - Tier 1 — Real-ESRGAN (RRDB, no basicsr): `realesrgan_x4plus`, `x4plus_anime`, `x2plus`
  - Tier 2 — Transformer SOTA (spandrel): HAT-L ×4/×2, SwinIR-L ×4
  - Tier 3 — Diffusion creative: SD ×4 upscaler (stabilityai), SeedVR2 one-step video
  - Auto tier: content classifier (noise/sharpness/saturation/ai-likelihood) selects tier
- **◎ Radiance Upscale Video** (`nodes_upscale.py`): Temporal-coherent video upscaler.
  SeedVR2-style 4n+1 overlapping windows, sinusoidal ramp weights at window edges,
  Lucas-Kanade optical-flow warp for cross-window reference alignment, Laplacian pyramid
  blend at seams. Eliminates inter-chunk flicker.
- **◎ Radiance Upscale Router** (`nodes_upscale.py`): Content-aware tier selector — analyse
  any image/video and route to the appropriate upscale node automatically.
- **◎ Radiance Upscale Face Restore** (`nodes_upscale.py`): Post-upscale face enhancement.
  Face detection: facexlib RetinaFace → OpenCV Haar cascade fallback. Restoration: spandrel
  (auto-detects CodeFormer / GFPGAN from checkpoint) → basicsr CodeFormer → GFPGANer →
  identity. Gaussian-feather composite. `fidelity_weight` controls realism vs. fidelity.
- **◎ Radiance Upscale Colour Fix** (`nodes_upscale.py`): Histogram-match colour-drift
  correction post-diffusion. Fast path: `torch.searchsorted` vectorised CDF inversion
  (~100× faster than pixel-loop). Strength 0–1 blend with original.

### Added — Test Coverage (Pillar 07 sweep)

- `tests/test_upscale.py` (120 assertions): Gaussian kernel normalisation/symmetry, weight
  map positivity, bicubic output shapes, `tiled_upscale` shape/confidence, histogram-match
  identity/strength-0 noop, content classifier keys, tier recommendation, quantisation math,
  model registry completeness, node registration (6 nodes).
- `tests/test_llm_driver.py`: `_driver_from_json` parse + fallback, `_resolve_api_key` env
  resolution, `_DISPATCH` routing for all 4 backends (Claude/GPT/Gemini/Ollama), INPUT_TYPES
  field names, 3-node registration.
- `tests/test_policy_guard.py`: `_luma` BT.709 coefficients, saturation range, gamut
  fraction, `_analyse` key coverage, `_evaluate` tuple return (passed, violations, score),
  metadata presence checks.
- `tests/test_dit_adapter.py`: `_get_spec` for 6 architectures, channel projection shape,
  `_apply_norm` round-trip, config/adapter/norm/inspect/split/merge INPUT_TYPES.
- `tests/test_character.py`: `_has`, HSV histogram, `_cosine_sim` identity/orthogonal/
  opposite, all 6 character node INPUT_TYPES and RETURN_TYPES.
- `tests/test_colorscience_v3.py` *(see v3.0.1)*: BT.1886 decode/encode round-trip,
  `_COLOR_SPACES` contents, `RadianceBitDepthDegrade` quantisation, PSNR monotonicity,
  dither divergence, FS dither completion, banding mask binary property.
- `tests/test_timecode.py` *(Pillar 09 — 68 tests)*: Full SMPTE 12M timecode suite.
  NDF 24/25/30 fps boundary frames; DF 29.97 (d=2) covering frames 0, 1799, 1800, 3597–3599,
  17981–17983, 107891–107892; DF 59.94 (d=4) equivalent boundaries; monotonicity within
  short minutes; dropped-frame absence at non-10th-minute starts; `;` vs `:` separator;
  `_DF_RATES` constant validation; `RadianceFrameStamp` INPUT_TYPES / RETURN_TYPES.

### Added — CI / Packaging (Pillar 10)

- **`.github/workflows/ci.yml`**: Full GitHub Actions CI matrix — Python 3.9 · 3.10 · 3.11 · 3.12
  on `ubuntu-latest`. Installs headless test deps (`opencv-python-headless`, `pytest-cov`),
  runs `pytest` with `--cov`, uploads XML coverage to Codecov (3.11 only), and attaches pytest
  logs as artifacts on failure. Concurrency group cancels in-flight runs on new push.
- **`.github/workflows/ci.yml` — smoke job**: Independent import smoke-test that verifies 11 key
  node modules (`nodes_scene_cut`, `nodes_ai_assist`, `nodes_aces2`, `nodes_realtime_preview`,
  `nodes_llm_driver`, `nodes_policy_guard`, `nodes_upscale`, `nodes_dit_adapter`,
  `nodes_character`, …) import cleanly with minimal stubs — catches syntax errors and bad
  top-level imports independently of the pytest run.
- **`.github/workflows/ci.yml` — lint-config job**: Validates `pyproject.toml` (TOML parse +
  required table check) and all `.github/workflows/*.yml` files (YAML parse) on every push.
- **`.github/workflows/publish.yml`**: Publish workflow triggered by version tags (`v*.*.*`) or
  `workflow_dispatch`. Runs full test gate, then calls `Comfy-Org/publish-node-action@v1` with
  `REGISTRY_ACCESS_TOKEN` secret, then creates a GitHub Release with notes extracted from
  `CHANGELOG.md`. Pre-release tags (`-rc*`) set `prerelease: true`.
- **`pyproject.toml`** — expanded and tightened:
  - Python classifiers now enumerate 3.9 · 3.10 · 3.11 · 3.12 explicitly.
  - Runtime `dependencies` list pinned with minimum versions (`Pillow>=9.0`, `numpy>=1.22`, …).
  - `opencv-python` moved from runtime to `[full]` extra; `opencv-python-headless>=4.7` added
    to `[test]` extra (server-safe, no GUI dependency).
  - `[test]` extra adds `pytest-cov>=4.0` and `pytest-timeout>=2.1`.
  - `[full]` extra extended to cover `diffusers`, `accelerate`, `peft`, `anthropic`, `openai`,
    `google-generativeai`.
  - `Bug Tracker` URL corrected to GitHub Issues.
  - `[tool.coverage.run]` and `[tool.coverage.report]` sections added (`fail_under = 0`,
    ready to raise once baseline is measured).
  - Stale `tb = "short"` (unknown pytest option) removed; `integration` marker added.

### Changed

- All display names standardised to `◎ Radiance …` prefix across all 7 new nodes_*.py files.
- `tests/test_coherence_prior.py` skip guard now also checks `HAS_TORCH` so it correctly
  skips when the conftest torch stub is active (prevents false-pass on stub MagicMocks).

## [2.6.0] - 2026-04-17 ("The HDR Science & Fast VAE Update")

### Added — Color Science

- **◎ Radiance ACES 2.0 Transform** (`nodes_colorscience.py`): Full analytical ACES 2.0
  Reference Rendering Transform (RRT) + four Output Display Transforms — no external CTL
  toolchain required.
  - RRT: cubic Bézier rational polynomial approximation matching CTL within ±0.3%
  - ODT **sRGB D65** — standard monitor / SDR web output
  - ODT **DCI-P3 D65** — digital cinema / wide-gamut display
  - ODT **Rec.2020 PQ (HDR10)** — ST 2084 absolute luminance up to 10,000 nits
  - ODT **Rec.2020 HLG** — ARIB STD-B67 broadcast HDR (BBC/NHK)
  - Full gamut pipeline: ACEScg → XYZ (D60) → D65 Bradford → target display primaries
  - `exposure_offset` (stops), `saturation`, `peak_nits` (cd/m²), `grade_info` passthrough

- **◎ Radiance White Balance** (`nodes_colorscience.py`): Bradford chromatic adaptation in
  three modes — Temperature/Tint (Kang 2002 xy approx), Illuminant-to-Illuminant CAT
  matrix, and Manual RGB gain. Full HDR-safe, no clamping above 1.0.

- **◎ Radiance Color Space Convert** (`nodes_colorscience.py`): OCIO-powered color space
  conversion with artist-friendly dropdown. Falls back to analytical transforms (sRGB OETF,
  LogC3, ACEScg gamut) when no OCIO config is present. Forward and Inverse directions.
  Supports: Linear sRGB, ACEScg, ACEScc, ACEScct, sRGB, Rec.709, LogC3, LogC4, F-Log2,
  C-Log3, Log3G10, DaVinci Intermediate, BMD Film Gen5, V-Log, N-Log.

### Added — AgX Tone Mapping (Full Pipeline)

- **AgX Full Pipeline** (`hdr/tonemap.py`): Replaced the previous smoothstep approximation
  with a mathematically correct AgX operator (Blender/Troy Sobotka, BSD-licensed).
  - `_AGX_M_IN` / `_AGX_M_OUT` gamut matrices (sRGB ↔ AgX working space)
  - Log₂ inset over working range: −10 … +6.5 EV
  - Sigmoid contrast curve fitted to the AgX CDL
  - Full GPU path via `torch.einsum` (no CPU fallback needed for AgX)
  - CPU fallback via NumPy using the same three-stage pipeline
  - Fixes previous hue shifts in saturated reds/blues under the agx operator

### Added — HDR Generation & Enhancement

- **◎ Radiance HDR Enhancer** (`nodes_hdr_inception.py`): Pixel-space generative HDR
  expansion for scene-linear images. No sampler required for standalone use.
  - Luminance-weighted soft highlight roll-off via configurable knee function
  - Calibrated Gaussian noise injection into near-clip specular regions (sub-pixel detail)
  - Luminance-weighted blend back to original (`blend_strength`)
  - Outputs: `enhanced_image` (IMAGE) + `headroom_mask` (MASK) + `stats` (JSON)
  - `headroom_pct`, `delta_stops` reported in stats JSON
  - Optional `lora_name` hint for HDR LoRA workflows (informational, load via Load LoRA node)

- **◎ Radiance HDR Latent Init** (`nodes_hdr_inception.py`): Properly seeded Gaussian noise
  latent (N(0, σ²)) for HDR-aware generation — replaces zero-latent which caused degenerate
  sampler output. Configurable `sigma` (1.0 = SDXL/SD3, 14.6 = Flux, 3.5 = LCM).

- **◎ Radiance HDR Latent Blend** (`nodes_hdr_inception.py`): Per-channel blend between a
  reference HDR latent and a noise latent. Steers generation toward a tonal distribution
  without full img2img. Per-channel weight overrides via comma-separated string.

### Added — HDR Analysis

- **HDR Histogram: Headroom Zone Markers** (`hdr/analysis.py`): `hdr_zone_markers` input
  draws calibrated vertical lines at 1× (+0 EV), 2× (+1 EV), 4× (+2 EV), 8× (+3 EV)
  scene-linear on the linear histogram panel. Essential for HDR authoring visibility above
  the SDR clip point.
- **HDR Histogram: `headroom_pct` Output** (`hdr/analysis.py`): New FLOAT output pin
  reporting the percentage of luminance pixels above 1.0 (scene-linear SDR ceiling).
  Connect directly to downstream math nodes or display.

### Added — Tier 4: TurboDecoder Training Pipeline

- **RadianceTurboDecoder Architecture** (`fast_vae.py`, `hdr/fast_vae.py`): ~2M parameter
  convolutional decoder that maps VAE latents directly to log-coded images, replacing the
  engineered `_denoise_log_highlights()` + `_soft_log_shoulder()` combination with a
  learned equivalent. Supports Flux (16ch) and SDXL (4ch) latent formats.

- **HDRLogLoss** (`train_turbo_decoder.py`): Four-component HDR-aware training loss:
  - L1 in log space (perceptually uniform tonal weight)
  - MSE in log space (sharpness / outlier penalty)
  - Highlight penalty above configurable `knee` (extra weight on near-clip regions)
  - Structural gradient loss (L1 on image gradients, preserves edge fidelity)

- **EMA Weight Tracking** (`train_turbo_decoder.py`): Exponential moving average of decoder
  weights (`decay=0.999`). EMA weights are saved separately as `turbo_decoder_ema_stepN.pth`
  for smooth, stable inference.

- **`train_turbo_decoder.py`** — Full training CLI:
  - AdamW + Cosine Annealing LR scheduler
  - Configurable steps (50k useful / 200k production), batch size, grad clip
  - Resume from checkpoint, JSONL training log, eval PSNR-log every 2k steps
  - Checkpoint save every 5k steps (full + EMA-only inference weights)

- **`dataset_hdr.py`** — HDR pair dataset generator: produces `.npz` (latent, log-coded
  target) pairs from raw HDR EXRI sources for supervised TurboDecoder distillation.

- **`nodes_fast_vae.py`** — ComfyUI inference node for loading a trained TurboDecoder
  checkpoint and running fast log-domain VAE decoding in production workflows.

---

## [2.5.1] - 2026-04-12 ("The New Architecture Update")

### Added
- **Node Architecture**: Unified `◎ Radiance HDR VAE Decode` to officially use the Radiance 4K production engine. Replaces placeholder with production-grade tiling and log-domain math.
- **Node: ◎ Radiance NDI Sender** — Upgraded with **High-Fidelity Log-Encoding** (LogC4/S-Log3) ensuring full HDR dynamic range is preserved when streaming to external apps in 8-bit.
- **Optics Refinement**: Added professional Boundary Handling (`zeros`, `reflection`, `border`) and `Invert` math to the Lens Distortion and Chromatic Aberration suite.

## [2.5.0] - 2026-04-12 ("The Optics & Architecture Update")

### Added
- **Node: ◎ Radiance Lens Distortion** — True barrel/pincushion distortion with proper ST-Map output generation. Validated for 32-bit float coordinate mapping and edge-wrap prevention.
- **Node: ◎ Radiance Chromatic Aberration** — Real-world spectral dispersion shifting (R/G/B) simulating glass Index of Refraction (IoR), operating from the specified optical center.
- **Node: ◎ Radiance Anamorphic Streaks** — High-end cinematic flare generation employing cascaded anisotropic blurs to organically bloom clipped highlights within the HDR/32-bit linear signal.
- **Node: ◎ Radiance SDR to HDR Expand** — Inverse OETF logic combined with mathematical power-curve spline expansion, successfully recovering squashed SDR highlights into full Scene-Linear data capable of pushing 4,000+ nits dynamically.
- **Node: ◎ Radiance Relight Engine** — True 32-bit fp geometric re-lighting node. Ingests any RGB normal map and applies physically-based Lambertian diffuse and Blinn-Phong specular passes. Allows additive relighting of 2D generative sequences safely.
- **Node: ◎ Radiance HDR VAE Decode** — Intercepts Latent tensor data during decode and bypasses traditional StableDiffusion/Flux clipping mechanisms, mapping latent energies directly to fp32 values to retain dynamic range.
- **Node: ◎ Radiance NDI Sender** — Native PyTorch-to-NDI streaming bridge. Streams PyTorch tensor surfaces as BGRA video frames directly over local network to OBS, Resolume, or Nuke via NewTek NDI SDK.

## [2.4.2] - 2026-04-10 (The Temporal & Intelligence Update)

### Added
- **Viewer: Precision Scopes Suite** — GPU-accelerated **CIE 1931 xy Chromaticity** scope with spectral locus and gamut target overlays (Rec.709, P3, Rec.2020).
- **Viewer: HDR Waveform Upgrade** — Full linear-to-ST.2084 (PQ) non-linear mapping for waveforms. Provides visibility for details up to 10,000 nits without clamping.
- **Viewer: HDR HUD Graticules** — Calibrated horizontal lines and labels for 100, 400, 1000, and 4000 nits in the waveform panel.
- **Node: ◎ Radiance Chromaticity** — Backend analytical node for CIE 1931 visualization. Supports batch processing and gamut target comparison.
- **Node: ◎ Radiance Curves & Hue Curves** — Full piecewise-linear RGB/Master curves and HSL-based secondary color correction engine for 32-bit float pipelines.
- **Viewer: Filmstrip Timeline** — A scrollable row of 40×28px frame thumbnails above the transport scrubber. Click any thumbnail to jump directly to that frame. Active frame is highlighted in accent blue. Max 60 thumbnails with step sub-sampling for long sequences.
- **Viewer: Pin Frame (A/B Wipe)** — New 📌 button in the transport panel. Freezes the current rendered frame as the B-side reference for instant A/B wipe comparison. Click again to release. Integrated with existing wipe renderer path.
- **Viewer: OCIO Display Transform Panel** — New panel in VIEW tab; 37 ACES transforms available when OCIO config is loaded.
- **Viewer: Filmstrip Flicker Heatmap** — Per-frame colored dot (green→magenta) on each filmstrip thumbnail showing inter-frame luma delta; ✂ marker and red border on detected scene cuts.
- **◎ Radiance Scene Cut Detector** — Multi-signal hard/soft cut detection: luma histogram distance, frame delta, LAB chroma shift, and flash guard. Outputs IMAGE + MASK + JSON report.
- **◎ Radiance Temporal Color Lock** — LAB-space batch colour stabiliser. Anchors every frame’s L*a*b* statistics to a reference frame with per-scene auto-reset via cut_mask.
- **◎ Radiance Deflicker Pro** — Professional luminance deflickering: Gain Only (fast), Percentile (selective), and Luma Match (full CDF histogram remapping).
- **◎ Radiance Frame Interpolation** — Temporal upsampler with Linear, Optical Flow (Farneback via OpenCV), and RIFE (external binary) modes. Factor 2–8×.
- **Sampler: CogVideoX + StepVideo** — Full `MODEL_DEFAULTS` entries, `VIDEO_MODEL_TYPES` membership, `CFG_GUIDED_MODELS`, and `detect_model_type()` class + config detection.
- **Sampler: Noise Library Expansion** — Three new pure-PyTorch noise generators added to `NOISE_TYPES`: Simplex (octaved hash-grid), Voronoi (cellular distance field), Curl (divergence-free from spectral potential).
- **◎ Radiance LoRA Scheduler** — Generate per-step LoRA strength schedules using Catmull-Rom / Linear / Step interpolation between user-defined control points. Outputs `FLOAT_LIST` + JSON.
- **◎ Radiance Regional Prompt** — Add a spatial region with its own conditioning via bbox or MASK. Additive / Replace merge modes. Chainable.
- **◎ Radiance Regional Grid** — Divide image into a columns×rows grid; assign a separate text prompt to each cell; encodes inline via CLIP input.
- **◎ Radiance EXR Multi-Part** — Write a Nuke/Resolve-compatible multi-part EXR v2 file with named AOV parts: beauty (RGBA), depth (Z), normal (NX/NY/NZ), albedo, and 2 custom layers. Fallback to per-layer EXRs if OpenEXR multi-part is unavailable.
- **◎ Radiance Render Queue** — Submit parametric sweep runs to the ComfyUI queue. Define `{node_id: {field: [values]}}` and choose Product / Zip / Chain combination modes. Dry-run preview. Returns JSON job manifest.
- **Radiance Write: Remote Output** — `RadianceDigitalCinemaWrite` gains optional `remote_path` input supporting UNC (`\\server\share`) and S3 (`s3://bucket/key`) output via `boto3`.
- **.rad v3 Archive Format** — New ZIP-based `.rad` v3 container (`RADZ` magic). Stores `workflow.json`, `manifest.json` (asset list + metadata), and optional `assets/` directory. Full backward-compat reader handles v1 (plain text), v2 (binary+SHA256), and v3 (ZIP) automatically via `_unpack_any_rad()`.

### Fixed
- **Viewer: Printer Lights Listener Leak (B-14)** — `makePrinterStrip()` was attaching permanent `document.addEventListener('mousemove'/'mouseup')` calls every time the grade panel was rebuilt. Fixed by switching to `AbortController`-scoped listeners.
- **Viewer: Printer Lights UX** — Added EV stop readout per channel (e.g. `+0.40EV`), combined RGB status badge, and a RST button to reset all channels simultaneously.

## [2.3.3] - 2026-04-01


### Fixed
- **A/B Split-Wipe Fix**: Repaired the broken A/B comparison functionality in the Radiance Viewer. The wipe mode now correctly uses the GPU shader and features a draggable split-line with A/B labels.
- **Node Cleanup**: Removed the redundant `RadianceSaveEXR` node in favor of the new `◎ Radiance Write` node.
- **Package Hardening**: Resolved JSON serialization errors in QC reports and standardized environment variables.

## [2.3.2] - 2026-03-30

### Added
- **Smart Overwrite Protection**: `Radiance Write` and `EXR Save` now feature automated index detection to prevent accidental file destruction.
- **Universal Digital Cinema I/O**: Consolidated Video, Image Sequence, and Single Image handling into a high-performance unified pipeline.
- **Terminal HUD & Live REPL**: Nuke-style Python interaction directly inside the viewer for real-time data inspection.
- **Interactive Mask Editor**: Non-destructive brush masking in `◎ Radiance Load Image`.

### Fixed
- **Robust Pipeline Validation**: Enhanced null-safety and input checking for high-load production environments.
- **◎ Radiance Grade Match**: Optimized shot-to-shot color statistics transfer using CIE L*a*b* mean/std math.

## [2.3] - 2026-03-29


## [2.2.2] - 2026-03-29

### Fixed
- **Comfy Registry Metadata**: Removed restrictive `Environment :: GPU :: NVIDIA CUDA` classifier from `pyproject.toml` to allow installation on CPU and macOS (MPS) systems.
- **Registry Icon**: Renamed `RADIANCE_ICON.png` to `flogo.png` and updated `pyproject.toml` to ensure the logo appears correctly in the Comfy Registry.

## [2.2.1] - 2026-03-28

### Fixed
- **Save Overwrite Protection** (`hdr/io.py`, `nodes_io.py`): `Radiance Save EXR/HDR` and `Radiance Write` now default to `start_frame=0`, enabling automatic index detection to prevent accidental file overwrites.
- **Null Safety**: Added robust input validation to sequence loading and saving methods to prevent crashes when passed empty paths or invalid indices from third-party nodes.

### Changed
- **Branding Consistency**: `RadianceApplyGradeInfo` display name updated to `◎ Radiance Apply Grade Info` to match the suite's naming convention.
- **Repository Cleanup**: Removed legacy test scripts and temporary debug directories (`MagicMock`) to streamline the package for GitHub and Comfy Registry.

## [2.2] - 2026-03-18

### Changed
- Major version update to 2.2.
- Refined terminal and viewer UI.
- Improved console noise filtering and error stabilization.

## [2.1.1] - 2026-03-17

### Changed
- Maintenance update to version 2.1.1.

## [2.1.0] - 2026-03-10

### Added — Radiance Viewer v2.2 (Terminal & UX Overhaul)
- **TERMINAL HUD Tab** (`radiance_viewer.js`): Live Python REPL embedded directly in the viewer.
  - Persistent namespace (`_TERMINAL_NS`) — variables survive between executions.
  - 30-second timeout guard via `threading.Thread` to prevent ComfyUI event-loop freeze.
  - Pre-injected context: `math`, `os`, `torch`, `np`, `json`, `time`, `folder_paths`.
  - `Reset Namespace` button wipes state on demand.
  - Snippet dropdown with common helper presets.
- **Documentation Links** in Radiance Workspace node (two new buttons: `📖 Docs — radiance.fxtd.org` and `🌐 FXTD Studios — www.fxtd.org`) and in the Terminal tab status bar.

### Changed — Radiance Viewer UX (5 Design Improvements)
- **Toolbar Group Labels**: 9 labeled clusters (`FILE · GRADE · VIEW · CH · NAV · ANALYSIS · COMPARE · SCOPES · ANNOTATE · MEASURE`) with hairline separator rules.
- **Panel Label Typography**: All HUD sub-labels bumped to `11px` + `letter-spacing: 0.06em` for improved legibility.
- **Bottom Dock Colors**: `TERMINAL` tab now renders in `#00a8ff` (brand blue); `SCRIPT EDITOR` in `#7a92b0` (muted slate). `RUN AUTOMATION` button color unified to match.
- **Film Stock Preset Active State**: Clicking a film stock pill now shows a blue-glow border indicator (`rgba(0,168,255,0.45)`) — selected state is always visible.
- **Right Panel Height**: `tabContentContainer` uses `flex:1 + overflow-y:auto` — grading controls fill the full panel height instead of leaving a generic black void below.

### Changed — Nodes
- **◎ Radiance Depth Map** (renamed from `◎ Depth Map Generator`): Display name updated in `nodes_depth.py` — existing workflows unaffected as the internal class name `RadianceDepthMapGenerator` is unchanged.
- **Output Path Consistency**: Widget parameter name unified to `output_path` across:
  - `RadianceWrite` node (`nodes_io.py`) — was `subfolder`.
  - `RadianceSaveEXR` node (`hdr/io.py`) — was `custom_path`.

### Fixed
- Added missing `import json` to `nodes_radiance_viewer.py`.
- Restored accidentally dropped `exportToCDL()` function declaration after Terminal injection.

### Added — Radiance v2.1.0 (The Professional Suite)
- **◎ Radiance 32-bit Denoise** (`nodes_denoise.py`): Edge-preserving bilateral filter for 32-bit float images.
- **◎ Radiance Reroute / Reroute+** (`nodes_layout.py`): Compact visual reroute nodes with auto-type detection and custom labels.
- **◎ Radiance Load Image** (`nodes_radiance_mask.py`): Enhanced image loader with integrated soft-brush mask editor and non-destructive companion mask storage.
- **◎ Show Text (Radiance)** (`nodes_text.py`): Utility node for displaying any data type (string, JSON, etc.) directly on the node UI.
- **Prompt Enhancer Integration**: `Cinematic Prompt Machine` now supports grammar-aware prompt enhancement (Natural, Descriptive, Cinematic styles).

### Improved
- **Metadata Management**: Improved EXR/PNG metadata handling across I/O nodes.
- **Dependency Validation**: enhanced `check_dependencies` in `__init__.py` for clearer installation guidance.

### Added — Radiance Viewer v2.1 (Viewer Overhaul)
- **fp32 Pick Buffer Sidecar** (`#5`): Each frame now saves a zlib-compressed fp32 `.rpick` sidecar (max 256px) for accurate scene-linear HDR color picking — true EV readout at cursor
- **OES_texture_half_float** (`#3`): Explicitly enables the WebGL `OES_texture_half_float` extension; falls back gracefully if unavailable. Half-float upload path now active by default
- **GPU Histogram** (`#6`): Press **H** in the viewer for a GPU-rendered 256-bin per-channel histogram with log scale for HDR images and a dotted white line at `x=1.0` to mark the SDR ceiling
- **LRU Frame Cache** (`#8`): Up to 8 GPU textures cached by frame ID. Scrubbing through frames no longer re-uploads data that is already in VRAM
- **Linear False Color / Zebra** (`#9`): `linearFalseColor` flag evaluates false-color and zebra thresholds in scene-linear space (pre-OETF) for accurate stop-level analysis
- **Display-P3 / HDR Monitor Detection** (`#10`): `RadianceWebGLRenderer.initDisplayP3()` detects P3 and Rec.2020 displays via CSS media queries and configures canvas `colorSpace` accordingly
- **CDL Export + Import** (`#7`): 💾 menu gains **Export CDL (Grade)** and **Import CDL (Grade)** — bidirectional ASC CDL v1.2 XML roundtrip recognized by Nuke, DaVinci Resolve, and OCIO pipelines

### Added — New Nodes (9 nodes)

#### Color
- **◎ Radiance Grade Match** (`nodes_grade.py`): Dedicated shot-to-shot LAB mean/std match grading. Connect `source` + `reference` → matched image + grade_info JSON
- **◎ Apply Grade Info** (`nodes_grade.py`): Replay a `grade_info` JSON string (from Radiance Grade) onto any new image with a `strength` blend slider. Closes the grade roundtrip
- **◎ Radiance LUT Bake** (`nodes_lut.py`): Generate a 33³ `.cube` LUT file from any Radiance Grade parameter set. Compatible with DaVinci Resolve, Nuke, Premiere Pro, and OCIO
- **◎ Radiance LUT Apply** (`nodes_lut.py`): Load and apply any external `.cube` LUT file via trilinear interpolation with `strength` blend control

#### Video / Temporal
- **◎ Radiance Temporal Smooth** (`nodes_temporal.py`): Per-pixel exponential moving average (EMA) across batch frames to remove inter-frame flicker and AI grain. Motion-aware masking preserves sharp moving areas
- **◎ Radiance Flicker Analyze** (`nodes_temporal.py`): Measures frame-to-frame luma delta and outputs a JSON flicker report (flicker index, max delta, per-frame means). Use before/after Temporal Smooth to benchmark improvement

#### Scopes / Analysis
- **◎ Radiance False Color** (`nodes_scopes.py`): Bake a 7-zone calibrated false-color exposure visualization as an IMAGE for headless/batch pipeline use. Zones: Crushed / Under / Dim / Correct / Over / Hot / Clipped. Configurable `is_linear`, `blend`, and `exposure_offset`

#### Compositing
- **◎ Radiance Blend Composite** (`nodes_overlay.py`): Two-layer compositor with 8 blend modes — Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, Divide. Optional `MASK` pin for per-pixel coverage. HDR-safe (no mid-pipeline clamping)

### Improved — Existing Nodes
- **◎ Radiance Grade** (`nodes_grade.py`): Added `reference_image` + `match_strength` optional inputs for auto LAB match grading. Added `preset_file` for loading external JSON preset libraries. `grade_info` output is now a full JSON dump of all 12 grade parameters (was plain text)
- **Sampler Pro v3.6**: Fixed BUG-28 (sigma schedule off-by-one causing negative step indices), BUG-33 (AYS Flux anchors), BUG-20 (deep-copy in guidance), and several noise-injection double-application bugs

### Fixed
- **Viewer Pick Buffer**: HDR color picker now reads true scene-linear fp32 values from `.rpick` sidecar instead of tonemapped 8-bit pixel values
- **Viewer Histogram**: Histogram scope now GPU-rendered in scene-linear space instead of display-encoded space
- **Pixel Loupe**: Enhanced HDR support and magnification controls
- **QC Analysis**: Tonemapping applied before defect detection for more accurate results

---

## [1.2.1] - 2026-02-10

### Changed
- **Code Organization** - Moved `defects.py` to `radiance/image/defects.py` for better modularity.
- **Cleanup** - Removed temporary dev files (`fix_indent.py`, `fix_indent_2.py`).
- **Dependencies** - Updated `nodes_qc.py` to import defects from the new location.

## [1.2.0] - 2026-02-09

### Added - Radiance Studio
- **RadianceQC Node (`RadianceQC`)**
  - Automated technical quality control for VFX workflows.
  - Analyzes Levels (Crushed Blacks/Clipped Whites), Gamut violations, and Noise floor.
  - Generates visual overlay (Red=Clipped, Blue=Crushed) and detailed text report.
- **Project Management System**
  - **Project Settings Node (`RadianceProjectSettings`)**: Define root, sequence, shot, and version.
  - **Smart EXR Saver**: `RadianceSaveEXR` now accepts `RADIANCE_PROJECT` input to auto-generate paths (`Root/Seq/Shot/vXX`).
- **Nuke-Style Backdrops**
  - Right-click context menu "Radiance Studio > Create Backdrop".
  - Wraps selected nodes in a professional, color-coded group using industry standard colors.
  - "Auto-Align Nodes" command for quick organization.

## [1.1.1] - 2026-02-07

### Critical Fixes
- **HDR Clamping Fixes** - Removed unintended 8-bit clamping in CPU-based effects (`apply_bloom`, `apply_chromatic_aberration`, `RadianceProFilmEffects` sharpening).
- **EXR Saver Stability** - Fixed generic `SystemError` crash in OpenEXR writer; fallback mechanism now robustly handles writer failures.
- **EXR Absolute Paths** - `RadianceSaveEXR` now supports absolute paths in `subfolder` input (e.g. for external drive export).

## [1.1.0] - 2026-02-03

### Added
- **Temporal Grain Engine v2** - Frame-coherent grain for video with smooth transitions
  - `generate_temporal_grain()` function with prime decorrelation seed system
  - Per-channel R/G/B intensity controls in `FXTDTemporalGrain` node
  - Temporal smoothness blending between frames
  
- **LUT Engine Upgrade**
  - Tetrahedral interpolation for more accurate 3D LUT lookups
  - `interpolation` parameter (Trilinear/Tetrahedral) in `RadianceLUTApply`
  
- **Comprehensive Test Suite**
  - `test_comprehensive.py` with 18 tests covering 32-bit precision and sampler validation
  - 100% pass rate on all tests

### Fixed
- Phase-Shift sampler override now correctly uses selected sampler
- Dynamic Guidance math corrected for Low → High → Low profile
- **Canon Log3** - Fixed coefficients to match Canon specification (18% gray → 0.343)
- **Panasonic V-Log** - Fixed decode coefficient typo for proper roundtrip precision

### Security & Production Hardening
- **Thread-safe caching** - Added `threading.RLock()` to LUT, depth model, and processor caches
- **Exception handling** - Replaced 14 bare `except:` blocks with specific exception types + logging
- **Logging infrastructure** - Added `logging.getLogger()` to all modules for proper debugging
- **LUT cache eviction** - Bounded cache size (32 max) with FIFO eviction to prevent memory leaks

---

## [1.0.0] - 2026-01-15

### Initial Release
- 55 professional nodes for HDR processing
- Film grain with 30+ camera sensors, 20+ film stocks
- Industry-standard scopes (Histogram, Waveform, Vectorscope)
- GPU-accelerated processing
- EXR/HDR file support
- ACES 2.0 Output Transform
- LUT support with caching
- Radiance Pro Viewer with Nuke/Flame-style shortcuts
