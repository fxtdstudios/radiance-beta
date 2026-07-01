# Radiance HDR Node Reference

Complete reference for all Radiance HDR nodes — encoding, decoding, smart
pre-processing, diagnostics, and LoRA training/inference.

---

## Table of Contents

1. [Encoding Nodes](#encoding-nodes)
   - [RadianceHDREncoder](#radiancehdrencoder)
   - [RadianceHDRPerChannelNorm](#radiancehdrperchannelnorm)
   - [RadianceHDRDecoder](#radiancehdrdecoder)
2. [Smart Pre-processing Nodes](#smart-pre-processing-nodes)
   - [RadianceHDRAutoLogSelect](#radiancehdrautologselect)
   - [RadianceHDRModelPresetLoader](#radiancehdrmodelpresetloader)
   - [RadianceHDRCoherencePrior](#radiancehdrcoherenceprior)
   - [RadianceHDRMetadataConditioner](#radiancehdrmetadataconditioner)
   - [RadianceHDRDiagnostics](#radiancehdrdiagnostics)
3. [LoRA Nodes](#lora-nodes)
   - [RadianceHDRLoRALoader](#radiancehdrloraloader)
   - [RadianceHDRLoRAApply](#radiancehdrloraapply)
4. [Model Preset Table](#model-preset-table)
5. [Pipeline Wiring Diagrams](#pipeline-wiring-diagrams)
   - [Minimal HDR encode → sample → decode](#minimal-hdr-encode--sample--decode)
   - [Full HDR pipeline with LoRA and diagnostics](#full-hdr-pipeline-with-lora-and-diagnostics)
   - [Multi-frame coherent video HDR](#multi-frame-coherent-video-hdr)
6. [Quick-start Recipes](#quick-start-recipes)
7. [Training Pipeline (CLI)](#training-pipeline-cli)

---

## Encoding Nodes

### RadianceHDREncoder

**Category:** `Radiance/HDR`

Converts scene-linear HDR imagery into VAE-ready display-referred output
using soft-knee Reinhard compression.  This is the mandatory first step
before VAE encoding — feeding raw HDR values directly into a VAE causes
clipping and highlight loss.

#### How it works

The encoder applies a per-pixel blend between hard clamp and Reinhard tone mapping:

```
y = (1 − r) · clamp(x, 0, 1) + r · x / (1 + x)
```

where `r = compression_ratio`, `x` is scene-linear and `y` is display-referred.
The result is always in `[0, 1]` so any downstream VAE sees valid SDR inputs.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Scene-linear input (values may exceed 1.0 for HDR) |
| `compression_ratio` | FLOAT | 0.5 | 0 = hard clamp, 1 = full Reinhard |
| `exposure_offset` | FLOAT | 0.0 | EV stops to apply before compression (positive = brighter) |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Display-referred image in [0, 1] — ready for VAE encode |
| `peak_linear` | FLOAT | 95th-percentile linear luminance (diagnostic) |

#### Tips

- Wire `compression_ratio` from `RadianceHDRLoRALoader` or `RadianceHDRModelPresetLoader`
  to guarantee the same value was used at training time.
- For LTX-Video, use `compression_ratio = 0.50` (the library's native default).
- For SDXL/SD 1.x, use lower values (0.35–0.40) — the 4-channel VAE has
  narrower dynamic range headroom.

---

### RadianceHDRPerChannelNorm

**Category:** `Radiance/HDR`

Mirrors LTX-Video's `vae_per_channel_normalize=True`.  Computes per-frame
per-channel mean and std, normalises the image, and stores the stats tensor
so `RadianceHDRDecoder` can invert the normalisation after VAE decode.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Output of `RadianceHDREncoder` |
| `norm_center` | FLOAT | 3.0 | ±N·σ clamping window |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Normalised image |
| `stats` | STATS | Per-channel mean/std dict — pass to decoder |

#### Tips

- `norm_center` should match the model preset (`norm_center` field in the
  preset table).  Flux/SD3 use 3.0; SDXL/SD15 use 2.5; Wan/HunyuanVideo use 3.5.
- Connect `stats` → `RadianceHDRDecoder.stats` to undo the normalisation
  after generation.

---

### RadianceHDRDecoder

**Category:** `Radiance/HDR`

Recovers scene-linear HDR from VAE-decoded output.  Inverts the per-channel
normalisation (if stats are supplied) and then inverts the soft-knee compression.

The inverse formula uses two regimes separated at the break point
`y_break = 1 − r/2`:

- **HDR regime** `y > y_break`: `x = (y − 1 + r) / (1 − y)`
- **SDR regime** `y ≤ y_break`: quadratic root of `(1−r)x² + (1−y)x − y = 0`

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | VAE-decoded output |
| `compression_ratio` | FLOAT | 0.5 | Must match the value used at encode time |
| `stats` | STATS | (optional) | Per-channel stats from `RadianceHDRPerChannelNorm` |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Recovered scene-linear HDR image |

---

## Smart Pre-processing Nodes

### RadianceHDRAutoLogSelect

**Category:** `◎ RADIANCE/HDR`

Histogram-based automatic log format selector.  Analyses the 95th-percentile
luminance and picks the log encoding whose dynamic-range knee best fits the
scene.  Optionally, overrides `compression_ratio` from the model preset table
when a `model_hint` is supplied.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Scene-linear input to analyse |
| `override` | ENUM | `auto` | Force a specific format or leave on `auto` |
| `model_hint` | STRING | `""` | Optional model name — e.g. `"flux"`, `"wan2.1"`, `"sdxl"` |

**Override choices:** `auto`, `LogC4`, `SLog3`, `VLog`, `LogC3`, `ACEScct`

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `log_format` | STRING | Selected format name |
| `compression_ratio` | FLOAT | Recommended ratio (model preset if hint given) |
| `stops_detected` | FLOAT | Estimated stops above 18% grey |
| `model_preset_used` | STRING | Canonical model key resolved from hint (empty if none) |

#### Automatic selection logic

| Stops above 18% grey | Selected format |
|---|---|
| ≤ 10 stops | `ACEScct` |
| 10–12 stops | `VLog` |
| 12–13 stops | `LogC3` |
| 13–14 stops | `LogC4` |
| > 14 stops | `SLog3` |

---

### RadianceHDRModelPresetLoader

**Category:** `◎ RADIANCE/HDR`

Exposes the full `RADIANCE_MODEL_PRESETS` table as a ComfyUI node.  Select a
model from the dropdown to instantly load all tuned HDR parameters.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_name` | ENUM | `ltx-video` | Model family to load |
| `model_hint_override` | STRING | `""` | Optional partial string to override dropdown selection |

**Model choices:** `ltx-video`, `flux`, `cogvideox`, `wan`, `hunyuanvideo`,
`sd3`, `sdxl`, `sd15`

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `compression_ratio` | FLOAT | Recommended HDR compression ratio |
| `norm_center` | FLOAT | ±N·σ window for `RadianceHDRPerChannelNorm` |
| `vae_spatial_factor` | INT | VAE spatial downscale factor |
| `vae_temporal_factor` | INT | VAE temporal downscale factor (1 = image model) |
| `latent_channels` | INT | Number of VAE latent channels |
| `notes` | STRING | Brief rationale for the preset values |

---

### RadianceHDRCoherencePrior

**Category:** `◎ RADIANCE/HDR`

Builds an EV-space temporal consistency map from a sequence of frames and
injects it into the latent noise tensor before sampling.  Eliminates temporal
flicker at the source rather than correcting it in post.

#### How it works

1. Converts each frame to log2-luminance (EV space).
2. Computes per-pixel temporal mean and std across the frame sequence.
3. Where std is low (temporally steady), coherence ≈ 1.  Where std is high
   (flickering), coherence → 0.
4. Adds coherence-scaled Gaussian noise to the latent: steady regions get
   no noise, flickery regions get noise that encourages consistency.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `latent` | LATENT | — | Noise latent from sampler or `EmptyLatentImage` |
| `image` | IMAGE | — | Scene-linear frames (B, H, W, C) |
| `noise_scale` | FLOAT | 0.1 | Coherence noise injection strength |
| `sigma` | FLOAT | 1.0 | Gaussian smoothing radius for temporal std map |
| `stats_passthrough` | STRING | `""` | JSON stats from upstream nodes (passed through) |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `latent` | LATENT | Noise latent modified with coherence hints |
| `coherence_map` | IMAGE | Visualisable coherence map (1 = steady, 0 = flickery) |
| `stats_json` | STRING | Coherence statistics as JSON |

---

### RadianceHDRMetadataConditioner

**Category:** `◎ RADIANCE/HDR`

Converts HDR scene metadata (compression ratio, peak luminance, log format)
into token embeddings that are concatenated to the CLIP conditioning.  This
gives the diffusion model explicit knowledge of the HDR capture parameters,
improving highlight reproduction consistency.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `conditioning` | CONDITIONING | — | Base CLIP conditioning to augment |
| `compression_ratio` | FLOAT | 0.5 | From `RadianceHDREncoder` or preset |
| `log_format` | STRING | `""` | From `RadianceHDRAutoLogSelect` |
| `peak_linear` | FLOAT | 1.0 | From `RadianceHDREncoder` output |
| `stops_detected` | FLOAT | 0.0 | From `RadianceHDRAutoLogSelect` |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `conditioning` | CONDITIONING | Augmented conditioning with HDR metadata tokens |

---

### RadianceHDRDiagnostics

**Category:** `◎ RADIANCE/HDR`

**OUTPUT_NODE = True**

In-memory codec health check and structured JSON report.  Runs the full
compress→decompress cycle on the input image without touching any external
services.  Writes a `HDR_DIAG` line to the `radiance.diagnostics` logger.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Scene-linear image to diagnose |
| `compression_ratio` | FLOAT | 0.5 | Ratio used in production pipeline |
| `model_preset_used` | STRING | `""` | From `RadianceHDRAutoLogSelect` (for display) |
| `stats_json` | STRING | `""` | JSON from `RadianceHDRCoherencePrior` (optional) |
| `coherence_map` | IMAGE | (opt) | Coherence map image for display |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `report_json` | STRING | Full structured JSON diagnostic report |
| `psnr_estimate` | FLOAT | In-memory codec PSNR (dB) |
| `peak_stops` | FLOAT | Peak luminance in stops above 18% grey |

#### Sample report

```json
{
  "psnr_db": 52.3,
  "peak_linear": 8.41,
  "peak_stops": 5.54,
  "compression_ratio": 0.5,
  "model_preset_used": "flux",
  "coherence_stats": {},
  "codec_healthy": true
}
```

---

## LoRA Nodes

### RadianceHDRLoRALoader

**Category:** `Radiance/HDR/LoRA`

Loads a `.safetensors` LoRA file produced by `train_hdr_lora.py`.  Extracts
Radiance training metadata embedded at export time so the `compression_ratio`
can flow automatically to `RadianceHDREncoder` — no manual entry required.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `lora_path` | STRING | `""` | Absolute path to the `.safetensors` file |
| `fallback_compression_ratio` | FLOAT | 0.5 | Used when metadata does not contain ratio |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `lora_dict` | LORA_DICT | Opaque dict of tensors + metadata — pass to `RadianceHDRLoRAApply` |
| `compression_ratio` | FLOAT | From LoRA metadata (wire to `RadianceHDREncoder`) |
| `model_name` | STRING | Model family the LoRA was trained on |
| `metadata_json` | STRING | Full metadata as a JSON string for display |

#### Metadata embedded in the LoRA file

| Key | Type | Description |
|-----|------|-------------|
| `radiance_model_name` | str | e.g. `"ltx-video"` |
| `radiance_compression_ratio` | float | e.g. `"0.5"` |
| `radiance_rank` | int | LoRA rank used during training |
| `radiance_alpha` | float | LoRA alpha used during training |
| `radiance_version` | str | Radiance version that exported the file |

---

### RadianceHDRLoRAApply

**Category:** `Radiance/HDR/LoRA`

Applies a Radiance HDR LoRA to a diffusion MODEL by adding low-rank weight
deltas `(strength × alpha/rank × B@A)` to each matching Linear projection.

Safe to chain: multiple LoRA applies accumulate deltas.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | MODEL | — | From `CheckpointLoader` or a prior `RadianceHDRLoRAApply` |
| `lora_dict` | LORA_DICT | — | From `RadianceHDRLoRALoader` |
| `strength` | FLOAT | 1.0 | Scale multiplier: 0 = skip, 1 = full, 1.5 = amplified |
| `model_hint` | STRING | `""` | Cross-check model family; warns on mismatch |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `MODEL` | MODEL | Patched model — drop-in replacement in the graph |
| `compression_ratio` | FLOAT | From LoRA metadata — wire to `RadianceHDREncoder` |

#### Mismatch warning

If `model_hint` is supplied and resolves to a different family than the LoRA
was trained on, a `WARNING` is logged to `radiance.hdr_lora` but processing
continues.  The node never blocks — you can intentionally cross-apply LoRAs
for creative effect.

---

## Model Preset Table

| Model | compression_ratio | norm_center | vae_spatial | vae_temporal | latent_ch |
|-------|:-----------------:|:-----------:|:-----------:|:------------:|:---------:|
| ltx-video | 0.50 | 3.0 | 8 | 8 | 128 |
| flux | 0.50 | 3.0 | 8 | 1 | 16 |
| cogvideox | 0.45 | 3.0 | 8 | 4 | 16 |
| wan | 0.60 | 3.5 | 8 | 4 | 16 |
| hunyuanvideo | 0.60 | 3.5 | 8 | 4 | 16 |
| sd3 | 0.50 | 3.0 | 8 | 1 | 16 |
| sdxl | 0.40 | 2.5 | 8 | 1 | 4 |
| sd15 | 0.35 | 2.5 | 8 | 1 | 4 |

**Aliases accepted by all nodes with `model_hint`:**

`ltx`, `flux1`, `flux.1`, `cogvideo`, `cogvideox5b`, `wanvideo`, `wan2`,
`wan2.1`, `hunyuan`, `sd3.5`, `stable-diffusion-3`, `sd1`, `sd2`

---

## Pipeline Wiring Diagrams

### Minimal HDR encode → sample → decode

```
[LoadImage (EXR/HDR)]
        │ image
        ▼
[RadianceHDREncoder]  ◀── compression_ratio (0.5)
        │ image
        ▼
[VAEEncode]
        │ latent
        ▼
[KSampler]  ◀── MODEL, positive, negative, ...
        │ latent
        ▼
[VAEDecode]
        │ image
        ▼
[RadianceHDRDecoder]  ◀── compression_ratio (0.5)
        │ image
        ▼
[SaveImage / PreviewImage]
```

### Full HDR pipeline with LoRA and diagnostics

```
[CheckpointLoader]
        │ MODEL
        ▼
[RadianceHDRLoRAApply] ◀── lora_dict ── [RadianceHDRLoRALoader] ◀── lora_path
        │ MODEL          ◀── strength
        │ compression_ratio ─────────────────────────────────────────────┐
        ▼                                                                  │
[KSampler] ◀── positive/negative (optional: HDRMetadataConditioner)      │
        │ latent                                                            │
        ▼                                                                  │
[VAEDecode]                                                                │
        │ image                                                             │
        ▼                                                                  │
[RadianceHDRDecoder] ◀────────────────────────────────────────────────────┘
        │ image
        ▼
[SaveImage]

── Encode branch (runs before KSampler) ──────────────────────────────────

[LoadImage]
        │ image
        ▼
[RadianceHDRAutoLogSelect]  ◀── model_hint ("flux")
        │ log_format, compression_ratio, stops_detected
        ▼
[RadianceHDREncoder]  ◀── compression_ratio
        │ image, peak_linear
        ▼
[RadianceHDRPerChannelNorm]  ◀── norm_center (3.0)
        │ image, stats
        ▼
[VAEEncode] → latent → [KSampler]

[RadianceHDRDiagnostics] ◀── image (from Encoder), compression_ratio, model_preset_used
        → report_json, psnr_estimate (connect to ShowText node for monitoring)
```

### Multi-frame coherent video HDR

```
[LoadImageBatch (EXR sequence)]
        │ image (B, H, W, 3)
        ▼
[RadianceHDREncoder]  ◀── compression_ratio
        │ image
        ▼
[RadianceHDRCoherencePrior]  ◀── latent (from EmptyLatentVideo)
        │ latent (noise-injected), coherence_map, stats_json
        │
        ├── coherence_map ──▶ [PreviewImage] (monitor temporal consistency)
        │
        ▼
[VideoKSampler]  ◀── MODEL (with or without LoRA)
        │ latent (denoised video)
        ▼
[VAEDecodeTiled]
        │ image
        ▼
[RadianceHDRDecoder]  ◀── compression_ratio
        │ image (recovered HDR video frames)
        ▼
[SaveImageBatch / VideoOutput]
```

---

## Quick-start Recipes

### Recipe 1 — LTX-Video HDR generation with preset

1. Add `RadianceHDRModelPresetLoader`, set model to `ltx-video`.
2. Wire `compression_ratio` → `RadianceHDREncoder`.
3. Wire `norm_center` → `RadianceHDRPerChannelNorm`.
4. Run standard LTX-Video sampler.
5. Wire same `compression_ratio` → `RadianceHDRDecoder`.

### Recipe 2 — Auto-detect format from footage

1. Load source EXR into `RadianceHDRAutoLogSelect`.
2. No `model_hint` needed — it returns the best-fit log format.
3. Wire `compression_ratio` to encoder and decoder.
4. Optionally wire `log_format` → `RadianceHDRMetadataConditioner` for richer conditioning.

### Recipe 3 — Apply trained HDR LoRA

1. `RadianceHDRLoRALoader` → set `lora_path` to your `.safetensors`.
2. Wire `lora_dict` → `RadianceHDRLoRAApply.lora_dict`.
3. Wire `CheckpointLoader` → `RadianceHDRLoRAApply.model`.
4. Wire `RadianceHDRLoRAApply.MODEL` → sampler.
5. Wire `RadianceHDRLoRAApply.compression_ratio` → `RadianceHDREncoder`
   and `RadianceHDRDecoder` — no manual entry needed.

### Recipe 4 — Video coherence + LoRA (production)

1. Load EXR batch → `RadianceHDREncoder` (compression_ratio from LoRALoader).
2. `RadianceHDRPerChannelNorm` → VAE encode.
3. `RadianceHDRCoherencePrior` injects coherence into the noise latent.
4. Video KSampler uses LoRA-patched model.
5. VAE decode → `RadianceHDRDecoder` (same compression_ratio).
6. `RadianceHDRDiagnostics` monitors PSNR in parallel (OUTPUT_NODE branch).

---

## Training Pipeline (CLI)

The Radiance LoRA training pipeline consists of three scripts:

### Step 1 — Cache the dataset

```bash
python -m radiance.dataset_hdr_lora \
    --exr_dirs /data/hdr_footage /data/more_exr \
    --cache_dir /data/latent_cache \
    --vae_path /models/ltxv_vae.safetensors \
    --model_name ltx-video \
    --size 512 \
    --n_frames 1 \
    --compression_ratio 0.5
```

`dataset_hdr_lora.py` reads `.exr` files, applies soft-knee compression,
encodes through the VAE once, and saves clean latents + null text embeddings
to `cache_dir`.  This only needs to run once per dataset.

### Step 2 — Train the LoRA

```bash
python -m radiance.train_hdr_lora \
    --cache_dir /data/latent_cache \
    --model_path /models/ltxv.safetensors \
    --output_dir /checkpoints/hdr_lora_ltxv \
    --model_name ltx-video \
    --rank 16 \
    --alpha 16.0 \
    --steps 5000 \
    --batch_size 2 \
    --lr 1e-4 \
    --save_every 500
```

Training injects LoRA into all attention projections (`to_q`, `to_k`, `to_v`,
`to_out`, feed-forward projections).  Checkpoints are saved as
`radiance_hdr_lora_step{N:06d}.safetensors` in kohya_ss-compatible format.

Key training features:
- **Flow-matching schedule** for LTX-Video, Flux, Wan, SD3 — with HDR
  bias toward high-noise timesteps (`t > 0.6`).
- **DDPM schedule** for SDXL / SD 1.x / 2.x.
- **Highlight-weighted loss** — extra MSE penalty on latent dimensions
  corresponding to bright regions (`compressed > 0.85`).
- **EMA checkpointing** — saves both current and EMA weights at each
  checkpoint interval.
- **Metadata embedding** — `radiance_compression_ratio`, `radiance_model_name`,
  `radiance_rank`, `radiance_alpha`, `radiance_version` stored in the
  `.safetensors` file header so `RadianceHDRLoRALoader` can read them.

### Step 3 — Inference in ComfyUI

Load the saved `.safetensors` with `RadianceHDRLoRALoader` and wire as
described in Recipe 3 above.  The `compression_ratio` flows automatically —
no configuration drift between training and inference.

---

### Diagnostics Logging

All Radiance HDR nodes write structured log lines to the
`radiance.diagnostics` logger.  To capture them in a file:

```python
import logging

handler = logging.FileHandler("radiance_hdr.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logging.getLogger("radiance.diagnostics").addHandler(handler)
logging.getLogger("radiance.diagnostics").setLevel(logging.INFO)
```

Log line formats:

| Node | Format |
|------|--------|
| `RadianceHDREncoder` | `HDR_ENCODE compression_ratio=… exposure_offset=… peak_linear=… peak_compressed=…` |
| `RadianceHDRDecoder` | `HDR_DECODE compression_ratio=… peak_decoded=…` |
| `RadianceHDRDiagnostics` | `HDR_DIAG psnr_db=… peak_linear=… peak_stops=… compression_ratio=… healthy=…` |
| `RadianceHDRLoRALoader` | `HDR_LORA_LOAD model=… compression_ratio=… rank=… tensors=…` |
| `RadianceHDRLoRAApply` | `HDR_LORA_APPLY_DONE lora_model=… hint=… strength=… compression_ratio=…` |

---

## ACES 2.0 Nodes (`nodes_aces2.py`)

Category: `◎ RADIANCE/Color/ACES`

### RadianceACES2Transform

Full analytical ACES 2.0 RRT + ODT in a single node. No external CTL required.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Scene-linear ACEScg input |
| `output_display` | COMBO | `sRGB D65` | Target display: sRGB D65, DCI-P3 D65, Rec.2020 PQ, Rec.2020 HLG |
| `exposure_offset` | FLOAT | 0.0 | Pre-RRT exposure trim in stops |
| `saturation` | FLOAT | 1.0 | Post-ODT saturation tweak |
| `peak_nits` | FLOAT | 1000.0 | PQ/HLG peak luminance (cd/m²) |

Outputs: `IMAGE`, `grade_info` (STRING JSON).

### RadianceACESInputTransform

25 IDT matrices covering ARRI ALEXA, RED, Sony Venice, Canon C-series, Nikon Z, Panasonic Varicam, Blackmagic, and GoPro camera systems.

| Input | Type | Description |
|-------|------|-------------|
| `image` | IMAGE | Camera-encoded input |
| `camera` | COMBO | Camera + log format (ARRI LogC3/LogC4, Sony SLog2/SLog3, RED Log3G10, Canon CLog3, Nikon NLog, Panasonic Vlog, BMD Film Gen5, GoPro Protune, …) |
| `invert` | BOOLEAN | Inverse (ACEScg → camera log) |

### RadianceACESLMT

Five LMT colour transforms applied after IDT and before RRT.

| Preset | Effect |
|--------|--------|
| `Linear` | No change — passthrough |
| `Blue Light Fix` | Reduces chromatic aberration in highlights |
| `Golden` | Warm golden tint |
| `Desaturate Highlights` | Roll off saturation above 70% luminance |
| `Kodak 2383` | Film print emulation (warm shadows, cyan highlights) |

### RadianceACESCDL

ASC CDL v1.2 in ACES scene-linear working space.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | ACEScg or scene-linear input |
| `slope_r/g/b` | FLOAT | 1.0 | Per-channel slope (gain) |
| `offset_r/g/b` | FLOAT | 0.0 | Per-channel lift offset |
| `power_r/g/b` | FLOAT | 1.0 | Per-channel gamma power |
| `saturation` | FLOAT | 1.0 | Luma-weighted global saturation (Rec.709) |
| `clamp` | BOOLEAN | True | Clamp output to [0, ∞) |

---

## Studio Integration Nodes (`nodes_studio_integrations.py`)

Category: `◎ RADIANCE/Studio`

### RadianceDaVinciSend

Export the current image to a DaVinci Resolve shared media folder. Supports 8-bit PNG, 16-bit TIFF, and EXR output formats.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Frame to export. Batch writes numbered files. |
| `resolve_folder` | STRING | `""` | Resolve shared media folder path |
| `filename` | STRING | `radiance_out` | Base filename (no extension) |
| `bit_depth` | COMBO | `16bit` | `8bit`, `16bit`, `EXR` |
| `frame_start` | INT | `1001` | Starting frame number |

Outputs: `status` (STRING), `render_path` (STRING).

### RadianceNukeSend

Export EXR + `.nk` Read-node snippet for direct Nuke import. Optionally push to a running Nuke instance via TCP.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Frame to export. Batch writes numbered EXR sequence. |
| `nuke_folder` | STRING | `""` | Output folder for image + .nk file |
| `filename` | STRING | `radiance_out` | Base name for EXR file(s) |
| `frame_start` | INT | `1001` | Starting frame number |
| `push_to_nuke` | BOOLEAN | `False` | Auto-create Read node in Nuke via TCP |
| `nuke_host` | STRING | `127.0.0.1` | Nuke listener host |
| `nuke_port` | INT | `1986` | Nuke listener port |
| `half_float` | BOOLEAN | `True` | 16-bit half EXR (True) or 32-bit float (False) |

Outputs: `status` (STRING), `render_path` (STRING).

### RadianceShotMetadata

Attach production metadata as a JSON sidecar and image comment.

| Input | Type | Description |
|-------|------|-------------|
| `image` | IMAGE | Source image |
| `scene` | STRING | Scene label |
| `shot` | STRING | Shot label |
| `take` | INT | Take number |
| `camera` | STRING | Camera body identifier |
| `lens` | STRING | Lens identifier |

Outputs: `IMAGE` (passthrough), `metadata` (STRING JSON).

### RadianceASCCDLExport

Serialize CDL values to ASC CDL v1.2 XML or JSON override file.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `slope_r/g/b` | FLOAT | 1.0 | Slope per channel |
| `offset_r/g/b` | FLOAT | 0.0 | Offset per channel |
| `power_r/g/b` | FLOAT | 1.0 | Power per channel |
| `saturation` | FLOAT | 1.0 | Global saturation |
| `output_path` | STRING | `/tmp/grade.cdl` | Output file path |
| `format` | COMBO | `xml` | `xml` or `json` |

---

## Real-Time Preview Nodes (`nodes_realtime_preview.py`)

Category: `◎ RADIANCE/Preview`

### RadianceFalseColorMonitor

Cinema false-colour overlay with 10 exposure zones. Zone colours and thresholds follow the standard cinema false-colour convention (black → dark → shadow → mid-shadow → mid → mid-high → high → near-clip → clip → super-white).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Scene-linear or log input |
| `enabled` | BOOLEAN | True | Toggle false-colour on/off |

### RadianceFocusPeaking

Sobel edge highlight overlay for critical focus verification.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Source image |
| `threshold` | FLOAT | 0.15 | Edge response threshold [0, 1] |
| `color` | COMBO | `red` | Peak colour: red, green, blue, white, yellow |
| `blend` | FLOAT | 0.7 | Overlay blend strength |

### RadianceSplitView

Side-by-side or top-bottom wipe comparison between two images.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image_a` | IMAGE | — | Left/top image |
| `image_b` | IMAGE | — | Right/bottom image |
| `split` | FLOAT | 0.5 | Wipe position [0, 1] |
| `direction` | COMBO | `horizontal` | `horizontal` or `vertical` |
| `guide_line` | BOOLEAN | True | Draw 1-pixel white guide at split |

### RadianceContactSheet

Auto-tiled thumbnail contact sheet from a frame batch.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `images` | IMAGE | — | Batch (B, H, W, 3) |
| `columns` | INT | 4 | Grid columns |
| `thumb_size` | INT | 256 | Thumbnail long-edge size (px) |
| `padding` | INT | 4 | Inter-cell padding (px) |

### RadianceFlipbookGIF

Assemble batch frames into an animated GIF preview.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `images` | IMAGE | — | Batch frames |
| `fps` | FLOAT | 12.0 | Playback rate |
| `scale` | FLOAT | 1.0 | Resize scale factor |
| `output_path` | STRING | `/tmp/flipbook.gif` | Output file |

Outputs: `output_path` (STRING), `frame_count` (INT).

### RadianceFrameStamp

Burn timecode, frame number, and label overlay into images.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Source image |
| `frame` | INT | 0 | Current frame number |
| `fps` | FLOAT | 24.0 | Frames per second (non-drop SMPTE timecode) |
| `label` | STRING | `` | Optional additional burn-in text |
| `position` | COMBO | `bottom-left` | Text anchor position |

### RadiancePreviewServer

Ephemeral HTTP daemon serving the last processed frame as JPEG on the local workstation.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Frame to serve |
| `port` | INT | 8765 | TCP port |
| `quality` | INT | 85 | JPEG quality [1, 100] |
| `enabled` | BOOLEAN | True | Enable/disable server |

Outputs: `IMAGE` (passthrough), `url` (STRING `http://127.0.0.1:{port}/frame`).

---

## AI Assist Nodes (`nodes_ai_assist.py`)

Category: `◎ RADIANCE/AI Assist`

### RadianceAutoGrade

Zone-based ASC CDL matching — automatically aligns a source image to a reference still.

**Algorithm**: 4-step zone matching: (1) shadow offset, (2) midtone slope, (3) highlight power, (4) global saturation. Strength parameter blends between ungraded and matched result.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | IMAGE | — | Image to grade |
| `reference` | IMAGE | — | Target look reference |
| `strength` | FLOAT | 1.0 | Blend strength [0, 1] |
| `shadow_thresh` | FLOAT | 0.15 | Luma threshold for shadow zone |
| `highlight_thresh` | FLOAT | 0.70 | Luma threshold for highlight zone |

Outputs: `IMAGE`, `cdl_params` (STRING JSON with `slope`, `offset`, `power`, `saturation` arrays).

### RadianceCLIPMatch

CLIP ViT-B/32 cosine similarity match between source and a pool of reference images.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | IMAGE | — | Query image |
| `references` | IMAGE | — | Pool of candidate images (batch) |
| `top_k` | INT | 1 | Number of best matches to return |

Falls back to Bhattacharyya 32-bin histogram similarity when `transformers` is unavailable.
Outputs: `best_match` (IMAGE), `similarity` (FLOAT), `report` (STRING JSON).

### RadianceContinuityCheck

Shot-to-shot continuity checker — flags photometric breaks between consecutive frames.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `frame_a` | IMAGE | — | Previous frame |
| `frame_b` | IMAGE | — | Current frame |
| `luma_thresh` | FLOAT | 0.05 | Max allowable luma drift |
| `color_thresh` | FLOAT | 0.03 | Max allowable colour cast |
| `sat_thresh` | FLOAT | 0.08 | Max allowable saturation drift |
| `contrast_thresh` | FLOAT | 0.10 | Max allowable contrast drift |
| `hist_thresh` | FLOAT | 0.82 | Min Bhattacharyya histogram similarity |

Outputs: `is_clean` (BOOLEAN), `report` (STRING JSON with per-metric values and issue list).

### RadianceGradePrompt

Natural language CDL grading — 35 intent rules + intensity modifiers parsed from free text.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Source image |
| `prompt` | STRING | `` | Grading instruction in plain English |
| `strength` | FLOAT | 1.0 | Global scale on all derived deltas |

**Intensity modifiers**: `very` → 1.6×, `slightly`/`a bit` → 0.4×, `a touch` → 0.3×, `less` → 0.5×.

**Built-in presets** (trigger with `preset: <name>`):

| Preset | Effect |
|--------|--------|
| `film` | Warm, slightly desaturated with lifted shadows |
| `bleach` | Desaturated, high-contrast bleach bypass |
| `day for night` | Dark, cool blue-shifted look |
| `instagram` | Bright, lifted, warm with slight fade |
| `neon` | Vivid, high saturation, cool blue shadows |
| `noir` | Crushed black-and-white |

Outputs: `IMAGE`, `cdl_params` (STRING JSON), `matched_rules` (STRING list of triggered rules).

---

## AI Upscaler Nodes

### RadianceUpscaleTiler
**Category:** `◎ RADIANCE/Upscale`

Anti-seam tiling engine. Splits the image into overlapping tiles, applies an upscale function to each, then blends with Gaussian-weighted overlap and 4-level Laplacian pyramid. Returns `upscaled` (IMAGE) and `confidence` (IMAGE, 1-channel weight map).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Input image (any resolution) |
| `upscale_fn` | UPSCALE_FN | — | Function from RadianceUpscaleImage/Router |
| `tile_size` | INT | 512 | Tile size before upscale |
| `overlap` | INT | 128 | Overlap pixels between tiles |
| `blend_mode` | CHOICE | `laplacian_pyramid` | `laplacian_pyramid` or `gaussian` |

### RadianceUpscaleImage
**Category:** `◎ RADIANCE/Upscale`

Three-tier image upscaler. Tier auto-selected by content classifier or overridden.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Input image |
| `model_tier` | CHOICE | `auto` | `auto`, `tier1_fast`, `tier2_quality`, `tier3_creative` |
| `scale` | INT | 4 | Upscale factor (1, 2, 4) |
| `tile_size` | INT | 512 | Tile size for tiled processing |
| `overlap` | INT | 128 | Tile overlap in pixels |

Outputs: `upscaled` (IMAGE), `confidence` (IMAGE), `tier_used` (STRING).

### RadianceUpscaleVideo
**Category:** `◎ RADIANCE/Upscale`

Temporal-coherent video upscaler. Uses SeedVR2-style 4n+1 overlapping windows with sinusoidal ramp weights, Lucas-Kanade optical flow for cross-window alignment, and Laplacian pyramid blending at seams. Eliminates inter-chunk flicker.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `images` | IMAGE | — | Batch of frames (B, H, W, C) |
| `model_tier` | CHOICE | `auto` | Upscale tier |
| `scale` | INT | 4 | Upscale factor |
| `window_size` | INT | 9 | Frames per SeedVR2 window |
| `overlap_frames` | INT | 2 | Overlap between windows |

### RadianceUpscaleFaceRestore
**Category:** `◎ RADIANCE/Upscale`

Post-upscale face enhancement. Detects faces (RetinaFace → Haar cascade fallback), restores each at 512×512 (CodeFormer → GFPGAN → identity), and composites back with Gaussian-feather mask.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Upscaled input image |
| `face_model` | CHOICE | `codeformer` | `codeformer` or `gfpgan_v1.4` |
| `fidelity_weight` | FLOAT | 0.5 | 0 = max enhancement, 1 = max fidelity |
| `blend_radius` | INT | 20 | Gaussian feather radius (pixels) |
| `min_face_px` | INT | 32 | Minimum face size to restore |

### RadianceUpscaleColourFix
**Category:** `◎ RADIANCE/Upscale`

Histogram-match colour-drift correction. Aligns per-channel CDF of the processed image to the original source. Uses `torch.searchsorted` vectorised CDF inversion (~100× faster than pixel loop).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Colour-drifted image (post-upscale/diffusion) |
| `reference` | IMAGE | — | Original source image |
| `strength` | FLOAT | 0.85 | Blend between corrected and drifted |
| `n_bins` | INT | 512 | Histogram bins for CDF estimation |

---

## Color Science Nodes (v3 Additions)

### RadianceBitDepthDegrade
**Category:** `◎ RADIANCE/Color/Science`

Quantize float images to N-bit precision and back — simulates quality loss from 8-bit/10-bit intermediary storage. Essential for demo workflows: ARRI LogC → 8-bit → AI HDR reconstruct → EXR.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Float input image |
| `bit_depth` | INT | 8 | Target bit depth (4–16) |
| `dither_mode` | CHOICE | `triangular` | `none`, `triangular` (TPDF), `floyd-steinberg` |
| `delta_gain` | FLOAT | 10.0 | Amplification for error visualisation output |
| `banding_threshold` | FLOAT | 0.004 | Per-pixel threshold for banding mask (~1 LSB at 8-bit) |

Outputs: `quantized` (IMAGE), `delta_amplified` (IMAGE), `banding_mask` (IMAGE), `metrics` (STRING JSON with PSNR dB, max error, DR loss in stops).

**Transfer functions added:**
- `"BT.1886 (TV γ2.4)"` in `_EOTF_MAP` — correct Rec.709 display EOTF for broadcast/TV work
- `"Rec.709 / BT.1886"` in `_COLOR_SPACES` — named IDT combining Rec.709 primaries with BT.1886 γ2.4 EOTF

---

## LLM Driver Nodes

### RadianceLLMDriver
**Category:** `◎ RADIANCE/AI`

Configure which LLM backend AI nodes use. One driver wire connects to any downstream LLM node.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | CHOICE | `Ollama (Local)` | Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google), Ollama (Local) |
| `model` | STRING | `llama3` | Model name (claude-sonnet-4-6, gpt-4o, gemini-2.0-flash, llama3…) |
| `max_tokens` | INT | 1024 | Max response tokens |
| `temperature` | FLOAT | 0.7 | Sampling temperature |
| `api_key_env` | STRING | `` | Optional environment variable name; defaults per backend when blank |
| `api_key` | STRING | `` | Legacy fallback only; direct values are serialized into workflow JSON |

Outputs: `driver_config` (STRING JSON).

### RadianceLLMPrompt / RadianceLLMImageQuery
**Category:** `◎ RADIANCE/AI`

Send text prompts or image+text queries to the configured backend. ImageQuery encodes the input image as base-64 PNG and uses the vision API.

---

## AI Pipeline Nodes

### RadiancePolicyGuard
**Category:** `◎ RADIANCE/QC`

Per-frame delivery compliance gate. Analyses peak, clipping fraction, black-crush fraction, mean saturation, and gamut violations. Compares against a policy dict (from `RadiancePolicyPreset`).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Frame to evaluate |
| `policy_json` | STRING | `` | Policy from RadiancePolicyPreset |
| `metadata_json` | STRING | `` | Frame metadata for metadata-key checks |

Outputs: `image` (IMAGE pass-through), `report` (STRING JSON), `passed` (BOOLEAN), `score` (INT 0–100).

**Built-in presets** (`RadiancePolicyPreset`): Broadcast SDR, Cinema DCP, Streaming HDR, Web SDR, Archive, Custom.

---

## DiT Adapter Nodes

### RadianceDiTModelConfig
**Category:** `◎ RADIANCE/DiT`

Per-architecture config for SD-VAE (4ch), SDXL-VAE (4ch), LTX-Video (128ch), HunyuanVideo (16ch), Wan 2.1 (16ch), CogVideoX (16ch), Mochi-1 (12ch). Outputs a `dit_config` JSON for downstream adapters.

### RadianceDiTLatentAdapter
**Category:** `◎ RADIANCE/DiT`

Adapt SD-VAE latents to DiT channel count (or vice versa) with per-architecture normalisation. Adaptive channel projection: expand by tiling, contract by folding.

### RadianceDiTLatentNorm / RadianceDiTInspect / RadianceDiTFrameSplit / RadianceDiTFrameMerge
Normalise latents to architecture statistics; inspect shape/dtype; split video latents into per-frame chunks; reassemble chunks into a video latent batch.

---

## Character Consistency Nodes

### RadianceCharacterAnchor
**Category:** `◎ RADIANCE/Character`

Build a character profile from a reference image — CLIP embedding + HSV histogram + face embedding + metadata. Saves to JSON sidecar.

| Input | Type | Description |
|-------|------|-------------|
| `reference_image` | IMAGE | Character reference still |
| `character_name` | STRING | Profile key / filename stem |
| `save_path` | STRING | Directory to persist JSON sidecar |

### RadianceCharacterEnforce
Inject character profile into CONDITIONING via `token_append`, `ip_adapter`, or `blend_text` strategy with configurable strength.

### RadianceCharacterChecker
Per-frame cosine-similarity QC against a stored profile. Returns `similarity` FLOAT (0–1), `passed` BOOLEAN, and `report` STRING JSON.

### RadianceCharacterBlend / RadianceCharacterGallery / RadianceCharacterScoreTimeline
Blend two profiles; browse a folder of profiles as a gallery; export per-shot similarity scores as a JSON timeline for trend analysis.

---

*Radiance v3.0.0 — © 2024–2026 Radiance*
