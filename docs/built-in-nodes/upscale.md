[← Back to Radiance docs](../README.md)

# Upscale

Image and video upscaling with HDR and colour awareness, tiling, and face
restoration.

## Typical graph

```text
Read → Upscale Image (tiled) → Write
```

## Before you use these nodes

- **Backends work in display-referred space.** Feeding scene-linear values
  straight in will not do what you want. Use the node's HDR and colour-encoding
  options so it encodes, upscales and decodes around the model.
- **Models download on first use** and are cached under your ComfyUI models
  directory. The first run of a tier is slow and needs network access.
- **Tiling is memory-safe by design.** Each tile is computed on the GPU and
  accumulated on the CPU, so peak VRAM is one tile regardless of output size.
- **`blend_mode`:** `gaussian_feather` and `linear` are genuinely different
  weightings. `laplacian_pyramid` is not implemented and falls back to the
  Gaussian feather, logging once when it does.

## Known limitations

The Tier 3 diffusion backend is fixed at 4× and ignores the `scale` widget; at
`scale = 2` the result is the top-left quarter of a 4× upscale. `mode =
"creative"` selects that tier regardless of the scale you asked for.

## Nodes in this section (4)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [Upscale Face Restore](#upscale-face-restore) | `RadianceUpscaleFaceRestore` | Restore and enhance facial detail using a face restoration model. |
| [Upscale Image](#upscale-image) | `RadianceUpscaleImage` | Upscale a still image using a selected super-resolution model. |
| [Upscale Tiler](#upscale-tiler) | `RadianceUpscaleTiler` | Tile large images into overlapping patches for memory-safe upscaling. |
| [Upscale Video](#upscale-video) | `RadianceUpscaleVideo` | Upscale a video sequence using a selected super-resolution model. |

---

## Upscale Face Restore

**Node key:** `RadianceUpscaleFaceRestore`  
**Menu:** `FXTD STUDIOS/Radiance/Upscale`  
**Source:** `nodes/upscale/upscale.py`  

Restore and enhance facial detail using a face restoration model.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  | Upscaled image batch (B,H,W,C) float32. |
| `face_model` | Yes | choice of `auto (CodeFormer → GFPGAN → skip)`, `codeformer`, `gfpgan_v1.4`, `skip (detection only)` | `auto (CodeFormer → GFPGAN → skip)` |  | Face restoration model. Auto tries CodeFormer first, falls back to GFPGAN, skips if neither is available. |
| `fidelity_weight` | Yes | `FLOAT` | `0.75` | 0.0 – 1.0, step 0.05 | CodeFormer fidelity: 0 = maximum enhancement (creative), 1 = faithful to input (precise). 0.5–0.8 is recommended for most upscaled content. |
| `blend_radius` | Yes | `INT` | `20` | 0 – 80, step 4 | Gaussian feather radius in pixels at face crop edge. Higher = softer transition. 0 = hard paste. |
| `face_pad_frac` | Yes | `FLOAT` | `0.25` | 0.0 – 0.6, step 0.05 | Extra padding around each detected face bbox (fraction of face width/height). 0.25 = 25%. |
| `min_face_px` | Yes | `INT` | `64` | 16 – 256, step 16 | Smallest face (in pixels) to process. Smaller faces are skipped. |
| `colour_correct` | Yes | `BOOLEAN` | `True` |  | Apply histogram-match colour correction after restoration to cancel diffusion colour drift. |
| `colour_strength` | Yes | `FLOAT` | `0.8` | 0.0 – 1.0, step 0.05 | Strength of histogram-match correction. 1.0 = full match to input colours. |
| `original_images` | No | `IMAGE` |  |  | Original (pre-upscale) images for colour reference. Used by histogram-match correction. Leave disconnected to use the restored images as self-reference. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `restored` | `IMAGE` |  |
| `face_mask` | `IMAGE` |  |
| `pass_info` | `STRING` |  |

---

## Upscale Image

**Node key:** `RadianceUpscaleImage`  
**Menu:** `FXTD STUDIOS/Radiance/Upscale`  
**Source:** `nodes/upscale/upscale.py`  

Upscale a still image using a selected super-resolution model.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `operation` | Yes | choice of `Upscale`, `Route` | `Upscale` |  |  |
| `images` | Yes | `IMAGE` |  |  | Input image batch. |
| `scale` | No | choice of `2×`, `4×`, `8× (tile cascade)` | `4×` |  |  |
| `hdr_mode` | No | choice of `auto`, `preserve`, `clamp` | `auto` |  | Scene-linear / HDR handling. auto: preserve range when input exceeds 1.0, else clamp. preserve: Reinhard tonemap before SR and re-expand after (keeps highlights >1.0). clamp: legacy [0,1] (LDR). |
| `color_encoding` | No | choice of `passthrough`, `linear<->sRGB`, `linear<->LogC3` | `passthrough` |  | Encode scene-linear -> display (sRGB/LogC3) before SR and decode after, so the LDR-trained network sees the domain it expects. passthrough: feed pixels unchanged. |
| `mode` | No | choice of `precise`, `creative`, `balanced` | `precise` |  | precise: Real-ESRGAN fidelity-first. creative: diffusion detail hallucination (requires VRAM). balanced: GAN upscale + light sharpening. |
| `tile_size` | No | `INT` | `512` | 128 – 1024, step 64 | Tile size in input pixels. Reduce if OOM. |
| `overlap` | No | `INT` | `128` | 32 – 256, step 32 |  |
| `sharpness_boost` | No | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.05 | Unsharp mask strength applied after upscale. |
| `denoise_pre` | No | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.05 | Gaussian pre-denoise strength. |
| `upscale_model` | No | `UPSCALE_MODEL` |  |  |  |
| `model_tier` | No | choice of `auto`, `tier1_fast    (Real-ESRGAN — GAN, ms/frame)`, `tier2_quality (HAT-L — transformer SOTA PSNR)`, `tier2_quality (SwinIR-L — transformer quality)`, `tier3_creative (SD x4 — diffusion hallucination)`, `tier3_creative (SeedVR2 — one-step video diffusion)` | `auto` |  | Model tier. 'auto' selects based on content analysis. |
| `diffusion_steps` | No | `INT` | `20` | 1 – 50 |  |
| `diffusion_noise_level` | No | `INT` | `20` | 0 – 350, step 10 |  |
| `guidance_scale` | No | `FLOAT` | `7.5` | 1.0 – 20.0, step 0.5 |  |
| `enhancement_prompt` | No | `STRING` | `` |  | Text prompt for creative mode diffusion steering. |
| `prefer_speed` | No | `BOOLEAN` | `False` |  | Always recommend Tier 1 fast regardless of content. |
| `sample_frame` | No | `INT` | `0` | 0 – 9999 | Index of frame to analyse (for Route operation). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image_a` | `IMAGE` |  |
| `image_b` | `IMAGE` |  |
| `info` | `STRING` |  |
| `data1` | `STRING` |  |
| `data2` | `STRING` |  |

---

## Upscale Tiler

**Node key:** `RadianceUpscaleTiler`  
**Menu:** `FXTD STUDIOS/Radiance/Upscale`  
**Source:** `nodes/upscale/upscale.py`  

Tile large images into overlapping patches for memory-safe upscaling.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `operation` | Yes | choice of `Tile`, `ColourFix` | `Tile` |  |  |
| `images` | No | `IMAGE` |  |  | Input image batch (B,H,W,C) float32. |
| `scale` | No | choice of `2×`, `4×`, `8× (tile cascade)` | `4×` |  | Upscale factor. 8× uses two cascaded 4× passes. |
| `tile_size` | No | `INT` | `512` | 128 – 2048, step 64 | Tile side in input pixels. Smaller = less VRAM. |
| `overlap` | No | `INT` | `128` | 32 – 512, step 32 | Tile overlap in input pixels. ≥20% of tile_size recommended. |
| `blend_mode` | No | choice of `laplacian_pyramid`, `gaussian_feather`, `linear` | `laplacian_pyramid` |  | laplacian_pyramid: best quality. gaussian_feather: fast. linear: simple. |
| `upscale_model` | No | `UPSCALE_MODEL` |  |  | Any ComfyUI UPSCALE_MODEL. Leave empty to use built-in Real-ESRGAN. |
| `model_tier` | No | choice of `auto`, `tier1_fast    (Real-ESRGAN — GAN, ms/frame)`, `tier2_quality (HAT-L — transformer SOTA PSNR)`, `tier2_quality (SwinIR-L — transformer quality)`, `tier3_creative (SD x4 — diffusion hallucination)`, `tier3_creative (SeedVR2 — one-step video diffusion)` | `tier1_fast (Real-ESRGAN — GAN, ms/frame)` |  | Built-in model tier when no upscale_model is connected. |
| `source` | No | `IMAGE` |  |  | Upscaled image with colour drift (ColourFix mode). |
| `reference` | No | `IMAGE` |  |  | Original pre-upscale image — colour reference (ColourFix mode). |
| `cf_strength` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.05 | ColourFix strength: 0 = off, 1 = full CDF match. |
| `n_bins` | No | `INT` | `512` | 64 – 2048, step 64 | Histogram resolution (ColourFix mode). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image_a` | `IMAGE` |  |
| `image_b` | `IMAGE` |  |
| `info` | `STRING` |  |

---

## Upscale Video

**Node key:** `RadianceUpscaleVideo`  
**Menu:** `FXTD STUDIOS/Radiance/Upscale`  
**Source:** `nodes/upscale/upscale.py`  

Upscale a video sequence using a selected super-resolution model.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `frames` | Yes | `IMAGE` |  |  | Video frame batch (B,H,W,C) float32. B = frame count. |
| `scale` | Yes | choice of `2×`, `4×`, `8× (tile cascade)` | `4×` |  |  |
| `tile_size` | Yes | `INT` | `512` | 128 – 1024, step 64 |  |
| `overlap_spatial` | Yes | `INT` | `128` | 32 – 256, step 32 | Spatial tile overlap in input pixels. |
| `window_size` | Yes | `INT` | `16` | 4 – 64, step 4 | Temporal window (frames processed together). Larger = better consistency but more VRAM. |
| `overlap_temporal` | Yes | `INT` | `4` | 1 – 16 | Frames shared between adjacent windows. Minimum 1 for seam-free stitching. |
| `flow_compensation` | Yes | `BOOLEAN` | `True` |  | Use Lucas-Kanade optical flow to warp reference frames before blending temporal window seams. |
| `sharpness_boost` | Yes | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.05 |  |
| `upscale_model` | No | `UPSCALE_MODEL` |  |  |  |
| `model_tier` | No | choice of `auto`, `tier1_fast    (Real-ESRGAN — GAN, ms/frame)`, `tier2_quality (HAT-L — transformer SOTA PSNR)`, `tier2_quality (SwinIR-L — transformer quality)`, `tier3_creative (SD x4 — diffusion hallucination)`, `tier3_creative (SeedVR2 — one-step video diffusion)` | `tier1_fast (Real-ESRGAN — GAN, ms/frame)` |  | Select 'SeedVR2' for best temporal consistency on video. Requires seedvr2 or diffusers package. |
| `enhancement_prompt` | No | `STRING` | `` |  | Text prompt for Tier 3 diffusion steering (e.g. 'cinematic film grain, detailed textures'). |
| `diffusion_steps` | No | `INT` | `1` | 1 – 50 | Diffusion inference steps. SeedVR2 uses 1 (one-step); SD x4 upscaler recommended 15-25. |
| `hdr_mode` | No | choice of `auto`, `preserve`, `clamp` | `auto` |  | Scene-linear / HDR handling. auto: preserve range when input exceeds 1.0, else clamp. preserve: Reinhard tonemap before SR and re-expand after. clamp: legacy [0,1] (LDR). |
| `color_encoding` | No | choice of `passthrough`, `linear<->sRGB`, `linear<->LogC3` | `passthrough` |  | Encode scene-linear -> display (sRGB/LogC3) before SR and decode after. passthrough: feed pixels unchanged. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `upscaled` | `IMAGE` |  |
| `confidence_map` | `IMAGE` |  |
| `pass_info` | `STRING` |  |

---
