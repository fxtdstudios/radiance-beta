[← Back to Radiance docs](../README.md)

# Upscale

Image and video upscaling, tiling, confidence outputs, and face restoration for high-resolution finishing workflows.

## Typical workflow

```text
Image/video batch -> Upscale Tiler -> Upscale Image / Video -> Face Restore -> Viewer / Write
```

## Before you use these nodes

- Tile large images when VRAM is tight.
- Inspect confidence and face masks; restoration can overcorrect identity or fine texture.
- Add grain after upscale/restoration when matching a plate.
- **Scene-linear / HDR input:** Upscale Image and Upscale Video expose `hdr_mode` (`auto`/`preserve`/`clamp`) — `preserve` tonemaps (Reinhard) before super-resolution and re-expands after, so highlights above 1.0 survive — and `color_encoding` (`passthrough` / `linear<->sRGB` / `linear<->LogC3`) which round-trips through a display transfer so the LDR-trained SR networks see the domain they expect. Leave both on defaults (`auto` / `passthrough`) for display-referred input.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Upscale Tiler](#upscale-tiler) | `RadianceUpscaleTiler` | Upscale Tiler. |
| [◎ Upscale Image](#upscale-image) | `RadianceUpscaleImage` | Upscale Image. |
| [◎ Upscale Video](#upscale-video) | `RadianceUpscaleVideo` | Upscale Video. |
| [◎ Upscale Face Restore](#upscale-face-restore) | `RadianceUpscaleFaceRestore` | Upscale Face Restore. |

## ◎ Upscale Tiler

**Internal key:** `RadianceUpscaleTiler`  
**Category:** `FXTD STUDIOS/Radiance/◎ Upscale`  
**Source:** `nodes/upscale/upscale.py`
**Function:** `run`

### What it does

Upscale Tiler.

### When to use it

Use `◎ Upscale Tiler` when the graph reaches the Upscale Tiler step in a upscale workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `operation` | Yes | `ENUM: Tile, ColourFix` | `Tile` | - |
| `images` | Optional | `IMAGE` | - | Input image batch (B,H,W,C) float32. |
| `scale` | Optional | `(_SCALE_CHOICES, {'default': '4×', 'tooltip': 'Upscale factor. 8× uses two cascaded 4× passes.'})` | - | - |
| `tile_size` | Optional | `INT` | `512` | Tile side in input pixels. Smaller = less VRAM. |
| `overlap` | Optional | `INT` | `128` | Tile overlap in input pixels. ≥20% of tile_size recommended. |
| `blend_mode` | Optional | `(_BLEND_CHOICES, {'default': 'laplacian_pyramid', 'tooltip': 'laplacian_pyramid: best quality. gaussian_feather: fast. linear: simple.'})` | - | - |
| `upscale_model` | Optional | `UPSCALE_MODEL` | - | Any ComfyUI UPSCALE_MODEL. Leave empty to use built-in Real-ESRGAN. |
| `model_tier` | Optional | `(_TIER_CHOICES, {'default': 'tier1_fast    (Real-ESRGAN — GAN, ms/frame)', 'tooltip': 'Built-in model tier when no upscale_model is connected.'})` | - | - |
| `source` | Optional | `IMAGE` | - | Upscaled image with colour drift (ColourFix mode). |
| `reference` | Optional | `IMAGE` | - | Original pre-upscale image — colour reference (ColourFix mode). |
| `cf_strength` | Optional | `FLOAT` | `1.0` | ColourFix strength: 0 = off, 1 = full CDF match. |
| `n_bins` | Optional | `INT` | `512` | Histogram resolution (ColourFix mode). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image_a` | `IMAGE` | Output produced by the `image_a` socket. |
| `image_b` | `IMAGE` | Output produced by the `image_b` socket. |
| `info` | `STRING` | Output produced by the `info` socket. |

### Practical notes

- The node returns `image_a` (`IMAGE`), `image_b` (`IMAGE`), `info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Upscale Image

**Internal key:** `RadianceUpscaleImage`  
**Category:** `FXTD STUDIOS/Radiance/◎ Upscale`  
**Source:** `nodes/upscale/upscale.py`
**Function:** `run`

### What it does

Upscale Image.

### When to use it

Use `◎ Upscale Image` when the graph reaches the Upscale Image step in a upscale workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `operation` | Yes | `ENUM: Upscale, Route` | `Upscale` | - |
| `images` | Yes | `IMAGE` | - | Input image batch. |
| `scale` | Optional | `(_SCALE_CHOICES, {'default': '4×'})` | - | - |
| `mode` | Optional | `(_MODE_CHOICES, {'default': 'precise', 'tooltip': 'precise: Real-ESRGAN fidelity-first. creative: diffusion detail hallucination (requires VRAM). balanced: GAN upscale + light sharpening.'})` | - | - |
| `tile_size` | Optional | `INT` | `512` | Tile size in input pixels. Reduce if OOM. |
| `overlap` | Optional | `INT` | `128` | - |
| `sharpness_boost` | Optional | `FLOAT` | `0.0` | Unsharp mask strength applied after upscale. |
| `denoise_pre` | Optional | `FLOAT` | `0.0` | Gaussian pre-denoise strength. |
| `upscale_model` | Optional | `UPSCALE_MODEL` | - | - |
| `model_tier` | Optional | `(_TIER_CHOICES, {'default': 'auto', 'tooltip': "Model tier. 'auto' selects based on content analysis."})` | - | - |
| `diffusion_steps` | Optional | `INT` | `20` | - |
| `diffusion_noise_level` | Optional | `INT` | `20` | - |
| `guidance_scale` | Optional | `FLOAT` | `7.5` | - |
| `enhancement_prompt` | Optional | `STRING` | `` | Text prompt for creative mode diffusion steering. |
| `prefer_speed` | Optional | `BOOLEAN` | `False` | Always recommend Tier 1 fast regardless of content. |
| `sample_frame` | Optional | `INT` | `0` | Index of frame to analyse (for Route operation). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image_a` | `IMAGE` | Output produced by the `image_a` socket. |
| `image_b` | `IMAGE` | Output produced by the `image_b` socket. |
| `info` | `STRING` | Output produced by the `info` socket. |
| `data1` | `STRING` | Output produced by the `data1` socket. |
| `data2` | `STRING` | Output produced by the `data2` socket. |

### Practical notes

- The node returns `image_a` (`IMAGE`), `image_b` (`IMAGE`), `info` (`STRING`), `data1` (`STRING`), `data2` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Upscale Video

**Internal key:** `RadianceUpscaleVideo`  
**Category:** `FXTD STUDIOS/Radiance/◎ Upscale`  
**Source:** `nodes/upscale/upscale.py`
**Function:** `upscale_video`

### What it does

Upscale Video.

### When to use it

Use `◎ Upscale Video` when the graph reaches the Upscale Video step in a upscale workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `frames` | Yes | `IMAGE` | - | Video frame batch (B,H,W,C) float32. B = frame count. |
| `scale` | Yes | `(_SCALE_CHOICES, {'default': '4×'})` | - | - |
| `tile_size` | Yes | `INT` | `512` | - |
| `overlap_spatial` | Yes | `INT` | `128` | Spatial tile overlap in input pixels. |
| `window_size` | Yes | `INT` | `16` | Temporal window (frames processed together). Larger = better consistency but more VRAM. |
| `overlap_temporal` | Yes | `INT` | `4` | Frames shared between adjacent windows. Minimum 1 for seam-free stitching. |
| `flow_compensation` | Yes | `BOOLEAN` | `True` | Use Lucas-Kanade optical flow to warp reference frames before blending temporal window seams. |
| `sharpness_boost` | Yes | `FLOAT` | `0.0` | - |
| `upscale_model` | Optional | `UPSCALE_MODEL` | - | - |
| `model_tier` | Optional | `(_TIER_CHOICES, {'default': 'tier1_fast    (Real-ESRGAN — GAN, ms/frame)', 'tooltip': "Select 'SeedVR2' for best temporal consistency on video. Requires seedvr2 or diffusers package."})` | - | - |
| `enhancement_prompt` | Optional | `STRING` | `` | Text prompt for Tier 3 diffusion steering (e.g. 'cinematic film grain, detailed textures'). |
| `diffusion_steps` | Optional | `INT` | `1` | Diffusion inference steps. SeedVR2 uses 1 (one-step); SD x4 upscaler recommended 15-25. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `upscaled` | `IMAGE` | Output produced by the `upscaled` socket. |
| `confidence_map` | `IMAGE` | Output produced by the `confidence_map` socket. |
| `pass_info` | `STRING` | Output produced by the `pass_info` socket. |

### Practical notes

- The node returns `upscaled` (`IMAGE`), `confidence_map` (`IMAGE`), `pass_info` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Upscale Face Restore

**Internal key:** `RadianceUpscaleFaceRestore`  
**Category:** `FXTD STUDIOS/Radiance/◎ Upscale`  
**Source:** `nodes/upscale/upscale.py`
**Function:** `restore_faces`

### What it does

Upscale Face Restore.

### When to use it

Use `◎ Upscale Face Restore` when the graph reaches the Upscale Face Restore step in a upscale workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` | - | Upscaled image batch (B,H,W,C) float32. |
| `face_model` | Yes | `(_FACE_MODEL_CHOICES, {'default': 'auto (CodeFormer → GFPGAN → skip)', 'tooltip': 'Face restoration model. Auto tries CodeFormer first, falls back to GFPGAN, skips if neither is available.'})` | - | - |
| `fidelity_weight` | Yes | `FLOAT` | `0.75` | CodeFormer fidelity: 0 = maximum enhancement (creative), 1 = faithful to input (precise). 0.5–0.8 is recommended for most upscaled content. |
| `blend_radius` | Yes | `INT` | `20` | Gaussian feather radius in pixels at face crop edge. Higher = softer transition. 0 = hard paste. |
| `face_pad_frac` | Yes | `FLOAT` | `0.25` | Extra padding around each detected face bbox (fraction of face width/height). 0.25 = 25%. |
| `min_face_px` | Yes | `INT` | `64` | Smallest face (in pixels) to process. Smaller faces are skipped. |
| `colour_correct` | Yes | `BOOLEAN` | `True` | Apply histogram-match colour correction after restoration to cancel diffusion colour drift. |
| `colour_strength` | Yes | `FLOAT` | `0.8` | Strength of histogram-match correction. 1.0 = full match to input colours. |
| `original_images` | Optional | `IMAGE` | - | Original (pre-upscale) images for colour reference. Used by histogram-match correction. Leave disconnected to use the restored images as self-reference. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `restored` | `IMAGE` | Output produced by the `restored` socket. |
| `face_mask` | `IMAGE` | Output produced by the `face_mask` socket. |
| `pass_info` | `STRING` | Output produced by the `pass_info` socket. |

### Practical notes

- The node returns `restored` (`IMAGE`), `face_mask` (`IMAGE`), `pass_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
