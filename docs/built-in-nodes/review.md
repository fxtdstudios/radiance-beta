[← Back to Radiance docs](../README.md)

# Review, Viewer, and Preview

The Viewer, the Lite Viewer, scopes, focus peaking, contact sheets, flipbooks,
frame stamps, QC and the preview server.

## Typical graph

```text
work → Radiance Viewer → (grade in the viewer) → RENDER → delivered master
```

## Before you use these nodes

- **The Viewer is also the delivery panel.** Grading done in the viewer is
  applied to the exported master. Before 3.2.0 six of those controls — shadows,
  highlights, hue, LUT and gamut compression — were silently dropped on export;
  they now reach the file, and a mismatch between the browser and the server is
  reported in the log. If you are upgrading, hard-refresh ComfyUI so the browser
  picks up the new JavaScript.
- **Scopes are computed on the ungraded texture.** They show you the incoming
  image, not your grade. That is deliberate, but it surprises people.
- **The Viewer always re-runs.** Its `IS_CHANGED` returns NaN so the delivery
  frame cache stays populated, which means it re-executes on every queue even
  when nothing upstream changed.
- **Keyboard shortcuts** are listed in the Viewer's own help overlay.

## Nodes in this section (10)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [Burn-In](#burn-in) | `RadianceFrameStamp` | Burn frame number, timecode, and metadata into an image for dailies. |
| [Contact Sheet](#contact-sheet) | `RadianceContactSheet` | Render a contact sheet grid of multiple images for review. |
| [Flipbook GIF](#flipbook-gif) | `RadianceFlipbookGIF` | Export a sequence as an animated GIF flipbook for quick review. |
| [Focus Peaking](#focus-peaking) | `RadianceFocusPeaking` | Overlay focus-peaking highlights on edges for sharpness assessment. |
| [HDR Monitor](#hdr-monitor) | `RadianceHDRMonitor` | HDR monitor / preview node |
| [Preview Server](#preview-server) | `RadiancePreviewServer` | Run a local HTTP preview server for browser-based image review. |
| [QC](#qc) | `RadianceQC` | Run a configurable suite of QC checks on an image or sequence. |
| [RadiancePolicyGuard](#radiancepolicyguard) | `RadiancePolicyGuard` | Enforce delivery policy constraints (legal range, gamut, loudness). |
| [Viewer](#viewer) | `RadianceViewer` | VFX Industry-Standard Viewer v3.0.0 — Temporal & Intelligence Update: • GPU Waveform / RGB Parade / Vectorscope / Histogram scopes • Power Windows masking (Radial + Box, feather, rotation) • Comparison Bridge — mouse-draggable wipe + reference shelf (8 stills) • Anamorphic lens streaks + Brown-Conrady k1/k2 distortion • Edge-preserving Bilateral Filter denoising (7×7 GPU kernel) • Channel viewing (RGB/R/G/B/Alpha/Luma), False Color, Zebra • 16-bit PNG + .rhdr HDR sidecar + .exr export • IMAGE passthrough — no longer a dead-end node |
| [Viewer (Lite)](#viewer-lite) | `RadianceLiteViewer` | Fast lightweight viewer for compare/check workflows |

---

## Burn-In

**Node key:** `RadianceFrameStamp`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes_realtime_preview.py`  

Burn frame number, timecode, and metadata into an image for dailies.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  |  |
| `start_frame` | Yes | `INT` | `1001` | 0 – 999999 | Frame number of the first frame in the batch. |
| `fps` | Yes | `FLOAT` | `24.0` | 1.0 – 120.0, step 0.001 | Frames per second (used for timecode calculation). |
| `drop_frame` | No | `BOOLEAN` | `False` |  | Use SMPTE drop-frame timecode (DF). Only meaningful at 29.97 or 59.94 fps. Uses ';' separator instead of ':' for the frame field. |
| `show_frame_number` | No | `BOOLEAN` | `True` |  |  |
| `show_timecode` | No | `BOOLEAN` | `True` |  |  |
| `custom_text` | No | `STRING` | `` |  | Additional text burned into each frame (e.g. shot name, version). |
| `position` | No | choice of `bottom_left`, `bottom_right`, `top_left`, `top_right`, `center` | `bottom_left` |  |  |
| `font_scale` | No | `FLOAT` | `1.0` | 0.5 – 4.0, step 0.1 |  |
| `opacity` | No | `FLOAT` | `0.85` | 0.1 – 1.0, step 0.05 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `stamped` | `IMAGE` |  |

---

## Contact Sheet

**Node key:** `RadianceContactSheet`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes_realtime_preview.py`  

Render a contact sheet grid of multiple images for review.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  |  |
| `thumb_width` | Yes | `INT` | `160` | 32 – 512, step 8 | Width of each thumbnail in pixels. |
| `max_cols` | Yes | `INT` | `8` | 1 – 32 | Maximum number of columns. Rows are computed automatically. |
| `label_frames` | No | `BOOLEAN` | `True` |  | Print the frame index below each thumbnail. |
| `background` | No | choice of `Black`, `Grey`, `White` | `Black` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `contact_sheet` | `IMAGE` |  |
| `grid_cols` | `INT` |  |
| `grid_rows` | `INT` |  |

---

## Flipbook GIF

**Node key:** `RadianceFlipbookGIF`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes_realtime_preview.py`  
**Output node** — runs even with nothing connected downstream.  

Export a sequence as an animated GIF flipbook for quick review.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  |  |
| `save_path` | Yes | `STRING` | `preview/flipbook.gif` |  | Output .gif path. A relative path is written under ComfyUI's output/ folder; absolute paths are used as given. The directory is created automatically. |
| `fps` | Yes | `FLOAT` | `12.0` | 1.0 – 60.0, step 0.5 | Playback speed. GIF frame delay = 1000/fps ms. |
| `max_width` | Yes | `INT` | `480` | 64 – 1920, step 8 | Resize frames to this width (preserves aspect ratio). Smaller = smaller file. |
| `loop` | No | `BOOLEAN` | `True` |  | Loop the animation indefinitely. |
| `dither` | No | `BOOLEAN` | `True` |  | Enable Floyd-Steinberg dithering for smoother gradients. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passthrough` | `IMAGE` |  |
| `status` | `STRING` |  |

---

## Focus Peaking

**Node key:** `RadianceFocusPeaking`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes_realtime_preview.py`  

Overlay focus-peaking highlights on edges for sharpness assessment.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `threshold` | Yes | `FLOAT` | `0.2` | 0.01 – 1.0, step 0.01 | Normalised Sobel magnitude above which a pixel is considered in-focus. |
| `peak_color` | Yes | choice of `Red`, `Green`, `White`, `Yellow`, `Cyan` | `Red` |  |  |
| `strength` | Yes | `FLOAT` | `0.85` | 0.0 – 1.0, step 0.05 | Blend factor for the peaking overlay. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passthrough` | `IMAGE` |  |
| `focus_peak` | `IMAGE` |  |

---

## HDR Monitor

**Node key:** `RadianceHDRMonitor`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes/hdr/delivery.py`  
**Output node** — runs even with nothing connected downstream.  

HDR monitor / preview node. SDR tone-map or direct PQ/HLG display encoding. Wire in parallel — does not affect your HDR pipeline.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `mode` | Yes | choice of `Preview (SDR)`, `Rec.2100 PQ`, `Rec.2100 HLG` | `Preview (SDR)` |  |  |
| `operator` | No | choice of `ACES (Narkowicz)`, `Filmic (Hejl-Burgess)`, `Reinhard Extended`, `Exposure + Gamma` | `ACES (Narkowicz)` |  | [Preview (SDR)] Tone-mapping operator. |
| `exposure` | No | `FLOAT` | `0.0` | -6.0 – 6.0, step 0.1 | [Preview (SDR)] EV offset applied before tone mapping. |
| `saturation` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | [Preview (SDR)] Post-tonemap saturation scale. |
| `gamma` | No | `FLOAT` | `2.2` | 1.0 – 3.0, step 0.05 | [Preview (SDR)] Display gamma. 2.2 ≈ sRGB, 2.4 = IEC 61966 precise. |
| `reinhard_white` | No | `FLOAT` | `4.0` | 0.5 – 100.0, step 0.5 | [Preview (SDR) / Reinhard Extended] White point. |
| `peak_nits` | No | `FLOAT` | `1000.0` | 100.0 – 10000.0, step 100.0 | [Rec.2100 PQ] Mastering display peak luminance in nits. |
| `gamma_correct_sdr` | No | `BOOLEAN` | `True` |  | [Preview (SDR)] Apply sRGB gamma encoding to output. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `preview` | `IMAGE` |  |

---

## Preview Server

**Node key:** `RadiancePreviewServer`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes_realtime_preview.py`  
**Output node** — runs even with nothing connected downstream.  

Run a local HTTP preview server for browser-based image review.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  |  |
| `port` | Yes | `INT` | `8765` | 1024 – 65535 | TCP port for the preview HTTP server. |
| `stream_name` | Yes | `STRING` | `radiance` |  | Stream identifier. Access at /frame/<stream_name>. |
| `jpeg_quality` | No | `INT` | `85` | 20 – 99 | JPEG compression quality (20=small, 99=lossless-ish). |
| `resize_width` | No | `INT` | `0` | 0 – 3840 | Resize frame before serving (0 = original size). Smaller = faster over network. |
| `enabled` | No | `BOOLEAN` | `True` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passthrough` | `IMAGE` |  |
| `server_url` | `STRING` |  |

---

## QC

**Node key:** `RadianceQC`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes/color/qc.py`  
**Output node** — runs even with nothing connected downstream.  

Run a configurable suite of QC checks on an image or sequence.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `mode` | Yes | choice of `Analyze`, `Export` | `Analyze` |  |  |
| `image` | No | `IMAGE` |  |  |  |
| `black_threshold` | No | `FLOAT` | `0.0` | -0.1 – 0.1, step 0.001 |  |
| `white_threshold` | No | `FLOAT` | `1.0` | 0.8 – 2.0, step 0.01 |  |
| `overlay_opacity` | No | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.1 |  |
| `banding_threshold` | No | `FLOAT` | `5.0` | 0.0 – 20.0, step 0.5 |  |
| `enable_focus_check` | No | `BOOLEAN` | `False` |  |  |
| `enable_artifacts_check` | No | `BOOLEAN` | `True` |  |  |
| `enable_noise_check` | No | `BOOLEAN` | `True` |  |  |
| `fail_on_errors` | No | `BOOLEAN` | `False` |  |  |
| `qc_report_json` | No | `STRING` |  |  |  |
| `output_path` | No | `STRING` | `` |  |  |
| `filename_prefix` | No | `STRING` | `qc_report` |  |  |
| `export_format` | No | choice of `json`, `csv`, `html`, `all` | `json` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `text_report` | `STRING` |  |
| `json_report` | `STRING` |  |
| `status` | `STRING` |  |

---

## RadiancePolicyGuard

**Node key:** `RadiancePolicyGuard`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes/color/qc.py`  

Enforce delivery policy constraints (legal range, gamut, loudness).

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `mode` | Yes | choice of `Preset`, `Guard` | `Guard` |  |  |
| `image` | Yes | `IMAGE` |  |  |  |
| `preset` | No | choice of `Broadcast SDR`, `Cinema HDR (P3-PQ)`, `OTT HDR10`, `Social Media`, `Custom` | `Broadcast SDR` |  |  |
| `policy_file` | No | `STRING` | `` |  |  |
| `custom_max_peak_nits` | No | `FLOAT` | `1000.0` | 0.0 – 10000.0, step 10.0 |  |
| `custom_max_clipping` | No | `FLOAT` | `0.01` | 0.0 – 1.0, step 0.001 |  |
| `custom_max_black_crush` | No | `FLOAT` | `0.05` | 0.0 – 1.0, step 0.001 |  |
| `custom_max_saturation` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 |  |
| `policy` | No | `STRING` |  |  |  |
| `max_clipping` | No | `FLOAT` | `0.01` | 0.0 – 1.0, step 0.001 |  |
| `max_black_crush` | No | `FLOAT` | `0.05` | 0.0 – 1.0, step 0.001 |  |
| `max_saturation` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 |  |
| `max_peak_nits` | No | `FLOAT` | `1000.0` | 0.0 – 10000.0, step 10.0 |  |
| `require_metadata` | No | `STRING` | `` |  |  |
| `metadata_present` | No | `STRING` | `` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `passed` | `BOOLEAN` |  |
| `data1` | `STRING` |  |
| `data2` | `STRING` |  |
| `score` | `INT` |  |

---

## Viewer

**Node key:** `RadianceViewer`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes/monitor/viewer.py`  
**Output node** — runs even with nothing connected downstream.  

VFX Industry-Standard Viewer v3.0.0 — Temporal & Intelligence Update:
• GPU Waveform / RGB Parade / Vectorscope / Histogram scopes
• Power Windows masking (Radial + Box, feather, rotation)
• Comparison Bridge — mouse-draggable wipe + reference shelf (8 stills)
• Anamorphic lens streaks + Brown-Conrady k1/k2 distortion
• Edge-preserving Bilateral Filter denoising (7×7 GPU kernel)
• Channel viewing (RGB/R/G/B/Alpha/Luma), False Color, Zebra
• 16-bit PNG + .rhdr HDR sidecar + .exr export
• IMAGE passthrough — no longer a dead-end node

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE,VIDEO` |  |  |  |
| `compare_image` | No | `IMAGE,VIDEO` |  |  |  |
| `zdepth` | No | `IMAGE,VIDEO` |  |  | Z-Depth map to display when pressing Z button |

Hidden inputs supplied by ComfyUI: `extra_pnginfo`, `prompt`, `unique_id`.

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## Viewer (Lite)

**Node key:** `RadianceLiteViewer`  
**Menu:** `FXTD STUDIOS/Radiance/Review`  
**Source:** `nodes/monitor/lite_viewer.py`  
**Output node** — runs even with nothing connected downstream.  

Fast lightweight viewer for compare/check workflows. Provides fit, 1:1, pan/zoom, wipe, split, diff, onion, clipping, alpha, and pixel inspect.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE,VIDEO` |  |  |  |
| `compare_image` | No | `IMAGE,VIDEO` |  |  | Optional B image for wipe, split, diff, and onion checks. |

Hidden inputs supplied by ComfyUI: `unique_id`.

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---
