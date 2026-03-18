<div align="center">

![RADIANCE](docs/RADIANCE.png)

![Version](https://img.shields.io/badge/version-2.2-blue)
![License](https://img.shields.io/badge/license-GPL--3.0-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-purple)
![Nodes](https://img.shields.io/badge/Nodes-79-blue?style=for-the-badge&logo=comfyui)
![Version](https://img.shields.io/badge/Version-2.2-orange?style=for-the-badge)

</div>

**Radiance** is a professional, VFX-grade 32-bit float color science suite for ComfyUI. Built for film editors, colorists, and VFX artists who require absolute precision in their AI-assisted workflows.

[Installation](#installation) · [Node Reference](#node-reference) · [Quick Start](#quick-start) · [Viewer Shortcuts](#viewer-shortcuts) · [Documentation](https://fxtd.org/radiance_docs/) · [What's New](#whats-new-v211)

---

## What's New - v2.2

### ◎ Radiance v2.2 (The Professional Suite)
- **Video-Native Workflow** - Radiance Viewer now accepts `VIDEO` and `IMAGE` inputs interchangeably with real-time frame extraction.
- **Cinematic Prompt Enhancer** - Built-in AI refinement widget in `Cinematic Prompt Machine` for physically-accurate camera/lighting prompts.
- **Terminal HUD & Live REPL** - Nuke-style Python interaction directly inside the viewer. Inspect tensors or run math in real-time.
- **Interactive Mask Editor** - `◎ Radiance Load Image` now includes a non-destructive soft-brush mask editor for immediate compositing.
- **◎ Radiance 32-bit Denoise** - new edge-preserving bilateral filter for 32-bit float images.
- **◎ Radiance Reroute / Reroute+** — compact visual reroute nodes with auto-type detection.
- **◎ Show Text (Radiance)** - display string, JSON, or any data type output directly on the node UI.

### ◎ New Color Nodes (4)
| Node | File | What it does |
|---|---|---|
| **◎ Radiance Grade Match** | `nodes_grade.py` | Match source image color statistics to a reference using CIE L\*a\*b\* mean/std |
| **◎ Apply Grade Info** | `nodes_grade.py` | Replay a `grade_info` JSON string onto any image — closes the grade roundtrip |
| **◎ Radiance LUT Bake** | `nodes_lut.py` | Bake grade params into a 33³ `.cube` LUT (Resolve / Nuke / Premiere ready) |
| **◎ Radiance LUT Apply** | `nodes_lut.py` | Load any `.cube` LUT and apply via trilinear interpolation |

## Node Reference (79 Total)

### ◎ Core & Viewers
| Node | File | Description |
| :--- | :--- | :--- |
| **◎ Radiance Viewer** | `nodes_radiance_viewer.py` | Professional v3.2 HUD with GPU scopes, LRU caching, and Terminal REPL |
| **◎ Radiance Grade Apply** | `nodes_radiance_viewer.py` | Bakes Viewer grading parameters into a 32-bit tensor |
| **◎ Radiance Manager** | `nodes_studio.py` | Centralized project and version management following VFX standards |
| **◎ Radiance .rad Workspace** | `nodes_workspace.py` | Save/Load state and custom node templates |
| **◎ Show Text (Radiance)** | `nodes_text.py` | Visual debugger for strings, JSON, and complex dictionaries |

### ◎ Color & HDR Pipeline
| Node | File | Description |
| :--- | :--- | :--- |
| **◎ Radiance Grade** | `nodes_grade.py` | Primary 32-bit Lift/Gamma/Gain/Offset grading engine |
| **◎ Radiance Grade Match** | `nodes_grade.py` | **New:** Shot-to-shot color transfer using CIE L\*a\*b\* statistics |
| **◎ Radiance LUT Apply** | `nodes_color.py` | High-precision trilinear LUT applicator (.cube / .3dl) |
| **◎ Radiance Workflow Presets**| `nodes_color.py` | Scene-linear setups for ACES, Log, and Display transforms |
| **◎ Radiance Log Curve Decode**| `nodes_color.py` | Precise decoding for LogC3, LogC4, S-Log3, REDLog, etc. |
| **◎ Radiance VAE Encode Pro** | `nodes_hdr.py` | 32-bit floating point VAE encoding with tiling support |
| **◎ Radiance VAE Decode Pro** | `nodes_hdr.py` | 32-bit floating point VAE decoding with tiling support |
| **◎ Radiance OCIO Transform** | `nodes_color.py` | Full OpenColorIO v2.x integration |
| **◎ Radiance Color Matrix** | `nodes_color.py` | GPU-accelerated 3x3 and 4x4 matrix operations |

### ◎ Camera & Film FX
| Node | File | Description |
| :--- | :--- | :--- |
| **◎ Radiance White Balance** | `nodes_camera.py` | Accurate Kelvin temperature and tint correction |
| **◎ Radiance Depth of Field** | `nodes_camera.py` | Physically-accurate bokeh synthesis with custom shapes |
| **◎ Radiance Motion Blur** | `nodes_camera.py` | Directional and radial 32-bit vector blur |
| **◎ Radiance Film Grain** | `nodes_filmgrain.py` | Scanned film grain synthesis with motion coherence |
| **◎ Radiance 32-bit Denoise** | `nodes_denoise.py` | Edge-preserving selective bilateral filtering |
| **◎ Radiance Rolling Shutter**| `nodes_camera.py` | Simulation of CMOS rolling shutter artifacts |

### ◎ Video & Temporal
| Node | File | Description |
| :--- | :--- | :--- |
| **◎ Radiance Read (Video)** | `nodes_io.py` | Professional high-bitrate video decoding (ProRes, DNx) |
| **◎ Radiance Read (Sequence)**| `nodes_io.py` | EXR/DPX/PNG frame sequence loader |
| **◎ Radiance Temporal Smooth**| `nodes_temporal.py` | Motion-aware flicker reduction for AI video |
| **◎ Radiance Flicker Analyze**| `nodes_temporal.py` | Measurement tool for frame-to-frame intensity variance |

### ◎ Analysis & QC
| Node | File | Description |
| :--- | :--- | :--- |
| **◎ Radiance QC Pro** | `nodes_qc.py` | Automated gamut, clipping, and noise violation detection |
| **◎ Radiance Waveform** | `nodes_scopes.py` | High-speed GPU-rendered RGB/Luma Parade |
| **◎ Radiance Vectorscope** | `nodes_scopes.py` | Professional hue/saturation phase monitoring |
| **◎ Radiance False Color** | `nodes_scopes.py` | Stop-accurate exposure visualization (Zebra/IRE) |
| **◎ Radiance Depth Map** | `nodes_depth.py` | Advanced Z-Depth generation for post-compositing |

### ◎ Pipeline & I/O
| Node | File | Description |
| :--- | :--- | :--- |
| **◎ Radiance Write** | `nodes_io.py` | Production-standard file writing (v01, v02, etc.) |
| **◎ Radiance Save EXR/HDR** | `nodes_exr.py` | Multi-channel 32-bit scanline/PIZ OpenEXR export |
| **◎ Radiance Nuke Bridge** | `nodes_nuke.py` | **Featured:** Live 2-way image transfer with The Foundry Nuke |
| **◎ Radiance Reroute+** | `nodes_layout.py` | Smart compact routing with auto-labeling |
| **◎ Radiance Metadata Overlay**| `nodes_overlay.py` | Custom bake-in for burn-ins and slates |
| **◎ Radiance DNA Reader** | `nodes_dna.py` | Read custom project/shot metadata from .dna files |

### ◎ Logic & AI
| Node | File | Description |
| :--- | :--- | :--- |
| **◎ Radiance Sampler Pro** | `nodes_sampler.py` | Flux-optimized sampling with phase-shift logic |
| **◎ Cinematic Encoder** | `nodes_prompt.py` | **Featured:** Turn simple prompts into professional lensing |
| **◎ Radiance AI Upscale** | `nodes_upscale.py` | 32-bit latent-preserving super-resolution |
| **◎ Radiance Highlight Synth**| `nodes_hdr.py` | AI-driven reconstruction of clipped HDR textures |

---

## Installation

### Option 1: ComfyUI Manager *(Recommended)*
1. Open **ComfyUI Manager**
2. Search for **radiance**
3. Click **Install**

### Option 2: Manual (Git Clone)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fxtdstudios/radiance.git
cd radiance
```

Then install dependencies for your OS:

| OS | Command |
|---|---|
| ◎ **Windows** | `pip install -r requirements_windows.txt` |
| ◎ **Linux** | `pip install -r requirements_linux.txt` |
| ◎ **macOS (Apple Silicon)** | `pip install -r requirements_mac_silicon.txt` |

> **Windows note:** If OpenEXR fails to install, you need [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or a pre-built wheel from [Gohlke's page](https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr).

> **Linux note:** Run `sudo apt-get install libopenexr-dev` before pip if the build fails.

### Dependencies
```
opencv-python   Pillow   imageio   OpenEXR   Imath
opencolorio   colour-science   transformers   scipy
```
`torch`, `numpy`, `safetensors` are bundled with ComfyUI — no separate install needed.

---

## Quick Start

### Basic HDR Workflow
```
Image → ImageToFloat32 → Float32ColorCorrect → HDRToneMap → SaveImageEXR
```

### Film Look Workflow
```
Image → FXTDFilmGrain → FXTDLensEffects → FXTDFilmLook → Output
```

### Grade Matching (New in v2.2)
```
SourceShot  ──┐
              ├─→ RadianceGradeMatch → matched_image
ReferenceShot ┘                     → grade_info → ApplyGradeInfo(OtherShot)
```

### LUT Pipeline (New in v2.2)
```
RadianceGrade → grade_info → RadianceLUTBake → grade.cube (share with Resolve)
any_image → RadianceLUTApply(grade.cube) → graded_image
```

### Video Flicker Fix (New in v2.2)
```
generated_video → RadianceFlickerAnalyze → flicker_report (before)
generated_video → RadianceTemporalSmooth(alpha=0.6) → clean_video
```

---

## Viewer Shortcuts

| Key | Action | Key | Action |
|---|---|---|---|
| `F` | Fit to view | `H` | GPU Histogram |
| `1` | 100% zoom | `W` | Waveform |
| `R` / `G` / `B` | View channel | `V` | Vectorscope |
| `L` | Luminance view | `E` | False Color |
| `C` | RGB (color) | `Z` | Zebra pattern |
| `A` | A/B comparison | `P` | Fullscreen |
| `T` | Export EXR | ◎ | Export PNG / EXR / CDL |

---

## Node Reference

**Total: 79 nodes** across 15 categories

### ◎ Color Grading

| Node | Description |
|---|---|
| **◎ Radiance Grade** | Per-channel Lift/Gamma/Gain/Offset with 13 cinematic presets, LAB match grading, and JSON export |
| **◎ Radiance Grade Match** | Shot-to-shot color match via CIE L\*a\*b\* mean/std transfer |
| **◎ Apply Grade Info** | Apply a saved `grade_info` JSON to any image |
| **◎ Radiance LUT Bake** | Bake grade parameters to a 33³ .cube LUT file |
| **◎ Radiance LUT Apply** | Apply any .cube LUT with trilinear interpolation + blend strength |
| Float32 Color Correct | Exposure, contrast, saturation, highlight/shadow in fp32 |
| HDR Tone Map | 10+ operators: ACES, Filmic, AgX, Reinhard, Hable |
| GPU HDR Tone Map | GPU-accelerated tone mapping |
| Color Space Convert | sRGB, ACEScg, Rec.2020, XYZ, DWG, AWG4 |
| ACES 2.0 Output Transform | Full ACES 2.0 pipeline |
| OCIO Color Transform | OpenColorIO integration |
| Log Curve Encode/Decode | ARRI LogC3/4, V-Log, ACEScct, S-Log3, Canon Log3, RED Log3G10 |

### ◎ Scopes & Analysis

| Node | Description |
|---|---|
| **◎ Radiance False Color** | 7-zone false-color baked as IMAGE — Crushed / Under / Dim / Correct / Over / Hot / Clipped |
| **◎ Radiance Waveform** | RGB Overlay, Luma, and Parade waveform monitor |
| **◎ Radiance Vectorscope** | YUV vectorscope with skin-tone line and graticule |
| **◎ Radiance QC Pro** | Automated defect detection (clipping, banding, noise, gamut) |
| HDR Histogram | HDR-aware histogram analysis |

### ◎️ Video & Temporal

| Node | Description |
|---|---|
| **◎ Radiance Temporal Smooth** | Per-pixel EMA flicker reduction with motion-aware masking |
| **◎ Radiance Flicker Analyze** | Frame-to-frame flicker index measurement and JSON report |
| Radiance Read Video | Read MP4/MOV/AVI with log color transform |
| Radiance Read Sequence | EXR/PNG/TIFF sequence reader |
| Radiance Write Video | Export ProRes 4444, H.264, H.265, EXR sequence |

### ◎ Compositing & Overlay

| Node | Description |
|---|---|
| **◎ Radiance Blend Composite** | 8 blend modes + optional MASK — Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, Divide |
| **◎ Radiance Metadata Overlay** | Burn-in timecode, shot info, and QC reports (slate/dailies style) |

### ◎ Film & Camera

| Node | Description |
|---|---|
| FXTD Film Grain | Photorealistic grain — 30+ camera sensors, 20+ film stocks |
| FXTD Lens Effects | Halation, bloom, chromatic aberration |
| FXTD Film Look | Complete film emulation in one node |
| FXTD Pro Film Effects | All-in-one: grain + lens + color |
| Camera Simulation | Sensor characteristics, white balance, presets |
| Radiance Camera | Real-world camera physics (ARRI, RED, Sony, Blackmagic) |

**Camera presets:** ARRI Alexa 35, RED V-Raptor, Sony Venice 2, Blackmagic URSA, Canon C300, Panasonic Varicam

**Film stock presets:** Kodak Vision3 250D/500T, Kodak Portra 400/800, Fuji Eterna, Ilford HP5, CineStill 800T

### ◎ HDR & Dynamic Range

| Node | Description |
|---|---|
| Image to Float32 | Convert any image to 32-bit float |
| HDR Expand Dynamic Range | Synthesize extended dynamic range |
| HDR Exposure Blend | Mertens fusion, Laplacian pyramid blending |
| Shadow/Highlight Recovery | Lift shadows, recover highlights |
| Radiance Highlight Synthesis | Reconstruct clipped highlights from neighbours |

### ◎ Export / Import

| Node | Description |
|---|---|
| Save EXR (32-bit) | OpenEXR export with metadata |
| Load EXR | OpenEXR import |
| Save 16-bit PNG/TIFF | 16-bit export with dithering |
| Save HDRI | HDR environment map |

### ◎ Upscale

| Node | Description |
|---|---|
| FXTD Pro Upscale | Lanczos / Mitchell — artifact-free |
| FXTD Upscale By Size | Target resolution with AR lock |
| FXTD Upscale Tiled | Tile-based for very large images |
| FXTD AI Upscale | RealESRGAN, SUPIR |
| FXTD Sharpen 32-bit | GPU sharpening, no HDR clamping |
| FXTD Downscale 32-bit | Anti-aliased downscaling |
| FXTD Bit Depth Convert | Dithered 8/16/32-bit conversion |

### ◎ Depth & AI

| Node | Description |
|---|---|
| Depth Anything V2 | Monocular depth estimation (auto-downloads from HuggingFace) |
| Depth Map Visualize | Normalize and colorize depth maps |

### ◎ Prompt & Generation

| Node | Description |
|---|---|
| Radiance Sampler Pro | Flux-optimized sampler with phase-shift, CFG++, PAG, dynamic guidance |
| Cinematic Prompt Machine | Prompt generator with real-world camera/lens physics |
| Simple to Flux | Convert natural language prompts for Flux models |
| Radiance DNA | Style DNA system for consistent look generation |

### ◎ Project & Pipeline

| Node | Description |
|---|---|
| Radiance Unified Loader | Load Checkpoint + CLIP + VAE in one node |
| Radiance Project Settings | Define root / sequence / shot / version |
| Radiance Save EXR (Project) | Auto-path EXR output from project settings |
| Radiance Studio | Nuke-style backdrops and node organization |
| Nuke Link | Send/receive images to/from Nuke |

### ◎ Filter & Denoise

| Node | Description |
|---|---|
| **◎ Radiance Denoise** | Edge-preserving bilateral filter for 32-bit float images |
| FXTD Gaussian Blur | High-precision Gaussian blur with HDR support |

### ◎ Layout

| Node | Description |
|---|---|
| **◎ Radiance Reroute** | Industry-standard reroute node |
| **◎ Radiance Reroute+** | Advanced reroute with labels and auto-type color detection |

### ◎ Masking

| Node | Description |
|---|---|
| **◎ Radiance Load Image** | Enhanced image loader with integrated soft-brush mask editor |

### ◎ Utilities

| Node | Description |
|---|---|
| **◎ Show Text (Radiance)** | Displays any input data (STRING, INT, DICT) as text on the node UI |
| Radiance Workspace | Define workspace-wide variables and settings |

---

## GPU Acceleration

All critical operations are GPU-accelerated (CUDA + Apple MPS) with automatic CPU fallback:

| Operation | GPU Speedup |
|---|---|
| Tone Mapping | 20–50× |
| Log Curves | 20× |
| Color Grading | 25× |
| Histogram | 10× |
| Waveform | 10× |
| LUT Application | 10× |
| Gaussian Blur | 20× |

---

## AI Models (Auto-Downloaded)

| Model | Purpose | Source |
|---|---|---|
| Depth Anything V2 (S/B/L) | Monocular depth | HuggingFace `LiheYoung/depth-anything` |
| RealESRGAN x2/x4 | AI upscaling | HuggingFace |
| SUPIR v0Q/v0F | High-fidelity restoration | HuggingFace |

---

## Libraries

| Library | Purpose |
|---|---|
| [OpenColorIO](https://opencolorio.org/) | Industry color management |
| [OpenEXR](https://openexr.com/) | Professional HDR format |
| [colour-science](https://www.colour-science.org/) | Color math and transforms |
| [PyTorch](https://pytorch.org/) | GPU acceleration |
| [OpenCV](https://opencv.org/) | Image processing |
| [Pillow](https://pillow.readthedocs.io/) | Image I/O |

---

## Credits

**Special Thanks**
- To all our dedicated beta testers and the ComfyUI community for their invaluable bug reports, feedback, and support in making Radiance v2.2 possible.


**Color Science**
- ACES Color System — Academy of Motion Picture Arts and Sciences
- Filmic Tone Mapping — John Hable (Uncharted 2), Stephen Hill, Krzysztof Narkowicz
- AgX Color Transform — Troy Sobotka
- Reinhard Tone Mapping — Erik Reinhard
- 3D LUT format — Adobe / DaVinci standard .cube

**Camera & Film Data**
- Log curves: ARRI (LogC3/C4), RED (Log3G10), Panasonic (V-Log), Canon (Log3)
- Film stocks: Kodak, Fujifilm reference data
- Lens presets: Panavision C-Series, Cooke Anamorphic, Zeiss Supreme Prime

**AI Models**
- Depth Anything V2 — HuggingFace Transformers
- RealESRGAN — Xintao Wang et al.
- SUPIR — Fanghua Yu et al.

---

## Documentation & Support

- ◎ **[Official Docs](https://radiance.fxtd.org)**
- ◎ **[Bug Tracker](https://radiance.fxtd.or/faq.html)**
- ◎ **[FXTD Studios](https://www.linkedin.com/company/fxtdstudios)**

---

## License

GPL-3.0 — See [LICENSE](LICENSE)

© FXTD Studios

---

[↑ Back to top](#radiance--professional-hdr-suite-for-comfyui)
