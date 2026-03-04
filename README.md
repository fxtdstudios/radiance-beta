<div align="center">

![RADIANCE](docs/RADIANCE.png)

![Version](https://img.shields.io/badge/version-2.1.0-blue)
![License](https://img.shields.io/badge/license-GPL--3.0-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-purple)
![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20Apple%20MPS-orange)
![Nodes](https://img.shields.io/badge/nodes-76-brightgreen)

</div>

# Radiance — Professional HDR Suite for ComfyUI

*Industry-grade color grading, film effects, HDR processing, and interactive viewing — built for VFX pipelines.*

[Installation](#installation) · [Node Reference](#node-reference) · [Quick Start](#quick-start) · [Viewer Shortcuts](#viewer-shortcuts) · [Documentation](https://fxtd.org/radiance_docs/) · [What's New](#whats-new-v210)

---

## What's New — v2.1.0

### ◎ Radiance Viewer Overhaul
- **CDL Export / Import** — save/load grade as ASC CDL v1.2 XML (Nuke, Resolve, OCIO compatible) via ◎ menu
- **GPU Histogram** — press `H` for a GPU-rendered 256-bin histogram with HDR log scale and SDR ceiling marker
- **fp32 Pick Buffer** — color picker now reads true scene-linear HDR values from a zlib-compressed `.rpick` sidecar (accurate EV readout at cursor)
- **LRU Frame Cache** — up to 8 GPU textures cached; zero re-uploads when scrubbing through frames
- **OES_texture_half_float** — explicit WebGL half-float upload path; HDR textures are now uploaded in fp16 instead of fp32
- **Linear False Color / Zebra** — `E` and `Z` now evaluate in scene-linear space (pre-OETF) for stop-accurate thresholds
- **Display-P3 / HDR Detection** — viewer detects P3 and Rec.2020 monitors and configures canvas color space automatically

### ◎ New Color Nodes (4)
| Node | File | What it does |
|---|---|---|
| **◎ Radiance Grade Match** | `nodes_grade.py` | Match source image color statistics to a reference using CIE L\*a\*b\* mean/std |
| **◎ Apply Grade Info** | `nodes_grade.py` | Replay a `grade_info` JSON string onto any image — closes the grade roundtrip |
| **◎ Radiance LUT Bake** | `nodes_lut.py` | Bake grade params into a 33³ `.cube` LUT (Resolve / Nuke / Premiere ready) |
| **◎ Radiance LUT Apply** | `nodes_lut.py` | Load any `.cube` LUT and apply via trilinear interpolation |

### ◎ New Video Nodes (2)
| Node | File | What it does |
|---|---|---|
| **◎ Radiance Temporal Smooth** | `nodes_temporal.py` | Per-pixel EMA across batch frames — removes flicker and AI grain; motion-aware mask preserves edges |
| **◎ Radiance Flicker Analyze** | `nodes_temporal.py` | Measures frame-to-frame flicker index; outputs JSON report for QC benchmarking |

### ◎ New Analysis + Compositing Nodes (3)
| Node | File | What it does |
|---|---|---|
| **◎ Radiance False Color** | `nodes_scopes.py` | 7-zone false-color overlay baked as IMAGE — pipeline-usable without opening the viewer |
| **◎ Radiance Blend Composite** | `nodes_overlay.py` | 8 blend modes (Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, Divide) with optional MASK |

### ◎ Improvements
- **Radiance Grade** — added `reference_image` match grading, external JSON preset file, and full JSON `grade_info` output
- **Sampler Pro v3.6** — fixed 6 critical bugs including sigma schedule off-by-one (BUG-28), guidance deep-copy overhead, and double noise injection
- **Requirements** — added `Pillow` (was used but undeclared), removed `scipy` (was declared but unused)

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
opencolorio   colour-science   transformers
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

### Grade Matching (New in v2.1)
```
SourceShot  ──┐
              ├─→ RadianceGradeMatch → matched_image
ReferenceShot ┘                     → grade_info → ApplyGradeInfo(OtherShot)
```

### LUT Pipeline (New in v2.1)
```
RadianceGrade → grade_info → RadianceLUTBake → grade.cube (share with Resolve)
any_image → RadianceLUTApply(grade.cube) → graded_image
```

### Video Flicker Fix (New in v2.1)
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

**Total: 76 nodes** across 12 categories

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
- To all our dedicated beta testers and the ComfyUI community for their invaluable bug reports, feedback, and support in making Radiance v2.1.0 possible.


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

- ◎ **[Official Docs](https://fxtd.org/radiance_docs/)**
- ◎ **[Bug Tracker](https://fxtd.org/radiance_docs/faq.html)**
- ◎ **[FXTD Studios](https://www.linkedin.com/company/fxtdstudios)**

---

## License

GPL-3.0 — See [LICENSE](LICENSE)

© FXTD Studios

---

[↑ Back to top](#radiance--professional-hdr-suite-for-comfyui)