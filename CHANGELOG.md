# Changelog

All notable changes to FXTD Radiance will be documented in this file.

## [2.1.0] - 2026-02-26

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
