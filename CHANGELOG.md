# Changelog

All notable changes to FXTD Radiance will be documented in this file.

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
