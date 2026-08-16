[← Back to Radiance docs](../README.md)

# Color

Grading, colour-space conversion, CDL, LUTs, curves, and QC. Everything here
assumes you know which space your image is in — the nodes will not guess, and a
transform applied to the wrong space is the single most common way to lose an
afternoon.

## Typical graph

```text
Read (decode to scene-linear) → Grade / CDL / Curves → Colorspace Convert (to display) → Viewer
```

## Before you use these nodes

- **Grade in scene-linear.** Exposure is a multiply, and a multiply only means
  "stops" in a linear space. Grading a gamma-encoded image gives you contrast
  changes you did not ask for.
- **Colorspace Convert is a primaries + transfer change**, not a look. It will
  not make an image "correct"; it moves it between two spaces you name.
- **CDL is ASC-standard** — slope, offset, power, saturation — and round-trips
  through `.cc`/`.cdl` files for handoff to a grading system.
- **QC before you deliver.** The QC node reports clipping, out-of-gamut pixels
  and banding risk. Policy Guard turns a house standard into a pass/fail.

## Known limitation

`chromatic_adaptation` on Colorspace Convert has no effect — the D60↔D65
adaptation is baked into the precomputed conversion matrices, and every option
on the widget produces identical output. The node logs a warning if you set it
to anything other than the default.

## Nodes in this section (13)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [Apply Grade Info](#apply-grade-info) | `RadianceApplyGradeInfo` | Apply a saved grade_info JSON to any image. |
| [CDL](#cdl) | `RadianceCDLTransform` | Apply an ASC CDL (Slope/Offset/Power/Saturation) colour transform. |
| [CDL Export](#cdl-export) | `RadianceCDLExport` | Export current CDL values to an ASC-compliant .cdl or .cc file. |
| [CDL Import](#cdl-import) | `RadianceCDLImport` | Import an ASC CDL (.cdl / .cc / .ccc) file into pipeline metadata. |
| [ColorLookup](#colorlookup) | `RadianceCurves` | RGB and luminance spline curve grading with customisable control points. |
| [Grade](#grade) | `RadianceGrade` | Professional color grading with per-channel Lift/Gamma/Gain/Offset, Contrast, Saturation, cinematic presets, optional grade matching, and JSON preset file loading. |
| [Grade Match](#grade-match) | `RadianceGradeMatch` | Match source image color statistics to a reference image using LAB mean/std matching. |
| [HueCorrect](#huecorrect) | `RadianceHueCurves` | Per-hue selective colour adjustment using spline curves. |
| [LUT](#lut) | `RadianceLUTApply` | Apply a 3D LUT (.cube) with trilinear or tetrahedral interpolation |
| [OCIO ColorSpace](#ocio-colorspace) | `RadianceColorSpaceConvert` | Convert images between named colour spaces. |
| [OCIO Context](#ocio-context) | `RadianceOCIOContext` | Set OpenColorIO context variables for environment-aware transforms. |
| [RadianceLUTBlend](#radiancelutblend) | `RadianceLUTBlend` | Blend two LUTs with various blend modes for creative color grading. |
| [White Balance](#white-balance) | `RadianceWhiteBalance` | Adjust white balance using a reference neutral or colour temperature. |

---

## Apply Grade Info

**Node key:** `RadianceApplyGradeInfo`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/grade.py`  

Apply a saved grade_info JSON to any image.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `grade_info` | Yes | `STRING` |  |  |  |
| `strength` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.05 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `grade_info` | `STRING` |  |

---

## CDL

**Node key:** `RadianceCDLTransform`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/cdl.py`  

Apply an ASC CDL (Slope/Offset/Power/Saturation) colour transform.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `slope_r` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.01 | Red channel slope (gain). 1.0 = unity. |
| `slope_g` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.01 | Green channel slope (gain). 1.0 = unity. |
| `slope_b` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.01 | Blue channel slope (gain). 1.0 = unity. |
| `offset_r` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 | Red channel offset. 0.0 = no shift. |
| `offset_g` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 | Green channel offset. 0.0 = no shift. |
| `offset_b` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 | Blue channel offset. 0.0 = no shift. |
| `power_r` | Yes | `FLOAT` | `1.0` | 0.01 – 4.0, step 0.01 | Red channel power (gamma). |
| `power_g` | Yes | `FLOAT` | `1.0` | 0.01 – 4.0, step 0.01 | Green channel power (gamma). |
| `power_b` | Yes | `FLOAT` | `1.0` | 0.01 – 4.0, step 0.01 | Blue channel power (gamma). |
| `saturation` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.01 | Global saturation. 1.0 = unity. |
| `cdl_data` | No | `STRING` |  |  | JSON CDL data from RadianceCDLImport. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `cdl_info` | `STRING` |  |

---

## CDL Export

**Node key:** `RadianceCDLExport`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/cdl.py`  
**Output node** — runs even with nothing connected downstream.  

Export current CDL values to an ASC-compliant .cdl or .cc file.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `file_path` | Yes | `STRING` | `grading/shot_01_output.cdl` |  | Destination .cdl path. A relative path is written under ComfyUI's output/ folder; absolute paths are used as given. |
| `slope_r` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.001 |  |
| `slope_g` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.001 |  |
| `slope_b` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.001 |  |
| `offset_r` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.0001 |  |
| `offset_g` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.0001 |  |
| `offset_b` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.0001 |  |
| `power_r` | Yes | `FLOAT` | `1.0` | 0.01 – 4.0, step 0.001 |  |
| `power_g` | Yes | `FLOAT` | `1.0` | 0.01 – 4.0, step 0.001 |  |
| `power_b` | Yes | `FLOAT` | `1.0` | 0.01 – 4.0, step 0.001 |  |
| `saturation` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.001 |  |
| `cdl_data` | No | `STRING` |  |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `STRING` |  |

---

## CDL Import

**Node key:** `RadianceCDLImport`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/cdl.py`  

Import an ASC CDL (.cdl / .cc / .ccc) file into pipeline metadata.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `file_path` | Yes | `STRING` | `grading/shot_01.cdl` |  | Path to a .cdl, .cc, or .ccc file. A relative path is looked for in ComfyUI's input/ then output/ folder; absolute paths are used as given. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `cdl_data` | `STRING` |  |
| `slope_r` | `FLOAT` |  |
| `slope_g` | `FLOAT` |  |
| `slope_b` | `FLOAT` |  |
| `offset_r` | `FLOAT` |  |
| `offset_g` | `FLOAT` |  |
| `offset_b` | `FLOAT` |  |
| `power_r` | `FLOAT` |  |
| `power_g` | `FLOAT` |  |
| `power_b` | `FLOAT` |  |
| `saturation` | `FLOAT` |  |

---

## ColorLookup

**Node key:** `RadianceCurves`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/curves.py`  

RGB and luminance spline curve grading with customisable control points.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `master` | Yes | `STRING` | `[[0.0,0.0],[0.25,0.25],[0.5,0.5],[0.75,0.75],[1.0,1.0]]` |  |  |
| `red` | Yes | `STRING` | `[[0.0,0.0],[0.25,0.25],[0.5,0.5],[0.75,0.75],[1.0,1.0]]` |  |  |
| `green` | Yes | `STRING` | `[[0.0,0.0],[0.25,0.25],[0.5,0.5],[0.75,0.75],[1.0,1.0]]` |  |  |
| `blue` | Yes | `STRING` | `[[0.0,0.0],[0.25,0.25],[0.5,0.5],[0.75,0.75],[1.0,1.0]]` |  |  |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `grade_info_in` | No | `STRING` |  |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `grade_info` | `STRING` |  |

---

## Grade

**Node key:** `RadianceGrade`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/grade.py`  

Professional color grading with per-channel Lift/Gamma/Gain/Offset, Contrast, Saturation, cinematic presets, optional grade matching, and JSON preset file loading.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | Input image to grade. |
| `preset` | Yes | choice of `None (Custom)`, `Cinematic Teal & Orange`, `Bleach Bypass`, `Cross Process`, `Film Noir`, `Vintage Film`, `Cool Blue Hour`, `Golden Hour`, … (+5 more) | `None (Custom)` |  |  |
| `preset_strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.05 |  |
| `reference_image` | No | `IMAGE` |  |  | Optional reference image for automatic grade matching. |
| `match_strength` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.05 |  |
| `preset_file` | No | `STRING` | `` |  |  |
| `lift_r` | No | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 |  |
| `lift_g` | No | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 |  |
| `lift_b` | No | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 |  |
| `gamma_r` | No | `FLOAT` | `1.0` | 0.01 – 5.0, step 0.001 |  |
| `gamma_g` | No | `FLOAT` | `1.0` | 0.01 – 5.0, step 0.001 |  |
| `gamma_b` | No | `FLOAT` | `1.0` | 0.01 – 5.0, step 0.001 |  |
| `gain_r` | No | `FLOAT` | `1.0` | 0.0 – 5.0, step 0.001 |  |
| `gain_g` | No | `FLOAT` | `1.0` | 0.0 – 5.0, step 0.001 |  |
| `gain_b` | No | `FLOAT` | `1.0` | 0.0 – 5.0, step 0.001 |  |
| `offset_r` | No | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 |  |
| `offset_g` | No | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 |  |
| `offset_b` | No | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.001 |  |
| `contrast` | No | `FLOAT` | `1.0` | 0.0 – 3.0, step 0.01 |  |
| `pivot` | No | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 |  |
| `saturation` | No | `FLOAT` | `1.0` | 0.0 – 3.0, step 0.01 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `grade_info` | `STRING` |  |

---

## Grade Match

**Node key:** `RadianceGradeMatch`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/grade.py`  

Match source image color statistics to a reference image using LAB mean/std matching.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `source` | Yes | `IMAGE` |  |  | Image to be matched. |
| `reference` | Yes | `IMAGE` |  |  | Target image. |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.05 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `matched_image` | `IMAGE` |  |
| `grade_info` | `STRING` |  |

---

## HueCorrect

**Node key:** `RadianceHueCurves`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/curves.py`  

Per-hue selective colour adjustment using spline curves.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `mode` | Yes | choice of `Hue vs Hue`, `Hue vs Saturation`, `Hue vs Luminance` | `Hue vs Hue` |  |  |
| `control_points` | Yes | `STRING` | `[[0.0,0.0],[0.167,0.0],[0.333,0.0],[0.5,0.0],[0.667,0.0],[0.833,0.0],[1.0,0.0]]` |  |  |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 |  |
| `grade_info` | No | `STRING` |  |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## LUT

**Node key:** `RadianceLUTApply`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `color/lut.py`  

Apply a 3D LUT (.cube) with trilinear or tetrahedral interpolation. Supports log-space input and HDR (unclamped) output.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `lut_file` | Yes | choice of `No LUTs found` | `No LUTs found` |  |  |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `log_space` | Yes | `BOOLEAN` | `False` |  |  |
| `log_encoding` | No | choice of `Log10`, `Log2`, `Natural Log (Ln)` | `Log10` |  |  |
| `clamp_output` | No | `BOOLEAN` | `False` |  | Clamp to 0-1. Disable for HDR. |
| `interpolation` | No | choice of `Trilinear`, `Tetrahedral` | `Trilinear` |  | Tetrahedral is more accurate but slightly slower |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## OCIO ColorSpace

**Node key:** `RadianceColorSpaceConvert`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/colorspace.py`  

Convert images between named colour spaces.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `src_space` | Yes | choice of `Linear sRGB (D65)`, `ACEScg`, `ACEScc`, `ACEScct`, `sRGB (OETF encoded)`, `Rec.709 (OETF encoded)`, `Rec.709 / BT.1886`, `LogC3 (ARRI EI800)`, … (+8 more) | `Linear sRGB (D65)` |  |  |
| `dst_space` | Yes | choice of `Linear sRGB (D65)`, `ACEScg`, `ACEScc`, `ACEScct`, `sRGB (OETF encoded)`, `Rec.709 (OETF encoded)`, `Rec.709 / BT.1886`, `LogC3 (ARRI EI800)`, … (+8 more) | `ACEScg` |  |  |
| `direction` | Yes | choice of `Forward`, `Inverse` | `Forward` |  |  |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `grade_info_in` | No | `STRING` |  |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `grade_info` | `STRING` |  |

---

## OCIO Context

**Node key:** `RadianceOCIOContext`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/ocio.py`  

Set OpenColorIO context variables for environment-aware transforms.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `config_path` | Yes | `STRING` | `C:/ACES/config.ocio` |  |  |
| `working_space` | Yes | `STRING` | `ACES - ACEScg` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `ocio_context` | `RADIANCE_OCIO` |  |

---

## RadianceLUTBlend

**Node key:** `RadianceLUTBlend`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `color/lut.py`  

Blend two LUTs with various blend modes for creative color grading.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `lut_a` | Yes | choice of `No LUTs found` | `No LUTs found` |  |  |
| `lut_b` | Yes | choice of `No LUTs found` | `No LUTs found` |  |  |
| `blend_factor` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | 0.0 = LUT A only, 1.0 = LUT B only |
| `blend_mode` | Yes | choice of `Linear`, `Luminosity`, `Saturation`, `Hue` | `Linear` |  |  |
| `strength` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `clamp_output` | No | `BOOLEAN` | `False` |  | Clamp to 0-1. Disable for HDR. |
| `log_space` | No | `BOOLEAN` | `False` |  | Decode log-encoded input before applying LUTs. |
| `log_encoding` | No | choice of `Log10`, `Log2`, `Natural Log (Ln)` | `Log10` |  |  |
| `interpolation` | No | choice of `Trilinear`, `Tetrahedral` | `Trilinear` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## White Balance

**Node key:** `RadianceWhiteBalance`  
**Menu:** `FXTD STUDIOS/Radiance/Color`  
**Source:** `nodes/color/colorspace.py`  

Adjust white balance using a reference neutral or colour temperature.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `mode` | Yes | choice of `Temperature / Tint`, `Illuminant Adapt`, `Manual RGB Gain` | `Temperature / Tint` |  |  |
| `preset` | Yes | choice of `Manual`, `Daylight (5500K)`, `Tungsten (3200K)`, `Fluorescent (4200K)`, `Flash (6000K)`, `Shade (7500K)` | `Manual` |  |  |
| `temperature` | Yes | `FLOAT` | `6500.0` | 1667.0 – 25000.0, step 50.0 |  |
| `tint` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.005 |  |
| `src_illuminant` | Yes | choice of `D50`, `D55`, `D60`, `D65`, `D75`, `A`, `B`, `C`, … (+1 more) | `D65` |  |  |
| `dst_illuminant` | Yes | choice of `D50`, `D55`, `D60`, `D65`, `D75`, `A`, `B`, `C`, … (+1 more) | `D50` |  |  |
| `gain_r` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.001 |  |
| `gain_g` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.001 |  |
| `gain_b` | Yes | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.001 |  |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `grade_info_in` | No | `STRING` |  |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `grade_info` | `STRING` |  |

---
