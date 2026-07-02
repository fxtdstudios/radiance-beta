[← Back to Radiance docs](../README.md)

# Color

Primary grading, CDL exchange, curves, white balance, color-space conversion, OCIO context, and QC checks.

## Typical workflow

```text
Radiance Read -> Color Space Convert / OCIO Context -> Grade / Curves / CDL -> QC -> Write
```

## Before you use these nodes

- Choose source, working, and output color spaces deliberately; do not rely on viewport appearance.
- Use CDL import/export for interchange with grading and comp tools.
- Run QC after grade and before final output when clipping, bit depth, or broadcast limits matter.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Radiance CDL Transform](#radiance-cdl-transform) | `RadianceCDLTransform` | Adjusts image color, tone, or grading metadata in a production-friendly way. |
| [◎ Radiance CDL Import](#radiance-cdl-import) | `RadianceCDLImport` | Adjusts image color, tone, or grading metadata in a production-friendly way. |
| [◎ Radiance CDL Export](#radiance-cdl-export) | `RadianceCDLExport` | Writes or hands off the current result to a file, folder, preview, or DCC destination. |
| [◎ Radiance White Balance](#radiance-white-balance) | `RadianceWhiteBalance` | Adjusts image color, tone, or grading metadata in a production-friendly way. |
| [◎ Radiance Colorspace Convert](#radiance-colorspace-convert) | `RadianceColorSpaceConvert` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Radiance ACES Transform](#radiance-aces-transform) | `RadianceACESTransform` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Radiance Hue Curves](#radiance-hue-curves) | `RadianceHueCurves` | Adjusts image color, tone, or grading metadata in a production-friendly way. |
| [◎ Radiance Curves](#radiance-curves) | `RadianceCurves` | Adjusts image color, tone, or grading metadata in a production-friendly way. |
| [◎ Radiance Grade](#radiance-grade) | `RadianceGrade` | Adjusts image color, tone, or grading metadata in a production-friendly way. |
| [◎ Radiance Apply Grade Info](#radiance-apply-grade-info) | `RadianceApplyGradeInfo` | Adjusts image color, tone, or grading metadata in a production-friendly way. |
| [◎ Radiance Grade Match](#radiance-grade-match) | `RadianceGradeMatch` | Adjusts image color, tone, or grading metadata in a production-friendly way. |
| [◎ Radiance OCIO Context](#radiance-ocio-context) | `RadianceOCIOContext` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Radiance QC](#radiance-qc) | `RadianceQC` | Analyzes the image or workflow state and returns reports that help catch delivery problems. |

## ◎ Radiance CDL Transform

**Internal key:** `RadianceCDLTransform`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/cdl.py`
**Function:** `apply`

### What it does

Adjusts image color, tone, or grading metadata in a production-friendly way.

### When to use it

Use `◎ Radiance CDL Transform` when the graph reaches the CDL Transform step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `slope_r` | Yes | `FLOAT` | `1.0` | Red channel slope (gain). 1.0 = unity. |
| `slope_g` | Yes | `FLOAT` | `1.0` | Green channel slope (gain). 1.0 = unity. |
| `slope_b` | Yes | `FLOAT` | `1.0` | Blue channel slope (gain). 1.0 = unity. |
| `offset_r` | Yes | `FLOAT` | `0.0` | Red channel offset. 0.0 = no shift. |
| `offset_g` | Yes | `FLOAT` | `0.0` | Green channel offset. 0.0 = no shift. |
| `offset_b` | Yes | `FLOAT` | `0.0` | Blue channel offset. 0.0 = no shift. |
| `power_r` | Yes | `FLOAT` | `1.0` | Red channel power (gamma). |
| `power_g` | Yes | `FLOAT` | `1.0` | Green channel power (gamma). |
| `power_b` | Yes | `FLOAT` | `1.0` | Blue channel power (gamma). |
| `saturation` | Yes | `FLOAT` | `1.0` | Global saturation. 1.0 = unity. |
| `cdl_data` | Optional | `STRING` | - | JSON CDL data from RadianceCDLImport. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `cdl_info` | `STRING` | Output produced by the `cdl_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `cdl_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance CDL Import

**Internal key:** `RadianceCDLImport`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/cdl.py`
**Function:** `load`

### What it does

Adjusts image color, tone, or grading metadata in a production-friendly way.

### When to use it

Use `◎ Radiance CDL Import` when the graph reaches the CDL Import step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `file_path` | Yes | `STRING` | `grading/shot_01.cdl` | Path to a .cdl, .cc, or .ccc file. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `cdl_data` | `STRING` | Output produced by the `cdl_data` socket. |
| `slope_r` | `FLOAT` | Output produced by the `slope_r` socket. |
| `slope_g` | `FLOAT` | Output produced by the `slope_g` socket. |
| `slope_b` | `FLOAT` | Output produced by the `slope_b` socket. |
| `offset_r` | `FLOAT` | Output produced by the `offset_r` socket. |
| `offset_g` | `FLOAT` | Output produced by the `offset_g` socket. |
| `offset_b` | `FLOAT` | Output produced by the `offset_b` socket. |
| `power_r` | `FLOAT` | Output produced by the `power_r` socket. |
| `power_g` | `FLOAT` | Output produced by the `power_g` socket. |
| `power_b` | `FLOAT` | Output produced by the `power_b` socket. |
| `saturation` | `FLOAT` | Output produced by the `saturation` socket. |

### Practical notes

- The node returns `cdl_data` (`STRING`), `slope_r` (`FLOAT`), `slope_g` (`FLOAT`), `slope_b` (`FLOAT`), `offset_r` (`FLOAT`), `offset_g` (`FLOAT`), `offset_b` (`FLOAT`), `power_r` (`FLOAT`), `power_g` (`FLOAT`), `power_b` (`FLOAT`), `saturation` (`FLOAT`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance CDL Export

**Internal key:** `RadianceCDLExport`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/cdl.py`
**Function:** `save`

### What it does

Writes or hands off the current result to a file, folder, preview, or DCC destination.

### When to use it

Use `◎ Radiance CDL Export` near the end of the graph after the image, sequence, or metadata is ready for delivery.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `file_path` | Yes | `STRING` | `grading/shot_01_output.cdl` | - |
| `slope_r` | Yes | `FLOAT` | `1.0` | - |
| `slope_g` | Yes | `FLOAT` | `1.0` | - |
| `slope_b` | Yes | `FLOAT` | `1.0` | - |
| `offset_r` | Yes | `FLOAT` | `0.0` | - |
| `offset_g` | Yes | `FLOAT` | `0.0` | - |
| `offset_b` | Yes | `FLOAT` | `0.0` | - |
| `power_r` | Yes | `FLOAT` | `1.0` | - |
| `power_g` | Yes | `FLOAT` | `1.0` | - |
| `power_b` | Yes | `FLOAT` | `1.0` | - |
| `saturation` | Yes | `FLOAT` | `1.0` | - |
| `cdl_data` | Optional | `STRING` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `STRING` | Output produced by the `file_path` socket. |

### Practical notes

- The node returns `file_path` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance White Balance

**Internal key:** `RadianceWhiteBalance`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/colorspace.py`
**Function:** `apply`

### What it does

Adjusts image color, tone, or grading metadata in a production-friendly way.

### When to use it

Use `◎ Radiance White Balance` when the graph reaches the White Balance step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `mode` | Yes | `ENUM: Temperature / Tint, Illuminant Adapt, Manual RGB Gain` | `Temperature / Tint` | - |
| `preset` | Yes | `ENUM: Manual, Daylight (5500K), Tungsten (3200K), Fluorescent (4200K), Flash (6000K), Shade (7500K)` | `Manual` | - |
| `temperature` | Yes | `FLOAT` | `6500.0` | - |
| `tint` | Yes | `FLOAT` | `0.0` | - |
| `src_illuminant` | Yes | `(list(_ILLUMINANT_XY.keys()), {'default': 'D65'})` | - | - |
| `dst_illuminant` | Yes | `(list(_ILLUMINANT_XY.keys()), {'default': 'D50'})` | - | - |
| `gain_r` | Yes | `FLOAT` | `1.0` | - |
| `gain_g` | Yes | `FLOAT` | `1.0` | - |
| `gain_b` | Yes | `FLOAT` | `1.0` | - |
| `strength` | Yes | `FLOAT` | `1.0` | - |
| `grade_info_in` | Optional | `STRING` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `grade_info` | `STRING` | Output produced by the `grade_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `grade_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Colorspace Convert

**Internal key:** `RadianceColorSpaceConvert`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/colorspace.py`
**Function:** `apply`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Radiance Colorspace Convert` when the graph reaches the Colorspace Convert step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `src_space` | Yes | `(cls._COLOR_SPACES, {'default': 'Linear sRGB (D65)'})` | - | - |
| `dst_space` | Yes | `(cls._COLOR_SPACES, {'default': 'ACEScg'})` | - | - |
| `direction` | Yes | `ENUM: Forward, Inverse` | `Forward` | - |
| `strength` | Yes | `FLOAT` | `1.0` | - |
| `grade_info_in` | Optional | `STRING` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `grade_info` | `STRING` | Output produced by the `grade_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `grade_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance ACES Transform

**Internal key:** `RadianceACESTransform`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/colorspace.py`
**Function:** `apply`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Radiance ACES Transform` when the graph reaches the ACES Transform step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | Scene-linear ACEScg image. |
| `odt` | Yes | `(cls._ODT_OPTIONS, {'default': 'sRGB D65'})` | - | - |
| `exposure_offset` | Yes | `FLOAT` | `0.0` | - |
| `peak_nits` | Yes | `FLOAT` | `1000.0` | - |
| `saturation` | Yes | `FLOAT` | `1.0` | - |
| `grade_info_in` | Optional | `STRING` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `aces_info` | `STRING` | Output produced by the `aces_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `aces_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Hue Curves

**Internal key:** `RadianceHueCurves`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/curves.py`
**Function:** `apply`

### What it does

Adjusts image color, tone, or grading metadata in a production-friendly way.

### When to use it

Use `◎ Radiance Hue Curves` when the graph reaches the Hue Curves step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `mode` | Yes | `ENUM: Hue vs Hue, Hue vs Saturation, Hue vs Luminance` | `Hue vs Hue` | - |
| `control_points` | Yes | `STRING` | `[[0.0,0.0],[0.167,0.0],[0.333,0.0],[0.5,0.0],[0.667,0.0],[0.833,0.0],[1.0,0.0]]` | - |
| `strength` | Yes | `FLOAT` | `1.0` | - |
| `grade_info` | Optional | `STRING` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Curves

**Internal key:** `RadianceCurves`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/curves.py`
**Function:** `apply`

### What it does

Adjusts image color, tone, or grading metadata in a production-friendly way.

### When to use it

Use `◎ Radiance Curves` when the graph reaches the Curves step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `master` | Yes | `('STRING', {'default': _default_pts, 'multiline': False})` | - | - |
| `red` | Yes | `('STRING', {'default': _default_pts, 'multiline': False})` | - | - |
| `green` | Yes | `('STRING', {'default': _default_pts, 'multiline': False})` | - | - |
| `blue` | Yes | `('STRING', {'default': _default_pts, 'multiline': False})` | - | - |
| `strength` | Yes | `FLOAT` | `1.0` | - |
| `grade_info_in` | Optional | `STRING` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `grade_info` | `STRING` | Output produced by the `grade_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `grade_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Grade

**Internal key:** `RadianceGrade`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/grade.py`
**Function:** `grade`

### What it does

Adjusts image color, tone, or grading metadata in a production-friendly way.

### When to use it

Use `◎ Radiance Grade` when the graph reaches the Grade step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | Input image to grade. |
| `preset` | Yes | `(preset_names, {'default': 'None (Custom)'})` | - | - |
| `preset_strength` | Yes | `FLOAT` | `1.0` | - |
| `reference_image` | Optional | `IMAGE` | - | Optional reference image for automatic grade matching. |
| `match_strength` | Optional | `FLOAT` | `1.0` | - |
| `preset_file` | Optional | `STRING` | `` | - |
| `lift_r` | Optional | `FLOAT` | `0.0` | - |
| `lift_g` | Optional | `FLOAT` | `0.0` | - |
| `lift_b` | Optional | `FLOAT` | `0.0` | - |
| `gamma_r` | Optional | `FLOAT` | `1.0` | - |
| `gamma_g` | Optional | `FLOAT` | `1.0` | - |
| `gamma_b` | Optional | `FLOAT` | `1.0` | - |
| `gain_r` | Optional | `FLOAT` | `1.0` | - |
| `gain_g` | Optional | `FLOAT` | `1.0` | - |
| `gain_b` | Optional | `FLOAT` | `1.0` | - |
| `offset_r` | Optional | `FLOAT` | `0.0` | - |
| `offset_g` | Optional | `FLOAT` | `0.0` | - |
| `offset_b` | Optional | `FLOAT` | `0.0` | - |
| `contrast` | Optional | `FLOAT` | `1.0` | - |
| `pivot` | Optional | `FLOAT` | `0.5` | - |
| `saturation` | Optional | `FLOAT` | `1.0` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `grade_info` | `STRING` | Output produced by the `grade_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `grade_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Apply Grade Info

**Internal key:** `RadianceApplyGradeInfo`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/grade.py`
**Function:** `apply`

### What it does

Adjusts image color, tone, or grading metadata in a production-friendly way.

### When to use it

Use `◎ Radiance Apply Grade Info` when the graph reaches the Apply Grade Info step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `grade_info` | Yes | `STRING` | - | - |
| `strength` | Optional | `FLOAT` | `1.0` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `grade_info` | `STRING` | Output produced by the `grade_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `grade_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Grade Match

**Internal key:** `RadianceGradeMatch`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/grade.py`
**Function:** `match`

### What it does

Adjusts image color, tone, or grading metadata in a production-friendly way.

### When to use it

Use `◎ Radiance Grade Match` when the graph reaches the Grade Match step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `source` | Yes | `IMAGE` | - | Image to be matched. |
| `reference` | Yes | `IMAGE` | - | Target image. |
| `strength` | Yes | `FLOAT` | `1.0` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `matched_image` | `IMAGE` | Output produced by the `matched_image` socket. |
| `grade_info` | `STRING` | Output produced by the `grade_info` socket. |

### Practical notes

- The node returns `matched_image` (`IMAGE`), `grade_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance OCIO Context

**Internal key:** `RadianceOCIOContext`  
**Category:** `FXTD STUDIOS/Radiance/◎ Color`  
**Source:** `nodes/color/ocio.py`
**Function:** `set_context`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Radiance OCIO Context` when the graph reaches the OCIO Context step in a color workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `config_path` | Yes | `STRING` | `C:/ACES/config.ocio` | - |
| `working_space` | Yes | `STRING` | `ACES - ACEScg` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `ocio_context` | `RADIANCE_OCIO` | Output produced by the `ocio_context` socket. |

### Practical notes

- The node returns `ocio_context` (`RADIANCE_OCIO`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance QC

**Internal key:** `RadianceQC`  
**Category:** `FXTD STUDIOS/Radiance/◎ QC & Debug`  
**Source:** `nodes/color/qc.py`
**Function:** `run`

### What it does

Analyzes the image or workflow state and returns reports that help catch delivery problems.

### When to use it

Use `◎ Radiance QC` before final output or when debugging an unexpected result.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `mode` | Yes | `(cls.MODES, {'default': 'Analyze'})` | - | - |
| `image` | Optional | `IMAGE` | - | - |
| `black_threshold` | Optional | `FLOAT` | `0.0` | - |
| `white_threshold` | Optional | `FLOAT` | `1.0` | - |
| `overlay_opacity` | Optional | `FLOAT` | `0.5` | - |
| `banding_threshold` | Optional | `FLOAT` | `5.0` | - |
| `enable_focus_check` | Optional | `BOOLEAN` | `False` | - |
| `enable_artifacts_check` | Optional | `BOOLEAN` | `True` | - |
| `enable_noise_check` | Optional | `BOOLEAN` | `True` | - |
| `fail_on_errors` | Optional | `BOOLEAN` | `False` | - |
| `qc_report_json` | Optional | `STRING` | - | - |
| `output_path` | Optional | `STRING` | `` | - |
| `filename_prefix` | Optional | `STRING` | `qc_report` | - |
| `export_format` | Optional | `ENUM: json, csv, html, all` | `json` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `text_report` | `STRING` | Output produced by the `text_report` socket. |
| `json_report` | `STRING` | Output produced by the `json_report` socket. |
| `status` | `STRING` | Output produced by the `status` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `text_report` (`STRING`), `json_report` (`STRING`), `status` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
