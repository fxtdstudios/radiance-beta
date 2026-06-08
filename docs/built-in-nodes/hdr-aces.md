[← Back to Radiance docs](../README.md)

# HDR and ACES

HDR analysis, tone mapping, ACES 2.0 transforms, SDR-to-HDR preparation, highlight recovery, relighting, and HDR latent support.

## Typical workflow

```text
Read EXR -> HDR Auto Log Select -> HDR Color Pipeline -> Generate/VFX -> HDR Diagnostics -> HDR Monitor / Write
```

## Before you use these nodes

- Use diagnostics before and after major HDR transforms so clipping is visible.
- Keep paired compression, log, and metadata settings consistent across encode/decode steps.
- Display transforms are not masters; save scene-linear or HDR EXR when finishing later.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ ACES 2.0 Tonescale](#aces-2-0-tonescale) | `RadianceACES2Tonescale` | Apply the Daniele Evo forward tonescale (ACES 2.0 official curve). |
| [◎ ACES 2.0 Gamut Compress](#aces-2-0-gamut-compress) | `RadianceACES2ReachGamutCompress` | ACES 2.0 Reach-Based Gamut Compression. |
| [◎ ACES 2.0 Output Transform](#aces-2-0-output-transform) | `RadianceACES2OutputTransformFull` | Complete ACES 2.0 Output Transform (reference-accurate). |
| [◎ HDR Color Pipeline](#hdr-color-pipeline) | `RadianceHDRColorPipeline` | HDR Color Pipeline. |
| [◎ HDR Encode](#hdr-encode) | `RadianceHDREncode` | HDR Encode. |
| [◎ HDR Monitor](#hdr-monitor) | `RadianceHDRMonitor` | HDR Monitor. |
| [◎ HDR Auto Log Select](#hdr-auto-log-select) | `RadianceHDRAutoLogSelect` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ HDR Diagnostics](#hdr-diagnostics) | `RadianceHDRDiagnostics` | Analyzes the image or workflow state and returns reports that help catch delivery problems. |
| [◎ Clip Detector](#clip-detector) | `RadianceClipDetector` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ SDR to HDR Prepare](#sdr-to-hdr-prepare) | `RadianceSDRToHDRPrepare` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ HDR Highlight Composite](#hdr-highlight-composite) | `RadianceHDRHighlightComposite` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ SDR to HDR Expand](#sdr-to-hdr-expand) | `RadianceSDRtoHDRExpand` | SDR to HDR Expand. |
| [◎ HDR Synthesis Engine](#hdr-synthesis-engine) | `RadianceHDRSynthesisEngine` | HDR Synthesis Engine. |
| [◎ Relight Engine](#relight-engine) | `RadianceRelightEngine` | Relight Engine. |
| [◎ HDR Latent Encoder](#hdr-latent-encoder) | `RadianceHDRLatentEncoder` | HDR Latent Encoder. |

## ◎ ACES 2.0 Tonescale

**Internal key:** `RadianceACES2Tonescale`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/aces2.py`
**Function:** `apply`

### What it does

Apply the Daniele Evo forward tonescale (ACES 2.0 official curve).

### When to use it

Use `◎ ACES 2.0 Tonescale` when the graph reaches the ACES 2.0 Tonescale step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | Scene-linear ACEScg (AP1) image batch. |
| `peak_nits` | Yes | `FLOAT` | `100.0` | Display peak luminance in cd/m². 100 = SDR, 1000/2000/4000 = HDR. |
| `mode` | Yes | `ENUM: luminance_preserving, per_channel` | `luminance_preserving` | luminance_preserving: tone-map luma then scale RGB — preserves hue/saturation. per_channel: apply curve independently to R, G, B — may introduce hue shifts but avoids colour casts. |
| `contrast_g` | Optional | `FLOAT` | `1.15` | Contrast exponent g. ACES 2.0 reference = 1.15. |
| `grey_target` | Optional | `FLOAT` | `0.1` | Display grey target as fraction of peak. Standard = 0.10 (10 nits of 100 SDR). |
| `toe_scene` | Optional | `FLOAT` | `0.04` | Scene luminance below which a linear toe is applied. Prevents gamma lift in deep shadows. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `curve_info` | `STRING` | Output produced by the `curve_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `curve_info` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ ACES 2.0 Gamut Compress

**Internal key:** `RadianceACES2ReachGamutCompress`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/aces2.py`
**Function:** `compress`

### What it does

ACES 2.0 Reach-Based Gamut Compression.

### When to use it

Use `◎ ACES 2.0 Gamut Compress` when the graph reaches the ACES 2.0 Gamut Compress step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | Scene-linear ACEScg (AP1) image batch. |
| `strength` | Yes | `FLOAT` | `1.0` | Compression strength. 0 = bypass. 1.0 = standard ACES 2.0. >1 = more aggressive squeeze for difficult footage. |
| `limit_cyan` | Optional | `FLOAT` | `1.147` | Reach gamut limit for cyan channel (R). |
| `limit_magenta` | Optional | `FLOAT` | `1.264` | Reach gamut limit for magenta channel (G). |
| `limit_yellow` | Optional | `FLOAT` | `1.312` | Reach gamut limit for yellow channel (B). |
| `threshold_cyan` | Optional | `FLOAT` | `0.815` | - |
| `threshold_magenta` | Optional | `FLOAT` | `0.803` | - |
| `threshold_yellow` | Optional | `FLOAT` | `0.88` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `compress_info` | `STRING` | Output produced by the `compress_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `compress_info` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ ACES 2.0 Output Transform

**Internal key:** `RadianceACES2OutputTransformFull`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/aces2.py`
**Function:** `transform`

### What it does

Complete ACES 2.0 Output Transform (reference-accurate).

### When to use it

Use `◎ ACES 2.0 Output Transform` when the graph reaches the ACES 2.0 Output Transform step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `input_colorspace` | Yes | `ENUM: ACEScg, ACES2065-1, Linear_sRGB, Linear_Rec2020` | `ACEScg` | - |
| `output_transform` | Yes | `(cls.OUTPUT_TRANSFORMS, {'default': 'ACES 2.0 SDR (sRGB/Rec.709)'})` | - | - |
| `peak_luminance` | Optional | `FLOAT` | `100.0` | SDR peak luminance (nits). Ignored for HDR outputs. |
| `surround` | Optional | `ENUM: Dark, Dim, Average` | `Dim` | Viewing environment — affects contrast parameter g. |
| `exposure_adjust` | Optional | `FLOAT` | `0.0` | Exposure adjustment in stops before transform. |
| `creative_white_scale` | Optional | `FLOAT` | `1.0` | - |
| `gamut_compress_strength` | Optional | `FLOAT` | `1.0` | ACES 2.0 reach gamut compression strength. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `transform_info` | `STRING` | Output produced by the `transform_info` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `transform_info` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Color Pipeline

**Internal key:** `RadianceHDRColorPipeline`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/colorspace.py`
**Function:** `pipeline`

### What it does

HDR Color Pipeline.

### When to use it

Use `◎ HDR Color Pipeline` when you want one higher-level node to wire several lower-level steps together.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `encoding` | Yes | `(list(_EOTF_MAP.keys()), {'default': 'sRGB'})` | - | - |
| `compression_ratio` | Yes | `FLOAT` | `0.5` | Compression ratio for soft-knee HDR highlight compression. |
| `source_primaries` | Optional | `(_PRIMARIES_LIST, {'default': 'Rec.709 (sRGB)'})` | - | - |
| `target_primaries` | Optional | `(_PRIMARIES_LIST, {'default': 'Rec.709 (sRGB)'})` | - | - |
| `chromatic_adaptation` | Optional | `(['None'] + list(_BRADFORD_CAT.keys()), {'default': 'None'})` | - | - |
| `pq_peak_nits` | Optional | `FLOAT` | `1000.0` | Reference peak luminance in nits for PQ decoding. Only used when encoding is 'PQ (ST.2084)'. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `vae_image` | `IMAGE` | Output produced by the `vae_image` socket. |
| `scene_linear` | `IMAGE` | Output produced by the `scene_linear` socket. |
| `peak_linear` | `FLOAT` | Output produced by the `peak_linear` socket. |
| `colorspace_json` | `STRING` | Output produced by the `colorspace_json` socket. |

### Practical notes

- The node returns `vae_image` (`IMAGE`), `scene_linear` (`IMAGE`), `peak_linear` (`FLOAT`), `colorspace_json` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Encode

**Internal key:** `RadianceHDREncode`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/delivery.py`
**Function:** `encode`

### What it does

HDR Encode.

### When to use it

Use `◎ HDR Encode` when the graph reaches the HDR Encode step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `format` | Yes | `(cls._FORMATS, {'default': 'PQ (HDR10)'})` | - | - |
| `peak_nits` | Optional | `ENUM: 100, 1000, 4000, 10000` | `1000` | [PQ] Mastering display peak luminance. 1000 = HDR10, 4000/10000 = Dolby Vision grade. |
| `reference_white_nits` | Optional | `FLOAT` | `203.0` | [PQ] Nits where scene-linear 1.0 maps. BT.2408 recommends 203. |
| `scene_linear_gain` | Optional | `FLOAT` | `1.0` | [HLG] Scale scene-linear before encoding. Useful for exposure trim. |
| `apply_bt2020` | Optional | `BOOLEAN` | `False` | Convert BT.709 primaries to BT.2020. Required for standards-compliant HDR10 / HLG delivery. Disable if source is already BT.2020. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `encoded_image` | `IMAGE` | Output produced by the `encoded_image` socket. |

### Practical notes

- The node returns `encoded_image` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Monitor

**Internal key:** `RadianceHDRMonitor`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/delivery.py`
**Function:** `monitor`

### What it does

HDR Monitor.

### When to use it

Use `◎ HDR Monitor` on a review branch so you can inspect output without changing the master path.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `mode` | Yes | `(cls._MODES, {'default': 'Preview (SDR)'})` | - | - |
| `operator` | Optional | `(list(_TONE_MAP_OPS.keys()), {'default': 'ACES (Narkowicz)', 'tooltip': '[Preview (SDR)] Tone-mapping operator.'})` | - | - |
| `exposure` | Optional | `FLOAT` | `0.0` | [Preview (SDR)] EV offset applied before tone mapping. |
| `saturation` | Optional | `FLOAT` | `1.0` | [Preview (SDR)] Post-tonemap saturation scale. |
| `gamma` | Optional | `FLOAT` | `2.2` | [Preview (SDR)] Display gamma. 2.2 ≈ sRGB, 2.4 = IEC 61966 precise. |
| `reinhard_white` | Optional | `FLOAT` | `4.0` | [Preview (SDR) / Reinhard Extended] White point. |
| `peak_nits` | Optional | `FLOAT` | `1000.0` | [Rec.2100 PQ] Mastering display peak luminance in nits. |
| `gamma_correct_sdr` | Optional | `BOOLEAN` | `True` | [Preview (SDR)] Apply sRGB gamma encoding to output. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `preview` | `IMAGE` | Output produced by the `preview` socket. |

### Practical notes

- The node returns `preview` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Auto Log Select

**Internal key:** `RadianceHDRAutoLogSelect`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/smart.py`
**Function:** `select`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ HDR Auto Log Select` when the graph reaches the HDR Auto Log Select step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `override` | Yes | `ENUM: auto, LogC4, SLog3, VLog, LogC3, ACEScct` | `auto` | - |
| `model_hint` | Optional | `STRING` | `` | Optional model name (e.g. 'flux', 'cogvideox', 'wan2.1', 'sdxl'). When set, the compression_ratio output is taken from the RADIANCE_MODEL_PRESETS table instead of the log-format default. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `log_format` | `STRING` | Output produced by the `log_format` socket. |
| `compression_ratio` | `FLOAT` | Output produced by the `compression_ratio` socket. |
| `stops_detected` | `FLOAT` | Output produced by the `stops_detected` socket. |
| `model_preset_used` | `STRING` | Output produced by the `model_preset_used` socket. |

### Practical notes

- The node returns `log_format` (`STRING`), `compression_ratio` (`FLOAT`), `stops_detected` (`FLOAT`), `model_preset_used` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Diagnostics

**Internal key:** `RadianceHDRDiagnostics`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/smart.py`
**Function:** `diagnose`

### What it does

Analyzes the image or workflow state and returns reports that help catch delivery problems.

### When to use it

Use `◎ HDR Diagnostics` before final output or when debugging an unexpected result.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `compression_ratio` | Optional | `FLOAT` | `0.5` | Must match the value used in RadianceHDRTurboEncoder. |
| `model_preset_used` | Optional | `STRING` | `` | Resolved model key from AutoLogSelect. |
| `stats_json` | Optional | `STRING` | `` | JSON from RadianceHDRPerChannelNorm (optional). |
| `coherence_map` | Optional | `IMAGE` | - | - |
| `colorspace` | Optional | `ENUM: Linear (sRGB), ACEScg, sRGB, Rec.709` | `Linear (sRGB)` | Input colour space for nit/EV-range estimation. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `report_json` | `STRING` | Output produced by the `report_json` socket. |
| `psnr_estimate` | `FLOAT` | Output produced by the `psnr_estimate` socket. |
| `peak_stops` | `FLOAT` | Output produced by the `peak_stops` socket. |
| `peak_nit` | `FLOAT` | Output produced by the `peak_nit` socket. |
| `ev_range` | `FLOAT` | Output produced by the `ev_range` socket. |
| `clipped_pct` | `FLOAT` | Output produced by the `clipped_pct` socket. |
| `is_hdr` | `BOOLEAN` | Output produced by the `is_hdr` socket. |

### Practical notes

- The node returns `report_json` (`STRING`), `psnr_estimate` (`FLOAT`), `peak_stops` (`FLOAT`), `peak_nit` (`FLOAT`), `ev_range` (`FLOAT`), `clipped_pct` (`FLOAT`), `is_hdr` (`BOOLEAN`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Clip Detector

**Internal key:** `RadianceClipDetector`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/uplift.py`
**Function:** `detect`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Clip Detector` when the graph reaches the Clip Detector step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `threshold` | Yes | `FLOAT` | `0.97` | Pixels brighter than this are marked as clipped. |
| `channel_mode` | Yes | `ENUM: any, all, luma` | `any` | - |
| `soft_edge` | Yes | `FLOAT` | `0.03` | Feathering width below the threshold. 0 = hard binary mask. |
| `dilate_px` | Yes | `INT` | `8` | Expand the clip mask outward to cover highlight fringing. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `clip_mask` | `MASK` | Output produced by the `clip_mask` socket. |
| `clip_fraction` | `FLOAT` | Output produced by the `clip_fraction` socket. |
| `visualization` | `IMAGE` | Output produced by the `visualization` socket. |

### Practical notes

- The node returns `clip_mask` (`MASK`), `clip_fraction` (`FLOAT`), `visualization` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ SDR to HDR Prepare

**Internal key:** `RadianceSDRToHDRPrepare`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/uplift.py`
**Function:** `prepare`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ SDR to HDR Prepare` when the graph reaches the SDR to HDR Prepare step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `clip_mask` | Yes | `MASK` | - | - |
| `compression_ratio` | Yes | `FLOAT` | `0.5` | Must match RadianceHDRDecoder. Wire from preset or LoRALoader. |
| `inverse_eotf` | Optional | `ENUM: sRGB, Rec.709, Gamma 2.2, Linear (no-op)` | `sRGB` | - |
| `highlight_boost` | Optional | `FLOAT` | `4.0` | How bright to push extrapolated highlights in scene-linear. 4.0 = 2 stops above white. Higher = more vivid reconstructed highlights. |
| `boost_gamma` | Optional | `FLOAT` | `1.5` | Power curve for highlight boost ramp. Higher = sharper specular peaks. |
| `mask_feather` | Optional | `INT` | `16` | Gaussian feather radius on the inpainting mask edge (pixels). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `mask` | `MASK` | Output produced by the `mask` socket. |
| `stats_json` | `STRING` | Output produced by the `stats_json` socket. |
| `peak_linear` | `FLOAT` | Output produced by the `peak_linear` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `mask` (`MASK`), `stats_json` (`STRING`), `peak_linear` (`FLOAT`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Highlight Composite

**Internal key:** `RadianceHDRHighlightComposite`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/uplift.py`
**Function:** `composite`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ HDR Highlight Composite` when the graph reaches the HDR Highlight Composite step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `original_image` | Yes | `IMAGE` | - | - |
| `hdr_image` | Yes | `IMAGE` | - | - |
| `clip_mask` | Yes | `MASK` | - | - |
| `inverse_eotf` | Optional | `ENUM: sRGB, Rec.709, Gamma 2.2, Linear (no-op)` | `sRGB` | - |
| `blend_softness` | Optional | `INT` | `24` | Feathering on composite edge in pixels. 0 = hard cut. |
| `highlight_strength` | Optional | `FLOAT` | `1.0` | How much of the AI highlight reconstruction to use. 1.0 = full. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ SDR to HDR Expand

**Internal key:** `RadianceSDRtoHDRExpand`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/synthesis.py`
**Function:** `apply`

### What it does

SDR to HDR Expand.

### When to use it

Use `◎ SDR to HDR Expand` when the graph reaches the SDR to HDR Expand step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `inverse_oetf` | Yes | `ENUM: None, sRGB, Rec.709` | `sRGB` | - |
| `threshold` | Yes | `FLOAT` | `0.8` | Luminance threshold above which HDR expansion begins. 0.8 = expand highlights above 80% SDR white. |
| `expansion_gain` | Yes | `FLOAT` | `5.0` | Peak luminance multiplier for expanded highlights. 5.0 = 500 nits from 100-nit SDR white. |
| `expansion_gamma` | Yes | `FLOAT` | `1.2` | Power curve applied to the expansion mask. Values > 1.0 create a harder shoulder; < 1.0 a softer roll-off. |
| `smoothness` | Yes | `FLOAT` | `0.1` | Feathering radius for the expansion mask edge. Higher values prevent harsh highlight boundaries. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Synthesis Engine

**Internal key:** `RadianceHDRSynthesisEngine`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/synthesis.py`
**Function:** `synthesize`

### What it does

HDR Synthesis Engine.

### When to use it

Use `◎ HDR Synthesis Engine` when the graph reaches the HDR Synthesis Engine step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `energy_target` | Yes | `FLOAT` | `10.0` | Target peak luminance multiplier (e.g. 10.0 = 10 stops above SDR white). |
| `recovery_iters` | Yes | `INT` | `3` | Number of Laplacian iterations to recover clipped detail. |
| `chroma_preservation` | Yes | `FLOAT` | `0.8` | Prevents expanded highlights from losing saturation or shifting hue. |
| `guidance_mask` | Optional | `MASK` | - | Per-pixel guidance mask from Radiance Luminance Guidance. |
| `guidance_nits` | Optional | `FLOAT` | `0.0` | Local target peak nits. 0 = use global energy_target only. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `highlight_mask` | `IMAGE` | Output produced by the `highlight_mask` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `highlight_mask` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Relight Engine

**Internal key:** `RadianceRelightEngine`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/synthesis.py`
**Function:** `apply`

### What it does

Relight Engine.

### When to use it

Use `◎ Relight Engine` when the graph reaches the Relight Engine step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `normal_map` | Yes | `IMAGE` | - | - |
| `light_dir_x` | Yes | `FLOAT` | `1.0` | Light direction X component. Normalized internally — sets the horizontal angle of the synthetic light. |
| `light_dir_y` | Yes | `FLOAT` | `1.0` | Light direction Y component. Positive = light from above. |
| `light_dir_z` | Yes | `FLOAT` | `1.0` | Light direction Z component. Positive = light in front of surface. |
| `light_color_r` | Yes | `FLOAT` | `1.0` | Red component of the synthetic light color. Values > 1.0 produce HDR emission. |
| `light_color_g` | Yes | `FLOAT` | `1.0` | Green component of the synthetic light color. |
| `light_color_b` | Yes | `FLOAT` | `1.0` | Blue component of the synthetic light color. |
| `diffuse_intensity` | Yes | `FLOAT` | `1.0` | Lambertian diffuse reflection strength. Controls broad, soft illumination. |
| `specular_intensity` | Yes | `FLOAT` | `0.5` | Specular highlight strength. Higher values produce brighter, more visible glints. |
| `specular_roughness` | Yes | `FLOAT` | `0.1` | Surface roughness for Blinn-Phong specular. Low = sharp glints (metallic), high = soft broad highlights (matte). |
| `camera` | Optional | `RADIANCE_CAMERA` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `lighting_pass_only` | `IMAGE` | Output produced by the `lighting_pass_only` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `lighting_pass_only` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Latent Encoder

**Internal key:** `RadianceHDRLatentEncoder`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/hdr/encoder.py`
**Function:** `encode`

### What it does

HDR Latent Encoder.

### When to use it

Use `◎ HDR Latent Encoder` when the graph reaches the HDR Latent Encoder step in a hdr and aces workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `vae` | Yes | `VAE` | - | - |
| `mode` | Yes | `(cls._MODES, {'default': 'Soft-Knee (LTX)'})` | - | - |
| `compression_ratio` | Optional | `FLOAT` | `0.5` | [Soft-Knee] 0 = hard clamp, 1 = full Reinhard. Mirrors LTX-Video tone_map_compression_ratio. |
| `exposure_offset` | Optional | `FLOAT` | `0.0` | [Soft-Knee] EV offset applied in scene-linear before compression. |
| `energy_normalization` | Optional | `FLOAT` | `1.0` | [Log Calibration] Scale applied before log1p encoding. Higher values compress brighter highlights more aggressively. |
| `normalize_channels` | Optional | `BOOLEAN` | `False` | Apply per-channel mean/std normalisation before VAE encode (mirrors LTX-Video vae_per_channel_normalize). Wire channel_stats → RadianceHDRPerChannelDenorm to invert. |
| `norm_center` | Optional | `FLOAT` | `3.0` | [Channel Norm] Window half-width in σ mapped to [0,1]. 3.0 captures ±3σ — suitable for most HDR content. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `latent` | `LATENT` | Output produced by the `latent` socket. |
| `channel_stats` | `STRING` | Output produced by the `channel_stats` socket. |

### Practical notes

- The node returns `latent` (`LATENT`), `channel_stats` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
