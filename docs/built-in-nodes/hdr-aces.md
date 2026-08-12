[← Back to Radiance docs](../README.md)

# HDR and ACES

ACES output transforms, OCIO, HDR encode and decode, tone mapping, SDR→HDR
uplift, and the HDR VAE path.

## Typical graph

```text
scene-linear → ACES 2.0 Output Transform → display
scene-linear → HDR Encode (PQ / HLG) → deliverable
SDR plate    → SDR to HDR Universal → scene-linear HDR
```

## Before you use these nodes

- **Know what "peak nits" means in each place.** For PQ it is an absolute
  display luminance and part of the encode. For HLG it is a property of the
  monitor, not of the file. For DCI it is the projector's luminance at code
  value 1.0. The nodes handle these differently because the standards do.
- **Tone mapping is one-way.** Once you have tone-mapped for a display you have
  thrown away highlight information. Keep the scene-linear master.
- **AgX emits display code values directly** and is not gamma-encoded a second
  time. The other operators are.
- **Alpha is never tone-mapped.** From 3.2.0 the tone-map and expansion nodes
  split alpha off and reattach it untouched.

## Known limitation — read this before trusting the ACES label

The ACES 2.0 tone scale here is a log-space contrast of 1.55 with a tanh
shoulder. It is **not** the Daniele Evo curve the ACES 2.0 specification
defines. In practice 18% grey renders about 0.84 stop above the ACES 2.0
reference on the SDR path, and HLG diffuse white lands at signal 0.915 rather
than the 0.75 that BT.2408 specifies.

The normalisation defects around it were fixed in 3.2.0 — Cinema no longer
flat-lines above 0.4 scene-linear, and HLG no longer pins diffuse white to the
display peak — but the curve itself has not been replaced. Grade against a
reference you trust rather than against the label.

## Nodes in this section (19)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [ACES 2.0 Gamut Compress](#aces-20-gamut-compress) | `RadianceACES2ReachGamutCompress` | Compress out-of-gamut values using the ACES 2.0 Reach Gamut method. |
| [ACES 2.0 Output Transform](#aces-20-output-transform) | `RadianceACES2OutputTransformFull` | Full ACES 2.0 Output Transform (RRT + ODT) for display rendering. |
| [ACES 2.0 Tonescale](#aces-20-tonescale) | `RadianceACES2Tonescale` | Apply the ACES 2.0 Tonescale operator with configurable parameters. |
| [ACES Transform](#aces-transform) | `RadianceACESTransform` | Apply ACES 1.x Input, Viewing, or Output Transform to an image. |
| [Clip Detector](#clip-detector) | `RadianceClipDetector` | Detect clipped highlights in SDR images |
| [HDR Auto Log Select](#hdr-auto-log-select) | `RadianceHDRAutoLogSelect` | Automatically select the optimal log encoding for an HDR input. |
| [HDR Color Pipeline](#hdr-color-pipeline) | `RadianceHDRColorPipeline` | Full colour pipeline: linearise → adapt → primaries → HDR compress |
| [HDR Diagnostics](#hdr-diagnostics) | `RadianceHDRDiagnostics` | Run full HDR diagnostic checks |
| [HDR Encode](#hdr-encode) | `RadianceHDREncode` | Scene-linear HDR → PQ (HDR10/DV) or HLG (Broadcast) delivery signal. |
| [HDR Expand Dynamic Range](#hdr-expand-dynamic-range) | `RadianceHDRExpandDynamicRange` | Expand SDR images to HDR dynamic range by recovering highlights and extending stops of exposure latitude. |
| [HDR Highlight Composite](#hdr-highlight-composite) | `RadianceHDRHighlightComposite` | Composite AI-reconstructed HDR highlights back onto original linearised SDR |
| [HDR LoRA Apply](#hdr-lora-apply) | `RadianceHDRLoRAApply` | Apply a Radiance HDR LoRA to a diffusion model |
| [HDR LoRA Loader](#hdr-lora-loader) | `RadianceHDRLoRALoader` | Load a Radiance HDR LoRA .safetensors and extract embedded training metadata |
| [HDR Synthesis Engine](#hdr-synthesis-engine) | `RadianceHDRSynthesisEngine` | Synthesise HDR imagery from SDR input and optional guidance signals. |
| [SDR to HDR Expand](#sdr-to-hdr-expand) | `RadianceSDRtoHDRExpand` | Expand an SDR image into HDR headroom via inverse OETF and mathematical highlight expansion |
| [SDR to HDR Prepare](#sdr-to-hdr-prepare) | `RadianceSDRToHDRPrepare` | Prepare an SDR image for AI HDR reconstruction |
| [SDR → HDR Recover](#sdr--hdr-recover) | `RadianceSDRToHDRRecover` | Learned SDR→HDR reconstruction for clipped highlights and crushed shadows |
| [SDR → HDR Universal](#sdr--hdr-universal) | `RadianceSDRToHDRUniversal` | SDR→HDR orchestrator: deterministic Expand, learned Recover, or Hybrid |
| [Tonemap](#tonemap) | `RadianceHDRToneMap` | Professional HDR tone mapping with presets and advanced controls |

---

## ACES 2.0 Gamut Compress

**Node key:** `RadianceACES2ReachGamutCompress`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/aces2.py`  

Compress out-of-gamut values using the ACES 2.0 Reach Gamut method.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | Scene-linear ACEScg (AP1) image batch. |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.5, step 0.05 | Compression strength. 0 = bypass. 1.0 = standard ACES 2.0. >1 = more aggressive squeeze for difficult footage. |
| `limit_cyan` | No | `FLOAT` | `1.147` | 1.0 – 2.0, step 0.01 | Reach gamut limit for cyan channel (R). |
| `limit_magenta` | No | `FLOAT` | `1.264` | 1.0 – 2.0, step 0.01 | Reach gamut limit for magenta channel (G). |
| `limit_yellow` | No | `FLOAT` | `1.312` | 1.0 – 2.0, step 0.01 | Reach gamut limit for yellow channel (B). |
| `threshold_cyan` | No | `FLOAT` | `0.815` | 0.0 – 1.0, step 0.01 |  |
| `threshold_magenta` | No | `FLOAT` | `0.803` | 0.0 – 1.0, step 0.01 |  |
| `threshold_yellow` | No | `FLOAT` | `0.88` | 0.0 – 1.0, step 0.01 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `compress_info` | `STRING` |  |

---

## ACES 2.0 Output Transform

**Node key:** `RadianceACES2OutputTransformFull`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/aces2.py`  

Full ACES 2.0 Output Transform (RRT + ODT) for display rendering.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `input_colorspace` | Yes | choice of `ACEScg`, `ACES2065-1`, `Linear_sRGB`, `Linear_Rec2020` | `ACEScg` |  |  |
| `output_transform` | Yes | choice of `ACES 2.0 SDR (sRGB/Rec.709)`, `ACES 2.0 SDR (P3-D65)`, `ACES 2.0 HDR (Rec.2100 PQ 1000 nits)`, `ACES 2.0 HDR (Rec.2100 PQ 2000 nits)`, `ACES 2.0 HDR (Rec.2100 PQ 4000 nits)`, `ACES 2.0 HDR (Rec.2100 HLG)`, `ACES 2.0 Cinema (DCI-P3 D60)`, `ACES 2.0 Cinema (DCI-P3 D65)` | `ACES 2.0 SDR (sRGB/Rec.709)` |  |  |
| `peak_luminance` | No | `FLOAT` | `100.0` | 48.0 – 10000.0 | SDR peak luminance (nits). Ignored for HDR outputs. |
| `surround` | No | choice of `Dark`, `Dim`, `Average` | `Dim` |  | Viewing environment — affects contrast parameter g. |
| `exposure_adjust` | No | `FLOAT` | `0.0` | -4.0 – 4.0, step 0.1 | Exposure adjustment in stops before transform. |
| `creative_white_scale` | No | `FLOAT` | `1.0` | 0.5 – 2.0, step 0.01 |  |
| `gamut_compress_strength` | No | `FLOAT` | `1.0` | 0.0 – 1.5, step 0.05 | ACES 2.0 reach gamut compression strength. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `transform_info` | `STRING` |  |

---

## ACES 2.0 Tonescale

**Node key:** `RadianceACES2Tonescale`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/aces2.py`  

Apply the ACES 2.0 Tonescale operator with configurable parameters.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | Scene-linear ACEScg (AP1) image batch. |
| `peak_nits` | Yes | `FLOAT` | `100.0` | 48.0 – 10000.0 | Display peak luminance in cd/m². 100 = SDR, 1000/2000/4000 = HDR. |
| `mode` | Yes | choice of `luminance_preserving`, `per_channel` | `luminance_preserving` |  | luminance_preserving: tone-map luma then scale RGB — preserves hue/saturation. per_channel: apply curve independently to R, G, B — may introduce hue shifts but avoids colour casts. |
| `contrast_g` | No | `FLOAT` | `1.15` | 0.8 – 1.6, step 0.01 | Contrast exponent g. ACES 2.0 reference = 1.15. |
| `grey_target` | No | `FLOAT` | `0.1` | 0.05 – 0.3, step 0.005 | Display grey target as fraction of peak. Standard = 0.10 (10 nits of 100 SDR). |
| `toe_scene` | No | `FLOAT` | `0.04` | 0.005 – 0.2, step 0.005 | Scene luminance below which a linear toe is applied. Prevents gamma lift in deep shadows. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `curve_info` | `STRING` |  |

---

## ACES Transform

**Node key:** `RadianceACESTransform`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/color/colorspace.py`  

Apply ACES 1.x Input, Viewing, or Output Transform to an image.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | Scene-linear ACEScg image. |
| `odt` | Yes | choice of `sRGB D65`, `DCI-P3 D65`, `Rec.2020 PQ (HDR10)`, `Rec.2020 HLG` | `sRGB D65` |  |  |
| `exposure_offset` | Yes | `FLOAT` | `0.0` | -4.0 – 4.0, step 0.1 |  |
| `peak_nits` | Yes | `FLOAT` | `1000.0` | 100.0 – 10000.0, step 100.0 |  |
| `saturation` | Yes | `FLOAT` | `1.0` | 0.0 – 1.5, step 0.02 |  |
| `grade_info_in` | No | `STRING` |  |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `aces_info` | `STRING` |  |

---

## Clip Detector

**Node key:** `RadianceClipDetector`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/uplift.py`  

Detect clipped highlights in SDR images. Wire clip_mask → RadianceSDRToHDRPrepare and RadianceHDRHighlightComposite.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `threshold` | Yes | `FLOAT` | `0.97` | 0.5 – 1.0, step 0.005 | Pixels brighter than this are marked as clipped. |
| `channel_mode` | Yes | choice of `any`, `all`, `luma` | `any` |  |  |
| `soft_edge` | Yes | `FLOAT` | `0.03` | 0.0 – 0.2, step 0.005 | Feathering width below the threshold. 0 = hard binary mask. |
| `dilate_px` | Yes | `INT` | `8` | 0 – 64 | Expand the clip mask outward to cover highlight fringing. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `clip_mask` | `MASK` |  |
| `clip_fraction` | `FLOAT` |  |
| `visualization` | `IMAGE` |  |

---

## HDR Auto Log Select

**Node key:** `RadianceHDRAutoLogSelect`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/smart.py`  

Automatically select the optimal log encoding for an HDR input.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `override` | Yes | choice of `auto`, `LogC4`, `SLog3`, `VLog`, `LogC3`, `ACEScct` | `auto` |  |  |
| `model_hint` | No | `STRING` | `` |  | Optional model name (e.g. 'flux', 'cogvideox', 'wan2.1', 'sdxl'). When set, the compression_ratio output is taken from the RADIANCE_MODEL_PRESETS table instead of the log-format default. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `log_format` | `STRING` |  |
| `compression_ratio` | `FLOAT` |  |
| `stops_detected` | `FLOAT` |  |
| `model_preset_used` | `STRING` |  |

---

## HDR Color Pipeline

**Node key:** `RadianceHDRColorPipeline`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/colorspace.py`  

Full colour pipeline: linearise → adapt → primaries → HDR compress. Outputs both VAE-ready and scene-linear images.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `encoding` | Yes | choice of `sRGB`, `Rec.709 (OETF)`, `BT.1886 (TV γ2.4)`, `Gamma 2.2`, `Gamma 2.4`, `PQ (ST.2084)`, `HLG`, `ARRI LogC4`, … (+2 more) | `sRGB` |  |  |
| `compression_ratio` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Compression ratio for soft-knee HDR highlight compression. |
| `source_primaries` | No | choice of `Rec.709 (sRGB)`, `BT.2020`, `ACEScg`, `DCI-P3 (D65)`, `XYZ (D65)` | `Rec.709 (sRGB)` |  |  |
| `target_primaries` | No | choice of `Rec.709 (sRGB)`, `BT.2020`, `ACEScg`, `DCI-P3 (D65)`, `XYZ (D65)` | `Rec.709 (sRGB)` |  |  |
| `chromatic_adaptation` | No | choice of `None`, `D65_to_D60`, `D60_to_D65`, `D65_to_D50`, `D50_to_D65` | `None` |  |  |
| `pq_peak_nits` | No | `FLOAT` | `1000.0` | 100.0 – 10000.0, step 100.0 | Reference peak luminance in nits for PQ decoding. Only used when encoding is 'PQ (ST.2084)'. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `vae_image` | `IMAGE` |  |
| `scene_linear` | `IMAGE` |  |
| `peak_linear` | `FLOAT` |  |
| `colorspace_json` | `STRING` |  |

---

## HDR Diagnostics

**Node key:** `RadianceHDRDiagnostics`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/smart.py`  
**Output node** — runs even with nothing connected downstream.  

Run full HDR diagnostic checks. Outputs a JSON report, estimated PSNR, peak stops, and live metric floats (peak_nit, ev_range, clipped_pct, is_hdr) — replaces the separate RadianceHDRAnalysis node.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `compression_ratio` | No | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.05 | Must match the value used in RadianceHDRTurboEncoder. |
| `model_preset_used` | No | `STRING` | `` |  | Resolved model key from AutoLogSelect. |
| `stats_json` | No | `STRING` | `` |  | JSON from RadianceHDRPerChannelNorm (optional). |
| `coherence_map` | No | `IMAGE` |  |  |  |
| `colorspace` | No | choice of `Linear (sRGB)`, `ACEScg`, `sRGB`, `Rec.709` | `Linear (sRGB)` |  | Input colour space for nit/EV-range estimation. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `report_json` | `STRING` |  |
| `psnr_estimate` | `FLOAT` |  |
| `peak_stops` | `FLOAT` |  |
| `peak_nit` | `FLOAT` |  |
| `ev_range` | `FLOAT` |  |
| `clipped_pct` | `FLOAT` |  |
| `is_hdr` | `BOOLEAN` |  |

---

## HDR Encode

**Node key:** `RadianceHDREncode`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/delivery.py`  
**Output node** — runs even with nothing connected downstream.  

Scene-linear HDR → PQ (HDR10/DV) or HLG (Broadcast) delivery signal.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `format` | Yes | choice of `PQ (HDR10)`, `HLG (Broadcast)` | `PQ (HDR10)` |  |  |
| `peak_nits` | No | choice of `100`, `1000`, `4000`, `10000` | `1000` |  | [PQ] Mastering display peak luminance. 1000 = HDR10, 4000/10000 = Dolby Vision grade. |
| `reference_white_nits` | No | `FLOAT` | `203.0` | 80.0 – 400.0 | [PQ] Nits where scene-linear 1.0 maps. BT.2408 recommends 203. |
| `scene_linear_gain` | No | `FLOAT` | `1.0` | 0.1 – 8.0, step 0.1 | [HLG] Scale scene-linear before encoding. Useful for exposure trim. |
| `apply_bt2020` | No | `BOOLEAN` | `False` |  | Convert BT.709 primaries to BT.2020. Required for standards-compliant HDR10 / HLG delivery. Disable if source is already BT.2020. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `encoded_image` | `IMAGE` |  |

---

## HDR Expand Dynamic Range

**Node key:** `RadianceHDRExpandDynamicRange`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `hdr/tonemap.py`  

Expand SDR images to HDR dynamic range by recovering highlights and extending stops of exposure latitude.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `source_gamma` | Yes | `FLOAT` | `2.2` | 1.0 – 3.0, step 0.1 |  |
| `highlight_recovery` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.1 |  |
| `black_point` | Yes | `FLOAT` | `0.0` | -0.1 – 0.1, step 0.001 |  |
| `target_stops` | Yes | `FLOAT` | `14.0` | 8.0 – 20.0, step 0.5 |  |
| `highlight_rolloff` | Yes | `FLOAT` | `1.5` | 1.0 – 3.0, step 0.1 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## HDR Highlight Composite

**Node key:** `RadianceHDRHighlightComposite`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/uplift.py`  

Composite AI-reconstructed HDR highlights back onto original linearised SDR. Non-clipped areas are pixel-perfect original. Only clipped regions come from the model.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `original_image` | Yes | `IMAGE` |  |  |  |
| `hdr_image` | Yes | `IMAGE` |  |  |  |
| `clip_mask` | Yes | `MASK` |  |  |  |
| `inverse_eotf` | No | choice of `sRGB`, `Rec.709`, `Gamma 2.2`, `Linear (no-op)` | `sRGB` |  |  |
| `blend_softness` | No | `INT` | `24` | 0 – 128 | Feathering on composite edge in pixels. 0 = hard cut. |
| `highlight_strength` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.05 | How much of the AI highlight reconstruction to use. 1.0 = full. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## HDR LoRA Apply

**Node key:** `RadianceHDRLoRAApply`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/generate/lora.py`  

Apply a Radiance HDR LoRA to a diffusion model. Outputs the patched MODEL and the matching compression_ratio for RadianceHDREncoder.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` |  |  |  |
| `lora_dict` | Yes | `LORA_DICT` |  |  |  |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | LoRA strength multiplier. 1.0 = trained weight. 0 = no effect. |
| `model_hint` | No | `STRING` | `` |  | Cross-checks the LoRA's trained model against this hint and warns if mismatched. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `MODEL` | `MODEL` |  |
| `compression_ratio` | `FLOAT` |  |

---

## HDR LoRA Loader

**Node key:** `RadianceHDRLoRALoader`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/generate/lora.py`  

Load a Radiance HDR LoRA .safetensors and extract embedded training metadata. Wire compression_ratio → RadianceHDREncoder to guarantee consistent HDR response.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `lora_path` | Yes | `STRING` | `` |  | Path to a Radiance HDR LoRA checkpoint (.safetensors or .pt). Leave blank to use the RADIANCE_HDR_LORA env var. |
| `fallback_compression_ratio` | No | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Used when LoRA metadata does not contain compression_ratio. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `lora_dict` | `LORA_DICT` |  |
| `compression_ratio` | `FLOAT` |  |
| `model_name` | `STRING` |  |
| `metadata_json` | `STRING` |  |

---

## HDR Synthesis Engine

**Node key:** `RadianceHDRSynthesisEngine`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/synthesis.py`  

Synthesise HDR imagery from SDR input and optional guidance signals.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `energy_target` | Yes | `FLOAT` | `10.0` | 1.0 – 100.0 | Target peak luminance multiplier (e.g. 10.0 = 10 stops above SDR white). |
| `recovery_iters` | Yes | `INT` | `3` | 0 – 8 | Number of Laplacian iterations to recover clipped detail. |
| `chroma_preservation` | Yes | `FLOAT` | `0.8` | 0.0 – 1.0, step 0.05 | Prevents expanded highlights from losing saturation or shifting hue. |
| `guidance_mask` | No | `MASK` |  |  | Per-pixel guidance mask from Radiance Luminance Guidance. |
| `guidance_nits` | No | `FLOAT` | `0.0` | 0.0 – 10000.0, step 50.0 | Local target peak nits. 0 = use global energy_target only. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `highlight_mask` | `IMAGE` |  |

---

## SDR to HDR Expand

**Node key:** `RadianceSDRtoHDRExpand`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/synthesis.py`  

Expand an SDR image into HDR headroom via inverse OETF and mathematical highlight expansion. Does not reconstruct clipped detail.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `inverse_oetf` | Yes | choice of `None`, `sRGB`, `Rec.709` | `sRGB` |  |  |
| `threshold` | Yes | `FLOAT` | `0.8` | 0.0 – 1.0, step 0.01 | Luminance threshold above which HDR expansion begins. 0.8 = expand highlights above 80% SDR white. |
| `expansion_gain` | Yes | `FLOAT` | `5.0` | 1.0 – 100.0, step 0.1 | Peak luminance multiplier for expanded highlights. 5.0 = 500 nits from 100-nit SDR white. |
| `expansion_gamma` | Yes | `FLOAT` | `1.2` | 0.1 – 5.0, step 0.01 | Power curve applied to the expansion mask. Values > 1.0 create a harder shoulder; < 1.0 a softer roll-off. |
| `smoothness` | Yes | `FLOAT` | `0.1` | 0.0 – 0.5, step 0.01 | Feathering radius for the expansion mask edge. Higher values prevent harsh highlight boundaries. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## SDR to HDR Prepare

**Node key:** `RadianceSDRToHDRPrepare`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/uplift.py`  

Prepare an SDR image for AI HDR reconstruction. Applies inverse EOTF, highlight extrapolation, and soft-knee compression.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `clip_mask` | Yes | `MASK` |  |  |  |
| `compression_ratio` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Must match RadianceHDRDecoder. Wire from preset or LoRALoader. |
| `inverse_eotf` | No | choice of `sRGB`, `Rec.709`, `Gamma 2.2`, `Linear (no-op)` | `sRGB` |  |  |
| `highlight_boost` | No | `FLOAT` | `4.0` | 1.0 – 32.0, step 0.5 | How bright to push extrapolated highlights in scene-linear. 4.0 = 2 stops above white. Higher = more vivid reconstructed highlights. |
| `boost_gamma` | No | `FLOAT` | `1.5` | 0.5 – 4.0, step 0.1 | Power curve for highlight boost ramp. Higher = sharper specular peaks. |
| `mask_feather` | No | `INT` | `16` | 0 – 128 | Gaussian feather radius on the inpainting mask edge (pixels). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `mask` | `MASK` |  |
| `stats_json` | `STRING` |  |
| `peak_linear` | `FLOAT` |  |

---

## SDR → HDR Recover

**Node key:** `RadianceSDRToHDRRecover`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/uplift_universal.py`  

Learned SDR→HDR reconstruction for clipped highlights and crushed shadows. Uses still RUDRA for images or motion-aligned temporal RUDRA for video; pixels outside the recovery masks are preserved exactly.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | SDR image or ordered 5+ frame video batch. |
| `inverse_oetf` | Yes | choice of `sRGB`, `Rec.709`, `Gamma 2.2`, `Gamma 2.4`, `None` | `sRGB` |  |  |
| `peak_nits` | Yes | `FLOAT` | `1000.0` | 200.0 – 10000.0, step 50.0 |  |
| `highlight_threshold` | Yes | `FLOAT` | `0.98` | 0.8 – 0.999, step 0.001 | SDR code-value threshold used to identify clipped luma or RGB channels. |
| `shadow_threshold` | Yes | `FLOAT` | `0.05` | 0.001 – 0.5, step 0.005 | Linear-luma threshold used to identify crushed shadows. |
| `highlight_strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `shadow_strength` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `output_encoding` | Yes | choice of `Linear`, `Linear ACES2065-1 (AP0)`, `PQ (HDR10)`, `HLG` | `Linear` |  |  |
| `rudra_size` | Yes | choice of `rudra_turbo`, `rudra_full` | `rudra_turbo` |  |  |
| `vae` | No | `VAE` |  |  | Required only for single-frame RUDRA recovery. |
| `model_meta` | No | `STRING` | `` |  | RadianceUnifiedLoader model_meta for exact model-family resolution. |
| `temporal_window` | No | choice of `5`, `7`, `9` | `5` |  | Adjacent frames used by the Phase 3 temporal residual model. |
| `temporal_checkpoint` | No | `STRING` | `` |  | Optional Phase 3 checkpoint path. Empty uses RADIANCE_TEMPORAL_RUDRA or models/radiance. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `highlight_mask` | `MASK` |  |
| `shadow_mask` | `MASK` |  |
| `highlight_confidence` | `MASK` |  |
| `shadow_confidence` | `MASK` |  |

---

## SDR → HDR Universal

**Node key:** `RadianceSDRToHDRUniversal`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `nodes/hdr/uplift_universal.py`  

SDR→HDR orchestrator: deterministic Expand, learned Recover, or Hybrid. Includes safe fallback, professional output transforms, and separate highlight/shadow masks.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | SDR still, image batch, or video frames [B,H,W,C]. |
| `inverse_oetf` | Yes | choice of `sRGB`, `Rec.709`, `Gamma 2.2`, `Gamma 2.4`, `None` | `sRGB` |  |  |
| `peak_nits` | Yes | `FLOAT` | `1000.0` | 200.0 – 10000.0, step 50.0 |  |
| `knee_mode` | Yes | choice of `adaptive`, `manual` | `adaptive` |  |  |
| `knee` | Yes | `FLOAT` | `0.75` | 0.05 – 0.99, step 0.01 |  |
| `shoulder_gamma` | Yes | `FLOAT` | `1.6` | 0.5 – 6.0, step 0.05 |  |
| `temporal_smoothing` | Yes | `FLOAT` | `0.85` | 0.0 – 0.98, step 0.01 |  |
| `output_encoding` | Yes | choice of `Linear`, `Linear ACES2065-1 (AP0)`, `PQ (HDR10)`, `HLG` | `Linear` |  |  |
| `vae` | No | `VAE` |  |  | Optional RUDRA recovery VAE. Recover/Hybrid falls back to Expand when unavailable. |
| `rudra_size` | No | choice of `rudra_turbo`, `rudra_full` | `rudra_turbo` |  |  |
| `rudra_blend` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `model_meta` | No | `STRING` | `` |  |  |
| `batch_mode` | No | choice of `Independent Images`, `Video Frames` | `Independent Images` |  |  |
| `shadow_threshold` | No | `FLOAT` | `0.05` | 0.001 – 0.5, step 0.005 |  |
| `processing_mode` | No | choice of `Expand`, `Recover`, `Hybrid` | `Hybrid` |  | Expand: deterministic only. Recover: learned reconstruction only. Hybrid: expansion plus masked learned recovery. |
| `highlight_threshold` | No | `FLOAT` | `0.98` | 0.8 – 0.999, step 0.001 | Clipping threshold for Recover/Hybrid. Appended for saved-workflow compatibility. |
| `temporal_window` | No | choice of `5`, `7`, `9` | `5` |  | Adjacent video frames used by temporal RUDRA. |
| `temporal_checkpoint` | No | `STRING` | `` |  | Optional Phase 3 checkpoint path. Empty uses the configured default. |
| `learned_backend` | No | choice of `Auto`, `Direct Pixel`, `Legacy RUDRA` | `Auto` |  | Auto prefers temporal video recovery, then the direct-pixel model, then legacy VAE RUDRA. |
| `pixel_checkpoint` | No | `STRING` | `` |  | Direct-pixel .pt checkpoint. Empty searches models/radiance and RADIANCE_SDR2HDR_PIXEL. |
| `pixel_tile_size` | No | `INT` | `512` | 128 – 2048, step 64 |  |
| `pixel_tile_overlap` | No | `INT` | `64` | 0 – 512, step 16 |  |
| `pixel_recovery_mode` | No | choice of `highlights`, `all`, `shadows`, `off` | `highlights` |  | Highlights is safest and avoids hallucinating chroma in deep shadows. |
| `pixel_strength` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `highlight_mask` | `MASK` |  |
| `shadow_mask` | `MASK` |  |
| `highlight_confidence` | `MASK` |  |
| `shadow_confidence` | `MASK` |  |

---

## Tonemap

**Node key:** `RadianceHDRToneMap`  
**Menu:** `FXTD STUDIOS/Radiance/HDR`  
**Source:** `hdr/tonemap.py`  

Professional HDR tone mapping with presets and advanced controls. GPU-accelerated.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | Input HDR or SDR image to tone map |
| `preset` | No | choice of `None (Custom)`, `◎ Cinematic Film`, `◎ HDR Display`, `◎ Web / Social`, `◎ Print Ready`, `◎ Game Engine`, `◎ Photography`, `◎ Low Key / Dark`, … (+1 more) | `◎ Cinematic Film` |  | Quick look preset. Overrides settings below. |
| `operator` | No | choice of `filmic_aces`, `filmic_uncharted2`, `agx`, `reinhard`, `reinhard_extended`, `reinhard_luminance`, `linear_clamp`, `exposure_only` | `filmic_aces` |  | Tone mapping algorithm: • filmic_aces: Industry-standard film curve • filmic_uncharted2: Game industry favorite • agx: Modern Blender default • reinhard: Classic, preserves color • linear_clamp: Simple clip |
| `exposure` | No | `FLOAT` | `0.0` | -5.0 – 5.0, step 0.1 | Exposure adjustment in stops. Negative = darker, Positive = brighter. |
| `gamma` | No | `FLOAT` | `2.2` | 1.0 – 3.0, step 0.1 | Display gamma. 2.2 = sRGB standard. Higher = darker midtones. |
| `white_point` | No | `FLOAT` | `1.0` | 0.5 – 10.0, step 0.1 | Maximum brightness that maps to white. Higher = more headroom for highlights. |
| `contrast` | No | `FLOAT` | `1.0` | 0.5 – 2.0, step 0.05 | Contrast adjustment around midpoint. >1 = more punch. |
| `saturation` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | Color saturation. 0 = grayscale, 1 = original, >1 = vivid. |
| `highlight_compression` | No | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.05 | Compress highlights to prevent clipping. 0 = no compression, 1 = full. |
| `shadow_lift` | No | `FLOAT` | `0.0` | 0.0 – 0.2, step 0.01 | Lift shadows to reveal detail. 0 = no lift, 0.1 = subtle. |
| `use_gpu` | No | `BOOLEAN` | `True` |  | Use GPU acceleration when available. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Tone-mapped SDR image ready for display or export. |

---
