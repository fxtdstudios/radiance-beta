[← Back to Radiance docs](../README.md)

# VFX, Masks, Optics, and Multipass

Plate preparation, masks and roto, depth, camera and lens optics, motion,
multipass extraction, real AOV ingestion, and relighting.

## Typical graph

```text
Read plate → Plate Prep → Multipass Master (or AOV Reader) → Relight / Composite → Write
```

## Before you use these nodes

- **Estimated passes are estimates.** Multipass Master derives albedo,
  roughness, ambient occlusion and a segmentation ID from a single image. That
  is genuinely useful for generated footage and 2D plates. It is not a
  substitute for render passes, and the segmentation output is a clustered
  matte, not a Cryptomatte. For ground truth, feed a multilayer EXR through the
  Multipass AOV Reader.
- **Premultiplication.** The compositing nodes unpremultiply before operating
  and re-premultiply afterwards. If you are bringing in a matte from elsewhere,
  know which convention it uses.
- **Depth needs `transformers`.** Depth Anything V2 weights download on first
  use and are cached; the cache holds one tier at a time.

## Known limitation

Optical flow is a single-scale Lucas–Kanade solve despite the DIS reference in
its docstring. It recovers roughly the full displacement at 1 pixel, about 17%
at 3 pixels, and about 1% at 5 — so mask propagation and vector blur are
effectively static above a couple of pixels of motion. Use it for subtle
stabilisation, not for fast action.

## Nodes in this section (29)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [Aberration](#aberration) | `RadianceChromaticAberration` | Simulate lateral chromatic aberration (fringe) as a lens artefact. |
| [Anamorphic Streaks](#anamorphic-streaks) | `RadianceAnamorphicStreaks` | Add anamorphic lens streak flares to bright highlights. |
| [Blend Composite](#blend-composite) | `RadianceBlendComposite` | Composite two images using Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, or Divide blend modes. |
| [Depth](#depth) | `RadianceDepthMapGenerator` | Depth Anything V2 monocular depth estimation |
| [Grain](#grain) | `RadianceFilmGrain` | Simulate photochemical film grain |
| [HDR Grain Matcher](#hdr-grain-matcher) | `RadianceHDRGrainMatcher` |  |
| [HDR Inpaint Crop](#hdr-inpaint-crop) | `RadianceHDRCrop` |  |
| [HDR Inpaint Stitch](#hdr-inpaint-stitch) | `RadianceHDRStitch` |  |
| [Keyer (SAM)](#keyer-sam) | `RadianceSAMGenerator` |  |
| [LensDistortion](#lensdistortion) | `RadianceLensDistortion` | Apply or remove lens distortion using Brown-Conrady coefficients. |
| [Matte](#matte) | `RadianceLinearMatting` |  |
| [MotionBlur](#motionblur) | `RadianceMotionBlur` | Apply physically-based motion blur using optical flow vectors. |
| [MotionVectors](#motionvectors) | `RadianceOpticalFlow` | Estimate dense optical flow between adjacent frames. |
| [Multipass AOV Reader](#multipass-aov-reader) | `RadianceMultipassAOVReader` | Read a real multilayer/AOV OpenEXR and split its named layers into the same passes as the Master extractor |
| [Multipass Composite](#multipass-composite) | `RadianceMultipassComposite` |  |
| [Multipass Extract](#multipass-extract) | `RadianceMultipassMaster` | Estimate utility, material, and lighting passes from beauty, while preserving connected renderer AOVs. |
| [Multipass Relight](#multipass-relight) | `RadianceMultipassRelight` |  |
| [RadianceBitDepthDegrade](#radiancebitdepthdegrade) | `RadianceBitDepthDegrade` | Simulate lower bit-depth quantisation for look development and QC. |
| [Read Mask](#read-mask) | `RadianceLoadImageMask` | Load an image with optional non-destructive mask override |
| [Relight Engine](#relight-engine) | `RadianceRelightEngine` | Relight an HDR or SDR image using environment map or directional light. |
| [Roto](#roto) | `RadianceVectorMaskDraw` |  |
| [SAM Model Loader](#sam-model-loader) | `RadianceSAMModelLoader` |  |
| [SAM Multi-Mask Picker](#sam-multi-mask-picker) | `RadianceMultiMaskVisualPicker` |  |
| [Scene Cut Detect](#scene-cut-detect) | `RadianceSceneCutDetect` | Automatically detect scene cuts in a video using visual change metrics. |
| [Scene Cut Split](#scene-cut-split) | `RadianceSceneCutSplit` | Split a video at detected scene cut points into discrete segments. |
| [Stabilize](#stabilize) | `RadianceSubpixelStabilizer` |  |
| [Temporal Stitch Stabilizer](#temporal-stitch-stabilizer) | `RadianceTemporalStitchStabilizer` |  |
| [Vignette](#vignette) | `RadianceVignette` | Apply a natural or stylised vignette to the image edges. |
| [Write EXR Passes](#write-exr-passes) | `RadianceEXRPassesWriter` | Write all passes inside the RADIANCE_PASSES bundle into a single-part multilayer or true multi-part OpenEXR file. |

---

## Aberration

**Node key:** `RadianceChromaticAberration`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/optics.py`  

Simulate lateral chromatic aberration (fringe) as a lens artefact.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `shift_r` | Yes | `FLOAT` | `0.005` | -0.1 – 0.1, step 0.001 | Radial scale for Red channel. Positive = pushed outward. |
| `shift_g` | Yes | `FLOAT` | `0.0` | -0.1 – 0.1, step 0.001 | Radial scale for Green channel. Typically 0 (reference channel). |
| `shift_b` | Yes | `FLOAT` | `-0.005` | -0.1 – 0.1, step 0.001 | Radial scale for Blue channel. Negative = pulled inward. |
| `center_x` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Horizontal center of the distortion/effect (0.0 = left, 1.0 = right). |
| `center_y` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Vertical center of the effect (0.0 = top, 1.0 = bottom). |
| `invert` | Yes | `BOOLEAN` | `False` |  | Invert shift for CA removal / undistortion pass. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## Anamorphic Streaks

**Node key:** `RadianceAnamorphicStreaks`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/optics.py`  

Add anamorphic lens streak flares to bright highlights.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `threshold` | Yes | `FLOAT` | `1.0` | 0.0 – 10.0, step 0.01 | Luminance threshold above which highlights generate streaks. |
| `streak_length` | Yes | `INT` | `64` | 1 – 512 | Maximum streak tail length in pixels. |
| `streak_color_r` | Yes | `FLOAT` | `0.0` | 0.0 – 5.0, step 0.01 | Red component of the anamorphic streak colour. Values > 1.0 produce HDR-energy streaks. |
| `streak_color_g` | Yes | `FLOAT` | `0.5` | 0.0 – 5.0, step 0.01 | Green component of the anamorphic streak colour. |
| `streak_color_b` | Yes | `FLOAT` | `1.0` | 0.0 – 5.0, step 0.01 | Streak color tint. Default is the cool blue-cyan typical of anamorphic lenses. |
| `intensity` | Yes | `FLOAT` | `1.0` | 0.0 – 10.0, step 0.01 | Vignette or streak intensity. Values > 1.0 add HDR energy. |
| `streak_direction` | Yes | choice of `Horizontal`, `Vertical`, `Diagonal +45`, `Diagonal -45` | `Horizontal` |  | Direction of streak propagation. |
| `streak_falloff` | Yes | `FLOAT` | `3.0` | 0.5 – 20.0, step 0.1 | Exponential decay rate of the streak kernel. Higher = tighter falloff (shorter effective tail). Lower = longer, softer streak. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Original image with streaks composited (add blend). |
| `streak_pass` | `IMAGE` | Isolated streak pass for manual compositing (pre-add). |

---

## Blend Composite

**Node key:** `RadianceBlendComposite`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/pipeline/overlay.py`  

Composite two images using Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, or Divide blend modes.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `base` | Yes | `IMAGE` |  |  | Bottom layer (background). |
| `blend` | Yes | `IMAGE` |  |  | Top layer (foreground). |
| `mode` | Yes | choice of `Normal`, `Add`, `Screen`, `Multiply`, `Overlay`, `Soft Light`, `Difference`, `Divide` | `Normal` |  |  |
| `opacity` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 | Overall strength of the blend layer. |
| `mask` | No | `MASK` |  |  | Optional per-pixel mask (grayscale). White = full blend, Black = base only. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## Depth

**Node key:** `RadianceDepthMapGenerator`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/depth.py`  

Depth Anything V2 monocular depth estimation. Video-safe — standardizes each frame with spatial-temporal alignment preventing flickering. Outputs 3-channel grayscale depth map. Connect to Depth of Field node for realistic defocus blur.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `model_size` | Yes | choice of `Small (25M - Fast)`, `Base (98M - Balanced)`, `Large (335M - Best)` | `Large (335M - Best)` |  | Depth Anything V2 model size. Small = fast previews, Large = best quality. |
| `normalize` | No | `BOOLEAN` | `True` |  | Normalize depth to 0-1 range. For video, frames are standardized for temporal consistency. |
| `invert` | No | `BOOLEAN` | `False` |  | Invert depth (white=far, black=near). |
| `blur_edges` | No | `FLOAT` | `0.0` | 0.0 – 5.0, step 0.5 | Gaussian blur to smooth depth discontinuities. |
| `use_gpu` | No | `BOOLEAN` | `True` |  | Run depth estimation on GPU. Requires a CUDA-capable device. Falls back to CPU if unavailable. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `depth_map` | `IMAGE` |  |

---

## Grain

**Node key:** `RadianceFilmGrain`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/optics.py`  

Simulate photochemical film grain. Luminance-weighted noise, HDR-safe — grain intensity scales with pixel brightness.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `grain_size` | Yes | `FLOAT` | `1.0` | 0.1 – 8.0, step 0.05 | Base grain size in pixels (Gaussian sigma). 1.0 ≈ ISO 400 medium format. |
| `grain_strength` | Yes | `FLOAT` | `0.04` | 0.0 – 1.0, step 0.001 | Grain amplitude. 0.015=fine, 0.04=medium, 0.12=heavy. |
| `grain_size_r_offset` | Yes | `FLOAT` | `0.1` | -1.0 – 1.0, step 0.05 | R channel grain size offset from base (pixels). Red grain is typically coarser. |
| `grain_size_g_offset` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.05 | G channel grain size offset. Green has finest grain (most photo-sites). |
| `grain_size_b_offset` | Yes | `FLOAT` | `-0.1` | -1.0 – 1.0, step 0.05 | B channel grain size offset. Blue grain is typically finest. |
| `hdr_aware` | Yes | `BOOLEAN` | `True` |  | Scale grain amplitude by 1/√luminance (Poisson shot-noise model). Makes highlights finer-grained than shadows — physically correct. Disable for flat/uniform grain. |
| `seed` | Yes | `INT` | `0` | 0 – 2147483647 | Random seed. 0 = random each run. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## HDR Grain Matcher

**Node key:** `RadianceHDRGrainMatcher`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/plate.py`  

◎ Radiance HDR Grain Matcher

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `target` | Yes | `IMAGE` |  |  |  |
| `reference` | Yes | `IMAGE` |  |  |  |
| `intensity` | Yes | `FLOAT` | `1.0` | 0.0 – 5.0, step 0.05 |  |
| `kernel_size` | Yes | `INT` | `3` | 1 – 15, step 2 |  |
| `r_gain` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 |  |
| `g_gain` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 |  |
| `b_gain` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `grained_image` | `IMAGE` |  |

---

## HDR Inpaint Crop

**Node key:** `RadianceHDRCrop`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/inpaint.py`  

◎ Radiance HDR Crop

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `mask` | Yes | `MASK` |  |  |  |
| `context_padding` | Yes | `FLOAT` | `1.5` | 1.0 – 4.0, step 0.05 |  |
| `force_multiple` | Yes | `INT` | `16` | 1 – 256 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `cropped_image` | `IMAGE` |  |
| `cropped_mask` | `MASK` |  |
| `stitcher_data` | `STITCHER_DATA` |  |

---

## HDR Inpaint Stitch

**Node key:** `RadianceHDRStitch`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/inpaint.py`  

◎ Radiance HDR Stitch

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `original_image` | Yes | `IMAGE` |  |  |  |
| `cropped_image` | Yes | `IMAGE` |  |  |  |
| `cropped_mask` | Yes | `MASK` |  |  |  |
| `stitcher_data` | Yes | `STITCHER_DATA` |  |  |  |
| `blend_mode` | Yes | choice of `Linear_Laplacian`, `Linear_Gaussian`, `Standard` | `Linear_Laplacian` |  |  |
| `feather_radius` | Yes | `INT` | `16` | 0 – 128 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `stitched_image` | `IMAGE` |  |
| `stitch_blend_mask` | `MASK` |  |

---

## Keyer (SAM)

**Node key:** `RadianceSAMGenerator`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/masking.py`  

◎ Radiance SAM Mask Generator

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `sam_model` | Yes | `SAM_MODEL` |  |  |  |
| `points` | Yes | `STRING` | `[[256, 256]]` | multiline |  |
| `point_labels` | Yes | `STRING` | `[1]` |  |  |
| `text_prompt` | No | `STRING` | `` |  |  |
| `bbox` | No | `STRING` | `` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `mask` | `MASK` |  |
| `masked_image` | `IMAGE` |  |

---

## LensDistortion

**Node key:** `RadianceLensDistortion`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/optics.py`  

Apply or remove lens distortion using Brown-Conrady coefficients.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `k1` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.005 | Primary radial distortion. k1<0 = barrel, k1>0 = pincushion. |
| `k2` | Yes | `FLOAT` | `0.0` | -1.0 – 1.0, step 0.005 | Secondary radial distortion. Affects extreme corners. Use sparingly. |
| `scale` | Yes | `FLOAT` | `1.0` | 0.1 – 2.0, step 0.01 | Uniform scale applied after distortion. Use to crop black borders. |
| `center_x` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Optical center X (0.5 = image center). |
| `center_y` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Optical center Y (0.5 = image center). |
| `padding_mode` | Yes | choice of `zeros`, `reflection`, `border` | `zeros` |  | Edge fill: zeros=black, reflection=mirrored, border=edge-clamped. |
| `invert` | Yes | `BOOLEAN` | `False` |  | Invert warp for plate undistortion. Uses exact closed-form inverse — valid for all k1/k2 combinations except the degenerate singularity at 1+k1·r²+k2·r⁴=0 (clamped automatically). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Distorted image. |
| `st_map` | `IMAGE` | ST-Map: R=U, G=V normalized [0,1] sample coordinates. Wire to Nuke STMap / Fusion DisplaceImage for downstream use. |

---

## Matte

**Node key:** `RadianceLinearMatting`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/masking.py`  

◎ Radiance Linear Alpha Matting

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `mask` | Yes | `MASK` |  |  |  |
| `method` | Yes | choice of `GuidedFilter`, `ViTMatte`, `RVM` | `GuidedFilter` |  |  |
| `trimap_dilation` | Yes | `INT` | `12` | 0 – 128 |  |
| `eps` | Yes | `FLOAT` | `0.0001` | 1e-06 – 0.1, step 1e-06 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `alpha_matte` | `MASK` |  |
| `foreground_image` | `IMAGE` |  |

---

## MotionBlur

**Node key:** `RadianceMotionBlur`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/motion_blur.py`  

Apply physically-based motion blur using optical flow vectors.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `motion_vectors` | Yes | `IMAGE` |  |  | 32-bit UV vectors from Radiance Optical Flow. |
| `shutter_angle` | Yes | `FLOAT` | `180.0` | 0.0 – 720.0 | Standard cinema is 180°. Higher = more blur. 360° = full frame motion blur. |
| `samples` | Yes | `INT` | `8` | 2 – 32 | Number of sub-frame integration samples. Higher = smoother streaks. |
| `energy_conservation` | Yes | `BOOLEAN` | `True` |  | Ensures that bright highlights maintain their intensity over the blur area. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## MotionVectors

**Node key:** `RadianceOpticalFlow`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/motion.py`  

Estimate dense optical flow between adjacent frames.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  | Batch of frames to analyze. |
| `preset` | Yes | choice of `Fast`, `Medium`, `Ultra` | `Medium` |  |  |
| `flow_scale` | Yes | `FLOAT` | `1.0` | 0.1 – 10.0, step 0.1 | Scale factor for output vectors. 1.0 = pixel units. |
| `visualize` | Yes | `BOOLEAN` | `False` |  | Outputs a color-coded visualization of the motion field. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `motion_vectors` | `IMAGE` |  |
| `visualization` | `IMAGE` |  |
| `stats` | `STRING` |  |

---

## Multipass AOV Reader

**Node key:** `RadianceMultipassAOVReader`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/multipass/aov_reader.py`  

Read a real multilayer/AOV OpenEXR and split its named layers into the same passes as the Master extractor. Ground-truth renderer passes — not estimates. Missing layers come through black.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `exr_path` | Yes | `STRING` | `` |  | Path to a multilayer/AOV EXR (Arnold/Redshift/Karma/Cycles/V-Ray). |
| `beauty_layer` | No | `STRING` | `auto` |  | Override the beauty layer name, or 'auto' to detect (RGBA / 'beauty'). |
| `albedo_layer` | No | `STRING` | `auto` |  | Override the albedo/diffuse layer name, or 'auto'. |
| `normal_layer` | No | `STRING` | `auto` |  | Override the normal layer name, or 'auto'. |
| `depth_layer` | No | `STRING` | `auto` |  | Override the depth (Z) layer name, or 'auto'. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passes` | `RADIANCE_PASSES` |  |
| `beauty` | `IMAGE` |  |
| `albedo` | `IMAGE` |  |
| `normal_map` | `IMAGE` |  |
| `depth` | `IMAGE` |  |
| `roughness` | `IMAGE` |  |
| `specular` | `IMAGE` |  |
| `metallic` | `IMAGE` |  |
| `ao` | `IMAGE` |  |
| `emission` | `IMAGE` |  |
| `transmission` | `IMAGE` |  |
| `highpass` | `IMAGE` |  |
| `world_position` | `IMAGE` |  |
| `curvature` | `IMAGE` |  |
| `shadow_mask` | `IMAGE` |  |
| `midtone_mask` | `IMAGE` |  |
| `highlight_mask` | `IMAGE` |  |
| `reflection_mask` | `IMAGE` |  |
| `motion_vector` | `IMAGE` |  |
| `segmentation_id` | `IMAGE` |  |
| `motion_visualization` | `IMAGE` |  |
| `alpha` | `IMAGE` |  |

---

## Multipass Composite

**Node key:** `RadianceMultipassComposite`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/multipass/relight_comp.py`  

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `foreground` | Yes | `IMAGE` |  |  |  |
| `alpha` | Yes | `IMAGE` |  |  |  |
| `background` | No | `IMAGE` |  |  |  |
| `relit_foreground` | No | `IMAGE` |  |  |  |
| `foreground_depth` | No | `IMAGE` |  |  |  |
| `background_depth` | No | `IMAGE` |  |  |  |
| `shadow_mask` | No | `IMAGE` |  |  |  |
| `alpha_invert` | No | `BOOLEAN` | `False` |  |  |
| `premultiplied_input` | No | `BOOLEAN` | `False` |  |  |
| `depth_near_is_white` | No | `BOOLEAN` | `True` |  |  |
| `depth_bias` | No | `FLOAT` | `0.01` | -0.25 – 0.25, step 0.001 |  |
| `shadow_strength` | No | `FLOAT` | `0.35` | 0.0 – 1.0, step 0.01 |  |
| `light_wrap` | No | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.01 |  |
| `light_wrap_radius` | No | `INT` | `8` | 0 – 64 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `composite` | `IMAGE` |  |
| `premultiplied_foreground` | `IMAGE` |  |
| `holdout_mask` | `IMAGE` |  |
| `depth_matte` | `IMAGE` |  |
| `comp_info` | `STRING` |  |

---

## Multipass Extract

**Node key:** `RadianceMultipassMaster`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/multipass/master.py`  

Estimate utility, material, and lighting passes from beauty, while preserving connected renderer AOVs.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `beauty` | Yes | `IMAGE` |  |  |  |
| `depth_map` | No | `IMAGE` |  |  |  |
| `normal_map` | No | `IMAGE` |  |  |  |
| `prev_frame` | No | `IMAGE` |  |  |  |
| `source_passes` | No | `RADIANCE_PASSES` |  |  |  |
| `luma_weights` | No | choice of `Rec.709 / sRGB`, `ACEScg / AP1`, `Rec.2020` | `Rec.709 / sRGB` |  |  |
| `auto_depth_model` | No | choice of `disabled`, `Depth Anything V2 — Small  (99 MB, fast)`, `Depth Anything V2 — Base   (390 MB, balanced)`, `Depth Anything V2 — Large  (1.3 GB, highest quality)` | `disabled` |  |  |
| `depth_near_is_white` | No | `BOOLEAN` | `True` |  |  |
| `depth_scale` | No | `FLOAT` | `10.0` | 0.01 – 1000.0, step 0.1 |  |
| `fov_degrees` | No | `FLOAT` | `60.0` | 10.0 – 120.0 |  |
| `dsine_model_path` | No | `STRING` | `auto` |  |  |
| `normal_strength` | No | `FLOAT` | `2.0` | 0.1 – 20.0, step 0.1 |  |
| `normal_convention` | No | choice of `OpenGL (Y-Up)`, `DirectX (Y-Down)` | `OpenGL (Y-Up)` |  |  |
| `albedo_shading_radius` | No | `FLOAT` | `80.0` | 10.0 – 300.0, step 5.0 |  |
| `albedo_eps` | No | `FLOAT` | `0.001` | 0.0001 – 0.1, step 0.0005 |  |
| `specular_floor` | No | `BOOLEAN` | `True` |  |  |
| `roughness_fine_radius` | No | `FLOAT` | `2.0` | 1.0 – 10.0, step 0.5 |  |
| `roughness_coarse_radius` | No | `FLOAT` | `15.0` | 3.0 – 60.0 |  |
| `transmission_sensitivity` | No | `FLOAT` | `2.0` | 0.5 – 10.0, step 0.25 |  |
| `highpass_radius` | No | `FLOAT` | `8.0` | 0.1 – 100.0, step 0.5 |  |
| `highpass_strength` | No | `FLOAT` | `1.0` | 0.1 – 10.0, step 0.1 |  |
| `highpass_contrast` | No | `FLOAT` | `1.0` | 0.1 – 5.0, step 0.1 |  |
| `shadow_threshold` | No | `FLOAT` | `0.2` | 0.0 – 0.6, step 0.01 |  |
| `highlight_threshold` | No | `FLOAT` | `0.75` | 0.4 – 1.0, step 0.01 |  |
| `mask_softness` | No | `FLOAT` | `0.15` | 0.01 – 0.5, step 0.01 |  |
| `ao_radius` | No | `FLOAT` | `15.0` | 0.0 – 100.0 |  |
| `ao_strength` | No | `FLOAT` | `1.0` | 0.0 – 4.0, step 0.1 |  |
| `ao_samples` | No | `INT` | `8` | 4 – 32, step 4 |  |
| `lk_window_radius` | No | `INT` | `7` | 1 – 32 |  |
| `motion_coherence` | No | `FLOAT` | `0.5` | 0.0 – 0.95, step 0.05 |  |
| `batch_is_sequence` | No | `BOOLEAN` | `False` |  |  |
| `object_id_segments` | No | `INT` | `16` | 2 – 64 |  |
| `object_id_spatial_weight` | No | `FLOAT` | `0.25` | 0.0 – 2.0, step 0.05 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passes` | `RADIANCE_PASSES` |  |
| `beauty` | `IMAGE` |  |
| `albedo` | `IMAGE` |  |
| `normal_map` | `IMAGE` |  |
| `depth` | `IMAGE` |  |
| `roughness` | `IMAGE` |  |
| `specular` | `IMAGE` |  |
| `metallic` | `IMAGE` |  |
| `ao` | `IMAGE` |  |
| `emission` | `IMAGE` |  |
| `transmission` | `IMAGE` |  |
| `highpass` | `IMAGE` |  |
| `world_position` | `IMAGE` |  |
| `curvature` | `IMAGE` |  |
| `shadow_mask` | `IMAGE` |  |
| `midtone_mask` | `IMAGE` |  |
| `highlight_mask` | `IMAGE` |  |
| `reflection_mask` | `IMAGE` |  |
| `motion_vector` | `IMAGE` |  |
| `segmentation_id` | `IMAGE` |  |
| `motion_visualization` | `IMAGE` |  |
| `alpha` | `IMAGE` |  |

---

## Multipass Relight

**Node key:** `RadianceMultipassRelight`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/multipass/relight_comp.py`  

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `albedo` | Yes | `IMAGE` |  |  |  |
| `normal_map` | Yes | `IMAGE` |  |  |  |
| `beauty` | No | `IMAGE` |  |  |  |
| `roughness` | No | `IMAGE` |  |  |  |
| `metallic` | No | `IMAGE` |  |  |  |
| `specular` | No | `IMAGE` |  |  |  |
| `ao` | No | `IMAGE` |  |  |  |
| `alpha` | No | `IMAGE` |  |  |  |
| `shadow_mask` | No | `IMAGE` |  |  |  |
| `depth_map` | No | `IMAGE` |  |  |  |
| `world_position` | No | `IMAGE` |  |  |  |
| `normal_convention` | No | choice of `OpenGL (Y-Up)`, `DirectX (Y-Down)` | `OpenGL (Y-Up)` |  |  |
| `light_type` | No | choice of `Directional`, `Point` | `Directional` |  |  |
| `light_x` | No | `FLOAT` | `-0.35` | -10.0 – 10.0, step 0.01 |  |
| `light_y` | No | `FLOAT` | `0.45` | -10.0 – 10.0, step 0.01 |  |
| `light_z` | No | `FLOAT` | `1.0` | -10.0 – 10.0, step 0.01 |  |
| `light_r` | No | `FLOAT` | `1.0` | 0.0 – 8.0, step 0.01 |  |
| `light_g` | No | `FLOAT` | `1.0` | 0.0 – 8.0, step 0.01 |  |
| `light_b` | No | `FLOAT` | `1.0` | 0.0 – 8.0, step 0.01 |  |
| `intensity` | No | `FLOAT` | `1.0` | 0.0 – 20.0, step 0.05 |  |
| `ambient` | No | `FLOAT` | `0.03` | 0.0 – 4.0, step 0.01 |  |
| `specular_intensity` | No | `FLOAT` | `1.0` | 0.0 – 8.0, step 0.05 |  |
| `depth_scale` | No | `FLOAT` | `10.0` | 0.01 – 1000.0, step 0.1 |  |
| `depth_near_is_white` | No | `BOOLEAN` | `True` |  |  |
| `mix_with_beauty` | No | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.01 |  |
| `output_premultiplied` | No | `BOOLEAN` | `False` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `relit` | `IMAGE` |  |
| `diffuse_light` | `IMAGE` |  |
| `specular_light` | `IMAGE` |  |
| `lighting` | `IMAGE` |  |
| `alpha` | `IMAGE` |  |
| `relight_info` | `STRING` |  |

---

## RadianceBitDepthDegrade

**Node key:** `RadianceBitDepthDegrade`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/color/colorspace.py`  

Simulate lower bit-depth quantisation for look development and QC.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `bit_depth` | Yes | `INT` | `8` | 4 – 16 |  |
| `dither_mode` | Yes | choice of `none`, `triangular`, `floyd-steinberg` | `triangular` |  |  |
| `delta_gain` | No | `FLOAT` | `10.0` | 1.0 – 100.0, step 0.5 |  |
| `banding_threshold` | No | `FLOAT` | `0.004` | 0.0005 – 0.05, step 0.0005 |  |
| `restore_from_quantized` | No | `BOOLEAN` | `False` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `quantized` | `IMAGE` |  |
| `delta_amplified` | `IMAGE` |  |
| `banding_mask` | `IMAGE` |  |
| `metrics` | `STRING` |  |

---

## Read Mask

**Node key:** `RadianceLoadImageMask`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/io/mask.py`  

Load an image with optional non-destructive mask override. If a companion '_radmask.png' file exists alongside the source image, its alpha channel is used as the mask instead of the image's own alpha. The Radiance mask editor frontend saves masks in this companion format.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | choice (empty) |  |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `mask` | `MASK` |  |

---

## Relight Engine

**Node key:** `RadianceRelightEngine`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/hdr/synthesis.py`  

Relight an HDR or SDR image using environment map or directional light.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `normal_map` | Yes | `IMAGE` |  |  |  |
| `light_dir_x` | Yes | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.01 | Light direction X component. Normalized internally — sets the horizontal angle of the synthetic light. |
| `light_dir_y` | Yes | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.01 | Light direction Y component. Positive = light from above. |
| `light_dir_z` | Yes | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.01 | Light direction Z component. Positive = light in front of surface. |
| `light_color_r` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 | Red component of the synthetic light color. Values > 1.0 produce HDR emission. |
| `light_color_g` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 | Green component of the synthetic light color. |
| `light_color_b` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 | Blue component of the synthetic light color. |
| `diffuse_intensity` | Yes | `FLOAT` | `1.0` | 0.0 – 10.0, step 0.01 | Lambertian diffuse reflection strength. Controls broad, soft illumination. |
| `specular_intensity` | Yes | `FLOAT` | `0.5` | 0.0 – 10.0, step 0.01 | Specular highlight strength. Higher values produce brighter, more visible glints. |
| `specular_roughness` | Yes | `FLOAT` | `0.1` | 0.01 – 1.0, step 0.01 | Surface roughness for Blinn-Phong specular. Low = sharp glints (metallic), high = soft broad highlights (matte). |
| `camera` | No | `RADIANCE_CAMERA` |  |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `lighting_pass_only` | `IMAGE` |  |

---

## Roto

**Node key:** `RadianceVectorMaskDraw`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/roto.py`  

◎ Radiance Vector Mask Draw

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `width` | Yes | `INT` | `512` | 64 – 4096, step 8 |  |
| `height` | Yes | `INT` | `512` | 64 – 4096, step 8 |  |
| `shape_type` | Yes | choice of `Polygon`, `Bezier_Spline` | `Polygon` |  |  |
| `points_data` | Yes | `STRING` | `[[128, 128], [384, 128], [384, 384], [128, 384]]` | multiline | Paste JSON coordinate list or Nuke control points block. |
| `anti_alias_width` | Yes | `FLOAT` | `1.5` | 0.0 – 8.0, step 0.1 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `vector_mask` | `MASK` |  |

---

## SAM Model Loader

**Node key:** `RadianceSAMModelLoader`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/masking.py`  

◎ Radiance SAM Model Loader

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `model_name` | Yes | choice of `sam2.1_hiera_large.pt`, `sam3_hiera_large.pt`, `sam2.1_hiera_base.pt` | `sam2.1_hiera_large.pt` |  |  |
| `device` | Yes | choice of `cuda`, `cpu`, `mps` | `cuda` |  |  |
| `offload_to_cpu` | Yes | `BOOLEAN` | `False` |  |  |
| `dtype` | Yes | choice of `float16`, `bfloat16`, `float32` | `float16` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `sam_model` | `SAM_MODEL` |  |

---

## SAM Multi-Mask Picker

**Node key:** `RadianceMultiMaskVisualPicker`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/masking.py`  

◎ Radiance Multi-Mask Picker

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `masks` | Yes | `MASK` |  |  |  |
| `picker_index` | Yes | `INT` | `0` | 0 – 5 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `selected_mask` | `MASK` |  |

---

## Scene Cut Detect

**Node key:** `RadianceSceneCutDetect`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/ai/scene_cut.py`  

Automatically detect scene cuts in a video using visual change metrics.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  | Full video sequence as IMAGE batch. |
| `threshold` | Yes | `FLOAT` | `0.35` | 0.05 – 1.0, step 0.01 | Cut sensitivity. Lower = detect more cuts. Typical range 0.25–0.45 for most footage. |
| `min_shot_frames` | Yes | `INT` | `12` | 1 – 500 | Minimum frames between detected cuts. |
| `method` | Yes | choice of `histogram`, `edge`, `combined` | `combined` |  | histogram: colour distribution diff (fast). edge: Sobel edge map diff (catches content cuts). combined: weighted blend of both (recommended). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `cut_data` | `STRING` |  |
| `shot_count` | `INT` |  |
| `score_plot` | `IMAGE` |  |

---

## Scene Cut Split

**Node key:** `RadianceSceneCutSplit`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/ai/scene_cut.py`  

Split a video at detected scene cut points into discrete segments.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  |  |
| `cut_data` | Yes | `STRING` |  |  | JSON from RadianceSceneCutDetect. |
| `shot_index` | Yes | `INT` | `0` | 0 – 9999 | Which shot to extract (0-based). Connect shot_count output to know the range. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `frames` | `IMAGE` |  |
| `shot_index` | `INT` |  |
| `start_frame` | `INT` |  |
| `end_frame` | `INT` |  |
| `shot_info` | `STRING` |  |

---

## Stabilize

**Node key:** `RadianceSubpixelStabilizer`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/plate.py`  

◎ Radiance Subpixel Stabilizer

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `anchor_frame` | Yes | `INT` | `0` | 0 – 1000 |  |
| `max_shift` | Yes | `INT` | `64` | 4 – 512, step 4 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `stabilized_sequence` | `IMAGE` |  |
| `displacements_xy` | `IMAGE` |  |

---

## Temporal Stitch Stabilizer

**Node key:** `RadianceTemporalStitchStabilizer`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/inpaint.py`  

◎ Radiance Temporal Stitch Stabilizer

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `masks` | Yes | `MASK` |  |  |  |
| `temporal_sigma` | Yes | `FLOAT` | `2.0` | 0.0 – 16.0, step 0.5 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `stabilized_masks` | `MASK` |  |

---

## Vignette

**Node key:** `RadianceVignette`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/optics.py`  

Apply a natural or stylised vignette to the image edges.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `strength` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Vignette intensity at corners. 0=no effect, 1=full black. |
| `power` | Yes | `FLOAT` | `2.0` | 0.5 – 8.0, step 0.1 | Falloff exponent. 2.0 = natural cos⁴ approximation. Higher = harder edge. |
| `center_x` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Horizontal center of the vignette (0 = left, 1 = right). |
| `center_y` | Yes | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Vertical center of the vignette (0 = top, 1 = bottom). |
| `feather` | Yes | `FLOAT` | `1.0` | 0.1 – 4.0, step 0.05 | Radial feather: scales the normalized radius. >1 = pushes falloff inward. |
| `tint_r` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 | Red multiplier for vignetted (dark) areas. 1.0=neutral. |
| `tint_g` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 | Green channel multiplier for vignette tint. 1.0 = neutral. |
| `tint_b` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.01 | Blue multiplier for vignetted areas. >1 = cool shadow edges. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Vignetted image. |
| `vignette_mask` | `IMAGE` | Grayscale vignette mask [0,1] — wire to downstream grades or multiply nodes. |

---

## Write EXR Passes

**Node key:** `RadianceEXRPassesWriter`  
**Menu:** `FXTD STUDIOS/Radiance/VFX`  
**Source:** `nodes/vfx/multipass/master.py`  
**Output node** — runs even with nothing connected downstream.  

Write all passes inside the RADIANCE_PASSES bundle into a single-part multilayer or true multi-part OpenEXR file.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `passes` | Yes | `RADIANCE_PASSES` |  |  |  |
| `filename_prefix` | Yes | `STRING` | `radiance_vfx_passes` |  |  |
| `bit_depth` | Yes | choice of `16-bit Half Float`, `32-bit Float` | `16-bit Half Float` |  |  |
| `compression` | Yes | choice of `ZIP`, `ZIPS`, `PIZ`, `RLE`, `Uncompressed`, `PXR24`, `B44`, `B44A`, … (+2 more) | `ZIP` |  |  |
| `output_path` | No | `STRING` | `` |  |  |
| `remote_path` | No | `STRING` | `` |  |  |
| `frame_index` | No | `INT` | `1001` | 0 – 999999 |  |
| `exr_layout` | No | choice of `Single-part multilayer`, `Multi-part` | `Single-part multilayer` |  |  |
| `custom_metadata` | No | `STRING` | `` | multiline |  |

Hidden inputs supplied by ComfyUI: `extra_pnginfo`, `prompt`.

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `output_path` | `STRING` |  |

---
