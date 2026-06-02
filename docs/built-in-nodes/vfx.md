# VFX, Masks, Optics, and Multipass

Depth, optical flow, motion blur, lens effects, SAM masking, matting, inpaint crop/stitch, roto, video mask propagation, multipass extraction, relight, composite, and EXR pass writing.

## Typical workflow

```text
Read plate -> Stabilize / Depth / Optical Flow / SAM -> Multipass / Inpaint / Composite -> EXR Passes Writer
```

## Before you use these nodes

- Preview masks and depth maps directly; silent mask inversion is a common source of bad composites.
- Use crop and stitch data from the same run when doing HDR inpaint workflows.
- Feed multipass EXR writing from the multipass master so the beauty pass and layer names are present.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Depth Map Generator](#depth-map-generator) | `RadianceDepthMapGenerator` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Optical Flow](#optical-flow) | `RadianceOpticalFlow` | Optical Flow. |
| [◎ Physical Motion Blur](#physical-motion-blur) | `RadianceMotionBlur` | Physical Motion Blur. |
| [◎ Lens Distortion](#lens-distortion) | `RadianceLensDistortion` | Lens Distortion. |
| [◎ Chromatic Aberration](#chromatic-aberration) | `RadianceChromaticAberration` | Chromatic Aberration. |
| [◎ Anamorphic Streaks](#anamorphic-streaks) | `RadianceAnamorphicStreaks` | Anamorphic Streaks. |
| [◎ Film Grain (Simple)](#film-grain-simple) | `RadianceFilmGrain` | Film Grain. |
| [◎ Vignette](#vignette) | `RadianceVignette` | Vignette. |
| [◎ SAM Model Loader](#sam-model-loader) | `RadianceSAMModelLoader` | SAM Model Loader. |
| [◎ SAM Mask Generator](#sam-mask-generator) | `RadianceSAMGenerator` | SAM Mask Generator. |
| [◎ SAM Multi-Mask Picker](#sam-multi-mask-picker) | `RadianceMultiMaskVisualPicker` | Multi-Mask Picker. |
| [◎ Linear Alpha Matting](#linear-alpha-matting) | `RadianceLinearMatting` | Linear Alpha Matting. |
| [◎ HDR Grain Matcher](#hdr-grain-matcher) | `RadianceHDRGrainMatcher` | HDR Grain Matcher. |
| [◎ Subpixel Plate Stabilizer](#subpixel-plate-stabilizer) | `RadianceSubpixelStabilizer` | Subpixel Stabilizer. |
| [◎ HDR Inpaint Crop](#hdr-inpaint-crop) | `RadianceHDRCrop` | HDR Crop. |
| [◎ HDR Inpaint Stitch](#hdr-inpaint-stitch) | `RadianceHDRStitch` | HDR Stitch. |
| [◎ Temporal Stitch Stabilizer](#temporal-stitch-stabilizer) | `RadianceTemporalStitchStabilizer` | Temporal Stitch Stabilizer. |
| [◎ Vector Mask Draw (Roto)](#vector-mask-draw-roto) | `RadianceVectorMaskDraw` | Vector Mask Draw. |
| [◎ Video Mask Propagator](#video-mask-propagator) | `RadianceVideoMaskPropagator` | Video Mask Propagator. |
| [◎ Multipass: Master VFX Extractor](#multipass-master-vfx-extractor) | `RadianceMultipassMaster` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Radiance EXR Passes Writer](#radiance-exr-passes-writer) | `RadianceEXRPassesWriter` | Writes or hands off the current result to a file, folder, preview, or DCC destination. |
| [◎ Multipass: Real PBR Relight](#multipass-real-pbr-relight) | `RadianceMultipassRelight` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Multipass: VFX Composite](#multipass-vfx-composite) | `RadianceMultipassComposite` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |

## ◎ Depth Map Generator

**Internal key:** `RadianceDepthMapGenerator`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/depth.py`
**Function:** `generate_depth`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Depth Map Generator` when the graph reaches the Depth Map Generator step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `model_size` | Yes | `(cls.MODEL_SIZES, {'default': 'Large (335M - Best)', 'tooltip': 'Depth Anything V2 model size. Small = fast previews, Large = best quality.'})` | - | - |
| `normalize` | Optional | `BOOLEAN` | `True` | Normalize depth to 0-1 range. For video, frames are standardized for temporal consistency. |
| `invert` | Optional | `BOOLEAN` | `False` | Invert depth (white=far, black=near). |
| `blur_edges` | Optional | `FLOAT` | `0.0` | Gaussian blur to smooth depth discontinuities. |
| `use_gpu` | Optional | `BOOLEAN` | `True` | Run depth estimation on GPU. Requires a CUDA-capable device. Falls back to CPU if unavailable. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `depth_map` | `IMAGE` | Output produced by the `depth_map` socket. |

### Practical notes

- The node returns `depth_map` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Optical Flow

**Internal key:** `RadianceOpticalFlow`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/motion.py`
**Function:** `analyze`

### What it does

Optical Flow.

### When to use it

Use `◎ Optical Flow` when the graph reaches the Optical Flow step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` | - | Batch of frames to analyze. |
| `preset` | Yes | `ENUM: Fast, Medium, Ultra` | `Medium` | - |
| `flow_scale` | Yes | `FLOAT` | `1.0` | Scale factor for output vectors. 1.0 = pixel units. |
| `visualize` | Yes | `BOOLEAN` | `False` | Outputs a color-coded visualization of the motion field. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `motion_vectors` | `IMAGE` | Output produced by the `motion_vectors` socket. |
| `visualization` | `IMAGE` | Output produced by the `visualization` socket. |
| `stats` | `STRING` | Output produced by the `stats` socket. |

### Practical notes

- The node returns `motion_vectors` (`IMAGE`), `visualization` (`IMAGE`), `stats` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Physical Motion Blur

**Internal key:** `RadianceMotionBlur`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/motion_blur.py`
**Function:** `apply`

### What it does

Physical Motion Blur.

### When to use it

Use `◎ Physical Motion Blur` when the graph reaches the Physical Motion Blur step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `motion_vectors` | Yes | `IMAGE` | - | 32-bit UV vectors from Radiance Optical Flow. |
| `shutter_angle` | Yes | `FLOAT` | `180.0` | Standard cinema is 180°. Higher = more blur. 360° = full frame motion blur. |
| `samples` | Yes | `INT` | `8` | Number of sub-frame integration samples. Higher = smoother streaks. |
| `energy_conservation` | Yes | `BOOLEAN` | `True` | Ensures that bright highlights maintain their intensity over the blur area. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Lens Distortion

**Internal key:** `RadianceLensDistortion`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/optics.py`
**Function:** `apply`

### What it does

Lens Distortion.

### When to use it

Use `◎ Lens Distortion` when the graph reaches the Lens Distortion step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `k1` | Yes | `FLOAT` | `0.0` | Primary radial distortion. k1<0 = barrel, k1>0 = pincushion. |
| `k2` | Yes | `FLOAT` | `0.0` | Secondary radial distortion. Affects extreme corners. Use sparingly. |
| `scale` | Yes | `FLOAT` | `1.0` | Uniform scale applied after distortion. Use to crop black borders. |
| `center_x` | Yes | `FLOAT` | `0.5` | Optical center X (0.5 = image center). |
| `center_y` | Yes | `FLOAT` | `0.5` | Optical center Y (0.5 = image center). |
| `padding_mode` | Yes | `ENUM: zeros, reflection, border` | `zeros` | Edge fill: zeros=black, reflection=mirrored, border=edge-clamped. |
| `invert` | Yes | `BOOLEAN` | `False` | Invert warp for plate undistortion. Uses exact closed-form inverse — valid for all k1/k2 combinations except the degenerate singularity at 1+k1·r²+k2·r⁴=0 (clamped automatically). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `st_map` | `IMAGE` | Output produced by the `st_map` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `st_map` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Chromatic Aberration

**Internal key:** `RadianceChromaticAberration`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/optics.py`
**Function:** `apply`

### What it does

Chromatic Aberration.

### When to use it

Use `◎ Chromatic Aberration` when the graph reaches the Chromatic Aberration step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `shift_r` | Yes | `FLOAT` | `0.005` | Radial scale for Red channel. Positive = pushed outward. |
| `shift_g` | Yes | `FLOAT` | `0.0` | Radial scale for Green channel. Typically 0 (reference channel). |
| `shift_b` | Yes | `FLOAT` | `-0.005` | Radial scale for Blue channel. Negative = pulled inward. |
| `center_x` | Yes | `FLOAT` | `0.5` | Horizontal center of the distortion/effect (0.0 = left, 1.0 = right). |
| `center_y` | Yes | `FLOAT` | `0.5` | Vertical center of the effect (0.0 = top, 1.0 = bottom). |
| `invert` | Yes | `BOOLEAN` | `False` | Invert shift for CA removal / undistortion pass. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Anamorphic Streaks

**Internal key:** `RadianceAnamorphicStreaks`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/optics.py`
**Function:** `apply`

### What it does

Anamorphic Streaks.

### When to use it

Use `◎ Anamorphic Streaks` when the graph reaches the Anamorphic Streaks step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `threshold` | Yes | `FLOAT` | `1.0` | Luminance threshold above which highlights generate streaks. |
| `streak_length` | Yes | `INT` | `64` | Maximum streak tail length in pixels. |
| `streak_color_r` | Yes | `FLOAT` | `0.0` | Red component of the anamorphic streak colour. Values > 1.0 produce HDR-energy streaks. |
| `streak_color_g` | Yes | `FLOAT` | `0.5` | Green component of the anamorphic streak colour. |
| `streak_color_b` | Yes | `FLOAT` | `1.0` | Streak color tint. Default is the cool blue-cyan typical of anamorphic lenses. |
| `intensity` | Yes | `FLOAT` | `1.0` | Vignette or streak intensity. Values > 1.0 add HDR energy. |
| `streak_direction` | Yes | `ENUM: Horizontal, Vertical, Diagonal +45, Diagonal -45` | `Horizontal` | Direction of streak propagation. |
| `streak_falloff` | Yes | `FLOAT` | `3.0` | Exponential decay rate of the streak kernel. Higher = tighter falloff (shorter effective tail). Lower = longer, softer streak. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `streak_pass` | `IMAGE` | Output produced by the `streak_pass` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `streak_pass` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Film Grain (Simple)

**Internal key:** `RadianceFilmGrain`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/optics.py`
**Function:** `apply`

### What it does

Film Grain.

### When to use it

Use `◎ Film Grain (Simple)` when the graph reaches the Film Grain (Simple) step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `grain_size` | Yes | `FLOAT` | `1.0` | Base grain size in pixels (Gaussian sigma). 1.0 ≈ ISO 400 medium format. |
| `grain_strength` | Yes | `FLOAT` | `0.04` | Grain amplitude. 0.015=fine, 0.04=medium, 0.12=heavy. |
| `grain_size_r_offset` | Yes | `FLOAT` | `0.1` | R channel grain size offset from base (pixels). Red grain is typically coarser. |
| `grain_size_g_offset` | Yes | `FLOAT` | `0.0` | G channel grain size offset. Green has finest grain (most photo-sites). |
| `grain_size_b_offset` | Yes | `FLOAT` | `-0.1` | B channel grain size offset. Blue grain is typically finest. |
| `hdr_aware` | Yes | `BOOLEAN` | `True` | Scale grain amplitude by 1/√luminance (Poisson shot-noise model). Makes highlights finer-grained than shadows — physically correct. Disable for flat/uniform grain. |
| `seed` | Yes | `('INT', {'default': 0, 'min': 0, 'max': 2 ** 31 - 1, 'step': 1, 'tooltip': 'Random seed. 0 = random each run.'})` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Vignette

**Internal key:** `RadianceVignette`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/optics.py`
**Function:** `apply`

### What it does

Vignette.

### When to use it

Use `◎ Vignette` when the graph reaches the Vignette step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `strength` | Yes | `FLOAT` | `0.5` | Vignette intensity at corners. 0=no effect, 1=full black. |
| `power` | Yes | `FLOAT` | `2.0` | Falloff exponent. 2.0 = natural cos⁴ approximation. Higher = harder edge. |
| `center_x` | Yes | `FLOAT` | `0.5` | Horizontal center of the vignette (0 = left, 1 = right). |
| `center_y` | Yes | `FLOAT` | `0.5` | Vertical center of the vignette (0 = top, 1 = bottom). |
| `feather` | Yes | `FLOAT` | `1.0` | Radial feather: scales the normalized radius. >1 = pushes falloff inward. |
| `tint_r` | Optional | `FLOAT` | `1.0` | Red multiplier for vignetted (dark) areas. 1.0=neutral. |
| `tint_g` | Optional | `FLOAT` | `1.0` | Green channel multiplier for vignette tint. 1.0 = neutral. |
| `tint_b` | Optional | `FLOAT` | `1.0` | Blue multiplier for vignetted areas. >1 = cool shadow edges. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `vignette_mask` | `IMAGE` | Output produced by the `vignette_mask` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `vignette_mask` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.


## ◎ SAM Model Loader

**Internal key:** `RadianceSAMModelLoader`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Masking`  
**Source:** `nodes/vfx/masking.py`
**Function:** `load`

### What it does

SAM Model Loader.

### When to use it

Use `◎ SAM Model Loader` at the start of a vfx, masks, optics, and multipass graph when this data needs to be loaded once and reused downstream.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `model_name` | Yes | `ENUM: sam2.1_hiera_large.pt, sam3_hiera_large.pt, sam2.1_hiera_base.pt` | - | - |
| `device` | Yes | `ENUM: cuda, cpu, mps` | `cuda` | - |
| `offload_to_cpu` | Yes | `BOOLEAN` | `False` | - |
| `dtype` | Yes | `ENUM: float16, bfloat16, float32` | `float16` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `sam_model` | `SAM_MODEL` | Output produced by the `sam_model` socket. |

### Practical notes

- The node returns `sam_model` (`SAM_MODEL`).
- Preview the mask output directly before using it to crop, inpaint, or composite.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ SAM Mask Generator

**Internal key:** `RadianceSAMGenerator`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Masking`  
**Source:** `nodes/vfx/masking.py`
**Function:** `generate`

### What it does

SAM Mask Generator.

### When to use it

Use `◎ SAM Mask Generator` when the graph reaches the SAM Mask Generator step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `sam_model` | Yes | `SAM_MODEL` | - | - |
| `points` | Yes | `STRING` | `[[256, 256]]` | - |
| `point_labels` | Yes | `STRING` | `[1]` | - |
| `text_prompt` | Optional | `STRING` | `` | - |
| `bbox` | Optional | `STRING` | `` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `mask` | `MASK` | Output produced by the `mask` socket. |
| `masked_image` | `IMAGE` | Output produced by the `masked_image` socket. |

### Practical notes

- The node returns `mask` (`MASK`), `masked_image` (`IMAGE`).
- Preview the mask output directly before using it to crop, inpaint, or composite.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ SAM Multi-Mask Picker

**Internal key:** `RadianceMultiMaskVisualPicker`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Masking`  
**Source:** `nodes/vfx/masking.py`
**Function:** `pick`

### What it does

Multi-Mask Picker.

### When to use it

Use `◎ SAM Multi-Mask Picker` when the graph reaches the SAM Multi-Mask Picker step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `masks` | Yes | `MASK` | - | - |
| `picker_index` | Yes | `INT` | `0` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `selected_mask` | `MASK` | Output produced by the `selected_mask` socket. |

### Practical notes

- The node returns `selected_mask` (`MASK`).
- Preview the mask output directly before using it to crop, inpaint, or composite.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Linear Alpha Matting

**Internal key:** `RadianceLinearMatting`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Masking`  
**Source:** `nodes/vfx/masking.py`
**Function:** `apply`

### What it does

Linear Alpha Matting.

### When to use it

Use `◎ Linear Alpha Matting` when the graph reaches the Linear Alpha Matting step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `mask` | Yes | `MASK` | - | - |
| `method` | Yes | `ENUM: GuidedFilter, ViTMatte, RVM` | `GuidedFilter` | - |
| `trimap_dilation` | Yes | `INT` | `12` | - |
| `eps` | Yes | `FLOAT` | `0.0001` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `alpha_matte` | `MASK` | Output produced by the `alpha_matte` socket. |
| `foreground_image` | `IMAGE` | Output produced by the `foreground_image` socket. |

### Practical notes

- The node returns `alpha_matte` (`MASK`), `foreground_image` (`IMAGE`).
- Preview the mask output directly before using it to crop, inpaint, or composite.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Grain Matcher

**Internal key:** `RadianceHDRGrainMatcher`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Plate Prep`  
**Source:** `nodes/vfx/plate.py`
**Function:** `apply`

### What it does

HDR Grain Matcher.

### When to use it

Use `◎ HDR Grain Matcher` when the graph reaches the HDR Grain Matcher step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `target` | Yes | `IMAGE` | - | - |
| `reference` | Yes | `IMAGE` | - | - |
| `intensity` | Yes | `FLOAT` | `1.0` | - |
| `kernel_size` | Yes | `INT` | `3` | - |
| `r_gain` | Yes | `FLOAT` | `1.0` | - |
| `g_gain` | Yes | `FLOAT` | `1.0` | - |
| `b_gain` | Yes | `FLOAT` | `1.0` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `grained_image` | `IMAGE` | Output produced by the `grained_image` socket. |

### Practical notes

- The node returns `grained_image` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Subpixel Plate Stabilizer

**Internal key:** `RadianceSubpixelStabilizer`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Plate Prep`  
**Source:** `nodes/vfx/plate.py`
**Function:** `apply`

### What it does

Subpixel Stabilizer.

### When to use it

Use `◎ Subpixel Plate Stabilizer` when the graph reaches the Subpixel Plate Stabilizer step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `anchor_frame` | Yes | `INT` | `0` | - |
| `max_shift` | Yes | `INT` | `64` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `stabilized_sequence` | `IMAGE` | Output produced by the `stabilized_sequence` socket. |
| `displacements_xy` | `IMAGE` | Output produced by the `displacements_xy` socket. |

### Practical notes

- The node returns `stabilized_sequence` (`IMAGE`), `displacements_xy` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Inpaint Crop

**Internal key:** `RadianceHDRCrop`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Inpainting`  
**Source:** `nodes/vfx/inpaint.py`
**Function:** `apply`

### What it does

HDR Crop.

### When to use it

Use `◎ HDR Inpaint Crop` when the graph reaches the HDR Inpaint Crop step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `mask` | Yes | `MASK` | - | - |
| `context_padding` | Yes | `FLOAT` | `1.5` | - |
| `force_multiple` | Yes | `INT` | `16` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `cropped_image` | `IMAGE` | Output produced by the `cropped_image` socket. |
| `cropped_mask` | `MASK` | Output produced by the `cropped_mask` socket. |
| `stitcher_data` | `STITCHER_DATA` | Output produced by the `stitcher_data` socket. |

### Practical notes

- The node returns `cropped_image` (`IMAGE`), `cropped_mask` (`MASK`), `stitcher_data` (`STITCHER_DATA`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR Inpaint Stitch

**Internal key:** `RadianceHDRStitch`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Inpainting`  
**Source:** `nodes/vfx/inpaint.py`
**Function:** `apply`

### What it does

HDR Stitch.

### When to use it

Use `◎ HDR Inpaint Stitch` when the graph reaches the HDR Inpaint Stitch step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `original_image` | Yes | `IMAGE` | - | - |
| `cropped_image` | Yes | `IMAGE` | - | - |
| `cropped_mask` | Yes | `MASK` | - | - |
| `stitcher_data` | Yes | `STITCHER_DATA` | - | - |
| `blend_mode` | Yes | `ENUM: Linear_Laplacian, Linear_Gaussian, Standard` | `Linear_Laplacian` | - |
| `feather_radius` | Yes | `INT` | `16` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `stitched_image` | `IMAGE` | Output produced by the `stitched_image` socket. |
| `stitch_blend_mask` | `MASK` | Output produced by the `stitch_blend_mask` socket. |

### Practical notes

- The node returns `stitched_image` (`IMAGE`), `stitch_blend_mask` (`MASK`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Temporal Stitch Stabilizer

**Internal key:** `RadianceTemporalStitchStabilizer`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Inpainting`  
**Source:** `nodes/vfx/inpaint.py`
**Function:** `apply`

### What it does

Temporal Stitch Stabilizer.

### When to use it

Use `◎ Temporal Stitch Stabilizer` when the graph reaches the Temporal Stitch Stabilizer step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `masks` | Yes | `MASK` | - | - |
| `temporal_sigma` | Yes | `FLOAT` | `2.0` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `stabilized_masks` | `MASK` | Output produced by the `stabilized_masks` socket. |

### Practical notes

- The node returns `stabilized_masks` (`MASK`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Vector Mask Draw (Roto)

**Internal key:** `RadianceVectorMaskDraw`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Masking`  
**Source:** `nodes/vfx/roto.py`
**Function:** `draw`

### What it does

Vector Mask Draw.

### When to use it

Use `◎ Vector Mask Draw (Roto)` when the graph reaches the Vector Mask Draw (Roto) step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `width` | Yes | `INT` | `512` | - |
| `height` | Yes | `INT` | `512` | - |
| `shape_type` | Yes | `ENUM: Polygon, Bezier_Spline` | `Polygon` | - |
| `points_data` | Yes | `STRING` | `[[128, 128], [384, 128], [384, 384], [128, 384]]` | Paste JSON coordinate list or Nuke control points block. |
| `anti_alias_width` | Yes | `FLOAT` | `1.5` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `vector_mask` | `MASK` | Output produced by the `vector_mask` socket. |

### Practical notes

- The node returns `vector_mask` (`MASK`).
- Preview the mask output directly before using it to crop, inpaint, or composite.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Mask Propagator

**Internal key:** `RadianceVideoMaskPropagator`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX/Masking`  
**Source:** `nodes/vfx/roto.py`
**Function:** `propagate`

### What it does

Video Mask Propagator.

### When to use it

Use `◎ Video Mask Propagator` when the graph reaches the Video Mask Propagator step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `masks` | Yes | `MASK` | - | - |
| `flow_vectors` | Yes | `IMAGE` | - | 32-bit flow vectors from Radiance Optical Flow. |
| `propagation_mode` | Yes | `ENUM: Forward, Backward, Bidirectional` | `Bidirectional` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `propagated_masks` | `MASK` | Output produced by the `propagated_masks` socket. |

### Practical notes

- The node returns `propagated_masks` (`MASK`).
- Preview the mask output directly before using it to crop, inpaint, or composite.
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Multipass: Master VFX Extractor

**Internal key:** `RadianceMultipassMaster`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/multipass/master.py`
**Function:** `extract`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Multipass: Master VFX Extractor` when the graph reaches the Multipass: Master VFX Extractor step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `beauty` | Yes | `IMAGE` | - | - |
| `depth_map` | Optional | `IMAGE` | - | - |
| `normal_map` | Optional | `IMAGE` | - | - |
| `prev_frame` | Optional | `IMAGE` | - | - |
| `luma_weights` | Optional | `(list(_LUMA_WEIGHTS.keys()), {'default': 'Rec.709 / sRGB'})` | - | - |
| `auto_depth_model` | Optional | `(_AUTO_DEPTH_CHOICES, {'default': 'disabled'})` | - | - |
| `depth_near_is_white` | Optional | `BOOLEAN` | `True` | - |
| `depth_scale` | Optional | `FLOAT` | `10.0` | - |
| `fov_degrees` | Optional | `FLOAT` | `60.0` | - |
| `dsine_model_path` | Optional | `STRING` | `auto` | - |
| `normal_strength` | Optional | `FLOAT` | `2.0` | - |
| `normal_convention` | Optional | `(_NORMAL_CONVENTIONS, {'default': 'OpenGL (Y-Up)'})` | - | - |
| `albedo_shading_radius` | Optional | `FLOAT` | `80.0` | - |
| `albedo_eps` | Optional | `FLOAT` | `0.001` | - |
| `specular_floor` | Optional | `BOOLEAN` | `True` | - |
| `roughness_fine_radius` | Optional | `FLOAT` | `2.0` | - |
| `roughness_coarse_radius` | Optional | `FLOAT` | `15.0` | - |
| `transmission_sensitivity` | Optional | `FLOAT` | `2.0` | - |
| `highpass_radius` | Optional | `FLOAT` | `8.0` | - |
| `highpass_strength` | Optional | `FLOAT` | `1.0` | - |
| `highpass_contrast` | Optional | `FLOAT` | `1.0` | - |
| `shadow_threshold` | Optional | `FLOAT` | `0.2` | - |
| `highlight_threshold` | Optional | `FLOAT` | `0.75` | - |
| `mask_softness` | Optional | `FLOAT` | `0.15` | - |
| `ao_radius` | Optional | `FLOAT` | `15.0` | - |
| `ao_strength` | Optional | `FLOAT` | `1.0` | - |
| `ao_samples` | Optional | `INT` | `8` | - |
| `lk_window_radius` | Optional | `INT` | `7` | - |
| `motion_coherence` | Optional | `FLOAT` | `0.5` | - |
| `object_id_segments` | Optional | `INT` | `16` | - |
| `object_id_spatial_weight` | Optional | `FLOAT` | `0.25` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passes` | `RADIANCE_PASSES` | Output produced by the `passes` socket. |
| `beauty` | `IMAGE` | Output produced by the `beauty` socket. |
| `albedo` | `IMAGE` | Output produced by the `albedo` socket. |
| `normal_map` | `IMAGE` | Output produced by the `normal_map` socket. |
| `depth` | `IMAGE` | Output produced by the `depth` socket. |
| `roughness` | `IMAGE` | Output produced by the `roughness` socket. |
| `specular` | `IMAGE` | Output produced by the `specular` socket. |
| `metallic` | `IMAGE` | Output produced by the `metallic` socket. |
| `ao` | `IMAGE` | Output produced by the `ao` socket. |
| `emission` | `IMAGE` | Output produced by the `emission` socket. |
| `transmission` | `IMAGE` | Output produced by the `transmission` socket. |
| `highpass` | `IMAGE` | Output produced by the `highpass` socket. |
| `world_position` | `IMAGE` | Output produced by the `world_position` socket. |
| `curvature` | `IMAGE` | Output produced by the `curvature` socket. |
| `shadow_mask` | `IMAGE` | Output produced by the `shadow_mask` socket. |
| `midtone_mask` | `IMAGE` | Output produced by the `midtone_mask` socket. |
| `highlight_mask` | `IMAGE` | Output produced by the `highlight_mask` socket. |
| `reflection_mask` | `IMAGE` | Output produced by the `reflection_mask` socket. |
| `motion_vector` | `IMAGE` | Output produced by the `motion_vector` socket. |
| `object_id` | `IMAGE` | Output produced by the `object_id` socket. |

### Practical notes

- The node returns `passes` (`RADIANCE_PASSES`), `beauty` (`IMAGE`), `albedo` (`IMAGE`), `normal_map` (`IMAGE`), `depth` (`IMAGE`), `roughness` (`IMAGE`), `specular` (`IMAGE`), `metallic` (`IMAGE`), `ao` (`IMAGE`), `emission` (`IMAGE`), `transmission` (`IMAGE`), `highpass` (`IMAGE`), `world_position` (`IMAGE`), `curvature` (`IMAGE`), `shadow_mask` (`IMAGE`), `midtone_mask` (`IMAGE`), `highlight_mask` (`IMAGE`), `reflection_mask` (`IMAGE`), `motion_vector` (`IMAGE`), `object_id` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance EXR Passes Writer

**Internal key:** `RadianceEXRPassesWriter`  
**Category:** `FXTD STUDIOS/Radiance/◎ IO & Delivery`  
**Source:** `nodes/vfx/multipass/master.py`
**Function:** `write_passes`

### What it does

Writes or hands off the current result to a file, folder, preview, or DCC destination.

### When to use it

Use `◎ Radiance EXR Passes Writer` near the end of the graph after the image, sequence, or metadata is ready for delivery.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `passes` | Yes | `RADIANCE_PASSES` | - | - |
| `filename_prefix` | Yes | `STRING` | `radiance_vfx_passes` | - |
| `bit_depth` | Yes | `ENUM: 16-bit Half Float, 32-bit Float` | `16-bit Half Float` | - |
| `compression` | Yes | `(compressions, {'default': 'ZIP'})` | - | - |
| `output_path` | Optional | `STRING` | `` | - |
| `remote_path` | Optional | `STRING` | `` | - |
| `frame_index` | Optional | `INT` | `1001` | - |
| `custom_metadata` | Optional | `STRING` | `` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `output_path` | `STRING` | Output produced by the `output_path` socket. |

### Practical notes

- The node returns `output_path` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Multipass: Real PBR Relight

**Internal key:** `RadianceMultipassRelight`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/multipass/relight_comp.py`
**Function:** `relight`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Multipass: Real PBR Relight` when the graph reaches the Multipass: Real PBR Relight step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `albedo` | Yes | `IMAGE` | - | - |
| `normal_map` | Yes | `IMAGE` | - | - |
| `beauty` | Optional | `IMAGE` | - | - |
| `roughness` | Optional | `IMAGE` | - | - |
| `metallic` | Optional | `IMAGE` | - | - |
| `specular` | Optional | `IMAGE` | - | - |
| `ao` | Optional | `IMAGE` | - | - |
| `alpha` | Optional | `IMAGE` | - | - |
| `shadow_mask` | Optional | `IMAGE` | - | - |
| `depth_map` | Optional | `IMAGE` | - | - |
| `normal_convention` | Optional | `(_NORMAL_INPUTS, {'default': 'OpenGL (Y-Up)'})` | - | - |
| `light_type` | Optional | `(_LIGHT_TYPES, {'default': 'Directional'})` | - | - |
| `light_x` | Optional | `FLOAT` | `-0.35` | - |
| `light_y` | Optional | `FLOAT` | `0.45` | - |
| `light_z` | Optional | `FLOAT` | `1.0` | - |
| `light_r` | Optional | `FLOAT` | `1.0` | - |
| `light_g` | Optional | `FLOAT` | `1.0` | - |
| `light_b` | Optional | `FLOAT` | `1.0` | - |
| `intensity` | Optional | `FLOAT` | `1.0` | - |
| `ambient` | Optional | `FLOAT` | `0.03` | - |
| `specular_intensity` | Optional | `FLOAT` | `1.0` | - |
| `depth_scale` | Optional | `FLOAT` | `10.0` | - |
| `mix_with_beauty` | Optional | `FLOAT` | `0.0` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `relit` | `IMAGE` | Output produced by the `relit` socket. |
| `diffuse_light` | `IMAGE` | Output produced by the `diffuse_light` socket. |
| `specular_light` | `IMAGE` | Output produced by the `specular_light` socket. |
| `lighting` | `IMAGE` | Output produced by the `lighting` socket. |
| `alpha` | `IMAGE` | Output produced by the `alpha` socket. |
| `relight_info` | `STRING` | Output produced by the `relight_info` socket. |

### Practical notes

- The node returns `relit` (`IMAGE`), `diffuse_light` (`IMAGE`), `specular_light` (`IMAGE`), `lighting` (`IMAGE`), `alpha` (`IMAGE`), `relight_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Multipass: VFX Composite

**Internal key:** `RadianceMultipassComposite`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/vfx/multipass/relight_comp.py`
**Function:** `composite`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Multipass: VFX Composite` when the graph reaches the Multipass: VFX Composite step in a vfx, masks, optics, and multipass workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `foreground` | Yes | `IMAGE` | - | - |
| `alpha` | Yes | `IMAGE` | - | - |
| `background` | Optional | `IMAGE` | - | - |
| `relit_foreground` | Optional | `IMAGE` | - | - |
| `foreground_depth` | Optional | `IMAGE` | - | - |
| `background_depth` | Optional | `IMAGE` | - | - |
| `shadow_mask` | Optional | `IMAGE` | - | - |
| `alpha_invert` | Optional | `BOOLEAN` | `False` | - |
| `premultiplied_input` | Optional | `BOOLEAN` | `False` | - |
| `depth_near_is_white` | Optional | `BOOLEAN` | `True` | - |
| `depth_bias` | Optional | `FLOAT` | `0.01` | - |
| `shadow_strength` | Optional | `FLOAT` | `0.35` | - |
| `light_wrap` | Optional | `FLOAT` | `0.0` | - |
| `light_wrap_radius` | Optional | `INT` | `8` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `composite` | `IMAGE` | Output produced by the `composite` socket. |
| `premultiplied_foreground` | `IMAGE` | Output produced by the `premultiplied_foreground` socket. |
| `holdout_mask` | `IMAGE` | Output produced by the `holdout_mask` socket. |
| `depth_matte` | `IMAGE` | Output produced by the `depth_matte` socket. |
| `comp_info` | `STRING` | Output produced by the `comp_info` socket. |

### Practical notes

- The node returns `composite` (`IMAGE`), `premultiplied_foreground` (`IMAGE`), `holdout_mask` (`IMAGE`), `depth_matte` (`IMAGE`), `comp_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
