[← Back to Radiance docs](../README.md)

# Generate, Loaders, and Sampling

Model loading, prompt construction, sampling, and HDR-aware latent decoding.

## Typical graph

```text
Radiance Loader → Cinematic Prompt Encoder → Radiance Sampler → HDR VAE Decode → Viewer
```

## Before you use these nodes

- **Radiance Sampler adapts to the model.** Widgets that do not apply to the
  detected architecture are hidden rather than ignored. If a control you expect
  is missing, check `model_meta` — the loader publishes what it detected.
- **`cfg` on Flux.** Flux models use a guidance embedding and want `cfg = 1.0`.
  Several of the sampler's extras (guidance rescale, CFG++) need `cfg > 1.0` to
  do anything, and will say so in the log.
- **HDR VAE decode needs a RUDRA decoder** matching your base model. Without
  one, decode falls back to the standard VAE and the result is display-referred,
  not scene-linear.
- **LoRA stack order matters.** Entries apply in list order; a style LoRA after
  a structural one behaves differently from the reverse.

## Known limitations

`PAG` here is a mid-block self-attention perturbation, not the full method from
Ahn et al. 2024 — there is no separate perturbed forward pass. Restart sampling
follows Xu et al. Alg. 2 as of 3.2.0, but it re-runs the whole tail of the
schedule, so it is not cheap.

## Nodes in this section (11)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [ControlNet](#controlnet) | `RadianceControlNetApply` | Apply a ControlNet conditioning signal to the Radiance sampler. |
| [Denoise](#denoise) | `RadianceDenoise` | Ultimate professional-grade 32-bit float spatial-temporal denoising node |
| [LoRA Stack](#lora-stack) | `RadianceLoraStack` | Compose up to 5 LoRAs into an accumulating LORA_STACK |
| [Loader](#loader) | `RadianceUnifiedLoader` | Universal loader v3.3 — streamlined to be extremely visual and modular |
| [Prompt](#prompt) | `RadianceCinematicPromptEncoder` | v3.2 — Professional cinematic encoder |
| [Regional Grid](#regional-grid) | `RadianceRegionalGrid` | Divide the canvas into a grid of independently prompted regions. |
| [Regional Prompt](#regional-prompt) | `RadianceRegionalPrompt` | Apply region-specific text prompts with spatial masks. |
| [Resolution](#resolution) | `RadianceResolution` | Professional resolution selector with internal preview card |
| [Sampler](#sampler) | `RadianceSamplerPro` | v3.0.0 — Universal diffusion sampler |
| [VAE Decode (HDR)](#vae-decode-hdr) | `RadianceHDRVAEDecode` | Decode HDR latents using a VAE conditioned for high-dynamic range. |
| [VAE Encode (HDR)](#vae-encode-hdr) | `RadianceHDRLatentEncoder` | Unified HDR latent encoder |

---

## ControlNet

**Node key:** `RadianceControlNetApply`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes_loader.py`  

Apply a ControlNet conditioning signal to the Radiance sampler.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `conditioning` | Yes | `CONDITIONING` |  |  |  |
| `control_net` | Yes | `CONTROL_NET` |  |  |  |
| `image` | Yes | `IMAGE` |  |  |  |
| `strength` | Yes | `FLOAT` | `1.0` | 0.0 – 10.0, step 0.05 | Global strength of the control effect. |
| `start_percent` | Yes | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.01 | Percentage of the generation where control starts (0.0 = beginning). |
| `end_percent` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 | Percentage of the generation where control ends (1.0 = end). |
| `control_type` | Yes | choice of `auto` | `auto` |  | For Union ControlNets (like Flux), select the specific control mode (Canny, Depth, etc.). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `conditioning` | `CONDITIONING` |  |

---

## Denoise

**Node key:** `RadianceDenoise`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes/generate/denoise.py`  

Ultimate professional-grade 32-bit float spatial-temporal denoising node. Splits Luma and Chroma, offers automatic hands-free noise profiling, applies 3-band multiscale frequency decomposition, provides joint guided chroma filtering, and implements multi-frame block-matching motion-compensated temporal stabilization.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `filter_type` | Yes | choice of `Bilateral`, `Guided` | `Bilateral` |  | The core spatial denoising algorithm. Bilateral preserves edges; Guided runs faster, preserves finer detail, and avoids halos. |
| `d` | Yes | `INT` | `9` | 1 – 50 | Filter diameter. In Bilateral mode, this defines the pixel neighborhood. In Guided mode, this translates to window radius. |
| `sigmaColor` | Yes | `FLOAT` | `0.15` | 0.0 – 10.0, step 0.01 | Color similarity threshold. High = smoother, but can lose edge detail. For HDR images, scale is auto-adjusted if hdr_auto_sigma is ON. This is bypassed if auto_profiling is enabled. |
| `sigmaSpace` | Yes | `FLOAT` | `75.0` | 0.1 – 500.0, step 0.5 | Spatial distance threshold. Higher = smoother across wider neighborhoods but slower in Bilateral mode. |
| `hdr_auto_sigma` | Yes | `BOOLEAN` | `True` |  | Highly recommended for HDR! Automatically scales the color similarity threshold to match the local maximum range of the image, keeping denoise strength uniform. |
| `auto_profiling` | Yes | `BOOLEAN` | `False` |  | Enables fully automatic, hands-free noise profiling. Scans the image for the flatest region (sensor noise floor) and dynamically scales all thresholds. |
| `profile_multiplier` | Yes | `FLOAT` | `1.0` | 0.1 – 5.0, step 0.1 | Adjusts the strength of the automatically profiled noise signature. Raise if noise remains; lower if details get soft. |
| `luma_strength` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | Multiplier for spatial denoise strength on brightness (Luma). Lower this (e.g. 0.2) to keep natural fine grain. |
| `chroma_strength` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | Multiplier for spatial denoise strength on colors (Chroma). Raise this to wash away annoying chromatic noise. |
| `high_freq_denoise` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | Denoise strength for fine details and high-frequency pixel grain. |
| `mid_freq_denoise` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | Denoise strength for medium textures and compression artifacts. |
| `low_freq_denoise` | Yes | `FLOAT` | `0.5` | 0.0 – 2.0, step 0.05 | Denoise strength for large coarse gradient splotches. |
| `joint_chroma_guidance` | Yes | `BOOLEAN` | `True` |  | Enables Joint Guided filtering. Uses the sharp structural boundaries of the Luma channel to guide Chroma smoothing, preventing color bleeding. |
| `temporal_blend` | Yes | `FLOAT` | `0.0` | 0.0 – 0.95, step 0.05 | Enables multi-frame temporal de-flickering. 0.0 is off. Higher values blend adjacent frames to stabilize video. |
| `temporal_radius` | Yes | `INT` | `1` | 1 – 4 | Temporal search window size. 1 searches 1 prev/next frame. 2 searches 2 prev/next frames, etc. Higher values de-flicker better but are slower. |
| `temporal_threshold` | Yes | `FLOAT` | `0.05` | 0.01 – 0.5, step 0.01 | Flicker delta threshold. Lower values prevent ghosting/trailing by only blending static or slow-moving areas. |
| `motion_compensation` | Yes | `BOOLEAN` | `True` |  | Enables 9-directional block-matching motion compensation to align adjacent frames, preventing ghosting on moving objects. |
| `detail_recovery` | Yes | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.05 | Blends high-frequency detail from the original image back into the denoised image to recover skin pores/grain. |
| `sharpen_strength` | Yes | `FLOAT` | `0.0` | 0.0 – 2.0, step 0.05 | Adds a subtle post-sharpening (unsharp mask) to recover perceived edge crispness. |
| `view_mode` | Yes | choice of `Denoised`, `Noise Residual`, `Luma (Y)`, `Chroma (Cb/Cr)` | `Denoised` |  | Diagnostic view options. 'Noise Residual' is extremely helpful to see exactly what details are being removed. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |

---

## LoRA Stack

**Node key:** `RadianceLoraStack`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes_loader.py`  

Compose up to 5 LoRAs into an accumulating LORA_STACK. Chain multiple stacks together. Feed into Radiance Read Models.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `lora_stack` | No | `LORA_STACK` |  |  | Chain an upstream LORA_STACK before these LoRAs. |
| `lora_1` | No | choice of `None` | `None` |  | LoRA 1 |
| `lora_1_model` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 1 model strength |
| `lora_1_clip` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 1 CLIP strength |
| `lora_2` | No | choice of `None` | `None` |  | LoRA 2 |
| `lora_2_model` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 2 model strength |
| `lora_2_clip` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 2 CLIP strength |
| `lora_3` | No | choice of `None` | `None` |  | LoRA 3 |
| `lora_3_model` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 3 model strength |
| `lora_3_clip` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 3 CLIP strength |
| `lora_4` | No | choice of `None` | `None` |  | LoRA 4 |
| `lora_4_model` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 4 model strength |
| `lora_4_clip` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 4 CLIP strength |
| `lora_5` | No | choice of `None` | `None` |  | LoRA 5 |
| `lora_5_model` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 5 model strength |
| `lora_5_clip` | No | `FLOAT` | `1.0` | -2.0 – 2.0, step 0.05 | LoRA 5 CLIP strength |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `lora_stack` | `LORA_STACK` |  |

---

## Loader

**Node key:** `RadianceUnifiedLoader`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes_loader.py`  

Universal loader v3.3 — streamlined to be extremely visual and modular. Auto-detects architecture, auto-tunes weights/offload, supports chainable LORA_STACK, and outputs JSON model metadata.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `preset` | Yes | choice of `Custom`, `AuraFlow`, `Chroma`, `Flux.1`, `Flux.1 (Low VRAM)`, `Flux.2`, `Flux.2 (Low VRAM)`, `Lumina2`, … (+5 more) | `Custom` |  | Quick-configure for common architectures. Overrides model_type, dtypes, offload_mode, and hints which CLIP slots are needed. |
| `unet_name` | Yes | choice (empty) |  |  | Main diffusion model (UNET / DiT / Transformer). |
| `weight_dtype` | Yes | choice of `default`, `fp8_e4m3fn`, `fp8_e5m2`, `fp16`, `bf16`, `fp32` | `default` |  | UNET weight precision. fp8_e4m3fn saves ~40% VRAM vs fp16. |
| `model_type` | Yes | choice of `Auto-Detect`, `flux`, `sd3`, `sd3.5`, `sdxl`, `sd1.5`, `lumina2`, `z_image`, … (+5 more) | `Auto-Detect` |  | 'Auto-Detect' reads the checkpoint's key names to determine architecture. Override manually if detection fails. |
| `vae_name` | Yes | choice of `Baked VAE (from UNET)` | `Baked VAE (from UNET)` |  | VAE for encoding/decoding latents. 'Baked VAE (from UNET)' extracts it from the checkpoint, for architectures whose standard release bundles the VAE into the main file instead of shipping it separately. |
| `clip_l` | No | choice of `None` | `None` |  | CLIP-L (text encoder). Used by: SD1.5, SDXL, Flux, SD3. |
| `clip_g` | No | choice of `None` | `None` |  | CLIP-G (text encoder). Used by: SDXL, SD3, SD3.5. |
| `t5xxl` | No | choice of `None`, `Baked (from UNET)` | `None` |  | T5-XXL (text encoder). Used by: Flux, SD3, SD3.5, Wan, PixArt, LTX (pre-2.3). 'Baked (from UNET)' loads it from the main checkpoint -- AuraFlow ships no standalone text encoder file. |
| `llm_encoder` | No | choice of `None` | `None` |  | LLM encoder. Used by: HunyuanVideo (Llava-Llama3), LTX 2.3 (Gemma 3), Lumina2 (Gemma-2), Z-Image (Qwen3), Flux.2 (Mistral-3/Qwen3). |
| `text_projection` | No | choice of `None`, `Baked (from UNET)` | `None` |  | Text projection matrix. Used by: LTX 2.3 (with Gemma 3 llm_encoder). 'Baked (from UNET)' loads it from the main LTX 2.3 checkpoint, like the native LTXV Audio Text Encoder Loader. |
| `clip_dtype` | No | choice of `default`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp32` | `default` |  | CLIP weight precision. Independent from UNET. For Flux T5XXL: fp8 saves ~4.7 GB vs fp16. |
| `offload_mode` | No | choice of `none`, `cpu_offload`, `sequential` | `none` |  | none = GPU only. cpu_offload = CLIP loaded to CPU RAM. sequential = enable ComfyUI sequential CPU offload (8–12 GB GPUs). |
| `lora_stack` | No | `LORA_STACK` |  |  | Accept a LORA_STACK from RadianceLoraStack node. |
| `check_vram` | No | choice of `On`, `Off` | `On` |  | Estimate VRAM before load and warn if tight. |
| `use_cache` | No | choice of `On`, `Off` | `On` |  | Cache loaded models. Skips disk I/O when re-running with the same files. Cache auto-invalidates if files change. |
| `lora_on_error` | No | choice of `warn`, `raise` | `raise` |  | 'warn' skips failed LoRA and continues. 'raise' stops execution. |
| `auto_download` | No | `BOOLEAN` | `False` |  | If a selected model is missing, automatically download it from Radiance mirrors. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `model` | `MODEL` |  |
| `clip` | `CLIP` |  |
| `vae` | `VAE` |  |
| `lora_stack` | `LORA_STACK` |  |
| `model_meta` | `STRING` |  |

---

## Prompt

**Node key:** `RadianceCinematicPromptEncoder`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes/generate/prompt.py`  

v3.2 — Professional cinematic encoder. Auto-detects architecture (T5/CLIP) and auto-tunes negatives and break token splits under the hood. No parameter clutter.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `clip` | Yes | `CLIP` |  |  | CLIP model for encoding. |
| `base_prompt` | No | `STRING` | `A cinematic scene...` | multiline | Primary subject/scene description. |
| `style_preset` | No | choice of `None (Custom)`, `→ Classic Hollywood`, `→ Film Noir`, `→ Sci-Fi Cinematic`, `→ Cyberpunk`, `→ Drama / Emotional`, `→ Epic Landscape`, `→ Portrait`, … (+21 more) | `None (Custom)` |  | One-click style preset. |
| `framing` | No | choice of `None`, `Extreme Close-Up (ECU)`, `Close-Up (CU)`, `Medium Close-Up (MCU)`, `Medium Shot (MS)`, `Medium Wide (MW)`, `Medium Full Shot (MFS)`, `Cowboy Shot (American Shot)`, … (+13 more) | `None` |  | Shot framing type. |
| `camera_type` | No | choice of `None`, `ARRI Alexa 65 (IMAX)`, `ARRI Alexa Mini LF`, `ARRI Alexa 35`, `Sony Venice 2`, `Sony FX9`, `Sony A7S III`, `RED V-Raptor XL`, … (+14 more) | `None` |  | Camera body. |
| `lens_focal` | No | choice of `None`, `14mm Ultra-Wide Angle`, `16mm Ultra-Wide Angle`, `24mm Wide Angle`, `28mm Wide Angle`, `35mm Classic Wide`, `40mm Semi-Wide`, `50mm Standard Prime`, … (+31 more) | `None` |  | Lens + focal length. |
| `aperture_dof` | No | choice of `None`, `f/0.95 (Razor Thin DoF)`, `f/1.2 (Dreamy Bokeh)`, `f/1.8 (Soft Background)`, `f/2.0 (Shallow Cinematic)`, `f/2.8 (Cinematic Separation)`, `f/4.0 (Balanced)`, `f/5.6 (Sharp Subject)`, … (+4 more) | `None` |  | Depth of field. |
| `lighting` | No | choice of `None`, `Rembrandt Lighting`, `Chiaroscuro (High Contrast)`, `Film Noir Lighting`, `Split Lighting`, `Butterfly Lighting`, `Paramount Lighting`, `Soft Window Light`, … (+18 more) | `None` |  | Lighting style. |
| `style_aesthetic` | No | choice of `None`, `Photorealistic (Raw)`, `Cinematic Movie Still`, `Hyper-Realism`, `Editorial Photography`, `National Geographic Style`, `Documentary Texture`, `Vintage 1990s VHS`, … (+23 more) | `None` |  | Visual aesthetic. |
| `color_grading` | No | choice of `None`, `Teal and Orange (Blockbuster)`, `Bleach Bypass (Gritty)`, `Technicolor (Vintage)`, `Cross Processed`, `Desaturated (Muted)`, `Vibrant High Contrast`, `Sepia Tone`, … (+7 more) | `None` |  | Color grading look. |
| `negative_strength` | No | choice of `Off`, `Soft`, `Standard`, `Aggressive` | `Standard` |  | Auto-negative strength. 'Soft' is recommended for Flux. |
| `negative_prompt` | No | `STRING` | `` | multiline | Custom negative prompt. Appended after auto-negatives. |
| `model_meta` | No | `STRING` | `` |  | Optional JSON metadata from Radiance Read Models. When connected, architecture detection uses this before tokenizer heuristics. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `positive` | `CONDITIONING` | Positive conditioning for the sampler. |
| `negative` | `CONDITIONING` | Negative conditioning for the sampler. |
| `positive_text` | `STRING` | Final positive prompt text that was encoded. |
| `negative_text` | `STRING` | Final negative prompt text that was encoded. |
| `resolved_arch` | `STRING` | Detected architecture used to choose prose vs CLIP-style prompting. |
| `token_count` | `INT` | Tokenizer-derived positive prompt token count after safety handling. |

---

## Regional Grid

**Node key:** `RadianceRegionalGrid`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes/generate/regional.py`  

Divide the canvas into a grid of independently prompted regions.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `base_cond` | Yes | `CONDITIONING` |  |  |  |
| `clip` | Yes | `CLIP` |  |  |  |
| `grid_prompts` | Yes | `STRING` | `["subject in left area", "background on right"]` | multiline | JSON array of prompts, one per grid cell, row-major order. |
| `columns` | Yes | `INT` | `2` | 1 – 8 | Number of columns in the regional prompt grid. |
| `rows` | Yes | `INT` | `1` | 1 – 8 | Number of rows in the regional prompt grid. |
| `cell_strength` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | Conditioning strength for each individual region cell. Higher values make the model follow regional prompts more closely. |
| `global_strength` | Yes | `FLOAT` | `0.3` | 0.0 – 2.0, step 0.05 | Conditioning strength for the global (full-image) prompt. Blended with cell conditioning at each step. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `conditioning` | `CONDITIONING` |  |
| `grid_info` | `STRING` |  |

---

## Regional Prompt

**Node key:** `RadianceRegionalPrompt`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes/generate/regional.py`  

Apply region-specific text prompts with spatial masks.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `base_cond` | Yes | `CONDITIONING` |  |  |  |
| `region_cond` | Yes | `CONDITIONING` |  |  |  |
| `region_label` | Yes | `STRING` | `region_1` |  | Human-readable label for this region (used in JSON output). |
| `x` | Yes | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.01 | Left edge of region as fraction of image width. |
| `y` | Yes | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.01 | Top edge of region as fraction of image height. |
| `w` | Yes | `FLOAT` | `0.5` | 0.01 – 1.0, step 0.01 | Width of region as fraction of image width. |
| `h` | Yes | `FLOAT` | `0.5` | 0.01 – 1.0, step 0.01 | Height of region as fraction of image height. |
| `region_strength` | Yes | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | Conditioning weight for this region vs global. |
| `global_strength` | Yes | `FLOAT` | `0.5` | 0.0 – 2.0, step 0.05 | Weight of the global base conditioning passed through. |
| `merge_mode` | Yes | choice of `Additive`, `Replace` | `Additive` |  | Additive: region added on top of global (default, safe). Replace: region replaces global in its area. |
| `mask` | No | `MASK` |  |  | Optional. When connected, overrides x/y/w/h with the mask's bounding box. |
| `ip_image` | No | `IMAGE` |  |  | Optional IP-Adapter reference image for this region. When connected, the image's visual features are injected into the region conditioning alongside the text prompt. Requires an IP-Adapter-enabled model hook to be active. |
| `ip_weight` | No | `FLOAT` | `0.6` | 0.0 – 1.5, step 0.05 | Strength of the IP-Adapter image influence for this region. 0 = text-only, 1 = equal image+text, >1 = image-dominant. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `conditioning` | `CONDITIONING` |  |
| `region_info` | `STRING` |  |

---

## Resolution

**Node key:** `RadianceResolution`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes/generate/resolution.py`  
**Output node** — runs even with nothing connected downstream.  

Professional resolution selector with internal preview card. Outputs empty LATENT for Flux/SDXL/SD/Cosmos/CogVideoX with correct channel count. 40+ cinema/social/AI/video presets. WAN frame count validation. LTX 32px alignment. Manual latent_channels override for custom architectures.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `preset` | Yes | choice of `Custom`, `4K DCI (4096×2160)`, `4K UHD (3840×2160)`, `2K DCI (2048×1080)`, `HD 1080p (1920×1080)`, `HD 720p (1280×720)`, `Anamorphic 2.39:1 (2048×856)`, `Anamorphic 2.39:1 (4096×1712)`, … (+9 more) | `Custom` |  | Resolution preset (Cinema or Social). The final width/height are automatically aligned for the selected 'model_type'. Select 'Custom' to use manual width/height. |
| `width` | Yes | `INT` | `1024` | 64 – 16384, step 8 | Custom width (only used when preset is 'Custom'). Auto-aligned to 8px (32px for LTX Video). |
| `height` | Yes | `INT` | `1024` | 64 – 16384, step 8 | Custom height (only used when preset is 'Custom'). Auto-aligned to 8px (32px for LTX Video). |
| `orientation` | Yes | choice of `As Preset`, `Landscape`, `Portrait`, `Square` | `As Preset` |  | Override orientation. 'As Preset' uses the preset's native orientation. |
| `model_type` | Yes | choice of `Manual`, `Flux / SD3 / Lumina2 / Z-Image (16ch)`, `SDXL / SD 1.5 / PixArt / Aura Flow (4ch)`, `Chroma (16ch)`, `Cosmos World (16ch)`, `CogVideoX (16ch)`, `Mochi (12ch)`, `LTXV (128ch)`, … (+4 more) | `Manual` |  | Drives pixel alignment, video-latent shape, frame-count rules, latent_format, and the Est. VRAM readout. Flux/SD3/Cosmos = 16ch. SDXL/SD 1.5 = 4ch. Mochi = 12ch. 'Manual': no alignment/frame-count constraints; use 'latent_channels' for experimental/unlisted models. Est. VRAM assumes a full load; actual usage may be lower with DynamicVRAM/CPU offload active. |
| `batch_size` | Yes | `INT` | `1` | 1 – 64 | Number of latent frames in batch. |
| `scale_factor` | No | `FLOAT` | `1.0` | 0.1 – 4.0, step 0.1 | Scale the resolution by this factor after preset/custom. 0.5 = half res, 2.0 = double res. Applied before alignment. |
| `latent_channels` | No | `INT` | `0` | 0 – 256 | Override latent channel count. 0 = use model_type default. Common: 4 (SD/SDXL), 12 (Mochi), 16 (Flux/SD3/Cosmos). Set manually for custom architectures. |
| `mp_target` | No | `FLOAT` | `0.0` | 0.0 – 64.0, step 0.1 | MEGAPIXEL TARGET: When > 0, auto-calculates W×H from this MP target and mp_aspect_ratio. Overrides preset and custom W/H. 0 = disabled. |
| `mp_aspect_ratio` | No | choice of `1:1`, `4:3`, `3:2`, `16:9`, `21:9`, `2.39:1`, `1.85:1`, `9:16`, … (+2 more) | `16:9` |  | Aspect ratio for megapixel target mode (only used when mp_target > 0). |
| `enable_video` | No | `BOOLEAN` | `False` |  | Enable video sequence mode (replaces batch parameter). |
| `frame_computation` | No | choice of `Manual (Frames)`, `Auto (Seconds)` | `Manual (Frames)` |  |  |
| `duration_seconds` | No | `FLOAT` | `5.0` | 0.1 – 120.0, step 0.1 | Target video duration in seconds (used when frame_computation = 'Auto (Seconds)'). |
| `video_frames` | No | `INT` | `81` | 1 – 100000 | Total number of video frames. 5D-latent models require (stride*k+1) — e.g. 4k+1 for WAN/HunyuanVideo (1, 5, 9, 13...), 8k+1 for LTXV (1, 9, 17...), 6k+1 for Mochi (1, 7, 13...). A warning is logged if this constraint is violated. Ignored when frame_computation = 'Auto (Seconds)'. |
| `frame_rate` | No | `FLOAT` | `24.0` | 1.0 – 120.0 | Playback frame rate. |

Hidden inputs supplied by ComfyUI: `unique_id`.

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `latent` | `LATENT` | Empty latent tensor at the selected resolution. |
| `width` | `INT` | Final image width (pixels). |
| `height` | `INT` | Final image height (pixels). |
| `channels` | `INT` | Latent channel count. |
| `info` | `STRING` | Resolution info string. |
| `frame_rate` | `FLOAT` | Playback frame rate. Always the widget value — never 0.0. |
| `frame_count` | `INT` | Total video frames (or batch size for images). |
| `latent_format` | `STRING` | Latent format string — wire to Sampler Pro latent_format input. |
| `duration_sec` | `FLOAT` | Duration in seconds (video_frames / frame_rate). 0.0 for images. |

---

## Sampler

**Node key:** `RadianceSamplerPro`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes_sampler.py`  

v3.0.0 — Universal diffusion sampler. Auto-detects model type (Flux, SD3, SDXL, WAN, LTX, HunyuanVideo, Lumina2, Chroma). Phase-shift sampling, AYS schedules, PAG, dynamic guidance, tiled sampling, multi-conditioning, noise types, refiner chain. Restart sampling (IRES style), noise alpha schedule, sigma plot output, custom AYS anchors, sigmas_remaining for multi-node chains. AVControl-style SDR reference conditioning: encode an SDR reference through the VAE and blend it into the initial latent + per-step post-CFG anchor for HDR structure preservation.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` |  |  |  |
| `positive` | Yes | `CONDITIONING` |  |  |  |
| `negative` | Yes | `CONDITIONING` |  |  |  |
| `latent_image` | Yes | `LATENT` |  |  |  |
| `preset` | Yes | choice of `Auto`, `Custom`, `[F] Flux txt2img`, `[F] Flux img2img`, `[F] Flux Inpaint`, `[F] Flux High-Res Fix`, `[F] Flux Fast (12 steps)`, `[F] Flux Quality (28 steps)`, … (+17 more) | `Auto` |  |  |
| `steps` | Yes | `INT` | `20` | 1 – 200 | Total denoising steps. More steps = higher quality but slower. 20–30 is typical for most samplers. |
| `start_step` | Yes | `INT` | `0` | 0 – 200 | Start step (0 = beginning) |
| `end_step` | Yes | `INT` | `0` | 0 – 200 | End step (0 = use total steps) |
| `cfg` | Yes | `FLOAT` | `1.0` | 0.0 – 20.0, step 0.1 |  |
| `sampler` | Yes | choice, supplied by the host at runtime |  |  |  |
| `sampler_mode` | Yes | choice of `Standard`, `Phase-Shift (Euler >> DPM)`, `Phase-Shift (Euler >> SGM)`, `CFG++ (Perpendicular)` | `Standard` |  |  |
| `phase_split` | Yes | `FLOAT` | `0.4` | 0.0 – 1.0, step 0.05 |  |
| `scheduler` | Yes | choice, supplied by the host at runtime |  |  |  |
| `scheduler_mode` | Yes | choice of `Manual`, `Auto (Match Steps)` | `Manual` |  |  |
| `denoise` | Yes | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `flux_shift` | Yes | `FLOAT` | `1.0` | 0.01 – 10.0, step 0.1 |  |
| `flux_guidance` | Yes | `FLOAT` | `3.5` | 0.0 – 20.0, step 0.1 |  |
| `flux_guidance_profile` | Yes | choice of `Static`, `Dynamic (Creative Start/End)` | `Static` |  |  |
| `seed` | Yes | `INT` | `0` | 0 – 18446744073709551615 | Random seed for reproducible results. Use the control below it (randomize / increment / fixed) to vary the seed between runs. |
| `pag_scale` | Yes | `FLOAT` | `0.0` | 0.0 – 5.0, step 0.1 | PAG strength (0=off). Perturbs attention for better prompt adherence. |
| `model_type` | Yes | choice of `auto`, `flux`, `flux2`, `flux2-klein`, `sd3`, `sd3.5`, `sdxl`, `sd1.5`, … (+11 more) | `auto` |  |  |
| `sigma_blend_steps` | Yes | `INT` | `0` | 0 – 10 | Smooth sigma transition steps at phase-shift boundary |
| `guidance_rescale_phi` | Yes | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.05 | Guidance rescale (Imagen). 0=off, 0.7=recommended for SDXL. Prevents oversaturation at high CFG. |
| `preview_method` | Yes | choice of `None`, `TAESD`, `Latent2RGB` | `None` |  |  |
| `noise_type` | Yes | choice of `Gaussian`, `Perlin`, `Uniform`, `Spectral`, `Brownian`, `Simplex`, `Voronoi`, `Curl` | `Gaussian` |  | Noise generation algorithm. Perlin=coherent structure, Spectral=pink/1f noise, Brownian=video-correlated, Uniform=flat distribution. |
| `conditioning_clip_target` | Yes | choice of `Auto`, `clip_l`, `clip_g`, `t5xxl` | `Auto` |  | Route conditioning to a specific encoder slot (clip_l, clip_g, t5xxl). Auto = no routing. |
| `add_noise` | Yes | `BOOLEAN` | `True` |  | Inject fresh noise at the start of sampling. Disable for img2img-style passes that should preserve structure. |
| `return_with_leftover_noise` | Yes | `BOOLEAN` | `False` |  | Return the latent with residual noise un-removed. Useful for multi-pass workflows. |
| `ays_schedule` | Yes | `BOOLEAN` | `False` |  | Use AYS (Align Your Steps) research-optimized sigma schedule. Best at 8-15 steps. |
| `tile_mode` | Yes | `BOOLEAN` | `False` |  | Enable tiled sampling for memory-efficient high-resolution generation. |
| `tile_size` | Yes | `INT` | `128` | 32 – 1024, step 32 | Tile size in latent pixels (128 latent ≈ 1024px output with VAE factor 8). |
| `tile_overlap` | Yes | `INT` | `16` | 0 – 256, step 8 | Overlap between adjacent tiles to reduce seam artifacts. |
| `tile_blend` | Yes | choice of `feather`, `average`, `gaussian` | `feather` |  | Seam blending method. feather=cosine fade, gaussian=bell curve, average=uniform. |
| `terminal_sigma_to_zero` | Yes | `BOOLEAN` | `False` |  | Ensure terminal step reaches zero noise even on truncated image-to-image runs. Vital for Flow Matching models. |
| `force_exact_steps` | Yes | `BOOLEAN` | `False` |  | Ensure precise step count in image-to-image runs, adjusting calculations rather than purely truncating stages. |
| `refiner_model` | No | `MODEL` |  |  |  |
| `refiner_start_step` | No | `INT` | `20` | 0 – 200 |  |
| `noise_override` | No | `LATENT` |  |  |  |
| `sigmas_override` | No | `SIGMAS` |  |  | Inject a pre-computed sigma schedule. Bypasses all internal sigma computation. |
| `_js_export_btn` | No | `STRING` | `` |  | JS serialization placeholder (not user-editable). |
| `_js_import_btn` | No | `STRING` | `` |  | JS serialization placeholder (not user-editable). |
| `_js_preset_info` | No | `STRING` | `` |  | JS serialization placeholder (not user-editable). |
| `restart_count` | No | `INT` | `0` | 0 – 4 | Number of restart iterations at each restart_schedule sigma. 0 = disabled. 1–2 restarts add ~5% extra steps but measurably improve high-frequency detail on Flux and WAN. |
| `noise_alpha_start` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.05 | Blend weight of the selected noise_type at step 0. 1.0 = pure noise_type. Cosine-interpolates to noise_alpha_end across the denoising trajectory. Set <1 to blend structured noise with Gaussian (e.g. 0.8 Perlin → 0.0 Gaussian for video). |
| `noise_alpha_end` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.05 | Blend weight of the selected noise_type at the final step. Set lower than noise_alpha_start to fade structured noise into pure Gaussian in late denoising steps. |
| `custom_ays_anchors` | No | `SIGMAS` |  |  | Optional model-specific AYS anchor schedule. Overrides the built-in AYS tables when ays_schedule=True. Must be a monotonically decreasing tensor ending at 0. |
| `restart_schedule` | No | `SIGMAS` |  |  | Optional list of sigma levels at which to re-inject noise and re-denoise (Restart / IRES style). Improves fine detail at fixed step count. Requires restart_count > 0. |
| `sdr_reference` | No | `IMAGE` |  |  |  |
| `sdr_vae` | No | `VAE` |  |  |  |
| `sdr_blend` | No | `FLOAT` | `0.35` | 0.0 – 1.0, step 0.05 |  |
| `sdr_inject_steps` | No | `INT` | `6` | 0 – 100 |  |
| `sdr_decay` | No | `FLOAT` | `0.65` | 0.0 – 1.0, step 0.05 |  |
| `model_meta` | No | `STRING` | `` |  | Optional: connect RadianceUnifiedLoader's model_meta output. Only used when preset='Auto'/'Custom' and model_type='auto'. Refines cfg/guidance/steps beyond what the loaded model's architecture alone can tell -- e.g. distinguishing Flux.2 Klein Base from Klein distilled, which are architecturally identical. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `latent` | `LATENT` | Denoised latent ready for VAE decode. |
| `sigmas` | `SIGMAS` | The full sigma schedule used — chain to another sampler or inspect. |
| `sigmas_remaining` | `SIGMAS` | Unused tail of the sigma schedule after end_step. Chain directly to a refiner or upscaler sampler. |
| `sigma_plot` | `IMAGE` | Visual plot of the sigma schedule as an IMAGE. Blank if matplotlib is unavailable. |

---

## VAE Decode (HDR)

**Node key:** `RadianceHDRVAEDecode`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes/generate/engine.py`  

Decode HDR latents using a VAE conditioned for high-dynamic range.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `samples` | Yes | `LATENT` |  |  |  |
| `vae` | Yes | `VAE` |  |  |  |
| `target_space` | Yes | choice of `Linear`, `ACEScg`, `ACES 2065-1`, `Rec.2020 Linear`, `sRGB`, `Raw`, `ARRI LogC3`, `ARRI LogC4`, … (+4 more) | `sRGB` |  | Output color space. 'sRGB' (default): correct for SaveImage, PreviewImage, and all standard ComfyUI nodes — output is display-ready [0,1] sRGB. 'Linear': scene-linear for VFX pipelines (Nuke, Resolve, Houdini). Requires hdr_output=True for true linear passthrough; with hdr_output=False the image is re-encoded to sRGB for ComfyUI compatibility. Log/ACEScg/Rec.2020 spaces are scene-referred — connect to a tonemap node before SaveImage. |
| `tile_size` | No | choice of `Auto`, `512`, `768`, `1024`, `1280`, `1536` | `Auto` |  | Tile size for decode. Auto queries VRAM. |
| `overlap` | No | `INT` | `128` | 32 – 256, step 16 | Overlap between tiles. 128px optimal for cosine blending. |
| `temporal_size` | No | `INT` | `0` | 0 – 256 | Temporal chunk size in LATENT frames (not pixel frames — unlike ComfyUI's native 'VAE Decode (Tiled)', which counts pixel frames and divides internally by the VAE's temporal compression). 0 = disabled (full video decoded at once). Splits video latents along the time axis to reduce peak VRAM during VAE decode. Useful when hitting OOM on long videos. For Mochi: try 4–6. For LTX-Video: try 8–16. Images (4D latents) are not affected. |
| `temporal_overlap` | No | `INT` | `0` | 0 – 32 | Overlap in latent frames between temporal chunks. 0 = no overlap. A small value (1–2) reduces temporal seams at chunk boundaries. Only active when temporal_size > 0. |
| `exposure_adjust` | No | `FLOAT` | `0.0` | -10.0 – 10.0, step 0.1 | Post-decode exposure in stops. |
| `alpha` | No | `IMAGE` |  |  | Alpha channel from Radiance VAE 4K Encode. |
| `hdr_mode` | No | choice of `Clip (SDR)`, `Soft Clip`, `Compress (Log)`, `Passthrough` | `Clip (SDR)` |  | Must match encode setting. 'Compress (Log)' (v2.3): true HDR — soft-shouldered, highlight-denoised, no hard clamp. Produces clean HDR output across the full log-curve dynamic range. 'Soft Clip': inverts tanh rolloff to recover highlights. 'Passthrough': extended sRGB range. |
| `display_tonemap` | No | choice of `Reinhard`, `ACES Filmic`, `None` | `None` |  | v2.3.8: display_tonemap is NOW THE SOLE TONEMAP CONTROL — independent of hdr_output. 'Reinhard' or 'ACES Filmic': tonemap ALWAYS fires for Compress(Log), even when hdr_output=True. Reinhard → smooth rolloff [0,∞)→[0,1), hue-preserving. ACES Filmic → filmic contrast, clips cleanly above ~10 lin. 'None': NO tonemap regardless of hdr_output. Scene-linear values far above 1.0 pass through — GUARANTEED OVEREXPOSURE in ComfyUI preview. Use only with an OCIO-aware downstream viewer. Ignored for all non-Compress(Log) hdr_modes. |
| `inverse_tonemap` | No | `BOOLEAN` | `False` |  | Expand SDR to HDR (recover highlights). |
| `target_stops` | No | `FLOAT` | `12.0` | 8.0 – 16.0, step 0.5 | Target dynamic range for inverse tonemap. |
| `export_rhdr` | No | `BOOLEAN` | `False` |  | Export .rhdr sidecar for Radiance Viewer. |
| `rhdr_precision` | No | choice of `f16`, `f32` | `f32` |  | RHDR export precision. 'f16' (fp16, default): smaller files, values capped at 65504 — adequate for most VFX material. 'f32' (fp32): full 32-bit range, ~2× file size — use for shots with extreme linear values (direct sun, fire VDBs). |
| `source_space` | No | choice of `Linear`, `ACEScg`, `ACES 2065-1`, `Rec.2020 Linear`, `sRGB`, `Raw`, `ARRI LogC3`, `ARRI LogC4`, … (+4 more) | `sRGB` |  | Source color space used during encode. ⚠ CRITICAL for hdr_mode='Compress (Log)': must exactly match the log curve selected at encode time. Wrong value = wrong decompression curve = incorrect colors even if exposure looks OK. Valid log spaces: ARRI LogC4, ARRI LogC3, Sony S-Log3, Panasonic V-Log, DaVinci Intermediate, RED Log3G10. If source_space is set to a non-log value (Linear, sRGB, ACEScg) with Compress(Log), a WARNING is logged and LogC4 fallback is used. For diffusion model output (SDXL/FLUX/SD1.5), use Clip(SDR) or Soft Clip instead — those models output sRGB, not log. |
| `hdr_output` | No | `BOOLEAN` | `False` |  | 32-BIT HDR OUTPUT — when True, skips the final [0,1] clamp so the output IMAGE tensor carries full 32-bit scene-linear values (>1.0 for bright highlights, <0.0 for below-black). Required for Linear/ACEScg/Log target spaces in HDR VFX pipelines. Disable when feeding SDR nodes (preview, PNG export, etc.). |
| `decode_noise_scale` | No | `FLOAT` | `0.0` | 0.0 – 0.1, step 0.001 | v2.4 — Decode-time noise injection. Mixes a small amount of Gaussian noise into the latent BEFORE VAE decode using lerp: noised = (1-scale)*latent + scale*noise. Breaks up coherent VAE decoder grid artifacts in flat highlight regions. Effect is most valuable for Compress (Log) mode where the log→linear decompression exponentially amplifies any residual decoder artifact in the highlight band. 0.0 = disabled (default — safe for existing workflows). Per-profile recommended starting points: ARRI LogC4: 0.018 Sony S-Log3: 0.018 ARRI LogC3: 0.025 (matches LTX default) Panasonic V-Log: 0.022 DaVinci Intermediate: 0.030 RED Log3G10: 0.035 Ignored for non-Compress(Log) hdr_modes. |
| `hdr_scale_factor` | No | `FLOAT` | `1.0` | 0.1 – 10.0, step 0.05 | Linear multiplier applied after decode. Only active for scene-referred target spaces (Linear, ACEScg, Log). Ignored (with a warning) for sRGB/Raw display-referred output to prevent blown highlights. |
| `rudra_decoder` | No | choice of `Disabled`, `Enabled` | `Disabled` |  | Enable to use the distilled RUDRA decoder weights instead of the standard VAE. Dramatically faster and handles high dynamic range without highlight noise or clamping. |
| `decoder_size` | No | choice of `rudra_turbo`, `rudra_full` | `rudra_turbo` |  | 'rudra_turbo': 2M param real-time dynamic range conditioned model. 'rudra_full': 32M param production-grade dynamic range conditioned model. |
| `model_meta` | No | `STRING` | `` |  | Optional: connect RadianceUnifiedLoader's model_meta output to resolve the exact model architecture for RUDRA, instead of guessing from the latent/VAE shape (which can't distinguish models that share the same VAE, e.g. Flux.2 Dev vs Flux.2 Klein). Only used when rudra_decoder='Enabled'. |
| `decode_mode` | No | choice of `Auto (Recommended)`, `Sampler (SDR-safe)`, `Direct HDR / RUDRA` | `Auto (Recommended)` |  | Auto uses direct HDR only when encode metadata survives; otherwise it is sampler-safe. Sampler mode always decodes standard diffusion latents without log inversion. Direct HDR / RUDRA forces LogC4 decode to scene-linear Linear output, disables display tonemapping, and preserves values above 1.0. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Decoded image tensor. |
| `metadata` | `STRING` | JSON string with decode settings and applied hdr_scale_factor. |

---

## VAE Encode (HDR)

**Node key:** `RadianceHDRLatentEncoder`  
**Menu:** `FXTD STUDIOS/Radiance/Generate`  
**Source:** `nodes/hdr/encoder.py`  

Unified HDR latent encoder.  Soft-Knee (LTX-aligned) or Log Calibration mode. Optional per-channel normalisation — wire channel_stats to RadianceHDRPerChannelDenorm.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `vae` | Yes | `VAE` |  |  |  |
| `mode` | Yes | choice of `Soft-Knee (LTX)`, `Log Calibration` | `Soft-Knee (LTX)` |  |  |
| `compression_ratio` | No | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.05 | [Soft-Knee] 0 = hard clamp, 1 = full Reinhard. Mirrors LTX-Video tone_map_compression_ratio. |
| `exposure_offset` | No | `FLOAT` | `0.0` | -10.0 – 10.0, step 0.1 | [Soft-Knee] EV offset applied in scene-linear before compression. |
| `energy_normalization` | No | `FLOAT` | `1.0` | 0.1 – 5.0, step 0.1 | [Log Calibration] Scale applied before log1p encoding. Higher values compress brighter highlights more aggressively. |
| `normalize_channels` | No | `BOOLEAN` | `False` |  | Apply per-channel mean/std normalisation before VAE encode (mirrors LTX-Video vae_per_channel_normalize). Wire channel_stats → RadianceHDRPerChannelDenorm to invert. |
| `norm_center` | No | `FLOAT` | `3.0` | 1.0 – 8.0, step 0.5 | [Channel Norm] Window half-width in σ mapped to [0,1]. 3.0 captures ±3σ — suitable for most HDR content. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `latent` | `LATENT` |  |
| `channel_stats` | `STRING` |  |

---
