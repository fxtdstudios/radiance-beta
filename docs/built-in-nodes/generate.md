[← Back to Radiance docs](../README.md)

# Generate, Loaders, and Sampling

Model loading, LoRA stacks, prompt conditioning, resolution setup, denoising, and sampling controls for Radiance generation workflows.

## Typical workflow

```text
Read Models -> Cinematic Prompt Encoder -> Resolution -> Sampler Pro -> HDR VAE Decode / Viewer
```

## Before you use these nodes

- Keep loader metadata and LoRA ratios connected through the graph so outputs are reproducible.
- Use the resolution node before sampling when dimensions must be exact.
- Denoise and HDR decode are late-stage operations; inspect the result before saving.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Radiance Sampler Pro](#radiance-sampler-pro) | `RadianceSamplerPro` | Advanced diffusion sampler with Radiance-oriented sigma and diagnostic outputs. |
| [◎ HDR VAE Decode](#hdr-vae-decode) | `RadianceHDRVAEDecode` | HDR VAE Decode. |
| [◎ LoRA Stack](#lora-stack) | `RadianceLoraStack` | Compose up to 5 LoRAs into a LORA_STACK for use with RadianceUnifiedLoader. |
| [◎ Radiance Read Models](#radiance-read-models) | `RadianceUnifiedLoader` | Main Radiance model loader for model, CLIP, VAE, LoRA stack, and metadata. |
| [◎ Video Loader](#video-loader) | `RadianceVideoLoader` | Alias — identical to RadianceUnifiedLoader. Kept for backward compatibility. |
| [◎ ControlNet Apply](#controlnet-apply) | `RadianceControlNetApply` | Advanced ControlNet application node for the Radiance suite. |
| [◎ HDR LoRA Loader](#hdr-lora-loader) | `RadianceHDRLoRALoader` | Load a Radiance HDR LoRA .safetensors file. |
| [◎ HDR LoRA Apply](#hdr-lora-apply) | `RadianceHDRLoRAApply` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Cinematic Prompt Encoder](#cinematic-prompt-encoder) | `RadianceCinematicPromptEncoder` | Encodes image or latent data into the representation expected by the next processing stage. |
| [◎ Regional Prompt](#regional-prompt) | `RadianceRegionalPrompt` | Regional Prompt. |
| [◎ Regional Grid](#regional-grid) | `RadianceRegionalGrid` | Regional Grid. |
| [◎ Resolution](#resolution) | `RadianceResolution` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Denoise](#denoise) | `RadianceDenoise` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |

## ◎ Radiance Sampler Pro

**Internal key:** `RadianceSamplerPro`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes_sampler.py`
**Function:** `sample`

### What it does

Advanced diffusion sampler with Radiance-oriented sigma and diagnostic outputs.

### When to use it

Use `◎ Radiance Sampler Pro` when the graph reaches the Sampler Pro step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` | - | - |
| `positive` | Yes | `CONDITIONING` | - | - |
| `negative` | Yes | `CONDITIONING` | - | - |
| `latent_image` | Yes | `LATENT` | - | - |
| `preset` | Yes | `(WORKFLOW_PRESETS, {'default': 'None'})` | - | - |
| `steps` | Yes | `INT` | `20` | Total denoising steps. More steps = higher quality but slower. 20–30 is typical for most samplers. |
| `start_step` | Yes | `INT` | `0` | Start step (0 = beginning) |
| `end_step` | Yes | `INT` | `0` | End step (0 = use total steps) |
| `cfg` | Yes | `FLOAT` | `1.0` | - |
| `sampler` | Yes | `(comfy.samplers.KSampler.SAMPLERS,)` | - | - |
| `sampler_mode` | Yes | `(SamplerMode.ALL, {'default': SamplerMode.STANDARD})` | - | - |
| `phase_split` | Yes | `FLOAT` | `0.4` | - |
| `scheduler` | Yes | `(comfy.samplers.KSampler.SCHEDULERS,)` | - | - |
| `scheduler_mode` | Yes | `ENUM: Manual, Auto (Match Steps)` | `Manual` | - |
| `denoise` | Yes | `FLOAT` | `1.0` | - |
| `flux_shift` | Yes | `FLOAT` | `1.0` | - |
| `flux_guidance` | Yes | `FLOAT` | `3.5` | - |
| `flux_guidance_profile` | Yes | `ENUM: Static, Dynamic (Creative Start/End)` | `Static` | - |
| `seed` | Yes | `INT` | `0` | Random seed for reproducible results. -1 = random each run. |
| `pag_scale` | Yes | `FLOAT` | `0.0` | PAG strength (0=off). Perturbs attention for better prompt adherence. |
| `model_type` | Yes | `(MODEL_TYPES, {'default': 'auto'})` | - | - |
| `sigma_blend_steps` | Yes | `INT` | `0` | Smooth sigma transition steps at phase-shift boundary |
| `guidance_rescale_phi` | Yes | `FLOAT` | `0.0` | Guidance rescale (Imagen). 0=off, 0.7=recommended for SDXL. Prevents oversaturation at high CFG. |
| `preview_method` | Yes | `(PREVIEW_METHODS, {'default': 'None'})` | - | - |
| `noise_type` | Yes | `(NOISE_TYPES, {'default': 'Gaussian', 'tooltip': 'Noise generation algorithm. Perlin=coherent structure, Spectral=pink/1f noise, Brownian=video-correlated, Uniform=flat distribution.'})` | - | - |
| `conditioning_clip_target` | Yes | `(CLIP_TARGETS, {'default': 'Auto', 'tooltip': 'Route conditioning to a specific encoder slot (clip_l, clip_g, t5xxl). Auto = no routing.'})` | - | - |
| `add_noise` | Yes | `BOOLEAN` | `True` | Inject fresh noise at the start of sampling. Disable for img2img-style passes that should preserve structure. |
| `return_with_leftover_noise` | Yes | `BOOLEAN` | `False` | Return the latent with residual noise un-removed. Useful for multi-pass workflows. |
| `ays_schedule` | Yes | `BOOLEAN` | `False` | Use AYS (Align Your Steps) research-optimized sigma schedule. Best at 8-15 steps. |
| `tile_mode` | Yes | `BOOLEAN` | `False` | Enable tiled sampling for memory-efficient high-resolution generation. |
| `tile_size` | Yes | `INT` | `128` | Tile size in latent pixels (128 latent ≈ 1024px output with VAE factor 8). |
| `tile_overlap` | Yes | `INT` | `16` | Overlap between adjacent tiles to reduce seam artifacts. |
| `tile_blend` | Yes | `(TILE_BLEND_MODES, {'default': 'feather', 'tooltip': 'Seam blending method. feather=cosine fade, gaussian=bell curve, average=uniform.'})` | - | - |
| `terminal_sigma_to_zero` | Yes | `BOOLEAN` | `False` | Ensure terminal step reaches zero noise even on truncated image-to-image runs. Vital for Flow Matching models. |
| `force_exact_steps` | Yes | `BOOLEAN` | `False` | Ensure precise step count in image-to-image runs, adjusting calculations rather than purely truncating stages. |
| `refiner_model` | Optional | `MODEL` | - | - |
| `refiner_start_step` | Optional | `INT` | `20` | - |
| `noise_override` | Optional | `LATENT` | - | - |
| `sigmas_override` | Optional | `SIGMAS` | - | Inject a pre-computed sigma schedule. Bypasses all internal sigma computation. |
| `_js_export_btn` | Optional | `STRING` | `` | JS serialization placeholder (not user-editable). |
| `_js_import_btn` | Optional | `STRING` | `` | JS serialization placeholder (not user-editable). |
| `_js_preset_info` | Optional | `STRING` | `` | JS serialization placeholder (not user-editable). |
| `restart_count` | Optional | `INT` | `0` | Number of restart iterations at each restart_schedule sigma. 0 = disabled. 1–2 restarts add ~5% extra steps but measurably improve high-frequency detail on Flux and WAN. |
| `noise_alpha_start` | Optional | `FLOAT` | `1.0` | Blend weight of the selected noise_type at step 0. 1.0 = pure noise_type. Cosine-interpolates to noise_alpha_end across the denoising trajectory. Set <1 to blend structured noise with Gaussian (e.g. 0.8 Perlin → 0.0 Gaussian for video). |
| `noise_alpha_end` | Optional | `FLOAT` | `1.0` | Blend weight of the selected noise_type at the final step. Set lower than noise_alpha_start to fade structured noise into pure Gaussian in late denoising steps. |
| `custom_ays_anchors` | Optional | `SIGMAS` | - | Optional model-specific AYS anchor schedule. Overrides the built-in AYS tables when ays_schedule=True. Must be a monotonically decreasing tensor ending at 0. |
| `restart_schedule` | Optional | `SIGMAS` | - | Optional list of sigma levels at which to re-inject noise and re-denoise (Restart / IRES style). Improves fine detail at fixed step count. Requires restart_count > 0. |
| `sdr_reference` | Optional | `IMAGE` | - | - |
| `sdr_vae` | Optional | `VAE` | - | - |
| `sdr_blend` | Optional | `FLOAT` | `0.35` | - |
| `sdr_inject_steps` | Optional | `INT` | `6` | - |
| `sdr_decay` | Optional | `FLOAT` | `0.65` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `latent` | `LATENT` | Output produced by the `latent` socket. |
| `sigmas` | `SIGMAS` | Output produced by the `sigmas` socket. |
| `sigmas_remaining` | `SIGMAS` | Output produced by the `sigmas_remaining` socket. |
| `sigma_plot` | `IMAGE` | Output produced by the `sigma_plot` socket. |

### Practical notes

- The node returns `latent` (`LATENT`), `sigmas` (`SIGMAS`), `sigmas_remaining` (`SIGMAS`), `sigma_plot` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR VAE Decode

**Internal key:** `RadianceHDRVAEDecode`  
**Category:** `FXTD STUDIOS/Radiance/◎ HDR`  
**Source:** `nodes/generate/engine.py`
**Function:** `apply`

### What it does

HDR VAE Decode.

### When to use it

Use `◎ HDR VAE Decode` when the graph reaches the HDR VAE Decode step in a generate, loaders, and sampling workflow.

### Inputs

This node does not expose static `INPUT_TYPES` metadata that can be read without importing the full ComfyUI runtime.

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ LoRA Stack

**Internal key:** `RadianceLoraStack`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes_loader.py`
**Function:** `build_stack`

### What it does

Compose up to 5 LoRAs into a LORA_STACK for use with RadianceUnifiedLoader.

### When to use it

Use `◎ LoRA Stack` when the graph reaches the LoRA Stack step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `lora_stack` | Optional | `LORA_STACK` | `None` | Chain an upstream LORA_STACK before these LoRAs. |
| `lora_1` | Optional | `lora_slot('LoRA 1')` | - | - |
| `lora_1_model` | Optional | `str_slot('LoRA 1 model strength')` | - | - |
| `lora_1_clip` | Optional | `str_slot('LoRA 1 CLIP strength')` | - | - |
| `lora_2` | Optional | `lora_slot('LoRA 2')` | - | - |
| `lora_2_model` | Optional | `str_slot('LoRA 2 model strength')` | - | - |
| `lora_2_clip` | Optional | `str_slot('LoRA 2 CLIP strength')` | - | - |
| `lora_3` | Optional | `lora_slot('LoRA 3')` | - | - |
| `lora_3_model` | Optional | `str_slot('LoRA 3 model strength')` | - | - |
| `lora_3_clip` | Optional | `str_slot('LoRA 3 CLIP strength')` | - | - |
| `lora_4` | Optional | `lora_slot('LoRA 4')` | - | - |
| `lora_4_model` | Optional | `str_slot('LoRA 4 model strength')` | - | - |
| `lora_4_clip` | Optional | `str_slot('LoRA 4 CLIP strength')` | - | - |
| `lora_5` | Optional | `lora_slot('LoRA 5')` | - | - |
| `lora_5_model` | Optional | `str_slot('LoRA 5 model strength')` | - | - |
| `lora_5_clip` | Optional | `str_slot('LoRA 5 CLIP strength')` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `lora_stack` | `LORA_STACK` | Output produced by the `lora_stack` socket. |

### Practical notes

- The node returns `lora_stack` (`LORA_STACK`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Read Models

**Internal key:** `RadianceUnifiedLoader`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes_loader.py`
**Function:** `load_radiance_stack`

### What it does

Main Radiance model loader for model, CLIP, VAE, LoRA stack, and metadata.

### When to use it

Use `◎ Radiance Read Models` at the start of a generate, loaders, and sampling graph when this data needs to be loaded once and reused downstream.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `preset` | Yes | `(list(CHECKPOINT_PRESETS.keys()), {'default': 'Custom', 'tooltip': 'Quick-configure for common architectures. Overrides model_type, dtypes, offload_mode, and hints which CLIP slots are needed.'})` | - | - |
| `unet_name` | Yes | `(folder_paths.get_filename_list('diffusion_models'), {'tooltip': 'Main diffusion model (UNET / DiT / Transformer).'})` | - | - |
| `weight_dtype` | Yes | `(WEIGHT_DTYPES, {'default': 'default', 'tooltip': 'UNET weight precision. fp8_e4m3fn saves ~40% VRAM vs fp16.'})` | - | - |
| `model_type` | Yes | `(MODEL_TYPES, {'default': 'Auto-Detect', 'tooltip': "'Auto-Detect' reads the checkpoint's key names to determine architecture. Override manually if detection fails."})` | - | - |
| `vae_name` | Yes | `(folder_paths.get_filename_list('vae'), {'tooltip': 'VAE for encoding/decoding latents.'})` | - | - |
| `clip_l` | Optional | `clip_slot('CLIP-L (text encoder). Used by: SD1.5, SDXL, Flux, SD3.')` | - | - |
| `clip_g` | Optional | `clip_slot('CLIP-G (text encoder). Used by: SDXL, SD3, SD3.5.')` | - | - |
| `t5xxl` | Optional | `clip_slot('T5-XXL (text encoder). Used by: Flux, SD3, SD3.5, Wan, LTX, PixArt.')` | - | - |
| `llm_encoder` | Optional | `clip_slot('LLM encoder (ChatGLM3 etc.). Used by: Kolors, HunyuanVideo.')` | - | - |
| `text_projection` | Optional | `clip_slot('Text projection matrix. Used by: LTX Video.')` | - | - |
| `clip_dtype` | Optional | `(CLIP_DTYPES, {'default': 'default', 'tooltip': 'CLIP weight precision. Independent from UNET. For Flux T5XXL: fp8 saves ~4.7 GB vs fp16.'})` | - | - |
| `offload_mode` | Optional | `(OFFLOAD_MODES, {'default': 'none', 'tooltip': 'none = GPU only. cpu_offload = CLIP loaded to CPU RAM. sequential = enable ComfyUI sequential CPU offload (8–12 GB GPUs).'})` | - | - |
| `lora_stack` | Optional | `LORA_STACK` | `None` | Accept a LORA_STACK from RadianceLoraStack node. |
| `check_vram` | Optional | `ENUM: On, Off` | `On` | Estimate VRAM before load and warn if tight. |
| `use_cache` | Optional | `ENUM: On, Off` | `On` | Cache loaded models. Skips disk I/O when re-running with the same files. Cache auto-invalidates if files change. |
| `lora_on_error` | Optional | `ENUM: warn, raise` | `raise` | 'warn' skips failed LoRA and continues. 'raise' stops execution. |
| `auto_download` | Optional | `BOOLEAN` | `False` | If a selected model is missing, automatically download it from Radiance mirrors. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `MODEL` | `MODEL` | Output produced by the `MODEL` socket. |
| `CLIP` | `CLIP` | Output produced by the `CLIP` socket. |
| `VAE` | `VAE` | Output produced by the `VAE` socket. |
| `lora_stack` | `LORA_STACK` | Output produced by the `lora_stack` socket. |
| `model_meta` | `STRING` | Output produced by the `model_meta` socket. |

### Practical notes

- The node returns `MODEL` (`MODEL`), `CLIP` (`CLIP`), `VAE` (`VAE`), `lora_stack` (`LORA_STACK`), `model_meta` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Loader

**Internal key:** `RadianceVideoLoader`  
**Category:** `FXTD STUDIOS/Radiance`  
**Source:** `nodes_loader.py`

### What it does

Alias — identical to RadianceUnifiedLoader. Kept for backward compatibility.

### When to use it

Use `◎ Video Loader` at the start of a generate, loaders, and sampling graph when this data needs to be loaded once and reused downstream.

### Inputs

This node does not expose static `INPUT_TYPES` metadata that can be read without importing the full ComfyUI runtime.

### Outputs

This node does not declare named runtime outputs in the source catalog.

### Practical notes

- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ ControlNet Apply

**Internal key:** `RadianceControlNetApply`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes_loader.py`
**Function:** `apply_controlnet`

### What it does

Advanced ControlNet application node for the Radiance suite.

### When to use it

Use `◎ ControlNet Apply` when the graph reaches the ControlNet Apply step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `conditioning` | Yes | `CONDITIONING` | - | - |
| `control_net` | Yes | `CONTROL_NET` | - | - |
| `image` | Yes | `IMAGE` | - | - |
| `strength` | Yes | `FLOAT` | `1.0` | Global strength of the control effect. |
| `start_percent` | Yes | `FLOAT` | `0.0` | Percentage of the generation where control starts (0.0 = beginning). |
| `end_percent` | Yes | `FLOAT` | `1.0` | Percentage of the generation where control ends (1.0 = end). |
| `control_type` | Yes | `(['auto'] + list(UNION_CONTROLNET_TYPES.keys()), {'default': 'auto', 'tooltip': 'For Union ControlNets (like Flux), select the specific control mode (Canny, Depth, etc.).'})` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `conditioning` | `CONDITIONING` | Output produced by the `conditioning` socket. |

### Practical notes

- The node returns `conditioning` (`CONDITIONING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR LoRA Loader

**Internal key:** `RadianceHDRLoRALoader`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes/generate/lora.py`
**Function:** `load`

### What it does

Load a Radiance HDR LoRA .safetensors file.

### When to use it

Use `◎ HDR LoRA Loader` at the start of a generate, loaders, and sampling graph when this data needs to be loaded once and reused downstream.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `lora_path` | Yes | `STRING` | `` | Path to a Radiance HDR LoRA checkpoint (.safetensors or .pt). Leave blank to use the RADIANCE_HDR_LORA env var. |
| `fallback_compression_ratio` | Optional | `FLOAT` | `0.5` | Used when LoRA metadata does not contain compression_ratio. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `lora_dict` | `LORA_DICT` | Output produced by the `lora_dict` socket. |
| `compression_ratio` | `FLOAT` | Output produced by the `compression_ratio` socket. |
| `model_name` | `STRING` | Output produced by the `model_name` socket. |
| `metadata_json` | `STRING` | Output produced by the `metadata_json` socket. |

### Practical notes

- The node returns `lora_dict` (`LORA_DICT`), `compression_ratio` (`FLOAT`), `model_name` (`STRING`), `metadata_json` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ HDR LoRA Apply

**Internal key:** `RadianceHDRLoRAApply`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes/generate/lora.py`
**Function:** `apply`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ HDR LoRA Apply` when the graph reaches the HDR LoRA Apply step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` | - | - |
| `lora_dict` | Yes | `LORA_DICT` | - | - |
| `strength` | Yes | `FLOAT` | `1.0` | LoRA strength multiplier. 1.0 = trained weight. 0 = no effect. |
| `model_hint` | Optional | `STRING` | `` | Cross-checks the LoRA's trained model against this hint and warns if mismatched. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `MODEL` | `MODEL` | Output produced by the `MODEL` socket. |
| `compression_ratio` | `FLOAT` | Output produced by the `compression_ratio` socket. |

### Practical notes

- The node returns `MODEL` (`MODEL`), `compression_ratio` (`FLOAT`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Cinematic Prompt Encoder

**Internal key:** `RadianceCinematicPromptEncoder`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes/generate/prompt.py`
**Function:** `encode_cinematic`

### What it does

Encodes image or latent data into the representation expected by the next processing stage.

### When to use it

Use `◎ Cinematic Prompt Encoder` when the graph reaches the Cinematic Prompt Encoder step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `clip` | Yes | `CLIP` | - | CLIP model for encoding. |
| `base_prompt` | Optional | `STRING` | `A cinematic scene...` | Primary subject/scene description. |
| `style_preset` | Optional | `(cls.STYLE_PRESETS, {'default': '→ Classic Hollywood', 'tooltip': 'One-click style preset.'})` | - | - |
| `framing` | Optional | `(cls.FRAMING, {'default': 'Medium Shot (MS)', 'tooltip': 'Shot framing type.'})` | - | - |
| `camera_type` | Optional | `(cls.CAMERAS, {'default': 'ARRI Alexa 35', 'tooltip': 'Camera body.'})` | - | - |
| `lens_focal` | Optional | `(cls.LENSES, {'default': '50mm Standard Prime', 'tooltip': 'Lens + focal length.'})` | - | - |
| `aperture_dof` | Optional | `(cls.APERTURES, {'default': 'f/2.8 (Cinematic Separation)', 'tooltip': 'Depth of field.'})` | - | - |
| `lighting` | Optional | `(cls.LIGHTING, {'default': 'Cinematic Haze / Volumetric Fog', 'tooltip': 'Lighting style.'})` | - | - |
| `style_aesthetic` | Optional | `(cls.STYLES, {'default': 'Photorealistic (Raw)', 'tooltip': 'Visual aesthetic.'})` | - | - |
| `color_grading` | Optional | `(cls.COLOR_GRADING, {'default': 'None', 'tooltip': 'Color grading look.'})` | - | - |
| `negative_strength` | Optional | `ENUM: Off, Soft, Standard, Aggressive` | `Standard` | Auto-negative strength. 'Soft' is recommended for Flux. |
| `negative_prompt` | Optional | `STRING` | `` | Custom negative prompt. Appended after auto-negatives. |
| `model_meta` | Optional | `STRING` | `` | Optional JSON metadata from Radiance Read Models. When connected, architecture detection uses this before tokenizer heuristics. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `positive` | `CONDITIONING` | Output produced by the `positive` socket. |
| `negative` | `CONDITIONING` | Output produced by the `negative` socket. |
| `positive_text` | `STRING` | Output produced by the `positive_text` socket. |
| `negative_text` | `STRING` | Output produced by the `negative_text` socket. |
| `resolved_arch` | `STRING` | Output produced by the `resolved_arch` socket. |
| `token_count` | `INT` | Output produced by the `token_count` socket. |

### Practical notes

- The node returns `positive` (`CONDITIONING`), `negative` (`CONDITIONING`), `positive_text` (`STRING`), `negative_text` (`STRING`), `resolved_arch` (`STRING`), `token_count` (`INT`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Regional Prompt

**Internal key:** `RadianceRegionalPrompt`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes/generate/regional.py`
**Function:** `apply`

### What it does

Regional Prompt.

### When to use it

Use `◎ Regional Prompt` when the graph reaches the Regional Prompt step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `base_cond` | Yes | `CONDITIONING` | - | - |
| `region_cond` | Yes | `CONDITIONING` | - | - |
| `region_label` | Yes | `STRING` | `region_1` | Human-readable label for this region (used in JSON output). |
| `x` | Yes | `FLOAT` | `0.0` | Left edge of region as fraction of image width. |
| `y` | Yes | `FLOAT` | `0.0` | Top edge of region as fraction of image height. |
| `w` | Yes | `FLOAT` | `0.5` | Width of region as fraction of image width. |
| `h` | Yes | `FLOAT` | `0.5` | Height of region as fraction of image height. |
| `region_strength` | Yes | `FLOAT` | `1.0` | Conditioning weight for this region vs global. |
| `global_strength` | Yes | `FLOAT` | `0.5` | Weight of the global base conditioning passed through. |
| `merge_mode` | Yes | `ENUM: Additive, Replace` | `Additive` | Additive: region added on top of global (default, safe). Replace: region replaces global in its area. |
| `mask` | Optional | `MASK` | - | Optional. When connected, overrides x/y/w/h with the mask's bounding box. |
| `ip_image` | Optional | `IMAGE` | - | Optional IP-Adapter reference image for this region. When connected, the image's visual features are injected into the region conditioning alongside the text prompt. Requires an IP-Adapter-enabled model hook to be active. |
| `ip_weight` | Optional | `FLOAT` | `0.6` | Strength of the IP-Adapter image influence for this region. 0 = text-only, 1 = equal image+text, >1 = image-dominant. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `conditioning` | `CONDITIONING` | Output produced by the `conditioning` socket. |
| `region_info` | `STRING` | Output produced by the `region_info` socket. |

### Practical notes

- The node returns `conditioning` (`CONDITIONING`), `region_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Regional Grid

**Internal key:** `RadianceRegionalGrid`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes/generate/regional.py`
**Function:** `apply_grid`

### What it does

Regional Grid.

### When to use it

Use `◎ Regional Grid` when the graph reaches the Regional Grid step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `base_cond` | Yes | `CONDITIONING` | - | - |
| `clip` | Yes | `CLIP` | - | - |
| `grid_prompts` | Yes | `STRING` | `["subject in left area", "background on right"]` | JSON array of prompts, one per grid cell, row-major order. |
| `columns` | Yes | `INT` | `2` | Number of columns in the regional prompt grid. |
| `rows` | Yes | `INT` | `1` | Number of rows in the regional prompt grid. |
| `cell_strength` | Yes | `FLOAT` | `1.0` | Conditioning strength for each individual region cell. Higher values make the model follow regional prompts more closely. |
| `global_strength` | Yes | `FLOAT` | `0.3` | Conditioning strength for the global (full-image) prompt. Blended with cell conditioning at each step. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `conditioning` | `CONDITIONING` | Output produced by the `conditioning` socket. |
| `grid_info` | `STRING` | Output produced by the `grid_info` socket. |

### Practical notes

- The node returns `conditioning` (`CONDITIONING`), `grid_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Resolution

**Internal key:** `RadianceResolution`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes/generate/resolution.py`
**Function:** `generate`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Resolution` when the graph reaches the Resolution step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `preset` | Yes | `(PRESET_NAMES, {'default': 'Flux Square (1024×1024)', 'tooltip': "Resolution preset. Cinema, Social, Flux, SDXL, SD 1.5, WAN, WAN 2.1, LTX, HunyuanVideo, Hunyuan I2V, CogVideoX presets. Select 'Custom' to use manual width/height."})` | - | - |
| `width` | Yes | `INT` | `1024` | Custom width (only used when preset is 'Custom'). Auto-aligned to 8px (32px for LTX Video). |
| `height` | Yes | `INT` | `1024` | Custom height (only used when preset is 'Custom'). Auto-aligned to 8px (32px for LTX Video). |
| `orientation` | Yes | `(ORIENTATIONS, {'default': 'As Preset', 'tooltip': "Override orientation. 'As Preset' uses the preset's native orientation."})` | - | - |
| `model_type` | Yes | `(MODEL_TYPES, {'default': 'Auto (Flux 16ch)', 'tooltip': 'Determines latent channel count. Flux/SD3/Cosmos/Kontext = 16ch. SDXL/SD 1.5 = 4ch. Mochi = 12ch.'})` | - | - |
| `batch_size` | Yes | `INT` | `1` | Number of latent frames in batch. |
| `scale_factor` | Optional | `FLOAT` | `1.0` | Scale the resolution by this factor after preset/custom. 0.5 = half res, 2.0 = double res. Applied before alignment. |
| `latent_channels` | Optional | `INT` | `0` | Override latent channel count. 0 = use model_type default. Common: 4 (SD/SDXL), 12 (Mochi), 16 (Flux/SD3/Cosmos). Set manually for custom architectures. |
| `mp_target` | Optional | `FLOAT` | `0.0` | MEGAPIXEL TARGET: When > 0, auto-calculates W×H from this MP target and mp_aspect_ratio. Overrides preset and custom W/H. 0 = disabled. |
| `mp_aspect_ratio` | Optional | `(MP_ASPECT_RATIOS, {'default': '16:9', 'tooltip': 'Aspect ratio for megapixel target mode (only used when mp_target > 0).'})` | - | - |
| `enable_video` | Optional | `BOOLEAN` | `False` | Enable video sequence mode (replaces batch parameter). |
| `video_frames` | Optional | `INT` | `81` | Total number of video frames. WAN/WAN 2.1: must satisfy (4k+1) — e.g. 1, 5, 9, 13, 17, 21, 49, 81. A warning is logged if this constraint is violated. |
| `frame_rate` | Optional | `FLOAT` | `24.0` | Playback frame rate. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `latent` | `LATENT` | Output produced by the `latent` socket. |

### Practical notes

- The node returns `latent` (`LATENT`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Denoise

**Internal key:** `RadianceDenoise`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes/generate/denoise.py`
**Function:** `denoise`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Denoise` when the graph reaches the Denoise step in a generate, loaders, and sampling workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `filter_type` | Yes | `ENUM: Bilateral, Guided` | `Bilateral` | The core spatial denoising algorithm. Bilateral preserves edges; Guided runs faster, preserves finer detail, and avoids halos. |
| `d` | Yes | `INT` | `9` | Filter diameter. In Bilateral mode, this defines the pixel neighborhood. In Guided mode, this translates to window radius. |
| `sigmaColor` | Yes | `FLOAT` | `0.15` | Color similarity threshold. High = smoother, but can lose edge detail. For HDR images, scale is auto-adjusted if hdr_auto_sigma is ON. This is bypassed if auto_profiling is enabled. |
| `sigmaSpace` | Yes | `FLOAT` | `75.0` | Spatial distance threshold. Higher = smoother across wider neighborhoods but slower in Bilateral mode. |
| `hdr_auto_sigma` | Yes | `BOOLEAN` | `True` | Highly recommended for HDR! Automatically scales the color similarity threshold to match the local maximum range of the image, keeping denoise strength uniform. |
| `auto_profiling` | Yes | `BOOLEAN` | `False` | Enables fully automatic, hands-free noise profiling. Scans the image for the flatest region (sensor noise floor) and dynamically scales all thresholds. |
| `profile_multiplier` | Yes | `FLOAT` | `1.0` | Adjusts the strength of the automatically profiled noise signature. Raise if noise remains; lower if details get soft. |
| `luma_strength` | Yes | `FLOAT` | `1.0` | Multiplier for spatial denoise strength on brightness (Luma). Lower this (e.g. 0.2) to keep natural fine grain. |
| `chroma_strength` | Yes | `FLOAT` | `1.0` | Multiplier for spatial denoise strength on colors (Chroma). Raise this to wash away annoying chromatic noise. |
| `high_freq_denoise` | Yes | `FLOAT` | `1.0` | Denoise strength for fine details and high-frequency pixel grain. |
| `mid_freq_denoise` | Yes | `FLOAT` | `1.0` | Denoise strength for medium textures and compression artifacts. |
| `low_freq_denoise` | Yes | `FLOAT` | `0.5` | Denoise strength for large coarse gradient splotches. |
| `joint_chroma_guidance` | Yes | `BOOLEAN` | `True` | Enables Joint Guided filtering. Uses the sharp structural boundaries of the Luma channel to guide Chroma smoothing, preventing color bleeding. |
| `temporal_blend` | Yes | `FLOAT` | `0.0` | Enables multi-frame temporal de-flickering. 0.0 is off. Higher values blend adjacent frames to stabilize video. |
| `temporal_radius` | Yes | `INT` | `1` | Temporal search window size. 1 searches 1 prev/next frame. 2 searches 2 prev/next frames, etc. Higher values de-flicker better but are slower. |
| `temporal_threshold` | Yes | `FLOAT` | `0.05` | Flicker delta threshold. Lower values prevent ghosting/trailing by only blending static or slow-moving areas. |
| `motion_compensation` | Yes | `BOOLEAN` | `True` | Enables 9-directional block-matching motion compensation to align adjacent frames, preventing ghosting on moving objects. |
| `detail_recovery` | Yes | `FLOAT` | `0.0` | Blends high-frequency detail from the original image back into the denoised image to recover skin pores/grain. |
| `sharpen_strength` | Yes | `FLOAT` | `0.0` | Adds a subtle post-sharpening (unsharp mask) to recover perceived edge crispness. |
| `view_mode` | Yes | `ENUM: Denoised, Noise Residual, Luma (Y), Chroma (Cb/Cr)` | `Denoised` | Diagnostic view options. 'Noise Residual' is extremely helpful to see exactly what details are being removed. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
