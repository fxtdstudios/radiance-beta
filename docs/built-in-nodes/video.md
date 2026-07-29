[← Back to Radiance docs](../README.md)

# Video

Video loading, prompt building, sampling, routing, batch decode, temporal
conditioning and export.

## Typical graph

```text
Video Loader → Video Prompt → Video Sampler → Video Batch Decode → Video Export
```

## Before you use these nodes

- **Frame batches are tensors, not streams.** A 240-frame 4K RGBA float batch
  is roughly 31 GB. Use `max_video_frames` and `proxy_scale` on Read while you
  are building the graph.
- **Export needs ffmpeg.** Radiance looks for it on PATH first, then falls back
  to the binary that `imageio-ffmpeg` installs, so a pip install is enough on
  Windows. Set `RADIANCE_FFMPEG` to point at a specific build.
- **Temporal windows overlap.** `overlap_temporal` controls the blend between
  processing windows; 1 is the minimum and produced black frames before 3.2.0.

## Nodes in this section (14)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [I2V Pipeline](#i2v-pipeline) | `RadianceI2VPipeline` | End-to-end image-to-video generation pipeline with motion control. |
| [T2V Pipeline](#t2v-pipeline) | `RadianceT2VPipeline` | End-to-end text-to-video generation pipeline with HDR support. |
| [Video Assembler](#video-assembler) | `RadianceVideoAssembler` | Assemble processed video frames back into a temporal sequence. |
| [Video Batch Decode](#video-batch-decode) | `RadianceVideoBatchDecode` | Batch-decode video latents to pixel frames with memory management. |
| [Video Cond Merge](#video-cond-merge) | `RadianceVideoCondMerge` | Merge multiple video conditioning signals into a unified tensor. |
| [Video Export](#video-export) | `RadianceVideoExport` | Export generated video frames to a file with format and codec options. |
| [Video Frame Router](#video-frame-router) | `RadianceVideoFrameRouter` | Route individual video frames to different processing branches. |
| [Video HDR Conditioner](#video-hdr-conditioner) | `RadianceVideoHDRConditioner` | Condition a video model on HDR metadata for luminance-aware sampling. |
| [Video HDR Decode](#video-hdr-decode) | `RadianceVideoHDRDecode` | Decode video latents to HDR pixel frames with colour space handling. |
| [Video Latent Noise](#video-latent-noise) | `RadianceVideoLatentNoise` | Generate temporally-coherent latent noise for video initialisation. |
| [Video Loader](#video-loader) | `RadianceVideoLoader` | Video loader v3.3 — for LTX 2.3, Wan, HunyuanVideo, etc |
| [Video Mask Propagator](#video-mask-propagator) | `RadianceVideoMaskPropagator` |  |
| [Video Model Info](#video-model-info) | `RadianceVideoModelInfo` | Display configuration and parameter info for a loaded video model. |
| [Video Sampler](#video-sampler) | `RadianceVideoSampler` | Run the diffusion sampler to generate video latents from pre-built noise and conditioning. |

---

## I2V Pipeline

**Node key:** `RadianceI2VPipeline`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/t2v.py`  

End-to-end image-to-video generation pipeline with motion control.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` |  |  |  |
| `clip` | Yes | `CLIP` |  |  |  |
| `vae` | Yes | `VAE` |  |  |  |
| `reference_image` | Yes | `IMAGE` |  |  |  |
| `positive_prompt` | Yes | `STRING` | `smooth camera motion, cinematic HDR, 4K` | multiline |  |
| `negative_prompt` | Yes | `STRING` | `watermark, blurry, flickering, sdr` | multiline |  |
| `frames` | Yes | `INT` | `25` | 1 – 512 |  |
| `seed` | Yes | `INT` | `0` | 0 – 2147483648 |  |
| `dit_config` | No | `STRING` | `{}` |  |  |
| `character_conditioning` | No | `CONDITIONING` |  |  |  |
| `cfg_schedule_json` | No | `STRING` | `` |  |  |
| `i2v_strategy` | No | choice of `auto`, `first_frame_lock`, `concat_channels`, `clip_vision_inject`, `prepend_latent` | `auto` |  |  |
| `image_strength` | No | `FLOAT` | `0.85` | 0.0 – 1.0, step 0.01 | How strongly the reference image anchors the generation |
| `motion_strength` | No | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.01 | Amount of motion / temporal variation (0=nearly static) |
| `steps` | No | `INT` | `0` | 0 – 200 |  |
| `cfg` | No | `FLOAT` | `0.0` | 0.0 – 30.0, step 0.1 |  |
| `sampler_name` | No | choice of `euler`, `euler_ancestral`, `heun`, `heunpp2`, `dpm_2`, `dpm_2_ancestral`, `lms`, `dpm_fast`, … (+17 more) | `euler` |  |  |
| `scheduler` | No | choice of `normal`, `karras`, `exponential`, `sgm_uniform`, `simple`, `ddim_uniform`, `beta` | `normal` |  |  |
| `peak_nits` | No | choice of `100`, `203`, `400`, `600`, `1000`, `4000`, `10000` | `1000` |  |  |
| `target_gamut` | No | choice of `BT.2020`, `P3-D65`, `P3-DCI`, `BT.709`, `ACEScg` | `BT.2020` |  |  |
| `hdr_eotf` | No | choice of `PQ (ST.2084)`, `HLG (BT.2100)`, `Linear`, `sRGB / BT.1886` | `PQ (ST.2084)` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `video_latent` | `LATENT` |  |
| `preview_frames` | `IMAGE` |  |
| `pipeline_report` | `STRING` |  |

---

## T2V Pipeline

**Node key:** `RadianceT2VPipeline`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/t2v.py`  

End-to-end text-to-video generation pipeline with HDR support.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` |  |  |  |
| `clip` | Yes | `CLIP` |  |  |  |
| `vae` | Yes | `VAE` |  |  |  |
| `positive_prompt` | Yes | `STRING` | `cinematic HDR video, stunning visuals, 4K, film grain` | multiline |  |
| `negative_prompt` | Yes | `STRING` | `watermark, blurry, low quality, sdr, flickering` | multiline |  |
| `width` | Yes | `INT` | `768` | 64 – 4096, step 8 |  |
| `height` | Yes | `INT` | `512` | 64 – 4096, step 8 |  |
| `frames` | Yes | `INT` | `25` | 1 – 512 |  |
| `seed` | Yes | `INT` | `0` | 0 – 2147483648 |  |
| `dit_config` | No | `STRING` | `{}` |  | JSON from RadianceVideoModelInfo — sets model-specific defaults |
| `character_conditioning` | No | `CONDITIONING` |  |  |  |
| `cfg_schedule_json` | No | `STRING` | `` |  | JSON float array from RadianceAudioCFGSchedule |
| `steps` | No | `INT` | `0` | 0 – 200 | 0 = use model default |
| `cfg` | No | `FLOAT` | `0.0` | 0.0 – 30.0, step 0.1 | 0 = use model default |
| `sampler_name` | No | choice of `euler`, `euler_ancestral`, `heun`, `heunpp2`, `dpm_2`, `dpm_2_ancestral`, `lms`, `dpm_fast`, … (+17 more) | `euler` |  |  |
| `scheduler` | No | choice of `normal`, `karras`, `exponential`, `sgm_uniform`, `simple`, `ddim_uniform`, `beta` | `normal` |  |  |
| `peak_nits` | No | choice of `100`, `203`, `400`, `600`, `1000`, `4000`, `10000` | `1000` |  |  |
| `target_gamut` | No | choice of `BT.2020`, `P3-D65`, `P3-DCI`, `BT.709`, `ACEScg` | `BT.2020` |  |  |
| `hdr_eotf` | No | choice of `PQ (ST.2084)`, `HLG (BT.2100)`, `Linear`, `sRGB / BT.1886` | `PQ (ST.2084)` |  |  |
| `hdr_strength` | No | `FLOAT` | `0.5` | 0.0 – 1.0, step 0.05 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `video_latent` | `LATENT` |  |
| `preview_frames` | `IMAGE` |  |
| `positive_cond` | `CONDITIONING` |  |
| `pipeline_report` | `STRING` |  |

---

## Video Assembler

**Node key:** `RadianceVideoAssembler`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/hdr.py`  

Assemble processed video frames back into a temporal sequence.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `frame` | Yes | `IMAGE` |  |  |  |
| `session_key` | Yes | `STRING` | `video_session_0` |  |  |
| `expected_total_frames` | Yes | `INT` | `24` | ≥ 1 |  |
| `flush` | No | `BOOLEAN` | `False` |  | Force output of accumulated frames now, even if incomplete |
| `reset` | No | `BOOLEAN` | `False` |  | Clear accumulated frames for this session key |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `video_image` | `IMAGE` |  |
| `frames_accumulated` | `INT` |  |
| `is_complete` | `BOOLEAN` |  |

---

## Video Batch Decode

**Node key:** `RadianceVideoBatchDecode`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/t2v.py`  

Batch-decode video latents to pixel frames with memory management.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `vae` | Yes | `VAE` |  |  |  |
| `latent` | Yes | `LATENT` |  |  |  |
| `dit_config` | No | `STRING` | `{}` |  |  |
| `tile_decode` | No | `BOOLEAN` | `False` |  | Tile the VAE decode to reduce VRAM on large videos |
| `tile_overlap` | No | `INT` | `64` | 0 – 256 | Pixel overlap between tiles (higher = smoother seams) |
| `output_linear` | No | `BOOLEAN` | `False` |  | Skip gamma correction — output linear-light frames for HDR pipeline |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `frames` | `IMAGE` |  |
| `frame_count` | `INT` |  |
| `decode_report` | `STRING` |  |

---

## Video Cond Merge

**Node key:** `RadianceVideoCondMerge`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/t2v.py`  

Merge multiple video conditioning signals into a unified tensor.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `text_conditioning` | Yes | `CONDITIONING` |  |  |  |
| `merge_mode` | Yes | choice of `concat`, `weighted`, `priority` | `concat` |  |  |
| `character_conditioning` | No | `CONDITIONING` |  |  |  |
| `hdr_conditioning` | No | `CONDITIONING` |  |  |  |
| `text_weight` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 |  |
| `character_weight` | No | `FLOAT` | `0.75` | 0.0 – 2.0, step 0.05 |  |
| `hdr_weight` | No | `FLOAT` | `0.5` | 0.0 – 2.0, step 0.05 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `merged_conditioning` | `CONDITIONING` |  |
| `merge_report` | `STRING` |  |

---

## Video Export

**Node key:** `RadianceVideoExport`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/t2v.py`  
**Output node** — runs even with nothing connected downstream.  

Export generated video frames to a file with format and codec options.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `frames` | Yes | `IMAGE` |  |  |  |
| `mode` | Yes | choice of `passthrough`, `hdr_decode`, `exr_sequence`, `preview_gif` | `passthrough` |  |  |
| `hdr_metadata_json` | No | `STRING` | `{"peak_nits":1000,"eotf":"PQ (ST.2084)"}` |  |  |
| `output_folder` | No | `STRING` | `` |  |  |
| `filename_prefix` | No | `STRING` | `radiance_video` |  |  |
| `fps` | No | `FLOAT` | `24.0` | 1.0 – 120.0 |  |
| `frame_offset` | No | `INT` | `0` | ≥ 0 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `frames` | `IMAGE` |  |
| `frame_count` | `INT` |  |
| `export_report` | `STRING` |  |

---

## Video Frame Router

**Node key:** `RadianceVideoFrameRouter`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/hdr.py`  

Route individual video frames to different processing branches.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `video_image` | Yes | `IMAGE` |  |  |  |
| `frame_index` | Yes | `INT` | `0` | 0 – 4096 |  |
| `wrap_index` | No | `BOOLEAN` | `True` |  | If frame_index >= total_frames, wrap around (modulo) |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `frame_image` | `IMAGE` |  |
| `frame_index` | `INT` |  |
| `total_frames` | `INT` |  |
| `passthrough` | `IMAGE` |  |

---

## Video HDR Conditioner

**Node key:** `RadianceVideoHDRConditioner`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/hdr.py`  

Condition a video model on HDR metadata for luminance-aware sampling.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `positive` | Yes | `CONDITIONING` |  |  |  |
| `peak_nits` | Yes | choice of `100`, `203`, `400`, `600`, `1000`, `4000`, `10000` | `1000` |  |  |
| `target_gamut` | Yes | choice of `BT.2020`, `P3-D65`, `P3-DCI`, `BT.709`, `ACEScg`, `ACES2065-1` | `BT.2020` |  |  |
| `eotf` | Yes | choice of `PQ (ST.2084)`, `HLG (BT.2100)`, `Linear`, `sRGB / BT.1886` | `PQ (ST.2084)` |  |  |
| `camera_move` | No | choice of `Handheld documentary`, `Locked off cinematic`, `Slow push-in`, `Drone aerial`, `Tracking shot`, `Static time-lapse`, `None` | `None` |  |  |
| `mood` | No | choice of `Golden hour`, `Blue hour / dusk`, `Night`, `Overcast flat`, `High contrast`, `Neon / cyberpunk`, `Natural daylight`, `None` | `None` |  |  |
| `extra_hdr_prompt` | No | `STRING` | `` | multiline | Additional HDR descriptors appended to conditioning tokens |
| `inject_metadata_embedding` | No | `BOOLEAN` | `True` |  | Add HDR metadata dict to conditioning['extra'] for compatible models |
| `token_strength` | No | `FLOAT` | `1.0` | 0.0 – 2.0, step 0.05 | Scale the appended token embeddings (1.0 = normal weight) |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `positive` | `CONDITIONING` |  |
| `hdr_metadata_json` | `STRING` |  |

---

## Video HDR Decode

**Node key:** `RadianceVideoHDRDecode`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/hdr.py`  

Decode video latents to HDR pixel frames with colour space handling.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  |  |
| `hdr_metadata_json` | Yes | `STRING` | `{"peak_nits":1000,"gamut":"BT.2020","eotf":"PQ (ST.2084)"}` |  | JSON from RadianceVideoHDRConditioner or manually entered |
| `tonemap` | Yes | choice of `Reinhard`, `Linear clip`, `Pass-through` | `Reinhard` |  |  |
| `exposure_compensation_ev` | No | `FLOAT` | `0.0` | -6.0 – 6.0, step 0.1 | EV adjustment before tone-mapping |
| `output_eotf` | No | choice of `PQ (ST.2084)`, `HLG (BT.2100)`, `Linear`, `sRGB / BT.1886` | `PQ (ST.2084)` |  |  |
| `sdr_preview_nits` | No | `FLOAT` | `100.0` | 1.0 – 203.0 | Scale factor for the SDR preview output |
| `gamut_clip` | No | `BOOLEAN` | `True` |  | Hard-clip out-of-gamut values before EOTF encode |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `hdr_image` | `IMAGE` |  |
| `sdr_preview` | `IMAGE` |  |
| `decode_report` | `STRING` |  |

---

## Video Latent Noise

**Node key:** `RadianceVideoLatentNoise`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/t2v.py`  

Generate temporally-coherent latent noise for video initialisation.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `dit_config` | Yes | `STRING` | `{}` |  | JSON from RadianceVideoModelInfo |
| `width` | Yes | `INT` | `512` | 64 – 4096, step 8 |  |
| `height` | Yes | `INT` | `512` | 64 – 4096, step 8 |  |
| `frames` | Yes | `INT` | `25` | 1 – 512 |  |
| `batch_size` | Yes | `INT` | `1` | 1 – 16 |  |
| `seed` | Yes | `INT` | `0` | 0 – 2147483648 |  |
| `noise_scale` | No | `FLOAT` | `1.0` | 0.01 – 4.0, step 0.01 | Multiply noise standard deviation (1.0 = unit Gaussian) |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `noise_latent` | `LATENT` |  |
| `shape_report` | `STRING` |  |

---

## Video Loader

**Node key:** `RadianceVideoLoader`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes_loader.py`  

Video loader v3.3 — for LTX 2.3, Wan, HunyuanVideo, etc. Supports Baked/standalone VAE, optional Audio VAE, and optional latent upscale model, on top of the universal loader features.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `preset` | Yes | choice of `Custom`, `HunyuanVideo`, `Wan 2.1`, `Wan 2.2`, `Wan 2.2 TI2V`, `LTX Video`, `LTX Video 13B`, `LTX Video 2.3`, … (+4 more) | `Custom` |  | Quick-configure for common architectures. Overrides model_type, dtypes, offload_mode, and hints which CLIP slots are needed. |
| `unet_name` | Yes | choice (empty) |  |  | Main diffusion model (UNET / DiT / Transformer). For WAN 2.2, select either the high_noise or low_noise file — the companion expert is detected automatically. The 'model' output always carries the high_noise expert and 'model_low_noise' always carries the low_noise expert, regardless of which file is selected. |
| `weight_dtype` | Yes | choice of `default`, `fp8_e4m3fn`, `fp8_e5m2`, `fp16`, `bf16`, `fp32` | `default` |  | UNET weight precision. fp8_e4m3fn saves ~40% VRAM vs fp16. |
| `model_type` | Yes | choice of `Auto-Detect`, `hunyuan_video`, `wan`, `ltxv`, `ltxav`, `cosmos`, `cogvideox`, `mochi` | `Auto-Detect` |  | 'Auto-Detect' reads the checkpoint's key names to determine architecture. Override manually if detection fails. |
| `vae_name` | Yes | choice of `Baked VAE (from UNET)` | `Baked VAE (from UNET)` |  | VAE for encoding/decoding latents. 'Baked VAE (from UNET)' extracts it from the checkpoint. |
| `audio_vae_name` | No | choice of `None`, `Baked Audio VAE (from UNET)` | `None` |  | Audio VAE for LTX 2.3. Choose 'Baked' or a standalone safetensors file. |
| `upscale_model_name` | No | choice of `None` | `None` |  | Latent Upscale Model (e.g. for LTX 2.3 or HunyuanVideo). |
| `clip_l` | No | choice of `None` | `None` |  | CLIP-L (text encoder). Used by: SD1.5, SDXL, Flux, SD3. |
| `clip_g` | No | choice of `None` | `None` |  | CLIP-G (text encoder). Used by: SDXL, SD3, SD3.5. |
| `t5xxl` | No | choice of `None` | `None` |  | T5-XXL (text encoder). Used by: Flux, SD3, SD3.5, Wan, PixArt, LTX (pre-2.3). |
| `llm_encoder` | No | choice of `None` | `None` |  | LLM encoder. Used by: Kolors/HunyuanVideo (ChatGLM3), LTX 2.3 (Gemma 3). |
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
| `model_low_noise` | `MODEL` |  |
| `clip` | `CLIP` |  |
| `vae` | `VAE` |  |
| `audio_vae` | `VAE` |  |
| `lora_stack` | `LORA_STACK` |  |
| `upscale_model` | `LATENT_UPSCALE_MODEL` |  |
| `model_meta` | `STRING` |  |

---

## Video Mask Propagator

**Node key:** `RadianceVideoMaskPropagator`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/vfx/roto.py`  

◎ Radiance Video Mask Propagator

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `masks` | Yes | `MASK` |  |  |  |
| `flow_vectors` | Yes | `IMAGE` |  |  | 32-bit flow vectors from Radiance Optical Flow. |
| `propagation_mode` | Yes | choice of `Forward`, `Backward`, `Bidirectional` | `Bidirectional` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `propagated_masks` | `MASK` |  |

---

## Video Model Info

**Node key:** `RadianceVideoModelInfo`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/t2v.py`  

Display configuration and parameter info for a loaded video model.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` |  |  |  |
| `model_preset` | Yes | choice of `LTX-Video (128ch)`, `HunyuanVideo (16ch)`, `Wan2.1 (16ch)`, `CogVideoX (16ch)`, `SD-VAE (4ch)` | `LTX-Video (128ch)` |  |  |
| `override_channels` | No | `INT` | `0` | 0 – 512 |  |
| `override_latent_scale` | No | `FLOAT` | `0.0` | 0.0 – 10.0 |  |
| `print_info` | No | `BOOLEAN` | `False` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `model` | `MODEL` |  |
| `dit_config` | `STRING` |  |
| `info_report` | `STRING` |  |

---

## Video Sampler

**Node key:** `RadianceVideoSampler`  
**Menu:** `FXTD STUDIOS/Radiance/Video`  
**Source:** `nodes/video/t2v.py`  

Run the diffusion sampler to generate video latents from pre-built noise and conditioning.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` |  |  |  |
| `positive` | Yes | `CONDITIONING` |  |  |  |
| `negative` | Yes | `CONDITIONING` |  |  |  |
| `latent_noise` | Yes | `LATENT` |  |  |  |
| `steps` | Yes | `INT` | `25` | 1 – 200 |  |
| `cfg` | Yes | `FLOAT` | `7.0` | 0.0 – 30.0, step 0.1 |  |
| `sampler_name` | Yes | choice of `euler`, `euler_ancestral`, `heun`, `heunpp2`, `dpm_2`, `dpm_2_ancestral`, `lms`, `dpm_fast`, … (+17 more) | `euler` |  |  |
| `scheduler` | Yes | choice of `normal`, `karras`, `exponential`, `sgm_uniform`, `simple`, `ddim_uniform`, `beta` | `normal` |  |  |
| `seed` | Yes | `INT` | `0` | 0 – 2147483648 |  |
| `dit_config` | No | `STRING` | `{}` |  | JSON from RadianceVideoModelInfo — when connected, overrides steps/cfg/sampler/scheduler with model-specific defaults. |
| `cfg_schedule_json` | No | `STRING` | `` |  | JSON float array from RadianceAudioCFGSchedule — first value overrides CFG |
| `denoise` | No | `FLOAT` | `1.0` | 0.0 – 1.0, step 0.01 |  |
| `tiling` | No | `BOOLEAN` | `False` |  | Enable tiled sampling for large resolutions (reduces VRAM) |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `samples` | `LATENT` |  |
| `sampler_report` | `STRING` |  |

---
