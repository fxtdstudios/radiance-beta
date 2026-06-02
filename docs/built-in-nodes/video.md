# Video

Video model inspection, latent noise, conditioning merge, sampling, T2V/I2V pipelines, batch decode, HDR video decode, frame routing, assembly, and export.

## Typical workflow

```text
Video Model Info -> Video Latent Noise + Video Cond Merge -> Video Sampler -> Batch Decode -> Video Export
```

## Before you use these nodes

- Start with model info so latent shapes match the model family.
- Track frame count through router, assembler, decode, and export nodes.
- Use HDR video conditioner/decode when tone and peak metadata must survive generation.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Video Model Info](#video-model-info) | `RadianceVideoModelInfo` | Inspect a ComfyUI MODEL object and produce a DiT config JSON describing. |
| [◎ Video Latent Noise](#video-latent-noise) | `RadianceVideoLatentNoise` | Generate correctly-shaped Gaussian noise for a given DiT video model. |
| [◎ Video Cond Merge](#video-cond-merge) | `RadianceVideoCondMerge` | Merge up to three conditioning inputs (text, character, HDR) into a. |
| [◎ Video Sampler](#video-sampler) | `RadianceVideoSampler` | Creates, selects, refines, or propagates masks for VFX, inpaint, and compositing work. |
| [◎ T2V Pipeline](#t2v-pipeline) | `RadianceT2VPipeline` | Builds, routes, samples, decodes, or exports video batches and video latents. |
| [◎ I2V Pipeline](#i2v-pipeline) | `RadianceI2VPipeline` | Builds, routes, samples, decodes, or exports video batches and video latents. |
| [◎ Video Batch Decode](#video-batch-decode) | `RadianceVideoBatchDecode` | Decode a DiT video LATENT tensor into an IMAGE frame batch. |
| [◎ Video Export](#video-export) | `RadianceVideoExport` | Route a decoded video IMAGE batch to the appropriate output format. |
| [◎ Video HDR Conditioner](#video-hdr-conditioner) | `RadianceVideoHDRConditioner` | Builds, routes, samples, decodes, or exports video batches and video latents. |
| [◎ Video HDR Decode](#video-hdr-decode) | `RadianceVideoHDRDecode` | Decodes latent, video, or HDR-oriented data back into image form for review or output. |
| [◎ Video Frame Router](#video-frame-router) | `RadianceVideoFrameRouter` | Extract individual frames from a decoded video IMAGE tensor. |
| [◎ Video Assembler](#video-assembler) | `RadianceVideoAssembler` | Collect per-frame IMAGE tensors into a single video batch tensor. |

## ◎ Video Model Info

**Internal key:** `RadianceVideoModelInfo`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/t2v.py`
**Function:** `inspect`

### What it does

Inspect a ComfyUI MODEL object and produce a DiT config JSON describing.

### When to use it

Use `◎ Video Model Info` when the graph reaches the Video Model Info step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` | - | - |
| `model_preset` | Yes | `(MODEL_NAMES, {'default': 'LTX-Video (128ch)'})` | - | - |
| `override_channels` | Optional | `INT` | `0` | - |
| `override_latent_scale` | Optional | `FLOAT` | `0.0` | - |
| `print_info` | Optional | `BOOLEAN` | `False` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `model` | `MODEL` | Output produced by the `model` socket. |
| `dit_config` | `STRING` | Output produced by the `dit_config` socket. |
| `info_report` | `STRING` | Output produced by the `info_report` socket. |

### Practical notes

- The node returns `model` (`MODEL`), `dit_config` (`STRING`), `info_report` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Latent Noise

**Internal key:** `RadianceVideoLatentNoise`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/t2v.py`
**Function:** `generate`

### What it does

Generate correctly-shaped Gaussian noise for a given DiT video model.

### When to use it

Use `◎ Video Latent Noise` when the graph reaches the Video Latent Noise step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `dit_config` | Yes | `STRING` | `{}` | JSON from RadianceVideoModelInfo |
| `width` | Yes | `INT` | `512` | - |
| `height` | Yes | `INT` | `512` | - |
| `frames` | Yes | `INT` | `25` | - |
| `batch_size` | Yes | `INT` | `1` | - |
| `seed` | Yes | `('INT', {'default': 0, 'min': 0, 'max': 2 ** 31})` | - | - |
| `noise_scale` | Optional | `FLOAT` | `1.0` | Multiply noise standard deviation (1.0 = unit Gaussian) |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `noise_latent` | `LATENT` | Output produced by the `noise_latent` socket. |
| `shape_report` | `STRING` | Output produced by the `shape_report` socket. |

### Practical notes

- The node returns `noise_latent` (`LATENT`), `shape_report` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Cond Merge

**Internal key:** `RadianceVideoCondMerge`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/t2v.py`
**Function:** `merge`

### What it does

Merge up to three conditioning inputs (text, character, HDR) into a.

### When to use it

Use `◎ Video Cond Merge` when the graph reaches the Video Cond Merge step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `text_conditioning` | Yes | `CONDITIONING` | - | - |
| `merge_mode` | Yes | `(cls.MODES, {'default': 'concat'})` | - | - |
| `character_conditioning` | Optional | `CONDITIONING` | - | - |
| `hdr_conditioning` | Optional | `CONDITIONING` | - | - |
| `text_weight` | Optional | `FLOAT` | `1.0` | - |
| `character_weight` | Optional | `FLOAT` | `0.75` | - |
| `hdr_weight` | Optional | `FLOAT` | `0.5` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `merged_conditioning` | `CONDITIONING` | Output produced by the `merged_conditioning` socket. |
| `merge_report` | `STRING` | Output produced by the `merge_report` socket. |

### Practical notes

- The node returns `merged_conditioning` (`CONDITIONING`), `merge_report` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Sampler

**Internal key:** `RadianceVideoSampler`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/t2v.py`
**Function:** `sample`

### What it does

Creates, selects, refines, or propagates masks for VFX, inpaint, and compositing work.

### When to use it

Use `◎ Video Sampler` when the graph reaches the Video Sampler step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` | - | - |
| `positive` | Yes | `CONDITIONING` | - | - |
| `negative` | Yes | `CONDITIONING` | - | - |
| `latent_noise` | Yes | `LATENT` | - | - |
| `dit_config` | Yes | `STRING` | `{}` | - |
| `steps` | Yes | `INT` | `25` | - |
| `cfg` | Yes | `FLOAT` | `7.0` | - |
| `sampler_name` | Yes | `(_SAMPLERS, {'default': 'euler'})` | - | - |
| `scheduler` | Yes | `(_SCHEDULERS, {'default': 'normal'})` | - | - |
| `seed` | Yes | `('INT', {'default': 0, 'min': 0, 'max': 2 ** 31})` | - | - |
| `cfg_schedule_json` | Optional | `STRING` | `` | JSON float array from RadianceAudioCFGSchedule — first value overrides CFG |
| `denoise` | Optional | `FLOAT` | `1.0` | - |
| `tiling` | Optional | `BOOLEAN` | `False` | Enable tiled sampling for large resolutions (reduces VRAM) |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `samples` | `LATENT` | Output produced by the `samples` socket. |
| `sampler_report` | `STRING` | Output produced by the `sampler_report` socket. |

### Practical notes

- The node returns `samples` (`LATENT`), `sampler_report` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ T2V Pipeline

**Internal key:** `RadianceT2VPipeline`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/t2v.py`
**Function:** `generate`

### What it does

Builds, routes, samples, decodes, or exports video batches and video latents.

### When to use it

Use `◎ T2V Pipeline` when you want one higher-level node to wire several lower-level steps together.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` | - | - |
| `clip` | Yes | `CLIP` | - | - |
| `vae` | Yes | `VAE` | - | - |
| `positive_prompt` | Yes | `STRING` | `cinematic HDR video, stunning visuals, 4K, film grain` | - |
| `negative_prompt` | Yes | `STRING` | `watermark, blurry, low quality, sdr, flickering` | - |
| `width` | Yes | `INT` | `768` | - |
| `height` | Yes | `INT` | `512` | - |
| `frames` | Yes | `INT` | `25` | - |
| `seed` | Yes | `('INT', {'default': 0, 'min': 0, 'max': 2 ** 31})` | - | - |
| `dit_config` | Optional | `STRING` | `{}` | JSON from RadianceVideoModelInfo — sets model-specific defaults |
| `character_conditioning` | Optional | `CONDITIONING` | - | - |
| `cfg_schedule_json` | Optional | `STRING` | `` | JSON float array from RadianceAudioCFGSchedule |
| `steps` | Optional | `INT` | `0` | 0 = use model default |
| `cfg` | Optional | `FLOAT` | `0.0` | 0 = use model default |
| `sampler_name` | Optional | `(_SAMPLERS, {'default': 'euler'})` | - | - |
| `scheduler` | Optional | `(_SCHEDULERS, {'default': 'normal'})` | - | - |
| `peak_nits` | Optional | `([str(n) for n in [100, 203, 400, 600, 1000, 4000, 10000]], {'default': '1000'})` | - | - |
| `target_gamut` | Optional | `ENUM: BT.2020, P3-D65, P3-DCI, BT.709, ACEScg` | `BT.2020` | - |
| `hdr_eotf` | Optional | `ENUM: PQ (ST.2084), HLG (BT.2100), Linear, sRGB / BT.1886` | `PQ (ST.2084)` | - |
| `hdr_strength` | Optional | `FLOAT` | `0.5` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `video_latent` | `LATENT` | Output produced by the `video_latent` socket. |
| `preview_frames` | `IMAGE` | Output produced by the `preview_frames` socket. |
| `positive_cond` | `CONDITIONING` | Output produced by the `positive_cond` socket. |
| `pipeline_report` | `STRING` | Output produced by the `pipeline_report` socket. |

### Practical notes

- The node returns `video_latent` (`LATENT`), `preview_frames` (`IMAGE`), `positive_cond` (`CONDITIONING`), `pipeline_report` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ I2V Pipeline

**Internal key:** `RadianceI2VPipeline`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/t2v.py`
**Function:** `generate`

### What it does

Builds, routes, samples, decodes, or exports video batches and video latents.

### When to use it

Use `◎ I2V Pipeline` when you want one higher-level node to wire several lower-level steps together.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `model` | Yes | `MODEL` | - | - |
| `clip` | Yes | `CLIP` | - | - |
| `vae` | Yes | `VAE` | - | - |
| `reference_image` | Yes | `IMAGE` | - | - |
| `positive_prompt` | Yes | `STRING` | `smooth camera motion, cinematic HDR, 4K` | - |
| `negative_prompt` | Yes | `STRING` | `watermark, blurry, flickering, sdr` | - |
| `frames` | Yes | `INT` | `25` | - |
| `seed` | Yes | `('INT', {'default': 0, 'min': 0, 'max': 2 ** 31})` | - | - |
| `dit_config` | Optional | `STRING` | `{}` | - |
| `character_conditioning` | Optional | `CONDITIONING` | - | - |
| `cfg_schedule_json` | Optional | `STRING` | `` | - |
| `i2v_strategy` | Optional | `(cls.I2V_STRATEGIES, {'default': 'auto'})` | - | - |
| `image_strength` | Optional | `FLOAT` | `0.85` | How strongly the reference image anchors the generation |
| `motion_strength` | Optional | `FLOAT` | `0.5` | Amount of motion / temporal variation (0=nearly static) |
| `steps` | Optional | `INT` | `0` | - |
| `cfg` | Optional | `FLOAT` | `0.0` | - |
| `sampler_name` | Optional | `(_SAMPLERS, {'default': 'euler'})` | - | - |
| `scheduler` | Optional | `(_SCHEDULERS, {'default': 'normal'})` | - | - |
| `peak_nits` | Optional | `([str(n) for n in [100, 203, 400, 600, 1000, 4000, 10000]], {'default': '1000'})` | - | - |
| `target_gamut` | Optional | `ENUM: BT.2020, P3-D65, P3-DCI, BT.709, ACEScg` | `BT.2020` | - |
| `hdr_eotf` | Optional | `ENUM: PQ (ST.2084), HLG (BT.2100), Linear, sRGB / BT.1886` | `PQ (ST.2084)` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `video_latent` | `LATENT` | Output produced by the `video_latent` socket. |
| `preview_frames` | `IMAGE` | Output produced by the `preview_frames` socket. |
| `pipeline_report` | `STRING` | Output produced by the `pipeline_report` socket. |

### Practical notes

- The node returns `video_latent` (`LATENT`), `preview_frames` (`IMAGE`), `pipeline_report` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Batch Decode

**Internal key:** `RadianceVideoBatchDecode`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/t2v.py`
**Function:** `decode`

### What it does

Decode a DiT video LATENT tensor into an IMAGE frame batch.

### When to use it

Use `◎ Video Batch Decode` when the graph reaches the Video Batch Decode step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `vae` | Yes | `VAE` | - | - |
| `latent` | Yes | `LATENT` | - | - |
| `dit_config` | Optional | `STRING` | `{}` | - |
| `tile_decode` | Optional | `BOOLEAN` | `False` | Tile the VAE decode to reduce VRAM on large videos |
| `tile_overlap` | Optional | `INT` | `64` | Pixel overlap between tiles (higher = smoother seams) |
| `output_linear` | Optional | `BOOLEAN` | `False` | Skip gamma correction — output linear-light frames for HDR pipeline |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `frames` | `IMAGE` | Output produced by the `frames` socket. |
| `frame_count` | `INT` | Output produced by the `frame_count` socket. |
| `decode_report` | `STRING` | Output produced by the `decode_report` socket. |

### Practical notes

- The node returns `frames` (`IMAGE`), `frame_count` (`INT`), `decode_report` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Export

**Internal key:** `RadianceVideoExport`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/t2v.py`
**Function:** `export`

### What it does

Route a decoded video IMAGE batch to the appropriate output format.

### When to use it

Use `◎ Video Export` near the end of the graph after the image, sequence, or metadata is ready for delivery.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `frames` | Yes | `IMAGE` | - | - |
| `mode` | Yes | `(cls.MODES, {'default': 'passthrough'})` | - | - |
| `hdr_metadata_json` | Optional | `STRING` | `{"peak_nits":1000,"eotf":"PQ (ST.2084)"}` | - |
| `output_folder` | Optional | `STRING` | `` | - |
| `filename_prefix` | Optional | `STRING` | `radiance_video` | - |
| `fps` | Optional | `FLOAT` | `24.0` | - |
| `frame_offset` | Optional | `INT` | `0` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `frames` | `IMAGE` | Output produced by the `frames` socket. |
| `frame_count` | `INT` | Output produced by the `frame_count` socket. |
| `export_report` | `STRING` | Output produced by the `export_report` socket. |

### Practical notes

- The node returns `frames` (`IMAGE`), `frame_count` (`INT`), `export_report` (`STRING`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video HDR Conditioner

**Internal key:** `RadianceVideoHDRConditioner`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/hdr.py`
**Function:** `condition`

### What it does

Builds, routes, samples, decodes, or exports video batches and video latents.

### When to use it

Use `◎ Video HDR Conditioner` when the graph reaches the Video HDR Conditioner step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `positive` | Yes | `CONDITIONING` | - | - |
| `peak_nits` | Yes | `([str(n) for n in PEAK_NITS], {'default': '1000'})` | - | - |
| `target_gamut` | Yes | `(GAMUT_OPTIONS, {'default': 'BT.2020'})` | - | - |
| `eotf` | Yes | `(EOTF_OPTIONS, {'default': 'PQ (ST.2084)'})` | - | - |
| `camera_move` | Optional | `(list(_CAMERA_TOKENS.keys()), {'default': 'None'})` | - | - |
| `mood` | Optional | `(list(_MOOD_TOKENS.keys()), {'default': 'None'})` | - | - |
| `extra_hdr_prompt` | Optional | `STRING` | `` | Additional HDR descriptors appended to conditioning tokens |
| `inject_metadata_embedding` | Optional | `BOOLEAN` | `True` | Add HDR metadata dict to conditioning['extra'] for compatible models |
| `token_strength` | Optional | `FLOAT` | `1.0` | Scale the appended token embeddings (1.0 = normal weight) |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `positive` | `CONDITIONING` | Output produced by the `positive` socket. |
| `hdr_metadata_json` | `STRING` | Output produced by the `hdr_metadata_json` socket. |

### Practical notes

- The node returns `positive` (`CONDITIONING`), `hdr_metadata_json` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video HDR Decode

**Internal key:** `RadianceVideoHDRDecode`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/hdr.py`
**Function:** `decode`

### What it does

Decodes latent, video, or HDR-oriented data back into image form for review or output.

### When to use it

Use `◎ Video HDR Decode` when the graph reaches the Video HDR Decode step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `hdr_metadata_json` | Yes | `STRING` | `{"peak_nits":1000,"gamut":"BT.2020","eotf":"PQ (ST.2084)"}` | JSON from RadianceVideoHDRConditioner or manually entered |
| `tonemap` | Yes | `(cls.TONEMAP_MODES, {'default': 'Reinhard'})` | - | - |
| `exposure_compensation_ev` | Optional | `FLOAT` | `0.0` | EV adjustment before tone-mapping |
| `output_eotf` | Optional | `(EOTF_OPTIONS, {'default': 'PQ (ST.2084)'})` | - | - |
| `sdr_preview_nits` | Optional | `FLOAT` | `100.0` | Scale factor for the SDR preview output |
| `gamut_clip` | Optional | `BOOLEAN` | `True` | Hard-clip out-of-gamut values before EOTF encode |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `hdr_image` | `IMAGE` | Output produced by the `hdr_image` socket. |
| `sdr_preview` | `IMAGE` | Output produced by the `sdr_preview` socket. |
| `decode_report` | `STRING` | Output produced by the `decode_report` socket. |

### Practical notes

- The node returns `hdr_image` (`IMAGE`), `sdr_preview` (`IMAGE`), `decode_report` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Frame Router

**Internal key:** `RadianceVideoFrameRouter`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/hdr.py`
**Function:** `route`

### What it does

Extract individual frames from a decoded video IMAGE tensor.

### When to use it

Use `◎ Video Frame Router` when the graph reaches the Video Frame Router step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `video_image` | Yes | `IMAGE` | - | - |
| `frame_index` | Yes | `INT` | `0` | - |
| `wrap_index` | Optional | `BOOLEAN` | `True` | If frame_index >= total_frames, wrap around (modulo) |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `frame_image` | `IMAGE` | Output produced by the `frame_image` socket. |
| `frame_index` | `INT` | Output produced by the `frame_index` socket. |
| `total_frames` | `INT` | Output produced by the `total_frames` socket. |
| `passthrough` | `IMAGE` | Output produced by the `passthrough` socket. |

### Practical notes

- The node returns `frame_image` (`IMAGE`), `frame_index` (`INT`), `total_frames` (`INT`), `passthrough` (`IMAGE`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Video Assembler

**Internal key:** `RadianceVideoAssembler`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/video/hdr.py`
**Function:** `assemble`

### What it does

Collect per-frame IMAGE tensors into a single video batch tensor.

### When to use it

Use `◎ Video Assembler` when the graph reaches the Video Assembler step in a video workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `frame` | Yes | `IMAGE` | - | - |
| `session_key` | Yes | `STRING` | `video_session_0` | - |
| `expected_total_frames` | Yes | `INT` | `24` | - |
| `flush` | Optional | `BOOLEAN` | `False` | Force output of accumulated frames now, even if incomplete |
| `reset` | Optional | `BOOLEAN` | `False` | Clear accumulated frames for this session key |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `video_image` | `IMAGE` | Output produced by the `video_image` socket. |
| `frames_accumulated` | `INT` | Output produced by the `frames_accumulated` socket. |
| `is_complete` | `BOOLEAN` | Output produced by the `is_complete` socket. |

### Practical notes

- The node returns `video_image` (`IMAGE`), `frames_accumulated` (`INT`), `is_complete` (`BOOLEAN`).
- Keep frame count and latent shape metadata consistent through the video graph.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
