[← Back to Radiance docs](../README.md)

# IO and Delivery

Load, inspect, save, and package production media. These nodes are the safest entry and exit points for EXR, sequences, masks, and delivery files.

## Typical workflow

```text
Radiance Read -> processing -> Radiance Write / EXR Multi-Part
```

## Before you use these nodes

- Use EXR for HDR or float masters; use PNG/JPEG only for review proxies.
- Keep mask outputs connected and preview them before crop, inpaint, or composite operations.
- For DCC handoff, write the file or sequence first, then send that output to Nuke or Resolve.
- **Alpha on EXR:** Radiance Write has an optional `mask` input. When connected and the format is EXR, the matte is written as the EXR alpha channel (RGBA); it is ignored for non-EXR formats. Write failures raise (the node turns red) rather than producing a silent empty file.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Radiance Read](#radiance-read) | `RadianceRead` | Universal reader for still images, EXR files, video, and numbered sequences. |
| [◎ Radiance Write](#radiance-write) | `RadianceWrite` | Production writer for frames, sequences, and delivery formats. |
| [◎ Radiance EXR Multi-Part](#radiance-exr-multi-part) | `RadianceEXRMultiPart` | Write a named multi-part EXR v2 combining up to 6 AOV layers into one. |
| [◎ Radiance Load Image Mask](#radiance-load-image-mask) | `RadianceLoadImageMask` | Advanced Image Loader + Mask Editor for Radiance. |

## ◎ Radiance Read

**Internal key:** `RadianceRead`  
**Category:** `FXTD STUDIOS/Radiance/◎ IO & Delivery`  
**Source:** `nodes_io.py`
**Function:** `read`

### What it does

Universal reader for still images, EXR files, video, and numbered sequences.

### When to use it

Use `◎ Radiance Read` at the start of a io and delivery graph when this data needs to be loaded once and reused downstream.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `browse` | Yes | `(_get_input_files(), {'image_upload': True, 'tooltip': "Browse or upload a file from disk.\n• Click the upload icon (📎) to open a native file picker.\n• Supports images (PNG, JPG, TIFF, EXR, DPX, HDR, WebP) and video (MP4, MOV, MXF, AVI, WebM, MKV).\n• Uploaded files are copied to ComfyUI's input• Leave blank and fill in 'path' below for absolute / network / sequence paths."})` | - | - |
| `media_type` | Optional | `ENUM: Auto, Image, Video, Sequence` | `Auto` | Override auto-detection. Auto infers from path extension and pattern. |
| `path` | Optional | `STRING` | `` | Optional — used only when 'browse' is left blank. Accepts any absolute path, UNC network path, or sequence pattern: Sequence patterns: /frames/f.%04d.exr · /frames/f.####.png · /dir/ Network paths: /mnt/nas/renders/shot or \\server\share\shot Format is auto-detected from extension. |
| `color_space` | Optional | `(INPUT_COLOR_SPACES, {'default': 'Auto / Linear (pass-through)', 'tooltip': 'Decode the input from this color space to scene-linear before processing.'})` | - | - |
| `start_frame` | Optional | `INT` | `1001` | First frame index (sequences only). |
| `end_frame` | Optional | `INT` | `0` | Last frame index (0 = read all frames). |
| `frame_step` | Optional | `INT` | `1` | Step size — e.g. 2 reads every other frame. |
| `max_video_frames` | Optional | `INT` | `0` | Cap on decoded video frames (0 = all frames). Large videos use a lot of RAM. |
| `proxy_scale` | Optional | `FLOAT` | `0.0` | Downscale factor for proxy preview (0 = full resolution). 0.5 = half res for faster iteration. |
| `missing_frames` | Optional | `ENUM: Error, Black, Skip` | `Skip` | How to handle missing sequence frames. Black inserts zero frames, Skip omits them, Error raises. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |
| `mask` | `MASK` | Output produced by the `mask` socket. |

### Practical notes

- The node returns `image` (`IMAGE`), `mask` (`MASK`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Write

**Internal key:** `RadianceWrite`  
**Category:** `FXTD STUDIOS/Radiance/◎ IO & Delivery`  
**Source:** `nodes_io.py`
**Function:** `write`

### What it does

Production writer for frames, sequences, and delivery formats.

### When to use it

Use `◎ Radiance Write` near the end of the graph after the image, sequence, or metadata is ready for delivery.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `*` | - | Accepts any ComfyUI output: IMAGE tensor, batched video frames, VHSor any dict/list carrying a video path. |
| `output_path` | Yes | `('STRING', {'default': str(Path.home() / 'radiance_output'), 'multiline': False, 'placeholder': '/output/render  or  //nas/share/render  or  Z:/renders/shot', 'tooltip': 'Output directory + filename stem.  Extension is appended automatically based on format.\nFor sequences: frame number and extension are appended (e.g. /out/frame_0001.exr).\n\nNetwork paths are fully supported — use the path as mounted on this machine:\n  Linux / Mac  →  /mnt/nas/renders/shot_001\n  Windows UNC  →  \\\\server\\share\\renders\\shot_001\n  Windows drive→  Z:\\renders\\shot_001\nThe directory is created automatically (mkdir -p) if it does not exist.\nWrite permissions on the share are required.'})` | - | - |
| `format` | Yes | `(WRITE_FORMATS, {'default': 'IMG │ EXR (16-bit half)', 'tooltip': 'Output format.  Extension is appended automatically.'})` | - | - |
| `filename` | Optional | `STRING` | `` | Output filename stem (version appended automatically). Leave empty to use output_path as the full stem. |
| `version` | Optional | `INT` | `1` | Version number appended to filename (e.g. shot_001_v0001). |
| `color_space` | Optional | `(OUTPUT_COLOR_SPACES, {'default': 'Linear (pass-through)', 'tooltip': 'Apply this color space transform before saving.'})` | - | - |
| `fps` | Optional | `FLOAT` | `24.0` | Frame rate for video and sequence outputs. |
| `quality` | Optional | `INT` | `18` | CRF quality for H.264/H.265 (lower = better). Also JPEG quality 0–100 (remapped). |
| `exr_compression` | Optional | `(EXR_COMPRESSIONS, {'default': 'ZIP', 'tooltip': 'EXR compression codec (EXR formats only).'})` | - | - |
| `start_frame` | Optional | `INT` | `1001` | First frame number for sequences. |
| `frame_padding` | Optional | `INT` | `4` | Zero-padding width for frame numbers (e.g. 4 → 0001). |
| `audio_source` | Optional | `STRING` | `` | Path to audio file to mux into video output (optional). |
| `broadcast_safe` | Optional | `BOOLEAN` | `False` | Clamp output to broadcast-legal range (16–235 luma) before saving. |
| `overwrite` | Optional | `BOOLEAN` | `True` | Overwrite existing files. When disabled, a unique suffix is appended. |
| `proxy_scale` | Optional | `FLOAT` | `0.0` | Downscale output by this factor for proxy preview (0 = full resolution). |
| `audio` | Optional | `AUDIO` | - | Audio tensor from RadianceVideoLoader (muxed into video output). |

### Outputs

This node does not declare named runtime outputs in the source catalog.

### Practical notes

- Keep this node in the part of the graph indicated by its group workflow and inspect all report or metadata outputs when debugging.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance EXR Multi-Part

**Internal key:** `RadianceEXRMultiPart`  
**Category:** `FXTD STUDIOS/Radiance/◎ IO & Delivery`  
**Source:** `nodes_io.py`
**Function:** `write_multipart`

### What it does

Write a named multi-part EXR v2 combining up to 6 AOV layers into one.

### When to use it

Use `◎ Radiance EXR Multi-Part` when the graph reaches the EXR Multi-Part step in a io and delivery workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `filename_prefix` | Yes | `STRING` | `radiance_multipart` | - |
| `beauty` | Yes | `IMAGE` | - | - |
| `bit_depth` | Yes | `(_EXR_BIT_DEPTHS, {'default': '16-bit Half Float'})` | - | - |
| `compression` | Yes | `(_EXR_COMPRESSIONS, {'default': 'ZIP'})` | - | - |
| `depth` | Optional | `IMAGE` | - | - |
| `normal` | Optional | `IMAGE` | - | - |
| `albedo` | Optional | `IMAGE` | - | - |
| `custom_1` | Optional | `IMAGE` | - | - |
| `custom_1_name` | Optional | `STRING` | `emission` | - |
| `custom_2` | Optional | `IMAGE` | - | - |
| `custom_2_name` | Optional | `STRING` | `specular` | - |
| `output_path` | Optional | `STRING` | `` | - |
| `remote_path` | Optional | `STRING` | `` | - |
| `frame_index` | Optional | `INT` | `1` | - |
| `custom_metadata` | Optional | `STRING` | `` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `output_path` | `STRING` | Output produced by the `output_path` socket. |

### Practical notes

- The node returns `output_path` (`STRING`).
- Preserve HDR masters as EXR when values above display white matter.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Load Image Mask

**Internal key:** `RadianceLoadImageMask`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/io/mask.py`
**Function:** `load_image`

### What it does

Advanced Image Loader + Mask Editor for Radiance.

### When to use it

Use `◎ Radiance Load Image Mask` when the graph reaches the Load Image Mask step in a io and delivery workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `(sorted(files), {'image_upload': True})` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `IMAGE` | `IMAGE` | Output produced by the `IMAGE` socket. |
| `MASK` | `MASK` | Output produced by the `MASK` socket. |

### Practical notes

- The node returns `IMAGE` (`IMAGE`), `MASK` (`MASK`).
- Preview the mask output directly before using it to crop, inpaint, or composite.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
