[← Back to Radiance docs](../README.md)

# IO and Delivery

Load, inspect, save, and package production media. These are the entry and exit
points of every Radiance graph, and the place where a colour-space mistake does
the most damage — what you decode wrongly here is wrong for the rest of the
graph, and nothing downstream will tell you.

## Typical graph

```text
Radiance Read → colour transform → work → Radiance Write
```

## Before you use these nodes

- **Use EXR for anything you intend to keep.** PNG and JPEG clip at 1.0 and
  quantise; they are for review proxies, not masters.
- **Set `color_space` on Read deliberately.** "Auto / Linear (pass-through)"
  means *no decode* — correct for an EXR that is already scene-linear, wrong for
  a camera log file, which will look flat and grade badly.
- **Alpha on EXR.** Write has an optional `mask` input. Connected, with an EXR
  format selected, the matte is written as the alpha channel. It is ignored for
  non-EXR formats. Write failures raise — the node turns red — rather than
  leaving an empty file behind.
- **Give Write a filename.** With `filename` blank, `output_path` is treated as
  the full path including the stem. A blank on both raises a clear error rather
  than writing somewhere surprising.
- **DPX needs OpenImageIO.** There is no Pillow plugin for it. The Digital
  Cinema nodes will tell you if the package is missing.

## Reading video

Video decodes through one raw ffmpeg pipe at the source's own bit depth — 10-bit
ProRes 422, 12-bit ProRes 4444, DNxHR HQX and 10-bit HEVC all survive intact.

- **ProRes 4444 alpha reaches the `mask` output.** 422 in any flavour has no
  alpha channel; only 4444 and 4444 XQ do.
- **`start_frame`, `end_frame` and `frame_step` apply to video**, on the clip's
  own zero-based numbering. `start_frame` defaults to 1001 because that is the
  sequence convention, so a start past the end of a clip is treated as unset
  rather than decoding nothing.
- **Colour tags are read.** On Auto, a file tagged `bt709`, `smpte2084` or
  `arib-std-b67` decodes through the matching curve and the node logs which one.
  An *untagged* file passes through unchanged and warns — a Rec.709 delivery is
  not scene-linear, and Radiance will not guess for you.
- **Read raises on failure.** A missing file, a corrupt MOV or a truncated clip
  turns the node red instead of yielding black frames.

## Known limitations

The EXR reader ignores the display window, so an overscan render (a standard
Nuke output) comes back offset and at the data-window resolution with no
warning. Crop to the display window in your comp application before bringing
overscan plates into Radiance.

A clip is decoded to one batched tensor, so RAM is the ceiling — 240 frames of
4K RGBA float32 is about 31 GB. Interlaced sources are reported but not
deinterlaced.

## Nodes in this section (5)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [DPX Read](#dpx-read) | `RadianceDigitalCinemaRead` |  |
| [DPX Write](#dpx-write) | `RadianceDigitalCinemaWrite` |  |
| [Read](#read) | `RadianceRead` | Read an image, EXR, video or numbered sequence into the pipeline |
| [Write](#write) | `RadianceWrite` | Write images or EXR sequences to disk with configurable format options. |
| [Write EXR](#write-exr) | `RadianceEXRMultiPart` | Read or write multi-part OpenEXR files with named channel layers. |

---

## DPX Read

**Node key:** `RadianceDigitalCinemaRead`  
**Menu:** `FXTD STUDIOS/Radiance/Load & Save`  
**Source:** `nodes_io.py`  

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `source_path` | Yes | `STRING` | `` |  |  |
| `read_mode` | Yes | choice of `Auto`, `Video`, `Sequence`, `EXR` | `Auto` |  |  |
| `start_frame` | Yes | `INT` | `1` | ≥ 1 |  |
| `frame_limit` | Yes | `INT` | `0` | ≥ 0 |  |
| `input_colorspace` | Yes | choice of `sRGB (Standard)` | `sRGB (Standard)` |  |  |
| `fps_override` | Yes | `FLOAT` | `0.0` | ≥ 0.0 |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` |  |
| `mask` | `MASK` |  |
| `shot_metadata` | `RADIANCE_SHOT` |  |

---

## DPX Write

**Node key:** `RadianceDigitalCinemaWrite`  
**Menu:** `FXTD STUDIOS/Radiance/Load & Save`  
**Source:** `nodes_io.py`  
**Output node** — runs even with nothing connected downstream.  

Backward-compatible Digital Cinema writer shim.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` |  |  |  |
| `output_path` | Yes | `STRING` | `/root/radiance_output` |  |  |
| `format` | No | choice of `IMG │ PNG (8-bit)`, `IMG │ PNG (16-bit)`, `IMG │ JPEG`, `IMG │ TIFF (16-bit)`, `IMG │ TIFF (32-bit float)`, `IMG │ DPX`, `IMG │ WEBP`, `IMG │ EXR (16-bit half)`, … (+14 more) | `IMG │ EXR (16-bit half)` |  |  |
| `filename` | No | `STRING` | `` |  |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `status` | `STRING` |  |

---

## Read

**Node key:** `RadianceRead`  
**Menu:** `FXTD STUDIOS/Radiance/Load & Save`  
**Source:** `nodes_io.py`  

Read an image, EXR, video or numbered sequence into the pipeline. Video decodes at the source bit depth through ffmpeg, keeps ProRes 4444 alpha on the mask output, and honours the file's colour tags.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `browse` | Yes | choice of ``, `x.jpg` | `` |  | Browse or upload a file from disk. • Click the upload icon (📎) to open a native file picker. • Supports images (PNG, JPG, TIFF, EXR, DPX, HDR, WebP) and video (MP4, MOV, MXF, AVI, WebM, MKV). • Uploaded files are copied to ComfyUI's input• Leave blank and fill in 'path' below for absolute / network / sequence paths. |
| `media_type` | No | choice of `Auto`, `Image`, `Video`, `Sequence` | `Auto` |  | Override auto-detection. Auto infers from path extension and pattern. |
| `path` | No | `STRING` | `` |  | Optional — used only when 'browse' is left blank. Accepts any absolute path, UNC network path, or sequence pattern: Sequence patterns: /frames/f.%04d.exr · /frames/f.####.png · /dir/ Network paths: /mnt/nas/renders/shot or \\server\share\shot Format is auto-detected from extension. |
| `color_space` | No | choice of `Auto / Linear (pass-through)`, `Rec.709 (BT.1886)`, `sRGB`, `ARRI LogC4`, `ARRI LogC3`, `Sony S-Log3`, `Panasonic V-Log`, `Canon Log 3`, … (+6 more) | `Auto / Linear (pass-through)` |  | Decode the input from this color space to scene-linear before processing. |
| `start_frame` | No | `INT` | `1001` | 0 – 99999 | First frame to read. • Sequence: the frame number in the filename (1001 is the usual VFX start). • Video: a 0-based offset into the clip. The 1001 default is a sequence convention, so it is ignored for any clip shorter than that rather than reading nothing. |
| `end_frame` | No | `INT` | `0` | 0 – 99999 | Last frame to read, inclusive. 0 = to the end. Applies to sequences and video. |
| `frame_step` | No | `INT` | `1` | 1 – 100 | Step size — e.g. 2 reads every other frame. Applies to sequences and video. |
| `max_video_frames` | No | `INT` | `0` | 0 – 99999 | Hard cap on decoded video frames (0 = all). Frames are float32 RGB in RAM: 240 frames of 4K RGBA is about 31 GB, so cap this while building a graph. |
| `proxy_scale` | No | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.05 | Downscale factor for proxy preview (0 = full resolution). 0.5 = half res for faster iteration. |
| `missing_frames` | No | choice of `Error`, `Black`, `Skip` | `Skip` |  | A frame inside a sequence that is not on disk. Black inserts a zero frame, Skip omits it, Error raises. For a read that fails outright, see on_error. |
| `layer` | No | `STRING` | `auto` |  | Which EXR layer to read. 'auto' takes the beauty, or the first colour layer, or — for a data-only file such as a Z-depth pass — the first layer of any kind. With the frontend loaded this is a dropdown listing the layers actually in the selected file. |
| `on_error` | No | choice of `Error`, `Black frame` | `Error` |  | What to do when the read fails — file missing, corrupt, unreadable. • Error: the node goes red and the queue stops. What Nuke and every other application does, and the default. • Black frame: return black and carry on. This is what 3.1.x always did, silently, which is how a black master got delivered. |
| `raw` | No | `BOOLEAN` | `False` |  | Hand back exactly what is stored in the file: no colour space decode, and no conform to the EXR display window (so overscan is preserved). Nuke's 'raw data'. |
| `premultiplied` | No | `BOOLEAN` | `False` |  | Tick when the file's RGB is already multiplied by its alpha — the EXR convention — and you want it divided back out on read. Off by default, matching Nuke, because turning it on changes pixels. |
| `reload` | No | `INT` | `0` | 0 – 2147483647 | Bump to force a re-read of the file, for when the contents changed but the timestamp did not. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Frames as a batch. Scene-linear once color_space has decoded them. |
| `mask` | `MASK` | Alpha. For a ProRes 4444, an RGBA EXR or an RGBA sequence this is the file's own matte; otherwise zeros. |
| `info` | `STRING` | JSON describing what was actually read: resolution, frame count and range, bit depth, codec, EXR layers and windows, colour tags, timecode. Nuke's metadata tab, as a wire you can plug in. |

---

## Write

**Node key:** `RadianceWrite`  
**Menu:** `FXTD STUDIOS/Radiance/Load & Save`  
**Source:** `nodes_io.py`  
**Output node** — runs even with nothing connected downstream.  

Write images or EXR sequences to disk with configurable format options.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `*` |  |  | Accepts any ComfyUI output: IMAGE tensor, batched video frames, VHSor any dict/list carrying a video path. |
| `output_path` | Yes | `STRING` | `/root/radiance_output` |  | Output directory + filename stem. Extension is appended automatically based on format. For sequences: frame number and extension are appended (e.g. /out/frame_0001.exr). Network paths are fully supported — use the path as mounted on this machine: Linux / Mac → /mnt/nas/renders/shot_001 Windows UNC → \\server\share\renders\shot_001 Windows drive→ Z:\renders\shot_001 The directory is created automatically (mkdir -p) if it does not exist. Write permissions on the share are required. |
| `format` | Yes | choice of `IMG │ PNG (8-bit)`, `IMG │ PNG (16-bit)`, `IMG │ JPEG`, `IMG │ TIFF (16-bit)`, `IMG │ TIFF (32-bit float)`, `IMG │ DPX`, `IMG │ WEBP`, `IMG │ EXR (16-bit half)`, … (+14 more) | `IMG │ EXR (16-bit half)` |  | Output format. Extension is appended automatically. |
| `filename` | No | `STRING` | `` |  | Output filename stem (version appended automatically). Leave empty to use output_path as the full stem. |
| `version` | No | `INT` | `1` | 0 – 9999 | Version number appended to filename (e.g. shot_001_v0001). |
| `color_space` | No | choice of `Linear (pass-through)`, `sRGB`, `Rec.709`, `Rec.2020`, `ACEScg`, `ARRI LogC4`, `ARRI LogC3`, `Sony S-Log3`, … (+2 more) | `Linear (pass-through)` |  | Apply this color space transform before saving. |
| `fps` | No | `FLOAT` | `0.0` | 0.0 – 240.0, step 0.001 | Frame rate for video and sequence outputs. 0 = auto-detect from the source video (falls back to 24 if unavailable). |
| `quality` | No | `INT` | `18` | 0 – 51 | CRF quality for H.264/H.265 (lower = better). Also JPEG quality 0–100 (remapped). |
| `exr_compression` | No | choice of `ZIP`, `ZIPS`, `PIZ`, `RLE`, `Uncompressed`, `DWAA`, `DWAB` | `ZIP` |  | EXR compression codec (EXR formats only). |
| `start_frame` | No | `INT` | `1001` | 0 – 99999 | First frame number for sequences. |
| `frame_padding` | No | `INT` | `4` | 1 – 8 | Zero-padding width for frame numbers (e.g. 4 → 0001). |
| `audio_source` | No | `STRING` | `` |  | Path to audio file to mux into video output (optional). Takes priority over the 'audio' input when both are set. |
| `broadcast_safe` | No | `BOOLEAN` | `False` |  | Clamp output to broadcast-legal range (16–235 luma) before saving. |
| `overwrite` | No | `BOOLEAN` | `True` |  | Overwrite existing files. When disabled, a unique suffix is appended. |
| `proxy_scale` | No | `FLOAT` | `0.0` | 0.0 – 1.0, step 0.05 | Downscale output by this factor for proxy preview (0 = full resolution). |
| `audio` | No | `AUDIO` |  |  | Audio tensor from RadianceVideoLoader (muxed into video output). |
| `mask` | No | `MASK` |  |  | Optional alpha/matte. When connected and the format is EXR or PNG, it is written as the alpha channel (RGBA). Ignored for other formats. |

Hidden inputs supplied by ComfyUI: `extra_pnginfo`, `prompt`.

### Outputs

None — this node is a terminal (it writes, sends, or displays).

---

## Write EXR

**Node key:** `RadianceEXRMultiPart`  
**Menu:** `FXTD STUDIOS/Radiance/Load & Save`  
**Source:** `nodes_io.py`  
**Output node** — runs even with nothing connected downstream.  

Read or write multi-part OpenEXR files with named channel layers.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `filename_prefix` | Yes | `STRING` | `radiance_multipart` |  |  |
| `beauty` | Yes | `IMAGE` |  |  |  |
| `bit_depth` | Yes | choice of `16-bit Half Float`, `32-bit Float` | `16-bit Half Float` |  |  |
| `compression` | Yes | choice of `ZIP`, `ZIPS`, `PIZ`, `RLE`, `Uncompressed`, `PXR24`, `B44`, `B44A`, … (+2 more) | `ZIP` |  |  |
| `depth` | No | `IMAGE` |  |  |  |
| `normal` | No | `IMAGE` |  |  |  |
| `albedo` | No | `IMAGE` |  |  |  |
| `custom_1` | No | `IMAGE` |  |  |  |
| `custom_1_name` | No | `STRING` | `emission` |  |  |
| `custom_2` | No | `IMAGE` |  |  |  |
| `custom_2_name` | No | `STRING` | `specular` |  |  |
| `output_path` | No | `STRING` | `` |  |  |
| `remote_path` | No | `STRING` | `` |  |  |
| `frame_index` | No | `INT` | `1` | ≥ 1 |  |
| `custom_metadata` | No | `STRING` | `` | multiline |  |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `output_path` | `STRING` |  |

---
