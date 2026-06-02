# Review, Viewer, and Preview

Interactive viewer, lightweight viewer, focus peaking, contact sheets, flipbook GIFs, frame stamps, and local preview server outputs.

## Typical workflow

```text
Processed image batch -> Viewer / Contact Sheet / Frame Stamp / Flipbook GIF / Preview Server
```

## Before you use these nodes

- Use review nodes on copies or branches so stamped/proxy outputs do not overwrite masters.
- Contact sheets and GIFs are for approval; use EXR sequences for final comp review.
- Keep preview servers local unless the workstation network policy says otherwise.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Radiance Lite Viewer](#radiance-lite-viewer) | `RadianceLiteViewer` | Fast viewer for compare/check workflows. |
| [◎ Radiance Viewer](#radiance-viewer) | `RadianceViewer` | Production review viewer with image passthrough and viewer tooling. |
| [◎ Focus Peaking](#focus-peaking) | `RadianceFocusPeaking` | Focus peaking monitor — highlight in-focus (sharp) regions with a. |
| [◎ Contact Sheet](#contact-sheet) | `RadianceContactSheet` | Generate a thumbnail contact sheet from an IMAGE batch. |
| [◎ Flipbook GIF](#flipbook-gif) | `RadianceFlipbookGIF` | Export an IMAGE batch as an animated GIF for quick preview sharing. |
| [◎ Frame Stamp](#frame-stamp) | `RadianceFrameStamp` | Burn timecode, frame number, and custom text into frames. |
| [◎ Preview Server](#preview-server) | `RadiancePreviewServer` | HTTP preview server — serve the most recent processed frame as JPEG to. |

## ◎ Radiance Lite Viewer

**Internal key:** `RadianceLiteViewer`  
**Category:** `FXTD STUDIOS/Radiance/◎ Review`  
**Source:** `nodes/monitor/lite_viewer.py`
**Function:** `view`

### What it does

Fast viewer for compare/check workflows.

### When to use it

Use `◎ Radiance Lite Viewer` on a review branch so you can inspect output without changing the master path.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `(image_video_type,)` | - | - |
| `compare_image` | Optional | `(image_video_type, {'tooltip': 'Optional B image for wipe, split, diff, and onion checks.'})` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Viewer

**Internal key:** `RadianceViewer`  
**Category:** `FXTD STUDIOS/Radiance/◎ Display`  
**Source:** `nodes/monitor/viewer.py`
**Function:** `view`

### What it does

Production review viewer with image passthrough and viewer tooling.

### When to use it

Use `◎ Radiance Viewer` on a review branch so you can inspect output without changing the master path.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `(image_video_type,)` | - | - |
| `compare_image` | Optional | `(image_video_type,)` | - | - |
| `zdepth` | Optional | `(image_video_type, {'tooltip': 'Z-Depth map to display when pressing Z button'})` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Focus Peaking

**Internal key:** `RadianceFocusPeaking`  
**Category:** `FXTD STUDIOS/Radiance/◎ QC & Debug`  
**Source:** `nodes_realtime_preview.py`
**Function:** `peak`

### What it does

Focus peaking monitor — highlight in-focus (sharp) regions with a.

### When to use it

Use `◎ Focus Peaking` when the graph reaches the Focus Peaking step in a review, viewer, and preview workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | - |
| `threshold` | Yes | `FLOAT` | `0.2` | Normalised Sobel magnitude above which a pixel is considered in-focus. |
| `peak_color` | Yes | `ENUM: Red, Green, White, Yellow, Cyan` | `Red` | - |
| `strength` | Yes | `FLOAT` | `0.85` | Blend factor for the peaking overlay. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passthrough` | `IMAGE` | Output produced by the `passthrough` socket. |
| `focus_peak` | `IMAGE` | Output produced by the `focus_peak` socket. |

### Practical notes

- The node returns `passthrough` (`IMAGE`), `focus_peak` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Contact Sheet

**Internal key:** `RadianceContactSheet`  
**Category:** `FXTD STUDIOS/Radiance/◎ QC & Debug`  
**Source:** `nodes_realtime_preview.py`
**Function:** `sheet`

### What it does

Generate a thumbnail contact sheet from an IMAGE batch.

### When to use it

Use `◎ Contact Sheet` when the graph reaches the Contact Sheet step in a review, viewer, and preview workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` | - | - |
| `thumb_width` | Yes | `INT` | `160` | Width of each thumbnail in pixels. |
| `max_cols` | Yes | `INT` | `8` | Maximum number of columns. Rows are computed automatically. |
| `label_frames` | Optional | `BOOLEAN` | `True` | Print the frame index below each thumbnail. |
| `background` | Optional | `ENUM: Black, Grey, White` | `Black` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `contact_sheet` | `IMAGE` | Output produced by the `contact_sheet` socket. |
| `grid_cols` | `INT` | Output produced by the `grid_cols` socket. |
| `grid_rows` | `INT` | Output produced by the `grid_rows` socket. |

### Practical notes

- The node returns `contact_sheet` (`IMAGE`), `grid_cols` (`INT`), `grid_rows` (`INT`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Flipbook GIF

**Internal key:** `RadianceFlipbookGIF`  
**Category:** `FXTD STUDIOS/Radiance/◎ QC & Debug`  
**Source:** `nodes_realtime_preview.py`
**Function:** `export_gif`

### What it does

Export an IMAGE batch as an animated GIF for quick preview sharing.

### When to use it

Use `◎ Flipbook GIF` when the graph reaches the Flipbook GIF step in a review, viewer, and preview workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` | - | - |
| `save_path` | Yes | `STRING` | `preview/flipbook.gif` | Output .gif path. Directory is created automatically. |
| `fps` | Yes | `FLOAT` | `12.0` | Playback speed. GIF frame delay = 1000/fps ms. |
| `max_width` | Yes | `INT` | `480` | Resize frames to this width (preserves aspect ratio). Smaller = smaller file. |
| `loop` | Optional | `BOOLEAN` | `True` | Loop the animation indefinitely. |
| `dither` | Optional | `BOOLEAN` | `True` | Enable Floyd-Steinberg dithering for smoother gradients. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passthrough` | `IMAGE` | Output produced by the `passthrough` socket. |
| `status` | `STRING` | Output produced by the `status` socket. |

### Practical notes

- The node returns `passthrough` (`IMAGE`), `status` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Frame Stamp

**Internal key:** `RadianceFrameStamp`  
**Category:** `FXTD STUDIOS/Radiance/◎ QC & Debug`  
**Source:** `nodes_realtime_preview.py`
**Function:** `stamp`

### What it does

Burn timecode, frame number, and custom text into frames.

### When to use it

Use `◎ Frame Stamp` when the graph reaches the Frame Stamp step in a review, viewer, and preview workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` | - | - |
| `start_frame` | Yes | `INT` | `1001` | Frame number of the first frame in the batch. |
| `fps` | Yes | `FLOAT` | `24.0` | Frames per second (used for timecode calculation). |
| `drop_frame` | Optional | `BOOLEAN` | `False` | Use SMPTE drop-frame timecode (DF). Only meaningful at 29.97 or 59.94 fps. Uses ';' separator instead of ':' for the frame field. |
| `show_frame_number` | Optional | `BOOLEAN` | `True` | - |
| `show_timecode` | Optional | `BOOLEAN` | `True` | - |
| `custom_text` | Optional | `STRING` | `` | Additional text burned into each frame (e.g. shot name, version). |
| `position` | Optional | `ENUM: bottom_left, bottom_right, top_left, top_right, center` | `bottom_left` | - |
| `font_scale` | Optional | `FLOAT` | `1.0` | - |
| `opacity` | Optional | `FLOAT` | `0.85` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `stamped` | `IMAGE` | Output produced by the `stamped` socket. |

### Practical notes

- The node returns `stamped` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Preview Server

**Internal key:** `RadiancePreviewServer`  
**Category:** `FXTD STUDIOS/Radiance/◎ QC & Debug`  
**Source:** `nodes_realtime_preview.py`
**Function:** `serve`

### What it does

HTTP preview server — serve the most recent processed frame as JPEG to.

### When to use it

Use `◎ Preview Server` on a review branch so you can inspect output without changing the master path.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` | - | - |
| `port` | Yes | `INT` | `8765` | TCP port for the preview HTTP server. |
| `stream_name` | Yes | `STRING` | `radiance` | Stream identifier. Access at /frame/<stream_name>. |
| `jpeg_quality` | Optional | `INT` | `85` | JPEG compression quality (20=small, 99=lossless-ish). |
| `resize_width` | Optional | `INT` | `0` | Resize frame before serving (0 = original size). Smaller = faster over network. |
| `enabled` | Optional | `BOOLEAN` | `True` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `passthrough` | `IMAGE` | Output produced by the `passthrough` socket. |
| `server_url` | `STRING` | Output produced by the `server_url` socket. |

### Practical notes

- The node returns `passthrough` (`IMAGE`), `server_url` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
