[← Back to Radiance docs](../README.md)

# Pipeline and Studio

Project, shot and asset management, and handoff to Nuke and DaVinci Resolve.

## Typical graph

```text
Project Manager (organise) → work → Send to Nuke / Send to Resolve
```

## Before you use these nodes

- **The dashboards open in-canvas**, as an overlay on the ComfyUI graph rather
  than a separate browser tab. Launch them from the Project Manager node.
- **The Nuke bridge is local and structured.** It binds to `127.0.0.1` and
  accepts a fixed set of production actions. It does not evaluate arbitrary
  Python — that path was removed in 3.2.0 after it was found to be exploitable
  through its own blocklist.
- **Resolve handoff is a folder drop.** Radiance writes PNG, TIFF or EXR into a
  directory that Resolve imports; there is no live link.
- **Write first, send second.** Both bridges hand over media that already
  exists on disk.

## Nodes in this section (4)

| Node | Key | What it does |
| :--- | :--- | :--- |
| [Export to Nuke](#export-to-nuke) | `RadianceNukeSend` | Export image as EXR and write a .nk Read-node snippet for direct Nuke import |
| [Export to Resolve](#export-to-resolve) | `RadianceDaVinciSend` | Export the current image to a DaVinci Resolve shared media folder for manual import |
| [MCP Bridge](#mcp-bridge) | `RadianceMCP` | MCP Bridge — Export frames as EXR/video for DCC consumption, or start a TCP bridge server for command/control between ComfyUI and DCC apps. |
| [Project Manager](#project-manager) | `RadianceProjectManager` | Save the current workflow graph as a .rad container with artist and version metadata. |

---

## Export to Nuke

**Node key:** `RadianceNukeSend`  
**Menu:** `FXTD STUDIOS/Radiance/Pipeline`  
**Source:** `nodes/pipeline/studio_integrations.py`  
**Output node** — runs even with nothing connected downstream.  

Export image as EXR and write a .nk Read-node snippet for direct Nuke import. Optionally push to a running Nuke instance via the Radiance TCP listener.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | Frame to export. A batch is written as a numbered EXR sequence. |
| `nuke_folder` | Yes | `STRING` | `` |  | Output folder for image + .nk file. Created if missing. |
| `filename` | Yes | `STRING` | `radiance_out` |  | Base name for the EXR file(s). |
| `frame_start` | No | `INT` | `1001` | 0 – 999999 | Starting frame number for the EXR sequence. |
| `push_to_nuke` | No | `BOOLEAN` | `False` |  | If True and Nuke listener is running, auto-create a Read node via TCP. |
| `nuke_host` | No | `STRING` | `127.0.0.1` |  | Nuke listener host (used only when push_to_nuke=True). |
| `nuke_port` | No | `INT` | `1986` | 1024 – 65535 | Nuke listener port (used only when push_to_nuke=True). |
| `half_float` | No | `BOOLEAN` | `True` |  | Write 16-bit half EXR (True) or 32-bit float EXR (False). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `status` | `STRING` |  |
| `render_path` | `STRING` |  |

---

## Export to Resolve

**Node key:** `RadianceDaVinciSend`  
**Menu:** `FXTD STUDIOS/Radiance/Pipeline`  
**Source:** `nodes/pipeline/studio_integrations.py`  
**Output node** — runs even with nothing connected downstream.  

Export the current image to a DaVinci Resolve shared media folder for manual import. Supports 8-bit PNG, 16-bit TIFF, and EXR output formats.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` |  |  | Frame to export. Batches write numbered files. |
| `resolve_folder` | Yes | `STRING` | `` |  | DaVinci Resolve shared media folder. Created if missing. |
| `filename` | Yes | `STRING` | `radiance_out` |  | Base filename (no extension). |
| `bit_depth` | Yes | choice of `16bit`, `8bit`, `EXR` | `16bit` |  | Output bit depth. EXR writes 16-bit half-float. |
| `frame_start` | No | `INT` | `1001` | 0 – 999999 | Starting frame number for numbered sequences. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `status` | `STRING` |  |
| `render_path` | `STRING` |  |

---

## MCP Bridge

**Node key:** `RadianceMCP`  
**Menu:** `FXTD STUDIOS/Radiance/Pipeline`  
**Source:** `nodes/pipeline/dcc.py`  
**Output node** — runs even with nothing connected downstream.  

MCP Bridge — Export frames as EXR/video for DCC consumption, or start a TCP bridge server for command/control between ComfyUI and DCC apps.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `mode` | Yes | choice of `Export Frames`, `Bridge Server` | `Export Frames` |  | Export Frames = save EXR/video for DCC. Bridge Server = start TCP control server. |
| `source` | Yes | choice of `Auto`, `Images`, `Video`, `Sequence` | `Auto` |  | Auto = try Images, then Video, then Sequence. Select explicitly to avoid ambiguity. |
| `target` | Yes | choice of `Nuke`, `Resolve`, `Fusion` | `Nuke` |  | Target DCC application (metadata hint). |
| `output_path` | Yes | `STRING` | `` |  | Output directory for EXR frames (Export mode) or bridge log (Bridge mode). |
| `format` | Yes | choice of `EXR (16-bit half)`, `EXR (32-bit float)`, `EXR + H.264 MP4`, `EXR + ProRes MOV` | `EXR (16-bit half)` |  | EXR bit depth. +H.264 or +ProRes also generates a video file. |
| `images` | No | `IMAGE` |  |  | Batch of frames to export (used when source is Images or Auto). |
| `video_path` | No | `STRING` | `` |  | Path to a video file (.mp4, .mov, etc.) to decode and export (source=Video or Auto). |
| `sequence_path` | No | `STRING` | `` |  | Path/pattern to an image sequence e.g. /frames/frame.%04d.exr (source=Sequence or Auto). |
| `fps` | No | `FLOAT` | `24.0` | 1.0 – 240.0, step 0.001 | Frame rate for video export. |
| `frame_start` | No | `INT` | `1001` | 0 – 999999 | Starting frame number for EXR sequence export. |
| `frame_end` | No | `INT` | `0` | 0 – 999999 | Last frame index (0 = read all found frames, for sequences only). |
| `filename_prefix` | No | `STRING` | `frame` |  | Prefix for EXR filenames (e.g. frame_1001.exr). |
| `bridge_port` | No | `INT` | `1987` | 1024 – 65535 | TCP port for Bridge Server (default 1987). |
| `bridge_host` | No | `STRING` | `127.0.0.1` |  | Bind address (127.0.0.1 = loopback only; 0.0.0.0 = all interfaces). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `status` | `STRING` |  |
| `render_path` | `STRING` |  |

---

## Project Manager

**Node key:** `RadianceProjectManager`  
**Menu:** `FXTD STUDIOS/Radiance/Core`  
**Source:** `nodes_workspace.py`  
**Output node** — runs even with nothing connected downstream.  

Save the current workflow graph as a .rad container with artist and version metadata.

### Inputs

| Input | Required | Type | Default | Range | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `filename` | No | `STRING` | `` |  | Workflow filename stem (version appended automatically). |
| `artist` | No | `STRING` | `` |  | Artist name saved in workflow metadata. |
| `version` | No | `INT` | `1` | 1 – 9999 | Version number for the saved workflow. |

Hidden inputs supplied by ComfyUI: `extra_pnginfo`, `prompt`.

### Outputs

None — this node is a terminal (it writes, sends, or displays).

---
