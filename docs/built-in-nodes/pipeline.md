# Pipeline and Studio

Project containers, audio cut data, blend composites, cinema prompt setup, local MCP bridge, Nuke send, Resolve handoff, and parameter history tracking.

## Typical workflow

```text
Project Manager -> processing graph -> Write -> Nuke Send / DaVinci Send / MCP Bridge
```

## Before you use these nodes

- Treat bridge nodes as local studio tools and verify host, port, and token settings.
- Use project containers for repeatable shot and version handling.
- Resolve handoff is folder based by default; live scripting requires the Resolve helper.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Audio Cut](#audio-cut) | `RadianceAudioCut` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Project Manager](#project-manager) | `RadianceProjectManager` | Pipeline project manager — save, list, load, delete, and inspect .rad workflow containers. |
| [◎ Blend Composite](#blend-composite) | `RadianceBlendComposite` | Composite two images using industry-standard blend modes. |
| [◎ Cinema Studio](#cinema-studio) | `RadianceCinemaStudio` | Performs the Radiance operation described by its inputs and outputs in the selected workflow group. |
| [◎ Radiance MCP Bridge](#radiance-mcp-bridge) | `RadianceMCP` | Local bridge node for pipeline and DCC handoff actions. |
| [◎ Radiance Send to Nuke](#radiance-send-to-nuke) | `RadianceNukeSend` | Exports and sends media to a local Nuke listener. |
| [◎ Radiance Send to DaVinci Resolve](#radiance-send-to-davinci-resolve) | `RadianceDaVinciSend` | Exports media into a DaVinci Resolve handoff folder. |
| [◎ Parameter History Tracker](#parameter-history-tracker) | `RadianceParamHistoryTracker` | Parameter History Tracker. |

## ◎ Audio Cut

**Internal key:** `RadianceAudioCut`  
**Category:** `FXTD STUDIOS/Radiance/◎ Video`  
**Source:** `nodes/pipeline/audio.py`
**Function:** `detect_cuts`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Audio Cut` when the graph reaches the Audio Cut step in a pipeline and studio workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `audio_filepath` | Yes | `STRING` | `/path/to/audio.wav` | Absolute path to audio file (WAV, MP3, FLAC, AAC, etc.) |
| `fps` | Yes | `FLOAT` | `24.0` | Frame rate of the target video sequence |
| `method` | Yes | `(cls.METHODS, {'default': 'beats'})` | - | - |
| `sensitivity` | Yes | `FLOAT` | `0.5` | 0 = only strong peaks, 1 = detect all micro-variations |
| `min_interval_frames` | Yes | `INT` | `12` | Minimum frames between consecutive cut points |
| `backend` | Yes | `(cls.BACKENDS, {'default': 'Auto'})` | - | - |
| `frame_offset` | Optional | `INT` | `0` | Add this value to every returned frame index (useful when audio starts mid-sequence) |
| `max_cuts` | Optional | `INT` | `0` | If > 0, keep only the strongest N cut points |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `cut_frames_json` | `STRING` | Output produced by the `cut_frames_json` socket. |
| `cut_times_json` | `STRING` | Output produced by the `cut_times_json` socket. |
| `cut_count` | `INT` | Output produced by the `cut_count` socket. |
| `analysis_report` | `STRING` | Output produced by the `analysis_report` socket. |

### Practical notes

- The node returns `cut_frames_json` (`STRING`), `cut_times_json` (`STRING`), `cut_count` (`INT`), `analysis_report` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Project Manager

**Internal key:** `RadianceProjectManager`  
**Category:** `FXTD STUDIOS/Radiance/◎ Pipeline`  
**Source:** `nodes_workspace.py`
**Function:** `run`

### What it does

Pipeline project manager — save, list, load, delete, and inspect .rad workflow containers.

### When to use it

Use `◎ Project Manager` when the graph reaches the Project Manager step in a pipeline and studio workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `filename` | Optional | `STRING` | `` | Workflow filename stem (version appended automatically). |
| `artist` | Optional | `STRING` | `` | Artist name saved in workflow metadata. |
| `version` | Optional | `INT` | `1` | Version number for the saved workflow. |

### Outputs

This node does not declare named runtime outputs in the source catalog.

### Practical notes

- Keep this node in the part of the graph indicated by its group workflow and inspect all report or metadata outputs when debugging.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Blend Composite

**Internal key:** `RadianceBlendComposite`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/pipeline/overlay.py`
**Function:** `composite`

### What it does

Composite two images using industry-standard blend modes.

### When to use it

Use `◎ Blend Composite` when the graph reaches the Blend Composite step in a pipeline and studio workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `base` | Yes | `IMAGE` | - | Bottom layer (background). |
| `blend` | Yes | `IMAGE` | - | Top layer (foreground). |
| `mode` | Yes | `(BLEND_MODES, {'default': 'Normal'})` | - | - |
| `opacity` | Yes | `FLOAT` | `1.0` | Overall strength of the blend layer. |
| `mask` | Optional | `MASK` | - | Optional per-pixel mask (grayscale). White = full blend, Black = base only. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Output produced by the `image` socket. |

### Practical notes

- The node returns `image` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Cinema Studio

**Internal key:** `RadianceCinemaStudio`  
**Category:** `FXTD STUDIOS/Radiance/◎ Generate`  
**Source:** `nodes/pipeline/studio.py`
**Function:** `generate_cinema_prompt`

### What it does

Performs the Radiance operation described by its inputs and outputs in the selected workflow group.

### When to use it

Use `◎ Cinema Studio` when the graph reaches the Cinema Studio step in a pipeline and studio workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `base_prompt` | Yes | `STRING` | `A cinematic shot of...` | - |
| `camera` | Yes | `(cls.CAMERA_LIST, {'default': cls.CAMERA_LIST[1] if len(cls.CAMERA_LIST) > 1 else 'None'})` | - | - |
| `lens_series` | Yes | `(cls.LENS_LIST, {'default': cls.LENS_LIST[1] if len(cls.LENS_LIST) > 1 else 'None'})` | - | - |
| `focal_length` | Yes | `(cls.FOCAL_LENGTHS, {'default': '50mm Standard'})` | - | - |
| `aperture` | Yes | `(APERTURES, {'default': 'T2.0 (Cinematic Separation)'})` | - | - |
| `shutter` | Yes | `(SHUTTER_ANGLES, {'default': '180° (Standard Motion - 1/48s)'})` | - | - |
| `iso` | Yes | `(ISO_SETTINGS, {'default': '800 ISO (Native Digital)'})` | - | - |
| `shot_type` | Optional | `ENUM: None, Extreme Wide Shot, Wide Shot, Full Shot, Medium Wide Shot, Medium Shot, Medium Close-Up, Close-Up...` | `Medium Shot` | - |
| `camera_movement` | Optional | `ENUM: None, Static Tripod, Handheld Shake, Steadicam Smooth, Dolly In, Dolly Out, Truck Left, Truck Right...` | `None` | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `prompt` | `STRING` | Output produced by the `prompt` socket. |
| `technical_data_str` | `STRING` | Output produced by the `technical_data_str` socket. |

### Practical notes

- The node returns `prompt` (`STRING`), `technical_data_str` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance MCP Bridge

**Internal key:** `RadianceMCP`  
**Category:** `FXTD STUDIOS/Radiance/07 Pipeline & DCC`  
**Source:** `nodes/pipeline/dcc.py`
**Function:** `run`

### What it does

Local bridge node for pipeline and DCC handoff actions.

### When to use it

Use `◎ Radiance MCP Bridge` when the graph reaches the MCP Bridge step in a pipeline and studio workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `mode` | Yes | `(_MCP_MODES, {'default': 'Export Frames', 'tooltip': 'Export Frames = save EXR/video for DCC. Bridge Server = start TCP control server.'})` | - | - |
| `source` | Yes | `(_SOURCES, {'default': 'Auto', 'tooltip': 'Auto = try Images, then Video, then Sequence. Select explicitly to avoid ambiguity.'})` | - | - |
| `target` | Yes | `(_DCC_TARGETS, {'default': 'Nuke', 'tooltip': 'Target DCC application (metadata hint).'})` | - | - |
| `output_path` | Yes | `STRING` | `` | Output directory for EXR frames (Export mode) or bridge log (Bridge mode). |
| `format` | Yes | `(_EXR_FORMATS, {'default': 'EXR (16-bit half)', 'tooltip': 'EXR bit depth. +H.264 or +ProRes also generates a video file.'})` | - | - |
| `images` | Optional | `IMAGE` | - | Batch of frames to export (used when source is Images or Auto). |
| `video_path` | Optional | `STRING` | `` | Path to a video file (.mp4, .mov, etc.) to decode and export (source=Video or Auto). |
| `sequence_path` | Optional | `STRING` | `` | Path/pattern to an image sequence e.g. /frames/frame.%04d.exr (source=Sequence or Auto). |
| `fps` | Optional | `FLOAT` | `24.0` | Frame rate for video export. |
| `frame_start` | Optional | `INT` | `1001` | Starting frame number for EXR sequence export. |
| `frame_end` | Optional | `INT` | `0` | Last frame index (0 = read all found frames, for sequences only). |
| `filename_prefix` | Optional | `STRING` | `frame` | Prefix for EXR filenames (e.g. frame_1001.exr). |
| `bridge_port` | Optional | `INT` | `1987` | TCP port for Bridge Server (default 1987). |
| `bridge_host` | Optional | `STRING` | `127.0.0.1` | Bind address (127.0.0.1 = loopback only; 0.0.0.0 = all interfaces). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `status` | `STRING` | Output produced by the `status` socket. |
| `render_path` | `STRING` | Output produced by the `render_path` socket. |

### Practical notes

- The node returns `status` (`STRING`), `render_path` (`STRING`).
- Confirm host, port, output path, and token settings before running bridge actions.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Send to Nuke

**Internal key:** `RadianceNukeSend`  
**Category:** `FXTD STUDIOS/Radiance/07 Pipeline & DCC`  
**Source:** `nodes/pipeline/studio_integrations.py`
**Function:** `run`

### What it does

Exports and sends media to a local Nuke listener.

### When to use it

Use `◎ Radiance Send to Nuke` near the end of the graph after the image, sequence, or metadata is ready for delivery.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | Frame to export. A batch is written as a numbered EXR sequence. |
| `nuke_folder` | Yes | `STRING` | `` | Output folder for image + .nk file. Created if missing. |
| `filename` | Yes | `STRING` | `radiance_out` | Base name for the EXR file(s). |
| `frame_start` | Optional | `INT` | `1001` | Starting frame number for the EXR sequence. |
| `push_to_nuke` | Optional | `BOOLEAN` | `False` | If True and Nuke listener is running, auto-create a Read node via TCP. |
| `nuke_host` | Optional | `STRING` | `127.0.0.1` | Nuke listener host (used only when push_to_nuke=True). |
| `nuke_port` | Optional | `INT` | `1986` | Nuke listener port (used only when push_to_nuke=True). |
| `half_float` | Optional | `BOOLEAN` | `True` | Write 16-bit half EXR (True) or 32-bit float EXR (False). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `status` | `STRING` | Output produced by the `status` socket. |
| `render_path` | `STRING` | Output produced by the `render_path` socket. |

### Practical notes

- The node returns `status` (`STRING`), `render_path` (`STRING`).
- Confirm host, port, output path, and token settings before running bridge actions.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Radiance Send to DaVinci Resolve

**Internal key:** `RadianceDaVinciSend`  
**Category:** `FXTD STUDIOS/Radiance/07 Pipeline & DCC`  
**Source:** `nodes/pipeline/studio_integrations.py`
**Function:** `run`

### What it does

Exports media into a DaVinci Resolve handoff folder.

### When to use it

Use `◎ Radiance Send to DaVinci Resolve` near the end of the graph after the image, sequence, or metadata is ready for delivery.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `image` | Yes | `IMAGE` | - | Frame to export. Batches write numbered files. |
| `resolve_folder` | Yes | `STRING` | `` | DaVinci Resolve shared media folder. Created if missing. |
| `filename` | Yes | `STRING` | `radiance_out` | Base filename (no extension). |
| `bit_depth` | Yes | `ENUM: 16bit, 8bit, EXR` | `16bit` | Output bit depth. EXR writes 16-bit half-float. |
| `frame_start` | Optional | `INT` | `1001` | Starting frame number for numbered sequences. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `status` | `STRING` | Output produced by the `status` socket. |
| `render_path` | `STRING` | Output produced by the `render_path` socket. |

### Practical notes

- The node returns `status` (`STRING`), `render_path` (`STRING`).
- Confirm host, port, output path, and token settings before running bridge actions.
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Parameter History Tracker

**Internal key:** `RadianceParamHistoryTracker`  
**Category:** `FXTD STUDIOS/Radiance/◎ Infrastructure`  
**Source:** `core/param_memory.py`
**Function:** `record`

### What it does

Parameter History Tracker.

### When to use it

Use `◎ Parameter History Tracker` when the graph reaches the Parameter History Tracker step in a pipeline and studio workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `node_name` | Yes | `STRING` | `RadianceHDREncoder` | - |
| `parameters_json` | Yes | `STRING` | `{}` | - |
| `trigger` | Optional | `*` | - | - |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `history_summary` | `STRING` | Output produced by the `history_summary` socket. |
| `parameter_diff` | `STRING` | Output produced by the `parameter_diff` socket. |

### Practical notes

- The node returns `history_summary` (`STRING`), `parameter_diff` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
