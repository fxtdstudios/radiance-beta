[← Back to Radiance docs](../README.md)

# AI Assist

Scene cut detection and shot splitting for per-shot processing and grade routing.

## Typical workflow

```text
Video/image batch -> Scene Cut Detect -> Scene Cut Split -> per-shot processing
```

## Before you use these nodes

- Run detection before per-shot grade, prompt, or export branches.
- Keep frame index metadata intact when splitting and reassembling shots.

## Nodes in this section

| Node | Internal key | Purpose |
| :--- | :--- | :--- |
| [◎ Scene Cut Detect](#scene-cut-detect) | `RadianceSceneCutDetect` | Detect hard shot cuts in a video sequence batch. |
| [◎ Scene Cut Split](#scene-cut-split) | `RadianceSceneCutSplit` | Split an IMAGE batch into per-shot sub-batches using cut_data from. |

## ◎ Scene Cut Detect

**Internal key:** `RadianceSceneCutDetect`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/ai/scene_cut.py`
**Function:** `detect`

### What it does

Detect hard shot cuts in a video sequence batch.

### When to use it

Use `◎ Scene Cut Detect` when the graph reaches the Scene Cut Detect step in a ai assist workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` | - | Full video sequence as IMAGE batch. |
| `threshold` | Yes | `FLOAT` | `0.35` | Cut sensitivity. Lower = detect more cuts. Typical range 0.25–0.45 for most footage. |
| `min_shot_frames` | Yes | `INT` | `12` | Minimum frames between detected cuts. |
| `method` | Yes | `ENUM: histogram, edge, combined` | `combined` | histogram: colour distribution diff (fast). edge: Sobel edge map diff (catches content cuts). combined: weighted blend of both (recommended). |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `cut_data` | `STRING` | Output produced by the `cut_data` socket. |
| `shot_count` | `INT` | Output produced by the `shot_count` socket. |
| `score_plot` | `IMAGE` | Output produced by the `score_plot` socket. |

### Practical notes

- The node returns `cut_data` (`STRING`), `shot_count` (`INT`), `score_plot` (`IMAGE`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.

## ◎ Scene Cut Split

**Internal key:** `RadianceSceneCutSplit`  
**Category:** `FXTD STUDIOS/Radiance/◎ VFX`  
**Source:** `nodes/ai/scene_cut.py`
**Function:** `split`

### What it does

Split an IMAGE batch into per-shot sub-batches using cut_data from.

### When to use it

Use `◎ Scene Cut Split` when the graph reaches the Scene Cut Split step in a ai assist workflow.

### Inputs

| Input | Required | Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Yes | `IMAGE` | - | - |
| `cut_data` | Yes | `STRING` | - | JSON from RadianceSceneCutDetect. |
| `shot_index` | Yes | `INT` | `0` | Which shot to extract (0-based). Connect shot_count output to know the range. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `frames` | `IMAGE` | Output produced by the `frames` socket. |
| `shot_index` | `INT` | Output produced by the `shot_index` socket. |
| `start_frame` | `INT` | Output produced by the `start_frame` socket. |
| `end_frame` | `INT` | Output produced by the `end_frame` socket. |
| `shot_info` | `STRING` | Output produced by the `shot_info` socket. |

### Practical notes

- The node returns `frames` (`IMAGE`), `shot_index` (`INT`), `start_frame` (`INT`), `end_frame` (`INT`), `shot_info` (`STRING`).
- If a result looks wrong, add a viewer, QC, or diagnostic node immediately after this node so the problem is isolated close to its source.
