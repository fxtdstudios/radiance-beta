<div align="center">
<img src="r_icon.png" width="76" alt="Radiance mark"><br>
<img src="RADIANCE.png" width="640" alt="Radiance">

# Radiance

**Professional VFX, HDR color science, review, and DCC handoff for ComfyUI.**

[![Version](https://img.shields.io/badge/version-3.1.1-c8a96e?style=for-the-badge)](https://github.com/fxtdstudios/radiance)
[![License](https://img.shields.io/badge/license-GPL--3.0-green?style=for-the-badge)](LICENSE)
[![Nodes](https://img.shields.io/badge/nodes-114-c8a96e?style=for-the-badge)](#node-map)
[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-Radiance-orange?style=for-the-badge)](https://registry.comfy.org/nodes/radiance)

Radiance is a production-grade node pack for ComfyUI built around 32-bit float and HDR/ACES image pipelines. It brings VFX plate prep, color management, review tooling, in-canvas studio dashboards, and Nuke / DaVinci Resolve handoff into one coherent toolkit — so you can take a shot from generation through finishing without leaving the graph.

Artists get 32-bit, HDR, and ACES image tools, professional viewers, and VFX nodes. Supervisors and coordinators get project, shot, asset, and workflow management built directly into the canvas.

[Install](#installation) · [Features](#features) · [Studio Dashboards](#studio-dashboards) · [Node Map](#node-map) · [Documentation](docs/README.md) · [DCC Handoff](#dcc-handoff) · [Support](#support)

</div>

---

## Features

- 32-bit float and EXR workflows for VFX and finishing, with lossless scene-linear round-trips.
- ACES, OCIO, log curves, LUTs, CDL, scopes, QC, and grade-transfer tools.
- VFX utilities for plate prep, masks, roto, depth, camera and optics, motion, multipass, real AOV ingestion, and relighting.
- Video and temporal workflow nodes for loading, routing, conditioning, sampling, and delivery.
- In-canvas studio dashboards — Project Manager, Workflow Library, and Assets — rendered over the ComfyUI graph, never in a separate browser tab.
- **Radiance Sampler** — a preset-driven sampler that hides irrelevant parameters and adapts to the selected model.
- A full-featured **Viewer** and a lightweight **Lite Viewer** with scopes, frame review, and keyboard shortcuts.
- HDR VAE decoders (Turbo and Full) and HDR LoRA tooling for scene-linear generation.
- Dynamic Gizmos — collapse any group of nodes into a single reusable custom node.
- Gaussian Splatting — load, render, and train 3D Gaussian Splatting scenes; renders feed the color/HDR/finishing pipeline.
- Secure-by-default handoff to Nuke and DaVinci Resolve.

## Studio Dashboards

Radiance includes three production dashboards that open in-canvas — as an overlay on top of the ComfyUI graph rather than a new browser tab — from the Radiance Project Manager node. All three share a clean, dark interface.

| Dashboard | Purpose |
| :--- | :--- |
| **Project Manager** | Show, sequence, and shot view with a status pipeline (WIP, Review, Approved, Retake), version history, a click-through shot panel, project storage, and recent outputs — backed by a live view of your saved workflows. |
| **Workflow Library** | Browse, search, preview, and load saved workflows back into the canvas, organized by production bins. |
| **Assets** | A media manager that scans your ComfyUI input and output folders and classifies images, videos, and image sequences (auto-grouped by frame range). Create custom bins, filter by type, search, drag and drop to import, and inspect each asset in a detail panel. |

The Project Manager node keeps its launchers (open, save, and links) in a single compact panel on the node.

## Smart Interface

- **Adaptive sampler.** The Radiance Sampler hides every parameter when no preset is selected, shows everything in Custom mode, and for a named preset shows only the parameters relevant to that model — so you only see the controls that matter.
- **In-canvas overlays.** Dashboards open over the graph and close with Esc, the dimmed background, or the close button, with an option to open in a full tab.
- **Dynamic Gizmos.** Collapse any selection of nodes into a single styled custom node that you can save and reuse like any other node.
- **Smart Backdrops.** Group nodes get a clear, tinted-glass background keyed to the node category instead of a near-invisible panel.

## Installation

### ComfyUI Manager / Comfy Registry

Search for **Radiance** in ComfyUI Manager, or install it from the Comfy Registry.

### Manual install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fxtdstudios/radiance.git
cd radiance
pip install -r requirements.txt
```

Windows users can use `requirements_windows.txt`; Apple Silicon users can use `requirements_mac_silicon.txt`.

Radiance relies on the same PyTorch installation that ComfyUI uses, so install it inside the same Python environment as ComfyUI.

### Optional: Gaussian Splatting

Loading, inspecting, editing, and exporting splats (`.ply` / `.splat`) and COLMAP import work out of the box. **Rendering and training** require [`gsplat`](https://github.com/nerfstudio-project/gsplat) and an NVIDIA CUDA GPU:

```bash
pip install gsplat plyfile
```

## Feature Spotlights

### Viewers

- **Viewer** — a full review surface with waveform and vectorscope, channel isolation, A/B compare, focus peaking, frame stepping, and keyboard shortcuts (see [Viewer Shortcuts](#viewer-shortcuts)).
- **Radiance Lite Viewer** — a lightweight inline viewer for quick frame inspection.

### HDR VAE Decoders

- **Turbo Decoder** — a lightweight, near-realtime decode to scene-linear for fast iteration.
- **Full Decoder** — a deep decoder for production-quality reconstruction.
- Both are available through the Radiance HDR VAE Decode node, which also reports the decode settings it used.

### HDR LoRA

- **HDR LoRA Loader / Apply** — load and apply LoRAs tuned for HDR and scene-linear generation.
- **LoRA Stack** — combine multiple LoRAs with individual model and CLIP strengths.

### Dynamic Gizmos

Select any group of nodes and collapse them into a single styled Gizmo node — a reusable, shareable custom node that loads automatically with Radiance.

## Node Map

Radiance nodes are organized under a single menu:

```text
FXTD STUDIOS/Radiance
├─ Core
├─ Load & Save
├─ Generate
├─ Color
├─ HDR
├─ VFX
├─ Video
├─ Upscale
├─ Review
├─ Pipeline
└─ Gaussian Splatting
```

Radiance provides **114 nodes** (plus any Gizmos you create). Some nodes depend on optional packages and your ComfyUI environment.

Node names follow standard compositing vocabulary under the **Radiance** menu — `Grade`, `CDL`, `OCIO ColorSpace`, `Roto`, `Defocus`, `Viewer`, `Read`/`Write` — so they read the way they do in Nuke or Flame. AI and generation nodes keep a `Radiance` prefix (`Radiance Sampler`, `Radiance VAE Decode`) to mark the diffusion layer. You can still find any node by typing "radiance" in the search.

| Group | Examples |
| :--- | :--- |
| Core | Project Manager / Workspace, Resolution, workspace utilities |
| Load & Save | Read, Write (EXR alpha and mask), image and mask loading, EXR multipart and sequence export |
| Generate | Radiance Loader, Radiance Sampler, VAE Decode (HDR), prompt tools, LoRA stack, HDR LoRA, regional prompts |
| Color | Grade, Grade Match, CDL, LUTs, Curves, Hue Curves, White Balance, Color Space Convert |
| HDR | ACES 2.0, OCIO, HDR VAE encode/decode, tone mapping, HDR synthesis, relight, QC |
| VFX | Plate prep, masks, roto, depth, optics, motion, multipass, AOV reader (real EXR layers), relight |
| Video | Video loader, prompt builder, sampler, text-to-video, image-to-video, routing, batch decode, export |
| Upscale | Image and video upscale (HDR and color aware), tiling, face restoration |
| Review | Viewer, Lite Viewer, scopes, focus peaking, contact sheets, flipbook, preview server |
| Pipeline | Project Manager, Send to Nuke, DaVinci Resolve handoff |
| Gaussian Splatting | Splat Load/Info/Export, Transform, Crop, Merge, Camera Orbit, Splat Render, COLMAP Load, Splat Train |

## DCC Handoff

### Nuke

Radiance can export EXR frames and send them to a running Nuke session over a local connection.

Inside Nuke, run:

```python
exec(open("/path/to/ComfyUI/custom_nodes/radiance/scripts/start_nuke_server.py").read())
```

Then use the Send to Nuke node from ComfyUI. The listener binds to `127.0.0.1` by default and only accepts structured production actions.

### DaVinci Resolve

Radiance supports DaVinci Resolve through a folder handoff: the Send to DaVinci Resolve node exports PNG, TIFF, or EXR media into a folder Resolve can import.

## Viewer Shortcuts

| Key | Action |
| :--- | :--- |
| Space | Toggle playback |
| Left / Right | Previous / next frame |
| F | Fit to view |
| 1 | 1:1 pixel zoom |
| C / R / G / B / L | Color, red, green, blue, luma channels |
| W | Toggle waveform |
| V | Toggle vectorscope |
| A | Cycle A/B compare modes |

## Good to Know

- **Estimated VFX passes.** The Multipass Master extractor derives passes (albedo, roughness, ambient occlusion, segmentation ID, and more) from a single image — handy for 2D and generated footage, but not a substitute for true render passes. For ground-truth passes, feed a multilayer EXR through the Multipass AOV Reader. The segmentation output is a clustered matte, not a Cryptomatte.
- **Super-resolution and color.** Upscale backends work in display-referred space. For scene-linear input, use the upscaler's HDR and color-encoding options to preserve your values.

## Documentation

Full documentation is available at [radiance.fxtd.org](https://radiance.fxtd.org) and in the repository docs at [docs/](docs/README.md). It covers setup, core concepts, workflow recipes, a complete node reference, and troubleshooting.

## Support

- Issues: [GitHub Issues](https://github.com/fxtdstudios/radiance/issues)
- Documentation: [radiance.fxtd.org](https://radiance.fxtd.org)
- Studio: [fxtd.org](https://fxtd.org)

## License

Radiance is released under the [GPL-3.0 license](LICENSE).
