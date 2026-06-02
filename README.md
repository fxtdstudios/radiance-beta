<div align="center">
<img src="RADIANCE.png" width="800" alt="Radiance Logo">

# ◎ Radiance

Professional VFX, HDR color science, review, and DCC handoff nodes for ComfyUI.

[![Version](https://img.shields.io/badge/version-3.1.0-00a8ff?style=for-the-badge)](https://github.com/fxtdstudios/radiance)
[![License](https://img.shields.io/badge/license-GPL--3.0-green?style=for-the-badge)](LICENSE)
[![Nodes](https://img.shields.io/badge/nodes-100%2B-blue?style=for-the-badge)](#node-map)
[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-Radiance-orange?style=for-the-badge)](https://registry.comfy.org/nodes/radiance)

**Radiance** is a production-oriented ComfyUI node pack for 32-bit image pipelines, HDR/ACES color management, VFX plate prep, video workflows, review tools, and Nuke/Resolve studio handoff.

[Install](#install) · [Documentation](docs/index.html) · [Node Map](#node-map) · [DCC Handoff](#dcc-handoff) · [Release Status](#release-status) · [Support](#support)

</div>

---

## Highlights

- 32-bit float image and EXR workflows for VFX and finishing.
- ACES, OCIO, log curves, LUTs, CDL, scopes, QC, and grade transfer tools.
- VFX utilities for plate prep, masks, roto, depth, camera/optics, motion, multipass, and relighting.
- Video and temporal workflow nodes for loading, routing, conditioning, sampling, and delivery.
- Radiance Pro Viewer with scopes, frame review, shortcuts, and optional local developer tools.
- DCC handoff for Nuke and DaVinci Resolve, with secure-by-default bridge behavior.

## Documentation

The documentation website starts at [docs/index.html](docs/index.html), with source Markdown in [docs/index.md](docs/index.md). It includes quickstart setup, production concepts, workflow recipes, a full node reference, troubleshooting, developer notes, and a coverage ledger for the registered node catalog.

## Install

### ComfyUI Manager / Comfy Registry

Search for **Radiance** in ComfyUI Manager or install from the Comfy Registry when published.

### Manual Git Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fxtdstudios/radiance.git
cd radiance
pip install -r requirements.txt
```

Windows users can use `requirements_windows.txt`; Apple Silicon users can use `requirements_mac_silicon.txt`.

> Radiance assumes ComfyUI already provides `torch`. Install Radiance inside the same Python environment used by ComfyUI.

## Node Map

Radiance is organized under:

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
└─ Developer
```

The source currently exposes **100+ node classes**. Runtime availability depends on installed optional dependencies and ComfyUI environment support.

### Core Groups

| Group | Examples |
| :--- | :--- |
| Core | Radiance Manager, Resolution, workspace utilities |
| Load & Save | Radiance Read, Radiance Write, image/mask loading, EXR and sequence export |
| Generate | Read Models, Sampler Pro, prompt tools, LoRA stack, regional prompts |
| Color | Grade, Grade Match, CDL, LUTs, Curves, White Balance, Color Space Convert |
| HDR | ACES, OCIO, HDR VAE, tone mapping, HDR encode/decode, QC |
| VFX | Plate prep, masks, roto, SAM, depth, optics, motion, multipass, relight |
| Video | Video loader, prompt builder, sampler, T2V/I2V, routing, export |
| Upscale | Image/video upscale, tiling, face restoration |
| Review | Viewer, scopes, contact sheets, preview server, policy guard |
| Pipeline | Project Manager, MCP Bridge, Nuke Send, DaVinci Resolve folder handoff |

## DCC Handoff

### Nuke

Radiance can export EXR frames and push them to a running Nuke session through the Radiance TCP listener.

Inside Nuke, run:

```python
exec(open("/path/to/ComfyUI/custom_nodes/radiance/scripts/start_nuke_server.py").read())
```

Then use **◎ Radiance Send to Nuke** or **◎ Radiance MCP Bridge** from ComfyUI.

Security defaults:

- Listener binds to `127.0.0.1` by default.
- Structured actions are enabled for normal production operations.
- Raw dynamic Python execution is disabled unless `RADIANCE_DEV=1`.
- Optional token auth uses `RADIANCE_DCC_AUTH_TOKEN`.

### DaVinci Resolve

Radiance currently supports Resolve as a **folder handoff/manual import** workflow through **◎ Radiance Send to DaVinci Resolve**. It exports PNG, TIFF, or EXR media into a Resolve-accessible folder.

The experimental `scripts/resolve_bridge.py` helper must be run inside DaVinci Resolve Studio when using Resolve scripting APIs.

## Release Status

| Area | Status |
| :--- | :--- |
| GitHub source layout | Ready |
| Comfy Registry metadata | Ready after publisher/token setup |
| Runtime dependency list | Ready |
| `.comfyignore` package cleanup | Ready |
| Node branding/menu taxonomy | Ready |
| Nuke bridge smoke test | Passed |
| MCP bridge smoke test | Passed |
| DaVinci Resolve live API push | Not included; folder handoff only |
| Local full pytest run | Not run in this shell because `pytest` is not installed |
| Full ComfyUI import test | Must be run inside the target ComfyUI environment before tagging |

## Publish Checklist

1. Confirm `pyproject.toml` has the final version.
2. Confirm your Comfy Registry publisher id is `fxtdstudios`.
3. Add the GitHub secret `REGISTRY_ACCESS_TOKEN`.
4. Run the lightweight local release check:

```bash
python tools/check_release_ready.py
```

5. Run CI on GitHub.
6. Tag the release:

```bash
git tag v3.1.0
git push origin v3.1.0
```

The publish workflow will publish to Comfy Registry and create a GitHub Release after the publish gate passes.

## Viewer Shortcuts

![Radiance Pro Viewer](viewer.png)
![Radiance Shortcuts](Viewer_shortcut.png)

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

## Support

- Issues: [GitHub Issues](https://github.com/fxtdstudios/radiance/issues)
- Documentation: [radiance.fxtd.org](https://radiance.fxtd.org)
- Studio: [fxtd.org](https://fxtd.org)

## License

Radiance is released under the [GPL-3.0 license](LICENSE).
