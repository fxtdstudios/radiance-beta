<div align="center">
<img src="r_icon.png" width="76" alt="Radiance mark"><br>
<img src="RADIANCE.png" width="640" alt="Radiance">

**Professional VFX, HDR color science, review, and DCC handoff for ComfyUI.**

[![Version](https://img.shields.io/badge/version-3.2.1-c8a96e?style=for-the-badge)](https://github.com/fxtdstudios/radiance)
[![License](https://img.shields.io/badge/license-GPL--3.0-green?style=for-the-badge)](LICENSE)
[![Nodes](https://img.shields.io/badge/nodes-110-c8a96e?style=for-the-badge)](#node-map)
[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-Radiance-orange?style=for-the-badge)](https://registry.comfy.org/nodes/radiance)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-RUDRA_models-ffd21e?style=for-the-badge)](https://huggingface.co/fxtdstudios/RUDRA)

Radiance is a production-grade node pack for ComfyUI built around 32-bit float and HDR/ACES image pipelines. It brings VFX plate prep, color management, review tooling, in-canvas studio dashboards, and Nuke / DaVinci Resolve handoff into one coherent toolkit, so you can take a shot from generation through finishing without leaving the graph.

Artists get 32-bit, HDR, and ACES image tools, professional viewers, and VFX nodes. Supervisors and coordinators get project, shot, asset, and workflow management built directly into the canvas.

[Install](#installation) · [Capabilities](#capabilities) · [Node Map](#node-map) · [DCC Handoff](#dcc-handoff) · [Known limitations](#known-limitations) · [Documentation](#documentation) · [Support](#support)

</div>

---

## Overview

- 32-bit float and EXR workflows for VFX and finishing, with lossless scene-linear round-trips.
- ACES, OCIO, log curves, LUTs, CDL, scopes, QC, and grade-transfer tools.
- VFX utilities for plate prep, masks, roto, depth, camera and optics, motion, multipass, real AOV ingestion, and relighting.
- Video and temporal workflow nodes for loading, routing, conditioning, sampling, and delivery.
- In-canvas studio dashboards — Project Manager, Workflow Library, and Assets — rendered over the ComfyUI graph, never in a separate browser tab.
- **Radiance Sampler** — a preset-driven sampler that hides irrelevant parameters and adapts to the selected model.
- A full-featured **Viewer** and a lightweight **Lite Viewer** with scopes, frame review, and keyboard shortcuts.
- HDR VAE decoders (Turbo and Full) and HDR LoRA tooling for scene-linear generation.
- Dynamic Gizmos — collapse any group of nodes into a single reusable custom node.
- Secure-by-default handoff to Nuke and DaVinci Resolve.

## In action

**The Radiance Viewer** — WebGPU-accelerated review at FP32/RGBA32F end to end: EXR channel and layer inspection, ACES 1.3 display pipeline, exposure/tone/colour controls with lift-gamma-gain wheels, scopes (histogram, waveform, vectorscope, parade), false colour, zebra, HDR heatmap, A/B compare with wipe, difference and blink, and a sequence timeline with per-frame thumbnails.

<div align="center">
<img src="viewer.png" width="920" alt="Radiance Viewer — inspector, colour transform, tone and colour wheels, scopes, sequence timeline">
</div>

**A generation-to-review graph** — Loader → Prompt → Resolution → Sampler → VAE Decode (HDR) in Direct HDR / RUDRA mode → Viewer. The decode stays scene-linear the whole way; the Viewer applies the display transform, so what you grade is what the file actually contains.

<div align="center">
<img src="basic_workflow.png" width="920" alt="Radiance graph: Loader, Prompt, Resolution, Sampler, HDR VAE Decode with RUDRA, and the Viewer node">
</div>

## Installation

### ComfyUI Manager / Comfy Registry

Search for **Radiance** in ComfyUI Manager, or install it from the Comfy Registry.

### Requirements

Install into the **same Python environment as ComfyUI** — Radiance relies on ComfyUI's existing PyTorch. Dependencies are pure Python (no CUDA toolkit or compiler required). Use the requirements file that matches your platform, shown below.

### Windows

```bat
cd ComfyUI\custom_nodes
git clone https://github.com/fxtdstudios/radiance-beta.git

cd radiance
pip install -r requirements_windows.txt
```

### Ubuntu / Linux

Install the system libraries OpenCV needs (OpenEXR and OCIO ship as self-contained wheels):

```bash
sudo apt update
sudo apt install -y git python3-venv build-essential libgl1 libglib2.0-0 ffmpeg
```

> On Ubuntu 22.04 the package is `libgl1-mesa-glx`; on 24.04+ it's `libgl1`.

Then, inside ComfyUI's environment:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fxtdstudios/radiance.git
cd radiance
pip install -r requirements_linux.txt
```

### WSL (Ubuntu on Windows)

WSL2 is Ubuntu with GPU passthrough, so follow the Ubuntu steps above (including `requirements_linux.txt`), plus:

- Install the NVIDIA driver on the **Windows host only** — never a Linux NVIDIA driver inside WSL. Verify with `nvidia-smi` inside WSL.
- No CUDA Toolkit needed — PyTorch's bundled CUDA runtime handles the GPU.
- Keep ComfyUI on the Linux filesystem (`~/ComfyUI`), not `/mnt/c/...`, for speed; open the UI from Windows at `http://localhost:8188`.

### macOS (Apple Silicon)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fxtdstudios/radiance.git
cd radiance
pip install -r requirements_mac_silicon.txt
```

### Verify

Start ComfyUI and look for `Radiance: successfully loaded 110 nodes` in the log.
A lower count means a node module failed to import — usually a missing optional
dependency; the Environment Guard table printed at startup shows which.

For a full machine-level check (every node executed on your GPU, VRAM peaks,
RUDRA decoder benchmarks and scores), run:

```bat
cd ComfyUI
python custom_nodes\radiance\tools\gpu_acceptance.py
```

It writes `gpu_acceptance_report.md` next to the script, and names any nodes
missing on your install.

### Models (RUDRA decoders)

The HDR VAE decoders (Turbo and Full) use trained **RUDRA** decoder weights, published on Hugging Face under Apache-2.0: [fxtdstudios/RUDRA](https://huggingface.co/fxtdstudios/RUDRA/tree/main).

Download the `.safetensors` files and place them in your ComfyUI models folder under a `radiance` subfolder — create it if it doesn't exist:

```
ComfyUI/models/radiance/
```

Radiance finds the checkpoints there automatically by filename, so keep the original names (for example `rudra_turbo_decoder_flux_ema.safetensors`). Download only the decoders for the models you use:

| Model | Turbo | Full |
| :--- | :---: | :---: |
| Flux.1 | `rudra_turbo_decoder_flux_ema` | `rudra_full_decoder_flux_ema` |
| Flux.2 | `rudra_turbo_decoder_flux2_ema` | — |
| Flux.2 Klein | `rudra_turbo_decoder_flux2-klein_ema` | — |
| SDXL | `rudra_turbo_decoder_sdxl_ema` | `rudra_full_decoder_sdxl_ema` |
| Qwen-Image | `rudra_turbo_decoder_qwen_ema` | — |
| Z-Image | `rudra_turbo_decoder_zimage_ema` | `rudra_full_decoder_zimage_ema` |
| Wan | `rudra_turbo_decoder_wan_ema` | `rudra_full_decoder_wan_ema` |
| LTX-Video | `rudra_turbo_decoder_ltx_ema` | `rudra_full_decoder_ltx-video_ema` |

To fetch everything at once with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download fxtdstudios/RUDRA --local-dir "ComfyUI/models/radiance"
```

### Example workflow

To get started quickly, drag [`workflows/start.json`](workflows/start.json) onto the ComfyUI canvas — a ready-made graph wiring the Radiance loader, Sampler Pro, HDR VAE decode, and viewers end to end.

## Capabilities

### Studio Dashboards

Radiance includes three production dashboards that open in-canvas — as an overlay on top of the ComfyUI graph rather than a new browser tab — from the Radiance Project Manager node. All three share a clean, dark interface.

| Dashboard | Purpose |
| :--- | :--- |
| **Project Manager** | Show, sequence, and shot view with a status pipeline (WIP, Review, Approved, Retake), version history, a click-through shot panel, project storage, and recent outputs — backed by a live view of your saved workflows. |
| **Workflow Library** | Browse, search, preview, and load saved workflows back into the canvas, organized by production bins. |
| **Assets** | A media manager that scans your ComfyUI input and output folders and classifies images, videos, and image sequences (auto-grouped by frame range). Create custom bins, filter by type, search, drag and drop to import, and inspect each asset in a detail panel. |

The Project Manager node keeps its launchers (open, save, and links) in a single compact panel on the node.

### Smart Interface

- **Adaptive sampler.** The Radiance Sampler hides every parameter when no preset is selected, shows everything in Custom mode, and for a named preset shows only the parameters relevant to that model — so you only see the controls that matter.
- **Inline video preview on Read.** MP4/MOV/WebM play directly on the node with scrubbing; production codecs the browser cannot decode (ProRes, DNxHR, MXF) fall back to a first-frame poster. Paths outside ComfyUI's folders need `RADIANCE_READ_ROOTS` (see Notes & Tips).
- **Resolved-path readout on Write.** The node shows the exact path that will land on disk — output_path, filename, version and format combined by the same code that writes — plus a note line ("frame 1 only" when an IMG format receives a batch, sequence start frame, unique-suffix behaviour). `overwrite` defaults **off**: existing files get a unique suffix instead of being destroyed.
- **In-canvas overlays.** Dashboards open over the graph and close with Esc, the dimmed background, or the close button, with an option to open in a full tab.
- **Dynamic Gizmos.** Collapse any selection of nodes into a single styled custom node that you can save and reuse like any other node.
- **Smart Backdrops.** Group nodes get a clear, tinted-glass background keyed to the node category instead of a near-invisible panel.

### Viewers

- **Viewer** — a full review surface with waveform and vectorscope, channel isolation, A/B compare, focus peaking, frame stepping, and keyboard shortcuts (below).
- **Radiance Lite Viewer** — a lightweight inline viewer for quick frame inspection.

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

### HDR VAE Decoders

- **Turbo Decoder** — a lightweight, near-realtime decode to scene-linear for fast iteration.
- **Full Decoder** — a deep decoder for production-quality reconstruction.
- Both are available through the Radiance HDR VAE Decode node, which also reports the decode settings it used.
- Decoder weights come from the [RUDRA models](#models-rudra-decoders) — see installation for the download and folder location.

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
└─ Pipeline
```

Radiance provides **110 nodes** (plus any Gizmos you create). Some nodes depend on optional packages and your ComfyUI environment.

Node names follow standard compositing vocabulary under the **Radiance** menu — `Grade`, `CDL`, `OCIO ColorSpace`, `Roto`, `Defocus`, `Viewer`, `Read`/`Write` — so they read the way they do in Nuke or Flame. AI and generation nodes keep a `Radiance` prefix (`Radiance Sampler`, `Radiance VAE Decode`) to mark the diffusion layer. You can still find any node by typing "radiance" in the search.

| Group | Examples |
| :--- | :--- |
| Core | Project Manager / Workspace, Resolution, workspace utilities |
| Load & Save | Read, Write (EXR alpha and mask), image and mask loading, EXR multipart and sequence export, Digital Cinema (DPX) read and write |
| Generate | Radiance Loader, Radiance Sampler, VAE Decode (HDR), prompt tools, LoRA stack, HDR LoRA, regional prompts |
| Color | Grade, Grade Match, CDL, LUT Apply / Blend, Curves, Hue Curves, White Balance, Color Space Convert, QC, Policy Guard |
| HDR | ACES 2.0, OCIO, HDR VAE encode/decode, tone mapping, HDR synthesis, relight, QC |
| VFX | Plate prep, masks, roto, depth, optics, motion, multipass, AOV reader (real EXR layers), relight |
| Video | Video loader, prompt builder, sampler, text-to-video, image-to-video, routing, batch decode, export |
| Upscale | Image and video upscale (HDR and color aware), tiling, face restoration |
| Review | Viewer, Lite Viewer, scopes, focus peaking, contact sheets, flipbook, preview server |
| Pipeline | Project Manager, Send to Nuke, DaVinci Resolve handoff |

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

## Notes & Tips

- **Estimated VFX passes.** The Multipass Master extractor derives passes (albedo, roughness, ambient occlusion, segmentation ID, and more) from a single image — handy for 2D and generated footage, but not a substitute for true render passes. For ground-truth passes, feed a multilayer EXR through the Multipass AOV Reader. The segmentation output is a clustered matte, not a Cryptomatte.
- **Super-resolution and color.** Upscale backends work in display-referred space. For scene-linear input, use the upscaler's HDR and color-encoding options to preserve your values.
- **Previews from a NAS or server path.** The Read node opens any absolute path (local, mapped drive, UNC), but its inline preview/info widgets are served over unauthenticated HTTP routes restricted to ComfyUI's own folders. To preview media elsewhere, allow those roots explicitly (`;`-separated on Windows) and restart ComfyUI:

  ```bat
  setx RADIANCE_READ_ROOTS "Z:\renders;\\server\share\plates"
  ```

- **Upgrading from ≤ 3.2.0: re-check graded masters.** `RadianceColorSpaceConvert` previously performed no conversion at all for 10 of its 16 spaces (all camera-log and ACES working spaces) whenever no OCIO config was loaded — the default install. Anything that passed through those conversions was graded on unconverted pixels. Details in the [changelog](CHANGELOG.md).

## Known limitations

Kept here rather than in a tracker, because a control that quietly does nothing
is worse than one that says so. Full detail in the [changelog](CHANGELOG.md).

- **The ACES 2.0 tone scale is not the Daniele Evo curve.** It is a log-space
  contrast of 1.55 with a tanh shoulder, so 18% grey sits about 0.84 stop above
  the ACES 2.0 reference on SDR, and HLG diffuse white lands at signal 0.915
  rather than 0.75. The normalisation defects around it were fixed in 3.2.0;
  the curve itself has not been replaced yet. Grade by eye against a reference,
  not by trusting the label.
- **`blend_mode = "laplacian_pyramid"`** falls back to the Gaussian feather.
  `"linear"` and `"gaussian_feather"` are genuinely different.
- **`chromatic_adaptation` has no effect.** The white-point adaptation is baked
  into the precomputed conversion matrices.
- **Optical flow is single-scale Lucas–Kanade**, not DIS. It recovers roughly
  1% of a 5-pixel displacement, so mask propagation is effectively static above
  about 2 pixels of motion.
- **Scene-cut detection normalises by the batch maximum**, so the threshold has
  no absolute meaning and cut-free footage will still report cuts.

## Status & to-do

Where the project actually stands, so nothing is carried in someone's head.
Ticked items are done and verified; unticked ones are the backlog.
Detail for anything here is in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) and the
[changelog](CHANGELOG.md).

### Blocking a release

- [ ] **Nothing has ever run in live ComfyUI on a GPU.** Every number in the
      audits is CPU and headless. Until a real graph renders a real frame, the
      verdict stays *ready with conditions*.
- [ ] **Viewer JavaScript is unverified.** `radiance_viewer.js` and the WebGL /
      WebGPU renderers have no automated coverage at all — the suite stops at
      the Python boundary.
- [ ] **Commit the working tree.** A stale `.git/index.lock` blocks every write;
      delete it, then commit. Several sessions of fixes are currently
      unprotected.
- [ ] **Reconcile with upstream.** `beta/main` is ~36 commits ahead and none of
      the bugs fixed here are fixed there. Commit locally first, then rebase;
      only `nodes/hdr/uplift_universal.py` and one `dcc.py` hunk conflict.

### Correctness backlog

- [ ] **ACES 2.0 tone scale is an approximation**, not the Daniele Evo curve —
      18% grey sits ~0.84 stop off reference on SDR. Grade against a reference,
      not the label.
- [ ] **RUDRA video decoders were trained on stills.** `wan` / `ltx-video` /
      `hunyuanvideo` checkpoints never saw multi-frame latents; real video falls
      back to math expansion. Needs retraining, not patching.
- [ ] **`rudra_full_decoder_ltx-video_ema.safetensors` is truncated at source**
      (23.0 MB against a declared ~36.0 MB). The loader detects it and degrades;
      the file still needs re-exporting.
- [ ] Scene-cut detection normalises by the batch maximum, so its threshold has
      no absolute meaning.
- [ ] Optical flow is single-scale Lucas–Kanade; mask propagation is effectively
      static above ~2 px of motion.
- [ ] Tiled VAE pastes tiles without blending.
- [ ] Tier-3 upscale ignores its `scale` input.
- [ ] Model auto-download happens without asking first.

### Structural debt

- [ ] **Retire the legacy `nodes_*.py` layer.** ~39 node keys are defined twice;
      this is the root cause of the aiohttp double-registration guards.
- [ ] **Replace heuristic menu classification** with an explicit per-node section
      declaration. `nodes/branding.py` keyword-matches and misfiles edge cases;
      `SECTION_OVERRIDES` is the current mitigation, not a fix.
- [ ] **Split the monoliths** — `nodes/monitor/viewer.py`, `nodes_io.py`,
      `hdr/vae.py`.
- [ ] **`delivery/handler.py` imports node classes**, inverting the library →
      UI layering and making the delivery path hard to test in isolation.
- [ ] Clarify the `Grade` / `Grade Apply` / `Apply Grade Info` naming overlap.
- [ ] Decide whether chained Energy Mask nodes should stack. Today the first one
      wins and the second is silently ignored.

### Done and verified

- [x] EXR 32-bit round-trip is bit-exact, including negatives and values > 1.
- [x] All 16 colour spaces hit published 18%-grey values (a P0 fix — 10 of them
      were silently identity without an OCIO config).
- [x] Video frame counts are exact, 1–100 frames, H.264 / H.265-10bit / ProRes.
- [x] Sequence reading is correct by frame number for `####`, `%04d` and ranges.
- [x] Memory is flat across 150 × 1080p runs.
- [x] Security: `weights_only` loads, sha256-pinned downloads, no `shell=True`.
- [x] Energy-Prioritized Sampling is reachable from a graph (#40) and no longer
      crashes on video latents.
- [x] Nodes no longer write into the ComfyUI install directory.
- [x] Suite: 2231 passed / 0 failed / 60 skipped.

## Documentation

Full documentation is available at [www.fxtdstudios.com](https://www.fxtdstudios.com) — setup, core concepts, workflow recipes, a complete node reference, and troubleshooting.

Every node also carries its own description and per-input tooltips, which ComfyUI shows on hover, so the parameter reference travels with the package.

## Support

- Issues: [GitHub Issues](https://github.com/fxtdstudios/radiance/issues)
- Documentation: [www.fxtdstudios.com](https://www.fxtdstudios.com)
- Studio: [www.fxtdstudios.com](https://www.fxtdstudios.com)

## License

Radiance is released under the [GPL-3.0 license](LICENSE).
