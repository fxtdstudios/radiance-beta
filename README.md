<div align="center">
<img src="r_icon.png" width="76" alt="Radiance mark"><br>
<img src="RADIANCE.png" width="640" alt="Radiance">

**Professional VFX, HDR color science, review, and DCC handoff for ComfyUI.**

[![Version](https://img.shields.io/badge/version-3.5.0-c8a96e?style=for-the-badge)](https://github.com/fxtdstudios/radiance)
[![License](https://img.shields.io/badge/license-GPL--3.0-green?style=for-the-badge)](LICENSE)
[![Nodes](https://img.shields.io/badge/nodes-156-c8a96e?style=for-the-badge)](#node-map)
[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-Radiance-orange?style=for-the-badge)](https://registry.comfy.org/nodes/radiance)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-RUDRA_models-ffd21e?style=for-the-badge)](https://huggingface.co/fxtdstudios/RUDRA)

Radiance is a production-grade node pack for ComfyUI built around 32-bit float and HDR/ACES image pipelines. It brings VFX plate prep, color management, review tooling, in-canvas studio dashboards, and Nuke / DaVinci Resolve handoff into one coherent toolkit, so you can take a shot from generation through finishing without leaving the graph.

Artists get 32-bit, HDR, and ACES image tools, professional viewers, and VFX nodes. Supervisors and coordinators get project, shot, asset, and workflow management built directly into the canvas.

[Install](#installation) · [What it does](#what-it-does) · [Node map](#node-map) · [DCC handoff](#dcc-handoff) · [Known limitations](#known-limitations) · [Notes](#notes--tips) · [Status](#status) · [Docs](#documentation)

</div>

---

## Overview

- 32-bit float and EXR workflows for VFX and finishing, with lossless scene-linear round-trips.
- ACES, OCIO, log curves, LUTs, CDL, scopes, QC, and grade-transfer tools.
- VFX utilities for plate prep, masks, roto, depth, camera and optics, motion, multipass, real AOV ingestion, and relighting.
- Video and temporal workflow nodes for loading, routing, conditioning, sampling, and delivery.
- In-canvas studio dashboards (Project Manager, Workflow Library, Assets) rendered over the ComfyUI graph, never in a separate browser tab.
- **Radiance Sampler**, a preset-driven sampler that hides irrelevant parameters and adapts to the selected model.
- A full-featured **Viewer** and a lightweight **Lite Viewer** with scopes, frame review, and keyboard shortcuts.
- HDR VAE decoders (Turbo and Full) and HDR LoRA tooling for scene-linear generation.
- Dynamic Gizmos: collapse any group of nodes into a single reusable custom node.
- Secure-by-default handoff to Nuke and DaVinci Resolve.

## In action

**The Radiance Viewer**: WebGPU-accelerated review at FP32/RGBA32F end to end: EXR channel and layer inspection, OpenColorIO 2.5 (load a show's config and the Display and View menus come from it, applied through OCIO's own GPU path), with a built-in ACES 1.3 pipeline as the fallback, exposure/tone/colour controls with lift-gamma-gain wheels, scopes (histogram, waveform, vectorscope, parade) with selectable scales (10-bit and 12-bit code value, percent, mV, and nits for ST.2084 and HLG), data or video levels, and a switch for whether they measure before or after the viewer colour transforms, a pixel probe with per-region min/max/mean/median and NaN/Inf reporting, false colour, zebra, a nit-accurate HDR heatmap anchored to BT.2408 reference white, safe areas labelled with the standard they come from (SMPTE ST 2046-1 / EBU R 95), aspect-ratio mattes, nearest-neighbour magnification on N, timecode, A/B compare with wipe, difference and blink, and a sequence timeline with per-frame thumbnails.

<div align="center">
<img src="viewer.png" width="920" alt="Radiance Viewer: inspector, colour transform, tone and colour wheels, scopes, sequence timeline">
</div>

**A generation-to-review graph**: Loader → Prompt → Resolution → Sampler → VAE Decode (HDR) in Direct HDR mode → Viewer. The decode stays scene-linear the whole way; the Viewer applies the display transform, so what you grade is what the file actually contains.

<div align="center">
<img src="basic_workflow.png" width="920" alt="Radiance graph: Loader, Prompt, Resolution, Sampler, HDR VAE Decode, and the Viewer node">
</div>

## Installation

### ComfyUI Manager / Comfy Registry

Search for **Radiance** in ComfyUI Manager, or install it from the Comfy Registry.

### Requirements

Install into the **same Python environment as ComfyUI**. Radiance relies on ComfyUI's existing PyTorch. Dependencies are pure Python (no CUDA toolkit or compiler required). Use the requirements file that matches your platform, shown below.

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

- Install the NVIDIA driver on the **Windows host only**, never a Linux NVIDIA driver inside WSL. Verify with `nvidia-smi` inside WSL.
- No CUDA Toolkit needed: PyTorch's bundled CUDA runtime handles the GPU.
- Keep ComfyUI on the Linux filesystem (`~/ComfyUI`), not `/mnt/c/...`, for speed; open the UI from Windows at `http://localhost:8188`.

### macOS (Apple Silicon)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fxtdstudios/radiance.git
cd radiance
pip install -r requirements_mac_silicon.txt
```

### Verify

Start ComfyUI and look for `Radiance: successfully loaded 156 nodes (v3.5.0)` in the log.
A lower count means a node module failed to import, usually a missing optional
dependency; the Environment Guard table printed at startup shows which.

For a full machine-level check (every node executed on your GPU, VRAM peaks,
and the RUDRA SDR → HDR pixel model benchmarked and scored), run:

```bat
cd ComfyUI
python custom_nodes\radiance\tools\gpu_acceptance.py
```

It writes `gpu_acceptance_report.md` next to the script, and names any nodes
missing on your install.

### Models (RUDRA SDR → HDR)

Learned SDR → HDR recovery uses the **RUDRA** pixel model: a compact network
that takes decoded 8-bit SDR pixels and returns scene-linear HDR, recovering
clipped highlights and crushed shadows. It works on any image or frame batch
and needs no VAE, so it applies equally to generated frames and to footage.
The weights are published on Hugging Face,
[fxtdstudios/RUDRA](https://huggingface.co/fxtdstudios/RUDRA).

**The RUDRA weights are licensed for non-commercial use only**, unlike
Radiance's code (GPL-3.0) and RUDRA's code (Apache 2.0). One training source
(HdM-HDR-2014 / HdM-HFR-2017) is free for academic use only, and FXTD Studios
cannot waive that term. Research, evaluation, teaching and personal projects
are fine; commercial production, client deliverables and paid services need a
commercial licence from [FXTD Studios](https://fxtdstudios.com). The full
terms are in the weights'
[licence](https://huggingface.co/fxtdstudios/RUDRA/blob/main/LICENSE). Expand
mode uses no weights and has no such restriction.

**It downloads automatically.** The first time SDR → HDR Universal or
SDR → HDR Recover needs the model and none is installed, Radiance fetches
RUDRA's shipped model, `sdr2hdr_shadow_v1` (about 5 MB), into your ComfyUI
models folder:

```
ComfyUI/models/radiance/sdr2hdr_shadow_v1.safetensors
```

The download is pinned to a fixed Hugging Face commit and checked by size and
SHA-256 before it is used; a file that does not match is discarded. It happens
once, and the console shows `[Radiance] Installed ...` when it is done. Leave
the node's `pixel_checkpoint` field empty to use it.

To install it by hand instead (offline or air-gapped machines), download
[`sdr2hdr_shadow_v1.safetensors`](https://huggingface.co/fxtdstudios/RUDRA/resolve/main/sdr2hdr/sdr2hdr_shadow_v1.safetensors)
and put it at the path above, creating the `radiance` folder if needed. To turn
automatic downloads off, set any of these in the ComfyUI launch environment:

| Variable | Effect |
| :--- | :--- |
| `RADIANCE_ALLOW_DOWNLOADS=0` | never download any Radiance model |
| `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1` | treat the machine as offline |

Radiance registers `models/radiance` with ComfyUI, so a `radiance:` entry in
`extra_model_paths.yaml` works too, and `RADIANCE_SDR2HDR_PIXEL` can point at
a file anywhere. Every RUDRA image checkpoint loads, as Hugging Face publishes
it (`.safetensors`, flat in the folder or under the `sdr2hdr/` subfolder
`hf download` creates) or as the training scripts write it (`.pt`); each file
carries its own architecture, so the shadow-gated and ungated models both
build correctly. When several are present `sdr2hdr_shadow_v1` wins, then the
other shadow seeds, then `sdr2hdr_image_v5`, then older `.pt` files.
`sdr2hdr_temporal_v1` is not an image model and is refused with a message
saying so.

| File | Used by | Link |
| :--- | :--- | :--- |
| `sdr2hdr_shadow_v1.safetensors` | SDR → HDR Universal (Recover / Hybrid) and SDR → HDR Recover, stills and frame batches | [download](https://huggingface.co/fxtdstudios/RUDRA/resolve/main/sdr2hdr/sdr2hdr_shadow_v1.safetensors) (auto) |
| `sdr2hdr_shadow_s2` / `_s3`, `sdr2hdr_image_v5` / `_v6` | alternatives: other seeds of the shipped recipe, and the backbone without the shadow gate | [RUDRA/sdr2hdr](https://huggingface.co/fxtdstudios/RUDRA/tree/main/sdr2hdr) |
| `temporal_rudra_residual_ema.safetensors` | the motion-aligned temporal model for ordered video (optional) | not yet published |

If the model can't be found or downloaded, Universal falls back to
deterministic expansion and says so in its `report` output; Recover raises
rather than silently expanding.

**What the model recovers, and what it doesn't.** In clipped highlights the
network supplies brightness; the colour stays the source's. A clipped channel
carries no information about its own value, and the released checkpoints were
trained on renders that almost never clipped, so their per-channel guesses in
blown areas are not reliable colour: letting them through drew false-colour
rings round a clipped sun and cast white areas red. The learned lift fades in
as the source goes to white (two or more channels near clip), is never darker
than the deterministic expansion, and no channel of it exceeds `peak_nits`,
so an HDR10 encode cannot clip it per channel. The model runs on the whole
frame when it fits in memory; `pixel_tile_size` only applies when it does
not, because the network normalises over its input and tiles shift the
result.

The other model downloads Radiance can make (Real-ESRGAN, HAT-L, SwinIR,
Depth Anything V2, DSINE; 67 MB to 2.4 GB) still ask first: set
`RADIANCE_ALLOW_DOWNLOADS=1` to allow them.

The latent-space RUDRA decoders that earlier releases loaded inside HDR VAE
Decode (`rudra_turbo_decoder_*` / `rudra_full_decoder_*`) were retired in
3.5.0; see the changelog. The files can be deleted from `models/radiance`.

### Example workflows

Drag a workflow onto the ComfyUI canvas to load it. Both ship in the
package's `workflows` folder.

| Workflow | What it does | Models it needs |
| :--- | :--- | :--- |
| [`workflows/start.json`](workflows/start.json) | The quick start: Radiance Loader, Cinematic Prompt, Resolution, Sampler Pro, HDR VAE Decode and the viewers wired end to end for FLUX.1-dev. | [`flux1-dev.safetensors`](https://huggingface.co/black-forest-labs/FLUX.1-dev) and `ae.safetensors` from the same page (accept the licence first) in `models/diffusion_models` and `models/vae`; [`clip_l.safetensors`](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors) and [`t5xxl_fp16.safetensors`](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors) in `models/text_encoders` |
| [`workflows/official/wan22_t2v_hdr_universal.json`](workflows/official/wan22_t2v_hdr_universal.json) | Wan 2.2 text-to-video through SDR → HDR Universal to an HDR master with HDR Encode and Write. | [`wan2.2_t2v_high_noise_14B_fp8_scaled`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors) and [`wan2.2_t2v_low_noise_14B_fp8_scaled`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors) in `models/diffusion_models`; [`umt5_xxl_fp8_e4m3fn_scaled`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) in `models/text_encoders`; [`wan_2.1_vae`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) in `models/vae`. The RUDRA model downloads itself. |

## What it does

### Reading and writing

Read opens images, EXRs, video and numbered sequences from any absolute path, including UNC and mapped drives. It works out what a path is from the extension and the pattern, so `/renders/shot.%04d.exr`, `/frames/####.png` and a bare directory all resolve to a sequence. Open one frame of a sequence and it offers you the whole range.

Video decodes through a single raw ffmpeg pipe at the source's own bit depth. 10, 12 and 16-bit survive. ProRes 4444 alpha comes out on the mask output, frame ranges are exact rather than approximate, and the container's colour tags are read instead of assumed. The `info` output is a JSON wire carrying resolution, frame count and range, bit depth, codec, EXR layers and windows, colour tags and timecode. Nuke's metadata tab, as something you can plug into.

Write covers PNG 8 and 16-bit, JPEG, TIFF 16 and 32-float, DPX, WebP, Radiance HDR, EXR half and float, H.264, H.265 10-bit, ProRes 422 and 4444, DNxHR, and numbered sequences of any of the still formats. It shows you the path that will actually land on disk before you run it, assembled by the same code that does the writing. `overwrite` is off by default; an existing file gets a unique suffix rather than being replaced.

### Colour

Load a show's OCIO config and the Display and View menus come from it. Nothing is re-implemented: the transform is OCIO's own, applied through its generated GPU shader in the Viewer. With no config loaded, the built-in ACES 1.3 pipeline runs as before. OCIO is a capability here, not a dependency.

The grading nodes are the ones you would expect from a compositing package. Grade, Grade Match, CDL, Curves, Hue Curves, White Balance, LUT Apply and Blend, Colour Space Convert, and a QC pass. Sixteen colour spaces, including the camera logs (ARRI LogC3 and LogC4, Sony S-Log3, Panasonic V-Log, Canon Log 3, RED Log3G10, DaVinci Intermediate), plus PQ, HLG, ACEScg and ACEScct.

Both ACES 2.0 tone scales are implemented against the published Output Transform table.

### HDR

Log encoding happens before the VAE, not after it, and the Compress Log profiles are clamp-free from decode through to the file. Highlights above 1.0 reach disk. That is the claim the package is built on and there is a test that writes negatives and values up to 64.0 through EXR and TIFF and requires them back exactly.

The HDR VAE Decode node decodes through the model's own VAE, in a sampler-safe mode or a Direct HDR mode that inverts the log encoding to scene-linear and keeps everything above 1.0. Learned recovery of clipped highlights and crushed shadows is the job of SDR → HDR Universal and SDR → HDR Recover, which run the RUDRA pixel model on any image or frame batch, no VAE required; the weights download on first use (see Models above).

There is also HDR LoRA loading and application, a LoRA stack with per-LoRA model and CLIP strengths, tone mapping, HDR synthesis, and relighting.

### The Viewer

FP32 and RGBA32F end to end, on WebGL2.

Scopes: histogram, waveform, vectorscope and parade. You pick the scale (10 or 12-bit code value, percent, millivolts, or nits for ST.2084 and HLG) and data or video levels, and whether the scopes measure before or after the viewer's colour transforms. Every graticule line carries its number.

The pixel probe samples a cursor, a region or the whole frame, and reports RGBA, luminance, EV, cd/m², HSV and hex, with min, max, mean and median per channel. NaN, Inf and negative counts are excluded from the statistics and reported separately, because a mean that quietly includes a NaN is worse than no mean.

Then the things you reach for while looking: false colour, zebra, a nit-accurate HDR heatmap anchored to BT.2408 reference white, safe areas labelled with the standard they come from, aspect-ratio mattes, nearest-neighbour magnification, timecode, A/B compare with wipe, difference and blink, EXR channel and layer inspection, focus peaking, and a sequence timeline with per-frame thumbnails.

A Lite Viewer exists for when you want a frame on the node and nothing else.

| Key | Action |
| :--- | :--- |
| Space | Play / pause |
| Left / Right | Previous / next frame |
| F | Fit to view |
| 1 | 1:1 pixels |
| C / R / G / B / L | Colour, red, green, blue, luma |
| W | Waveform |
| V | Vectorscope |
| A | Cycle A/B compare |
| N | Nearest-neighbour / linear |

### VFX

Plate prep, masks, roto, depth, optics, motion, and multipass. Motion estimation is DIS optical flow, which stays dense out to about 20 px of movement; the older Lucas–Kanade solver is still selectable. The Multipass Master extractor derives passes from a single image, which is useful for generated footage and is not a render pass; when you have real AOVs, the Multipass AOV Reader takes a multilayer EXR. Relighting works off either.

### Video

Loader, prompt builder, sampler, text-to-video, image-to-video, routing, batch decode and export. Upscaling is available for stills and for video, with HDR and colour-encoding options so scene-linear input survives a backend that works in display space.

### Studio dashboards

Three of them, opened from the Project Manager node. They render over the ComfyUI graph rather than in another tab, and close with Esc.

| Dashboard | What it holds |
| :--- | :--- |
| **Project Manager** | Shows, sequences and shots, with a WIP / Review / Approved / Retake pipeline, version history, a shot panel, project storage and recent outputs, read from your saved workflows. |
| **Workflow Library** | Search, preview and load saved workflows back into the canvas, organised into bins. |
| **Assets** | Scans the ComfyUI input and output folders, sorts images, video and sequences (grouped by frame range), and gives you bins, filters, search, drag-and-drop import and a detail panel. |

### Things that make the graph easier to live with

The Sampler shows nothing until you pick a preset, everything in Custom, and only the parameters that matter to the model you named. Read plays MP4, MOV and WebM on the node with scrubbing, and falls back to a first-frame poster for codecs the browser cannot decode. Group nodes get a tinted backdrop keyed to category instead of a near-invisible panel. And any selection of nodes can be collapsed into a Gizmo: one node, saved and reloaded like any other.

## Node map

Everything lives under one menu.

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

**156 nodes**, plus whatever Gizmos you build. A few depend on optional packages.

Compositing nodes use compositing names (`Grade`, `CDL`, `OCIO ColorSpace`, `Roto`, `Defocus`, `Viewer`, `Read`, `Write`), so they read the way they do in Nuke or Flame. The diffusion layer keeps a `Radiance` prefix, so `Radiance Sampler` and `Radiance VAE Decode` are obviously the AI ones. Typing "radiance" in the search still finds everything.

| Group | What's in it |
| :--- | :--- |
| Core | Project Manager, Workspace, Resolution, workspace utilities |
| Load & Save | Read, Write, EXR alpha and mask, EXR multipart, sequence export, DPX read and write |
| Generate | Loader, Sampler, VAE Decode (HDR), prompt tools, LoRA stack, HDR LoRA, regional prompts |
| Color | Grade, Grade Match, CDL, LUT Apply / Blend, Curves, Hue Curves, White Balance, Colour Space Convert, QC, Policy Guard |
| HDR | ACES 2.0, OCIO, HDR VAE encode and decode, tone mapping, HDR synthesis, relight, QC |
| VFX | Plate prep, masks, roto, depth, optics, motion, multipass, AOV reader, relight |
| Video | Loader, prompt builder, sampler, text-to-video, image-to-video, routing, batch decode, export |
| Upscale | Image and video upscale, tiling, face restoration |
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
- **Optical flow works to about 20 px of motion, not beyond.** The solver is
  DIS (dense inverse search), and it holds a dense field, 99% or more of the
  frame within half a pixel, out to 20 px on the test plates. Past roughly 28
  px it fails, as does every other solver here, because the correspondence
  becomes ambiguous rather than merely hard.

  Pyramidal Lucas–Kanade is still selectable and is the fallback if OpenCV
  cannot be imported. It is bounded by its 15×15 integration window: the
  fraction of the field landing within half a pixel falls from 100% at 3 px to
  84% at 8, 71% at 12, 58% at 16 and 29% at 20, and on aperiodic detail it is
  already at 9% by 12 px. What degrades is the *field*, not the median, which
  is the worst way for a flow solver to fail: the number you would check looks
  right while the mask you propagate tears.

  Deepening the LK pyramid does not help, and that was measured rather than
  assumed: the coarsest level has to stay larger than the window, and going
  from four levels to five on a 256×512 plate made every displacement worse (at
  8 px, 96% of the field within half a pixel became 32%). Shrinking the window
  with the level to buy depth was tried too, and measured worse for the same
  reason.

- **No SAM runtime.** `SAM Loader` and `SAM Mask Generator` are hidden and
  raise when executed; they never ran a segmentation model. Use a SAM2 node
  pack and feed its MASK into Radiance's matting, roto and propagation nodes.
- **Upscale `confidence` is a tile weight.** It is 1 at tile centres and lower
  toward tile edges; no backend reports per-pixel hallucination.
- **CFG schedules are static.** The video pipelines use the first value of
  `cfg_schedule_json` as the CFG; it does not vary per step.

## Notes & Tips

- **Estimated VFX passes.** The Multipass Master extractor derives passes (albedo, roughness, ambient occlusion, segmentation ID, and more) from a single image, handy for 2D and generated footage, but not a substitute for true render passes. For ground-truth passes, feed a multilayer EXR through the Multipass AOV Reader. The segmentation output is a clustered matte, not a Cryptomatte.
- **Super-resolution and color.** Upscale backends work in display-referred space. For scene-linear input, use the upscaler's HDR and color-encoding options to preserve your values.
- **Previews from a NAS or server path.** The Read node opens any absolute path (local, mapped drive, UNC), but its inline preview/info widgets are served over unauthenticated HTTP routes restricted to ComfyUI's own folders. To preview media elsewhere, allow those roots explicitly (`;`-separated on Windows) and restart ComfyUI:

  ```bat
  setx RADIANCE_READ_ROOTS "Z:\renders;\\server\share\plates"
  ```

- **Upgrading from ≤ 3.2.0: re-check graded masters.** `RadianceColorSpaceConvert` previously performed no conversion at all for 10 of its 16 spaces (all camera-log and ACES working spaces) whenever no OCIO config was loaded, which is the default install. Anything that passed through those conversions was graded on unconverted pixels. Details in the [changelog](CHANGELOG.md).

## Status

Radiance is **ready with conditions**. Everything below is measured rather than
asserted; the full history is in the [changelog](CHANGELOG.md), and per-defect
detail in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### Verified

Standing properties of the shipped package. Pinned by a test unless the row
says otherwise. An audit number that no test holds is a number that can
quietly stop being true.

A note on what changed here in 3.4.0. Until this release the job that installs
real torch, OpenEXR and OpenColorIO was gated on `github.repository ==
'fxtdstudios/radiance'`, and development happens on `radiance-beta`, so the lane
that executes every row below had never run against this code. The numbers in
this table used to be what the suite *would* report. They are now what it does
report, from a lane that runs. Where the two differed, the row was rewritten and
the gap is recorded under Open rather than quietly corrected.

| Area | What is verified |
| :-- | :-- |
| **EXR** | 32-bit float round-trips bit-exactly through EXR and TIFF, negatives and over-range highlights included, the clamp-free HDR claim, as a write-and-read rather than an assertion. 16-bit half holds to 1e-3. |
| **Colour** | All 16 colour spaces are pinned to a published 18%-grey value: 15 transfer curves plus the linear identity, each held to 1e-4 and cross-checked against colour-science's independent implementation of the same specification. Both ACES 2.0 tone scales place 18% scene grey where the ACES Output Transform publishes it: 10.000 / 13.193 / 14.512 / 15.747 / 16.824 nits for peaks of 100 / 500 / 1000 / 2000 / 4000, held to 1e-6, with intermediate peaks checked against the geometric-mean form of the log-log rule. The shipped table is compared against a transcription of the published one rather than against the interpolator that reads it, which is what made the old test unfalsifiable. HLG keeps its BT.2408 anchor. The OCIO bake is checked against OCIO's own CPU processor, exactly rather than approximately. |
| **Transfer** | PQ encodes absolute luminance against ST.2084's fixed 10 000 cd/m² ceiling, pinned at five mastering peaks and against an absolute-luminance ladder, and cross-checked against the package's other PQ encoder. Rec.709 and Rec.2020 are the real BT.709-6 and BT.2020-2 OETFs with the BT.2020 primaries matrix, pinned by value, and every offered output colour space is asserted to change the data, so a missing conversion cannot pass as a successful write. |
| **Video** | Frame counts are exact from 1 to 100 frames across H.264, H.265 10-bit and ProRes 422 HQ, by encoding and reading back real media. The default suite covers 17 lengths per codec, chosen around the 1/2/3 degenerate cases and both sides of every GOP boundary, and asserts the identity and order of each frame as well as the count, at `core.video` and again at the Read node. The exhaustive 1-to-100 sweep runs under `-m slow`. Sequences read correctly by frame number for `####`, `%04d` and explicit ranges. |
| **Duration** | The write path, the HDR VAE encode and decode, the viewer and the sampler's noise generation all hold a working window rather than the clip. Measured, not asserted: writing 32 frames and writing 512 frames peak within 0.1 MB of each other, and enabling a colour transform costs 1.6 MB rather than a second copy of the shot. The VAE's decode overhead is flat at 5.7 MB from 4 frames to 32 where it used to grow by a whole extra clip. Sequence length is bounded by disk. Generation is the exception and has its own control, see below. |
| **Memory** | Flat across 150 consecutive 1080p runs, an audit measurement rather than a standing test. |
| **Security** | `weights_only` loads, sha256-pinned downloads, no `shell=True`, and no model weight downloads without `RADIANCE_ALLOW_DOWNLOADS=1` through a gate every downloader shares. Nodes never write into the ComfyUI install directory. |
| **Catalog** | All 156 nodes declare their menu section explicitly; a test fails if a registered node is missing from the table. Withholding a node from the menu requires a named entry with a written reason a test reads and checks the length of. Separately, an AST walk of every file in the distribution finds every `NODE_CLASS_MAPPINGS` and asserts each class in it is registered as the class that ships, so a node stranded in a package the catalog does not load turns the suite red. That is how 26 finished nodes stayed out of the menu until 3.4.0. |
| **Isolation** | Every one of the eleven node groups imports with `aiohttp` and `server` blocked, proven in a subprocess rather than for one hand-listed module. The blocker uses `find_spec`; it previously used `find_module`, which Python 3.12 removed, so on the 3.12 leg of the matrix it silently blocked nothing and the test passed while measuring nothing. The harness now proves it is blocking before it reports anything. |
| **Layering** | `radiance/io/writer.py` and `radiance/io/reader.py` import nothing above them, checked by AST walk *and* by running them in a bare interpreter with no ComfyUI present. |
| **Suite** | 3755 Python tests and 260 JavaScript tests. On the full dependency lane: 3667 pass, 84 skip, 3 are `slow` and deselected by default; JS is 256 pass, 4 skip, 0 todo. Coverage is **59.6% of 28,018 statements** (56.0% counting branches, which is what the floor gates on). The old 53% was the figure the full lane would have produced had it run; CI's lightweight lane was really reporting 22% against a `--cov-fail-under=15` that overrode the project's own floor. There is one floor now, in `pyproject.toml`, enforced on the lane that can execute the code. The JS side includes a GPU lane that compiles the real shaders in both GLSL and WGSL and compares them against the CPU implementations they were generated from, and a browser lane that builds all fourteen Viewer panels and operates their controls. Verified from a checkout named `radiance-beta` as well as `radiance`. |

### Open

**Blocking a release**

- [ ] **One full GPU render in live ComfyUI.** Checked on 2026-09-24 in a real
      ComfyUI 0.32 with frontend 1.48 (CPU): all 156 nodes register, every
      Radiance node can be created, saved and reloaded with no frontend error,
      and `workflows/start.json` passes ComfyUI's own prompt validation. The
      Viewer and the pixel SDR → HDR model have run live on the RTX 4080. What
      is still owed is one real graph (sampler → HDR VAE Decode → Write)
      rendering a frame on the GPU.
- [ ] **The public repo is behind, and it is not a fast-forward.**
      `fxtdstudios/radiance` `main` (`16e885f`) is 5 commits the beta line
      never took. They were reviewed hunk by hunk on 2026-09-23: PR #18's
      two real fixes are ported (Write reports its files to ComfyUI history;
      no upper version caps in the platform requirements or `pyproject.toml`),
      the rest are already fixed here, patch files that no longer exist, or
      are features (movable controls panel, extra Save nodes) left for later.
      PR #20 edits a README section that no longer exists. What remains is
      the publishing decision: replace public `main` with this line.

**Needs a GPU to confirm**

- [ ] **Long-video windowing is unproven on a real model.** `temporal_window` on
      the sampler denoises a long clip in overlapping latent windows, blended at
      every step rather than after each window is finished, so peak VRAM follows
      the window size instead of the clip length. CPU tests pin the parts that
      are checkable without a model: the schedule covers every frame at full
      weight, the blend weights form a partition of unity, step accounting and
      seeding are stable across a rerun, and a single window reproduces the
      unwindowed result. What CPU cannot show is whether the output is
      temporally coherent on a real DiT. Default is 0, off, so nothing changes
      until it is switched on deliberately. Treat it as ready to test, not ready
      to deliver from, until a long clip has run on the 4080.

**Correctness**

- [x] **Legacy latent RUDRA decoders retired (3.5.0).** The video decoders
      trained on stills, the truncated `ltx-video` full decoder and the
      per-model checkpoint matrix are gone with them. Learned SDR → HDR is the
      pixel model, one checkpoint for every model family, plus the temporal
      residual model for video.
- [x] **The Viewer rendered black (3.5.0), two causes, both verified live on
      ComfyUI 0.32 / frontend 1.48.** (1) The WebGL renderer's `setMask` and
      its mask uniforms addressed a `this.mask` object that the v3.1 refactor
      had replaced with flat fields, so every `render()` threw before the draw
      call; hidden while WebGPU auto-upgraded, exposed when 3.4.0 made WebGL
      the default. (2) The Vue node frontend grew the node to the height of
      the sidebar and inspector content (1180x760 became 1480x2286), the
      canvas stretched with it, and `resize()` never refit, so the frame was
      centred below the visible area. The container now has `contain: size`
      and `resize()` refits an auto-fitted view. Pinned by browser tests that
      load a real frame through `onExecuted`, read the canvas back, and host
      the viewer in an auto-height parent.
- [x] **Read / Write colour management and precision (3.5.0).** Every
      encoding decodes and encodes transfer AND primaries to a selectable
      working space (Linear Rec.709, ACEScg, Linear Rec.2020, Linear P3-D65,
      ACES2065-1), through OpenColorIO's ACES studio config or any
      `ocio_colorspace` / `ocio_config`, with an analytic fallback held to
      OCIO in tests. Video uses the correct YUV matrix and is tagged; EXR and
      DPX carry their colour metadata; 16-bit grey, half-float TIFF and
      alpha read correctly. 65 write-and-read-back tests.
- [x] **Honest release pass (3.5.0).** Every control, option and output was
      checked against the code that reads it. Fixed: I2V strategies now write
      the conditioning keys ComfyUI's Wan models read; T2V/I2V latents follow
      the connected model instead of an LTX default; Video HDR Conditioner and
      Decode reach the model and convert gamut; upscale reports what actually
      ran and Face Restore `auto` restores; mask propagation follows motion;
      Bezier roto, motion-blur energy, Policy Guard (all frames, HDR peak),
      Diagnostics colorspace, Digital Cinema Read colorspace and fps, Audio
      Cut and Camera Sync errors, Regional `Replace`, NDI batches, joint
      bilateral chroma. SAM withheld; ViTMatte/RVM removed. Each has a test.
- [x] **HDR VAE Decode and SDR → HDR (3.5.0).** Auto no longer log-inverts a
      latent a sampler touched (HDR Encode now fingerprints its latent);
      hidden widgets no longer steer the decode; Direct HDR honours
      scene-referred targets, applies exposure after reconstruction and writes
      RHDR from the returned image. One HDR convention everywhere: linear
      1.0 = 203 nits (BT.2408), HLG the BT.2100 1000-nit transcode OCIO uses,
      camera log targets in their camera gamut, AP0 matrix corrected. Checked
      against OpenColorIO and the shipped pixel checkpoint.
- [x] **RUDRA pixel model: no false colour, no over-peak channels (3.5.0).**
      From a user report on a clipped Flux.2 sunset. Recovered highlights keep
      the source colour (no rings, no red cast, source-level chroma noise),
      no channel exceeds `peak_nits`, the whole frame runs untiled when it
      fits, and every published checkpoint loads, with the shipped
      `sdr2hdr_shadow_v1` as the default and the auto-download.
- [x] **Viewer and Lite Viewer, phase 1 (3.5.0).**
  - **Colour.** The node tags every frame: a ComfyUI IMAGE is shown exactly as ComfyUI shows it, and a linear source goes through OpenColorIO ACES 2.0. A normal image used to be read as linear and sRGB-encoded twice, which washed it out, and its input colour space was guessed from brightness.
  - **View menu.** Every entry is real (ACES 2.0 and 1.3 through OCIO, sRGB, Rec.709 BT.1886). The same view is baked into the PNG previews.
  - **Units.** Everything reads 203 nits for 1.0, and the DaVinci Intermediate curve matches the spec again.
  - **Compare and playback.** Compare works on WebGL and follows the playhead. Revisited frames no longer go black. Playback follows the source fps. Timecode is SMPTE, with drop-frame at 29.97 and 59.94.
  - **Export and transport.** The graded EXR is scene-linear and tagged with its primaries. Frames travel as half-float by default, and an unchanged viewer no longer re-runs everything downstream.
  - **Lite Viewer.** Readout, clip check and diff use float source values. 1:1 is exact on scaled displays. It has play/loop.
- [x] **Cinematic Encoder, step 1 (3.5.0).** Long SDXL prompts keep their camera and lighting (no `BREAK`, no 77-token cut); Wan, Flux.2, Z-Image, Lumina2, Qwen-Image, AuraFlow and every other T5 / LLM encoder get prose instead of SDXL tags; the subject is never rewritten; Flux, Flux.2 and MiniMax skip the unused negative encode.
- [x] **Viewer phase 2 (3.5.0).**
  - **Scopes.** Waveform, vectorscope and histogram measure the picture as displayed (graded, through the active view), not the ungraded texture or the 8-bit canvas. The vectorscope is BT.709 Cb/Cr with 75 % and 100 % targets and a skin line.
  - **Warnings.** False colour uses ARRI's bands on the Rec.709 signal. Clip and gamut warnings run before the display clamp, so they fire.
  - **Viewer-only exposure and gamma.** `f/` and `γ` in the viewer bar (and `-` / `=`, `0` to reset) change only what you see, never scopes, readout or export.
  - **Pixels.** The canvas is device-pixel sized; zoom above 1:1 is nearest-neighbour by default. Output is dithered, and a P3 monitor gets a Display P3 view and canvas.
  - **Keys and transport.** Keys go to the viewer under the pointer only. In/out (`I` / `O`), J/K/L shuttle, ping-pong and play-once, play every frame with a dropped-frame count.
- [x] **Sequential offload never engaged.** `setup_offload_mode("sequential")`
      called a `comfy.model_management.set_lowvram_mode` that ComfyUI never
      shipped, so it warned and did nothing on every run. It sets ComfyUI's
      `vram_state` now.

**Structural debt**

- [ ] **Split the remaining monoliths.** Largest first: `hdr/vae.py` (3527
      lines), `nodes/upscale/upscale.py` (3029), `image/upscale.py` (2741),
      `nodes/generate/sampler.py` (2111), `nodes/pipeline/workspace.py` (1913),
      `sampler_utils.py` (1897), `nodes/generate/prompt.py` (1830),
      `nodes/io/write.py` (1261), `nodes/monitor/viewer.py` (1233).

**Test coverage**

59.6% of 28,018 statements, measured on the full dependency lane. Every node has
structural coverage, though note what that does and does not mean: the smoke
tests check that the method named by `FUNCTION` exists, they do not call it.
Calling every registered node is `test_node_functional.py`'s job, and it now
runs in CI.

- [ ] **`image/upscale.py`, 1111 statements, 31%.** The one module that cannot
      be finished on CPU: what remains is `RadianceAIUpscale`, the SUPIR path,
      and the tiling code, all of which need model weights or a GPU. Newly
      registered in 3.4.0, so this is the first release in which its coverage
      counts for anything.

**Colour management**

- [x] **Sampler speed (3.5.0).** No second model load per stage, no silent
      cfg 1.0 → base CFG (the extra unconditional pass that doubled turbo
      runs), no cfg boost at 1.0 in Dynamic CFG, no gc / cache flush before
      sampling, no debug statistics with DEBUG off. Flux with an empty
      `clip_l` no longer falls back to Mochi's T5 encoder, and the Loader
      reports when the weights cannot stay in VRAM.
- [x] **OCIO is configured automatically (3.5.0).** OpenColorIO is a
      required dependency (installed by ComfyUI-Manager via `requirements.txt`,
      or by `install.py`). At startup Radiance uses `$OCIO` when you have one
      and otherwise OpenColorIO's built-in ACES 2.0 studio config (55
      colorspaces: every ACES space, the major camera logs, Rec.709 / Rec.2020
      / P3 / PQ / HLG), written to `ACES/studio-config.ocio`, exported as
      `$OCIO` for the process and made OCIO's current config, so every node
      resolves the same names. Nothing is downloaded (the old startup fetch
      from GitHub is gone). `RadianceColorSpaceConvert` now maps its names to
      that config (the old targets existed in no config, so OCIO never ran)
      and `RadianceHDROCIOTransform`'s defaults resolve. OpenCV's EXR codec is
      forced on (`OPENCV_IO_ENABLE_OPENEXR=1`) before anything imports cv2.

## Documentation

Full documentation is available at [www.fxtdstudios.com](https://www.fxtdstudios.com): setup, core concepts, workflow recipes, a complete node reference, and troubleshooting.

Every node also carries its own description and per-input tooltips, which ComfyUI shows on hover, so the parameter reference travels with the package.

## Support

- Issues: [GitHub Issues](https://github.com/fxtdstudios/radiance/issues)
- Documentation: [www.fxtdstudios.com](https://www.fxtdstudios.com)
- Studio: [www.fxtdstudios.com](https://www.fxtdstudios.com)

## License

Radiance is released under the [GPL-3.0 license](LICENSE). The RUDRA model weights that
SDR → HDR Universal and Recover download are a separate work under their own
non-commercial licence; see [Models](#models-rudra-sdr--hdr).
