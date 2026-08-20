<div align="center">
<img src="r_icon.png" width="76" alt="Radiance mark"><br>
<img src="RADIANCE.png" width="640" alt="Radiance">

**Professional VFX, HDR color science, review, and DCC handoff for ComfyUI.**

[![Version](https://img.shields.io/badge/version-3.3.0-c8a96e?style=for-the-badge)](https://github.com/fxtdstudios/radiance)
[![License](https://img.shields.io/badge/license-GPL--3.0-green?style=for-the-badge)](LICENSE)
[![Nodes](https://img.shields.io/badge/nodes-131-c8a96e?style=for-the-badge)](#node-map)
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

**The Radiance Viewer** — WebGPU-accelerated review at FP32/RGBA32F end to end: EXR channel and layer inspection, OpenColorIO 2.5 — load a show's config and the Display and View menus come from it, applied through OCIO's own GPU path — with a built-in ACES 1.3 pipeline as the fallback, exposure/tone/colour controls with lift-gamma-gain wheels, scopes (histogram, waveform, vectorscope, parade) with selectable scales — 10-bit and 12-bit code value, percent, mV, and nits for ST.2084 and HLG — data or video levels, and a switch for whether they measure before or after the viewer colour transforms, a pixel probe with per-region min/max/mean/median and NaN/Inf reporting, false colour, zebra, a nit-accurate HDR heatmap anchored to BT.2408 reference white, safe areas labelled with the standard they come from (SMPTE ST 2046-1 / EBU R 95), aspect-ratio mattes, nearest-neighbour magnification on N, timecode, A/B compare with wipe, difference and blink, and a sequence timeline with per-frame thumbnails.

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

Start ComfyUI and look for `Radiance: successfully loaded 131 nodes` in the log.
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

- **Viewer** — a full review surface with OpenColorIO config support, waveform and vectorscope, a pixel probe (cursor, region and full-frame statistics, source or rendered values), channel isolation, A/B compare, focus peaking, frame stepping, and keyboard shortcuts (below).
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
| N | Nearest-neighbour / linear magnification |

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

Radiance provides **131 nodes** (plus any Gizmos you create). Some nodes depend on optional packages and your ComfyUI environment.

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
- **Optical flow is pyramidal Lucas–Kanade**, not DIS or a learned method, and
  it thins out above roughly 8 px of motion. What degrades is the *field*, not
  the estimate: the median displacement stays within a few percent out to about
  20 px, but the fraction of the field landing within half a pixel falls from
  100% at 3 px to 84% at 8, 71% at 12, 58% at 16 and 29% at 20. Mask
  propagation tears where the field is patchy, so treat ~8 px as the working
  limit rather than the point of failure.

  Making the pyramid deeper does not fix it, and this was measured rather than
  assumed. The ceiling is the integration window — the coarsest level has to
  stay larger than the 15×15 window, so a short side of *S* allows about
  log₂(*S*/15) levels. On a 256×512 plate, going from four levels to five made
  every displacement worse (at 8 px, 96% of the field within half a pixel
  became 32%), because a 15×15 window on a 16-pixel-tall level is solving over
  most of the frame and that estimate propagates back down. Shrinking the
  window with the level to buy depth was tried too, and measured worse for the
  same reason.
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
- [x] **The WGSL grade is compiled and compared, and the panels are driven
      rather than grepped.** 256 JavaScript tests, all green. The claim that no
      CI environment exposes `navigator.gpu` was true of the browser and wrong
      about the environment: headless Chromium has no `navigator.gpu` at all —
      absent, not blocked, under `--enable-unsafe-webgpu` with SwiftShader,
      with `--use-vulkan=swiftshader`, and with the Blink runtime flag, while
      WebGL2 works in the same browser — but Deno ships WebGPU, and Mesa's
      lavapipe gives it a software Vulkan device, so the runner still needs no
      GPU. The emitted WGSL now runs the same 60-case matrix as the GLSL
      against the same JS functions: 58 comparable samples, worst deviation
      3.1e-7 against a 2e-6 tolerance, the two excluded for the same fp32 range
      reason. Checked by mutation, not just by passing — reverting the WGSL to
      flat lift, to unguarded gamma, or to the power-form contrast each turns
      the suite red, and those are precisely the three ways the WebGPU path
      used to disagree with WebGL. The panels are loaded in a real browser
      against stubbed ComfyUI modules, built, and then operated: selects
      changed, toggles clicked, state and store and label checked afterwards.
      That immediately found one: switching the safe-area preset to Legacy
      480-line left the caption below it still reading "SMPTE ST 2046-1 and EBU
      R 95 specify the same two boxes" — right boxes, wrong standard named, in
      a QC guide. Fixed. Still open, and smaller: the panel harness drives the
      framing, scope, probe, OCIO and view sections, not every control in the
      viewer.
- [ ] **The public repo is 2.5 months behind.** `fxtdstudios/radiance` `main`
      is still at `64fee41` (2026-06-04); everything since lives in
      `fxtdstudios/radiance-beta`. Decide when beta merges down to public.

### Correctness backlog

- [ ] **RUDRA video decoders were trained on stills.** `wan` / `ltx-video` /
      `hunyuanvideo` checkpoints never saw multi-frame latents; real video falls
      back to math expansion. Needs retraining, not patching.
- [ ] **`rudra_full_decoder_ltx-video_ema.safetensors` is truncated at source**
      (23.0 MB against a declared ~36.0 MB). The loader detects it and degrades;
      the file still needs re-exporting.
- [x] **Optical flow above ~8 px is characterised and bounded** — measured,
      not estimated. The median displacement holds within a few percent out to
      ~20 px; what degrades is field density inside half a pixel, which falls
      100% / 84% / 71% / 58% / 29% at 3 / 8 / 12 / 16 / 20 px, and mask
      propagation tears where the field is patchy. Deepening the pyramid is not
      the fix: its ceiling is the 15x15 integration window, and going from four
      levels to five made every displacement worse (at 8 px, 96% of the field
      within half a pixel became 32%). Closed here as a property of pyramidal
      Lucas-Kanade rather than an open defect, and written up under
      [Known limitations](#known-limitations). Replacing the solver — DIS, or a
      learned method — is the only route past it, and that is a new item rather
      than a continuation of this one.

### Structural debt

- [ ] **Split the monoliths** — `hdr/vae.py` (3324 lines),
      `nodes/io/write.py` (2972), `nodes/monitor/viewer.py` (1220). The middle
      one is the only thing still holding the item below open.
- [ ] **`delivery/handler.py` still imports `RadianceWrite`.** The upscale half
      of this is closed — the handler calls `radiance.image.upscale` directly
      now, and the stated reason it could not (a `tests/conftest.py` stub that
      shadowed the whole `radiance.image` package rather than just `defects`)
      turned out to be a fixable stub, not a fact about the code. What is left
      is the writer: `RadianceWrite.write` is 118 lines leaning on module-level
      helpers in the same 2972-line file, so it comes out with that split
      rather than before it.

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
- [x] **Scene-cut detection reports one calibrated 0–1 confidence, whichever
      method you pick.** Unifying the two scales turned up the larger defect
      underneath: `edge` was an *absolute* difference of gradient magnitudes,
      so it tracked the footage's own contrast rather than the cut — the same
      cut graded two stops down scored a quarter as high (0.0369 → 0.0092),
      and grain on a detailed frame outscored a real cut in soft content. No
      threshold constant could have fixed that, so it is a relative
      (Bray–Curtis) distance now, with a 5 px pre-blur so a high-pass metric
      stops measuring grain. `combined` is finally a real 60/40 blend instead
      of a sum of incompatible units. Widget renamed to `cut_confidence`, and
      `KNOWN_ISSUES.md` records what the edge method still cannot see.
- [x] Tier-3 upscale honours `scale` — 2x no longer returns the top-left
      quarter of a 4x render.
- [x] Optical flow is pyramidal: 1–5 px displacements recover to within 10%,
      with 84–100% of the field inside half a pixel (was ~100% at 1 px, 17% at
      3 px, 1% at 5 px).
- [x] Model weights are never downloaded without consent —
      `RADIANCE_ALLOW_DOWNLOADS=1` — through one gate shared by every
      downloader. The multipass path previously had none.
- [x] Tiled VAE blending verified seamless and pinned by a test (the open note
      was stale — the cosine ramp had already landed).
- [x] First JavaScript coverage: 27 tests over the shared DOM and widget
      helpers, on `node --test`, wired into CI. Since grown to 256, including a
      GPU lane that renders through the real shaders in both dialects and a
      browser lane that builds and operates the panels.
- [x] Four commits merged to `radiance-beta` `main` via PR #43.
- [x] **Both ACES 2.0 tone scales hit the published reference.** 18% grey now
      lands at 10.000 / 13.193 / 14.512 / 15.747 / 16.824 nits at peaks of
      100 / 500 / 1000 / 2000 / 4000, matching the ACES Output Transform table
      exactly. Previously the Daniele Evo node scaled grey with the display
      (400 nits at a 4000-nit peak, ~24x reference) and the legacy transform
      held it at ~18 nits everywhere (~0.85 stop bright at SDR). HLG keeps its
      BT.2408 anchor, which is a different and equally deliberate reference.
- [x] Menu placement is declared per node, not guessed. `NODE_SECTIONS` covers
      all 131; the keyword classifier is a warning-only fallback, and a test
      fails if a registered node is missing from the table.
- [x] **The legacy `nodes_*.py` layer is gone.** Forty pure re-export shims
      deleted; the six modules that were still the real home of their code —
      `nodes_io`, `nodes_sampler`, `nodes_workspace`, `nodes_realtime_preview`,
      `nodes_loader`, `nodes_gizmo` — moved into `nodes/`, so the organized
      package no longer imports *backwards* into the layer it was meant to
      replace. `RadianceViewer`, the one genuinely dual-published node, now has
      a single entry point. Nothing in the catalog moved: 111 keys in, 111 keys
      out, verified key-by-key against a pre-refactor snapshot.
- [x] **Twenty finished nodes published that had never appeared in anyone's
      menu.** Each group's `__init__.py` hand-copied a selection of its
      modules' node keys, and whatever nobody remembered to copy did not exist
      as far as ComfyUI was concerned — the same defect as #40 and as
      `RadianceGradeApply`, at scale. Found by widening the source scan once
      the root layer stopped hiding it. `nodes/aggregate.py` inverts the
      default: writing the node publishes it, and withholding one now requires
      a named `WITHHELD_NODES` entry that a test reads. Catalog 111 → 131.
- [x] `RadianceGradeApply` published — a complete node with 16 documented
      inputs that had been invisible in the menu, found because it had a
      branding override but no registration. Now `Bake Viewer Grade`, which
      also settles the Grade naming overlap. `SECTION_OVERRIDES` is empty for
      the first time; both entries it ever held were unpublished nodes.
- [x] **Every node group imports without aiohttp or a running ComfyUI.**
      `nodes/monitor/viewer.py` and `delivery/handler.py` imported
      `aiohttp`/`server` at module scope and read `PromptServer.instance`
      there, which the flat `nodes_realtime_preview.py` had hidden — moving it
      into a package made importing any Review module run the viewer, and CI's
      minimal-dependency job failed. `gizmo.py` and `workspace.py` had guarded
      the same import for years. `tests/test_import_isolation.py` now proves
      the property for all eleven groups in a subprocess with aiohttp and
      `server` blocked, rather than for one hand-listed module in CI.
- [x] **A guard that guarded nothing.** `io/formats.py` caught a missing `cv2`
      and set `HAS_CV2 = False` without binding `cv2`, so `viewer_utils.py`'s
      `from radiance.io.formats import HAS_CV2, cv2` died anyway — with a
      worse error than the ImportError the guard was written to absorb. It
      only surfaced once retiring the flat layer made a Review import pull the
      whole chain in.
- [x] Three compatibility alias keys (`RadianceImageLoader`,
      `RadianceControlApply`, `RadianceWorkspace`) ship as `DEPRECATED`
      subclasses: workflows saved against the old key still open, and the menu
      shows one entry per node instead of two.
- [x] Repository trimmed to what a user or contributor needs: the six internal
      audit and review write-ups are gone, `.comfyignore` no longer lists files
      that stopped existing, and the generated GPU report is ignored rather
      than committed.
- [x] Suite: 2421 passed / 0 failed / 69 skipped, plus 27 JS tests. Verified
      from a checkout named `radiance-beta` as well as `radiance` — CI clones
      the mirror under that name, and the suite has to mean the same thing there.
- [x] Released as **3.3.0**, not a patch: five changes alter what an unchanged
      graph does, and the changelog leads with them.

## Documentation

Full documentation is available at [www.fxtdstudios.com](https://www.fxtdstudios.com) — setup, core concepts, workflow recipes, a complete node reference, and troubleshooting.

Every node also carries its own description and per-input tooltips, which ComfyUI shows on hover, so the parameter reference travels with the package.

## Support

- Issues: [GitHub Issues](https://github.com/fxtdstudios/radiance/issues)
- Documentation: [www.fxtdstudios.com](https://www.fxtdstudios.com)
- Studio: [www.fxtdstudios.com](https://www.fxtdstudios.com)

## License

Radiance is released under the [GPL-3.0 license](LICENSE).
