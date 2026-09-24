# Radiance development record

For contributors and reviewers: what has been verified, and what is open
before and after this release.


Radiance is **ready with conditions**. Everything below is measured rather than
asserted; the full history is in the [changelog](../CHANGELOG.md), and per-defect
detail in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).

## Verified

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
| **Security** | `weights_only` loads, sha256-pinned downloads, no `shell=True`, and no third-party weight downloads without `RADIANCE_ALLOW_DOWNLOADS=1`, through a gate every downloader shares. Two exceptions: Radiance's own ~5 MB RUDRA checkpoint, fetched on first use, pinned to a commit and SHA-256 checked; and Multipass Estimate's MoGe-2 and Marigold weights, fetched when the node runs with its `download_missing_models` switch on, pinned to commits and hash-checked by Hugging Face. Both are off with `RADIANCE_ALLOW_DOWNLOADS=0`. Nodes never write into the ComfyUI install directory. |
| **Catalog** | All 156 nodes declare their menu section explicitly; a test fails if a registered node is missing from the table. Withholding a node from the menu requires a named entry with a written reason a test reads and checks the length of. Separately, an AST walk of every file in the distribution finds every `NODE_CLASS_MAPPINGS` and asserts each class in it is registered as the class that ships, so a node stranded in a package the catalog does not load turns the suite red. That is how 26 finished nodes stayed out of the menu until 3.4.0. |
| **Isolation** | Every one of the eleven node groups imports with `aiohttp` and `server` blocked, proven in a subprocess rather than for one hand-listed module. The blocker uses `find_spec`; it previously used `find_module`, which Python 3.12 removed, so on the 3.12 leg of the matrix it silently blocked nothing and the test passed while measuring nothing. The harness now proves it is blocking before it reports anything. |
| **Layering** | `radiance/io/writer.py` and `radiance/io/reader.py` import nothing above them, checked by AST walk *and* by running them in a bare interpreter with no ComfyUI present. |
| **Suite** | 3755 Python tests and 311 JavaScript tests. On the full dependency lane: 3667 pass, 84 skip, 3 are `slow` and deselected by default; JS is 307 pass, 4 skip, 0 todo. Coverage is **59.6% of 28,018 statements** (56.0% counting branches, which is what the floor gates on). The old 53% was the figure the full lane would have produced had it run; CI's lightweight lane was really reporting 22% against a `--cov-fail-under=15` that overrode the project's own floor. There is one floor now, in `pyproject.toml`, enforced on the lane that can execute the code. The JS side includes a GPU lane that compiles the real shaders in both GLSL and WGSL and compares them against the CPU implementations they were generated from, and a browser lane that builds all fourteen Viewer panels and operates their controls. Verified from a checkout named `radiance-beta` as well as `radiance`. |

## Open

**Blocking a release**

- [ ] **One full GPU render in live ComfyUI.** Checked on 2026-09-24 in a real
      ComfyUI 0.32 with frontend 1.48 (CPU): all 158 nodes register, every
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
- [x] **Clean install from the registry package (3.5.0).** Packed with
      `comfy node pack`, installed into a fresh ComfyUI 0.32 on Python 3.13:
      157 nodes, OCIO configured, the RUDRA model fetched and applied on first
      run, the suite green there. VAE Encode (HDR) registered as the encoder
      VAE Decode (HDR) inverts; the two legacy HDR latent encoders labelled.
- [x] **RUDRA pixel model: no false colour, no over-peak channels (3.5.0).**
      From a user report on a clipped Flux.2 sunset. Recovered highlights keep
      the source colour (no rings, no red cast, source-level chroma noise),
      no channel exceeds `peak_nits`, the whole frame runs untiled when it
      fits, and every published checkpoint loads, with the shipped
      `sdr2hdr_shadow_v1` as the default and the auto-download.
- [x] **Documentation complete (3.5.0).** Every menu node has a description
      and every input a tooltip, written from the code and spot-checked by a
      separate reviewer (43 of 45 sampled tooltips accurate, the other two
      reworded). The node reference in `docs/nodes` is generated and tested
      for freshness; five example workflows were run from their saved files
      on a clean install. About 45 controls found not to match their names
      are listed in KNOWN_ISSUES for the next pass.
- [x] **Slow tests are opt-in (3.5.0).** The 1-to-100 video frame-count
      sweep ran on every `pytest` although it is marked `slow`; it is now
      deselected by default (`pytest -m slow` runs it, and CI runs it in its
      own step) and runs in parallel. Default suite about 68 s faster.
- [x] **Release feature test from the registry package (3.5.0).** Packed
      with `comfy node pack` (238 files, 3.5 MB, no tests or dev tools),
      installed into a clean ComfyUI 0.32 on Python 3.13 the way
      ComfyUI-Manager does it: 158 nodes, OCIO configured, the RUDRA model
      and MoGe-2 fetched on first use. Run through the API: SDR → HDR
      Universal to 32-bit EXR (HDR to 4.9, 19 % of pixels above 1.0) and the
      Viewer; Grade, CDL, colour-space convert, OCIO and tone map to 16-bit
      PNG; VAE Encode (HDR) to VAE Decode (HDR) with a real SD VAE (median
      error 0.09 stops, 96 % of over-range values kept); 8-frame H.264 and
      ProRes 422 HQ (8 frames each); Multipass Estimate to EXR passes, Read
      AOVs and Relight. In a headless browser every node creates, and the
      graph saves and reloads, with no Radiance console error. Three bugs
      this found are fixed (see the changelog).
- [x] **Release clean-up (3.5.0).** Removed 26 files (21, then 5 in a second pass) that nothing loaded,
      called or documented, including a stale offline manual that ComfyUI
      loaded as an extension on every page in git installs and a front-end
      extension for a node that does not exist. `tools/check_release_ready.py`
      is fixed and now runs in the suite, so version, node count, licence
      and packaging drift fail a test.
- [x] **Multipass passes are real or removed (3.5.0).** Multipass Extract's
      image-filter passes (Retinex albedo, blur-difference specular, contrast
      roughness, emission, transmission, reflection, k-means object ID) are
      gone. Multipass Estimate predicts geometry with MoGe-2 and materials and
      lighting with Marigold IID, computes GTAO and metric curvature from the
      geometry, and fits the lighting to the plate. Run on the real models on
      a photo: FOV, metric depth, normals, AO and curvature checked visually
      and on analytic scenes (plane, 90 degree crease, unit sphere), lighting
      rebuilds the plate to 11 percent RMS.
- [x] **Viewer and Lite Viewer, phase 1 (3.5.0).**
  - **Colour.** The node tags every frame: a ComfyUI IMAGE is shown exactly as ComfyUI shows it, and a linear source goes through OpenColorIO ACES 2.0. A normal image used to be read as linear and sRGB-encoded twice, which washed it out, and its input colour space was guessed from brightness.
  - **View menu.** Every entry is real (ACES 2.0 and 1.3 through OCIO, sRGB, Rec.709 BT.1886). The same view is baked into the PNG previews.
  - **Units.** Everything reads 203 nits for 1.0, and the DaVinci Intermediate curve matches the spec again.
  - **Compare and playback.** Compare works on WebGL and follows the playhead. Revisited frames no longer go black. Playback follows the source fps. Timecode is SMPTE, with drop-frame at 29.97 and 59.94.
  - **Export and transport.** The graded EXR is scene-linear and tagged with its primaries. Frames travel as half-float by default, and an unchanged viewer no longer re-runs everything downstream.
  - **Lite Viewer** (the node was removed later in 3.5.0, see below). Readout, clip check and diff use float source values. 1:1 is exact on scaled displays. It has play/loop.
- [x] **Cinematic Encoder, step 1 (3.5.0).** Long SDXL prompts keep their camera and lighting (no `BREAK`, no 77-token cut); Wan, Flux.2, Z-Image, Lumina2, Qwen-Image, AuraFlow and every other T5 / LLM encoder get prose instead of SDXL tags; the subject is never rewritten; Flux, Flux.2 and MiniMax skip the unused negative encode.
- [x] **Viewer phase 2 (3.5.0).**
  - **Scopes.** Waveform, vectorscope and histogram measure the picture as displayed (graded, through the active view), not the ungraded texture or the 8-bit canvas. The vectorscope is BT.709 Cb/Cr with 75 % and 100 % targets and a skin line.
  - **Warnings.** False colour uses ARRI's bands on the Rec.709 signal. Clip and gamut warnings run before the display clamp, so they fire.
  - **Viewer-only exposure and gamma.** `f/` and `γ` in the viewer bar (and `-` / `=`, `0` to reset) change only what you see, never scopes, readout or export.
  - **Pixels.** The canvas is device-pixel sized; zoom above 1:1 is nearest-neighbour by default. Output is dithered, and a P3 monitor gets a Display P3 view and canvas.
  - **Keys and transport.** Keys go to the viewer under the pointer only. In/out (`I` / `O`), J/K/L shuttle, ping-pong and play-once, play every frame with a dropped-frame count.
- [x] **Viewer player checked end to end (3.5.0).** Load, paused seeks, fast scrub, arrow keys, Space, J/K/L, in/out loop, ping-pong, play-once, whole-clip wrap, Home/End and memory over a full loop, on MP4, a 240-frame clip and a PNG sequence, with the frame read back from the pixels. Fixed: playback freezing at the end of the range, the node growing without limit, J on a directly loaded video, and the software-backed 2D canvas. Playback fps on real hardware is still to be measured (the check machine renders WebGL in software).
- [x] **Viewer Simple / Advanced, one compare, Lite Viewer removed (3.5.0).** A switch in the title bar: Simple is picture, transport and compare; Advanced is every panel. Saved per node; graphs saved before it open in Advanced. The Lite Viewer node is deleted; a saved one is converted to a Viewer in Simple mode when the graph loads (before ComfyUI's missing-node check), links re-pointed by socket name. Compare is one controller (A, B, Wipe, Diff, Blink, pin and release) behind every button in both modes. Checked in a browser with B = the clip through ImageInvert, so every mode has a known picture: 29 of 29 checks, including B following the playhead, a pin surviving a new run, save and reload, and an old Lite Viewer graph running with its widget values in place.
- [x] **VFX efficiency, the severe six (3.5.0).** An audit of the 31 VFX nodes (report in the project: `vfx_efficiency_3.5.0.md`) found six that broke or crawled at production size. Fixed, each checked against the old maths: Motion Blur (the vector blur in the menu: +1 GB -> +0.48 GB, identical; the unregistered rival in film/camera.py: out of memory -> 3.7 s), Multipass Relight (killed out of memory -> 4.0 s, output identical), Linear Matting (107 s -> 13.5 s, separable box filters, identical to 2e-6), HDR Stitch (23.6 s -> 3.1 s), Floyd-Steinberg dither (8.2 s a frame -> 0.06 s a frame in a batch, bit-identical), Depth Map Generator (batched, GPU preprocessing identical to the processor, fp16 on CUDA, depth no longer kept on the GPU). CPU timings; the GPU gains are larger and still to be measured on the 4080. `tests/test_vfx_efficiency.py` (28).
- [ ] **VFX efficiency, the rest.** A shared GPU-in-chunks pass for the optics, grain, defocus, stabilizer, grain matcher and relight / composite nodes and Multipass Estimate's geometry; then the EXR passes writer, Compression Artifacts and Scene Cut Detect. See the project report.
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
