# Known Issues — Radiance (beta)

Honest, tracked list of current limitations and tech debt. None of these block
normal use; they're documented so the beta ships transparently and the team has a
clear backlog.

## Architecture / tech debt

- **Route registration keeps an idempotency guard.** aiohttp routes register
  through `_radiance_route_once`, which records what it has already registered
  on the PromptServer singleton, so a module imported twice cannot crash
  startup with "method HEAD is already registered". The dual-module structure
  that made this necessary is gone — the legacy `nodes_*.py` layer was removed
  in 3.3.0 and there is one entry point now — but ComfyUI puts `custom_nodes/`
  on `sys.path`, so a user script doing `import viewer` can still produce a
  second copy of a module. One file (`nodes/monitor/viewer.py`) carries the
  guard; two more read the same registry. Kept as cheap insurance, no longer a
  symptom of anything.

- **Masks and Qualifiers are inert on the WebGPU backend.** `js/radiance_webgpu.js`
  carries no WGSL for either one. The shared base class, `js/radiance_renderer.js`,
  implements `setMask` and `setQualifier` and stores the state, so the Viewer's
  calls succeed and nothing changes on screen: both tabs are fully interactive
  and completely inert on that backend. WebGPU is opt-in since 3.4.0
  (`localStorage.radiance_prefer_webgpu = "1"`); WebGL, the default, renders
  both.
  OCIO has the same WGSL gap and at least says so, returning "OpenColorIO needs
  the WebGL backend; this renderer has no WGSL path for it"; these two say
  nothing. *Planned fix:* none scheduled. The two resolutions are porting the
  shaders to WGSL, or disabling the tabs on WebGPU with that message. The gap is
  listed in `js/tests/backend_parity.test.mjs`, which fails if this entry and
  the shipped source stop agreeing, so neither resolution can land without
  updating this list.
  Beyond masks, qualifiers and OCIO (3.5.0 audit), the WebGPU backend also
  differs from WebGL in: no sRGB decode for 8-bit input; a 3D LUT read with the
  wrong stride; the red curve applied to every channel; a transposed hue
  matrix; misregistered bloom; and no heatmap, gamut/clip warnings, scope
  signal, viewer f-stop or scene-linear graded EXR. `FEATURE_PARITY` now says
  false. Porting these to WGSL is the remaining work; until then WebGL is the
  default and the tested path.

- **Monolithic files.** `hdr/vae.py` (3324 lines), `nodes/io/write.py` (2972)
  and `nodes/monitor/viewer.py` (1220, with dynamic `globals().update()`
  injection) carry several responsibilities each and are the hardest files to
  change safely. The `nodes/io/write.py` split is also what unblocks
  `delivery/handler.py`, which still reaches up into the node layer for
  `RadianceWrite`. Tracked in the README's Status & to-do list.

- **Learned SDR → HDR is one pixel model.** The latent-space RUDRA decoders
  (per-model `rudra_turbo_decoder_*` / `rudra_full_decoder_*` checkpoints,
  the video decoders trained on stills, and the truncated `ltx-video` full
  decoder) were retired in 3.5.0. `SDR → HDR Universal` and `Recover` run the
  pixel model (`sdr2hdr_pixel_image.pt`) on any image or frame batch and the
  temporal residual model on ordered video when its checkpoint is installed.
  The pixel model processes video frames independently; a clip may need
  downstream deflicker until the temporal checkpoint ships.

## Placeholders and labelled limits (3.5.0 honest release pass)

- **SAM Loader / SAM Mask Generator are not shipped.** Radiance bundles no
  SAM runtime; the generator used to return discs drawn around the click
  points. Both nodes are hidden from the menu (`DEPRECATED`), still load in
  saved graphs, and raise a clear error when executed. *Planned fix:* a real
  SAM2 backend, or removal in 4.0.
- **Upscale `confidence` output is geometric.** 1 at tile centres, lower
  toward tile edges. It locates seam blending; it does not measure
  hallucination, which none of the backends report.
- **Upscale Video is windowed, not temporal.** Tier 1/2 models upscale each
  frame independently; overlap frames are flow-aligned and blended at window
  seams. GAN flicker between frames is not removed.
- **`cfg_schedule_json` is a static override.** T2V, I2V and Video Sampler use
  its first value as CFG. A per-step schedule needs a sampler CFG hook and is
  not implemented.
- **Video HDR Conditioner needs `clip`.** Without it the conditioning passes
  through unchanged and `hdr_metadata_json.applied_to_model` is false.
- **I2V on Wan 2.2 TI2V 5B uses `first_frame_lock`.** That model has no
  image-concat input; ComfyUI's own path for it is `Wan22ImageToVideoLatent`
  (image in the latent with a noise mask), which the pipeline does not yet
  reproduce.
- **Audio Cut `method` is honoured by librosa only.** scipy finds energy
  onsets and ffmpeg finds silence ends whatever the method; the report says
  which ran. `max_cuts` keeps an evenly spaced subset (no strength is
  reported by any backend).
- **Camera Sync reads JSON only.** Alembic (`.abc`) raises with a message.
- **HDR Synthesis Engine does not reconstruct clipped detail.** It lifts the
  low-pass base toward `energy_target`; `recovery_iters` is the pyramid
  depth. Learned recovery is SDR → HDR Universal / Recover.
- **Relight Engine adds one directional light.** Lambert + Blinn-Phong added
  to the image, not an albedo relight, and no environment map.
- **SDR → HDR Expand keeps SDR midtones at their SDR display level.** Below
  the knee the signal stays at 1.0 = 100 nits (18 % grey at 18 nits) and only
  the knee-to-white span opens up to reference white; BT.2446 Method B would
  also lift midtones by ~2x. Raise `reference_white_nits` or grade after for a
  brighter mid-scale.
- **Pixel SDR → HDR runs frame by frame.** Ordered video gets the temporal
  model only when its checkpoint is installed; otherwise downstream deflicker
  may be needed. `pixel_tile_size` changes the result slightly (each tile sees
  less context); keep it at 512 or above. On CPU a 1080p frame takes ~19 s
  (0.16 s for Expand); GPU timing is still to be measured on the 4080.
- **HLG is referenced to a 1000-nit display.** BT.2100 / BT.2408 / OCIO
  convention: highlights mastered above 1000 nits clip in the HLG signal. Use
  PQ for 4000- and 10000-nit masters.

## Viewer: remaining (phase 3)

- **No HDR output.** The canvas is SDR (sRGB or Display P3, 8-bit, dithered). A Rec.2100 PQ / HLG canvas needs a float16 HDR drawing buffer in both the WebGL and the 2D compositing canvas; browser support is still settling and it cannot be verified without an HDR display. Use HDR Monitor or an external HDR display for HDR review.
- **Compare B side.** It is the node's display-referred preview, baked with ACES 2.0 or untouched for sRGB. Switching the A side to another view does not re-render B. Side-by-side and difference are 2D-fallback only.
- **Non-OCIO views on non-Rec.709 sources.** A source tagged ACEScg or Rec.2020 is shown with Rec.709 primaries in the sRGB, Rec.709 and Filmic views. The Auto and ACES views go through OCIO with the right source.
- **Annotations are screen-space.** They do not follow pan and zoom.
- **No audio.** Playback is picture only.
- **Pro scope "False colour" mode** is a display-level zone map, not the ARRI bands of the viewer overlay.

## Minor

- **`RadianceSDRToHDRUniversal`'s learned modes need a checkpoint you must
  install yourself.** `Recover` and `Hybrid` are the reason the node is called
  Universal, and with no `sdr2hdr_pixel_image.pt` in `models/radiance` and no
  `RADIANCE_SDR2HDR_PIXEL`, they produce output bit-identical to `Expand`.
  That is a safe fallback rather than a failure, and the node's `report`
  output names the path that ran and, when the learned path did not, why and
  what to install; a WARNING is also logged. The dedicated `Recover` node
  raises instead. *Planned fix:* auto-fetch the default checkpoint, gated by
  `RADIANCE_ALLOW_DOWNLOADS` like every other downloader.


- **Naming overlap:** `Grade` / `Grade Apply` / `Apply Grade Info` read similarly;
  to be clarified during the Color cleanup.

- **Scene-cut `edge` method is weak on soft and grainy footage.** It compares
  gradient magnitudes, so it cannot see a cut between two frames that both lack
  edges — a soft gradient cutting to a different soft gradient, or black
  cutting to white — and heavy grain still moves it more than a subtle cut
  does, because a gradient operator is a high-pass filter and grain is
  high-frequency. The 5 px pre-blur reduces that but does not remove it. This
  is why `combined` is the default and is weighted 60/40 toward the histogram.
  Use `histogram` alone on soft material; use `edge` alone only for the case
  the histogram cannot see — the palette held, the framing changed. *Planned
  fix:* none scheduled. The two methods cover each other's blind spots, and a
  metric that handled both would be a different detector — feature matching or
  a learned embedding — not a tuned constant.

## Environment notes (not bugs)

- **GPU optimized kernels** prefer a `cu130` PyTorch build; on `cu128` the
  `comfy_kitchen` CUDA backend falls back to "eager" (works, slightly slower fp8/fp4).
- **`Imath`** is provided by the `OpenEXR` wheel — do not add a separate `Imath`
  dependency (PyPI `imath` maxes at 0.0.2).
