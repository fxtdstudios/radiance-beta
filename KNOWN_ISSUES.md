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

- **Monolithic files.** `hdr/vae.py` (3324 lines), `nodes/io/write.py` (2972)
  and `nodes/monitor/viewer.py` (1220, with dynamic `globals().update()`
  injection) carry several responsibilities each and are the hardest files to
  change safely. The `nodes/io/write.py` split is also what unblocks
  `delivery/handler.py`, which still reaches up into the node layer for
  `RadianceWrite`. Tracked in the README's Status & to-do list.

- **RUDRA video checkpoints trained on stills (wan / ltx-video / hunyuanvideo).**
  `scripts/training/dataset_hdr.py` pads every video model to T=1 before VAE
  encoding, so the shipped video-model RUDRA decoders never saw real multi-frame
  temporal latents. Symptom: abstract-noise output on real video (confirmed on
  LTX 2.3; wan/hunyuan share the code path). Multi-frame input therefore skips
  the learned path (SDR → HDR Universal falls back to math expansion, no VAE
  compute wasted). *Planned fix:* retrain the video checkpoints on real
  multi-frame sequences.

  A retrained checkpoint no longer inherits the warning by filename: the loader
  reads the frame count the training run stamped into the safetensors metadata
  (`radiance_train_frames` and three aliases) and warns only when the
  checkpoint declares one frame or declares nothing. Two node-side gates still
  stand in the way of using retrained weights on video: `SDR → HDR Universal`
  routes the Legacy RUDRA backend only when `img.shape[0] == 1`, and
  `_rudra_reconstruct` refuses multi-frame outright.

- **`rudra_full_decoder_ltx-video_ema.safetensors` truncated at the source.**
  The distributed file is 23,044,260 bytes while its header declares
  ~36,035,756 (45 of 124 tensors out of bounds) — confirmed identical across
  independent downloads. The loader now detects this before deserialization
  and falls back to the standard VAE decode with a clear log message.
  *Planned fix:* re-export the checkpoint from the training environment and
  re-upload (expected size ≈ 36.0 MB).

## Minor

- **`RadianceSDRToHDRUniversal`'s learned modes need a checkpoint you must
  install yourself.** `Recover` and `Hybrid` are the reason the node is called
  Universal, and with no RUDRA checkpoint in `models/radiance`, no
  `RADIANCE_SDR2HDR_PIXEL`, and no VAE connected, they produce output
  bit-identical to `Expand` — verified. That is a safe fallback rather than a
  failure, but it used to be silent. From 3.4 the node's `report` output names
  the path that ran and, when the learned path did not, why and what to
  install; a WARNING is also logged. *Planned fix:* ship or auto-fetch a
  default checkpoint, gated by `RADIANCE_ALLOW_DOWNLOADS` like every other
  downloader.


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
