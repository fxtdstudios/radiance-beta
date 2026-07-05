# Known Issues — Radiance (beta)

Honest, tracked list of current limitations and tech debt. None of these block
normal use; they're documented so the beta ships transparently and the team has a
clear backlog.

## Architecture / tech debt

- **Dual-module structure (~39 duplicate node keys).** Many nodes are defined in
  both a legacy top-level `nodes_*.py` and the organized `nodes/<group>/` package.
  Only one wins at registration (no duplicate menu entries), but maintaining two
  source files per node is the root cause of the route-registration issue below.
  *Planned fix:* Phase 4 of `docs/dev/REFACTOR_PLAN.md` — retire the legacy shims,
  keep the organized packages.

- **Route registration depends on idempotency guards.** Because the package can be
  imported under two names, aiohttp routes are guarded against double-registration
  (`_radiance_registered_routes` on the PromptServer singleton). This prevents the
  "method HEAD is already registered" startup crash, but it is a guard around the
  dual-module symptom, not a structural fix. Removed once Phase 4 lands.

- **Monolithic files.** `nodes/monitor/viewer.py` (~1k lines, dynamic
  `globals().update()` injection), `nodes_io.py`, and `hdr/vae.py` carry several
  responsibilities and are the hardest files to change safely. *Planned fix:*
  Phase 1 of the refactor plan.

- **Heuristic menu classification.** `nodes/branding.py` keyword-classifies nodes
  into menu sections; edge cases can misfile. Mitigated by `SECTION_OVERRIDES`.
  *Planned fix:* Phase 3 — explicit per-node section declaration.

- **RUDRA video checkpoints trained on stills (wan / ltx-video / hunyuanvideo).**
  `scripts/training/dataset_hdr.py` pads every video model to T=1 before VAE
  encoding, so the shipped video-model RUDRA decoders never saw real multi-frame
  temporal latents. Symptom: abstract-noise output on real video (confirmed on
  LTX 2.3; wan/hunyuan share the code path). Multi-frame input therefore skips
  the learned path (SDR → HDR Universal falls back to math expansion, no VAE
  compute wasted). *Planned fix:* retrain the video checkpoints on real
  multi-frame sequences.

- **`rudra_full_decoder_ltx-video_ema.safetensors` truncated at the source.**
  The distributed file is 23,044,260 bytes while its header declares
  ~36,035,756 (45 of 124 tensors out of bounds) — confirmed identical across
  independent downloads. The loader now detects this before deserialization
  and falls back to the standard VAE decode with a clear log message.
  *Planned fix:* re-export the checkpoint from the training environment and
  re-upload (expected size ≈ 36.0 MB).

## Minor

- **Naming overlap:** `Grade` / `Grade Apply` / `Apply Grade Info` read similarly;
  to be clarified during the Color cleanup.

## Environment notes (not bugs)

- **GPU optimized kernels** prefer a `cu130` PyTorch build; on `cu128` the
  `comfy_kitchen` CUDA backend falls back to "eager" (works, slightly slower fp8/fp4).
- **`Imath`** is provided by the `OpenEXR` wheel — do not add a separate `Imath`
  dependency (PyPI `imath` maxes at 0.0.2).
