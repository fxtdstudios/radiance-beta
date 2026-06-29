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

## Minor

- **Naming overlap:** `Grade` / `Grade Apply` / `Apply Grade Info` read similarly;
  to be clarified during the Color cleanup.

## Environment notes (not bugs)

- **GPU optimized kernels** prefer a `cu130` PyTorch build; on `cu128` the
  `comfy_kitchen` CUDA backend falls back to "eager" (works, slightly slower fp8/fp4).
- **`Imath`** is provided by the `OpenEXR` wheel — do not add a separate `Imath`
  dependency (PyPI `imath` maxes at 0.0.2).
