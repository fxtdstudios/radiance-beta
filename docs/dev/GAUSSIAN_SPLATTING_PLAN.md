# Gaussian Splatting — Implementation Plan

Adds a **Gaussian Splatting** menu to Radiance, wrapping `gsplat` (Apache-2.0)
for rendering/training. Render-first, training-second; the rasterizer is never
reimplemented.

## Layout
- `nodes/splatting/` — ComfyUI node classes (registered via `nodes/catalog.py`).
- `splatting/` — implementation library (no ComfyUI imports): `data.py` (SPLAT
  type), `ply.py` (.ply IO), `cameras.py`, `backend.py` (gsplat), `colmap.py`.
- Menu `FXTD STUDIOS/Radiance/Gaussian Splatting` via `nodes/branding.py`.

## SPLAT type
`splatting.data.Splat`: means (N,3), scales (N,3), quats (N,4), opacities (N,),
sh (N,K,3) with K=(degree+1)^2, sh_degree, meta. Flows between splat nodes.

## Phases
- **1 — render & IO (CPU for IO, GPU for render):** Splat Load, Splat Info,
  Splat Export (this milestone, CPU-only); then Camera Orbit/Path + Splat Render
  (gsplat). Render output (IMAGE+depth) feeds Grade/OCIO/HDR/Write/Viewer/DCC.
- **2 — training (GPU):** COLMAP Poses, Splat Train, Splat Crop/Transform.
- **3 — VFX:** depth/AOV extraction, relight, mesh export, 4D/video splats.

## Dependencies (all optional, gated in config/dependencies.py)
`gsplat` (CUDA; render+train), `pycolmap` (poses), `plyfile` (optional IO accel).
The bundled `.ply` codec is pure-Python (no deps). gsplat is NVIDIA-CUDA only
(AMD via ROCm fork; Apple Silicon cannot render/train) — gate + document.

## Tests / CI
CPU logic (PLY round-trip, SPLAT schema, camera math) runs in CI. Render/train
tests guard on `HAS_GSPLAT` + GPU and skip in CI (like the torch-full tests).
Keep gsplat out of CI deps.
