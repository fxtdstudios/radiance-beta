[← Back to Radiance docs](../README.md)

# Gaussian Splatting

Radiance includes a 3D Gaussian Splatting pipeline under the **Gaussian Splatting**
menu — load, inspect, and export `.ply` splats, render novel views, import COLMAP
reconstructions, and train splats from posed images. Rendered views and depth flow
straight into the rest of Radiance (Grade, OCIO, HDR, Write, Viewer, DCC handoff).

## Requirements

- **`.ply` load / info / export and COLMAP import are CPU-only** — no GPU needed.
- **Render and Train require [`gsplat`](https://github.com/nerfstudio-project/gsplat)
  and an NVIDIA CUDA GPU** (`pip install gsplat`). On a machine without them, those
  two nodes raise a clear message; the rest of the menu still works.

## Nodes

| Node | Purpose |
| :--- | :--- |
| **Splat Load** | Read a Gaussian Splatting `.ply` or `.splat` into a `SPLAT`. |
| **Splat Info** | Report point count, SH degree, bounds, and opacity range. |
| **Splat Export** | Write a `SPLAT` to a binary `.ply` or web `.splat`. |
| **Splat Transform** | Translate, rotate, and uniformly scale a `SPLAT`. |
| **Splat Crop** | Keep only gaussians inside an axis-aligned box. |
| **Splat Merge** | Concatenate two `SPLAT`s into one. |
| **Camera Orbit** | Build an orbit camera rig (`RAD_CAMERAS`) around a center point. |
| **Splat Render** | Render a `SPLAT` through a camera rig → image, depth, alpha *(gsplat/CUDA)*. |
| **Splat Viewer 3D** | Interactive in-node viewer — orbit, pan, zoom (WebGL, no GPU/gsplat needed). |
| **COLMAP Load** | Read a COLMAP sparse model (`.bin`/`.txt`) → cameras, camera-ordered images, and a point-initialized `SPLAT`. |
| **Splat Train** | Optimize a `SPLAT` against posed images with adaptive densification and a live preview every N steps *(gsplat/CUDA)*. |

## Workflows

**Render an existing splat**

```text
Splat Load → Camera Orbit → Splat Render → (Grade / HDR / Viewer / Write)
```

**Train a splat from a COLMAP dataset**

```text
COLMAP Load (model_dir + images_dir) → Splat Train → Splat Render → Write
```

`COLMAP Load` loads the source images in the **same order as the cameras**, so the
image/camera correspondence Splat Train needs is guaranteed — no manual matching.

## Notes

- The `SPLAT` type carries means, scales, quats, opacities, and spherical-harmonic
  colour; it is passed between splat nodes like `IMAGE`/`LATENT`.
- `Splat Train` uses gsplat's adaptive density control (clone/split/prune) plus an
  L1 + SSIM loss. It is a strong baseline; tune steps and learning rates per scene.
- `Splat Viewer 3D` renders with WebGL2 for maximum browser compatibility. Planned:
  a WebGPU backend (GPU compute sort) for smooth interaction on multi-million-splat
  scenes, with automatic fallback to the current WebGL2 renderer.
