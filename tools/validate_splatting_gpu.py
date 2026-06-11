#!/usr/bin/env python3
"""GPU smoke test for Radiance Gaussian Splatting (render + train).

Run this on a machine with an NVIDIA CUDA GPU and gsplat installed:

    cd ComfyUI/custom_nodes/radiance
    pip install gsplat plyfile          # in ComfyUI's Python env
    python tools/validate_splatting_gpu.py

It builds a tiny random splat, renders an orbit, then runs a few training
steps against the rendered frames. No data files or ComfyUI needed.
If anything errors, copy the full output back and it can be fixed.
"""
from __future__ import annotations

import os
import sys
import traceback

# Make `import radiance...` work whether run from the repo root or tools/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_PARENT = os.path.dirname(_REPO)
for p in (_PARENT, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    _section("1. Environment")
    try:
        import torch
    except Exception:
        print("FAIL: torch is not installed in this environment.")
        return 1
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    else:
        print("FAIL: no CUDA device visible to torch. Render/train need a GPU.")
        return 1
    try:
        import gsplat  # noqa: F401
        print("gsplat:", getattr(gsplat, "__version__", "(version unknown)"))
    except Exception:
        print("FAIL: gsplat not installed. Run: pip install gsplat")
        return 1

    # Import the library (works via either 'radiance.splatting' or 'splatting').
    try:
        from radiance.splatting.init import init_from_points
        from radiance.splatting.cameras import orbit
        from radiance.splatting.backend import render
        from radiance.splatting.train import train, TrainConfig
    except Exception:
        from splatting.init import init_from_points          # type: ignore
        from splatting.cameras import orbit                  # type: ignore
        from splatting.backend import render                 # type: ignore
        from splatting.train import train, TrainConfig       # type: ignore

    rng = np.random.default_rng(0)

    _section("2. Build a tiny random splat")
    n = 2000
    points = rng.normal(0.0, 0.4, size=(n, 3)).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
    splat = init_from_points(points, colors, sh_degree=1)
    print("splat:", splat.count, "gaussians, sh_degree", splat.sh_degree)

    _section("3. Render an orbit")
    W, H = 256, 256
    cams = orbit(num_frames=4, radius=3.0, elevation_deg=15.0, width=W, height=H)
    print("cameras:", len(cams), f"@ {cams.width}x{cams.height}")
    image, depth, alpha = render(splat, cams, background=(0.0, 0.0, 0.0))
    print("image:", tuple(image.shape), image.dtype,
          "range", float(image.min()), "->", float(image.max()))
    print("depth:", tuple(depth.shape), " alpha:", tuple(alpha.shape))
    assert image.shape[0] == len(cams) and image.shape[1:3] == (H, W), "image shape mismatch"
    print("RENDER OK")

    _section("4. Train a few steps against the rendered frames")
    target = image.detach().to("cpu").numpy().astype(np.float32)  # (B,H,W,3) 0..1
    cfg = TrainConfig(steps=50, sh_degree=1, densify=True,
                      refine_start=10, refine_stop=40, refine_every=10)

    def _progress(step, total, loss):
        if step == 1 or step % 10 == 0 or step == total:
            print(f"  step {step:4d}/{total}  loss={loss:.5f}")

    trained = train(target, cams, splat, config=cfg, progress=_progress)
    print("trained splat:", trained.count, "gaussians")
    print("final_loss:", trained.meta.get("final_loss"))
    print("TRAIN OK")

    _section("5. Re-render the trained splat")
    img2, _, _ = render(trained, cams)
    err = float(np.abs(img2.detach().cpu().numpy() - target).mean())
    print("mean abs error vs target after training:", round(err, 5))
    print("RE-RENDER OK")

    _section("RESULT")
    print("ALL CHECKS PASSED — render + train run on this GPU.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n" + "!" * 60)
        print("VALIDATION FAILED — copy everything below back for a fix:")
        print("!" * 60)
        traceback.print_exc()
        sys.exit(1)
