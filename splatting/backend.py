"""gsplat rendering backend (GPU).

Isolated so the rest of the splatting package stays CPU/CUDA-free. gsplat is an
optional, NVIDIA-CUDA dependency; everything here imports it lazily and raises a
clear, actionable error when it (or a GPU) is unavailable.
"""
from __future__ import annotations

import importlib.util
from typing import Tuple

import numpy as np

from radiance.splatting.cameras import Cameras
from radiance.splatting.data import Splat

HAS_GSPLAT = importlib.util.find_spec("gsplat") is not None


def _require_backend():
    import torch  # local import: torch is a core dep but keep this module light
    try:
        from gsplat import rasterization
    except Exception as exc:  # pragma: no cover - exercised only without gsplat
        raise RuntimeError(
            "Gaussian Splatting rendering needs the 'gsplat' package (NVIDIA CUDA). "
            "Install it with: pip install gsplat"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Gaussian Splatting rendering requires a CUDA GPU; none is available."
        )
    return torch, rasterization


def render(splat: Splat, cameras: Cameras,
           background: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Render a SPLAT from a camera rig.

    Returns (image, depth, alpha) as torch tensors:
      image (B,H,W,3) 0..1, depth (B,H,W,1), alpha (B,H,W,1).
    """
    torch, rasterization = _require_backend()
    splat.validate()
    dev = "cuda"

    means = torch.from_numpy(np.ascontiguousarray(splat.means)).to(dev)
    quats = torch.from_numpy(np.ascontiguousarray(splat.quats)).to(dev)
    # 3DGS .ply stores log-scale and logit-opacity; activate for the rasterizer.
    scales = torch.exp(torch.from_numpy(np.ascontiguousarray(splat.scales)).to(dev))
    opac = torch.sigmoid(torch.from_numpy(np.ascontiguousarray(splat.opacities)).to(dev))
    colors = torch.from_numpy(np.ascontiguousarray(splat.sh)).to(dev)      # (N,K,3) SH
    viewmats = torch.from_numpy(np.ascontiguousarray(cameras.viewmats)).to(dev)
    Ks = torch.from_numpy(np.ascontiguousarray(cameras.Ks)).to(dev)
    bg = torch.tensor([background], dtype=torch.float32, device=dev).expand(len(cameras), 3)

    render_colors, render_alphas, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
        viewmats=viewmats, Ks=Ks, width=cameras.width, height=cameras.height,
        sh_degree=int(splat.sh_degree), render_mode="RGB+D", backgrounds=bg,
    )

    # RGB+D packs depth as the 4th channel; fall back gracefully if RGB-only.
    if render_colors.shape[-1] >= 4:
        image = render_colors[..., :3]
        depth = render_colors[..., 3:4]
    else:
        image = render_colors[..., :3]
        depth = torch.zeros(*image.shape[:3], 1, device=dev)
    image = image.clamp(0.0, 1.0).contiguous().cpu()
    depth = depth.contiguous().cpu()
    alpha = render_alphas.contiguous().cpu()
    return image, depth, alpha


__all__ = ["HAS_GSPLAT", "render"]
