"""gsplat optimization loop for fitting a SPLAT to posed images. GPU + gsplat.

This is a minimal, correct-by-construction trainer (L1 photometric loss over the
posed views). Densification / pruning / SSIM / LR scheduling are deliberately
left as extension points — the goal here is a working, gated baseline that the
render+IO plumbing can build on. Validate and tune on a CUDA box.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from radiance.splatting.cameras import Cameras
from radiance.splatting.data import Splat

HAS_GSPLAT = importlib.util.find_spec("gsplat") is not None


@dataclass
class TrainConfig:
    steps: int = 3000
    sh_degree: int = 3
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3


def train(images, cameras: Cameras, init_splat: Splat,
          config: Optional[TrainConfig] = None,
          progress: Optional[Callable[[int, int, float], None]] = None,
          interrupt: Optional[Callable[[], None]] = None) -> Splat:
    """Optimize `init_splat` against `images` (B,H,W,3 in 0..1) posed by `cameras`."""
    config = config or TrainConfig()
    import torch
    try:
        from gsplat import rasterization
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Splat Train needs 'gsplat' (CUDA). Install: pip install gsplat") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Splat Train requires a CUDA GPU.")

    init_splat.validate()
    dev = "cuda"
    P = lambda a: torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(a)).to(dev))
    means, scales = P(init_splat.means), P(init_splat.scales)
    quats, opac, sh = P(init_splat.quats), P(init_splat.opacities), P(init_splat.sh)

    gt = images if torch.is_tensor(images) else torch.from_numpy(np.asarray(images, np.float32))
    gt = gt.to(dev).float()
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
    viewmats = torch.from_numpy(np.ascontiguousarray(cameras.viewmats)).to(dev)
    Ks = torch.from_numpy(np.ascontiguousarray(cameras.Ks)).to(dev)
    n_views = min(len(cameras), gt.shape[0])
    if n_views == 0:
        raise ValueError("Splat Train: no matching images/cameras")

    opt = torch.optim.Adam([
        {"params": [means], "lr": config.lr_means},
        {"params": [scales], "lr": config.lr_scales},
        {"params": [quats], "lr": config.lr_quats},
        {"params": [opac], "lr": config.lr_opacity},
        {"params": [sh], "lr": config.lr_sh},
    ])

    last = 0.0
    for step in range(int(config.steps)):
        if interrupt is not None:
            interrupt()
        i = step % n_views
        rc, _, _ = rasterization(
            means=means, quats=quats, scales=torch.exp(scales),
            opacities=torch.sigmoid(opac), colors=sh,
            viewmats=viewmats[i:i + 1], Ks=Ks[i:i + 1],
            width=cameras.width, height=cameras.height,
            sh_degree=int(config.sh_degree), render_mode="RGB",
        )
        pred = rc[..., :3].clamp(0.0, 1.0)
        loss = torch.abs(pred - gt[i:i + 1]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.item())
        if progress is not None:
            progress(step + 1, int(config.steps), last)

    return Splat(
        means.detach().cpu().numpy(), scales.detach().cpu().numpy(),
        quats.detach().cpu().numpy(), opac.detach().cpu().numpy(),
        sh.detach().cpu().numpy(), int(config.sh_degree),
        meta={"source": "train", "steps": int(config.steps), "final_loss": last},
    ).validate()


__all__ = ["TrainConfig", "train", "HAS_GSPLAT"]
