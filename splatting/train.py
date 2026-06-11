"""gsplat optimization for fitting a SPLAT to posed images (GPU + gsplat).

Uses gsplat's DefaultStrategy for adaptive density control (clone/split/prune/
opacity-reset) when available, falling back to a plain Adam loop on older gsplat
or when densification is disabled. Loss is (1-lambda)*L1 + lambda*(1-SSIM), the
standard 3DGS objective. GPU-only; validate/tune on a CUDA box.
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
    steps: int = 7000
    sh_degree: int = 3
    ssim_lambda: float = 0.2
    # learning rates
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3
    # adaptive density control
    densify: bool = True
    refine_start: int = 500
    refine_stop: int = 15000
    refine_every: int = 100
    reset_every: int = 3000
    # live preview: every N steps hand the current render to the preview callback (0 = off)
    preview_every: int = 0


def _ssim(pred, target):
    """Lightweight SSIM on (B,3,H,W) tensors in 0..1."""
    import torch
    import torch.nn.functional as F
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    win = 11
    coords = torch.arange(win, dtype=torch.float32, device=pred.device) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] @ g[None, :])[None, None].expand(3, 1, win, win)

    def filt(x):
        return F.conv2d(x, kernel, padding=win // 2, groups=3)

    mu_x, mu_y = filt(pred), filt(target)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = filt(pred * pred) - mu_x2
    sy = filt(target * target) - mu_y2
    sxy = filt(pred * target) - mu_xy
    ssim = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx + sy + c2))
    return ssim.mean()


def _scene_scale(cameras: Cameras) -> float:
    c = cameras.centers()
    if c.shape[0] <= 1:
        return 1.0
    return float(np.linalg.norm(c - c.mean(0), axis=1).mean()) or 1.0


def train(images, cameras: Cameras, init_splat: Splat,
          config: Optional[TrainConfig] = None,
          progress: Optional[Callable[[int, int, float], None]] = None,
          interrupt: Optional[Callable[[], None]] = None,
          preview: Optional[Callable[[np.ndarray], None]] = None) -> Splat:
    """Optimize `init_splat` against `images` (B,H,W,3 in 0..1), order-matched to cameras."""
    config = config or TrainConfig()
    import torch
    try:
        from gsplat import rasterization
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Splat Train needs 'gsplat' (CUDA). Install: pip install gsplat") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Splat Train requires a CUDA GPU.")
    from radiance.splatting.backend import _gsplat_cuda_ready, _GSPLAT_DISABLED_HINT
    if not _gsplat_cuda_ready():
        raise RuntimeError(_GSPLAT_DISABLED_HINT)

    init_splat.validate()
    dev = "cuda"

    def par(a):
        return torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(a)).to(dev))

    # gsplat strategy expects SH split into DC (sh0) and the rest (shN).
    params = torch.nn.ParameterDict({
        "means": par(init_splat.means),
        "scales": par(init_splat.scales),
        "quats": par(init_splat.quats),
        "opacities": par(init_splat.opacities),
        "sh0": par(init_splat.sh[:, :1, :]),
        "shN": par(init_splat.sh[:, 1:, :] if init_splat.sh_coeffs > 1
                   else np.zeros((init_splat.count, 0, 3), np.float32)),
    }).to(dev)
    lrs = {"means": config.lr_means, "scales": config.lr_scales, "quats": config.lr_quats,
           "opacities": config.lr_opacity, "sh0": config.lr_sh, "shN": config.lr_sh / 20.0}
    optimizers = {k: torch.optim.Adam([{"params": [params[k]], "lr": lrs[k]}], eps=1e-15)
                  for k in params}

    strategy = None
    state = None
    if config.densify:
        try:
            from gsplat.strategy import DefaultStrategy
            strategy = DefaultStrategy(
                verbose=False,
                refine_start_iter=int(config.refine_start),
                refine_stop_iter=int(min(config.refine_stop, config.steps)),
                refine_every=int(config.refine_every),
                reset_every=int(config.reset_every),
            )
            state = strategy.initialize_state(scene_scale=_scene_scale(cameras))
            strategy.check_sanity(params, optimizers)
        except Exception:
            strategy = None  # fall back to the plain loop on older gsplat

    gt = images if torch.is_tensor(images) else torch.from_numpy(np.asarray(images, np.float32))
    gt = gt.to(dev).float()
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
    viewmats = torch.from_numpy(np.ascontiguousarray(cameras.viewmats)).to(dev)
    Ks = torch.from_numpy(np.ascontiguousarray(cameras.Ks)).to(dev)
    n_views = min(len(cameras), gt.shape[0])
    if n_views == 0:
        raise ValueError("Splat Train: no matching images/cameras")

    last = 0.0
    for step in range(int(config.steps)):
        if interrupt is not None:
            interrupt()
        i = step % n_views
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        rc, _, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]), opacities=torch.sigmoid(params["opacities"]),
            colors=colors, viewmats=viewmats[i:i + 1], Ks=Ks[i:i + 1],
            width=cameras.width, height=cameras.height,
            sh_degree=int(config.sh_degree), render_mode="RGB",
        )
        if strategy is not None:
            strategy.step_pre_backward(params, optimizers, state, step, info)

        pred = rc[..., :3].clamp(0.0, 1.0)
        tgt = gt[i:i + 1]
        l1 = torch.abs(pred - tgt).mean()
        ssim = _ssim(pred.permute(0, 3, 1, 2), tgt.permute(0, 3, 1, 2))
        loss = (1.0 - config.ssim_lambda) * l1 + config.ssim_lambda * (1.0 - ssim)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if strategy is not None:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step()

        last = float(loss.item())
        if (preview is not None and config.preview_every > 0
                and (step + 1) % int(config.preview_every) == 0):
            try:  # never let a preview failure kill a training run
                preview(pred[0].detach().clamp(0.0, 1.0).cpu().numpy())
            except Exception:
                pass
        if progress is not None:
            progress(step + 1, int(config.steps), last)

    sh0 = params["sh0"].detach().cpu().numpy()
    shN = params["shN"].detach().cpu().numpy()
    sh = np.concatenate([sh0, shN], axis=1) if shN.shape[1] else sh0
    return Splat(
        params["means"].detach().cpu().numpy(), params["scales"].detach().cpu().numpy(),
        params["quats"].detach().cpu().numpy(), params["opacities"].detach().cpu().numpy(),
        sh, int(config.sh_degree),
        meta={"source": "train", "steps": int(config.steps), "final_loss": last,
              "densified": strategy is not None},
    ).validate()


__all__ = ["TrainConfig", "train", "HAS_GSPLAT"]
