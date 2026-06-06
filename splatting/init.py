"""Initialize a SPLAT from a sparse point cloud (the standard 3DGS init).

Pure-numpy / CI-testable. Mirrors the reference initialization: SH DC term from
RGB, identity rotations, logit opacity, and per-point log-scale from the local
nearest-neighbour spacing.
"""
from __future__ import annotations

import numpy as np

from radiance.splatting.data import Splat

_SH_C0 = 0.28209479177387814  # 0th-order SH basis value


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, np.float32) - 0.5) / _SH_C0


def inverse_sigmoid(x: float) -> float:
    x = float(np.clip(x, 1e-6, 1.0 - 1e-6))
    return float(np.log(x / (1.0 - x)))


def _mean_nn_dist(points: np.ndarray, k: int = 3, cap: int = 20000) -> np.ndarray:
    n = points.shape[0]
    if n <= 1:
        return np.full(n, 0.01, np.float32)
    if n > cap:  # avoid O(N^2); coarse estimate from scene extent
        ext = float(np.linalg.norm(points.max(0) - points.min(0)))
        return np.full(n, max(ext / (n ** (1 / 3) + 1) * 0.5, 1e-4), np.float32)
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    kk = min(k, n - 1)
    nn = np.sort(d2, axis=1)[:, :kk]
    return np.sqrt(nn.mean(axis=1)).astype(np.float32)


def init_from_points(points, colors=None, sh_degree: int = 3,
                     init_opacity: float = 0.1) -> Splat:
    """Build an initial SPLAT from sparse points (+ optional 0..1 or 0..255 colors)."""
    points = np.asarray(points, np.float32).reshape(-1, 3)
    n = points.shape[0]
    if n == 0:
        raise ValueError("init_from_points: empty point cloud")
    if colors is None:
        colors = np.full((n, 3), 0.5, np.float32)
    colors = np.asarray(colors, np.float32).reshape(-1, 3)
    if colors.max() > 1.0 + 1e-3:
        colors = colors / 255.0

    k = (sh_degree + 1) ** 2
    sh = np.zeros((n, k, 3), np.float32)
    sh[:, 0, :] = rgb_to_sh(colors)

    scales = np.log(np.maximum(_mean_nn_dist(points), 1e-7))
    scales = scales[:, None].repeat(3, axis=1).astype(np.float32)
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1.0  # identity (w, x, y, z)
    opac = np.full(n, inverse_sigmoid(init_opacity), np.float32)

    return Splat(points, scales, quats, opac, sh, sh_degree,
                 meta={"source": "init_from_points", "init": True}).validate()


__all__ = ["init_from_points", "rgb_to_sh", "inverse_sigmoid"]
