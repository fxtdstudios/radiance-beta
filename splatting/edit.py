"""Edit operations on a SPLAT — transform, crop, merge. Pure-numpy / CI-testable."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from radiance.splatting.data import Splat


def _euler_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """XYZ Euler degrees -> unit quaternion (w, x, y, z)."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, cy, cz = np.cos([rx / 2, ry / 2, rz / 2])
    sx, sy, sz = np.sin([rx / 2, ry / 2, rz / 2])
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], np.float32)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float32)


def _quat_mul_batch(q: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Left-multiply each row of Q (N,4) by quaternion q (4,). wxyz."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    return np.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], axis=1).astype(np.float32)


def transform(splat: Splat, translate: Sequence[float] = (0, 0, 0),
              rotate_euler: Sequence[float] = (0, 0, 0), scale: float = 1.0) -> Splat:
    """Rigid+uniform-scale transform: rotate, uniformly scale, then translate."""
    scale = float(scale)
    q = _euler_to_quat(*rotate_euler)
    R = _quat_to_R(q)
    means = (splat.means * scale) @ R.T + np.asarray(translate, np.float32)
    quats = _quat_mul_batch(q, splat.quats)
    scales = (splat.scales + np.log(max(scale, 1e-8))).astype(np.float32)
    return Splat(means.astype(np.float32), scales, quats, splat.opacities.copy(),
                 splat.sh.copy(), splat.sh_degree,
                 meta={**splat.meta, "edited": "transform"}).validate()


def crop(splat: Splat, aabb_min: Sequence[float], aabb_max: Sequence[float]) -> Splat:
    """Keep only gaussians whose centres lie inside the axis-aligned box."""
    lo = np.asarray(aabb_min, np.float32)
    hi = np.asarray(aabb_max, np.float32)
    m = splat.means
    mask = np.all((m >= lo) & (m <= hi), axis=1)
    if not mask.any():
        raise ValueError("crop removed all gaussians (empty box)")
    return Splat(m[mask], splat.scales[mask], splat.quats[mask], splat.opacities[mask],
                 splat.sh[mask], splat.sh_degree,
                 meta={**splat.meta, "edited": "crop"}).validate()


def merge(a: Splat, b: Splat) -> Splat:
    """Concatenate two splats; SH is padded to the higher degree."""
    deg = max(a.sh_degree, b.sh_degree)
    K = (deg + 1) ** 2

    def pad(s: Splat) -> np.ndarray:
        if s.sh_coeffs == K:
            return s.sh
        out = np.zeros((s.count, K, 3), np.float32)
        out[:, :s.sh_coeffs, :] = s.sh
        return out

    return Splat(
        np.concatenate([a.means, b.means]),
        np.concatenate([a.scales, b.scales]),
        np.concatenate([a.quats, b.quats]),
        np.concatenate([a.opacities, b.opacities]),
        np.concatenate([pad(a), pad(b)]),
        deg, meta={"source": "merge"},
    ).validate()


__all__ = ["transform", "crop", "merge"]
