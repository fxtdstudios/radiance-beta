"""The SPLAT container — a 3D Gaussian Splatting scene.

Pure-numpy, GPU-free, so it imports anywhere. Render/train phases convert these
arrays to torch tensors on demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class Splat:
    """A set of 3D Gaussians.

    means      (N, 3)  float32  centre positions
    scales     (N, 3)  float32  log-scale (as stored in 3DGS .ply)
    quats      (N, 4)  float32  rotation quaternion (rot_0..3)
    opacities  (N,)    float32  logit opacity (pre-sigmoid)
    sh         (N, K, 3) float32  spherical-harmonic colour; index 0 is the DC
                                  term. K = (sh_degree + 1) ** 2.
    """

    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray
    sh_degree: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32).reshape(-1, 3)
        self.scales = np.asarray(self.scales, dtype=np.float32).reshape(-1, 3)
        self.quats = np.asarray(self.quats, dtype=np.float32).reshape(-1, 4)
        self.opacities = np.asarray(self.opacities, dtype=np.float32).reshape(-1)
        sh = np.asarray(self.sh, dtype=np.float32)
        if sh.ndim == 2:                       # (N, 3) DC-only convenience
            sh = sh.reshape(sh.shape[0], 1, 3)
        self.sh = sh

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_coeffs(self) -> int:
        return int(self.sh.shape[1])

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            z = np.zeros(3, np.float32)
            return z, z
        return self.means.min(axis=0), self.means.max(axis=0)

    def validate(self) -> "Splat":
        n = self.count
        if self.scales.shape != (n, 3):
            raise ValueError(f"SPLAT scales shape {self.scales.shape} != ({n}, 3)")
        if self.quats.shape != (n, 4):
            raise ValueError(f"SPLAT quats shape {self.quats.shape} != ({n}, 4)")
        if self.opacities.shape != (n,):
            raise ValueError(f"SPLAT opacities shape {self.opacities.shape} != ({n},)")
        if self.sh.shape[0] != n or self.sh.shape[2] != 3:
            raise ValueError(f"SPLAT sh shape {self.sh.shape} invalid for N={n}")
        expect = (self.sh_degree + 1) ** 2
        if self.sh_coeffs != expect:
            raise ValueError(
                f"SPLAT sh has {self.sh_coeffs} coeffs but degree {self.sh_degree} "
                f"expects {expect}"
            )
        return self

    def info(self) -> str:
        lo, hi = self.aabb()
        size = hi - lo
        op = self.opacities
        omin = float(op.min()) if self.count else 0.0
        omax = float(op.max()) if self.count else 0.0
        return (
            "Gaussian Splat\n"
            f"  points     : {self.count:,}\n"
            f"  SH degree  : {self.sh_degree}  ({self.sh_coeffs} coeffs/channel)\n"
            f"  bounds min : [{lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f}]\n"
            f"  bounds max : [{hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f}]\n"
            f"  size       : [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]\n"
            f"  opacity    : min {omin:.3f}  max {omax:.3f}  (logit)\n"
            f"  source     : {self.meta.get('source', '-')}"
        )


__all__ = ["Splat"]
