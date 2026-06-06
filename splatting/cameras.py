"""Camera rig for splat rendering — intrinsics + world-to-camera view matrices.

Pure-numpy and CI-testable. OpenCV camera convention (x right, y down, z forward),
which matches what gsplat's rasterizer expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class Cameras:
    """A batch of cameras. viewmats: (B,4,4) world->camera; Ks: (B,3,3) intrinsics."""

    viewmats: np.ndarray
    Ks: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        self.viewmats = np.asarray(self.viewmats, dtype=np.float32).reshape(-1, 4, 4)
        self.Ks = np.asarray(self.Ks, dtype=np.float32).reshape(-1, 3, 3)
        self.width = int(self.width)
        self.height = int(self.height)

    def __len__(self) -> int:
        return int(self.viewmats.shape[0])

    def centers(self) -> np.ndarray:
        """Camera positions in world space (B,3)."""
        R = self.viewmats[:, :3, :3]
        t = self.viewmats[:, :3, 3]
        return -np.einsum("bij,bj->bi", np.transpose(R, (0, 2, 1)), t)


def look_at(eye: Sequence[float], target: Sequence[float],
            up: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """World->camera 4x4 view matrix (OpenCV convention)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - eye                      # forward (+z in camera space)
    nz = np.linalg.norm(z)
    z = z / nz if nz > 1e-12 else np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)                    # right (+x)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)                     # down (+y, OpenCV)

    R_c2w = np.stack([x, y, z], axis=1)    # columns are camera axes in world
    R = R_c2w.T                            # world->camera rotation
    t = -R @ eye
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = R
    view[:3, 3] = t
    return view.astype(np.float32)


def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """3x3 pinhole intrinsics from a vertical field of view."""
    f = 0.5 * height / np.tan(0.5 * np.radians(fov_deg))
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def orbit(num_frames: int = 60, radius: float = 3.0, elevation_deg: float = 15.0,
          center: Sequence[float] = (0.0, 0.0, 0.0), fov_deg: float = 50.0,
          width: int = 960, height: int = 540,
          up: Sequence[float] = (0.0, 1.0, 0.0)) -> Cameras:
    """Generate a horizontal orbit of cameras looking at `center`."""
    num_frames = max(1, int(num_frames))
    center = np.asarray(center, dtype=np.float64)
    el = np.radians(elevation_deg)
    K = intrinsics(width, height, fov_deg)
    views, Ks = [], []
    for az in np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False):
        eye = center + radius * np.array([
            np.cos(el) * np.cos(az),
            np.sin(el),
            np.cos(el) * np.sin(az),
        ])
        views.append(look_at(eye, center, up))
        Ks.append(K)
    return Cameras(np.stack(views, 0), np.stack(Ks, 0), width, height)


__all__ = ["Cameras", "look_at", "intrinsics", "orbit"]
