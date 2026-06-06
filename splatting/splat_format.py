"""Read/write the web '.splat' format (32 bytes/gaussian: pos, scale, rgba, rot).

Lossy relative to .ply (colour/opacity/rotation are 8-bit, only the SH DC term is
kept), but it is the common interchange format for web viewers. Pure-numpy.
"""
from __future__ import annotations

import os

import numpy as np

from radiance.splatting.data import Splat

_SH_C0 = 0.28209479177387814
_DTYPE = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3),
                   ("rgba", "u1", 4), ("rot", "u1", 4)])


def save_splat(splat: Splat, path: str) -> str:
    splat.validate()
    n = splat.count
    color = np.clip(_SH_C0 * splat.sh[:, 0, :] + 0.5, 0.0, 1.0)
    opacity = 1.0 / (1.0 + np.exp(-splat.opacities))  # sigmoid(logit)
    q = splat.quats / (np.linalg.norm(splat.quats, axis=1, keepdims=True) + 1e-9)

    arr = np.zeros(n, dtype=_DTYPE)
    arr["pos"] = splat.means.astype("<f4")
    arr["scale"] = np.exp(splat.scales).astype("<f4")
    arr["rgba"][:, :3] = np.round(color * 255).astype(np.uint8)
    arr["rgba"][:, 3] = np.round(opacity * 255).astype(np.uint8)
    arr["rot"] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    return path


def load_splat(path: str) -> Splat:
    data = np.fromfile(path, dtype=_DTYPE)
    if data.size == 0:
        raise ValueError(f"Empty or invalid .splat file: {path}")
    means = data["pos"].astype(np.float32)
    scales = np.log(np.maximum(data["scale"].astype(np.float32), 1e-9))
    rgba = data["rgba"].astype(np.float32)
    sh = ((rgba[:, :3] / 255.0 - 0.5) / _SH_C0)[:, None, :].astype(np.float32)
    op = np.clip(rgba[:, 3] / 255.0, 1e-6, 1 - 1e-6)
    opac = np.log(op / (1 - op)).astype(np.float32)
    quats = (data["rot"].astype(np.float32) - 128.0) / 128.0
    quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9)
    return Splat(means, scales, quats, opac, sh, 0,
                 meta={"source": os.path.basename(path), "format": "splat"}).validate()


__all__ = ["save_splat", "load_splat"]
