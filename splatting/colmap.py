"""Read a COLMAP sparse reconstruction into Radiance camera + point data.

Supports both text (cameras.txt / images.txt / points3D.txt) and binary
(.bin) models — the two formats COLMAP emits. Pure-Python (numpy + struct),
so it runs in CI with no GPU or COLMAP install. The output feeds Splat Train.
"""
from __future__ import annotations

import os
import struct
from typing import Dict, Tuple

import numpy as np

from radiance.splatting.cameras import Cameras

# COLMAP camera model id -> (name, num_params)
_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
_MODEL_BY_NAME = {name: (mid, n) for mid, (name, n) in _CAMERA_MODELS.items()}


def _K_from_params(model: str, params) -> np.ndarray:
    p = list(params)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                 "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    else:  # PINHOLE, OPENCV, FULL_OPENCV, *_FISHEYE with 4 leading params
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)


def _quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], np.float32)


# ── text format ──────────────────────────────────────────────────────────────
def _read_cameras_txt(path):
    cams = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t = line.split()
        cid = int(t[0]); model = t[1]; w = int(t[2]); h = int(t[3])
        params = [float(x) for x in t[4:]]
        cams[cid] = (model, w, h, params)
    return cams


def _read_images_txt(path):
    out = []
    lines = [ln for ln in open(path)]
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if not ln or ln.startswith("#"):
            continue
        t = ln.split()
        qw, qx, qy, qz = map(float, t[1:5])
        tx, ty, tz = map(float, t[5:8])
        cam_id = int(t[8]); name = t[9] if len(t) > 9 else ""
        i += 1  # skip the POINTS2D line
        out.append((qw, qx, qy, qz, tx, ty, tz, cam_id, name))
    return out


def _read_points3D_txt(path):
    pts, cols = [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t = line.split()
        pts.append([float(t[1]), float(t[2]), float(t[3])])
        cols.append([int(t[4]), int(t[5]), int(t[6])])
    return np.array(pts, np.float32).reshape(-1, 3), np.array(cols, np.float32).reshape(-1, 3)


# ── binary format ────────────────────────────────────────────────────────────
def _read_next(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))


def _read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        num = _read_next(f, "<Q")[0]
        for _ in range(num):
            cid, model_id, w, h = _read_next(f, "<iiQQ")
            name, npar = _CAMERA_MODELS.get(model_id, ("PINHOLE", 4))
            params = _read_next(f, "<" + "d" * npar)
            cams[cid] = (name, w, h, list(params))
    return cams


def _read_images_bin(path):
    out = []
    with open(path, "rb") as f:
        num = _read_next(f, "<Q")[0]
        for _ in range(num):
            _id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read_next(f, "<idddddddi")
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            npts = _read_next(f, "<Q")[0]
            f.read(npts * struct.calcsize("<ddq"))  # skip 2D points
            out.append((qw, qx, qy, qz, tx, ty, tz, cam_id, name.decode("utf-8", "replace")))
    return out


def _read_points3D_bin(path):
    pts, cols = [], []
    with open(path, "rb") as f:
        num = _read_next(f, "<Q")[0]
        for _ in range(num):
            _pid, x, y, z, r, g, b, _err = _read_next(f, "<QdddBBBd")
            tl = _read_next(f, "<Q")[0]
            f.read(tl * struct.calcsize("<ii"))
            pts.append([x, y, z]); cols.append([r, g, b])
    return np.array(pts, np.float32).reshape(-1, 3), np.array(cols, np.float32).reshape(-1, 3)


def load_colmap(model_dir: str) -> Tuple[Cameras, np.ndarray, np.ndarray]:
    """Load a COLMAP sparse model dir -> (Cameras, points (M,3), colors (M,3) 0..255)."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"COLMAP model dir not found: {model_dir}")
    has_bin = os.path.isfile(os.path.join(model_dir, "cameras.bin"))
    if has_bin:
        cams = _read_cameras_bin(os.path.join(model_dir, "cameras.bin"))
        imgs = _read_images_bin(os.path.join(model_dir, "images.bin"))
        pts, cols = _read_points3D_bin(os.path.join(model_dir, "points3D.bin"))
    elif os.path.isfile(os.path.join(model_dir, "cameras.txt")):
        cams = _read_cameras_txt(os.path.join(model_dir, "cameras.txt"))
        imgs = _read_images_txt(os.path.join(model_dir, "images.txt"))
        pts, cols = _read_points3D_txt(os.path.join(model_dir, "points3D.txt"))
    else:
        raise FileNotFoundError(f"No cameras.bin or cameras.txt in {model_dir}")

    if not imgs:
        raise ValueError("COLMAP model has no images")
    viewmats, Ks = [], []
    width = height = 0
    for (qw, qx, qy, qz, tx, ty, tz, cam_id, _name) in imgs:
        model, w, h, params = cams[cam_id]
        width, height = w, h
        v = np.eye(4, dtype=np.float32)
        v[:3, :3] = _quat_to_R(qw, qx, qy, qz)
        v[:3, 3] = [tx, ty, tz]
        viewmats.append(v)
        Ks.append(_K_from_params(model, params))
    cameras = Cameras(np.stack(viewmats, 0), np.stack(Ks, 0), width, height)
    return cameras, pts, cols


__all__ = ["load_colmap"]
