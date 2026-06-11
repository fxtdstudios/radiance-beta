"""Read/write 3D Gaussian Splatting .ply files (binary_little_endian).

Self-contained (numpy only) so basic load/save works with no extra dependency.
The standard 3DGS vertex layout is supported: x,y,z, nx,ny,nz, f_dc_0..2,
f_rest_*, opacity, scale_0..2, rot_0..3. Higher-order SH (f_rest) uses the
channel-major flattening from the reference implementation.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from radiance.splatting.data import Splat

_PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


def load_ply(path: str) -> Splat:
    """Load a Gaussian Splatting .ply into a :class:`Splat`."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        count = 0
        props: List[Tuple[str, str]] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in PLY header")
            tok = line.strip().split()
            if not tok:
                continue
            if tok[0] == b"format":
                fmt = tok[1].decode()
            elif tok[0] == b"element" and tok[1] == b"vertex":
                count = int(tok[2])
            elif tok[0] == b"property":
                if tok[1] == b"list":
                    raise ValueError("List properties are not supported in splat .ply")
                props.append((tok[2].decode(), tok[1].decode()))
            elif tok[0] == b"end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format '{fmt}' (need binary_little_endian)")
        dtype = np.dtype([(n, _PLY_TYPES[t]) for n, t in props])
        buf = f.read(count * dtype.itemsize)
    data = np.frombuffer(buf, dtype=dtype, count=count)
    return _splat_from_struct(data, source=os.path.basename(path))


def _splat_from_struct(data: np.ndarray, source: str = "") -> Splat:
    names = set(data.dtype.names or ())
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = required - names
    if missing:
        raise ValueError(f"PLY missing Gaussian-splat properties: {sorted(missing)}")

    means = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32)
    quats = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32)
    opac = data["opacity"].astype(np.float32)
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1).astype(np.float32)

    rest_names = sorted(
        (n for n in data.dtype.names if n.startswith("f_rest_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_names:
        rest = np.stack([data[n] for n in rest_names], axis=1).astype(np.float32)  # (N, 3*(K-1))
        if rest.shape[1] % 3:
            raise ValueError(f"f_rest count {rest.shape[1]} is not a multiple of 3")
        krest = rest.shape[1] // 3
        rest = rest.reshape(-1, 3, krest).transpose(0, 2, 1)  # (N, K-1, 3)
        sh = np.concatenate([f_dc[:, None, :], rest], axis=1)  # (N, K, 3)
    else:
        sh = f_dc[:, None, :]

    k = sh.shape[1]
    sh_degree = int(round(k ** 0.5)) - 1
    return Splat(means, scales, quats, opac, sh, sh_degree, meta={"source": source}).validate()


def save_ply(splat: Splat, path: str) -> str:
    """Write a :class:`Splat` to a binary 3DGS .ply. Returns the path."""
    splat.validate()
    n = splat.count
    k = splat.sh_coeffs
    f_dc = splat.sh[:, 0, :]

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
              ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
    rest_flat = None
    if k > 1:
        rest = splat.sh[:, 1:, :]                       # (N, K-1, 3)
        rest_flat = rest.transpose(0, 2, 1).reshape(n, -1)  # channel-major (N, 3*(K-1))
        fields += [(f"f_rest_{i}", "<f4") for i in range(rest_flat.shape[1])]
    fields += [("opacity", "<f4"),
               ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
               ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")]

    arr = np.zeros(n, dtype=np.dtype(fields))
    arr["x"], arr["y"], arr["z"] = splat.means.T
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc.T
    if rest_flat is not None:
        for i in range(rest_flat.shape[1]):
            arr[f"f_rest_{i}"] = rest_flat[:, i]
    arr["opacity"] = splat.opacities
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = splat.scales.T
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = splat.quats.T

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name, _ in fields]
    header.append("end_header\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header)).encode("ascii"))
        f.write(arr.tobytes())
    return path


__all__ = ["load_ply", "save_ply"]
