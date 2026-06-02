import os
import numpy as np
import torch
import logging
from functools import lru_cache

logger = logging.getLogger("radiance.lut")

MAX_LUT_SIZE = 65
MAX_ENTRIES = MAX_LUT_SIZE ** 3 + 100

@lru_cache(maxsize=4)
def _parse_cube_cached(path: str, _mtime: float, _size: int):
    size = None
    is_3d = True
    entries = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                is_3d = True
            elif line.upper().startswith("LUT_1D_SIZE"):
                size = int(line.split()[-1])
                is_3d = False
            elif line.upper().startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX")):
                continue
            else:
                try:
                    vals = list(map(float, line.split()))
                    if len(vals) == 3:
                        entries.append(vals)
                        if len(entries) > MAX_ENTRIES:
                            raise ValueError(f"LUT file too large: {len(entries)} entries (max {MAX_ENTRIES})")
                except ValueError:
                    continue

    if size is None:
        size = int(round(len(entries) ** (1/3))) if is_3d else len(entries)

    table = np.array(entries, dtype=np.float32)
    if is_3d:
        # QA-004 FIX: reshape gives [R,G,B] order natively - no transpose needed.
        table = table.reshape(size, size, size, 3)
    
    return table, size, is_3d

def parse_cube(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LUT file not found: {path}")
    st = os.stat(path)
    return _parse_cube_cached(path, st.st_mtime, st.st_size)

def apply_3d_lut(img: np.ndarray, table: np.ndarray, size: int) -> np.ndarray:
    """
    Vectorized tetrahedral interpolation for 3D LUT over arbitrary batch dims.
    img: (..., 3) float32 in [0, 1].
    table: (S, S, S, 3) float32.
    """
    s1 = size - 1
    
    r = np.clip(img[..., 0], 0.0, 1.0) * s1
    g = np.clip(img[..., 1], 0.0, 1.0) * s1
    b = np.clip(img[..., 2], 0.0, 1.0) * s1

    r0 = np.floor(r).astype(np.int32).clip(0, s1 - 1)
    g0 = np.floor(g).astype(np.int32).clip(0, s1 - 1)
    b0 = np.floor(b).astype(np.int32).clip(0, s1 - 1)
    r1 = (r0 + 1).clip(0, s1)
    g1 = (g0 + 1).clip(0, s1)
    b1 = (b0 + 1).clip(0, s1)

    dr = (r - r0)[..., np.newaxis]
    dg = (g - g0)[..., np.newaxis]
    db = (b - b0)[..., np.newaxis]

    c000 = table[r0, g0, b0]
    c100 = table[r1, g0, b0]
    c010 = table[r0, g1, b0]
    c110 = table[r1, g1, b0]
    c001 = table[r0, g0, b1]
    c101 = table[r1, g0, b1]
    c011 = table[r0, g1, b1]
    c111 = table[r1, g1, b1]

    mask_rg = dr >= dg
    mask_gb = dg >= db
    mask_rb = dr >= db

    out = np.where(
        mask_rg & mask_gb,
        c000 + dr * (c100 - c000) + dg * (c110 - c100) + db * (c111 - c110),
        np.where(
            mask_rg & mask_rb,
            c000 + dr * (c100 - c000) + db * (c101 - c100) + dg * (c111 - c101),
            np.where(
                mask_rb & ~mask_rg,
                c000 + db * (c001 - c000) + dr * (c101 - c001) + dg * (c111 - c101),
                np.where(
                    ~mask_rb & ~mask_gb,
                    c000 + db * (c001 - c000) + dg * (c011 - c001) + dr * (c111 - c011),
                    np.where(
                        ~mask_rg & mask_gb,
                        c000 + dg * (c010 - c000) + db * (c011 - c010) + dr * (c111 - c011),
                        c000 + dg * (c010 - c000) + dr * (c110 - c010) + db * (c111 - c110),
                    ),
                ),
            ),
        ),
    ).astype(np.float32)

    return out

def apply_1d_lut(img: np.ndarray, table: np.ndarray, size: int) -> np.ndarray:
    out = np.empty_like(img)
    s1 = size - 1
    for c in range(3):
        x = np.clip(img[..., c], 0.0, 1.0) * s1
        x0 = np.floor(x).astype(np.int32).clip(0, s1 - 1)
        x1 = (x0 + 1).clip(0, s1)
        t = (x - x0)
        out[..., c] = table[x0, c] * (1.0 - t) + table[x1, c] * t
    return out
