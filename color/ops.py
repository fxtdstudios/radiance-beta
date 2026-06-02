"""
color/ops.py — Shared GPU-accelerated colour math primitives (torch).

This module is the **single source of truth** for:
  - 3×3 matrix application to image tensors
  - PQ (ST.2084) and HLG (ARIB STD-B67) encoding constants
  - BT.2408 reference-white PQ encode/decode (scene-linear ↔ PQ signal)
  - HLG encode (scene-linear → signal)
  - Soft-knee Reinhard compression for VAE ingestion
  - Authoritative torch primary-colour matrices (matching color/matrices.py values)

Why this module exists
──────────────────────
Prior to this file, PQ constants (_PQ_M1, _PQ_M2 …), the BT.2020 matrix tensor,
and soft_knee_compress() were each duplicated in nodes_hdr_delivery.py,
nodes_hdr_colorspace.py, nodes_hdr_uplift.py, and nodes/color/colorspace.py —
sometimes with subtly different precision.  Any node file that needs these
primitives should import from here instead of redefining them.

Usage
──────
    from radiance.color.ops import (
        apply_matrix_3x3,
        linear_to_pq_bt2408,
        pq_bt2408_to_linear,
        linear_to_hlg,
        soft_knee_compress,
        M_REC709_TO_BT2020,
        M_REC709_TO_ACESCG,
    )
"""
from __future__ import annotations

import torch

# ─────────────────────────────────────────────────────────────────────────────
# PQ (ST.2084) constants — SMPTE ST.2084-2014 §5
# ─────────────────────────────────────────────────────────────────────────────

#: 2610 / 4096 / 4
PQ_M1: float = 0.1593017578125
#: 2523 / 4096 * 128
PQ_M2: float = 78.84375
#: 3424 / 4096
PQ_C1: float = 0.8359375
#: 2413 / 4096 * 32
PQ_C2: float = 18.8515625
#: 2392 / 4096 * 32
PQ_C3: float = 18.6875

#: BT.2408 reference-white luminance: scene-linear 1.0 ≡ 203 cd/m².
#: Used by the delivery and colorspace nodes when working in scene-linear.
PQ_REF_WHITE_NITS: float = 203.0


# ─────────────────────────────────────────────────────────────────────────────
# HLG (ARIB STD-B67) constants
# ─────────────────────────────────────────────────────────────────────────────

HLG_A: float = 0.17883277
HLG_B: float = 0.28466892
HLG_C: float = 0.55991073


# ─────────────────────────────────────────────────────────────────────────────
# Authoritative torch primary-colour matrices
# Values are kept consistent with color/matrices.py (numpy reference).
# ─────────────────────────────────────────────────────────────────────────────

# Rec.709 (sRGB D65) → BT.2020 — Bradford-adapted, D65
# Source: ITU-R BT.2020 / IEC 61966-2-1
M_REC709_TO_BT2020 = torch.tensor([
    [ 0.6274040,  0.3292820,  0.0433136],
    [ 0.0690970,  0.9195400,  0.0113612],
    [ 0.0163916,  0.0880132,  0.8955950],
], dtype=torch.float32)

M_BT2020_TO_REC709 = torch.tensor([
    [ 1.6604910, -0.5876411, -0.0728499],
    [-0.1245505,  1.1328999, -0.0083494],
    [-0.0181508, -0.1005789,  1.1187297],
], dtype=torch.float32)

# Rec.709 (sRGB) → ACEScg (AP1) — Bradford D65→D60 adapted
# Matches color/matrices.py SRGB_TO_ACESCG for cross-consistency.
M_REC709_TO_ACESCG = torch.tensor([
    [0.613097, 0.339523, 0.047379],
    [0.070194, 0.916354, 0.013452],
    [0.020616, 0.109570, 0.869815],
], dtype=torch.float32)

M_ACESCG_TO_REC709 = torch.tensor([
    [ 1.7050509, -0.6217921, -0.0832588],
    [-0.1302564,  1.1408047, -0.0105483],
    [-0.0240033, -0.1289690,  1.1529723],
], dtype=torch.float32)

# DCI-P3 (D65) → BT.2020
M_P3D65_TO_BT2020 = torch.tensor([
    [ 0.7539397,  0.1986815,  0.0473788],
    [ 0.0458697,  0.9416312,  0.0124991],
    [-0.0011204,  0.0176004,  0.9835200],
], dtype=torch.float32)

# ACEScg (AP1) → XYZ (D60)
M_ACESCG_TO_XYZ_D60 = torch.tensor([
    [ 0.6624541811,  0.1340042065,  0.1561876744],
    [ 0.2722287168,  0.6740817658,  0.0536895174],
    [-0.0055746495,  0.0040607335,  1.0103391003],
], dtype=torch.float32)

# XYZ (D60) → ACEScg (AP1)
M_XYZ_D60_TO_ACESCG = torch.tensor([
    [ 1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587,  1.6153315917,  0.0167563477],
    [ 0.0117218943, -0.0082844420,  0.9883948585],
], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix application
# ─────────────────────────────────────────────────────────────────────────────

def apply_matrix_3x3(
    img: torch.Tensor,
    matrix: torch.Tensor | list,
) -> torch.Tensor:
    """
    Apply a 3×3 colour matrix to an image tensor.

    Handles (B, H, W, 3), (H, W, 3), or any shape ending in 3.
    Row-vector convention: rgb_out = rgb_in @ M.T

    Args:
        img:    Input image tensor whose last dimension is 3.
        matrix: A 3×3 torch.Tensor or nested Python list.

    Returns:
        Tensor with the same shape as ``img``.
    """
    mat = (
        torch.tensor(matrix, dtype=img.dtype, device=img.device)
        if not isinstance(matrix, torch.Tensor)
        else matrix.to(dtype=img.dtype, device=img.device)
    )
    return (img.reshape(-1, 3) @ mat.T).reshape(img.shape)


# ─────────────────────────────────────────────────────────────────────────────
# PQ encode/decode — BT.2408 reference-white model
# ─────────────────────────────────────────────────────────────────────────────

def linear_to_pq_bt2408(
    y: torch.Tensor,
    peak_nits: float = 1000.0,
) -> torch.Tensor:
    """
    Scene-linear → ST.2084 PQ signal, BT.2408 reference-white model.

    Scene-linear 1.0 = 203 cd/m² (PQ_REF_WHITE_NITS).
    The signal is clipped to [0, 1] before encoding, so values above
    ``peak_nits / 203`` are clipped to white.

    Args:
        y:          Scene-linear tensor, non-negative.
        peak_nits:  Mastering-display peak luminance in cd/m².

    Returns:
        PQ-encoded signal in [0, 1].
    """
    # Normalise scene-linear to absolute display luminance, then to [0, 1].
    L = (y.clamp(min=0.0) * PQ_REF_WHITE_NITS / peak_nits).clamp(0.0, 1.0)
    Lm1 = L ** PQ_M1
    return ((PQ_C1 + PQ_C2 * Lm1) / (1.0 + PQ_C3 * Lm1)) ** PQ_M2


def pq_bt2408_to_linear(
    v: torch.Tensor,
    peak_nits: float = 1000.0,
) -> torch.Tensor:
    """
    ST.2084 PQ signal → scene-linear, BT.2408 reference-white model.

    Inverse of :func:`linear_to_pq_bt2408`.

    Args:
        v:          PQ-encoded signal in [0, 1].
        peak_nits:  Mastering-display peak luminance in cd/m².

    Returns:
        Scene-linear values (1.0 = 203 cd/m²).  May exceed 1.0 for HDR.
    """
    Vm2 = v.clamp(min=0.0) ** (1.0 / PQ_M2)
    L = (
        (Vm2 - PQ_C1).clamp(min=0.0)
        / (PQ_C2 - PQ_C3 * Vm2).clamp(min=1e-7)
    ) ** (1.0 / PQ_M1)
    return L * (peak_nits / PQ_REF_WHITE_NITS)


# ─────────────────────────────────────────────────────────────────────────────
# HLG encode — ARIB STD-B67
# ─────────────────────────────────────────────────────────────────────────────

def linear_to_hlg(y: torch.Tensor) -> torch.Tensor:
    """
    Scene-linear [0, 12] → HLG signal [0, 1].

    Reference: ARIB STD-B67 §3.5
    Scene-linear range is [0, 12] where 1.0 is reference white.

    Args:
        y:  Scene-linear tensor, non-negative.

    Returns:
        HLG-encoded signal in [0, 1].
    """
    y_safe = y.clamp(min=0.0)
    lin_segment = torch.sqrt(3.0 * y_safe)
    log_segment = HLG_A * torch.log(12.0 * y_safe - HLG_B + 1e-7) + HLG_C
    return torch.where(y_safe <= 1.0 / 12.0, lin_segment, log_segment).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# VAE soft-knee compression
# ─────────────────────────────────────────────────────────────────────────────

def soft_knee_compress(img: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Soft-knee Reinhard blend — compresses scene-linear HDR into [0, 1] for VAE.

    Allows the model to "see" highlight structure above 1.0 without hard clipping.

    Args:
        img:   Scene-linear tensor (values may exceed 1.0).
        ratio: Blend weight in [0, 1].
                 0.0  →  hard clamp to [0, 1]  (no compression)
                 1.0  →  full Reinhard: x / (1 + x)
                 0–1  →  interpolated blend between the two

    Returns:
        Compressed tensor in [0, 1].
    """
    img_pos  = img.clamp(min=0.0)
    reinhard = img_pos / (1.0 + img_pos)
    clamped  = img_pos.clamp(max=1.0)
    if ratio <= 0.0:
        return clamped
    if ratio >= 1.0:
        return reinhard
    return torch.lerp(clamped, reinhard, ratio)


__all__ = [
    # PQ constants
    "PQ_M1", "PQ_M2", "PQ_C1", "PQ_C2", "PQ_C3", "PQ_REF_WHITE_NITS",
    # HLG constants
    "HLG_A", "HLG_B", "HLG_C",
    # Torch matrices
    "M_REC709_TO_BT2020", "M_BT2020_TO_REC709",
    "M_REC709_TO_ACESCG", "M_ACESCG_TO_REC709",
    "M_P3D65_TO_BT2020",
    "M_ACESCG_TO_XYZ_D60", "M_XYZ_D60_TO_ACESCG",
    # Functions
    "apply_matrix_3x3",
    "linear_to_pq_bt2408", "pq_bt2408_to_linear",
    "linear_to_hlg",
    "soft_knee_compress",
]
