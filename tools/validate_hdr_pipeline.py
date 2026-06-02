#!/usr/bin/env python3
"""
tools/validate_hdr_pipeline.py
================================
Real-world HDR pipeline validation tool for Radiance / radiance_color.

Tests four critical aspects of a production HDR color pipeline:

  1. ARRI ALEXA sensor simulation
     14-stop exposure ramp with photon shot noise.  Per-stop SNR is measured
     over 512 synthetic pixels per stop.  Midtone SNR must be > 30 dB.

  2. Macbeth ColorChecker round-trip chromatic accuracy
     24 Macbeth chart patches (D50 XYZ) through sRGB → ACEScg → sRGB.
     In-gamut patches must have ΔE00 < 1.0; neutrals must be < 0.5 ΔE00.
     Out-of-gamut patches are reported but not failed.

  3. 14-stop highlight rolloff
     Daniele Evo tonescale (1000 nit) must be monotone, must roll off
     smoothly above 2 stops over reference white with no oscillation.

  4. ACES 2.0 pipeline self-consistency regression
     LogC4 → ACEScg → reach-gamut-compress → Daniele Evo → sRGB against
     pre-computed reference values (from same pipeline, pinned at v1.0.0).
     Tolerance: 0.001 sRGB (sub-half-DN in 8-bit).

Usage:
    python tools/validate_hdr_pipeline.py
    python tools/validate_hdr_pipeline.py --output report.json
    python tools/validate_hdr_pipeline.py --verbose

Exit codes:
    0  All tests passed
    1  One or more tests failed
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import numpy as np

# ── ensure radiance_color is importable from the repo root ──────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import radiance_color as rc
from radiance_color import (
    linear_to_logc4, logc4_to_linear,
    linear_to_srgb, srgb_to_linear,
    apply_matrix,
    SRGB_TO_ACESCG, ACESCG_TO_SRGB,
    LUMA_BT709,
    DanieleEvoParams,
    daniele_evo_fwd,
    reach_gamut_compress,
)


# ═══════════════════════════════════════════════════════════════════════════
# § 0  Result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    name: str
    passed: bool
    metrics: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class ValidationReport:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    version: str
    timestamp: str
    radiance_color_version: str
    results: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# ═══════════════════════════════════════════════════════════════════════════
# § 1  ARRI ALEXA sensor simulation
# ═══════════════════════════════════════════════════════════════════════════

def _alexa_photon_noise_batch(
    scene_linear: np.ndarray,      # (S,) scene-linear values, one per stop
    n_pixels: int = 512,
    base_iso: int = 800,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate ARRI ALEXA photon shot noise.

    Returns (S, n_pixels) float32 array — `n_pixels` noisy realisations
    for each of the S scene-linear input values.

    Noise model: σ²(x) = x * k_gain + σ_read²

    k_gain is calibrated so that 18% grey (x=0.18) gives ~40 dB SNR, which
    matches ARRI ALEXA at EI 800 (≈ 100:1 signal-to-noise at midtones).
      k_gain = x_grey / SNR² = 0.18 / 10000 = 1.8e-5
    ISO scaling adjusts k proportionally (higher ISO → more gain → more noise).
    """
    rng     = np.random.default_rng(seed=seed)
    k_gain  = (800.0 / base_iso) * 1.8e-5
    sigma_r = 5e-5

    # Broadcast (S, 1) * (S, n_pixels)
    linear_tiled = scene_linear[:, np.newaxis] * np.ones((1, n_pixels), dtype=np.float32)
    var   = np.maximum(linear_tiled, 0.0) * k_gain + sigma_r ** 2
    noise = rng.normal(0.0, np.sqrt(var)).astype(np.float32)
    return np.clip(linear_tiled + noise, 0.0, None)


def test_alexa_sensor_simulation(verbose: bool = False) -> TestResult:
    """
    Test 1: ARRI ALEXA sensor — 14-stop exposure ramp + LogC4 round-trip.

    For each stop, 512 synthetic pixels with photon shot noise are generated.
    Per-stop SNR = mean(signal) / std(noise) in scene-linear space after
    LogC4 encode → decode.

    Pass criteria:
      - Clean round-trip PSNR > 80 dB (numerical precision)
      - Midtone SNR (stops −3 to +3) > 30 dB each
      - LogC4 encode is monotonically non-decreasing
      - 18% grey encodes to LogC4 value in [0.22, 0.35]
    """
    t0 = time.perf_counter()
    errors: List[str] = []
    metrics: dict = {}

    # 15-stop ramp: -7 … +7, centred at 18% grey
    stops        = np.arange(-7, 8, dtype=np.float32)          # (15,)
    scene_linear = (0.18 * 2.0 ** stops).astype(np.float32)    # (15,)

    # A) Clean round-trip PSNR (no noise)
    v_clean   = linear_to_logc4(scene_linear)
    decoded   = logc4_to_linear(v_clean)
    mse_clean = float(np.mean((scene_linear - decoded) ** 2))
    peak      = float(scene_linear.max())
    psnr_clean = 20.0 * math.log10(peak / math.sqrt(max(mse_clean, 1e-30)))
    metrics["logc4_roundtrip_psnr_db"] = round(psnr_clean, 2)
    if psnr_clean < 80.0:
        errors.append(f"LogC4 clean round-trip PSNR {psnr_clean:.1f} dB < 80 dB")

    # B) Noisy pixels → LogC4 → decoded; measure per-stop SNR
    noisy_batch  = _alexa_photon_noise_batch(scene_linear, n_pixels=512)  # (15, 512)
    noisy_v      = linear_to_logc4(noisy_batch)
    noisy_dec    = logc4_to_linear(noisy_v)                                # (15, 512)

    per_stop_snr: List[float] = []
    for i in range(len(stops)):
        sig_mean  = float(np.mean(noisy_dec[i]))
        noise_std = float(np.std(noisy_dec[i]))
        snr_db    = 20.0 * math.log10(sig_mean / max(noise_std, 1e-12))
        per_stop_snr.append(round(snr_db, 1))

    metrics["per_stop_snr_db"] = per_stop_snr

    # Midtone stops: indices 4 to 10 → stops -3 to +3
    mid_snr = np.array(per_stop_snr[4:11])
    metrics["midtone_min_snr_db"] = round(float(mid_snr.min()), 1)
    if np.any(mid_snr < 30.0):
        bad = [(int(stops[4+i]), per_stop_snr[4+i]) for i, v in enumerate(mid_snr) if v < 30.0]
        errors.append(f"Midtone SNR < 30 dB at stops {bad}")

    # C) Monotonicity of encode
    is_monotone = bool(np.all(np.diff(v_clean) >= -1e-7))
    metrics["logc4_monotone"] = is_monotone
    if not is_monotone:
        errors.append("LogC4 encode is not monotonically non-decreasing")

    # D) 18% grey value
    grey_v = float(linear_to_logc4(np.array([0.18], dtype=np.float32))[0])
    metrics["grey_18pct_logc4"] = round(grey_v, 4)
    if not (0.22 < grey_v < 0.35):
        errors.append(f"18% grey LogC4 value {grey_v:.4f} outside expected 0.22–0.35")

    return TestResult(
        name="ARRI ALEXA sensor simulation (LogC4 + shot noise)",
        passed=len(errors) == 0,
        metrics=metrics,
        errors=errors,
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# § 2  Macbeth ColorChecker chromatic accuracy
# ═══════════════════════════════════════════════════════════════════════════

# Macbeth ColorChecker Classic — D50 XYZ (CIE 2°), normalised to D50 white
# Source: BabelColor (https://babelcolor.com/colorchecker-2.htm)
_MACBETH_XYZ_D50 = np.array([
    [0.4002, 0.3502, 0.0493],   # 1  Dark Skin
    [0.5960, 0.5228, 0.1695],   # 2  Light Skin
    [0.2000, 0.2179, 0.4138],   # 3  Blue Sky
    [0.2360, 0.3182, 0.1111],   # 4  Foliage
    [0.3061, 0.2817, 0.5457],   # 5  Blue Flower
    [0.2416, 0.4053, 0.4523],   # 6  Bluish Green
    [0.6174, 0.4136, 0.0291],   # 7  Orange
    [0.1468, 0.1283, 0.5399],   # 8  Purplish Blue
    [0.4787, 0.2834, 0.1450],   # 9  Moderate Red
    [0.1559, 0.0905, 0.1950],   # 10 Purple
    [0.3672, 0.4435, 0.0926],   # 11 Yellow Green
    [0.6064, 0.4605, 0.0609],   # 12 Orange Yellow
    [0.0985, 0.0757, 0.4234],   # 13 Blue
    [0.2012, 0.3101, 0.0923],   # 14 Green
    [0.3913, 0.1523, 0.0522],   # 15 Red
    [0.6023, 0.5528, 0.0743],   # 16 Yellow
    [0.4254, 0.2220, 0.3562],   # 17 Magenta
    [0.0797, 0.1588, 0.4148],   # 18 Cyan
    [0.8785, 0.8188, 0.7059],   # 19 White  (9.5)
    [0.5717, 0.5335, 0.4609],   # 20 Neutral 8
    [0.3605, 0.3363, 0.2914],   # 21 Neutral 6.5
    [0.1955, 0.1817, 0.1575],   # 22 Neutral 5
    [0.0884, 0.0820, 0.0709],   # 23 Neutral 3.5
    [0.0346, 0.0316, 0.0267],   # 24 Black  (2)
], dtype=np.float64)

_MACBETH_NAMES = [
    "Dark Skin", "Light Skin", "Blue Sky", "Foliage", "Blue Flower",
    "Bluish Green", "Orange", "Purplish Blue", "Moderate Red", "Purple",
    "Yellow Green", "Orange Yellow", "Blue", "Green", "Red", "Yellow",
    "Magenta", "Cyan", "White", "Neutral 8", "Neutral 6.5",
    "Neutral 5", "Neutral 3.5", "Black",
]

# Neutral patches: indices 18-23 (White through Black)
_NEUTRAL_IDX = list(range(18, 24))

# Bradford D50 → D65
_BRADFORD_D50_D65 = np.array([
    [ 0.9554734, -0.0230545,  0.0631633],
    [-0.0283728,  1.0099954,  0.0210224],
    [ 0.0123140, -0.0205262,  1.3299359],
], dtype=np.float64)

# D65 XYZ → linear sRGB (IEC 61966-2-1)
_XYZ_D65_TO_SRGB_LINEAR = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)

_SRGB_LIN_TO_XYZ_D65 = np.linalg.inv(_XYZ_D65_TO_SRGB_LINEAR)
_BRADFORD_D65_D50     = np.linalg.inv(_BRADFORD_D50_D65)


def _xyz_to_lab(xyz: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    """CIE XYZ → CIELAB."""
    r = xyz / illuminant
    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, t ** (1/3), 7.787 * t + 16/116)
    fx, fy, fz = f(r[..., 0]), f(r[..., 1]), f(r[..., 2])
    return np.stack([116*fy - 16, 500*(fx - fy), 200*(fy - fz)], axis=-1)


def _delta_e_00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE ΔE00 (vectorised)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.sqrt(a1**2 + b1**2); C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2; C7 = C_avg**7
    G = 0.5 * (1 - np.sqrt(C7 / (C7 + 25**7)))
    a1p = a1 * (1 + G); a2p = a2 * (1 + G)
    C1p = np.sqrt(a1p**2 + b1**2); C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = np.where(np.abs(h2p - h1p) <= 180, h2p - h1p,
          np.where(h2p <= h1p, h2p - h1p + 360, h2p - h1p - 360))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lp_avg = (L1 + L2) / 2; Cp_avg = (C1p + C2p) / 2
    hp_avg = np.where(np.abs(h1p - h2p) <= 180, (h1p + h2p) / 2,
             np.where(h1p + h2p < 360, (h1p + h2p + 360) / 2, (h1p + h2p - 360) / 2))
    T = (1 - 0.17*np.cos(np.radians(hp_avg-30)) + 0.24*np.cos(np.radians(2*hp_avg))
         + 0.32*np.cos(np.radians(3*hp_avg+6)) - 0.20*np.cos(np.radians(4*hp_avg-63)))
    SL = 1 + 0.015*(Lp_avg-50)**2 / np.sqrt(20+(Lp_avg-50)**2)
    SC = 1 + 0.045*Cp_avg; SH = 1 + 0.015*Cp_avg*T
    d_th = 30*np.exp(-((hp_avg-275)/25)**2)
    RC = 2*np.sqrt(Cp_avg**7 / (Cp_avg**7 + 25**7))
    RT = -np.sin(np.radians(2*d_th))*RC
    return np.sqrt((dLp/(SL))**2 + (dCp/(SC))**2 + (dHp/(SH))**2
                   + RT*(dCp/SC)*(dHp/SH))


def test_macbeth_chromatic_accuracy(verbose: bool = False) -> TestResult:
    """
    Test 2: Macbeth ColorChecker sRGB ↔ ACEScg round-trip.

    For each of the 24 Macbeth patches:
      D50 XYZ → D65 XYZ → linear sRGB → ACEScg → sRGB → display sRGB
      Compare output display sRGB (→ Lab) against reference (input D50 XYZ → Lab).

    In-gamut patches (sRGB linear ∈ [0, 1]): must have ΔE00 < 1.0.
    Neutral patches (White–Black):             must have ΔE00 < 0.5.
    Out-of-gamut patches are reported but do not fail the test.
    """
    t0 = time.perf_counter()
    errors: List[str] = []
    metrics: dict = {}

    D50_white = np.array([0.9642, 1.0000, 0.8251])

    # Reference Lab from D50 XYZ
    ref_lab = _xyz_to_lab(_MACBETH_XYZ_D50, D50_white)

    # Forward pipeline
    xyz_d65  = _MACBETH_XYZ_D50 @ _BRADFORD_D50_D65.T          # D50 → D65
    srgb_lin = (xyz_d65 @ _XYZ_D65_TO_SRGB_LINEAR.T)           # XYZ D65 → sRGB linear
    srgb_f32 = srgb_lin.astype(np.float32)

    acescg   = apply_matrix(srgb_f32, SRGB_TO_ACESCG)           # sRGB → ACEScg
    srgb_out = apply_matrix(acescg, ACESCG_TO_SRGB)             # ACEScg → sRGB
    srgb_display = linear_to_srgb(srgb_out.clip(0.0, 1.0))     # OETF

    # Recover XYZ from display sRGB for ΔE00
    srgb_dec  = srgb_to_linear(srgb_display.clip(0.0, 1.0)).astype(np.float64)
    xyz_rec   = srgb_dec @ _SRGB_LIN_TO_XYZ_D65.T @ _BRADFORD_D65_D50.T
    out_lab   = _xyz_to_lab(xyz_rec, D50_white)
    de00      = _delta_e_00(ref_lab, out_lab)

    # Classify patches
    in_gamut  = np.all((srgb_lin >= -0.001) & (srgb_lin <= 1.001), axis=-1)
    patch_info = []
    for i in range(24):
        info = {
            "patch": _MACBETH_NAMES[i],
            "in_gamut": bool(in_gamut[i]),
            "delta_e00": round(float(de00[i]), 3),
        }
        patch_info.append(info)
    metrics["patches"] = patch_info

    ig_de   = de00[in_gamut]
    metrics["in_gamut_count"]      = int(np.sum(in_gamut))
    metrics["in_gamut_mean_de00"]  = round(float(ig_de.mean()) if len(ig_de) else 0.0, 3)
    metrics["in_gamut_max_de00"]   = round(float(ig_de.max())  if len(ig_de) else 0.0, 3)

    # In-gamut: ΔE00 < 1.0
    for i in range(24):
        if not in_gamut[i]:
            continue
        de = float(de00[i])
        tol = 0.5 if i in _NEUTRAL_IDX else 1.0
        if de > tol:
            errors.append(f"Patch {_MACBETH_NAMES[i]} (in-gamut): ΔE00 {de:.3f} > {tol}")

    # Summary of out-of-gamut patches (informational)
    oog_patches = [_MACBETH_NAMES[i] for i in range(24) if not in_gamut[i]]
    metrics["out_of_gamut_patches"] = oog_patches

    return TestResult(
        name="Macbeth ColorChecker chromatic accuracy (ΔE00, sRGB↔ACEScg)",
        passed=len(errors) == 0,
        metrics=metrics,
        errors=errors,
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# § 3  14-stop highlight rolloff (Daniele Evo)
# ═══════════════════════════════════════════════════════════════════════════

def test_highlight_rolloff(verbose: bool = False) -> TestResult:
    """
    Test 3: 14-stop highlight rolloff via Daniele Evo tonescale (1000 nit).

    Pass criteria:
      - Monotonically non-decreasing across all 14 stops
      - 18% grey → 10% of display peak (±0.5%)
      - +7 stops (23.04 scene-linear) → 90–99.5% of display peak
      - No oscillation in highlights (slope never goes negative)
    """
    t0 = time.perf_counter()
    errors: List[str] = []
    metrics: dict = {}

    params = DanieleEvoParams(peak_nits=1000.0)

    x = np.logspace(np.log10(0.18 * 2**-7), np.log10(0.18 * 2**7), 10000, dtype=np.float64)
    y = daniele_evo_fwd(x.astype(np.float32), params).astype(np.float64)
    dy = np.diff(y)

    # A) Monotone
    metrics["monotone"] = bool(np.all(dy >= -1e-8))
    if not metrics["monotone"]:
        errors.append(f"Tonescale not monotone — worst drop {dy.min():.2e}")

    # B) Grey point
    grey_out = float(daniele_evo_fwd(np.array([0.18], dtype=np.float32), params)[0])
    metrics["grey_18pct_output"] = round(grey_out, 4)
    if abs(grey_out - 0.10) > 0.005:
        errors.append(f"18% grey → {grey_out:.4f}, expected 0.100 ± 0.005")

    # C) +7 stops
    peak_scene = 0.18 * 2.0**7
    peak_out   = float(daniele_evo_fwd(np.array([peak_scene], dtype=np.float32), params)[0])
    metrics["plus7_stop_output"] = round(peak_out, 4)
    if peak_out > 1.0:
        errors.append(f"+7 stop output {peak_out:.4f} > 1.0 — hard clipping")
    if peak_out < 0.90:
        errors.append(f"+7 stop output {peak_out:.4f} < 0.90 — insufficient highlight rolloff")

    # D) No negative slope in highlights
    hi_mask = x > 0.18 * 2.0**3
    hi_dy   = np.diff(y[hi_mask])
    metrics["highlight_min_slope"] = float(f"{hi_dy.min():.2e}")
    if hi_dy.min() < -1e-6:
        errors.append(f"Highlight slope oscillation: min slope {hi_dy.min():.2e}")

    # E) Per-stop output table
    stops = np.arange(-7, 8, dtype=float)
    stop_vals = (0.18 * 2.0 ** stops).astype(np.float32)
    metrics["per_stop_output"] = [
        round(float(v), 4) for v in daniele_evo_fwd(stop_vals, params)
    ]

    return TestResult(
        name="14-stop highlight rolloff (Daniele Evo, 1000 nit)",
        passed=len(errors) == 0,
        metrics=metrics,
        errors=errors,
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# § 4  ACES 2.0 pipeline self-consistency regression
# ═══════════════════════════════════════════════════════════════════════════

# Reference values computed from the pipeline at radiance_color v1.0.0.
# Pipeline: ACEScg → reach_gamut_compress → daniele_evo_fwd(per-ch)
#           → ACESCG_TO_SRGB clip(0,1) → linear_to_srgb
# Pinned tolerance: 0.001 sRGB (sub-half-DN in 8-bit).
_ACES2_REFERENCE: list[tuple[str, tuple, tuple]] = [
    ("18pct_grey",         (0.18, 0.18, 0.18), (0.3492, 0.3492, 0.3492)),
    ("2stop_overexpose",   (0.72, 0.72, 0.72), (0.6292, 0.6292, 0.6292)),
    ("peak_white_8x",      (1.44, 1.44, 1.44), (0.7664, 0.7664, 0.7664)),
    ("red_primary",        (1.0,  0.0,  0.0),  (0.877,  0.0,    0.0   )),
    ("green_primary",      (0.0,  1.0,  0.0),  (0.0,    0.7373, 0.0   )),
    ("blue_primary",       (0.0,  0.0,  1.0),  (0.0,    0.1372, 0.7409)),
    ("near_black",         (0.01, 0.01, 0.01), (0.0385, 0.0385, 0.0385)),
]


def _run_aces2_pipeline(rgb_in: tuple, params: DanieleEvoParams) -> tuple:
    x    = np.array([rgb_in], dtype=np.float32)
    x_gc = reach_gamut_compress(x, strength=1.0)
    x_tm = daniele_evo_fwd(x_gc, params)
    x_srgb = apply_matrix(x_tm, ACESCG_TO_SRGB).clip(0.0, 1.0)
    x_out  = linear_to_srgb(x_srgb)
    return tuple(round(float(v), 4) for v in x_out[0])


def test_aces2_pipeline_reference(verbose: bool = False) -> TestResult:
    """
    Test 4: ACES 2.0 pipeline self-consistency regression (1000 nit).

    Verifies that the current pipeline produces bit-stable output against
    values pinned at radiance_color v1.0.0.  Tolerance: 0.001 sRGB.

    This catches regressions in: reach_gamut_compress, daniele_evo_fwd,
    ACESCG_TO_SRGB matrix, and linear_to_srgb OETF.
    """
    t0 = time.perf_counter()
    errors: List[str] = []
    metrics: dict = {}

    params = DanieleEvoParams(peak_nits=1000.0)
    tol = 0.001

    patch_results = []
    for name, rgb_in, rgb_ref in _ACES2_REFERENCE:
        out   = _run_aces2_pipeline(rgb_in, params)
        max_e = max(abs(o - r) for o, r in zip(out, rgb_ref))
        ok    = max_e <= tol
        patch_results.append({
            "patch":     name,
            "output":    out,
            "reference": rgb_ref,
            "max_err":   round(max_e, 5),
            "passed":    ok,
        })
        if not ok:
            errors.append(
                f"{name}: output {out} vs ref {rgb_ref}, err={max_e:.5f} > tol={tol}"
            )

    metrics["patch_results"]  = patch_results
    metrics["tolerance_srgb"] = tol

    return TestResult(
        name="ACES 2.0 pipeline regression (1000 nit, v1.0.0 pin)",
        passed=len(errors) == 0,
        metrics=metrics,
        errors=errors,
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# § 5  Runner + report
# ═══════════════════════════════════════════════════════════════════════════

_ALL_TESTS = [
    test_alexa_sensor_simulation,
    test_macbeth_chromatic_accuracy,
    test_highlight_rolloff,
    test_aces2_pipeline_reference,
]


def _print_report(report: ValidationReport, verbose: bool) -> None:
    width = 72
    print("=" * width)
    print(f"  Radiance HDR Pipeline Validation  v{report.version}")
    print(f"  radiance_color {report.radiance_color_version}   {report.timestamp}")
    print("=" * width)

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        sym    = "✓" if r.passed else "✗"
        print(f"\n  [{sym}] {status}  {r.name}  ({r.duration_ms:.0f} ms)")

        if r.errors:
            for e in r.errors:
                print(f"       ERR: {e}")

        if verbose:
            for k, v in r.metrics.items():
                if isinstance(v, (int, float, bool)):
                    print(f"       {k}: {v}")
                elif isinstance(v, list) and len(v) <= 24 and all(isinstance(x, (int, float)) for x in v):
                    print(f"       {k}: {v}")
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    for item in v:
                        ok  = item.get("passed", item.get("in_gamut", True))
                        sym2 = "✓" if ok else "~"
                        core = {kk: vv for kk, vv in item.items() if kk not in ("passed",)}
                        print(f"         [{sym2}] {core}")
                else:
                    print(f"       {k}: {v}")

    print()
    print("-" * width)
    print(f"  Passed: {report.n_passed}/{len(report.results)}")
    if report.passed:
        print("  Overall: PASS — pipeline is production-ready")
    else:
        print(f"  Overall: FAIL — {report.n_failed} test(s) failed")
    print("=" * width)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Radiance HDR pipeline validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Write JSON report to FILE")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed metrics")
    args = parser.parse_args()

    import datetime
    report = ValidationReport(
        version="1.0.0",
        timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        radiance_color_version=rc.__version__,
    )

    for test_fn in _ALL_TESTS:
        result = test_fn(verbose=args.verbose)
        report.results.append(result)

    _print_report(report, verbose=args.verbose)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version":                  report.version,
            "timestamp":                report.timestamp,
            "radiance_color_version":   report.radiance_color_version,
            "overall_passed":           report.passed,
            "n_passed":                 report.n_passed,
            "n_failed":                 report.n_failed,
            "results":                  [asdict(r) for r in report.results],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nJSON report written to: {out_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
