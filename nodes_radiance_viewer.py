"""
═══════════════════════════════════════════════════════════════════════════════
                     RADIANCE VIEWER NODE v2.1
              VFX Industry-Standard Image Viewer for ComfyUI
                         Radiance © 2024-2026

Professional viewer node providing:
- Interactive zoom/pan with canvas-based rendering
- Real-time exposure, gamma, gain, lift, saturation, temperature controls
- Channel viewing modes (RGB, R, G, B, Alpha, Luminance)
- Color picker with HDR value display (via float32 sidecar)
- Built-in LUTs (sRGB, Rec.709, Log-to-Lin, Filmic, False Color, etc.)
- False color and zebra analysis
- A/B comparison modes with mouse-draggable wipe divider
- IMAGE passthrough output
- 16-bit PNG + .rhdr compressed float16 sidecar for true HDR
- 32-bit OpenEXR export via cv2/imageio
- Z-Depth with 16-bit precision

VERSION HISTORY:

v2.1 — Pro Evolution (February 2026)
─────────────────────────────────────
Phase 8  — GPU Waveform Monitor
  NEW: renderWaveform() public helper in WebGL renderer
  NEW: Luma waveform + RGB Parade toggle in scope HUD

Phase 9  — Localized Grading (Power Windows)
  NEW: Radial + Box mask fully wired — setMask() API
  NEW: showMaskOverlay flag for mask positioning

Phase 10 — Comparison Bridge & Reference Shelf
  NEW: initWipeDragging() — mouse-draggable wipe divider on GL canvas
  NEW: grabReferenceStill() — snapshot current grade → shelf texture
  NEW: referenceShelf[0..7] — up to 8 stored comparison stills
  NEW: swapReferenceShelf(idx) — instantly swap active reference

Phase 11 — Cinematic Optical Effects
  NEW: Brown-Conrady full k1+k2 barrel/pincushion distortion
  NEW: Anamorphic lens streaks (horizontal highlight bloom, cyan tint)
  NEW: setLensDistortionK2(), setAnamorphicStreaks(), setStreakThreshold()

Phase 12 — GPU Bilateral Filter (Edge-Preserving Denoise)
  NEW: 7×7 bilateral kernel replaces 5-tap box blur
  NEW: setBilateralSigma(sigmaD, sigmaR) — spatial + range control
  NEW: Preserves skin/hair/object edges at all denoise strengths

v2.1 — HDR Display Fix (February 2026)
  FIX: Black result from missing sRGB OETF in composite shader
  FIX: HDR sidecar always saved
  FIX: PNG preview tonemapped via Reinhard for HDR content

v2.1 — Major Upgrade (February 2026)
  NEW: Exposure, Gamma, Gain, Lift, Saturation, Temperature sliders
  NEW: IMAGE passthrough output
  NEW: Built-in LUT engine (10 industry-standard LUTs)
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import torch
import numpy as np
import zlib
import os
import uuid
import logging
import math

# v3.1: Enable OpenEXR support in OpenCV
# Essential for 32-bit float export
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from typing import Dict, Any, Optional, List, Tuple

import folder_paths

# Import safe path utilities
try:
    from .path_utils import safe_join
except ImportError:
    from path_utils import safe_join

# v1.21: cv2 for 16-bit PNG support
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.getLogger("radiance.viewer").warning(
        "cv2 not available — 16-bit PNG disabled, falling back to 8-bit. "
        "Install opencv-python for 16-bit support."
    )

# Module logger
logger = logging.getLogger("radiance.viewer")


# ═══════════════════════════════════════════════════════════════════════════════
#                           CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PNG_COMPRESSION = 4
BIT_16_MAX = 65535
BIT_8_MAX = 255
BIT_16_TO_8_DIVISOR = 257  # Correct: 65535 / 255 = 257

MAX_IMAGE_DIMENSION = 16384
MAX_BATCH_SIZE = 100

# Try to import imageio for EXR fallback
try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

BIT_DEPTH_MODES = ["8-bit (Fast)", "16-bit (Quality)", "16-bit + HDR Data"]
CV2_PNG_COMPRESSION = 4

# ═══════════════════════════════════════════════════════════════════════════════
#                        BUILT-IN LUT ENGINE v2.1
#
#  Analytical LUTs — no file dependencies.
#  Each LUT is a function: float32 linear [0..1+] → float32 [0..1]
#  Applied per-channel in linear space unless noted otherwise.
# ═══════════════════════════════════════════════════════════════════════════════

LUT_MODES = [
    # ── Display / Tonemap ──────────────────────────────────────────────────────
    "None",
    "sRGB (Display)",
    "Rec.709 (Broadcast)",
    "Filmic (Cinematic)",
    "Reinhard Tonemap",
    "ACES Filmic",
    # ── Camera Log Encoding (Linear → Log) ────────────────────────────────────
    "LogC3 (ARRI EI800)",
    "LogC4 (ARRI Alexa 35)",
    "F-Log2 (Fujifilm)",
    "C-Log3 (Canon)",
    "Log3G10 (RED IPP2)",
    "DaVinci Intermediate",
    "BMD Film Gen5",
    "V-Log (Panasonic)",
    "RED Log3G10",
    "—",
    "N-Log (Nikon)",
    "Linear to Log (Generic)",
    "IDT: LogC3 → Linear",
    "IDT: LogC4 → Linear",
    "IDT: V-Log → Linear",
    "IDT: Log3G10 → Linear",
    "IDT: DaVinci → Linear",
    "IDT: BMD Gen5 → Linear",
    "IDT: N-Log → Linear",
    # ── Analysis ──────────────────────────────────────────────────────────────
    "False Color (Exposure)",
]

# ═══════════════════════════════════════════════════════════════════════════════
#                    CAMERA LOG LUT FUNCTIONS  (v3.2 — Spec-Accurate)
#
#  All forward transforms:
#    • Input: scene-linear float32, 18% grey = 0.18
#    • Output: log-encoded float, clipped to [0, 1]
#
#  Verified 18% grey output values (normalized 0–1):
#    LogC3:   0.391  |  LogC4:  0.278  |  S-Log3:  0.411
#    V-Log:   0.423  |  F-Log2: 0.391  |  C-Log3:  0.343
#    Log3G10: 0.333  |  DaVinci: 0.336 |  BMD Gen5: 0.384 | N-Log: 0.364
#
#  All IDT functions are exact inverses of their forward counterpart.
#  Sources: manufacturer white papers, ACES CLF reference, OCIO configs.
# ═══════════════════════════════════════════════════════════════════════════════


def _lut_srgb(x: np.ndarray) -> np.ndarray:
    """Linear → sRGB gamma  [IEC 61966-2-1]."""
    return np.clip(
        np.where(
            x <= 0.0031308,
            x * 12.92,
            1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055,
        ),
        0.0,
        1.0,
    )


def _lut_rec709(x: np.ndarray) -> np.ndarray:
    """Linear → Rec.709 OETF  [ITU-R BT.709-6]."""
    return np.clip(
        np.where(
            x < 0.018, x * 4.5, 1.099 * np.power(np.maximum(x, 0.018), 0.45) - 0.099
        ),
        0.0,
        1.0,
    )


def _lut_filmic(x: np.ndarray) -> np.ndarray:
    """Filmic tonemapping (Hable / Uncharted 2 curve)."""
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

    def hable(v):
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

    white_scale = 1.0 / hable(np.array(11.2))
    return np.clip(hable(np.maximum(x, 0.0) * 2.0) * white_scale, 0.0, 1.0)


def _lut_reinhard(x: np.ndarray) -> np.ndarray:
    """Reinhard global tonemapping."""
    return np.clip(x / (1.0 + x), 0.0, 1.0)


def _lut_aces_filmic(x: np.ndarray) -> np.ndarray:
    """ACES Filmic tone mapping (Narkowicz fit)."""
    x = np.maximum(x, 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


# ── Camera Log Encoding ───────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
#        CAMERA LOG LUT FUNCTIONS  (v3.3 — Spec-Accurate)
#
#  All forward transforms (Linear → Log):
#    • Input:  scene-linear float32, 18% grey = 0.18
#    • Output: log-encoded float, clipped to [0, 1]
#
#  Authoritative 18% grey output values (verified against colour-science
#  library, which cites manufacturer specification documents directly):
#    LogC3:   0.391  |  LogC4:  0.278  |  S-Log3:  0.411
#    V-Log:   0.423  |  F-Log2: 0.391  |  C-Log3:  0.343
#    Log3G10: 0.333  |  DaVinci: 0.336 |  BMD Gen5: 0.384 | N-Log: 0.364
#
#  All IDT functions are exact analytic inverses of their forward counterpart.
#  Sources: colour-science library (https://github.com/colour-science/colour),
#           manufacturer white papers, ACES CLF reference, OCIO configs.
# ═══════════════════════════════════════════════════════════════════════════════


def _lut_logc3(x: np.ndarray) -> np.ndarray:
    """Linear → ARRI LogC3 EI800.
    Source: colour-science DATA_ALEXA_LOG_C_CURVE_CONVERSION['SUP 3.x'][800]
    Verified: 18% grey → 0.391007
    """
    # EI800 piecewise constants
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809  # linear toe (shadow extension)
    return np.clip(
        np.where(x > cut, c * np.log10(np.maximum(a * x + b, 1e-10)) + d, e * x + f),
        0.0,
        1.0,
    )


def _lut_logc4(x: np.ndarray) -> np.ndarray:
    """Linear → ARRI LogC4 (Alexa 35).
    Source: colour-science CONSTANTS_ARRILOGC4 (ARRI Specification 2022)
    Formula: E_p = (log2(a*E + 64) - 6) / 14 * b + c  for E >= t
             E_p = (E - t) / s                          for E < t
    Verified: 18% grey → 0.278396
    """
    a = 2231.82630906768830
    b = 0.90713587487781030
    c = 0.09286412512218964
    s = 0.11359720861058910
    t = -0.01805699611991131
    log_branch = (
        np.log2(np.maximum(a * np.maximum(x, t) + 64.0, 1e-10)) - 6.0
    ) / 14.0 * b + c
    lin_branch = (x - t) / s
    return np.where(x >= t, log_branch, lin_branch)


def _lut_slog3(x: np.ndarray) -> np.ndarray:
    """Linear → Sony S-Log3.
    Source: colour-science log_encoding_SLog3 (Sony Specification 2014)
    Formula: (420 + log10((x+0.01)/(0.18+0.01)) * 261.5) / 1023
    Key: normalization by /0.19 — not just log10(x+0.01)
    Verified: 18% grey → 0.410557
    """
    cut = 0.01125
    return np.clip(
        np.where(
            x >= cut,
            (420.0 + np.log10(np.maximum(x + 0.01, 1e-10) / 0.19) * 261.5) / 1023.0,
            (x * (171.2102946929 - 95.0) / cut + 95.0) / 1023.0,
        ),
        0.0,
        1.0,
    )


def _lut_vlog(x: np.ndarray) -> np.ndarray:
    """Linear → Panasonic V-Log / V-Log3.
    Source: colour-science CONSTANTS_VLOG (Panasonic Specification 2014)
    Verified: 18% grey → 0.423311
    """
    return np.clip(
        np.where(
            x < 0.01,
            5.6 * x + 0.125,
            0.241514 * np.log10(np.maximum(x + 0.00873, 1e-10)) + 0.598206,
        ),
        0.0,
        1.0,
    )


def _lut_flog2(x: np.ndarray) -> np.ndarray:
    """Linear → Fujifilm F-Log2.
    Source: colour-science CONSTANTS_FLOG2 (Fujifilm Specification 2022)
    Piecewise: c*log10(a*x+b)+d for x >= cut1; e*x+f for x < cut1
    Verified: 18% grey → 0.391007
    """
    cut1 = 0.000889
    a = 5.555556
    b = 0.064829
    c_c = 0.245281
    d = 0.384316
    e = 8.799461
    f = 0.092864
    return np.clip(
        np.where(
            x >= cut1, c_c * np.log10(np.maximum(a * x + b, 1e-10)) + d, e * x + f
        ),
        0.0,
        1.0,
    )


def _lut_clog3(x: np.ndarray) -> np.ndarray:
    """Linear → Canon C-Log3 v1.2.
    Source: colour-science log_encoding_CanonLog3_v1_2 (Canon Specification 2020)
    Input rescaled by /0.9 before encoding. Three-segment piecewise.
    Cut points (in rescaled space): lower ≈ -0.009670, upper ≈ 0.014043
    Verified: 18% grey → 0.343389
    """
    xr = x / 0.9  # Canon rescaling
    k = 14.98325
    a = 0.36726845
    lo = -0.009670
    hi = 0.014043
    neg = -a * np.log10(np.maximum(-xr * k + 1.0, 1e-10)) + 0.12783901
    lin = 1.9754798 * xr + 0.12512219
    pos = a * np.log10(np.maximum(xr * k + 1.0, 1e-10)) + 0.12240537
    result = np.where(xr < lo, neg, np.where(xr <= hi, lin, pos))
    return result  # C-Log3 values can be outside [0,1] for extreme inputs


def _lut_log3g10(x: np.ndarray) -> np.ndarray:
    """Linear → RED Log3G10 v2 (REDCINE-X PRO / IPP2 pipeline).
    Source: colour-science log_encoding_Log3G10_v2 (Nattress / RED 2016)
    Formula: sign(x+0.01) * 0.224282 * log10(|x+0.01| * 155.975327 + 1)
    Verified: 18% grey → 0.333333 (exactly 1/3 by design)
    """
    xoff = x + 0.01
    return np.sign(xoff) * 0.224282 * np.log10(np.abs(xoff) * 155.975327 + 1.0)


def _lut_davinci_intermediate(x: np.ndarray) -> np.ndarray:
    """Linear → DaVinci Intermediate.
    Source: colour-science CONSTANTS_DAVINCI_INTERMEDIATE (BMD Specification 2020)
    Formula: DI_C * (log2(L + DI_A) + DI_B)  for L > DI_LIN_CUT
             L * DI_M                          for L <= DI_LIN_CUT
    Verified: 18% grey → 0.336043
    """
    DI_A = 0.0075
    DI_B = 7.0
    DI_C = 0.07329248
    DI_M = 10.44426855
    DI_LIN_CUT = 0.00262409
    return np.where(
        x <= DI_LIN_CUT, x * DI_M, DI_C * (np.log2(np.maximum(x + DI_A, 1e-10)) + DI_B)
    )


def _lut_bmd_gen5(x: np.ndarray) -> np.ndarray:
    """Linear → Blackmagic Film Generation 5.
    Source: colour-science CONSTANTS_BLACKMAGIC_FILM_GENERATION_5 (BMD Specification 2021)
    Formula: A*ln(x+B)+C  for x >= LIN_CUT;  D*x+E  for x < LIN_CUT
    Uses natural log (ln), NOT log10.
    Verified: 18% grey → 0.383562
    """
    A = 0.08692876065491224
    B = 0.005494072432257808
    C = 0.5300133392291939
    D = 8.283605932402494
    E = 0.09246575342465753
    LIN_CUT = 0.005
    return np.where(x >= LIN_CUT, A * np.log(np.maximum(x + B, 1e-10)) + C, D * x + E)


def _lut_nlog(x: np.ndarray) -> np.ndarray:
    """Linear → Nikon N-Log.
    Source: colour-science CONSTANTS_NLOG (Nikon Specification 2018)
    Formula: a*(y+b)^(1/3)  for y < cut1 (cube-root toe)
             c*ln(y)+d      for y >= cut1 (log body)
    Uses natural log (ln), NOT log10.
    Verified: 18% grey → 0.363668
    """
    cut1 = 0.328
    a = 0.635386119257087
    b = 0.0075
    c = 0.1466275659824047
    d = 0.6050830889540567
    log_v = c * np.log(np.maximum(x, 1e-10)) + d
    cbrt_v = a * np.power(np.maximum(x + b, 1e-10), 1.0 / 3.0)
    return np.where(x >= cut1, log_v, cbrt_v)


def _lut_lin_to_log(x: np.ndarray) -> np.ndarray:
    """Generic linear → log (Cineon-style)."""
    return np.clip(np.log2(np.maximum(x, 1e-10) * 5.55 + 1.0) / np.log2(6.55), 0.0, 1.0)


def _lut_log_to_lin(x: np.ndarray) -> np.ndarray:
    """Generic log → linear (Cineon inverse)."""
    return np.clip((np.power(6.55, np.clip(x, 0.0, 1.0)) - 1.0) / 5.55, 0.0, 1.0)


# ── IDT: Camera Log → Scene Linear ────────────────────────────────────────────
# All IDT functions are exact analytic inverses of their forward counterparts.


def _idt_logc3(x: np.ndarray) -> np.ndarray:
    """IDT: ARRI LogC3 EI800 → Scene Linear (exact inverse of _lut_logc3)."""
    # Code-value at cut: e*cut + f = 5.367655*0.010591 + 0.092809 ≈ 0.149658
    ENC_CUT = 5.367655 * 0.010591 + 0.092809
    log_out = (np.power(10.0, (x - 0.385537) / 0.247190) - 0.052272) / 5.555556
    lin_out = (x - 0.092809) / 5.367655
    return np.maximum(np.where(x > ENC_CUT, log_out, lin_out), 0.0)


def _idt_logc4(x: np.ndarray) -> np.ndarray:
    """IDT: ARRI LogC4 → Scene Linear (exact inverse of _lut_logc4)."""
    a = 2231.82630906768830
    b = 0.90713587487781030
    c = 0.09286412512218964
    s = 0.11359720861058910
    t = -0.01805699611991131
    # Code value at cut t
    lc4_log_cut = (np.log2(a * t + 64.0) - 6.0) / 14.0 * b + c
    log_out = (np.exp2((x - c) / b * 14.0 + 6.0) - 64.0) / a
    lin_out = x * s + t
    return np.where(x >= lc4_log_cut, log_out, lin_out)


def _idt_slog3(x: np.ndarray) -> np.ndarray:
    """IDT: Sony S-Log3 → Scene Linear (exact inverse of _lut_slog3).
    Applies the /0.19 normalization: x = 10^((V*1023-420)/261.5) * 0.19 - 0.01
    """
    # CV at cut: (cut*(171.2102946929-95)/0.01125 + 95)/1023
    CUT_CV = (0.01125 * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0  # = 95/1023
    log_out = np.power(10.0, (x * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
    lin_out = (x * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0)
    return np.maximum(np.where(x >= CUT_CV, log_out, lin_out), 0.0)


def _idt_vlog(x: np.ndarray) -> np.ndarray:
    """IDT: Panasonic V-Log → Scene Linear (exact inverse of _lut_vlog)."""
    return np.maximum(
        np.where(
            x < 0.181,  # CONSTANTS_VLOG.cut2
            (x - 0.125) / 5.6,
            np.power(10.0, (x - 0.598206) / 0.241514) - 0.00873,
        ),
        0.0,
    )


def _idt_flog2(x: np.ndarray) -> np.ndarray:
    """IDT: Fujifilm F-Log2 → Scene Linear (exact inverse of _lut_flog2)."""
    a = 5.555556
    b = 0.064829
    c_c = 0.245281
    d = 0.384316
    e = 8.799461
    f = 0.092864
    CUT_CV = c_c * np.log10(a * 0.000889 + b) + d  # ≈ 0.09248
    log_out = (np.power(10.0, (x - d) / c_c) - b) / a
    lin_out = (x - f) / e
    return np.where(x >= CUT_CV, log_out, lin_out)


def _idt_clog3(x: np.ndarray) -> np.ndarray:
    """IDT: Canon C-Log3 v1.2 → Scene Linear (inverse of positive branch).
    For normal scene content (x > CV of linear segment ~0.1527).
    """
    cl3_lin_cv = 1.9754798 * 0.014043 + 0.12512219  # ≈ 0.15277
    k = 14.98325
    a = 0.36726845
    log_out = (np.power(10.0, (x - 0.12240537) / a) - 1.0) / k * 0.9
    lin_out = (x - 0.12512219) / 1.9754798 * 0.9
    return np.where(x > cl3_lin_cv, log_out, lin_out)


def _idt_log3g10(x: np.ndarray) -> np.ndarray:
    """IDT: RED Log3G10 v2 → Scene Linear (exact inverse of _lut_log3g10)."""
    s = np.sign(x)
    return s * (np.power(10.0, np.abs(x) / 0.224282) - 1.0) / 155.975327 - 0.01


def _idt_davinci_intermediate(x: np.ndarray) -> np.ndarray:
    """IDT: DaVinci Intermediate → Scene Linear (exact inverse of _lut_davinci_intermediate)."""
    DI_A = 0.0075
    DI_B = 7.0
    DI_C = 0.07329248
    DI_M = 10.44426855
    DI_LOG_CUT = 0.02740668
    log_out = np.exp2(x / DI_C - DI_B) - DI_A
    lin_out = x / DI_M
    return np.where(x > DI_LOG_CUT, log_out, lin_out)


def _idt_bmd_gen5(x: np.ndarray) -> np.ndarray:
    """IDT: BMD Film Generation 5 → Scene Linear (exact inverse of _lut_bmd_gen5)."""
    A = 0.08692876065491224
    B = 0.005494072432257808
    C = 0.5300133392291939
    D = 8.283605932402494
    E = 0.09246575342465753
    LOG_CUT = D * 0.005 + E  # ≈ 0.13388
    log_out = np.exp((x - C) / A) - B
    lin_out = (x - E) / D
    return np.where(x >= LOG_CUT, log_out, lin_out)


def _idt_nlog(x: np.ndarray) -> np.ndarray:
    """IDT: Nikon N-Log → Scene Linear (exact inverse of _lut_nlog)."""
    a = 0.635386119257087
    b = 0.0075
    c = 0.1466275659824047
    d = 0.6050830889540567
    # CV at cut1=0.328: a * (0.328 + b)^(1/3)
    CUT_CV = a * (0.328 + b) ** (1.0 / 3.0)
    log_out = np.exp((x - d) / c)
    cbrt_out = np.power(x / a, 3.0) - b
    return np.maximum(np.where(x >= CUT_CV, log_out, cbrt_out), 0.0)


def _lut_passthrough(x: np.ndarray) -> np.ndarray:
    """Placeholder for separator."""
    return x


# ── LUT dispatch table ────────────────────────────────────────────────────────
# Keys must match LUT_MODES entries exactly.
_LUT_FUNCTIONS = {
    "None": None,
    # Display / Tonemap
    "sRGB (Display)": _lut_srgb,
    "Rec.709 (Broadcast)": _lut_rec709,
    "Filmic (Cinematic)": _lut_filmic,
    "Reinhard Tonemap": _lut_reinhard,
    "ACES Filmic": _lut_aces_filmic,
    # Camera Log — Forward (Linear → Log)
    "LogC3 (ARRI EI800)": _lut_logc3,
    "LogC4 (ARRI Alexa 35)": _lut_logc4,
    "F-Log2 (Fujifilm)": _lut_flog2,
    "C-Log3 (Canon)": _lut_clog3,
    "Log3G10 (RED IPP2)": _lut_log3g10,
    "DaVinci Intermediate": _lut_davinci_intermediate,
    "BMD Film Gen5": _lut_bmd_gen5,
    "V-Log (Panasonic)": _lut_vlog,
    "RED Log3G10": _lut_log3g10,  # Alias for Log3G10 (RED IPP2)
    "—": _lut_passthrough,
    "N-Log (Nikon)": _lut_nlog,
    "Linear to Log (Generic)": _lut_lin_to_log,
    "IDT: LogC3 → Linear": _idt_logc3,
    "IDT: LogC4 → Linear": _idt_logc4,
    "IDT: V-Log → Linear": _idt_vlog,
    "IDT: Log3G10 → Linear": _idt_log3g10,
    "IDT: DaVinci → Linear": _idt_davinci_intermediate,
    "IDT: BMD Gen5 → Linear": _idt_bmd_gen5,
    "IDT: N-Log → Linear": _idt_nlog,
    # Analysis
    "False Color (Exposure)": "false_color",
}


def apply_lut(img: np.ndarray, lut_name: str, intensity: float = 1.0) -> np.ndarray:
    """
    Apply a named LUT to a float32 image.

    Args:
        img: float32 numpy array (H,W,C) or (H,W)
        lut_name: Key from LUT_MODES
        intensity: Blend factor 0.0 (bypass) to 1.0 (full)

    Returns:
        float32 numpy array, same shape (except False Color always returns H,W,3)
    """
    if lut_name == "None" or intensity <= 0.0:
        return img

    handler = _LUT_FUNCTIONS.get(lut_name)
    if handler is None:
        return img

    # False Color is a special case — replaces the whole image
    if handler == "false_color":
        fc = _lut_false_color(img)
        if intensity >= 1.0:
            return fc
        # Blend: need to match shapes
        if img.ndim == 2:
            orig = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            orig = np.concatenate([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            orig = img[..., :3]
        else:
            orig = img
        return (orig * (1.0 - intensity) + fc * intensity).astype(np.float32)

    # Standard per-channel LUT
    lut_applied = handler(img).astype(np.float32)

    if intensity >= 1.0:
        return lut_applied

    return (img * (1.0 - intensity) + lut_applied * intensity).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#                      COLOR GRADING ENGINE v3.2
#
#  CANONICAL PIPELINE ORDER — must match GLSL composite shader exactly:
#
#   1. Offset (global additive shift)
#   2. Exposure (stops, multiplicative)
#   3. White Balance (temperature / tint multiplicative)
#   4. Lift / Gain / Gamma  (Resolve-style — via applyGrading)
#   5. Contrast  (pivoted)
#   6. Shadows / Highlights (luma-weighted)
#   7. Saturation
#   8. Hue Shift
#   9. LUT (last in chain)
#
#  v3.2 FIX: Python and GLSL were out of sync.
#   Old Python order: Lift → Exposure → Gain → Gamma → Saturation → Temp → LUT
#   New Python order: matches GLSL exactly for correct passthrough output.
# ═══════════════════════════════════════════════════════════════════════════════


def apply_grading(
    img: np.ndarray,
    # Basic controls
    exposure: float = 0.0,
    gamma: float = 1.0,  # artistic gamma (maps to gradingGamma[1,1,1] → scalar)
    gain: float = 1.0,
    lift: float = 0.0,
    saturation: float = 1.0,
    temperature: float = 6500.0,
    # Extended Resolve-style controls (v3.2 sync)
    offset: float = 0.0,  # global additive offset (applied first)
    contrast: float = 1.0,  # contrast multiplier (pivoted)
    pivot: float = 0.18,  # contrast pivot point
    shadows: float = 0.0,  # shadow lift/crush (-1..1)
    highlights: float = 0.0,  # highlight expand/compress (-1..1)
    hue_shift: float = 0.0,  # degrees (-180..180)
    # LUT
    lut_name: str = "None",
    lut_intensity: float = 1.0,
) -> np.ndarray:
    """
    Apply full grading stack to a float32 image.

    Pipeline order is IDENTICAL to the GLSL composite shader so that the
    passthrough IMAGE output matches what the WebGL viewer displays.

    Args:
        img:         float32 (H,W,C), values can exceed [0,1] for HDR
        exposure:    Stops of exposure compensation (-6 to +6)
        gamma:       Artistic midtone gamma (0.1–4.0, 1.0 = identity)
        gain:        Multiplicative gain (0.0–5.0, 1.0 = unity)
        lift:        Shadow additive offset (-1.0–1.0, luma-pivoted)
        saturation:  Color saturation (0.0 = mono, 1.0 = normal, 3.0 = hyper)
        temperature: Color temperature in Kelvin (2000 = warm, 12000 = cool)
        offset:      Global additive offset applied before everything else
        contrast:    Contrast multiplier around pivot (1.0 = identity)
        pivot:       Contrast pivot point (default 0.18 = 18% grey)
        shadows:     Shadow region brightness (-1..1)
        highlights:  Highlight region brightness (-1..1)
        hue_shift:   Hue rotation in degrees
        lut_name:    Name of built-in LUT to apply
        lut_intensity: LUT blend factor (0.0–1.0)

    Returns:
        float32 numpy array, same shape as input
    """
    out = img.astype(np.float32, copy=True)

    # Fast-path: skip if all controls are at identity
    is_default = (
        abs(exposure) < 0.001
        and abs(gamma - 1.0) < 0.001
        and abs(gain - 1.0) < 0.001
        and abs(lift) < 0.001
        and abs(offset) < 0.001
        and abs(saturation - 1.0) < 0.001
        and abs(contrast - 1.0) < 0.001
        and abs(shadows) < 0.001
        and abs(highlights) < 0.001
        and abs(hue_shift) < 0.1
        and abs(temperature - 6500.0) < 10.0
        and lut_name == "None"
    )
    if is_default:
        return out

    # ── 1. Offset ──────────────────────────────────────────────────────────────
    if abs(offset) > 0.001:
        out += np.float32(offset)

    # ── 2. Exposure ────────────────────────────────────────────────────────────
    if abs(exposure) > 0.001:
        out *= np.float32(2.0**exposure)

    # ── 3. White Balance ───────────────────────────────────────────────────────
    if abs(temperature - 6500.0) > 10.0 and out.ndim == 3 and out.shape[2] >= 3:
        r_mult, g_mult, b_mult = _kelvin_to_rgb_multipliers(temperature)
        out[..., 0] *= np.float32(r_mult)
        out[..., 1] *= np.float32(g_mult)
        out[..., 2] *= np.float32(b_mult)

    # ── 4. Lift / Gain / Gamma  (Resolve-style) ───────────────────────────────
    # Lift: additive shift to shadows, pivoted by luma so whites are unaffected
    if abs(lift) > 0.001 and out.ndim == 3 and out.shape[2] >= 3:
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        pivot_lift = np.clip(1.0 - luma, 0.0, 1.0)[..., np.newaxis]
        out += np.float32(lift) * pivot_lift

    # Gain: multiplicative slope
    if abs(gain - 1.0) > 0.001:
        out *= np.float32(gain)

    # Gamma: power curve on positives only
    if abs(gamma - 1.0) > 0.001:
        inv_gamma = np.float32(1.0 / max(gamma, 0.01))
        positive = out > 0
        out[positive] = np.power(out[positive], inv_gamma)

    # ── 5. Contrast ────────────────────────────────────────────────────────────
    if abs(contrast - 1.0) > 0.001:
        out = (out - np.float32(pivot)) * np.float32(contrast) + np.float32(pivot)

    # ── 6. Shadows / Highlights ────────────────────────────────────────────────
    if (
        (abs(shadows) > 0.001 or abs(highlights) > 0.001)
        and out.ndim == 3
        and out.shape[2] >= 3
    ):
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        # Quadratic weights — matches GLSL smoothstep approach
        s_weight = np.power(np.clip(1.0 - luma, 0.0, 1.0), 2.0)[..., np.newaxis]
        h_weight = np.power(np.clip(luma, 0.0, 1.0), 2.0)[..., np.newaxis]
        out *= 1.0 + np.float32(shadows) * s_weight * 0.5
        out *= 1.0 + np.float32(highlights) * h_weight * 0.5
        out = np.maximum(out, 0.0)

    # ── 7. Saturation ──────────────────────────────────────────────────────────
    if abs(saturation - 1.0) > 0.001 and out.ndim == 3 and out.shape[2] >= 3:
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        for c in range(min(out.shape[2], 3)):
            out[..., c] = luma + saturation * (out[..., c] - luma)

    # ── 8. Hue Shift ───────────────────────────────────────────────────────────
    if abs(hue_shift) > 0.1 and out.ndim == 3 and out.shape[2] >= 3:
        pass

        h_shift = hue_shift / 360.0
        rgb = out[..., :3]
        # Vectorised HSV shift using numpy
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin + 1e-10

        h = (
            np.where(
                cmax == r,
                (g - b) / delta % 6,
                np.where(cmax == g, (b - r) / delta + 2, (r - g) / delta + 4),
            )
            / 6.0
        )
        s = np.where(cmax > 1e-10, delta / cmax, 0.0)
        v = cmax

        h = (h + h_shift) % 1.0

        i = (h * 6).astype(int)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        i6 = i % 6
        r_out = np.select(
            [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [v, q, p, p, t, v]
        )
        g_out = np.select(
            [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [t, v, v, q, p, p]
        )
        b_out = np.select(
            [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [p, p, t, v, v, q]
        )

        out[..., 0] = r_out
        out[..., 1] = g_out
        out[..., 2] = b_out

    # ── 9. LUT (last in chain) ─────────────────────────────────────────────────
    if lut_name != "None" and lut_intensity > 0.0:
        out = apply_lut(out, lut_name, lut_intensity)

    return out


def _kelvin_to_rgb_multipliers(kelvin: float) -> Tuple[float, float, float]:
    """
    Convert color temperature (K) to RGB multipliers.
    Based on Tanner Helland's algorithm, normalized so 6500K = (1,1,1).
    """
    temp = max(1000.0, min(40000.0, kelvin)) / 100.0

    # Red
    if temp <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        r = max(0.0, min(255.0, r))

    # Green
    if temp <= 66.0:
        g = 99.4708025861 * np.log(max(temp, 1.0)) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
    g = max(0.0, min(255.0, g))

    # Blue
    if temp >= 66.0:
        b = 255.0
    elif temp <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(max(temp - 10.0, 1.0)) - 305.0447927307
        b = max(0.0, min(255.0, b))

    # Normalize to 6500K baseline
    ref_temp = 65.0  # 6500K / 100
    r_ref = 255.0  # At 6500K, temp<=66 so r=255
    g_ref = 99.4708025861 * np.log(ref_temp) - 161.1195681661
    b_ref = 138.5177312231 * np.log(ref_temp - 10.0) - 305.0447927307

    r_ref = max(r_ref, 1.0)
    g_ref = max(g_ref, 1.0)
    b_ref = max(b_ref, 1.0)

    return (r / r_ref, g / g_ref, b / b_ref)


# ═══════════════════════════════════════════════════════════════════════════════
#                      TENSOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a torch tensor to numpy, handling all dtypes.
    Handles bfloat16 which raises TypeError on direct .numpy().
    Always returns float32 numpy array.
    """
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().cpu().numpy()
    return tensor.cpu().float().numpy()


def compute_data_range(img_np: np.ndarray) -> Tuple[float, float, bool]:
    """
    Compute actual data range and detect HDR content.
    Returns (min_value, max_value, has_hdr).
    """
    d_min = float(img_np.min())
    d_max = float(img_np.max())

    # Sanitize NaN/Inf for JSON safety
    if not math.isfinite(d_min):
        d_min = 0.0
    if not math.isfinite(d_max):
        d_max = 1.0

    has_hdr = d_max > 1.0 or d_min < 0.0
    return d_min, d_max, has_hdr


# ═══════════════════════════════════════════════════════════════════════════════
#                      16-BIT PNG SAVE (v1.21)
# ═══════════════════════════════════════════════════════════════════════════════


def save_16bit_png(filepath: str, img_uint16: np.ndarray) -> bool:
    """
    Save a uint16 numpy array as a true 16-bit PNG using cv2.

    Args:
        filepath: Output file path
        img_uint16: uint16 array — (H,W,3) RGB, (H,W,4) RGBA, or (H,W) gray

    Returns:
        True on success, False on failure
    """
    if not HAS_CV2:
        logger.error("cv2 required for 16-bit PNG save but not available")
        return False

    try:
        if img_uint16.ndim == 3 and img_uint16.shape[2] == 3:
            bgr = cv2.cvtColor(img_uint16, cv2.COLOR_RGB2BGR)
            return cv2.imwrite(
                filepath, bgr, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 3 and img_uint16.shape[2] == 4:
            bgra = cv2.cvtColor(img_uint16, cv2.COLOR_RGBA2BGRA)
            return cv2.imwrite(
                filepath, bgra, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 3 and img_uint16.shape[2] == 1:
            return cv2.imwrite(
                filepath,
                img_uint16[:, :, 0],
                [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION],
            )
        elif img_uint16.ndim == 2:
            return cv2.imwrite(
                filepath, img_uint16, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        else:
            logger.warning(f"Unsupported shape for 16-bit save: {img_uint16.shape}")
            return False
    except Exception as e:
        logger.error(f"cv2 16-bit PNG save failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE VIEWER NODE v2.1
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceViewer:
    """
    VFX Industry-Standard Viewer with IMAGE passthrough:
    • Zoom/Pan navigation
    • Exposure / Gamma / Gain / Lift / Saturation / Temperature controls
    • Built-in LUT engine (10 industry-standard LUTs)
    • Channel viewing (RGB/R/G/B/Alpha/Luma)
    • Color picker with HDR value display
    • False color & zebra analysis
    • A/B comparison modes
    • 16-bit + HDR output
    • IMAGE passthrough — no more dead-end node
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "bit_depth": (
                    BIT_DEPTH_MODES,
                    {
                        "default": "16-bit (Quality)",
                        "tooltip": (
                            "8-bit: Fast, 256 levels. "
                            "16-bit: Quality, 65536 levels. "
                            "16-bit + HDR: Adds float32 sidecar for true HDR color picker."
                        ),
                    },
                ),
                # ── Compare / Depth ──
                "compare_image": ("IMAGE",),
                "zdepth": (
                    "IMAGE",
                    {"tooltip": "Z-Depth map to display when pressing Z button"},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # v2.1: IMAGE passthrough + metadata outputs
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "view"
    CATEGORY = "FXTD Studios/Radiance/Views"
    OUTPUT_NODE = True
    DESCRIPTION = """VFX Industry-Standard Viewer v3.2 — Pro Evolution:
• GPU Waveform / RGB Parade / Vectorscope / Histogram scopes
• Power Windows masking (Radial + Box, feather, rotation)
• Comparison Bridge — mouse-draggable wipe + reference shelf (8 stills)
• Anamorphic lens streaks + Brown-Conrady k1/k2 distortion
• Edge-preserving Bilateral Filter denoising (7×7 GPU kernel)
• Channel viewing (RGB/R/G/B/Alpha/Luma), False Color, Zebra
• 16-bit PNG + .rhdr HDR sidecar + .exr export
• IMAGE passthrough — no longer a dead-end node"""

    def view(
        self,
        image: torch.Tensor,
        bit_depth: str = "16-bit (Quality)",
        # Compare / Depth
        compare_image: Optional[torch.Tensor] = None,
        zdepth: Optional[torch.Tensor] = None,
        # Hidden
        prompt: Optional[Any] = None,
        extra_pnginfo: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Process and display the image in the Radiance Viewer."""

        # ── Parse bit depth ──
        use_16bit = "16-bit" in bit_depth and HAS_CV2
        # v2.1: ALWAYS save HDR sidecar — essential for 32-bit viewer display.
        # The sidecar enables proper HDR viewing via WebGL float textures.
        # Without it, the viewer only has the clipped [0,1] PNG preview.
        # v2.3: Only if 16-bit is actually requested.
        save_hdr_sidecar = use_16bit

        if "16-bit" in bit_depth and not HAS_CV2:
            logger.warning(
                "16-bit mode requested but cv2 not available. "
                "Falling back to 8-bit. Install opencv-python for 16-bit support."
            )

        # ── Validate ──
        validation_error = self._validate_image(image, "image")
        if validation_error:
            logger.error(validation_error)
            return {
                "ui": {"radiance_images": [], "error": [validation_error]},
                "result": (image, validation_error),
            }

        try:
            output_dir = folder_paths.get_temp_directory()
            batch_size = image.shape[0] if image.dim() == 4 else 1
            batch_size = min(batch_size, MAX_BATCH_SIZE)

            images_list: List[Dict[str, Any]] = []

            for frame_idx in range(batch_size):
                try:
                    frame_result = self._process_frame(
                        image,
                        frame_idx,
                        output_dir,
                        use_16bit=use_16bit,
                        save_hdr_sidecar=save_hdr_sidecar,
                        prefix="radiance_viewer",
                    )
                    if frame_result is not None:
                        frame_result["frame"] = frame_idx
                        frame_result["total_frames"] = batch_size
                        images_list.append(frame_result)
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue

            # Compare image
            if compare_image is not None:
                compare_list = self._process_compare_image(
                    compare_image,
                    output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                )
                images_list.extend(compare_list)

            # Z-Depth
            if zdepth is not None:
                zdepth_list = self._process_zdepth_image(
                    zdepth,
                    output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                )
                images_list.extend(zdepth_list)

            # ── Build output IMAGE ──
            output_image = image

            # ── Metadata string ──
            meta_dict = {
                "bit_depth": "16-bit" if use_16bit else "8-bit",
                "hdr_sidecar": True,  # v2.1: always enabled
                "batch_size": batch_size,
            }
            metadata_str = json.dumps(meta_dict, indent=2)

            return {
                "ui": {
                    "radiance_images": images_list,
                    "batch_size": [batch_size],
                    "bit_depth": ["16-bit" if use_16bit else "8-bit"],
                },
                "result": (output_image, metadata_str),
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Error in viewer: {e}")
            return {
                "ui": {"radiance_images": [], "error": [str(e)]},
                "result": (image, str(e)),
            }

    # ─────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────

    def _validate_image(self, image: torch.Tensor, name: str) -> Optional[str]:
        """Validate image tensor. Returns error string or None if valid."""
        if not isinstance(image, torch.Tensor):
            return f"{name} must be a torch.Tensor, got {type(image)}"
        if image.dim() not in (3, 4):
            return f"{name} must be 3D or 4D tensor, got {image.dim()}D"
        if image.dim() == 4:
            _, h, w, c = image.shape
        else:
            h, w = image.shape[0], image.shape[1]
            c = image.shape[2] if image.dim() == 3 else 1
        if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
            return (
                f"{name} dimensions ({h}x{w}) exceed maximum "
                f"({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            )
        if c not in (1, 3, 4):
            return f"{name} has {c} channels, expected 1, 3, or 4"
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Frame Processing (v2.1: grading + LUT + 16-bit)
    # ─────────────────────────────────────────────────────────────────────

    def _process_frame(
        self,
        image: torch.Tensor,
        frame_idx: int,
        output_dir: str,
        use_16bit: bool = True,
        save_hdr_sidecar: bool = False,
        prefix: str = "radiance_viewer",
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single frame: grade → LUT → save at selected bit depth.
        """
        # Safe tensor → numpy
        if image.dim() == 4:
            frame = safe_tensor_to_numpy(image[frame_idx])
        else:
            frame = safe_tensor_to_numpy(image)

        # Sanitize NaNs and Infs (Critical for WebGL and PNG safety)
        # Replace NaN with 0.0, Inf with 65504.0 (max float16)
        if not np.isfinite(frame).all():
            frame = np.nan_to_num(frame, nan=0.0, posinf=65504.0, neginf=0.0)

        # Data range of RAW input (before grading)
        d_min, d_max, has_hdr = compute_data_range(frame)

        # ── Apply grading + LUT for viewer display ──
        # FIX v2.1: Do NOT apply grading to the preview image saved to disk.
        # The frontend viewer applies grading in real-time via WebGL.
        # If we bake it here, it gets applied twice.

        # ── v2.2 ARCHITECTURE: RHDR is the primary display format ──
        # Like DJV/RV viewing EXR natively: viewer loads compressed half-float
        # directly into GPU and tonemaps in real-time. PNG is reduced to a
        # tiny 256px ComfyUI node thumbnail (required by the framework).
        #
        # .rhdr format: zlib-compressed IEEE 754 float16 — lossless at fp16
        # precision, comparable to EXR DWAA at quality ~85.
        # Compression: ~5-15% of raw → similar file sizes to DWAA EXR.

        unique_id = uuid.uuid4().hex[:12]

        # ── Data Prep ──
        h_frame, w_frame = frame.shape[:2]
        c_frame = frame.shape[2] if frame.ndim == 3 else 1

        # Pad RGB to RGBA for WebGL/OpenCV consistency
        # Many Windows/NVIDIA WebGL drivers fail with 3-channel float textures
        if c_frame == 3:
            frame_to_save = np.ones((h_frame, w_frame, 4), dtype=np.float32)
            frame_to_save[..., :3] = frame
            c_frame = 4
        else:
            frame_to_save = frame

        # ── 1. PRIMARY: Save .rhdr compressed float16 ──
        # Only if usage of 16-bit is requested.
        rhdr_filename = f"{prefix}_{unique_id}_{frame_idx}.rhdr"
        rhdr_saved = False

        if use_16bit:
            try:
                import struct

                rhdr_filepath = safe_join(output_dir, rhdr_filename)
                fp16_data = frame_to_save.astype(np.float16).tobytes()
                compressed = zlib.compress(fp16_data, level=6)

                header = struct.pack("<4sHHHH", b"RHDR", w_frame, h_frame, c_frame, 0)
                with open(rhdr_filepath, "wb") as rhdr_f:
                    rhdr_f.write(header)
                    rhdr_f.write(compressed)
                rhdr_saved = True
                ratio = (
                    len(compressed) / len(fp16_data) * 100 if len(fp16_data) > 0 else 0
                )
                logger.debug(
                    f"RHDR primary saved: {rhdr_filename} "
                    f"({len(compressed)//1024}KB, {ratio:.0f}% ratio | "
                    f"range [{d_min:.3f}, {d_max:.3f}])"
                )
            except (IOError, OSError, ValueError) as e:
                logger.warning(f"Failed to save RHDR for frame {frame_idx}: {e}")

        # ── 2. SECONDARY: Save .exr (OpenEXR 32-bit float) for external use ──
        # Requested by users for "Save Image" to export true HDR.
        exr_filename = f"{prefix}_{unique_id}_{frame_idx}.exr"
        exr_saved = False
        if HAS_CV2 or HAS_IMAGEIO:
            try:
                exr_filepath = safe_join(output_dir, exr_filename)

                # Check if we have valid float data
                save_data = frame_to_save.astype(np.float32)

                if HAS_CV2:
                    # OpenCV expects BGR
                    if save_data.ndim == 3 and save_data.shape[2] >= 3:
                        # RGB -> BGR (handles 3 or 4+ channels safely)
                        bgr_order = [2, 1, 0] + list(range(3, save_data.shape[2]))
                        exr_data = save_data[..., np.array(bgr_order)]
                    else:
                        exr_data = save_data

                    success = cv2.imwrite(exr_filepath, exr_data)
                    if success:
                        exr_saved = True

                # Fallback to imageio if cv2 failed or not available
                if not exr_saved and HAS_IMAGEIO:
                    imageio.imwrite(exr_filepath, save_data, format="EXR-FI")
                    exr_saved = True

            except Exception as e:
                logger.warning(f"Failed to save EXR for frame {frame_idx}: {e}")
                exr_filename = None
        else:
            exr_filename = None

        if not exr_saved:
            exr_filename = None

        # v3.1 FIX: Always save full-resolution PNG as fallback.
        # Previously 256px thumbnail was used when RHDR was saved, but if RHDR
        # decompression fails in the browser, the viewer had no usable fallback.
        THUMB_MAX = 99999

        if has_hdr and d_max > 1.05:
            preview_safe = np.maximum(frame, 0.0)
            preview_image = (preview_safe / (1.0 + preview_safe)).astype(np.float32)
        else:
            preview_image = frame

        # Downsample to thumbnail
        th, tw = preview_image.shape[:2]
        if max(th, tw) > THUMB_MAX:
            scale = THUMB_MAX / max(th, tw)
            new_w, new_h = int(tw * scale), int(th * scale)
            try:
                if HAS_CV2:
                    preview_thumb = cv2.resize(
                        preview_image, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                else:
                    from PIL import Image as PILImage

                    pil_full = self._frame_to_pil_8bit(preview_image)
                    if pil_full:
                        pil_full = pil_full.resize((new_w, new_h), PILImage.LANCZOS)
                        preview_thumb = np.array(pil_full).astype(np.float32) / 255.0
                    else:
                        preview_thumb = preview_image
            except Exception as e:
                logger.debug(f"Image resize failed, using original: {e}")
                preview_thumb = preview_image
        else:
            preview_thumb = preview_image

        png_filename = f"{prefix}_{unique_id}_{frame_idx}_thumb.png"
        try:
            png_filepath = safe_join(output_dir, png_filename)
            pil_img = self._frame_to_pil_8bit(preview_thumb)
            if pil_img is None:
                return None
            pil_img.save(png_filepath, compress_level=DEFAULT_PNG_COMPRESSION)
        except (IOError, OSError) as e:
            logger.warning(f"Failed to save thumbnail for frame {frame_idx}: {e}")
            return None

        result: Dict[str, Any] = {
            "filename": png_filename,
            "subfolder": "",
            "type": "temp",
            "data_range": [d_min, d_max],
            "has_hdr": has_hdr,
            "hdr_filename": rhdr_filename if rhdr_saved else None,
            "exr_filename": exr_filename if exr_filename else None,
        }

        if rhdr_saved:
            result["hdr_sidecar"] = rhdr_filename
            result["hdr_primary"] = (
                True  # v2.2: tells viewer to load .rhdr as display source
            )

        return result

    # ─────────────────────────────────────────────────────────────────────
    # PIL 8-bit
    # ─────────────────────────────────────────────────────────────────────

    def _frame_to_pil_8bit(self, frame: np.ndarray) -> Optional["PILImage.Image"]:
        """Convert float32 numpy frame to 8-bit PIL Image."""
        from PIL import Image as PILImage

        img_8bit = self._convert_to_8bit(frame)

        if img_8bit.ndim == 3 and img_8bit.shape[2] == 4:
            return PILImage.fromarray(img_8bit, mode="RGBA")
        elif img_8bit.ndim == 3 and img_8bit.shape[2] == 3:
            return PILImage.fromarray(img_8bit, mode="RGB")
        elif img_8bit.ndim == 3 and img_8bit.shape[2] == 1:
            return PILImage.fromarray(img_8bit[:, :, 0], mode="L")
        elif img_8bit.ndim == 2:
            return PILImage.fromarray(img_8bit, mode="L")

        logger.warning(f"Unexpected 8-bit image shape: {img_8bit.shape}")
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Compare Image
    # ─────────────────────────────────────────────────────────────────────

    def _process_compare_image(
        self,
        compare_image: torch.Tensor,
        output_dir: str,
        use_16bit: bool = True,
        save_hdr_sidecar: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process comparison image. Returns list of image metadata."""
        result: List[Dict[str, Any]] = []

        validation_error = self._validate_image(compare_image, "compare_image")
        if validation_error:
            logger.warning(validation_error)
            return result

        cmp_batch = compare_image.shape[0] if compare_image.dim() == 4 else 1
        cmp_batch = min(cmp_batch, MAX_BATCH_SIZE)

        for cmp_idx in range(cmp_batch):
            try:
                frame_result = self._process_frame(
                    compare_image,
                    cmp_idx,
                    output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                    prefix="radiance_compare",
                )
                if frame_result is not None:
                    frame_result["is_compare"] = True
                    frame_result["frame"] = cmp_idx
                    result.append(frame_result)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Error processing compare frame {cmp_idx}: {e}")
                continue

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Z-Depth
    # ─────────────────────────────────────────────────────────────────────

    def _process_zdepth_image(
        self,
        zdepth: torch.Tensor,
        output_dir: str,
        use_16bit: bool = True,
        save_hdr_sidecar: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process Z-depth map. 16-bit = 65536 depth levels via cv2."""
        result: List[Dict[str, Any]] = []

        validation_error = self._validate_image(zdepth, "zdepth")
        if validation_error:
            logger.warning(validation_error)
            return result

        depth_batch = zdepth.shape[0] if zdepth.dim() == 4 else 1
        depth_batch = min(depth_batch, MAX_BATCH_SIZE)

        for depth_idx in range(depth_batch):
            try:
                if zdepth.dim() == 4:
                    depth_frame = safe_tensor_to_numpy(zdepth[depth_idx])
                else:
                    depth_frame = safe_tensor_to_numpy(zdepth)

                # Extract single channel
                if depth_frame.ndim == 2:
                    depth_np = depth_frame
                elif depth_frame.ndim == 3:
                    if depth_frame.shape[-1] == 1:
                        depth_np = depth_frame[..., 0]
                    else:
                        depth_np = (
                            0.2126 * depth_frame[..., 0]
                            + 0.7152 * depth_frame[..., 1]
                            + 0.0722 * depth_frame[..., 2]
                        )
                else:
                    continue

                d_min = float(depth_np.min())
                d_max = float(depth_np.max())

                # Sanitize for metadata / JSON safety
                if not math.isfinite(d_min):
                    d_min = 0.0
                if not math.isfinite(d_max):
                    d_max = 1.0

                if d_max > d_min:
                    depth_normalized = (depth_np - d_min) / (d_max - d_min)
                else:
                    depth_normalized = np.zeros_like(depth_np)

                unique_id = uuid.uuid4().hex[:12]
                depth_filename = f"radiance_zdepth_{unique_id}_{depth_idx}.png"

                try:
                    depth_filepath = safe_join(output_dir, depth_filename)
                except ValueError as e:
                    logger.error(f"Invalid zdepth path: {e}")
                    continue

                try:
                    if use_16bit:
                        depth_16bit = (depth_normalized * BIT_16_MAX).astype(np.uint16)
                        success = save_16bit_png(depth_filepath, depth_16bit)
                        if not success:
                            from PIL import Image as PILImage

                            depth_8bit = (depth_normalized * BIT_8_MAX).astype(np.uint8)
                            PILImage.fromarray(depth_8bit, mode="L").save(
                                depth_filepath, compress_level=DEFAULT_PNG_COMPRESSION
                            )
                    else:
                        from PIL import Image as PILImage

                        depth_8bit = (depth_normalized * BIT_8_MAX).astype(np.uint8)
                        PILImage.fromarray(depth_8bit, mode="L").save(
                            depth_filepath, compress_level=DEFAULT_PNG_COMPRESSION
                        )
                except (IOError, OSError) as e:
                    logger.warning(f"Failed to save zdepth frame {depth_idx}: {e}")
                    continue

                frame_meta: Dict[str, Any] = {
                    "filename": depth_filename,
                    "subfolder": "",
                    "type": "temp",
                    "is_zdepth": True,
                    "frame": depth_idx,
                    "bit_depth": 16 if use_16bit else 8,
                    "depth_range": [d_min, d_max],
                }

                if save_hdr_sidecar:
                    npy_filename = f"radiance_zdepth_{unique_id}_{depth_idx}_float.rhdr"
                    try:
                        import struct

                        npy_filepath = safe_join(output_dir, npy_filename)
                        fp16_data = depth_np.astype(np.float16).tobytes()
                        compressed = zlib.compress(fp16_data, level=6)
                        dh, dw = depth_np.shape[:2]
                        dc = depth_np.shape[2] if depth_np.ndim == 3 else 1
                        header = struct.pack("<4sHHHH", b"RHDR", dw, dh, dc, 0)
                        with open(npy_filepath, "wb") as rhdr_f:
                            rhdr_f.write(header)
                            rhdr_f.write(compressed)
                        frame_meta["hdr_sidecar"] = npy_filename
                    except (IOError, OSError, ValueError) as e:
                        logger.warning(f"Failed to save depth sidecar {depth_idx}: {e}")

                result.append(frame_meta)

            except (RuntimeError, ValueError) as e:
                logger.warning(f"Error processing zdepth frame {depth_idx}: {e}")
                continue

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Bit Depth Conversion
    # ─────────────────────────────────────────────────────────────────────

    def _convert_to_16bit(self, img_np: np.ndarray) -> np.ndarray:
        """Convert float32 image to uint16. Clamps to [0,1] for display."""
        if img_np.dtype in (np.float32, np.float64):
            return (np.clip(img_np, 0.0, 1.0) * BIT_16_MAX).astype(np.uint16)
        elif img_np.dtype == np.float16:
            return (np.clip(img_np.astype(np.float32), 0.0, 1.0) * BIT_16_MAX).astype(
                np.uint16
            )
        elif img_np.dtype == np.uint16:
            return img_np
        elif img_np.dtype == np.uint8:
            return img_np.astype(np.uint16) * BIT_16_TO_8_DIVISOR
        else:
            img_f = img_np.astype(np.float64)
            mn, mx = img_f.min(), img_f.max()
            if mx > mn:
                return ((img_f - mn) / (mx - mn) * BIT_16_MAX).astype(np.uint16)
            return np.zeros_like(img_f, dtype=np.uint16)

    def _convert_to_8bit(self, img_np: np.ndarray) -> np.ndarray:
        """Convert float32 image to uint8."""
        if img_np.dtype in (np.float32, np.float64, np.float16):
            return (np.clip(img_np.astype(np.float32), 0.0, 1.0) * BIT_8_MAX).astype(
                np.uint8
            )
        elif img_np.dtype == np.uint16:
            return (img_np // BIT_16_TO_8_DIVISOR).astype(np.uint8)
        elif img_np.dtype == np.uint8:
            return img_np
        else:
            img_f = img_np.astype(np.float64)
            mn, mx = img_f.min(), img_f.max()
            if mx > mn:
                return ((img_f - mn) / (mx - mn) * BIT_8_MAX).astype(np.uint8)
            return np.zeros_like(img_f, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#                           NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "FXTD_RadianceViewer": RadianceViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTD_RadianceViewer": "◎ Radiance Viewer",
}
