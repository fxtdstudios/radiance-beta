"""
═══════════════════════════════════════════════════════════════════════════════
                    Radiance - Shared Color Utilities
                    Professional Color Space Conversions
═══════════════════════════════════════════════════════════════════════════════

Centralized color space conversion functions used across Radiance nodes.
All functions operate on numpy float32 arrays and torch tensors.
"""

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
#                          COLOR SPACE MATRICES
# ═══════════════════════════════════════════════════════════════════════════════

# sRGB (Rec.709) to ACEScg (AP1)
SRGB_TO_ACESCG = np.array(
    [
        [0.613097, 0.339523, 0.047379],
        [0.070194, 0.916354, 0.013452],
        [0.020616, 0.109570, 0.869815],
    ],
    dtype=np.float32,
)

# ACEScg (AP1) to sRGB (Rec.709)
ACESCG_TO_SRGB = np.array(
    [
        [1.7050509, -0.6217921, -0.0832588],
        [-0.1302564, 1.1408047, -0.0105483],
        [-0.0240033, -0.1289690, 1.1529723],
    ],
    dtype=np.float32,
)

# ACES AP0 (2065-1) to AP1 (ACEScg)
ACES_AP0_TO_AP1 = np.array(
    [
        [1.4514393161, -0.2365107469, -0.2149285693],
        [-0.0765537734, 1.1762296998, -0.0996759264],
        [0.0083161484, -0.0060324498, 0.9977163014],
    ],
    dtype=np.float32,
)

# ACEScg (AP1) to Rec.2020
ACESCG_TO_REC2020 = np.array(
    [
        [1.0258246, -0.0200540, -0.0057706],
        [-0.0023054, 1.0045847, -0.0022793],
        [-0.0050569, -0.0252857, 1.0303426],
    ],
    dtype=np.float32,
)

# ACEScg (AP1) to P3-D65
ACESCG_TO_P3D65 = np.array(
    [
        [1.3792141, -0.3088546, -0.0703595],
        [-0.0693257, 1.0823507, -0.0130250],
        [-0.0021522, -0.0454616, 1.0476138],
    ],
    dtype=np.float32,
)

# ARRI Wide Gamut 3 (LogC3) to ACEScg
# Source: ARRI White Paper / colour-science
AWG3_TO_ACESCG = np.array(
    [
        [0.672418, 0.153626, 0.173956],
        [0.005363, 1.006965, -0.012328],
        [-0.082291, -0.038573, 1.120864],
    ],
    dtype=np.float32,
)

# ARRI Wide Gamut 4 (LogC4) to ACEScg
# Source: ARRI LogC4 Specification
AWG4_TO_ACESCG = np.array(
    [
        [0.725807, 0.158580, 0.115613],
        [0.021028, 0.957640, 0.021332],
        [-0.003504, 0.001925, 1.001579],
    ],
    dtype=np.float32,
)

# Sony S-Gamut3.Cine to ACEScg
# Common choice for S-Log3
SGAMUT3_CINE_TO_ACESCG = np.array(
    [
        [0.599084, 0.292670, 0.108247],
        [0.063251, 0.893690, 0.043059],
        [0.016334, 0.077673, 0.905993],
    ],
    dtype=np.float32,
)

# Panasonic V-Gamut to ACEScg
VGAMUT_TO_ACESCG = np.array(
    [
        [0.627256, 0.280459, 0.092285],
        [0.031267, 0.923483, 0.045250],
        [-0.026490, 0.062086, 0.964404],
    ],
    dtype=np.float32,
)

# Canon Cinema Gamut to ACEScg
CINEMA_GAMUT_TO_ACESCG = np.array(
    [
        [0.686567, 0.245239, 0.068194],
        [0.041019, 0.928738, 0.030243],
        [0.003595, 0.053802, 0.942603],
    ],
    dtype=np.float32,
)

# REDWideGamutRGB to ACEScg
REDWIDEGAMUT_TO_ACESCG = np.array(
    [[0.5510, 0.2114, 0.2376], [0.0089, 0.9996, -0.0086], [-0.0886, -0.0583, 1.1470]],
    dtype=np.float32,
)

# DaVinci Wide Gamut to ACEScg
DAVINCI_WIDE_TO_ACESCG = np.array(
    [
        [0.718049, 0.237248, 0.044703],
        [0.055964, 0.925547, 0.018489],
        [0.088324, 0.056073, 0.855604],
    ],
    dtype=np.float32,
)


# ═══════════════════════════════════════════════════════════════════════════════
#                          GAMMA / TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear color space (numpy)."""
    return np.where(
        img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0) + 0.055) / 1.055, 2.4)
    ).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB color space (numpy)."""
    return np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 1e-10), 1 / 2.4) - 0.055,
    ).astype(np.float32)


def tensor_srgb_to_linear(tensor: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Convert sRGB tensor to linear color space (GPU-compatible).

    HDR FIX: Previous implementation used tensor.clamp(min=0) inside the pow()
    branch, which silently destroyed all values above 1.0 — correct for SDR but
    wrong for HDR pipelines where decoded values legitimately exceed 1.0
    (Soft Clip recovery, Passthrough, scene-linear highlights).

    Fix: sign-preserving absolute-value extension so the EOTF is continuous
    and monotonic across the full float32 range, matching IEC 61966-2-1 for
    [0, 1] and extending smoothly outside. Identical to _safe_srgb_to_linear_extended
    in vae.py — consolidated here so all Radiance nodes share one implementation.
    """
    if gamma == 2.2:
        # IEC 61966-2-1 sRGB EOTF — extended to full float32 range
        abs_t = tensor.abs()
        sign  = tensor.sign()
        low   = abs_t / 12.92
        high  = torch.pow((abs_t + 0.055) / 1.055, 2.4)
        return torch.where(abs_t <= 0.04045, low, high) * sign
    else:
        # Simple gamma — sign-preserving for HDR symmetry
        abs_t = tensor.abs()
        return torch.pow(abs_t.clamp(min=1e-10), gamma) * tensor.sign()


def tensor_linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    """Convert linear tensor to sRGB color space (GPU-compatible)."""
    low = tensor * 12.92
    high = 1.055 * torch.pow(tensor.clamp(min=1e-10), 1 / 2.4) - 0.055
    return torch.where(tensor <= 0.0031308, low, high)


# ═══════════════════════════════════════════════════════════════════════════════
#                          COLOR SPACE CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════════


def linear_srgb_to_acescg(img: np.ndarray) -> np.ndarray:
    """Convert Linear sRGB (Rec.709) to ACEScg (AP1)."""
    return np.dot(img, SRGB_TO_ACESCG.T)


def acescg_to_linear_srgb(img: np.ndarray) -> np.ndarray:
    """Convert ACEScg (AP1) to Linear sRGB (Rec.709)."""
    return np.dot(img, ACESCG_TO_SRGB.T)


def apply_matrix_transform(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 color matrix to an image (H, W, 3)."""
    return np.dot(img, matrix.T)


# ═══════════════════════════════════════════════════════════════════════════════
#                          LOG ENCODING CURVES
# ═══════════════════════════════════════════════════════════════════════════════

# ARRI LogC3 Exposure Index (EI) Constants
# Source: ARRI LogC Curve Usage whitepaper
LOGC3_EI_PARAMS = {
    # EI: (cut, a, b, c, d, e, f)
    160: (0.005561, 5.061087, 0.089004, 0.269035, 0.391007, 6.332427, 0.108361),
    200: (0.006208, 5.168208, 0.076621, 0.265275, 0.391007, 5.842037, 0.099519),
    250: (0.006871, 5.282072, 0.065521, 0.261620, 0.391007, 5.397270, 0.091111),
    320: (0.007622, 5.399335, 0.055194, 0.257766, 0.391007, 4.969419, 0.083295),
    400: (0.008318, 5.510883, 0.046585, 0.254174, 0.391007, 4.606965, 0.076257),
    500: (0.009031, 5.618393, 0.039023, 0.250758, 0.391007, 4.282556, 0.069776),
    640: (0.009840, 5.737055, 0.031538, 0.247070, 0.391007, 3.946374, 0.063409),
    800: (
        0.010591,
        5.555556,
        0.052272,
        0.247190,
        0.385537,
        5.367655,
        0.092809,
    ),  # Standard
    1000: (0.011361, 5.944966, 0.019018, 0.240020, 0.391007, 3.369506, 0.051759),
    1280: (0.012235, 6.056760, 0.013804, 0.236500, 0.391007, 3.088156, 0.046447),
    1600: (0.013047, 6.161541, 0.009677, 0.233182, 0.391007, 2.852200, 0.041773),
    2000: (0.013901, 6.260724, 0.006210, 0.230014, 0.391007, 2.643126, 0.037413),
    2560: (0.014842, 6.362496, 0.002995, 0.226764, 0.391007, 2.440085, 0.033198),
    3200: (0.015711, 6.456037, 0.000295, 0.223740, 0.391007, 2.265605, 0.029493),
}


def linear_to_logc3(img: np.ndarray, ei: int = 800) -> np.ndarray:
    """
    ARRI LogC3 encoding with Exposure Index support.

    Args:
        img: Linear image data
        ei: Exposure Index (160-3200, default 800)

    Returns:
        LogC3 encoded image
    """
    # Get EI params, fallback to 800
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params

    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut
    out[mask] = c * np.log10(a * img[mask] + b) + d
    out[~mask] = e * img[~mask] + f
    return out


def logc3_to_linear(img: np.ndarray, ei: int = 800) -> np.ndarray:
    """
    ARRI LogC3 decoding with Exposure Index support.

    Args:
        img: LogC3 encoded image
        ei: Exposure Index (160-3200, default 800)

    Returns:
        Linear image data
    """
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    cut_encoded = e * cut + f

    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut_encoded
    out[mask] = (np.power(10.0, (img[mask] - d) / c) - b) / a
    out[~mask] = (img[~mask] - f) / e
    return out


def linear_to_logc4(img: np.ndarray) -> np.ndarray:
    """
    ARRI LogC4 encoding (ALEXA 35).
    Approximate implementation fitted to spec:
    - Black (0.0) -> 0.0
    - Mid Gray (0.18) -> 0.32
    - Peak White (~46.0) -> 1.0
    """
    # Constants optimized for continuity and key values
    A = 4296.65
    D = 11.593

    # Pure logarithmic curve (no linear toe needed for standard dynamic range)
    # y = (log2(A * x + 64) - 6) / D

    # Handle negative values safely?
    # LogC4 is technically defined for negative values via linear toe,
    # but for this implementation we clamp or handle similarly.
    # Since A is large, A*x will dominate.
    # Minimum valid input for log2 is -64/A = -0.014...

    # Let's use a small linear toe for very dark/negative values to avoid NaN
    cut = 0.0  # Use pure log for positive

    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut

    # Log segment
    # log2(A * x + 64) - 6
    # Note: 6 is log2(64). So this is log2((A*x+64)/64) ?
    # Yes -> log2(A/64 * x + 1).
    # A/64 = 4296.65 / 64 = 67.135

    # Using the fitted params directly:
    out[mask] = (np.log2(A * img[mask] + 64.0) - 6.0) / D

    # Linear segment for <= 0
    # Slope at 0: derivative of log form
    # dy/dx = 1/(D*ln(2)) * A/(A*0 + 64) = A / (D * ln(2) * 64)
    # = 4296.65 / (11.593 * 0.693 * 64) = 4296 / 514 = ~8.35

    slope = A / (D * np.log(2) * 64.0)
    out[~mask] = img[~mask] * slope

    return out


def logc4_to_linear(img: np.ndarray) -> np.ndarray:
    """ARRI LogC4 decoding (ALEXA 35)."""
    # Inverse of encoding
    A = 4296.65
    D = 11.593
    slope = A / (D * np.log(2) * 64.0)

    out = np.empty_like(img, dtype=np.float32)

    # Threshold is 0 (since cut was 0)
    # But checking if encoded value is > 0
    mask = img > 0.0

    # x = (2^(y * D + 6) - 64) / A
    out[mask] = (np.power(2.0, img[mask] * D + 6.0) - 64.0) / A

    # Linear
    out[~mask] = img[~mask] / slope

    return out


def linear_to_slog3(img: np.ndarray) -> np.ndarray:
    """Sony S-Log3 encoding."""
    cut = 0.011250
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut
    out[mask] = (420.0 + np.log10((img[mask] + 0.01) / 0.19) * 261.5) / 1023.0
    out[~mask] = (img[~mask] + 0.01) * (4.0 * 261.5) / 1023.0 + (95.0 / 1023.0)
    return out


def slog3_to_linear(img: np.ndarray) -> np.ndarray:
    """Sony S-Log3 decoding."""
    cut_v = 171.2102946929 / 1023.0
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut_v
    out[mask] = 0.19 * np.power(10.0, (img[mask] * 1023.0 - 420.0) / 261.5) - 0.01
    out[~mask] = (img[~mask] * 1023.0 - 95.0) / (4.0 * 261.5) - 0.01
    return out


def linear_to_vlog(img: np.ndarray) -> np.ndarray:
    """Panasonic V-Log encoding."""
    cut = 0.01
    b, c, d = 0.00873, 0.241514, 0.598206
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut
    out[mask] = c * np.log10(img[mask] + b) + d
    out[~mask] = 5.625 * img[~mask] + 0.125
    return out


def vlog_to_linear(img: np.ndarray) -> np.ndarray:
    """
    Panasonic V-Log decoding.

    Inverse of V-Log encoding for Panasonic VariCam, S1H, GH6.
    """
    # Use same constants as encoder for perfect roundtrip
    b, c, d = 0.00873, 0.241514, 0.598206
    cut_linear = 0.01
    # Calculate the encoded cut point: c * log10(cut + b) + d
    cut_encoded = c * np.log10(cut_linear + b) + d  # Should be ~0.181

    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut_encoded
    # Inverse log: x = 10^((y - d) / c) - b
    out[mask] = np.power(10.0, (img[mask] - d) / c) - b
    # Inverse linear: x = (y - 0.125) / 5.625
    out[~mask] = (img[~mask] - 0.125) / 5.625
    return out


def linear_to_canonlog3(img: np.ndarray) -> np.ndarray:
    """
    Canon Log 3 encoding.

    Official Canon Specification (Canon Log Gamma Curves whitepaper):
    - 18% gray → 0.343 (34.3 IRE)
    - Base ISO: 800
    - Dynamic range optimized for cinema production

    Formula: For x >= cut:
      y = c * log10(a * x + 1) + d
    For x < cut:
      y = e * x + f (linear toe)
    """
    # Canon Log 3 constants (from Canon specification)
    # These are derived to place 18% gray at 0.343
    cut = 0.014
    a = 14.98325  # Adjusted for correct 18% gray mapping
    c = 0.36726845  # Log coefficient
    d = 0.12783901  # Offset
    e = 5.449285  # Linear slope
    f = 0.073059361  # Linear offset

    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut
    out[mask] = c * np.log10(a * img[mask] + 1.0) + d
    out[~mask] = e * img[~mask] + f
    return out


def canonlog3_to_linear(img: np.ndarray) -> np.ndarray:
    """
    Canon Log 3 decoding.

    Inverse of Canon Log 3 encoding. Properly restores linear values
    for HDR workflows and color grading.
    """
    # Same constants as encoding
    cut_encoded = 0.14926  # Encoded value at linear cut point
    a = 14.98325
    c = 0.36726845
    d = 0.12783901
    e = 5.449285
    f = 0.073059361

    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut_encoded
    # Inverse: x = (10^((y - d) / c) - 1) / a
    out[mask] = (np.power(10.0, (img[mask] - d) / c) - 1.0) / a
    # Linear segment inverse: x = (y - f) / e
    out[~mask] = (img[~mask] - f) / e
    return out


def linear_to_log3g10(img: np.ndarray) -> np.ndarray:
    """
    RED Log3G10 encoding.

    REDLogFilm / Log3G10 for RED cameras (DSMC2, Komodo, V-Raptor).
    Maps 18% gray to 1/3 code value (0.333).
    """
    # Log3G10 constants (from RED tech specs)
    a = 0.224282  # log coefficient
    b = 155.975327  # linear scale
    c = 0.01  # offset
    cut = 0.01

    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut
    # Log region: y = a * log10(b * x + 1) + c
    out[mask] = a * np.log10(b * img[mask] + 1.0) + c
    # Linear toe for low values
    out[~mask] = (img[~mask] / cut) * (a * np.log10(b * cut + 1.0) + c)
    return out


def log3g10_to_linear(img: np.ndarray) -> np.ndarray:
    """
    RED Log3G10 decoding.

    Inverse of Log3G10 for RED camera footage.
    """
    a = 0.224282
    b = 155.975327
    c = 0.01
    cut = 0.01
    cut_encoded = a * np.log10(b * cut + 1.0) + c

    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut_encoded
    # Inverse log: x = (10^((y - c) / a) - 1) / b
    out[mask] = (np.power(10.0, (img[mask] - c) / a) - 1.0) / b
    # Inverse linear toe
    out[~mask] = img[~mask] / cut_encoded * cut
    return out


def linear_to_acescct(img: np.ndarray) -> np.ndarray:
    """ACEScct encoding (ACES Central Transform)."""
    cut = 0.0078125
    a, b = 10.5402377416545, 0.0729055341958355
    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut
    out[mask] = (np.log2(img[mask]) + 9.72) / 17.52
    out[~mask] = a * img[~mask] + b
    return out


def acescct_to_linear(img: np.ndarray) -> np.ndarray:
    """ACEScct decoding."""
    cut_encoded = 0.155251141552511
    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut_encoded
    out[mask] = np.power(2.0, img[mask] * 17.52 - 9.72)
    out[~mask] = (img[~mask] - 0.0729055341958355) / 10.5402377416545
    return out


def linear_to_davinci_intermediate(img: np.ndarray) -> np.ndarray:
    """DaVinci Intermediate encoding."""
    A, B, C = 0.0075, 7.0, 0.07329248
    cut = 0.00262409
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut
    out[mask] = np.log10(img[mask] + A) * C + 0.5
    out[~mask] = B * img[~mask] + 0.07329248
    return out


def davinci_intermediate_to_linear(img: np.ndarray) -> np.ndarray:
    """DaVinci Intermediate decoding."""
    A, C = 0.0075, 0.07329248
    cut_v = 0.07329248
    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut_v
    out[mask] = np.power(10.0, (img[mask] - 0.5) / C) - A
    out[~mask] = (img[~mask] - 0.07329248) / 7.0
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#                          HDR TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def linear_to_pq(img: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    """Encode to ST.2084 PQ (Perceptual Quantizer)."""
    L = np.clip(img * peak_nits / 10000.0, 0, 1)
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = np.power(L, m1)
    return np.power((c1 + c2 * Lm1) / (1 + c3 * Lm1), m2)


def pq_to_linear(img: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    """Decode from ST.2084 PQ to linear."""
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Vm2 = np.power(np.maximum(img, 0), 1 / m2)
    return (
        np.power(np.maximum(Vm2 - c1, 0) / (c2 - c3 * Vm2), 1 / m1)
        * 10000.0
        / peak_nits
    )


def linear_to_hlg(img: np.ndarray) -> np.ndarray:
    """Encode to ARIB STD-B67 HLG (Hybrid Log-Gamma)."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    out = np.where(
        img <= 1 / 12,
        np.sqrt(3 * np.maximum(img, 0)),
        a * np.log(np.maximum(12 * img - b, 1e-10)) + c,
    )
    return np.clip(out, 0, 1)


def hlg_to_linear(img: np.ndarray) -> np.ndarray:
    """Decode from HLG to linear."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    out = np.where(img <= 0.5, (img**2) / 3, (np.exp((img - c) / a) + b) / 12)
    return np.maximum(out, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#                          ACES 2.0 FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def aces_approx_tonemap(
    img: np.ndarray,
    peak_luminance: float = 1000.0,
    mid_gray: float = 0.18,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    Approximate filmic tone mapping curve inspired by ACES aesthetics.

    NOTE: This is NOT the ACES 2.0 Reference Rendering Transform (RRT).
    It uses a Michaelis-Menten (Naka-Rushton) S-curve that produces
    ACES-like results but is not spec-compliant. For true ACES RRT output
    use PyOpenColorIO with an ACES config (RadianceOCIODisplayView).

    Args:
        img: Linear ACEScg image (float32)
        peak_luminance: Display peak luminance in nits (default 1000 for HDR)
        mid_gray: Scene mid-gray value (default 0.18)
        contrast: Contrast adjustment (1.0 = default)

    Returns:
        Tone-mapped image ready for display encoding
    """
    # Normalize to display range based on peak luminance
    # ACES uses 48 nits as reference diffuse white
    display_scale = 48.0 / peak_luminance

    # ACES 2.0 uses a modified Michaelis-Menten (Naka-Rushton) curve
    # Simplified form: y = (x^p) / (x^p + 1) where p controls contrast
    p = 1.2 * contrast

    # Prevent division by zero and handle negative values
    img_safe = np.maximum(img, 1e-10)

    # Apply per-channel tone curve
    img_pow = np.power(img_safe / mid_gray, p)
    tonemapped = img_pow / (img_pow + 1.0)

    # Scale to display range
    result = tonemapped * display_scale

    # Soft highlight rolloff (prevents hard clipping)
    highlight_threshold = 0.9
    highlight_mask = result > highlight_threshold
    if np.any(highlight_mask):
        excess = result[highlight_mask] - highlight_threshold
        result[highlight_mask] = highlight_threshold + 0.1 * np.tanh(excess * 10)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# Backward-compatibility alias — old name was misleadingly labelled "ACES 2.0"
aces2_tonemap = aces_approx_tonemap


def aces2_gamut_compress(
    img: np.ndarray, threshold: float = 0.75, power: float = 1.2
) -> np.ndarray:
    """
    ACES 2.0 perceptual gamut compression.

    Compresses out-of-gamut colors smoothly back into the target gamut
    while preserving hue and relative colorfulness.

    Args:
        img: Linear image in target gamut
        threshold: Where compression starts (0.75 = 75% of gamut boundary)
        power: Compression curve power (higher = more aggressive)

    Returns:
        Gamut-compressed image
    """
    # Calculate per-pixel luminance (Rec.709 coefficients for sRGB gamut)
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    luma = np.maximum(luma, 1e-10)[..., np.newaxis]

    # Calculate distance from achromatic (saturation-like metric)
    rgb_norm = img / luma
    distance = np.max(np.abs(rgb_norm - 1.0), axis=-1, keepdims=True)

    # Compress distances beyond threshold
    compress_mask = distance > threshold
    if np.any(compress_mask):
        excess = (distance - threshold) / (1.0 - threshold + 1e-10)
        compressed_distance = threshold + (1.0 - threshold) * (
            1.0 - np.power(1.0 - excess, power)
        )

        # Apply compression by scaling saturation
        scale = np.where(compress_mask, compressed_distance / (distance + 1e-10), 1.0)
        img_compressed = luma + (img - luma) * scale
    else:
        img_compressed = img

    return img_compressed.astype(np.float32)


def linear_to_jmh(img: np.ndarray, white_point: np.ndarray = None) -> np.ndarray:
    """
    Convert linear RGB to simplified JMh (Lightness, Colorfulness, Hue) space.

    This is inspired by CIECAM16 but simplified for real-time processing.
    Used in ACES 2.0 for perceptual uniformity in grading operations.

    Args:
        img: Linear RGB image (ACEScg or sRGB primaries)
        white_point: Adaptation white point (default D65)

    Returns:
        JMh array where:
        - J: Lightness (0-100 scale)
        - M: Colorfulness (unbounded positive)
        - h: Hue angle in degrees (0-360)
    """
    if white_point is None:
        white_point = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)  # D65

    # Simplified Hunt-Pointer-Estevez transform (RGB to LMS-like)
    M_hpe = np.array(
        [
            [0.38971, 0.68898, -0.07868],
            [-0.22981, 1.18340, 0.04641],
            [0.00000, 0.00000, 1.00000],
        ],
        dtype=np.float32,
    )

    lms = np.dot(img, M_hpe.T)
    lms = np.maximum(lms, 1e-10)

    # Chromatic adaptation (simplified von Kries)
    lms_w = np.dot(white_point, M_hpe.T)
    lms_adapted = lms / lms_w

    # Power compression for perceptual uniformity
    lms_c = np.power(lms_adapted, 0.42)

    # Calculate J (lightness)
    J = 100.0 * ((lms_c[..., 0] + lms_c[..., 1]) / 2.0)

    # Calculate a, b opponent channels
    a = lms_c[..., 0] - lms_c[..., 1]
    b = 0.5 * (lms_c[..., 0] + lms_c[..., 1]) - lms_c[..., 2]

    # Calculate M (colorfulness) and h (hue)
    M = np.sqrt(a**2 + b**2) * 100.0
    h = np.degrees(np.arctan2(b, a)) % 360.0

    return np.stack([J, M, h], axis=-1).astype(np.float32)


def jmh_to_linear(jmh: np.ndarray, white_point: np.ndarray = None) -> np.ndarray:
    """
    Convert JMh back to linear RGB.

    Inverse of linear_to_jmh for round-trip color operations.

    Args:
        jmh: JMh array (Lightness, Colorfulness, Hue)
        white_point: Adaptation white point (default D65)

    Returns:
        Linear RGB image
    """
    if white_point is None:
        white_point = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

    J, M, h = jmh[..., 0], jmh[..., 1], jmh[..., 2]

    # Reconstruct a, b from M, h
    h_rad = np.radians(h)
    a = M / 100.0 * np.cos(h_rad)
    b = M / 100.0 * np.sin(h_rad)

    # Reconstruct LMS from J, a, b
    lms_sum = J / 100.0 * 2.0
    lms_c_0 = (lms_sum + a) / 2.0
    lms_c_1 = (lms_sum - a) / 2.0
    lms_c_2 = lms_sum / 2.0 - b

    lms_c = np.stack([lms_c_0, lms_c_1, lms_c_2], axis=-1)

    # Reverse power compression
    lms_adapted = np.power(np.maximum(lms_c, 1e-10), 1.0 / 0.42)

    # Reverse chromatic adaptation
    M_hpe = np.array(
        [
            [0.38971, 0.68898, -0.07868],
            [-0.22981, 1.18340, 0.04641],
            [0.00000, 0.00000, 1.00000],
        ],
        dtype=np.float32,
    )
    lms_w = np.dot(white_point, M_hpe.T)
    lms = lms_adapted * lms_w

    # Inverse HPE transform
    M_hpe_inv = np.linalg.inv(M_hpe)
    rgb = np.dot(lms, M_hpe_inv.T)

    return np.maximum(rgb, 0.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#                          TENSOR/NUMPY CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════


def tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy float32 array."""
    return tensor.detach().cpu().numpy().astype(np.float32)


def numpy_to_tensor_float32(array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor."""
    return torch.from_numpy(array.astype(np.float32))

def savePng16(filepath, image_data):
    """
    Save 16-bit PNG image using cv2.
    Provided for backwards compatibility with existing workflows.
    """
    try:
        import cv2
        # Ensure data is numpy array [0,1] float or [0,65535] uint16
        if isinstance(image_data, torch.Tensor):
            image_data = image_data.detach().cpu().numpy()
        
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = np.clip(image_data, 0, 1)
            image_data = (image_data * 65535).astype(np.uint16)
        
        # Convert RGB to BGR for cv2
        if image_data.shape[-1] == 3:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        elif image_data.shape[-1] == 4:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGRA)
            
        cv2.imwrite(filepath, image_data)
        return True
    except Exception as e:
        import logging
        logging.getLogger("radiance.color_utils").error(f"Failed to save 16-bit PNG: {e}")
        return False



# ═══════════════════════════════════════════════════════════════════════════════
#                          GPU-ACCELERATED LOG CURVES (TORCH)
# ═══════════════════════════════════════════════════════════════════════════════


def aces_tonemap(img: np.ndarray) -> np.ndarray:
    """
    Narkowicz-style ACES approximate filmic tonemapper.
    Matches the GLSL implementation in Radiance Viewer.
    Assumes image is scene-linear 18% grey = 0.18.
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    # Apply curve: (x * (ax + b)) / (x * (cx + d) + e)
    res = (img * (a * img + b)) / (img * (c * img + d) + e)
    return np.clip(res, 0.0, 1.0)


def tensor_linear_to_logc3(tensor: torch.Tensor, ei: int = 800) -> torch.Tensor:
    """
    GPU-accelerated ARRI LogC3 encoding.

    Args:
        tensor: Linear image tensor (B, H, W, C) or (H, W, C)
        ei: Exposure Index (160-3200)

    Returns:
        LogC3 encoded tensor
    """
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params

    log_val = c * torch.log10(a * tensor + b) + d
    lin_val = e * tensor + f
    return torch.where(tensor > cut, log_val, lin_val)


def tensor_logc3_to_linear(tensor: torch.Tensor, ei: int = 800) -> torch.Tensor:
    """
    GPU-accelerated ARRI LogC3 decoding.
    """
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    cut_encoded = e * cut + f

    log_val = (torch.pow(10.0, (tensor - d) / c) - b) / a
    lin_val = (tensor - f) / e
    return torch.where(tensor > cut_encoded, log_val, lin_val)


def tensor_linear_to_logc4(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated ARRI LogC4 encoding (Alexa 35)."""
    A = 4296.65
    D = 11.593
    slope = A / (D * 0.693147 * 64.0)  # ln(2) ≈ 0.693147

    log_val = (torch.log2(A * tensor + 64.0) - 6.0) / D
    lin_val = tensor * slope
    return torch.where(tensor > 0, log_val, lin_val)


def tensor_logc4_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated ARRI LogC4 decoding."""
    A = 4296.65
    D = 11.593
    slope = A / (D * 0.693147 * 64.0)

    log_val = (torch.pow(2.0, tensor * D + 6.0) - 64.0) / A
    lin_val = tensor / slope
    return torch.where(tensor > 0, log_val, lin_val)


def tensor_linear_to_slog3(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated Sony S-Log3 encoding.

    Official Sony S-Log3 spec (S-Log3 Technical Note v1.1):
      For x >= 0.011250: y = (420 + log10((x + 0.01) / 0.19) × 261.5) / 1023
      For x  < 0.011250: y = (x × (171.2102946929 − 95) / 0.01125 + 95) / 1023

    BUG-1 FIX: Previous tensor code had operator-precedence error:
      log10(tensor + 0.01) / 0.19 * 261.5   ← divides the LOG RESULT by 0.19
    Correct form requires 0.19 INSIDE the log argument:
      log10((tensor + 0.01) / 0.19) * 261.5  ← divides the SCENE VALUE inside log
    At 18% grey (x=0.18) this was producing −0.56 (negative log!) instead of 0.41.
    """
    cut = 0.011250
    log_val = (420.0 + torch.log10((tensor + 0.01) / 0.19) * 261.5) / 1023.0
    lin_val = (tensor + 0.01) * (171.2102946929 - 95.0) / (0.01125 * 1023.0) + (95.0 / 1023.0)
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_slog3_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated Sony S-Log3 decoding."""
    cut_v = 171.2102946929 / 1023.0
    log_val = 0.19 * torch.pow(10.0, (tensor * 1023.0 - 420.0) / 261.5) - 0.01
    lin_val = (tensor * 1023.0 - 95.0) / (4.0 * 261.5) - 0.01
    return torch.where(tensor >= cut_v, log_val, lin_val)


def tensor_linear_to_log3g10(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated RED Log3G10 encoding.

    BUG-4 FIX: Previous code used torch.tensor(b * cut + 1.0) to compute
    cut_encoded, creating a CPU scalar tensor inside a GPU function. On CUDA
    this triggers a device-mismatch in torch.where. Replaced with a
    pre-computed Python float constant.

    Pre-computed: a × log10(b × cut + 1) + c
                = 0.224282 × log10(155.975327 × 0.01 + 1) + 0.01
                = 0.224282 × log10(2.55975327) + 0.01
                ≈ 0.101551
    """
    a, b, c = 0.224282, 155.975327, 0.01
    cut = 0.01
    # Pre-computed encode(cut) — avoids torch.tensor() CPU allocation inside GPU fn
    cut_encoded = 0.101551  # a * log10(b * cut + 1) + c

    log_val = a * torch.log10(b * tensor + 1.0) + c
    lin_val = (tensor / cut) * cut_encoded
    return torch.where(tensor > cut, log_val, lin_val)


def tensor_log3g10_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated RED Log3G10 decoding.

    BUG-3 FIX: Previous code used cut_encoded = a × 0.50514998 ≈ 0.1133.
    The correct threshold is encode(cut_linear) = a × log10(b × cut + 1) + c ≈ 0.1016.
    Using 0.1133 caused values in [0.1016, 0.1133] — encoded by the LOG formula —
    to be decoded with the LINEAR formula, creating a discontinuity at the cut point.

    Pre-computed: a × log10(155.975327 × 0.01 + 1) + c
                = 0.224282 × log10(2.55975327) + 0.01
                ≈ 0.101551
    """
    a, b, c = 0.224282, 155.975327, 0.01
    cut = 0.01
    # Correct cut in the encoded (Log3G10) domain — encode(0.01)
    cut_encoded = 0.101551  # a * log10(b * cut + 1) + c

    log_val = (torch.pow(10.0, (tensor - c) / a) - 1.0) / b
    lin_val = tensor / cut_encoded * cut
    return torch.where(tensor > cut_encoded, log_val, lin_val)


def tensor_linear_to_vlog(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated Panasonic V-Log encoding."""
    cut = 0.01
    b, c, d = 0.00873, 0.241514, 0.598206

    log_val = c * torch.log10(tensor + b) + d
    lin_val = 5.625 * tensor + 0.125
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_vlog_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated Panasonic V-Log decoding.

    BUG-2 FIX: Previous code used cut_encoded = c * 0.33893 ≈ 0.0819.
    The correct threshold is encode(cut_linear) = c × log10(cut + b) + d ≈ 0.1810.
    Using 0.0819 caused values in the range [0.0819, 0.1810] — encoded by the LOG
    formula — to be decoded with the LINEAR formula, producing a visible kink in
    the dark/midtone range of any V-Log source.

    Pre-computed: c × log10(0.01 + 0.00873) + d
                = 0.241514 × log10(0.01873) + 0.598206
                = 0.241514 × (−1.72757) + 0.598206
                ≈ 0.181000
    """
    b, c, d = 0.00873, 0.241514, 0.598206
    # Correct cut in the encoded (V-Log) domain — encode(0.01)
    cut_encoded = 0.181000  # c * log10(0.01 + b) + d

    log_val = torch.pow(10.0, (tensor - d) / c) - b
    lin_val = (tensor - 0.125) / 5.625
    return torch.where(tensor >= cut_encoded, log_val, lin_val)


def tensor_linear_to_davinci_intermediate(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated DaVinci Intermediate encoding.

    Official Blackmagic DaVinci Intermediate (DWG) spec:
      For x >= 0.00262409: y = log10(x + 0.0075) × 0.07329248 + 0.5
      For x  < 0.00262409: y = slope × x + intercept
        where slope     = C / ((cut + A) × ln(10))  [C1 tangent continuity at cut]
              intercept = log10(cut + A) × C + 0.5 − slope × cut

    BUG-5 FIX: Previous code used slope = 7.0 and intercept = 0.07329248.
    This gave a 0.262-magnitude discontinuity at the cut point (log value 0.354 vs
    linear value 0.092 — the linear toe was essentially in a completely different
    range). The correct C1-continuous slope is ~3.1440, intercept ~0.3456.

    Pre-computed constants (A=0.0075, C=0.07329248, cut=0.00262409):
      slope     = 0.07329248 / ((0.00262409 + 0.0075) × ln(10)) ≈ 3.14404
      intercept = log10(0.01012409) × 0.07329248 + 0.5 − 3.14404 × 0.00262409
               ≈ 0.35381 − 0.00825 ≈ 0.34556
    """
    A, C = 0.0075, 0.07329248
    cut = 0.00262409
    # C1-continuous linear toe constants (pre-computed, no runtime allocation)
    _slope     = 3.14403760   # C / ((cut + A) * ln(10))
    _intercept = 0.34555736   # log10(cut + A) * C + 0.5 - slope * cut

    log_val = torch.log10(tensor + A) * C + 0.5
    lin_val = _slope * tensor + _intercept
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_davinci_intermediate_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated DaVinci Intermediate decoding.

    BUG-5 FIX (decode side): Inverts the corrected C1-continuous linear toe.
    Previous decode had cut_v = 0.07329248 (the log coefficient C, which is
    clearly wrong as a threshold in encoded space — the actual encoded cut is
    log10(0.00262409 + 0.0075) × 0.07329248 + 0.5 ≈ 0.35381).

    Pre-computed encoded cut ≈ 0.353808.
    """
    A, C = 0.0075, 0.07329248
    # Correct encoded cut: log10(cut + A) * C + 0.5 ≈ 0.353808
    cut_v      = 0.353808
    _slope     = 3.14403760
    _intercept = 0.34555736

    log_val = torch.pow(10.0, (tensor - 0.5) / C) - A
    lin_val = (tensor - _intercept) / _slope
    return torch.where(tensor > cut_v, log_val, lin_val)


def tensor_linear_to_acescct(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated ACEScct encoding."""
    cut = 0.0078125
    a, b = 10.5402377416545, 0.0729055341958355
    log_val = (torch.log2(tensor.clamp(min=1e-10)) + 9.72) / 17.52
    lin_val = a * tensor + b
    return torch.where(tensor > cut, log_val, lin_val)


def tensor_acescct_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated ACEScct decoding."""
    cut_encoded = 0.155251141552511
    log_val = torch.pow(2.0, tensor * 17.52 - 9.72)
    lin_val = (tensor - 0.0729055341958355) / 10.5402377416545
    return torch.where(tensor > cut_encoded, log_val, lin_val)


# ─────────────────────────────────────────────────────────────────────────────
# GPU TENSOR VARIANTS — Canon Log 3, PQ, HLG
# Previously only numpy implementations existed; GPU path forces a full
# tensor→CPU→numpy→GPU round-trip on every frame in video workflows.
# ─────────────────────────────────────────────────────────────────────────────

def tensor_linear_to_canonlog3(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated Canon Log 3 encoding.

    Canon Specification constants — same as the numpy variant.
    18% gray → 0.343 (34.3 IRE), Base ISO 800.
    """
    cut = 0.014
    a   = 14.98325
    c   = 0.36726845
    d   = 0.12783901
    e   = 5.449285
    f   = 0.073059361

    log_val = c * torch.log10(a * tensor + 1.0) + d
    lin_val = e * tensor + f
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_canonlog3_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated Canon Log 3 decoding."""
    cut_encoded = 0.14926   # encoded value at linear cut point
    a = 14.98325
    c = 0.36726845
    d = 0.12783901
    e = 5.449285
    f = 0.073059361

    log_val = (torch.pow(10.0, (tensor - d) / c) - 1.0) / a
    lin_val = (tensor - f) / e
    return torch.where(tensor > cut_encoded, log_val, lin_val)


def tensor_linear_to_pq(tensor: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """GPU-accelerated ST.2084 PQ (Perceptual Quantizer) encoding.

    Encodes scene-linear light (relative to display peak) to PQ signal [0, 1].
    Standard for HDR10, Dolby Vision mastering.

    Args:
        tensor: Scene-linear image, 1.0 = display peak / peak_nits
        peak_nits: Display peak luminance (default 1000 nits for HDR10)
    """
    # Normalise to [0, 1] relative to 10 000 nits (PQ absolute reference)
    L = torch.clamp(tensor * peak_nits / 10000.0, min=0.0)
    m1 = torch.tensor(0.1593017578125, dtype=tensor.dtype, device=tensor.device)
    m2 = torch.tensor(78.84375,        dtype=tensor.dtype, device=tensor.device)
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    Lm1 = torch.pow(L, m1)
    return torch.pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), m2)


def tensor_pq_to_linear(tensor: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """GPU-accelerated ST.2084 PQ decoding to scene-linear."""
    m1 = torch.tensor(0.1593017578125, dtype=tensor.dtype, device=tensor.device)
    m2 = torch.tensor(78.84375,        dtype=tensor.dtype, device=tensor.device)
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    Vm2 = torch.pow(torch.clamp(tensor, min=0.0), 1.0 / m2)
    return (
        torch.pow(torch.clamp(Vm2 - c1, min=0.0) / (c2 - c3 * Vm2), 1.0 / m1)
        * 10000.0 / peak_nits
    )


def tensor_linear_to_hlg(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated ARIB STD-B67 HLG (Hybrid Log-Gamma) encoding.

    HLG is backward-compatible with SDR displays and used in broadcast HDR
    (BBC, NHK, YouTube HDR). Output range [0, 1].
    """
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    t = torch.clamp(tensor, min=0.0)

    lin_val = torch.sqrt(3.0 * t)
    log_val = a * torch.log(torch.clamp(12.0 * t - b, min=1e-10)) + c
    result  = torch.where(t <= 1.0 / 12.0, lin_val, log_val)
    return torch.clamp(result, 0.0, 1.0)


def tensor_hlg_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated ARIB STD-B67 HLG decoding to scene-linear."""
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    lin_val = (tensor ** 2) / 3.0
    log_val = (torch.exp((tensor - c) / a) + b) / 12.0
    result  = torch.where(tensor <= 0.5, lin_val, log_val)
    return torch.clamp(result, min=0.0)
