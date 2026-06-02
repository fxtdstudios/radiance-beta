"""Log and gamma transfer functions — both numpy and GPU-accelerated torch variants.

Every function has a numpy (..._to_linear / linear_to_...) and a tensor
(tensor_..._to_linear / tensor_linear_to_...) form.  The tensor variants avoid
device round-trips by operating directly on GPU tensors.
"""
from __future__ import annotations

import numpy as np
import torch

# ── sRGB Gamma ────────────────────────────────────────────────────────────────


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    return np.where(
        img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0) + 0.055) / 1.055, 2.4)
    ).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    return np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 1e-10), 1 / 2.4) - 0.055,
    ).astype(np.float32)


def tensor_srgb_to_linear(tensor: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    if gamma == 2.2:
        abs_t = tensor.abs()
        sign = tensor.sign()
        low = abs_t / 12.92
        high = torch.pow((abs_t + 0.055) / 1.055, 2.4)
        return torch.where(abs_t <= 0.04045, low, high) * sign
    abs_t = tensor.abs()
    return torch.pow(abs_t.clamp(min=1e-10), gamma) * tensor.sign()


def tensor_linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    abs_t = tensor.abs()
    sign = tensor.sign()
    low = abs_t * 12.92
    high = 1.055 * torch.pow(abs_t.clamp(min=1e-10), 1 / 2.4) - 0.055
    return torch.where(abs_t <= 0.0031308, low, high) * sign


# ── ARRI LogC3 ────────────────────────────────────────────────────────────────

LOGC3_EI_PARAMS = {
    160: (0.005561, 5.061087, 0.089004, 0.269035, 0.391007, 6.332427, 0.108361),
    200: (0.006208, 5.168208, 0.076621, 0.265275, 0.391007, 5.842037, 0.099519),
    250: (0.006871, 5.282072, 0.065521, 0.261620, 0.391007, 5.397270, 0.091111),
    320: (0.007622, 5.399335, 0.055194, 0.257766, 0.391007, 4.969419, 0.083295),
    400: (0.008318, 5.510883, 0.046585, 0.254174, 0.391007, 4.606965, 0.076257),
    500: (0.009031, 5.618393, 0.039023, 0.250758, 0.391007, 4.282556, 0.069776),
    640: (0.009840, 5.737055, 0.031538, 0.247070, 0.391007, 3.946374, 0.063409),
    800: (0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809),
    1000: (0.011361, 5.944966, 0.019018, 0.240020, 0.391007, 3.369506, 0.051759),
    1280: (0.012235, 6.056760, 0.013804, 0.236500, 0.391007, 3.088156, 0.046447),
    1600: (0.013047, 6.161541, 0.009677, 0.233182, 0.391007, 2.852200, 0.041773),
    2000: (0.013901, 6.260724, 0.006210, 0.230014, 0.391007, 2.643126, 0.037413),
    2560: (0.014842, 6.362496, 0.002995, 0.226764, 0.391007, 2.440085, 0.033198),
    3200: (0.015711, 6.456037, 0.000295, 0.223740, 0.391007, 2.265605, 0.029493),
}


def linear_to_logc3(img: np.ndarray, ei: int = 800) -> np.ndarray:
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut
    out[mask] = c * np.log10(a * img[mask] + b) + d
    out[~mask] = e * img[~mask] + f
    return out


def logc3_to_linear(img: np.ndarray, ei: int = 800) -> np.ndarray:
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    cut_encoded = e * cut + f
    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut_encoded
    out[mask] = (np.power(10.0, (img[mask] - d) / c) - b) / a
    out[~mask] = (img[~mask] - f) / e
    return out


def tensor_linear_to_logc3(tensor: torch.Tensor, ei: int = 800) -> torch.Tensor:
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    return torch.where(tensor > cut,
                       c * torch.log10(a * tensor + b) + d,
                       e * tensor + f)


def tensor_logc3_to_linear(tensor: torch.Tensor, ei: int = 800) -> torch.Tensor:
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    cut_encoded = e * cut + f
    return torch.where(tensor > cut_encoded,
                       (torch.pow(10.0, (tensor - d) / c) - b) / a,
                       (tensor - f) / e)


# ── ARRI LogC4 ────────────────────────────────────────────────────────────────

_LOGC4_A = 2231.826309067637
_LOGC4_B = 0.9071358691330627
_LOGC4_C = 0.0928641308669373
_LOGC4_T = -0.0180569961199123
_LOGC4_S = 0.1135773173772412


def linear_to_logc4(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= _LOGC4_T
    out[mask] = ((np.log2(_LOGC4_A * img[mask] + 64.0) - 6.0) / 14.0) * _LOGC4_B + _LOGC4_C
    out[~mask] = (img[~mask] - _LOGC4_T) / _LOGC4_S
    return out


def logc4_to_linear(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= 0.0
    out[mask] = (np.power(2.0, ((img[mask] - _LOGC4_C) / _LOGC4_B) * 14.0 + 6.0) - 64.0) / _LOGC4_A
    out[~mask] = img[~mask] * _LOGC4_S + _LOGC4_T
    return out


def tensor_linear_to_logc4(tensor: torch.Tensor) -> torch.Tensor:
    log_val = ((torch.log2((_LOGC4_A * tensor + 64.0).clamp(min=1e-10)) - 6.0) / 14.0) * _LOGC4_B + _LOGC4_C
    lin_val = (tensor - _LOGC4_T) / _LOGC4_S
    return torch.where(tensor >= _LOGC4_T, log_val, lin_val)


def tensor_logc4_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    log_val = (torch.pow(2.0, ((tensor - _LOGC4_C) / _LOGC4_B) * 14.0 + 6.0) - 64.0) / _LOGC4_A
    lin_val = tensor * _LOGC4_S + _LOGC4_T
    return torch.where(tensor >= 0.0, log_val, lin_val)


# ── Sony S-Log3 ───────────────────────────────────────────────────────────────

_TOE_SLOPE_SLOG3 = 76.2102946929 / (0.02125 * 1023.0)  # ≈ 3.5058, C0-continuous


def linear_to_slog3(img: np.ndarray) -> np.ndarray:
    img = np.maximum(img, -0.01)
    cut = 0.011250
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut
    out[mask] = (420.0 + np.log10((img[mask] + 0.01) / 0.19) * 261.5) / 1023.0
    out[~mask] = _TOE_SLOPE_SLOG3 * (img[~mask] + 0.01) + 95.0 / 1023.0
    return out


def slog3_to_linear(img: np.ndarray) -> np.ndarray:
    cut_v = 171.2102946929 / 1023.0
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut_v
    out[mask] = 0.19 * np.power(10.0, (img[mask] * 1023.0 - 420.0) / 261.5) - 0.01
    out[~mask] = (img[~mask] - 95.0 / 1023.0) / _TOE_SLOPE_SLOG3 - 0.01
    return out


def tensor_linear_to_slog3(tensor: torch.Tensor) -> torch.Tensor:
    cut = 0.011250
    tensor_clamped = tensor.clamp(min=-0.01)
    log_val = (420.0 + torch.log10((tensor_clamped + 0.01).clamp(min=1e-10) / 0.19) * 261.5) / 1023.0
    lin_val = _TOE_SLOPE_SLOG3 * (tensor_clamped + 0.01) + 95.0 / 1023.0
    return torch.where(tensor_clamped >= cut, log_val, lin_val)


def tensor_slog3_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    cut_v = 171.2102946929 / 1023.0
    log_val = 0.19 * torch.pow(10.0, (tensor * 1023.0 - 420.0) / 261.5) - 0.01
    lin_val = (tensor - 95.0 / 1023.0) / _TOE_SLOPE_SLOG3 - 0.01
    return torch.where(tensor >= cut_v, log_val, lin_val)


# ── Panasonic V-Log ───────────────────────────────────────────────────────────

def linear_to_vlog(img: np.ndarray) -> np.ndarray:
    cut = 0.01
    b, c, d = 0.00873, 0.241514, 0.598206
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut
    out[mask] = c * np.log10(img[mask] + b) + d
    out[~mask] = 5.6 * img[~mask] + 0.125
    return out


def vlog_to_linear(img: np.ndarray) -> np.ndarray:
    b, c, d = 0.00873, 0.241514, 0.598206
    cut_encoded = 0.181  # 5.6 * 0.01 + 0.125
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut_encoded
    out[mask] = np.power(10.0, (img[mask] - d) / c) - b
    out[~mask] = (img[~mask] - 0.125) / 5.6
    return out


def tensor_linear_to_vlog(tensor: torch.Tensor) -> torch.Tensor:
    cut = 0.01
    b, c, d = 0.00873, 0.241514, 0.598206
    log_val = c * torch.log10(tensor + b) + d
    lin_val = 5.6 * tensor + 0.125
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_vlog_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    b, c, d = 0.00873, 0.241514, 0.598206
    cut_encoded = 0.181000
    log_val = torch.pow(10.0, (tensor - d) / c) - b
    lin_val = (tensor - 0.125) / 5.6
    return torch.where(tensor >= cut_encoded, log_val, lin_val)


# ── Canon Log 3 ───────────────────────────────────────────────────────────────

_CANON_CUT = 0.014
_CANON_A = 14.98325
_CANON_C = 0.36726845
_CANON_D = 0.12783901
_CANON_E = 5.449285
_CANON_F = 0.073059361
_CANON_CUT_ENCODED = 0.14926


def linear_to_canonlog3(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= _CANON_CUT
    out[mask] = _CANON_C * np.log10(_CANON_A * img[mask] + 1.0) + _CANON_D
    out[~mask] = _CANON_E * img[~mask] + _CANON_F
    return out


def canonlog3_to_linear(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _CANON_CUT_ENCODED
    out[mask] = (np.power(10.0, (img[mask] - _CANON_D) / _CANON_C) - 1.0) / _CANON_A
    out[~mask] = (img[~mask] - _CANON_F) / _CANON_E
    return out


def tensor_linear_to_canonlog3(tensor: torch.Tensor) -> torch.Tensor:
    log_val = _CANON_C * torch.log10(_CANON_A * tensor + 1.0) + _CANON_D
    lin_val = _CANON_E * tensor + _CANON_F
    return torch.where(tensor >= _CANON_CUT, log_val, lin_val)


def tensor_canonlog3_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    log_val = (torch.pow(10.0, (tensor - _CANON_D) / _CANON_C) - 1.0) / _CANON_A
    lin_val = (tensor - _CANON_F) / _CANON_E
    return torch.where(tensor > _CANON_CUT_ENCODED, log_val, lin_val)


# ── RED Log3G10 ───────────────────────────────────────────────────────────────

_LOG3G10_A = 0.224282
_LOG3G10_B = 155.975327
_LOG3G10_C = 0.01
_LOG3G10_CUT = 0.01
_LOG3G10_CUT_ENCODED = 0.101551  # encode(_LOG3G10_CUT)


def linear_to_log3g10(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _LOG3G10_CUT
    out[mask] = _LOG3G10_A * np.log10(_LOG3G10_B * img[mask] + 1.0) + _LOG3G10_C
    out[~mask] = (img[~mask] / _LOG3G10_CUT) * _LOG3G10_CUT_ENCODED
    return out


def log3g10_to_linear(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _LOG3G10_CUT_ENCODED
    out[mask] = (np.power(10.0, (img[mask] - _LOG3G10_C) / _LOG3G10_A) - 1.0) / _LOG3G10_B
    out[~mask] = img[~mask] / _LOG3G10_CUT_ENCODED * _LOG3G10_CUT
    return out


def tensor_linear_to_log3g10(tensor: torch.Tensor) -> torch.Tensor:
    log_val = _LOG3G10_A * torch.log10(_LOG3G10_B * tensor + 1.0) + _LOG3G10_C
    lin_val = (tensor / _LOG3G10_CUT) * _LOG3G10_CUT_ENCODED
    return torch.where(tensor > _LOG3G10_CUT, log_val, lin_val)


def tensor_log3g10_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    log_val = (torch.pow(10.0, (tensor - _LOG3G10_C) / _LOG3G10_A) - 1.0) / _LOG3G10_B
    lin_val = tensor / _LOG3G10_CUT_ENCODED * _LOG3G10_CUT
    return torch.where(tensor > _LOG3G10_CUT_ENCODED, log_val, lin_val)


# ── ACEScct ───────────────────────────────────────────────────────────────────

_ACESCCT_CUT = 0.0078125
_ACESCCT_A = 10.5402377416545
_ACESCCT_B = 0.0729055341958355
_ACESCCT_CUT_ENCODED = 0.155251141552511


def linear_to_acescct(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _ACESCCT_CUT
    out[mask] = (np.log2(img[mask]) + 9.72) / 17.52
    out[~mask] = _ACESCCT_A * img[~mask] + _ACESCCT_B
    return out


def acescct_to_linear(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _ACESCCT_CUT_ENCODED
    out[mask] = np.power(2.0, img[mask] * 17.52 - 9.72)
    out[~mask] = (img[~mask] - _ACESCCT_B) / _ACESCCT_A
    return out


def tensor_linear_to_acescct(tensor: torch.Tensor) -> torch.Tensor:
    log_val = (torch.log2(tensor.clamp(min=1e-10)) + 9.72) / 17.52
    lin_val = _ACESCCT_A * tensor + _ACESCCT_B
    return torch.where(tensor > _ACESCCT_CUT, log_val, lin_val)


def tensor_acescct_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    log_val = torch.pow(2.0, tensor * 17.52 - 9.72)
    lin_val = (tensor - _ACESCCT_B) / _ACESCCT_A
    return torch.where(tensor > _ACESCCT_CUT_ENCODED, log_val, lin_val)


# ── DaVinci Intermediate ──────────────────────────────────────────────────────

_DI_A = 0.24928
_DI_B = 444.8616
_DI_C = 0.0139
_DI_D = 0.4
_DI_CUT = 0.00262409
_DI_SLOPE = 3.14403760  # C1-continuous toe slope
_DI_INTERCEPT = 0.34555736
_DI_CUT_ENCODED = 0.353808


def linear_to_davinci_intermediate(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= _DI_CUT
    out[mask] = _DI_A * np.log10(img[mask] * _DI_B + _DI_C) + _DI_D
    out[~mask] = _DI_SLOPE * img[~mask] + _DI_INTERCEPT
    return out


def davinci_intermediate_to_linear(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= _DI_CUT_ENCODED
    out[mask] = (np.power(10.0, (img[mask] - _DI_D) / _DI_A) - _DI_C) / _DI_B
    out[~mask] = (img[~mask] - _DI_INTERCEPT) / _DI_SLOPE
    return out


def tensor_linear_to_davinci_intermediate(tensor: torch.Tensor) -> torch.Tensor:
    log_val = torch.log10(tensor * _DI_B + _DI_C) * _DI_A + _DI_D
    lin_val = _DI_SLOPE * tensor + _DI_INTERCEPT
    return torch.where(tensor >= _DI_CUT, log_val, lin_val)


def tensor_davinci_intermediate_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    log_val = torch.pow(10.0, (tensor - _DI_D) / _DI_A) - _DI_C
    lin_val = (tensor - _DI_INTERCEPT) / _DI_SLOPE
    return torch.where(tensor > _DI_CUT_ENCODED, log_val / _DI_B, lin_val)


# ── HDR Transfer Functions (PQ / HLG) ─────────────────────────────────────────

def linear_to_pq(img: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    L = np.clip(img * peak_nits / 10000.0, 0, 1)
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = np.power(L, m1)
    return np.power((c1 + c2 * Lm1) / (1 + c3 * Lm1), m2)


def pq_to_linear(img: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Vm2 = np.power(np.maximum(img, 0), 1 / m2)
    return np.power(np.maximum(Vm2 - c1, 0) / (c2 - c3 * Vm2), 1 / m1) * 10000.0 / peak_nits


def linear_to_hlg(img: np.ndarray) -> np.ndarray:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    out = np.where(
        img <= 1 / 12,
        np.sqrt(3 * np.maximum(img, 0)),
        a * np.log(np.maximum(12 * img - b, 1e-10)) + c,
    )
    return np.clip(out, 0, 1)


def hlg_to_linear(img: np.ndarray) -> np.ndarray:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    out = np.where(img <= 0.5, (img**2) / 3, (np.exp((img - c) / a) + b) / 12)
    return np.maximum(out, 0)


def tensor_linear_to_pq(tensor: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    L = torch.clamp(tensor * peak_nits / 10000.0, min=0.0)
    m1 = torch.tensor(0.1593017578125, dtype=tensor.dtype, device=tensor.device)
    m2 = torch.tensor(78.84375, dtype=tensor.dtype, device=tensor.device)
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = torch.pow(L, m1)
    return torch.pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), m2)


def tensor_pq_to_linear(tensor: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    m1 = torch.tensor(0.1593017578125, dtype=tensor.dtype, device=tensor.device)
    m2 = torch.tensor(78.84375, dtype=tensor.dtype, device=tensor.device)
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Vm2 = torch.pow(torch.clamp(tensor, min=0.0), 1.0 / m2)
    return torch.pow(torch.clamp(Vm2 - c1, min=0.0) / (c2 - c3 * Vm2), 1.0 / m1) * 10000.0 / peak_nits


def tensor_linear_to_hlg(tensor: torch.Tensor) -> torch.Tensor:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    t = torch.clamp(tensor, min=0.0)
    lin_val = torch.sqrt(3.0 * t)
    log_val = a * torch.log(torch.clamp(12.0 * t - b, min=1e-10)) + c
    return torch.clamp(torch.where(t <= 1.0 / 12.0, lin_val, log_val), 0.0, 1.0)


def tensor_hlg_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    lin_val = (tensor ** 2) / 3.0
    log_val = (torch.exp((tensor - c) / a) + b) / 12.0
    return torch.clamp(torch.where(tensor <= 0.5, lin_val, log_val), min=0.0)
