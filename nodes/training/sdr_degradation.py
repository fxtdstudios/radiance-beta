"""
◎ Radiance v3.0.0 — SDR Degradation Pipeline
nodes_sdr_degradation.py

Training data generation for IC-LoRA HDR conditioning.

PURPOSE
───────
IC-LoRA (Image Conditioning LoRA) learns to generate HDR output conditioned
on an SDR reference image of the same scene.  The model sees:

    (sdr_reference, noise) → generate → hdr_output

For the training pairs to generalise, the SDR reference must realistically
simulate how real cameras, broadcast chains, and streaming codecs render the
same scene in SDR.  A single tonemap operator with fixed parameters produces
pairs that overfit to one "look" — the model learns the specific curve, not
the HDR→SDR mapping in general.

This module provides `run_sdr_degradation_pipeline`, a stochastic pipeline
that applies a randomised but reproducible sequence of degradations to a
scene-linear HDR IMAGE, producing a realistic SDR conditioning image together
with a JSON metadata string that records every operation and its exact
parameters for training reproducibility and dataset analysis.

DEGRADATION STAGES (applied in order)
───────────────────────────────────────
  1.  White-point normalisation   — scale so white_point maps to 1.0
  2.  Tone mapping               — operator drawn from pool (or fixed)
  3.  Highlight clip             — hard clip or soft shoulder
  4.  OETF (sRGB gamma)         — display-referred encoding
  5.  Color shift                — hue rotation, saturation crush, per-channel gain
  6.  Sensor noise               — Gaussian + Poisson shot noise
  7.  JPEG codec simulation      — PIL encode/decode at randomised quality
  8.  Bit-depth quantisation     — simulate 8-bit banding (optional)
  9.  Vignette                   — radial brightness roll-off (optional)

All stages run on the compute device (CUDA if available).
JPEG codec simulation falls back gracefully when PIL is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import math
import random
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("radiance.sdr_degradation")

# torch is imported lazily inside functions so the module loads in test
# environments where only a stub is present.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None          # type: ignore[assignment]
    _TORCH_AVAILABLE = False

import numpy as np

_AGX_M_IN = np.array([
    [0.842479062253094, 0.0784335999999992, 0.0792237451477643],
    [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
    [0.0423756549057051, 0.0784336000000002, 0.879142973793104],
], dtype=np.float32).T

_AGX_M_IN_T:  Optional[object] = None
_AGX_M_OUT_T: Optional[object] = None


def _agx_matrices(device):
    global _AGX_M_IN_T, _AGX_M_OUT_T
    if _AGX_M_IN_T is None or _AGX_M_IN_T.device != device:
        m_in  = _torch.from_numpy(_AGX_M_IN).to(device)
        m_out = _torch.from_numpy(np.linalg.inv(_AGX_M_IN)).to(device)
        _AGX_M_IN_T, _AGX_M_OUT_T = m_in, m_out
    return _AGX_M_IN_T, _AGX_M_OUT_T


# ═════════════════════════════════════════════════════════════════════════════
#  DEGRADATION PRIMITIVES
#  All functions operate on float32 tensors (B, H, W, 3), no side-effects.
# ═════════════════════════════════════════════════════════════════════════════

# ── Tonemapping ───────────────────────────────────────────────────────────────

_TONEMAP_OPERATORS = [
    "reinhard",
    "reinhard_luminance",
    "filmic_aces",
    "filmic_uncharted2",
    "agx",
    "linear_clamp",
]


def _tonemap(x, operator: str, white_point: float):
    """
    Apply a tone mapping operator to scene-linear input x (B, H, W, 3).

    All operators map [0, ∞) → [0, 1].  Values are clamped to [0, 1] after
    mapping.  `white_point` is the scene-linear luminance value that should
    map to SDR 1.0 (used differently by each operator — see per-operator docs).

    Returns float32 tensor, same shape, values in [0, 1].
    """
    wp = max(white_point, 1e-6)

    if operator == "reinhard":
        # Global Reinhard: maps x → x/(1+x), independently per channel.
        # white_point used as scene exposure prescale: input = x / wp.
        y = x / wp
        return _torch.clamp(y / (1.0 + y), 0.0, 1.0)

    elif operator == "reinhard_luminance":
        # Luminance-only Reinhard — preserves hue & saturation.
        y    = x / wp
        luma = (0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]).clamp(min=1e-6)
        # Extended Reinhard on luma
        luma_tm = (luma * (1.0 + luma / 4.0)) / (1.0 + luma)
        scale   = (luma_tm / luma).unsqueeze(-1)
        return _torch.clamp(y * scale, 0.0, 1.0)

    elif operator == "filmic_aces":
        # Narkowicz 2015 ACES approximation.  white_point scales the input.
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        y = x / wp
        return _torch.clamp((y * (a * y + b)) / (y * (c * y + d) + e), 0.0, 1.0)

    elif operator == "filmic_uncharted2":
        # Uncharted 2 filmic — slight warm/film look; strong shoulder.
        A, B, C, D, E_, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

        def _curve(v: _torch.Tensor) -> _torch.Tensor:
            return (v * (A * v + C * B) + D * E_) / (v * (A * v + B) + D * F) - E_ / F

        y          = x / wp
        white_t    = _torch.tensor(wp, device=x.device, dtype=x.dtype)
        white_scale = 1.0 / (_curve(white_t) + 1e-8)
        return _torch.clamp(_curve(y) * white_scale, 0.0, 1.0)

    elif operator == "agx":
        # Full AgX pipeline: sRGB linear → AgX working space → log-sigmoid → sRGB.
        m_in, m_out = _agx_matrices(x.device)
        y     = (x / wp).clamp(min=0.0)
        y_agx = _torch.einsum("ij,...j->...i", m_in, y)
        y_log = (_torch.log2(y_agx.clamp(min=1e-10)) - (-10.0)) / (6.5 - (-10.0))
        y_log = y_log.clamp(0.0, 1.0)
        # Approximate AgX CDL sigmoid
        y_sig = y_log / (1.0 + (y_log - 0.5).abs() * 2.0)
        y_sig = ((y_sig - 0.5) * 1.5 + 0.5).clamp(0.0, 1.0)
        return _torch.einsum("ij,...j->...i", m_out, y_sig).clamp(0.0, 1.0)

    elif operator == "linear_clamp":
        # Straight linear scale + hard clip — no tone mapping.
        return _torch.clamp(x / wp, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown tonemap operator: {operator!r}")


# ── Highlight clipping ────────────────────────────────────────────────────────

def _highlight_clip(x: _torch.Tensor, mode: str) -> _torch.Tensor:
    """
    Clip values above 1.0.

    mode="hard"         : _torch.clamp(x, 0, 1)
    mode="soft_shoulder": smooth roll-off above `knee` (0.9) to 1.0.
                          Prevents hard transitions at the clip boundary.
    """
    if mode == "hard":
        return _torch.clamp(x, 0.0, 1.0)

    # soft_shoulder — cubic roll-off from `knee` to 1.0
    knee = 0.90
    y    = x.clone()
    mask = (x > knee) & (x <= 1.2)   # region to soften
    t    = (x[mask] - knee) / (1.2 - knee)   # [0, 1]
    # Cubic ease-in-out: 3t² - 2t³
    soft = knee + (1.0 - knee) * (3.0 * t * t - 2.0 * t * t * t)
    y[mask] = soft
    return _torch.clamp(y, 0.0, 1.0)


# ── sRGB OETF (gamma encoding) ────────────────────────────────────────────────

def _oetf_srgb(x: _torch.Tensor) -> _torch.Tensor:
    """
    Apply the sRGB Opto-Electronic Transfer Function (IEC 61966-2-1).
    Input: scene-linear [0, 1].  Output: display-referred [0, 1].
    """
    x    = x.clamp(0.0, 1.0)
    low  = x * 12.92
    high = 1.055 * x.pow(1.0 / 2.4) - 0.055
    return _torch.where(x <= 0.0031308, low, high)


# ── Colour shift ──────────────────────────────────────────────────────────────

def _rgb_to_hsv(x: _torch.Tensor) -> _torch.Tensor:
    """Convert (B, H, W, 3) RGB [0,1] → HSV [0,1]."""
    r, g, b  = x[..., 0], x[..., 1], x[..., 2]
    v        = _torch.max(x, dim=-1).values
    s_denom  = v.clamp(min=1e-7)
    mn       = _torch.min(x, dim=-1).values
    delta    = v - mn
    s        = delta / s_denom
    s[v < 1e-7] = 0.0

    h = _torch.zeros_like(v)
    d  = delta.clamp(min=1e-7)
    rc = (v == r)
    gc = (v == g) & ~rc
    bc = ~rc & ~gc
    h[rc] = ((g[rc] - b[rc]) / d[rc]) % 6.0
    h[gc] = (b[gc] - r[gc]) / d[gc] + 2.0
    h[bc] = (r[bc] - g[bc]) / d[bc] + 4.0
    h = h / 6.0 % 1.0
    return _torch.stack([h, s, v], dim=-1)


def _hsv_to_rgb(hsv: _torch.Tensor) -> _torch.Tensor:
    """Convert (B, H, W, 3) HSV [0,1] → RGB [0,1]."""
    h, s, v = hsv[..., 0] * 6.0, hsv[..., 1], hsv[..., 2]
    i   = h.long() % 6
    f   = h - h.floor()
    p   = v * (1.0 - s)
    q   = v * (1.0 - f * s)
    t   = v * (1.0 - (1.0 - f) * s)

    rgb = _torch.zeros_like(hsv)
    for ch, (a, b, c) in enumerate([
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]):
        mask = (i == ch)
        rgb[..., 0][mask] = a[mask]
        rgb[..., 1][mask] = b[mask]
        rgb[..., 2][mask] = c[mask]
    return rgb.clamp(0.0, 1.0)


def _color_shift(
    x: _torch.Tensor,
    hue_shift:   float,   # [-0.5, 0.5]  — fraction of full hue circle
    sat_scale:   float,   # [0.5, 1.5]   — saturation multiplier
    gain_r:      float,   # [0.8, 1.2]   — per-channel brightness offset
    gain_g:      float,
    gain_b:      float,
) -> _torch.Tensor:
    """
    Apply randomised colour grading shifts to simulate imperfect SDR rendering.

    Operates in HSV space for hue/saturation, then per-channel gain in RGB.
    """
    hsv = _rgb_to_hsv(x)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0      # hue wrap
    hsv[..., 1] = hsv[..., 1].mul(sat_scale).clamp(0.0, 1.0)
    x2  = _hsv_to_rgb(hsv)
    gains = _torch.tensor([gain_r, gain_g, gain_b], device=x.device, dtype=x.dtype)
    return (x2 * gains).clamp(0.0, 1.0)


# ── Sensor noise ──────────────────────────────────────────────────────────────

def _sensor_noise(
    x: _torch.Tensor,
    gaussian_std: float,   # additive read noise
    shot_scale:   float,   # Poisson shot noise scale (0 = off)
    rng:          _torch.Generator,
) -> _torch.Tensor:
    """
    Simulate digital camera sensor noise:
      • Gaussian read noise  — signal-independent (dark current, ADC quantisation)
      • Poisson shot noise   — signal-dependent (∝ √signal), stronger in shadows
    """
    y = x.clone()
    if gaussian_std > 0.0:
        noise = _torch.zeros_like(x).normal_(0.0, gaussian_std, generator=rng)
        y = y + noise

    if shot_scale > 0.0:
        # Shot noise: variance ∝ signal.  Approximate with Gaussian(0, √(x·scale)).
        sigma_map = (x.abs() * shot_scale).sqrt().clamp(min=0.0)
        noise     = _torch.zeros_like(x).normal_(0.0, 1.0, generator=rng) * sigma_map
        y = y + noise

    return y.clamp(0.0, 1.0)


# ── JPEG codec simulation ─────────────────────────────────────────────────────

def _jpeg_compress(x: _torch.Tensor, quality: int) -> _torch.Tensor:
    """
    Simulate JPEG codec degradation via PIL encode → decode.

    quality: 1 (worst) … 95 (best).  0 = no-op.
    Operates frame-by-frame on CPU; the result is moved back to x.device.
    """
    if quality <= 0:
        return x
    try:
        from PIL import Image as _PIL_Image
        import io
    except ImportError:
        logger.debug("[SDRDegradation] PIL not available — skipping JPEG stage.")
        return x

    device = x.device
    x_cpu  = (x.cpu().float().clamp(0.0, 1.0) * 255.0).to(_torch.uint8)
    results = []
    for i in range(x_cpu.shape[0]):
        frame_np = x_cpu[i].numpy()
        img      = _PIL_Image.fromarray(frame_np, mode="RGB")
        buf      = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, subsampling=2)
        buf.seek(0)
        decoded  = _PIL_Image.open(buf).convert("RGB")
        results.append(_torch.from_numpy(np.array(decoded)).float() / 255.0)

    return _torch.stack(results, dim=0).to(device)


# ── Bit-depth quantisation ────────────────────────────────────────────────────

def _quantize(x: _torch.Tensor, bits: int) -> _torch.Tensor:
    """
    Simulate bit-depth reduction banding.  bits=8 → 256 levels.  0 = no-op.
    """
    if bits <= 0:
        return x
    levels = float(2 ** bits - 1)
    return (x * levels).round() / levels


# ── Vignette ──────────────────────────────────────────────────────────────────

def _vignette(x: _torch.Tensor, strength: float, power: float = 2.0) -> _torch.Tensor:
    """
    Apply a radial vignette (darkening toward the edges).

    strength: 0 = no effect, 1 = corners fully black.
    power:    exponent of the radial gradient (2.0 = circular, 1.0 = linear).
    """
    if strength <= 0.0:
        return x
    B, H, W, C = x.shape
    ys = _torch.linspace(-1.0, 1.0, H, device=x.device)
    xs = _torch.linspace(-1.0, 1.0, W, device=x.device)
    grid_y, grid_x = _torch.meshgrid(ys, xs, indexing="ij")
    dist   = (grid_x ** 2 + grid_y ** 2).sqrt()   # 0..√2
    dist   = (dist / math.sqrt(2.0)).clamp(0.0, 1.0)
    weight = 1.0 - strength * dist.pow(power)
    weight = weight.unsqueeze(0).unsqueeze(-1)      # (1, H, W, 1)
    return (x * weight).clamp(0.0, 1.0)


# ── Gaussian blur (optional spatial softening) ────────────────────────────────

def _gaussian_blur(x: _torch.Tensor, sigma: float) -> _torch.Tensor:
    """
    Apply a separable Gaussian blur.  sigma=0 = no-op.
    Operates per-batch via F.conv2d.
    """
    if sigma <= 0.0:
        return x
    ksize = max(3, 2 * int(3.0 * sigma) + 1)
    # Build 1-D Gaussian kernel
    coords = _torch.arange(ksize, dtype=_torch.float32, device=x.device) - ksize // 2
    kernel = _torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # Separate horizontal / vertical convolution
    C = x.shape[-1]
    kh = kernel.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
    kv = kernel.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)
    y  = x.permute(0, 3, 1, 2)   # (B, C, H, W)
    pad = ksize // 2
    y   = _F_conv.conv2d(y, kh, padding=(0, pad), groups=C)
    y   = _F_conv.conv2d(y, kv, padding=(pad, 0), groups=C)
    return y.permute(0, 2, 3, 1).clamp(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_sdr_degradation_pipeline(
    image:              _torch.Tensor,     # (B, H, W, 3) scene-linear HDR
    white_point:        float  = 8.0,
    tonemap_op:         str    = "random",
    clip_mode:          str    = "hard",
    apply_oetf:         bool   = True,
    color_shift_str:    float  = 0.05,
    noise_strength:     float  = 0.03,
    jpeg_quality:       int    = 0,
    quantize_bits:      int    = 0,
    vignette_strength:  float  = 0.0,
    blur_sigma:         float  = 0.0,
    augment_probability: float = 1.0,
    seed:               int    = -1,
) -> Tuple[_torch.Tensor, Dict]:
    """
    Apply the full SDR degradation pipeline to a scene-linear HDR tensor.

    Args:
        image:               (B, H, W, 3) float32, scene-linear.  Values ≥ 0.
        white_point:         Scene-linear luminance that maps to SDR 1.0.
        tonemap_op:          Operator name or "random" to sample from pool.
        clip_mode:           "hard" or "soft_shoulder".
        apply_oetf:          Apply sRGB gamma after tonemapping.
        color_shift_str:     Max magnitude of hue/sat/gain perturbation (0 = off).
        noise_strength:      Gaussian noise std-dev (0 = off).
        jpeg_quality:        JPEG quality level 1–95 (0 = off).
        quantize_bits:       Bit depth to quantise to (0 = off).
        vignette_strength:   Vignette amount (0 = off).
        blur_sigma:          Gaussian blur sigma in pixels (0 = off).
        augment_probability: Fraction of batch items to augment (0–1).
        seed:                RNG seed.  -1 = random.

    Returns:
        (sdr_image, metadata_dict)
    """
    t0    = time.monotonic()
    rng_s = seed if seed >= 0 else random.randint(0, 2**31 - 1)
    rng   = _torch.Generator(device=image.device)
    rng.manual_seed(rng_s)
    py_rng = random.Random(rng_s)

    # Resolve "random" tonemap operator
    if tonemap_op == "random":
        op = py_rng.choice(_TONEMAP_OPERATORS)
    else:
        op = tonemap_op

    meta: Dict = {
        "seed":          rng_s,
        "white_point":   white_point,
        "stages":        [],
    }

    def _record(stage: str, **params):
        meta["stages"].append({"stage": stage, **{k: round(v, 6) if isinstance(v, float) else v
                                                   for k, v in params.items()}})

    x = image.float().clamp(min=0.0)

    # Per-item augment mask (supports augment_probability < 1)
    B = x.shape[0]
    aug_mask = _torch.zeros(B, dtype=_torch.bool, device=x.device)
    for i in range(B):
        aug_mask[i] = py_rng.random() < augment_probability
    _record("augment_mask", mask=[bool(aug_mask[i].item()) for i in range(B)])

    # For unaug items, remember the clean sRGB (simple clip + OETF)
    if not aug_mask.all():
        x_clean = _oetf_srgb(_torch.clamp(x / max(white_point, 1e-6), 0.0, 1.0)) if apply_oetf \
                  else _torch.clamp(x / max(white_point, 1e-6), 0.0, 1.0)

    # ── Stage 1: Tonemap ─────────────────────────────────────────────────────
    y = _tonemap(x, op, white_point)
    _record("tonemap", operator=op, white_point=white_point)

    # ── Stage 2: Highlight clip ──────────────────────────────────────────────
    y = _highlight_clip(y, clip_mode)
    _record("highlight_clip", mode=clip_mode)

    # ── Stage 3: OETF ────────────────────────────────────────────────────────
    if apply_oetf:
        y = _oetf_srgb(y)
        _record("oetf", encoding="sRGB_IEC61966-2-1")

    # ── Stage 4: Colour shift ────────────────────────────────────────────────
    if color_shift_str > 0.0:
        hue_s   = py_rng.uniform(-color_shift_str, color_shift_str)
        sat_s   = py_rng.uniform(max(0.5, 1.0 - color_shift_str * 4),
                                  min(1.5, 1.0 + color_shift_str * 4))
        gain_r  = py_rng.uniform(1.0 - color_shift_str, 1.0 + color_shift_str)
        gain_g  = py_rng.uniform(1.0 - color_shift_str * 0.5, 1.0 + color_shift_str * 0.5)
        gain_b  = py_rng.uniform(1.0 - color_shift_str, 1.0 + color_shift_str)
        y = _color_shift(y, hue_s, sat_s, gain_r, gain_g, gain_b)
        _record("color_shift",
                hue_shift=hue_s, sat_scale=sat_s,
                gain_r=gain_r, gain_g=gain_g, gain_b=gain_b)

    # ── Stage 5: Sensor noise ────────────────────────────────────────────────
    if noise_strength > 0.0:
        gauss_std  = noise_strength * py_rng.uniform(0.5, 1.5)
        shot_scale = noise_strength * py_rng.uniform(0.0, 0.5)
        y = _sensor_noise(y, gauss_std, shot_scale, rng)
        _record("sensor_noise", gaussian_std=gauss_std, shot_scale=shot_scale)

    # ── Stage 6: JPEG codec ──────────────────────────────────────────────────
    if jpeg_quality > 0:
        # Randomise around the requested quality ±10 for diversity
        q = max(1, min(95, jpeg_quality + py_rng.randint(-10, 10)))
        y = _jpeg_compress(y, q)
        _record("jpeg_codec", quality=q)

    # ── Stage 7: Bit-depth quantisation ─────────────────────────────────────
    if quantize_bits > 0:
        y = _quantize(y, quantize_bits)
        _record("quantize", bits=quantize_bits)

    # ── Stage 8: Vignette ────────────────────────────────────────────────────
    if vignette_strength > 0.0:
        vig_s = vignette_strength * py_rng.uniform(0.5, 1.5)
        power  = py_rng.uniform(1.5, 3.0)
        y = _vignette(y, vig_s, power)
        _record("vignette", strength=vig_s, power=power)

    # ── Stage 9: Blur ────────────────────────────────────────────────────────
    if blur_sigma > 0.0:
        sigma = blur_sigma * py_rng.uniform(0.5, 2.0)
        y = _gaussian_blur(y, sigma)
        _record("blur", sigma=sigma)

    # ── Blend augmented / clean ───────────────────────────────────────────────
    if not aug_mask.all():
        mask_bchw = aug_mask.view(B, 1, 1, 1).float()
        y = y * mask_bchw + x_clean * (1.0 - mask_bchw)

    meta["elapsed_ms"] = round((time.monotonic() - t0) * 1000.0, 2)
    return y.clamp(0.0, 1.0), meta


NODE_CLASS_MAPPINGS = {}

NODE_DISPLAY_NAME_MAPPINGS = {}