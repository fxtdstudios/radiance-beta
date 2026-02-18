"""
═══════════════════════════════════════════════════════════════════════════════
    Radiance Film Grain v3.0 — HDR-Native Photographic Grain Simulation
                        Radiance © 2024-2026 FXTD STUDIOS

One unified node. Camera profiles + film stock emulation.
Full float32/HDR pipeline — zero output clamping.

Physically-based grain model:
 - Exposure-dependent grain density (simplified H&D characteristic curve)
 - HDR-aware luminance masking (Reinhard tonemap for mask, image untouched)
 - Per-channel spectral sensitivity matching real emulsion layers
 - Silver halide cluster simulation (non-Gaussian grain shape)
 - Halation as part of image formation (before grain compositing)
 - Photon noise √(exposure) scaling for digital sensor profiles
 - Gate weave / film base texture options

v3.0 vs v2.0:
 - MERGED: RadianceFilmGrain + RadianceCameraNoise + RadianceLensEffects → 1 node
 - FIX: 13 clamp operations that destroyed HDR — all removed or made mask-only
 - FIX: Overlay/Soft Light blend modes now HDR-safe (decompose → blend → restore)
 - FIX: Luma masking uses Reinhard tonemap (works for values 0→∞)
 - FIX: Halation detects HDR highlights (not just 0.7-1.0 range)
 - FIX: CameraNoise hdr_safe now defaults to True
 - NEW: Exposure-dependent grain density (H&D curve)
 - NEW: Non-Gaussian grain shape (grain_character parameter)
 - NEW: Gate weave simulation
 - NEW: Film base density / color cast
 - REMOVED: 3 separate nodes → 1 unified node
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("radiance.film.grain")


# ═══════════════════════════════════════════════════════════════════════════════
#                              PROFILES
# ═══════════════════════════════════════════════════════════════════════════════

# Each profile contains the complete grain character.
# Film stocks model silver halide emulsion grain.
# Camera profiles model digital sensor photon/read noise.

PROFILES = {
    # ─── FILM STOCKS ──────────────────────────────────────────────────────

    # Kodak Motion Picture Negative
    "Kodak Vision3 500T 5219": {
        "type": "film",
        "description": "Kodak Vision3 500T — Tungsten balanced, high sensitivity. The workhorse of modern cinema.",
        "iso": 500,
        "grain_intensity": 0.22,
        "grain_size": 1.2,
        "grain_softness": 0.45,
        "grain_character": 0.6,        # 0=Gaussian, 1=full silver-halide clumping
        "color_sensitivity": [1.15, 1.0, 0.90],  # RGB emulsion response
        "shadow_grain_boost": 2.0,      # Underexposed film = more visible grain
        "highlight_shoulder": 0.70,     # Where grain starts to roll off (HDR-aware)
        "halation": 0.35,              # Red-channel antihalation layer bleed
        "halation_color": [1.0, 0.25, 0.08],
        "film_base": [1.02, 0.99, 0.96],  # Orange mask (neg stock)
        "contrast": 1.1,
        "saturation": 0.95,
        "gate_weave": 0.0003,          # Sub-pixel film registration jitter
    },
    "Kodak Vision3 250D 5207": {
        "type": "film",
        "description": "Kodak Vision3 250D — Daylight balanced, fine grain. Exteriors / beauty work.",
        "iso": 250,
        "grain_intensity": 0.16,
        "grain_size": 0.9,
        "grain_softness": 0.40,
        "grain_character": 0.55,
        "color_sensitivity": [1.08, 1.0, 0.95],
        "shadow_grain_boost": 1.7,
        "highlight_shoulder": 0.75,
        "halation": 0.28,
        "halation_color": [1.0, 0.28, 0.10],
        "film_base": [1.02, 0.99, 0.96],
        "contrast": 1.05,
        "saturation": 1.0,
        "gate_weave": 0.0002,
    },
    "Kodak Vision3 50D 5203": {
        "type": "film",
        "description": "Kodak Vision3 50D — Ultra fine grain, daylight. VFX plates / maximum detail.",
        "iso": 50,
        "grain_intensity": 0.08,
        "grain_size": 0.6,
        "grain_softness": 0.30,
        "grain_character": 0.4,
        "color_sensitivity": [1.05, 1.0, 0.98],
        "shadow_grain_boost": 1.3,
        "highlight_shoulder": 0.85,
        "halation": 0.18,
        "halation_color": [1.0, 0.30, 0.12],
        "film_base": [1.01, 0.99, 0.97],
        "contrast": 1.12,
        "saturation": 1.05,
        "gate_weave": 0.00015,
    },

    # Fujifilm
    "Fuji Eterna 500T 8573": {
        "type": "film",
        "description": "Fuji Eterna 500T — Rich blacks, cool shadows. Japanese cinema staple.",
        "iso": 500,
        "grain_intensity": 0.20,
        "grain_size": 1.1,
        "grain_softness": 0.42,
        "grain_character": 0.55,
        "color_sensitivity": [1.0, 1.05, 1.12],
        "shadow_grain_boost": 1.9,
        "highlight_shoulder": 0.72,
        "halation": 0.30,
        "halation_color": [0.95, 0.30, 0.15],
        "film_base": [1.01, 1.0, 0.98],
        "contrast": 1.15,
        "saturation": 0.92,
        "gate_weave": 0.0003,
    },
    "Fuji Eterna Vivid 500T 8547": {
        "type": "film",
        "description": "Fuji Eterna Vivid — Saturated variant, music videos / commercials.",
        "iso": 500,
        "grain_intensity": 0.21,
        "grain_size": 1.15,
        "grain_softness": 0.40,
        "grain_character": 0.55,
        "color_sensitivity": [1.08, 1.02, 1.10],
        "shadow_grain_boost": 1.8,
        "highlight_shoulder": 0.70,
        "halation": 0.28,
        "halation_color": [0.98, 0.28, 0.12],
        "film_base": [1.01, 1.0, 0.98],
        "contrast": 1.2,
        "saturation": 1.1,
        "gate_weave": 0.0003,
    },

    # CineStill (remjet-removed motion picture stock)
    "CineStill 800T": {
        "type": "film",
        "description": "CineStill 800T — Remjet-removed 500T, extreme halation. Neon / night scenes.",
        "iso": 800,
        "grain_intensity": 0.26,
        "grain_size": 1.3,
        "grain_softness": 0.48,
        "grain_character": 0.65,
        "color_sensitivity": [1.12, 1.0, 0.88],
        "shadow_grain_boost": 2.2,
        "highlight_shoulder": 0.55,
        "halation": 0.65,              # Extreme — no antihalation backing
        "halation_color": [1.0, 0.20, 0.05],
        "film_base": [1.03, 0.98, 0.94],
        "contrast": 1.05,
        "saturation": 0.98,
        "gate_weave": 0.00035,
    },
    "CineStill 50D": {
        "type": "film",
        "description": "CineStill 50D — Daylight remjet-removed. Clean with subtle halation.",
        "iso": 50,
        "grain_intensity": 0.09,
        "grain_size": 0.65,
        "grain_softness": 0.28,
        "grain_character": 0.4,
        "color_sensitivity": [1.06, 1.0, 0.96],
        "shadow_grain_boost": 1.3,
        "highlight_shoulder": 0.82,
        "halation": 0.25,
        "halation_color": [1.0, 0.25, 0.08],
        "film_base": [1.02, 0.99, 0.96],
        "contrast": 1.1,
        "saturation": 1.02,
        "gate_weave": 0.00015,
    },

    # B&W
    "Kodak Double-X 5222": {
        "type": "film",
        "description": "Kodak Double-X — Classic B&W cinema. Lighthouse, Schindler's List.",
        "iso": 250,
        "grain_intensity": 0.18,
        "grain_size": 1.0,
        "grain_softness": 0.40,
        "grain_character": 0.7,
        "color_sensitivity": [1.0, 1.0, 1.0],
        "shadow_grain_boost": 1.8,
        "highlight_shoulder": 0.75,
        "halation": 0.20,
        "halation_color": [0.9, 0.9, 0.9],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.25,
        "saturation": 0.0,            # B&W
        "gate_weave": 0.0003,
    },
    "Kodak Tri-X 400 7266": {
        "type": "film",
        "description": "Kodak Tri-X 400 — Iconic B&W. Gritty, contrasty, photojournalism grain.",
        "iso": 400,
        "grain_intensity": 0.24,
        "grain_size": 1.15,
        "grain_softness": 0.38,
        "grain_character": 0.75,
        "color_sensitivity": [1.0, 1.0, 1.0],
        "shadow_grain_boost": 2.1,
        "highlight_shoulder": 0.68,
        "halation": 0.22,
        "halation_color": [0.9, 0.9, 0.9],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.35,
        "saturation": 0.0,
        "gate_weave": 0.00035,
    },

    # Vintage / Specialty
    "Super 8mm (Kodachrome)": {
        "type": "film",
        "description": "Super 8mm Kodachrome look — Heavy grain, warm, home movie texture.",
        "iso": 200,
        "grain_intensity": 0.40,
        "grain_size": 2.0,
        "grain_softness": 0.60,
        "grain_character": 0.8,
        "color_sensitivity": [1.20, 1.0, 0.85],
        "shadow_grain_boost": 2.8,
        "highlight_shoulder": 0.50,
        "halation": 0.60,
        "halation_color": [1.0, 0.22, 0.06],
        "film_base": [1.06, 0.98, 0.90],
        "contrast": 0.90,
        "saturation": 0.80,
        "gate_weave": 0.002,
    },
    "16mm (Ektachrome)": {
        "type": "film",
        "description": "16mm Ektachrome reversal — Saturated, documentary grain.",
        "iso": 160,
        "grain_intensity": 0.30,
        "grain_size": 1.5,
        "grain_softness": 0.50,
        "grain_character": 0.7,
        "color_sensitivity": [1.10, 1.05, 0.92],
        "shadow_grain_boost": 2.4,
        "highlight_shoulder": 0.60,
        "halation": 0.35,
        "halation_color": [1.0, 0.30, 0.10],
        "film_base": [1.03, 1.0, 0.95],
        "contrast": 1.2,
        "saturation": 1.15,
        "gate_weave": 0.001,
    },

    # ─── DIGITAL CAMERA SENSORS ───────────────────────────────────────────

    # ARRI
    "ARRI Alexa 35": {
        "type": "digital",
        "description": "ARRI Alexa 35 — Super 35 4.6K. The gold standard.",
        "iso": 800,
        "grain_intensity": 0.12,
        "grain_size": 0.8,
        "grain_softness": 0.30,
        "grain_character": 0.15,        # Digital = mostly Gaussian
        "color_sensitivity": [1.0, 0.95, 1.1],  # Bayer pattern sensitivity
        "shadow_grain_boost": 1.4,
        "highlight_shoulder": 0.85,
        "halation": 0.0,               # No halation on digital
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],  # No film base
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,             # No gate weave
        "noise_floor": 0.008,          # Digital-specific: read noise floor
        "color_science": "LogC4",
    },
    "ARRI Alexa Mini LF": {
        "type": "digital",
        "description": "ARRI Alexa Mini LF — Large Format. Lower noise, shallower DoF.",
        "iso": 800,
        "grain_intensity": 0.10,
        "grain_size": 0.7,
        "grain_softness": 0.25,
        "grain_character": 0.12,
        "color_sensitivity": [1.0, 0.98, 1.05],
        "shadow_grain_boost": 1.3,
        "highlight_shoulder": 0.88,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.006,
        "color_science": "LogC3",
    },
    "ARRI Alexa Classic": {
        "type": "digital",
        "description": "ARRI Alexa Classic — Original ALEV III sensor. Warm, organic digital.",
        "iso": 800,
        "grain_intensity": 0.15,
        "grain_size": 0.9,
        "grain_softness": 0.35,
        "grain_character": 0.18,
        "color_sensitivity": [1.05, 1.0, 1.1],
        "shadow_grain_boost": 1.5,
        "highlight_shoulder": 0.82,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.010,
        "color_science": "LogC3",
    },

    # RED
    "RED V-Raptor XL 8K": {
        "type": "digital",
        "description": "RED V-Raptor XL — 8K Vista Vision. Clean, detailed, VFX-friendly.",
        "iso": 800,
        "grain_intensity": 0.08,
        "grain_size": 0.5,
        "grain_softness": 0.20,
        "grain_character": 0.10,
        "color_sensitivity": [1.0, 1.0, 1.0],
        "shadow_grain_boost": 1.2,
        "highlight_shoulder": 0.90,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.005,
        "color_science": "IPP2",
    },
    "RED Komodo 6K": {
        "type": "digital",
        "description": "RED Komodo — 6K Super 35. Compact body, global shutter.",
        "iso": 800,
        "grain_intensity": 0.11,
        "grain_size": 0.65,
        "grain_softness": 0.25,
        "grain_character": 0.12,
        "color_sensitivity": [1.02, 1.0, 1.03],
        "shadow_grain_boost": 1.35,
        "highlight_shoulder": 0.85,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.007,
        "color_science": "IPP2",
    },
    "RED Monstro 8K VV": {
        "type": "digital",
        "description": "RED Monstro 8K — Full Frame Vista Vision. Helium successor.",
        "iso": 800,
        "grain_intensity": 0.09,
        "grain_size": 0.55,
        "grain_softness": 0.22,
        "grain_character": 0.10,
        "color_sensitivity": [1.0, 1.0, 1.02],
        "shadow_grain_boost": 1.25,
        "highlight_shoulder": 0.88,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.006,
        "color_science": "IPP2",
    },

    # Sony
    "Sony Venice 2": {
        "type": "digital",
        "description": "Sony Venice 2 — 8.6K Full Frame. Dual ISO, 16 stops DR.",
        "iso": 800,
        "grain_intensity": 0.09,
        "grain_size": 0.6,
        "grain_softness": 0.28,
        "grain_character": 0.12,
        "color_sensitivity": [0.98, 1.0, 1.05],
        "shadow_grain_boost": 1.2,
        "highlight_shoulder": 0.90,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.005,
        "color_science": "S-Log3",
    },
    "Sony FX9": {
        "type": "digital",
        "description": "Sony FX9 — 6K Full Frame. Documentary workhorse.",
        "iso": 800,
        "grain_intensity": 0.12,
        "grain_size": 0.7,
        "grain_softness": 0.30,
        "grain_character": 0.14,
        "color_sensitivity": [0.98, 1.0, 1.08],
        "shadow_grain_boost": 1.35,
        "highlight_shoulder": 0.85,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.008,
        "color_science": "S-Log3",
    },
    "Sony FX6": {
        "type": "digital",
        "description": "Sony FX6 — 4K Full Frame. Budget cinema body.",
        "iso": 800,
        "grain_intensity": 0.14,
        "grain_size": 0.75,
        "grain_softness": 0.32,
        "grain_character": 0.15,
        "color_sensitivity": [0.97, 1.0, 1.10],
        "shadow_grain_boost": 1.4,
        "highlight_shoulder": 0.82,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.009,
        "color_science": "S-Log3",
    },
    "Sony A7S III": {
        "type": "digital",
        "description": "Sony A7S III — 12MP Full Frame. Low-light king, large photosites.",
        "iso": 800,
        "grain_intensity": 0.07,
        "grain_size": 0.85,
        "grain_softness": 0.35,
        "grain_character": 0.10,
        "color_sensitivity": [0.96, 1.0, 1.12],
        "shadow_grain_boost": 1.1,
        "highlight_shoulder": 0.80,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.004,
        "color_science": "S-Log3",
    },

    # Blackmagic
    "Blackmagic URSA Mini Pro 12K": {
        "type": "digital",
        "description": "Blackmagic URSA Mini Pro 12K — Highest resolution S35 sensor.",
        "iso": 800,
        "grain_intensity": 0.13,
        "grain_size": 0.45,
        "grain_softness": 0.20,
        "grain_character": 0.15,
        "color_sensitivity": [1.05, 1.0, 1.08],
        "shadow_grain_boost": 1.5,
        "highlight_shoulder": 0.80,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.010,
        "color_science": "BMD Film Gen5",
    },
    "Blackmagic Pocket 6K Pro": {
        "type": "digital",
        "description": "Blackmagic Pocket 6K Pro — Super 35 indie cinema camera.",
        "iso": 800,
        "grain_intensity": 0.16,
        "grain_size": 0.70,
        "grain_softness": 0.28,
        "grain_character": 0.18,
        "color_sensitivity": [1.08, 1.0, 1.12],
        "shadow_grain_boost": 1.6,
        "highlight_shoulder": 0.75,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.012,
        "color_science": "BMD Film Gen5",
    },

    # Canon
    "Canon C70": {
        "type": "digital",
        "description": "Canon C70 — Super 35 DGO sensor. Dual gain, low noise.",
        "iso": 800,
        "grain_intensity": 0.11,
        "grain_size": 0.72,
        "grain_softness": 0.28,
        "grain_character": 0.12,
        "color_sensitivity": [1.0, 0.98, 1.05],
        "shadow_grain_boost": 1.3,
        "highlight_shoulder": 0.88,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.007,
        "color_science": "Canon Log 3",
    },

    # Panasonic
    "Panasonic Varicam LT": {
        "type": "digital",
        "description": "Panasonic Varicam LT — Super 35 cinema sensor. Dual ISO 800/5000.",
        "iso": 800,
        "grain_intensity": 0.10,
        "grain_size": 0.72,
        "grain_softness": 0.28,
        "grain_character": 0.13,
        "color_sensitivity": [1.0, 1.0, 1.06],
        "shadow_grain_boost": 1.3,
        "highlight_shoulder": 0.86,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.007,
        "color_science": "V-Log",
    },

    # IMAX
    "IMAX Digital": {
        "type": "digital",
        "description": "IMAX Digital — Dual 2K/4K projection system. Ultra clean.",
        "iso": 800,
        "grain_intensity": 0.06,
        "grain_size": 0.40,
        "grain_softness": 0.18,
        "grain_character": 0.08,
        "color_sensitivity": [1.0, 1.0, 1.0],
        "shadow_grain_boost": 1.1,
        "highlight_shoulder": 0.92,
        "halation": 0.0,
        "halation_color": [1.0, 0.3, 0.1],
        "film_base": [1.0, 1.0, 1.0],
        "contrast": 1.0,
        "saturation": 1.0,
        "gate_weave": 0.0,
        "noise_floor": 0.004,
        "color_science": "IMAX DMR",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                          HDR-SAFE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _hdr_luma(img: torch.Tensor) -> torch.Tensor:
    """
    Compute luminance from HDR image using Reinhard tonemap for mask range.
    
    Critical difference from v2.0: does NOT clamp. Uses Reinhard L/(1+L)
    to compress [0, ∞) → [0, 1) for mask calculation. This means:
    - SDR content (0-1): maps ~linearly (L/(1+L) ≈ L for small L)
    - HDR highlights (1-10): smoothly compressed, still get unique mask values
    - Extreme HDR (>10): asymptotically approaches 1.0
    
    The image itself is NEVER modified — only the mask operates in [0,1).
    """
    c = img.shape[-1]
    if c >= 3:
        # BT.709 luminance
        luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        luma = img[..., 0]

    # Reinhard tonemap for mask only — handles negative values too
    luma_abs = torch.abs(luma)
    luma_mapped = luma_abs / (1.0 + luma_abs)
    return luma_mapped


def _gaussian_blur_2d(
    tensor: torch.Tensor, kernel_size: int, sigma: float
) -> torch.Tensor:
    """
    Gaussian blur on BHWC tensor. HDR-safe (no clamping).
    """
    if kernel_size <= 1 or sigma <= 0:
        return tensor
    kernel_size = kernel_size | 1  # Ensure odd

    coords = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device)
    coords -= kernel_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()

    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    x = tensor.permute(0, 3, 1, 2)
    c = x.shape[1]
    kernel = kernel_2d.expand(c, 1, kernel_size, kernel_size)
    padding = kernel_size // 2

    blurred = F.conv2d(x, kernel, padding=padding, groups=c)
    return blurred.permute(0, 2, 3, 1)


def _overlay_blend_hdr(bg: torch.Tensor, fg_centered: torch.Tensor) -> torch.Tensor:
    """
    HDR-safe Overlay blend.
    
    v2.0 problem: clamped both inputs to [0,1] — destroyed HDR.
    v3.0 fix: decompose image into [0,1] body + HDR residual,
    blend only the body, then restore residual.
    """
    # Decompose: body (clamped for blend) + residual (HDR headroom)
    bg_body = torch.clamp(bg, 0.0, 1.0)
    bg_residual = bg - bg_body  # HDR headroom preserved

    # Shift centered grain [-x,+x] → [0,1] for overlay formula
    fg_01 = (fg_centered * 0.5 + 0.5).clamp(0.0, 1.0)

    # Standard overlay
    mask = bg_body < 0.5
    result = torch.where(
        mask,
        2.0 * bg_body * fg_01,
        1.0 - 2.0 * (1.0 - bg_body) * (1.0 - fg_01)
    )

    # Restore HDR residual
    return result + bg_residual


def _soft_light_blend_hdr(bg: torch.Tensor, fg_centered: torch.Tensor) -> torch.Tensor:
    """
    HDR-safe Soft Light blend (Pegtop formula).
    Same decompose/restore strategy as overlay.
    """
    bg_body = torch.clamp(bg, 0.0, 1.0)
    bg_residual = bg - bg_body

    fg_01 = (fg_centered * 0.5 + 0.5).clamp(0.0, 1.0)
    result = (1.0 - 2.0 * fg_01) * bg_body * bg_body + 2.0 * fg_01 * bg_body

    return result + bg_residual


def _generate_grain_texture(
    b: int, h: int, w: int, c: int,
    grain_size: float,
    grain_character: float,
    color_sensitivity: list,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Generate grain texture with physically-based characteristics.
    
    grain_character controls the distribution:
      0.0 = Pure Gaussian (digital sensor noise)
      0.5 = Moderate clumping (fine grain film)
      1.0 = Heavy silver halide clusters (coarse/vintage film)
    
    Real film grain is NOT Gaussian — silver halide crystals form
    irregular clusters. We approximate this by:
    1. Base Gaussian noise at grain-scaled resolution
    2. Squared to create positive-biased clumps (silver crystals are additive)
    3. Re-centered to zero mean
    4. Blended with original Gaussian based on grain_character
    """
    noise_h = max(1, int(h / max(grain_size, 0.1)))
    noise_w = max(1, int(w / max(grain_size, 0.1)))

    # Base Gaussian noise
    noise = torch.randn((b, noise_h, noise_w, c), device=device, generator=generator)

    # Silver halide clumping: square → re-center
    if grain_character > 0.01:
        # Squared noise creates positive-biased clumps
        clumped = noise * torch.abs(noise)  # Preserves sign, adds weight to extremes
        clumped = clumped - clumped.mean()  # Re-center to zero
        # Normalize variance to match original
        clumped = clumped / (clumped.std() + 1e-8) * noise.std()
        # Blend
        noise = noise * (1.0 - grain_character) + clumped * grain_character

    # Per-channel spectral sensitivity
    if c >= 3 and len(color_sensitivity) >= 3:
        noise[..., 0] *= color_sensitivity[0]
        noise[..., 1] *= color_sensitivity[1]
        noise[..., 2] *= color_sensitivity[2]
    if c == 4:
        noise[..., 3] = 0.0  # Never grain the alpha

    # Resize to image dimensions
    if noise_h != h or noise_w != w:
        noise = F.interpolate(
            noise.permute(0, 3, 1, 2),
            size=(h, w), mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)

    return noise


def _apply_halation(
    img: torch.Tensor,
    luma: torch.Tensor,
    halation: float,
    halation_color: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Halation: light scattering through film base around bright areas.
    
    v2.0 problem: used luma.clamp(0.7, 1.0) — missed all HDR highlights.
    v3.0 fix: uses Reinhard-mapped luma, so HDR values 1→∞ produce
    progressively brighter halation (physically correct).
    """
    if halation <= 0.001:
        return img

    c = img.shape[-1]
    if c < 3:
        return img

    # Halation source: highlights above ~0.5 in tonemapped luma
    # Since luma is already Reinhard-mapped [0,1), threshold at 0.5
    # corresponds to real value of 1.0 (perfect for HDR)
    hal_source = (luma - 0.5).clamp(min=0.0) * 2.0  # [0, ~1]

    # Blur to create glow
    kernel = max(15, int(halation * 50) | 1)
    sigma = halation * 15.0
    hal_expanded = hal_source.unsqueeze(-1).expand(img.shape[0], -1, -1, min(c, 3))
    glow = _gaussian_blur_2d(hal_expanded, kernel, sigma)

    # Apply with halation color
    hc = torch.tensor(halation_color[:3], device=device, dtype=torch.float32)
    img = img.clone()
    img[..., :3] = img[..., :3] + glow[..., :3] * hc * halation

    return img


def _apply_gate_weave(
    img: torch.Tensor,
    weave_amount: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """
    Simulate film gate registration jitter (gate weave).
    Sub-pixel random translation per frame.
    """
    if weave_amount <= 0.0:
        return img

    b, h, w, c = img.shape

    # Random offset in pixels (weave_amount is fraction of frame height)
    max_shift_px = weave_amount * h
    dx = (torch.rand(1, generator=generator, device=device).item() - 0.5) * 2.0 * max_shift_px
    dy = (torch.rand(1, generator=generator, device=device).item() - 0.5) * 2.0 * max_shift_px

    # Normalized shift for grid_sample
    shift_x = dx / w * 2.0
    shift_y = dy / h * 2.0

    # Build identity grid + offset
    theta = torch.tensor([
        [1.0, 0.0, shift_x],
        [0.0, 1.0, shift_y],
    ], device=device, dtype=torch.float32).unsqueeze(0).expand(b, -1, -1)

    grid = F.affine_grid(theta, [b, c, h, w], align_corners=False)
    img_bchw = img.permute(0, 3, 1, 2)
    shifted = F.grid_sample(img_bchw, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return shifted.permute(0, 2, 3, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#                         UNIFIED GRAIN NODE
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceFilmGrain:
    """
    Unified photographic grain simulation.
    
    Combines film stock emulation and digital camera noise profiles
    into one node with full float32/HDR support.
    
    Pipeline order (matches real image formation):
    1. Film base color cast (if film stock)
    2. Halation (light scatter before grain)
    3. Contrast / saturation adjustment
    4. Generate grain texture (non-Gaussian for film, Gaussian for digital)
    5. Exposure-dependent grain density mask (H&D curve)
    6. Luminance-aware compositing (shadow boost + highlight shoulder)
    7. Blend (Additive / Overlay / Soft Light — all HDR-safe)
    8. Gate weave (if film stock)
    9. Output — ZERO clamping, full float32 passthrough
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "profile": (list(PROFILES.keys()), {
                    "default": "Kodak Vision3 500T 5219",
                    "tooltip": "Camera or film stock profile. Film stocks include halation, gate weave, and silver halide grain character.",
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Global grain intensity multiplier. 1.0 = profile default.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible grain patterns.",
                }),
            },
            "optional": {
                "size": ("FLOAT", {
                    "default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1,
                    "tooltip": "Grain particle size multiplier. >1 = coarser, <1 = finer.",
                }),
                "iso": ("INT", {
                    "default": 0, "min": 0, "max": 12800, "step": 100,
                    "tooltip": "Override ISO (0 = use profile default). Higher ISO = more noise.",
                }),
                "halation": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Override halation strength (0 = use profile default).",
                }),
                "blend_mode": (["Additive", "Overlay", "Soft Light"], {
                    "default": "Additive",
                    "tooltip": "Grain compositing mode. All modes are HDR-safe in v3.0.",
                }),
                "gate_weave": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.01, "step": 0.0001,
                    "tooltip": "Override gate weave (0 = use profile default). Film registration jitter.",
                }),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "FXTD Studios/Radiance/Filter"
    DESCRIPTION = (
        "Unified photographic grain simulation with camera and film stock profiles. "
        "Full float32/HDR pipeline — zero output clamping. Physically-based grain "
        "with silver halide clumping, exposure-dependent density, and HDR-safe compositing."
    )

    def apply_grain(
        self,
        image: torch.Tensor,
        profile: str,
        intensity: float,
        seed: int,
        size: float = 1.0,
        iso: int = 0,
        halation: float = 0.0,
        blend_mode: str = "Additive",
        gate_weave: float = 0.0,
        use_gpu: bool = True,
    ) -> tuple:

        # ── Validate ──
        if not isinstance(image, torch.Tensor):
            logger.error(f"Expected torch.Tensor, got {type(image)}")
            return (image,)
        if image.dim() not in (3, 4):
            logger.error(f"Expected 3D/4D tensor, got {image.dim()}D")
            return (image,)

        # ── Device ──
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        img = image.to(device).float()
        if img.dim() == 3:
            img = img.unsqueeze(0)

        b, h, w, c = img.shape

        # ── Load Profile ──
        if profile not in PROFILES:
            logger.warning(f"Unknown profile '{profile}', using Kodak Vision3 500T 5219")
            profile = "Kodak Vision3 500T 5219"
        prof = PROFILES[profile]

        grain_intensity = prof["grain_intensity"] * intensity
        grain_size = prof["grain_size"] * size
        grain_softness = prof["grain_softness"]
        grain_character = prof["grain_character"]
        color_sensitivity = prof["color_sensitivity"]
        shadow_boost = prof["shadow_grain_boost"]
        highlight_shoulder = prof["highlight_shoulder"]
        hal_strength = halation if halation > 0 else prof["halation"]
        hal_color = prof["halation_color"]
        film_base = prof["film_base"]
        contrast = prof.get("contrast", 1.0)
        saturation = prof.get("saturation", 1.0)
        weave = gate_weave if gate_weave > 0 else prof.get("gate_weave", 0.0)

        # ISO scaling (digital profiles use photon noise model)
        if prof["type"] == "digital":
            noise_floor = prof.get("noise_floor", 0.008)
            profile_iso = prof.get("iso", 800)
            actual_iso = iso if iso > 0 else profile_iso
            # Photon noise ∝ √(ISO) — fundamental physics
            iso_scale = math.sqrt(actual_iso / profile_iso)
            grain_intensity = noise_floor * iso_scale * intensity
        elif iso > 0:
            # Film stocks: ISO push/pull changes grain character
            profile_iso = prof.get("iso", 500)
            push_stops = math.log2(max(iso, 1) / max(profile_iso, 1))
            grain_intensity *= 2.0 ** (push_stops * 0.5)  # ~√2 per stop push

        # ── Seed ──
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # ── Step 1: Film base color cast ──
        # Film stocks have an orange mask (neg) or tint (reversal)
        if film_base != [1.0, 1.0, 1.0] and c >= 3:
            fb = torch.tensor(film_base[:3], device=device, dtype=torch.float32)
            img = img.clone()
            img[..., :3] = img[..., :3] * fb

        # ── Step 2: HDR-aware luminance for all masking ──
        luma = _hdr_luma(img)

        # ── Step 3: Halation (before grain — part of image formation) ──
        img = _apply_halation(img, luma, hal_strength, hal_color, device)

        # ── Step 4: Contrast / saturation (film stock character) ──
        if contrast != 1.0 and c >= 3:
            # Apply contrast around midpoint (0.5 in tonemapped space)
            # For HDR: apply in Reinhard-mapped space, then unmapped
            # Simpler: just scale relative to mean luminance
            mean_luma = luma.mean()
            img = img.clone() if not img.requires_grad else img
            # Scale channels relative to their mean
            for ch in range(min(c, 3)):
                ch_mean = img[..., ch].mean()
                img[..., ch] = ch_mean + (img[..., ch] - ch_mean) * contrast

        if saturation != 1.0 and c >= 3:
            # BT.709 desaturation in linear space
            luma_3ch = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).unsqueeze(-1)
            img = img.clone() if not img.requires_grad else img
            img[..., :3] = luma_3ch + (img[..., :3] - luma_3ch) * saturation

        # ── Step 5: Generate grain texture ──
        noise = _generate_grain_texture(
            b, h, w, c,
            grain_size, grain_character, color_sensitivity,
            device, generator
        )

        # ── Step 6: Grain softness (emulsion scatter / sensor read blur) ──
        if grain_softness > 0:
            kernel_size = max(3, int(grain_softness * 14) | 1)
            sigma = 0.5 + grain_softness * 3.5
            noise = _gaussian_blur_2d(noise, kernel_size, sigma)

        # ── Step 7: Exposure-dependent grain density mask ──
        # This is the H&D (Hurter-Driffield) characteristic curve approximation.
        #
        # Real film has MORE visible grain in shadows (underexposed silver
        # halide clusters are larger, more randomly distributed) and LESS
        # grain in highlights (fully-developed crystals are dense, uniform).
        #
        # For digital sensors: read noise dominates in shadows,
        # shot noise is uniform, highlights clip hard.
        #
        # We use the Reinhard-mapped luma so HDR values get proper treatment:
        # - Shadow (luma < 0.3): boosted by shadow_boost factor
        # - Midtone (0.3 - shoulder): baseline grain
        # - Highlight (> shoulder): smooth rolloff to protect

        # Shadow boost: quadratic curve peaking in deep shadows
        shadow_factor = 1.0 + (shadow_boost - 1.0) * torch.pow(1.0 - luma, 2.0)

        # Highlight shoulder: smoothstep rolloff
        shoulder_t = ((luma - highlight_shoulder) / max(1.0 - highlight_shoulder, 0.01))
        shoulder_t = shoulder_t.clamp(0.0, 1.0)
        # Smoothstep: 3t² - 2t³
        highlight_fade = 1.0 - (3.0 * shoulder_t ** 2 - 2.0 * shoulder_t ** 3)

        density_mask = (shadow_factor * highlight_fade).unsqueeze(-1)

        # ── Step 8: Composite grain ──
        final_grain = noise * grain_intensity * density_mask

        if blend_mode == "Overlay":
            output = _overlay_blend_hdr(img, final_grain)
        elif blend_mode == "Soft Light":
            output = _soft_light_blend_hdr(img, final_grain)
        else:
            # Additive (default, most physically accurate)
            output = img + final_grain

        # ── Step 9: Gate weave (film only) ──
        if weave > 0:
            output = _apply_gate_weave(output, weave, generator, device)

        # ── Step 10: Output — ZERO clamping ──
        # Full float32 passthrough. HDR values preserved exactly.
        # Downstream nodes (viewer, export, color management) handle range.

        # Cleanup GPU intermediates
        del img, noise, final_grain, density_mask, luma

        return (output.cpu(),)


# ═══════════════════════════════════════════════════════════════════════════════
#                          NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceFilmGrain": RadianceFilmGrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceFilmGrain": "◎ Radiance Film Grain",
}
