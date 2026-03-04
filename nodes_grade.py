"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE GRADE NODE v2.1
              Professional Color Grading for ComfyUI
                       Radiance © 2024-2026

 v2.0 — Grade Matching & Automation
   - NEW: reference_image + match_strength → LAB mean/std match grade
   - NEW: preset_file → load presets from external JSON
   - NEW: grade_info is now a full JSON dump of grade parameters
   - NEW: ApplyGradeInfo node — apply grade_info JSON to any image
   - NEW: RadianceGradeMatch — dedicated match-grading node
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import logging
import torch
from typing import Tuple, Dict, Any, Optional

try:
    from .exceptions import validate_image_input
except ImportError:
    from exceptions import validate_image_input

logger = logging.getLogger("radiance.grade")

# =============================================================================
# GRADE PRESETS - Common cinematic looks
# =============================================================================

GRADE_PRESETS = {
    "None (Custom)": {
        "description": "No preset - use manual controls",
        "lift": (0.0, 0.0, 0.0),
        "gamma": (1.0, 1.0, 1.0),
        "gain": (1.0, 1.0, 1.0),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.0,
        "saturation": 1.0,
    },
    "Cinematic Teal & Orange": {
        "description": "Popular blockbuster look with teal shadows and orange highlights",
        "lift": (0.0, 0.02, 0.05),
        "gamma": (1.0, 0.98, 0.95),
        "gain": (1.05, 0.98, 0.90),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.15,
        "saturation": 1.1,
    },
    "Bleach Bypass": {
        "description": "Desaturated, high contrast - popular in war and thriller films",
        "lift": (0.02, 0.02, 0.02),
        "gamma": (0.95, 0.95, 0.95),
        "gain": (1.1, 1.1, 1.1),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.3,
        "saturation": 0.5,
    },
    "Cross Process": {
        "description": "Film cross-processing look with color shifts",
        "lift": (0.05, -0.02, 0.08),
        "gamma": (0.95, 1.05, 0.9),
        "gain": (1.1, 0.95, 1.15),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.2,
        "saturation": 1.2,
    },
    "Film Noir": {
        "description": "High contrast black and white with deep shadows",
        "lift": (-0.02, -0.02, -0.02),
        "gamma": (0.9, 0.9, 0.9),
        "gain": (1.2, 1.2, 1.2),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.4,
        "saturation": 0.0,
    },
    "Vintage Film": {
        "description": "Warm, faded look reminiscent of aged film prints",
        "lift": (0.03, 0.02, 0.0),
        "gamma": (1.05, 1.0, 0.92),
        "gain": (1.0, 0.98, 0.88),
        "offset": (0.02, 0.01, -0.02),
        "contrast": 0.9,
        "saturation": 0.85,
    },
    "Cool Blue Hour": {
        "description": "Blue-tinted look for dusk/dawn scenes",
        "lift": (0.0, 0.01, 0.04),
        "gamma": (0.98, 1.0, 1.05),
        "gain": (0.95, 1.0, 1.1),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.05,
        "saturation": 0.9,
    },
    "Golden Hour": {
        "description": "Warm golden tones for sunset scenes",
        "lift": (0.02, 0.01, -0.02),
        "gamma": (0.98, 1.0, 1.05),
        "gain": (1.1, 1.02, 0.88),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.1,
        "saturation": 1.15,
    },
    "Matrix Green": {
        "description": "Green-tinted cyberpunk look",
        "lift": (0.0, 0.02, 0.0),
        "gamma": (0.95, 1.05, 0.95),
        "gain": (0.9, 1.1, 0.9),
        "offset": (0.0, 0.01, 0.0),
        "contrast": 1.2,
        "saturation": 0.8,
    },
    "High Key Bright": {
        "description": "Bright, airy look for fashion and beauty",
        "lift": (0.03, 0.03, 0.03),
        "gamma": (0.9, 0.9, 0.9),
        "gain": (1.0, 1.0, 1.0),
        "offset": (0.05, 0.05, 0.05),
        "contrast": 0.85,
        "saturation": 0.95,
    },
    "Low Key Moody": {
        "description": "Dark, moody look with crushed blacks",
        "lift": (-0.03, -0.03, -0.03),
        "gamma": (1.1, 1.1, 1.1),
        "gain": (0.95, 0.95, 0.95),
        "offset": (-0.02, -0.02, -0.02),
        "contrast": 1.25,
        "saturation": 0.9,
    },
    "Sci-Fi Cold": {
        "description": "Cold, sterile look for sci-fi environments",
        "lift": (0.0, 0.01, 0.03),
        "gamma": (1.0, 1.0, 0.98),
        "gain": (0.95, 0.98, 1.05),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.15,
        "saturation": 0.75,
    },
    "Horror Desaturated": {
        "description": "Desaturated with green tint for horror",
        "lift": (0.0, 0.02, 0.0),
        "gamma": (1.0, 1.02, 1.0),
        "gain": (0.95, 1.0, 0.95),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.3,
        "saturation": 0.4,
    },
}


# =============================================================================
# SHARED GRADE MATH
# =============================================================================

def _apply_grade(
    image: torch.Tensor,
    lift_r: float, lift_g: float, lift_b: float,
    gamma_r: float, gamma_g: float, gamma_b: float,
    gain_r: float, gain_g: float, gain_b: float,
    offset_r: float, offset_g: float, offset_b: float,
    contrast: float, pivot: float, saturation: float,
) -> torch.Tensor:
    img = image.float().clone()

    # 1. Lift
    img[..., 0] += lift_r;  img[..., 1] += lift_g;  img[..., 2] += lift_b
    # 2. Gain
    img[..., 0] *= gain_r;  img[..., 1] *= gain_g;  img[..., 2] *= gain_b
    # 3. Offset
    img[..., 0] += offset_r;  img[..., 1] += offset_g;  img[..., 2] += offset_b
    # 4. Gamma (sign-preserving)
    eps = 1e-8
    for ch, g in enumerate([gamma_r, gamma_g, gamma_b]):
        if g != 1.0:
            img[..., ch] = torch.sign(img[..., ch]) * torch.pow(
                torch.abs(img[..., ch]) + eps, 1.0 / g
            )
    # 5. Contrast
    if contrast != 1.0:
        img = (img - pivot) * contrast + pivot
    # 6. Saturation
    if saturation != 1.0 and img.shape[-1] >= 3:
        luma = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).unsqueeze(-1)
        img = luma + saturation * (img - luma)

    return img


def _rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """Rough sRGB → CIE L*a*b* for mean/std matching. img shape (B,H,W,3) [0,1]."""
    # Linear → XYZ (D65)
    rgb = img.float().clamp(0.0, 1.0)
    # sRGB linearize (approx)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # RGB → XYZ (sRGB D65)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    # Normalize by D65 white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx = _lab_f(X / Xn);  fy = _lab_f(Y / Yn);  fz = _lab_f(Z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_out = 200.0 * (fy - fz)
    return torch.stack([L, a, b_out], dim=-1)


def _lab_f(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)


def _match_grade_params(
    source: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute CDL-compatible grade offsets to match source statistics to target.
    Works in CIE LAB space: mean offset → Lift/Gain, std ratio → Gain, saturation.
    Returns a dict of grade params that make source look like target.
    """
    src_lab = _rgb_to_lab(source)  # (B,H,W,3)
    tgt_lab = _rgb_to_lab(target)

    # Per-channel stats in LAB
    src_mean = src_lab.mean(dim=[0, 1, 2])  # (3,)
    tgt_mean = tgt_lab.mean(dim=[0, 1, 2])
    src_std  = src_lab.std(dim=[0, 1, 2]).clamp(min=1e-6)
    tgt_std  = tgt_lab.std(dim=[0, 1, 2]).clamp(min=1e-6)

    # Scale factor per LAB channel
    scale = (tgt_std / src_std).cpu().tolist()
    shift = ((tgt_mean - src_mean * (tgt_std / src_std)) / 100.0).cpu().tolist()

    # Map LAB channel 0 (L*) → luma → gain/offset
    # Map LAB channels 1,2 (a*,b*) → color cast → offset R,G,B (approx)
    # L scale → uniform gain; L shift → uniform offset
    L_scale = float(scale[0]);   L_shift = float(shift[0])
    a_shift = float(shift[1]) / 200.0  # a* maps to R−G
    b_shift = float(shift[2]) / 200.0  # b* maps to y−B

    return {
        "gain_r":   L_scale + a_shift,
        "gain_g":   L_scale - a_shift + b_shift * 0.5,
        "gain_b":   L_scale - b_shift,
        "offset_r": L_shift + a_shift * 0.5,
        "offset_g": L_shift,
        "offset_b": L_shift - b_shift * 0.5,
        "lift_r": 0.0, "lift_g": 0.0, "lift_b": 0.0,
        "gamma_r": 1.0, "gamma_g": 1.0, "gamma_b": 1.0,
        "contrast": 1.0, "pivot": 0.5, "saturation": 1.0,
    }


# =============================================================================
# RADIANCE GRADE (v2.0 — adds matching, JSON preset loading, JSON grade_info)
# =============================================================================

class RadianceGrade:
    """
    Professional color grading with per-channel Lift/Gamma/Gain/Offset, Contrast, Saturation,
    cinematic presets, optional grade matching, and JSON preset file loading.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        preset_names = list(GRADE_PRESETS.keys())
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to grade. Processed in 32-bit float precision."}),
                "preset": (
                    preset_names,
                    {"default": "None (Custom)", "tooltip": "Load a cinematic preset look."},
                ),
                "preset_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Blend between original (0) and preset (1)."},
                ),
            },
            "optional": {
                # v2.0: Grade matching
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "Optional reference image. When connected, grade parameters are computed automatically to match its color statistics (LAB mean/std)."},
                ),
                "match_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Blend between manual grade (0) and matched grade (1)."},
                ),
                # v2.0: External preset file
                "preset_file": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "Path to a JSON file with custom presets in GRADE_PRESETS format. Loaded presets override built-in ones of the same name."},
                ),
                # Lift
                "lift_r":     ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Red channel lift (shadow offset)."}),
                "lift_g":     ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Green channel lift."}),
                "lift_b":     ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Blue channel lift."}),
                # Gamma
                "gamma_r":    ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001, "tooltip": "Red channel gamma (midtone power)."}),
                "gamma_g":    ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001, "tooltip": "Green channel gamma."}),
                "gamma_b":    ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001, "tooltip": "Blue channel gamma."}),
                # Gain
                "gain_r":     ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 5.0, "step": 0.001, "tooltip": "Red channel gain (highlight multiplier)."}),
                "gain_g":     ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 5.0, "step": 0.001, "tooltip": "Green channel gain."}),
                "gain_b":     ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 5.0, "step": 0.001, "tooltip": "Blue channel gain."}),
                # Offset
                "offset_r":   ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Red channel global offset."}),
                "offset_g":   ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Green channel global offset."}),
                "offset_b":   ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Blue channel global offset."}),
                # Contrast / Saturation
                "contrast":   ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 3.0, "step": 0.01, "tooltip": "Contrast multiplier around pivot."}),
                "pivot":      ("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0, "step": 0.01, "tooltip": "Contrast pivot point (v2.1: fixed to 0.18 for scene-linear correctness)."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 3.0, "step": 0.01, "tooltip": "Luminance-preserving saturation."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")
    OUTPUT_TOOLTIPS = (
        "Graded image in 32-bit float.",
        "JSON string of all applied grade parameters — connect to ApplyGradeInfo to replicate.",
    )
    FUNCTION = "grade"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Professional color grading with per-channel LGG/Offset, cinematic presets, LAB match grading, and JSON grade export."

    def grade(
        self,
        image: torch.Tensor,
        preset: str = "None (Custom)",
        preset_strength: float = 1.0,
        reference_image: Optional[torch.Tensor] = None,
        match_strength: float = 1.0,
        preset_file: str = "",
        lift_r: float = 0.0, lift_g: float = 0.0, lift_b: float = 0.0,
        gamma_r: float = 1.0, gamma_g: float = 1.0, gamma_b: float = 1.0,
        gain_r: float = 1.0, gain_g: float = 1.0, gain_b: float = 1.0,
        offset_r: float = 0.0, offset_g: float = 0.0, offset_b: float = 0.0,
        contrast: float = 1.0, pivot: float = 0.5, saturation: float = 1.0,
    ) -> Tuple[torch.Tensor, str]:

        # --- Input validation ---
        validate_image_input(image, "RadianceGrade", require_batch=True)
        if image.shape[-1] < 3:
            logger.warning(
                f"[Grade] Image has {image.shape[-1]} channel(s), expected >= 3. "
                f"Grading requires RGB input."
            )
            # Auto-broadcast single channel to RGB
            if image.shape[-1] == 1:
                image = image.repeat(1, 1, 1, 3)

        # --- v2.0: Load external preset file ---
        presets = dict(GRADE_PRESETS)
        if preset_file and os.path.isfile(preset_file):
            try:
                with open(preset_file, encoding="utf-8") as f:
                    extra = json.load(f)
                presets.update(extra)
                logger.info(f"[Grade] Loaded {len(extra)} presets from {preset_file}")
            except Exception as e:
                logger.warning(f"[Grade] Failed to load preset file: {e}")

        # --- Apply preset blend ---
        if preset != "None (Custom)" and preset in presets and preset_strength > 0:
            p = presets[preset];  s = preset_strength
            lift_r   = lift_r   * (1-s) + p["lift"][0]   * s
            lift_g   = lift_g   * (1-s) + p["lift"][1]   * s
            lift_b   = lift_b   * (1-s) + p["lift"][2]   * s
            gamma_r  = gamma_r  * (1-s) + p["gamma"][0]  * s
            gamma_g  = gamma_g  * (1-s) + p["gamma"][1]  * s
            gamma_b  = gamma_b  * (1-s) + p["gamma"][2]  * s
            gain_r   = gain_r   * (1-s) + p["gain"][0]   * s
            gain_g   = gain_g   * (1-s) + p["gain"][1]   * s
            gain_b   = gain_b   * (1-s) + p["gain"][2]   * s
            offset_r = offset_r * (1-s) + p["offset"][0] * s
            offset_g = offset_g * (1-s) + p["offset"][1] * s
            offset_b = offset_b * (1-s) + p["offset"][2] * s
            contrast   = contrast   * (1-s) + p["contrast"]   * s
            saturation = saturation * (1-s) + p["saturation"] * s

        # --- v2.0: Grade matching ---
        if reference_image is not None and match_strength > 0:
            try:
                mp = _match_grade_params(image, reference_image)
                ms = match_strength
                gain_r   = gain_r   * (1-ms) + mp["gain_r"]   * ms
                gain_g   = gain_g   * (1-ms) + mp["gain_g"]   * ms
                gain_b   = gain_b   * (1-ms) + mp["gain_b"]   * ms
                offset_r = offset_r * (1-ms) + mp["offset_r"] * ms
                offset_g = offset_g * (1-ms) + mp["offset_g"] * ms
                offset_b = offset_b * (1-ms) + mp["offset_b"] * ms
            except Exception as e:
                logger.warning(f"[Grade] Match grading failed: {e}")

        # --- Apply grade ---
        img = _apply_grade(
            image,
            lift_r, lift_g, lift_b,
            gamma_r, gamma_g, gamma_b,
            gain_r, gain_g, gain_b,
            offset_r, offset_g, offset_b,
            contrast, pivot, saturation,
        )

        # --- v2.0: Full JSON grade_info ---
        grade_info = json.dumps({
            "preset": preset,
            "preset_strength": round(preset_strength, 4),
            "match_strength": round(match_strength, 4) if reference_image is not None else 0.0,
            "lift":   [round(lift_r, 6),   round(lift_g, 6),   round(lift_b, 6)],
            "gamma":  [round(gamma_r, 6),  round(gamma_g, 6),  round(gamma_b, 6)],
            "gain":   [round(gain_r, 6),   round(gain_g, 6),   round(gain_b, 6)],
            "offset": [round(offset_r, 6), round(offset_g, 6), round(offset_b, 6)],
            "contrast":   round(contrast, 6),
            "pivot":      round(pivot, 6),
            "saturation": round(saturation, 6),
        })

        return (img, grade_info)


# =============================================================================
# APPLY GRADE INFO  (v2.0 — closes the grade_info JSON roundtrip)
# =============================================================================

class RadianceApplyGradeInfo:
    """
    Apply a grade_info JSON string (from RadianceGrade) to any image.
    Enables grade replication across shots, or loading saved grades from disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grade_info": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect to the grade_info output of RadianceGrade.",
                    },
                ),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "Blend original (0) vs graded (1).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")
    FUNCTION = "apply"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Apply a saved grade_info JSON to any image. Replicates a RadianceGrade exactly."

    def apply(self, image: torch.Tensor, grade_info: str, strength: float = 1.0):
        # --- Input validation ---
        validate_image_input(image, "RadianceApplyGradeInfo", require_batch=True)
        if image.shape[-1] < 3:
            if image.shape[-1] == 1:
                image = image.repeat(1, 1, 1, 3)

        try:
            p = json.loads(grade_info)
        except json.JSONDecodeError as e:
            logger.error(f"[ApplyGradeInfo] Invalid JSON: {e}")
            return (image, grade_info)

        def g(key, default):
            return p.get(key, default)

        lift   = g("lift",   [0, 0, 0])
        gamma  = g("gamma",  [1, 1, 1])
        gain   = g("gain",   [1, 1, 1])
        offset = g("offset", [0, 0, 0])

        graded = _apply_grade(
            image,
            lift[0], lift[1], lift[2],
            gamma[0], gamma[1], gamma[2],
            gain[0], gain[1], gain[2],
            offset[0], offset[1], offset[2],
            g("contrast", 1.0), g("pivot", 0.5), g("saturation", 1.0),
        )

        if strength < 1.0:
            graded = image.float() * (1.0 - strength) + graded * strength

        return (graded, grade_info)


# =============================================================================
# RADIANCE GRADE MATCH  (standalone node for shot-to-shot matching)
# =============================================================================

class RadianceGradeMatch:
    """
    Match the color statistics of a source image to a reference (target) image
    using CIE L*a*b* mean and standard deviation matching.
    Outputs both the matched image and a grade_info JSON for chain use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source":    ("IMAGE", {"tooltip": "Image to be matched (will be modified)."}),
                "reference": ("IMAGE", {"tooltip": "Target image whose look is applied to source."}),
                "strength":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "0 = no change, 1 = full match."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "grade_info")
    OUTPUT_TOOLTIPS = (
        "Source image color-matched to reference.",
        "JSON grade parameters used — connect to ApplyGradeInfo for batch use.",
    )
    FUNCTION = "match"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Match source image color statistics to a reference image using LAB mean/std matching."

    def match(self, source: torch.Tensor, reference: torch.Tensor, strength: float = 1.0):
        # --- Input validation ---
        validate_image_input(source, "RadianceGradeMatch (source)", require_batch=True)
        validate_image_input(reference, "RadianceGradeMatch (reference)", require_batch=True)
        if source.shape[-1] < 3:
            if source.shape[-1] == 1:
                source = source.repeat(1, 1, 1, 3)
        if reference.shape[-1] < 3:
            if reference.shape[-1] == 1:
                reference = reference.repeat(1, 1, 1, 3)

        try:
            mp = _match_grade_params(source, reference)
        except Exception as e:
            logger.error(f"[GradeMatch] Failed: {e}")
            return (source, "{}")

        mixed = {k: v * strength + (1.0 if "gain" in k else 0.0) * (1.0 - strength)
                 for k, v in mp.items()}

        graded = _apply_grade(
            source,
            mixed["lift_r"], mixed["lift_g"], mixed["lift_b"],
            mixed["gamma_r"], mixed["gamma_g"], mixed["gamma_b"],
            mixed["gain_r"], mixed["gain_g"], mixed["gain_b"],
            mixed["offset_r"], mixed["offset_g"], mixed["offset_b"],
            mixed["contrast"], mixed["pivot"], mixed["saturation"],
        )

        info = json.dumps({
            "preset": "grade_match",
            "preset_strength": 0.0,
            "match_strength": round(strength, 4),
            "lift":   [round(mixed["lift_r"], 6),   round(mixed["lift_g"], 6),   round(mixed["lift_b"], 6)],
            "gamma":  [round(mixed["gamma_r"], 6),  round(mixed["gamma_g"], 6),  round(mixed["gamma_b"], 6)],
            "gain":   [round(mixed["gain_r"], 6),   round(mixed["gain_g"], 6),   round(mixed["gain_b"], 6)],
            "offset": [round(mixed["offset_r"], 6), round(mixed["offset_g"], 6), round(mixed["offset_b"], 6)],
            "contrast":   1.0,
            "pivot":      0.5,
            "saturation": 1.0,
        })

        return (graded, info)


# =============================================================================
NODE_CLASS_MAPPINGS = {
    "RadianceGrade":          RadianceGrade,
    "RadianceApplyGradeInfo": RadianceApplyGradeInfo,
    "RadianceGradeMatch":     RadianceGradeMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceGrade":          "◎ Radiance Grade",
    "RadianceApplyGradeInfo": "◎ Apply Grade Info",
    "RadianceGradeMatch":     "◎ Radiance Grade Match",
}
