"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE LUT NODES v1.0
              3D LUT Bake & Apply for ComfyUI Pipelines
                       Radiance © 2024-2026

 Nodes:
   RadianceLUTBake  — Bake RadianceGrade parameters into a 33³ .cube LUT file
   RadianceLUTApply — Apply an external .cube LUT file to any IMAGE tensor

 .cube format is DaVinci Resolve / Nuke / Premiere compatible.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import logging
import math
from typing import Tuple

import torch
import numpy as np

logger = logging.getLogger("radiance.lut")

# ─────────────────────────────────────────────────────────────────────────────
#                           CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

LUT_SIZES = [17, 33, 65]          # Standard .cube grid sizes
DEFAULT_LUT_SIZE = 33             # 33³ = Resolve / ACES standard


# ─────────────────────────────────────────────────────────────────────────────
#                       PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_grade_to_tensor(
    img: torch.Tensor,
    lift_r: float, lift_g: float, lift_b: float,
    gamma_r: float, gamma_g: float, gamma_b: float,
    gain_r: float, gain_g: float, gain_b: float,
    offset_r: float, offset_g: float, offset_b: float,
    contrast: float, pivot: float, saturation: float,
) -> torch.Tensor:
    """Apply RadianceGrade math to an arbitrary tensor. Output shape = input shape."""
    out = img.clone().float()

    # 1. Lift
    out[..., 0] += lift_r;  out[..., 1] += lift_g;  out[..., 2] += lift_b

    # 2. Gain
    out[..., 0] *= gain_r;  out[..., 1] *= gain_g;  out[..., 2] *= gain_b

    # 3. Offset
    out[..., 0] += offset_r;  out[..., 1] += offset_g;  out[..., 2] += offset_b

    # 4. Gamma (sign-preserving)
    eps = 1e-8
    for ch, g in enumerate([gamma_r, gamma_g, gamma_b]):
        if g != 1.0:
            out[..., ch] = torch.sign(out[..., ch]) * torch.pow(
                torch.abs(out[..., ch]) + eps, 1.0 / g
            )

    # 5. Contrast
    if contrast != 1.0:
        out = (out - pivot) * contrast + pivot

    # 6. Saturation
    if saturation != 1.0 and out.shape[-1] >= 3:
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        luma = luma.unsqueeze(-1)
        out = luma + saturation * (out - luma)

    return out


def _generate_cube_lut(
    size: int,
    lift_r: float, lift_g: float, lift_b: float,
    gamma_r: float, gamma_g: float, gamma_b: float,
    gain_r: float, gain_g: float, gain_b: float,
    offset_r: float, offset_g: float, offset_b: float,
    contrast: float, pivot: float, saturation: float,
) -> torch.Tensor:
    """Build size³×3 LUT tensor from grade parameters. Returns float32 CPU tensor."""
    # Build identity grid [0,1] in B-fast (R varies fastest in .cube)
    lin = torch.linspace(0.0, 1.0, size)
    # .cube order: R fastest, B slowest
    r_grid = lin.repeat(size * size)
    g_grid = lin.repeat_interleave(size).repeat(size)
    b_grid = lin.repeat_interleave(size * size)

    grid = torch.stack([r_grid, g_grid, b_grid], dim=-1)  # (size³, 3)

    out = _apply_grade_to_tensor(
        grid,
        lift_r, lift_g, lift_b,
        gamma_r, gamma_g, gamma_b,
        gain_r, gain_g, gain_b,
        offset_r, offset_g, offset_b,
        contrast, pivot, saturation,
    )
    # Clamp to [0,1] — .cube standard requires values in this range for SDR LUTs
    return torch.clamp(out, 0.0, 1.0)


def _write_cube_file(path: str, lut: torch.Tensor, size: int, title: str = "Radiance Grade") -> None:
    """Write a .cube file from a (size³, 3) float32 tensor."""
    lut_np = lut.cpu().numpy()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TITLE \"{title}\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
        for i in range(len(lut_np)):
            r, g, b = lut_np[i]
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")


def _parse_cube_file(path: str) -> Tuple[torch.Tensor, int]:
    """Parse a .cube file. Returns (lut_tensor [N³,3], size)."""
    MAX_LUT_SIZE = 128  # Security: prevent memory DoS (128³×3×4 = 25MB max)
    size = None
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                if size > MAX_LUT_SIZE:
                    raise ValueError(
                        f"LUT_3D_SIZE {size} exceeds maximum allowed size of {MAX_LUT_SIZE}. "
                        f"Industry standard is 33 or 65."
                    )
                if size < 2:
                    raise ValueError(f"LUT_3D_SIZE {size} is too small (minimum 2)")
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    values.append([float(x) for x in parts])
                except ValueError:
                    pass
    if size is None:
        raise ValueError("No LUT_3D_SIZE found in .cube file")
    if len(values) != size ** 3:
        raise ValueError(f"Expected {size**3} entries, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32), size


def _apply_cube_lut_to_image(img: torch.Tensor, lut: torch.Tensor, lut_size: int, clamp_input: bool = True) -> torch.Tensor:
    """
    Trilinear lookup of a (N³, 3) LUT on an (..., H, W, 3) image tensor.
    Returns same shape as img.
    """
    orig_shape = img.shape
    flat = img.reshape(-1, 3)
    if clamp_input:
        flat = flat.clamp(0.0, 1.0)
    n = lut_size

    # .cube: R fastest, B slowest → index = B*n² + G*n + R
    rc = flat[:, 0] * (n - 1)
    gc = flat[:, 1] * (n - 1)
    bc = flat[:, 2] * (n - 1)

    r0 = rc.long().clamp(0, n - 2);  r1 = r0 + 1
    g0 = gc.long().clamp(0, n - 2);  g1 = g0 + 1
    b0 = bc.long().clamp(0, n - 2);  b1 = b0 + 1

    rf = rc - r0.float()
    gf = gc - g0.float()
    bf = bc - b0.float()

    def idx(r, g, b):
        return b * (n * n) + g * n + r

    lut = lut.to(img.device)
    c000 = lut[idx(r0, g0, b0)]
    c100 = lut[idx(r1, g0, b0)]
    c010 = lut[idx(r0, g1, b0)]
    c110 = lut[idx(r1, g1, b0)]
    c001 = lut[idx(r0, g0, b1)]
    c101 = lut[idx(r1, g0, b1)]
    c011 = lut[idx(r0, g1, b1)]
    c111 = lut[idx(r1, g1, b1)]

    rf = rf.unsqueeze(1);  gf = gf.unsqueeze(1);  bf = bf.unsqueeze(1)

    out = (
        c000 * (1 - rf) * (1 - gf) * (1 - bf) +
        c100 *      rf  * (1 - gf) * (1 - bf) +
        c010 * (1 - rf) *      gf  * (1 - bf) +
        c110 *      rf  *      gf  * (1 - bf) +
        c001 * (1 - rf) * (1 - gf) *      bf  +
        c101 *      rf  * (1 - gf) *      bf  +
        c011 * (1 - rf) *      gf  *      bf  +
        c111 *      rf  *      gf  *      bf
    )
    return out.reshape(orig_shape)


# ─────────────────────────────────────────────────────────────────────────────
#                           NODES
# ─────────────────────────────────────────────────────────────────────────────

class RadianceLUTBake:
    """
    Bake a RadianceGrade into a .cube LUT file (33³ by default).
    Compatible with DaVinci Resolve, Nuke, ACES, Premiere Pro.
    Also outputs the IMAGE with the LUT applied as a preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image for preview output."}),
                "output_path": (
                    "STRING",
                    {
                        "default": "output/grade.cube",
                        "multiline": False,
                        "tooltip": "Path to write the .cube file. Relative paths resolve from ComfyUI root.",
                    },
                ),
                "lut_title": (
                    "STRING",
                    {"default": "Radiance Grade", "multiline": False},
                ),
                "lut_size": (
                    [str(s) for s in LUT_SIZES],
                    {"default": "33", "tooltip": "LUT grid size. 33 = industry standard. 65 = high precision."},
                ),
            },
            "optional": {
                "lift_r":     ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "lift_g":     ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "lift_b":     ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "gamma_r":    ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001}),
                "gamma_g":    ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001}),
                "gamma_b":    ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001}),
                "gain_r":     ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 5.0, "step": 0.001}),
                "gain_g":     ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 5.0, "step": 0.001}),
                "gain_b":     ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 5.0, "step": 0.001}),
                "offset_r":   ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "offset_g":   ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "offset_b":   ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "contrast":   ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 3.0, "step": 0.01}),
                "pivot":      ("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 3.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "cube_path")
    OUTPUT_TOOLTIPS = (
        "Input image with LUT applied (identical to RadianceGrade output).",
        "Absolute path of the saved .cube file.",
    )
    FUNCTION = "bake"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Bake Radiance grade parameters into a .cube LUT file for Resolve / Nuke / Premiere."

    def bake(
        self, image, output_path, lut_title, lut_size,
        lift_r=0.0, lift_g=0.0, lift_b=0.0,
        gamma_r=1.0, gamma_g=1.0, gamma_b=1.0,
        gain_r=1.0, gain_g=1.0, gain_b=1.0,
        offset_r=0.0, offset_g=0.0, offset_b=0.0,
        contrast=1.0, pivot=0.5, saturation=1.0,
    ):
        size = int(lut_size)

        # Resolve output path
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        # Bake LUT
        lut = _generate_cube_lut(
            size,
            lift_r, lift_g, lift_b,
            gamma_r, gamma_g, gamma_b,
            gain_r, gain_g, gain_b,
            offset_r, offset_g, offset_b,
            contrast, pivot, saturation,
        )
        _write_cube_file(abs_path, lut, size, title=lut_title)
        logger.info(f"[Radiance LUT] Saved {size}³ .cube → {abs_path}")

        # Apply to input image for preview
        preview = _apply_cube_lut_to_image(image.float(), lut, size, clamp_input=True)

        return (preview, abs_path)


class RadianceLUTApply:
    """
    Apply an external .cube LUT file to any IMAGE tensor.
    Uses CPU trilinear interpolation (GPU-safe via torch ops).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cube_path": (
                    "STRING",
                    {
                        "default": "output/grade.cube",
                        "multiline": False,
                        "tooltip": "Path to the .cube LUT file to apply.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blend strength: 0 = bypass, 1 = full LUT.",
                    },
                ),
                "clamp_input": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Clamp (SDR)",
                        "label_off": "No Clamp (HDR)",
                        "tooltip": "Clamp input image to 0-1 range before LUT application. Standard .cube LUTs expect 0-1 range.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_lut"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Apply an external .cube LUT file (Resolve / Nuke / Premiere format) to an image."

    def apply_lut(self, image: torch.Tensor, cube_path: str, strength: float = 1.0, clamp_input: bool = True):
        abs_path = os.path.abspath(cube_path)

        # Security: validate file extension
        if not abs_path.lower().endswith(".cube"):
            logger.error(f"[Radiance LUT] Not a .cube file: {abs_path}")
            return (image,)

        if not os.path.isfile(abs_path):
            logger.error(f"[Radiance LUT] .cube file not found: {abs_path}")
            return (image,)

        try:
            lut, lut_size = _parse_cube_file(abs_path)
        except Exception as e:
            logger.error(f"[Radiance LUT] Failed to parse .cube: {e}")
            return (image,)

        img_f = image.float()
        graded = _apply_cube_lut_to_image(img_f, lut, lut_size, clamp_input=clamp_input)

        if strength < 1.0:
            graded = img_f * (1.0 - strength) + graded * strength

        return (graded,)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "RadianceLUTBake":  RadianceLUTBake,
    "RadianceLUTApply": RadianceLUTApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLUTBake": "◎ Radiance LUT Bake",
    "RadianceLUTApply": "◎ Radiance LUT Apply",
}
