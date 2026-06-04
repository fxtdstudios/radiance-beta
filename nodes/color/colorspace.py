from __future__ import annotations

import torch
import logging
import json
import math
from typing import Tuple

from radiance.radiance_ocio import get_ocio_manager

# Shared torch-math primitives — single source of truth for matrix ops.
# Replaces the fragile try/except private-API import from nodes_hdr_colorspace.
from radiance.color.ops import apply_matrix_3x3, M_REC709_TO_ACESCG, M_ACESCG_TO_REC709

logger = logging.getLogger("radiance.colorscience")

_BRADFORD = torch.tensor([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
], dtype=torch.float64)

_BRADFORD_INV = torch.inverse(_BRADFORD)

_ILLUMINANT_XY = {
    "D50": (0.3457, 0.3585), "D55": (0.3324, 0.3474), "D60": (0.3217, 0.3377),
    "D65": (0.3127, 0.3290), "D75": (0.2990, 0.3149), "A": (0.4476, 0.4074),
    "B": (0.3484, 0.3516), "C": (0.3101, 0.3162), "E": (0.3333, 0.3333),
}


def _xy_to_XYZ(xy: Tuple[float, float]) -> torch.Tensor:
    x, y = xy
    return torch.tensor([x / y, 1.0, (1.0 - x - y) / y], dtype=torch.float64)


def _build_bradford_matrix(src_illuminant: str, dst_illuminant: str) -> torch.Tensor:
    src_XYZ = _xy_to_XYZ(_ILLUMINANT_XY.get(src_illuminant, _ILLUMINANT_XY["D65"]))
    dst_XYZ = _xy_to_XYZ(_ILLUMINANT_XY.get(dst_illuminant, _ILLUMINANT_XY["D65"]))
    src_cone = _BRADFORD @ src_XYZ
    dst_cone = _BRADFORD @ dst_XYZ
    scale = dst_cone / src_cone.clamp(min=1e-9)
    M = _BRADFORD_INV @ torch.diag(scale) @ _BRADFORD
    return M.float()


def _temperature_to_xy(kelvin: float) -> Tuple[float, float]:
    T = max(1667.0, min(kelvin, 25000.0))
    if T <= 4000:
        x = (-0.2661239e9 / T**3 - 0.2343580e6 / T**2 + 0.8776956e3 / T + 0.179910)
    else:
        x = (-3.0258469e9 / T**3 + 2.1070379e6 / T**2 + 0.2226347e3 / T + 0.240390)
    if T <= 2222:
        y = (-1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683)
    elif T <= 4000:
        y = (-0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867)
    else:
        y = (3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483)
    return (x, y)


class RadianceWhiteBalance:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Adjust white balance using a reference neutral or colour temperature."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["Temperature / Tint", "Illuminant Adapt", "Manual RGB Gain"], {"default": "Temperature / Tint"}),
                "preset": (["Manual", "Daylight (5500K)", "Tungsten (3200K)", "Fluorescent (4200K)", "Flash (6000K)", "Shade (7500K)"], {"default": "Manual"}),
                "temperature": ("FLOAT", {"default": 6500.0, "min": 1667.0, "max": 25000.0, "step": 50.0}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.005}),
                "src_illuminant": (list(_ILLUMINANT_XY.keys()), {"default": "D65"}),
                "dst_illuminant": (list(_ILLUMINANT_XY.keys()), {"default": "D50"}),
                "gain_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001}),
                "gain_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001}),
                "gain_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "grade_info_in": ("STRING", {"forceInput": True}),
            },
        }

    def apply(self, image: torch.Tensor, mode: str, preset: str = "Manual",
              temperature: float = 6500.0, tint: float = 0.0,
              src_illuminant: str = "D65", dst_illuminant: str = "D50",
              gain_r: float = 1.0, gain_g: float = 1.0, gain_b: float = 1.0,
              strength: float = 1.0, grade_info_in: str = None):
        img = image.clone()
        device = img.device

        if preset != "Manual" and mode == "Temperature / Tint":
            preset_map = {
                "Daylight (5500K)": 5500.0, "Tungsten (3200K)": 3200.0,
                "Fluorescent (4200K)": 4200.0, "Flash (6000K)": 6000.0, "Shade (7500K)": 7500.0,
            }
            temperature = preset_map.get(preset, temperature)

        if mode == "Temperature / Tint":
            target_xy = _temperature_to_xy(temperature)
            src_xy = _ILLUMINANT_XY["D65"]
            src_XYZ = _xy_to_XYZ(src_xy)
            dst_XYZ = _xy_to_XYZ(target_xy)
            src_cone = _BRADFORD @ src_XYZ
            dst_cone = _BRADFORD @ dst_XYZ
            scale = dst_cone / src_cone.clamp(min=1e-9)
            M = (_BRADFORD_INV @ torch.diag(scale) @ _BRADFORD).float().to(device)
            tint_gain = torch.tensor([1.0, math.pow(2.0, -tint * 0.5), 1.0], dtype=torch.float32, device=device)
            result = torch.einsum('ij,...j->...i', M, img)
            result = result * tint_gain
        elif mode == "Illuminant Adapt":
            M = _build_bradford_matrix(src_illuminant, dst_illuminant).to(device)
            result = torch.einsum('ij,...j->...i', M, img)
        else:
            gains = torch.tensor([gain_r, gain_g, gain_b], dtype=torch.float32, device=device)
            result = img * gains

        if strength < 1.0:
            result = torch.lerp(image, result, strength)

        grade_info = json.dumps({
            "node": "RadianceWhiteBalance", "mode": mode, "preset": preset,
            "temperature": temperature, "tint": tint,
            "src_illuminant": src_illuminant, "dst_illuminant": dst_illuminant,
            "gain_rgb": [gain_r, gain_g, gain_b], "strength": strength,
            "upstream": grade_info_in,
        })
        return (result, grade_info)


class RadianceColorSpaceConvert:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Convert images between named colour spaces."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")

    _COLOR_SPACES = [
        "Linear sRGB (D65)", "ACEScg", "ACEScc", "ACEScct",
        "sRGB (OETF encoded)", "Rec.709 (OETF encoded)", "Rec.709 / BT.1886",
        "LogC3 (ARRI EI800)", "LogC4 (ARRI Alexa 35)", "F-Log2 (Fujifilm)",
        "C-Log3 (Canon)", "Log3G10 (RED IPP2)", "DaVinci Intermediate",
        "BMD Film Gen5", "V-Log (Panasonic)", "N-Log (Nikon)",
    ]

    # Reference the module-level imports — avoids a second tensor allocation and
    # keeps precision consistent with color/matrices.py (6 d.p. vs 4 d.p. here).
    _M_709_TO_ACESCG = M_REC709_TO_ACESCG
    _M_ACESCG_TO_709 = M_ACESCG_TO_REC709

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "src_space": (cls._COLOR_SPACES, {"default": "Linear sRGB (D65)"}),
                "dst_space": (cls._COLOR_SPACES, {"default": "ACEScg"}),
                "direction": (["Forward", "Inverse"], {"default": "Forward"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "grade_info_in": ("STRING", {"forceInput": True}),
            },
        }

    @staticmethod
    def _lin_to_logc3(img: torch.Tensor) -> torch.Tensor:
        cut = 0.010591
        a, b = 5.555556, 0.052272
        c, d = 0.247190, 0.385537
        e, f = 5.367655, 0.092809
        lin = img * a + b
        return torch.where(lin >= cut, c * torch.log10(lin + a * b) + d, e * lin + f)

    @staticmethod
    def _logc3_to_lin(img: torch.Tensor) -> torch.Tensor:
        cut = 0.010591
        a, b = 5.555556, 0.052272
        c, d = 0.247190, 0.385537
        e, f = 5.367655, 0.092809
        log_cut = e * (cut * a + b) + f
        return torch.where(img >= log_cut, (torch.pow(10.0, (img - d) / c) - a * b) / a, (img - f) / e)

    @staticmethod
    def _lin_to_srgb(img: torch.Tensor) -> torch.Tensor:
        return torch.where(img <= 0.0031308, img * 12.92, 1.055 * img.clamp(min=0.0) ** (1.0 / 2.4) - 0.055)

    @staticmethod
    def _srgb_to_lin(img: torch.Tensor) -> torch.Tensor:
        return torch.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

    def _try_ocio(self, img: torch.Tensor, src: str, dst: str) -> torch.Tensor | None:
        mgr = get_ocio_manager()
        if not mgr.is_loaded:
            return None
        try:
            config = mgr.config

            def _resolve_name(n):
                try:
                    if config.getColorSpace(n):
                        return n
                except Exception:
                    pass
                _NAME_MAP = {
                    "Linear sRGB (D65)": "Linear", "ACEScg": "acescg",
                    "ACEScc": "acescc", "ACEScct": "acescct",
                    "sRGB (OETF encoded)": "sRGB", "Rec.709 (OETF encoded)": "Rec.709",
                    "Rec.709 / BT.1886": "Output - Rec.709",
                    "LogC3 (ARRI EI800)": "ARRI LogC3", "LogC4 (ARRI Alexa 35)": "ARRI LogC4",
                }
                return _NAME_MAP.get(n, n)

            ocio_src = _resolve_name(src)
            ocio_dst = _resolve_name(dst)
            processor = mgr.get_processor(ocio_src, ocio_dst)
            if not processor:
                return None
            cpu = processor.getDefaultCPUProcessor()
            import numpy as np
            arr = img.cpu().numpy().astype("float32")
            B, H, W, _ = arr.shape
            flat = arr.reshape(-1, 3)
            cpu.applyRGB(flat)
            return torch.from_numpy(flat.reshape(B, H, W, 3)).to(img.device)
        except Exception as e:
            logger.debug(f"[CSConvert] OCIO failed ({e}), using analytical fallback")
            return None

    def apply(self, image: torch.Tensor, src_space: str, dst_space: str,
              direction: str = "Forward", strength: float = 1.0, grade_info_in: str = None):
        if src_space == dst_space:
            return (image, json.dumps({"node": "RadianceColorSpaceConvert", "src": src_space,
                                       "dst": dst_space, "direction": direction, "noop": True,
                                       "upstream": grade_info_in}))

        effective_src = src_space if direction == "Forward" else dst_space
        effective_dst = dst_space if direction == "Forward" else src_space

        result = self._try_ocio(image, effective_src, effective_dst)

        if result is None:
            result = image.clone()

            def _encode(img, space):
                if space in ("sRGB (OETF encoded)", "Rec.709 (OETF encoded)"):
                    return self._lin_to_srgb(img)
                elif space == "Rec.709 / BT.1886":
                    return img.clamp(min=0.0) ** (1.0 / 2.4)
                elif space == "LogC3 (ARRI EI800)":
                    return self._lin_to_logc3(img)
                elif space == "ACEScg":
                    M = self._M_709_TO_ACESCG.to(img.device)
                    return torch.einsum('ij,...j->...i', M, img)
                return img

            def _decode(img, space):
                if space in ("sRGB (OETF encoded)", "Rec.709 (OETF encoded)"):
                    return self._srgb_to_lin(img)
                elif space == "Rec.709 / BT.1886":
                    return img.clamp(min=0.0) ** 2.4
                elif space == "LogC3 (ARRI EI800)":
                    return self._logc3_to_lin(img)
                elif space == "ACEScg":
                    M = self._M_ACESCG_TO_709.to(img.device)
                    return torch.einsum('ij,...j->...i', M, img)
                return img

            linear = _decode(result, effective_src)
            result = _encode(linear, effective_dst)

        if strength < 1.0:
            result = torch.lerp(image, result, strength)

        grade_info = json.dumps({
            "node": "RadianceColorSpaceConvert", "src": src_space, "dst": dst_space,
            "direction": direction, "strength": strength, "upstream": grade_info_in,
        })
        return (result, grade_info)


# ACEScg → XYZ (D60) imported from color.ops (authoritative source).
from radiance.color.ops import M_ACESCG_TO_XYZ_D60 as _M_ACESCG_TO_XYZ

# The remaining display-referred matrices are small enough to live here —
# they are specific to the ACES ODT logic below and not shared elsewhere.
_M_D60_TO_D65 = torch.tensor([
    [0.9872240, 0.0000000, 0.0128491],
    [0.0000000, 1.0000000, 0.0000000],
    [0.0000000, 0.0000000, 1.0000000],
], dtype=torch.float32)

_M_XYZ_TO_SRGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=torch.float32)

_M_XYZ_TO_P3 = torch.tensor([
    [ 2.4934969, -0.9313836, -0.4027108],
    [-0.8294890,  1.7626641,  0.0236247],
    [ 0.0358458, -0.0761724,  0.9568845],
], dtype=torch.float32)

_M_XYZ_TO_REC2020 = torch.tensor([
    [ 1.7166512, -0.3556708, -0.2533663],
    [-0.6666844,  1.6164812,  0.0157685],
    [ 0.0176399, -0.0427706,  0.9421031],
], dtype=torch.float32)


class RadianceACESTransform:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Apply ACES 1.x Input, Viewing, or Output Transform to an image."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "aces_info")

    _ODT_OPTIONS = ["sRGB D65", "DCI-P3 D65", "Rec.2020 PQ (HDR10)", "Rec.2020 HLG"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Scene-linear ACEScg image."}),
                "odt": (cls._ODT_OPTIONS, {"default": "sRGB D65"}),
                "exposure_offset": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.1}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 100.0, "max": 10000.0, "step": 100.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.02}),
            },
            "optional": {
                "grade_info_in": ("STRING", {"forceInput": True}),
            },
        }

    @staticmethod
    def _rrt(x: torch.Tensor) -> torch.Tensor:
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

    @staticmethod
    def _oetf_srgb(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=0.0) ** (1.0 / 2.4) - 0.055).clamp(0, 1)

    @staticmethod
    def _oetf_pq(linear: torch.Tensor, peak_nits: float) -> torch.Tensor:
        """Scene-linear (in nits) → PQ signal. Note: input is already in nits here."""
        # Import shared constants rather than redefining them.
        from radiance.color.ops import PQ_M1, PQ_M2, PQ_C1, PQ_C2, PQ_C3
        lum = linear.clamp(min=0.0)
        Ym = (lum * peak_nits / 10000.0).clamp(min=0.0) ** PQ_M1
        return ((PQ_C1 + PQ_C2 * Ym) / (1.0 + PQ_C3 * Ym)) ** PQ_M2

    @staticmethod
    def _oetf_hlg(linear: torch.Tensor) -> torch.Tensor:
        """Scene-linear → HLG signal. Delegates to the canonical implementation."""
        from radiance.color.ops import linear_to_hlg
        return linear_to_hlg(linear)

    def apply(self, image: torch.Tensor, odt: str = "sRGB D65", exposure_offset: float = 0.0,
              peak_nits: float = 1000.0, saturation: float = 1.0, grade_info_in: str = None) -> Tuple[torch.Tensor, str]:
        device = image.device
        img = image.float().clone()

        if exposure_offset != 0.0:
            img = img * (2.0 ** exposure_offset)

        img = self._rrt(img)

        if saturation != 1.0 and img.shape[-1] >= 3:
            luma = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).unsqueeze(-1)
            img = luma + saturation * (img - luma)
            img = img.clamp(0, 1)

        M_to_xyz = _M_ACESCG_TO_XYZ.to(device)
        M_d60_d65 = _M_D60_TO_D65.to(device)
        xyz_d65 = torch.einsum('ij,...j->...i', M_d60_d65 @ M_to_xyz, img)

        if odt == "sRGB D65":
            M_out = _M_XYZ_TO_SRGB.to(device)
            rgb = torch.einsum('ij,...j->...i', M_out, xyz_d65).clamp(0, 1)
            result = self._oetf_srgb(rgb)
        elif odt == "DCI-P3 D65":
            M_out = _M_XYZ_TO_P3.to(device)
            rgb = torch.einsum('ij,...j->...i', M_out, xyz_d65).clamp(0, 1)
            result = rgb ** (1.0 / 2.6)
        elif odt == "Rec.2020 PQ (HDR10)":
            M_out = _M_XYZ_TO_REC2020.to(device)
            rgb = torch.einsum('ij,...j->...i', M_out, xyz_d65).clamp(0, 1)
            result = self._oetf_pq(rgb, peak_nits)
        else:
            M_out = _M_XYZ_TO_REC2020.to(device)
            rgb = torch.einsum('ij,...j->...i', M_out, xyz_d65).clamp(0, 1)
            result = self._oetf_hlg(rgb)

        aces_info = json.dumps({
            "node": "RadianceACESTransform", "odt": odt,
            "exposure_offset_stops": exposure_offset, "peak_nits": peak_nits,
            "saturation": saturation, "upstream": grade_info_in,
        }, indent=2)
        return (result, aces_info)


class RadianceBitDepthDegrade:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Simulate lower bit-depth quantisation for look development and QC."
    FUNCTION = "degrade"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("quantized", "delta_amplified", "banding_mask", "metrics")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bit_depth": ("INT", {"default": 8, "min": 4, "max": 16, "step": 1, "display": "slider"}),
                "dither_mode": (["none", "triangular", "floyd-steinberg"], {"default": "triangular"}),
            },
            "optional": {
                "delta_gain": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "banding_threshold": ("FLOAT", {"default": 0.004, "min": 0.0005, "max": 0.05, "step": 0.0005}),
                "restore_from_quantized": ("BOOLEAN", {"default": False}),
            },
        }

    @staticmethod
    def _tpdf_dither(img: torch.Tensor, levels: int) -> torch.Tensor:
        lsb = 1.0 / levels
        noise = (torch.rand_like(img) + torch.rand_like(img) - 1.0) * lsb
        return img + noise

    @staticmethod
    def _floyd_steinberg(img: torch.Tensor, levels: int) -> torch.Tensor:
        import numpy as np
        step = 1.0 / (levels - 1)
        B, H, W, C = img.shape
        out = img.cpu().float().numpy().copy()
        for b in range(B):
            for c in range(C):
                p = out[b, :, :, c]
                for y in range(H):
                    for x in range(W):
                        old = p[y, x]
                        new = np.round(old / step) * step
                        err = old - new
                        p[y, x] = new
                        if x + 1 < W:
                            p[y, x + 1] += err * 7 / 16
                        if y + 1 < H:
                            if x > 0:
                                p[y + 1, x - 1] += err * 3 / 16
                            p[y + 1, x] += err * 5 / 16
                            if x + 1 < W:
                                p[y + 1, x + 1] += err * 1 / 16
                out[b, :, :, c] = p
        return torch.from_numpy(out).to(img.device)

    @staticmethod
    def _psnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
        mse = ((original - quantized) ** 2).mean().item()
        if mse == 0.0:
            return float("inf")
        return 10.0 * math.log10(1.0 / mse)

    def degrade(self, image: torch.Tensor, bit_depth: int = 8, dither_mode: str = "triangular",
                delta_gain: float = 10.0, banding_threshold: float = 0.004,
                restore_from_quantized: bool = False):
        img = image[..., :3].float()
        levels = (2 ** bit_depth) - 1
        step = 1.0 / levels

        if dither_mode == "triangular":
            dithered = self._tpdf_dither(img, levels)
        elif dither_mode == "floyd-steinberg":
            dithered = self._floyd_steinberg(img, levels)
        else:
            dithered = img.clone()

        quantized = (dithered / step).round() * step
        quantized = quantized.clamp(0.0, 1.0)

        delta = (img - quantized).abs()
        delta_amp = (delta * delta_gain).clamp(0.0, 1.0)
        banding = (delta.max(dim=-1, keepdim=True).values > banding_threshold).float().expand_as(img)

        psnr_val = self._psnr(img, quantized)
        max_err = delta.max().item()
        dr_loss_stops = math.log2(max(levels, 1) / (2 ** 16 - 1) + 1e-9)

        metrics = json.dumps({
            "bit_depth": bit_depth, "levels": levels, "dither_mode": dither_mode,
            "psnr_dB": round(psnr_val, 2), "max_error": round(max_err, 6),
            "dynamic_range_loss_stops": round(abs(dr_loss_stops), 2),
        }, indent=2)

        def _reattach(out3c):
            if image.shape[-1] == 4:
                return torch.cat([out3c, image[..., 3:4]], dim=-1)
            return out3c

        return (_reattach(quantized), _reattach(delta_amp), _reattach(banding), metrics)
