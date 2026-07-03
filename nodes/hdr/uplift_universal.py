"""
◎ Radiance SDR → HDR Universal
═══════════════════════════════

One-click SDR→HDR conversion for any image or video (frame batch).

Pipeline
--------
1. Inverse OETF          — decode the SDR transfer curve to scene-linear.
2. Adaptive soft-knee    — per-frame knee from a luma percentile (or manual),
   inverse tone mapping    EMA-smoothed across frames so video doesn't flicker.
3. Highlight expansion   — hue-preserving luminance expansion above the knee
                           up to peak_nits, hard shoulder controlled by gamma.
4. Output encoding       — scene-linear (EXR-ready), PQ / ST.2084 (HDR10),
                           or HLG (ARIB STD-B67) via the shared encoders in
                           radiance.nodes.hdr.aces2.

Works on IMAGE tensors of shape [B,H,W,C]; a video is simply a batch of
frames, so the same node handles stills, batches, and footage. B == 1
automatically behaves as a still image (temporal smoothing is a no-op).

No model checkpoints required — pure GPU math.
"""

from __future__ import annotations

import torch

from radiance.nodes.hdr.aces2 import _torch_pq_encode, _torch_hlg_encode

_EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
#  Transfer-function helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inverse_oetf(img: torch.Tensor, curve: str) -> torch.Tensor:
    """Decode an SDR-encoded image to scene-linear [0, 1]."""
    if curve == "sRGB":
        return torch.where(img <= 0.04045, img / 12.92,
                           ((img + 0.055) / 1.055).clamp(min=_EPS) ** 2.4)
    if curve == "Rec.709":
        return torch.where(img < 0.0812, img / 4.5,
                           ((img + 0.099) / 1.099).clamp(min=_EPS) ** (1.0 / 0.45))
    if curve == "Gamma 2.2":
        return img.clamp(min=0.0) ** 2.2
    if curve == "Gamma 2.4":
        return img.clamp(min=0.0) ** 2.4
    return img  # "None" — already linear


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luminance from linear RGB, shape [..., H, W]."""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


# ─────────────────────────────────────────────────────────────────────────────
#  Core expansion math (kept as free functions so they are unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_knees(luma: torch.Tensor, percentile: float,
                    smoothing: float) -> torch.Tensor:
    """
    Per-frame knee = `percentile` of frame luma, EMA-smoothed along the batch
    (time) axis with weight `smoothing` in [0, 1). smoothing=0 → per-frame.
    Returns a tensor of shape [B].
    """
    b = luma.shape[0]
    flat = luma.reshape(b, -1)
    q = torch.quantile(flat, percentile, dim=1).clamp(0.05, 0.99)
    if smoothing <= 0.0 or b == 1:
        return q
    knees = torch.empty_like(q)
    ema = q[0]
    for i in range(b):
        ema = smoothing * ema + (1.0 - smoothing) * q[i]
        knees[i] = ema
    return knees


def _soft_knee_expand(luma: torch.Tensor, knee: torch.Tensor,
                      peak_scale: float, shoulder_gamma: float) -> torch.Tensor:
    """
    Inverse tone map: identity below the knee, smooth power-curve expansion
    from the knee to `peak_scale` at L == 1. `knee` broadcasts per frame [B].
    Returns the expanded luminance (same shape as `luma`).
    """
    k = knee.view(-1, *([1] * (luma.dim() - 1)))
    t = ((luma - k) / (1.0 - k).clamp(min=_EPS)).clamp(0.0, 1.0)
    expanded = k + (peak_scale - k) * t.clamp(min=0.0) ** shoulder_gamma
    return torch.where(luma > k, expanded, luma)


# ─────────────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSDRToHDRUniversal:
    """
    ◎ Radiance SDR → HDR Universal

    Convert any SDR image or video (frame batch) to HDR in one node:
    inverse OETF → adaptive hue-preserving highlight expansion (temporally
    smoothed for video) → scene-linear, PQ (HDR10), or HLG output.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Universal one-click SDR→HDR for stills and video. Decodes the SDR "
        "transfer curve, expands highlights toward peak_nits with an adaptive "
        "flicker-free soft knee, and outputs scene-linear, PQ, or HLG."
    )
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "highlight_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR still, image batch, or video frames [B,H,W,C]."}),
                "inverse_oetf": (["sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.4", "None"], {
                    "default": "sRGB",
                    "tooltip": "Transfer curve the SDR source was encoded with. 'None' if input is already linear."}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 200.0, "max": 10000.0, "step": 50.0,
                    "tooltip": "Target HDR peak luminance. SDR reference white stays at 100 nits."}),
                "knee_mode": (["adaptive", "manual"], {"default": "adaptive",
                    "tooltip": "adaptive: knee follows a per-frame luma percentile (recommended). manual: fixed knee."}),
                "knee": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 0.99, "step": 0.01,
                    "tooltip": "Manual knee, or the percentile (as fraction) used in adaptive mode. Luma below the knee is preserved."}),
                "shoulder_gamma": ("FLOAT", {"default": 1.6, "min": 0.5, "max": 6.0, "step": 0.05,
                    "tooltip": ">1 confines the strongest expansion to the very brightest pixels (speculars); <1 lifts highlights broadly."}),
                "temporal_smoothing": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.98, "step": 0.01,
                    "tooltip": "EMA weight for the adaptive knee across frames. Prevents highlight flicker in video. Ignored for single frames."}),
                "output_encoding": (["Linear", "PQ (HDR10)", "HLG"], {"default": "Linear",
                    "tooltip": "Linear: scene-linear floats (EXR/delivery-ready, 1.0 = 100 nits). PQ: ST.2084 for HDR10. HLG: ARIB STD-B67."}),
            },
        }

    @torch.no_grad()
    def convert(self, image: torch.Tensor, inverse_oetf: str, peak_nits: float,
                knee_mode: str, knee: float, shoulder_gamma: float,
                temporal_smoothing: float, output_encoding: str):
        img = image.clone().float()
        if img.dim() == 3:                      # single HWC frame → batch of 1
            img = img.unsqueeze(0)

        rgb, extra = img[..., :3], img[..., 3:]
        peak_scale = max(peak_nits, 100.0) / 100.0

        # 1 ── decode to scene-linear
        lin = _inverse_oetf(rgb.clamp(0.0, 1.0), inverse_oetf)

        # 2 ── knee per frame (adaptive + temporally smoothed, or manual)
        luma = _luma(lin).clamp(0.0, 1.0)
        if knee_mode == "adaptive":
            knees = _adaptive_knees(luma, float(knee), float(temporal_smoothing))
        else:
            knees = torch.full((img.shape[0],), float(knee),
                               dtype=lin.dtype, device=lin.device)

        # 3 ── hue-preserving highlight expansion
        luma_exp = _soft_knee_expand(luma, knees, peak_scale, float(shoulder_gamma))
        gain = luma_exp / luma.clamp(min=_EPS)
        hdr = lin * gain.unsqueeze(-1)

        mask = ((luma_exp - luma) / max(peak_scale - 1.0, _EPS)).clamp(0.0, 1.0)

        # 4 ── output encoding
        if output_encoding.startswith("PQ"):
            out = _torch_pq_encode(hdr)   # convention: 1.0 linear = 100 nits
        elif output_encoding == "HLG":
            out = _torch_hlg_encode((hdr / peak_scale).clamp(0.0, 1.0))
        else:
            out = hdr

        if extra.shape[-1] > 0:                 # pass alpha / extra channels through
            out = torch.cat([out, extra], dim=-1)
        return (out, mask)


NODE_CLASS_MAPPINGS = {
    "RadianceSDRToHDRUniversal": RadianceSDRToHDRUniversal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSDRToHDRUniversal": "◎ Radiance SDR → HDR Universal",
}
