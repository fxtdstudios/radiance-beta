"""
◎ Radiance SDR → HDR Recover / Universal
═════════════════════════════════════════

One-click SDR→HDR conversion for any image or video (frame batch).

Pipeline
--------
1. Inverse OETF          — decode the SDR transfer curve to scene-linear.
2. Adaptive soft-knee    — per-frame knee from a luma percentile (or manual),
   inverse tone mapping    EMA-smoothed across frames so video doesn't flicker.
3. Highlight expansion   — hue-preserving luminance expansion above the knee
                           up to peak_nits, hard shoulder controlled by gamma.
4. Output transform      — scene-linear Rec.709, ACES2065-1/AP0 for EXR,
                           or Rec.2020 PQ / HLG delivery encoding.

Works on IMAGE tensors of shape [B,H,W,C]; a video is simply a batch of
frames, so the same node handles stills, batches, and footage. B == 1
automatically behaves as a still image (temporal smoothing is a no-op).

Pure GPU math by default — no checkpoints required. The dedicated Recover
node requires a VAE and trained RUDRA checkpoint. Universal exposes Expand,
Recover, and Hybrid modes; its learned modes reconstruct only clipped
highlights/crushed shadows and safely fall back to deterministic expansion.
Still RUDRA applies to single frames. Ordered video batches use a separate
Phase 3 motion-aligned residual checkpoint over 5/7/9 adjacent RGB frames;
missing or incompatible temporal weights fall back to deterministic expansion.
"""

from __future__ import annotations

import logging

import torch

from radiance.nodes.hdr.aces2 import _torch_pq_encode, _torch_hlg_encode
from radiance.color.ops import (
    M_BT2020_TO_REC709,
    M_REC709_TO_ACES2065_1,
    M_REC709_TO_BT2020,
    apply_matrix_3x3,
)

logger = logging.getLogger("radiance.nodes.hdr.uplift_universal")

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


def _soft_peak_limit(rgb: torch.Tensor, peak_scale: float,
                     knee_ratio: float = 0.9) -> torch.Tensor:
    """Hue-preserving luminance shoulder bounded by ``peak_scale``.

    Values below ``knee_ratio * peak_scale`` are unchanged. Brighter values
    approach the requested peak smoothly; a final numerical guard guarantees
    that learned reconstruction can never exceed the mastering target.
    """
    peak = max(float(peak_scale), _EPS)
    knee = peak * float(knee_ratio)
    y = _luma(rgb).clamp(min=0.0)
    span = max(peak - knee, _EPS)
    compressed = knee + span * torch.tanh((y - knee).clamp(min=0.0) / span)
    target_y = torch.where(y > knee, compressed, y).clamp(max=peak)
    gain = target_y / y.clamp(min=_EPS)
    limited = rgb.clamp(min=0.0) * gain.unsqueeze(-1)

    # Floating-point roundoff can leave luma a few ulps above the target.
    limited_y = _luma(limited).clamp(min=_EPS)
    guard = torch.clamp(peak / limited_y, max=1.0)
    return limited * guard.unsqueeze(-1)


def _shadow_mask(luma: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft mask for crushed-shadow recovery/QC, 1 at black and 0 at threshold."""
    t = max(float(threshold), _EPS)
    return ((t - luma) / t).clamp(0.0, 1.0)


def _clipped_highlight_mask(sdr_rgb: torch.Tensor,
                            threshold: float) -> torch.Tensor:
    """Mask display-referred pixels that contain little recoverable headroom.

    Channel clipping is included alongside luma so saturated red, green, or
    blue highlights are not missed. The smoothstep ramp is exactly zero below
    ``threshold`` and reaches one at SDR code value 1.0.
    """
    t = min(max(float(threshold), 0.0), 1.0 - _EPS)
    evidence = torch.maximum(_luma(sdr_rgb), sdr_rgb.max(dim=-1).values)
    x = ((evidence - t) / max(1.0 - t, _EPS)).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _encode_output(hdr: torch.Tensor, output_encoding: str,
                   peak_scale: float) -> torch.Tensor:
    """Apply the Phase 1 standards-correct mastering output transform."""
    if output_encoding.startswith("PQ"):
        delivery = apply_matrix_3x3(hdr, M_REC709_TO_BT2020).clamp(min=0.0)
        return _torch_pq_encode(delivery)
    if output_encoding == "HLG":
        delivery = apply_matrix_3x3(hdr, M_REC709_TO_BT2020).clamp(min=0.0)
        return _torch_hlg_encode((delivery / peak_scale).clamp(0.0, 1.0))
    if output_encoding == "Linear ACES2065-1 (AP0)":
        return apply_matrix_3x3(hdr, M_REC709_TO_ACES2065_1)
    return hdr


# ─────────────────────────────────────────────────────────────────────────────
#  Shared learned-recovery core
# ─────────────────────────────────────────────────────────────────────────────

class _RudraRecoveryCore:
    """Shared RUDRA execution used by Recover and Universal."""

    @staticmethod
    def _rudra_reconstruct(sdr_pixels: torch.Tensor, base_hdr: torch.Tensor,
                           mask: torch.Tensor, vae, rudra_size: str,
                           blend: float, peak_scale: float,
                           model_meta: str = "") -> torch.Tensor:
        """
        SDR pixels → VAE encode → RUDRA decode → scene-linear reconstruction,
        gain-matched to `base_hdr` in well-exposed regions and blended into the
        supplied recovery mask. Raises on any failure so the owning node can
        apply its documented error/fallback policy.
        """
        if base_hdr.shape[0] > 1:
            raise RuntimeError(
                "RUDRA reconstruction supports single frames only — video VAEs "
                "compress time (frame counts cannot be matched 1:1) and current "
                "video checkpoints were trained on stills."
            )
        import torch.nn.functional as F
        from radiance.fast_vae import (
            decode_to_linear_realtime,
            load_radiance_decoder_weights,
            resolve_rudra_model_type,
        )

        latent = vae.encode(sdr_pixels[..., :3])
        if isinstance(latent, dict):                       # some wrappers
            latent = latent.get("samples", next(iter(latent.values())))

        model_type = resolve_rudra_model_type(latent.shape[1], latent.ndim == 5, vae, model_meta=model_meta)
        decoder = load_radiance_decoder_weights(model_type=model_type,
                                                model_size=rudra_size)
        if decoder is None:
            raise RuntimeError(f"no RUDRA checkpoint for model_type={model_type!r}")
        decoder = decoder.to(latent.device)

        scale_factor = getattr(vae, "scale_factor", None)
        if not isinstance(scale_factor, (int, float)) or scale_factor == 0:
            scale_factor = None                            # resolve from config
        rec = decode_to_linear_realtime(
            latent=latent, decoder=decoder, model_type=model_type,
            scale_factor=scale_factor, precision="bf16",
        ).float().to(base_hdr.device)
        rec = torch.nan_to_num(
            rec, nan=0.0, posinf=float(peak_scale), neginf=0.0,
        ).clamp(min=0.0)

        # VAE rounding can shift resolution — match the base exactly.
        if rec.shape[1:3] != base_hdr.shape[1:3]:
            rec = F.interpolate(rec.permute(0, 3, 1, 2), size=base_hdr.shape[1:3],
                                mode="bilinear", align_corners=False
                                ).permute(0, 2, 3, 1)
        if rec.shape[0] != base_hdr.shape[0]:
            raise RuntimeError(f"RUDRA frame count {rec.shape[0]} != input {base_hdr.shape[0]}")

        # Gain-match in well-exposed, non-highlight regions so the two paths
        # agree on exposure before blending.
        base_l, rec_l = _luma(base_hdr), _luma(rec)
        ref = (mask < 0.05) & (base_l > 0.05) & (base_l < 0.8)
        if ref.any():
            gain = (base_l[ref].median() / rec_l[ref].median().clamp(min=_EPS)).clamp(0.1, 10.0)
            rec = rec * gain

        # Learned inverse-log reconstruction can contain extreme radiance
        # values. Constrain it before blending so peak_nits remains a real
        # mastering limit, not merely a hint used by the deterministic path.
        rec = _soft_peak_limit(rec, peak_scale)
        w = (mask * float(blend)).unsqueeze(-1)
        return _soft_peak_limit(base_hdr * (1.0 - w) + rec * w, peak_scale)

    @staticmethod
    def _temporal_reconstruct(
        source: torch.Tensor,
        deterministic: torch.Tensor,
        highlight_mask: torch.Tensor,
        shadow_mask: torch.Tensor,
        peak_scale: float,
        window_size: int,
        checkpoint_path: str,
        preserve_source_outside: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from radiance.temporal_rudra import (
            load_temporal_rudra_weights,
            recover_temporal_residual,
        )

        if source.shape[0] < 5:
            raise RuntimeError("temporal RUDRA requires at least five ordered frames")
        model = load_temporal_rudra_weights(checkpoint_path, source.device)
        if model is None:
            raise RuntimeError(
                "no Phase 3 temporal checkpoint; set RADIANCE_TEMPORAL_RUDRA "
                "or place temporal_rudra_residual_ema.safetensors in models/radiance"
            )
        out, h_conf, s_conf = recover_temporal_residual(
            source, deterministic, highlight_mask, shadow_mask, model,
            peak_scale, window_size, preserve_source_outside,
        )
        return _soft_peak_limit(out, peak_scale), h_conf, s_conf

    @staticmethod
    def _pixel_reconstruct(
        sdr_linear: torch.Tensor,
        base_hdr: torch.Tensor,
        mask: torch.Tensor,
        checkpoint_path: str,
        blend: float,
        peak_scale: float,
        tile_size: int,
        tile_overlap: int,
        recovery_mode: str,
        strength: float,
    ) -> torch.Tensor:
        """Run the direct-pixel SDR2HDRNet and blend it into evidence masks.

        The network consumes canonical sRGB code values and returns normalized
        scene-linear Rec.2020 with 1.0 == 10,000 nits. Radiance's internal
        working space is scene-linear Rec.709 with 1.0 == 100 nits.
        """
        from radiance.pixel_sdr2hdr import linear_to_srgb, predict_pixel_sdr2hdr

        canonical_srgb = linear_to_srgb(sdr_linear.clamp(0.0, 1.0))
        recovered_2020 = predict_pixel_sdr2hdr(
            canonical_srgb, checkpoint_path=checkpoint_path,
            tile_size=int(tile_size), tile_overlap=int(tile_overlap),
            recovery_mode=str(recovery_mode), strength=float(strength),
        )
        recovered_709 = apply_matrix_3x3(
            recovered_2020 * 100.0, M_BT2020_TO_REC709,
        ).clamp(min=0.0)
        recovered_709 = _soft_peak_limit(recovered_709, peak_scale)
        weight = (mask * float(blend)).unsqueeze(-1)
        return _soft_peak_limit(
            base_hdr * (1.0 - weight) + recovered_709 * weight,
            peak_scale,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Dedicated learned recovery product
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSDRToHDRRecover(_RudraRecoveryCore):
    """Recover clipped highlights and crushed shadows with a RUDRA decoder.

    Unlike deterministic expansion, this node is explicitly reconstructive.
    Still recovery requires a VAE/RUDRA checkpoint. Video recovery uses the
    Phase 3 temporal residual checkpoint without a VAE. Both change pixels
    only inside the returned recovery masks.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Learned SDR→HDR reconstruction for clipped highlights and crushed "
        "shadows. Uses still RUDRA for images or motion-aligned temporal RUDRA "
        "for video; pixels outside the recovery masks are preserved exactly."
    )
    FUNCTION = "recover"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "highlight_mask", "shadow_mask",
                    "highlight_confidence", "shadow_confidence")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR image or ordered 5+ frame video batch."}),
                "inverse_oetf": (["sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.4", "None"], {"default": "sRGB"}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 200.0, "max": 10000.0, "step": 50.0}),
                "highlight_threshold": ("FLOAT", {"default": 0.98, "min": 0.8, "max": 0.999, "step": 0.001,
                    "tooltip": "SDR code-value threshold used to identify clipped luma or RGB channels."}),
                "shadow_threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.005,
                    "tooltip": "Linear-luma threshold used to identify crushed shadows."}),
                "highlight_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_encoding": (["Linear", "Linear ACES2065-1 (AP0)", "PQ (HDR10)", "HLG"], {"default": "Linear"}),
                "rudra_size": (["rudra_turbo", "rudra_full"], {"default": "rudra_turbo"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Required only for single-frame RUDRA recovery."}),
                "model_meta": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "RadianceUnifiedLoader model_meta for exact model-family resolution."}),
                "temporal_window": ([5, 7, 9], {"default": 5,
                    "tooltip": "Adjacent frames used by the Phase 3 temporal residual model."}),
                "temporal_checkpoint": ("STRING", {"default": "",
                    "tooltip": "Optional Phase 3 checkpoint path. Empty uses RADIANCE_TEMPORAL_RUDRA or models/radiance."}),
            },
        }

    def recover(self, image: torch.Tensor, inverse_oetf: str,
                peak_nits: float, highlight_threshold: float,
                shadow_threshold: float, highlight_strength: float,
                shadow_strength: float, output_encoding: str,
                rudra_size: str, vae=None, model_meta: str = "",
                temporal_window: int = 5, temporal_checkpoint: str = ""):
        img = torch.nan_to_num(image.clone().float(), nan=0.0, posinf=1.0, neginf=0.0)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        rgb, extra = img[..., :3], img[..., 3:]
        peak_scale = max(float(peak_nits), 100.0) / 100.0
        lin = _inverse_oetf(rgb.clamp(0.0, 1.0), inverse_oetf)
        highlights = _clipped_highlight_mask(rgb.clamp(0.0, 1.0), highlight_threshold)
        shadows = _shadow_mask(_luma(lin).clamp(0.0, 1.0), shadow_threshold)
        recovery_mask = torch.maximum(
            highlights * float(highlight_strength),
            shadows * float(shadow_strength),
        )
        try:
            if img.shape[0] > 1:
                knees = _adaptive_knees(_luma(lin).clamp(0.0, 1.0), 0.75, 0.85)
                expanded_luma = _soft_knee_expand(
                    _luma(lin).clamp(0.0, 1.0), knees, peak_scale, 1.6,
                )
                deterministic = lin * (
                    expanded_luma / _luma(lin).clamp(min=_EPS)
                ).unsqueeze(-1)
                hdr, h_conf, s_conf = self._temporal_reconstruct(
                    lin, deterministic,
                    highlights * float(highlight_strength),
                    shadows * float(shadow_strength),
                    peak_scale, int(temporal_window), str(temporal_checkpoint), True,
                )
            else:
                if vae is None:
                    raise RuntimeError("single-frame recovery requires a VAE")
                hdr = self._rudra_reconstruct(
                    rgb, lin, recovery_mask, vae, str(rudra_size), 1.0,
                    peak_scale, model_meta=model_meta,
                )
                h_conf = highlights * float(highlight_strength)
                s_conf = shadows * float(shadow_strength)
        except Exception as exc:  # noqa: BLE001 — convert to an actionable node error
            raise RuntimeError(
                "SDR → HDR Recover needs a compatible trained RUDRA checkpoint; "
                f"recovery could not run: {exc}"
            ) from exc

        out = _encode_output(hdr, output_encoding, peak_scale)
        if extra.shape[-1] > 0:
            out = torch.cat([out, extra], dim=-1)
        return (out, highlights, shadows, h_conf, s_conf)


# ─────────────────────────────────────────────────────────────────────────────
#  Universal orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSDRToHDRUniversal(_RudraRecoveryCore):
    """Orchestrate deterministic expansion, learned recovery, or a hybrid."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "SDR→HDR orchestrator: deterministic Expand, learned Recover, or "
        "Hybrid. Includes safe fallback, professional output transforms, and "
        "separate highlight/shadow masks."
    )
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "highlight_mask", "shadow_mask",
                    "highlight_confidence", "shadow_confidence")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR still, image batch, or video frames [B,H,W,C]."}),
                "inverse_oetf": (["sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.4", "None"], {"default": "sRGB"}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 200.0, "max": 10000.0, "step": 50.0}),
                "knee_mode": (["adaptive", "manual"], {"default": "adaptive"}),
                "knee": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 0.99, "step": 0.01}),
                "shoulder_gamma": ("FLOAT", {"default": 1.6, "min": 0.5, "max": 6.0, "step": 0.05}),
                "temporal_smoothing": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.98, "step": 0.01}),
                "output_encoding": (["Linear", "Linear ACES2065-1 (AP0)", "PQ (HDR10)", "HLG"], {"default": "Linear"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Optional RUDRA recovery VAE. Recover/Hybrid falls back to Expand when unavailable."}),
                "rudra_size": (["rudra_turbo", "rudra_full"], {"default": "rudra_turbo"}),
                "rudra_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "model_meta": ("STRING", {"default": "", "forceInput": True}),
                "batch_mode": (["Independent Images", "Video Frames"], {"default": "Independent Images"}),
                "shadow_threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.005}),
                "processing_mode": (["Expand", "Recover", "Hybrid"], {"default": "Hybrid",
                    "tooltip": "Expand: deterministic only. Recover: learned reconstruction only. Hybrid: expansion plus masked learned recovery."}),
                "highlight_threshold": ("FLOAT", {"default": 0.98, "min": 0.8, "max": 0.999, "step": 0.001,
                    "tooltip": "Clipping threshold for Recover/Hybrid. Appended for saved-workflow compatibility."}),
                "temporal_window": ([5, 7, 9], {"default": 5,
                    "tooltip": "Adjacent video frames used by temporal RUDRA."}),
                "temporal_checkpoint": ("STRING", {"default": "",
                    "tooltip": "Optional Phase 3 checkpoint path. Empty uses the configured default."}),
                "learned_backend": (["Auto", "Direct Pixel", "Legacy RUDRA"], {"default": "Auto",
                    "tooltip": "Auto prefers temporal video recovery, then the direct-pixel model, then legacy VAE RUDRA."}),
                "pixel_checkpoint": ("STRING", {"default": "",
                    "tooltip": "Direct-pixel .pt checkpoint. Empty searches models/radiance and RADIANCE_SDR2HDR_PIXEL."}),
                "pixel_tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "pixel_tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16}),
                "pixel_recovery_mode": (["highlights", "all", "shadows", "off"], {"default": "highlights",
                    "tooltip": "Highlights is safest and avoids hallucinating chroma in deep shadows."}),
                "pixel_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    @torch.no_grad()
    def convert(self, image: torch.Tensor, inverse_oetf: str, peak_nits: float,
                knee_mode: str, knee: float, shoulder_gamma: float,
                temporal_smoothing: float, output_encoding: str,
                vae=None, rudra_size: str = "rudra_turbo", rudra_blend: float = 1.0,
                model_meta: str = "", batch_mode: str = "Independent Images",
                shadow_threshold: float = 0.05,
                processing_mode: str = "Hybrid",
                highlight_threshold: float = 0.98,
                temporal_window: int = 5,
                temporal_checkpoint: str = "",
                learned_backend: str = "Auto",
                pixel_checkpoint: str = "",
                pixel_tile_size: int = 512,
                pixel_tile_overlap: int = 64,
                pixel_recovery_mode: str = "highlights",
                pixel_strength: float = 1.0):
        img = torch.nan_to_num(
            image.clone().float(), nan=0.0, posinf=1.0, neginf=0.0,
        )
        if img.dim() == 3:                      # single HWC frame → batch of 1
            img = img.unsqueeze(0)

        rgb, extra = img[..., :3], img[..., 3:]
        peak_scale = max(peak_nits, 100.0) / 100.0

        # 1 ── decode to scene-linear
        lin = _inverse_oetf(rgb.clamp(0.0, 1.0), inverse_oetf)

        # 2 ── knee per frame (adaptive + temporally smoothed, or manual)
        luma = _luma(lin).clamp(0.0, 1.0)
        if knee_mode == "adaptive":
            smoothing = float(temporal_smoothing) if batch_mode == "Video Frames" else 0.0
            knees = _adaptive_knees(luma, float(knee), smoothing)
        else:
            knees = torch.full((img.shape[0],), float(knee),
                               dtype=lin.dtype, device=lin.device)

        # 3 ── hue-preserving highlight expansion
        luma_exp = _soft_knee_expand(luma, knees, peak_scale, float(shoulder_gamma))
        gain = luma_exp / luma.clamp(min=_EPS)
        expanded_hdr = lin * gain.unsqueeze(-1)

        mask = ((luma_exp - luma) / max(peak_scale - 1.0, _EPS)).clamp(0.0, 1.0)
        shadows = _shadow_mask(luma, float(shadow_threshold))
        clipped = _clipped_highlight_mask(rgb.clamp(0.0, 1.0), highlight_threshold)

        # Expand is deterministic and never invokes a learned model. Recover
        # starts from decoded SDR and changes only clipped highlights/crushed
        # shadows. Hybrid starts from the deterministic expansion and blends
        # learned reconstruction only inside those evidence masks.
        mode = processing_mode if processing_mode in {"Expand", "Recover", "Hybrid"} else "Hybrid"
        hdr = lin if mode == "Recover" else expanded_hdr
        recovery_applied = False
        h_conf = torch.zeros_like(clipped)
        s_conf = torch.zeros_like(shadows)

        # 3b ── direct-pixel, legacy VAE RUDRA, or temporal reconstruction.
        wants_recovery = mode in {"Recover", "Hybrid"}
        if wants_recovery and rudra_blend > 0.0:
            backend = learned_backend if learned_backend in {
                "Auto", "Direct Pixel", "Legacy RUDRA",
            } else "Auto"

            # A trained temporal model remains the preferred video backend.
            if img.shape[0] > 1 and batch_mode == "Video Frames" and backend != "Direct Pixel":
                try:
                    hdr, h_conf, s_conf = self._temporal_reconstruct(
                        lin, expanded_hdr,
                        clipped * float(rudra_blend),
                        shadows * float(rudra_blend),
                        peak_scale, int(temporal_window), str(temporal_checkpoint),
                        mode == "Recover",
                    )
                    recovery_applied = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Temporal RUDRA unavailable (%s).", exc)

            # The direct-pixel checkpoint supports stills and frame batches.
            # Until the temporal direct model is trained, video frames are
            # independent and may need downstream deflicker/temporal QC.
            use_pixel = backend == "Direct Pixel" or (
                backend == "Auto" and bool(str(pixel_checkpoint).strip())
            )
            if not recovery_applied and use_pixel:
                try:
                    recovery_mask = torch.maximum(clipped, shadows)
                    if pixel_recovery_mode == "highlights":
                        recovery_mask = clipped
                    elif pixel_recovery_mode == "shadows":
                        recovery_mask = shadows
                    elif pixel_recovery_mode == "off":
                        recovery_mask = torch.zeros_like(clipped)
                    hdr = self._pixel_reconstruct(
                        lin, hdr, recovery_mask, str(pixel_checkpoint),
                        float(rudra_blend), peak_scale, int(pixel_tile_size),
                        int(pixel_tile_overlap), str(pixel_recovery_mode),
                        float(pixel_strength),
                    )
                    h_conf = clipped * float(rudra_blend) if pixel_recovery_mode in {"highlights", "all"} else torch.zeros_like(clipped)
                    s_conf = shadows * float(rudra_blend) if pixel_recovery_mode in {"shadows", "all"} else torch.zeros_like(shadows)
                    recovery_applied = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Direct-pixel recovery unavailable (%s).", exc)

            if (not recovery_applied and backend in {"Auto", "Legacy RUDRA"}
                    and img.shape[0] == 1 and vae is not None):
                try:
                    recovery_mask = torch.maximum(clipped, shadows)
                    hdr = self._rudra_reconstruct(
                        rgb, hdr, recovery_mask, vae, str(rudra_size),
                        float(rudra_blend), peak_scale, model_meta=model_meta,
                    )
                    h_conf = clipped * float(rudra_blend)
                    s_conf = shadows * float(rudra_blend)
                    recovery_applied = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Legacy RUDRA recovery unavailable (%s).", exc)

        # Universal always produces usable HDR. Recover mode therefore falls
        # back to deterministic expansion if the learned path cannot run.
        if mode == "Recover" and not recovery_applied:
            hdr = expanded_hdr

        # 4 ── output encoding
        out = _encode_output(hdr, output_encoding, peak_scale)

        if extra.shape[-1] > 0:                 # pass alpha / extra channels through
            out = torch.cat([out, extra], dim=-1)
        return (out, mask, shadows, h_conf, s_conf)


NODE_CLASS_MAPPINGS = {
    "RadianceSDRToHDRRecover": RadianceSDRToHDRRecover,
    "RadianceSDRToHDRUniversal": RadianceSDRToHDRUniversal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSDRToHDRRecover": "◎ Radiance SDR → HDR Recover",
    "RadianceSDRToHDRUniversal": "◎ Radiance SDR → HDR Universal",
}
