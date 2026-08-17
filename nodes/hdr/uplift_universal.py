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

#: torch.quantile refuses inputs above 2**24 elements along the reduced axis.
#: A single 8K frame is 33.2M pixels, so the default adaptive knee mode raised
#: "quantile() input tensor is too large" on anything past roughly 4K -- on the
#: node whose entire purpose is film-resolution HDR work.
_QUANTILE_MAX_ELEMS = 2 ** 24


def _row_quantile(flat: torch.Tensor, percentile: float) -> torch.Tensor:
    """
    Per-row quantile that works at any resolution.

    Falls back to ``torch.kthvalue`` (which has no size cap) for rows too large
    for ``torch.quantile``. kthvalue returns the exact k-th order statistic
    rather than interpolating between neighbours; on image-sized inputs the two
    agree to ~1e-5, which is far below the precision a tone-mapping knee needs.
    """
    n = flat.shape[1]
    if n == 0:
        return flat.new_zeros((flat.shape[0],))
    if n <= _QUANTILE_MAX_ELEMS:
        return torch.quantile(flat, percentile, dim=1)
    k = max(1, min(n, int(round(float(percentile) * (n - 1))) + 1))
    return torch.kthvalue(flat, k, dim=1).values


def _adaptive_knees(luma: torch.Tensor, percentile: float,
                    smoothing: float) -> torch.Tensor:
    """
    Per-frame knee = `percentile` of frame luma, EMA-smoothed along the batch
    (time) axis with weight `smoothing` in [0, 1). smoothing=0 → per-frame.
    Returns a tensor of shape [B].
    """
    b = luma.shape[0]
    if b == 0:                                   # empty batch — nothing to do
        return luma.new_zeros((0,))
    flat = luma.reshape(b, -1)
    q = _row_quantile(flat, percentile).clamp(0.05, 0.99)
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
    Inverse tone map: identity below the knee, expansion from the knee to
    `peak_scale` at L == 1, with the gradient continuous across the join.
    `knee` broadcasts per frame [B]. Returns the expanded luminance.

    The join used to be a bare power curve, ``k + (peak - k) * t**gamma``.
    That is continuous in value and discontinuous in slope: below the knee the
    curve is the identity, gradient 1; immediately above it the gradient is
    ``(peak - k) * gamma * t**(gamma - 1) / (1 - k)``, which for any gamma > 1
    goes to *zero* as t → 0. Measured at the shipped defaults (knee 0.75,
    peak_scale 10.0):

        shoulder_gamma   gradient below   gradient above
              1.0             1.000            37.05
              1.6             1.000             0.083
              2.5             1.000             0.000

    A gradient that drops from 1.0 to 0.08 across a single code value is a
    visible ridge in any smooth gradient crossing the knee — a sky, a skin
    falloff, the soft edge of a practical — and in adaptive mode the knee moves
    per frame, so on video the ridge crawls. BT.2446 Method B specifies the
    opposite: "the gradient of the exponential function is set to unity at the
    breakpoint", and suggests a Bezier blend "to avoid artefacts at the join".
    This module's own `_soft_peak_limit` has always been C¹ at its knee; only
    the expansion was not.

    The shoulder is now

        f(t) = s·t + (1 − s)·t^gamma,   s = (1 − k) / (peak_scale − k)

    which satisfies f(0) = 0, f(1) = 1 and f'(0) = s — and s is exactly the
    value that makes the gradient in luminance space equal 1 at the knee. It
    is monotonic for gamma ≥ 1 and keeps `shoulder_gamma`'s meaning: how hard
    the expansion ramps once past the knee.

    gamma = 1 collapses to a straight line from (k, k) to (1, peak_scale),
    which has an unavoidable gradient step — a linear ramp to peak is what
    that asks for. Values at or above 2 land within 0.2% of unity.
    """
    k = knee.view(-1, *([1] * (luma.dim() - 1)))
    span = (1.0 - k).clamp(min=_EPS)
    t = ((luma - k) / span).clamp(0.0, 1.0)

    # The gradient the shoulder must start with for the join to be C¹.
    s = (span / (peak_scale - k).clamp(min=_EPS)).clamp(0.0, 1.0)

    g = max(float(shoulder_gamma), 1.0)
    shaped = s * t + (1.0 - s) * t.clamp(min=0.0) ** g

    expanded = k + (peak_scale - k) * shaped
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


def _pixel_checkpoint_available(checkpoint_path: str) -> bool:
    """
    Whether a direct-pixel checkpoint can actually be resolved.

    `resolve_pixel_checkpoint` searches, in order: the explicit path, the
    RADIANCE_SDR2HDR_PIXEL environment variable, then models/radiance. The
    Auto backend used to gate on `bool(pixel_checkpoint.strip())` instead of
    asking the resolver, so a user with an installed checkpoint but an empty
    path widget -- the stock defaults -- never got the direct-pixel model at
    all. It silently fell through to legacy VAE RUDRA or plain expansion,
    contradicting both the `learned_backend` tooltip ("Auto prefers ... the
    direct-pixel model ...") and the `pixel_checkpoint` tooltip ("Empty
    searches models/radiance and RADIANCE_SDR2HDR_PIXEL").
    """
    try:
        from radiance.pixel_sdr2hdr import resolve_pixel_checkpoint
    except Exception as exc:  # noqa: BLE001 — optional dependency chain
        logger.debug("Direct-pixel backend unavailable: %s", exc)
        return False
    try:
        return resolve_pixel_checkpoint(str(checkpoint_path)) is not None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Direct-pixel checkpoint lookup failed: %s", exc)
        return False


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
        # Limit the LEARNED signal only, then blend -- do not re-limit the
        # result. Two reasons:
        #   1. _soft_peak_limit is a tanh compressor, so it is not idempotent.
        #      Applying it to `rec` and again to the blend compounded the
        #      compression and pulled highlights below the peak they should sit
        #      at.
        #   2. Re-limiting the blend altered pixels where the mask is zero,
        #      contradicting this node's documented contract that pixels
        #      outside the recovery masks are preserved exactly.
        # The peak guarantee still holds: luma is linear in RGB and the blend is
        # convex, so a blend of two signals that each respect `peak_scale` also
        # respects it. Callers must pass a `base_hdr` that already does -- every
        # in-tree caller does (deterministic expansion tops out at peak_scale,
        # and decoded SDR tops out at 1.0).
        return base_hdr * (1.0 - w) + rec * w

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
        # Limit the learned signal only — see the note in _rudra_reconstruct.
        # Re-limiting the blend double-compressed highlights and modified
        # pixels the mask had excluded.
        return base_hdr * (1.0 - weight) + recovered_709 * weight


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

    # Runs VAE encode + a temporal model. Its sibling convert() has carried this guard
    # all along; without it the whole encoder activation stack is retained as an
    # autograd graph -- several GB of VRAM on a 4K frame.
    @torch.no_grad()
    def recover(self, image: torch.Tensor, inverse_oetf: str,
                peak_nits: float, highlight_threshold: float,
                shadow_threshold: float, highlight_strength: float,
                shadow_strength: float, output_encoding: str,
                rudra_size: str, vae=None, model_meta: str = "",
                temporal_window: int = 5, temporal_checkpoint: str = ""):
        img = torch.nan_to_num(image.clone().float(), nan=0.0, posinf=1.0, neginf=0.0)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.shape[0] == 0:                   # empty batch → empty result
            zeros = img[..., 0]
            return (img, zeros, zeros, zeros, zeros)
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
        "Hybrid. Recover and Hybrid need a learned checkpoint or a RUDRA VAE — "
        "without one they fall back to Expand, and the report output says so. "
        "Professional output transforms and separate highlight/shadow masks."
    )
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "highlight_mask", "shadow_mask",
                    "highlight_confidence", "shadow_confidence", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR still, image batch, or video frames [B,H,W,C]."}),
                "inverse_oetf": (["sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.4", "None"], {"default": "sRGB"}),
                "peak_nits": ("FLOAT", {
                    "default": 1000.0, "min": 200.0, "max": 10000.0, "step": 50.0,
                    "tooltip": (
                        "Mastering display peak. This is the ceiling the output "
                        "is limited to and the value the PQ/HLG encode targets — "
                        "not where SDR white lands. See reference_white_nits."
                    ),
                }),
                "reference_white_nits": ("FLOAT", {
                    "default": 203.0, "min": 100.0, "max": 1000.0, "step": 1.0,
                    "tooltip": (
                        "Where SDR diffuse white (code 1.0) lands, in nits. "
                        "203 is the ITU-R BT.2408 HDR Reference White — a white "
                        "shirt, a page, a cloud. Headroom above it belongs to "
                        "speculars. Raise it for a brighter grade; setting it to "
                        "peak_nits restores the pre-3.4 behaviour of slamming "
                        "SDR white to the display peak."
                    ),
                }),
                "knee_mode": (["adaptive", "manual"], {"default": "adaptive"}),
                "knee": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 0.99, "step": 0.01}),
                "shoulder_gamma": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 6.0, "step": 0.05,
                    "tooltip": (
                        "How hard highlights ramp once past the knee. The join "
                        "keeps a continuous gradient at any value ≥ 1; 2.0 and "
                        "above sit within 0.2% of unity there. 1.0 is a "
                        "straight line to peak and steps the gradient."
                    ),
                }),
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
                # Appended rather than slotted in beside peak_nits, for the same
                # reason highlight_threshold below is: ComfyUI passes required
                # inputs by keyword, so widget order is free, but a positional
                # caller — every existing test, and any script driving the node
                # directly — binds by position.
                reference_white_nits: float = 203.0,
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
        if img.shape[0] == 0:                   # empty batch → empty result
            zeros = img[..., 0]
            return (img, zeros, zeros, zeros, zeros, "mode: empty batch\nframes: 0")

        rgb, extra = img[..., :3], img[..., 3:]
        peak_scale = max(peak_nits, 100.0) / 100.0

        # Where SDR diffuse white lands, as distinct from the display ceiling.
        #
        # The expansion used to target `peak_scale` at SDR code 1.0, so a white
        # shirt or a page of text came out at the full mastering peak: measured
        # 1000.00 nits at the shipped defaults, against the 203 nits ITU-R
        # BT.2408 defines as HDR Reference White. Nearly five times reference,
        # on the value the eye uses to judge exposure for the whole image.
        #
        # BT.2446 Method B, which is this node's method done to the standard,
        # scales SDR by ~2 to reach 203 and then expands highlights above the
        # breakpoint by 2.3x in display light. Separating the two numbers is
        # what makes that possible: `peak_nits` is the ceiling and the encode
        # target, `reference_white_nits` is where SDR white sits, and the range
        # between them is headroom for speculars the learned path recovers.
        white_scale = max(float(reference_white_nits), 100.0) / 100.0
        white_scale = min(white_scale, peak_scale)

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

        # 3 ── hue-preserving highlight expansion, up to reference white
        luma_exp = _soft_knee_expand(luma, knees, white_scale, float(shoulder_gamma))
        gain = luma_exp / luma.clamp(min=_EPS)
        expanded_hdr = lin * gain.unsqueeze(-1)

        mask = ((luma_exp - luma) / max(white_scale - 1.0, _EPS)).clamp(0.0, 1.0)
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

        # What actually ran, reported rather than left to the console.
        #
        # Every learned backend below is wrapped in try/except and logs a
        # warning on the way past, so a user with no checkpoint installed gets
        # deterministic Expand output from Hybrid or Recover — bit-identical,
        # with five normal-looking outputs and nothing on the graph to say the
        # node's headline feature never engaged. Measured on a clean install:
        # Hybrid and Recover both return exactly the Expand result.
        #
        # `report` names the path that executed and, when the learned path did
        # not, why. It is a STRING output so it can be wired to a preview or
        # read at a glance, and it is appended last so existing links, which
        # ComfyUI stores by index, are unaffected.
        attempts: List[str] = []
        path = "deterministic expansion"

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
                    path = "temporal RUDRA"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Temporal RUDRA unavailable (%s).", exc)
                    attempts.append(f"temporal RUDRA unavailable ({exc})")

            # The direct-pixel checkpoint supports stills and frame batches.
            # Until the temporal direct model is trained, video frames are
            # independent and may need downstream deflicker/temporal QC.
            # Ask the resolver, don't just look for a non-empty widget: an
            # installed checkpoint in models/radiance (or RADIANCE_SDR2HDR_PIXEL)
            # is discoverable with the path left blank, which is the default.
            use_pixel = backend == "Direct Pixel" or (
                backend == "Auto" and _pixel_checkpoint_available(pixel_checkpoint)
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
                    path = f"direct-pixel ({pixel_recovery_mode})"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Direct-pixel recovery unavailable (%s).", exc)
                    attempts.append(f"direct-pixel unavailable ({exc})")

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
                    path = f"legacy RUDRA ({rudra_size})"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Legacy RUDRA recovery unavailable (%s).", exc)
                    attempts.append(f"legacy RUDRA unavailable ({exc})")

        # Universal always produces usable HDR. Recover mode therefore falls
        # back to deterministic expansion if the learned path cannot run.
        if mode == "Recover" and not recovery_applied:
            hdr = expanded_hdr

        # 4 ── output encoding
        out = _encode_output(hdr, output_encoding, peak_scale)

        if extra.shape[-1] > 0:                 # pass alpha / extra channels through
            out = torch.cat([out, extra], dim=-1)

        report = self._build_report(
            mode=mode, path=path, recovery_applied=recovery_applied,
            attempts=attempts, frames=int(img.shape[0]),
            reference_white_nits=float(reference_white_nits),
            peak_nits=float(peak_nits), output_encoding=str(output_encoding),
            rudra_blend=float(rudra_blend), vae_connected=vae is not None,
        )
        return (out, mask, shadows, h_conf, s_conf, report)

    @staticmethod
    def _build_report(*, mode, path, recovery_applied, attempts, frames,
                      reference_white_nits, peak_nits, output_encoding,
                      rudra_blend, vae_connected) -> str:
        """One human-readable line per fact about what this run actually did.

        Recover and Hybrid are the reason this node is called Universal, and on
        a machine with no checkpoint installed they are bit-identical to Expand
        — silently, because every backend is wrapped in try/except and only
        logs. This is the output that makes the difference visible on the graph
        instead of in a console the user is not reading.
        """
        lines = [
            f"mode: {mode}",
            f"path: {path}",
            f"frames: {frames}",
            f"reference white: {reference_white_nits:.0f} nits "
            f"(BT.2408 reference is 203)",
            f"mastering peak: {peak_nits:.0f} nits",
            f"output: {output_encoding}",
        ]

        if mode == "Expand":
            lines.append("learned recovery: not requested")
            return "\n".join(lines)

        if recovery_applied:
            lines.append("learned recovery: applied")
            return "\n".join(lines)

        # Requested but did not run. Say why, and say what it cost.
        if rudra_blend <= 0.0:
            why = "rudra_blend is 0"
        elif attempts:
            why = "; ".join(attempts)
        elif not vae_connected:
            why = ("no learned checkpoint found and no VAE connected — install a "
                   "RUDRA checkpoint in models/radiance, set "
                   "RADIANCE_SDR2HDR_PIXEL, or connect a VAE")
        else:
            why = "no backend was eligible for this input"

        lines.append(f"learned recovery: NOT APPLIED — {why}")
        lines.append(
            f"result: identical to Expand. {mode} did nothing this run."
        )
        logger.warning(
            "[SDR→HDR Universal] %s mode produced deterministic Expand output: %s",
            mode, why,
        )
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "RadianceSDRToHDRRecover": RadianceSDRToHDRRecover,
    "RadianceSDRToHDRUniversal": RadianceSDRToHDRUniversal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSDRToHDRRecover": "◎ Radiance SDR → HDR Recover",
    "RadianceSDRToHDRUniversal": "◎ Radiance SDR → HDR Universal",
}
