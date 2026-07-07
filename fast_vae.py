"""
fast_vae.py — Radiance Turbo Decoder
════════════════════════════════════════════════════════════════════════════════

Architecture: RadianceTurboDecoder — lightweight TAESD-style distilled decoder
               optimised for the Radiance log-domain codec pipeline.

v1.1 FIXES vs original:
  - decode_to_linear_realtime() now returns TRUE SCENE-LINEAR values.
    Previous version applied soft-shoulder but never called the inverse log
    curve — the output was still log-coded. The function name was misleading.
    Fix: inverse log (LogC4 by default, configurable) applied after shoulder.

  - load_turbo_weights() now loads from a checkpoint file if one exists.
    Previous version always returned a randomly-initialised model with a
    TODO comment. If no checkpoint is found it emits a clear warning
    (not silent failure) so operators know the decoder is uninitialised.

  -     _TRAINED_DECODER_CACHE: module-level singleton avoids reloading the
    decoder on every call.

TRAINING:
  See training/train_turbo_decoder.py for the full training pipeline.
  After training, point TURBO_DECODER_PATH to your checkpoint:

      export RADIANCE_TURBO_DECODER=/path/to/turbo_decoder_ema_step050000.pth
"""
from __future__ import annotations

import os
import json
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from radiance.config.model_map import resolve_model_vae_config

logger = logging.getLogger("radiance.fast_vae")

# ── Environment-configurable checkpoint path ──────────────────────────────────
# Operators can override via: export RADIANCE_TURBO_DECODER=/path/to/ckpt.pth
_ENV_CKPT = os.environ.get("RADIANCE_TURBO_DECODER", "")

# Module-level decoder cache keyed on (latent_channels, n_upsample, is_full_model).
# A dict allows flux (16ch) and sdxl (4ch) models, turbo vs full variants, and
# same-channel-count architectures with different upsample factors (e.g. 128ch
# ltx-video vs flux2-klein) to coexist in memory without collisions.
# ALBABIT-FIX: a cached value of None means "no compatible RUDRA checkpoint"
# (FiLM architecture, missing file, or failed strict load) — so repeated calls
# don't repeat the same failing filesystem lookup / warning log.
_TRAINED_DECODER_CACHE: dict = {}

# ALBABIT-FIX: additional model_type tokens to try for checkpoint filenames
# when the primary model_type has no matching file on disk — covers both
# filename inconsistencies (ltx-video's turbo checkpoint is named
# "..._ltx_ema..." without "-video") and documented cross-model decoder reuse
# (RUDRA README: "Z-Image: use the Flux decoder").
_DECODER_TYPE_FALLBACKS: dict[str, list[str]] = {
    "flux": ["wan"], "wan": ["flux"],
    "ltx-video": ["ltx"],
    "chroma": ["flux"], "zimage": ["flux"], "z_image": ["flux"], "qwen": ["flux"],
    "cosmos": ["flux"], "sd3": ["flux"], "lumina2": ["flux"],
    "hunyuanvideo": ["wan"], "cogvideox": ["wan"],
    "sd15": ["sdxl"], "pixart": ["sdxl"], "kolors": ["sdxl"], "aura_flow": ["sdxl"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#                      TURBO-DECODER ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """Residual 3-conv block (same as TAESD)."""
    def __init__(self, ni: int, no: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ni, no, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(no, no, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(no, no, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class BlockFull(nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """Deep residual block for Full Decoder."""
    def __init__(self, ni: int, no: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ni, no, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(no, no, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(no, no, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(no, no, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(no, no, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class DynamicRangePredictor(nn.Module):
    """Predict a compact dynamic-range conditioning vector from latent stats."""

    def __init__(self, latent_channels: int, dr_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(latent_channels * 2, dr_dim),
            nn.SiLU(),
            nn.Linear(dr_dim, dr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.pool(x).flatten(1)
        std = torch.sqrt(self.pool((x - mean[..., None, None]) ** 2).flatten(1) + 1e-6)
        return self.net(torch.cat([mean, std], dim=1))


class _RUDRAConditionedDecoderMixin:
    dr_dim: int | None
    projection: nn.Module | None
    film_in: nn.Module | None
    film_out: nn.Module | None
    predictor: nn.Module | None

    def _init_rudra_conditioning(
        self,
        latent_channels: int,
        output_channels: int,
        dr_dim: int | None,
    ) -> None:
        self.dr_dim = dr_dim
        if dr_dim is None:
            self.projection = None
            self.film_in = None
            self.film_out = None
            self.predictor = None
            return

        self.projection = nn.Sequential(
            nn.Linear(dr_dim, dr_dim),
            nn.SiLU(),
            nn.Linear(dr_dim, dr_dim),
        )
        self.film_in = nn.Linear(dr_dim, latent_channels * 2)
        self.film_out = nn.Linear(dr_dim, output_channels * 2)
        self.predictor = DynamicRangePredictor(latent_channels, dr_dim)

    def _condition_vector(
        self,
        x: torch.Tensor,
        dr_proj: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.dr_dim is None:
            return None

        if dr_proj is None:
            if self.predictor is None:
                raise RuntimeError("RUDRA decoder is missing its predictor module.")
            dr_proj = self.predictor(x)

        if dr_proj.ndim == 1:
            dr_proj = dr_proj.unsqueeze(0)
        dr_proj = dr_proj.to(device=x.device, dtype=x.dtype)
        if dr_proj.shape[0] != x.shape[0]:
            if x.shape[0] % dr_proj.shape[0] != 0:
                raise ValueError(
                    f"RUDRA dr_proj batch {dr_proj.shape[0]} cannot condition latent batch {x.shape[0]}."
                )
            dr_proj = dr_proj.repeat_interleave(x.shape[0] // dr_proj.shape[0], dim=0)

        if self.projection is None:
            return dr_proj
        return self.projection(dr_proj)

    def _apply_input_film(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None or self.film_in is None:
            return x
        scale, bias = self.film_in(cond).chunk(2, dim=-1)
        scale = 1.0 + 0.05 * torch.tanh(scale).view(x.shape[0], x.shape[1], 1, 1)
        bias = 0.05 * bias.view(x.shape[0], x.shape[1], 1, 1)
        return x * scale + bias

    def _apply_output_film(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None or self.film_out is None:
            return x
        scale, bias = self.film_out(cond).chunk(2, dim=-1)
        scale = 1.0 + 0.05 * torch.tanh(scale).view(x.shape[0], x.shape[1], 1, 1)
        bias = 0.05 * bias.view(x.shape[0], x.shape[1], 1, 1)
        return x * scale + bias


class RadianceTurboDecoder(_RUDRAConditionedDecoderMixin, nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """
    Distilled HDR VAE Decoder.

    Learns to predict log-coded images from VAE latents.
    The inverse log curve (applied externally or by decode_to_linear_realtime)
    converts the output to scene-linear.

    Architecture: stem + n_upsample × (upsample + conv + residual block), 64ch
    Parameters (measured from deployed checkpoints, 2026-07-03 review):
      ~0.57M (flux 16ch, 3 stages), ~0.56M (sdxl 4ch), ~0.78M (klein 128ch, 4 stages)
    Inference: ~4ms on A100 for a 64×64 latent → 512×512 output (fp16)
    """

    def __init__(
        self,
        latent_channels: int = 16,
        output_channels: int = 3,
        n_upsample: int = 3,
        dr_dim: int | None = None,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.n_upsample = n_upsample
        self._init_rudra_conditioning(latent_channels, output_channels, dr_dim)
        # n_upsample stages of ×2 → total upscale 2**n_upsample. Default 3 = 8×
        # (reproduces the original fixed layer order, so old checkpoints load).
        # Klein etc. (16× VAE) use n_upsample=4.
        layers = [nn.Conv2d(latent_channels, 64, 3, padding=1), Block(64, 64)]
        for _ in range(n_upsample):
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(64, 64, 3, padding=1), Block(64, 64)]
        layers += [nn.Conv2d(64, output_channels, 3, padding=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, dr_proj: torch.Tensor | None = None) -> torch.Tensor:
        cond = self._condition_vector(x, dr_proj)
        x = self._apply_input_film(x, cond)
        out = self.layers(x)
        return self._apply_output_film(out, cond)


class RadianceFullDecoder(_RUDRAConditionedDecoderMixin, nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """
    High-Fidelity Production HDR VAE Decoder.

    Optimised for visual accuracy and texture preservation.
    Architecture: stem + n_upsample × (upsample + conv + 2× deep blocks)
    Channels: 128 (vs 64 in Turbo)
    Parameters (measured from deployed checkpoints, 2026-07-03 review):
      ~5.63M (flux 16ch, 3 stages), ~9.01M (ltx-video 128ch, 5 stages)
      — NOT the ~32M previously claimed here (see DECODER_CHEATSHEET.md).
    """

    def __init__(
        self,
        latent_channels: int = 16,
        output_channels: int = 3,
        n_upsample: int = 3,
        dr_dim: int | None = None,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.n_upsample = n_upsample
        self._init_rudra_conditioning(latent_channels, output_channels, dr_dim)
        # Default 3 = 8× and reproduces the original layer order (old checkpoints
        # load). 16× backbones (Flux.2 Klein) use n_upsample=4.
        layers = [nn.Conv2d(latent_channels, 128, 3, padding=1), BlockFull(128, 128)]
        for _ in range(n_upsample):
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(128, 128, 3, padding=1), BlockFull(128, 128), BlockFull(128, 128)]
        layers += [nn.Conv2d(128, output_channels, 3, padding=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, dr_proj: torch.Tensor | None = None) -> torch.Tensor:
        cond = self._condition_vector(x, dr_proj)
        x = self._apply_input_film(x, cond)
        out = self.layers(x)
        return self._apply_output_film(out, cond)


# ═══════════════════════════════════════════════════════════════════════════════
#                      HDR TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_soft_shoulder(
    log_coded: torch.Tensor,
    knee: float,
    ceiling: float,
) -> torch.Tensor:
    """
    Soft-shoulder compressor matching vae.py:_soft_log_shoulder().
    Keeps midtones unchanged, gently compresses values above `knee`
    toward `ceiling` via tanh.
    """
    log_coded = torch.clamp(log_coded, min=0.0)
    above = log_coded > knee
    if above.any():
        result = log_coded.clone()
        rng    = ceiling - knee
        excess = (log_coded[above] - knee) / (rng + 1e-8)
        result[above] = knee + rng * torch.tanh(excess)
        return result
    return log_coded


def _apply_inverse_log(
    log_coded: torch.Tensor,
    log_curve: str = "ARRI LogC4",
) -> torch.Tensor:
    """
    Apply inverse log curve to convert log-coded → scene-linear.
    Matches the decode path in vae.py.
    """
    from radiance.color.transfer import (
        tensor_logc4_to_linear,
        tensor_logc3_to_linear,
        tensor_slog3_to_linear,
        tensor_vlog_to_linear,
        tensor_davinci_intermediate_to_linear,
        tensor_log3g10_to_linear,
    )

    converters = {
        "ARRI LogC4":           tensor_logc4_to_linear,
        "ARRI LogC3":           tensor_logc3_to_linear,
        "Sony S-Log3":          tensor_slog3_to_linear,
        "Panasonic V-Log":      tensor_vlog_to_linear,
        "DaVinci Intermediate": tensor_davinci_intermediate_to_linear,
        "RED Log3G10":          tensor_log3g10_to_linear,
    }
    fn = converters.get(log_curve, tensor_logc4_to_linear)
    return fn(log_coded)


def _decode_with_optional_conditioning(
    decoder: nn.Module,
    x: torch.Tensor,
    dr_proj: torch.Tensor | None = None,
) -> torch.Tensor:
    if getattr(decoder, "dr_dim", None) is not None:
        return decoder(x, dr_proj)
    return decoder(x)


# ═══════════════════════════════════════════════════════════════════════════════
#                      INFERENCE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def decode_to_linear_realtime(
    latent: torch.Tensor,
    decoder: nn.Module,
    model_type: str = "flux",
    scale_factor: float | None = None,
    profile_params: Tuple[float, float, float, float] = (0.96, 1.08, 0.80, 0.55),
    precision: str = "bf16",
    log_curve: str | None = None,
    return_log_coded: bool = False,
    tiled: bool = False,
    tile_size: int = 512,
    overlap: int = 64,
    dr_proj: torch.Tensor | None = None,
    latent_space: str = "raw",
) -> torch.Tensor:
    """
    High-speed decode: latent → scene-linear.
    Supports tiled inference for high-resolution images.

    LATENT CONVENTION (2026-07-03 review, empirically verified):
    The decoders were trained on RAW VAE latents — pair generation stored
    `vae.encode(...)`/`latent_dist.sample()` output with NO scale/shift
    (dataset_hdr.py), and train_turbo_decoder.py fed them unchanged. Decoding
    stored flux training pairs through the deployed turbo checkpoint measures:
        raw           ~29-34 dB PSNR(log)   <-- training convention
        latent/scale  ~23 dB                 (the old behavior of this function)
        latent*scale  ~22-25 dB
    So the previous unconditional `x = x / scale_factor` was a train/inference
    mismatch that degraded every decode (its symptom — a global gain error —
    was being compensated downstream by uplift_universal's gain-match).

    Args:
        latent_space: "raw" (default) — the latent is already in the VAE's
            native (unscaled) space; it is fed to the decoder unchanged.
            "scaled" — the latent is in scaled/model space; it is divided by
            the per-model scale_factor first (the legacy behavior; only for
            callers that verified their latents really are scaled).
    """
    # Resolve per-model defaults from unified config when not overridden
    cfg = resolve_model_vae_config(model_type) or {}
    _scale = scale_factor if scale_factor is not None else cfg.get("scale_factor", 0.18215)
    _curve = log_curve if log_curve is not None else cfg.get("log_curve", "ARRI LogC4")

    device = latent.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)

    # 1. Handle 5D latents (video) [B, C, T, H, W]
    is_video = latent.ndim == 5
    if is_video:
        B, C, T, H, W = latent.shape
        # Fold temporal into batch: [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
        x = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        x = latent

    # 2. Map the latent into the decoder's training space (raw VAE latents).
    if latent_space == "scaled":
        x = (x / _scale).to(dtype)   # legacy path for verified scaled-space callers
    else:
        x = x.to(dtype)              # raw: the space the decoders were trained on

    # 2. Distilled decode pass
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            if not tiled:
                log_coded_bchw = _decode_with_optional_conditioning(decoder, x, dr_proj)
            else:
                # Tiled inference
                B, C, H, W = x.shape
                spatial_scale = int(2 ** getattr(decoder, "n_upsample", 3))
                latent_tile = max(1, tile_size // spatial_scale)
                latent_overlap = max(0, overlap // spatial_scale)
                log_coded_bchw = torch.zeros(
                    (B, 3, H * spatial_scale, W * spatial_scale),
                    device=device,
                    dtype=dtype,
                )
                
                # Account for the model-specific VAE spatial factor in tile coordinates.
                for i in range(0, H, latent_tile):
                    for j in range(0, W, latent_tile):
                        # Extract tile with overlap
                        si = max(0, i - latent_overlap)
                        sj = max(0, j - latent_overlap)
                        ei = min(H, i + latent_tile + latent_overlap)
                        ej = min(W, j + latent_tile + latent_overlap)
                        
                        tile = x[:, :, si:ei, sj:ej]
                        tile_out = _decode_with_optional_conditioning(decoder, tile, dr_proj)
                        
                        # Calculate crop for overlap
                        oi = (i - si) * spatial_scale
                        oj = (j - sj) * spatial_scale
                        wi = min(tile_size, (ei - i) * spatial_scale)
                        wj = min(tile_size, (ej - j) * spatial_scale)
                        
                        out_i = i * spatial_scale
                        out_j = j * spatial_scale
                        log_coded_bchw[:, :, out_i:out_i+wi, out_j:out_j+wj] = tile_out[:, :, oi:oi+wi, oj:oj+wj]

    # 3. Convert (B, 3, H, W) → (B, H, W, 3) float32
    log_coded = log_coded_bchw.permute(0, 2, 3, 1).float()

    # 4. Soft-shoulder
    knee, ceiling, _, _ = profile_params
    log_coded = _apply_soft_shoulder(log_coded, knee, ceiling)

    if return_log_coded:
        return log_coded

    # 5. Apply inverse log curve → scene-linear
    linear = _apply_inverse_log(log_coded, _curve)

    return linear


# ═══════════════════════════════════════════════════════════════════════════════
#                      WEIGHT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

# ALBABIT-FIX: shared model_type detection for RUDRA decoder selection — single
# source of truth for RadianceHDRVAEDecode and RadianceNDISender, replacing the
# previous "_ch == 16 -> wan/flux else -> sdxl" heuristics that misclassified
# any 12ch (mochi) or 128ch (ltx-video, flux2/flux2-klein) latent as 4ch sdxl.
def detect_rudra_model_type(latent_channels: int, is_video: bool, vae=None) -> str:
    """
    Map a latent's channel count (+ video/image shape) to the closest
    MODEL_VAE_CONFIG canonical model_type for RUDRA decoder selection.

    Raises ValueError for channel counts with no RUDRA-compatible config.
    """
    _vae_cls = type(getattr(vae, "first_stage_model", vae)).__name__.lower() if vae is not None else ""

    if latent_channels == 128:
        # Only ltx-video (8x VAE, 5D) and flux2/flux2-klein (16x VAE, 4D) are 128ch.
        return "ltx-video" if is_video else "flux2-klein"
    if latent_channels == 16:
        if is_video:
            if "cogvideo" in _vae_cls:
                return "cogvideox"
            if "cosmos" in _vae_cls:
                return "cosmos"
            return "wan"  # WanVAE, or HunyuanVideo's generic AutoencodingEngine
        return "flux"  # Chroma/Z-Image/Qwen/SD3/Lumina2 share Flux's 16ch VAE shape
    if latent_channels == 12:
        return "mochi"  # only 12ch entry in MODEL_VAE_CONFIG
    if latent_channels == 4:
        # ALBABIT-FIX: SDXL and SD 1.5 share the same VAE architecture (4ch,
        # identical structure) — ComfyUI wraps both in the same generic class,
        # so no VAE-level signal can distinguish them. "sdxl" is the only
        # 4ch RUDRA checkpoint that exists, so return it unconditionally.
        return "sdxl"
    raise ValueError(f"RUDRA decoder: unsupported latent channel count {latent_channels}.")


def resolve_rudra_model_type(
    latent_channels: int, is_video: bool, vae=None, model_meta: str = "",
) -> str:
    """
    Determine the RUDRA model_type, preferring the Loader's model_meta JSON
    (unambiguous) over latent-shape auto-detection when it's connected.

    ALBABIT-FIX: detect_rudra_model_type() can't distinguish Flux.2 Dev
    ("flux2") from Flux.2 Klein ("flux2-klein") -- both share the identical
    128ch/16x VAE, so the latent shape alone always resolved to "flux2-klein"
    even when the model actually running was Flux.2 Dev. When model_meta is
    connected, its "arch" field (set by RadianceUnifiedLoader) is the exact
    canonical model_type and is used directly instead of guessing.
    """
    if model_meta:
        try:
            arch = json.loads(model_meta).get("arch", "")
            if arch and arch != "unknown":
                return arch.lower()
        except Exception:
            pass
    return detect_rudra_model_type(latent_channels, is_video, vae=vae)


def _infer_rudra_dr_dim(state_dict: dict) -> int | None:
    """Infer the dynamic-range conditioning size from a RUDRA checkpoint."""
    if not isinstance(state_dict, dict):
        return None

    for key, tensor in state_dict.items():
        name = key[7:] if key.startswith("module.") else key
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        if name.endswith("film_in.weight") or name.endswith("film_out.weight"):
            return int(shape[1])
        if name.endswith("projection.0.weight") or name.endswith("projection.weight"):
            return int(shape[1])
        if name.endswith("predictor.net.0.weight"):
            return int(shape[0])
    return None


def _validate_safetensors_size(path: str) -> Optional[str]:
    """
    Cheap integrity check for a .safetensors file BEFORE attempting to load it.

    A safetensors file is: 8-byte little-endian header length, JSON header,
    then the tensor data blob. The header declares every tensor's
    [start, end) data_offsets, so the expected total file size is knowable
    without reading the data. A truncated upload/download (e.g. the
    ltx-video full-decoder checkpoint shipped at 23 MB when its header
    declares ~36 MB) otherwise fails deep inside deserialization with a
    cryptic out-of-bounds error.

    Returns a human-readable problem description, or None if the file looks
    structurally sound.
    """
    import json as _json
    import struct as _struct

    try:
        actual = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8:
                return f"file is only {actual} bytes — not a safetensors file"
            header_len = _struct.unpack("<Q", head)[0]
            if header_len <= 0 or 8 + header_len > actual:
                return (f"header length {header_len:,} exceeds file size "
                        f"{actual:,} — file is corrupt")
            header = _json.loads(fh.read(header_len))
    except Exception as exc:  # noqa: BLE001 — any parse failure means corrupt
        return f"unreadable safetensors header ({exc})"

    data_start = 8 + header_len
    declared_end = 0
    n_tensors = 0
    n_oob = 0
    for key, meta in header.items():
        if key == "__metadata__" or not isinstance(meta, dict):
            continue
        offsets = meta.get("data_offsets")
        if not offsets or len(offsets) != 2:
            continue
        n_tensors += 1
        declared_end = max(declared_end, int(offsets[1]))
        if data_start + int(offsets[1]) > actual:
            n_oob += 1

    expected = data_start + declared_end
    if expected > actual:
        return (f"truncated: {actual:,} bytes on disk but header declares "
                f"{expected:,} ({n_oob} of {n_tensors} tensors out of bounds). "
                f"The file is incomplete at its source — re-export/re-upload "
                f"a full ~{expected / 1e6:.1f} MB checkpoint")
    return None


def load_radiance_decoder_weights(
    model_type: str = "flux",
    model_size: str = "turbo",
    checkpoint_path: Optional[str] = None,
) -> Optional[nn.Module]:
    """
    Load a trained Radiance Decoder (Turbo or Full) checkpoint.

    Priority order for weight source:
      1. explicit `checkpoint_path` argument
      2. RADIANCE_TURBO_DECODER environment variable
      3. Default locations (ComfyUI models/radiance/ directory)

    Args:
        model_type:       Canonical MODEL_VAE_CONFIG key, e.g. "flux", "sdxl",
                           "wan", "mochi", "ltx-video", "flux2-klein" — see
                           detect_rudra_model_type().
        model_size:       "rudra_turbo"/"turbo" (lightweight) or
                           "rudra_full"/"full" (production).
        checkpoint_path:  Optional explicit path to a checkpoint file.

    Returns the loaded nn.Module in eval mode, or None if no compatible
    checkpoint is available (FiLM-conditioned architecture, no checkpoint on
    disk, or a strict state_dict load failure) — callers should fall back to
    the standard VAE decode in that case.
    """
    import math as _math

    cfg = resolve_model_vae_config(model_type)
    expected_channels = cfg.get("latent_channels", 16) if cfg else 16
    # Number of ×2 upsample stages = log2(VAE spatial factor). 8× → 3, 16× → 4.
    _spatial = cfg.get("vae_spatial_factor", 8) if cfg else 8
    n_upsample = max(1, int(round(_math.log2(_spatial))))
    # Accept "full" or the node's dropdown value "rudra_full" (substring match) —
    # an exact "== full" check silently built the Turbo arch for rudra_full and
    # then failed the strict load of full-decoder weights.
    request_is_full = ("full" in str(model_size).lower())

    # Resolve checkpoint path
    ckpt_path = checkpoint_path or _ENV_CKPT
    # ALBABIT-FIX: track which _types entry actually resolved, so callers can
    # tell "used model_type's own checkpoint" from "silently substituted a
    # cross-architecture one" (e.g. wan -> flux) -- stays None when
    # checkpoint_path/_ENV_CKPT was given explicitly (nothing to substitute).
    resolved_type = None

    if not ckpt_path:
        # Try default ComfyUI models directory
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
            # ALBABIT-FIX: generalised fallback table (was a hardcoded flux<->wan
            # if/elif) — also covers the "ltx-video" -> "ltx" filename mismatch
            # and the RUDRA README's documented cross-model decoder reuse
            # (e.g. "Z-Image: use the Flux decoder").
            _types = [model_type] + _DECODER_TYPE_FALLBACKS.get(model_type, [])

            for t in _types:
                # v3.1.2: Expanded candidates to handle common user naming errors (double extensions)
                # and added support for more specific model identifiers.
                found = None
                for ext in [".safetensors", ".pth", ".pth.pth", ".st"]:
                    for cand in (
                        os.path.join(models_dir, "radiance", f"{model_size}_decoder_{t}_ema{ext}"),
                        os.path.join(models_dir, "radiance", f"{model_size}_decoder_{t}{ext}"),
                    ):
                        if os.path.exists(cand):
                            found = cand
                            break
                    if found:
                        break
                if found:
                    ckpt_path = found
                    resolved_type = t
                    break
        except ImportError:
            pass

    # ALBABIT-FIX: checkpoint path is part of the cache identity. This prevents
    # a prior "no checkpoint" lookup from hiding an explicit checkpoint load.
    cache_key = (
        expected_channels,
        n_upsample,
        request_is_full,
        os.path.abspath(ckpt_path) if ckpt_path else "",
        model_type,
    )
    if cache_key in _TRAINED_DECODER_CACHE:
        return _TRAINED_DECODER_CACHE[cache_key]

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            if ckpt_path.endswith(".safetensors"):
                # Fail fast with a precise message on truncated/corrupt files
                # instead of a cryptic tensor-out-of-bounds error mid-load.
                _corrupt = _validate_safetensors_size(ckpt_path)
                if _corrupt:
                    logger.error(
                        f"[Radiance {model_size.upper()}] Checkpoint "
                        f"{ckpt_path} is unusable — {_corrupt}. "
                        f"Falling back to the standard VAE decode."
                    )
                    _TRAINED_DECODER_CACHE[cache_key] = None
                    return None
                import safetensors.torch
                state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                # Support EMA-only .pth (just state_dict) or full checkpoint
                if isinstance(ckpt, dict) and "ema_shadow" in ckpt:
                    state_dict = ckpt["ema_shadow"]
                elif isinstance(ckpt, dict) and "model" in ckpt:
                    state_dict = ckpt["model"]
                else:
                    state_dict = ckpt
            dr_dim = _infer_rudra_dr_dim(state_dict)
            if request_is_full:
                model = RadianceFullDecoder(
                    latent_channels=expected_channels,
                    n_upsample=n_upsample,
                    dr_dim=dr_dim,
                )
            else:
                model = RadianceTurboDecoder(
                    latent_channels=expected_channels,
                    n_upsample=n_upsample,
                    dr_dim=dr_dim,
                )
            model.load_state_dict(state_dict, strict=True)
            logger.info(
                f"[Radiance {model_size.upper()}] Loaded trained decoder: {ckpt_path} "
                f"({expected_channels}ch / {model_type})"
            )
            # ALBABIT-FIX: resolved_type != model_type means no checkpoint was
            # found for the requested architecture and a cross-model one was
            # substituted (_DECODER_TYPE_FALLBACKS) -- warn explicitly (the INFO
            # log above doesn't make the substitution obvious) and tag the
            # model so RadianceHDRVAEDecode can surface it to the UI.
            if resolved_type and resolved_type != model_type:
                logger.warning(
                    f"[Radiance {model_size.upper()}] No RUDRA checkpoint for "
                    f"model_type={model_type!r} — substituting the {resolved_type!r} "
                    f"checkpoint instead ({ckpt_path})."
                )
                model._radiance_resolved_type = resolved_type
            if model_type == "ltx-video":
                logger.warning(
                    "[Radiance RUDRA] LTX-Video decoder was trained on isolated still "
                    "images encoded as 1-frame videos via ltx_vae.safetensors (LTX v1). "
                    "At inference, LTX 2.3 video latents carry full causal temporal "
                    "context across 81 frames — a distribution the decoder was never "
                    "trained on. Output quality may be severely degraded (abstract noise). "
                    "Retrain the decoder on actual LTX 2.3 video latents for correct results."
                )
        except Exception as e:
            # ALBABIT-FIX: previously fell through and returned the model with
            # random weights ("output will be garbage"). Cache and return None
            # so callers fall back to the standard VAE decoder instead.
            logger.error(
                f"[Radiance {model_size.upper()}] Failed to load checkpoint {ckpt_path}: {e}\n"
                f"Falling back to the standard VAE decoder."
            )
            _TRAINED_DECODER_CACHE[cache_key] = None
            return None
    else:
        # ALBABIT-FIX: previously returned a randomly-initialised decoder
        # ("output will be garbage"). Cache and return None so callers fall
        # back to the standard VAE decoder instead of producing noise.
        logger.warning(
            f"[Radiance {model_size.upper()}] No trained checkpoint found for "
            f"model_type={model_type!r}. Falling back to the standard VAE decoder.\n"
            f"Train a decoder with: python training/train_turbo_decoder.py --model_size {model_size}\n"
            f"Then set: export RADIANCE_TURBO_DECODER=/path/to/checkpoint.pth\n"
            f"Or place checkpoint at: models/radiance/{model_size}_decoder_{model_type}_ema.pth"
        )
        _TRAINED_DECODER_CACHE[cache_key] = None
        return None

    model.eval()
    _TRAINED_DECODER_CACHE[cache_key] = model
    return model


def load_turbo_weights(model_type: str = "flux", checkpoint_path: Optional[str] = None) -> Optional[nn.Module]:
    """Legacy alias for backward compatibility."""
    return load_radiance_decoder_weights(model_type=model_type, model_size="turbo", checkpoint_path=checkpoint_path)
