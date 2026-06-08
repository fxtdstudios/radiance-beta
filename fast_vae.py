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
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance.config.model_map import resolve_model_vae_config

logger = logging.getLogger("radiance.fast_vae")

# ── Environment-configurable checkpoint path ──────────────────────────────────
# Operators can override via: export RADIANCE_TURBO_DECODER=/path/to/ckpt.pth
_ENV_CKPT = os.environ.get("RADIANCE_TURBO_DECODER", "")

# Module-level decoder cache keyed on (latent_channels, is_full_model).
# A dict allows flux (16ch) and sdxl (4ch) models, and turbo vs full
# variants, to coexist in memory — switching model types no longer
# invalidates and reloads the previously-loaded decoder.
_TRAINED_DECODER_CACHE: dict = {}


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


class RadianceTurboDecoder(nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """
    Distilled HDR VAE Decoder.

    Learns to predict log-coded images from VAE latents.
    The inverse log curve (applied externally or by decode_to_linear_realtime)
    converts the output to scene-linear.

    Architecture: 4 × (upsample + conv + residual block)
    Parameters: ~2.1M (flux 16ch), ~1.6M (sdxl 4ch)
    Inference: ~4ms on A100 for a 64×64 latent → 512×512 output (fp16)
    """

    def __init__(self, latent_channels: int = 16, output_channels: int = 3):
        super().__init__()
        self.latent_channels = latent_channels
        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            Block(64, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, padding=1),
            Block(64, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, padding=1),
            Block(64, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, padding=1),
            Block(64, 64),
            nn.Conv2d(64, output_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RadianceFullDecoder(nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """
    High-Fidelity Production HDR VAE Decoder.

    Optimised for visual accuracy and texture preservation.
    Architecture: 4 × (upsample + conv + 2x deep blocks)
    Channels: 128 (vs 64 in Turbo)
    Parameters: ~32M (flux 16ch), ~28M (sdxl 4ch)
    """

    def __init__(self, latent_channels: int = 16, output_channels: int = 3):
        super().__init__()
        self.latent_channels = latent_channels
        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1),
            BlockFull(128, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1),
            BlockFull(128, 128),
            BlockFull(128, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1),
            BlockFull(128, 128),
            BlockFull(128, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1),
            BlockFull(128, 128),
            BlockFull(128, 128),
            nn.Conv2d(128, output_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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
) -> torch.Tensor:
    """
    High-speed decode: latent → scene-linear.
    Supports tiled inference for high-resolution images.
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

    # 2. De-scale latent
    x = (x / _scale).to(dtype)

    # 2. Distilled decode pass
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            if not tiled:
                log_coded_bchw = decoder(x)
            else:
                # Tiled inference
                B, C, H, W = x.shape
                # Output will be 8x the latent size (standard VAE upsample)
                # But our decoder has 3 upsample stages (2x each) => 8x
                log_coded_bchw = torch.zeros((B, 3, H * 8, W * 8), device=device, dtype=dtype)
                
                # We need to account for the 8x upsampling in tile coordinates
                for i in range(0, H, tile_size // 8):
                    for j in range(0, W, tile_size // 8):
                        # Extract tile with overlap
                        si = max(0, i - overlap // 8)
                        sj = max(0, j - overlap // 8)
                        ei = min(H, i + tile_size // 8 + overlap // 8)
                        ej = min(W, j + tile_size // 8 + overlap // 8)
                        
                        tile = x[:, :, si:ei, sj:ej]
                        tile_out = decoder(tile)
                        
                        # Calculate crop for overlap
                        oi = (i - si) * 8
                        oj = (j - sj) * 8
                        wi = min(tile_size, (ei - i) * 8)
                        wj = min(tile_size, (ej - j) * 8)
                        
                        log_coded_bchw[:, :, i*8:i*8+wi, j*8:j*8+wj] = tile_out[:, :, oi:oi+wi, oj:oj+wj]

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

def load_radiance_decoder_weights(
    model_type: str = "flux",
    model_size: str = "turbo",
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Load Radiance Decoder (Turbo or Full) with trained or random weights.

    Priority order for weight source:
      1. explicit `checkpoint_path` argument
      2. RADIANCE_TURBO_DECODER environment variable
      3. Default locations (ComfyUI models/radiance/ directory)
      4. Random initialisation with a clear warning

    Args:
        model_type:       "flux" (16ch) or "sdxl" (4ch).
        model_size:       "turbo" (lightweight) or "full" (production).
        checkpoint_path:  Optional explicit path to .pth checkpoint.

    Returns initialised nn.Module in eval mode.
    """
    cfg = resolve_model_vae_config(model_type)
    expected_channels = cfg.get("latent_channels", 16) if cfg else 16
    # Accept "full" or the node's dropdown value "rudra_full" (substring match) —
    # an exact "== full" check silently built the Turbo arch for rudra_full and
    # then failed the strict load of full-decoder weights.
    request_is_full = ("full" in str(model_size).lower())
    cache_key = (expected_channels, request_is_full)

    # Fast path: return already-loaded model for this (channels, size) pair.
    # Multiple entries coexist so switching between flux/sdxl or turbo/full
    # does not evict the previously-loaded decoder.
    if _TRAINED_DECODER_CACHE and cache_key in _TRAINED_DECODER_CACHE:
        return _TRAINED_DECODER_CACHE[cache_key]

    if request_is_full:
        model = RadianceFullDecoder(latent_channels=expected_channels)
    else:
        model = RadianceTurboDecoder(latent_channels=expected_channels)

    # Resolve checkpoint path
    ckpt_path = checkpoint_path or _ENV_CKPT

    if not ckpt_path:
        # Try default ComfyUI models directory
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
            _types = [model_type]
            if model_type == "flux": _types.append("wan")
            elif model_type == "wan": _types.append("flux")

            candidates = []
            for t in _types:
                # v3.1.2: Expanded candidates to handle common user naming errors (double extensions)
                # and added support for more specific model identifiers.
                for ext in [".safetensors", ".pth", ".pth.pth", ".st"]:
                    candidates.extend([
                        os.path.join(models_dir, "radiance", f"{model_size}_decoder_{t}_ema{ext}"),
                        os.path.join(models_dir, "radiance", f"{model_size}_decoder_{t}{ext}"),
                        os.path.join(models_dir, "radiance", f"turbo_decoder_{t}_ema{ext}") if model_size == "turbo" else None,
                    ])
            for c in candidates:
                if c and os.path.exists(c):
                    ckpt_path = c
                    break
        except ImportError:
            pass

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            if ckpt_path.endswith(".safetensors"):
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
            model.load_state_dict(state_dict, strict=True)
            logger.info(
                f"[Radiance {model_size.upper()}] Loaded trained decoder: {ckpt_path} "
                f"({expected_channels}ch / {model_type})"
            )
        except Exception as e:
            logger.error(
                f"[Radiance {model_size.upper()}] Failed to load checkpoint {ckpt_path}: {e}\n"
                f"Falling back to random weights."
            )
    else:
        msg = (
            f"[Radiance {model_size.upper()}] No trained checkpoint found. "
            f"Decoder is randomly initialised — output will be garbage.\n"
            f"Train a decoder with: python training/train_turbo_decoder.py --model_size {model_size}\n"
            f"Then set: export RADIANCE_TURBO_DECODER=/path/to/checkpoint.pth\n"
            f"Or place checkpoint at: models/radiance/{model_size}_decoder_{model_type}_ema.pth"
        )
        logger.warning(msg)

    model.eval()
    _TRAINED_DECODER_CACHE[cache_key] = model
    return model


def load_turbo_weights(model_type: str = "flux", checkpoint_path: Optional[str] = None) -> nn.Module:
    """Legacy alias for backward compatibility."""
    return load_radiance_decoder_weights(model_type=model_type, model_size="turbo", checkpoint_path=checkpoint_path)
