"""Radiance VAE Decoder models — Turbo (lightweight) and Full (production).

All CATEGORY attributes have been removed — these are pure nn.Module subclasses,
not ComfyUI nodes. Node wrappers live in the nodes/ package.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from radiance.config.env import ENV, get_env
from radiance.config.model_map import resolve_model_vae_config

logger = logging.getLogger("radiance.model.vae")

_ENV_CKPT = get_env(ENV.RADIANCE_TURBO_DECODER, "")
_TRAINED_DECODER_CACHE: dict = {}


class Block(nn.Module):
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
    """Distilled HDR VAE Decoder — lightweight, optimized for speed."""

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
    """High-Fidelity Production HDR VAE Decoder."""

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


def _apply_soft_shoulder(log_coded: torch.Tensor, knee: float, ceiling: float) -> torch.Tensor:
    log_coded = torch.clamp(log_coded, min=0.0)
    above = log_coded > knee
    if above.any():
        result = log_coded.clone()
        rng = ceiling - knee
        excess = (log_coded[above] - knee) / (rng + 1e-8)
        result[above] = knee + rng * torch.tanh(excess)
        return result
    return log_coded


def _apply_inverse_log(log_coded: torch.Tensor, log_curve: str = "ARRI LogC4") -> torch.Tensor:
    from radiance.color.transfer import (
        logc4_to_linear,
        logc3_to_linear,
        slog3_to_linear,
        vlog_to_linear,
        davinci_intermediate_to_linear,
        log3g10_to_linear,
    )
    converters = {
        "ARRI LogC4": logc4_to_linear,
        "ARRI LogC3": logc3_to_linear,
        "Sony S-Log3": slog3_to_linear,
        "Panasonic V-Log": vlog_to_linear,
        "DaVinci Intermediate": davinci_intermediate_to_linear,
        "RED Log3G10": log3g10_to_linear,
    }
    fn = converters.get(log_curve, logc4_to_linear)
    return fn(log_coded)


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
    # Resolve per-model defaults from unified config when not overridden
    cfg = resolve_model_vae_config(model_type) or {}
    _scale = scale_factor if scale_factor is not None else cfg.get("scale_factor", 0.18215)
    _curve = log_curve if log_curve is not None else cfg.get("log_curve", "ARRI LogC4")

    device = latent.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)

    is_video = latent.ndim == 5
    if is_video:
        B, C, T, H, W = latent.shape
        x = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        x = latent

    x = (x / _scale).to(dtype)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            if not tiled:
                log_coded_bchw = decoder(x)
            else:
                B, C, H, W = x.shape
                log_coded_bchw = torch.zeros((B, 3, H * 8, W * 8), device=device, dtype=dtype)
                for i in range(0, H, tile_size // 8):
                    for j in range(0, W, tile_size // 8):
                        si = max(0, i - overlap // 8)
                        sj = max(0, j - overlap // 8)
                        ei = min(H, i + tile_size // 8 + overlap // 8)
                        ej = min(W, j + tile_size // 8 + overlap // 8)
                        tile = x[:, :, si:ei, sj:ej]
                        tile_out = decoder(tile)
                        oi = (i - si) * 8
                        oj = (j - sj) * 8
                        wi = min(tile_size, (ei - i) * 8)
                        wj = min(tile_size, (ej - j) * 8)
                        log_coded_bchw[:, :, i*8:i*8+wi, j*8:j*8+wj] = tile_out[:, :, oi:oi+wi, oj:oj+wj]

    log_coded = log_coded_bchw.permute(0, 2, 3, 1).float()

    knee, ceiling, _, _ = profile_params
    log_coded = _apply_soft_shoulder(log_coded, knee, ceiling)

    if return_log_coded:
        return log_coded

    return _apply_inverse_log(log_coded, _curve)


def load_radiance_decoder_weights(
    model_type: str = "flux",
    model_size: str = "turbo",
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    cfg = resolve_model_vae_config(model_type)
    expected_channels = cfg.get("latent_channels", 16) if cfg else 16
    request_is_full = (model_size == "full")
    cache_key = (expected_channels, request_is_full)

    if _TRAINED_DECODER_CACHE and cache_key in _TRAINED_DECODER_CACHE:
        return _TRAINED_DECODER_CACHE[cache_key]

    if model_size == "full":
        model = RadianceFullDecoder(latent_channels=expected_channels)
    else:
        model = RadianceTurboDecoder(latent_channels=expected_channels)

    ckpt_path = checkpoint_path or _ENV_CKPT

    if not ckpt_path:
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
            candidates = []
            for t in set([model_type, "wan" if model_type == "flux" else "flux" if model_type == "wan" else model_type]):
                for ext in [".safetensors", ".pth", ".pth.pth", ".st"]:
                    candidates.extend([
                        os.path.join(models_dir, "radiance", f"{model_size}_decoder_{t}_ema{ext}"),
                        os.path.join(models_dir, "radiance", f"{model_size}_decoder_{t}{ext}"),
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
                if isinstance(ckpt, dict) and "ema_shadow" in ckpt:
                    state_dict = ckpt["ema_shadow"]
                elif isinstance(ckpt, dict) and "model" in ckpt:
                    state_dict = ckpt["model"]
                else:
                    state_dict = ckpt
            model.load_state_dict(state_dict, strict=True)
            logger.info(
                "[Radiance %s] Loaded trained decoder: %s (%dch / %s)",
                model_size.upper(), ckpt_path, expected_channels, model_type,
            )
        except Exception as e:
            logger.error(
                "[Radiance %s] Failed to load checkpoint %s: %s. Falling back to random weights.",
                model_size.upper(), ckpt_path, e,
            )
    else:
        logger.warning(
            "[Radiance %s] No trained checkpoint found. Decoder is randomly initialised — output will be garbage. "
            "Train a decoder with: python training/train_turbo_decoder.py --model_size %s",
            model_size.upper(), model_size,
        )

    model.eval()
    _TRAINED_DECODER_CACHE[cache_key] = model
    return model


def load_turbo_weights(model_type: str = "flux", checkpoint_path: Optional[str] = None) -> nn.Module:
    """Legacy alias for backward compatibility."""
    return load_radiance_decoder_weights(model_type=model_type, model_size="turbo", checkpoint_path=checkpoint_path)
