"""Direct 8-bit SDR to scene-linear HDR image and video models.

Unlike the latent decoders, these modules consume decoded SDR RGB pixels.  The
image network predicts a residual over a deterministic inverse-tone-map
baseline plus explicit highlight and shadow recovery masks.  The temporal
network is deliberately small and refines image-model predictions across a
short clip without requiring a multi-billion-parameter video backbone.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_MODEL_CACHE: dict[tuple[str, str], "SDR2HDRNet"] = {}


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def canonicalize_sdr(sdr: torch.Tensor, transfer: str = "srgb", value_range: str = "full") -> torch.Tensor:
    """Convert common 8-bit SDR encodings to the model's canonical sRGB code."""
    x = sdr.float()
    if value_range == "limited":
        x = (x - 16.0 / 255.0) / (219.0 / 255.0)
    elif value_range != "full":
        raise ValueError("value_range must be 'full' or 'limited'")
    x = x.clamp(0.0, 1.0)
    if transfer == "srgb":
        return x
    if transfer == "rec709":
        linear = torch.where(x < 0.081, x / 4.5, ((x + 0.099) / 1.099).pow(1.0 / 0.45))
    elif transfer == "gamma22":
        linear = x.pow(2.2)
    elif transfer == "gamma24":
        linear = x.pow(2.4)
    else:
        raise ValueError("transfer must be srgb, rec709, gamma22, or gamma24")
    return linear_to_srgb(linear)


def inverse_aces_approx(display_linear: torch.Tensor) -> torch.Tensor:
    """Invert the ACES approximation used by ``prepare_training_data.py``.

    Values at exactly one are clipped and unknowable; clamping them just below
    one produces a finite baseline which the learned residual can extend.
    """
    y = display_linear.clamp(0.0, 0.995)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    qa = y * c - a
    qb = y * d - b
    qc = y * e
    disc = (qb.square() - 4.0 * qa * qc).clamp_min(0.0)
    root_a = (-qb - torch.sqrt(disc)) / (2.0 * qa).clamp(max=-1e-7)
    root_b = (-qb + torch.sqrt(disc)) / (2.0 * qa).clamp(max=-1e-7)
    return torch.maximum(root_a, root_b).clamp_min(0.0)


def sdr_to_baseline_hdr(sdr: torch.Tensor) -> torch.Tensor:
    """Map sRGB SDR to the normalized scene-linear convention used by G:\\data.

    Preparation applies -1 EV before the ACES curve, then stores HDR as scene
    linear * 203/10000.  Therefore inverse-ACES output is multiplied by
    ``2 * 203/10000``.  The learned network handles other camera/tone curves.
    """
    display_linear = srgb_to_linear(sdr)
    return inverse_aces_approx(display_linear) * (2.0 * 203.0 / 10000.0)


def luminance(x: torch.Tensor) -> torch.Tensor:
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


@dataclass
class SDR2HDROutput:
    hdr: torch.Tensor
    baseline: torch.Tensor
    log_residual: torch.Tensor
    highlight_mask: torch.Tensor
    shadow_mask: torch.Tensor
    highlight_logits: torch.Tensor
    shadow_logits: torch.Tensor


class SDR2HDRNet(nn.Module):
    """Compact U-Net for direct SDR pixel to HDR radiance recovery."""

    def __init__(self, base_channels: int = 32, log_scale: float = 16.0, max_hdr: float = 4.0):
        super().__init__()
        c = base_channels
        self.log_scale = float(log_scale)
        self.max_hdr = float(max_hdr)
        self.stem = nn.Conv2d(6, c, 3, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.down1 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResidualBlock(c * 2), ResidualBlock(c * 2))
        self.down2 = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(c * 4), ResidualBlock(c * 4))
        self.up2 = nn.Conv2d(c * 6, c * 2, 3, padding=1)
        self.dec2 = nn.Sequential(ResidualBlock(c * 2), ResidualBlock(c * 2))
        self.up1 = nn.Conv2d(c * 3, c, 3, padding=1)
        self.dec1 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.head = nn.Conv2d(c, 5, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        sdr: torch.Tensor,
        preserve_outside: bool = False,
        recovery_mode: str = "all",
        residual_strength: float = 1.0,
    ) -> SDR2HDROutput:
        if sdr.ndim != 4 or sdr.shape[1] != 3:
            raise ValueError(f"Expected SDR tensor (B,3,H,W), got {tuple(sdr.shape)}")
        sdr = sdr.float().clamp(0.0, 1.0)
        baseline = sdr_to_baseline_hdr(sdr)
        x = torch.cat((sdr, baseline), dim=1)
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(self.up2(torch.cat((u2, e2), dim=1)))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(self.up1(torch.cat((u1, e1), dim=1)))
        raw = self.head(features)

        residual = raw[:, :3]
        highlight_logits = raw[:, 3:4]
        shadow_logits = raw[:, 4:5]
        highlight = torch.sigmoid(highlight_logits)
        shadow = torch.sigmoid(shadow_logits)
        base_log = torch.log1p(baseline * self.log_scale)
        sdr_y = luminance(sdr)
        highlight_prior = torch.sigmoid((sdr_y - 0.82) * 24.0)
        shadow_prior = torch.sigmoid((0.10 - sdr_y) * 24.0)
        # Preserve the physically strong inverse-tone-map baseline through
        # ordinary midtones. Most learned capacity is applied where clipping or
        # crushed shadows made the SDR mapping non-invertible.
        if recovery_mode == "all":
            residual_gate = torch.maximum(highlight_prior, shadow_prior)
        elif recovery_mode == "highlights":
            residual_gate = highlight_prior
        elif recovery_mode == "shadows":
            residual_gate = shadow_prior
        elif recovery_mode == "off":
            residual_gate = torch.zeros_like(highlight_prior)
        else:
            raise ValueError("recovery_mode must be all, highlights, shadows, or off")
        residual_gate = residual_gate * float(residual_strength)
        pred_log = (base_log + residual * residual_gate).clamp(0.0, torch.log1p(torch.tensor(
            self.max_hdr * self.log_scale, device=sdr.device, dtype=sdr.dtype
        )))
        pred = torch.expm1(pred_log) / self.log_scale
        if preserve_outside:
            recovery = torch.maximum(highlight, shadow)
            pred = baseline + recovery * (pred - baseline)
        return SDR2HDROutput(
            pred, baseline, residual, highlight, shadow,
            highlight_logits, shadow_logits,
        )


class TemporalHDRRefiner(nn.Module):
    """Small residual 3D CNN for short-clip temporal consistency.

    Input layouts are ``(B,T,3,H,W)``.  The frozen or jointly trained image
    model supplies the initial HDR frames; this module only learns a bounded
    log-radiance correction using neighboring frames.
    """

    def __init__(self, channels: int = 24, log_scale: float = 16.0):
        super().__init__()
        self.log_scale = float(log_scale)
        self.net = nn.Sequential(
            nn.Conv3d(6, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, 3, (3, 3, 3), padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, sdr: torch.Tensor, image_hdr: torch.Tensor) -> torch.Tensor:
        if sdr.shape != image_hdr.shape or sdr.ndim != 5 or sdr.shape[2] != 3:
            raise ValueError("Expected matching SDR/HDR tensors with shape (B,T,3,H,W)")
        x = torch.cat((sdr, image_hdr), dim=2).permute(0, 2, 1, 3, 4)
        correction = self.net(x).permute(0, 2, 1, 3, 4)
        base_log = torch.log1p(image_hdr.clamp_min(0.0) * self.log_scale)
        return torch.expm1((base_log + 0.25 * torch.tanh(correction)).clamp_min(0.0)) / self.log_scale


def recovery_masks(sdr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft luminance masks used as supervision for recovery regions."""
    y = luminance(sdr)
    highlight = torch.sigmoid((y - 0.82) * 24.0)
    shadow = torch.sigmoid((0.10 - y) * 24.0)
    return highlight, shadow


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px, tx = pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    py, ty = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def sdr2hdr_loss(output: SDR2HDROutput, sdr: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Stable HDR recovery objective in log-radiance and masked regions."""
    scale = 16.0
    pred_log = torch.log1p(output.hdr.clamp_min(0.0) * scale)
    target_log = torch.log1p(target.clamp_min(0.0) * scale)
    hi, sh = recovery_masks(sdr)
    base = F.l1_loss(pred_log, target_log)
    highlight = ((pred_log - target_log).abs() * hi).sum() / (hi.sum() * 3.0 + 1e-6)
    shadow = ((pred_log - target_log).abs() * sh).sum() / (sh.sum() * 3.0 + 1e-6)
    mask = F.binary_cross_entropy_with_logits(output.highlight_logits, hi) + F.binary_cross_entropy_with_logits(output.shadow_logits, sh)
    edge = _gradient_loss(pred_log, target_log)
    pred_chroma = output.hdr / (output.hdr.sum(1, keepdim=True) + 1e-4)
    target_chroma = target / (target.sum(1, keepdim=True) + 1e-4)
    chroma = F.l1_loss(pred_chroma, target_chroma)
    recovery = torch.maximum(hi, sh)
    outside = ((pred_log - target_log).abs() * (1.0 - recovery)).mean()
    residual_outside = (output.log_residual.abs() * (1.0 - recovery)).mean()
    total = (base + 0.50 * highlight + 0.20 * shadow + 0.10 * chroma +
             0.10 * edge + 0.05 * mask + 0.05 * outside + 0.10 * residual_outside)
    return {
        "total": total, "log_l1": base, "highlight": highlight,
        "shadow": shadow, "chroma": chroma, "edge": edge,
        "mask": mask, "outside": outside, "residual_outside": residual_outside,
    }


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match adjacent-frame log-radiance changes without assuming optical flow."""
    p = torch.log1p(pred.clamp_min(0.0) * 16.0)
    t = torch.log1p(target.clamp_min(0.0) * 16.0)
    return F.l1_loss(p[:, 1:] - p[:, :-1], t[:, 1:] - t[:, :-1])


def resolve_pixel_checkpoint(checkpoint_path: str = "") -> Path | None:
    """Resolve an explicit or installed direct-pixel SDR-to-HDR checkpoint."""
    candidates: list[Path] = []
    if checkpoint_path.strip():
        candidates.append(Path(checkpoint_path).expanduser())
    env_path = os.environ.get("RADIANCE_SDR2HDR_PIXEL", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    model_dir = Path(__file__).resolve().parents[2] / "models" / "radiance"
    candidates.extend((
        model_dir / "sdr2hdr_pixel_image.pt",
        model_dir / "sdr2hdr_image_50k.pt",
    ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_pixel_sdr2hdr_weights(checkpoint_path: str, device: torch.device) -> SDR2HDRNet | None:
    """Load and cache a direct-pixel checkpoint on the requested device."""
    resolved = resolve_pixel_checkpoint(checkpoint_path)
    if resolved is None:
        return None
    key = (str(resolved), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    checkpoint = torch.load(str(resolved), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = SDR2HDRNet(base_channels=int(config.get("base_channels", 32)))
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _tile_weight(height: int, width: int, overlap: int, y: int, x: int,
                 full_h: int, full_w: int, device: torch.device) -> torch.Tensor:
    wy = torch.ones(height, device=device, dtype=torch.float32)
    wx = torch.ones(width, device=device, dtype=torch.float32)
    fy, fx = min(overlap, height // 2), min(overlap, width // 2)
    if y > 0 and fy:
        wy[:fy] = torch.linspace(1e-3, 1.0, fy, device=device)
    if y + height < full_h and fy:
        wy[-fy:] = torch.linspace(1.0, 1e-3, fy, device=device)
    if x > 0 and fx:
        wx[:fx] = torch.linspace(1e-3, 1.0, fx, device=device)
    if x + width < full_w and fx:
        wx[-fx:] = torch.linspace(1.0, 1e-3, fx, device=device)
    return (wy[:, None] * wx[None, :])[None, None]


@torch.inference_mode()
def _predict_frame(model: SDR2HDRNet, frame: torch.Tensor, tile_size: int,
                   overlap: int, recovery_mode: str, strength: float) -> torch.Tensor:
    _, _, height, width = frame.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if frame.is_cuda else contextlib.nullcontext()
        with amp:
            return model(frame, recovery_mode=recovery_mode,
                         residual_strength=strength).hdr.float()
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("pixel tile overlap must be >= 0 and smaller than tile size")
    result = torch.zeros((1, 3, height, width), device=frame.device, dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), device=frame.device, dtype=torch.float32)
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile = frame[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if tile.is_cuda else contextlib.nullcontext()
            with amp:
                prediction = model(tile, recovery_mode=recovery_mode,
                                   residual_strength=strength).hdr.float()
            weight = _tile_weight(tile.shape[-2], tile.shape[-1], overlap, y, x,
                                  height, width, frame.device)
            result[..., y:y + tile.shape[-2], x:x + tile.shape[-1]] += prediction * weight
            weights[..., y:y + tile.shape[-2], x:x + tile.shape[-1]] += weight
    return result / weights.clamp_min(1e-6)


@torch.inference_mode()
def predict_pixel_sdr2hdr(sdr_bhwc: torch.Tensor, checkpoint_path: str = "",
                          tile_size: int = 512, tile_overlap: int = 64,
                          recovery_mode: str = "highlights",
                          strength: float = 1.0) -> torch.Tensor:
    """Run the installed image model on an IMAGE batch.

    The returned BHWC tensor follows the model contract: scene-linear RGB,
    where 1.0 equals 10,000 nits. Frames are processed independently until a
    compatible temporal direct-pixel model is trained.
    """
    if sdr_bhwc.ndim != 4 or sdr_bhwc.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(sdr_bhwc.shape)}")
    model = load_pixel_sdr2hdr_weights(checkpoint_path, sdr_bhwc.device)
    if model is None:
        raise RuntimeError(
            "no direct-pixel checkpoint; set pixel_checkpoint, "
            "RADIANCE_SDR2HDR_PIXEL, or install models/radiance/sdr2hdr_pixel_image.pt"
        )
    frames = sdr_bhwc[..., :3].float().clamp(0.0, 1.0).permute(0, 3, 1, 2)
    outputs = [
        _predict_frame(model, frame.unsqueeze(0), int(tile_size), int(tile_overlap),
                       recovery_mode, float(strength))[0]
        for frame in frames
    ]
    return torch.stack(outputs, dim=0).permute(0, 2, 3, 1)
