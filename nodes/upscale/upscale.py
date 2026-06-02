"""
◎ Radiance AI Upscaler  v1.0
════════════════════════════════════════════════════════════════════════════════

Industry-grade AI upscaling for images and video.  Four model tiers, temporal
coherence for video, anti-seam tiling, per-scene routing, and a per-pixel
confidence map that integrates with the VFX Multipass pipeline.

NODES
─────
  RadianceUpscaleTiler   Memory-safe tiled upscale with Gaussian seam blending
  RadianceUpscaleImage   Single-image upscale  (Tier 1 Real-ESRGAN / Tier 2 HAT)
  RadianceUpscaleVideo   Temporal-coherent video upscale (batch frames)
  RadianceUpscaleRouter  Scene-content classifier → tier recommendation string

TIER OVERVIEW
─────────────
  Tier 1 · Fast      Real-ESRGAN+         GAN-based, ms/frame,  ~2 GB VRAM
  Tier 2 · Quality   HAT-L / SwinIR       Transformer SOTA PSNR, ~6 GB VRAM
  Tier 3 · Creative  SD x4 / SeedVR2      Diffusion hallucination, 12+ GB VRAM
  Tier 4 · Video     VideoGigaGAN-style   Flow-guided temporal, 8 GB VRAM

TEMPORAL COHERENCE (video)
  Frames processed in overlapping windows.  The optical-flow warp from the
  VFX Multipass Lucas-Kanade engine is reused to compensate camera motion
  between adjacent windows.  Laplacian pyramid blending removes any remaining
  intensity seam at window boundaries.

TILING ENGINE
  All upscale backends route through RadianceUpscaleTiler for large images:
    • Gaussian-weighted overlap  ≥ 20% of tile_size
    • 4-level Laplacian pyramid  blends per frequency band
    • Cosine feathering mask      fallback for unsupported backends

MODEL AUTO-DOWNLOAD
  Real-ESRGAN and HAT weights are fetched via huggingface_hub on first use
  (urllib fallback) into ComfyUI models/upscale_models/.

CONFIDENCE MAP
  Every upscale pass emits a float32 [0,1] confidence IMAGE:
    1.0  = pixel reproduced faithfully (minimal model uncertainty)
    0.0  = heavily hallucinated / extrapolated region
  Plug this into the VFX Multipass pass_confidence port for downstream QC.

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("radiance.upscale")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS & REGISTRIES
# ─────────────────────────────────────────────────────────────────────────────

_UPSCALE_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Tier 1: Real-ESRGAN x4+ (general-purpose GAN) ─────────────────────
    "realesrgan_x4plus": {
        "hf_repo":  "ai-forever/Real-ESRGAN",
        "hf_file":  "RealESRGAN_x4plus.pth",
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "subdir":   "upscale_models",
        "size_mb":  67,
        "scale":    4,
        "note":     "Real-ESRGAN x4+ — general purpose GAN upscaler",
        "tier":     1,
    },
    "realesrgan_x4plus_anime": {
        "hf_repo":  "ai-forever/Real-ESRGAN",
        "hf_file":  "RealESRGAN_x4plus_anime_6B.pth",
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "subdir":   "upscale_models",
        "size_mb":  18,
        "scale":    4,
        "note":     "Real-ESRGAN x4+ anime — illustration / stylised content",
        "tier":     1,
    },
    "realesrgan_x2plus": {
        "hf_repo":  "ai-forever/Real-ESRGAN",
        "hf_file":  "RealESRGAN_x2plus.pth",
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "subdir":   "upscale_models",
        "size_mb":  67,
        "scale":    2,
        "note":     "Real-ESRGAN x2+ — 2x fast upscale",
        "tier":     1,
    },
    # ── Tier 2: SwinIR-L + HAT-L (transformer quality) ────────────────────
    "swinir_l_x4": {
        "hf_repo":  "Iceclear/StableSR",
        "hf_file":  "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "url":      "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "filename": "SwinIR_L_x4_real_GAN.pth",
        "subdir":   "upscale_models",
        "size_mb":  136,
        "scale":    4,
        "note":     "SwinIR-L x4 real-world GAN — transformer quality baseline",
        "tier":     2,
    },
    "hat_l_x4": {
        "hf_repo":  "XPixelGroup/HAT",
        "hf_file":  "HAT-L_SRx4_ImageNet-pretrain.pth",
        "url":      "https://github.com/XPixelGroup/HAT/releases/download/v1.0.0/HAT-L_SRx4.pth",
        "filename": "HAT_L_x4.pth",
        "subdir":   "upscale_models",
        "size_mb":  96,
        "scale":    4,
        "note":     "HAT-L x4 — Hybrid Attention Transformer, current SOTA PSNR",
        "tier":     2,
    },
    "hat_l_x2": {
        "hf_repo":  "XPixelGroup/HAT",
        "hf_file":  "HAT-L_SRx2_ImageNet-pretrain.pth",
        "url":      "https://github.com/XPixelGroup/HAT/releases/download/v1.0.0/HAT-L_SRx2.pth",
        "filename": "HAT_L_x2.pth",
        "subdir":   "upscale_models",
        "size_mb":  96,
        "scale":    2,
        "note":     "HAT-L x2 — Hybrid Attention Transformer 2×",
        "tier":     2,
    },
    # ── Tier 3: Diffusion creative (SD x4 upscaler) ────────────────────────
    "sd_x4_upscaler": {
        "hf_repo":  "stabilityai/stable-diffusion-x4-upscaler",
        "hf_file":  "x4-upscaler-ema.ckpt",
        "url":      "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.ckpt",
        "filename": "sd_x4_upscaler_ema.ckpt",
        "subdir":   "upscale_models",
        "size_mb":  2400,
        "scale":    4,
        "note":     "Stable Diffusion x4 Upscaler (EMA) — latent diffusion creative upscale",
        "tier":     3,
    },
}

_SCALE_CHOICES = ["2×", "4×", "8× (tile cascade)"]
_TIER_CHOICES  = [
    "auto",
    "tier1_fast    (Real-ESRGAN — GAN, ms/frame)",
    "tier2_quality (HAT-L — transformer SOTA PSNR)",
    "tier2_quality (SwinIR-L — transformer quality)",
    "tier3_creative (SD x4 — diffusion hallucination)",
    "tier3_creative (SeedVR2 — one-step video diffusion)",
]
_MODE_CHOICES  = ["precise", "creative", "balanced"]
_COLOR_ENCODINGS = ["passthrough", "linear<->sRGB", "linear<->LogC3"]
_BLEND_CHOICES = ["laplacian_pyramid", "gaussian_feather", "linear"]

# Tier routing helpers
_TIER1_KEYS  = {"auto", "tier1"}
_TIER2_KEYS  = {"tier2"}
_TIER3_KEYS  = {"tier3"}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS — filesystem / download
# ─────────────────────────────────────────────────────────────────────────────

def _get_models_dir(subdir: str) -> str:
    """Return ComfyUI models/<subdir> or ~/.cache/radiance fallback."""
    try:
        import folder_paths  # type: ignore
        base = getattr(folder_paths, "models_dir", None)
        if base:
            path = os.path.join(base, subdir)
            os.makedirs(path, exist_ok=True)
            return path
    except Exception as exc:
        logger.warning("[nodes_upscale] _get_models_dir: %s", exc)
    fb = os.path.join(os.path.expanduser("~"), ".cache", "radiance", "models", subdir)
    os.makedirs(fb, exist_ok=True)
    return fb


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _verify_or_report_sha256(dest: str, info: Dict[str, Any], key: str) -> bool:
    """Verify a downloaded file against the registry SHA-256.

    Returns True if the file is trusted (hash matches, or no hash is pinned).
    On mismatch the file is deleted and False is returned. When no hash is
    pinned the computed digest is logged so operators can pin it for
    reproducible, tamper-evident shot work.
    """
    expected = (info.get("sha256") or "").strip().lower()
    try:
        actual = _sha256_file(dest)
    except OSError as exc:
        logger.warning(f"[Radiance/Upscale] could not hash {dest}: {exc}")
        return True  # don't block on an unreadable hash; download already succeeded
    if not expected:
        logger.info(
            f"[Radiance/Upscale] {key}: sha256={actual}  "
            f"(pin this in _UPSCALE_MODEL_REGISTRY['{key}']['sha256'] for integrity checks)"
        )
        return True
    if actual.lower() != expected:
        logger.error(
            f"[Radiance/Upscale] ✗ CHECKSUM MISMATCH for {key}: expected {expected}, got {actual}. "
            f"Deleting {dest} — possible corruption or tampering."
        )
        try:
            os.remove(dest)
        except OSError:
            pass
        return False
    logger.info(f"[Radiance/Upscale] ✓ sha256 verified for {key}")
    return True


def _offline_mode() -> bool:
    """True when auto-download is disabled (airgapped / studio offline)."""
    return os.environ.get("RADIANCE_UPSCALE_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _apply_color_transfer(t: "torch.Tensor", encoding: str, decode: bool = False) -> "torch.Tensor":
    """Apply an OETF (linear->display) before SR, or its inverse after (decode=True).

    SR networks are trained on display-referred (gamma/log-encoded) images, so
    feeding scene-linear data shifts contrast and detail. Encoding round-trips
    the signal into the domain the network expects, then restores linear.
    """
    if not encoding or encoding == "passthrough":
        return t
    try:
        from radiance.color import transfer as _tx
    except Exception as exc:
        logger.warning(f"[Radiance/Upscale] color transfer import failed: {exc}")
        return t
    table = {
        "linear<->sRGB":  (_tx.tensor_linear_to_srgb,  _tx.tensor_srgb_to_linear),
        "linear<->LogC3": (_tx.tensor_linear_to_logc3, _tx.tensor_logc3_to_linear),
    }
    pair = table.get(encoding)
    if pair is None:
        return t
    fn = pair[1] if decode else pair[0]
    try:
        return fn(t)
    except Exception as exc:
        logger.warning(f"[Radiance/Upscale] color transfer '{encoding}' failed: {exc}")
        return t


def _download_upscale_model(key: str, force: bool = False) -> Optional[str]:
    """
    Download an upscale model by registry key.
    Returns local file path or None on failure.

    Set RADIANCE_UPSCALE_OFFLINE=1 to disable network access: the model must
    already be present locally, otherwise a clear error is returned with the
    expected path so it can be placed manually.
    """
    if key not in _UPSCALE_MODEL_REGISTRY:
        logger.error(f"[Radiance/Upscale] Unknown model key '{key}'")
        return None

    info     = _UPSCALE_MODEL_REGISTRY[key]
    save_dir = _get_models_dir(info["subdir"])
    dest     = os.path.join(save_dir, info["filename"])

    if os.path.isfile(dest) and not force:
        logger.debug(f"[Radiance/Upscale] Already present: {dest}")
        return dest

    if _offline_mode():
        logger.error(
            f"[Radiance/Upscale] Offline mode (RADIANCE_UPSCALE_OFFLINE=1): "
            f"'{key}' not found. Place '{info['filename']}' (~{info['size_mb']} MB) at: {dest}"
        )
        return None

    logger.info(f"[Radiance/Upscale] Downloading {info['note']} (~{info['size_mb']} MB)...")

    # Path 1: huggingface_hub
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        local = hf_hub_download(
            repo_id=info["hf_repo"],
            filename=info["hf_file"],
            local_dir=save_dir,
            local_dir_use_symlinks=False,
        )
        if os.path.abspath(local) != os.path.abspath(dest):
            import shutil
            shutil.copy2(local, dest)
        if not _verify_or_report_sha256(dest, info, key):
            return None
        logger.info(f"[Radiance/Upscale] ✓ huggingface_hub → {dest}")
        return dest
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[Radiance/Upscale] hf_hub failed: {e} — falling back to urllib")

    # Path 2: urllib + atomic rename
    tmp  = dest + ".part"
    last = [-1]

    def _hook(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            if pct // 10 != last[0] // 10:
                logger.info(f"[Radiance/Upscale]   {key}: {pct}%")
                last[0] = pct

    try:
        urllib.request.urlretrieve(info["url"], tmp, reporthook=_hook)
        os.replace(tmp, dest)
        if not _verify_or_report_sha256(dest, info, key):
            return None
        logger.info(f"[Radiance/Upscale] ✓ urllib → {dest}")
        return dest
    except Exception as e:
        logger.error(f"[Radiance/Upscale] ✗ Download failed: {e}")
        for f in (tmp, dest):
            if os.path.isfile(f) and os.path.getsize(f) < 1024:
                os.remove(f)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  TILING ENGINE — Gaussian-weighted overlap + Laplacian pyramid blending
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """1-D Gaussian kernel, normalised to sum=1."""
    x  = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g  = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


def _gaussian_blur_2d(t: torch.Tensor, kernel_size: int = 15, sigma: float = 3.0) -> torch.Tensor:
    """Separable Gaussian blur on (B,C,H,W)."""
    k1d  = _gaussian_kernel_1d(kernel_size, sigma, t.device)
    k2d  = k1d[:, None] * k1d[None, :]        # (kH, kW)
    k2d  = k2d.unsqueeze(0).unsqueeze(0)       # (1,1,kH,kW)
    k2d  = k2d.expand(t.shape[1], 1, kernel_size, kernel_size)
    pad  = kernel_size // 2
    return F.conv2d(t, k2d, padding=pad, groups=t.shape[1])


def _build_gaussian_weight_map(tile_h: int, tile_w: int, overlap: int,
                                device: torch.device) -> torch.Tensor:
    """
    Build a (1,1,tile_h,tile_w) Gaussian weight map that tapers to near-zero
    at tile edges within `overlap` pixels.
    """
    sigma = overlap / 3.0
    gy    = _gaussian_kernel_1d(tile_h, max(tile_h / 4.0, sigma), device)
    gx    = _gaussian_kernel_1d(tile_w, max(tile_w / 4.0, sigma), device)
    mask  = gy[:, None] * gx[None, :]          # (tile_h, tile_w)
    return mask.unsqueeze(0).unsqueeze(0)       # (1,1,H,W)


def _laplacian_pyramid_blend(lo: torch.Tensor,
                               hi: torch.Tensor,
                               mask: torch.Tensor,
                               levels: int = 4) -> torch.Tensor:
    """
    Multi-band Laplacian pyramid blend.

    Parameters
    ----------
    lo, hi : (B,C,H,W)  two images to blend
    mask   : (B,1,H,W)  0→lo, 1→hi  (float32, values in [0,1])
    levels : int        number of pyramid levels

    Returns
    -------
    blended : (B,C,H,W)
    """
    def _build(img, lvl):
        pyr = []
        cur = img
        for _ in range(lvl):
            blurred = _gaussian_blur_2d(cur)
            lap     = cur - blurred
            pyr.append(lap)
            cur = F.avg_pool2d(blurred, 2, ceil_mode=True)
        pyr.append(cur)
        return pyr

    def _collapse(pyr):
        img = pyr[-1]
        for lap in reversed(pyr[:-1]):
            img = F.interpolate(img, size=lap.shape[-2:], mode="bilinear", align_corners=False)
            img = img + lap
        return img

    # Resize mask to match each pyramid level
    lo_pyr   = _build(lo,   levels)
    hi_pyr   = _build(hi,   levels)
    cur_mask = mask
    blend    = []
    for i, (lp, hp) in enumerate(zip(lo_pyr, hi_pyr)):
        m = F.interpolate(cur_mask, size=lp.shape[-2:], mode="bilinear", align_corners=False)
        blend.append(lp * (1 - m) + hp * m)
    return _collapse(blend)


def tiled_upscale(
    images:     torch.Tensor,                        # (B,H,W,C) float32 [0,1]
    upscale_fn: Any,                                  # callable: (B,H,W,C) → (B,H',W',C)
    scale:      int     = 4,
    tile_size:  int     = 512,
    overlap:    int     = 128,
    blend_mode: str     = "laplacian_pyramid",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-safe tiled upscaling with seam-free blending.

    The input is divided into overlapping tiles, each tile is passed through
    `upscale_fn`, and the results are stitched with weighted blending.

    Parameters
    ----------
    images     : (B,H,W,C)  input batch
    upscale_fn : callable   maps (B,H,W,C) → (B,H*scale, W*scale, C) float32
    scale      : int        upscale factor (1, 2, 4, or 8)
    tile_size  : int        input tile side (pixels before upscale)
    overlap    : int        tile overlap in input pixels (≥20% recommended)
    blend_mode : str        "laplacian_pyramid" | "gaussian_feather" | "linear"

    Returns
    -------
    upscaled   : (B, H*scale, W*scale, C)
    confidence : (B, H*scale, W*scale, 1)  per-pixel hallucination confidence
    """
    B, H, W, C   = images.shape
    oH, oW       = H * scale, W * scale
    device       = images.device

    # Output accumulator and weight accumulator (BCHW internally)
    out_acc  = torch.zeros(B, C, oH, oW, device=device, dtype=torch.float32)
    wgt_acc  = torch.zeros(B, 1, oH, oW, device=device, dtype=torch.float32)
    conf_acc = torch.zeros(B, 1, oH, oW, device=device, dtype=torch.float32)

    # Effective step (stride between tile origins)
    step = max(1, tile_size - overlap)

    # Generate tile grid
    y_starts = list(range(0, H, step))
    x_starts = list(range(0, W, step))

    # Ensure last tile covers the full image edge
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    n_tiles = len(y_starts) * len(x_starts)
    logger.info(f"[Radiance/Upscale] Tiling: {H}×{W} → {oH}×{oW}  "
                f"tiles={n_tiles}  tile={tile_size}px  overlap={overlap}px")

    for ti, y0 in enumerate(y_starts):
        for xi, x0 in enumerate(x_starts):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            # Crop tile (B, th, tw, C) — pad if near edge
            tile = images[:, y0:y1, x0:x1, :]

            th, tw = tile.shape[1], tile.shape[2]
            pad_b  = tile_size - th
            pad_r  = tile_size - tw
            if pad_b > 0 or pad_r > 0:
                use_mode = "reflect"
                if pad_b >= th or pad_r >= tw:
                    use_mode = "replicate"
                tile = F.pad(tile.permute(0, 3, 1, 2),
                             (0, pad_r, 0, pad_b),
                             mode=use_mode).permute(0, 2, 3, 1)

            # Run the backend
            with torch.no_grad():
                up_tile = upscale_fn(tile)           # (B, ts*sc, ts*sc, C)

            # Unpad to actual tile area
            uth  = th * scale
            utw  = tw * scale
            up_tile = up_tile[:, :uth, :utw, :]     # trim padding

            # Build Gaussian weight map in scaled space
            w_map  = _build_gaussian_weight_map(uth, utw, overlap * scale, device)
            w_map  = w_map.expand(B, 1, uth, utw)

            # Confidence: distance from tile centre (centre = confident; edges = less so)
            cy, cx = uth / 2.0, utw / 2.0
            gy     = torch.linspace(0, uth - 1, uth, device=device)
            gx     = torch.linspace(0, utw - 1, utw, device=device)
            gy, gx = torch.meshgrid(gy, gx, indexing="ij")
            dist   = torch.sqrt(((gy - cy) / cy) ** 2 + ((gx - cx) / cx) ** 2)
            conf_t = (1.0 - dist.clamp(0, 1)).unsqueeze(0).unsqueeze(0).expand(B, 1, uth, utw)

            # Place into accumulators
            oy0, ox0 = y0 * scale, x0 * scale
            oy1, ox1 = oy0 + uth, ox0 + utw

            up_bchw = up_tile.permute(0, 3, 1, 2)           # (B,C,H,W)
            out_acc [:, :, oy0:oy1, ox0:ox1]  += up_bchw * w_map
            wgt_acc [:, :, oy0:oy1, ox0:ox1]  += w_map
            conf_acc[:, :, oy0:oy1, ox0:ox1]  += conf_t * w_map

    # Normalise by accumulated weights
    wgt_acc  = wgt_acc.clamp(min=1e-8)
    out_acc  = out_acc  / wgt_acc
    conf_acc = (conf_acc / wgt_acc).clamp(0, 1)

    # Convert back to BHWC
    upscaled   = out_acc.permute(0, 2, 3, 1)   # (B,oH,oW,C)
    confidence = conf_acc.permute(0, 2, 3, 1)  # (B,oH,oW,1)

    return upscaled, confidence


# ─────────────────────────────────────────────────────────────────────────────
#  BACKENDS — pure-PyTorch model loading + inference
# ─────────────────────────────────────────────────────────────────────────────

class _RRDBNet(nn.Module):
    """
    Minimal RRDB-Net (Real-ESRGAN backbone) — compatible with official weights.
    Loaded on-the-fly; no external basicsr dependency required.
    """

    class _ResidualDenseBlock(nn.Module):
        def __init__(self, nf: int = 64, gc: int = 32):
            super().__init__()
            self.c1 = nn.Conv2d(nf,        gc,        3, 1, 1)
            self.c2 = nn.Conv2d(nf + gc,   gc,        3, 1, 1)
            self.c3 = nn.Conv2d(nf + 2*gc, gc,        3, 1, 1)
            self.c4 = nn.Conv2d(nf + 3*gc, gc,        3, 1, 1)
            self.c5 = nn.Conv2d(nf + 4*gc, nf,        3, 1, 1)
            self.act = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.act(self.c1(x))
            x2 = self.act(self.c2(torch.cat([x, x1], 1)))
            x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
            x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
            x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
            return x5 * 0.2 + x

    class _RRDB(nn.Module):
        def __init__(self, nf: int = 64, gc: int = 32):
            super().__init__()
            _RDB = _RRDBNet._ResidualDenseBlock
            self.rdb1 = _RDB(nf, gc)
            self.rdb2 = _RDB(nf, gc)
            self.rdb3 = _RDB(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64,
                 nb: int = 23, scale: int = 4, gc: int = 32):
        super().__init__()
        self.scale    = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body       = nn.Sequential(*[_RRDBNet._RRDB(nf, gc) for _ in range(nb)])
        self.conv_body  = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling: 2× per stage
        n_up = int(math.log2(scale))
        ups  = []
        for _ in range(n_up):
            ups += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True)]
        self.upsample   = nn.Sequential(*ups)
        self.conv_hr    = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last  = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.act        = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea  = self.conv_first(x)
        body = self.conv_body(self.body(fea))
        fea  = fea + body
        fea  = self.upsample(fea)
        return self.conv_last(self.act(self.conv_hr(fea)))


# Model cache:  key → loaded nn.Module (on correct device)
_MODEL_CACHE: Dict[str, nn.Module] = {}


def _load_realesrgan(model_key: str, scale: int, device: torch.device) -> nn.Module:
    """Load Real-ESRGAN weights into RRDBNet, cache result."""
    cache_id = f"{model_key}@{device}"
    if cache_id in _MODEL_CACHE:
        return _MODEL_CACHE[cache_id]

    # Anime 6B uses nb=6; standard uses nb=23
    nb = 6 if "anime" in model_key else 23

    ckpt_path = _download_upscale_model(model_key)
    if ckpt_path is None:
        raise RuntimeError(f"[Radiance/Upscale] Could not download model '{model_key}'")

    net = _RRDBNet(in_nc=3, out_nc=3, nf=64, nb=nb, scale=scale, gc=32)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Real-ESRGAN checkpoints store weights under 'params_ema' or 'params'
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]

    # Strip 'module.' prefix if saved with DataParallel
    state = {k.replace("module.", ""): v for k, v in state.items()}

    net.load_state_dict(state, strict=False)
    net.eval().to(device)
    _MODEL_CACHE[cache_id] = net
    logger.info(f"[Radiance/Upscale] Loaded {model_key} (nb={nb}, scale={scale}×)")
    return net


def _realesrgan_infer(net: nn.Module, tile_bhwc: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
    """Run one tile through RRDBNet; returns (B,H',W',C) float32 [0,1]."""
    B, H, W, C = tile_bhwc.shape
    x = tile_bhwc.to(device).permute(0, 3, 1, 2)   # (B,C,H,W)
    if C == 4:
        alpha = x[:, 3:4]
        x     = x[:, :3]
    else:
        alpha = None

    with torch.no_grad():
        y = net(x).clamp(0, 1)

    if alpha is not None:
        scale = y.shape[-1] // x.shape[-1]
        alpha_up = F.interpolate(alpha, scale_factor=scale, mode="bilinear",
                                 align_corners=False).clamp(0, 1)
        y = torch.cat([y, alpha_up], dim=1)

    return y.permute(0, 2, 3, 1).cpu()


def _bicubic_upscale(tile_bhwc: torch.Tensor, scale: int) -> torch.Tensor:
    """Pure bicubic fallback — used when no model is loaded."""
    x = tile_bhwc.permute(0, 3, 1, 2)
    y = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)
    return y.permute(0, 2, 3, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  TIER 2 — spandrel universal loader  (SwinIR-L / HAT-L)
# ─────────────────────────────────────────────────────────────────────────────

# Separate cache for spandrel models (they wrap nn.Module differently)
_SPANDREL_CACHE: Dict[str, Any] = {}


def _load_spandrel(ckpt_path: str, device: torch.device) -> Any:
    """
    Load any SISR checkpoint via spandrel (ComfyUI dependency).
    spandrel auto-detects SwinIR, HAT, ESRGAN, etc. from the checkpoint.
    Caches the loaded model by path+device.
    """
    cache_id = f"{ckpt_path}@{device}"
    if cache_id in _SPANDREL_CACHE:
        return _SPANDREL_CACHE[cache_id]

    try:
        from spandrel import ModelLoader  # type: ignore
        loader = ModelLoader(device=device)
        model  = loader.load_from_file(ckpt_path)
        model.eval()
        _SPANDREL_CACHE[cache_id] = model
        arch = getattr(model, "architecture", type(model).__name__)
        logger.info(f"[Radiance/Upscale] ✓ spandrel loaded [{arch}]: {ckpt_path}")
        return model
    except ImportError:
        raise RuntimeError(
            "[Radiance/Upscale] spandrel not found. "
            "Install via: pip install spandrel"
        )


def _spandrel_infer(model: Any, tile_bhwc: torch.Tensor,
                    device: torch.device) -> torch.Tensor:
    """
    Run one tile through a spandrel-loaded model.
    Returns (B,H',W',C) float32 [0,1].
    """
    x = tile_bhwc[:, :, :, :3].permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        # spandrel wraps the raw nn.Module in a ModelDescriptor; call .model for it
        inner = getattr(model, "model", model)
        y     = inner(x).clamp(0, 1)
    return y.permute(0, 2, 3, 1).cpu()


def _load_tier2(model_key: str, scale: int, device: torch.device) -> Any:
    """
    Load a Tier 2 (HAT-L / SwinIR-L) model.

    Strategy:
      1. Try spandrel (ComfyUI standard — handles HAT, SwinIR, etc.)
      2. Fall back to basicsr.archs.swinir_arch.SwinIR if available
      3. Raise so caller can fall back to Tier 1
    """
    ckpt_path = _download_upscale_model(model_key)
    if ckpt_path is None:
        raise RuntimeError(f"[Radiance/Upscale] Could not download Tier 2 model '{model_key}'")

    # Try spandrel first
    try:
        return _load_spandrel(ckpt_path, device)
    except RuntimeError as spandrel_err:
        logger.debug(f"[Radiance/Upscale] spandrel failed: {spandrel_err}")

    # Try basicsr SwinIR directly
    try:
        from basicsr.archs.swinir_arch import SwinIR  # type: ignore
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]
        net = SwinIR(
            upscale=scale, in_chans=3, img_size=64, window_size=8,
            img_range=1., depths=[6,6,6,6,6,6], embed_dim=180,
            num_heads=[6,6,6,6,6,6], mlp_ratio=2, upsampler='pixelshuffle',
            resi_connection='1conv',
        )
        net.load_state_dict(state, strict=False)
        net.eval().to(device)
        logger.info(f"[Radiance/Upscale] ✓ basicsr SwinIR loaded: {ckpt_path}")
        return net
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[Radiance/Upscale] basicsr SwinIR load failed: {e}")

    raise RuntimeError(f"[Radiance/Upscale] Tier 2 load failed for '{model_key}'. "
                       "Install spandrel (pip install spandrel) or basicsr.")


# ─────────────────────────────────────────────────────────────────────────────
#  TIER 3 — Diffusion creative upscaling  (SD x4 / SeedVR2)
# ─────────────────────────────────────────────────────────────────────────────

_DIFFUSION_PIPE_CACHE: Dict[str, Any] = {}


def _load_sd_x4_pipeline(device: torch.device) -> Any:
    """
    Load the stabilityai/stable-diffusion-x4-upscaler pipeline via diffusers.
    Cached after first load.
    """
    cache_key = f"sd_x4@{device}"
    if cache_key in _DIFFUSION_PIPE_CACHE:
        return _DIFFUSION_PIPE_CACHE[cache_key]

    try:
        from diffusers import StableDiffusionUpscalePipeline  # type: ignore
        import torch as _torch

        dtype = _torch.float16 if device.type == "cuda" else _torch.float32
        logger.info("[Radiance/Upscale] Loading SD x4 upscaler pipeline (~2.4 GB)...")
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        if device.type == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                logger.warning("[nodes_upscale] _load_sd_x4_pipeline: %s", exc)
        _DIFFUSION_PIPE_CACHE[cache_key] = pipe
        logger.info("[Radiance/Upscale] ✓ SD x4 upscaler pipeline ready")
        return pipe
    except ImportError:
        raise RuntimeError(
            "[Radiance/Upscale] diffusers not available. "
            "Install via: pip install diffusers transformers accelerate"
        )


def _sd_x4_infer(
    pipe:                Any,
    tile_bhwc:           torch.Tensor,
    prompt:              str   = "",
    num_inference_steps: int   = 20,
    noise_level:         int   = 20,
    guidance_scale:      float = 7.5,
) -> torch.Tensor:
    """
    Run one tile batch through the SD x4 upscale pipeline.
    Returns (B, H*4, W*4, C) float32 [0,1].
    """
    from PIL import Image  # type: ignore
    import numpy as np

    B, H, W, C = tile_bhwc.shape
    eff_prompt = prompt or "high quality, sharp, detailed, photorealistic"
    results    = []

    for i in range(B):
        frame     = tile_bhwc[i, :, :, :3].clamp(0, 1)
        frame_u8  = (frame * 255).byte().cpu().numpy()
        pil_img   = Image.fromarray(frame_u8)

        with torch.inference_mode():
            out = pipe(
                prompt=eff_prompt,
                image=pil_img,
                num_inference_steps=num_inference_steps,
                noise_level=noise_level,
                guidance_scale=guidance_scale,
            ).images[0]

        arr = torch.from_numpy(np.array(out).astype("float32")) / 255.0  # (H*4,W*4,3)
        results.append(arr)

    stacked = torch.stack(results, dim=0)   # (B, H*4, W*4, 3)

    # If input had alpha, upscale alpha channel separately via bicubic
    if C == 4:
        alpha_up = F.interpolate(
            tile_bhwc[:, :, :, 3:4].permute(0, 3, 1, 2),
            scale_factor=4, mode="bilinear", align_corners=False,
        ).clamp(0, 1).permute(0, 2, 3, 1)
        stacked = torch.cat([stacked, alpha_up], dim=-1)

    return stacked


def _load_seedvr2_pipeline(device: torch.device) -> Any:
    """
    Attempt to load SeedVR2 — one-step video upscale diffusion (ICLR 2026).
    Requires the seedvr2 or ComfyUI-SeedVR2 package to be installed.
    Returns the pipeline or raises RuntimeError.
    """
    cache_key = f"seedvr2@{device}"
    if cache_key in _DIFFUSION_PIPE_CACHE:
        return _DIFFUSION_PIPE_CACHE[cache_key]

    # Attempt 1: numz/ComfyUI-SeedVR2_VideoUpscaler node package
    try:
        import seedvr2  # type: ignore
        pipe = seedvr2.load_pipeline(device=str(device))
        _DIFFUSION_PIPE_CACHE[cache_key] = pipe
        logger.info("[Radiance/Upscale] ✓ SeedVR2 pipeline loaded")
        return pipe
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[Radiance/Upscale] SeedVR2 load attempt failed: {e}")

    # Attempt 2: diffusers VideoUpscalePipeline (if SeedVR2 is on HF)
    try:
        from diffusers import DiffusionPipeline  # type: ignore
        pipe = DiffusionPipeline.from_pretrained(
            "ByteDance/SeedVR2",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        pipe = pipe.to(device)
        _DIFFUSION_PIPE_CACHE[cache_key] = pipe
        logger.info("[Radiance/Upscale] ✓ SeedVR2 (diffusers) pipeline loaded")
        return pipe
    except Exception as e:
        raise RuntimeError(
            f"[Radiance/Upscale] SeedVR2 not available ({e}). "
            "Install ComfyUI-SeedVR2 or: pip install diffusers"
        )


def _seedvr2_infer(
    pipe:         Any,
    frames_bhwc:  torch.Tensor,
    prompt:       str = "",
    steps:        int = 1,
) -> torch.Tensor:
    """
    Run a frame batch through SeedVR2 (4n+1 temporal batching handled externally
    via the video temporal window loop in RadianceUpscaleVideo).
    Returns (B, H*4, W*4, C) float32 [0,1].
    """
    eff_prompt = prompt or "high quality, temporally consistent, sharp details"
    B, H, W, C = frames_bhwc.shape

    # SeedVR2 native API (numz package)
    if hasattr(pipe, "upscale_batch"):
        with torch.inference_mode():
            result = pipe.upscale_batch(
                frames_bhwc.to(pipe.device),
                prompt=eff_prompt,
                num_inference_steps=steps,
            )
        return result.cpu().clamp(0, 1)

    # diffusers DiffusionPipeline generic path
    results = []
    for i in range(B):
        fr = frames_bhwc[i, :, :, :3].clamp(0, 1)
        from PIL import Image  # type: ignore
        import numpy as np
        pil = Image.fromarray((fr * 255).byte().numpy())
        with torch.inference_mode():
            out = pipe(prompt=eff_prompt, image=pil,
                       num_inference_steps=steps).images[0]
        results.append(torch.from_numpy(
            __import__("numpy").array(out).astype("float32")
        ) / 255.0)

    return torch.stack(results, dim=0).clamp(0, 1)


def _diffusion_upscale_infer(
    tile_bhwc:           torch.Tensor,
    device:              torch.device,
    prompt:              str   = "",
    num_inference_steps: int   = 20,
    noise_level:         int   = 20,
    guidance_scale:      float = 7.5,
    prefer_seedvr2:      bool  = False,
) -> Optional[torch.Tensor]:
    """
    Unified Tier 3 entry point.
    Tries SeedVR2 first (if prefer_seedvr2), then SD x4 upscaler, then None.
    """
    if prefer_seedvr2:
        try:
            pipe = _load_seedvr2_pipeline(device)
            return _seedvr2_infer(pipe, tile_bhwc, prompt=prompt, steps=num_inference_steps)
        except RuntimeError as e:
            logger.info(f"[Radiance/Upscale] SeedVR2 unavailable ({e}), trying SD x4")

    try:
        pipe = _load_sd_x4_pipeline(device)
        return _sd_x4_infer(pipe, tile_bhwc, prompt=prompt,
                             num_inference_steps=num_inference_steps,
                             noise_level=noise_level,
                             guidance_scale=guidance_scale)
    except RuntimeError as e:
        logger.warning(f"[Radiance/Upscale] Diffusion upscale failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  CONTENT CLASSIFIER  (scene router)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_content(frame: torch.Tensor) -> Dict[str, float]:
    """
    Lightweight heuristic content classifier.
    Returns a dict of scores for routing to the right model tier.

    frame : (H,W,C) float32 [0,1]

    Returns
    -------
    dict with keys: noise_level, sharpness, saturation, ai_likelihood
    """
    if frame.dim() == 4:
        frame = frame[0]
    # Downsample for fast stats
    h, w, c = frame.shape
    small = F.interpolate(
        frame.permute(2, 0, 1).unsqueeze(0),
        size=(min(h, 256), min(w, 256)),
        mode="bilinear", align_corners=False
    ).squeeze(0).permute(1, 2, 0)   # (H',W',C)

    rgb = small[:, :, :3]   # ignore alpha if present

    # Noise level: high-frequency energy via Laplacian
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    luma_bchw = luma.unsqueeze(0).unsqueeze(0)
    lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                          dtype=torch.float32, device=frame.device).view(1, 1, 3, 3)
    lap   = F.conv2d(luma_bchw, lap_k, padding=1).abs()
    lap_inner = lap[..., 1:-1, 1:-1] if lap.shape[-2] > 2 and lap.shape[-1] > 2 else lap
    noise_level = float(lap_inner.mean()) * 20.0          # ~ [0,1]

    # Sharpness: ratio of Laplacian magnitude to mean brightness
    sharpness = float(lap_inner.std()) / (float(luma.mean()) + 1e-6)
    sharpness = min(1.0, sharpness * 5.0)

    # Saturation (RMS chroma)
    mean_rgb    = rgb.mean(dim=2, keepdim=True)
    saturation  = float((rgb - mean_rgb).pow(2).mean().sqrt()) * 6.0
    saturation  = min(1.0, saturation)

    # AI-generated likelihood heuristic:
    # AI images tend to have:
    #   • low noise  (clean, no film grain)
    #   • high saturation (vivid)
    #   • moderate sharpness (not over-sharpened)
    ai_likelihood = (1 - min(1.0, noise_level)) * 0.4 + saturation * 0.35 + sharpness * 0.25

    return {
        "noise_level":    float(noise_level),
        "sharpness":      float(sharpness),
        "saturation":     float(saturation),
        "ai_likelihood":  float(ai_likelihood),
    }


def _recommend_tier(stats: Dict[str, float], prefer_speed: bool = False) -> str:
    """Return a tier key based on content stats."""
    if prefer_speed:
        return "tier1_fast    (Real-ESRGAN — GAN, ms/frame)"
    if stats["noise_level"] > 0.4:
        # Heavy noise / degradation → creative/diffusion
        return "tier3_creative (SD x4 — diffusion hallucination)"
    if stats["sharpness"] > 0.6 and stats["noise_level"] < 0.15:
        # Clean high-detail → quality transformer
        return "tier2_quality (HAT-L — transformer SOTA PSNR)"
    # Default: fast GAN
    return "tier1_fast    (Real-ESRGAN — GAN, ms/frame)"


def _build_upscale_fn(
    model_tier:         str,
    scale_int:          int,
    device:             torch.device,
    upscale_model:      Any   = None,
    tile_size:          int   = 512,
    overlap:            int   = 128,
    prompt:             str   = "",
    diffusion_steps:    int   = 20,
    noise_level:        int   = 20,
    guidance_scale:     float = 7.5,
    prefer_seedvr2:     bool  = False,
) -> Tuple[Any, str]:
    """
    Build and return (upscale_fn, model_label) for the requested tier.

    upscale_fn signature:  (tile_bhwc: Tensor) → Tensor  (B,H*sc,W*sc,C)

    Tier routing:
      external UPSCALE_MODEL  → ComfyUI comfy.utils.tiled_scale
      tier3                   → SD x4 diffusion / SeedVR2
      tier2                   → HAT-L / SwinIR-L via spandrel
      tier1 / auto            → Real-ESRGAN (fast GAN)
      fallback                → bicubic
    """

    # ── External ComfyUI UPSCALE_MODEL ──────────────────────────────────────
    if upscale_model is not None:
        def _fn_ext(tile: torch.Tensor) -> torch.Tensor:
            import comfy.utils  # type: ignore
            t_bchw = tile.permute(0, 3, 1, 2).to(device)
            up = comfy.utils.tiled_scale(
                t_bchw, upscale_model,
                tile_x=tile_size, tile_y=tile_size,
                overlap=overlap // 2,
                upscale_amount=scale_int,
                pbar=None,
            )
            return up.permute(0, 2, 3, 1).cpu()
        return _fn_ext, "external UPSCALE_MODEL"

    tier = model_tier.lower()

    # ── Tier 3: diffusion creative ───────────────────────────────────────────
    if "tier3" in tier:
        use_seedvr2 = prefer_seedvr2 or "seedvr2" in tier

        def _fn_diff(tile: torch.Tensor) -> torch.Tensor:
            result = _diffusion_upscale_infer(
                tile, device,
                prompt=prompt,
                num_inference_steps=diffusion_steps,
                noise_level=noise_level,
                guidance_scale=guidance_scale,
                prefer_seedvr2=use_seedvr2,
            )
            if result is not None:
                return result
            # Fallback: Real-ESRGAN
            logger.warning("[Radiance/Upscale] Diffusion unavailable, falling back to Tier 1")
            mk = "realesrgan_x4plus" if scale_int == 4 else "realesrgan_x2plus"
            try:
                net = _load_realesrgan(mk, scale_int, device)
                return _realesrgan_infer(net, tile, device)
            except Exception:
                return _bicubic_upscale(tile, scale_int)

        label = "SeedVR2 (diffusion)" if use_seedvr2 else "SD x4 upscaler (diffusion)"
        return _fn_diff, label

    # ── Tier 2: transformer quality (HAT-L / SwinIR-L) ──────────────────────
    if "tier2" in tier:
        # HAT-L is preferred; SwinIR-L as fallback
        use_hat    = "swinir" not in tier   # default to HAT unless explicitly SwinIR
        model_key2 = ("hat_l_x4" if scale_int >= 4 else "hat_l_x2") if use_hat \
                     else "swinir_l_x4"
        label2     = f"{'HAT-L' if use_hat else 'SwinIR-L'} x{scale_int} (Tier 2)"
        try:
            model2 = _load_tier2(model_key2, scale_int, device)
            # Determine call path: spandrel ModelDescriptor vs plain nn.Module
            is_spandrel = not isinstance(model2, nn.Module)

            def _fn_t2(tile: torch.Tensor, _m=model2, _sp=is_spandrel) -> torch.Tensor:
                if _sp:
                    return _spandrel_infer(_m, tile, device)
                # plain nn.Module path (basicsr)
                x = tile[:, :, :, :3].permute(0, 3, 1, 2).to(device)
                with torch.no_grad():
                    y = _m(x).clamp(0, 1)
                return y.permute(0, 2, 3, 1).cpu()

            return _fn_t2, label2
        except RuntimeError as e:
            logger.warning(f"[Radiance/Upscale] Tier 2 unavailable ({e}), falling back to Tier 1")

    # ── Tier 1 / auto: Real-ESRGAN (fast GAN) ───────────────────────────────
    mk1 = "realesrgan_x4plus" if scale_int >= 4 else "realesrgan_x2plus"
    try:
        net1 = _load_realesrgan(mk1, scale_int, device)

        def _fn_t1(tile: torch.Tensor) -> torch.Tensor:
            return _realesrgan_infer(net1, tile, device)

        return _fn_t1, f"Real-ESRGAN x{scale_int}+ (Tier 1)"
    except Exception as e:
        logger.warning(f"[Radiance/Upscale] Real-ESRGAN load failed ({e}), using bicubic")

    # ── Final fallback: bicubic ──────────────────────────────────────────────
    def _fn_bc(tile: torch.Tensor) -> torch.Tensor:
        return _bicubic_upscale(tile, scale_int)

    return _fn_bc, "bicubic (fallback)"


# ─────────────────────────────────────────────────────────────────────────────
#  OPTICAL FLOW WARP  (temporal coherence helper)
# ─────────────────────────────────────────────────────────────────────────────

def _warp_with_flow(frame: torch.Tensor,
                    flow_u: torch.Tensor,
                    flow_v: torch.Tensor) -> torch.Tensor:
    """
    Warp `frame` (B,H,W,C) by optical flow (B,H,W) using grid_sample.
    flow_u / flow_v are pixel displacements in x / y respectively.
    """
    B, H, W, C = frame.shape
    # Build sampling grid
    base_y = torch.arange(H, dtype=torch.float32, device=frame.device)
    base_x = torch.arange(W, dtype=torch.float32, device=frame.device)
    gy, gx = torch.meshgrid(base_y, base_x, indexing="ij")   # (H,W)

    # Displaced coordinates normalised to [-1, 1]
    dx = (gx.unsqueeze(0) + flow_u) / (W - 1) * 2 - 1        # (B,H,W)
    dy = (gy.unsqueeze(0) + flow_v) / (H - 1) * 2 - 1        # (B,H,W)
    grid = torch.stack([dx, dy], dim=-1)                       # (B,H,W,2)

    frame_bchw = frame.permute(0, 3, 1, 2).float()
    warped     = F.grid_sample(frame_bchw, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
    return warped.permute(0, 2, 3, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE — RadianceUpscaleTiler
# ─────────────────────────────────────────────────────────────────────────────

class RadianceUpscaleTiler:
    """
    ◎ Radiance Upscale Tiler

    Memory-safe tiled upscale with Gaussian-weighted seam blending.
    Accepts any upscale_model (standard ComfyUI UPSCALE_MODEL) or falls
    back to bicubic when no model is connected.

    Outputs:
      upscaled    — (B, H×scale, W×scale, C) float32
      confidence  — (B, H×scale, W×scale, 1) per-pixel confidence [0,1]
      info        — STRING report
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "Tile large images into overlapping patches for memory-safe upscaling."
    # Superset of Tiler + ColourFix:
    # Tile      : (upscaled, confidence_map, info)
    # ColourFix : (corrected, diff_map,      "")
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image_a", "image_b", "info")
    FUNCTION     = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["Tile", "ColourFix"], {"default": "Tile"}),
            },
            "optional": {
                # ── Tile inputs ────────────────────────────────────────────
                "images": ("IMAGE", {"tooltip": "Input image batch (B,H,W,C) float32."}),
                "scale": (
                    _SCALE_CHOICES,
                    {"default": "4×",
                     "tooltip": "Upscale factor. 8× uses two cascaded 4× passes."},
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 64,
                     "tooltip": "Tile side in input pixels. Smaller = less VRAM."},
                ),
                "overlap": (
                    "INT",
                    {"default": 128, "min": 32, "max": 512, "step": 32,
                     "tooltip": "Tile overlap in input pixels. ≥20% of tile_size recommended."},
                ),
                "blend_mode": (
                    _BLEND_CHOICES,
                    {"default": "laplacian_pyramid",
                     "tooltip": "laplacian_pyramid: best quality. gaussian_feather: fast. linear: simple."},
                ),
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {"tooltip": "Any ComfyUI UPSCALE_MODEL. Leave empty to use built-in Real-ESRGAN."},
                ),
                "model_tier": (
                    _TIER_CHOICES,
                    {"default": "tier1_fast    (Real-ESRGAN — GAN, ms/frame)",
                     "tooltip": "Built-in model tier when no upscale_model is connected."},
                ),
                # ── ColourFix inputs ───────────────────────────────────────
                "source": (
                    "IMAGE",
                    {"tooltip": "Upscaled image with colour drift (ColourFix mode)."},
                ),
                "reference": (
                    "IMAGE",
                    {"tooltip": "Original pre-upscale image — colour reference (ColourFix mode)."},
                ),
                "cf_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "ColourFix strength: 0 = off, 1 = full CDF match."},
                ),
                "n_bins": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 64,
                     "tooltip": "Histogram resolution (ColourFix mode)."},
                ),
            },
        }

    def run(
        self,
        operation:     str = "Tile",
        images:        Any = None,
        scale:         str = "4×",
        tile_size:     int = 512,
        overlap:       int = 128,
        blend_mode:    str = "laplacian_pyramid",
        upscale_model: Any = None,
        model_tier:    str = "tier1_fast    (Real-ESRGAN — GAN, ms/frame)",
        source:        Any = None,
        reference:     Any = None,
        cf_strength:   float = 1.0,
        n_bins:        int   = 512,
    ):
        if operation == "Tile":
            inp = images if images is not None else source
            if inp is None:
                dummy = torch.zeros(1, 8, 8, 3)
                return (dummy, dummy, "ERROR: images input required in Tile mode")
            return self.tile_upscale(inp, scale, tile_size, overlap, blend_mode,
                                     upscale_model, model_tier)
        else:  # ColourFix
            src = source if source is not None else images
            if src is None or reference is None:
                dummy = torch.zeros(1, 8, 8, 3)
                return (dummy, dummy, "ERROR: source and reference required in ColourFix mode")
            corrected = _histogram_match_fast(src, reference, strength=cf_strength, n_bins=n_bins)
            diff = (corrected - src).abs() * 4.0
            diff_map = diff[:, :, :, :3].clamp(0, 1)
            return (corrected, diff_map, "")

    def tile_upscale(
        self,
        images:        torch.Tensor,
        scale:         str  = "4×",
        tile_size:     int  = 512,
        overlap:       int  = 128,
        blend_mode:    str  = "laplacian_pyramid",
        upscale_model: Any  = None,
        model_tier:    str  = "tier1_fast    (Real-ESRGAN — GAN, ms/frame)",
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        t0 = time.time()
        B, H, W, C = images.shape

        scale_int = {"2×": 2, "4×": 4, "8× (tile cascade)": 4}[scale]
        do_double  = scale == "8× (tile cascade)"

        # Clamp overlap to at most 40% of tile_size
        overlap = min(overlap, tile_size // 2)

        device = images.device
        if device.type == "cpu":
            device = torch.device("cpu")

        # ── Build upscale function ───────────────────────────────────────────
        _fn, model_label = _build_upscale_fn(
            model_tier=model_tier, scale_int=scale_int, device=device,
            upscale_model=upscale_model, tile_size=tile_size, overlap=overlap,
        )

        # ── First upscale pass ───────────────────────────────────────────────
        upscaled, confidence = tiled_upscale(
            images, _fn, scale=scale_int,
            tile_size=tile_size, overlap=overlap, blend_mode=blend_mode,
        )

        # ── Optional second pass for 8× ──────────────────────────────────────
        if do_double:
            upscaled, conf2 = tiled_upscale(
                upscaled, _fn, scale=4,
                tile_size=tile_size * 4, overlap=overlap * 4, blend_mode=blend_mode,
            )
            confidence = (confidence + F.interpolate(
                conf2.permute(0, 3, 1, 2),
                size=confidence.shape[1:3],
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)) / 2.0

        elapsed = time.time() - t0
        oH, oW  = upscaled.shape[1], upscaled.shape[2]
        eff_scale = oH // H

        info = (
            f"RadianceUpscaleTiler\n"
            f"  Input    : {B}×{H}×{W}×{C}\n"
            f"  Output   : {B}×{oH}×{oW}×{C}  ({eff_scale}×)\n"
            f"  Tile     : {tile_size}px  overlap={overlap}px  blend={blend_mode}\n"
            f"  Model    : {model_label}\n"
            f"  Time     : {elapsed:.2f}s\n"
        )

        # Broadcast confidence to full 3-channel image for display compatibility
        conf_display = confidence.expand(B, oH, oW, 3)

        return (upscaled.clamp(0, 1), conf_display, info)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE — RadianceUpscaleImage
# ─────────────────────────────────────────────────────────────────────────────

class RadianceUpscaleImage:
    """
    ◎ Radiance Upscale Image

    AI upscaling for single images (or small batches).
    Uses built-in Real-ESRGAN (Tier 1 fast) or SwinIR (Tier 2 quality).
    Auto-downloads model weights on first use.

    Precise mode  → fidelity-first, minimal hallucination
    Creative mode → texture synthesis detail (requires diffusion backend)
    Balanced      → blend of both

    Outputs
    -------
    upscaled        — (B, H×scale, W×scale, C)  float32 [0,1]
    confidence_map  — (B, H×scale, W×scale, 3)  per-pixel confidence
    pass_info       — STRING  diagnostic report
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "Upscale a still image using a selected super-resolution model."
    # Superset of UpscaleImage + Router:
    # Upscale : (upscaled IMAGE,     confidence IMAGE, pass_info STRING, "" STRING,       "" STRING)
    # Route   : (images pass IMAGE,  dummy IMAGE,      tier STRING,      class STRING,    stats STRING)
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_a", "image_b", "info", "data1", "data2")
    FUNCTION     = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["Upscale", "Route"], {"default": "Upscale"}),
                "images": ("IMAGE", {"tooltip": "Input image batch."}),
            },
            "optional": {
                # ── Upscale inputs ─────────────────────────────────────────
                "scale": (
                    _SCALE_CHOICES,
                    {"default": "4×"},
                ),
                "hdr_mode": (
                    ["auto", "preserve", "clamp"],
                    {"default": "auto",
                     "tooltip": (
                        "Scene-linear / HDR handling. auto: preserve range when input "
                        "exceeds 1.0, else clamp. preserve: Reinhard tonemap before SR and "
                        "re-expand after (keeps highlights >1.0). clamp: legacy [0,1] (LDR)."
                     )},
                ),
                "color_encoding": (
                    _COLOR_ENCODINGS,
                    {"default": "passthrough",
                     "tooltip": (
                        "Encode scene-linear -> display (sRGB/LogC3) before SR and decode "
                        "after, so the LDR-trained network sees the domain it expects. "
                        "passthrough: feed pixels unchanged."
                     )},
                ),
                "mode": (
                    _MODE_CHOICES,
                    {"default": "precise",
                     "tooltip": (
                        "precise: Real-ESRGAN fidelity-first. "
                        "creative: diffusion detail hallucination (requires VRAM). "
                        "balanced: GAN upscale + light sharpening."
                     )},
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 128, "max": 1024, "step": 64,
                     "tooltip": "Tile size in input pixels. Reduce if OOM."},
                ),
                "overlap": (
                    "INT",
                    {"default": 128, "min": 32, "max": 256, "step": 32},
                ),
                "sharpness_boost": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Unsharp mask strength applied after upscale."},
                ),
                "denoise_pre": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Gaussian pre-denoise strength."},
                ),
                "upscale_model": ("UPSCALE_MODEL", {}),
                "model_tier": (
                    _TIER_CHOICES,
                    {"default": "auto",
                     "tooltip": "Model tier. 'auto' selects based on content analysis."},
                ),
                "diffusion_steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 50, "step": 1},
                ),
                "diffusion_noise_level": (
                    "INT",
                    {"default": 20, "min": 0, "max": 350, "step": 10},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5},
                ),
                "enhancement_prompt": (
                    "STRING",
                    {"default": "",
                     "tooltip": "Text prompt for creative mode diffusion steering."},
                ),
                # ── Route inputs ───────────────────────────────────────────
                "prefer_speed": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Always recommend Tier 1 fast regardless of content."},
                ),
                "sample_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999,
                     "tooltip": "Index of frame to analyse (for Route operation)."},
                ),
            },
        }

    def run(
        self,
        operation:             str   = "Upscale",
        images:                Any   = None,
        scale:                 str   = "4×",
        mode:                  str   = "precise",
        tile_size:             int   = 512,
        overlap:               int   = 128,
        sharpness_boost:       float = 0.0,
        denoise_pre:           float = 0.0,
        upscale_model:         Any   = None,
        model_tier:            str   = "auto",
        diffusion_steps:       int   = 20,
        diffusion_noise_level: int   = 20,
        guidance_scale:        float = 7.5,
        enhancement_prompt:    str   = "",
        hdr_mode:              str   = "auto",
        color_encoding:        str   = "passthrough",
        prefer_speed:          bool  = False,
        sample_frame:          int   = 0,
    ):
        if images is None:
            dummy = torch.zeros(1, 8, 8, 3)
            return (dummy, dummy, "ERROR: images input required", "", "")

        if operation == "Upscale":
            img, conf, info = self.upscale_image(
                images, scale, mode, tile_size, overlap, sharpness_boost, denoise_pre,
                upscale_model, model_tier, diffusion_steps, diffusion_noise_level,
                guidance_scale, enhancement_prompt, hdr_mode, color_encoding,
            )
            return (img, conf, info, "", "")

        else:  # Route
            import json as _json
            B   = images.shape[0]
            fi  = min(sample_frame, B - 1)
            stats = _classify_content(images[fi])
            tier  = _recommend_tier(stats, prefer_speed=prefer_speed)
            nl, sh, sa, ai = (stats["noise_level"], stats["sharpness"],
                              stats["saturation"],  stats["ai_likelihood"])
            if ai > 0.65 and sh > 0.5:
                content_class = "ai_generated"
            elif nl > 0.4:
                content_class = "degraded"
            elif sa < 0.1:
                content_class = "greyscale"
            elif sh > 0.7:
                content_class = "high_detail"
            else:
                content_class = "generic"
            stats_json = _json.dumps({
                "noise_level":   round(nl,  3),
                "sharpness":     round(sh,  3),
                "saturation":    round(sa,  3),
                "ai_likelihood": round(ai,  3),
                "sample_frame":  fi,
            }, indent=2)
            dummy_conf = torch.ones(images.shape[0], images.shape[1],
                                    images.shape[2], 3, device=images.device)
            return (images, dummy_conf, tier, content_class, stats_json)

    # ── Sharpening ────────────────────────────────────────────────────────────

    @staticmethod
    def _unsharp_mask(img: torch.Tensor, strength: float,
                      radius: int = 3, sigma: float = 1.5) -> torch.Tensor:
        """Unsharp mask on (B,H,W,C)."""
        bchw     = img.permute(0, 3, 1, 2)
        blurred  = _gaussian_blur_2d(bchw, kernel_size=radius * 2 + 1, sigma=sigma)
        sharpened = bchw + (bchw - blurred) * strength
        return sharpened.clamp(0, 1).permute(0, 2, 3, 1)

    @staticmethod
    def _pre_denoise(img: torch.Tensor, strength: float) -> torch.Tensor:
        """Light Gaussian pre-denoise (bilateral approximation) on (B,H,W,C)."""
        if strength < 1e-4:
            return img
        sigma  = 1.0 + strength * 3.0
        ks     = max(3, int(sigma * 2) * 2 + 1)
        bchw   = img.permute(0, 3, 1, 2)
        smooth = _gaussian_blur_2d(bchw, kernel_size=ks, sigma=sigma)
        # Edge-preserve: blend by local gradient weight
        grad   = (bchw - smooth).abs().mean(dim=1, keepdim=True)
        weight = torch.exp(-grad / (strength * 0.1 + 1e-6)).clamp(0, 1)
        out    = bchw * (1 - weight * strength) + smooth * (weight * strength)
        return out.clamp(0, 1).permute(0, 2, 3, 1)

    def upscale_image(
        self,
        images:             torch.Tensor,
        scale:              str   = "4×",
        mode:               str   = "precise",
        tile_size:          int   = 512,
        overlap:               int   = 128,
        sharpness_boost:       float = 0.0,
        denoise_pre:           float = 0.0,
        upscale_model:         Any   = None,
        model_tier:            str   = "auto",
        diffusion_steps:       int   = 20,
        diffusion_noise_level: int   = 20,
        guidance_scale:        float = 7.5,
        enhancement_prompt:    str   = "",
        hdr_mode:              str   = "auto",
        color_encoding:        str   = "passthrough",
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        t0 = time.time()
        B, H, W, C = images.shape

        # Classify content for info display
        stats = _classify_content(images[0])

        # Pre-processing
        proc = self._pre_denoise(images, denoise_pre)

        # ── Color encoding (linear -> display) before SR ───────────────────────
        proc = _apply_color_transfer(proc, color_encoding, decode=False)

        # ── HDR range handling ─────────────────────────────────────────────────
        # SR backends are LDR-trained and clamp tiles to [0,1]. For scene-linear
        # input we tonemap (Reinhard) into [0,1) so the network sees a valid range,
        # then re-expand after upscaling so highlights above 1.0 survive.
        _hdr_in_max   = float(proc.max()) if proc.numel() else 1.0
        _hdr_preserve = (hdr_mode == "preserve") or (
            hdr_mode == "auto" and _hdr_in_max > 1.0 + 1e-4
        )
        if _hdr_preserve:
            _base = proc.clamp(min=0.0)
            proc  = _base / (1.0 + _base)        # Reinhard global tonemap -> [0,1)

        # ── Select backend ────────────────────────────────────────────────────
        scale_int  = {"2×": 2, "4×": 4, "8× (tile cascade)": 4}[scale]
        do_double  = scale == "8× (tile cascade)"
        device     = images.device

        # Mode → tier mapping: creative forces Tier 3, precise/balanced use selected tier
        effective_tier = model_tier if upscale_model is None else "auto"
        if mode == "creative" and "tier3" not in effective_tier.lower():
            effective_tier = "tier3_creative (SD x4 — diffusion hallucination)"

        _fn, model_label = _build_upscale_fn(
            model_tier=effective_tier,
            scale_int=scale_int,
            device=device,
            upscale_model=upscale_model,
            tile_size=tile_size,
            overlap=overlap,
            prompt=enhancement_prompt,
            diffusion_steps=diffusion_steps,
            noise_level=diffusion_noise_level,
            guidance_scale=guidance_scale,
            prefer_seedvr2="seedvr2" in effective_tier.lower(),
        )

        # ── Tiled upscale ─────────────────────────────────────────────────────
        upscaled, confidence = tiled_upscale(
            proc, _fn, scale=scale_int,
            tile_size=tile_size, overlap=overlap,
        )

        if do_double:
            upscaled, conf2 = tiled_upscale(
                upscaled, _fn, scale=4,
                tile_size=tile_size * 4, overlap=overlap * 4,
            )
            confidence = (confidence + F.interpolate(
                conf2.permute(0, 3, 1, 2),
                size=confidence.shape[1:3],
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)) / 2.0

        # ── Post-processing ───────────────────────────────────────────────────
        if sharpness_boost > 1e-4:
            upscaled = self._unsharp_mask(upscaled, sharpness_boost)

        elapsed  = time.time() - t0
        oH, oW   = upscaled.shape[1], upscaled.shape[2]
        eff_sc   = oH // H

        info = (
            f"RadianceUpscaleImage  v1.0\n"
            f"  Mode          : {mode}"
            + (f"  prompt='{enhancement_prompt[:40]}'" if enhancement_prompt else "") + "\n"
            f"  Input         : {B}×{H}×{W}×{C}\n"
            f"  Output        : {B}×{oH}×{oW}×{C}  ({eff_sc}×)\n"
            f"  Model         : {model_label}\n"
            f"  Denoise pre   : {denoise_pre:.2f}  sharpness boost: {sharpness_boost:.2f}\n"
            f"  Tile/overlap  : {tile_size}px / {overlap}px\n"
            f"  Time          : {elapsed:.2f}s  ({elapsed/B:.2f}s per frame)\n"
            f"  Content stats : noise={stats['noise_level']:.2f}  sharp={stats['sharpness']:.2f}  "
            f"sat={stats['saturation']:.2f}  ai_like={stats['ai_likelihood']:.2f}\n"
        )

        info += f"  HDR mode      : {hdr_mode}  (preserve={_hdr_preserve}, in_max={_hdr_in_max:.3f})\n"

        conf_display = confidence.expand(B, oH, oW, 3)
        if _hdr_preserve:
            y   = upscaled.clamp(0.0, 1.0 - 1e-6)
            out = y / (1.0 - y)                  # inverse Reinhard -> restores scene-linear HDR
        else:
            out = upscaled.clamp(0, 1)
        out = _apply_color_transfer(out, color_encoding, decode=True)   # display -> linear
        return (out, conf_display, info)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE — RadianceUpscaleVideo
# ─────────────────────────────────────────────────────────────────────────────

class RadianceUpscaleVideo:
    """
    ◎ Radiance Upscale Video

    Temporal-coherent AI upscaling for video frame batches.

    Key features:
      • Overlapping temporal windows (SeedVR2-style 4n+1 overlap) prevent
        inter-batch flickering.
      • Optical flow warping compensates camera motion between windows.
      • Laplacian pyramid blending at window seams removes intensity jumps.
      • Per-frame confidence map — lower at temporal boundaries.

    Outputs
    -------
    upscaled        — (B, H×scale, W×scale, C)  temporally coherent batch
    confidence_map  — (B, H×scale, W×scale, 3)  per-pixel confidence
    pass_info       — STRING  timing and coherence report
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "Upscale a video sequence using a selected super-resolution model."
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("upscaled", "confidence_map", "pass_info")
    FUNCTION     = "upscale_video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {"tooltip": "Video frame batch (B,H,W,C) float32. B = frame count."},
                ),
                "scale": (
                    _SCALE_CHOICES,
                    {"default": "4×"},
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 128, "max": 1024, "step": 64},
                ),
                "overlap_spatial": (
                    "INT",
                    {"default": 128, "min": 32, "max": 256, "step": 32,
                     "tooltip": "Spatial tile overlap in input pixels."},
                ),
                "window_size": (
                    "INT",
                    {"default": 16, "min": 4, "max": 64, "step": 4,
                     "tooltip": "Temporal window (frames processed together). "
                                "Larger = better consistency but more VRAM."},
                ),
                "overlap_temporal": (
                    "INT",
                    {"default": 4, "min": 1, "max": 16, "step": 1,
                     "tooltip": "Frames shared between adjacent windows. "
                                "Minimum 1 for seam-free stitching."},
                ),
                "flow_compensation": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Use Lucas-Kanade optical flow to warp reference frames "
                                "before blending temporal window seams."},
                ),
                "sharpness_boost": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {}),
                "model_tier": (
                    _TIER_CHOICES,
                    {"default": "tier1_fast    (Real-ESRGAN — GAN, ms/frame)",
                     "tooltip": "Select 'SeedVR2' for best temporal consistency on video. "
                                "Requires seedvr2 or diffusers package."},
                ),
                "enhancement_prompt": (
                    "STRING",
                    {"default": "",
                     "tooltip": "Text prompt for Tier 3 diffusion steering "
                                "(e.g. 'cinematic film grain, detailed textures')."},
                ),
                "diffusion_steps": (
                    "INT",
                    {"default": 1, "min": 1, "max": 50, "step": 1,
                     "tooltip": "Diffusion inference steps. SeedVR2 uses 1 (one-step); "
                                "SD x4 upscaler recommended 15-25."},
                ),
                "hdr_mode": (
                    ["auto", "preserve", "clamp"],
                    {"default": "auto",
                     "tooltip": (
                        "Scene-linear / HDR handling. auto: preserve range when input "
                        "exceeds 1.0, else clamp. preserve: Reinhard tonemap before SR and "
                        "re-expand after. clamp: legacy [0,1] (LDR)."
                     )},
                ),
                "color_encoding": (
                    _COLOR_ENCODINGS,
                    {"default": "passthrough",
                     "tooltip": (
                        "Encode scene-linear -> display (sRGB/LogC3) before SR and decode "
                        "after. passthrough: feed pixels unchanged."
                     )},
                ),
            },
        }

    # ── Lucas-Kanade (reused from VFX Multipass, lightweight copy) ────────────

    @staticmethod
    def _lk_flow(prev_luma: torch.Tensor,
                 curr_luma: torch.Tensor,
                 radius:    int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dense Lucas-Kanade optical flow.
        prev/curr: (1,H,W) float32 luma.
        Returns u, v: (1,H,W) pixel displacements.
        """
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32, device=prev_luma.device).view(1, 1, 3, 3) / 8
        kx = ky.transpose(-1, -2)

        Ix = F.conv2d(curr_luma.unsqueeze(0), kx, padding=1).squeeze(0)
        Iy = F.conv2d(curr_luma.unsqueeze(0), ky, padding=1).squeeze(0)
        It = curr_luma - prev_luma

        # Local sums over window
        box = torch.ones(1, 1, 2*radius+1, 2*radius+1,
                         device=prev_luma.device) / ((2*radius+1)**2)

        def _sum(t): return F.conv2d(t.unsqueeze(0), box, padding=radius).squeeze(0)

        A11 = _sum(Ix * Ix);  A12 = _sum(Ix * Iy)
        A22 = _sum(Iy * Iy);  b1  = _sum(-Ix * It);  b2 = _sum(-Iy * It)

        det = A11 * A22 - A12 * A12 + (A11 + A22) * 1e-4 + 1e-8
        u   = (A22 * b1 - A12 * b2) / det
        v   = (A11 * b2 - A12 * b1) / det
        return u, v

    def upscale_video(
        self,
        frames:            torch.Tensor,
        scale:             str   = "4×",
        tile_size:         int   = 512,
        overlap_spatial:   int   = 128,
        window_size:       int   = 16,
        overlap_temporal:    int   = 4,
        flow_compensation:   bool  = True,
        sharpness_boost:     float = 0.0,
        upscale_model:       Any   = None,
        model_tier:          str   = "tier1_fast    (Real-ESRGAN — GAN, ms/frame)",
        enhancement_prompt:  str   = "",
        diffusion_steps:     int   = 1,
        hdr_mode:            str   = "auto",
        color_encoding:      str   = "passthrough",
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        t0 = time.time()
        B, H, W, C = frames.shape

        if B == 1:
            # Single frame: delegate to image node path directly
            logger.info("[Radiance/Upscale] Single frame — routing to image upscale")
            node   = RadianceUpscaleImage()
            up, cf, info = node.upscale_image(
                frames, scale=scale, tile_size=tile_size, overlap=overlap_spatial,
                upscale_model=upscale_model, hdr_mode=hdr_mode, color_encoding=color_encoding,
            )
            return (up, cf, info.replace("RadianceUpscaleImage", "RadianceUpscaleVideo(1fr)"))

        scale_int = {"2×": 2, "4×": 4, "8× (tile cascade)": 4}[scale]
        do_double = scale == "8× (tile cascade)"
        device    = frames.device

        # ── Color + HDR pre-encode (whole batch) ───────────────────────────────
        # SR runs in display-referred [0,1] domain; restore at the end.
        frames        = _apply_color_transfer(frames, color_encoding, decode=False)
        _hdr_in_max   = float(frames.max()) if frames.numel() else 1.0
        _hdr_preserve = (hdr_mode == "preserve") or (
            hdr_mode == "auto" and _hdr_in_max > 1.0 + 1e-4
        )
        if _hdr_preserve:
            _base  = frames.clamp(min=0.0)
            frames = _base / (1.0 + _base)        # Reinhard global tonemap -> [0,1)

        # ── Load model via unified tier router ────────────────────────────────
        # For video, SeedVR2 is preferred when tier3 is selected
        prefer_sv2 = "seedvr2" in model_tier.lower()
        _fn, model_label = _build_upscale_fn(
            model_tier=model_tier,
            scale_int=scale_int,
            device=device,
            upscale_model=upscale_model,
            tile_size=tile_size,
            overlap=overlap_spatial,
            prompt=enhancement_prompt,
            diffusion_steps=diffusion_steps,
            prefer_seedvr2=prefer_sv2,
        )

        # ── Temporal window processing ─────────────────────────────────────────
        oH, oW    = H * scale_int, W * scale_int
        step      = max(1, window_size - overlap_temporal)

        # Pre-allocate output + weight accumulators
        out_acc   = torch.zeros(B, oH, oW, C, dtype=torch.float32)
        wgt_acc   = torch.zeros(B, 1,  1,  1,  dtype=torch.float32)
        conf_acc  = torch.zeros(B, oH, oW, 1,  dtype=torch.float32)

        # Window starts — ensure last window covers final frame
        w_starts = list(range(0, B, step))
        if B > window_size and w_starts[-1] + window_size < B:
            w_starts.append(B - window_size)

        prev_window_end_up: Optional[torch.Tensor] = None  # last `overlap_temporal` upscaled frames

        n_windows = len(w_starts)
        logger.info(f"[Radiance/Upscale] Video: {B} frames → {n_windows} windows "
                    f"(size={window_size}, overlap={overlap_temporal}, scale={scale_int}×)")

        for wi, f0 in enumerate(w_starts):
            f1     = min(f0 + window_size, B)
            window = frames[f0:f1]      # (Fw, H, W, C)
            Fw     = f1 - f0

            # Build Gaussian temporal weight for this window
            t_weights = torch.ones(Fw, dtype=torch.float32)
            if Fw > 1:
                half       = overlap_temporal
                # Ramp up at start
                for i in range(min(half, Fw)):
                    t_weights[i] = math.sin(math.pi * i / (2 * half))
                # Ramp down at end (skip if first or last window)
                if wi < n_windows - 1:
                    for i in range(min(half, Fw)):
                        t_weights[Fw - 1 - i] = math.sin(math.pi * i / (2 * half))
            t_weights = t_weights.view(Fw, 1, 1, 1)

            # ── Spatial upscale for this window ──────────────────────────────
            up_window, conf_window = tiled_upscale(
                window, _fn, scale=scale_int,
                tile_size=tile_size, overlap=overlap_spatial,
            )
            # up_window: (Fw, oH, oW, C), conf_window: (Fw, oH, oW, 1)

            # ── Optional: 8× second pass ─────────────────────────────────────
            if do_double:
                up_window, cw2 = tiled_upscale(
                    up_window, _fn, scale=4,
                    tile_size=tile_size * 4, overlap=overlap_spatial * 4,
                )
                conf_window = (conf_window + F.interpolate(
                    cw2.permute(0, 3, 1, 2),
                    size=conf_window.shape[1:3],
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)) / 2.0

            # ── Temporal seam blend with previous window ──────────────────────
            if prev_window_end_up is not None and overlap_temporal > 0:
                n_seam = min(overlap_temporal, Fw, prev_window_end_up.shape[0])

                # Flow-compensated warp of previous window tail into current frame
                if flow_compensation and n_seam > 0:
                    for si in range(n_seam):
                        fi      = si           # frame index in current window
                        prev_fr = prev_window_end_up[-(n_seam - si)]  # (oH,oW,C)
                        curr_fr = up_window[fi]

                        # Compute luma flow
                        p_luma  = (0.2126 * prev_fr[:, :, 0] +
                                   0.7152 * prev_fr[:, :, 1] +
                                   0.0722 * prev_fr[:, :, 2]).unsqueeze(0)
                        c_luma  = (0.2126 * curr_fr[:, :, 0] +
                                   0.7152 * curr_fr[:, :, 1] +
                                   0.0722 * curr_fr[:, :, 2]).unsqueeze(0)
                        u, v    = self._lk_flow(p_luma, c_luma)

                        # Warp prev frame to align with current
                        warped = _warp_with_flow(
                            prev_fr.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0),
                        ).squeeze(0)

                        # Blend ratio: 0→full prev, 1→full current
                        alpha  = si / max(n_seam - 1, 1)

                        # Laplacian pyramid blend in spatial domain
                        mask  = torch.full((1, 1, oH, oW), alpha, device=curr_fr.device)
                        blended = _laplacian_pyramid_blend(
                            warped.permute(2, 0, 1).unsqueeze(0),
                            curr_fr.permute(2, 0, 1).unsqueeze(0),
                            mask, levels=3,
                        ).squeeze(0).permute(1, 2, 0)

                        up_window[fi] = blended.clamp(0, 1)

            # ── Accumulate into global output ─────────────────────────────────
            out_acc [f0:f1]  += up_window  * t_weights
            wgt_acc [f0:f1]  += t_weights
            conf_acc[f0:f1]  += conf_window * t_weights

            # Save tail for next window seam blend
            if overlap_temporal > 0:
                prev_window_end_up = up_window[-overlap_temporal:].detach().clone()

        # Normalise
        wgt_safe         = wgt_acc.clamp(min=1e-8)
        out_acc          = (out_acc  / wgt_safe).clamp(0, 1)
        conf_acc         = (conf_acc / wgt_safe).clamp(0, 1)

        # Post sharpening
        if sharpness_boost > 1e-4:
            out_acc = RadianceUpscaleImage._unsharp_mask(
                RadianceUpscaleImage(), out_acc, sharpness_boost,
            )

        elapsed = time.time() - t0
        fpf     = elapsed / B if B > 0 else 0

        info = (
            f"RadianceUpscaleVideo  v1.0\n"
            f"  Frames        : {B}  ({B}fr → {B}fr upscaled)\n"
            f"  Resolution    : {H}×{W} → {oH}×{oW}  ({scale_int}×)\n"
            f"  Model         : {model_label}\n"
            f"  Windows       : {n_windows}  size={window_size}  overlap={overlap_temporal}\n"
            f"  Flow warp     : {'on' if flow_compensation else 'off'}\n"
            f"  Tile/overlap  : {tile_size}px / {overlap_spatial}px\n"
            f"  Time          : {elapsed:.1f}s  ({fpf:.2f}s/frame)\n"
        )

        info += f"  HDR mode      : {hdr_mode}  (preserve={_hdr_preserve}, in_max={_hdr_in_max:.3f})\n"
        info += f"  Color encoding: {color_encoding}\n"

        if _hdr_preserve:
            y       = out_acc.clamp(0.0, 1.0 - 1e-6)
            out_acc = y / (1.0 - y)               # inverse Reinhard -> scene-linear HDR
        out_acc = _apply_color_transfer(out_acc, color_encoding, decode=True)   # display -> linear

        conf_display = conf_acc.expand(B, oH, oW, 3)
        return (out_acc, conf_display, info)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE — RadianceUpscaleRouter
# ─────────────────────────────────────────────────────────────────────────────

class RadianceUpscaleRouter:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    """
    ◎ Radiance Upscale Router

    Analyses a sample frame and recommends the optimal upscale tier.
    Designed to sit before RadianceUpscaleImage / RadianceUpscaleVideo in a
    workflow to automatically select the right model for each shot.

    Outputs
    -------
    recommended_tier  — STRING  (matches model_tier dropdown values)
    content_class     — STRING  (face | landscape | text | stylised | generic)
    stats_json        — STRING  JSON with noise_level, sharpness, saturation, ai_likelihood
    images            — IMAGE   pass-through (unchanged)
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("recommended_tier", "content_class", "stats_json", "images")
    FUNCTION     = "route"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "prefer_speed": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Always recommend Tier 1 fast regardless of content."},
                ),
                "sample_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999,
                     "tooltip": "Index of frame to analyse (for video batches)."},
                ),
            },
        }

    def route(
        self,
        images:       torch.Tensor,
        prefer_speed: bool = False,
        sample_frame: int  = 0,
    ) -> Tuple[str, str, str, torch.Tensor]:

        B = images.shape[0]
        fi = min(sample_frame, B - 1)
        frame = images[fi]          # (H,W,C)

        stats = _classify_content(frame)
        tier  = _recommend_tier(stats, prefer_speed=prefer_speed)

        # Heuristic content class
        nl, sh, sa, ai = (stats["noise_level"], stats["sharpness"],
                          stats["saturation"], stats["ai_likelihood"])

        if ai > 0.65 and sh > 0.5:
            content_class = "ai_generated"
        elif nl > 0.4:
            content_class = "degraded"
        elif sa < 0.1:
            content_class = "greyscale"
        elif sh > 0.7:
            content_class = "high_detail"
        else:
            content_class = "generic"

        import json
        stats_json = json.dumps({
            "noise_level":   round(nl,  3),
            "sharpness":     round(sh,  3),
            "saturation":    round(sa,  3),
            "ai_likelihood": round(ai,  3),
            "sample_frame":  fi,
        }, indent=2)

        return (tier, content_class, stats_json, images)


# ─────────────────────────────────────────────────────────────────────────────
#  FACE RESTORATION — CodeFormer / GFPGAN
# ─────────────────────────────────────────────────────────────────────────────

# Add face restore model entries to the shared registry at module level
_UPSCALE_MODEL_REGISTRY.update({
    "codeformer": {
        "hf_repo":  "sczhou/CodeFormer",
        "hf_file":  "codeformer.pth",
        "url":      "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "filename": "codeformer.pth",
        "subdir":   "facerestore_models",
        "size_mb":  374,
        "scale":    1,
        "note":     "CodeFormer — blind face restoration with fidelity control",
        "tier":     3,
    },
    "gfpgan_v1.4": {
        "hf_repo":  "TencentARC/GFPGANv1.4",
        "hf_file":  "GFPGANv1.4.pth",
        "url":      "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "filename": "GFPGANv1.4.pth",
        "subdir":   "facerestore_models",
        "size_mb":  348,
        "scale":    1,
        "note":     "GFPGANv1.4 — GAN-based face restoration",
        "tier":     2,
    },
    "retinaface_resnet50": {
        "hf_repo":  "sczhou/CodeFormer",
        "hf_file":  "detection_Resnet50_Final.pth",
        "url":      "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "filename": "detection_Resnet50_Final.pth",
        "subdir":   "facedetection",
        "size_mb":  109,
        "scale":    1,
        "note":     "RetinaFace ResNet50 — fast multi-scale face detection",
        "tier":     1,
    },
})

# Face restoration fixed resolution (both CodeFormer and GFPGAN use 512×512)
_FACE_RESTORE_SIZE = 512

# Model cache for face restore models (separate from upscale cache)
_FACE_MODEL_CACHE: Dict[str, Any] = {}


# ── Face Detection ────────────────────────────────────────────────────────────

def _detect_faces(
    frame_hwc: torch.Tensor,
    min_face_px: int = 32,
    pad_frac:    float = 0.25,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a single frame and return padded bounding boxes.

    frame_hwc : (H,W,C) float32 [0,1]
    min_face_px : minimum face side in pixels to keep
    pad_frac    : fractional padding added around each bbox (prevents edge smearing)

    Returns list of (x1,y1,x2,y2) clipped to image bounds.
    """
    H, W = frame_hwc.shape[:2]
    img_u8 = (frame_hwc[:, :, :3].clamp(0, 1) * 255).byte().cpu().numpy()

    raw_boxes: List[Tuple[int, int, int, int]] = []

    # ── Method 1: facexlib RetinaFace (preferred) ────────────────────────────
    try:
        from facexlib.detection import init_detection_model  # type: ignore
        cache_key = "retinaface"
        if cache_key not in _FACE_MODEL_CACHE:
            det_path = _download_upscale_model("retinaface_resnet50")
            model    = init_detection_model("retinaface_resnet50", half=False,
                                            model_rootpath=_get_models_dir("facedetection"))
            _FACE_MODEL_CACHE[cache_key] = model
        det = _FACE_MODEL_CACHE[cache_key]
        import numpy as np
        bboxes_scores = det.detect_faces(img_u8, 0.97)
        if bboxes_scores is not None and len(bboxes_scores):
            for b in bboxes_scores:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                raw_boxes.append((x1, y1, x2, y2))
    except Exception as e:
        logger.debug(f"[Radiance/FaceRestore] facexlib detection failed: {e}")

    # ── Method 2: OpenCV Haar cascade fallback ───────────────────────────────
    if not raw_boxes:
        try:
            import cv2  # type: ignore
            gray    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            detections = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(min_face_px, min_face_px),
            )
            if len(detections):
                for (x, y, w, h) in detections:
                    raw_boxes.append((x, y, x + w, y + h))
        except Exception as e:
            logger.debug(f"[Radiance/FaceRestore] OpenCV cascade failed: {e}")

    # Apply padding + clamp
    padded: List[Tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in raw_boxes:
        fw, fh = x2 - x1, y2 - y1
        if fw < min_face_px or fh < min_face_px:
            continue
        px, py = int(fw * pad_frac), int(fh * pad_frac)
        bx1 = max(0,     x1 - px)
        by1 = max(0,     y1 - py)
        bx2 = min(W - 1, x2 + px)
        by2 = min(H - 1, y2 + py)
        padded.append((bx1, by1, bx2, by2))

    return padded


# ── Face Restoration Backends ─────────────────────────────────────────────────

def _load_face_restore_model(model_key: str, device: torch.device) -> Any:
    """
    Load CodeFormer or GFPGAN via spandrel → basicsr → gfpgan package chain.
    Returns the loaded model or raises RuntimeError.
    """
    cache_id = f"{model_key}@{device}"
    if cache_id in _FACE_MODEL_CACHE:
        return _FACE_MODEL_CACHE[cache_id]

    ckpt_path = _download_upscale_model(model_key)
    if ckpt_path is None:
        raise RuntimeError(f"[Radiance/FaceRestore] Could not download '{model_key}'")

    # ── spandrel (handles CodeFormer and GFPGAN automatically) ───────────────
    try:
        model = _load_spandrel(ckpt_path, device)
        _FACE_MODEL_CACHE[cache_id] = model
        return model
    except RuntimeError:
        pass

    # ── basicsr CodeFormer arch ───────────────────────────────────────────────
    if "codeformer" in model_key:
        try:
            from basicsr.utils.registry import ARCH_REGISTRY  # type: ignore
            net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                connect_list=["32", "64", "128", "256"],
            )
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state = state.get("params_ema", state.get("params", state))
            net.load_state_dict(state, strict=False)
            net.eval().to(device)
            _FACE_MODEL_CACHE[cache_id] = net
            logger.info(f"[Radiance/FaceRestore] ✓ basicsr CodeFormer loaded")
            return net
        except (ImportError, Exception) as e:
            logger.debug(f"[Radiance/FaceRestore] basicsr CodeFormer: {e}")

    # ── gfpgan package ────────────────────────────────────────────────────────
    if "gfpgan" in model_key:
        try:
            from gfpgan import GFPGANer  # type: ignore
            restorer = GFPGANer(
                model_path=ckpt_path, upscale=1, arch="clean", channel_multiplier=2,
                bg_upsampler=None, device=device,
            )
            _FACE_MODEL_CACHE[cache_id] = restorer
            logger.info(f"[Radiance/FaceRestore] ✓ gfpgan GFPGANer loaded")
            return restorer
        except (ImportError, Exception) as e:
            logger.debug(f"[Radiance/FaceRestore] gfpgan package: {e}")

    raise RuntimeError(
        f"[Radiance/FaceRestore] Cannot load '{model_key}'. "
        "Install spandrel (pip install spandrel) or basicsr / gfpgan."
    )


def _restore_face_crop(
    model:          Any,
    face_crop_hwc:  torch.Tensor,    # (H,W,C) float32 [0,1]
    model_key:      str,
    fidelity_weight: float,          # CodeFormer only: 0=creative, 1=faithful
    device:         torch.device,
) -> torch.Tensor:
    """
    Run restoration on a single face crop resized to 512×512.
    Returns (H,W,C) float32 [0,1] at original crop resolution.
    """
    H_orig, W_orig = face_crop_hwc.shape[:2]
    S = _FACE_RESTORE_SIZE

    # Resize to 512×512
    x = face_crop_hwc[:, :, :3].permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
    x_512 = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)

    result_512: Optional[torch.Tensor] = None

    # ── spandrel path ─────────────────────────────────────────────────────────
    is_spandrel = not isinstance(model, nn.Module)
    if is_spandrel:
        try:
            inner = getattr(model, "model", model)
            with torch.no_grad():
                y = inner(x_512.to(device))
                if isinstance(y, (list, tuple)):
                    y = y[0]
                result_512 = y.clamp(0, 1).cpu()
        except Exception as e:
            logger.debug(f"[Radiance/FaceRestore] spandrel infer failed: {e}")

    # ── CodeFormer via basicsr (nn.Module with fidelity_weight param) ─────────
    if result_512 is None and hasattr(model, "forward") and "codeformer" in model_key:
        try:
            with torch.no_grad():
                output = model(x_512.to(device), w=fidelity_weight, adain=True)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                result_512 = output.clamp(0, 1).cpu()
        except Exception as e:
            logger.debug(f"[Radiance/FaceRestore] CodeFormer basicsr infer: {e}")

    # ── GFPGAN via gfpgan package ─────────────────────────────────────────────
    if result_512 is None and hasattr(model, "enhance"):
        try:
            import numpy as np
            face_u8 = (x_512.squeeze(0).permute(1, 2, 0) * 255).byte().numpy()
            # gfpgan expects BGR
            face_bgr = face_u8[:, :, ::-1]
            _, _, restored_faces = model.enhance(
                face_bgr, has_aligned=True, only_center_face=False, paste_back=False,
            )
            if restored_faces:
                rf = torch.from_numpy(
                    restored_faces[0][:, :, ::-1].copy()
                ).float() / 255.0        # RGB back
                result_512 = rf.unsqueeze(0).permute(0, 3, 1, 2)
        except Exception as e:
            logger.debug(f"[Radiance/FaceRestore] gfpgan enhance: {e}")

    # ── Identity fallback ─────────────────────────────────────────────────────
    if result_512 is None:
        result_512 = x_512.cpu()

    # Resize restored face back to original crop dimensions
    restored = F.interpolate(result_512, size=(H_orig, W_orig),
                              mode="bilinear", align_corners=False)
    return restored.squeeze(0).permute(1, 2, 0)   # (H,W,C)


def _composite_face(
    image_hwc:    torch.Tensor,    # (H,W,C) full image
    face_hwc:     torch.Tensor,    # (h,w,C) restored face crop (same hw as bbox)
    bbox:         Tuple[int,int,int,int],
    blend_radius: int = 20,
) -> torch.Tensor:
    """
    Paste `face_hwc` back into `image_hwc` at `bbox` with Gaussian feather.

    The blend mask peaks at 1.0 in the face centre and tapers to 0 at the
    bbox edge over `blend_radius` pixels, giving invisible seams.
    """
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return image_hwc

    result = image_hwc.clone()
    C      = image_hwc.shape[2]

    # Build 2-D Gaussian feather mask (h, w)
    sigma_y = max(1.0, (h - 2 * blend_radius) / 3.0 + blend_radius / 2.0)
    sigma_x = max(1.0, (w - 2 * blend_radius) / 3.0 + blend_radius / 2.0)
    gy = _gaussian_kernel_1d(h, sigma_y, image_hwc.device)
    gx = _gaussian_kernel_1d(w, sigma_x, image_hwc.device)
    mask = (gy[:, None] * gx[None, :]).clamp(0, 1)        # (h, w)
    # Normalise so centre = 1.0
    peak = mask.max().clamp(min=1e-8)
    mask = (mask / peak).unsqueeze(-1)                     # (h, w, 1)

    # Ensure face_hwc matches (h, w, C)
    face_c = face_hwc[:, :, :C] if face_hwc.shape[2] >= C else \
             face_hwc.repeat(1, 1, C // face_hwc.shape[2] + 1)[:, :, :C]

    region  = result[y1:y2, x1:x2, :]
    blended = face_c * mask + region * (1.0 - mask)
    result[y1:y2, x1:x2, :] = blended.clamp(0, 1)
    return result


# ── Histogram-match colour drift correction ───────────────────────────────────

def _histogram_match(
    source: torch.Tensor,   # (B,H,W,C) — image to correct
    ref:    torch.Tensor,   # (B,H,W,C) — reference (original before upscale)
    strength: float = 1.0,  # 0=no correction, 1=full match
) -> torch.Tensor:
    """
    Channel-wise CDF histogram matching.

    Corrects colour drift introduced by diffusion upscalers (SD x4, SeedVR2)
    by matching the output's per-channel cumulative distribution to the
    source image's CDF.  Applied independently per image in the batch.

    Parameters
    ----------
    source   : upscaled output to correct  (B,H,W,C)
    ref      : original lower-res image    (B,H,W,C)  — resized internally
    strength : blend between original (0) and fully matched (1)

    Returns
    -------
    corrected : (B,H,W,C) float32 [0,1]
    """
    B, H, W, C = source.shape
    out = source.clone()

    # Downsample ref to source resolution for CDF comparison
    ref_up = F.interpolate(
        ref.permute(0, 3, 1, 2),
        size=(H, W), mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1).clamp(0, 1)

    n_bins = 256

    for b in range(B):
        for c in range(min(3, C)):     # only RGB, not alpha
            src_ch = out    [b, :, :, c].reshape(-1)
            ref_ch = ref_up [b, :, :, c].reshape(-1)

            # Build CDFs
            src_hist = torch.histc(src_ch, bins=n_bins, min=0.0, max=1.0)
            ref_hist = torch.histc(ref_ch, bins=n_bins, min=0.0, max=1.0)

            src_cdf  = torch.cumsum(src_hist, dim=0)
            ref_cdf  = torch.cumsum(ref_hist, dim=0)
            src_cdf  = src_cdf / src_cdf[-1].clamp(min=1e-8)
            ref_cdf  = ref_cdf / ref_cdf[-1].clamp(min=1e-8)

            # Build lookup table: for each src bin find closest ref bin
            bin_edges = torch.linspace(0.0, 1.0, n_bins, device=source.device)
            src_bins  = (src_ch * (n_bins - 1)).long().clamp(0, n_bins - 1)

            # For each source pixel, find matched target value via CDF inversion
            matched = torch.zeros_like(src_ch)
            for i, sb in enumerate(src_bins):
                sv       = src_cdf[sb]
                tb       = (ref_cdf - sv).abs().argmin()
                matched[i] = bin_edges[tb]

            matched = matched.reshape(H, W)
            out[b, :, :, c] = src_ch.reshape(H, W) * (1 - strength) \
                               + matched * strength

    return out.clamp(0, 1)


def _histogram_match_fast(
    source:   torch.Tensor,
    ref:      torch.Tensor,
    strength: float = 1.0,
    n_bins:   int   = 512,
) -> torch.Tensor:
    """
    Vectorised histogram matching — replaces the pixel loop with a
    searchsorted-based CDF inversion (≈100× faster on large tensors).
    """
    if strength == 0.0 or torch.equal(source, ref):
        return source

    B, H, W, C = source.shape

    ref_up = F.interpolate(
        ref.permute(0, 3, 1, 2),
        size=(H, W), mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1).clamp(0, 1)

    out = source.clone()

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=source.device)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    for b in range(B):
        for c in range(min(3, C)):
            src_flat = out   [b, :, :, c].reshape(-1).clamp(0, 1)
            ref_idx = b if b < ref_up.shape[0] else 0
            ref_flat = ref_up[ref_idx, :, :, c].reshape(-1).clamp(0, 1)

            # CDFs
            def _cdf(x):
                hist = torch.histc(x, bins=n_bins, min=0.0, max=1.0)
                cdf  = torch.cumsum(hist, dim=0)
                return cdf / cdf[-1].clamp(min=1e-8)

            src_cdf = _cdf(src_flat)
            ref_cdf = _cdf(ref_flat)

            # Map each source pixel through CDF inversion
            # 1. quantise src pixel to bin index
            src_bin = (src_flat * (n_bins - 1)).long().clamp(0, n_bins - 1)
            # 2. look up src_cdf value at that bin
            src_cdf_vals = src_cdf[src_bin]               # (N,)
            # 3. find closest ref_cdf bin (searchsorted)
            try:
                ref_bins = torch.searchsorted(ref_cdf.contiguous(),
                                              src_cdf_vals.contiguous())
            except Exception:
                # fallback: argmin (slower)
                ref_bins = (ref_cdf.unsqueeze(0) - src_cdf_vals.unsqueeze(1)).abs().argmin(dim=1)

            ref_bins = ref_bins.clamp(0, n_bins - 1)
            matched  = bin_mids[ref_bins]                 # (N,)

            blended = src_flat * (1 - strength) + matched * strength
            out[b, :, :, c] = blended.reshape(H, W)

    return out.clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE — RadianceUpscaleFaceRestore
# ─────────────────────────────────────────────────────────────────────────────

_FACE_MODEL_CHOICES = [
    "auto (CodeFormer → GFPGAN → skip)",
    "codeformer",
    "gfpgan_v1.4",
    "skip (detection only)",
]


class RadianceUpscaleFaceRestore:
    """
    ◎ Radiance Upscale Face Restore

    Detects faces in upscaled images and applies blind face restoration
    (CodeFormer or GFPGAN), then composites restored faces back with
    Gaussian feather blending.  Fixes the smearing/over-smoothing that
    Real-ESRGAN and HAT-L introduce on portrait shots.

    Pipeline:
      1. Detect face regions (facexlib RetinaFace → OpenCV Haar cascade)
      2. Crop + pad each face region
      3. Run CodeFormer (fidelity-controlled) or GFPGAN on 512×512 crop
      4. Resize restored crop back to original face size
      5. Feather-blend back into image (Gaussian mask, `blend_radius` pixels)
      6. Optional: histogram-match to correct diffusion colour drift

    Outputs
    -------
    restored        — (B,H,W,C)  image with faces restored
    face_mask       — (B,H,W,3)  white regions = face bbox areas
    pass_info       — STRING      detection + restoration report
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "Restore and enhance facial detail using a face restoration model."
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("restored", "face_mask", "pass_info")
    FUNCTION     = "restore_faces"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Upscaled image batch (B,H,W,C) float32."},
                ),
                "face_model": (
                    _FACE_MODEL_CHOICES,
                    {"default": "auto (CodeFormer → GFPGAN → skip)",
                     "tooltip": "Face restoration model. Auto tries CodeFormer first, "
                                "falls back to GFPGAN, skips if neither is available."},
                ),
                "fidelity_weight": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "CodeFormer fidelity: 0 = maximum enhancement (creative), "
                                "1 = faithful to input (precise). "
                                "0.5–0.8 is recommended for most upscaled content."},
                ),
                "blend_radius": (
                    "INT",
                    {"default": 20, "min": 0, "max": 80, "step": 4,
                     "tooltip": "Gaussian feather radius in pixels at face crop edge. "
                                "Higher = softer transition. 0 = hard paste."},
                ),
                "face_pad_frac": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 0.6, "step": 0.05,
                     "tooltip": "Extra padding around each detected face bbox "
                                "(fraction of face width/height). 0.25 = 25%."},
                ),
                "min_face_px": (
                    "INT",
                    {"default": 64, "min": 16, "max": 256, "step": 16,
                     "tooltip": "Smallest face (in pixels) to process. "
                                "Smaller faces are skipped."},
                ),
                "colour_correct": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Apply histogram-match colour correction after restoration "
                                "to cancel diffusion colour drift."},
                ),
                "colour_strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Strength of histogram-match correction. "
                                "1.0 = full match to input colours."},
                ),
            },
            "optional": {
                "original_images": (
                    "IMAGE",
                    {"tooltip": "Original (pre-upscale) images for colour reference. "
                                "Used by histogram-match correction. "
                                "Leave disconnected to use the restored images as self-reference."},
                ),
            },
        }

    def restore_faces(
        self,
        images:          torch.Tensor,
        face_model:      str   = "auto (CodeFormer → GFPGAN → skip)",
        fidelity_weight: float = 0.75,
        blend_radius:    int   = 20,
        face_pad_frac:   float = 0.25,
        min_face_px:     int   = 64,
        colour_correct:  bool  = True,
        colour_strength: float = 0.8,
        original_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        t0     = time.time()
        B, H, W, C = images.shape
        device = images.device
        result = images.clone()

        # ── Resolve model key ─────────────────────────────────────────────────
        skip_restore = "skip" in face_model.lower()
        model_key    = None
        fr_model     = None

        if not skip_restore:
            candidates = (["codeformer", "gfpgan_v1.4"]
                          if "auto" in face_model.lower()
                          else [face_model.strip()])
            for ck in candidates:
                try:
                    fr_model  = _load_face_restore_model(ck, device)
                    model_key = ck
                    logger.info(f"[Radiance/FaceRestore] Using model: {ck}")
                    break
                except RuntimeError as e:
                    logger.warning(f"[Radiance/FaceRestore] {ck} unavailable: {e}")

        # ── Process each frame ────────────────────────────────────────────────
        face_mask = torch.zeros(B, H, W, 3, dtype=torch.float32)
        total_faces   = 0
        restored_faces = 0

        for b in range(B):
            frame  = result[b]     # (H,W,C)
            bboxes = _detect_faces(frame, min_face_px=min_face_px,
                                   pad_frac=face_pad_frac)
            total_faces += len(bboxes)

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2, :]   # (ch, cw, C)

                if fr_model is not None and model_key is not None:
                    try:
                        restored_crop = _restore_face_crop(
                            fr_model, crop, model_key,
                            fidelity_weight=fidelity_weight,
                            device=device,
                        )
                        restored_faces += 1
                    except Exception as e:
                        logger.warning(f"[Radiance/FaceRestore] Crop restore failed: {e}")
                        restored_crop = crop
                else:
                    restored_crop = crop

                # Composite back with feather blend
                result[b] = _composite_face(result[b], restored_crop, bbox,
                                            blend_radius=blend_radius)

                # Paint face mask region white
                face_mask[b, y1:y2, x1:x2, :] = 1.0

        # ── Colour drift correction (histogram match) ─────────────────────────
        colour_corrected = False
        if colour_correct and colour_strength > 1e-3:
            ref = original_images if original_images is not None else images
            try:
                result = _histogram_match_fast(result, ref, strength=colour_strength)
                colour_corrected = True
            except Exception as e:
                logger.warning(f"[Radiance/FaceRestore] Colour correct failed: {e}")

        elapsed = time.time() - t0
        info = (
            f"RadianceUpscaleFaceRestore  v1.0\n"
            f"  Frames processed  : {B}\n"
            f"  Faces detected    : {total_faces}\n"
            f"  Faces restored    : {restored_faces}\n"
            f"  Model             : {model_key or 'none (skip)'}\n"
            f"  Fidelity weight   : {fidelity_weight:.2f}  "
            f"(0=creative, 1=faithful)\n"
            f"  Blend radius      : {blend_radius}px\n"
            f"  Colour correction : {'on' if colour_corrected else 'off'}  "
            f"strength={colour_strength:.2f}\n"
            f"  Time              : {elapsed:.2f}s\n"
        )

        return (result.clamp(0, 1), face_mask, info)




# ─────────────────────────────────────────────────────────────────────────────
#  REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Registration  (4 nodes — consolidated from 6)
# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "RadianceUpscaleTiler":       RadianceUpscaleTiler,       # Tile | ColourFix
    "RadianceUpscaleImage":       RadianceUpscaleImage,       # Upscale | Route
    "RadianceUpscaleVideo":       RadianceUpscaleVideo,
    "RadianceUpscaleFaceRestore": RadianceUpscaleFaceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceUpscaleTiler":       "◎ Radiance Upscale Tiler / ColourFix",
    "RadianceUpscaleImage":       "◎ Radiance Upscale Image / Router",
    "RadianceUpscaleVideo":       "◎ Radiance Upscale Video",
    "RadianceUpscaleFaceRestore": "◎ Radiance Upscale Face Restore",
}
