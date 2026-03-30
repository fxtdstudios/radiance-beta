import torch
import torch.nn.functional as F
import json
import math
import logging
import struct
import zlib
import os
import uuid
from typing import Tuple, Dict, Any, Optional

import comfy.model_management
import comfy.utils
import folder_paths

logger = logging.getLogger("Radiance")

# Local imports
from .utils import tensor_srgb_to_linear, tensor_linear_to_srgb

# Log curve imports (optional)
_HAS_LOG_CURVES = False
try:
    from ..color_utils import (
        tensor_logc4_to_linear,
        tensor_linear_to_logc4,
        tensor_slog3_to_linear,
        tensor_linear_to_slog3,
        tensor_vlog_to_linear,
        tensor_linear_to_vlog,
        tensor_davinci_intermediate_to_linear,
        tensor_linear_to_davinci_intermediate,
        tensor_log3g10_to_linear,
        tensor_linear_to_log3g10,
    )

    # ARRI LogC3 — Alexa Classic / Mini / SXT / LF (LogC3 mode).
    # Imported separately so absence does not disable the other log curves.
    _HAS_LOGC3 = False
    try:
        from ..color_utils import tensor_logc3_to_linear, tensor_linear_to_logc3
        _HAS_LOGC3 = True
    except ImportError:
        # Provide passthrough stubs so the converter dicts always have a value;
        # the warning below fires once at encode/decode time when actually used.
        def tensor_logc3_to_linear(t):  # noqa: E301
            logger.warning(
                "[Radiance VAE 4K] tensor_logc3_to_linear not found in color_utils — "
                "ARRI LogC3 input will NOT be linearized. Add LogC3 to color_utils.py."
            )
            return t

        def tensor_linear_to_logc3(t):  # noqa: E301
            logger.warning(
                "[Radiance VAE 4K] tensor_linear_to_logc3 not found in color_utils — "
                "ARRI LogC3 output will be LINEAR data. Add LogC3 to color_utils.py."
            )
            return t

    _HAS_LOG_CURVES = True
except ImportError:
    # FIX-1: Emit a visible warning at module load time so operators immediately
    # know that log encode/decode will NOT work. Previously this was silent and
    # log-space nodes would appear to succeed while producing wrong output.
    logger.warning(
        "[Radiance VAE 4K] color_utils not found — log curve encode/decode is DISABLED. "
        "Log input spaces (LogC3, LogC4, S-Log3, V-Log, DaVinci Intermediate, Log3G10) will pass "
        "through without linearization, and log output spaces will emit linear data. "
        "Install color_utils.py alongside this package to enable log support."
    )
    _HAS_LOGC3 = False

    # Passthrough stubs so converter dicts are always populated.
    def tensor_logc3_to_linear(t): return t  # noqa: E704
    def tensor_linear_to_logc3(t): return t  # noqa: E704


# ═══════════════════════════════════════════════════════════════════════════════
#                    v2.0 CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Feature 2: Model-architecture → latent downscale factor
VAE_FACTOR_DEFAULT = 8
VAE_FACTOR_MAP: Dict[str, int] = {
    # Video models that may use non-8 factors
    "StableCascadeStageC": 42,
    "StableCascadeStageB": 4,
    # Most modern image models: 8
}

# Feature 1: channel count → format label
LATENT_FORMAT_MAP: Dict[int, str] = {
    4:  "sd_4ch",     # SD1.x, SD2.x
    8:  "sd3_8ch",    # SD3 medium (8-ch)
    16: "flux_16ch",  # Flux, SD3 large, WAN, LTX-V
    32: "cascade_32ch",
}

# Feature 4: latent distribution sampling modes
LATENT_SAMPLING_MODES = ["sample", "mean", "mode"]

# Feature 7: Extended color space lists
EXTENDED_SOURCE_SPACES = [
    "Linear",
    "ACEScg",
    "ACES 2065-1",
    "Rec.2020 Linear",
    "sRGB",
    "Raw",
    "ARRI LogC3",   # Alexa Classic / Mini / SXT / LF (LogC3 mode)
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]

EXTENDED_TARGET_SPACES = [
    "Linear",
    "ACEScg",
    "ACES 2065-1",
    "Rec.2020 Linear",
    "sRGB",
    "Raw",
    "ARRI LogC3",   # Alexa Classic / Mini / SXT / LF (LogC3 mode)
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]

EXTENDED_LOG_SPACES = [
    "ARRI LogC3",   # Alexa Classic / Mini / SXT / LF (LogC3 mode)
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]

# ─────────────────────────────────────────────────────────────────────────────
# V-1 FIX: Precomputed color space matrices (module-level constants)
# Previously allocated as new torch.tensor() on EVERY call, causing 6+
# allocations per frame for video. Now created once, moved to device at use.
# ─────────────────────────────────────────────────────────────────────────────
# Encode: working space → Rec.709 linear
_AP1_TO_REC709 = torch.tensor([
    [1.7050509, -0.6217921, -0.0832588],
    [-0.1302564, 1.1408047, -0.0105483],
    [-0.0240033, -0.1289690, 1.1529723],
], dtype=torch.float32).T

_AP0_TO_REC709 = torch.tensor([
    [ 2.5216494, -1.1368885, -0.3847609],
    [-0.2752136,  1.3697052, -0.0944916],
    [-0.0159027, -0.1478148,  1.1637175],
], dtype=torch.float32).T

_REC2020_TO_REC709 = torch.tensor([
    [ 1.6604910, -0.5876411, -0.0728499],
    [-0.1245505,  1.1328999, -0.0083494],
    [-0.0181508, -0.1005789,  1.1187297],
], dtype=torch.float32).T

# Decode: Rec.709 linear → working space
_REC709_TO_AP1 = torch.tensor([
    [0.613097, 0.339523, 0.047379],
    [0.070194, 0.916354, 0.013452],
    [0.020616, 0.109570, 0.869815],
], dtype=torch.float32).T

_REC709_TO_AP0 = torch.tensor([
    [0.4339316, 0.3762584, 0.1898100],
    [0.0886227, 0.8131989, 0.0981784],
    [0.0177087, 0.1095613, 0.8727300],
], dtype=torch.float32).T

_REC709_TO_REC2020 = torch.tensor([
    [0.6274039, 0.3292830, 0.0433131],
    [0.0690973, 0.9195404, 0.0113623],
    [0.0163914, 0.0880132, 0.8955954],
], dtype=torch.float32).T


# ─────────────────────────────────────────────────────────────────────────────
# v2.2 Helpers: HDR-safe color transforms
# ─────────────────────────────────────────────────────────────────────────────

def _safe_matrix_transform(img: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """Apply a 3×3 color matrix to an image tensor, safe for 4+ channels.

    v2.2 FIX (BUG-48): Previous code did ``img.reshape(-1, 3) @ mat`` which
    crashes with RuntimeError if the image has alpha or other extra channels.
    This helper slices to the first 3 channels, transforms, then recombines.

    Args:
        img: (B, H, W, C) where C ≥ 3.  C > 3 channels pass through untouched.
        mat: (3, 3) color matrix, already transposed for right-multiply.
    """
    if img.shape[-1] == 3:
        shape = img.shape
        return (img.reshape(-1, 3) @ mat).reshape(shape)
    # 4+ channels: split, transform RGB, recombine
    rgb = img[..., :3]
    extra = img[..., 3:]
    shape = rgb.shape
    rgb = (rgb.reshape(-1, 3) @ mat).reshape(shape)
    return torch.cat([rgb, extra], dim=-1)


def _safe_srgb_to_linear_extended(img: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF with correct handling for values outside [0, 1].

    v2.2 FIX (BUG-51): Standard sRGB EOTF is only defined for [0, 1].
    For HDR decode (Soft Clip, Passthrough), recovered values can exceed 1.0.
    This helper:
      • Values in [0, 1]: standard IEC 61966-2-1 EOTF
      • Values > 1.0: continuous power-function extension (no clamp)
      • Values < 0.0: mirrored for symmetry (preserves negatives)

    If the imported ``tensor_srgb_to_linear`` already handles extended range
    correctly (i.e. does NOT clamp to [0, 1]), this wrapper adds negligible
    overhead since the torch.where branches are fused.
    """
    # Decompose: sign-preserving absolute value for the power branch
    abs_img = img.abs()
    sign = img.sign()

    # Standard sRGB EOTF extended to full range
    low_mask = abs_img <= 0.04045
    low = abs_img / 12.92
    high = ((abs_img + 0.055) / 1.055).pow(2.4)
    result = torch.where(low_mask, low, high) * sign

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2: Dynamic VAE scale factor detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_vae_factor(vae: Any) -> int:
    """
    v2.0: Read the spatial downscale factor from the VAE object.
    Falls back to 8 for unknown model classes.

    ComfyUI exposes this as:
      vae.downscale_ratio          (int, most models)
      vae.latent_format.downscale_factor  (some wrappers)
    """
    for attr in ("downscale_ratio", "latent_downscale_factor"):
        val = getattr(vae, attr, None)
        if isinstance(val, int) and val > 0:
            return val

    # Check through latent_format object
    lf = getattr(vae, "latent_format", None)
    if lf is not None:
        for attr in ("downscale_factor", "scale_factor"):
            val = getattr(lf, attr, None)
            if isinstance(val, (int, float)) and val > 0:
                return int(val)

    # Fallback: check class name against known architectures
    cls_name = type(getattr(vae, "first_stage_model", vae)).__name__
    for key, factor in VAE_FACTOR_MAP.items():
        if key in cls_name:
            return factor

    return VAE_FACTOR_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1: Latent format label detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_latent_format(vae: Any) -> str:
    """
    v2.0: Return a format string e.g. 'flux_16ch' based on VAE latent channels.
    Compatible with the Sampler Pro v4.0 latent_format input socket.
    """
    # Try to get channel count from VAE
    channels = None
    model = getattr(vae, "first_stage_model", None)
    if model is not None:
        # Typical attr names across ComfyUI VAE wrappers
        for attr in ("z_channels", "latent_channels", "out_channels"):
            val = getattr(model, attr, None)
            if isinstance(val, int) and val > 0:
                channels = val
                break
        # Try encoder output shape heuristic
        if channels is None:
            enc = getattr(model, "encoder", None)
            if enc is not None:
                for attr in ("z_channels", "out_channels"):
                    val = getattr(enc, attr, None)
                    if isinstance(val, int) and val > 0:
                        channels = val
                        break

    if channels is not None:
        return LATENT_FORMAT_MAP.get(channels, f"unknown_{channels}ch")

    # Fallback: try class name
    cls = type(getattr(vae, "first_stage_model", vae)).__name__.lower()
    if "flux" in cls:
        return "flux_16ch"
    if "cascade" in cls:
        return "cascade_32ch"
    return "sd_4ch"  # Safe default


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5: Encode quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_encode_quality_report(
    pixels_before: torch.Tensor,
    pixels_after: torch.Tensor,
    latent: torch.Tensor,
    vae_factor: int,
    latent_fmt: str,
    source_space: str,
    hdr_mode: str,
    tile_size: int,
    total_tiles: int,
    encode_time_ms: int,
) -> str:
    """Build a quality metrics JSON for the encode node output."""
    n_total = pixels_before.numel()
    n_clipped = (pixels_before > 1.0).sum().item() + (pixels_before < 0.0).sum().item()
    clip_pct = round(n_clipped / max(1, n_total) * 100, 3)
    nan_count = int(torch.isnan(latent).sum().item() + torch.isinf(latent).sum().item())

    b, c_lat, lh, lw = latent.shape
    ph, pw = pixels_before.shape[1:3]

    report = {
        "version": "2.0",
        "input_resolution": f"{pw}x{ph}",
        "latent_resolution": f"{lw}x{lh}",
        "latent_format": latent_fmt,
        "latent_channels": c_lat,
        "vae_factor": vae_factor,
        "source_space": source_space,
        "hdr_mode": hdr_mode,
        "pixel_range": [round(float(pixels_after.min()), 4),
                        round(float(pixels_after.max()), 4)],
        "latent_range": [round(float(latent.min()), 4),
                         round(float(latent.max()), 4)],
        "latent_std": round(float(latent.std()), 4),
        "clipping_pct": clip_pct,
        "nan_inf_count": nan_count,
        "tile_size_px": tile_size,
        "total_tiles": total_tiles,
        "encode_time_ms": encode_time_ms,
    }
    return json.dumps(report, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: VAE posterior sampling modes
# ─────────────────────────────────────────────────────────────────────────────

def _encode_with_sampling_mode(vae: Any, pixels: torch.Tensor, mode: str) -> torch.Tensor:
    """
    v2.2: Encode pixels with explicit distribution sampling control.

    mode = "sample"  — standard (random sample from posterior, default ComfyUI)
    mode = "mean"    — use posterior mean (deterministic, best for img2img)
    mode = "mode"    — use mean (same as mean for Gaussian posterior)

    v2.2 FIX (BUG-54): The mean/mode path previously called ``fsm.encode(pixels)``
    directly, but ``pixels`` is in ComfyUI's BHWC [0,1] format. The raw model's
    encoder expects BCHW [-1,1]. This caused mean/mode to silently produce garbage
    latents (or crash on shape mismatch), while "sample" worked correctly because
    it goes through ``vae.encode()`` which handles the conversion internally.

    Fix: preprocess pixels the same way ComfyUI does before calling the raw model.
    """
    if mode == "sample":
        result = vae.encode(pixels)
        if isinstance(result, dict):
            return result["samples"]
        return result

    # ── mean / mode path ──────────────────────────────────────────────

    # Strategy 1: Try ComfyUI's own encode_raw / encode_to_distribution
    # Some ComfyUI versions expose these directly.
    for method_name in ("encode_raw", "encode_to_distribution"):
        fn = getattr(vae, method_name, None)
        if callable(fn):
            try:
                posterior = fn(pixels)
                if hasattr(posterior, "mean"):
                    return posterior.mean
                if hasattr(posterior, "mode"):
                    return posterior.mode()
                if isinstance(posterior, torch.Tensor):
                    return posterior
            except Exception:
                pass

    # Strategy 2: Access the raw first_stage_model with correct preprocessing.
    # ComfyUI's vae.encode() internally does:
    #   1. BHWC → BCHW  (movedim or permute)
    #   2. [0,1] → [-1,1] scaling
    #   3. Device placement
    #   4. fsm.encode() → posterior
    #   5. posterior.sample()
    # We replicate steps 1-4 and then extract .mean instead of .sample().
    try:
        fsm = vae.first_stage_model
        device = next(fsm.parameters()).device

        # Preprocess: BHWC [0,1] → BCHW [-1,1]
        x = pixels.movedim(-1, 1).to(device=device, dtype=torch.float32)
        x = x * 2.0 - 1.0

        posterior = fsm.encode(x)

        # Handle different return types across model architectures
        if hasattr(posterior, "latent_dist"):
            # Diffusers-style: returns an object with .latent_dist
            posterior = posterior.latent_dist

        if hasattr(posterior, "mean"):
            latent = posterior.mean
        elif hasattr(posterior, "mode"):
            latent = posterior.mode() if callable(posterior.mode) else posterior.mode
        elif isinstance(posterior, torch.Tensor):
            # Some encoders return the latent directly (no distribution)
            latent = posterior
        else:
            raise TypeError(f"Unknown posterior type: {type(posterior)}")

        # Apply scaling factor if the VAE uses one (SD models use 0.18215, etc.)
        scale_factor = getattr(vae, "scale_factor", None)
        if scale_factor is None:
            # Try to find it on the config
            config = getattr(vae, "config", None)
            if config is not None:
                scale_factor = getattr(config, "scaling_factor", None)
        if scale_factor is not None and isinstance(scale_factor, (int, float)):
            latent = latent * scale_factor

        return latent

    except Exception as e:
        logger.warning(
            f"[Radiance 4K Encode v2.2] mean/mode encode failed ({e}), "
            f"falling back to standard sample mode."
        )

    # Strategy 3: Fall back to standard sample (always works)
    result = vae.encode(pixels)
    if isinstance(result, dict):
        return result["samples"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#                         TILING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════



class TileEngine:
    """
    Production-grade tiling engine with cosine blend weights.
    Handles arbitrary image sizes, pad-to-multiple, and VRAM-aware sizing.
    """

    @staticmethod
    def get_optimal_tile_size(
        image_h: int,
        image_w: int,
        min_tile: int = 512,
        max_tile: int = 1536,
        vram_budget_gb: float = None,
    ) -> int:
        """
        Determine optimal tile size based on image dimensions and available VRAM.

        Rules:
        - If image fits in a single tile (≤max_tile), use full image
        - Otherwise, pick largest tile that fits in VRAM budget
        - Always returns multiple of 8 (VAE requirement)
        """
        # If image fits in one tile, no tiling needed
        if image_h <= max_tile and image_w <= max_tile:
            # Round up to multiple of 8
            tile = max(image_h, image_w)
            tile = ((tile + 7) // 8) * 8
            return min(tile, max_tile)

        # Query available VRAM
        if vram_budget_gb is None:
            try:
                device = comfy.model_management.get_torch_device()
                if device.type == "cuda":
                    free_mem, total_mem = torch.cuda.mem_get_info(device)
                    # Use 60% of free VRAM for safety (VAE + intermediate buffers)
                    vram_budget_gb = (free_mem * 0.6) / (1024**3)
                else:
                    vram_budget_gb = 4.0  # Conservative default for CPU
            except Exception:
                vram_budget_gb = 4.0

        # Estimate VRAM per tile:
        # Encode: tile_pixels(fp32) + tile_latent(fp32) + VAE weights + intermediates
        # Rough formula: ~20 bytes per pixel for VAE encode/decode
        bytes_per_pixel = 20
        max_pixels = int(vram_budget_gb * (1024**3) / bytes_per_pixel)
        max_side = int(math.sqrt(max_pixels))

        # Clamp to range and round down to multiple of 8
        tile = max(min_tile, min(max_side, max_tile))
        tile = (tile // 8) * 8

        logger.info(
            f"[Radiance 4K] VRAM budget: {vram_budget_gb:.1f}GB → "
            f"tile size: {tile}px (image: {image_w}×{image_h})"
        )
        return tile

    @staticmethod
    def compute_tiles(total: int, tile_size: int, overlap: int) -> list:
        """
        Compute tile start positions along one axis.
        Returns list of (start, end) tuples.
        Ensures full coverage with consistent overlap.
        """
        if total <= tile_size:
            return [(0, total)]

        stride = tile_size - overlap
        positions = []
        pos = 0
        while pos < total:
            end = min(pos + tile_size, total)
            start = max(0, end - tile_size)
            positions.append((start, end))
            if end >= total:
                break
            pos += stride

        return positions

    @staticmethod
    def make_cosine_blend_weight_2d(
        tile_h: int,
        tile_w: int,
        overlap_top: int,
        overlap_bottom: int,
        overlap_left: int,
        overlap_right: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Create 2D cosine blend weight mask for a tile.

        Cosine blending eliminates visible seams — the weight transitions
        smoothly from 0→1 at borders, following a raised cosine curve.
        This is the standard approach used in DJV, Nuke, and Flame for
        tiled compositing.

        Returns: (1, tile_h, tile_w, 1) weight tensor
        """
        weight = torch.ones(tile_h, tile_w, device=device)

        # Cosine ramp: 0 → 1 over `n` pixels
        def cosine_ramp(n):
            if n <= 0:
                return torch.ones(0, device=device)
            t = torch.linspace(0, math.pi / 2, n, device=device)
            return torch.sin(t) ** 2  # Raised cosine (power-of-sine window)

        # Top edge
        if overlap_top > 0:
            ramp = cosine_ramp(overlap_top)
            weight[:overlap_top, :] *= ramp.unsqueeze(1)

        # Bottom edge
        if overlap_bottom > 0:
            ramp = cosine_ramp(overlap_bottom).flip(0)
            weight[-overlap_bottom:, :] *= ramp.unsqueeze(1)

        # Left edge
        if overlap_left > 0:
            ramp = cosine_ramp(overlap_left)
            weight[:, :overlap_left] *= ramp.unsqueeze(0)

        # Right edge
        if overlap_right > 0:
            ramp = cosine_ramp(overlap_right).flip(0)
            weight[:, -overlap_right:] *= ramp.unsqueeze(0)

        return weight.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

    @staticmethod
    def pad_to_multiple(
        tensor: torch.Tensor, multiple: int = 8, mode: str = "reflect"
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Pad image tensor to nearest multiple of `multiple`.
        Input: (B, H, W, C) — ComfyUI IMAGE format
        Returns: padded tensor, (pad_h, pad_w) for later cropping
        """
        b, h, w, c = tensor.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0)

        # F.pad expects (N, C, H, W) and padding as (left, right, top, bottom)
        t = tensor.permute(0, 3, 1, 2)  # → (B, C, H, W)
        t = F.pad(t, (0, pad_w, 0, pad_h), mode=mode)
        return t.permute(0, 2, 3, 1), (pad_h, pad_w)  # → (B, H+padH, W+padW, C)


# ═══════════════════════════════════════════════════════════════════════════════
#                         4K VAE ENCODE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceVAE4KEncode:
    """
    Encode images up to 8K+ to VAE latents using production-grade tiling.

    Features:
    - VRAM-aware auto tile sizing (no OOM)
    - Cosine blend weights (invisible seams)
    - Pad-to-8 for VAE compatibility
    - Color space aware (Linear, sRGB, ACEScg, Log formats)
    - Alpha channel preservation
    - Progress reporting
    """

    SOURCE_SPACES = EXTENDED_SOURCE_SPACES
    LOG_SPACES = EXTENDED_LOG_SPACES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "source_space": (
                    cls.SOURCE_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": "Input color space. Auto-linearized before VAE encode.",
                    },
                ),
            },
            "optional": {
                "tile_size": (
                    ["Auto", "512", "768", "1024", "1280", "1536"],
                    {
                        "default": "Auto",
                        "tooltip": "Tile size in pixels. Auto queries VRAM.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 128, "min": 32, "max": 256, "step": 16,
                        "tooltip": "Overlap between tiles in pixels. 128px optimal for cosine blending.",
                    },
                ),
                "exposure": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                        "tooltip": "Exposure adjustment in stops (linear space).",
                    },
                ),
                "alpha_handling": (
                    ["Preserve", "Ignore"],
                    {"default": "Preserve",
                     "tooltip": "Preserve alpha channel for roundtrip."},
                ),
                "hdr_mode": (
                    ["Clip (SDR)", "Soft Clip", "Compress (Log)", "Passthrough"],
                    {
                        "default": "Soft Clip",
                        "tooltip": "HDR handling before VAE encode.",
                    },
                ),
                # v2.0 Feature 4: Latent sampling mode
                "latent_sampling": (
                    LATENT_SAMPLING_MODES,
                    {
                        "default": "sample",
                        "tooltip": (
                            "How to sample from the VAE posterior distribution. "
                            "'mean' gives deterministic, lowest-noise results for img2img. "
                            "'sample' gives diverse outputs (default ComfyUI behaviour)."
                        ),
                    },
                ),
                # v2.0 Feature 8: Tile processing mode
                "processing_mode": (
                    ["sequential", "batched"],
                    {
                        "default": "sequential",
                        "tooltip": (
                            "'sequential' uses minimal VRAM (one tile at a time). "
                            "'batched' groups tiles for 2-4x faster encode on high-VRAM GPUs."
                        ),
                    },
                ),
            },
        }

    # v2.0: 5 outputs — latent_format and quality_report added
    RETURN_TYPES = ("LATENT", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("samples", "alpha", "metadata", "latent_format", "quality_report")
    OUTPUT_TOOLTIPS = (
        "Encoded latent — wire to Sampler Pro or save node.",
        "Alpha channel tensor — wire to Radiance VAE 4K Decode alpha input.",
        "Encode metadata JSON — wire to Decode crop_padding for auto-crop.",
        "Latent format string (e.g. 'flux_16ch') — wire to Sampler Pro latent_format input.",
        "Quality metrics JSON: clipping %, NaN count, latent range, tile info.",
    )
    FUNCTION = "encode"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = (
        "v2.0 — Universal 4K+ VAE encode. Auto VAE-factor detection, "
        "video 5D latent support, 11 color spaces, mean/sample/mode latent sampling, "
        "batched tile processing, quality report, latent_format output."
    )

    # FIX-2: Map each log space to its linear→log converter so Compress (Log)
    # uses the correct curve per source/target space instead of always LogC4.
    _LOG_TO_LINEAR_CONVERTERS = None  # populated lazily after import succeeds
    _LINEAR_TO_LOG_CONVERTERS = None

    @classmethod
    def _get_log_to_linear(cls):
        if not _HAS_LOG_CURVES:
            return {}
        if cls._LOG_TO_LINEAR_CONVERTERS is None:
            cls._LOG_TO_LINEAR_CONVERTERS = {
                "ARRI LogC3": tensor_logc3_to_linear,   # Alexa Classic/Mini/SXT/LF
                "ARRI LogC4": tensor_logc4_to_linear,
                "Sony S-Log3": tensor_slog3_to_linear,
                "Panasonic V-Log": tensor_vlog_to_linear,
                "DaVinci Intermediate": tensor_davinci_intermediate_to_linear,
                "RED Log3G10": tensor_log3g10_to_linear,
            }
        return cls._LOG_TO_LINEAR_CONVERTERS

    @classmethod
    def _get_linear_to_log(cls):
        if not _HAS_LOG_CURVES:
            return {}
        if cls._LINEAR_TO_LOG_CONVERTERS is None:
            cls._LINEAR_TO_LOG_CONVERTERS = {
                "ARRI LogC3": tensor_linear_to_logc3,   # Alexa Classic/Mini/SXT/LF
                "ARRI LogC4": tensor_linear_to_logc4,
                "Sony S-Log3": tensor_linear_to_slog3,
                "Panasonic V-Log": tensor_linear_to_vlog,
                "DaVinci Intermediate": tensor_linear_to_davinci_intermediate,
                "RED Log3G10": tensor_linear_to_log3g10,
            }
        return cls._LINEAR_TO_LOG_CONVERTERS

    def _prepare_for_vae(
        self, img: torch.Tensor, source_space: str, exposure: float, hdr_mode: str
    ) -> torch.Tensor:
        """Full color pipeline: linearize → expose → VAE-space transform.

        v2.2 FIX (BUG-56): "Raw" source space is now an explicit early return.
        Previously "Raw" fell through all branches without matching, then had
        the hdr_mode transform applied (linear_to_srgb + clamp), which made
        "Raw" behave identically to "Linear" — defeating the purpose.
        """

        # Raw: complete passthrough — no linearization, no exposure, no hdr_mode.
        # The data goes to the VAE exactly as the operator provided it.
        if source_space == "Raw":
            return img

        # 1. Linearize log/gamma inputs
        if source_space in self.LOG_SPACES:
            if _HAS_LOG_CURVES:
                img = self._get_log_to_linear()[source_space](img)
            else:
                logger.warning(
                    f"[Radiance 4K Encode] Log curves unavailable — {source_space} "
                    f"will NOT be linearized. Install color_utils.py to fix this."
                )
        elif source_space == "sRGB":
            img = tensor_srgb_to_linear(img)
        elif source_space == "ACEScg":
            # ACEScg → Rec.709 linear (V-1: use precomputed matrix)
            # v2.2 FIX (BUG-48): Safe reshape — only transform first 3 channels.
            # reshape(-1, 3) crashes if img has 4+ channels (alpha from upstream).
            mat = _AP1_TO_REC709.to(img.device)
            img = _safe_matrix_transform(img, mat)
        elif source_space == "ACES 2065-1":
            mat = _AP0_TO_REC709.to(img.device)
            img = _safe_matrix_transform(img, mat)
        elif source_space == "Rec.2020 Linear":
            mat = _REC2020_TO_REC709.to(img.device)
            img = _safe_matrix_transform(img, mat)

        # 2. Exposure (linear domain)
        if exposure != 0.0:
            img = img * (2.0**exposure)

        # 2b. Clamp negatives — but only for modes that need it.
        #
        # v2.2 FIX (BUG-49): Mode-aware negative clamp.
        #   • Compress (Log): Log curves (LogC4, S-Log3, etc.) have well-defined
        #     mappings for scene-linear values below zero (below-black detail,
        #     down to ~−0.01 for LogC4). Clamping at 0.0 destroys this data.
        #     Only clamp at the minimum that the log curve can encode.
        #   • All sRGB-based modes (Soft Clip, Passthrough, Clip SDR):
        #     linear_to_srgb uses pow() which produces NaN for negative inputs.
        #     Must clamp at 0.0 to prevent NaN propagation through the VAE.
        if hdr_mode == "Compress (Log)":
            # Allow small negatives for below-black detail. LogC4 encodes values
            # from approximately −0.015 scene-linear. Clamp well below that so
            # the log curve itself decides the floor, not this pre-clamp.
            img = torch.clamp(img, min=-0.02)
        else:
            img = torch.clamp(img, min=0.0)

        # 3. Transform to VAE input space
        if hdr_mode == "Compress (Log)":
            if _HAS_LOG_CURVES:
                # FIX-2: Use the log curve that matches the source space so the
                # encode and decode curves are symmetric. Previously this was
                # hardcoded to LogC4 regardless of source_space, causing a
                # mismatched encode/decode when shooting S-Log3, V-Log, etc.
                #
                # Preference order:
                #   1. If source_space is itself a log space → use its own curve
                #      (data is already linear after step 1, re-encode to same log)
                #   2. Otherwise default to LogC4 as a neutral choice
                if source_space in self.LOG_SPACES:
                    compress_fn = self._get_linear_to_log()[source_space]
                else:
                    compress_fn = tensor_linear_to_logc4  # neutral default
                img = compress_fn(img)
            else:
                logger.warning(
                    "[Radiance 4K Encode] Compress (Log) HDR mode selected but "
                    "log curves are unavailable — falling back to SDR clamp."
                )
                img = torch.clamp(img, 0.0, 1.0)
        elif hdr_mode == "Passthrough":
            img = tensor_linear_to_srgb(img)
            # v2.2 FIX (BUG-50): Widened from [−0.05, 1.05] → [−0.05, 1.5].
            # Previous ceiling of 1.05 only preserved 1.12 linear (0.17 stops
            # of HDR headroom). New ceiling of 1.5 preserves up to ~2.5 linear
            # (~1.3 stops above SDR white). The VAE may introduce minor
            # reconstruction noise for values above 1.0, but this is a far
            # better trade-off than hard-clipping all HDR content.
            # Values above ~1.5 sRGB push the VAE too far out-of-distribution
            # and produce unacceptable artifacts, so 1.5 is the practical ceiling.
            img = torch.clamp(img, -0.05, 1.5)
        elif hdr_mode == "Soft Clip":
            img = tensor_linear_to_srgb(img)
            # Tanh soft clip at knee=0.85
            # V-5 FIX: Use consistent range constant for encode/decode symmetry.
            knee = 0.85
            rng = 1.0 - knee  # 0.15
            above = img > knee
            if above.any():
                result = img.clone()
                excess = (img[above] - knee) / rng
                result[above] = knee + rng * torch.tanh(excess)
                img = torch.clamp(result, min=0.0)
            else:
                img = torch.clamp(img, 0.0, 1.0)
        else:
            # Clip (SDR)
            img = tensor_linear_to_srgb(img)
            img = torch.clamp(img, 0.0, 1.0)

        return img

    def _tiled_encode(
        self,
        pixels: torch.Tensor,
        vae: Any,
        tile_size: int,
        overlap: int,
        pbar: Any = None,
        latent_sampling: str = "sample",
        processing_mode: str = "sequential",
        vae_factor: int = 8,
    ) -> Dict[str, Any]:
        """
        v2.0: Encode full image in overlapping tiles with cosine blend.

        Args:
            pixels: (B, H, W, 3) float32, already in VAE-space
            vae: ComfyUI VAE model
            tile_size: Tile size in pixels (must be multiple of 8)
            overlap: Overlap in pixels
            pbar: Optional progress bar
            latent_sampling: "sample" / "mean" / "mode"
            processing_mode: "sequential" / "batched"
            vae_factor: Spatial downscale factor (auto-detected, default 8)

        Returns:
            {"samples": (B, C, latH, latW), "radiance_meta": {...}} latent dict
        """
        b, h, w, c = pixels.shape
        device = pixels.device
        scale = vae_factor  # v2.0: no longer hardcoded

        # Compute tile grid
        tiles_y = TileEngine.compute_tiles(h, tile_size, overlap)
        tiles_x = TileEngine.compute_tiles(w, tile_size, overlap)
        total_tiles = len(tiles_y) * len(tiles_x)

        logger.info(
            f"[Radiance 4K Encode] {w}×{h} → {len(tiles_x)}×{len(tiles_y)} "
            f"tiles ({tile_size}px, {overlap}px overlap, {total_tiles} total, "
            f"mode={processing_mode}, factor={scale})"
        )

        lat_c = None
        lat_h = h // scale
        lat_w = w // scale
        lat_overlap = overlap // scale

        output = None
        weight_acc = None

        # Build flat list of all tile coords for batched mode
        all_tiles = [(yi, xi, y1, y2, x1, x2)
                     for yi, (y1, y2) in enumerate(tiles_y)
                     for xi, (x1, x2) in enumerate(tiles_x)]

        tile_idx = 0

        if processing_mode == "batched" and total_tiles > 1:
            # Encode all tiles in one GPU batch (uses more VRAM, much faster)
            tile_list = [pixels[:, y1:y2, x1:x2, :3].contiguous()
                         for (_, _, y1, y2, x1, x2) in all_tiles]
            # Pad tiles to same size (last row/col may be smaller)
            max_th = max(t.shape[1] for t in tile_list)
            max_tw = max(t.shape[2] for t in tile_list)

            padded = []
            for t in tile_list:
                pad_h2 = max_th - t.shape[1]
                pad_w2 = max_tw - t.shape[2]
                if pad_h2 > 0 or pad_w2 > 0:
                    t = F.pad(t.permute(0, 3, 1, 2), (0, pad_w2, 0, pad_h2)).permute(0, 2, 3, 1)
                padded.append(t)

            batch_pixels = torch.cat(padded, dim=0)  # (B*T, H, W, C)
            try:
                # Defensive no_grad: _tiled_encode may be called from a subclass or
                # test harness without the caller's no_grad context. Inference only.
                with torch.no_grad():
                    batch_result = _encode_with_sampling_mode(vae, batch_pixels, latent_sampling)
                if isinstance(batch_result, dict):
                    batch_result = batch_result["samples"]
                batch_result = batch_result.float().cpu()
                lat_c = batch_result.shape[1]
                tile_latents = list(batch_result.chunk(total_tiles, dim=0))
            except Exception as e:
                logger.warning(f"[Radiance] Batched encode failed ({e}), falling back to sequential")
                tile_latents = None
        else:
            tile_latents = None

        for tile_idx_i, (yi, xi, y1, y2, x1, x2) in enumerate(all_tiles):
            if tile_latents is not None:
                # Use pre-computed batch result
                tile_samples = tile_latents[tile_idx_i]
                if tile_samples.ndim == 3:
                    tile_samples = tile_samples.unsqueeze(0)
            else:
                # Sequential: encode one tile at a time
                # Defensive no_grad: idempotent if caller already has it,
                # protective if _tiled_encode is ever invoked standalone.
                tile = pixels[:, y1:y2, x1:x2, :3].contiguous()
                with torch.no_grad():
                    tile_samples = _encode_with_sampling_mode(vae, tile, latent_sampling)
                if isinstance(tile_samples, dict):
                    tile_samples = tile_samples["samples"]
                tile_samples = tile_samples.float().cpu()
                del tile

            # Lazy-init accumulators
            if output is None:
                lat_c = tile_samples.shape[1]
                output = torch.zeros(
                    (b, lat_c, lat_h, lat_w), dtype=torch.float32, device="cpu"
                )
                weight_acc = torch.zeros(
                    (1, 1, lat_h, lat_w), dtype=torch.float32, device="cpu"
                )

            th, tw = tile_samples.shape[2], tile_samples.shape[3]

            # Per-edge overlap flags
            ov_top   = lat_overlap if yi > 0 else 0
            ov_bot   = lat_overlap if yi < len(tiles_y) - 1 else 0
            ov_left  = lat_overlap if xi > 0 else 0
            ov_right = lat_overlap if xi < len(tiles_x) - 1 else 0

            blend_w = TileEngine.make_cosine_blend_weight_2d(
                th, tw, ov_top, ov_bot, ov_left, ov_right, device="cpu"
            )
            blend_w = blend_w.squeeze(-1).unsqueeze(1)

            lx1 = x1 // scale
            ly1 = y1 // scale
            lx2 = lx1 + tw
            ly2 = ly1 + th

            output[:, :, ly1:ly2, lx1:lx2] += tile_samples * blend_w
            weight_acc[:, :, ly1:ly2, lx1:lx2] += blend_w

            tile_idx += 1
            if pbar:
                pbar.update_absolute(tile_idx, total_tiles)

            if tile_latents is None:
                del tile_samples, blend_w

        # V-6 FIX: Call empty_cache once after all tiles, not per-tile.
        # Per-tile empty_cache is a CUDA sync barrier (~1-2ms each) that adds
        # 24-48ms of pure overhead for a 4K image split into 24 tiles.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        weight_acc = torch.clamp(weight_acc, min=1e-3)
        output = output / weight_acc

        return {"samples": output.to(device), "_total_tiles": total_tiles}


    def encode(
        self,
        pixels: torch.Tensor,
        vae: Any,
        source_space: str = "Linear",
        tile_size: str = "Auto",
        overlap: int = 128,
        exposure: float = 0.0,
        alpha_handling: str = "Preserve",
        hdr_mode: str = "Soft Clip",
        latent_sampling: str = "sample",
        processing_mode: str = "sequential",
    ) -> Tuple:
        """v2.0: Universal encode supporting 4D image and 5D video latents."""
        import time as _time
        t_start = _time.time()

        # v2.0 Feature 2: Auto-detect VAE factor and format
        vae_factor = detect_vae_factor(vae)
        latent_fmt = detect_latent_format(vae)

        # v2.0 Feature 3: Handle 5D video latents (B, F, H, W, C)
        is_video = pixels.ndim == 5
        
        # Check if the VAE natively supports 3D latents (e.g., Wan Video, Cosmos)
        is_3d_vae = False
        if hasattr(vae, "latent_dim") and getattr(vae, "latent_dim") == 3:
            is_3d_vae = True
            
        if is_video and not is_3d_vae:
            B, F, H, W, C = pixels.shape
            logger.info(
                f"[Radiance 4K Encode v2.0] Video: {B}×{F}×{W}×{H}, space={source_space}"
            )
            # Encode frame by frame, stack temporal dim
            frame_latents = []
            for fi in range(F):
                frame = pixels[:, fi, ...]  # (B, H, W, C)
                latent_frame, _, _, _, _ = self.encode(
                    frame, vae, source_space, tile_size, overlap, exposure,
                    alpha_handling, hdr_mode, latent_sampling, processing_mode
                )
                frame_latents.append(latent_frame["samples"])
            # Stack: (B, C, F, latH, latW)
            all_frames = torch.stack(frame_latents, dim=2)
            video_latent = {"samples": all_frames, "latent_format": latent_fmt}
            # Build a minimal quality/meta output for video
            total_time_ms = int((_time.time() - t_start) * 1000)
            # Match device of input pixels so downstream nodes don't hit a device
            # mismatch when alpha is moved to GPU alongside the image tensor.
            alpha_out = torch.ones((B, H, W, 1), dtype=torch.float32, device=pixels.device)
            meta = json.dumps({"node": "RadianceVAE4KEncode", "video": True,
                               "frames": F, "latent_format": latent_fmt})
            qr = json.dumps({"version": "2.0", "video": True, "frames": F,
                             "latent_format": latent_fmt, "encode_time_ms": total_time_ms})
            return (video_latent, alpha_out, meta, latent_fmt, qr)

        b, h, w, c = pixels.shape
        logger.info(
            f"[Radiance 4K Encode v2.0] Input: {w}×{h} ({b} frames), "
            f"space={source_space}, sampling={latent_sampling}, vae_factor={vae_factor}"
        )

        # Extract alpha
        has_alpha = c == 4
        if has_alpha and alpha_handling == "Preserve":
            alpha = pixels[..., 3:4].clone().float()
        else:
            alpha = torch.ones((b, h, w, 1), dtype=torch.float32, device=pixels.device)

        target_device = comfy.model_management.get_torch_device()
        img = pixels[..., :3].clone().float().to(target_device)

        # V-3 FIX: Clone to CPU immediately instead of keeping a 50MB+ GPU copy.
        # The original pixels are only needed for post-encode quality metrics,
        # which run on CPU anyway. Keeping them on GPU competes with VAE for VRAM.
        pixels_orig = img.detach().cpu()

        # Color pipeline
        img = self._prepare_for_vae(img, source_space, exposure, hdr_mode)

        # Pad to multiple of vae_factor (minimum 8 for all known VAEs)
        pad_multiple = max(vae_factor, 8)
        img, (pad_h, pad_w) = TileEngine.pad_to_multiple(img, pad_multiple)
        padded_h, padded_w = img.shape[1], img.shape[2]

        # Determine tile size
        if tile_size == "Auto":
            ts = TileEngine.get_optimal_tile_size(padded_h, padded_w)
        else:
            ts = int(tile_size)
        ts = (ts // pad_multiple) * pad_multiple

        overlap = min(overlap, ts // 2)
        overlap = (overlap // 8) * 8
        overlap = max(16, overlap)

        # NaN/Inf guard — mode-aware for Passthrough's extended range
        if torch.isnan(img).any() or torch.isinf(img).any():
            logger.warning("[Radiance 4K Encode v2.2] NaN/Inf detected — sanitizing")
            if hdr_mode == "Passthrough":
                img = torch.nan_to_num(img, nan=0.0, posinf=1.5, neginf=-0.05)
            else:
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        pbar = comfy.utils.ProgressBar(100)
        total_tiles = 1

        with torch.no_grad():
            if padded_h <= ts and padded_w <= ts:
                # Single tile — use latent_sampling mode
                enc_samples = _encode_with_sampling_mode(vae, img, latent_sampling)
                if isinstance(enc_samples, dict):
                    enc_samples = enc_samples["samples"]
                latent = {
                    "samples": enc_samples,
                    "_total_tiles": 1,
                }
                pbar.update_absolute(100, 100)
            else:
                latent = self._tiled_encode(
                    img, vae, ts, overlap, pbar,
                    latent_sampling=latent_sampling,
                    processing_mode=processing_mode,
                    vae_factor=vae_factor,
                )
                total_tiles = latent.pop("_total_tiles", 1)

        # v2.0 Feature 6: Embed radiance_meta in latent dict for auto-crop in decode
        radiance_meta = {
            "pad_h": pad_h, "pad_w": pad_w,
            "vae_factor": vae_factor, "latent_format": latent_fmt,
            "source_space": source_space, "hdr_mode": hdr_mode,
        }
        latent["radiance_meta"] = radiance_meta

        # Standard metadata JSON (backward compat wire via crop_padding)
        metadata = {
            "node": "RadianceVAE4KEncode",
            "resolution": f"{w}×{h}",
            "padded": f"{padded_w}×{padded_h}" if pad_h or pad_w else "none",
            "tile_size": ts, "overlap": overlap,
            "source_space": source_space, "hdr_mode": hdr_mode,
            "exposure": exposure, "pad_h": pad_h, "pad_w": pad_w,
            "vae_factor": vae_factor, "latent_format": latent_fmt,
            "latent_sampling": latent_sampling,
        }

        # v2.0 Feature 5: Quality metrics
        encode_time_ms = int((_time.time() - t_start) * 1000)
        quality_report = build_encode_quality_report(
            pixels_orig, img.cpu(), latent["samples"].cpu(),
            vae_factor, latent_fmt, source_space, hdr_mode,
            ts, total_tiles, encode_time_ms,
        )

        return (latent, alpha, json.dumps(metadata, indent=2), latent_fmt, quality_report)



# ═══════════════════════════════════════════════════════════════════════════════
#                         4K VAE DECODE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceVAE4KDecode:
    """
    Decode VAE latents to 4K+ images using production-grade tiling.

    Features:
    - VRAM-aware auto tile sizing
    - Cosine blend weights (invisible seams)
    - Direct .rhdr export for Radiance Viewer
    - Color space output (Linear, sRGB, ACEScg, Log formats)
    - Alpha channel restoration
    - Inverse tonemap for HDR recovery
    """

    TARGET_SPACES = EXTENDED_TARGET_SPACES
    LOG_SPACES = EXTENDED_LOG_SPACES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "target_space": (
                    cls.TARGET_SPACES,
                    {"default": "Linear", "tooltip": "Output color space."},
                ),
            },
            "optional": {
                "tile_size": (
                    ["Auto", "512", "768", "1024", "1280", "1536"],
                    {
                        "default": "Auto",
                        "tooltip": "Tile size for decode. Auto queries VRAM.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 256,
                        "step": 16,
                        "tooltip": "Overlap between tiles. 128px optimal for cosine blending.",
                    },
                ),
                "exposure_adjust": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Post-decode exposure in stops.",
                    },
                ),
                "alpha": (
                    "IMAGE",
                    {"tooltip": "Alpha channel from Radiance VAE 4K Encode."},
                ),
                "hdr_mode": (
                    ["Clip (SDR)", "Soft Clip", "Compress (Log)", "Passthrough"],
                    {
                        "default": "Clip (SDR)",
                        "tooltip": "Must match encode setting. 'Soft Clip' inverts tanh rolloff to recover highlights.",
                    },
                ),
                "inverse_tonemap": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Expand SDR to HDR (recover highlights).",
                    },
                ),
                "target_stops": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 8.0,
                        "max": 16.0,
                        "step": 0.5,
                        "tooltip": "Target dynamic range for inverse tonemap.",
                    },
                ),
                "crop_padding": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "JSON metadata from encode (auto-crops padding). Leave empty to skip.",
                    },
                ),
                "export_rhdr": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Export .rhdr sidecar for Radiance Viewer.",
                    },
                ),
                # BUG-G FIX: Precision was previously invisible to the operator
                # (always fp16). Expose it so production shots with values > 65504
                # (sun, explosions, VDB fire) can be delivered in full 32-bit range.
                "rhdr_precision": (
                    ["f16", "f32"],
                    {
                        "default": "f16",
                        "tooltip": (
                            "RHDR export precision. "
                            "'f16' (fp16, default): smaller files, values capped at 65504 — "
                            "adequate for most VFX material. "
                            "'f32' (fp32): full 32-bit range, ~2× file size — "
                            "use for shots with extreme linear values (direct sun, fire VDBs)."
                        ),
                    },
                ),
                "source_space": (
                    EXTENDED_SOURCE_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": (
                            "Source space used during encode. Required when hdr_mode is "
                            "'Compress (Log)' to invert the exact same log curve."
                        ),
                    },
                ),
                # v2.0 Feature 8: Tile processing mode for decode
                # v2.1 FIX (BUG-44): Removed "batched" — decode is much more VRAM-intensive
                # than encode and was never actually implemented for decode. The parameter
                # existed in the UI but silently ran sequential. Encode batched still works.
                "processing_mode": (
                    ["sequential"],
                    {
                        "default": "sequential",
                        "tooltip": "'sequential' processes one tile at a time with minimal VRAM usage.",
                    },
                ),
                # ── HDR-FIX: 32-bit HDR output controls ───────────────────────────────
                "force_hdr_decode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "HDR DIRECT PATH — enables 32-bit HDR output for direct "
                            "encode→decode pipelines (no sampler between nodes). "
                            "When True: hdr_mode is always respected, even if radiance_meta "
                            "is absent from the latent. "
                            "Keep False when decoding after diffusion sampling — the sampler "
                            "produces sRGB-encoded latents and Log/SoftClip inversion would "
                            "produce incorrect color output."
                        ),
                    },
                ),
                "hdr_output": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "32-BIT HDR OUTPUT — when True, skips the final [0,1] clamp "
                            "so the output IMAGE tensor carries full 32-bit scene-linear "
                            "values (>1.0 for bright highlights, <0.0 for below-black). "
                            "Required for Linear/ACEScg/Log target spaces in HDR VFX pipelines. "
                            "Disable when feeding SDR nodes (preview, PNG export, etc.)."
                        ),
                    },
                ),
            },
        }

    # v2.0: 3 outputs — latent_format added
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "metadata", "latent_format")
    OUTPUT_TOOLTIPS = (
        "Decoded image tensor.",
        "Decode metadata JSON.",
        "Latent format detected from VAE (e.g. 'flux_16ch').",
    )
    FUNCTION = "decode"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = (
        "v2.0 — Universal 4K+ VAE decode. Auto VAE-factor, 11 color spaces, "
        "video 5D support, auto-crop from radiance_meta, batched tile mode."
    )

    def _tiled_decode(
        self,
        samples: Dict[str, Any],
        vae: Any,
        tile_size_px: int,
        overlap_px: int,
        pbar: Any = None,
        vae_factor: int = 8,
    ) -> torch.Tensor:
        """
        Decode full latent in overlapping tiles with cosine blend.

        Called only after decode() has already handled the frame-loop path for
        non-3D-native VAEs. By the time we arrive here, samples["samples"] is
        always 4D (B, C, H, W) — a single image, a single extracted frame, or a
        3D-native VAE latent whose 5D output is reshaped inside the tile loop.

        Args:
            samples:      {"samples": (B, C, H, W)} latent dict
            vae:          ComfyUI VAE model
            tile_size_px: Tile size in pixel space (divided by vae_factor for latent space)
            overlap_px:   Overlap in pixel space
            pbar:         Optional ComfyUI ProgressBar
            vae_factor:   Spatial downscale factor (auto-detected by caller)

        Returns:
            Tuple of:
              - (B, pixH, pixW, C) decoded image tensor
              - int | None  — frame count if a 3D-native VAE produced 5D output, else None
        """
        latent = samples["samples"]

        b, c = latent.shape[0], latent.shape[1]
        lat_h, lat_w = latent.shape[-2], latent.shape[-1]
        device = latent.device
        # v2.1 FIX (BUG-42): Use detected vae_factor instead of hardcoded 8.
        # Stable Cascade Stage C uses factor=42 — hardcoded 8 gave completely
        # wrong tile pixel coordinates, corrupting the output.
        scale = vae_factor

        pix_h = lat_h * scale
        pix_w = lat_w * scale

        # Tile in latent space
        lat_tile = tile_size_px // scale
        lat_overlap = overlap_px // scale

        tiles_y = TileEngine.compute_tiles(lat_h, lat_tile, lat_overlap)
        tiles_x = TileEngine.compute_tiles(lat_w, lat_tile, lat_overlap)
        total_tiles = len(tiles_y) * len(tiles_x)

        logger.info(
            f"[Radiance 4K Decode] {pix_w}×{pix_h} → {len(tiles_x)}×{len(tiles_y)} "
            f"tiles ({tile_size_px}px/{lat_tile}lat, {total_tiles} total)"
        )

        # FIX-3: Do NOT pre-allocate output with a hardcoded 3 channels.
        # The VAE channel count is unknown until the first tile is decoded —
        # inpainting VAEs, future model variants, or custom VAEs may return 4+
        # channels. Allocate lazily from the first tile's actual shape.
        output = None
        weight_acc = None
        out_c = None  # Will be set from first tile

        tile_idx = 0
        _tile_video_frames = None  # Populated on first tile if 3D VAE detected
        for yi, (ly1, ly2) in enumerate(tiles_y):
            for xi, (lx1, lx2) in enumerate(tiles_x):
                # Extract latent tile
                # Use ellipsis to gracefully handle both 4D (H, W) and 5D (F, H, W) slicing
                tile_lat = latent[..., ly1:ly2, lx1:lx2].contiguous()

                # Decode on GPU — defensive no_grad: idempotent if caller has it,
                # protective if _tiled_decode is ever invoked standalone.
                with torch.no_grad():
                    tile_decoded = vae.decode(tile_lat).float().cpu()

                # FIX-4: 3D (temporal) VAEs return (B, F, H, W, C) — a 5D tensor.
                # Reshape to (B*F, H, W, C) so all downstream accumulation logic
                # treats frames as batch elements (standard ComfyUI IMAGE format).
                if tile_decoded.ndim == 5:
                    _B5, _F5, _H5, _W5, _C5 = tile_decoded.shape
                    if _tile_video_frames is None:
                        _tile_video_frames = _F5
                    tile_decoded = tile_decoded.reshape(_B5 * _F5, _H5, _W5, _C5)
                    # Also update b so the accumulator is sized for all frames
                    b = tile_decoded.shape[0]

                # Pixel coordinates
                px1 = lx1 * scale
                py1 = ly1 * scale
                th, tw = tile_decoded.shape[1], tile_decoded.shape[2]

                # FIX-3: Lazy-init accumulators on first tile so channel count is
                # taken from the actual VAE output, not assumed to be 3.
                if output is None:
                    out_c = tile_decoded.shape[3]
                    output = torch.zeros(
                        (b, pix_h, pix_w, out_c), dtype=torch.float32, device="cpu"
                    )
                    weight_acc = torch.zeros(
                        (1, pix_h, pix_w, 1), dtype=torch.float32, device="cpu"
                    )
                    logger.info(f"[Radiance 4K Decode] VAE output channels: {out_c}")

                # Per-edge overlap (pixel space)
                pix_overlap = lat_overlap * scale
                ov_top = pix_overlap if yi > 0 else 0
                ov_bot = pix_overlap if yi < len(tiles_y) - 1 else 0
                ov_left = pix_overlap if xi > 0 else 0
                ov_right = pix_overlap if xi < len(tiles_x) - 1 else 0

                blend_w = TileEngine.make_cosine_blend_weight_2d(
                    th, tw, ov_top, ov_bot, ov_left, ov_right, device="cpu"
                )

                # Accumulate
                output[:, py1 : py1 + th, px1 : px1 + tw, :] += tile_decoded * blend_w
                weight_acc[:, py1 : py1 + th, px1 : px1 + tw, :] += blend_w

                tile_idx += 1
                if pbar:
                    pbar.update_absolute(tile_idx, total_tiles)

                # Free tile references (GPU texture freed by Python GC)
                del tile_lat, tile_decoded, blend_w

        # V-6 FIX: Single empty_cache after all tiles instead of per-tile.
        # Note: If decode OOMs on very large images, restore per-tile cleanup
        # as a fallback — but for most workloads this saves 24-48ms of sync overhead.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Normalize by accumulated weight (guard against near-zero at tile edges)
        weight_acc = torch.clamp(weight_acc, min=1e-3)
        output = output / weight_acc
        return output.to(device), _tile_video_frames

    def _vae_output_to_target(
        self,
        img: torch.Tensor,
        target_space: str,
        hdr_mode: str,
        exposure: float,
        inverse_tonemap: bool,
        target_stops: float,
        source_space: str = "Linear",
        hdr_output: bool = False,
    ) -> torch.Tensor:
        """Convert VAE output to target color space.

        v2.2 production pipeline — all HDR clamps audited for 32-bit precision.

        Pipeline: VAE output → Linearize → Inverse Tonemap → Exposure → Target Space

        All operations happen in linear space to prevent double-gamma errors.

        v2.2 FIX audit:
          - BUG-48: Color matrices safe for 4+ channels (alpha passthrough)
          - BUG-50: Passthrough clamp widened from [−0.05, 1.1] → [−0.05, 1.6]
          - BUG-52: Soft Clip atanh safety raised from 0.9999 → 0.999999
                    (highlight recovery ceiling 2.9 lin → 4.6 lin)
          - BUG-51: Extended sRGB EOTF for Soft Clip / Passthrough values > 1.0

        HDR-FIX (hdr_output):
          When hdr_output=True, the final result is NOT clamped to [0,1].
          Scene-linear values > 1.0 (bright highlights, sky, fire, explosions)
          and < 0.0 (below-black, out-of-gamut in ACEScg) are preserved as-is
          in the output fp32 tensor. Use target_space="Linear" or "ACEScg" for
          full HDR scene-referred output.
        """

        # Step 1: VAE output → Linear (ALWAYS linearize)
        #
        # Pre-clamp: Remove VAE reconstruction noise outside the encoded domain.
        # Each hdr_mode has a different valid range in the encoded space.
        # The clamp MUST match the encode-side range to avoid data loss.
        if hdr_mode == "Compress (Log)":
            # Log-encoded domain: the log curve compresses scene linear into
            # approximately [0, 1]. The hard clamp at [0, 1] is correct because:
            #   • Values above 1.0 are VAE noise, not signal. The log curve is
            #     extremely steep near 1.0 (LogC4: 1.0→~38 lin, 1.01→~43 lin),
            #     so even tiny noise amplifies to huge linear errors.
            #   • The below-zero linear content is preserved by the encode-side
            #     change (BUG-49) which lets small negatives into the log curve.
            #     Those map to log values slightly above 0, not below 0.
            img = torch.clamp(img, 0.0, 1.0)
        elif hdr_mode == "Soft Clip":
            # Tanh-encoded domain: max encoded value is exactly 1.0 (tanh → 1).
            # Clamp to [0, 1] is correct — anything outside is VAE noise.
            img = torch.clamp(img, 0.0, 1.0)
        elif hdr_mode == "Passthrough":
            # v2.2 FIX (BUG-50): Widened from [−0.05, 1.1] → [−0.05, 1.6].
            # Must match the widened encode clamp [−0.05, 1.5] plus a small
            # margin for tile-boundary reconstruction blur. Previous 1.1 ceiling
            # destroyed all HDR above 1.24 linear (~0.3 stops above SDR white).
            img = torch.clamp(img, -0.05, 1.6)
        else:
            # Clip (SDR): [0, 1] by definition.
            img = torch.clamp(img, 0.0, 1.0)

        # Step 1b: Decode from encoded space → linear
        if hdr_mode == "Compress (Log)":
            if _HAS_LOG_CURVES:
                # FIX-2 (decode side): Decompress using the curve that matches
                # the encode-side source space, not always LogC4.
                log_to_linear = RadianceVAE4KEncode._get_log_to_linear()
                decompress_fn = log_to_linear.get(source_space, tensor_logc4_to_linear)
                img = decompress_fn(img)
            else:
                logger.warning(
                    "[Radiance 4K Decode] Compress (Log) hdr_mode selected but log "
                    "curves are unavailable — falling back to sRGB linearization. "
                    "Output will be incorrect. Install color_utils.py to fix this."
                )
                img = tensor_srgb_to_linear(img)
        elif hdr_mode == "Soft Clip":
            # v2.2 FIX (BUG-52): Invert tanh soft clip with raised atanh ceiling.
            #
            # Encode:  excess = (v − knee) / rng;  coded = knee + rng × tanh(excess)
            # Decode:  t = (coded − knee) / rng;    excess = atanh(t);  v = knee + excess × rng
            #
            # Previous atanh clamp of 0.9999 gave max recovery of:
            #   atanh(0.9999) ≈ 4.95 → sRGB 1.59 → linear 2.9
            # Raised to 0.999999:
            #   atanh(0.999999) ≈ 7.25 → sRGB 1.94 → linear 4.6
            # This nearly doubles the effective HDR ceiling.
            knee = 0.85
            rng = 1.0 - knee  # 0.15
            above = img > knee
            if above.any():
                recovered = img.clone()
                t = torch.clamp((img[above] - knee) / rng, 0.0, 0.999999)
                excess = torch.atanh(t)
                recovered[above] = knee + excess * rng
                img = recovered
            # v2.2 FIX (BUG-51): Use extended sRGB EOTF because recovered
            # values range from [0, ~1.94] — standard srgb_to_linear may clamp
            # inputs to [0, 1] internally, destroying all recovered highlights.
            img = _safe_srgb_to_linear_extended(img)
        elif hdr_mode == "Passthrough":
            # v2.2 FIX (BUG-51): Passthrough values can be up to 1.6 after
            # the widened clamp. Use extended EOTF to preserve HDR headroom.
            img = _safe_srgb_to_linear_extended(img)
        else:
            # Clip (SDR) / default: standard linearization.
            # Data is in [0, 1] from clamp above, so standard EOTF is safe.
            img = tensor_srgb_to_linear(img)

        # Step 2: Inverse tonemap (SDR→HDR expansion)
        if inverse_tonemap and hdr_mode != "Compress (Log)":
            target_peak = 2.0 ** (target_stops - 8.0)
            if target_peak > 1.0:
                # v2.2 FIX (BUG-48): Safe channel indexing for 4+ channels.
                luma = (
                    0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
                )
                threshold = 0.75
                a = (target_peak - 1.0) / ((1.0 - threshold) ** 2)
                luma_c = torch.clamp(luma, max=1.0)
                expanded = torch.where(
                    luma > threshold,
                    threshold + (luma_c - threshold) + a * (luma_c - threshold) ** 2,
                    luma,
                )
                slope = 1.0 + 2 * a * (1.0 - threshold)
                expanded = torch.where(
                    luma > 1.0, target_peak + (luma - 1.0) * slope, expanded
                )
                ratio = (expanded / (luma + 1e-8)).unsqueeze(-1)
                # Apply ratio to RGB only, leave any extra channels untouched
                if img.shape[-1] > 3:
                    img_rgb = img[..., :3] * ratio
                    img = torch.cat([img_rgb, img[..., 3:]], dim=-1)
                else:
                    img = img * ratio

        # Step 3: Exposure (linear domain)
        if exposure != 0.0:
            # Apply only to RGB channels, leave alpha/extra untouched
            if img.shape[-1] > 3:
                img_rgb = img[..., :3] * (2.0**exposure)
                img = torch.cat([img_rgb, img[..., 3:]], dim=-1)
            else:
                img = img * (2.0**exposure)

        # Step 4: Linear → target space
        if target_space == "Linear":
            pass  # Already linear
        elif target_space == "sRGB":
            img = tensor_linear_to_srgb(img)
        elif target_space == "ACEScg":
            mat = _REC709_TO_AP1.to(img.device)
            img = _safe_matrix_transform(img, mat)
        elif target_space == "ACES 2065-1":
            mat = _REC709_TO_AP0.to(img.device)
            img = _safe_matrix_transform(img, mat)
        elif target_space == "Rec.2020 Linear":
            mat = _REC709_TO_REC2020.to(img.device)
            img = _safe_matrix_transform(img, mat)
        elif target_space in self.LOG_SPACES and _HAS_LOG_CURVES:
            converters = RadianceVAE4KEncode._get_linear_to_log()
            img = converters[target_space](img)
        elif target_space in self.LOG_SPACES and not _HAS_LOG_CURVES:
            logger.warning(
                f"[Radiance 4K Decode] Target space '{target_space}' is a log format "
                f"but color_utils is not installed — outputting LINEAR data instead. "
                f"Install color_utils.py to enable log output."
            )
        elif target_space == "Raw":
            pass

        # HDR-FIX: When hdr_output=False (default/SDR mode), clamp to [0,1] so
        # downstream SDR nodes (preview, PNG export, etc.) receive valid data.
        # When hdr_output=True, skip this clamp — the caller explicitly wants the
        # full 32-bit scene-linear range preserved in the output tensor.
        if not hdr_output:
            # Only clamp for display-referred target spaces.
            # Linear/ACEScg/Rec.2020/Log spaces are scene-referred; clamping them
            # destroys HDR content even in SDR mode (they need a tonemap first).
            # Safe to clamp only sRGB and Raw which are already display-referred.
            if target_space in ("sRGB",):
                img = torch.clamp(img, 0.0, 1.0)

        # BUG-C FIX: Guarantee fp32 output regardless of which path was taken.
        return img.float()

    @staticmethod
    def _save_rhdr(
        img_np, output_dir: str, prefix: str = "radiance_4k", precision: str = "f16"
    ) -> Optional[str]:
        """Save image as .rhdr compressed float.

        Args:
            img_np:     (H, W, C) numpy array, any float dtype.
            output_dir: Target directory.
            prefix:     Filename stem.
            precision:  "f16" (fp16, default — smaller file, max value 65504) or
                        "f32" (fp32 — full 32-bit range, ~2× file size).
                        BUG-F/G FIX: Previously hard-coded fp16 with no operator
                        control. Production VFX content (sun, explosions, VDBs) can
                        exceed 65504 in linear light — use "f32" for those shots.
        """
        try:
            h, w = img_np.shape[:2]
            c = img_np.shape[2] if img_np.ndim == 3 else 1

            # V-9 FIX: Guard against uint16 header overflow
            if w > 65535 or h > 65535:
                logger.warning(
                    f"[Radiance 4K] RHDR export skipped: dimensions {w}×{h} "
                    f"exceed uint16 header limit (65535)"
                )
                return None

            unique_id = uuid.uuid4().hex[:12]
            filename = f"{prefix}_{unique_id}.rhdr"

            # V-7 FIX: Use safe_join to prevent path traversal via malicious prefix
            try:
                from .path_utils import safe_join
                filepath = safe_join(output_dir, filename)
            except ImportError:
                filepath = os.path.join(output_dir, filename)

            # BUG-F FIX: Support fp32 for scenes with linear values > 65504.
            # Header precision flag: 0 = fp16 (legacy), 1 = fp32.
            if precision == "f32":
                payload = img_np.astype("float32").tobytes()
                prec_flag = 1
            else:
                payload = img_np.astype("float16").tobytes()
                prec_flag = 0

            compressed = zlib.compress(payload, level=6)

            # BUG-G FIX: Precision flag stored in the reserved header byte (was 0).
            header = struct.pack("<4sHHHH", b"RHDR", w, h, c, prec_flag)

            with open(filepath, "wb") as f:
                f.write(header)
                f.write(compressed)

            ratio = len(compressed) / len(payload) * 100
            size_mb = (len(header) + len(compressed)) / (1024 * 1024)
            dtype_str = "fp32" if precision == "f32" else "fp16"
            logger.info(
                f"[Radiance 4K] RHDR export: {filename} "
                f"({w}×{h}, {dtype_str}, {size_mb:.1f}MB, {ratio:.0f}% ratio)"
            )
            return filename
        except Exception as e:
            logger.warning(f"[Radiance 4K] RHDR export failed: {e}")
            return None

    def decode(
        self,
        samples: Dict[str, Any],
        vae: Any,
        target_space: str = "Linear",
        tile_size: str = "Auto",
        overlap: int = 128,
        exposure_adjust: float = 0.0,
        alpha: torch.Tensor = None,
        hdr_mode: str = "Clip (SDR)",
        inverse_tonemap: bool = False,
        target_stops: float = 12.0,
        crop_padding: str = "",
        export_rhdr: bool = False,
        rhdr_precision: str = "f16",
        source_space: str = "Linear",
        processing_mode: str = "sequential",
        force_hdr_decode: bool = False,
        hdr_output: bool = False,
    ) -> Tuple:
        """v2.2 HDR-FIX: Universal decode with 32-bit HDR output support.

        New parameters vs v2.1:
          force_hdr_decode: When True, respects hdr_mode even without radiance_meta
                            (direct encode→decode, no sampler). Required for HDR output
                            with Compress(Log) or Soft Clip modes post-encoding.
          hdr_output:       When True, disables the final [0,1] normalization clamp so
                            the output tensor carries full 32-bit scene-linear values.
                            Combine with target_space="Linear" or "ACEScg" for HDR VFX.
        """

        # v2.0 Feature 2: Auto-detect VAE factor
        vae_factor = detect_vae_factor(vae)
        latent_fmt = detect_latent_format(vae)

        # v2.0 Feature 6: Read radiance_meta embedded by encode() if present
        radiance_meta = samples.get("radiance_meta", {})
        if radiance_meta:
            latent_fmt = radiance_meta.get("latent_format", latent_fmt)
            vae_factor = radiance_meta.get("vae_factor", vae_factor)

            # v2.2 FIX (BUG-47): Auto-read hdr_mode and source_space from
            # radiance_meta so encode→decode are always symmetric. Previously
            # the encode stored these in radiance_meta but the decode ignored
            # them, relying solely on the user's manual widget selection.
            meta_hdr = radiance_meta.get("hdr_mode")
            meta_src = radiance_meta.get("source_space")
            if meta_hdr and meta_hdr != hdr_mode:
                logger.info(
                    f"[Radiance 4K Decode v2.2] radiance_meta hdr_mode='{meta_hdr}' "
                    f"overrides widget hdr_mode='{hdr_mode}' for encode↔decode symmetry."
                )
                hdr_mode = meta_hdr
            if meta_src and meta_src != source_space:
                logger.info(
                    f"[Radiance 4K Decode v2.2] radiance_meta source_space='{meta_src}' "
                    f"overrides widget source_space='{source_space}' for encode↔decode symmetry."
                )
                source_space = meta_src
        else:
            # BUG-47 / HDR-FIX: No radiance_meta in latent dict.
            # Two legitimate scenarios reach this branch:
            #   A) Latent passed through a sampler — sampler strips custom dict keys.
            #      After diffusion, the VAE output is sRGB-like regardless of encode
            #      settings. Applying Log/SoftClip inversion here → wrong color output.
            #   B) Direct encode→decode with no sampler, but radiance_meta was stripped
            #      by a third-party node. User WANTS the HDR mode respected.
            #
            # Resolution: force_hdr_decode overrides the safety fallback.
            #   force_hdr_decode=False (default) → original BUG-47 safety: warn + Clip(SDR)
            #   force_hdr_decode=True  (HDR mode) → respect user hdr_mode, no override.
            if hdr_mode in ("Compress (Log)", "Soft Clip") and not force_hdr_decode:
                logger.warning(
                    f"[Radiance 4K Decode v2.2] hdr_mode='{hdr_mode}' selected but "
                    f"radiance_meta is absent — latent likely passed through a sampler "
                    f"which strips encode metadata. After diffusion sampling the VAE "
                    f"decodes to sRGB; Log/SoftClip inversion would corrupt color. "
                    f"Falling back to 'Clip (SDR)'. "
                    f"To preserve HDR for direct encode→decode (no sampler): enable "
                    f"force_hdr_decode=True on this node."
                )
                hdr_mode = "Clip (SDR)"
            elif hdr_mode in ("Compress (Log)", "Soft Clip") and force_hdr_decode:
                logger.info(
                    f"[Radiance 4K Decode v2.2] force_hdr_decode=True: respecting "
                    f"hdr_mode='{hdr_mode}' despite absent radiance_meta. "
                    f"Ensure this is a direct encode→decode path (no sampler between "
                    f"encode and decode) for correct results."
                )

        latent = samples["samples"]
        
        # v2.1 Video Support: Check for 5D video latent (B, C, F, H, W)
        is_video = latent.ndim == 5
        
        # Check if the VAE natively supports 3D latents (e.g., Wan Video, Cosmos)
        is_3d_vae = False
        if hasattr(vae, "latent_dim") and getattr(vae, "latent_dim") == 3:
            is_3d_vae = True
            
        if is_video and not is_3d_vae:
            B, C, F, H, W = latent.shape
            logger.info(
                f"[Radiance 4K Decode v2.1] Video Latent: {W}×{H} ({B} batches, {F} frames), format={latent_fmt}"
            )
            decoded_frames = []
            frame_alphas = None
            if alpha is not None:
                if alpha.shape[0] == B * F:
                    frame_alphas = alpha.chunk(F, dim=0)
                else:
                    frame_alphas = [alpha] * F

            for fi in range(F):
                frame_latent = latent[:, :, fi, ...]
                frame_samples = dict(samples)
                frame_samples["samples"] = frame_latent
                
                frame_alpha = frame_alphas[fi] if frame_alphas else None

                decoded_frame, meta_str, fmt = self.decode(
                    samples=frame_samples,
                    vae=vae,
                    target_space=target_space,
                    tile_size=tile_size,
                    overlap=overlap,
                    exposure_adjust=exposure_adjust,
                    alpha=frame_alpha,
                    hdr_mode=hdr_mode,
                    inverse_tonemap=inverse_tonemap,
                    target_stops=target_stops,
                    crop_padding=crop_padding,
                    export_rhdr=False,   # RHDR handled on concatenated output below
                    rhdr_precision=rhdr_precision,
                    source_space=source_space,
                    processing_mode=processing_mode,
                    force_hdr_decode=force_hdr_decode,
                    hdr_output=hdr_output,
                )
                decoded_frames.append(decoded_frame)

            all_frames = torch.cat(decoded_frames, dim=0)

            try:
                meta_json = json.loads(meta_str)
            except (json.JSONDecodeError, TypeError, ValueError):
                meta_json = {}
            meta_json["video"] = True
            meta_json["frames"] = F

            # v2.1 FIX (BUG-46): Export RHDR sidecars for video frames.
            # Previously skipped because per-frame decode passed export_rhdr=False.
            # Now runs on the concatenated output after all frames are decoded.
            rhdr_filenames = []
            if export_rhdr:
                output_dir = folder_paths.get_temp_directory()
                for fi in range(all_frames.shape[0]):
                    frame_np = all_frames[fi, ..., :3].cpu().numpy()
                    while frame_np.ndim > 3:
                        frame_np = frame_np[0]
                    fname = self._save_rhdr(
                        frame_np, output_dir,
                        prefix=f"radiance_4k_f{fi:04d}",
                        precision=rhdr_precision,
                    )
                    if fname:
                        rhdr_filenames.append(fname)
                if rhdr_filenames:
                    meta_json["rhdr_export"] = rhdr_filenames
                    meta_json["rhdr_precision"] = rhdr_precision

            return (all_frames, json.dumps(meta_json, indent=2), latent_fmt)

        b, c = latent.shape[0], latent.shape[1]
        lat_h, lat_w = latent.shape[-2], latent.shape[-1]
        pix_h, pix_w = lat_h * vae_factor, lat_w * vae_factor

        logger.info(
            f"[Radiance 4K Decode v2.0] Latent: {lat_w}×{lat_h} → "
            f"Output: {pix_w}×{pix_h} ({b} frames, factor={vae_factor}, fmt={latent_fmt})"
        )

        target_device = comfy.model_management.get_torch_device()
        if latent.device != target_device:
            latent = latent.to(target_device)
            samples = dict(samples)
            samples["samples"] = latent

        pad_multiple = max(vae_factor, 8)

        # Tile size in pixel space (must be aligned to vae_factor)
        if tile_size == "Auto":
            ts_px = TileEngine.get_optimal_tile_size(pix_h, pix_w)
        else:
            ts_px = int(tile_size)
        ts_px = (ts_px // pad_multiple) * pad_multiple

        overlap = min(overlap, ts_px // 2)
        overlap = (overlap // 8) * 8
        overlap = max(16, overlap)

        pbar = comfy.utils.ProgressBar(100)

        decoded_video_frames = None  # Set when 3D VAE returns 5D output
        with torch.no_grad():
            if pix_h <= ts_px and pix_w <= ts_px:
                img = vae.decode(latent).float()
                # FIX-4: 3D (temporal) VAEs return (B, F, H, W, C) — reshape to
                # (B*F, H, W, C) so downstream code handles it as a frame batch.
                if img.ndim == 5:
                    _B5, _F5, _H5, _W5, _C5 = img.shape
                    decoded_video_frames = _F5
                    img = img.reshape(_B5 * _F5, _H5, _W5, _C5)
                pbar.update_absolute(100, 100)
            else:
                img, _tiled_video_frames = self._tiled_decode(samples, vae, ts_px, overlap, pbar, vae_factor=vae_factor)
                if _tiled_video_frames is not None:
                    decoded_video_frames = _tiled_video_frames

        img = img.float()

        if torch.isnan(img).any() or torch.isinf(img).any():
            logger.warning("[Radiance 4K Decode v2.2] VAE produced NaN/Inf — sanitizing")
            # v2.2 FIX (BUG-53): Pre-transform guard must be mode-aware.
            #   • posinf → 1.0 is correct for all modes (encoded domain max = 1.0,
            #     or 1.5 for Passthrough, but Inf is never valid signal).
            #   • neginf → −0.05 for Passthrough (which encodes small negatives).
            #     neginf → 0.0 for all other modes (log/sRGB domain floor = 0).
            if hdr_mode == "Passthrough":
                img = torch.nan_to_num(img, nan=0.0, posinf=1.5, neginf=-0.05)
            else:
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        img = self._vae_output_to_target(
            img, target_space, hdr_mode,
            exposure_adjust, inverse_tonemap, target_stops,
            source_space=source_space,
            hdr_output=hdr_output,
        )

        # v2.2 FIX (BUG-B): Guard for NaN/Inf *introduced by* the color transform.
        # Log decompression of VAE noise near 1.0 → exp-domain can produce +Inf.
        # Inverse tonemap with high target_stops can also amplify noise to Inf.
        #
        # HDR-FIX: posinf cap is mode-aware:
        #   hdr_output=False: cap at 65504 (fp16 max — safe for RHDR export)
        #   hdr_output=True:  cap at fp32 max (~3.4e38) — only true Inf is replaced,
        #                     preserving all valid HDR scene-linear values.
        if torch.isnan(img).any() or torch.isinf(img).any():
            logger.warning(
                "[Radiance 4K Decode v2.2] NaN/Inf after color transform — sanitizing. "
                "Check VAE reconstruction quality and hdr_mode/source_space pairing."
            )
            posinf_val = 3.4e38 if hdr_output else 65504.0
            img = torch.nan_to_num(img, nan=0.0, posinf=posinf_val, neginf=0.0)

        # v2.0 Feature 6: Auto-crop — prefer radiance_meta, fall back to crop_padding string
        pad_h, pad_w = 0, 0
        if radiance_meta:
            pad_h = radiance_meta.get("pad_h", 0)
            pad_w = radiance_meta.get("pad_w", 0)
        elif crop_padding:
            try:
                meta = json.loads(crop_padding)
                pad_h = meta.get("pad_h", 0)
                pad_w = meta.get("pad_w", 0)
            except (json.JSONDecodeError, TypeError):
                pass

        if pad_h > 0 or pad_w > 0:
            crop_h = img.shape[1] - pad_h
            crop_w = img.shape[2] - pad_w
            img = img[:, :crop_h, :crop_w, :]
            logger.info(f"[Radiance 4K v2.0] Cropped padding: {pad_h}h, {pad_w}w → {crop_w}×{crop_h}")

        # Restore alpha
        if alpha is not None:
            alpha_f = alpha.float()
            if alpha_f.dim() == 3:
                alpha_f = alpha_f.unsqueeze(0)
            alpha_ch = alpha_f[..., :1]
            if alpha_ch.shape[1:3] != img.shape[1:3]:
                alpha_ch = F.interpolate(
                    alpha_ch.permute(0, 3, 1, 2),
                    size=(img.shape[1], img.shape[2]),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
            if alpha_ch.shape[0] != img.shape[0]:
                alpha_ch = alpha_ch.expand(img.shape[0], -1, -1, -1)
            img = torch.cat([img, alpha_ch], dim=-1)

        # Export .rhdr if requested — one sidecar per frame for video, single file for stills
        rhdr_filenames = []
        if export_rhdr:
            output_dir = folder_paths.get_temp_directory()
            num_frames = img.shape[0]
            for fi in range(num_frames):
                frame_np = img[fi, ..., :3].cpu().numpy()
                # FIX-4: Guard against any residual extra dims
                while frame_np.ndim > 3:
                    frame_np = frame_np[0]
                prefix = f"radiance_4k_f{fi:04d}" if num_frames > 1 else "radiance_4k"
                fname = self._save_rhdr(
                    frame_np, output_dir, prefix=prefix, precision=rhdr_precision
                )
                if fname:
                    rhdr_filenames.append(fname)

        final_h, final_w = img.shape[1], img.shape[2]
        metadata = {
            "node": "RadianceVAE4KDecode",
            "version": "2.0",
            "resolution": f"{final_w}×{final_h}",
            "tile_size": ts_px, "overlap": overlap,
            "target_space": target_space, "hdr_mode": hdr_mode,
            "exposure_adjust": exposure_adjust,
            "inverse_tonemap": inverse_tonemap,
            "target_stops": target_stops if inverse_tonemap else "N/A",
            "alpha_restored": alpha is not None,
            "vae_factor": vae_factor, "latent_format": latent_fmt,
            "processing_mode": processing_mode,
        }
        # Video metadata: populated for both 3D-native VAE path and frame-loop path
        if decoded_video_frames is not None:
            metadata["video"] = True
            metadata["frames"] = decoded_video_frames
        elif img.shape[0] > 1:
            # Batch of images (e.g. from frame-loop path arriving here as concatenated frames)
            metadata["video"] = True
            metadata["frames"] = img.shape[0]
        if rhdr_filenames:
            metadata["rhdr_export"] = rhdr_filenames[0] if len(rhdr_filenames) == 1 else rhdr_filenames
            metadata["rhdr_precision"] = rhdr_precision

        return (img, json.dumps(metadata, indent=2), latent_fmt)



# ═══════════════════════════════════════════════════════════════════════════════
#                    4K DIRECT ENCODE → DECODE (ROUNDTRIP)
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceVAE4KRoundtrip:
    """
    Single-node 4K VAE roundtrip: Encode → Decode in one step.

    Use case: Direct 4K export without diffusion — tests VAE reconstruction
    quality, applies color transforms, or serves as a high-quality resampler.

    Also useful for preparing reference frames: encode 4K plate, decode to
    see what the VAE preserves vs destroys, then adjust workflow accordingly.
    """

    SOURCE_SPACES = RadianceVAE4KEncode.SOURCE_SPACES
    TARGET_SPACES = RadianceVAE4KDecode.TARGET_SPACES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            },
            "optional": {
                "source_space": (cls.SOURCE_SPACES, {"default": "Linear"}),
                "target_space": (cls.TARGET_SPACES, {"default": "Linear"}),
                "tile_size": (
                    ["Auto", "512", "768", "1024", "1280", "1536"],
                    {"default": "Auto"},
                ),
                "overlap": ("INT", {"default": 128, "min": 32, "max": 256, "step": 16}),
                "exposure": (
                    "FLOAT",
                    {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "hdr_mode": (
                    ["Clip (SDR)", "Soft Clip", "Compress (Log)", "Passthrough"],
                    {"default": "Soft Clip"},
                ),
                "export_rhdr": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Export .rhdr for Radiance Viewer."},
                ),
                "rhdr_precision": (
                    ["f16", "f32"],
                    {
                        "default": "f16",
                        "tooltip": (
                            "RHDR export precision. "
                            "'f16': smaller files, values capped at 65504. "
                            "'f32': full 32-bit range — use for extreme HDR content."
                        ),
                    },
                ),
                # HDR-FIX: Roundtrip is always direct encode→decode (no sampler),
                # so force_hdr_decode and hdr_output default to True here.
                "hdr_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "32-BIT HDR OUTPUT — preserves full scene-linear range "
                            "(values >1.0 and <0.0) in the output tensor. "
                            "Default True for Roundtrip since there is no sampler. "
                            "Disable only when feeding SDR-only downstream nodes."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("image", "latent", "metadata")
    FUNCTION = "roundtrip"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = "4K VAE roundtrip in one node: encode → decode with tiling. Test reconstruction or apply color transforms."

    def roundtrip(
        self,
        pixels: torch.Tensor,
        vae: Any,
        source_space: str = "Linear",
        target_space: str = "Linear",
        tile_size: str = "Auto",
        overlap: int = 128,
        exposure: float = 0.0,
        hdr_mode: str = "Soft Clip",
        export_rhdr: bool = True,
        rhdr_precision: str = "f16",
        hdr_output: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any], str]:

        # V-4 FIX: Handle both 4D images (B,H,W,C) and 5D video (B,F,H,W,C).
        # Previously both ternary branches returned pixels.shape, crashing on 5D.
        if pixels.ndim == 5:
            b, f, h, w, c = pixels.shape
            logger.info(f"[Radiance 4K Roundtrip v2.0] Video: {w}\u00D7{h} ({f} frames), {source_space} \u2192 {target_space}")
        else:
            b, h, w, c = pixels.shape
            logger.info(f"[Radiance 4K Roundtrip v2.0] {w}\u00D7{h}, {source_space} \u2192 {target_space}")

        # Encode (returns 5-tuple in v2.0)
        encoder = RadianceVAE4KEncode()
        latent, alpha, enc_meta, latent_fmt, _quality = encoder.encode(
            pixels, vae, source_space, tile_size, overlap, exposure, "Preserve", hdr_mode,
        )

        # Map encode hdr_mode to decode hdr_mode
        decode_hdr = {
            "Clip (SDR)": "Clip (SDR)",
            "Soft Clip": "Soft Clip",
            "Compress (Log)": "Compress (Log)",
            "Passthrough": "Passthrough",
        }.get(hdr_mode, "Clip (SDR)")

        # Decode — reverse encode exposure so VAE roundtrip is neutral (returns 3-tuple in v2.0)
        decoder = RadianceVAE4KDecode()
        img, dec_meta, _fmt = decoder.decode(
            latent, vae, target_space, tile_size, overlap,
            exposure_adjust=-exposure, alpha=alpha,
            hdr_mode=decode_hdr,
            crop_padding=enc_meta,
            export_rhdr=export_rhdr,
            rhdr_precision=rhdr_precision,
            source_space=source_space,
            force_hdr_decode=True,   # Roundtrip is always direct — no sampler in path
            hdr_output=hdr_output,
        )

        # Combined metadata
        combined_meta = {
            "node": "RadianceVAE4KRoundtrip",
            "version": "2.0",
            "latent_format": latent_fmt,
            "encode": json.loads(enc_meta),
            "decode": json.loads(dec_meta),
        }

        return (img, latent, json.dumps(combined_meta, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
#                          NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceVAE4KEncode": RadianceVAE4KEncode,
    "RadianceVAE4KDecode": RadianceVAE4KDecode,
    "RadianceVAE4KRoundtrip": RadianceVAE4KRoundtrip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceVAE4KEncode": "◎ Radiance VAE 4K Encode",
    "RadianceVAE4KDecode": "◎ Radiance VAE 4K Decode",
    "RadianceVAE4KRoundtrip": "◎ Radiance VAE 4K Roundtrip",
}
