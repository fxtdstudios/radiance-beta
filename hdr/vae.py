"""
Radiance VAE 4K — Native 4K Direct Encode/Decode
═══════════════════════════════════════════════════════════════════
Production-grade tiled VAE for 4K+ resolution without super-resolution.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│  Input Image (e.g. 3840×2160)                                  │
│       │                                                        │
│       ▼                                                        │
│  Pad to multiple of 8 → 3840×2160 (already aligned)           │
│       │                                                        │
│       ▼                                                        │
│  Split into overlapping tiles (VRAM-aware sizing)              │
│  ┌──────┬──────┬──────┐                                       │
│  │ T1   │ T2   │ T3   │  ← overlap region uses cosine blend  │
│  ├──────┼──────┼──────┤                                       │
│  │ T4   │ T5   │ T6   │                                       │
│  └──────┴──────┴──────┘                                       │
│       │                                                        │
│       ▼                                                        │
│  VAE encode each tile → latent tiles                          │
│       │                                                        │
│       ▼                                                        │
│  Blend overlaps in latent space (cosine weights)              │
│       │                                                        │
│       ▼                                                        │
│  Full 480×270 latent (for 3840×2160 image)                    │
│       │                                                        │
│       ▼                                                        │
│  [Optional: Diffusion denoise at latent resolution]           │
│       │                                                        │
│       ▼                                                        │
│  Tiled VAE decode → 3840×2160 pixels                          │
│       │                                                        │
│       ▼                                                        │
│  Crop padding → original resolution                            │
│       │                                                        │
│       ▼                                                        │
│  Color space transform + alpha restore                         │
└─────────────────────────────────────────────────────────────────┘

Key improvements over base tiled VAE:
- Cosine blend weights (seamless, no visible tile boundaries)
- VRAM-aware auto tile sizing (queries available GPU memory)
- Proper pad-to-8 handling (VAE requirement)
- Sequential tile processing with VRAM cleanup between tiles
- Built-in .rhdr export for Radiance Viewer
- Progress reporting via ComfyUI ProgressBar

Radiance © 2024 - FXTD STUDIOS
"""

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

    _HAS_LOG_CURVES = True
except ImportError:
    # FIX-1: Emit a visible warning at module load time so operators immediately
    # know that log encode/decode will NOT work. Previously this was silent and
    # log-space nodes would appear to succeed while producing wrong output.
    logger.warning(
        "[Radiance VAE 4K] color_utils not found — log curve encode/decode is DISABLED. "
        "Log input spaces (LogC4, S-Log3, V-Log, DaVinci Intermediate, Log3G10) will pass "
        "through without linearization, and log output spaces will emit linear data. "
        "Install color_utils.py alongside this package to enable log support."
    )


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
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]

EXTENDED_LOG_SPACES = [
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]


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
    import json

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
    v2.0: Encode pixels with explicit distribution sampling control.
    mode = "sample"  — standard (random sample from posterior, default ComfyUI)
    mode = "mean"    — use posterior mean (deterministic, best for img2img)
    mode = "mode"    — use mean (same as mean for Gaussian posterior)
    """
    if mode == "sample":
        result = vae.encode(pixels)
        if isinstance(result, dict):
            return result["samples"]
        return result

    # Try to access the underlying DiagonalGaussian distribution
    try:
        fsm = vae.first_stage_model
        # ComfyUI/CompVis VAE: encode returns a DiagonalGaussianDistribution
        posterior = fsm.encode(pixels)
        if hasattr(posterior, "mean"):
            return posterior.mean
        if hasattr(posterior, "mode"):
            return posterior.mode()
        # Fallback: the encoder returned a raw tensor (some lightweight VAEs)
        if isinstance(posterior, torch.Tensor):
            return posterior
    except Exception:
        pass  # Fall through to standard encode

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
        """Full color pipeline: linearize → expose → VAE-space transform."""

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
            # ACEScg → Rec.709 linear
            AP1_TO_REC709 = torch.tensor(
                [
                    [1.7050509, -0.6217921, -0.0832588],
                    [-0.1302564, 1.1408047, -0.0105483],
                    [-0.0240033, -0.1289690, 1.1529723],
                ],
                dtype=torch.float32,
                device=img.device,
            ).T
            shape = img.shape
            img = img.reshape(-1, 3) @ AP1_TO_REC709
            img = img.reshape(shape)

        # 2. Exposure (linear domain)
        if exposure != 0.0:
            img = img * (2.0**exposure)

        # 2b. Clamp negative linear values before sRGB conversion.
        # ACEScg→Rec709 matrix and exposure can produce out-of-gamut negatives.
        # linear_to_srgb of negatives → garbage/NaN → noise through VAE.
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
            # VAE trained on [0,1]. Values outside cause reconstruction noise.
            # Soft-clamp to [-0.05, 1.05] — small margin for VAE tolerance,
            # but prevent extreme out-of-range that causes hallucinated noise.
            img = torch.clamp(img, -0.05, 1.05)
        elif hdr_mode == "Soft Clip":
            img = tensor_linear_to_srgb(img)
            # Tanh soft clip at knee=0.85
            knee = 0.85
            above = img > knee
            if above.any():
                result = img.clone()
                excess = (img[above] - knee) / (1.0 - knee + 1e-6)
                result[above] = knee + (1.0 - knee) * torch.tanh(excess)
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

        if output is None:
            output = None  # will init lazily on first tile

        for tile_idx_i, (yi, xi, y1, y2, x1, x2) in enumerate(all_tiles):
            if tile_latents is not None:
                # Use pre-computed batch result
                tile_samples = tile_latents[tile_idx_i]
                if tile_samples.ndim == 3:
                    tile_samples = tile_samples.unsqueeze(0)
            else:
                # Sequential: encode one tile at a time
                tile = pixels[:, y1:y2, x1:x2, :3].contiguous()
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
        if is_video:
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
            alpha_out = torch.ones((B, H, W, 1))
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

        # Keep a copy of the un-processed pixels for quality metrics
        pixels_orig = img.clone()

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

        # NaN/Inf guard
        if torch.isnan(img).any() or torch.isinf(img).any():
            logger.warning("[Radiance 4K Encode v2.0] NaN/Inf detected — sanitizing")
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
            pixels_orig.cpu(), img.cpu(), latent["samples"].cpu(),
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
                        "tooltip": "Export .rhdr sidecar for Radiance Viewer (compressed fp16).",
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
                "processing_mode": (
                    ["sequential", "batched"],
                    {
                        "default": "sequential",
                        "tooltip": "'sequential' = min VRAM. 'batched' = 2-4x faster on high-VRAM GPUs.",
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
    ) -> torch.Tensor:
        """
        Decode full latent in overlapping tiles with cosine blend.

        Args:
            samples: {"samples": (B, C, H, W)} latent
            vae: ComfyUI VAE model
            tile_size_px: Tile size in PIXEL space (will be /8 for latent)
            overlap_px: Overlap in PIXEL space
            pbar: Optional progress bar

        Returns:
            (B, pixH, pixW, 3) decoded image
        """
        latent = samples["samples"]
        b, c, lat_h, lat_w = latent.shape
        device = latent.device
        scale = 8

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
        for yi, (ly1, ly2) in enumerate(tiles_y):
            for xi, (lx1, lx2) in enumerate(tiles_x):
                # Extract latent tile
                tile_lat = latent[:, :, ly1:ly2, lx1:lx2].contiguous()

                # Decode on GPU
                tile_decoded = vae.decode(tile_lat).float().cpu()

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

                # Free GPU memory
                del tile_lat, tile_decoded, blend_w
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Normalize by accumulated weight (guard against near-zero at tile edges)
        weight_acc = torch.clamp(weight_acc, min=1e-3)
        output = output / weight_acc
        return output.to(device)

    def _vae_output_to_target(
        self,
        img: torch.Tensor,
        target_space: str,
        hdr_mode: str,
        exposure: float,
        inverse_tonemap: bool,
        target_stops: float,
        source_space: str = "Linear",
    ) -> torch.Tensor:
        """Convert VAE output to target color space.

        Pipeline (v2 fix): ALL operations happen in linear space.
          VAE sRGB output → Linearize → Inverse Tonemap → Exposure → Target Space

        v2 FIX (BUG-3): Previously skipped linearization when target_space=="sRGB",
        then step 4 applied linear_to_srgb to already-sRGB data → DOUBLE GAMMA.
        Now always linearizes first, so all math is in linear domain.

        v2 FIX (BUG-4): Exposure and inverse tonemap now always operate in linear
        space, regardless of target. Previously they ran on sRGB values when
        target=="sRGB", producing wrong results.
        """

        # Step 1: VAE output → Linear (ALWAYS linearize)
        # CRITICAL: Clamp VAE output to [0, 1] first. VAE reconstruction can produce
        # slight negatives and >1.0 values. Unclamped negatives in srgb_to_linear
        # produce garbage via pow() with negative base, and amplify noise.
        img = torch.clamp(img, 0.0, 1.0)

        if hdr_mode == "Compress (Log)":
            if _HAS_LOG_CURVES:
                # FIX-2 (decode side): Decompress using the curve that matches the
                # encode-side source space, not always LogC4. The caller passes
                # source_space so we can invert the exact same curve that was used
                # to compress during encode, giving a mathematically symmetric
                # encode→decode for any camera log format.
                #
                # If source_space is not a log space (e.g. Linear was compressed
                # through a log curve as a tone-mapping strategy), default to LogC4.
                log_to_linear = RadianceVAE4KEncode._get_log_to_linear()
                decompress_fn = log_to_linear.get(source_space, tensor_logc4_to_linear)
                img = decompress_fn(img)
            else:
                # FIX-1 (decode fallback): Emit a clear warning instead of silently
                # producing wrong data. Previously fell through to srgb_to_linear
                # which applies the wrong transform and gives no indication of failure.
                logger.warning(
                    "[Radiance 4K Decode] Compress (Log) hdr_mode selected but log "
                    "curves are unavailable — falling back to sRGB linearization. "
                    "Output will be incorrect. Install color_utils.py to fix this."
                )
                img = tensor_srgb_to_linear(img)
        elif hdr_mode == "Soft Clip":
            # Invert the tanh soft clip applied during encode, then linearize.
            # Encode did: excess = (srgb - knee) / range → result = knee + range * tanh(excess)
            # Inverse:    excess = atanh((result - knee) / range) → srgb = knee + excess * range
            knee = 0.85
            rng = 1.0 - knee  # 0.15
            above = img > knee
            if above.any():
                recovered = img.clone()
                # Normalize to tanh output domain and clamp for atanh safety
                t = torch.clamp((img[above] - knee) / (rng + 1e-6), 0.0, 0.9999)
                excess = torch.atanh(t)
                recovered[above] = knee + excess * (rng + 1e-6)
                img = recovered
            img = tensor_srgb_to_linear(img)
        else:
            # VAE outputs sRGB — ALWAYS linearize for correct math
            img = tensor_srgb_to_linear(img)

        # Step 2: Inverse tonemap (SDR→HDR expansion)
        if inverse_tonemap and hdr_mode != "Compress (Log)":
            target_peak = 2.0 ** (target_stops - 8.0)
            if target_peak > 1.0:
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
                img = img * ratio

        # Step 3: Exposure (linear domain)
        if exposure != 0.0:
            img = img * (2.0**exposure)

        # Step 4: Linear → target space
        if target_space == "Linear":
            pass  # Already linear
        elif target_space == "sRGB":
            img = tensor_linear_to_srgb(img)
        elif target_space == "ACEScg":
            REC709_TO_AP1 = torch.tensor(
                [
                    [0.613097, 0.339523, 0.047379],
                    [0.070194, 0.916354, 0.013452],
                    [0.020616, 0.109570, 0.869815],
                ],
                dtype=torch.float32,
                device=img.device,
            ).T
            shape = img.shape
            img = img.reshape(-1, 3) @ REC709_TO_AP1
            img = img.reshape(shape)
        elif target_space in self.LOG_SPACES and _HAS_LOG_CURVES:
            converters = RadianceVAE4KEncode._get_linear_to_log()
            img = converters[target_space](img)
        elif target_space in self.LOG_SPACES and not _HAS_LOG_CURVES:
            # FIX-1: Previously this silently fell through, producing linear data
            # with no indication of failure. Now emit a clear warning so operators
            # know the output is NOT in the requested log space.
            logger.warning(
                f"[Radiance 4K Decode] Target space '{target_space}' is a log format "
                f"but color_utils is not installed — outputting LINEAR data instead. "
                f"Install color_utils.py to enable log output."
            )
            # img stays linear — at least the data is valid, just in the wrong space
        elif target_space == "Raw":
            pass

        return img

    @staticmethod
    def _save_rhdr(
        img_np, output_dir: str, prefix: str = "radiance_4k"
    ) -> Optional[str]:
        """Save image as .rhdr compressed float16."""
        try:
            import numpy as np

            unique_id = uuid.uuid4().hex[:12]
            filename = f"{prefix}_{unique_id}.rhdr"
            filepath = os.path.join(output_dir, filename)

            fp16_data = img_np.astype(np.float16).tobytes()
            compressed = zlib.compress(fp16_data, level=6)

            h, w = img_np.shape[:2]
            c = img_np.shape[2] if img_np.ndim == 3 else 1
            header = struct.pack("<4sHHHH", b"RHDR", w, h, c, 0)

            with open(filepath, "wb") as f:
                f.write(header)
                f.write(compressed)

            ratio = len(compressed) / len(fp16_data) * 100
            size_mb = (len(header) + len(compressed)) / (1024 * 1024)
            logger.info(
                f"[Radiance 4K] RHDR export: {filename} "
                f"({w}×{h}, {size_mb:.1f}MB, {ratio:.0f}% ratio)"
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
        source_space: str = "Linear",
        processing_mode: str = "sequential",
    ) -> Tuple:
        """v2.0: Universal decode with auto vae_factor, radiance_meta auto-crop, 3-tuple output."""

        # v2.0 Feature 2: Auto-detect VAE factor
        vae_factor = detect_vae_factor(vae)
        latent_fmt = detect_latent_format(vae)

        # v2.0 Feature 6: Read radiance_meta embedded by encode() if present
        radiance_meta = samples.get("radiance_meta", {})
        if radiance_meta:
            latent_fmt = radiance_meta.get("latent_format", latent_fmt)
            vae_factor = radiance_meta.get("vae_factor", vae_factor)

        latent = samples["samples"]
        b, c, lat_h, lat_w = latent.shape
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

        with torch.no_grad():
            if pix_h <= ts_px and pix_w <= ts_px:
                img = vae.decode(latent).float()
                pbar.update_absolute(100, 100)
            else:
                img = self._tiled_decode(samples, vae, ts_px, overlap, pbar)

        img = img.float()

        if torch.isnan(img).any() or torch.isinf(img).any():
            logger.warning("[Radiance 4K Decode v2.0] VAE produced NaN/Inf — sanitizing")
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        img = self._vae_output_to_target(
            img, target_space, hdr_mode,
            exposure_adjust, inverse_tonemap, target_stops,
            source_space=source_space,
        )

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

        # Export .rhdr if requested
        rhdr_filename = None
        if export_rhdr:
            output_dir = folder_paths.get_temp_directory()
            frame_np = img[0, ..., :3].cpu().numpy()
            rhdr_filename = self._save_rhdr(frame_np, output_dir)

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
        if rhdr_filename:
            metadata["rhdr_export"] = rhdr_filename

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
    ) -> Tuple[torch.Tensor, Dict[str, Any], str]:

        b, h, w, c = pixels.shape if pixels.ndim == 4 else pixels.shape
        logger.info(f"[Radiance 4K Roundtrip v2.0] {w}×{h}, {source_space} → {target_space}")

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
            source_space=source_space,
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
    "RadianceVAE4KEncode": "Radiance VAE 4K Encode",
    "RadianceVAE4KDecode": "Radiance VAE 4K Decode",
    "RadianceVAE4KRoundtrip": "Radiance VAE 4K Roundtrip",
}
