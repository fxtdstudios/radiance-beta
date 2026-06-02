# ============================================================
# FXTD STUDIOS — Radiance v3.0.0
# nodes_dit_adapter.py  —  DiT Latent Space Adapter
# ============================================================
# Bridges Radiance's HDR color pipeline to Diffusion Transformer
# video models running inside ComfyUI.
#
# Problem this solves
# -------------------
# SD-VAE latents:   [B, 4,  H/8,  W/8]     scale ≈ 0.18215
# LTX-2.x latents:  [B, 128, T/8, H/32, W/32]  scale ≈ 1.0
# HunyuanVideo:     [B, 16,  T/4, H/8,  W/8]   scale ≈ 0.476986
# Wan2.1:           [B, 16,  T/4, H/8,  W/8]   scale ≈ 1.0   (dense DiT)
# Wan2.2:           [B, 16,  T/4, H/8,  W/8]   scale ≈ 1.0   (MoE, 8-expert/2-active, NOT LoRA-compatible with 2.1)
# CogVideoX:        [B, 16,  T,   H/8,  W/8]   scale ≈ 1.15258426
#
# ============================================================

__version__ = "3.1.0"

import json
import math
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Per-model latent specifications
# ---------------------------------------------------------------------------

# Each entry: {channels, spatial_compression, temporal_compression,
#              latent_scale, mean (per-ch or scalar), std (per-ch or scalar)}
# Mean/std are empirical estimates from public model cards and papers.

_MODEL_SPECS: Dict[str, Dict] = {
    "SD-VAE (4ch)": {
        "channels": 4,
        "spatial_compression": 8,
        "temporal_compression": 1,
        "latent_scale": 0.18215,
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std":  [1.0, 1.0, 1.0, 1.0],
        "temporal": False,
        "description": "Stable Diffusion 1.x / 2.x VAE",
    },
    "SDXL-VAE (4ch)": {
        "channels": 4,
        "spatial_compression": 8,
        "temporal_compression": 1,
        "latent_scale": 0.13025,
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std":  [1.0, 1.0, 1.0, 1.0],
        "temporal": False,
        "description": "Stable Diffusion XL VAE",
    },
    "LTX-Video (128ch)": {
        "channels": 128,
        "spatial_compression": 32,
        "temporal_compression": 8,
        "latent_scale": 1.0,
        # Per-channel stats not published; empirical approximations
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Lightricks LTX-2.x / LTX-2.3",
    },
    "HunyuanVideo (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "latent_scale": 0.476986,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Tencent HunyuanVideo",
    },
    "Wan2.1 (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Alibaba Wan2.1 dense DiT",
    },
    "Wan2.2-T2V-14B (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "moe": True,
        "moe_experts": 8,
        "moe_active_experts": 2,
        "description": "Alibaba Wan2.2 T2V 14B MoE — NOT weight-compatible with Wan2.1 LoRAs",
    },
    "Wan2.2-I2V-14B (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "moe": True,
        "moe_experts": 8,
        "moe_active_experts": 2,
        "description": "Alibaba Wan2.2 I2V 14B MoE — NOT weight-compatible with Wan2.1 LoRAs",
    },
    "CogVideoX (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 1,   # CogVideoX encodes each frame independently
        "latent_scale": 1.15258426,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Zhipu AI CogVideoX-5B",
    },
    "Mochi-1 (12ch)": {
        "channels": 12,
        "spatial_compression": 8,
        "temporal_compression": 6,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Genmo Mochi-1",
    },
}

MODEL_NAMES = list(_MODEL_SPECS.keys())


def _get_spec(name: str) -> Dict:
    return _MODEL_SPECS.get(name, _MODEL_SPECS["SD-VAE (4ch)"])


# ===========================================================================
# Registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {}

NODE_DISPLAY_NAME_MAPPINGS = {}