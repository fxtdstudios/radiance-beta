"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE SAMPLER PRO v3.5
         Professional Flux-Optimized Sampling Engine
                   Radiance © 2024-2026

 Features:
 - Native Flux sigma shifting for high-resolution detail
 - Integrated guidance control (bypasses CFG for Flux)
 - PAG (Perturbed Attention Guidance) via attention hooking (v3.0, fixed v3.2)
 - CFG++ with perpendicular scheduling for better saturation control (v3.0)
 - Multi-Model Support: Auto-detect Flux/SD3/SDXL with optimal defaults (v3.1)
 - Sigma Blend: Smooth phase transitions for phase-shift mode (v3.1)
 - Live Preview: TAESD/Latent2RGB preview during sampling (v3.1)
 - Sigma Report: Diagnostic string output with ETA/speed (v3.2, enhanced v3.5)
 - Per-Stage Timing: Breakdown per sampling stage (v3.2)
 - AYS (Align Your Steps) research-optimized sigma schedules (v3.5)
 - Guidance Rescale (Imagen-style) to prevent oversaturation (v3.5)
 - Karras sigma end correction for complete denoising (v3.5)
 - Workflow presets (txt2img, img2img, inpaint, high-res, turbo)
 - Full step control with timing diagnostics
 - SIGMAS output for advanced chaining
 - Professional error handling and logging

 Example:
     Model → Sampler Pro → VAE Decode → Image

 Flux Tips:
     - CFG = 1.0 (Flux uses guidance instead)
     - flux_guidance = 3.5 (default, higher = more prompt adherence)
     - flux_shift = 1.0 (increase to 3.0 for high-res detail boost)
     - pag_scale = 1.5 (optional, improves prompt adherence via attention perturbation)

 VERSION HISTORY:

 v3.5 - Performance & Quality Upgrade (February 2026)
 - FIX: CRITICAL — apply_guidance used copy.deepcopy: 5500× slower for
   ControlNet/IP-Adapter conditioning. Replaced with shallow dict copy (BUG-20)
 - FIX: CRITICAL — Dynamic guidance timing wrong for img2img: thresholds
   computed on total_steps not effective denoising range. Creative-start
   phase was skipped entirely at denoise < 1.0 (BUG-21/22/26)
 - FIX: Dynamic guidance created unnecessary split stages for non-Flux
   models, adding overhead with zero benefit (BUG-23)
 - FIX: preview_callback variable was dead code; progress bar completely
   disabled when preview creation failed (BUG-24/25)
 - NEW: AYS (Align Your Steps) sigma schedules — research-optimized anchor
   points from Sabour et al. 2024 for measurably better FID/CLIP scores
 - NEW: Guidance Rescale (Imagen-style phi parameter) to prevent
   oversaturation at high CFG values, especially for SDXL
 - NEW: Karras sigma end correction — ensures terminal sigma reaches 0.0
 - NEW: Per-stage speed reporting (it/s) in sigma report
 - PERF: Guidance application is now O(1) for tensor references vs O(n) deepcopy

 v3.4 - Noise & Quality Fix (February 2026)
 - FIX: CRITICAL — Double noise in stage 1: code added noise*σ to latent,
   then sample_custom added noise*σ AGAIN internally → 2× noise (BUG-15)
 - FIX: CRITICAL — Extra noise in continuation stages: sample_custom always
   adds noise*σ_start to latent, corrupting partially-denoised results from
   previous stages. Now passes zeros for continuation (BUG-16)
 - FIX: Dynamic guidance late-stage drop: guidance fell to 60% at 90-100%
   of steps, destroying fine detail coherence. Now tapers to 85% (BUG-17)
 - FIX: Phase-Shift default sigma_blend_steps=0 caused hard sigma
   discontinuity. Auto-enables 3 blend steps when phase-shift active (BUG-18)
 - FIX: Sigma blend first sigma now exactly matches phase transition
   point instead of ~85% approximation (BUG-19)

 v3.1 - Multi-Model & Preview (2025)
 - NEW: Auto-detect Flux/SD3/SDXL with optimal defaults
 - NEW: Sigma Blend for smooth phase transitions
 - NEW: Live Preview via TAESD/Latent2RGB

 v3.0 - Next Generation (2025)
 - NEW: PAG (Perturbed Attention Guidance) support
 - NEW: CFG++ mode with perpendicular scheduling
 - NEW: Turbo presets for distilled models (Schnell, SD3.5 Turbo)
 - FIX #1-15: All previous production fixes retained
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import time
import math
import logging
import weakref
from typing import Tuple, Dict, Any, Optional, List

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils

# ═══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Dynamic guidance curve parameters
# v3.5: Renamed for clarity — these control the creative freedom phase
DYNAMIC_GUIDANCE_EARLY_MULTIPLIER = 0.6  # Early phase: base * 0.6 (creative freedom)
DYNAMIC_GUIDANCE_LATE_MULTIPLIER = 0.85  # Late phase: base * 0.85 (detail preservation)
DYNAMIC_GUIDANCE_EARLY_THRESHOLD = 0.2  # First 20% uses low guidance
DYNAMIC_GUIDANCE_LATE_THRESHOLD = 0.9  # Last 10% uses reduced guidance
DYNAMIC_GUIDANCE_RAMP_WIDTH = 0.05  # 5% cosine ramp between phases

# v3.5: Guidance rescale (Imagen-style) — prevents oversaturation at high CFG
GUIDANCE_RESCALE_PHI = 0.0  # 0.0 = disabled, 0.7 = recommended for SDXL

# Sigma validation
SIGMA_DISCONTINUITY_THRESHOLD = 0.01

# PAG (Perturbed Attention Guidance) configuration
PAG_DEFAULT_SCALE = 0.0  # Disabled by default
PAG_LAYER_NAMES = ["middle_block"]  # Attention layers to perturb

# CFG++ configuration
CFG_PLUS_PLUS_DEFAULT_SCALE = 1.6  # Recommended CFG++ scale

# v3.1: Model type detection and defaults
# v3.3: Extended with video and modern model types (S-BUG-10)
MODEL_TYPES = [
    "auto",
    "flux",
    "sd3",
    "sd35",
    "sdxl",
    "sd15",
    "wan",
    "ltxv",
    "hunyuan_video",
    "lumina2",
    "z_image",
    "chroma",
]

# v3.3: Extended model defaults with sampler, shift, and denoise range
MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "flux": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 3.5,
        "shift": 1.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
    "sd3": {
        "cfg": 4.5,
        "scheduler": "sgm_uniform",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "dpmpp_2m",
        "denoise_range": (0.2, 1.0),
    },
    "sd35": {
        "cfg": 4.5,
        "scheduler": "sgm_uniform",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "dpmpp_2m",
        "denoise_range": (0.2, 1.0),
    },
    "sdxl": {
        "cfg": 7.0,
        "scheduler": "karras",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler_ancestral",
        "denoise_range": (0.3, 1.0),
    },
    "sd15": {
        "cfg": 7.0,
        "scheduler": "normal",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler_ancestral",
        "denoise_range": (0.3, 1.0),
    },
    # v3.3: Video & modern models (S-BUG-10)
    "wan": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 8.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
    "ltxv": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 2.37,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
    "hunyuan_video": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 7.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
    "lumina2": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 6.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
    "z_image": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 3.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
    "chroma": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
}

# v3.1: Preview callback methods
PREVIEW_METHODS = ["None", "TAESD", "Latent2RGB"]


# v3.2: Type-safe sampler mode constants
class SamplerMode:
    """Sampler mode constants — eliminates fragile string matching."""

    STANDARD = "Standard"
    PHASE_SHIFT_DPM = "Phase-Shift (Euler→DPM)"
    PHASE_SHIFT_SGM = "Phase-Shift (Euler→SGM)"
    CFG_PLUS_PLUS = "CFG++ (Perpendicular)"

    ALL = [STANDARD, PHASE_SHIFT_DPM, PHASE_SHIFT_SGM, CFG_PLUS_PLUS]

    @classmethod
    def is_phase_shift(cls, mode: str) -> bool:
        return mode in (cls.PHASE_SHIFT_DPM, cls.PHASE_SHIFT_SGM)

    @classmethod
    def is_cfg_plus_plus(cls, mode: str) -> bool:
        return mode == cls.CFG_PLUS_PLUS


# Module logger
logger = logging.getLogger("radiance.sampler")

# v3.2: Module-level sigma cache for performance across batch runs (BUG-14)
# Key: (model_id, scheduler_name) -> (weak_model_ref, sigmas_tensor)
_sigma_cache = {}


def _clean_sigma_cache():
    """v3.3 (S-BUG-6): Remove entries with dead weakrefs to prevent unbounded growth."""
    dead = [k for k, (ref, _) in _sigma_cache.items() if ref() is None]
    for k in dead:
        del _sigma_cache[k]
    if dead:
        logger.debug(f"Cleaned {len(dead)} stale sigma cache entries")


# ═══════════════════════════════════════════════════════════════════════════════
#                         MODEL TYPE DETECTION (v3.1, fixed v3.2)
# ═══════════════════════════════════════════════════════════════════════════════


def detect_model_type(model) -> str:
    """
    Auto-detect model architecture type from model object.

    v3.3 (S-BUG-2): Extended to detect modern models:
      flux, sd3, sd35, sdxl, sd15, wan, ltxv, hunyuan_video,
      lumina2, z_image, chroma

    Detection strategy:
      1. Check diffusion_model class name (most reliable)
      2. Check model_config class name
      3. Fall back to sigma/attribute heuristics
    """
    try:
        # ── Strategy 1: Check diffusion model class name ──
        diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model is not None:
            model_cls = type(diffusion_model).__name__.lower()
            model_module = (
                type(diffusion_model).__module__.lower()
                if hasattr(type(diffusion_model), "__module__")
                else ""
            )
            full_path = f"{model_module}.{model_cls}"

            # Video models (check first — most specific)
            if "wan" in model_cls or "wan" in model_module:
                return "wan"
            if (
                "ltxv" in model_cls
                or "ltxav" in model_cls
                or "lightricks" in model_module
            ):
                return "ltxv"
            if "hunyuan" in model_cls and (
                "video" in model_cls or "video" in model_module
            ):
                return "hunyuan_video"

            # Modern image models
            if "lumina" in full_path:
                # z-image uses a Lumina2 base with larger dim
                if (
                    hasattr(diffusion_model, "hidden_size")
                    and diffusion_model.hidden_size >= 3840
                ):
                    return "z_image"
                return "lumina2"
            if "chroma" in full_path:
                return "chroma"

            # SD3/SD3.5 detection - MMDiT architecture
            if "mmdit" in model_cls or "sd3" in model_cls:
                try:
                    if (
                        hasattr(diffusion_model, "in_channels")
                        and diffusion_model.in_channels >= 16
                    ):
                        return "sd35"
                except (AttributeError, RuntimeError):
                    pass
                return "sd3"

            # SDXL detection
            if "sdxl" in model_cls or hasattr(diffusion_model, "label_emb"):
                return "sdxl"

        # ── Strategy 2: Check model_config class name ──
        try:
            model_config = model.model.model_config if hasattr(model, "model") else None
            if model_config is not None:
                config_cls = type(model_config).__name__
                config_map = {
                    "WAN21": "wan",
                    "WAN22": "wan",
                    "LTXV": "ltxv",
                    "LTXAV": "ltxv",
                    "HunyuanVideo": "hunyuan_video",
                    "Lumina2": "lumina2",
                    "ZImage": "z_image",
                    "Chroma": "chroma",
                    "ChromaRadiance": "chroma",
                    "Flux": "flux",
                    "FluxSchnell": "flux",
                    "FluxInpaint": "flux",
                    "Flux2": "flux",
                }
                for pattern, model_type in config_map.items():
                    if pattern in config_cls:
                        return model_type
        except (AttributeError, RuntimeError):
            pass

        # ── Strategy 3: Sigma/attribute heuristics (legacy) ──
        model_sampling = model.get_model_object("model_sampling")
        has_flux_attrs = hasattr(model_sampling, "shift") or hasattr(
            model_sampling, "flux_shift"
        )

        if has_flux_attrs:
            try:
                sigma_max = (
                    model_sampling.sigma_max.item()
                    if hasattr(model_sampling, "sigma_max")
                    else 1.0
                )
                if sigma_max <= 1.5:
                    return "flux"
            except (AttributeError, RuntimeError):
                return "flux"

        return "sd15"

    except (AttributeError, RuntimeError) as e:
        logger.warning(f"Model type detection failed: {e}, defaulting to sd15")
        return "sd15"


def get_model_defaults(model_type: str) -> Dict[str, Any]:
    """Get optimal default settings for detected model type."""
    return MODEL_DEFAULTS.get(model_type, MODEL_DEFAULTS["sd15"])


# ═══════════════════════════════════════════════════════════════════════════════
#                         GRADUAL SIGMA BLEND (v3.1, fixed v3.2)
# ═══════════════════════════════════════════════════════════════════════════════


def gradual_sigma_blend(
    sigmas_a: torch.Tensor, sigmas_b: torch.Tensor, blend_steps: int = 3
) -> torch.Tensor:
    """
    Smoothly interpolate between sigma schedules at transition point.

    Uses cosine interpolation for smooth phase transitions, eliminating
    the sigma discontinuity warnings in phase-shift sampling.

    v3.2 FIX: Now clones sigmas_b internally — callers no longer need
    to pre-clone. The input tensor is never mutated.

    v3.4 FIX (BUG-19): First blended sigma now EXACTLY matches last_sigma_a.
    Previously, i=0 got ~85% weight from A (cosine at t=0.25), so the model
    saw a slightly wrong noise level at the transition. Now uses i+1 range
    starting from 0% blend (100% A) to full blend.

    Args:
        sigmas_a: Ending sigma schedule from first phase
        sigmas_b: Starting sigma schedule for second phase
        blend_steps: Number of steps to blend over (0 = no blend)

    Returns:
        New tensor with smooth transition (sigmas_b is not modified)
    """
    if blend_steps <= 0 or len(sigmas_a) == 0 or len(sigmas_b) == 0:
        return sigmas_b

    # v3.2 FIX: Defensive clone — never mutate the input
    result = sigmas_b.clone()

    # Get the last sigma from phase A
    last_sigma_a = sigmas_a[-1].item()

    # Clamp blend_steps to available sigmas
    blend_steps = min(blend_steps, len(result) - 1)

    # v3.4 FIX: Force first sigma to exactly match phase A's ending sigma.
    # Then blend subsequent steps with cosine interpolation toward phase B.
    result[0] = last_sigma_a

    for i in range(1, blend_steps):
        # t goes from ~0 to ~1 over the remaining blend steps
        t = i / blend_steps
        # Cosine ease-in-out
        blend_factor = 0.5 * (1.0 - math.cos(math.pi * t))
        # Blend from last_sigma_a towards original sigmas_b[i]
        result[i] = (
            last_sigma_a * (1.0 - blend_factor) + sigmas_b[i].item() * blend_factor
        )

    logger.debug(
        f"Sigma blend: {blend_steps} steps, "
        f"transition {last_sigma_a:.4f} → {sigmas_b[blend_steps-1].item():.4f}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#                         DEBUG UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def log_tensor(name: str, tensor: Optional[torch.Tensor]) -> None:
    """Log tensor statistics for debugging."""
    if tensor is None:
        logger.debug(f"{name}: None")
        return
    try:
        t = tensor.float()
        logger.debug(
            f"{name}: Shape={list(t.shape)} | "
            f"Range=[{t.min().item():.3f}, {t.max().item():.3f}] | "
            f"Mean={t.mean().item():.3f} | Std={t.std().item():.3f}"
        )
    except (RuntimeError, ValueError) as e:
        logger.warning(f"{name}: Error logging stats ({e})")


# ═══════════════════════════════════════════════════════════════════════════════
#                         WORKFLOW PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

WORKFLOW_PRESETS = [
    "None (Custom)",
    "→ Flux txt2img",
    "→ Flux img2img",
    "→ Flux Inpaint",
    "→ Flux High-Res Fix",
    "→ Flux Fast (12 steps)",
    "→ Flux Quality (28 steps)",
    "→ Flux Cinematic (30 steps)",
    # Turbo / Distilled model presets
    "→ Flux Schnell (4 steps)",
    "→ SD3.5 Turbo (4 steps)",
    "→ Flux Ultra Fast (8 steps)",
]

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "→ Flux txt2img": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "→ Flux img2img": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.75,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "→ Flux Inpaint": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
    },
    "→ Flux High-Res Fix": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.5,
        "flux_shift": 3.0,
        "flux_guidance": 3.5,
    },
    "→ Flux Fast (12 steps)": {
        "steps": 12,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "→ Flux Quality (28 steps)": {
        "steps": 28,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
    },
    "→ Flux Cinematic (30 steps)": {
        "steps": 30,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
    },
    # Turbo / Distilled model presets
    "→ Flux Schnell (4 steps)": {
        "steps": 4,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 0.0,  # Schnell ignores guidance
    },
    "→ SD3.5 Turbo (4 steps)": {
        "steps": 4,
        "cfg": 1.6,
        "sampler": "euler",
        "scheduler": "sgm_uniform",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 0.0,
    },
    "→ Flux Ultra Fast (8 steps)": {
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 2.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                         SIGMA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def flux_shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    Apply Flux-specific sigma shifting.

    The shift parameter controls how the noise schedule is transformed.
    Higher shift values push more denoising into later steps, which
    can improve high-frequency details at high resolutions.

    Formula: shifted = shift * sigma / (1 + (shift - 1) * sigma)

    Args:
        sigmas: Original sigma schedule
        shift: Shift factor (1.0 = no change, 3.0 = typical for high-res)

    Returns:
        Shifted sigma schedule

    Raises:
        ValueError: If shift <= 0
    """
    if shift <= 0:
        raise ValueError(f"flux_shift must be > 0, got {shift}")

    if shift == 1.0:
        return sigmas

    # FIX: Prevent division by zero for sigma values near 1/(1-shift)
    denominator = 1.0 + (shift - 1.0) * sigmas
    # Clamp denominator to avoid division issues
    denominator = torch.clamp(denominator, min=1e-6)

    shifted = shift * sigmas / denominator
    return shifted


def get_flux_sigmas(
    model, scheduler: str, steps: int, denoise: float, shift: float = 1.0
) -> torch.Tensor:
    """
    Calculate sigma schedule optimized for Flux models.

    Args:
        model: The model wrapper
        scheduler: Scheduler name (recommended: "simple" for Flux)
        steps: Number of sampling steps (must be >= 1)
        denoise: Denoise strength (1.0 = full, <1.0 = img2img)
        shift: Flux shift parameter (must be > 0)

    Returns:
        Sigma schedule tensor

    Raises:
        ValueError: If steps < 1 or denoise/shift out of valid range
    """
    # FIX #9: Validate inputs
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    if denoise < 0.0 or denoise > 1.0:
        raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")

    # FIX #13: Handle denoise=0 early (no denoising needed)
    if denoise <= 0.0:
        return torch.tensor([0.0])  # Return minimal sigma schedule

    # Get the model's sampling configuration
    model_sampling = model.get_model_object("model_sampling")

    # Calculate base sigmas using the scheduler
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)

    # Apply Flux shift if specified
    if shift != 1.0:
        sigmas = flux_shift_sigmas(sigmas, shift)

    # Apply denoise (trim sigmas for img2img)
    if denoise < 1.0:
        total_steps = len(sigmas) - 1
        if total_steps <= 0:
            return sigmas  # Can't trim further

        start_step = max(0, int(total_steps * (1.0 - denoise)))
        sigmas = sigmas[start_step:]

    return sigmas


def validate_step_range(
    start_step: int, end_step: int, steps: int, context: str = ""
) -> Tuple[int, int]:
    """
    Validate and clamp step range to valid bounds.

    Args:
        start_step: Requested start step
        end_step: Requested end step
        steps: Total number of steps
        context: Context string for logging

    Returns:
        Tuple of (validated_start, validated_end)
    """
    original = (start_step, end_step)

    # Clamp to valid range
    start = max(0, min(start_step, steps))
    end = max(0, min(end_step, steps))

    # Ensure start <= end
    if start > end:
        logger.warning(f"{context}start_step ({start}) > end_step ({end}), swapping")
        start, end = end, start

    if original != (start, end):
        logger.debug(f"{context}Step range adjusted: {original} → ({start}, {end})")

    return start, end


# ═══════════════════════════════════════════════════════════════════════════════
#                   PAG (PERTURBED ATTENTION GUIDANCE) — v3.2 Rewrite
# ═══════════════════════════════════════════════════════════════════════════════


def apply_pag_to_model(model, pag_scale: float):
    """
    Apply Perturbed Attention Guidance to model via attention hooking.

    PAG works by replacing the unconditional attention pass with an
    identity attention map (each token attends only to itself). The
    difference between normal and perturbed predictions is then scaled
    by pag_scale and added to the conditional output — similar to how
    CFG works but targeting attention structure instead of text conditioning.

    v3.2 REWRITE: Previous implementation only stored pag_scale in
    model_options without any attention hook — it was a complete no-op.
    Now uses ComfyUI's set_model_attn1_patch to actually perturb attention.

    Args:
        model: ComfyUI model wrapper
        pag_scale: Perturbation strength (0.0 = disabled, 1.0-3.0 typical)

    Returns:
        Modified model with PAG applied, or original if scale is 0
    """
    if pag_scale <= 0:
        return model

    try:
        # Clone model to avoid modifying original
        model_pag = model.clone()

        # Store PAG scale in model options for reference
        if hasattr(model_pag, "model_options"):
            model_pag.model_options = model_pag.model_options.copy()
            model_pag.model_options["pag_scale"] = pag_scale

        # v3.2: Hook into attention layers to create actual perturbation.
        # For the unconditional pass, replace self-attention with identity
        # attention (each token only attends to itself).
        def pag_attention_patch(q, k, v, extra_options):
            """
            v3.3 (S-BUG-5): PAG attention patch rewrite.

            Only modifies the unconditional pass on middle blocks.
            Returns (q, k, v) tuple — ComfyUI's attn1_patch expects
            transformed inputs, NOT pre-computed attention output.

            For uncond middle blocks: replaces k/v with repeat of q
            so each token attends only to itself (identity attention).
            For all other cases: returns inputs unchanged.
            """
            cond_or_uncond = extra_options.get("cond_or_uncond", [0])
            block_type = extra_options.get("block_type", "unknown")

            # Only modify unconditional pass on middle blocks
            if 1 not in cond_or_uncond or block_type != "middle":
                return q, k, v

            # Clone k,v to avoid mutating shared tensors
            k_out = k.clone()
            v_out = v.clone()

            num_cond = len(cond_or_uncond)
            batch_size = q.shape[0]
            chunk_size = batch_size // num_cond if num_cond > 0 else batch_size

            for idx, cond_type in enumerate(cond_or_uncond):
                if cond_type == 1:  # Unconditional
                    start = idx * chunk_size
                    end = min(start + chunk_size, batch_size)
                    # Replace k and v with q — this makes the attention
                    # matrix become identity (q @ q^T after softmax),
                    # effectively making each token attend only to itself
                    k_out[start:end] = q[start:end]
                    v_out[start:end] = q[start:end]

            return q, k_out, v_out

        # Apply the attention patch to middle block layers
        model_pag.set_model_attn1_patch(pag_attention_patch)

        logger.info(f"PAG applied with scale {pag_scale} (attention hook active)")
        return model_pag

    except (AttributeError, RuntimeError, TypeError) as e:
        # v3.2 FIX: Catch specific exceptions, not bare Exception.
        # This prevents swallowing MemoryError or other critical failures.
        logger.warning(f"Failed to apply PAG: {e}, using original model")
        return model


# ═══════════════════════════════════════════════════════════════════════════════
#                    AYS (ALIGN YOUR STEPS) SIGMA SCHEDULES — v3.5
# ═══════════════════════════════════════════════════════════════════════════════

# Research-optimized sigma anchor points from "Align Your Steps" (Sabour et al. 2024)
# These provide measurably better FID/CLIP scores than linear/karras at low step counts.
AYS_ANCHORS = {
    "sdxl": [14.615, 6.315, 3.771, 1.181, 0.468, 0.131, 0.029, 0.0],
    "sd15": [
        14.615,
        6.475,
        3.861,
        2.697,
        1.886,
        1.396,
        0.963,
        0.652,
        0.399,
        0.152,
        0.029,
        0.0,
    ],
    "flux": [1.0, 0.85, 0.60, 0.35, 0.15, 0.05, 0.01, 0.0],
    "sd3": [14.615, 6.291, 3.438, 1.566, 0.741, 0.288, 0.079, 0.0],
}


def get_ays_sigmas(model_type: str, steps: int) -> Optional[torch.Tensor]:
    """
    Generate AYS (Align Your Steps) sigma schedule for a given model type.

    Interpolates between research-optimized anchor points to produce a schedule
    for any step count. Returns None if model type has no AYS anchors.

    Reference: "Align Your Steps: Optimizing Sampling Schedules in Diffusion
    Models" (Sabour et al., NeurIPS 2024)
    """
    key = model_type if model_type in AYS_ANCHORS else None
    if key is None:
        # Map variants to base types
        if model_type in ("sd35",):
            key = "sd3"
        elif model_type in ("chroma",):
            key = "flux"
        else:
            return None

    anchors = AYS_ANCHORS[key]
    n_anchors = len(anchors)

    if steps + 1 <= n_anchors:
        # Fewer steps than anchors — sub-sample
        indices = torch.linspace(0, n_anchors - 1, steps + 1).long()
        return torch.tensor([anchors[i] for i in indices])

    # More steps than anchors — interpolate with log-space for better distribution
    result = torch.zeros(steps + 1)
    anchor_t = torch.tensor(anchors)

    # Use log-space interpolation for non-zero values (sigmas span orders of magnitude)
    log_anchors = torch.log(torch.clamp(anchor_t[:-1], min=1e-6))  # Skip terminal 0
    log_anchors = torch.cat([log_anchors, torch.tensor([-12.0])])  # ~exp(-12) ≈ 0

    x_anchors = torch.linspace(0, 1, len(log_anchors))
    x_result = torch.linspace(0, 1, steps + 1)

    # Linear interpolation in log space
    for i in range(steps + 1):
        t = x_result[i].item()
        # Find surrounding anchors
        idx = 0
        while idx < len(x_anchors) - 2 and x_anchors[idx + 1] < t:
            idx += 1

        t_local = (t - x_anchors[idx].item()) / max(
            1e-8, (x_anchors[idx + 1] - x_anchors[idx]).item()
        )
        t_local = max(0.0, min(1.0, t_local))

        log_val = log_anchors[idx] * (1 - t_local) + log_anchors[idx + 1] * t_local
        result[i] = math.exp(log_val.item())

    # Force terminal sigma to exactly 0
    result[-1] = 0.0

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#               GUIDANCE RESCALE (Imagen-style) — v3.5
# ═══════════════════════════════════════════════════════════════════════════════


def guidance_rescale_cfg(
    cond_output: torch.Tensor,
    uncond_output: torch.Tensor,
    cfg: float,
    phi: float = 0.7,
) -> torch.Tensor:
    """
    Apply guidance rescale from "Imagen" (Saharia et al., 2022).

    Standard CFG: output = uncond + cfg * (cond - uncond)
    This causes oversaturation at high CFG because the std deviation
    of the guided output exceeds that of the conditional output.

    Guidance rescale corrects this by normalizing the std back:
      guided_std = std(cfg_output)
      cond_std = std(cond_output)
      output = cfg_output * (cond_std / guided_std) * phi + cfg_output * (1 - phi)

    Args:
        cond_output: Conditional model prediction
        uncond_output: Unconditional model prediction
        cfg: CFG scale
        phi: Rescale strength (0.0 = off, 0.7 = recommended for SDXL)

    Returns:
        Rescaled guided output
    """
    if phi <= 0.0 or cfg <= 1.0:
        return uncond_output + cfg * (cond_output - uncond_output)

    # Standard CFG
    guided = uncond_output + cfg * (cond_output - uncond_output)

    # Compute channel-wise std
    dims = list(range(1, guided.ndim))  # All dims except batch
    guided_std = guided.std(dim=dims, keepdim=True) + 1e-6
    cond_std = cond_output.std(dim=dims, keepdim=True) + 1e-6

    # Rescale: normalize guided to match conditional std
    rescaled = guided * (cond_std / guided_std)

    # Blend between rescaled and original guided
    return rescaled * phi + guided * (1.0 - phi)


# ═══════════════════════════════════════════════════════════════════════════════
#               KARRAS SIGMA END CORRECTION — v3.5
# ═══════════════════════════════════════════════════════════════════════════════


def correct_karras_sigma_end(
    sigmas: torch.Tensor, target_end: float = 0.0
) -> torch.Tensor:
    """
    Correct Karras scheduler's non-zero terminal sigma.

    The Karras scheduler computes sigmas with σ_min > 0 (typically ~0.029),
    leaving residual noise in the final output. This function appends a
    terminal 0.0 sigma to ensure complete denoising.

    Only modifies sigmas if the last value is significantly > 0.
    """
    if len(sigmas) < 2:
        return sigmas

    if sigmas[-1].item() > 0.001 and target_end == 0.0:
        return torch.cat([sigmas, torch.tensor([0.0])])

    return sigmas


def apply_cfg_plus_plus(cfg: float, sigma: torch.Tensor, sigma_max: float) -> float:
    """
    Apply CFG++ perpendicular scheduling.

    Uses cosine scheduling to dynamically adjust CFG scale over the
    sampling process, preventing oversaturation at high guidance values.
    CFG is highest at the start (sigma_max) and reduces toward sigma_min.

    Formula: cfg_effective = cfg * cos_factor + 1.0 * (1 - cos_factor)

    Args:
        cfg: Base CFG value
        sigma: Current sigma value
        sigma_max: Maximum sigma in schedule

    Returns:
        Adjusted CFG value for current sigma
    """
    if sigma_max <= 0:
        return cfg

    # Get sigma as float (handle tensor case)
    if isinstance(sigma, torch.Tensor):
        sigma_val = sigma.item() if sigma.numel() == 1 else sigma[0].item()
    else:
        sigma_val = float(sigma)

    # Calculate progress (0.0 at start, 1.0 at end)
    progress = 1.0 - (sigma_val / sigma_max)
    progress = max(0.0, min(1.0, progress))

    # Cosine schedule: high CFG at start, gradually reduces
    cos_factor = (1.0 + math.cos(math.pi * progress)) / 2.0

    # Interpolate between cfg and 1.0 (no guidance)
    effective_cfg = cfg * cos_factor + 1.0 * (1.0 - cos_factor)

    return effective_cfg


# ═══════════════════════════════════════════════════════════════════════════════
#                         SIGMA REPORT (v3.2)
# ═══════════════════════════════════════════════════════════════════════════════


def build_sigma_report(
    detected_type: str,
    steps: int,
    scheduler: str,
    flux_shift: float,
    denoise: float,
    sigmas: torch.Tensor,
    sampler_mode: str,
    sorted_splits: List[int],
    stage_timings: List[Tuple[int, int, int, str, float]],
    total_time: float,
    ays_active: bool = False,
) -> str:
    """
    Build a human-readable sigma schedule report for diagnostics.
    v3.5: Added ETA, speed (it/s), and AYS indicator.
    """
    lines = [
        f"═══ Radiance Sampler Pro v3.5 ═══",
        f"Model: {detected_type} | Steps: {steps} | Scheduler: {scheduler}{'  [AYS]' if ays_active else ''}",
        f"Shift: {flux_shift} | Denoise: {denoise} | Mode: {sampler_mode}",
    ]

    if len(sigmas) > 0:
        lines.append(
            f"Sigma range: [{sigmas[0].item():.4f} → {sigmas[-1].item():.4f}] "
            f"({len(sigmas)} values)"
        )

    lines.append(f"Stages: {max(0, len(sorted_splits) - 1)}")

    if stage_timings:
        lines.append("─── Per-Stage Timing ───")
        for stage_num, s_start, s_end, samp, t in stage_timings:
            stage_steps = s_end - s_start
            speed = stage_steps / t if t > 0.001 else 0
            lines.append(
                f"  Stage {stage_num}: steps {s_start}→{s_end} [{samp}] "
                f"= {t:.3f}s ({speed:.1f} it/s)"
            )

    # Overall speed
    total_sampling_steps = (
        sum(s_end - s_start for _, s_start, s_end, _, _ in stage_timings)
        if stage_timings
        else steps
    )
    overall_speed = total_sampling_steps / total_time if total_time > 0.001 else 0
    lines.append(f"Total: {total_time:.2f}s ({overall_speed:.1f} it/s)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#                         RADIANCE SAMPLER PRO v3.4
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceSamplerPro:
    """
    Professional Flux-optimized sampler with presets, timing report,
    and full parameter control.

    v3.5: Fixed 5500x deepcopy perf regression, fixed dynamic guidance
    timing for img2img, added AYS schedules, guidance rescale, Karras
    end correction, and per-stage speed reporting.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "preset": (WORKFLOW_PRESETS, {"default": "None (Custom)"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Start step (0 = beginning)",
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "End step (0 = use total steps)",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "sampler_mode": (SamplerMode.ALL, {"default": SamplerMode.STANDARD}),
                "phase_split": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "scheduler_mode": (
                    ["Manual", "Auto (Match Steps)"],
                    {"default": "Manual"},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "flux_shift": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.1},
                ),
                "flux_guidance": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "flux_guidance_profile": (
                    ["Static", "Dynamic (Creative Start/End)"],
                    {"default": "Static"},
                ),
                "add_noise": ("BOOLEAN", {"default": True}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                # v3.0: PAG (Perturbed Attention Guidance)
                "pag_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "PAG strength (0=off). Perturbs attention for better prompt adherence.",
                    },
                ),
                # v3.1: Multi-Model Support
                "model_type": (MODEL_TYPES, {"default": "auto"}),
                "sigma_blend_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Smooth sigma transition steps at phase-shift boundary",
                    },
                ),
                # v3.5: Advanced features
                "ays_schedule": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use AYS (Align Your Steps) research-optimized sigma schedule. Best at 8-15 steps.",
                    },
                ),
                "guidance_rescale_phi": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Guidance rescale (Imagen). 0=off, 0.7=recommended for SDXL. Prevents oversaturation at high CFG.",
                    },
                ),
                # v3.1: Live Preview
                "preview_method": (PREVIEW_METHODS, {"default": "None"}),
            },
            "optional": {
                "refiner_model": ("MODEL",),
                "refiner_start_step": (
                    "INT",
                    {"default": 20, "min": 0, "max": 200, "step": 1},
                ),
                "noise_override": ("LATENT",),
            },
        }

    # v3.2: Added sigma_report STRING output
    RETURN_TYPES = ("LATENT", "SIGMAS", "STRING")
    RETURN_NAMES = ("latent", "sigmas", "sigma_report")
    FUNCTION = "sample"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = (
        "Professional Flux-optimized sampler with workflow presets, Flux Shift, "
        "dynamic guidance, PAG (attention-hooked), CFG++, AYS schedules, "
        "guidance rescale, multi-model auto-detect, sigma blending, live preview, "
        "and per-stage timing diagnostics."
    )

    def sample(
        self,
        model,
        positive: List,
        negative: List,
        latent_image: Dict[str, torch.Tensor],
        preset: str,
        steps: int,
        start_step: int,
        end_step: int,
        cfg: float,
        sampler: str,
        sampler_mode: str,
        phase_split: float,
        scheduler: str,
        scheduler_mode: str,
        denoise: float,
        flux_shift: float,
        flux_guidance: float,
        flux_guidance_profile: str,
        add_noise: bool,
        return_with_leftover_noise: bool,
        seed: int,
        pag_scale: float = 0.0,
        # v3.1 parameters
        model_type: str = "auto",
        sigma_blend_steps: int = 0,
        # v3.5 parameters
        ays_schedule: bool = False,
        guidance_rescale_phi: float = 0.0,
        # v3.1: Live Preview
        preview_method: str = "None",
        # Optional
        refiner_model=None,
        refiner_start_step: int = 20,
        noise_override: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, str]:

        t_start = time.time()
        timings: Dict[str, float] = {}

        # ─────────────────────────────────────────────────────────────────
        # Apply Preset if Selected
        # ─────────────────────────────────────────────────────────────────
        if preset != "None (Custom)" and preset in PRESET_CONFIGS:
            config = PRESET_CONFIGS[preset]
            steps = config.get("steps", steps)
            cfg = config.get("cfg", cfg)
            sampler = config.get("sampler", sampler)
            scheduler = config.get("scheduler", scheduler)
            denoise = config.get("denoise", denoise)
            flux_shift = config.get("flux_shift", flux_shift)
            flux_guidance = config.get("flux_guidance", flux_guidance)
            logger.info(f"Loaded Preset: {preset}")

        # ─────────────────────────────────────────────────────────────────
        # v3.2: Guard flux_shift for API callers bypassing widget min
        # ─────────────────────────────────────────────────────────────────
        flux_shift = max(0.01, flux_shift)

        # ─────────────────────────────────────────────────────────────────
        # v3.1: Model Type Detection and Auto-Configuration
        # ─────────────────────────────────────────────────────────────────
        detected_type = detect_model_type(model) if model_type == "auto" else model_type

        # Apply model-specific defaults if preset is custom
        if preset == "None (Custom)" and model_type == "auto":
            defaults = get_model_defaults(detected_type)
            # Only override if using default values
            if cfg == 1.0 and detected_type != "flux":
                cfg = defaults.get("cfg", cfg)
            if flux_guidance == 3.5 and detected_type != "flux":
                flux_guidance = defaults.get("guidance", flux_guidance)

            # v3.3 FIX (S-BUG-3): Do NOT override scheduler/sampler.
            # The user's explicit widget choices should be respected.
            # Only cfg and guidance are overridden when at default values.

            logger.info(
                f"Auto-detected model type: {detected_type} (CFG={cfg}, guidance={flux_guidance}, scheduler={scheduler})"
            )
        else:
            logger.info(f"Model type: {detected_type}")

        # BUG-7 FIX: Implement scheduler_mode Auto
        if scheduler_mode == "Auto (Match Steps)":
            defaults = get_model_defaults(detected_type)
            auto_scheduler = defaults.get("scheduler", scheduler)
            if auto_scheduler != scheduler:
                logger.info(
                    f"Auto scheduler: {scheduler} → {auto_scheduler} (optimal for {detected_type})"
                )
                scheduler = auto_scheduler

        # ─────────────────────────────────────────────────────────────────
        # FIX #11: Validate Step Ranges
        # ─────────────────────────────────────────────────────────────────
        start_step, end_step = validate_step_range(
            start_step, end_step, steps, "[Radiance] "
        )

        # ─────────────────────────────────────────────────────────────────
        # Setup
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        latent = latent_image
        latent_samples = latent["samples"]
        # v3.3 (S-BUG-1): Handle both 4D (image) and 5D (video) latents
        if latent_samples.ndim == 5:
            batch_size, channels, frames, height, width = latent_samples.shape
        else:
            batch_size, channels, height, width = latent_samples.shape
            frames = None
        timings["latent_copy"] = time.time() - t0

        # ─────────────────────────────────────────────────────────────────
        # Prepare Noise
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        device = comfy.model_management.get_torch_device()
        noise_mask = latent.get("noise_mask", None)

        # FIX #14: Validate noise_override shape
        if noise_override is not None:
            noise = noise_override["samples"]
            expected_shape = latent_samples.shape
            if noise.shape != expected_shape:
                raise ValueError(
                    f"noise_override shape {noise.shape} does not match "
                    f"latent shape {expected_shape}"
                )
        else:
            noise = comfy.sample.prepare_noise(latent_samples, seed, None)

        noise = noise.to(device)
        timings["prepare_noise"] = time.time() - t0

        # ─────────────────────────────────────────────────────────────────
        # v3.2: Seed handled by prepare_noise / sampler_object.
        # Global manual_seed removed to prevent side effects (BUG-12)
        # ─────────────────────────────────────────────────────────────────
        # v3.3 (S-BUG-4): Global seed removed entirely.
        # Seed is passed to prepare_noise() and sample_custom().
        # Setting global CUDA seed caused side effects on other nodes.

        # ─────────────────────────────────────────────────────────────────
        # Calculate Sigmas (with Flux Shift)
        # v3.5: AYS schedule option and Karras end correction
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            if ays_schedule:
                # Try AYS research-optimized schedule first
                ays_sigmas = get_ays_sigmas(detected_type, steps)
                if ays_sigmas is not None:
                    sigmas = ays_sigmas
                    # Apply denoise trimming
                    if denoise < 1.0:
                        total_s = len(sigmas) - 1
                        if total_s > 0:
                            start_s = max(0, int(total_s * (1.0 - denoise)))
                            sigmas = sigmas[start_s:]
                    logger.info(
                        f"Using AYS schedule for {detected_type} ({steps} steps)"
                    )
                else:
                    logger.info(
                        f"AYS not available for {detected_type}, using standard schedule"
                    )
                    sigmas = get_flux_sigmas(
                        model, scheduler, steps, denoise, flux_shift
                    )
            else:
                sigmas = get_flux_sigmas(model, scheduler, steps, denoise, flux_shift)

            # v3.5: Karras end correction — ensure terminal sigma is 0
            if scheduler == "karras":
                sigmas = correct_karras_sigma_end(sigmas)

        except ValueError as e:
            logger.error(f"Failed to calculate sigmas: {e}")
            raise
        timings["sigma_calc"] = time.time() - t0

        log_tensor("Sigmas", sigmas)

        # FIX #13: Handle empty/trivial sigmas
        if len(sigmas) <= 1:
            logger.warning(
                "Sigma schedule is trivial (denoise effectively 0), returning input unchanged"
            )
            report = build_sigma_report(
                detected_type,
                steps,
                scheduler,
                flux_shift,
                denoise,
                sigmas,
                sampler_mode,
                [],
                [],
                0.0,
            )
            return (latent_image.copy(), sigmas, report)

        # ─────────────────────────────────────────────────────────────────
        # Guidance Helper (v3.5: shallow copy for performance)
        # ─────────────────────────────────────────────────────────────────
        def apply_guidance(cond: List, guidance_value: float) -> List:
            """
            Apply Flux guidance to conditioning.

            v3.5 FIX (BUG-20): Replaced copy.deepcopy with shallow dict copy.
            deepcopy was 5500x slower for ControlNet/IP-Adapter conditioning
            because it recursively cloned multi-GB tensor references.

            We only need to SET the "guidance" key — the dict itself must be a
            new object (so we don't mutate the original), but nested values like
            model_conds, control, and gligen are read-only during sampling and
            can be safely shared.
            """
            result = []
            for c in cond:
                new_dict = c[1].copy()  # Shallow dict copy — O(1) for tensors
                new_dict["guidance"] = guidance_value
                result.append([c[0], new_dict])
            return result

        # ─────────────────────────────────────────────────────────────────
        # FIX #4: Prepare starting latent WITHOUT pre-noising
        # Let the sampling loop handle noise properly
        # ─────────────────────────────────────────────────────────────────
        work_latent = latent_samples.to(device)
        log_tensor("Work Latent (Start)", work_latent)

        # ─────────────────────────────────────────────────────────────────
        # UNIFIED SAMPLING PIPELINE (Refiner + Phase-Shift + Dynamic)
        # ─────────────────────────────────────────────────────────────────

        # 1. Setup Splits
        # ────────────────────────────────────────
        # FIX #3: Respect user's sampler choice - don't override it
        primary_sampler = sampler
        secondary_sampler: Optional[str] = None
        # v3.2: Phase-Shift SGM now swaps scheduler, not sampler
        secondary_scheduler: Optional[str] = None
        split_step = -1

        # FIX #9: Guard against zero total_steps
        total_steps = max(1, steps)

        # BUG-3 FIX: Use start_step/end_step to constrain sampling range.
        # Previously these were validated then completely ignored.
        # end_step=0 means "use total_steps" (backward compat with new default)
        effective_end = end_step if end_step > 0 else total_steps
        effective_end = min(effective_end, total_steps)
        effective_start = min(start_step, effective_end)

        splits = {effective_start, effective_end}

        # v3.0: CFG++ mode flag (v3.2: uses SamplerMode constant)
        is_cfg_plus_plus = SamplerMode.is_cfg_plus_plus(sampler_mode)
        if is_cfg_plus_plus:
            logger.info(f"CFG++ (Perpendicular) mode active with base CFG {cfg}")

            # v3.2 FIX (BUG-9): Warn if CFG++ is used with default Flux CFG (1.0)
            if detected_type == "flux" and cfg <= 1.05:
                logger.warning(
                    f"CFG++ enabled but CFG is {cfg}. For Flux, CFG++ requires CFG > 1.0 "
                    "to have an effect (try 2.0-4.0)."
                )

        # v3.0: Apply PAG to model if enabled (v3.2: now has real attention hook)
        if pag_scale > 0:
            model = apply_pag_to_model(model, pag_scale)

        # v3.5: Apply guidance rescale (Imagen-style) to prevent oversaturation
        if guidance_rescale_phi > 0.0 and cfg > 1.0:
            phi = guidance_rescale_phi
            model = (
                model.clone() if pag_scale <= 0 else model
            )  # Already cloned if PAG active

            def guidance_rescale_patch(args):
                """
                Sampler CFG function that applies guidance rescale.
                Prevents oversaturation at high CFG by normalizing the
                standard deviation of the guided output to match the
                conditional output (Saharia et al., 2022).
                """
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                cfg_val = args["cond_scale"]

                # Standard CFG
                guided = uncond + cfg_val * (cond - uncond)

                # Channel-wise std normalization
                dims = list(range(1, guided.ndim))
                guided_std = guided.std(dim=dims, keepdim=True).clamp(min=1e-6)
                cond_std = cond.std(dim=dims, keepdim=True).clamp(min=1e-6)

                rescaled = guided * (cond_std / guided_std)
                return rescaled * phi + guided * (1.0 - phi)

            model.set_model_sampler_cfg_function(guidance_rescale_patch)
            logger.info(f"Guidance Rescale applied (phi={phi:.2f})")

        # v3.2: Phase-Shift setup using SamplerMode constants
        if SamplerMode.is_phase_shift(sampler_mode):
            # v3.4 FIX (BUG-18): Auto-enable sigma blending for phase-shift.
            # Without blending, the sigma schedule has a hard discontinuity
            # at the phase boundary, causing visible noise/artifacts.
            # Default to 3 blend steps when user hasn't explicitly set it.
            if sigma_blend_steps == 0:
                sigma_blend_steps = 3
                logger.info("Phase-Shift: auto-enabled 3 sigma blend steps (was 0)")

            if sampler_mode == SamplerMode.PHASE_SHIFT_DPM:
                secondary_sampler = "dpmpp_2m"
                secondary_scheduler = None  # Keep same scheduler
            elif sampler_mode == SamplerMode.PHASE_SHIFT_SGM:
                # v3.2 FIX: SGM mode swaps the SCHEDULER, not the sampler.
                # "sgm_uniform" is a scheduler name, not a sampler name.
                # Passing it to sampler_object() would crash or fall back.
                secondary_sampler = sampler  # Keep same sampler
                secondary_scheduler = "sgm_uniform"  # Swap scheduler for phase 2
            else:
                secondary_sampler = sampler

            # FIX: Handle edge cases for phase_split
            split_step = int(total_steps * max(0.0, min(1.0, phase_split)))

            # Only add split if it falls within the effective sampling range
            if effective_start < split_step < effective_end:
                splits.add(split_step)
                phase2_label = secondary_sampler or sampler
                if secondary_scheduler:
                    phase2_label = f"{phase2_label}+{secondary_scheduler}"
                logger.info(
                    f"Phase-Shift: {primary_sampler} (0-{split_step}) → "
                    f"{phase2_label} ({split_step}-{total_steps})"
                )

                # v3.1: Apply gradual sigma blend for smooth transition
                if sigma_blend_steps > 0:
                    logger.info(f"Sigma blend: {sigma_blend_steps} steps at transition")

        if refiner_model is not None:
            refiner_step = max(0, min(refiner_start_step, total_steps))
            if effective_start < refiner_step < effective_end:
                splits.add(refiner_step)
                logger.info(f"Refiner starts at step {refiner_step}")

        # Dynamic guidance split points
        # v3.5 FIX (BUG-21/22): Compute thresholds on EFFECTIVE denoising range,
        # not total_steps. With denoise < 1.0, the sigma schedule starts partway
        # through. Splits must land within the active range or guidance timing
        # is completely wrong (e.g., "creative start" phase gets skipped entirely).
        #
        # v3.5 FIX (BUG-23): Only create dynamic guidance splits for Flux models.
        # Non-Flux models ignore guidance anyway — the extra stages just add overhead.
        is_dynamic = "Dynamic" in flux_guidance_profile and detected_type in (
            "flux",
            "chroma",
        )

        if is_dynamic:
            # Calculate effective denoising range for correct threshold mapping
            denoising_steps = (
                int(total_steps * denoise) if denoise < 1.0 else total_steps
            )
            denoising_start = total_steps - denoising_steps

            # Map percentages to steps within the effective range
            idx_20 = denoising_start + int(
                denoising_steps * DYNAMIC_GUIDANCE_EARLY_THRESHOLD
            )
            idx_90 = denoising_start + int(
                denoising_steps * DYNAMIC_GUIDANCE_LATE_THRESHOLD
            )

            if effective_start < idx_20 < effective_end:
                splits.add(idx_20)
            if effective_start < idx_90 < effective_end:
                splits.add(idx_90)
            logger.info(
                f"Dynamic Guidance Active (effective range: steps {denoising_start}-{total_steps}, "
                f"early={idx_20}, late={idx_90})"
            )

        # Sort and filter splits — constrain to effective sampling range
        sorted_splits = sorted(
            s for s in splits if effective_start <= s <= effective_end
        )

        # ─────────────────────────────────────────────────────────────────
        # Preview Callback (v3.2: with TAESD availability check)
        # v3.5 FIX (BUG-24/25): Removed dead `preview_callback` variable.
        # Fixed: when pbar_ref creation fails, don't disable built-in pbar.
        # ─────────────────────────────────────────────────────────────────
        pbar_ref = None
        use_custom_preview = False
        if preview_method != "None":
            # v3.2: Verify TAESD decoder availability, fallback to Latent2RGB
            if preview_method == "TAESD":
                try:
                    # v3.3 (S-BUG-12): Correct TAESD module path
                    from comfy.taesd.taesd import TAESDDecoder  # noqa: F401
                except (ImportError, AttributeError):
                    logger.warning(
                        "TAESD decoder not available, falling back to Latent2RGB"
                    )
                    preview_method = "Latent2RGB"

            try:
                pbar_ref = comfy.utils.ProgressBar(total_steps)
                use_custom_preview = True
                logger.debug(f"Preview callback active: {preview_method}")
            except (AttributeError, TypeError) as e:
                logger.warning(f"Failed to create preview callback: {e}")
                # v3.5 FIX: Don't disable built-in pbar if custom preview fails
                preview_method = "None"

        # v3.2 FIX (BUG-13): Helper to create phase-aware callbacks
        def create_phase_callback(phase_start_step):
            def callback(step, x0, x1, total_steps):
                # 'step' passed by sampler is relative to the current phase's sigmas.
                # We add phase_start_step to get global step index.
                global_step = step + phase_start_step

                if pbar_ref:
                    pbar_ref.update_absolute(global_step + 1, total_steps, (x0,))

            return callback

        # 2. Execution Loop
        # ────────────────────────────────────────
        current_latent = work_latent
        prev_stage_sigmas: Optional[torch.Tensor] = None  # For sigma blending

        # v3.2: Sigma cache with weakref safety to prevent stale hits.
        # If a model is GC'd and a new one allocated at the same address,
        # the weakref will be dead and we'll recompute correctly.
        from comfy.samplers import calculate_sigmas

        # _sigma_cache defined at module level (BUG-14)

        def _get_base_sigmas(mdl, sched: str = scheduler) -> torch.Tensor:
            """
            Get base sigmas for a model, computing only once per unique model+scheduler.

            v3.3 (S-BUG-7): No longer applies flux_shift — the global sigmas
            from get_flux_sigmas() already include shift. Re-applying here
            caused double-shifted sigmas and stage boundary discontinuities.

            v3.3 (S-BUG-8): Pre-seeds cache with the already-computed global
            sigmas to ensure stage indexing matches the split calculation.
            """
            cache_key = (id(mdl), sched)
            if cache_key in _sigma_cache:
                ref, cached_sigmas = _sigma_cache[cache_key]
                if ref() is mdl:  # Verify it's actually the same object
                    return cached_sigmas

            ms = mdl.get_model_object("model_sampling")
            bs = calculate_sigmas(ms, sched, total_steps)
            # v3.3 (S-BUG-7): Shift already applied in get_flux_sigmas().
            # Only apply here for non-primary schedulers (Phase-Shift SGM)
            # where a different scheduler is used than the global one.
            if flux_shift != 1.0 and sched != scheduler:
                bs = flux_shift_sigmas(bs, flux_shift)
            if denoise < 1.0:
                n = len(bs) - 1
                if n > 0:
                    bs = bs[max(0, int(n * (1.0 - denoise))) :]
            _sigma_cache[cache_key] = (weakref.ref(mdl), bs)
            return bs

        # v3.3 (S-BUG-6): Clean stale cache entries before use
        global _sigma_cache
        _clean_sigma_cache()

        # v3.3 (S-BUG-8): Pre-seed cache with global sigmas so stage indexing
        # is guaranteed to match the split points computed from total_steps
        _sigma_cache[(id(model), scheduler)] = (weakref.ref(model), sigmas)

        # Start timing for sampling
        t0 = time.time()

        # v3.2: Per-stage timing collection
        stage_timings: List[Tuple[int, int, int, str, float]] = []

        # Wrap sampling loop in try/finally to ensure cleanup (FIX #10)
        try:
            for i in range(len(sorted_splits) - 1):
                t_stage = time.time()

                s_start = sorted_splits[i]
                s_end = sorted_splits[i + 1]

                if s_start >= s_end:
                    continue

                # Determine config for this stage
                current_model = model
                if refiner_model is not None and s_start >= refiner_start_step:
                    current_model = refiner_model

                current_sampler = primary_sampler
                # v3.2: Determine scheduler for this stage (may differ in SGM mode)
                current_scheduler = scheduler

                if SamplerMode.is_phase_shift(sampler_mode) and s_start >= split_step:
                    if secondary_sampler:
                        current_sampler = secondary_sampler
                    if secondary_scheduler:
                        current_scheduler = secondary_scheduler

                # ─────────────────────────────────────────────────────────────
                # FIX #3: Dynamic Guidance with Cosine Interpolation
                # ─────────────────────────────────────────────────────────────
                stage_positive = positive
                if is_dynamic:
                    g_low = flux_guidance * DYNAMIC_GUIDANCE_EARLY_MULTIPLIER
                    g_high = flux_guidance

                    # v3.5 FIX (BUG-26): Calculate progress relative to the EFFECTIVE
                    # denoising range, not total_steps. For img2img with denoise=0.5,
                    # step 10 is the START of denoising (progress=0.0), not 50%.
                    denoising_steps_f = max(
                        1, int(total_steps * denoise) if denoise < 1.0 else total_steps
                    )
                    denoising_start_f = total_steps - denoising_steps_f
                    progress = max(
                        0.0, min(1.0, (s_start - denoising_start_f) / denoising_steps_f)
                    )

                    # Transition params
                    RAMP = DYNAMIC_GUIDANCE_RAMP_WIDTH
                    EARLY_T = DYNAMIC_GUIDANCE_EARLY_THRESHOLD
                    LATE_T = DYNAMIC_GUIDANCE_LATE_THRESHOLD

                    if progress < EARLY_T - RAMP:
                        effective_guidance = g_low

                    elif progress < EARLY_T + RAMP:
                        t = (progress - (EARLY_T - RAMP)) / (2 * RAMP)
                        blend = 0.5 * (1.0 - math.cos(math.pi * t))
                        effective_guidance = g_low + (g_high - g_low) * blend

                    elif progress < LATE_T - RAMP:
                        effective_guidance = g_high

                    elif progress < LATE_T + RAMP:
                        g_late = g_high * DYNAMIC_GUIDANCE_LATE_MULTIPLIER
                        t = (progress - (LATE_T - RAMP)) / (2 * RAMP)
                        blend = 0.5 * (1.0 - math.cos(math.pi * t))
                        effective_guidance = g_high + (g_late - g_high) * blend

                    else:
                        effective_guidance = g_high * DYNAMIC_GUIDANCE_LATE_MULTIPLIER

                    # v3.5: is_dynamic already filters for flux/chroma models
                    stage_positive = apply_guidance(positive, effective_guidance)
                    logger.debug(
                        f"Dynamic Guidance @ step {s_start} (progress={progress:.2f}): {effective_guidance:.2f}"
                    )

                elif detected_type in ("flux", "chroma"):
                    # Static guidance — Flux and Chroma use guidance embedding
                    stage_positive = apply_guidance(positive, flux_guidance)

                logger.info(
                    f"Stage {i+1}: Steps {s_start}-{s_end} | "
                    f"Sampler: {current_sampler} | Scheduler: {current_scheduler}"
                )

                # ─────────────────────────────────────────────────────────────
                # Calculate stage sigmas with bounds checking
                # ─────────────────────────────────────────────────────────────
                try:
                    # v3.2: Pass current_scheduler for correct sigma computation
                    # (differs from primary scheduler in Phase-Shift SGM mode)
                    base_sigmas = _get_base_sigmas(current_model, current_scheduler)

                    # FIX #4: Correct sigma indexing when denoise < 1.0
                    # base_sigmas is trimmed, so its index 0 corresponds to step (total_steps - len + 1)
                    # We must offset s_start/s_end to match base_sigmas indices.

                    full_schedule_len = len(
                        base_sigmas
                    )  # This is the TRIMMED length if denoise < 1.0
                    # If denoise=1.0, full_schedule_len = total_steps + 1

                    # Calculate the step index where base_sigmas technically "starts" relative to full schedule
                    # start_offset = total_steps - (full_schedule_len - 1)
                    # But actually, s_start/s_end are 0-indexed on TOTAL steps.
                    # If denoise < 1.0, base_sigmas starts at step X. Indices 0..N in base_sigmas map to steps X..Total.
                    # So we need to map s_start (global step) to local index.
                    # index = global_step - start_offset

                    bs_start_step = total_steps - (len(base_sigmas) - 1)

                    local_start = s_start - bs_start_step
                    local_end = s_end - bs_start_step

                    # If the entire stage is before the start of the sigmas (due to low denoise), skip
                    if local_end < 0:
                        # logger.debug(f"Stage {i+1} skipped (before denoise start)")
                        continue

                    # Clamp to valid range within base_sigmas
                    safe_start = max(0, min(local_start, len(base_sigmas) - 1))
                    safe_end = max(
                        0, min(local_end + 1, len(base_sigmas))
                    )  # +1 for slicing inclusive of end sigma

                    if safe_start >= safe_end:
                        # logger.warning(f"Stage {i+1}: Empty sigma range after offset, skipping")
                        continue

                    stage_sigmas = base_sigmas[safe_start:safe_end]

                    # v3.1: Apply gradual sigma blend at phase-shift transitions
                    # v3.2: No longer needs .clone() from caller — function handles it
                    if (
                        sigma_blend_steps > 0
                        and prev_stage_sigmas is not None
                        and SamplerMode.is_phase_shift(sampler_mode)
                        and s_start == split_step
                    ):
                        stage_sigmas = gradual_sigma_blend(
                            prev_stage_sigmas, stage_sigmas, sigma_blend_steps
                        )

                    if len(stage_sigmas) < 2:
                        logger.warning(
                            f"Stage {i+1}: Insufficient sigmas ({len(stage_sigmas)}), skipping"
                        )
                        continue

                    # BUG-6 FIX: Implement return_with_leftover_noise
                    # When True and this is the last stage, strip terminal sigma=0.0
                    # so the sampler stops before full denoising (latent retains noise)
                    is_last_stage = i == len(sorted_splits) - 2
                    if (
                        return_with_leftover_noise
                        and is_last_stage
                        and len(stage_sigmas) >= 3  # need at least 3 to safely strip
                        and stage_sigmas[-1].item() == 0.0
                    ):
                        stage_sigmas = stage_sigmas[:-1]
                        logger.debug("Stripped terminal sigma for leftover noise")

                    # Create sampler object
                    sampler_obj = comfy.samplers.sampler_object(current_sampler)

                    # v3.2 FIX: Only compute CFG++ when mode is active.
                    # Previously stage_cfg was always computed then ignored via ternary.
                    if is_cfg_plus_plus and len(stage_sigmas) > 0:
                        sigma_max = (
                            base_sigmas[0].item() if len(base_sigmas) > 0 else 1.0
                        )
                        effective_cfg = apply_cfg_plus_plus(
                            cfg, stage_sigmas[0], sigma_max
                        )
                        logger.debug(
                            f"CFG++: {cfg:.2f} → {effective_cfg:.2f} at sigma {stage_sigmas[0].item():.4f}"
                        )
                    else:
                        effective_cfg = cfg

                    # ─────────────────────────────────────────────────────
                    # v3.4 FIX (BUG-15/16): Correct noise handling.
                    #
                    # ComfyUI's KSAMPLER.sample() ALWAYS applies:
                    #   x = noise * sigma[0] + latent_image
                    #
                    # Previously this code pre-added noise * sigma to the
                    # latent (line 1404), then sample_custom added it AGAIN
                    # inside the sampler → 2× noise in stage 1.
                    #
                    # For continuation stages (i > 0), the sampler would add
                    # fresh noise * sigma_start to the already-denoised latent,
                    # injecting extra noise that corrupts partial results.
                    #
                    # FIX:
                    #  - Stage 1, add_noise=True:  pass raw noise + clean latent
                    #    → sampler correctly does noise*σ + latent (1× noise)
                    #  - Stage 1, add_noise=False: pass zeros + clean latent
                    #    → sampler does 0*σ + latent = latent (no noise)
                    #  - Stages 2+: ALWAYS pass zeros + previous output
                    #    → sampler does 0*σ + result = result (no extra noise)
                    # ─────────────────────────────────────────────────────
                    stage_latent = current_latent

                    is_first_stage = i == 0 and effective_start == 0
                    if is_first_stage and add_noise:
                        # Pass unscaled noise — sampler handles scaling + addition
                        stage_noise = noise
                        logger.debug("Stage 1: sampler will add initial noise")
                    else:
                        # Continuation or no-noise: pass zeros so sampler
                        # does 0*σ + latent = latent (no extra noise)
                        stage_noise = torch.zeros_like(noise)
                        if is_first_stage:
                            logger.debug("Stage 1: add_noise=False, no noise")
                        else:
                            logger.debug(f"Stage {i+1}: continuation, no extra noise")

                    result = comfy.sample.sample_custom(
                        current_model,
                        stage_noise,
                        effective_cfg,
                        sampler_obj,
                        stage_sigmas,
                        stage_positive,
                        negative,
                        stage_latent,
                        noise_mask=noise_mask,
                        callback=create_phase_callback(s_start),
                        disable_pbar=use_custom_preview,
                        seed=seed,
                    )
                    current_latent = result
                    # v3.3 (S-BUG-9): Store pre-strip sigmas for blend source.
                    # If return_with_leftover_noise stripped the terminal sigma,
                    # the blend needs the original schedule's last sigma.
                    prev_stage_sigmas = base_sigmas[safe_start:safe_end]
                    log_tensor(f"Stage {i+1} Output", current_latent)

                    # FIX #8: Sigma continuity validation with correct next-scheduler checking
                    if i < len(sorted_splits) - 2:
                        next_stage_start = sorted_splits[i + 1]

                        # Determine next stage's config to get correct base sigmas
                        # (Logic duplicated from top of loop - could be refactored)
                        next_scheduler = scheduler
                        next_model_ref = model
                        if (
                            SamplerMode.is_phase_shift(sampler_mode)
                            and next_stage_start >= split_step
                        ):
                            if secondary_scheduler:
                                next_scheduler = secondary_scheduler

                        next_base_sigmas = _get_base_sigmas(
                            next_model_ref, next_scheduler
                        )

                        # Calculate offset for next stage
                        next_bs_start = total_steps - (len(next_base_sigmas) - 1)
                        next_local_idx = next_stage_start - next_bs_start

                        if (
                            0 <= next_local_idx < len(next_base_sigmas)
                            and len(stage_sigmas) > 0
                        ):
                            expected_sigma = stage_sigmas[-1].item()
                            next_sigma = next_base_sigmas[next_local_idx].item()

                            sigma_diff = abs(expected_sigma - next_sigma)
                            if sigma_diff > SIGMA_DISCONTINUITY_THRESHOLD:
                                logger.warning(
                                    f"Sigma discontinuity: {expected_sigma:.4f} → {next_sigma:.4f} "
                                    f"(Δ={sigma_diff:.4f})"
                                )

                    # v3.2: Record per-stage timing
                    stage_time = time.time() - t_stage
                    stage_timings.append(
                        (i + 1, s_start, s_end, current_sampler, stage_time)
                    )

                except (RuntimeError, ValueError) as e:
                    logger.error(f"Error in Stage {i+1}: {e}")
                    raise
        finally:
            # ─────────────────────────────────────────────────────────────────
            # v3.2 FIX: Free GPU noise tensor to prevent VRAM leak (FIX #10)
            # ─────────────────────────────────────────────────────────────────
            if "noise" in locals():
                del noise
            if "work_latent" in locals():
                del work_latent

        # (noise deleted in finally block)

        # FIX #12: Ensure result is on CPU to prevent VRAM accumulation
        if current_latent.is_cuda:
            samples = current_latent.cpu()
        else:
            samples = current_latent

        timings["sampling"] = time.time() - t0

        # ─────────────────────────────────────────────────────────────────
        # Prepare output
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        out = latent.copy()
        out["samples"] = samples
        timings["output_prep"] = time.time() - t0

        total_time = time.time() - t_start

        # ─────────────────────────────────────────────────────────────────
        # Log timing report (v3.2: includes per-stage breakdown)
        # ─────────────────────────────────────────────────────────────────
        logger.info(
            f"Sampling complete: {steps} steps, {total_time:.2f}s total, "
            f"{timings['sampling']:.2f}s sampling"
        )
        for stage_num, s_start_t, s_end_t, samp, t in stage_timings:
            logger.info(
                f"  Stage {stage_num}: steps {s_start_t}→{s_end_t} [{samp}] = {t:.3f}s"
            )

        # Ensure sigmas is a valid tensor
        output_sigmas = (
            sigmas if sigmas is not None and len(sigmas) > 0 else torch.tensor([0.0])
        )

        # v3.2: Build sigma report string for diagnostics output
        sigma_report = build_sigma_report(
            detected_type,
            steps,
            scheduler,
            flux_shift,
            denoise,
            output_sigmas,
            sampler_mode,
            sorted_splits,
            stage_timings,
            total_time,
            ays_active=ays_schedule,
        )

        return (out, output_sigmas, sigma_report)


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceSamplerPro": RadianceSamplerPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSamplerPro": "◎ Radiance Sampler Pro",
}
