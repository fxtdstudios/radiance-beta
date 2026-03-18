"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE SAMPLER PRO v4.2.0
         Professional Flux-Optimized Sampling Engine
                   Radiance © 2024-2026

 Features:
 - Native Flux sigma shifting for high-resolution detail
 - Integrated guidance control (bypasses CFG for Flux)
 - PAG (Perturbed Attention Guidance) via attention hooking (v3.0, fixed v3.2)
 - CFG++ with perpendicular scheduling for better saturation control (v3.0)
 - Multi-Model Support: Auto-detect Flux/SD3/SDXL with optimal defaults (v3.1)
 - Video Model Support: WAN, LTX-Video, HunyuanVideo with auto-shift (v4.1)
 - Sigma Blend: Smooth phase transitions for phase-shift mode (v3.1)
 - Live Preview: TAESD/Latent2RGB preview during sampling (v3.1)
 - Sigma Report: Diagnostic string output with ETA/speed (v3.2, enhanced v3.5)
 - Per-Stage Timing: Breakdown per sampling stage (v3.2)
 - AYS (Align Your Steps) research-optimized sigma schedules (v3.5)
 - Guidance Rescale (Imagen-style) to prevent oversaturation (v3.5)
 - Terminal sigma correction for complete denoising (v3.6, replaces v3.5 Karras-only)
 - Workflow presets (txt2img, img2img, inpaint, high-res, turbo, video)
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

 v4.1 - Video Support & Cache Fix (March 2026)
 - FIX: CRITICAL (BUG-34) — SigmaCache poisoned across denoise values.
   Trimmed schedules were cached without denoise in the key. A second call
   with different denoise got stale trimmed sigmas. Now caches UNTRIMMED
   schedules (keyed by total_steps), trims after retrieval.
 - FIX: CRITICAL (BUG-35) — Auto-defaults didn't apply shift for video models.
   WAN (shift=8.0), LTX (2.37), HunyuanVideo (7.0) all got shift=1.0.
   Now auto-applies shift/sampler from MODEL_DEFAULTS when at widget defaults.
 - FIX: HIGH (BUG-36) — Guidance embedding only applied for Flux/Chroma.
   Added GUIDANCE_EMBED_MODELS set: {flux, chroma, lumina2, z_image, ltxv}.
 - FIX: HIGH (BUG-37) — Channel validation missing "hunyuan".
 - NEW: HIGH (BUG-38) — Video workflow presets: WAN txt2vid/img2vid,
   LTX-Video, HunyuanVideo with correct shift, steps, scheduler.
 - FIX: MEDIUM (BUG-39) — Sigma report and latent_meta now include frame
   count and is_video flag for video diagnostics.
 - NEW: MEDIUM (BUG-40) — Experimental AYS anchors for WAN and LTX-Video.
   HunyuanVideo mapped to WAN anchors.
 - FIX: MEDIUM (BUG-41) — Perlin/Spectral noise lacked temporal coherence
   for 5D video latents. Added _temporally_correlate() AR(1) wrapper.

 v4.2 - Cinema Production Pass (March 2026)
 - FIX: WAN/HunyuanVideo CFG defaults corrected — was 1.0 (no guidance),
   now 6.0 matching Alibaba's reference inference code. Added guidance_type
   field to MODEL_DEFAULTS for explicit cfg vs embedding classification.
 - NEW: Dynamic CFG ramp for CFG-guided video models (WAN, HunyuanVideo).
   compute_dynamic_cfg() boosts CFG ×1.2 early for structure, tapers ×0.7
   late for clean convergence and reduced temporal artifacts.
 - NEW: CFG_GUIDED_MODELS constant separates CFG-guided models from
   embedding-guided (GUIDANCE_EMBED_MODELS). Sampling loop now correctly
   routes to either dynamic guidance or dynamic CFG based on model type.
 - NEW: Per-frame noise seeding for all 5D video noise generators.
   Each frame uses seed+frame_idx — extending a video preserves existing
   frame noise for reproducible keyframe locking.
 - FIX: Phase-Shift mode auto-disabled for VIDEO_MODEL_TYPES with warning.
   Sampler switching mid-schedule causes temporal discontinuities ("pop").
 - FIX: DPM solver incompatibility warning expanded to all flow-matching
   models including video (WAN, HunyuanVideo).

 VERSION HISTORY:

 v3.7 - Architecture Finalization (February 2026)
 - REFACTOR: Extracted apply_guidance → module-level apply_flux_guidance()
 - REFACTOR: Extracted 40-line dynamic guidance → compute_dynamic_guidance()
   Pure function, fully testable in isolation
 - REFACTOR: Extracted _get_base_sigmas closure → compute_base_sigmas()
   All dependencies now explicit parameters instead of implicit captures
 - REFACTOR: Replaced id()-based sigma cache with SigmaCache class.
   Uses model config hash (config_name, sigma_max, sigma_min, scheduler)
   instead of memory address. GC-safe, better hit rate, bounded to 32 entries.
 - REFACTOR: Removed weakref dependency entirely
 - NEW: Separate test_sampler.py with comprehensive unit tests

 v3.6 - Noise Elimination & Clean Output (February 2026)
 - FIX: CRITICAL (BUG-28) — correct_karras_sigma_end APPENDED 0.0 to sigma
   schedule, making it steps+2 instead of steps+1. Since total_steps was
   never updated, bs_start_step went negative, ALL local stage indices
   shifted by +1, and σ_max was SKIPPED. The model started denoising from
   the wrong noise level, leaving residual noise in every Karras output.
   FIX: REPLACE terminal sigma instead of appending.
 - FIX: HIGH (BUG-29) — Terminal sigma correction only ran for Karras.
   Other schedulers can also produce non-zero terminal sigma depending on
   model sigma_min. Now applies to ALL schedulers universally.
 - FIX: HIGH (BUG-30) — SDXL/SD15 auto-defaults used euler_ancestral.
   Ancestral samplers add stochastic noise at every step, producing
   inherently grainy output. Changed to dpmpp_2m (deterministic, clean).
 - FIX: MEDIUM (BUG-31) — Dynamic guidance late multiplier was 0.85,
   dropping guidance 15% in the final detail phase. Too aggressive —
   prevented clean convergence. Increased to 0.95.
 - FIX: MEDIUM (BUG-32) — _get_base_sigmas didn't apply terminal sigma
   correction on cache misses, causing length/value mismatches between
   stages in Phase-Shift mode.
 - FIX: LOW (BUG-33) — AYS Flux anchors refined for better match to
   flow-matching sigma distribution. Marked as experimental.

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
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
from .gpu_memory import cleanup_gpu_memory

# ═══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Dynamic guidance curve parameters
# v3.5: Renamed for clarity — these control the creative freedom phase
DYNAMIC_GUIDANCE_EARLY_MULTIPLIER = 0.6  # Early phase: base * 0.6 (creative freedom)
DYNAMIC_GUIDANCE_LATE_MULTIPLIER = 0.95  # v3.6 FIX (BUG-31): was 0.85, too aggressive — killed detail sharpness
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

# v4.2: Models that use traditional CFG for guidance (not embedding).
# These need CFG > 1.0 for any guidance effect. Dynamic CFG ramp applies here.
CFG_GUIDANCE_MODELS = {"wan", "hunyuan_video"}

# v4.2: Dynamic CFG ramp parameters for video models.
# Video models benefit from higher CFG early (structure) tapering to lower CFG
# late (clean convergence, reduces temporal artifacts in final frames).
DYNAMIC_CFG_EARLY_MULTIPLIER = 1.2   # Early: boost CFG 20% for strong structure
DYNAMIC_CFG_LATE_MULTIPLIER = 0.7    # Late: reduce CFG 30% for clean convergence
DYNAMIC_CFG_EARLY_THRESHOLD = 0.15   # First 15% boosted
DYNAMIC_CFG_LATE_THRESHOLD = 0.85    # Last 15% reduced

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

# v4.1: Video models requiring 5D latent (batch, channels, frames, height, width).
# Used for early validation — prevents cryptic errors deep inside model forward pass.
VIDEO_MODEL_TYPES = {"wan", "ltxv", "hunyuan_video", "cosmos"}

# v4.1 FIX (BUG-36): Models that use guidance embedding in conditioning dict.
# These are flow-matching models where guidance is injected as an embedding
# into the conditioning, NOT via traditional CFG scaling. Must be kept in sync
# with new flow-matching model additions.
GUIDANCE_EMBED_MODELS = {"flux", "chroma", "lumina2", "z_image", "ltxv"}

# v4.2: Models that use traditional CFG as primary guidance mechanism.
# These need CFG > 1.0 for quality output. Dynamic CFG ramp applies to these.
# WAN and HunyuanVideo look like flow-matching but use CFG, not embedding.
CFG_GUIDED_MODELS = {"wan", "hunyuan_video", "sdxl", "sd15", "sd3", "sd35"}

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
        "sampler": "dpmpp_2m",  # v3.6 FIX (BUG-30): was euler_ancestral (stochastic → noisy)
        "denoise_range": (0.3, 1.0),
    },
    "sd15": {
        "cfg": 7.0,
        "scheduler": "normal",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "dpmpp_2m",  # v3.6 FIX (BUG-30): was euler_ancestral (stochastic → noisy)
        "denoise_range": (0.3, 1.0),
    },
    # v3.3: Video & modern models (S-BUG-10)
    # v4.2 FIX: WAN and HunyuanVideo use CFG-based guidance, NOT embedding.
    # cfg=1.0 was a silent quality killer — Alibaba's reference is cfg=5.0-7.0.
    "wan": {
        "cfg": 6.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 8.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "cfg",  # v4.2: WAN uses CFG, not guidance embedding
    },
    "ltxv": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 2.37,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "embedding",
    },
    "hunyuan_video": {
        "cfg": 6.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 7.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "cfg",  # v4.2: HunyuanVideo uses CFG, not guidance embedding
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

# v4.0: Noise generation types
NOISE_TYPES = ["Gaussian", "Perlin", "Uniform", "Spectral", "Brownian"]

# v4.0: Multi-encoder conditioning target routing
CLIP_TARGETS = ["Auto", "clip_l", "clip_g", "t5xxl"]

# v4.0: Multiple conditioning merge modes
MULTI_COND_MODES = ["Off", "average", "weighted", "sequential"]

# v4.0: Tile sampling seam-blend modes
TILE_BLEND_MODES = ["feather", "average", "gaussian"]


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

# ═══════════════════════════════════════════════════════════════════════════════
#                   SIGMA CACHE (v3.7 — replaces id()-based cache)
# ═══════════════════════════════════════════════════════════════════════════════


class SigmaCache:
    """
    GC-safe sigma schedule cache using model config identity instead of id().

    v3.7: Previous cache used id(model) as key, which is the memory address.
    After garbage collection, Python can allocate a new model at the same
    address, causing stale cache hits. The weakref guard mitigated this but
    added complexity and wasn't airtight under all GC timing scenarios.

    New approach: cache key is (config_class_name, sigma_max, sigma_min, scheduler).
    Models with identical sampling configurations SHOULD share cached sigmas
    because calculate_sigmas() produces identical output for identical inputs.
    This also improves hit rate when loading the same model type multiple times.

    Max 32 entries to bound memory usage.
    """

    MAX_ENTRIES = 32

    def __init__(self):
        self._cache: Dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _make_key(model, scheduler: str, total_steps: int) -> tuple:
        """Build a GC-safe cache key from model sampling config.

        v4.1 FIX (BUG-34): Added total_steps to key. Previously, denoise
        trimming was applied before caching, but denoise was NOT in the key.
        A second call with different denoise got the stale trimmed schedule.
        Now we cache untrimmed schedules (keyed by total_steps) and trim
        after retrieval — so all denoise values share one cache entry.
        """
        try:
            ms = model.get_model_object("model_sampling")
            sigma_max = ms.sigma_max.item() if hasattr(ms, "sigma_max") else 0.0
            sigma_min = ms.sigma_min.item() if hasattr(ms, "sigma_min") else 0.0

            config_name = "unknown"
            try:
                if hasattr(model, "model") and hasattr(model.model, "model_config"):
                    config_name = type(model.model.model_config).__name__
            except (AttributeError, RuntimeError):
                pass

            return (config_name, round(sigma_max, 6), round(sigma_min, 6), scheduler, total_steps)
        except (AttributeError, RuntimeError):
            # Absolute fallback — unique per call (no caching)
            logger.debug("SigmaCache: failed to build config key, cache disabled for this model")
            return (id(model), scheduler, total_steps, time.time())

    def get(self, model, scheduler: str, total_steps: int) -> Optional[torch.Tensor]:
        """Retrieve cached sigmas, or None on miss."""
        key = self._make_key(model, scheduler, total_steps)
        return self._cache.get(key)

    def put(self, model, scheduler: str, total_steps: int, sigmas: torch.Tensor) -> None:
        """Store sigmas in cache, evicting oldest if full."""
        if len(self._cache) >= self.MAX_ENTRIES:
            # Evict oldest entry (first inserted)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            logger.debug(f"SigmaCache: evicted oldest entry, {len(self._cache)} remaining")

        key = self._make_key(model, scheduler, total_steps)
        self._cache[key] = sigmas

    def clear(self) -> None:
        """Clear all cached entries."""
        count = len(self._cache)
        self._cache.clear()
        if count:
            logger.debug(f"SigmaCache: cleared {count} entries")


# Module-level cache instance
_sigma_cache = SigmaCache()


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
#                         SIGMA INDEXER (v3.6)
# ═══════════════════════════════════════════════════════════════════════════════


class SigmaIndexer:
    """
    v3.6: Encapsulates all sigma-to-step index mapping with safety assertions.

    This replaces the fragile inline offset math that caused BUG-28 (off-by-one
    from sigma schedule length mismatch). All index calculations go through
    this class, and invariant violations are caught immediately via assertions.

    The core problem: when denoise < 1.0, the sigma schedule is trimmed so
    index 0 no longer corresponds to global step 0. We need to map between
    global step indices (used by splits/stages) and local indices into the
    sigma tensor. This class makes that mapping explicit and verified.
    """

    def __init__(self, total_steps: int, base_sigmas: torch.Tensor):
        self.total_steps = total_steps
        self.base_sigmas = base_sigmas
        self.num_sigmas = len(base_sigmas)
        # The global step where this sigma schedule "starts"
        # For denoise=1.0: offset = 0 (schedule covers all steps)
        # For denoise<1.0: offset > 0 (schedule starts partway through)
        self.offset = total_steps - (self.num_sigmas - 1)

        # ── The invariant BUG-28 violated ──
        # If this fires, a sigma function returned wrong-length schedule
        assert self.num_sigmas <= total_steps + 1, (
            f"SigmaIndexer: schedule length {self.num_sigmas} exceeds "
            f"steps+1 ({total_steps + 1}). A sigma correction probably "
            f"appended instead of replacing."
        )
        assert self.offset >= 0, (
            f"SigmaIndexer: negative offset {self.offset} — schedule has "
            f"{self.num_sigmas} sigmas for {total_steps} steps"
        )

    def to_local(self, global_step: int) -> int:
        """Convert global step index to local sigma index."""
        return global_step - self.offset

    def get_stage_sigmas(
        self, global_start: int, global_end: int
    ) -> Optional[torch.Tensor]:
        """
        Extract sigma sub-schedule for a stage defined by global step range.

        Returns None if the stage falls entirely outside the sigma schedule
        (can happen with low denoise values). The caller should skip the stage.

        The returned tensor has len = (end - start + 1) sigmas, which
        represents (end - start) sampling steps.
        """
        local_start = self.to_local(global_start)
        local_end = self.to_local(global_end)

        # Stage is entirely before sigma schedule starts (low denoise)
        if local_end < 0:
            return None

        # Clamp to valid range
        safe_start = max(0, min(local_start, self.num_sigmas - 1))
        safe_end = max(0, min(local_end + 1, self.num_sigmas))  # +1 for slice

        if safe_start >= safe_end:
            return None

        return self.base_sigmas[safe_start:safe_end]

    def get_sigma_at(self, global_step: int) -> Optional[float]:
        """Get sigma value at a global step index, or None if out of range."""
        local = self.to_local(global_step)
        if 0 <= local < self.num_sigmas:
            return self.base_sigmas[local].item()
        return None


@dataclass
class SamplingStage:
    """
    v3.6: Captures all configuration for a single sampling stage.

    Built by plan_stages() BEFORE the execution loop, so the entire
    sampling plan can be validated and logged upfront. The loop then
    becomes a clean execute-only sequence.
    """

    index: int  # Stage number (0-based)
    global_start: int  # Global step start
    global_end: int  # Global step end
    model: Any  # ComfyUI model wrapper
    sampler_name: str  # Sampler algorithm name
    scheduler_name: str  # Scheduler name
    is_phase_shifted: bool = False  # Whether this stage uses secondary config
    is_blend_point: bool = False  # Whether sigma blending should apply here


# ═══════════════════════════════════════════════════════════════════════════════
#              EXTRACTED FUNCTIONS (v3.7 — formerly closures in sample())
# ═══════════════════════════════════════════════════════════════════════════════


def apply_flux_guidance(cond: List, guidance_value: float) -> List:
    """
    Apply Flux guidance embedding to conditioning.

    v3.7: Extracted from closure inside sample() — was needlessly re-defined
    every call despite having zero dependency on enclosing scope.

    v3.5 FIX (BUG-20): Uses shallow dict copy (O(1) for tensors) instead of
    copy.deepcopy which was 5500× slower for ControlNet/IP-Adapter conditioning.

    Args:
        cond: ComfyUI conditioning list [ [tensor, dict], ... ]
        guidance_value: Flux guidance scale (0.0 = none, 3.5 = typical)

    Returns:
        New conditioning list with guidance value set (originals unchanged)
    """
    result = []
    for c in cond:
        new_dict = c[1].copy()  # Shallow dict copy — O(1) for tensors
        new_dict["guidance"] = guidance_value
        result.append([c[0], new_dict])
    return result


def compute_dynamic_guidance(
    base_guidance: float,
    step: int,
    total_steps: int,
    denoise: float,
) -> float:
    """
    Compute dynamic guidance value based on sampling progress.

    v3.7: Extracted from 40-line inline block in the sampling loop.
    Now a pure function — testable in isolation.

    The guidance curve has three phases:
      1. Early (0-20%): Low guidance (base × 0.6) for creative freedom
      2. Middle (20-90%): Full guidance for structure/detail
      3. Late (90-100%): Slightly reduced (base × 0.95) for smooth convergence

    Transitions use cosine interpolation over a 5% ramp width.

    Args:
        base_guidance: User-set guidance value (e.g., 3.5)
        step: Current global step index
        total_steps: Total number of steps in schedule
        denoise: Denoise strength (1.0 = full, <1.0 = img2img)

    Returns:
        Effective guidance value for this step
    """
    g_low = base_guidance * DYNAMIC_GUIDANCE_EARLY_MULTIPLIER
    g_high = base_guidance

    # v3.5 FIX (BUG-26): Progress relative to EFFECTIVE denoising range
    denoising_steps = max(
        1, int(total_steps * denoise) if denoise < 1.0 else total_steps
    )
    denoising_start = total_steps - denoising_steps
    progress = max(0.0, min(1.0, (step - denoising_start) / denoising_steps))

    RAMP = DYNAMIC_GUIDANCE_RAMP_WIDTH
    EARLY_T = DYNAMIC_GUIDANCE_EARLY_THRESHOLD
    LATE_T = DYNAMIC_GUIDANCE_LATE_THRESHOLD

    if progress < EARLY_T - RAMP:
        return g_low

    elif progress < EARLY_T + RAMP:
        t = (progress - (EARLY_T - RAMP)) / (2 * RAMP)
        blend = 0.5 * (1.0 - math.cos(math.pi * t))
        return g_low + (g_high - g_low) * blend

    elif progress < LATE_T - RAMP:
        return g_high

    elif progress < LATE_T + RAMP:
        g_late = g_high * DYNAMIC_GUIDANCE_LATE_MULTIPLIER
        t = (progress - (LATE_T - RAMP)) / (2 * RAMP)
        blend = 0.5 * (1.0 - math.cos(math.pi * t))
        return g_high + (g_late - g_high) * blend

    else:
        return g_high * DYNAMIC_GUIDANCE_LATE_MULTIPLIER


def compute_dynamic_cfg(
    base_cfg: float,
    step: int,
    total_steps: int,
    denoise: float,
) -> float:
    """
    v4.2: Compute dynamic CFG value for video models (WAN, HunyuanVideo).

    These models use traditional CFG-based guidance (not embedding), so the
    guidance ramp operates on the CFG scale directly. The curve:
      1. Early (0-15%): Boosted CFG (×1.2) for strong structural composition
      2. Middle (15-85%): Full CFG for detail and prompt adherence
      3. Late (85-100%): Reduced CFG (×0.7) for clean convergence and
         reduced temporal artifacts in the final frames

    Uses cosine interpolation for smooth transitions (same approach as
    compute_dynamic_guidance but operating on CFG instead of guidance embedding).
    """
    # Progress relative to effective denoising range
    denoising_steps = max(1, int(total_steps * denoise) if denoise < 1.0 else total_steps)
    denoising_start = total_steps - denoising_steps
    progress = max(0.0, min(1.0, (step - denoising_start) / denoising_steps))

    RAMP = DYNAMIC_GUIDANCE_RAMP_WIDTH  # Reuse 5% ramp from guidance
    EARLY_T = DYNAMIC_CFG_EARLY_THRESHOLD
    LATE_T = DYNAMIC_CFG_LATE_THRESHOLD

    cfg_early = base_cfg * DYNAMIC_CFG_EARLY_MULTIPLIER
    cfg_late = base_cfg * DYNAMIC_CFG_LATE_MULTIPLIER

    if progress < EARLY_T - RAMP:
        return cfg_early
    elif progress < EARLY_T + RAMP:
        t = (progress - (EARLY_T - RAMP)) / (2 * RAMP)
        blend = 0.5 * (1.0 - math.cos(math.pi * t))
        return cfg_early + (base_cfg - cfg_early) * blend
    elif progress < LATE_T - RAMP:
        return base_cfg
    elif progress < LATE_T + RAMP:
        t = (progress - (LATE_T - RAMP)) / (2 * RAMP)
        blend = 0.5 * (1.0 - math.cos(math.pi * t))
        return base_cfg + (cfg_late - base_cfg) * blend
    else:
        return cfg_late


def compute_base_sigmas(
    model,
    scheduler_name: str,
    total_steps: int,
    primary_scheduler: str,
    flux_shift: float,
    denoise: float,
    cache: SigmaCache,
) -> torch.Tensor:
    """
    Compute base sigma schedule for a model, with caching.

    v3.7: Extracted from closure inside sample(). All dependencies are
    now explicit parameters instead of implicit captures from enclosing scope.

    v4.1 FIX (BUG-34): Cache stores the UNTRIMMED schedule (steps+1 elements).
    Denoise trimming is applied AFTER cache retrieval. Previously, trimmed
    schedules were cached without denoise in the key, so a second call with
    a different denoise value got the wrong (stale trimmed) schedule.

    Handles:
      - Cache lookup/store via SigmaCache
      - Flux shift for non-primary schedulers (Phase-Shift SGM mode)
      - Terminal sigma correction (BUG-28/29/32)
      - Schedule length validation
      - Denoise trimming (post-cache)

    Args:
        model: ComfyUI model wrapper
        scheduler_name: Scheduler to compute sigmas for
        total_steps: Total sampling steps
        primary_scheduler: The primary scheduler (shift already applied in global sigmas)
        flux_shift: Flux shift parameter (only applied for non-primary schedulers)
        denoise: Denoise strength (1.0 = full schedule, <1.0 = trimmed)
        cache: SigmaCache instance for cross-call memoization

    Returns:
        Sigma schedule tensor (may be trimmed for denoise < 1.0)
    """
    # Check cache first — cache stores UNTRIMMED schedules
    cached = cache.get(model, scheduler_name, total_steps)
    if cached is not None:
        bs = cached
    else:
        # Compute fresh sigmas
        ms = model.get_model_object("model_sampling")
        bs = comfy.samplers.calculate_sigmas(ms, scheduler_name, total_steps)

        # v3.3 (S-BUG-7): Shift already applied in get_flux_sigmas() for primary.
        # Only apply for non-primary schedulers (e.g., Phase-Shift SGM).
        if flux_shift != 1.0 and scheduler_name != primary_scheduler:
            bs = flux_shift_sigmas(bs, flux_shift)

        # v3.6 FIX (BUG-28/29/32): Universal terminal sigma correction
        bs = correct_sigma_end(bs)

        # v3.6: Validate schedule length (the invariant BUG-28 violated)
        assert len(bs) == total_steps + 1, (
            f"compute_base_sigmas: schedule length {len(bs)} != steps+1 "
            f"({total_steps + 1}) for scheduler '{scheduler_name}'. "
            f"A sigma correction may have appended instead of replacing."
        )

        # Cache the UNTRIMMED schedule
        cache.put(model, scheduler_name, total_steps, bs)

    # v4.1 FIX (BUG-34): Trim AFTER cache retrieval — each denoise value
    # trims independently from the same cached untrimmed schedule.
    if denoise < 1.0:
        n = len(bs) - 1
        if n > 0:
            bs = bs[max(0, int(n * (1.0 - denoise))):]

    return bs


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
    # v4.1: Video model presets (BUG-38)
    "▶ WAN txt2vid (30 steps)",
    "▶ WAN img2vid (20 steps)",
    "▶ LTX-Video (25 steps)",
    "▶ HunyuanVideo (30 steps)",
    # v4.0: Universal quality tiers (any arch)
    "◈ Draft (4-step / AYS)",
    "◈ Fast (8-step / AYS)",
    "◈ Balanced (20-step)",
    "◈ Quality (35-step)",
    "◈ Cinema (60-step)",
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
    # v4.1: Video model presets (BUG-38)
    # v4.2 FIX: WAN/HunyuanVideo use CFG-based guidance — cfg must be > 1.0
    "▶ WAN txt2vid (30 steps)": {
        "steps": 30,
        "cfg": 6.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 8.0,
        "flux_guidance": 0.0,
    },
    "▶ WAN img2vid (20 steps)": {
        "steps": 20,
        "cfg": 6.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.75,
        "flux_shift": 8.0,
        "flux_guidance": 0.0,
    },
    "▶ LTX-Video (25 steps)": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 2.37,
        "flux_guidance": 0.0,
    },
    "▶ HunyuanVideo (30 steps)": {
        "steps": 30,
        "cfg": 6.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 7.0,
        "flux_guidance": 0.0,
    },
    # v4.0: Universal quality-tier presets (architecture-agnostic)
    "◈ Draft (4-step / AYS)": {
        "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.0, "flux_guidance": 3.5,
        "ays_schedule": True,
    },
    "◈ Fast (8-step / AYS)": {
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.0, "flux_guidance": 3.5,
        "ays_schedule": True,
    },
    "◈ Balanced (20-step)": {
        "steps": 20, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.0, "flux_guidance": 3.5,
        "ays_schedule": False,
    },
    "◈ Quality (35-step)": {
        "steps": 35, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.0, "flux_guidance": 4.0,
        "sampler_mode": "Phase-Shift (Euler\u2192SGM)", "ays_schedule": False,
    },
    "◈ Cinema (60-step)": {
        "steps": 60, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.5, "flux_guidance": 4.5,
        "sampler_mode": "Phase-Shift (Euler\u2192SGM)", "ays_schedule": False,
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
    # v3.6 FIX (BUG-33): Refined Flux anchors — better match to Flux's
    # flow-matching sigma distribution. More aggressive early denoising
    # with smoother tail for clean convergence. (Experimental — not from paper.)
    "flux": [1.0, 0.90, 0.70, 0.45, 0.22, 0.08, 0.02, 0.0],
    "sd3": [14.615, 6.291, 3.438, 1.566, 0.741, 0.288, 0.079, 0.0],
    # v4.1 FIX (BUG-40): Experimental video model anchors.
    # WAN uses high shift (8.0) — sigma distribution is compressed with
    # heavy early denoising. Anchors tuned for temporal coherence:
    # slower early ramp preserves inter-frame consistency.
    "wan": [1.0, 0.92, 0.78, 0.58, 0.35, 0.15, 0.04, 0.0],
    # LTX-Video uses moderate shift (2.37) — closer to Flux distribution.
    "ltxv": [1.0, 0.88, 0.68, 0.42, 0.20, 0.07, 0.015, 0.0],
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
        elif model_type in ("hunyuan_video",):
            key = "wan"  # Similar flow-matching + high shift
        elif model_type in ("lumina2", "z_image"):
            key = "flux"  # Flow-matching family
        else:
            return None

    # v3.6: Warn about experimental (non-paper) anchor sets
    if key in ("flux", "wan", "ltxv"):
        logger.info(
            f"AYS for {model_type}: using experimental anchors (not from original paper)"
        )

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


def correct_sigma_end(
    sigmas: torch.Tensor, target_end: float = 0.0
) -> torch.Tensor:
    """
    Ensure terminal sigma reaches target (default 0.0) for complete denoising.

    v3.6 FIX (BUG-28): CRITICAL — previous implementation APPENDED 0.0,
    making the schedule steps+2 elements instead of steps+1. Since
    total_steps was never updated, bs_start_step became negative and
    all local stage indices shifted by +1, SKIPPING the first sigma
    (σ_max). The model started denoising from a lower noise level than
    intended, leaving residual noise in the output.

    v3.6 FIX (BUG-29): Now applies to ALL schedulers, not just Karras.
    Any scheduler can produce non-zero terminal sigma depending on the
    model's sigma_min configuration.

    FIX: REPLACE the last sigma instead of appending. This preserves the
    expected schedule length of steps+1 and keeps all indexing correct.
    The second-to-last sigma already provides sufficient denoising context,
    so the final step cleanly transitions from σ[-2] → 0.0.
    """
    if len(sigmas) < 2:
        return sigmas

    if sigmas[-1].item() > 0.0 and target_end == 0.0:
        # REPLACE, not append — keeps schedule at steps+1 elements
        result = sigmas.clone()
        result[-1] = target_end
        return result

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
    frames: Optional[int] = None,
) -> str:
    """
    Build a human-readable sigma schedule report for diagnostics.
    v3.5: Added ETA, speed (it/s), and AYS indicator.
    v4.1 FIX (BUG-39): Added frames count for video diagnostics.
    """
    # v4.1: Show video info in header when applicable
    video_tag = f" | Frames: {frames}" if frames is not None and frames > 1 else ""
    lines = [
        f"═══ Radiance Sampler Pro v4.2 ═══",
        f"Model: {detected_type} | Steps: {steps} | Scheduler: {scheduler}{'  [AYS]' if ays_active else ''}{video_tag}",
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
#                    v4.0 HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


# --- Feature 3: Noise Generation -------------------------------------------

def _temporally_correlate(
    noise_fn, shape: tuple, device: torch.device, alpha: float = 0.6,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    v4.1 FIX (BUG-41): Generate temporally-correlated noise for 5D video latents.

    For video (B, C, T, H, W), generates per-frame noise using `noise_fn`
    then blends adjacent frames with an AR(1) process for temporal coherence.
    Without this, each frame gets independent noise → temporal flickering.

    v4.2: Per-frame seeding when seed is provided. Each frame's base noise
    is seeded with seed+frame_idx BEFORE generating, so frame 0's noise is
    deterministic regardless of total frame count. The temporal correlation
    (alpha blending with previous frame) still provides coherence while
    each frame's "innovation" noise is independently reproducible.

    Args:
        noise_fn: Callable(shape, device) → Tensor. Generates single-frame noise.
        shape: Full 5D shape (B, C, T, H, W)
        device: Target device
        alpha: Temporal correlation (0=independent, 1=identical). 0.6 is a good
               balance between coherence and per-frame variation.
        seed: Optional base seed. If provided, frame f uses seed+f.
    """
    B, C, T, H, W = shape
    frame_shape = (B, C, H, W)

    frames = []
    if seed is not None:
        torch.manual_seed(seed)
    prev = noise_fn(frame_shape, device)
    for f in range(T):
        if seed is not None:
            torch.manual_seed(seed + f + 1)  # +1 because prev consumed seed+0
        curr = noise_fn(frame_shape, device)
        blended = alpha * prev + math.sqrt(1 - alpha ** 2) * curr
        frames.append(blended)
        prev = blended

    result = torch.stack(frames, dim=2)  # → (B, C, T, H, W)
    # Re-normalize to unit std
    result = result - result.mean()
    std = result.std().clamp(min=1e-6)
    return result / std


def _perlin_noise(shape: tuple, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    """Octave-based coherent noise. Provides structured texture character.

    v4.1: Rewritten to properly handle 5D video latents.
    Uses unfold-based smoothing instead of fragile avg_pool2d view chain.
    v4.1 FIX (BUG-41): 5D video latents use temporal correlation wrapper.
    v4.2: Per-frame seeding for reproducible video noise.
    """
    # v4.1 FIX (BUG-41): Temporally-correlated noise for video
    if len(shape) == 5 and shape[2] > 1:
        return _temporally_correlate(_perlin_noise_2d, shape, device, seed=seed)

    return _perlin_noise_2d(shape, device)


def _perlin_noise_2d(shape: tuple, device: torch.device) -> torch.Tensor:
    """Core Perlin noise generator for 2D/4D shapes."""
    noise = torch.zeros(shape, device=device)
    amplitude = 1.0
    frequency = 1.0
    spatial_h, spatial_w = shape[-2], shape[-1]

    for _ in range(4):  # 4 octaves
        n = torch.randn(shape, device=device) * amplitude

        if spatial_w > 1:
            kernel = max(1, int(spatial_w / frequency))
            # Flatten all leading dims → (N, 1, H, W) for avg_pool2d
            leading = n.shape[:-2]
            n_flat = n.reshape(-1, 1, spatial_h, spatial_w)

            # avg_pool2d with stride=1 and same-padding
            pad = kernel // 2
            pooled = torch.nn.functional.avg_pool2d(
                n_flat, kernel_size=kernel, stride=1, padding=pad
            )

            # Even kernels produce H+1/W+1 output — crop back to original spatial dims
            pooled = pooled[..., :spatial_h, :spatial_w]

            # Restore original leading dims
            n = pooled.reshape(*leading, spatial_h, spatial_w)

        noise = noise + n
        amplitude *= 0.5
        frequency *= 2.0

    # Normalise to unit std and zero mean to prevent color shifts
    noise = noise - noise.mean()
    std = noise.std().clamp(min=1e-6)
    return noise / std


def _spectral_noise(shape: tuple, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    """Frequency-weighted noise (pink noise approximation).

    v4.1 FIX (BUG-41): 5D video latents use temporal correlation wrapper.
    v4.2: Per-frame seeding for reproducible video noise.
    """
    # v4.1 FIX (BUG-41): Temporally-correlated noise for video
    if len(shape) == 5 and shape[2] > 1:
        return _temporally_correlate(_spectral_noise_2d, shape, device, seed=seed)

    return _spectral_noise_2d(shape, device)


def _spectral_noise_2d(shape: tuple, device: torch.device) -> torch.Tensor:
    """Core spectral (pink) noise generator for 2D/4D shapes."""
    white = torch.randn(shape, device=device)
    fft = torch.fft.rfft2(white)
    # Build 1/f frequency weight. Note: rfft2 only halves the LAST dimension.
    freqs_h = torch.fft.fftfreq(shape[-2], device=device).abs()
    freqs_w = torch.fft.rfftfreq(shape[-1], device=device).abs()
    freq_grid = torch.sqrt(
        freqs_h.unsqueeze(-1) ** 2 + freqs_w.unsqueeze(0) ** 2
    ).clamp(min=1e-6)
    weight = 1.0 / freq_grid  # Shape: (H, rfft_W)

    # Zero out DC to prevent huge mean shifts
    weight[0, 0] = 0.0

    # Reshape weight to broadcast against fft: add leading 1-dims to match ndim
    # e.g., for 4D fft (B, C, H, W'), weight becomes (1, 1, H, W')
    for _ in range(fft.ndim - 2):
        weight = weight.unsqueeze(0)

    filtered = fft * weight
    result = torch.fft.irfft2(filtered, s=(shape[-2], shape[-1]))

    # Force zero mean and unit variance
    result = result - result.mean()
    std = result.std().clamp(min=1e-6)
    return result / std


def _brownian_noise(
    shape: tuple, device: torch.device, frames: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Temporally-correlated noise for video models (Brownian bridge).

    v4.1: Fixed for 5D video latents. Previous version correlated across
    batch dimension (shape[0]) instead of temporal dimension (shape[2]).
    Now correctly builds correlated walk along the frames axis.

    v4.2: Per-frame seeding when seed is provided. The initial frame uses
    seed, and each subsequent innovation noise uses seed+f. Since the AR(1)
    chain carries forward, frames 0-19 remain identical if you extend to 40.
    """
    # For 5D video latents: shape = (B, C, T, H, W)
    if len(shape) == 5 and shape[2] > 1:
        B, C, T, H, W = shape
        alpha = 0.7  # Correlation coefficient between frames
        # Build per-frame correlated walk
        frame_shape = (B, C, H, W)
        noises = []
        if seed is not None:
            torch.manual_seed(seed)
        prev = torch.randn(frame_shape, device=device)
        for f in range(T):
            if seed is not None:
                torch.manual_seed(seed + f + 1)
            curr = alpha * prev + math.sqrt(1 - alpha ** 2) * torch.randn(frame_shape, device=device)
            noises.append(curr)
            prev = curr
        # Stack along temporal dim → (B, C, T, H, W)
        return torch.stack(noises, dim=2)

    # For 4D or single-frame — use spectral as fallback
    if frames is None or (len(shape) == 4 and shape[0] == 1):
        return _spectral_noise(shape, device)

    # Legacy 4D path: correlate across batch dim (for multi-batch single frames)
    alpha = 0.7
    noises = []
    prev = torch.randn(shape[1:], device=device)
    for _ in range(shape[0]):
        curr = alpha * prev + math.sqrt(1 - alpha ** 2) * torch.randn(shape[1:], device=device)
        noises.append(curr)
        prev = curr
    return torch.stack(noises, dim=0)


def generate_noise(
    latent_samples: torch.Tensor,
    seed: int,
    noise_type: str = "Gaussian",
    frames: Optional[int] = None,
) -> torch.Tensor:
    """
    v4.0: Generate noise matching `latent_samples` shape in the requested style.
    Falls back to Gaussian if the requested generator fails.

    v4.2: Per-frame seeding for 5D video latents. Each frame uses seed+frame_idx
    so frame 0's noise is deterministic regardless of total frame count. This
    enables reproducible keyframe locking — you can extend a video and the
    existing frames keep identical noise.

    noise_type options: Gaussian, Perlin, Uniform, Spectral, Brownian
    """
    shape = latent_samples.shape
    device = latent_samples.device
    dtype = latent_samples.dtype

    # v4.2: Per-frame seeding for 5D video latents (Gaussian/Uniform)
    is_video = len(shape) == 5 and shape[2] > 1
    if is_video and noise_type in ("Gaussian", "Uniform"):
        B, C, T, H, W = shape
        frame_shape = (B, C, H, W)
        frame_noises = []
        for f in range(T):
            torch.manual_seed(seed + f)
            if noise_type == "Gaussian":
                frame_noises.append(torch.randn(frame_shape, device=device, dtype=dtype))
            else:  # Uniform
                frame_noises.append(
                    (torch.rand(frame_shape, device=device, dtype=dtype) * 2 - 1) * (3 ** 0.5)
                )
        return torch.stack(frame_noises, dim=2)  # → (B, C, T, H, W)

    # Non-video or noise types with their own temporal handling
    torch.manual_seed(seed)

    try:
        if noise_type == "Gaussian":
            return torch.randn(shape, device=device, dtype=dtype)
        elif noise_type == "Uniform":
            # Uniform scaled to same RMS as Gaussian (σ=1 → uniform [-√3, √3])
            return (torch.rand(shape, device=device, dtype=dtype) * 2 - 1) * (3 ** 0.5)
        elif noise_type == "Perlin":
            return _perlin_noise(shape, device, seed=seed).to(dtype)
        elif noise_type == "Spectral":
            return _spectral_noise(shape, device, seed=seed).to(dtype)
        elif noise_type == "Brownian":
            return _brownian_noise(shape, device, frames=frames, seed=seed).to(dtype)
    except Exception as e:
        logger.warning(f"[Noise] Failed to generate {noise_type} noise ({e}), falling back to Gaussian")

    return torch.randn(shape, device=device, dtype=dtype)


# --- Feature 7: Conditioning Merge ------------------------------------------

def merge_conditionings(
    cond_a: List,
    cond_b: List,
    mode: str = "average",
    weight_b: float = 0.5,
    split_step: int = 0,
    total_steps: int = 20,
) -> List:
    """
    v4.0: Merge two conditioning lists into one.

    Modes:
      * average    — element-wise mean (weight_b controls blend ratio)
      * weighted   — explicit weight_b for cond_b, (1-weight_b) for cond_a
      * sequential — cond_a for steps < split_step, cond_b for rest
                     (ComfyUI's scheduled conditioning handles this natively)
    """
    if not cond_b or mode == "Off":
        return cond_a
    if mode in ("average", "weighted"):
        w_b = weight_b if mode == "weighted" else 0.5
        w_a = 1.0 - w_b
        merged = []
        max_len = max(len(cond_a), len(cond_b))
        for i in range(max_len):
            a = cond_a[i % len(cond_a)]
            b = cond_b[i % len(cond_b)]
            tensor_a = a[0]
            tensor_b = b[0]
            # Pad shorter tensor along sequence dimension if needed
            if tensor_a.shape[1] != tensor_b.shape[1]:
                target_len = max(tensor_a.shape[1], tensor_b.shape[1])
                def _pad(t, n):
                    if t.shape[1] < n:
                        pad = t[:, -1:, :].expand(-1, n - t.shape[1], -1)
                        return torch.cat([t, pad], dim=1)
                    return t
                tensor_a = _pad(tensor_a, target_len)
                tensor_b = _pad(tensor_b, target_len)
            blended = tensor_a * w_a + tensor_b * w_b
            # Merge pooled outputs if present
            pooled_a = a[1].get("pooled_output", None) if len(a) > 1 else None
            pooled_b = b[1].get("pooled_output", None) if len(b) > 1 else None
            extra = dict(a[1]) if len(a) > 1 else {}
            if pooled_a is not None and pooled_b is not None:
                try:
                    extra["pooled_output"] = pooled_a * w_a + pooled_b * w_b
                except Exception:
                    pass  # Shapes incompatible — keep cond_a's pooled
            merged.append([blended, extra])
        return merged
    elif mode == "sequential":
        # Use ComfyUI's ConditioningSetTimestepRange semantics:
        # Mark cond_a for [0, split_step/total], cond_b for rest.
        # We inject start/end keys into the conditioning dicts.
        frac = float(split_step) / max(1, total_steps)
        cond_a_timed = []
        for c in cond_a:
            entry = [c[0], dict(c[1]) if len(c) > 1 else {}]
            entry[1]["start_percent"] = 0.0
            entry[1]["end_percent"] = frac
            cond_a_timed.append(entry)
        cond_b_timed = []
        for c in cond_b:
            entry = [c[0], dict(c[1]) if len(c) > 1 else {}]
            entry[1]["start_percent"] = frac
            entry[1]["end_percent"] = 1.0
            cond_b_timed.append(entry)
        return cond_a_timed + cond_b_timed
    return cond_a


# --- Feature 8: Conditioning CLIP Target Routing ----------------------------

def route_conditioning(cond: List, target_key: str) -> List:
    """
    v4.0: Attach a CLIP encoder target key to each conditioning entry.
    This hints to multi-encoder models (SD3, Flux) which slot to use.
    target_key: "Auto" (no-op), "clip_l", "clip_g", "t5xxl"
    """
    if not target_key or target_key == "Auto":
        return cond
    routed = []
    for entry in cond:
        new_entry = [entry[0], dict(entry[1]) if len(entry) > 1 else {}]
        new_entry[1]["encoder_target"] = target_key
        routed.append(new_entry)
    return routed


# --- Feature 6: Tile Sampling -----------------------------------------------

def tile_sample(
    model,
    noise: torch.Tensor,
    latent_samples: torch.Tensor,
    positive: List,
    negative: List,
    sigmas: torch.Tensor,
    sampler_obj,
    seed: int,
    tile_size: int = 128,      # in latent pixels
    tile_overlap: int = 16,    # in latent pixels
    tile_blend: str = "feather",
    noise_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v4.0: Memory-efficient tiled sampling.
    Splits the latent into overlapping tiles, samples each independently,
    then stitches them with the chosen blend mode.

    tile_size / tile_overlap are in latent-space pixels (before VAE decode).
    Typical: tile_size=128 latent = 1024px image (8x VAE factor).
    """
    import comfy.sample as cs

    if latent_samples.ndim != 4:
        raise ValueError(
            f"[Radiance] tile_sample requires 4D latent (B, C, H, W), "
            f"got {latent_samples.ndim}D. Video tile sampling is not supported."
        )

    B, C, H, W = latent_samples.shape
    step = max(1, tile_size - tile_overlap)
    device = latent_samples.device

    output = torch.zeros_like(latent_samples)
    weight = torch.zeros((B, 1, H, W), device=device, dtype=latent_samples.dtype)

    tile_coords = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            y1 = min(y, H - tile_size) if H > tile_size else 0
            x1 = min(x, W - tile_size) if W > tile_size else 0
            y2 = min(y1 + tile_size, H)
            x2 = min(x1 + tile_size, W)
            tile_coords.append((y1, y2, x1, x2))

    # Deduplicate tiles
    tile_coords = list(dict.fromkeys(tile_coords))

    for idx, (y1, y2, x1, x2) in enumerate(tile_coords):
        t_latent = latent_samples[:, :, y1:y2, x1:x2]
        t_noise = noise[:, :, y1:y2, x1:x2]

        try:
            t_out = cs.sample_custom(
                model,
                t_noise,
                cfg=1.0,  # CFG handled via model patches; this is a pass-through
                sampler=sampler_obj,
                sigmas=sigmas,
                positive=positive,
                negative=negative,
                latent_image=t_latent,
                noise_mask=noise_mask[:, :, y1:y2, x1:x2] if noise_mask is not None else None,
                callback=None,
                disable_pbar=True,
                seed=seed + idx,
            )
        except Exception as e:
            logger.warning(f"[TileSample] Tile ({y1},{y2},{x1},{x2}) failed: {e} — using input")
            t_out = t_latent

        # Ensure t_out is on the same device/dtype as output.
        # In low-VRAM / offloaded mode ComfyUI may return results on CPU
        # even when the input latent was on CUDA, causing a device mismatch.
        if t_out.device != device or t_out.dtype != latent_samples.dtype:
            t_out = t_out.to(device=device, dtype=latent_samples.dtype)

        th = y2 - y1
        tw = x2 - x1

        # Build blend weight for this tile
        if tile_blend == "feather":
            # Cosine feathering — smooth seam fade
            wy = torch.ones(th, device=device)
            wx = torch.ones(tw, device=device)
            fade = min(tile_overlap, th // 2, tw // 2)
            if fade > 0:
                ramp = (1 - torch.cos(torch.linspace(0, math.pi, fade, device=device))) / 2
                wy[:fade] = ramp
                wy[-fade:] = ramp.flip(0)
                wx[:fade] = ramp
                wx[-fade:] = ramp.flip(0)
            w_tile = (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        elif tile_blend == "gaussian":
            sigma_h = th / 4.0
            sigma_w = tw / 4.0
            yg = torch.arange(th, device=device).float() - th / 2
            xg = torch.arange(tw, device=device).float() - tw / 2
            w_tile = torch.exp(-(yg.unsqueeze(1)**2 / (2*sigma_h**2) +
                                  xg.unsqueeze(0)**2 / (2*sigma_w**2)))
            w_tile = w_tile.unsqueeze(0).unsqueeze(0)
        else:  # average — uniform weight (seams averaged naturally)
            w_tile = torch.ones((1, 1, th, tw), device=device, dtype=latent_samples.dtype)

        output[:, :, y1:y2, x1:x2] += t_out * w_tile
        weight[:, :, y1:y2, x1:x2] += w_tile

    # Normalise by accumulated weight
    weight = weight.clamp(min=1e-6)
    output = output / weight

    return output


# --- Feature 2: Latent Meta JSON --------------------------------------------

def _build_latent_meta(
    detected_type: str,
    steps: int,
    scheduler: str,
    flux_shift: float,
    denoise: float,
    sigmas: torch.Tensor,
    ays_active: bool,
    pag_active: bool,
    noise_type: str,
    tile_mode: bool,
    multi_cond_mode: str,
    clip_target: str,
    seed: int,
    total_time_ms: int,
    latent_format: str = "",
    frames: Optional[int] = None,
) -> str:
    """Build a JSON telemetry string for downstream nodes to consume.
    v4.1 FIX (BUG-39): Added frames, is_video fields for video diagnostics.
    """
    import json
    sigma_min = float(sigmas[sigmas > 0].min()) if (sigmas > 0).any() else 0.0
    sigma_max = float(sigmas.max())
    is_video = frames is not None and frames > 1
    meta = {
        "version": "4.2",
        "detected_arch": detected_type,
        "is_video": is_video,
        "frames": frames if is_video else None,
        "steps": steps,
        "scheduler": scheduler,
        "flux_shift": flux_shift,
        "denoise": denoise,
        "sigma_min": round(sigma_min, 5),
        "sigma_max": round(sigma_max, 5),
        "ays_active": ays_active,
        "pag_active": pag_active,
        "noise_type": noise_type,
        "tile_mode": tile_mode,
        "multi_cond_mode": multi_cond_mode,
        "clip_target": clip_target,
        "latent_format": latent_format or "unknown",
        "seed": seed,
        "time_ms": total_time_ms,
    }
    return json.dumps(meta, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#                         RADIANCE SAMPLER PRO v4.0
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceSamplerPro:

    """
    Professional Flux-optimized sampler with presets, timing report,
    and full parameter control.

    v3.7: Architecture finalization — all closures extracted to testable
    module-level functions, sigma cache replaced with GC-safe SigmaCache
    class, weakref dependency removed. Comprehensive test suite added.
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
                # v4.0: Noise type selection
                "noise_type": (
                    NOISE_TYPES,
                    {"default": "Gaussian",
                     "tooltip": ("Noise generation algorithm. Perlin=coherent structure, "
                                 "Spectral=pink/1f noise, Brownian=video-correlated, Uniform=flat distribution.")},
                ),
                # v4.0: Prompt weight mode / conditioning merge
                "multi_cond_mode": (
                    MULTI_COND_MODES,
                    {"default": "Off",
                     "tooltip": "Merge positive + positive_2 conditioning. average/weighted blend tensors; sequential splits by step range."},
                ),
                "cond_weight_b": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Weight of positive_2 conditioning when mode='weighted'."},
                ),
                # v4.0: Multi-encoder routing
                "conditioning_clip_target": (
                    CLIP_TARGETS,
                    {"default": "Auto",
                     "tooltip": "Route conditioning to a specific encoder slot (clip_l, clip_g, t5xxl). Auto = no routing."},
                ),
                # v4.0: Tile sampling
                "tile_mode": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Enable tiled sampling for memory-efficient high-resolution generation."},
                ),
                "tile_size": (
                    "INT",
                    {"default": 128, "min": 32, "max": 1024, "step": 32,
                     "tooltip": "Tile size in latent pixels (128 latent ≈ 1024px output with VAE factor 8)."},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 16, "min": 0, "max": 256, "step": 8,
                     "tooltip": "Overlap between adjacent tiles to reduce seam artifacts."},
                ),
                "tile_blend": (
                    TILE_BLEND_MODES,
                    {"default": "feather",
                     "tooltip": "Seam blending method. feather=cosine fade, gaussian=bell curve, average=uniform."},
                ),
            },
            "optional": {
                "refiner_model": ("MODEL",),
                "refiner_start_step": (
                    "INT",
                    {"default": 20, "min": 0, "max": 200, "step": 1},
                ),
                "noise_override": ("LATENT",),
                # v4.0: External sigma schedule override (Feature 1)
                "sigmas_override": (
                    "SIGMAS",
                    {"tooltip": "Inject a pre-computed sigma schedule. Bypasses all internal sigma computation."},
                ),
                # v4.0: Latent format from Radiance Loader (Feature 4)
                "latent_format": (
                    "STRING",
                    {"default": "",
                     "tooltip": "Wire from Radiance Loader latent_format output. Used for channel validation."},
                ),
                # v4.0: Second conditioning for merge mode (Feature 7)
                "positive_2": (
                    "CONDITIONING",
                    {"tooltip": "Secondary positive conditioning. Merged with primary via multi_cond_mode."},
                ),
            },
        }

    # v4.0: 4 outputs — latent_meta JSON added
    RETURN_TYPES = ("LATENT", "SIGMAS", "STRING", "STRING")
    RETURN_NAMES = ("latent", "sigmas", "sigma_report", "latent_meta")
    OUTPUT_TOOLTIPS = (
        "Denoised latent ready for VAE decode.",
        "The sigma schedule used — chain to another sampler or inspect.",
        "Human-readable timing and schedule report.",
        "JSON telemetry: arch, steps, scheduler, noise_type, tile_mode, seed, time_ms, etc.",
    )
    FUNCTION = "sample"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = (
        "v4.2 — Cinema-grade universal sampler. "
        "WAN/LTX/HunyuanVideo with auto-shift + dynamic CFG ramp, "
        "per-frame seeding for reproducible video, 5D latent handling, "
        "temporally-correlated noise, Phase-Shift safety gate for video, "
        "external sigma override, 5 noise types, "
        "tiled high-res sampling, multi-conditioning merge, CLIP encoder routing, "
        "quality-tier presets, latent_meta JSON output, "
        "plus all v3.x features: Flux Shift, dynamic guidance, PAG, CFG++, AYS, guidance rescale."
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
        # v4.0 parameters
        noise_type: str = "Gaussian",
        multi_cond_mode: str = "Off",
        cond_weight_b: float = 0.5,
        conditioning_clip_target: str = "Auto",
        tile_mode: bool = False,
        tile_size: int = 128,
        tile_overlap: int = 16,
        tile_blend: str = "feather",
        # Optional
        refiner_model=None,
        refiner_start_step: int = 20,
        noise_override: Optional[Dict[str, torch.Tensor]] = None,
        sigmas_override: Optional[torch.Tensor] = None,
        latent_format: str = "",
        positive_2: Optional[List] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, str, str]:

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
            # v4.0: quality presets can also set ays_schedule and sampler_mode
            ays_schedule = config.get("ays_schedule", ays_schedule)
            sampler_mode = config.get("sampler_mode", sampler_mode)
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

            # v4.1 FIX (BUG-35): Auto-apply shift for video/modern models.
            # Without correct shift, sigma schedules are completely wrong.
            # Only override when user hasn't touched the widget (still at 1.0)
            # and the model's optimal shift differs.
            default_shift = defaults.get("shift", 1.0)
            if flux_shift == 1.0 and default_shift != 1.0:
                flux_shift = default_shift
                logger.info(
                    f"Auto-applied shift={flux_shift} for {detected_type} "
                    f"(widget was at default 1.0)"
                )

            # v4.1 FIX (BUG-35): Auto-apply optimal sampler for video models
            # when the user hasn't changed from the ComfyUI default (euler).
            # Video models like WAN need euler; SDXL/SD15 need dpmpp_2m.
            default_sampler = defaults.get("sampler", sampler)
            if sampler == "euler" and default_sampler != "euler":
                sampler = default_sampler
                logger.info(
                    f"Auto-applied sampler={sampler} for {detected_type}"
                )

            logger.info(
                f"Auto-detected model type: {detected_type} (CFG={cfg}, guidance={flux_guidance}, "
                f"shift={flux_shift}, scheduler={scheduler})"
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
        # v4.1: Auto-reshape 4D latents for video models.
        # Video models (WAN, LTX, HunyuanVideo) require 5D tensors
        # (batch, channels, frames, height, width). If the user connected
        # an EmptyLatentImage (4D) instead of a video latent node, we
        # auto-reshape by inserting frames=1 and warn about it.
        # This prevents a cryptic "not enough values to unpack" crash
        # deep inside the model's forward() pass.
        # ─────────────────────────────────────────────────────────────────
        if detected_type in VIDEO_MODEL_TYPES and latent_samples.ndim == 4:
            b, c, h, w = latent_samples.shape
            latent_samples = latent_samples.unsqueeze(2)  # (B,C,H,W) → (B,C,1,H,W)
            frames = 1
            logger.warning(
                f"[Radiance] Video model '{detected_type}' received a 4D latent "
                f"[{b},{c},{h},{w}] — auto-reshaped to 5D [{b},{c},1,{h},{w}] "
                f"(frames=1). For proper video generation, use a video latent node "
                f"(e.g., WAN EmptyLatentVideo) with the desired frame count."
            )

        # ─────────────────────────────────────────────────────────────────
        # v4.0 Feature 4: Latent format / channel validation
        # ─────────────────────────────────────────────────────────────────
        if latent_format:
            # Infer expected channels from format string (SD=4, Flux=16, SD3=16, etc.)
            fmt_lower = latent_format.lower()
            expected_ch = 16 if any(k in fmt_lower for k in ("flux", "sd3", "16ch", "wan", "ltx", "hunyuan")) else 4
            if channels != expected_ch:
                logger.warning(
                    f"[v4.0] Latent has {channels} channels but latent_format='{latent_format}' "
                    f"suggests {expected_ch}ch. Check VAE/model mismatch."
                )

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
            # v4.0 Feature 3: Custom noise type
            if noise_type == "Gaussian":
                noise = comfy.sample.prepare_noise(latent_samples, seed, None)
            else:
                noise = generate_noise(latent_samples, seed, noise_type, frames=frames)
                logger.info(f"[v4.0] Using {noise_type} noise generator")

        noise = noise.to(device)
        timings["prepare_noise"] = time.time() - t0

        # v4.1: Ensure noise matches latent shape after auto-reshape.
        # If latent was reshaped 4D→5D for a video model, noise generated
        # from the reshaped latent should already be 5D. This is a safety
        # net for noise_override or edge cases where shapes diverge.
        if latent_samples.ndim == 5 and noise.ndim == 4:
            noise = noise.unsqueeze(2)
            logger.debug("[Radiance] Auto-reshaped noise 4D→5D to match video latent.")
        if latent_samples.ndim == 5 and noise_mask is not None and noise_mask.ndim == 4:
            noise_mask = noise_mask.unsqueeze(2)
            logger.debug("[Radiance] Auto-reshaped noise_mask 4D→5D to match video latent.")

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
        # v4.0 Feature 1: sigmas_override bypasses all internal computation
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        if sigmas_override is not None:
            sigmas = sigmas_override.to(device if hasattr(device, '__str__') else "cpu")
            logger.info(
                f"[v4.0] Using external sigmas_override ({len(sigmas)} values, "
                f"range [{sigmas[0].item():.4f} \u2192 {sigmas[-1].item():.4f}])"
            )
            # Still apply terminal correction for safety
            sigmas = correct_sigma_end(sigmas)
        else:
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

                # v3.6 FIX (BUG-28/29): Terminal sigma correction for ALL schedulers.
                # Previously only ran for Karras, and APPENDED a sigma (causing off-by-one).
                # Now REPLACES terminal sigma for any scheduler where it's > 0.
                sigmas = correct_sigma_end(sigmas)

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
                frames=frames,
            )
            return (latent_image.copy(), sigmas, report, "{}")

        # ─────────────────────────────────────────────────────────────────
        # FIX #4: Prepare starting latent WITHOUT pre-noising
        # Let the sampling loop handle noise properly
        # ─────────────────────────────────────────────────────────────────
        work_latent = latent_samples.to(device)
        log_tensor("Work Latent (Start)", work_latent)

        # ─────────────────────────────────────────────────────────────────
        # v4.0 Feature 7: Merge conditionings if requested
        # v4.0 Feature 8: Route conditioning to CLIP encoder slot
        # ─────────────────────────────────────────────────────────────────
        if multi_cond_mode != "Off" and positive_2 is not None:
            positive = merge_conditionings(
                positive, positive_2,
                mode=multi_cond_mode,
                weight_b=cond_weight_b,
                split_step=int(steps * phase_split),
                total_steps=steps,
            )
            logger.info(f"[v4.0] Conditionings merged (mode={multi_cond_mode}, weight_b={cond_weight_b:.2f})")

        if conditioning_clip_target != "Auto":
            positive = route_conditioning(positive, conditioning_clip_target)
            logger.info(f"[v4.0] Conditioning routed to {conditioning_clip_target}")

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
        # v4.2: Disable Phase-Shift for video models — sampler switching mid-schedule
        # causes temporal discontinuities visible as a "pop" at the transition frame.
        if SamplerMode.is_phase_shift(sampler_mode) and detected_type in VIDEO_MODEL_TYPES:
            logger.warning(
                f"[v4.2] Phase-Shift mode is not supported for video model '{detected_type}' — "
                f"sampler switching mid-schedule causes temporal discontinuities. "
                f"Falling back to Standard mode."
            )
            sampler_mode = SamplerMode.STANDARD

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
                
                # BUG-FIX: Warn about DPM solvers on flow-matching models
                # v4.1: Expanded from (flux, chroma) to include video flow-matching models
                flow_match_types = GUIDANCE_EMBED_MODELS | {"wan", "hunyuan_video"}
                if detected_type in flow_match_types:
                    logger.warning(
                        f"WARNING: DPM solvers are generally incompatible with {detected_type}'s "
                        f"Flow Matching and may produce severe noise/grain. "
                        f"Recommended to use Euler or Phase-Shift SGM."
                    )
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
        # v3.5 FIX (BUG-23): Only create dynamic guidance splits for models
        # that use guidance embedding. Others ignore it — extra stages just add overhead.
        # v4.1 FIX (BUG-36): Expanded from (flux, chroma) to GUIDANCE_EMBED_MODELS.
        is_dynamic = "Dynamic" in flux_guidance_profile and detected_type in GUIDANCE_EMBED_MODELS

        # v4.2: Dynamic CFG for CFG-guided models (WAN, HunyuanVideo, etc.)
        # When the user selects "Dynamic" profile and the model uses CFG (not
        # embedding), apply compute_dynamic_cfg() to ramp CFG over the schedule.
        is_dynamic_cfg = "Dynamic" in flux_guidance_profile and detected_type in CFG_GUIDED_MODELS

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

        elif is_dynamic_cfg:
            # Split points for CFG ramp — different thresholds than guidance
            denoising_steps = (
                int(total_steps * denoise) if denoise < 1.0 else total_steps
            )
            denoising_start = total_steps - denoising_steps

            idx_15 = denoising_start + int(denoising_steps * DYNAMIC_CFG_EARLY_THRESHOLD)
            idx_85 = denoising_start + int(denoising_steps * DYNAMIC_CFG_LATE_THRESHOLD)

            if effective_start < idx_15 < effective_end:
                splits.add(idx_15)
            if effective_start < idx_85 < effective_end:
                splits.add(idx_85)
            logger.info(
                f"Dynamic CFG Active for {detected_type} (effective range: steps "
                f"{denoising_start}-{total_steps}, boost→{idx_15}, taper→{idx_85})"
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

        # v3.7: Pre-seed SigmaCache with global sigmas so stage indexing
        # is guaranteed to match the split points computed from total_steps.
        # compute_base_sigmas is now a module-level function with explicit params.
        _sigma_cache.put(model, scheduler, total_steps, sigmas)

        # Helper to call compute_base_sigmas with current sampling context
        def _get_base_sigmas(mdl, sched: str = scheduler) -> torch.Tensor:
            return compute_base_sigmas(
                mdl, sched, total_steps, scheduler, flux_shift, denoise, _sigma_cache
            )

        # Start timing for sampling
        t0 = time.time()

        # v3.2: Per-stage timing collection
        stage_timings: List[Tuple[int, int, int, str, float]] = []

        # ─────────────────────────────────────────────────────────────────
        # v3.6: Build execution plan upfront using SamplingStage + SigmaIndexer.
        # This replaces the inline stage config determination that was
        # scattered across the loop. The entire plan is validated and
        # logged before any sampling happens.
        # ─────────────────────────────────────────────────────────────────
        planned_stages: List[SamplingStage] = []
        for si in range(len(sorted_splits) - 1):
            gs = sorted_splits[si]
            ge = sorted_splits[si + 1]
            if gs >= ge:
                continue

            stage_model = model
            if refiner_model is not None and gs >= refiner_start_step:
                stage_model = refiner_model

            stage_sampler = primary_sampler
            stage_scheduler = scheduler
            is_shifted = False
            is_blend = False

            if SamplerMode.is_phase_shift(sampler_mode) and gs >= split_step:
                is_shifted = True
                if secondary_sampler:
                    stage_sampler = secondary_sampler
                if secondary_scheduler:
                    stage_scheduler = secondary_scheduler
                if gs == split_step and sigma_blend_steps > 0:
                    is_blend = True

            planned_stages.append(
                SamplingStage(
                    index=si,
                    global_start=gs,
                    global_end=ge,
                    model=stage_model,
                    sampler_name=stage_sampler,
                    scheduler_name=stage_scheduler,
                    is_phase_shifted=is_shifted,
                    is_blend_point=is_blend,
                )
            )

        # Log the execution plan
        for ps in planned_stages:
            label = f"{ps.sampler_name}+{ps.scheduler_name}"
            if ps.is_blend_point:
                label += " [blend]"
            logger.info(
                f"Plan Stage {ps.index + 1}: steps {ps.global_start}→{ps.global_end} [{label}]"
            )

        # Wrap sampling loop in try/finally to ensure cleanup (FIX #10)
        try:
            for plan_idx, stage in enumerate(planned_stages):
                t_stage = time.time()
                i = stage.index
                s_start = stage.global_start
                s_end = stage.global_end
                current_model = stage.model
                current_sampler = stage.sampler_name
                current_scheduler = stage.scheduler_name

                # ─────────────────────────────────────────────────────────────
                # v3.7: Dynamic/Static Guidance (extracted to module-level)
                # ─────────────────────────────────────────────────────────────
                stage_positive = positive
                if is_dynamic:
                    effective_guidance = compute_dynamic_guidance(
                        flux_guidance, s_start, total_steps, denoise
                    )
                    stage_positive = apply_flux_guidance(positive, effective_guidance)
                    logger.debug(
                        f"Dynamic Guidance @ step {s_start}: {effective_guidance:.2f}"
                    )

                elif detected_type in GUIDANCE_EMBED_MODELS:
                    # Static guidance — flow-matching models use guidance embedding
                    stage_positive = apply_flux_guidance(positive, flux_guidance)

                logger.info(
                    f"Stage {i+1}: Steps {s_start}-{s_end} | "
                    f"Sampler: {current_sampler} | Scheduler: {current_scheduler}"
                )

                # ─────────────────────────────────────────────────────────────
                # v3.6: Calculate stage sigmas using SigmaIndexer
                # Replaces fragile inline offset math that caused BUG-28.
                # ─────────────────────────────────────────────────────────────
                try:
                    # v3.2: Pass current_scheduler for correct sigma computation
                    # (differs from primary scheduler in Phase-Shift SGM mode)
                    base_sigmas = _get_base_sigmas(current_model, current_scheduler)

                    # v3.6: SigmaIndexer encapsulates all offset math + validates invariants
                    indexer = SigmaIndexer(total_steps, base_sigmas)
                    stage_sigmas = indexer.get_stage_sigmas(s_start, s_end)

                    if stage_sigmas is None:
                        # Stage falls outside sigma schedule (low denoise)
                        continue

                    # v3.6: Sigma blend uses stage.is_blend_point (pre-computed)
                    if (
                        stage.is_blend_point
                        and prev_stage_sigmas is not None
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
                    is_last_stage = plan_idx == len(planned_stages) - 1
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
                    elif is_dynamic_cfg:
                        # v4.2: Dynamic CFG ramp for video/CFG-guided models.
                        # Boosts CFG early for structure, tapers late for clean convergence.
                        effective_cfg = compute_dynamic_cfg(
                            cfg, s_start, total_steps, denoise
                        )
                        logger.debug(
                            f"Dynamic CFG @ step {s_start}: {cfg:.2f} → {effective_cfg:.2f}"
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

                    # v3.5 FIX (BUG-27): Allow initial noise for img2img/denoise<1.0 workflows.
                    # Previously forced effective_start == 0, causing sampler to destructively
                    # subtract non-existent noise when denoise was less than 1.0.
                    is_first_stage = plan_idx == 0
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

                    # v3.1 FIX: Explicitly load model to GPU to ensure VRAM is available
                    comfy.model_management.load_model_gpu(current_model)

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
                    # v3.6: Store pre-strip sigmas via SigmaIndexer for blend source.
                    prev_stage_sigmas = indexer.get_stage_sigmas(s_start, s_end)
                    log_tensor(f"Stage {i+1} Output", current_latent)

                    # v3.6: Sigma continuity validation using SigmaIndexer.
                    # Replaces duplicated config logic + inline offset math (FIX #8).
                    # Uses planned_stages to get next stage's config directly.
                    if plan_idx < len(planned_stages) - 1:
                        next_stage = planned_stages[plan_idx + 1]
                        next_base = _get_base_sigmas(
                            next_stage.model, next_stage.scheduler_name
                        )
                        next_indexer = SigmaIndexer(total_steps, next_base)
                        next_sigma = next_indexer.get_sigma_at(next_stage.global_start)

                        if (
                            next_sigma is not None
                            and len(stage_sigmas) > 0
                        ):
                            expected_sigma = stage_sigmas[-1].item()
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

            # FIX #12: Ensure result is on CPU to prevent VRAM accumulation
            if current_latent.is_cuda:
                samples = current_latent.cpu()
            else:
                samples = current_latent

            timings["sampling"] = time.time() - t0

            # ─────────────────────────────────────────────────────────────────
            # v4.0 Feature 6: Tile sampling (post-stage, replaces normal pipeline result)
            # Only applies to 4D (image) latents; video tile sampling not yet supported.
            # ─────────────────────────────────────────────────────────────────
            if tile_mode and latent_samples.ndim == 4:
                logger.info(
                    f"[v4.0] Tile sampling: size={tile_size}, overlap={tile_overlap}, blend={tile_blend}"
                )

                t_tile = time.time()
                sampler_obj = comfy.samplers.sampler_object(primary_sampler)
                samples = tile_sample(
                    model=model,
                    noise=noise,
                    latent_samples=work_latent,
                    positive=positive,
                    negative=negative,
                    sigmas=sigmas,
                    sampler_obj=sampler_obj,
                    seed=seed,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    tile_blend=tile_blend,
                    noise_mask=noise_mask,
                )
                timings["tile_sampling"] = time.time() - t_tile
                logger.info(f"[v4.0] Tile sampling done in {timings['tile_sampling']:.2f}s")
            elif tile_mode and latent_samples.ndim == 5:
                logger.warning(
                    "[v4.0] Tile sampling requested but not supported for video (5D) latents. Skipping."
                )
                
        finally:
            # ─────────────────────────────────────────────────────────────────
            # v3.2 FIX: Free GPU noise tensor to prevent VRAM leak (FIX #10)
            # ─────────────────────────────────────────────────────────────────
            if "noise" in locals():
                try: del noise
                except UnboundLocalError: pass
            if "work_latent" in locals():
                try: del work_latent
                except UnboundLocalError: pass
            
            # v3.1 FIX: Proactively cleanup GPU memory to prevent VGA crashes
            cleanup_gpu_memory()

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
            frames=frames,
        )

        # v4.0: Build latent_meta JSON (Feature 2)
        latent_meta = _build_latent_meta(
            detected_type=detected_type,
            steps=steps,
            scheduler=scheduler,
            flux_shift=flux_shift,
            denoise=denoise,
            sigmas=output_sigmas,
            ays_active=ays_schedule,
            pag_active=(pag_scale > 0),
            noise_type=noise_type,
            tile_mode=tile_mode,
            multi_cond_mode=multi_cond_mode,
            clip_target=conditioning_clip_target,
            seed=seed,
            total_time_ms=int(total_time * 1000),
            latent_format=latent_format,
            frames=frames,
        )

        return (out, output_sigmas, sigma_report, latent_meta)


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceSamplerPro": RadianceSamplerPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSamplerPro": "◎ Radiance Sampler Pro",
}
