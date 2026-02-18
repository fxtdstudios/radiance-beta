"""
RADIANCE - LOADER NODES v2.1.0
-----------------------------------
Professional unified loader for Diffusion models, CLIPs, VAEs, LoRAs, and ControlNets.

v2.1.0 Fixes & Features (February 2026):
- FIX: LoRA now has separate model_strength and clip_strength per slot
- FIX: Preset enforces single/dual CLIP mode (prevents SD1.5 + dual CLIP crash)
- FIX: ControlNet now has strength, start_percent, end_percent controls
- FIX: CLIP weight dtype is independently controllable (critical for Flux T5XXL VRAM)
- FIX: Explicit CLIP type mapping for all architectures (no silent fallthrough)
- FIX: VRAM estimation now accounts for CLIP dtype separately
- NEW: Model caching — skip reload if same model already loaded
- NEW: Expanded presets (HunyuanVideo, Wan 2.1, LTX, PixArt, Kolors, AuraFlow)
- NEW: Progress logging during multi-model load sequence
- NEW: Load summary with timing

v2.0.1 Fixes:
- Fixed VRAM reporting (was returning total instead of free)
- Fixed built-in `type` shadowing → renamed to `model_type`
- Fixed bare except clause
- Added GGUF dtype handling
- Added input path validation
- ControlNet output gracefully returns None without breaking downstream
- LoRA failures raise errors instead of silent swallowing

v2.0 Features:
- Checkpoint presets with auto-configuration
- VRAM estimation before loading
- ControlNet bundling
- Model caching option

v1.1 Features:
- Single CLIP mode (optional clip_name2)
- GGUF quantization detection
- LoRA stacking support
"""

import os
import time
import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management

logger = logging.getLogger("radiance.loader")


# ═══════════════════════════════════════════════════════════════════════════════
#                         CHECKPOINT PRESETS (v2.1)
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PRESETS = {
    "None (Manual)": {},
    # ── Flux ──
    "→ Flux Dev": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": True,
        "vram_gb": 12,
    },
    "→ Flux Schnell": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": True,
        "vram_gb": 10,
    },
    # ── SD3.x ──
    "→ SD3.5 Large": {
        "model_type": "sd3.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": True,
        "vram_gb": 16,
    },
    "→ SD3.5 Medium": {
        "model_type": "sd3.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": True,
        "vram_gb": 10,
    },
    # ── SDXL ──
    "→ SDXL Base": {
        "model_type": "sdxl",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": True,
        "vram_gb": 8,
    },
    "→ SDXL Turbo": {
        "model_type": "sdxl",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": True,
        "vram_gb": 6,
    },
    # ── SD 1.5 ──
    "→ SD 1.5": {
        "model_type": "sd1.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": False,  # SD1.5 = single CLIP only
        "vram_gb": 4,
    },
    # ── Video Models ──
    "→ HunyuanVideo": {
        "model_type": "hunyuan_video",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": True,
        "vram_gb": 24,
    },
    "→ Wan 2.1": {
        "model_type": "wan",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": False,
        "vram_gb": 16,
    },
    "→ LTX Video": {
        "model_type": "ltx",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": False,
        "vram_gb": 12,
    },
    # ── Other Image Models ──
    "→ PixArt Sigma": {
        "model_type": "pixart",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": False,
        "vram_gb": 8,
    },
    "→ AuraFlow": {
        "model_type": "aura_flow",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": False,
        "vram_gb": 10,
    },
    "→ Kolors": {
        "model_type": "kolors",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "device": "default",
        "dual_clip": False,
        "vram_gb": 10,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                         VRAM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_vram_usage(
    model_type: str,
    weight_dtype: str,
    clip_dtype: str = "fp16",
    has_loras: bool = False,
    has_controlnet: bool = False,
) -> float:
    """
    Estimate VRAM usage in GB based on model configuration.
    v2.1: Now accounts for CLIP dtype separately.
    """
    base_vram = {
        "flux": 12.0,
        "sd3": 10.0,
        "sd3.5": 12.0,
        "sdxl": 6.5,
        "sd1.5": 3.5,
        "hunyuan_video": 20.0,
        "wan": 14.0,
        "ltx": 10.0,
        "pixart": 6.0,
        "aura_flow": 8.0,
        "kolors": 8.0,
    }.get(model_type, 8.0)

    # UNET dtype multiplier
    unet_multiplier = {
        "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
        "fp8_e4m3fn": 0.6, "fp8_e5m2": 0.6, "default": 1.0,
    }.get(weight_dtype, 1.0)

    # CLIP contribution (T5XXL is huge — dtype matters)
    clip_vram = {
        "flux": 4.5, "sd3": 3.0, "sd3.5": 3.5,
        "sdxl": 1.5, "sd1.5": 0.8,
        "hunyuan_video": 4.5, "wan": 3.0, "ltx": 2.0,
        "pixart": 2.0, "aura_flow": 2.0, "kolors": 3.0,
    }.get(model_type, 2.0)

    clip_multiplier = {
        "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
        "fp8_e4m3fn": 0.55, "fp8_e5m2": 0.55, "default": 1.0,
    }.get(clip_dtype, 1.0)

    # Total: UNET (scaled) + CLIP (scaled) + overhead
    vram = (base_vram * unet_multiplier) + (clip_vram * clip_multiplier)

    if has_loras:
        vram += 0.5
    if has_controlnet:
        vram += 2.0

    return round(vram, 1)


def get_available_vram() -> float:
    """Get available (free) VRAM in GB."""
    try:
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info(0)
            return round(free_mem / (1024**3), 1)
    except Exception:
        pass
    return 0.0


def get_total_vram() -> float:
    """Get total VRAM in GB."""
    try:
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            return round(total_mem / (1024**3), 1)
    except Exception:
        pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#                        CLIP TYPE MAPPING (v2.1 — explicit)
# ═══════════════════════════════════════════════════════════════════════════════

def get_clip_type_enum(model_type: str):
    """
    Map model_type string to comfy.sd.CLIPType enum.
    v2.1: Explicit mapping for every supported type — no silent fallthrough.
    """
    mapping = {
        "flux": comfy.sd.CLIPType.FLUX,
        "sd3": comfy.sd.CLIPType.SD3,
        "sd3.5": comfy.sd.CLIPType.SD3,
        "sdxl": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sd1.5": comfy.sd.CLIPType.STABLE_DIFFUSION,
    }

    # Newer model types — try to resolve, fall back to STABLE_DIFFUSION
    # These may have dedicated CLIPType enums in newer ComfyUI versions
    for name in ("hunyuan_video", "wan", "ltx", "pixart", "aura_flow", "kolors"):
        enum_name = name.upper().replace(".", "_")
        if hasattr(comfy.sd.CLIPType, enum_name):
            mapping[name] = getattr(comfy.sd.CLIPType, enum_name)
        # Also try common variants
        for variant in (enum_name, name.upper(), name.title().replace("_", "")):
            if hasattr(comfy.sd.CLIPType, variant):
                mapping[name] = getattr(comfy.sd.CLIPType, variant)
                break

    clip_type = mapping.get(model_type)
    if clip_type is None:
        # Try dynamic lookup as last resort
        enum_name = model_type.upper().replace(".", "_")
        if hasattr(comfy.sd.CLIPType, enum_name):
            clip_type = getattr(comfy.sd.CLIPType, enum_name)
        else:
            logger.warning(
                f"⚠ No CLIPType mapping for '{model_type}', "
                f"falling back to STABLE_DIFFUSION"
            )
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

    return clip_type


# ═══════════════════════════════════════════════════════════════════════════════
#                        MODEL CACHE (v2.1)
# ═══════════════════════════════════════════════════════════════════════════════

class _LRUCache:
    """
    Least-Recently-Used cache to prevent memory leaks.
    Maintains a fixed number of items (default=4).
    """

    def __init__(self, max_size=4):
        self._cache = {}
        self._access_order = []
        self._max_size = max_size

    def get(self, key: str):
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, obj):
        if key in self._cache:
             # Update existing
             self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
             # Evict least recently used (first in list)
             oldest_key = self._access_order.pop(0)
             del self._cache[oldest_key]
             logger.info(f"Evicted from cache: {oldest_key}")
        
        self._cache[key] = obj
        self._access_order.append(key)

    def has(self, key: str) -> bool:
        return key in self._cache

    def clear(self):
        self._cache.clear()
        self._access_order.clear()
        logger.info("Model cache cleared")

    @property
    def size(self) -> int:
        return len(self._cache)


_cache = _LRUCache(max_size=4)


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE UNIFIED LOADER v2.1
# ═══════════════════════════════════════════════════════════════════════════════

# Supported model types — keep in sync with presets + CLIPType mapping
MODEL_TYPES = ["sdxl", "sd3", "flux", "sd3.5", "sd1.5",
               "hunyuan_video", "wan", "ltx", "pixart", "aura_flow", "kolors"]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"]

CLIP_DTYPES = ["default", "fp16", "bf16", "fp8_e4m3fn", "fp32"]


class RadianceUnifiedLoader:
    """
    Unified Professional Loader v2.1 for Diffusion models.
    UNET + Dual CLIP + VAE + LoRA (separate model/clip strengths) + ControlNet (with controls).
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        clip_list = ["None"] + folder_paths.get_filename_list("text_encoders")
        controlnet_list = ["None"] + folder_paths.get_filename_list("controlnet")

        return {
            "required": {
                # ── Preset ──
                "preset": (list(CHECKPOINT_PRESETS.keys()), {
                    "default": "None (Manual)",
                    "tooltip": (
                        "Quick preset for common models. Auto-configures type, dtype, "
                        "device, and CLIP mode. Overridden values are logged."
                    ),
                }),
                # ── UNET ──
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Select the main diffusion model (UNET/Transformer).",
                }),
                "weight_dtype": (WEIGHT_DTYPES, {
                    "default": "default",
                    "tooltip": "UNET weight precision. 'fp8_e4m3fn' saves ~40% VRAM on Flux.",
                }),
                # ── CLIP ──
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Primary Text Encoder (e.g., T5XXL for Flux, CLIP-L for SDXL).",
                }),
                "model_type": (MODEL_TYPES, {
                    "default": "flux",
                    "tooltip": "Model architecture type. Controls CLIP loading strategy.",
                }),
                "clip_dtype": (CLIP_DTYPES, {
                    "default": "default",
                    "tooltip": (
                        "CLIP weight precision. Independent from UNET dtype. "
                        "For Flux: T5XXL fp8 saves ~4.7GB vs fp16."
                    ),
                }),
                "device": (["default", "cpu"], {
                    "default": "default",
                    "tooltip": "Device for CLIP loading. 'cpu' offloads CLIP to save VRAM.",
                }),
                # ── VAE ──
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "VAE for encoding/decoding latents.",
                }),
            },
            "optional": {
                # ── CLIP (secondary) ──
                "clip_name2": (clip_list, {
                    "default": "None",
                    "tooltip": (
                        "Secondary CLIP (e.g., CLIP-L for Flux, CLIP-G for SDXL). "
                        "Leave 'None' for single-CLIP architectures (SD1.5, PixArt, etc.)."
                    ),
                }),
                # ── LoRA 1 (separate model/clip strengths) ──
                "lora_1": (lora_list, {"default": "None"}),
                "lora_1_model_str": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA 1 strength applied to the UNET/diffusion model.",
                }),
                "lora_1_clip_str": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA 1 strength applied to CLIP text encoder.",
                }),
                # ── LoRA 2 ──
                "lora_2": (lora_list, {"default": "None"}),
                "lora_2_model_str": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                }),
                "lora_2_clip_str": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                }),
                # ── LoRA 3 ──
                "lora_3": (lora_list, {"default": "None"}),
                "lora_3_model_str": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                }),
                "lora_3_clip_str": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                }),
                # ── ControlNet (v2.1: with application controls) ──
                "controlnet_name": (controlnet_list, {
                    "default": "None",
                    "tooltip": "Optional ControlNet to load alongside model.",
                }),
                "controlnet_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "ControlNet application strength.",
                }),
                "controlnet_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "ControlNet start percent (0.0 = from beginning).",
                }),
                "controlnet_end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "ControlNet end percent (1.0 = until end).",
                }),
                # ── Options ──
                "check_vram": (["On", "Off"], {
                    "default": "On",
                    "tooltip": "Estimate VRAM and warn if insufficient before loading.",
                }),
                "use_cache": (["On", "Off"], {
                    "default": "On",
                    "tooltip": (
                        "Cache loaded models in memory. Skips disk reload when "
                        "re-queuing with the same configuration."
                    ),
                }),
                "lora_on_error": (["warn", "raise"], {
                    "default": "raise",
                    "tooltip": "'warn' logs and continues, 'raise' stops execution on LoRA failure.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONTROL_NET", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "CONTROLNET", "load_info")
    OUTPUT_IS_LIST = (False, False, False, False, False)
    FUNCTION = "load_radiance_stack"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = (
        "Unified Professional Loader v2.1 — UNET + CLIP + VAE + LoRA + ControlNet. "
        "Separate model/clip LoRA strengths, CLIP dtype control, ControlNet strength/range, "
        "model caching, expanded presets."
    )

    def load_radiance_stack(
        self,
        preset, unet_name, weight_dtype, clip_name1, model_type, clip_dtype, device, vae_name,
        clip_name2="None",
        lora_1="None", lora_1_model_str=1.0, lora_1_clip_str=1.0,
        lora_2="None", lora_2_model_str=1.0, lora_2_clip_str=1.0,
        lora_3="None", lora_3_model_str=1.0, lora_3_clip_str=1.0,
        controlnet_name="None",
        controlnet_strength=1.0, controlnet_start=0.0, controlnet_end=1.0,
        check_vram="On", use_cache="On", lora_on_error="raise",
    ):
        load_start = time.time()
        info_lines = []

        # ═══════════════════════════════════════════════════════════════
        # 0. APPLY PRESET (v2.1: also enforces CLIP mode)
        # ═══════════════════════════════════════════════════════════════
        if preset != "None (Manual)" and preset in CHECKPOINT_PRESETS:
            config = CHECKPOINT_PRESETS[preset]
            overrides = []

            prev_type = model_type
            prev_wdtype = weight_dtype
            prev_cdtype = clip_dtype
            prev_device = device

            model_type = config.get("model_type", model_type)
            weight_dtype = config.get("weight_dtype", weight_dtype)
            clip_dtype = config.get("clip_dtype", clip_dtype)
            device = config.get("device", device)

            # v2.1: Enforce CLIP mode from preset
            if "dual_clip" in config:
                if not config["dual_clip"] and clip_name2 != "None":
                    logger.info(
                        f"  ↳ Preset '{preset}' is single-CLIP architecture — "
                        f"ignoring clip_name2='{clip_name2}'"
                    )
                    clip_name2 = "None"
                    overrides.append("clip_name2: forced to None (single-CLIP)")

            if prev_type != model_type:
                overrides.append(f"model_type: {prev_type}→{model_type}")
            if prev_wdtype != weight_dtype:
                overrides.append(f"weight_dtype: {prev_wdtype}→{weight_dtype}")
            if prev_cdtype != clip_dtype:
                overrides.append(f"clip_dtype: {prev_cdtype}→{clip_dtype}")
            if prev_device != device:
                overrides.append(f"device: {prev_device}→{device}")

            if overrides:
                msg = f"✓ Preset '{preset}' overrode: {', '.join(overrides)}"
            else:
                msg = f"✓ Applied preset: {preset} (no overrides needed)"
            logger.info(msg)
            info_lines.append(msg)

        # ═══════════════════════════════════════════════════════════════
        # 1. VRAM ESTIMATION (v2.1: includes CLIP dtype)
        # ═══════════════════════════════════════════════════════════════
        has_loras = any(l != "None" for l in [lora_1, lora_2, lora_3])
        has_controlnet = controlnet_name and controlnet_name != "None"

        if check_vram == "On":
            estimated_vram = estimate_vram_usage(
                model_type, weight_dtype, clip_dtype, has_loras, has_controlnet
            )
            available_vram = get_available_vram()
            total_vram = get_total_vram()

            vram_msg = f"📊 VRAM: ~{estimated_vram}GB needed | {available_vram}GB free / {total_vram}GB total"
            logger.info(vram_msg)
            info_lines.append(vram_msg)

            if available_vram > 0 and estimated_vram > available_vram * 0.9:
                warn_msg = (
                    f"⚠ VRAM may be tight! Estimated {estimated_vram}GB needed, "
                    f"only {available_vram}GB free. Consider fp8 dtype or CPU offload."
                )
                logger.warning(warn_msg)
                info_lines.append(warn_msg)

        caching = use_cache == "On"

        # ═══════════════════════════════════════════════════════════════
        # 2. LOAD UNET (with caching)
        # ═══════════════════════════════════════════════════════════════
        t0 = time.time()
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if not unet_path:
            raise FileNotFoundError(
                f"❌ UNET model not found: '{unet_name}'. Was it deleted or moved?"
            )

        unet_cache_key = f"unet:{unet_path}:{weight_dtype}"
        if caching and _cache.has(unet_cache_key):
            model = _cache.get(unet_cache_key)
            logger.info(f"⚡ UNET from cache: {unet_name}")
            info_lines.append(f"⚡ UNET: {unet_name} (cached)")
        else:
            model_options = {}
            is_gguf = unet_name.lower().endswith('.gguf')

            if is_gguf:
                logger.info(f"✓ GGUF quantized model detected: {unet_name}")
                if weight_dtype not in ("default",):
                    logger.info(
                        f"  ↳ Ignoring weight_dtype='{weight_dtype}' for GGUF "
                        f"(uses embedded quantization)"
                    )
            else:
                dtype_map = {
                    "fp8_e4m3fn": torch.float8_e4m3fn,
                    "fp8_e5m2": torch.float8_e5m2,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "fp32": torch.float32,
                }
                if weight_dtype in dtype_map:
                    model_options["dtype"] = dtype_map[weight_dtype]

            try:
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                elapsed = time.time() - t0
                logger.info(f"✓ Loaded UNET: {unet_name} ({elapsed:.1f}s)")
                info_lines.append(f"✓ UNET: {unet_name} [{weight_dtype}] ({elapsed:.1f}s)")
                if caching:
                    _cache.put(unet_cache_key, model)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load UNET '{unet_name}': {e}")

        # ═══════════════════════════════════════════════════════════════
        # 3. LOAD CLIP (v2.1: explicit type mapping + CLIP dtype)
        # ═══════════════════════════════════════════════════════════════
        t0 = time.time()
        clip_path1 = folder_paths.get_full_path("text_encoders", clip_name1)
        if not clip_path1:
            raise FileNotFoundError(
                f"❌ CLIP encoder not found: '{clip_name1}'. Was it deleted or moved?"
            )

        use_dual_clip = clip_name2 and clip_name2 != "None"
        if use_dual_clip:
            clip_path2 = folder_paths.get_full_path("text_encoders", clip_name2)
            if not clip_path2:
                raise FileNotFoundError(
                    f"❌ Secondary CLIP not found: '{clip_name2}'. Was it deleted or moved?"
                )
            clip_paths = [clip_path1, clip_path2]
        else:
            clip_paths = [clip_path1]

        clip_cache_key = f"clip:{':'.join(clip_paths)}:{model_type}:{clip_dtype}:{device}"
        if caching and _cache.has(clip_cache_key):
            clip = _cache.get(clip_cache_key)
            clip_label = f"{clip_name1}" + (f" + {clip_name2}" if use_dual_clip else "")
            logger.info(f"⚡ CLIP from cache: {clip_label}")
            info_lines.append(f"⚡ CLIP: {clip_label} (cached)")
        else:
            # v2.1: Explicit CLIP type resolution
            clip_type_enum = get_clip_type_enum(model_type)

            clip_load_device = "cpu" if device == "cpu" else None

            # v2.1: Build CLIP model_options with dtype
            clip_model_options = {}
            if clip_load_device:
                clip_model_options["load_device"] = clip_load_device

            clip_dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp8_e4m3fn": torch.float8_e4m3fn,
                "fp32": torch.float32,
            }
            if clip_dtype in clip_dtype_map:
                clip_model_options["dtype"] = clip_dtype_map[clip_dtype]

            try:
                clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type_enum,
                    model_options=clip_model_options if clip_model_options else {},
                )
                elapsed = time.time() - t0
                clip_label = f"{clip_name1}" + (f" + {clip_name2}" if use_dual_clip else "")
                logger.info(
                    f"✓ Loaded CLIP: {clip_label} "
                    f"(type={model_type}, dtype={clip_dtype}, {elapsed:.1f}s)"
                )
                info_lines.append(f"✓ CLIP: {clip_label} [{clip_dtype}] ({elapsed:.1f}s)")
                if caching:
                    _cache.put(clip_cache_key, clip)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load CLIP: {e}")

        # ═══════════════════════════════════════════════════════════════
        # 4. LOAD VAE (with caching)
        # ═══════════════════════════════════════════════════════════════
        t0 = time.time()
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path:
            raise FileNotFoundError(
                f"❌ VAE not found: '{vae_name}'. Was it deleted or moved?"
            )

        vae_cache_key = f"vae:{vae_path}"
        if caching and _cache.has(vae_cache_key):
            vae = _cache.get(vae_cache_key)
            logger.info(f"⚡ VAE from cache: {vae_name}")
            info_lines.append(f"⚡ VAE: {vae_name} (cached)")
        else:
            try:
                sd = comfy.utils.load_torch_file(vae_path)
                vae = comfy.sd.VAE(sd=sd)
                elapsed = time.time() - t0
                logger.info(f"✓ Loaded VAE: {vae_name} ({elapsed:.1f}s)")
                info_lines.append(f"✓ VAE: {vae_name} ({elapsed:.1f}s)")
                if caching:
                    _cache.put(vae_cache_key, vae)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load VAE '{vae_name}': {e}")

        # ═══════════════════════════════════════════════════════════════
        # 5. APPLY LoRA STACK (v2.1: separate model/clip strengths)
        # ═══════════════════════════════════════════════════════════════
        loras_to_apply = [
            (lora_1, lora_1_model_str, lora_1_clip_str),
            (lora_2, lora_2_model_str, lora_2_clip_str),
            (lora_3, lora_3_model_str, lora_3_clip_str),
        ]

        for i, (lora_name, model_str, clip_str) in enumerate(loras_to_apply, 1):
            if not lora_name or lora_name == "None":
                continue
            if model_str == 0 and clip_str == 0:
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                msg = f"❌ LoRA not found: '{lora_name}'. Was it deleted or moved?"
                if lora_on_error == "raise":
                    raise FileNotFoundError(msg)
                logger.warning(f"⚠ {msg} — Skipping.")
                info_lines.append(f"⚠ LoRA {i}: {lora_name} (not found, skipped)")
                continue

            try:
                t0 = time.time()
                lora_data = comfy.utils.load_torch_file(lora_path)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_data, model_str, clip_str
                )
                elapsed = time.time() - t0
                logger.info(
                    f"✓ Applied LoRA {i}: {lora_name} "
                    f"(model={model_str}, clip={clip_str}, {elapsed:.1f}s)"
                )
                info_lines.append(
                    f"✓ LoRA {i}: {lora_name} [model={model_str}, clip={clip_str}] ({elapsed:.1f}s)"
                )
            except Exception as e:
                msg = f"Failed to apply LoRA '{lora_name}': {e}"
                if lora_on_error == "raise":
                    raise RuntimeError(f"❌ {msg}")
                logger.warning(f"⚠ {msg} — Skipping.")
                info_lines.append(f"⚠ LoRA {i}: {lora_name} (failed, skipped)")

        # ═══════════════════════════════════════════════════════════════
        # 6. LOAD CONTROLNET (v2.1: with strength/start/end metadata)
        # ═══════════════════════════════════════════════════════════════
        controlnet = None
        if controlnet_name and controlnet_name != "None":
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
            if not controlnet_path:
                logger.warning(
                    f"⚠ ControlNet not found: '{controlnet_name}'. Was it deleted or moved?"
                )
                info_lines.append(f"⚠ ControlNet: {controlnet_name} (not found)")
            else:
                try:
                    t0 = time.time()
                    controlnet = comfy.sd.load_controlnet(controlnet_path)
                    elapsed = time.time() - t0
                    logger.info(
                        f"✓ Loaded ControlNet: {controlnet_name} "
                        f"(str={controlnet_strength}, range={controlnet_start}-{controlnet_end}, "
                        f"{elapsed:.1f}s)"
                    )
                    info_lines.append(
                        f"✓ ControlNet: {controlnet_name} "
                        f"[str={controlnet_strength}, {controlnet_start}-{controlnet_end}] "
                        f"({elapsed:.1f}s)"
                    )
                    
                    # Attach metadata for downstream use (RadianceControlNetApply)
                    # We bypass standard ComfyUI immutability by setting attributes directly
                    # This works because ControlNetModel is a Python object.
                    try:
                        controlnet.radiance_strength = float(controlnet_strength)
                        controlnet.radiance_start = float(controlnet_start)
                        controlnet.radiance_end = float(controlnet_end)
                    except Exception as e:
                        logger.warning(f"Could not attach metadata to ControlNet: {e}")

                except Exception as e:
                    logger.warning(f"⚠ Failed to load ControlNet '{controlnet_name}': {e}")
                    info_lines.append(f"⚠ ControlNet: {controlnet_name} (load failed)")

        # ═══════════════════════════════════════════════════════════════
        # 7. LOAD SUMMARY
        # ═══════════════════════════════════════════════════════════════
        total_elapsed = time.time() - load_start
        summary = f"✓ Load complete in {total_elapsed:.1f}s"
        if caching:
            summary += f" (cache: {_cache.size} items)"
        logger.info(summary)
        info_lines.append(summary)

        load_info = "\n".join(info_lines)

        return (model, clip, vae, controlnet, load_info)


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceUnifiedLoader": RadianceUnifiedLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceUnifiedLoader": "◎ Radiance Unified Loader",
}
