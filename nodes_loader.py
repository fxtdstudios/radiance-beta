


# ═══════════════════════════════════════════════════════════════════════════════
#  Radiance Read Models  —  nodes_loader.py
#  v3.1.1: Deduplicated — all shared utilities live in loader_utils.py (SSOT).
#  This file contains only ComfyUI node class definitions and registration.
# ═══════════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import time

import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

# ── Single Source of Truth: all model-detection / latent / cache utilities ──
from .loader_utils import (
    RADIANCE_MODEL_MAP,
    CHECKPOINT_PRESETS,
    CLIP_SLOT_ORDER,
    LATENT_CHANNELS,
    detect_model_type as _detect_model_type,
    latent_format as _latent_format,
    file_fingerprint as _file_fingerprint,
    assemble_clip_paths as _assemble_clip_paths,
    ensure_model_exists as _ensure_model_exists,
    estimate_vram_usage,
    get_available_vram,
    get_total_vram,
    get_clip_type_enum,
    _unet_cache,
    _clip_cache,
    _vae_cache,
)

logger = logging.getLogger("radiance.loader")
from radiance.core.logging import print_premium_loader_hud

# ── Node-local UI option lists ──
MODEL_TYPES = [
    "Auto-Detect",
    "flux", "sd3", "sd3.5",
    "sdxl", "sd1.5",
    "hunyuan_video", "wan", "ltx", "ltxav",
    "lumina2", "z_image",
    "pixart", "aura_flow", "kolors",
]
WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"]
CLIP_DTYPES   = ["default", "fp16", "bf16", "fp8_e4m3fn", "fp32"]
OFFLOAD_MODES = ["none", "cpu_offload", "sequential"]




# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE LORA STACK NODE  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceLoraStack:
    """
    Compose up to 5 LoRAs into a LORA_STACK for use with RadianceUnifiedLoader.
    Can accept an upstream LORA_STACK to chain stacks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        lora_slot = lambda tooltip: (
            lora_list,
            {"default": "None", "tooltip": tooltip},
        )
        str_slot = lambda tooltip: (
            "FLOAT",
            {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
             "tooltip": tooltip},
        )
        return {
            "required": {},
            "optional": {
                "lora_stack":     ("LORA_STACK", {"default": None,
                    "tooltip": "Chain an upstream LORA_STACK before these LoRAs."}),
                "lora_1":         lora_slot("LoRA 1"),
                "lora_1_model":   str_slot("LoRA 1 model strength"),
                "lora_1_clip":    str_slot("LoRA 1 CLIP strength"),
                "lora_2":         lora_slot("LoRA 2"),
                "lora_2_model":   str_slot("LoRA 2 model strength"),
                "lora_2_clip":    str_slot("LoRA 2 CLIP strength"),
                "lora_3":         lora_slot("LoRA 3"),
                "lora_3_model":   str_slot("LoRA 3 model strength"),
                "lora_3_clip":    str_slot("LoRA 3 CLIP strength"),
                "lora_4":         lora_slot("LoRA 4"),
                "lora_4_model":   str_slot("LoRA 4 model strength"),
                "lora_4_clip":    str_slot("LoRA 4 CLIP strength"),
                "lora_5":         lora_slot("LoRA 5"),
                "lora_5_model":   str_slot("LoRA 5 model strength"),
                "lora_5_clip":    str_slot("LoRA 5 CLIP strength"),
            },
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = (
        "Compose up to 5 LoRAs into an accumulating LORA_STACK. "
        "Chain multiple stacks together. Feed into Radiance Read Models."
    )

    def build_stack(
        self,
        lora_stack=None,
        lora_1="None", lora_1_model=1.0, lora_1_clip=1.0,
        lora_2="None", lora_2_model=1.0, lora_2_clip=1.0,
        lora_3="None", lora_3_model=1.0, lora_3_clip=1.0,
        lora_4="None", lora_4_model=1.0, lora_4_clip=1.0,
        lora_5="None", lora_5_model=1.0, lora_5_clip=1.0,
    ) -> tuple:
        stack = list(lora_stack) if lora_stack else []
        for name, ms, cs in [
            (lora_1, lora_1_model, lora_1_clip),
            (lora_2, lora_2_model, lora_2_clip),
            (lora_3, lora_3_model, lora_3_clip),
            (lora_4, lora_4_model, lora_4_clip),
            (lora_5, lora_5_model, lora_5_clip),
        ]:
            if name and name != "None":
                stack.append((name, float(ms), float(cs)))
        return (stack,)


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE READ MODELS v2.1.0
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_TYPES = [
    "Auto-Detect",
    "flux", "sd3", "sd3.5",
    "sdxl", "sd1.5",
    "hunyuan_video", "wan", "ltx",
    "lumina2", "z_image",
    "pixart", "aura_flow", "kolors",
]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"]
CLIP_DTYPES   = ["default", "fp16", "bf16", "fp8_e4m3fn", "fp32"]
OFFLOAD_MODES = ["none", "cpu_offload", "sequential"]



class RadianceUnifiedLoader:
    """
    Universal diffusion model loader v3.3 — Streamlined and Optimized.
    Zero canvas clutter: LoRAs and ControlNets decoupled to external modular nodes.
    Outputs: MODEL, CLIP, VAE, LORA_STACK.
    """

    @classmethod
    def INPUT_TYPES(cls):
        clip_list = ["None"] + folder_paths.get_filename_list("text_encoders")
        clip_slot = lambda tip: (clip_list, {"default": "None", "tooltip": tip})

        return {
            "required": {
                # ── Preset ──
                "preset": (
                    list(CHECKPOINT_PRESETS.keys()),
                    {"default": "Custom",
                     "tooltip": "Quick-configure for common architectures. "
                                "Overrides model_type, dtypes, offload_mode, and hints "
                                "which CLIP slots are needed."},
                ),
                # ── UNET ──
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "Main diffusion model (UNET / DiT / Transformer)."},
                ),
                "weight_dtype": (
                    WEIGHT_DTYPES,
                    {"default": "default",
                     "tooltip": "UNET weight precision. fp8_e4m3fn saves ~40% VRAM vs fp16."},
                ),
                # ── Architecture ──
                "model_type": (
                    MODEL_TYPES,
                    {"default": "Auto-Detect",
                     "tooltip": "'Auto-Detect' reads the checkpoint's key names to determine "
                                "architecture. Override manually if detection fails."},
                ),
                # ── VAE ──
                "vae_name": (
                    folder_paths.get_filename_list("vae"),
                    {"tooltip": "VAE for encoding/decoding latents."},
                ),
            },
            "optional": {
                # ── Named CLIP slots ──
                "clip_l":      clip_slot(
                    "CLIP-L (text encoder). Used by: SD1.5, SDXL, Flux, SD3."),
                "clip_g":      clip_slot(
                    "CLIP-G (text encoder). Used by: SDXL, SD3, SD3.5."),
                "t5xxl":       clip_slot(
                    "T5-XXL (text encoder). Used by: Flux, SD3, SD3.5, Wan, LTX, PixArt."),
                "llm_encoder": clip_slot(
                    "LLM encoder (ChatGLM3 etc.). Used by: Kolors, HunyuanVideo."),
                "text_projection": clip_slot(
                    "Text projection matrix. Used by: LTX Video."),
                # ── CLIP precision ──
                "clip_dtype": (
                    CLIP_DTYPES,
                    {"default": "default",
                     "tooltip": "CLIP weight precision. Independent from UNET. "
                                "For Flux T5XXL: fp8 saves ~4.7 GB vs fp16."},
                ),
                # ── Offload ──
                "offload_mode": (
                    OFFLOAD_MODES,
                    {"default": "none",
                     "tooltip": "none = GPU only. "
                                "cpu_offload = CLIP loaded to CPU RAM. "
                                "sequential = enable ComfyUI sequential CPU offload (8–12 GB GPUs)."},
                ),
                # ── LoRA (external stack input) ──
                "lora_stack":      ("LORA_STACK", {"default": None,
                    "tooltip": "Accept a LORA_STACK from RadianceLoraStack node."}),
                # ── Options ──
                "check_vram":  (["On", "Off"], {"default": "On",
                    "tooltip": "Estimate VRAM before load and warn if tight."}),
                "use_cache":   (["On", "Off"], {"default": "On",
                    "tooltip": "Cache loaded models. Skips disk I/O when re-running "
                               "with the same files. Cache auto-invalidates if files change."}),
                "lora_on_error": (["warn", "raise"], {"default": "raise",
                    "tooltip": "'warn' skips failed LoRA and continues. "
                               " 'raise' stops execution."}),
                "auto_download": ("BOOLEAN", {"default": False,
                    "tooltip": "If a selected model is missing, automatically download it from Radiance mirrors."}),
            },
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "VAE", "LORA_STACK", "STRING")
    RETURN_NAMES  = ("MODEL", "CLIP", "VAE", "lora_stack", "model_meta")
    FUNCTION      = "load_radiance_stack"
    CATEGORY      = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION   = (
        "Universal loader v3.3 — streamlined to be extremely visual and modular. "
        "Auto-detects architecture, auto-tunes weights/offload, supports "
        "chainable LORA_STACK, and outputs JSON model metadata."
    )

    @staticmethod
    def _apply_preset_override(cfg, field, key, cur, overrides):
        new = cfg.get(key, cur)
        if new != cur:
            overrides.append(f"{field}: {cur}→{new}")
        return new

    def load_radiance_stack(
        self,
        preset,
        unet_name,
        weight_dtype,
        model_type,
        vae_name,
        clip_l="None",
        clip_g="None",
        t5xxl="None",
        llm_encoder="None",
        text_projection="None",
        clip_dtype="default",
        offload_mode="none",
        lora_stack=None,
        check_vram="On",
        use_cache="On",
        lora_on_error="raise",
        auto_download=False,
    ):
        def _none(val):
            return None if val in ("None", "", None) else val

        clip_l = _none(clip_l)
        clip_g = _none(clip_g)
        t5xxl = _none(t5xxl)
        llm_encoder = _none(llm_encoder)
        text_projection = _none(text_projection)

        load_start  = time.time()
        info_lines  = []
        caching     = use_cache == "On"
        overrides   = []

        # ════════════════════════════════════════════════════════════════
        # 0. APPLY PRESET
        # ════════════════════════════════════════════════════════════════
        if preset not in ("Custom",) and preset in CHECKPOINT_PRESETS:
            cfg = CHECKPOINT_PRESETS[preset]

            model_type   = self._apply_preset_override(cfg, "model_type",   "model_type",   model_type,   overrides)
            weight_dtype = self._apply_preset_override(cfg, "weight_dtype",  "weight_dtype", weight_dtype, overrides)
            clip_dtype   = self._apply_preset_override(cfg, "clip_dtype",    "clip_dtype",   clip_dtype,   overrides)
            offload_mode = self._apply_preset_override(cfg, "offload_mode",  "offload_mode", offload_mode, overrides)

            msg = f"Preset '{preset}'" + (f" (overrode: {', '.join(overrides)})" if overrides else " (no overrides)")
            logger.info(msg)
            info_lines.append(msg)

        # ════════════════════════════════════════════════════════════════
        # 1. RESOLVE ARCHITECTURE
        # ════════════════════════════════════════════════════════════════
        unet_path = _ensure_model_exists(unet_name, "diffusion_models", auto_download)
        if not unet_path:
            raise FileNotFoundError(
                f"❌ UNET not found: '{unet_name}'. Enable auto_download or install it manually."
            )

        from radiance.core.logging import supports_unicode
        divider = "│" if supports_unicode() else "|"

        detected_type = None
        if model_type == "Auto-Detect":
            detected_type = _detect_model_type(unet_path)
            if detected_type:
                resolved_type = detected_type
                info_lines.append(f"Auto-detected: {resolved_type}")
            else:
                resolved_type = "sdxl"   # safe fallback
                logger.warning(
                    "Architecture auto-detect failed. Falling back to 'sdxl'. "
                    "Set model_type manually if this is wrong."
                )
                info_lines.append("Auto-detect failed — fallback: sdxl")
        else:
            resolved_type = model_type

        latent_fmt  = _latent_format(resolved_type)
        lat_msg     = f"Latent format: {latent_fmt} ({resolved_type})"
        logger.info(lat_msg)
        info_lines.append(lat_msg)

        # ════════════════════════════════════════════════════════════════
        # 2. OFFLOAD MODE
        # ════════════════════════════════════════════════════════════════
        if offload_mode == "sequential":
            try:
                comfy.model_management.set_lowvram_mode(True)
                logger.info("Sequential CPU offload enabled")
                info_lines.append("Offload: sequential")
            except Exception as e:
                logger.warning(f"Could not enable sequential offload: {e}")

        # FIX 6: ComfyUI model_options["load_device"] expects torch.device, not str.
        clip_load_device = torch.device("cpu") if offload_mode == "cpu_offload" else None

        # ════════════════════════════════════════════════════════════════
        # 3. VRAM ESTIMATION
        # ════════════════════════════════════════════════════════════════
        has_loras = bool(lora_stack)

        if check_vram == "On":
            est   = estimate_vram_usage(resolved_type, weight_dtype, clip_dtype,
                                        has_loras, False)
            avail = get_available_vram()
            total = get_total_vram()
            vram_msg = f"VRAM: ~{est} GB needed {divider} {avail:.2f} GB free / {total:.2f} GB total"
            logger.info(vram_msg)
            info_lines.append(vram_msg)
            if avail > 0 and est > avail * 0.9:
                warn = f"VRAM tight! {est} GB estimated, {avail:.2f} GB free. Consider fp8 dtype or cpu_offload."
                logger.warning(warn)
                info_lines.append(warn)
        else:
            est = estimate_vram_usage(resolved_type, weight_dtype, clip_dtype,
                                      has_loras, False)

        # ════════════════════════════════════════════════════════════════
        # 4. LOAD UNET  (mtime + size cache key)
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        unet_fp   = _file_fingerprint(unet_path)
        unet_key  = f"unet:{unet_path}:{weight_dtype}:{unet_fp}"

        unet_time = 0.0
        unet_cache_hit = caching and _unet_cache.has(unet_key)
        if unet_cache_hit:
            model = _unet_cache.get(unet_key)
            logger.info(f"UNET loaded from cache: {unet_name}")
            info_lines.append(f"UNET: {unet_name} (cached)")
        else:
            model_options = {}
            is_gguf = unet_name.lower().endswith(".gguf")
            if is_gguf:
                logger.info(f"GGUF detected: {unet_name} (embedded quant)")
            else:
                dtype_map = {
                    "fp8_e4m3fn": torch.float8_e4m3fn,
                    "fp8_e5m2":   torch.float8_e5m2,
                    "fp16":       torch.float16,
                    "bf16":       torch.bfloat16,
                    "fp32":       torch.float32,
                }
                if weight_dtype in dtype_map:
                    model_options["dtype"] = dtype_map[weight_dtype]

            try:
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                unet_time = time.time() - t0
                logger.info(f"UNET loaded {divider} {unet_name} [{weight_dtype}] {divider} {unet_time:.1f}s")
                info_lines.append(f"UNET: {unet_name} [{weight_dtype}] ({unet_time:.1f}s)")
                if caching:
                    _unet_cache.put(unet_key, model)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load UNET '{unet_name}': {e}")

        # ════════════════════════════════════════════════════════════════
        # 5. LOAD CLIP  (named slots → ordered paths → mtime cache key)
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        
        # Ensure all selected CLIPs exist/downloaded
        for slot, val in [("clip_l", clip_l), ("clip_g", clip_g), 
                          ("t5xxl", t5xxl), ("llm_encoder", llm_encoder),
                          ("text_projection", text_projection)]:
             _ensure_model_exists(val, "text_encoders", auto_download)

        clip_paths = _assemble_clip_paths(resolved_type, clip_l=clip_l, clip_g=clip_g, t5xxl=t5xxl, llm_encoder=llm_encoder, text_projection=text_projection)

        if not clip_paths:
            raise ValueError(
                f"❌ No CLIP encoders provided for architecture '{resolved_type}'. "
                f"Fill the required slot(s): "
                f"{', '.join(CLIP_SLOT_ORDER.get(resolved_type, ['clip_l']))}"
            )

        clip_fps     = ":".join(_file_fingerprint(p) for p in clip_paths)
        clip_key     = f"clip:{':'.join(clip_paths)}:{resolved_type}:{clip_dtype}:{clip_fps}"

        clip_slot_used = []
        for slot, val in [("clip_l", clip_l), ("clip_g", clip_g),
                          ("t5xxl", t5xxl), ("llm_encoder", llm_encoder),
                          ("text_projection", text_projection)]:
            if val is not None:
                clip_slot_used.append(slot)

        clip_time = 0.0
        clip_cache_hit = caching and _clip_cache.has(clip_key)
        if clip_cache_hit:
            clip = _clip_cache.get(clip_key)
            logger.info(f"CLIP loaded from cache: {' + '.join(clip_slot_used)}")
            info_lines.append(f"CLIP: {'+'.join(clip_slot_used)} (cached)")
        else:
            clip_type_enum = get_clip_type_enum(resolved_type)
            clip_model_opts = {}
            if clip_load_device:
                clip_model_opts["load_device"] = clip_load_device
            dtype_map = {
                "fp16": torch.float16, "bf16": torch.bfloat16,
                "fp8_e4m3fn": torch.float8_e4m3fn, "fp32": torch.float32,
            }
            if clip_dtype in dtype_map:
                clip_model_opts["dtype"] = dtype_map[clip_dtype]

            try:
                clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type_enum,
                    model_options=clip_model_opts if clip_model_opts else {},
                )
                clip_time = time.time() - t0
                logger.info(
                    f"CLIP loaded {divider} {' + '.join(clip_slot_used)} "
                    f"[{clip_dtype}] {divider} {clip_time:.1f}s"
                )
                info_lines.append(
                    f"CLIP: {'+'.join(clip_slot_used)} [{clip_dtype}] ({clip_time:.1f}s)"
                )
                if caching:
                    _clip_cache.put(clip_key, clip)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load CLIP: {e}")

        # ════════════════════════════════════════════════════════════════
        # 6. LOAD VAE  (mtime cache key)
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        vae_path = _ensure_model_exists(vae_name, "vae", auto_download)
        if not vae_path:
            raise FileNotFoundError(f"❌ VAE not found: '{vae_name}'. Enable auto_download or install it manually.")

        vae_fp  = _file_fingerprint(vae_path)
        vae_key = f"vae:{vae_path}:{vae_fp}"

        vae_time = 0.0
        vae_cache_hit = caching and _vae_cache.has(vae_key)
        if vae_cache_hit:
            vae = _vae_cache.get(vae_key)
            logger.info(f"VAE loaded from cache: {vae_name}")
            info_lines.append(f"VAE: {vae_name} (cached)")
        else:
            try:
                sd  = comfy.utils.load_torch_file(vae_path)
                vae = comfy.sd.VAE(sd=sd)
                vae_time = time.time() - t0
                logger.info(f"VAE loaded {divider} {vae_name} {divider} {vae_time:.1f}s")
                info_lines.append(f"VAE: {vae_name} ({vae_time:.1f}s)")
                if caching:
                    _vae_cache.put(vae_key, vae)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load VAE '{vae_name}': {e}")

        # ════════════════════════════════════════════════════════════════
        # 7. APPLY LoRA STACK
        #    Priority: upstream lora_stack input stack
        # ════════════════════════════════════════════════════════════════
        combined_loras = list(lora_stack) if lora_stack else []

        applied_loras = []
        for i, (lora_name, model_str, clip_str) in enumerate(combined_loras, 1):
            if model_str == 0 and clip_str == 0:
                continue
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                msg = f"LoRA not found: '{lora_name}'"
                if lora_on_error == "raise":
                    raise FileNotFoundError(msg)
                logger.warning(f"{msg} — Skipping.")
                info_lines.append(f"LoRA {i}: {lora_name} (not found)")
                continue
            try:
                t0 = time.time()
                lora_data = comfy.utils.load_torch_file(lora_path)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_data, model_str, clip_str
                )
                elapsed = time.time() - t0
                logger.info(
                    f"LoRA {i} applied {divider} {lora_name} [model={model_str}, clip={clip_str}] {divider} {elapsed:.1f}s"
                )
                info_lines.append(
                    f"LoRA {i}: {lora_name} [m={model_str} c={clip_str}] ({elapsed:.1f}s)"
                )
                applied_loras.append({"name": lora_name, "model_str": model_str,
                                      "clip_str": clip_str})
            except Exception as e:
                msg = f"Failed to apply LoRA '{lora_name}': {e}"
                if lora_on_error == "raise":
                    raise RuntimeError(f"❌ {msg}")
                logger.warning(f"{msg} — Skipping.")
                info_lines.append(f"LoRA {i}: {lora_name} (failed)")

        # ════════════════════════════════════════════════════════════════
        # 8. BUILD OUTPUTS
        # ════════════════════════════════════════════════════════════════
        total_ms = round((time.time() - load_start) * 1000)
        cache_info = f" (cache: U{_unet_cache.size} C{_clip_cache.size} V{_vae_cache.size})" if caching else ""
        summary = f"System load complete in {total_ms / 1000:.1f}s{cache_info}"
        logger.info(summary)
        info_lines.append(summary)

        # Draw a beautiful independent console HUD card that represents visual Apple-style dashboard loading
        print_premium_loader_hud(
            preset=preset,
            overrides=overrides if preset in CHECKPOINT_PRESETS else None,
            resolved_type=resolved_type,
            latent_fmt=latent_fmt,
            est_vram=est,
            avail_vram=avail if check_vram == "On" else 0.0,
            total_vram=total if check_vram == "On" else 0.0,
            unet_name=unet_name,
            unet_dtype=weight_dtype,
            unet_time=unet_time,
            unet_cached=unet_cache_hit,
            clip_slots=clip_slot_used,
            clip_dtype=clip_dtype,
            clip_time=clip_time,
            clip_cached=clip_cache_hit,
            vae_name=vae_name,
            vae_time=vae_time,
            vae_cached=vae_cache_hit,
            loras=applied_loras,
            total_time_ms=total_ms,
            caching=caching
        )

        load_info = "\n".join(info_lines)

        # model_meta — structured JSON for downstream QC / analytics nodes
        model_meta = {
            "arch":          resolved_type,
            "detected":      detected_type is not None,
            "unet_file":     unet_name,
            "weight_dtype":  weight_dtype,
            "clip_slots":    clip_slot_used,
            "clip_dtype":    clip_dtype,
            "offload_mode":  offload_mode,
            "latent_ch":     LATENT_CHANNELS.get(resolved_type, 4),
            "latent_format": latent_fmt,
            "vram_est_gb":   est,
            "loras":         applied_loras,
            "load_ms":       total_ms,
            "cached_unet":   unet_cache_hit,
        }

        # Output the accumulated lora list
        out_lora_stack = [(e["name"], e["model_str"], e["clip_str"])
                          for e in applied_loras] if applied_loras else None

        return (
            model,
            clip,
            vae,
            out_lora_stack,
            json.dumps(model_meta, sort_keys=True),
        )


class RadianceVideoLoader(RadianceUnifiedLoader):
    """Alias — identical to RadianceUnifiedLoader. Kept for backward compatibility."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
class RadianceControlNetApply:
    """
    Advanced ControlNet application node for the Radiance suite.
    - Gracefully bypasses if CONTROL_NET is None (prevents AttributeError crashes).
    - Supports standard start/end percent and strength controls.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global strength of the control effect."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Percentage of the generation where control starts (0.0 = beginning)."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Percentage of the generation where control ends (1.0 = end)."}),
                "control_type": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()), {"default": "auto",
                    "tooltip": "For Union ControlNets (like Flux), select the specific control mode (Canny, Depth, etc.)."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)  # FIX 7: was missing — ComfyUI showed generic labels
    FUNCTION = "apply_controlnet"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Apply a ControlNet conditioning signal to the Radiance sampler."

    def apply_controlnet(self, conditioning, control_net, image, strength, start_percent, end_percent, control_type="auto"):
        # 1. Graceful Bypass: If no control_net or zero strength, just return the input conditioning.
        if control_net is None:
            logger.info("◎ Radiance Control: No ControlNet connected. Bypassing.")
            return (conditioning, )
            
        if strength == 0:
            return (conditioning, )

        # 2. Set Union ControlNet Type (if applicable)
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(control_type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        # 3. Advanced Application (Standard ComfyUI Logic with added safety)
        c = []
        try:
            for t in conditioning:
                n = [t[0], t[1].copy()]
                c_net = control_net.copy().set_cond_hint(image, strength, (start_percent, end_percent))
                if 'control' in n[1]:
                    c_net.set_previous_controlnet(n[1]['control'])
                n[1]['control'] = c_net
                n[1]['control_apply_strength'] = strength
                c.append(n)
            return (c, )
        except Exception as e:
            logger.error(f"❌ Radiance Control: Failed to apply ControlNet: {e}")
            return (conditioning, )


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

# FIX 1: NODE_CLASS_MAPPINGS keys must be plain ASCII identifiers.
# The ◎ prefix belongs only in NODE_DISPLAY_NAME_MAPPINGS (the user-visible label).
# Having it in the type key breaks ComfyUI workflow JSON serialization and node lookup.
NODE_CLASS_MAPPINGS = {
    "RadianceUnifiedLoader": RadianceUnifiedLoader,
    "RadianceImageLoader":   RadianceUnifiedLoader, # Alias
    "RadianceVideoLoader":   RadianceVideoLoader,
    "RadianceLoraStack":     RadianceLoraStack,
    "RadianceControlNetApply": RadianceControlNetApply,
    "RadianceControlApply":    RadianceControlNetApply, # Alias
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceUnifiedLoader": "◎ Radiance Read Models",
    "RadianceImageLoader":   "◎ Radiance Image Loader",
    "RadianceVideoLoader":   "◎ Radiance Video Loader",
    "RadianceLoraStack":     "◎ Radiance LoRA Stack",
    "RadianceControlNetApply": "◎ Radiance ControlNet Apply",
    "RadianceControlApply":    "◎ Radiance Control",
}
