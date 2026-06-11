"""Model architecture detection from safetensors key heuristics."""
from __future__ import annotations

import logging

import comfy.sd
import folder_paths

logger = logging.getLogger("radiance.model.detect")

_SAFETENSORS_PEEK = 200

_ARCH_HEURISTICS = [
    (lambda ks: any("double_blocks" in k for k in ks), "flux"),
    (lambda ks: any("joint_blocks" in k for k in ks), "sd3"),
    (lambda ks: any("img_in" in k and "proj" in k for k in ks), "hunyuan_video"),
    (lambda ks: any("cap_v_projection.weight" in k for k in ks), "lumina2"),
    (lambda ks: any("chatglm" in k.lower() for k in ks), "kolors"),
    (lambda ks: any("auraflow" in k.lower() for k in ks), "aura_flow"),
    (lambda ks: any("patch_embedding" in k for k in ks)
     and any("time_embedding" in k for k in ks)
     and not any("joint_blocks" in k for k in ks), "wan"),
    (lambda ks: any("patchify_proj" in k for k in ks), "ltx"),
    (lambda ks: any("patch_embedding" in k for k in ks)
     and any("adaln_single" in k for k in ks)
     and not any("time_embedding" in k for k in ks), "ltx"),
    (lambda ks: any("adaln_single" in k for k in ks)
     and not any("patchify_proj" in k for k in ks), "pixart"),
    (lambda ks: any("down_blocks.0" in k for k in ks)
     and any("add_embedding" in k for k in ks), "sdxl"),
    (lambda ks: any("input_blocks.0" in k for k in ks), "sd1.5"),
    (lambda ks: any("down_blocks.0" in k for k in ks), "sdxl"),
]

LATENT_CHANNELS = {
    # ALBABIT-FIX: LTX-Video VAE (incl. LTX 2.3) uses 128 latent channels, not 16
    "flux": 16, "sd3": 16, "sd3.5": 16, "ltx": 128, "ltxav": 128,
    "hunyuan_video": 16, "wan": 16, "lumina2": 16, "z_image": 16,
    "sdxl": 4, "sd1.5": 4, "pixart": 4, "aura_flow": 4, "kolors": 4,
}

_FORMAT_MAP = {
    "flux": "flux_16ch", "sd3": "sd3_16ch", "sd3.5": "sd3_16ch",
    "ltx": "ltx_128ch", "ltxav": "ltx_128ch",  # ALBABIT-FIX: match 128ch VAE
    "hunyuan_video": "hunyuan_16ch", "wan": "wan_16ch",
    "lumina2": "lumina_16ch", "z_image": "z_image_16ch",
    "sdxl": "sd_4ch", "sd1.5": "sd_4ch", "pixart": "sd_4ch",
    "aura_flow": "sd_4ch", "kolors": "sd_4ch",
}

CLIP_SLOT_ORDER = {
    "flux": ["clip_l", "t5xxl"],
    "sd3": ["clip_l", "clip_g", "t5xxl"],
    "sd3.5": ["clip_l", "clip_g", "t5xxl"],
    "sdxl": ["clip_l", "clip_g"],
    "sd1.5": ["clip_l"],
    "hunyuan_video": ["llm_encoder", "clip_l"],
    "wan": ["t5xxl"],
    "ltx": ["llm_encoder", "text_projection"],
    "ltxav": ["llm_encoder", "text_projection"],
    "lumina2": ["t5xxl"],
    "z_image": ["t5xxl"],
    "pixart": ["t5xxl"],
    "aura_flow": ["clip_l"],
    "kolors": ["llm_encoder"],
}

_CLIP_TYPE_VARIANTS = {
    "ltx": ["LTX_VIDEO", "LTXV", "LTX"],
    "ltxav": ["LTX_VIDEO", "LTXV", "LTX"],
    "hunyuan_video": ["HUNYUAN_VIDEO", "HUNYUANVIDEO"],
    "wan": ["WAN", "WAN2", "WAN_VIDEO"],
    "aura_flow": ["AURA_FLOW", "AURAFLOW"],
}

_BASE_CLIP_VRAM = {
    "flux": 4.5, "sd3": 3.0, "sd3.5": 3.5, "sdxl": 1.5, "sd1.5": 0.8,
    "hunyuan_video": 4.5, "wan": 3.0, "ltx": 2.5, "ltxav": 8.0,
    "pixart": 2.0, "aura_flow": 2.0, "kolors": 3.0, "lumina2": 3.0, "z_image": 3.0,
}

_DTYPE_MULT = {
    "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
    "fp8_e4m3fn": 0.6, "fp8_e5m2": 0.6,
}

_CLIP_DTYPE_MULT = {
    "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
    "fp8_e4m3fn": 0.55, "fp8_e5m2": 0.55,
}

_BASE_VRAM = {
    "flux": 12.0, "sd3": 10.0, "sd3.5": 12.0,
    "sdxl": 6.5, "sd1.5": 3.5,
    "hunyuan_video": 20.0, "wan": 14.0, "ltx": 11.0, "ltxav": 15.0,
    "pixart": 6.0, "aura_flow": 8.0, "kolors": 8.0,
    "lumina2": 12.0, "z_image": 14.0,
}


def detect_model_type(unet_path: str) -> str | None:
    try:
        from safetensors import safe_open
        with safe_open(unet_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())[:_SAFETENSORS_PEEK]
        for test_fn, arch in _ARCH_HEURISTICS:
            if test_fn(keys):
                logger.info(
                    "Auto-detected architecture: %s from %s",
                    arch, unet_path.rsplit("/", 1)[-1],
                )
                return arch
    except ImportError:
        logger.debug("safetensors not available — skipping auto-detect")
    except Exception as e:
        logger.debug("Auto-detect failed: %s", e)
    return None


def latent_format(arch: str) -> str:
    return _FORMAT_MAP.get(arch, f"{arch}_{LATENT_CHANNELS.get(arch, 4)}ch")


def assemble_clip_paths(arch: str, **slots) -> list[str]:
    slot_map = slots
    order = CLIP_SLOT_ORDER.get(arch, list(slot_map.keys()))
    paths = []
    for slot in order:
        val = slot_map.get(slot)
        if val and val not in ("None", ""):
            p = folder_paths.get_full_path("text_encoders", val)
            if p:
                paths.append(p)
            else:
                logger.warning("CLIP slot '%s' file not found: %s", slot, val)
    return paths


def get_clip_type_enum(model_type: str):
    mapping = {
        "flux": comfy.sd.CLIPType.FLUX,
        "sd3": comfy.sd.CLIPType.SD3,
        "sd3.5": comfy.sd.CLIPType.SD3,
        "sdxl": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sd1.5": comfy.sd.CLIPType.STABLE_DIFFUSION,
    }

    for name in ("hunyuan_video", "wan", "ltx", "ltxav", "pixart", "aura_flow", "kolors", "lumina2", "z_image"):
        enum_name = name.upper().replace(".", "_")
        auto_variants = [enum_name, name.upper(), name.title().replace("_", "")]
        extra = _CLIP_TYPE_VARIANTS.get(name, [])
        all_variants = extra + [v for v in auto_variants if v not in extra]

        for variant in all_variants:
            if hasattr(comfy.sd.CLIPType, variant):
                mapping[name] = getattr(comfy.sd.CLIPType, variant)
                break
        else:
            mapping.setdefault(name, comfy.sd.CLIPType.STABLE_DIFFUSION)

    clip_type = mapping.get(model_type)
    if clip_type is None:
        enum_name = model_type.upper().replace(".", "_")
        if hasattr(comfy.sd.CLIPType, enum_name):
            clip_type = getattr(comfy.sd.CLIPType, enum_name)
        else:
            logger.warning("No CLIPType mapping for '%s', falling back to STABLE_DIFFUSION", model_type)
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

    return clip_type


def estimate_vram_usage(
    model_type: str,
    weight_dtype: str,
    clip_dtype: str = "fp16",
    has_loras: bool = False,
    has_controlnet: bool = False,
) -> float:
    base = _BASE_VRAM.get(model_type, 8.0)
    unet_mult = _DTYPE_MULT.get(weight_dtype, 1.0)
    clip_base = _BASE_CLIP_VRAM.get(model_type, 2.0)
    clip_mult = _CLIP_DTYPE_MULT.get(clip_dtype, 1.0)

    vram = (base * unet_mult) + (clip_base * clip_mult)
    if has_loras:
        vram += 0.5
    if has_controlnet:
        vram += 2.0
    return round(vram, 1)
