"""Model architecture detection from safetensors key heuristics."""
from __future__ import annotations

import logging

import comfy.sd
import folder_paths

logger = logging.getLogger("radiance.model.detect")


def _tensor_dim0(f, ks: list[str], substr: str) -> int | None:
    """Return shape[0] of the first key containing `substr`, or None.

    ALBABIT-FIX: used to distinguish architectures that share identical key
    names but differ by tensor width (e.g. Lumina2 vs Z-Image).
    """
    for k in ks:
        if substr in k:
            try:
                return f.get_slice(k).get_shape()[0]
            except Exception:
                return None
    return None


def _block_count(ks: list[str], prefix: str) -> int:
    """Count distinct numbered blocks for a key prefix (e.g. "single_blocks.").

    ALBABIT-FIX: used to distinguish architectures that share identical key
    names but differ by depth (e.g. Flux.2 Dev vs Flux.2 Klein).
    """
    indices = set()
    for k in ks:
        if k.startswith(prefix):
            idx = k[len(prefix):].split(".", 1)[0]
            if idx.isdigit():
                indices.add(idx)
    return len(indices)


_ARCH_HEURISTICS = [
    # ALBABIT-FIX: Chroma (and Chroma Radiance) share Flux's double_blocks/
    # img_in keys but add a distilled_guidance_layer — must be checked before
    # "flux" (different CLIP slots: chroma=["t5xxl"] vs flux=["clip_l","t5xxl"]).
    # Chroma Radiance (pixel-space, nerf_blocks.*) is excluded — unsupported,
    # falls through like before rather than being mislabeled "chroma" (16ch VAE).
    (lambda ks, f: any("double_blocks" in k for k in ks)
     and any("distilled_guidance_layer" in k for k in ks)
     and not any("nerf_blocks" in k for k in ks), "chroma"),
    # ALBABIT-FIX: Flux.2 Klein shares Flux.2 Dev's double_stream_modulation_img
    # key but has far fewer single_blocks (measured: Dev=48, Klein 9B=24,
    # Klein Base 4B=20) — must be checked before "flux2" (different CLIP:
    # mistral_3_small vs qwen3). Both share Flux's double_blocks/img_in keys
    # too, so this whole group must be checked before "flux".
    (lambda ks, f: any("double_stream_modulation_img" in k for k in ks)
     and _block_count(ks, "single_blocks.") < 40, "flux2-klein"),
    (lambda ks, f: any("double_stream_modulation_img" in k for k in ks), "flux2"),
    (lambda ks, f: any("double_blocks" in k for k in ks), "flux"),
    (lambda ks, f: any("joint_blocks" in k for k in ks), "sd3"),
    (lambda ks, f: any("img_in" in k and "proj" in k for k in ks), "hunyuan_video"),
    # ALBABIT-FIX: Lumina2 and Z-Image share the same NextDiT architecture and
    # key names (cap_embedder.1.weight + noise_refiner.0.attention.k_norm.weight,
    # per comfy/model_detection.py); they differ only by the output dim of
    # cap_embedder.1.weight (2304 = Lumina2, 3840 = Z-Image). The previous
    # "cap_v_projection.weight" heuristic never matched any real checkpoint.
    (lambda ks, f: any("cap_embedder.1.weight" in k for k in ks)
     and any("noise_refiner.0.attention.k_norm.weight" in k for k in ks)
     and _tensor_dim0(f, ks, "cap_embedder.1.weight") == 3840, "z_image"),
    (lambda ks, f: any("cap_embedder.1.weight" in k for k in ks)
     and any("noise_refiner.0.attention.k_norm.weight" in k for k in ks), "lumina2"),
    (lambda ks, f: any("auraflow" in k.lower() for k in ks), "aura_flow"),
    # ALBABIT-FIX: Mochi (Genmo preview) UNET.
    (lambda ks, f: any("t5_yproj.weight" in k for k in ks), "mochi"),
    # ALBABIT-FIX: Cosmos World (T2V/I2V, stride 8) UNET.
    (lambda ks, f: any("blocks.block0.blocks.0.block.attn.to_q.0.weight" in k for k in ks), "cosmos"),
    # ALBABIT-FIX: CogVideoX UNET.
    (lambda ks, f: any("blocks.0.norm1.linear.weight" in k for k in ks), "cogvideox"),
    (lambda ks, f: any("patch_embedding" in k for k in ks)
     and any("time_embedding" in k for k in ks)
     and not any("joint_blocks" in k for k in ks), "wan"),
    # ALBABIT-FIX: return "ltxv" (not "ltx") — matches sampler_utils.py vocabulary
    (lambda ks, f: any("patchify_proj" in k for k in ks), "ltxv"),
    (lambda ks, f: any("patch_embedding" in k for k in ks)
     and any("adaln_single" in k for k in ks)
     and not any("time_embedding" in k for k in ks), "ltxv"),
    (lambda ks, f: any("adaln_single" in k for k in ks)
     and not any("patchify_proj" in k for k in ks), "pixart"),
    (lambda ks, f: any("down_blocks.0" in k for k in ks)
     and any("add_embedding" in k for k in ks), "sdxl"),
    (lambda ks, f: any("input_blocks.0" in k for k in ks), "sd1.5"),
    (lambda ks, f: any("down_blocks.0" in k for k in ks), "sdxl"),
]

LATENT_CHANNELS = {
    # ALBABIT-FIX: LTX-Video VAE (incl. LTX 2.3) uses 128 latent channels, not 16
    "flux": 16, "sd3": 16, "sd3.5": 16, "ltxv": 128, "ltxav": 128,  # ALBABIT-FIX: "ltx" → "ltxv"
    "hunyuan_video": 16, "wan": 16, "lumina2": 16, "z_image": 16,
    "sdxl": 4, "sd1.5": 4, "pixart": 4, "aura_flow": 4,
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi latent channels
    "cosmos": 16, "cogvideox": 16, "mochi": 12,
    # ALBABIT-FIX: Chroma (distilled Flux, 16ch) and Flux.2 / Flux.2 Klein (128ch)
    "chroma": 16, "flux2": 128, "flux2-klein": 128,
}

_FORMAT_MAP = {
    "flux": "flux_16ch", "sd3": "sd3_16ch", "sd3.5": "sd3_16ch",
    "ltxv": "ltx_128ch", "ltxav": "ltx_128ch",  # ALBABIT-FIX: "ltx" → "ltxv"; match 128ch VAE
    "hunyuan_video": "hunyuan_16ch", "wan": "wan_16ch",
    "lumina2": "lumina_16ch", "z_image": "z_image_16ch",
    "sdxl": "sd_4ch", "sd1.5": "sd_4ch", "pixart": "sd_4ch",
    "aura_flow": "sd_4ch",
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi latent formats
    "cosmos": "cosmos_16ch", "cogvideox": "cogvideox_16ch", "mochi": "mochi_12ch",
    # ALBABIT-FIX: Chroma and Flux.2 / Flux.2 Klein (share the 128ch VAE)
    "chroma": "chroma_16ch", "flux2": "flux2_128ch", "flux2-klein": "flux2_128ch",
}

CLIP_SLOT_ORDER = {
    "flux": ["clip_l", "t5xxl"],
    "sd3": ["clip_l", "clip_g", "t5xxl"],
    "sd3.5": ["clip_l", "clip_g", "t5xxl"],
    "sdxl": ["clip_l", "clip_g"],
    "sd1.5": ["clip_l"],
    "hunyuan_video": ["llm_encoder", "clip_l"],
    "wan": ["t5xxl"],
    "ltxv": ["llm_encoder", "text_projection"],  # ALBABIT-FIX: "ltx" → "ltxv"
    "ltxav": ["llm_encoder", "text_projection"],
    # ALBABIT-FIX: Lumina2 (Gemma-2 2B) and Z-Image (Qwen3-4B) are routed to
    # llm_encoder by their presets, not t5xxl — fixes "No CLIP encoders
    # provided" error when only llm_encoder is filled.
    "lumina2": ["llm_encoder"],
    "z_image": ["llm_encoder"],
    "pixart": ["t5xxl"],
    # ALBABIT-FIX: AuraFlow's real encoder is a T5 variant (comfy.text_encoders.
    # aura_t5.AuraT5Model, TEModel.T5_XL) -- "clip_l" was wrong, no file matching
    # that slot's naming convention exists for AuraFlow anywhere.
    "aura_flow": ["t5xxl"],
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi all use a single T5XXL text encoder
    "cosmos": ["t5xxl"], "cogvideox": ["t5xxl"], "mochi": ["t5xxl"],
    # ALBABIT-FIX: Chroma — distilled Flux, single T5XXL (no clip_l). Flux.2 /
    # Flux.2 Klein — single LLM encoder (Mistral-3 24B / Qwen3-4B).
    "chroma": ["t5xxl"], "flux2": ["llm_encoder"], "flux2-klein": ["llm_encoder"],
}

_CLIP_TYPE_VARIANTS = {
    "ltxv": ["LTX_VIDEO", "LTXV", "LTX"],   # ALBABIT-FIX: "ltx" → "ltxv"
    "ltxav": ["LTX_VIDEO", "LTXV", "LTX"],
    "hunyuan_video": ["HUNYUAN_VIDEO", "HUNYUANVIDEO"],
    "wan": ["WAN", "WAN2", "WAN_VIDEO"],
    "aura_flow": ["AURA_FLOW", "AURAFLOW"],
    # ALBABIT-FIX: Flux.2 Klein shares CLIPType.FLUX2 with Flux.2 Dev (the
    # auto-generated "FLUX2-KLEIN" enum name doesn't exist).
    "flux2-klein": ["FLUX2"],
}

_BASE_CLIP_VRAM = {
    "flux": 4.5, "sd3": 3.0, "sd3.5": 3.5, "sdxl": 1.5, "sd1.5": 0.8,
    "hunyuan_video": 4.5, "wan": 3.0, "ltxv": 2.5, "ltxav": 8.0,  # ALBABIT-FIX: "ltx" → "ltxv"
    "pixart": 2.0, "aura_flow": 2.0, "lumina2": 3.0, "z_image": 3.0,
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi — single T5XXL encoder, similar to Wan
    "cosmos": 3.0, "cogvideox": 3.0, "mochi": 3.0,
    # ALBABIT-FIX: Chroma — single T5XXL (no clip_l). Flux.2 Dev — Mistral-3 24B
    # encoder (much heavier). Flux.2 Klein — Qwen3-4B, same as Z-Image.
    "chroma": 3.5, "flux2": 8.0, "flux2-klein": 3.0,
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
    "hunyuan_video": 20.0, "wan": 14.0, "ltxv": 11.0, "ltxav": 15.0,  # ALBABIT-FIX: "ltx" → "ltxv"
    "pixart": 6.0, "aura_flow": 8.0,
    "lumina2": 12.0, "z_image": 14.0,
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi base VRAM estimates
    "cosmos": 14.0, "cogvideox": 12.0, "mochi": 16.0,
    # ALBABIT-FIX: Chroma (~8.9B distilled Flux) and Flux.2 / Flux.2 Klein (128ch)
    "chroma": 10.0, "flux2": 20.0, "flux2-klein": 14.0,
}


def detect_model_type(unet_path: str) -> str | None:
    try:
        from safetensors import safe_open
        with safe_open(unet_path, framework="pt", device="cpu") as f:
            # ALBABIT-FIX: no key-count limit — listing keys only reads the
            # safetensors header (cheap), and some architectures (e.g. Mochi's
            # t5_yproj.weight) sort late alphabetically among hundreds of keys.
            keys = list(f.keys())
            for test_fn, arch in _ARCH_HEURISTICS:
                if test_fn(keys, f):
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


def assemble_clip_paths(arch: str, unet_path: str | None = None, **slots) -> list[str]:
    slot_map = slots
    order = CLIP_SLOT_ORDER.get(arch, list(slot_map.keys()))
    paths = []
    for slot in order:
        val = slot_map.get(slot)
        if not val or val in ("None", ""):
            continue
        # ALBABIT-FIX: "Baked (from UNET)" for text_projection — LTX 2.3's
        # text_embedding_projection weights ship inside the main UNET
        # checkpoint, mirroring the native "LTXV Audio Text Encoder Loader"
        # node (which loads the UNET checkpoint a second time as a CLIP source).
        if val == "Baked (from UNET)":
            if unet_path:
                paths.append(unet_path)
            else:
                logger.warning("CLIP slot '%s' set to 'Baked (from UNET)' but no UNET path available.", slot)
            continue
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

    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi resolve via CLIPType.{COSMOS,COGVIDEOX,MOCHI}.
    # Chroma -> CLIPType.CHROMA, Flux.2 -> CLIPType.FLUX2 (Flux.2 Klein via
    # _CLIP_TYPE_VARIANTS override above, since "FLUX2-KLEIN" isn't a real enum).
    for name in ("hunyuan_video", "wan", "ltxv", "ltxav", "pixart", "aura_flow", "lumina2", "z_image",  # ALBABIT-FIX: "ltx" → "ltxv"
                  "cosmos", "cogvideox", "mochi", "chroma", "flux2", "flux2-klein"):
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
