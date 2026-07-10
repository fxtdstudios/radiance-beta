"""Model URL map and checkpoint presets — single source of truth for model resources.

Every model URL, preset configuration, and architecture heuristic lives here
so loader_utils and other modules never hardcode URLs or heuristics.
"""
from __future__ import annotations

RADIANCE_MODEL_MAP: dict = {
    "flux1-schnell-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-schnell-fp8.safetensors",
        "type": "diffusion_models",
    },
    "flux1-dev-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors",
        "type": "diffusion_models",
    },
    "sd_xl_base_1.0.safetensors": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "type": "diffusion_models",
    },
    "t5xxl_fp8_e4m3fn.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "type": "text_encoders",
    },
    "clip_l.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "type": "text_encoders",
    },
    "ae.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "type": "vae",
    },
    # LTX-2.3 transformer (46 GB) — only needed for generation, NOT for decoder training.
    "ltx-2.3-22b-dev.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors",
        "type": "diffusion_models",
    },
    "ltx-2.3-22b-distilled-1.1.safetensors": {  # ALBABIT-FIX: v1.1 — better audio + aesthetics over original distilled
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors",
        "type": "diffusion_models",
    },
    # LTX-2 Video-VAE (AutoencoderKLLTX2Video, 32x/8x/128ch, 2.44 GB) — shared by LTX-2
    # and LTX-2.3. This is all the RUDRA decoder needs to encode HDR -> latents.
    "ltx2_vae_config.json": {
        "url": "https://huggingface.co/Lightricks/LTX-2/resolve/main/vae/config.json",
        "type": "vae",
    },
    "ltx2_vae.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2/resolve/main/vae/diffusion_pytorch_model.safetensors",
        "type": "vae",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Unified per-model VAE / training configuration  (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Canonical keys match the user-facing names in RADIANCE_MODEL_PRESETS.
#  All training scripts, decoders, and HDR nodes should resolve through
#  resolve_model_vae_config() rather than hard-coding values.

MODEL_VAE_CONFIG: dict[str, dict] = {
    "ltx-video": {
        "latent_channels":     128,
        "scale_factor":        1.0,
        "log_curve":           "Sony S-Log3",
        "compression_ratio":     0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  32,
        "vae_temporal_factor": 8,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["llm_encoder", "text_projection"],
        "notes":               "LTX-Video / LTX-2.3 Video-VAE (AutoencoderKLLTX2Video): "
                               "spatial_compression_ratio=32, temporal=8, latent_channels=128, "
                               "scaling_factor=1.0. 32x spatial -> log2(32)=5 decoder upsample "
                               "stages (was wrongly 8 -> 3 stages, which broke real-LTX decode).",
    },
    "flux": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   4096,
        "clip_slots":          ["clip_l", "t5xxl"],
        "notes":               "Flux.1 image model — validated against Bradford sRGB derivation.",
    },
    # ALBABIT-FIX: Chroma (distilled Flux.1, single T5-XXL encoder)
    "chroma": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Chroma — distilled Flux.1 variant, single T5-XXL encoder (no clip_l), reuses the Flux.1 VAE (16ch).",
    },
    "zimage": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "shift_factor":        0.1159,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   2560,
        "clip_slots":          ["qwen3_4b"],
        "notes":               "Z-Image (Tongyi) — uses the FLUX.1 VAE (16ch). The flux decoder works directly.",
    },
    "qwen": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   3584,
        "clip_slots":          ["qwen2.5_vl_7b"],
        "notes":               "Qwen-Image — 16ch VAE (qwen_image_vae.safetensors), 8x. Standard decoder.",
    },
    # ALBABIT-FIX: Flux.2 Dev (Mistral-3 24B encoder) — shares the 128ch VAE with Flux.2 Klein.
    "flux2": {
        "latent_channels":     128,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  16,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  512,
        "text_embed_hidden":   7680,
        "clip_slots":          ["mistral_3_small"],
        "notes":               "Flux.2 Dev — Mistral-3 24B encoder, 128ch latent (post pixel-shuffle), shares the same VAE/DiT conditioning dim as Flux.2 Klein.",
    },
    "flux2-klein": {
        "latent_channels":     128,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  16,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  512,
        "text_embed_hidden":   7680,
        "clip_slots":          ["qwen3_4b"],
        "notes":               "Flux.2 Klein — 128ch latent (post pixel-shuffle), 16x -> 4 upsample stages.",
    },
    # ALBABIT-FIX: Cosmos (16ch, T5XXL-old encoder) and Mochi (12ch, T5XXL encoder)
    "cosmos": {
        "latent_channels":     16,
        "scale_factor":        1.0,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 8,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  512,
        "text_embed_hidden":   1024,
        "clip_slots":          ["t5xxl"],
        "notes":               "NVIDIA Cosmos; 16ch causal video VAE, T5-XXL (old) text encoder.",
    },
    "mochi": {
        "latent_channels":     12,
        "scale_factor":        1.0,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 6,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Genmo Mochi-1; 12ch latent, T5-XXL text encoder.",
    },
    "cogvideox": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.45,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "CogVideoX 3D causal VAE; tighter latent distribution → lower ratio.",
    },
    "wan": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Wan 2.1 Flow-VAE; different latent mean target → higher compression.",
    },
    "hunyuanvideo": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["llm_encoder", "clip_l"],
        "notes":               "HunyuanVideo; similar latent conventions to Wan.",
    },
    "sd3": {
        "latent_channels":     16,
        "scale_factor":        1.5305,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  154,
        "text_embed_hidden":   4096,
        "clip_slots":          ["clip_l", "clip_g", "t5xxl"],
        "notes":               "SD 3 / SD 3.5 — same VAE as Flux, different scale factor.",
    },
    "sdxl": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   2048,
        "clip_slots":          ["clip_l", "clip_g"],
        "notes":               "SDXL classic 4ch VAE; tighter DR headroom → lower ratio.",
    },
    "sd15": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.35,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   768,
        "clip_slots":          ["clip_l"],
        "notes":               "SD 1.x / 2.x — narrowest latent dynamic range.",
    },
    "lumina2": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Lumina-Next; 16ch latent flow-matching model.",
    },
    "pixart": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "PixArt-Σ / PixArt-α; 4ch latent DiT with T5 conditioning.",
    },
    "aura_flow": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   768,
        "clip_slots":          ["clip_l"],
        "notes":               "AuraFlow; 4ch latent flow model.",
    },
}

# Aliases so that architecture detectors (model/detect.py) and CLI args
# resolve to the same canonical config regardless of naming convention.
_MODEL_VAE_ALIASES: dict[str, str] = {
    # LTX variants
    "ltx":         "ltx-video",
    "ltxv":        "ltx-video",
    "ltx-video":   "ltx-video",
    "ltxav":       "ltx-video",
    # Flux variants
    "flux1":       "flux",
    "flux.1":      "flux",
    "flux-dev":    "flux",
    "flux-schnell":"flux",
    # Z-Image variants
    "z_image":     "zimage",
    "z-image":     "zimage",
    # CogVideoX variants
    "cogvideo":    "cogvideox",
    "cogvideox5b": "cogvideox",
    # Cosmos variants
    "cosmos1":     "cosmos",
    "cosmos-1":    "cosmos",
    # Mochi variants
    "mochi1":      "mochi",
    "mochi-1":     "mochi",
    "genmo-mochi": "mochi",
    # Wan variants
    "wanvideo":    "wan",
    "wan2":        "wan",
    "wan2.1":      "wan",
    "wan-2.1":     "wan",
    # Hunyuan variants
    "hunyuan":     "hunyuanvideo",
    "hunyuan_video":"hunyuanvideo",
    "hyvideo":     "hunyuanvideo",
    # SD3 variants
    "sd3.5":       "sd3",
    "sd3":         "sd3",
    "stable-diffusion-3": "sd3",
    # SDXL variants
    "sd_xl":       "sdxl",
    "sdxl-base":   "sdxl",
    # SD1.x variants
    "sd1":         "sd15",
    "sd2":         "sd15",
    "sd1.5":       "sd15",
    "sd-1.5":      "sd15",
    "v1-5":        "sd15",
    # Lumina2
    "lumina":      "lumina2",
    "lumina-next": "lumina2",
    # PixArt
    "pixart_sigma":"pixart",
    "pixart-alpha":"pixart",
    # AuraFlow
    "auraflow":    "aura_flow",
    "aura":        "aura_flow",
}


def resolve_model_vae_config(model_hint: str) -> dict | None:
    """
    Look up the unified VAE/training config for a model name or alias.

    Args:
        model_hint: Any model name string (e.g. 'flux', 'wan2.1',
                    'hunyuan_video', 'sd1.5'). Case-insensitive.

    Returns:
        A dict with all VAE/training parameters, or None if not found.
    """
    key = model_hint.strip().lower()
    if key in MODEL_VAE_CONFIG:
        return MODEL_VAE_CONFIG[key]
    if key in _MODEL_VAE_ALIASES:
        canonical = _MODEL_VAE_ALIASES[key]
        return MODEL_VAE_CONFIG.get(canonical)
    # Partial substring match against aliases
    for alias, canonical in _MODEL_VAE_ALIASES.items():
        if alias in key:
            return MODEL_VAE_CONFIG.get(canonical)
    # Partial match against canonical names
    for canonical in MODEL_VAE_CONFIG:
        if canonical in key:
            return MODEL_VAE_CONFIG[canonical]
    return None


def get_model_vae_param(model_hint: str, param: str, default=None):
    """
    Convenience: get a single parameter from the unified config.

    Example:
        scale = get_model_vae_param("flux", "scale_factor", default=0.18215)
    """
    cfg = resolve_model_vae_config(model_hint)
    if cfg is None:
        return default
    return cfg.get(param, default)


# ALBABIT-FIX: each preset only carries model_type/weight_dtype/clip_dtype —
# the only fields _apply_preset_override() (nodes_loader.py) actually reads;
# file-matching hints and CLIP slot layout live in js/radiance_loader.js
# (PRESET_CONFIGS/PRESET_SLOTS), the single source of truth for the loader
# UI's auto-fill. "default" dtypes (no dtype_map entry) let ComfyUI's own
# VRAM-aware auto-selection apply. Keys sorted alphabetically ("Custom"
# pinned first) to match the preset dropdown order.
CHECKPOINT_PRESETS: dict = {
    "Custom": {},
    "AuraFlow": {
        "model_type": "aura_flow",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    "Chroma": {
        "model_type": "chroma",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "CogVideoX": {
        "model_type": "cogvideox",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "Cosmos World": {
        "model_type": "cosmos",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: Dev and Schnell merged -- both already shared this exact
    # model_type/weight_dtype/clip_dtype, and are architecturally identical
    # (unlike Flux.2 Klein); the Sampler's model_meta mechanism already tells
    # them apart by filename for guidance/steps, so no reason to make the
    # user pick manually here either.
    "Flux.1": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp16",
    },
    "Flux.1 (Low VRAM)": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp8_e4m3fn",
    },
    # ALBABIT-FIX: Dev and Klein merged into one preset -- model/detect.py's
    # Auto-Detect can now tell them apart on its own (single_blocks count),
    # so there's no need for the user to pick the right one manually.
    "Flux.2": {
        "model_type": "Auto-Detect",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: clip_dtype forced (unlike "Flux.2"'s "default") -- Dev's
    # Mistral-3 24B encoder is heavy enough that VRAM-constrained users
    # benefit from an explicit push, on top of the offload_mode escape hatch.
    "Flux.2 (Low VRAM)": {
        "model_type": "Auto-Detect",
        "weight_dtype": "default",
        "clip_dtype": "fp8_e4m3fn",
    },
    "HunyuanVideo": {
        "model_type": "hunyuan_video",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "default",
    },
    "LTX Video": {
        "model_type": "ltxv",  # ALBABIT-FIX: "ltx" → "ltxv" — matches sampler_utils.py
        "weight_dtype": "fp16",
        "clip_dtype": "default",
    },
    "LTX Video 13B": {
        "model_type": "ltxv",  # ALBABIT-FIX: "ltx" → "ltxv" — matches sampler_utils.py
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "default",
    },
    "LTX Video 2.3": {
        "model_type": "ltxav",
        # ALBABIT-FIX: ltx-2.3-22b-dev.safetensors is bf16-native; "fp16"
        # forced a bf16->fp16 cast with no VRAM benefit and overflow risk
        # (same issue as the Z-Image clip_dtype fix above).
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "LTX Video 2.3 (Low VRAM)": {
        "model_type": "ltxav",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp8_e4m3fn",
    },
    "Lumina2": {
        "model_type": "lumina2",
        "weight_dtype": "fp16",
        "clip_dtype": "default",
    },
    "Mochi": {
        "model_type": "mochi",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "PixArt Sigma": {
        "model_type": "pixart",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    "SD 1.5": {
        "model_type": "sd1.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    # ALBABIT-FIX: Large and Turbo merged -- same model_type/weight_dtype/
    # clip_dtype already, Turbo is Large's distilled variant (Sampler tells
    # them apart by filename). Kept "Large" in the name since "SD3.5 Medium"
    # is a genuinely different-sized sibling, not a merge candidate.
    "SD3.5 Large": {
        "model_type": "sd3.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    "SD3.5 Medium": {
        "model_type": "sd3.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    # ALBABIT-FIX: Base and Turbo merged -- same reasoning as Flux.1/SD3.5
    # Large above.
    "SDXL": {
        "model_type": "sdxl",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    "Wan 2.1": {
        "model_type": "wan",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: separate preset for Wan 2.2 checkpoints — same CLIP
    # slot layout as Wan 2.1, distinct unet_hints (js/radiance_loader.js) to
    # avoid matching the wrong version when both are installed.
    "Wan 2.2": {
        "model_type": "wan",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: TI2V-5B is a single-UNET WAN 2.2 variant (no high/low_noise pair)
    # and requires wan2.2_vae.safetensors (48ch), not wan_2.1_vae (16ch).
    "Wan 2.2 TI2V": {
        "model_type": "wan",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "Z-Image": {
        "model_type": "z_image",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
}

# ALBABIT-FIX: presets for video-generation architectures — used to filter
# the Loader's "preset" dropdown (RadianceVideoLoader shows these,
# RadianceUnifiedLoader shows the rest).
VIDEO_PRESET_NAMES: set = {
    "CogVideoX",
    "Cosmos World",
    "HunyuanVideo",
    "LTX Video",
    "LTX Video 13B",
    "LTX Video 2.3",
    "LTX Video 2.3 (Low VRAM)",
    "Mochi",
    "Wan 2.1",
    "Wan 2.2",
    "Wan 2.2 TI2V",
}

# ALBABIT-FIX: model_type analog of VIDEO_PRESET_NAMES — filters the
# "model_type" dropdown (Custom/Auto-Detect) the same way "preset" is
# filtered. "ltx" renamed to "ltxv" to match sampler_utils.py.
VIDEO_MODEL_TYPES: set = {
    "hunyuan_video", "wan", "ltxv", "ltxav",
    "cosmos", "cogvideox", "mochi",
}
