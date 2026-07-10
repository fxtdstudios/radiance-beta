/**
 * radiance_loader.js
 * v3.3 — Streamlined Radiance Loader Widget sync, dynamic visibility & smart auto-fill
 */

import { app } from "../../scripts/app.js";

// Node definition identifiers
const LOADER_NODES = ["RadianceUnifiedLoader", "RadianceImageLoader", "RadianceVideoLoader"];

// Dynamic visibility rules for CLIP slots per preset.
// ALBABIT-FIX: keys sorted alphabetically ("Custom" pinned first) to match
// the preset dropdown order (config/model_map.py's CHECKPOINT_PRESETS).
const PRESET_SLOTS = {
    "Custom": ["clip_l", "clip_g", "t5xxl", "llm_encoder", "text_projection"],
    "AuraFlow": ["clip_l"],
    "Chroma": ["t5xxl"],
    "CogVideoX": ["t5xxl"],
    "Cosmos World": ["t5xxl"],
    "Flux.1": ["clip_l", "t5xxl"],
    "Flux.1 (Low VRAM)": ["clip_l", "t5xxl"],
    "Flux.2": ["llm_encoder"],
    "Flux.2 (Low VRAM)": ["llm_encoder"],
    "HunyuanVideo": ["clip_l", "llm_encoder"],
    "Kolors": ["llm_encoder"],
    "LTX Video": ["llm_encoder", "text_projection"],
    "LTX Video 13B": ["llm_encoder", "text_projection"],
    "LTX Video 2.3": ["llm_encoder", "text_projection"],
    "LTX Video 2.3 (Low VRAM)": ["llm_encoder", "text_projection"],
    "Lumina2": ["llm_encoder"],
    "Mochi": ["t5xxl"],
    "PixArt Sigma": ["t5xxl"],
    "SD 1.5": ["clip_l"],
    "SD3.5 Large": ["clip_l", "clip_g", "t5xxl"],
    "SD3.5 Medium": ["clip_l", "clip_g", "t5xxl"],
    "SDXL": ["clip_l", "clip_g"],
    "Wan 2.1": ["t5xxl"],
    "Wan 2.2": ["t5xxl"],
    "Wan 2.2 TI2V": ["t5xxl"],
    "Z-Image": ["llm_encoder"],
};

// Full hints configs for automatic local file selection matching.
// ALBABIT-FIX: keys sorted alphabetically to match PRESET_SLOTS/the preset
// dropdown order (config/model_map.py's CHECKPOINT_PRESETS).
const PRESET_CONFIGS = {
    "AuraFlow": {
        "unet_hints":    ["auraflow", "aura_flow", "aura-flow"],
        "vae_hints":     ["aura_vae", "sd_vae"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
        },
    },
    "Chroma": {
        "unet_hints":    ["chroma-unlocked", "chroma_unlocked", "chroma"],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "CogVideoX": {
        "unet_hints":    ["cogvideox-5b", "cogvideox_5b", "CogVideoX", "cogvideox"],
        "vae_hints":     ["cogvideox_vae", "cogvideox-vae", "cogvideo_vae"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "Cosmos World": {
        "unet_hints":    ["cosmos-1_0-diffusion", "Cosmos-1_0", "cosmos_world", "cosmos"],
        "vae_hints":     ["cosmos_vae", "cosmos-tokenizer", "cosmos"],
        "clip_hints":    {
            // ALBABIT-FIX: Cosmos uses the "old" T5-XXL (T5 1.0) encoder,
            // distinct from the t5xxl_fp8/fp16 (T5 1.1) used by Flux/SD3/etc.
            // Prioritize oldt5_xxl_*, fall back to t5xxl_* if absent.
            "t5xxl": ["oldt5_xxl_fp8_e4m3fn", "oldt5_xxl_fp16", "oldt5_xxl", "t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    // ALBABIT-FIX: Dev and Schnell merged -- architecturally identical (no
    // single_blocks-style split like Flux.2 Klein), and the Sampler's
    // model_meta mechanism already tells them apart by filename for
    // guidance/steps. unet_hints combines both lists, Dev first.
    "Flux.1": {
        "unet_hints":    [
            "flux1-dev-fp8", "flux1-dev", "flux1-krea-dev", "krea-dev", "flux-dev",
            "flux1-schnell-fp8", "flux1-schnell", "flux-schnell",
        ],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl":  ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "Flux.1 (Low VRAM)": {
        "unet_hints":    ["flux1-dev-fp8", "flux1-dev", "flux1-krea-dev", "krea-dev", "flux-dev"],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl":  ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
        // ALBABIT-FIX: "Low VRAM" presets force offload_mode — expose the
        // widget so the user can still override it (e.g. on a higher-VRAM
        // GPU where cpu_offload is unnecessarily slow).
        "extra_widgets": ["offload_mode"],
        "offload_mode": "cpu_offload",
    },
    // ALBABIT-FIX: Dev and Klein merged into one preset now that Auto-Detect
    // can tell them apart on its own (model/detect.py, single_blocks count).
    // unet_hints cover both families -- Dev first (flagship, quality-first),
    // then Klein's sizes/distillation states (4B/9B, Base/distilled).
    "Flux.2": {
        "unet_hints": [
            "flux2-dev.safetensors", "flux2_dev_fp8mixed.safetensors",
            "flux-2-klein-9b.safetensors",
            "flux-2-klein-base-4b.safetensors",
            "flux-2-klein-base-9b-fp8.safetensors",
            "klein-9b-kv", "klein-base", "klein-9b", "klein-4b",
            "flux2-dev", "flux2_dev", "flux.2-dev",
            "flux2-klein", "flux2_klein", "flux.2-klein", "klein",
        ],
        // full_encoder_small_decoder is a lighter/faster decoder (same
        // encoder) -- only used if the full-quality flux2-vae isn't present.
        "vae_hints":     ["flux2-vae", "flux2_vae", "flux2_ae", "full_encoder_small_decoder"],
        "clip_hints":    {
            // Dev's encoder (Mistral) -- also the fallback when unet_name
            // isn't a recognized Klein size (see clip_size_hints, which
            // takes priority whenever a Klein 4B/9B file is detected).
            "llm_encoder": ["mistral_3_small_flux2_bf16", "mistral_3_small_flux2_fp8", "mistral_3_small_flux2", "mistral_3", "mistral"],
        },
        // ALBABIT-FIX: Klein's encoder (Qwen) must match its size (4B->qwen_3_4b,
        // 9B->qwen_3_8b*) -- resolved dynamically from the size token detected
        // in unet_name. Quality-first: bf16 before fp8/fp4mixed.
        "clip_size_hints": {
            "9b": ["qwen_3_8b.safetensors", "qwen_3_8b", "qwen_3_8b_fp8mixed", "qwen_3_8b_fp4mixed", "qwen3_8b"],
            "4b": ["qwen_3_4b.safetensors", "qwen_3_4b", "qwen3_4b"],
        },
    },
    "Flux.2 (Low VRAM)": {
        "unet_hints": [
            "flux2-dev.safetensors", "flux2_dev_fp8mixed.safetensors",
            "flux-2-klein-9b.safetensors",
            "flux-2-klein-base-4b.safetensors",
            "flux-2-klein-base-9b-fp8.safetensors",
            "klein-9b-kv", "klein-base", "klein-9b", "klein-4b",
            "flux2-dev", "flux2_dev", "flux.2-dev",
            "flux2-klein", "flux2_klein", "flux.2-klein", "klein",
        ],
        "vae_hints":     ["flux2-vae", "flux2_vae", "flux2_ae", "full_encoder_small_decoder"],
        "clip_hints":    {
            "llm_encoder": ["mistral_3_small_flux2_bf16", "mistral_3_small_flux2_fp8", "mistral_3_small_flux2", "mistral_3", "mistral"],
        },
        "clip_size_hints": {
            "9b": ["qwen_3_8b.safetensors", "qwen_3_8b", "qwen_3_8b_fp8mixed", "qwen_3_8b_fp4mixed", "qwen3_8b"],
            "4b": ["qwen_3_4b.safetensors", "qwen_3_4b", "qwen3_4b"],
        },
        // ALBABIT-FIX: "Low VRAM" presets force offload_mode — expose the
        // widget so the user can still override it (e.g. on a higher-VRAM
        // GPU where cpu_offload is unnecessarily slow).
        "extra_widgets": ["offload_mode"],
        "offload_mode": "cpu_offload",
    },
    "HunyuanVideo": {
        "unet_hints":    ["hunyuan_video", "hunyuanvideo", "hyvideo"],
        "vae_hints":     ["hunyuan_video_vae", "hunyuan_vae", "hyvideo_vae"],
        "clip_hints":    {
            "llm_encoder": ["llava_llama3", "llm_llava", "hunyuan_llm"],
            "clip_l":      ["clip_l.safetensors", "clip_l"],
        },
    },
    "Kolors": {
        "unet_hints":    ["kolors", "Kolors"],
        "vae_hints":     ["kolors_vae", "sdxl_vae"],
        "clip_hints":    {
            "llm_encoder": ["chatglm3", "chatglm", "kolors_clip"],
        },
    },
    "LTX Video": {
        "unet_hints":    ["ltx-video-2b", "ltxv-2b", "ltx_video", "ltxv"],
        "vae_hints":     ["Baked VAE (from UNET)", "ltxvideo_vae", "ltx_vae", "ltxv_vae", "causal_vae"],
        "clip_hints":    {
            "llm_encoder": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
        "extra_widgets": ["upscale_model_name"],
        "upscale_hints": ["ltxv", "ltx_video", "latent_upsampler", "upsampler"],
    },
    "LTX Video 13B": {
        "unet_hints":    ["ltx-video-13b", "ltxv-13b", "ltx_13b"],
        "vae_hints":     ["Baked VAE (from UNET)", "ltxvideo_vae", "ltx_vae", "ltxv_vae", "causal_vae"],
        "clip_hints":    {
            "llm_encoder": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
        "extra_widgets": ["upscale_model_name"],
        "upscale_hints": ["ltxv-13b", "ltx_13b", "latent_upsampler", "upsampler"],
    },
    "LTX Video 2.3": {
        // ALBABIT-FIX: distilled-1.1 added in second position — auto-matches if user has it
        "unet_hints":    ["ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled-1.1.safetensors", "ltx-2.3-22b-distilled.safetensors", "ltx-2.3-22b-dev", "ltx-2.3", "ltx_2.3"],
        "vae_hints":     ["LTX23_video_vae_bf16.safetensors", "LTX23_video_vae", "ltx23_video", "ltx_23_video", "Baked VAE (from UNET)"],
        "audio_vae_hints": ["LTX23_audio_vae_bf16.safetensors", "LTX23_audio_vae", "ltx23_audio", "ltx_23_audio"],
        "clip_hints":    {
            "llm_encoder":     ["gemma_3_12B_it.safetensors", "gemma_3_12B_it_fp4", "gemma_3_12B_it", "gemma_3", "gemma"],
            // ALBABIT-FIX: fall back to "Baked (from UNET)" if the standalone
            // text_projection file isn't present (mirrors the Low VRAM preset).
            "text_projection": ["ltx-2.3_text_projection_bf16.safetensors", "ltx-2.3_text_projection", "text_projection", "Baked (from UNET)"],
        },
        "extra_widgets": ["upscale_model_name", "audio_vae_name"],
        "upscale_hints": ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "ltx-2.3", "ltx_2.3", "latent_upsampler", "upsampler"],
    },
    "LTX Video 2.3 (Low VRAM)": {
        // ALBABIT-FIX: distilled-1.1 replaces dev-fp8 as primary Low VRAM model
        "unet_hints":    ["ltx-2.3-22b-distilled-1.1.safetensors", "ltx-2.3-22b-distilled.safetensors", "ltx-2.3-22b-distilled-1.1", "ltx-2.3-22b-distilled", "ltx-2.3-22b-dev-fp8.safetensors", "ltx-2.3-22b-dev-fp8", "ltx-2.3", "ltx_2.3"],
        "vae_hints":     ["Baked VAE (from UNET)", "LTX23_video_vae", "ltx23_video", "ltx_23_video"],
        // ALBABIT-FIX: without this, autoFillPresetFiles() falls back to
        // audio_vae_name = "None" (no hints), so extract_audio_vae is False
        // and the AUDIO_VAE output stays None, failing downstream with
        // "Audio VAE model is required" (nodes_lt_audio.py).
        "audio_vae_hints": ["Baked Audio VAE (from UNET)"],
        "clip_hints":    {
            "llm_encoder":     ["gemma_3_12B_it_fp4_mixed.safetensors", "gemma_3_12B_it_fp4", "gemma_3_12B_it", "gemma_3", "gemma"],
            "text_projection": ["Baked (from UNET)"],
        },
        // ALBABIT-FIX: "Low VRAM" presets force offload_mode — expose the
        // widget so the user can still override it (e.g. on a higher-VRAM
        // GPU where cpu_offload is unnecessarily slow).
        "extra_widgets": ["upscale_model_name", "offload_mode"],
        "offload_mode": "cpu_offload",
        "upscale_hints": ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "ltx-2.3", "ltx_2.3", "latent_upsampler", "upsampler"],
    },
    "Lumina2": {
        "unet_hints":    ["lumina2", "lumina-2", "lumina_2"],
        "vae_hints":     ["ae.safetensors", "flux_ae", "sd3_vae", "sd_vae", "lumina_vae"],
        "clip_hints":    {
            "llm_encoder": ["gemma_2_2b", "gemma2_2b", "gemma_2"],
        },
    },
    "Mochi": {
        "unet_hints":    ["mochi_preview", "mochi-1-preview", "genmo_mochi", "mochi"],
        "vae_hints":     ["mochi_vae", "mochi-vae"],
        "clip_hints":    {
            // ALBABIT-FIX: prioritize fp16 t5xxl for Mochi, fp8 as fallback.
            "t5xxl": ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "PixArt Sigma": {
        "unet_hints":    ["pixart_sigma", "pixart-sigma", "PixArt-Sigma"],
        "vae_hints":     ["pixart_sigma_sdxlvae", "sdxl_vae", "sd_vae", "pixart_vae", "vae-ft-mse"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "SD 1.5": {
        "unet_hints":    ["v1-5", "v1_5", "sd15", "sd-1-5", "sd_1.5"],
        "vae_hints":     ["vae-ft-mse", "sd15_vae", "kl-f8"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
        },
    },
    // ALBABIT-FIX: Large and Turbo merged -- same vae_hints/clip_hints
    // already, Turbo is Large's distilled variant (Sampler tells them apart
    // by filename via _deriveDistillationOverride). "Medium" stays separate
    // (a genuinely different-sized sibling, not a distillation pair).
    "SD3.5 Large": {
        "unet_hints":    ["sd3.5_large", "sd3-5_large", "sd3.5_large_turbo", "sd3.5_turbo", "sd3-5_turbo"],
        "vae_hints":     ["sd3_vae", "sd3.5_vae", "sd3"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
            "t5xxl":  ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "SD3.5 Medium": {
        "unet_hints":    ["sd3.5_medium", "sd3-5_medium"],
        "vae_hints":     ["sd3_vae", "sd3.5_vae", "sd3"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
            "t5xxl":  ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    // ALBABIT-FIX: Base and Turbo merged -- same reasoning as SD3.5 Large above.
    "SDXL": {
        "unet_hints":    ["sd_xl_base", "sdxl_base", "sdxl-base", "sdxl_turbo", "sdxl-turbo", "turbo"],
        "vae_hints":     ["sdxl_vae", "vae-ft-mse", "xl_vae"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
        },
    },
    "Wan 2.1": {
        "unet_hints":    ["wan2.1", "wan_2.1", "wan-2.1", "Wan2.1"],
        "vae_hints":     ["wan_2.1_vae", "wan2.1_vae", "wan_vae", "wan2_vae", "open_wan"],
        "clip_hints":    {
            "t5xxl": ["umt5_xxl", "umt5-xxl", "umt5xxl", "t5xxl"],
        },
    },
    "Wan 2.2": {
        "unet_hints":    [
            "wan2.2_t2v_high_noise_14B_fp8_scaled",
            "wan2.2_i2v_high_noise_14B_fp8_scaled",
            "wan2.2_t2v_high_noise",
            "wan2.2_i2v_high_noise",
            "wan2.2_high_noise",
            "wan2.2",
            "wan_2.2",
            "wan-2.2",
            "Wan2.2",
        ],
        "vae_hints":     ["wan_2.1_vae", "wan2.1_vae", "wan_vae", "wan2_vae", "open_wan"],
        "clip_hints":    {
            "t5xxl": ["umt5_xxl", "umt5-xxl", "umt5xxl", "t5xxl"],
        },
        // ALBABIT-FIX: either the high_noise or low_noise UNET works --
        // _find_wan_moe_companion() (nodes_loader.py) auto-loads the other
        // one server-side regardless of which one is picked here.
        "companion_linked": true,
    },
    "Wan 2.2 TI2V": {
        "unet_hints":    ["wan2.2_ti2v_5B_fp16", "wan2.2_ti2v_5B", "wan2.2_ti2v"],
        "vae_hints":     ["wan2.2_vae", "wan_2.2_vae"],
        "clip_hints":    {
            "t5xxl": ["umt5_xxl", "umt5-xxl", "umt5xxl", "t5xxl"],
        },
    },
    "Z-Image": {
        "unet_hints":    ["z_image", "z-image", "zimage"],
        "vae_hints":     ["flux_vae", "ae.safetensors", "flux_ae", "sd3_vae", "sd_vae"],
        "clip_hints":    {
            "llm_encoder": ["qwen_3_4b", "qwen3_4b", "qwen_3"],
        },
    },
};

const ALL_CLIP_WIDGETS = ["clip_l", "clip_g", "t5xxl", "llm_encoder", "text_projection"];

function getWidget(node, name) {
    return node.widgets?.find(w => w.name === name) ?? null;
}

/**
 * Fuzzy search through options values list for any item containing the hints
 */
function findMatchingFile(hints, values) {
    if (!hints || !values) return null;
    for (const hint of hints) {
        const h = hint.toLowerCase();
        for (const val of values) {
            if (val.toLowerCase().includes(h)) {
                return val;
            }
        }
    }
    return null;
}

/**
 * Extract a size token like "4b"/"9b" from a filename (e.g. Flux.2 Klein
 * variants), used to pick a differently-sized paired text encoder.
 */
function _detectSizeToken(filename) {
    const m = /(?:^|[-_])(\d+b)(?:[-_.]|$)/i.exec((filename || "").toLowerCase());
    return m ? m[1] : null;
}

/**
 * Like _detectSizeToken, but only returns a token the preset actually knows
 * about (config.clip_size_hints) -- an unrelated UNET (e.g. Cosmos "...7B...")
 * can contain a size-shaped substring without being a real Klein size.
 */
function _knownSizeToken(config, filename) {
    const token = _detectSizeToken(filename);
    return (token && config?.clip_size_hints?.[token]) ? token : null;
}

/**
 * Resolve the CLIP hint list for a given slot, applying a preset's
 * clip_size_hints (size-aware override) when present, falling back to its
 * static clip_hints otherwise.
 */
function _resolveClipHints(node, config, wName) {
    if (wName === "llm_encoder" && config?.clip_size_hints) {
        const sizeToken = _knownSizeToken(config, getWidget(node, "unet_name")?.value || "");
        if (sizeToken) {
            return config.clip_size_hints[sizeToken];
        }
    }
    return config?.clip_hints?.[wName];
}

/**
 * Resolve which file (if any) currently matches a CLIP slot's hints.
 */
function _resolveClipMatch(node, config, wName) {
    const w = getWidget(node, wName);
    const hints = _resolveClipHints(node, config, wName);
    return (w && hints && w.options?.values) ? findMatchingFile(hints, w.options.values) : null;
}

/**
 * Performs client-side smart auto-fill matching of files based on selected preset
 */
function autoFillPresetFiles(node, cleanPreset) {
    const config = PRESET_CONFIGS[cleanPreset];
    if (!config) return;

    // 1. Match UNET
    const unetW = getWidget(node, "unet_name");
    if (unetW && unetW.options?.values && config.unet_hints) {
        const matched = findMatchingFile(config.unet_hints, unetW.options.values);
        if (matched) unetW.value = matched;
    }

    // 2. Match VAE
    const vaeW = getWidget(node, "vae_name");
    if (vaeW && vaeW.options?.values && config.vae_hints) {
        const matched = findMatchingFile(config.vae_hints, vaeW.options.values);
        if (matched) vaeW.value = matched;
    }

    // 3. Match CLIP Slots
    for (const wName of ALL_CLIP_WIDGETS) {
        const clipW = getWidget(node, wName);
        if (!clipW) continue;

        const hints = _resolveClipHints(node, config, wName);
        if (hints && clipW.options?.values) {
            const matched = findMatchingFile(hints, clipW.options.values);
            clipW.value = matched || "None";
        } else {
            clipW.value = "None";
        }
    }

    // 4. Match Audio VAE (LTX 2.3, Radiance Video Loader only)
    const audioVaeW = getWidget(node, "audio_vae_name");
    if (audioVaeW && audioVaeW.options?.values) {
        if (config.audio_vae_hints) {
            const matched = findMatchingFile(config.audio_vae_hints, audioVaeW.options.values);
            audioVaeW.value = matched || "None";
        } else {
            audioVaeW.value = "None";
        }
    }

    // 5. Match Latent Upscale Model (Radiance Video Loader only)
    const upscaleW = getWidget(node, "upscale_model_name");
    if (upscaleW && upscaleW.options?.values) {
        if (config.upscale_hints) {
            const matched = findMatchingFile(config.upscale_hints, upscaleW.options.values);
            upscaleW.value = matched || "None";
        } else {
            upscaleW.value = "None";
        }
    }

    // 6. Default offload_mode (visible widget for "Low VRAM" presets — see
    // extra_widgets — but the user can change it afterward; not preset-locked)
    const offloadW = getWidget(node, "offload_mode");
    if (offloadW) {
        offloadW.value = config.offload_mode || "none";
    }
}

// ALBABIT-FIX: unet_name/vae_name/CLIP slots/audio_vae_name/upscale_model_name
// stay visible and editable while a preset is active (unlike model_type/
// weight_dtype/clip_dtype, hidden by design — loader_utils.py). Flag any that
// no longer match what autoFillPresetFiles() would pick right now, with a "✎"
// label marker (same pattern as radiance_sampler.js/radiance_prompt.js).
const PRESET_MARKER = " ✎";
// ALBABIT-FIX: "🧲" marks unet_name whenever the file is a recognized preset
// variant -- unet_name always has a real link to other magnet-marked widgets
// (the Sampler's model_meta-driven defaults, and llm_encoder for Klein), so
// picking any recognized file is never a mistake, unlike a genuinely
// unrelated file (e.g. Flux1 Fill under "Flux.1"), which still gets "✎".
const LINKED_MARKER = " 🧲";
// ALBABIT-FIX: for companion_linked presets (Wan 2.2), either the high_noise
// or low_noise UNET is a valid pick -- "⛓" marks unet_name instead of "✎"
// to signal its companion is auto-loaded server-side, not a manual mistake.
const COMPANION_MARKER = " ⛓";

// unet_name/vae_name are left untouched by autoFillPresetFiles() on no
// match; CLIP slots/audio_vae/upscale fall back to "None" instead.
const NO_NONE_FALLBACK_FIELDS = new Set(["unet_name", "vae_name"]);

function _markFileWidget(widget, markerText) {
    if (!widget) return false;
    if (widget._radOrigLabel === undefined && !markerText) return false;
    if (widget._radOrigLabel === undefined) widget._radOrigLabel = widget.label ?? widget.name;
    const wanted = markerText ? widget._radOrigLabel + markerText : widget._radOrigLabel;
    if (widget.label === wanted) return false;
    widget.label = wanted;
    return true;
}

function updatePresetDivergenceMarkers(node) {
    if (!node.widgets) return;
    const presetW = getWidget(node, "preset");
    const presetVal = presetW ? presetW.value : "Custom";
    const cleanPreset = presetVal ? presetVal.replace("→ ", "").replace("▶ ", "").replace("◈ ", "").trim() : "Custom";
    const config = cleanPreset === "Custom" ? null : PRESET_CONFIGS[cleanPreset];
    const activeSlots = config ? (PRESET_SLOTS[cleanPreset] || ALL_CLIP_WIDGETS) : [];
    const unetVal = getWidget(node, "unet_name")?.value || "";
    const unetRecognized = !!findMatchingFile(config?.unet_hints, [unetVal]);
    // ALBABIT-FIX: unetVariantLinked (any recognized file -- unet_name always
    // feeds the Sampler's model_meta magnets) vs clipLinked (llm_encoder also
    // depends on it, Klein-only) -- previously conflated under one check.
    const unetVariantLinked = !!(config && unetRecognized);
    const clipLinked = !!(config?.clip_size_hints && unetRecognized);
    const companionLinked = !!(config?.companion_linked && unetRecognized);

    let changed = false;

    const check = (widgetName, hints) => {
        const w = getWidget(node, widgetName);
        if (!w) return;
        let markerText = null;
        if (config) {
            if (!hints || hints.length === 0) {
                if (String(w.value) !== "None") markerText = PRESET_MARKER;
            } else {
                const matched = findMatchingFile(hints, w.options?.values);
                if (matched !== null) {
                    if (String(w.value) !== String(matched)) markerText = PRESET_MARKER;
                } else if (!NO_NONE_FALLBACK_FIELDS.has(widgetName)) {
                    if (String(w.value) !== "None") markerText = PRESET_MARKER;
                }
            }
        }
        if (_markFileWidget(w, markerText)) changed = true;
    };

    // unet_name shows a link marker instead of the divergence marker while
    // companionLinked/unetVariantLinked -- the point is to signal a
    // relationship (auto-loaded companion, recognized preset variant), not
    // flag a mistake. companionLinked checked first: more specific than the
    // general "recognized" case.
    if (companionLinked) {
        if (_markFileWidget(getWidget(node, "unet_name"), COMPANION_MARKER)) changed = true;
    } else if (unetVariantLinked) {
        if (_markFileWidget(getWidget(node, "unet_name"), LINKED_MARKER)) changed = true;
    } else {
        check("unet_name", config?.unet_hints);
    }
    check("vae_name", config?.vae_hints);
    for (const wName of ALL_CLIP_WIDGETS) {
        if (config && !activeSlots.includes(wName)) continue; // hidden slot
        if (wName === "llm_encoder" && clipLinked) {
            const w = getWidget(node, "llm_encoder");
            const matched = _resolveClipMatch(node, config, "llm_encoder");
            const markerText = matched === null ? null
                : (String(w.value) === String(matched) ? LINKED_MARKER : PRESET_MARKER);
            if (_markFileWidget(w, markerText)) changed = true;
            continue;
        }
        check(wName, config ? _resolveClipHints(node, config, wName) : null);
    }
    check("audio_vae_name", config?.audio_vae_hints);
    check("upscale_model_name", config?.upscale_hints);

    if (changed) node.setDirtyCanvas(true, true);
}

/**
 * ALBABIT-FIX: Vue 3's virtual-DOM differ reuses the existing widget component
 * instance when the same object reference stays in node.widgets — a plain
 * `splice(0, 0)` no-op notifies Vue "something changed" but Vue doesn't
 * re-read `type`/`options.hidden` on that reused instance, so a widget
 * restored from `type === "hidden"` can stay invisible/zero-height (e.g. when
 * switching back to "Custom"). Removing and re-inserting the widget at the
 * same index forces Vue to destroy and remount its component.
 */
function _forceWidgetReinsert(widget, node) {
    if (!node?.widgets) return;
    const idx = node.widgets.indexOf(widget);
    if (idx === -1) return;
    node.widgets.splice(idx, 1);
    node.widgets.splice(idx, 0, widget);
}

/**
 * Collapsible widget visibility helper
 */
function setWidgetVisible(widget, visible, node) {
    if (!widget) return;

    if (!widget.options) widget.options = {};
    widget.options.hidden = !visible;

    widget.hidden = !visible;

    if (visible) {
        if (widget.type === "hidden") {
            widget.type = widget._origType || "combo";
            delete widget.computeSize;
            delete widget._origComputeSize;
            if (widget._origDraw !== undefined) {
                widget.draw = widget._origDraw;
                delete widget._origDraw;
            } else {
                delete widget.draw;
            }
            if (widget.inputEl) widget.inputEl.style.display = "";
            if (widget.element)  widget.element.style.display  = "";
            if (widget._origComputedHeight !== undefined) {
                widget.computedHeight = widget._origComputedHeight;
                delete widget._origComputedHeight;
            } else {
                widget.computedHeight = 32;
            }
        }
    } else {
        if (widget.type !== "hidden") {
            widget._origType        = widget.type;
            widget._origComputeSize = widget.computeSize;
            widget._origComputedHeight = widget.computedHeight;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
            if (widget.draw) widget._origDraw = widget.draw;
            widget.draw = function() {};
            if (widget.inputEl) widget.inputEl.style.display = "none";
            if (widget.element)  widget.element.style.display  = "none";
            widget.computedHeight = 4;
        }
    }

    // ALBABIT-FIX: always force a remove+reinsert, even if type/hidden didn't
    // change this call. Empirically, once a widget's Vue component has been
    // (re)mounted, it stops reacting to later type/hidden changes via a
    // no-op splice(0,0) alone -- it keeps rendering its previous state until
    // reinserted again (e.g. after "LTX -> Custom -> LTX", the 2nd LTX still
    // showed every widget, even though widget.type was already correctly
    // "hidden" again). Reinserting unconditionally guarantees every widget's
    // component reflects its current state regardless of how many times it
    // toggled before.
    _forceWidgetReinsert(widget, node);
}

/**
 * Recalculate node dimensions and refresh the canvas layout cleanly
 */
function refreshNodeSize(node) {
    if (!node.computeSize) return;

    const sz = node.computeSize();
    // ALBABIT-FIX: directly mutating node.size[i] updates the LiteGraph
    // model but Vue's node component never observes it, so the rendered
    // box keeps its old (larger) height forever. node.setSize(...) is the
    // API Vue's resize handling actually reacts to, and computeSize() is
    // already correct synchronously here (no DOM-timing issue), so a
    // single immediate call is enough.
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    app.graph.setDirtyCanvas(true, true);
}

function updateLoaderUI(node, forceAutoFill = false) {
    if (!node.widgets) return;

    const presetW = getWidget(node, "preset");
    if (!presetW) return;

    const presetVal = presetW.value;
    // Strip cosmetic menu symbol prefixes (e.g. arrow icons)
    const cleanPreset = presetVal ? presetVal.replace("→ ", "").replace("▶ ", "").replace("◈ ", "").trim() : "Custom";

    const isCustom = cleanPreset === "Custom";

    if (isCustom) {
        // SHOW ALL WIDGETS
        node.widgets.forEach(w => {
            setWidgetVisible(w, true, node);
        });
        refreshNodeSize(node);
        updatePresetDivergenceMarkers(node);
        return;
    }

    // --- Specific Model Preset mode ---
    // Hide: model_type, weight_dtype, clip_dtype, offload_mode (preset manages them)
    // Hide general utilities to keep the UI clean: check_vram, use_cache, lora_on_error, auto_download
    // Show only: preset, unet_name, vae_name, and active CLIP slots!
    const activeSlots = PRESET_SLOTS[cleanPreset] || ALL_CLIP_WIDGETS;
    const extraWidgets = (PRESET_CONFIGS[cleanPreset] && PRESET_CONFIGS[cleanPreset].extra_widgets) || [];

    if (forceAutoFill) {
        autoFillPresetFiles(node, cleanPreset);
    }

    node.widgets.forEach(w => {
        if (w.name === "preset" || w.name === "unet_name" || w.name === "vae_name") {
            setWidgetVisible(w, true, node);
        } else if (ALL_CLIP_WIDGETS.includes(w.name)) {
            const shouldShow = activeSlots.includes(w.name);
            setWidgetVisible(w, shouldShow, node);
        } else if (extraWidgets.includes(w.name)) {
            setWidgetVisible(w, true, node);
        } else {
            setWidgetVisible(w, false, node);
        }
    });

    refreshNodeSize(node);
    updatePresetDivergenceMarkers(node);
}

// ALBABIT-FIX: app.registerExtension({ nodeCreated, loadedGraphNode }) wraps
// presetW.callback / modelTypeW.callback AFTER Vue (Nodes 2.0) has already
// mounted the combo widget components, so the wrapped callback is never
// invoked when the user changes the dropdown (no console output, nothing is
// folded/auto-filled). The beforeRegisterNodeDef + prototype.onNodeCreated /
// onConfigure pattern (as used in radiance_sampler.js) hooks the node before
// widget construction, so the wrapped callbacks are the ones Vue captures and
// actually fire on user interaction. The preset/auto-fill logic itself
// (updateLoaderUI, autoFillPresetFiles, PRESET_SLOTS, ...) is unchanged.
app.registerExtension({
    name: "Radiance.UnifiedLoaderSync",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!LOADER_NODES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;
            const presetW = getWidget(node, "preset");
            const modelTypeW = getWidget(node, "model_type");

            if (presetW) {
                const origPresetCallback = presetW.callback;
                presetW.callback = function(value) {
                    if (origPresetCallback) origPresetCallback.call(this, value);
                    // Trigger updates and execute smart file matching
                    setTimeout(() => updateLoaderUI(node, true), 10);
                };
            }

            if (modelTypeW) {
                const origModelCallback = modelTypeW.callback;
                modelTypeW.callback = function(value) {
                    if (origModelCallback) origModelCallback.call(this, value);
                    setTimeout(() => updateLoaderUI(node, false), 10);
                };
            }

            // ALBABIT-FIX: for clip_size_hints presets (Flux.2 Klein), keep
            // llm_encoder in sync when the user manually changes unet_name.
            // No-op for every other preset.
            const unetW = getWidget(node, "unet_name");
            if (unetW) {
                const origUnetCallback = unetW.callback;
                unetW.callback = function(value) {
                    if (origUnetCallback) origUnetCallback.call(this, value);
                    setTimeout(() => {
                        const presetVal = getWidget(node, "preset")?.value;
                        const cleanPreset = presetVal ? presetVal.replace("→ ", "").replace("▶ ", "").replace("◈ ", "").trim() : "Custom";
                        const cfg = PRESET_CONFIGS[cleanPreset];
                        if (cfg?.clip_size_hints) {
                            const clipW = getWidget(node, "llm_encoder");
                            const matched = _resolveClipMatch(node, cfg, "llm_encoder");
                            if (clipW && matched) {
                                clipW.value = matched;
                                node.setDirtyCanvas(true, true);
                            }
                        }
                        updatePresetDivergenceMarkers(node);
                    }, 10);
                };
            }

            // Apply initial layout folding on creation, unless onConfigure
            // (loaded workflow) is about to do it with the restored values.
            setTimeout(() => {
                if (node._configuredByLoad) return;
                updateLoaderUI(node, false);
            }, 50);

            // File widgets aren't individually wrapped, so poll for manual
            // edits (same pattern as radiance_sampler.js/radiance_prompt.js).
            // Calls only the lightweight marker check, never updateLoaderUI()
            // itself, which remounts every widget on each call.
            node._presetMarkerInterval = setInterval(() => updatePresetDivergenceMarkers(node), 250);
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                if (node._presetMarkerInterval) {
                    clearInterval(node._presetMarkerInterval);
                    node._presetMarkerInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            this._configuredByLoad = true;
            const node = this;
            // ALBABIT-FIX: a single 100ms reapply can fire before graph.configure()
            // has finished applying the saved "preset" widget value, so
            // updateLoaderUI reads the still-default value and folds as if
            // "Custom" (showing every widget) — and nothing corrects it
            // afterwards. Mirror radiance_sampler.js's onConfigure: 150ms for
            // Vue's first layout pass, 600ms as a safety net for heavy
            // workflows where configure() takes longer than 150ms.
            setTimeout(() => updateLoaderUI(node, false), 150);
            setTimeout(() => updateLoaderUI(node, false), 600);
            return r;
        };
    }
});
