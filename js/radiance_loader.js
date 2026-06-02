/**
 * radiance_loader.js
 * v3.3 — Streamlined Radiance Loader Widget sync, dynamic visibility & smart auto-fill
 */

import { app } from "../../scripts/app.js";

// Node definition identifiers
const LOADER_NODES = ["RadianceUnifiedLoader", "RadianceImageLoader", "RadianceVideoLoader"];

// Dynamic visibility rules for CLIP slots per preset
const PRESET_SLOTS = {
    "Custom": ["clip_l", "clip_g", "t5xxl", "llm_encoder", "text_projection"],
    "Flux Dev": ["clip_l", "t5xxl"],
    "Flux Schnell": ["clip_l", "t5xxl"],
    "Flux Dev (Low VRAM)": ["clip_l", "t5xxl"],
    "SD3.5 Large": ["clip_l", "clip_g", "t5xxl"],
    "SD3.5 Medium": ["clip_l", "clip_g", "t5xxl"],
    "SD3.5 Turbo": ["clip_l", "clip_g", "t5xxl"],
    "SDXL Base": ["clip_l", "clip_g"],
    "SDXL Turbo": ["clip_l", "clip_g"],
    "SD 1.5": ["clip_l"],
    "HunyuanVideo": ["clip_l", "llm_encoder"],
    "Wan 2.1": ["t5xxl"],
    "LTX Video": ["llm_encoder", "text_projection"],
    "LTX Video 13B": ["llm_encoder", "text_projection"],
    "LTX Video 2.3": ["llm_encoder", "text_projection"],
    "LTX Video 2.3 (Low VRAM)": ["llm_encoder", "text_projection"],
    "PixArt Sigma": ["t5xxl"],
    "AuraFlow": ["clip_l"],
    "Kolors": ["llm_encoder"],
    "Lumina2": ["t5xxl"],
    "Z-Image": ["t5xxl"],
};

// Dynamic visibility rules for CLIP slots per model_type
const MODEL_SLOTS = {
    "Auto-Detect": ["clip_l", "clip_g", "t5xxl", "llm_encoder", "text_projection"], // Show all for safety
    "flux": ["clip_l", "t5xxl"],
    "sd3": ["clip_l", "clip_g", "t5xxl"],
    "sd3.5": ["clip_l", "clip_g", "t5xxl"],
    "sdxl": ["clip_l", "clip_g"],
    "sd1.5": ["clip_l"],
    "hunyuan_video": ["clip_l", "llm_encoder"],
    "wan": ["t5xxl"],
    "ltx": ["llm_encoder", "text_projection"],
    "ltxav": ["llm_encoder", "text_projection"],
    "lumina2": ["t5xxl"],
    "z_image": ["t5xxl"],
    "pixart": ["t5xxl"],
    "aura_flow": ["clip_l"],
    "kolors": ["llm_encoder"],
};

// Full hints configs for automatic local file selection matching
const PRESET_CONFIGS = {
    "Flux Dev": {
        "unet_hints":    ["flux1-dev-fp8", "flux1-dev", "flux-dev"],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl":  ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "Flux Schnell": {
        "unet_hints":    ["flux1-schnell-fp8", "flux1-schnell", "flux-schnell"],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl":  ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "Flux Dev (Low VRAM)": {
        "unet_hints":    ["flux1-dev-fp8", "flux1-dev", "flux-dev"],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl":  ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "SD3.5 Large": {
        "unet_hints":    ["sd3.5_large_turbo", "sd3.5_large", "sd3-5_large"],
        "vae_hints":     ["sd3_vae", "sd3.5_vae", "sd3"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
            "t5xxl":  ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "SD3.5 Medium": {
        "unet_hints":    ["sd3.5_medium", "sd3-5_medium"],
        "vae_hints":     ["sd3_vae", "sd3.5_vae", "sd3"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
            "t5xxl":  ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "SD3.5 Turbo": {
        "unet_hints":    ["sd3.5_large_turbo", "sd3.5_turbo", "sd3-5_turbo"],
        "vae_hints":     ["sd3_vae", "sd3.5_vae", "sd3"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
            "t5xxl":  ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "SDXL Base": {
        "unet_hints":    ["sd_xl_base", "sdxl_base", "sdxl-base"],
        "vae_hints":     ["sdxl_vae", "vae-ft-mse", "xl_vae"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
        },
    },
    "SDXL Turbo": {
        "unet_hints":    ["sdxl_turbo", "sdxl-turbo", "turbo"],
        "vae_hints":     ["sdxl_vae", "vae-ft-mse", "xl_vae"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
        },
    },
    "SD 1.5": {
        "unet_hints":    ["v1-5", "v1_5", "sd15", "sd-1-5", "sd_1.5"],
        "vae_hints":     ["vae-ft-mse", "sd15_vae", "kl-f8"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
        },
    },
    "HunyuanVideo": {
        "unet_hints":    ["hunyuan_video", "hunyuanvideo", "hyvideo"],
        "vae_hints":     ["hunyuan_video_vae", "hunyuan_vae", "hyvideo_vae"],
        "clip_hints":    {
            "llm_encoder": ["llava_llama3", "llm_llava", "hunyuan_llm"],
            "clip_l":      ["clip_l.safetensors", "clip_l"],
        },
    },
    "Wan 2.1": {
        "unet_hints":    ["wan2.1", "wan_2.1", "wan-2.1", "Wan2.1"],
        "vae_hints":     ["wan_vae", "wan2_vae", "open_wan"],
        "clip_hints":    {
            "t5xxl": ["umt5-xxl", "umt5xxl", "t5xxl"],
        },
    },
    "LTX Video": {
        "unet_hints":    ["ltx-video-2b", "ltxv-2b", "ltx_video", "ltxv"],
        "vae_hints":     ["ltx_vae", "ltxv_vae", "causal_vae"],
        "clip_hints":    {
            "llm_encoder": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "LTX Video 13B": {
        "unet_hints":    ["ltx-video-13b", "ltxv-13b", "ltx_13b"],
        "vae_hints":     ["ltx_vae", "ltxv_vae", "causal_vae"],
        "clip_hints":    {
            "llm_encoder": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "LTX Video 2.3": {
        "unet_hints":    ["ltx-2.3-22b-dev", "ltx-2.3", "ltx_2.3"],
        "vae_hints":     ["LTX23_video_vae", "ltx23_video", "ltx_23_video"],
        "clip_hints":    {
            "llm_encoder":     ["gemma_3_12B_it_fp4", "gemma_3_12B_it", "gemma_3", "gemma"],
            "text_projection": ["ltx-2.3_text_projection", "text_projection"],
        },
    },
    "LTX Video 2.3 (Low VRAM)": {
        "unet_hints":    ["ltx-2.3-22b-dev-fp8", "ltx-2.3", "ltx_2.3"],
        "vae_hints":     ["LTX23_video_vae", "ltx23_video", "ltx_23_video"],
        "clip_hints":    {
            "llm_encoder":     ["gemma_3_12B_it_fp4", "gemma_3_12B_it", "gemma_3", "gemma"],
            "text_projection": ["ltx-2.3_text_projection", "text_projection"],
        },
    },
    "PixArt Sigma": {
        "unet_hints":    ["pixart_sigma", "pixart-sigma", "PixArt-Sigma"],
        "vae_hints":     ["sd_vae", "pixart_vae", "vae-ft-mse"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "AuraFlow": {
        "unet_hints":    ["auraflow", "aura_flow", "aura-flow"],
        "vae_hints":     ["aura_vae", "sd_vae"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
        },
    },
    "Kolors": {
        "unet_hints":    ["kolors", "Kolors"],
        "vae_hints":     ["kolors_vae", "sdxl_vae"],
        "clip_hints":    {
            "llm_encoder": ["chatglm3", "chatglm", "kolors_clip"],
        },
    },
    "Lumina2": {
        "unet_hints":    ["lumina2", "lumina-2", "lumina_2"],
        "vae_hints":     ["sd3_vae", "sd_vae", "lumina_vae"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "Z-Image": {
        "unet_hints":    ["z_image", "z-image", "zimage"],
        "vae_hints":     ["sd3_vae", "sd_vae"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
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
    const clipHints = config.clip_hints || {};
    for (const wName of ALL_CLIP_WIDGETS) {
        const clipW = getWidget(node, wName);
        if (!clipW) continue;

        if (clipHints[wName] && clipW.options?.values) {
            const matched = findMatchingFile(clipHints[wName], clipW.options.values);
            if (matched) {
                clipW.value = matched;
            } else {
                clipW.value = "None";
            }
        } else {
            clipW.value = "None";
        }
    }
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
    if (node?.widgets) node.widgets.splice(0, 0);
}

/**
 * Recalculate node dimensions and refresh the canvas layout cleanly
 */
function refreshNodeSize(node) {
    if (node.computeSize) {
        const sz = node.computeSize();
        node.size[0] = Math.max(node.size[0], sz[0]);
        node.size[1] = sz[1];
        app.graph.setDirtyCanvas(true, true);
    }
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
        return;
    }

    // --- Specific Model Preset mode ---
    // Hide: model_type, weight_dtype, clip_dtype, offload_mode (preset manages them)
    // Hide general utilities to keep the UI clean: check_vram, use_cache, lora_on_error, auto_download
    // Show only: preset, unet_name, vae_name, and active CLIP slots!
    const activeSlots = PRESET_SLOTS[cleanPreset] || ALL_CLIP_WIDGETS;

    if (forceAutoFill) {
        autoFillPresetFiles(node, cleanPreset);
    }

    node.widgets.forEach(w => {
        if (w.name === "preset" || w.name === "unet_name" || w.name === "vae_name") {
            setWidgetVisible(w, true, node);
        } else if (ALL_CLIP_WIDGETS.includes(w.name)) {
            const shouldShow = activeSlots.includes(w.name);
            setWidgetVisible(w, shouldShow, node);
        } else {
            setWidgetVisible(w, false, node);
        }
    });

    refreshNodeSize(node);
}

app.registerExtension({
    name: "Radiance.UnifiedLoaderSync",

    nodeCreated(node) {
        const nodeId = node.type ?? node.comfyClass ?? "";
        if (!LOADER_NODES.includes(nodeId)) return;

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

        // Apply initial layout folding immediately on creation (defaults to None, so everything folds)
        setTimeout(() => updateLoaderUI(node, false), 50);
    },

    loadedGraphNode(node) {
        const nodeId = node.type ?? node.comfyClass ?? "";
        if (!LOADER_NODES.includes(nodeId)) return;
        setTimeout(() => updateLoaderUI(node, false), 100);
    }
});
