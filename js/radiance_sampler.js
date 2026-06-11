import { app } from "../../scripts/app.js";

const PRESET_CONFIGS = {
    "→ Flux txt2img": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Standard Flux text-to-image. Optimal for 1024×1024 images.",
    },
    "→ Flux img2img": {
        steps: 20, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 0.75, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Image-to-image refinement. denoise=0.75 for balanced changes.",
    },
    "→ Flux Inpaint": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 4.0,
        description: "Inpainting with guidance=4.0 for strong detail matching.",
    },
    "→ Flux High-Res Fix": {
        steps: 20, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 0.5, flux_shift: 3.0, flux_guidance: 3.5,
        description: "2× upscale. shift=3.0 enhances high-frequency detail.",
    },
    "→ Flux Fast (12 steps)": {
        steps: 12, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Quick generation for prompt testing and iteration.",
    },
    "→ Flux Quality (28 steps)": {
        steps: 28, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 4.0,
        description: "Maximum quality final outputs. Higher guidance for adherence.",
    },
    "→ Flux Cinematic (30 steps)": {
        steps: 30, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 4.0,
        description: "Cinema-grade 30-step Flux run with strong guidance.",
    },
    "→ Flux Schnell (4 steps)": {
        steps: 4, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 0.0,
        description: "Flux Schnell distilled model — 4 steps, no guidance needed.",
    },
    "→ SD3.5 Turbo (4 steps)": {
        steps: 4, cfg: 1.6, sampler: "euler", scheduler: "sgm_uniform",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 0.0,
        description: "SD3.5 Turbo — 4-step distilled, CFG=1.6.",
    },
    "→ Flux Ultra Fast (8 steps)": {
        steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 2.0,
        description: "8-step Flux with reduced guidance for fast drafts.",
    },
    "▶ WAN txt2vid (30 steps)": {
        steps: 30, cfg: 6.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 8.0, flux_guidance: 0.0,
        description: "WAN text-to-video. shift=8 is critical for correct temporal dynamics.",
    },
    "▶ WAN img2vid (20 steps)": {
        steps: 20, cfg: 6.0, sampler: "euler", scheduler: "simple",
        denoise: 0.75, flux_shift: 8.0, flux_guidance: 0.0,
        description: "WAN image-to-video. 20 steps at denoise=0.75.",
    },
    "▶ LTX-Video (25 steps)": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 2.37, flux_guidance: 0.0,
        description: "LTX-V standard — shift=2.37 per spec.",
    },
    "▶ LTX 2.3 LowRes (20 steps)": {
        steps: 20, start_step: 0, end_step: 0, cfg: 3.0, sampler: "euler",
        sampler_mode: "Standard", phase_split: 0.0, scheduler: "beta",
        scheduler_mode: "Manual", denoise: 1.0, flux_shift: 3.0,
        flux_guidance: 0.0, flux_guidance_profile: "Static", add_noise: true,
        return_with_leftover_noise: false, seed: 0, control_after_generate: "fixed",
        pag_scale: 0.0, model_type: "ltxav", sigma_blend_steps: 0, ays_schedule: false,
        guidance_rescale_phi: 0.0, preview_method: "None", noise_type: "Gaussian",
        conditioning_clip_target: "Auto",
        tile_mode: false, refiner_start_step: 0, latent_format: "",
        terminal_sigma_to_zero: true, force_exact_steps: true,
        description: "LTX 2.3 LowRes. Optimal settings for 720p base generation.",
    },
    "▶ LTX 2.3 HighRes (40 steps)": {
        steps: 40, start_step: 0, end_step: 0, cfg: 3.0, sampler: "euler",
        sampler_mode: "Standard", phase_split: 0.0, scheduler: "beta",
        scheduler_mode: "Manual", denoise: 0.45, flux_shift: 6.0,
        flux_guidance: 0.0, flux_guidance_profile: "Static", add_noise: true,
        return_with_leftover_noise: false, seed: 0, control_after_generate: "fixed",
        pag_scale: 0.0, model_type: "ltxav", sigma_blend_steps: 0, ays_schedule: false,
        guidance_rescale_phi: 0.0, preview_method: "None", noise_type: "Gaussian",
        conditioning_clip_target: "Auto",
        tile_mode: false, refiner_start_step: 0, latent_format: "",
        terminal_sigma_to_zero: true, force_exact_steps: true,
        description: "High-Res upscale. Uses Euler by default. If using a LoRA, adjust denoise as needed.",
    },
    "▶ HunyuanVideo (30 steps)": {
        steps: 30, cfg: 6.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 7.0, flux_guidance: 0.0,
        description: "HunyuanVideo — shift=7, CFG=6.",
    },
    "◈ Draft (4-step / AYS)": {
        steps: 4, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Ultra-fast AYS draft. Great for rough composition checks.",
    },
    "◈ Fast (8-step / AYS)": {
        steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "AYS 8-step — good quality/speed balance.",
    },
    "◈ Balanced (20-step)": {
        steps: 20, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Default 20-step production run.",
    },
    "◈ Quality (35-step)": {
        steps: 35, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 4.0,
        description: "High-quality 35-step with Phase-Shift SGM in the backend.",
    },
    "◈ Cinema (60-step)": {
        steps: 60, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.5, flux_guidance: 4.5,
        description: "Cinema-grade 60-step with Phase-Shift SGM. Maximum fidelity.",
    },
    "◈ z_image (25 steps)": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 3.0, flux_guidance: 3.5,
        description: "z_image / Lumina variant — shift=3.",
    },
    "◈ Lumina2 (25 steps)": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 6.0, flux_guidance: 3.5,
        description: "Lumina2 — shift=6, guidance=3.5.",
    },
};

const LTX_PRESETS = [
    "▶ LTX 2.3 LowRes (20 steps)",
    "▶ LTX 2.3 HighRes (40 steps)"
];

// Model taxonomy — mirrors sampler_utils.py so the UI folds the same way the
// backend resolves models. GUIDANCE_EMBED models use flux_guidance; CFG_GUIDED
// models drive denoising with plain CFG and ignore the guidance-embed widgets.
const GUIDANCE_EMBED_MODELS = new Set(["flux", "lumina2", "z_image", "ltxv"]);
const CFG_GUIDED_MODELS = new Set([
    "wan", "hunyuan_video", "sdxl", "sd15", "sd3", "sd35",
    "ltxav", "cogvideox", "stepvideo"
]);
const LTX_MODEL_TYPES = new Set(["ltxv", "ltxav"]);

// Infer the effective model type when model_type is left on "auto" by reading
// the chosen preset name. Returns "auto" when nothing matches (treated as a
// generic flux-style flow model — guidance widgets stay visible).
// The dropdown values come from the backend (WORKFLOW_PRESETS) as "[F] …",
// "[V] …", "[Q] …", but the PRESET_CONFIGS table above is keyed with "→/▶/◈"
// markers. Match by the label after the marker so either naming resolves to the
// same config — without this, selecting a preset silently applies nothing.
function normPresetKey(name) {
    return String(name || "")
        .replace(/^\s*\[[A-Za-z]\]\s*/, "")
        .replace(/^\s*[^A-Za-z0-9]+\s*/, "")
        .trim()
        .toLowerCase();
}
function getPresetConfig(name) {
    if (!name) return null;
    if (PRESET_CONFIGS[name]) return PRESET_CONFIGS[name];
    const target = normPresetKey(name);
    for (const k in PRESET_CONFIGS) {
        if (normPresetKey(k) === target) return PRESET_CONFIGS[k];
    }
    return null;
}

function resolveModelType(presetVal, modelTypeVal) {
    // ALBABIT-FIX: prioritise preset name FIRST — it is always more reliable than
    // modelTypeVal, which may carry a stale backend default ("ltxav") from an earlier
    // LTX workflow, causing Flux/WAN presets to be falsely classified as isLTX and
    // hiding flux_guidance / tile widgets even for non-LTX presets.
    const p = (presetVal || "").toLowerCase();
    if (LTX_PRESETS.includes(presetVal) || p.includes("ltx 2.3")) return "ltxav";
    if (p.includes("ltx"))      return "ltxv";
    if (p.includes("wan"))      return "wan";
    if (p.includes("hunyuan"))  return "hunyuan_video";
    if (p.includes("z_image"))  return "z_image";
    if (p.includes("lumina"))   return "lumina2";
    if (p.includes("sd3.5") || p.includes("sd35")) return "sd35";
    if (p.includes("flux") || p.includes("schnell") || p.includes("draft") ||
        p.includes("fast")    || p.includes("balanced") || p.includes("quality") ||
        p.includes("cinema")  || p.includes("txt2img")  || p.includes("img2img") ||
        p.includes("inpaint") || p.includes("high-res")) return "flux";
    // Preset name didn't match a known model family: fall back to the widget value
    if (modelTypeVal && modelTypeVal !== "auto") return modelTypeVal;
    return "auto";
}

const LTX_INCOMPATIBLE_WIDGETS = [
    "flux_guidance",
    "flux_guidance_profile",
    "preview_method",
    "tile_mode",
    "tile_size",
    "tile_overlap",
    "tile_stride",
    "tile_blend"
];

// ── 1. Widget visibility helpers ──
// ALBABIT-FIX: three-mechanism pattern for Nodes 2.0 + Legacy LiteGraph:
//   1. widget.options.hidden  — Nodes 2.0 Vue filter
//   2. widget.hidden          — LiteGraph getLayoutWidgets() exclusion
//   3. widget.type="hidden" + computeSize=[0,-4] + computedHeight=4 — physical height collapse

// Force Vue to destroy and recreate a widget's component instance by doing a real
// remove+re-insert in the reactive array. A splice(0,0) no-op only notifies Vue that
// the array changed but Vue's vdom differ may reuse the existing component instance
// (same object reference) and skip re-reading changed properties like `type`.
// A true remove+insert forces Vue to treat it as a new item → fresh component mount.
function _forceWidgetReinsert(widget, node) {
    if (!node?.widgets) return;
    const idx = node.widgets.indexOf(widget);
    if (idx === -1) return;
    node.widgets.splice(idx, 1);          // remove → Vue destroys component instance
    node.widgets.splice(idx, 0, widget);  // re-insert → Vue creates fresh instance
}

function setWidgetVisible(widget, visible, node) {
    if (!widget) return;

    if (!widget.options) widget.options = {};
    widget.options.hidden = !visible;
    widget.hidden = !visible;

    if (visible) {
        if (widget.type === "hidden") {
            widget.type = widget._origType || "number";
            if (widget._origComputeSize !== undefined) {
                widget.computeSize = widget._origComputeSize;
            } else {
                delete widget.computeSize;
            }
            delete widget._origComputeSize;
            // ALBABIT-FIX: set a positive default height BEFORE reinserting so Vue
            // renders the widget at a valid size on first mount (undefined → "undefinedpx"
            // in CSS collapses to 0 on page load when Vue hasn't computed heights yet).
            // The deferred cleanup in toggleFields() deletes this after Vue's first pass
            // so Vue can recompute the real height without ghost-space artefacts.
            widget.computedHeight = widget._origComputedHeight ?? 32;
            delete widget._origComputedHeight;
            // Force Vue to destroy + recreate the component instance (see comment on
            // _forceWidgetReinsert). Must run AFTER type/computedHeight are corrected.
            _forceWidgetReinsert(widget, node);
        } else {
            if (node?.widgets) node.widgets.splice(0, 0);
        }
    } else {
        if (widget.type !== "hidden") {
            widget._origType = widget.type;
            widget._origComputeSize = widget.computeSize;
            widget._origComputedHeight = widget.computedHeight;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
            widget.computedHeight = 4;
            if (node?.widgets) node.widgets.splice(0, 0);
        }
    }
}

// ── 2. Resize and redraw helper (verbatim from standard radiance_upscale.js) ──
function refreshNodeSize(node) {
    if (node.computeSize) {
        const sz = node.computeSize();
        node.size[0] = Math.max(node.size[0], sz[0]);
        node.size[1] = sz[1]; // Set height directly to let it shrink cleanly!
        app.graph.setDirtyCanvas(true, true);
    }
}

// ── 3. Dynamic folding logic ──
// Self-heal + diagnostic. A widget marked visible (widget.hidden === false) but
// still typed "hidden" means a restore was missed by the frontend — force it back
// so parameters can never silently vanish. Set window.__RADIANCE_SAMPLER_DEBUG = true
// in the browser console to log the folded state on every toggle.
function auditSamplerWidgets(node) {
    if (!node.widgets) return;
    const stuck = node.widgets
        .filter(w => w && w.name && w.hidden !== true && w.type === "hidden")
        .map(w => w.name);
    if (stuck.length) {
        stuck.forEach(name => {
            const w = node.widgets.find(x => x.name === name);
            setWidgetVisible(w, true, node);
        });
        refreshNodeSize(node);
        console.warn("[Radiance Sampler] self-healed stranded widgets:", stuck.join(", "));
    }
    if (window.__RADIANCE_SAMPLER_DEBUG) {
        const hidden = node.widgets.filter(w => w && (w.hidden || w.type === "hidden")).map(w => w.name);
        const pv = (node.widgets.find(w => w.name === "preset") || {}).value;
        console.debug("[Radiance Sampler] preset=%s | hidden=[%s]", pv, hidden.join(", "));
    }
}

function toggleFields(node) {
    if (!node.widgets) return;
    applyFolding(node);
    auditSamplerWidgets(node);
    // ALBABIT-FIX: after Vue processes this render cycle, clear the synthetic
    // computedHeight = 32 we set so Vue can recompute the real per-widget height.
    // The 32 was needed as a safe initial value for the _forceWidgetReinsert mount;
    // after Vue's first layout pass the value would be wrong if the real height ≠ 32.
    setTimeout(() => {
        if (!node.widgets) return;
        let changed = false;
        node.widgets.forEach(w => {
            if (!w.options?.hidden && !w.hidden && w.type !== "hidden") {
                if (w.computedHeight === 32) {
                    delete w.computedHeight;
                    changed = true;
                }
            }
        });
        if (changed) node.widgets.splice(0, 0);
    }, 50);
}

function applyFolding(node) {
    if (!node.widgets) return;

    const find = (name) => node.widgets.find(w => w.name === name);

    // Get key widget references
    const presetW = find("preset");
    const presetVal = presetW ? presetW.value : "None";
    const isNone = presetVal === "None";
    const isCustom = presetVal === "Custom";

    // Dummy compatibility absorbers that are always hidden
    const dummyWidgets = ["_js_export_btn", "_js_import_btn", "_js_preset_info"];
    const controlAfterW = find("control_after_generate");

    // ── None: simple default mode → hide EVERYTHING except the essentials ──
    if (isNone) {
        const alwaysVisible = ["preset", "preset_info", "› Export Preset", "› Import Preset"];
        node.widgets.forEach(w => setWidgetVisible(w, alwaysVisible.includes(w.name), node));
        if (controlAfterW) setWidgetVisible(controlAfterW, false, node);
        refreshNodeSize(node);
        return;
    }

    // ── Custom: full manual control → show everything, only tile sub-options follow tile_mode ──
    if (isCustom) {
        node.widgets.forEach(w => {
            const hide = (w.name === "preset_info") || dummyWidgets.includes(w.name);
            setWidgetVisible(w, !hide, node);
        });
        if (controlAfterW) setWidgetVisible(controlAfterW, true, node);
        // ALBABIT-FIX: tile sub-options are meaningless when tile_mode is off — keep them
        // hidden in Custom too so the UI stays uncluttered. tile_mode callback triggers
        // toggleFields, so toggling tile_mode immediately shows/hides these.
        const tileModeW = find("tile_mode");
        const isTiled = tileModeW && tileModeW.value === true;
        setWidgetVisible(find("tile_size"),    isTiled, node);
        setWidgetVisible(find("tile_overlap"), isTiled, node);
        setWidgetVisible(find("tile_blend"),   isTiled, node);
        refreshNodeSize(node);
        return;
    }

    // ── Preset: show all by default, then apply smart, model-aware folding ──
    node.widgets.forEach(w => {
        const hide = dummyWidgets.includes(w.name);
        setWidgetVisible(w, !hide, node);
    });
    if (controlAfterW) setWidgetVisible(controlAfterW, true, node);

    const tileModeW = find("tile_mode");
    const restartCountW = find("restart_count");
    const aysScheduleW = find("ays_schedule");
    const modelTypeW = find("model_type");
    const samplerModeW = find("sampler_mode");

    // Check optional link states using node.inputs
    const hasRefinerModel = node.inputs && node.inputs.some(i => i.name === "refiner_model" && i.link !== null);

    const modelType = modelTypeW ? modelTypeW.value : "auto";
    const samplerMode = samplerModeW ? samplerModeW.value : "Standard";

    // Resolve the effective model so folding matches what the backend will run.
    const effectiveModel = resolveModelType(presetVal, modelType);
    const isLTX = LTX_MODEL_TYPES.has(effectiveModel) || LTX_PRESETS.includes(presetVal);

    // 3.1. Refiner: visible if refiner_model input port is wired up
    setWidgetVisible(find("refiner_start_step"), hasRefinerModel, node);

    // 3.2. Tiled latent sampling: visible if tile_mode is checked
    const isTiled = tileModeW && tileModeW.value === true;
    setWidgetVisible(find("tile_size"), isTiled, node);
    setWidgetVisible(find("tile_overlap"), isTiled, node);
    setWidgetVisible(find("tile_blend"), isTiled, node);

    // 3.3. Restart schedules: visible if restart_count > 0
    const hasRestartCount = restartCountW && parseInt(restartCountW.value, 10) > 0;
    setWidgetVisible(find("noise_alpha_start"), hasRestartCount, node);
    setWidgetVisible(find("noise_alpha_end"), hasRestartCount, node);

    // 3.4. Sigma blend steps: visible if Phase-Shift sampler_mode OR ays_schedule is active
    const isPhaseShift = samplerMode.includes("Phase-Shift");
    const isAys = aysScheduleW && aysScheduleW.value === true;
    setWidgetVisible(find("sigma_blend_steps"), isPhaseShift || isAys, node);

    // 3.5. Model-aware shift/guidance folding.
    //  - flux_shift (flow-match shift) is used by every modern flow model → show in preset mode.
    //  - flux_guidance / profile only apply to guidance-embed models; CFG-guided
    //    models (WAN, Hunyuan, SDXL, SD1.5/3/3.5, LTX-AV, CogVideoX, StepVideo) ignore them.
    const usesGuidanceEmbed =
        GUIDANCE_EMBED_MODELS.has(effectiveModel) ||
        (effectiveModel === "auto" && !CFG_GUIDED_MODELS.has(effectiveModel));
    setWidgetVisible(find("flux_shift"), true, node);
    setWidgetVisible(find("flux_guidance"), usesGuidanceEmbed, node);
    setWidgetVisible(find("flux_guidance_profile"), usesGuidanceEmbed, node);

    // 3.5b. LTX models can't use the flux-style guidance / tiling / preview widgets.
    if (isLTX) {
        LTX_INCOMPATIBLE_WIDGETS.forEach(name => setWidgetVisible(find(name), false, node));
    }

    // 3.6. Preset info text box is shown for named presets.
    setWidgetVisible(find("preset_info"), true, node);

    refreshNodeSize(node);
}

function updateUILocks(node, presetName) {
    if (!node.widgets) return;
    const isLTX = LTX_PRESETS.includes(presetName);
    const isCustom = presetName === "None" || presetName === "Custom";

    node.widgets.forEach((widget) => {
        if (widget.name === "preset" || widget.name === "preset_info") return;

        const wName = widget.name ? widget.name.toLowerCase() : "";
        const isTargetWidget = LTX_INCOMPATIBLE_WIDGETS.some(t => t.toLowerCase() === wName);

        if (isTargetWidget) {
            if (!isCustom && isLTX) {
                widget.disabled = true;
                if (widget.inputEl) {
                    widget.inputEl.disabled = true;
                    widget.inputEl.style.opacity = "0.4";
                    widget.inputEl.style.pointerEvents = "none";
                }
            } else {
                widget.disabled = false;
                if (widget.inputEl) {
                    widget.inputEl.disabled = false;
                    widget.inputEl.style.opacity = "1.0";
                    widget.inputEl.style.pointerEvents = "auto";
                }
            }
        }
    });

    node.setDirtyCanvas(true, true);
}

// ALBABIT-FIX: infer the model_type that best matches the preset name, used when
// the preset config doesn't explicitly include model_type. Mirrors resolveModelType
// so the UI widget reflects the model family the preset targets.
function inferModelTypeForPreset(presetName) {
    if (LTX_PRESETS.includes(presetName)) return "ltxav";
    const p = (presetName || "").toLowerCase();
    if (p.includes("ltx 2.3") || p.includes("ltxav")) return "ltxav";
    if (p.includes("ltx"))      return "ltxv";
    if (p.includes("wan"))      return "wan";
    if (p.includes("hunyuan"))  return "hunyuan_video";
    if (p.includes("z_image"))  return "z_image";
    if (p.includes("lumina"))   return "lumina2";
    if (p.includes("sd3.5") || p.includes("sd35")) return "sd35";
    if (p.includes("flux") || p.includes("draft") || p.includes("fast") ||
        p.includes("balanced") || p.includes("quality") || p.includes("cinema")) return "flux";
    return null;  // unknown, leave unchanged
}

function applyPreset(node, presetName) {
    if (presetName === "None" || presetName === "Custom") return;

    const config = getPresetConfig(presetName);
    if (!config) return;

    const widgets = node.widgets;
    if (!widgets) return;

    // Apply values silently without triggering loops
    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
        }
    }

    // ALBABIT-FIX: ensure model_type matches the preset's model family so
    // resolveModelType (which returns early if model_type != "auto") picks the
    // correct effective model and doesn't falsely flag Flux presets as isLTX.
    if (!config.model_type) {
        const inferred = inferModelTypeForPreset(presetName);
        if (inferred) {
            const modelTypeW = widgets.find(w => w.name === "model_type");
            if (modelTypeW) modelTypeW.value = inferred;
        }
    }

    node.setDirtyCanvas(true);
}

// Safely extract tracking values
function getTrackedState(node) {
    const state = {};
    if (!node.widgets) return state;
    const trackedFields = [
        "steps", "cfg", "sampler", "scheduler", "denoise",
        "flux_shift", "flux_guidance", "force_exact_steps",
        "terminal_sigma_to_zero"
    ];
    for (const w of node.widgets) {
        if (trackedFields.includes(w.name)) {
            state[w.name] = w.value;
        }
    }
    return state;
}

async function exportPreset(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    const exportFields = ["steps", "cfg", "sampler", "scheduler", "denoise", "flux_shift", "flux_guidance"];
    const settings = {};
    for (const widget of widgets) {
        if (exportFields.includes(widget.name)) {
            settings[widget.name] = widget.value;
        }
    }

    const presetName = await promptSamplerAction("Sampler Preset", "Enter a name for this sampler preset.", "My Sampler Preset", "Export");
    if (!presetName) return;

    const preset = {
        name: presetName,
        created: new Date().toISOString(),
        version: "1.0",
        settings,
    };

    const jsonStr = JSON.stringify(preset, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${presetName.replace(/[^a-z0-9]/gi, "_")}_sampler.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function showSamplerToast(message, tone = "info") {
    const toast = document.createElement("div");
    const toneColor = tone === "error" ? "#ff6b6b" : tone === "success" ? "#4cd964" : "#00a8ff";
    Object.assign(toast.style, {
        position: "fixed",
        left: "50%",
        bottom: "24px",
        zIndex: "10000",
        transform: "translateX(-50%) translateY(12px)",
        opacity: "0",
        maxWidth: "420px",
        padding: "10px 14px",
        color: "#f5f5f7",
        background: "rgba(18, 18, 24, 0.94)",
        border: `1px solid ${toneColor}55`,
        borderRadius: "8px",
        boxShadow: "0 12px 36px rgba(0,0,0,0.45)",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        fontSize: "12px",
        lineHeight: "1.35",
        pointerEvents: "none",
        transition: "opacity 160ms ease, transform 160ms ease",
    });
    toast.textContent = message;
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
        toast.style.opacity = "1";
        toast.style.transform = "translateX(-50%) translateY(0)";
    });

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(-50%) translateY(12px)";
        setTimeout(() => toast.remove(), 180);
    }, 3200);
}

function promptSamplerAction(titleText, message, defaultValue = "", confirmLabel = "Continue") {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        Object.assign(overlay.style, {
            position: "fixed",
            inset: "0",
            zIndex: "10001",
            display: "grid",
            placeItems: "center",
            background: "rgba(0,0,0,0.55)",
            backdropFilter: "blur(8px)",
        });

        const dialog = document.createElement("div");
        Object.assign(dialog.style, {
            width: "min(420px, calc(100vw - 32px))",
            padding: "18px",
            color: "#f5f5f7",
            background: "rgba(18,18,24,0.96)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "8px",
            boxShadow: "0 18px 60px rgba(0,0,0,0.65)",
            fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            fontSize: "13px",
        });

        const title = document.createElement("div");
        title.textContent = titleText;
        title.style.cssText = "font-weight:700;font-size:15px;margin-bottom:8px;";

        const copy = document.createElement("div");
        copy.textContent = message;
        copy.style.cssText = "color:#b8c0cc;margin-bottom:12px;line-height:1.45;";

        const input = document.createElement("input");
        input.type = "text";
        input.value = defaultValue;
        input.style.cssText = "width:100%;box-sizing:border-box;height:36px;margin-bottom:16px;border-radius:8px;border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.06);color:#f5f5f7;padding:0 10px;outline:none;";

        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;gap:10px;justify-content:flex-end;";

        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.textContent = "Cancel";
        cancel.style.cssText = "height:32px;padding:0 12px;border-radius:6px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.05);color:#f5f5f7;cursor:pointer;";

        const confirm = document.createElement("button");
        confirm.type = "button";
        confirm.textContent = confirmLabel;
        confirm.style.cssText = "height:32px;padding:0 12px;border-radius:6px;border:1px solid rgba(0,168,255,0.45);background:rgba(0,168,255,0.16);color:#9fdcff;cursor:pointer;font-weight:700;";

        const close = (value) => {
            overlay.remove();
            resolve(value);
        };

        cancel.addEventListener("click", () => close(null));
        confirm.addEventListener("click", () => close(input.value.trim()));
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") close(input.value.trim());
            if (event.key === "Escape") close(null);
        });
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) close(null);
        });

        actions.append(cancel, confirm);
        dialog.append(title, copy, input, actions);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        input.focus();
        input.select();
    });
}

function importPreset(node) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    fileInput.onchange = (e) => {
        document.body.removeChild(fileInput);

        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const preset = JSON.parse(event.target.result);

                if (!preset.settings || typeof preset.settings !== "object") {
                    showSamplerToast("Invalid preset file: missing settings object.", "error");
                    return;
                }

                const widgets = node.widgets;
                let appliedCount = 0;

                for (const widget of widgets) {
                    if (preset.settings[widget.name] !== undefined) {
                        widget.value = preset.settings[widget.name];
                        appliedCount++;
                    }
                }

                const presetWidget = widgets.find(w => w.name === "preset");
                if (presetWidget) {
                    presetWidget.value = "Custom";
                    updateUILocks(node, "Custom");
                }

                node.setDirtyCanvas(true);
                showSamplerToast(`Preset "${preset.name || "Unnamed"}" imported. ${appliedCount} settings applied.`, "success");

            } catch (error) {
                console.error("[Radiance Sampler] Failed to import preset:", error);
                showSamplerToast(`Failed to import preset: ${error.message}`, "error");
            }
        };

        reader.readAsText(file);
    };

    fileInput.click();
}

app.registerExtension({
    name: "FXTD.RadianceSampler",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceSamplerPro") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onPropertyChanged = nodeType.prototype.onPropertyChanged;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const self = this;

            this.addWidget("button", "› Export Preset", null, () => exportPreset(this), { serialize: false });
            this.addWidget("button", "› Import Preset", null, () => importPreset(this), { serialize: false });

            const presetWidget = this.widgets?.find(w => w.name === "preset");
            if (!presetWidget) return;

            let descWidget = this.widgets?.find(w => w.name === "preset_info");
            if (!descWidget) {
                descWidget = this.addWidget("text", "preset_info", "", () => { }, {
                    multiline: true,
                    serialize: false
                });
            }

            setTimeout(() => {
                if (descWidget && descWidget.inputEl) {
                    descWidget.inputEl.style.fontFamily = "monospace";
                    descWidget.inputEl.style.fontSize = "11px";
                    descWidget.inputEl.style.color = "#888";
                    descWidget.inputEl.style.fontStyle = "italic";
                    descWidget.inputEl.style.height = "60px";
                    descWidget.inputEl.readOnly = true;
                }
            }, 100);

            const updateDescription = (presetName) => {
                const config = getPresetConfig(presetName);
                let text = "Manual / Custom Mode. All widgets are unlocked.";
                if (presetName === "None") {
                    text = "Default simple mode. Parameters are hidden and use standard defaults. Select 'Custom' to manually tweak settings.";
                } else if (config && config.description) {
                    text = config.description;
                }

                if (descWidget) {
                    descWidget.value = text;
                    if (descWidget.inputEl) {
                        descWidget.inputEl.value = text;
                    }
                }
            };

            let lastPresetValue = presetWidget.value;

            // Handle Preset changes explicitly
            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(presetWidget, value);

                if (window.app && window.app.configuringGraph) return;

                if (value !== lastPresetValue && value !== "None" && value !== "Custom") {
                    lastPresetValue = value;
                    applyPreset(this, value);
                    updateUILocks(this, value);
                    updateDescription(value);
                    toggleFields(this);
                } else if (value !== lastPresetValue) {
                    lastPresetValue = value;
                    // ALBABIT-FIX: reset model_type to "auto" on manual switch to None/Custom
                    // (named presets set it via applyPreset/inferModelTypeForPreset above)
                    if (value === "None" || value === "Custom") {
                        const modelTypeW = this.widgets?.find(w => w.name === "model_type");
                        if (modelTypeW) modelTypeW.value = "auto";
                    }
                    updateUILocks(this, value);
                    updateDescription(value);
                    toggleFields(this);
                }
            };

            // Hook manual widget changes to toggle Custom and refresh fields
            this.onPropertyChanged = function (property, value, prevValue) {
                if (onPropertyChanged) onPropertyChanged.apply(this, arguments);

                // Ignore backend-only or system properties
                if (window.app && window.app.configuringGraph) return;

                const pWidget = this.widgets?.find(wd => wd.name === "preset");
                if (!pWidget || pWidget.value === "Custom" || pWidget.value === "None") return;

                // If it's a property managed by the preset, verify if it diverges
                const currentPreset = getPresetConfig(pWidget.value);
                if (currentPreset && currentPreset[property] !== undefined) {
                    if (currentPreset[property] != value) {
                        console.log(`[Radiance Sampler] Manual override detected on '${property}'. Switching to Custom.`);
                        pWidget.value = "Custom";
                        lastPresetValue = "Custom";
                        // ALBABIT-FIX: reset model_type to "auto" on auto-switch to Custom
                        const modelTypeW = this.widgets?.find(w => w.name === "model_type");
                        if (modelTypeW) modelTypeW.value = "auto";
                        updateUILocks(this, "Custom");
                        updateDescription("Custom");
                        toggleFields(this);
                        this.setDirtyCanvas(true);
                    }
                }
            };

            // Wire up callbacks for dynamic folding on change
            const foldTriggers = ["preset", "tile_mode", "restart_count", "ays_schedule", "model_type", "sampler_mode"];
            foldTriggers.forEach(name => {
                const w = self.widgets?.find(x => x.name === name);
                if (w) {
                    const origCallback = w.callback;
                    w.callback = function(...args) {
                        const res = origCallback ? origCallback.apply(this, args) : undefined;
                        toggleFields(self);
                        return res;
                    };
                }
            });

            // Hook connection change events (optional ports linked/unlinked)
            const origConnect = this.onConnectionsChange;
            this.onConnectionsChange = function (...args) {
                if (origConnect) origConnect.apply(this, args);
                toggleFields(this);
            };

            // Initialize UI state immediately (safe — no widget visibility changes)
            const val = presetWidget.value;
            if (val) {
                lastPresetValue = val;
                updateUILocks(this, val);
                updateDescription(val);
            }

            // ALBABIT-FIX: _configuredByLoad is set in onConfigure (loaded workflow).
            // For a loaded node, onConfigure fires right after onNodeCreated with correct values;
            // its 150ms timer handles initial folding so we skip this one to avoid a race where
            // this fires while configure() is still running (large workflows take > 100ms).
            // For a freshly added node, onConfigure never fires, so this timer is the only one.
            const nodeRef = this;
            setTimeout(() => {
                if (nodeRef._configuredByLoad) return;
                const val = presetWidget.value;
                if (val) {
                    lastPresetValue = val;
                    updateUILocks(nodeRef, val);
                    updateDescription(val);
                }
                toggleFields(nodeRef);
            }, 150);
        };

        // Re-apply folding after a saved workflow restores this node. onNodeCreated
        // runs BEFORE ComfyUI deserializes widget values, so the preset value isn't
        // known at creation time; without this hook a node saved with a non-None
        // preset (or in Custom mode) could load stuck-collapsed.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            // ALBABIT-FIX: signal to the onNodeCreated timer that onConfigure ran,
            // so the timer skips and avoids a potential race with in-progress configuration.
            self._configuredByLoad = true;
            const reapply = () => {
                const presetW = self.widgets?.find(w => w.name === "preset");
                if (presetW) updateUILocks(self, presetW.value);
                toggleFields(self);
            };
            // ALBABIT-FIX: 150ms for Vue's first layout pass; 600ms safety net for
            // heavy workflows where graph.configure() stalls the main thread > 100ms.
            setTimeout(reapply, 150);
            setTimeout(reapply, 600);
        };
    }
});

console.log("[Radiance Sampler] Extension loaded");
