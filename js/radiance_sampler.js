import { app } from "../../../scripts/app.js";
// FIX 2: was "../../scripts/app.js" — extensions in custom_nodes/Radiance/web/
// need three ../ to reach ComfyUI's scripts/ directory.

/**
 * Radiance Sampler Pro — Widget Management (v2.3)
 *
 * FIX 1: Node name corrected to "RadianceSamplerPro" (matches NODE_CLASS_MAPPINGS
 *         in nodes_sampler.py). Was "FXTD_Radiance_Sampler_Pro" — extension was dead.
 * FIX 2: Import path fixed (see above).
 * FIX 3: PRESET_CONFIGS synced with nodes_sampler.py PRESET_CONFIGS. All missing
 *         presets added; stale values corrected.
 * FIX 4: fileInput removed INSIDE onchange after processing, not before the dialog.
 * FIX 5: URL.revokeObjectURL deferred via setTimeout to avoid cancelling the download.
 * FIX 6: preset_info uses a "note"-style div overlay instead of a serialized widget,
 *         preventing duplicate fields on workflow restore.
 */

// FIX 3: Full preset table — synced exactly with nodes_sampler.py PRESET_CONFIGS.
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
    "▶ LTX 2.3 LowRes (32 steps)": {
        steps: 32, cfg: 3.0, sampler: "euler", scheduler: "beta",
        denoise: 1.0, flux_shift: 3.0, flux_guidance: 0.0,
        description: "LTX 2.3 low-res pass. Beta scheduler, shift=3.",
    },
    "▶ LTX 2.3 HighRes (40 steps)": {
        steps: 40, cfg: 3.0, sampler: "euler", scheduler: "beta",
        denoise: 0.45, flux_shift: 6.0, flux_guidance: 0.0,
        description: "LTX 2.3 high-res refinement pass. denoise=0.45, shift=6.",
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

// Apply preset to node widgets
function applyPreset(node, presetName) {
    if (presetName === "None (Custom)") return;

    const config = PRESET_CONFIGS[presetName];
    if (!config) {
        console.warn(`[Radiance Sampler] Preset "${presetName}" not in JS table — skipping widget update.`);
        return;
    }

    const widgets = node.widgets;
    if (!widgets) return;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
            if (widget.callback) widget.callback(config[widget.name]);
        }
    }

    node.setDirtyCanvas(true);
}

// Export current settings as preset JSON
function exportPreset(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    const exportFields = ["steps", "cfg", "sampler", "scheduler", "denoise", "flux_shift", "flux_guidance"];
    const settings = {};
    for (const widget of widgets) {
        if (exportFields.includes(widget.name)) {
            settings[widget.name] = widget.value;
        }
    }

    const presetName = prompt("Enter preset name:", "My Sampler Preset");
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

    // FIX 5: Defer revoke — synchronous revoke cancels the download on some browsers
    // before the browser has read the blob data.
    setTimeout(() => URL.revokeObjectURL(url), 1000);

    console.log(`[Radiance Sampler] Preset "${presetName}" exported`);
}

// Import preset from JSON file
function importPreset(node) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    fileInput.onchange = (e) => {
        // FIX 4: Remove input INSIDE onchange after processing — not before the
        // dialog closes. Removing it before onchange fires cancels the dialog on
        // Firefox and Safari.
        document.body.removeChild(fileInput);

        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const preset = JSON.parse(event.target.result);

                if (!preset.settings || typeof preset.settings !== "object") {
                    alert("Invalid preset file: Missing settings object");
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
                if (presetWidget) presetWidget.value = "None (Custom)";

                node.setDirtyCanvas(true);
                console.log(`[Radiance Sampler] Preset "${preset.name || "Unnamed"}" imported (${appliedCount} settings)`);
                alert(`Preset "${preset.name || "Unnamed"}" imported!\n${appliedCount} settings applied.`);

            } catch (error) {
                console.error("[Radiance Sampler] Failed to import preset:", error);
                alert("Failed to import preset: " + error.message);
            }
        };

        reader.readAsText(file);
    };

    fileInput.click();
}

// Register the extension
app.registerExtension({
    name: "FXTD.RadianceSampler",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // FIX 1: was "FXTD_Radiance_Sampler_Pro" — correct name is "RadianceSamplerPro"
        // as registered in nodes_sampler.py NODE_CLASS_MAPPINGS.
        if (nodeData.name !== "RadianceSamplerPro") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            // Export / Import buttons
            this.addWidget("button", "› Export Preset", null, () => exportPreset(this));
            this.addWidget("button", "› Import Preset", null, () => importPreset(this));

            const presetWidget = this.widgets?.find(w => w.name === "preset");
            if (!presetWidget) {
                console.warn("[Radiance Sampler] preset widget not found");
                return;
            }

            // FIX 6: Use a DOM overlay for the description instead of a serialized
            // text widget. ComfyUI serializes every widget added via addWidget() into
            // the workflow JSON. On restore it creates a second "preset_info" widget
            // on top of the first — duplicate read-only fields stack up on each open.
            // A DOM label element attached to the canvas node avoids serialization.
            let descEl = null;

            const updateDescription = (presetName) => {
                const config = PRESET_CONFIGS[presetName];
                const text = (config && config.description) ? config.description : "";

                if (!descEl) {
                    // Lazy-create a floating label anchored near the node
                    descEl = document.createElement("div");
                    descEl.style.cssText = [
                        "position:absolute",
                        "pointer-events:none",
                        "font:italic 11px/1.4 monospace",
                        "color:#aaa",
                        "background:rgba(0,0,0,.45)",
                        "padding:3px 6px",
                        "border-radius:3px",
                        "max-width:260px",
                        "white-space:pre-wrap",
                        "z-index:10",
                    ].join(";");
                    // Attach to the ComfyUI canvas container
                    const container = document.querySelector("#graph-canvas")?.parentElement
                        || document.body;
                    container.appendChild(descEl);

                    // Remove element when node is removed
                    const origOnRemoved = this.onRemoved?.bind(this);
                    this.onRemoved = () => {
                        if (descEl && descEl.parentNode) descEl.parentNode.removeChild(descEl);
                        descEl = null;
                        if (origOnRemoved) origOnRemoved();
                    };
                }

                descEl.textContent = text;
                descEl.style.display = text ? "block" : "none";

                // Position below the node using LiteGraph bounding box
                const canvas = app.canvas;
                if (canvas && this.pos && this.size) {
                    const [nx, ny] = canvas.convertOffsetToCanvas
                        ? canvas.convertOffsetToCanvas([this.pos[0], this.pos[1] + this.size[1] + 4])
                        : [this.pos[0], this.pos[1] + this.size[1] + 4];
                    descEl.style.left = nx + "px";
                    descEl.style.top  = ny + "px";
                }
            };

            // Intercept preset changes
            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(presetWidget, value);
                applyPreset(this, value);
                updateDescription(value);
            };

            // Initialise on creation
            setTimeout(() => {
                const val = presetWidget.value;
                if (val && val !== "None (Custom)") {
                    applyPreset(this, val);
                    updateDescription(val);
                }
            }, 100);
        };
    }
});

console.log("[Radiance Sampler] Extension loaded");
