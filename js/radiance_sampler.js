import { app } from "../../scripts/app.js";

// Radiance Sampler Preset Configurations (synced with nodes_sampler.py)
const PRESET_CONFIGS = {
    "→ Flux txt2img": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
        "description": "Standard Flux text-to-image. Optimal for 1024x1024 images.",
    },
    "→ Flux img2img": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.75,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
        "description": "Image-to-image refinement with denoise=0.75 for balanced changes.",
    },
    "→ Flux Inpaint": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
        "description": "Inpainting with higher guidance (4.0) for detail matching.",
    },
    "→ Flux High-Res Fix": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.5,
        "flux_shift": 3.0,
        "flux_guidance": 3.5,
        "description": "2x upscale with shift=3.0 for enhanced detail. Magic for upscaling!",
    },
    "→ Flux Fast (12 steps)": {
        "steps": 12,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
        "description": "Quick generation for testing prompts and iteration.",
    },
    "→ Flux Quality (28 steps)": {
        "steps": 28,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
        "description": "Maximum quality for final outputs. Higher guidance for adherence.",
    },
    "→ Flux Cinematic (30 steps)": {
        "steps": 30,
        "cfg": 1.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 1.0,
        "flux_shift": 2.0,
        "flux_guidance": 4.5,
        "description": "Film-grade with enhanced detail (shift=2.0) and strong guidance.",
    },
};

// Apply preset to node widgets
function applyPreset(node, presetName) {
    if (presetName === "None (Custom)") return;

    if (!PRESET_CONFIGS[presetName]) {
        console.warn(`[Radiance Sampler] Preset "${presetName}" not found`);
        return;
    }

    const config = PRESET_CONFIGS[presetName];
    const widgets = node.widgets;
    if (!widgets) return;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
            if (widget.callback) {
                widget.callback(config[widget.name]);
            }
        }
    }

    node.setDirtyCanvas(true);
}

// Export current settings as preset
function exportPreset(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    const preset = {
        name: "Custom Sampler Preset",
        created: new Date().toISOString(),
        version: "1.0",
        settings: {}
    };

    const exportFields = ["steps", "cfg", "sampler", "scheduler", "denoise", "flux_shift", "flux_guidance"];

    for (const widget of widgets) {
        if (exportFields.includes(widget.name)) {
            preset.settings[widget.name] = widget.value;
        }
    }

    const presetName = prompt("Enter preset name:", "My Sampler Preset");
    if (!presetName) return;

    preset.name = presetName;

    const jsonStr = JSON.stringify(preset, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${presetName.replace(/[^a-z0-9]/gi, '_')}_sampler.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[Radiance Sampler] Preset "${presetName}" exported`);
}

// Import preset from file
function importPreset(node) {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json';
    fileInput.style.display = 'none';

    fileInput.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const preset = JSON.parse(event.target.result);

                if (!preset.settings || typeof preset.settings !== 'object') {
                    alert('Invalid preset file: Missing settings object');
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
                    presetWidget.value = "None (Custom)";
                }

                node.setDirtyCanvas(true);

                console.log(`[Radiance Sampler] Preset "${preset.name || 'Unnamed'}" imported (${appliedCount} settings)`);
                alert(`Preset "${preset.name || 'Unnamed'}" imported!\n${appliedCount} settings applied.`);

            } catch (error) {
                console.error('[Radiance Sampler] Failed to import preset:', error);
                alert('Failed to import preset: ' + error.message);
            }
        };

        reader.readAsText(file);
    };

    document.body.appendChild(fileInput);
    fileInput.click();
    document.body.removeChild(fileInput);
}

// Register the extension
app.registerExtension({
    name: "FXTD.RadianceSampler",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only process our target node
        if (nodeData.name !== "FXTD_Radiance_Sampler_Pro") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            // Add Export and Import buttons
            this.addWidget("button", "› Export Preset", null, () => {
                exportPreset(this);
            });

            this.addWidget("button", "› Import Preset", null, () => {
                importPreset(this);
            });

            // Find the preset widget
            const presetWidget = this.widgets?.find(w => w.name === "preset");
            if (!presetWidget) {
                console.warn("[Radiance Sampler] preset widget not found");
                return;
            }

            // Add description display
            const descWidget = this.addWidget("text", "preset_info", "", () => { }, {
                multiline: true,
            });

            // Make description read-only
            setTimeout(() => {
                if (descWidget.inputEl) {
                    descWidget.inputEl.style.fontFamily = "monospace";
                    descWidget.inputEl.style.fontSize = "11px";
                    descWidget.inputEl.style.color = "#888";
                    descWidget.inputEl.style.fontStyle = "italic";
                    descWidget.inputEl.readOnly = true;
                }
            }, 100);

            // Update description when preset changes
            const updateDescription = (presetName) => {
                const config = PRESET_CONFIGS[presetName];
                if (config && config.description) {
                    descWidget.value = config.description;
                    if (descWidget.inputEl) {
                        descWidget.inputEl.value = config.description;
                    }
                } else {
                    descWidget.value = "";
                    if (descWidget.inputEl) {
                        descWidget.inputEl.value = "";
                    }
                }
            };

            // Handle preset changes
            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) {
                    originalCallback.call(presetWidget, value);
                }
                applyPreset(this, value);
                updateDescription(value);
            };

            // Apply initial preset on node creation
            setTimeout(() => {
                if (presetWidget.value && presetWidget.value !== "None (Custom)") {
                    applyPreset(this, presetWidget.value);
                    updateDescription(presetWidget.value);
                }
            }, 100);
        };
    }
});

console.log("[Radiance Sampler] Extension loaded");
