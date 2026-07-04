import { app } from "../../scripts/app.js";

// ALBABIT-FIX: apply_style_preset() used to overwrite these 7 widgets on
// every execution, not just on selection (same bug class as the Sampler's
// _apply_presets — see radiance_sampler.js). Python now respects the live
// widget values; this file fills them once on selection and flags a later
// edit with a "✎" marker. film_stock/shutter_speed/aspect_ratio have no
// widget here, so Python keeps applying the preset for those.
const PRESET_CONFIGS = {
    "→ Classic Hollywood": {
        framing: "Medium Shot (MS)",
        camera_type: "Panavision Panaflex Gold II (35mm)",
        lens_focal: "50mm Standard Prime",
        aperture_dof: "f/2.8 (Cinematic Separation)",
        lighting: "Paramount Lighting",
        style_aesthetic: "Cinematic Movie Still",
        color_grading: "Technicolor (Vintage)",
    },
    "→ Film Noir": {
        framing: "Low Angle (Hero Shot)",
        camera_type: "ARRI Alexa 35",
        lens_focal: "35mm Classic Wide",
        aperture_dof: "f/2.8 (Cinematic Separation)",
        lighting: "Film Noir Lighting",
        style_aesthetic: "Monochrome Noir",
        color_grading: "Bleach Bypass (Gritty)",
    },
    "→ Sci-Fi Cinematic": {
        framing: "Extreme Wide Shot (EWS)",
        camera_type: "ARRI Alexa 65 (IMAX)",
        lens_focal: "ARRI Master Anamorphic",
        aperture_dof: "f/4.0 (Balanced)",
        lighting: "Cinematic Haze / Volumetric Fog",
        style_aesthetic: "Blade Runner Atmosphere",
        color_grading: "Teal and Orange (Blockbuster)",
    },
    "→ Cyberpunk": {
        framing: "Dutch Angle (Canted)",
        camera_type: "Sony Venice 2",
        lens_focal: "Anamorphic Lens",
        aperture_dof: "f/1.8 (Soft Background)",
        lighting: "Neon Cyberpunk Lighting",
        style_aesthetic: "Cyberpunk 2077 Aesthetic",
        color_grading: "Cyberpunk Neon Grading",
    },
    "→ Drama / Emotional": {
        framing: "Close-Up (CU)",
        camera_type: "ARRI Alexa Mini LF",
        lens_focal: "85mm Portrait Prime",
        aperture_dof: "f/1.2 (Dreamy Bokeh)",
        lighting: "Rembrandt Lighting",
        style_aesthetic: "Cinematic Movie Still",
        color_grading: "Desaturated (Muted)",
    },
    "→ Epic Landscape": {
        framing: "Extreme Wide Shot (EWS)",
        camera_type: "ARRI Alexa 65 (IMAX)",
        lens_focal: "14mm Ultra-Wide Angle",
        aperture_dof: "f/11 (Landscape Sharpness)",
        lighting: "Golden Hour (Magic Hour)",
        style_aesthetic: "National Geographic Style",
        color_grading: "Vibrant High Contrast",
    },
    "→ Portrait": {
        framing: "Medium Close-Up (MCU)",
        camera_type: "Sony A7S III",
        lens_focal: "85mm Portrait Prime",
        aperture_dof: "f/1.2 (Dreamy Bokeh)",
        lighting: "Soft Window Light",
        style_aesthetic: "Editorial Photography",
        color_grading: "Pastel Soft Tones",
    },
    "→ Documentary": {
        framing: "Medium Shot (MS)",
        camera_type: "Canon C700 FF",
        lens_focal: "35mm Classic Wide",
        aperture_dof: "f/4.0 (Balanced)",
        lighting: "Practical Lighting",
        style_aesthetic: "Documentary Texture",
        color_grading: "Desaturated (Muted)",
    },
    "→ Artistic / Painterly": {
        framing: "Medium Shot (MS)",
        camera_type: "None",
        lens_focal: "Petzval 85mm (Classic Swirl)",
        aperture_dof: "f/1.8 (Soft Background)",
        lighting: "Soft Window Light",
        style_aesthetic: "Oil Painting (Classic)",
        color_grading: "Pastel Soft Tones",
    },
    "→ Retro VHS": {
        framing: "Medium Shot (MS)",
        camera_type: "Super 8mm Camera",
        lens_focal: "50mm Standard Prime",
        aperture_dof: "f/4.0 (Balanced)",
        lighting: "Practical Lighting",
        style_aesthetic: "Vintage 1990s VHS",
        color_grading: "Cross Processed",
    },
    "→ Golden Hour Magic": {
        framing: "Full Body Shot (Wide)",
        camera_type: "Sony Venice 2",
        lens_focal: "85mm Portrait Prime",
        aperture_dof: "f/1.8 (Soft Background)",
        lighting: "Golden Hour (Magic Hour)",
        style_aesthetic: "Photorealistic (Raw)",
        color_grading: "Vibrant High Contrast",
    },
    "→ Moody Night": {
        framing: "Medium Shot (MS)",
        camera_type: "Sony A7S III",
        lens_focal: "35mm Classic Wide",
        aperture_dof: "f/1.2 (Dreamy Bokeh)",
        lighting: "Moonlight",
        style_aesthetic: "Cinematic Movie Still",
        color_grading: "Teal and Orange (Blockbuster)",
    },
    "→ Action / Dynamic": {
        framing: "Low Angle (Hero Shot)",
        camera_type: "RED V-Raptor XL",
        lens_focal: "24mm Wide Angle",
        aperture_dof: "f/5.6 (Sharp Subject)",
        lighting: "Harsh Sunlight",
        style_aesthetic: "Hyper-Realism",
        color_grading: "Teal and Orange (Blockbuster)",
    },
    "→ Wes Anderson": {
        framing: "Symmetrical Composition",
        camera_type: "ARRI Alexa 35",
        lens_focal: "35mm Classic Wide",
        aperture_dof: "f/8.0 (Deep Focus)",
        lighting: "Soft Window Light",
        style_aesthetic: "Wes Anderson Symmetric",
        color_grading: "Pastel Soft Tones",
    },
    "→ 1970s New Hollywood": {
        framing: "Medium Shot (MS)",
        camera_type: "Panavision Panaflex Gold II (35mm)",
        lens_focal: "35mm Classic Wide",
        aperture_dof: "f/2.8 (Cinematic Separation)",
        lighting: "Practical Lighting",
        style_aesthetic: "Cinematic Movie Still",
        color_grading: "Technicolor (Vintage)",
    },
    "→ 1980s Retro Action": {
        framing: "Low Angle (Hero Shot)",
        camera_type: "ARRI Alexa 35",
        lens_focal: "Anamorphic Lens",
        aperture_dof: "f/4.0 (Balanced)",
        lighting: "Cinematic Haze / Volumetric Fog",
        style_aesthetic: "Hyper-Realism",
        color_grading: "Teal and Orange (Blockbuster)",
    },
    "→ 1990s Music Video": {
        framing: "Extreme Close-Up (ECU)",
        camera_type: "Super 8mm Camera",
        lens_focal: "Fish-Eye Lens",
        aperture_dof: "f/1.8 (Soft Background)",
        lighting: "Neon Cyberpunk Lighting",
        style_aesthetic: "Vintage 1990s VHS",
        color_grading: "Cross Processed",
    },
    "→ 2000s Digital Look": {
        framing: "Medium Shot (MS)",
        camera_type: "Sony A7S III",
        lens_focal: "24mm Wide Angle",
        aperture_dof: "f/5.6 (Sharp Subject)",
        lighting: "Harsh Sunlight",
        style_aesthetic: "Editorial Photography",
        color_grading: "Vibrant High Contrast",
    },
    "→ Horror / Thriller": {
        framing: "Low Angle (Hero Shot)",
        camera_type: "ARRI Alexa Mini LF",
        lens_focal: "16mm Ultra-Wide Angle",
        aperture_dof: "f/2.8 (Cinematic Separation)",
        lighting: "Chiaroscuro (High Contrast)",
        style_aesthetic: "Film Noir Aesthetic",
        color_grading: "Bleach Bypass (Gritty)",
    },
    "→ Romance / Soft Focus": {
        framing: "Medium Close-Up (MCU)",
        camera_type: "Sony A7S III",
        lens_focal: "50mm Standard Prime",
        aperture_dof: "f/1.2 (Dreamy Bokeh)",
        lighting: "Soft Window Light",
        style_aesthetic: "Dreamy Soft Focus",
        color_grading: "Pastel Soft Tones",
    },
    "→ Christopher Nolan": {
        framing: "Wide Shot (WS)",
        camera_type: "IMAX 15/70mm Film Camera",
        lens_focal: "28mm Wide Angle",
        aperture_dof: "f/8.0 (Deep Focus)",
        lighting: "Natural Ambient Light",
        style_aesthetic: "Photorealistic (Raw)",
        color_grading: "Neutral ACES Workflow",
    },
    "→ Denis Villeneuve": {
        framing: "Extreme Wide Shot (EWS)",
        camera_type: "ARRI Alexa 65 (IMAX)",
        lens_focal: "24mm Wide Angle",
        aperture_dof: "f/4.0 (Balanced)",
        lighting: "Volumetric Fog / Atmospheric Haze",
        style_aesthetic: "Atmospheric Cinematic",
        color_grading: "Desaturated Cool Tones",
    },
    "→ Quentin Tarantino": {
        framing: "Medium Shot (MS)",
        camera_type: "Panavision Panaflex Gold II (35mm)",
        lens_focal: "40mm Semi-Wide",
        aperture_dof: "f/2.8 (Cinematic Separation)",
        lighting: "High-Key Lighting",
        style_aesthetic: "Vintage Kodachrome Look",
        color_grading: "Vibrant High Contrast",
    },
    "→ Prestige Drama (HBO)": {
        framing: "Medium Shot (MS)",
        camera_type: "ARRI Alexa Mini LF",
        lens_focal: "35mm Classic Wide",
        aperture_dof: "f/2.0 (Shallow Cinematic)",
        lighting: "Naturalistic Interior Light",
        style_aesthetic: "Cinematic TV Drama",
        color_grading: "Moody Shadows",
    },
    "→ True Crime Documentary": {
        framing: "Medium Close-Up (MCU)",
        camera_type: "Canon C300 Mark III",
        lens_focal: "50mm Standard Prime",
        aperture_dof: "f/2.8 (Cinematic Separation)",
        lighting: "Interview 3-Point Setup",
        style_aesthetic: "Documentary Realism",
        color_grading: "Desaturated Cool Tones",
    },
    "→ Western": {
        framing: "Medium Full Shot (MFS)",
        camera_type: "Panavision Panaflex Gold II (35mm)",
        lens_focal: "35mm Classic Wide",
        aperture_dof: "f/5.6 (Sharp Subject)",
        lighting: "Harsh Sunlight",
        style_aesthetic: "Vintage Kodachrome Look",
        color_grading: "Orange and Teal (Hollywood)",
    },
    "→ Mystery / Detective": {
        framing: "Over-The-Shoulder (OTS)",
        camera_type: "ARRI Alexa 35",
        lens_focal: "50mm Standard Prime",
        aperture_dof: "f/2.8 (Cinematic Separation)",
        lighting: "Film Noir Lighting",
        style_aesthetic: "Neo-Noir Modern",
        color_grading: "Desaturated Cool Tones",
    },
    "→ Fantasy / Magical": {
        framing: "Medium Wide (MW)",
        camera_type: "ARRI Alexa 65 (IMAX)",
        lens_focal: "Anamorphic Lens",
        aperture_dof: "f/2.0 (Shallow Cinematic)",
        lighting: "God Rays (Crepuscular Rays)",
        style_aesthetic: "Magical Realism",
        color_grading: "Technicolor (Vintage)",
    },
};

const PRESET_MARKER = " ✎";
const TRACKED_FIELDS = [
    "framing", "camera_type", "lens_focal", "aperture_dof",
    "lighting", "style_aesthetic", "color_grading",
];

function getPresetConfig(name) {
    return PRESET_CONFIGS[name] || null;
}

function applyPreset(node, presetName) {
    const config = getPresetConfig(presetName);
    if (!config) return;

    const widgets = node.widgets;
    if (!widgets) return;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
        }
    }

    // Widgets now match the preset again — clear any "✎" markers.
    updatePresetDivergenceMarkers(node);
    node.setDirtyCanvas(true);
}

// Every tracked dataset (FRAMING/CAMERAS/LENSES/APERTURES/LIGHTING/STYLES/
// COLOR_GRADING) has "None" as its first, valid entry — switching to
// "None (Custom)" blanks all 7 style widgets back to it, same as selecting
// a named preset overwrites them with that preset's values.
function resetToCustomDefaults(node) {
    const widgets = node.widgets;
    if (!widgets) return;
    for (const widget of widgets) {
        if (TRACKED_FIELDS.includes(widget.name)) {
            widget.value = "None";
        }
    }
    updatePresetDivergenceMarkers(node);
    node.setDirtyCanvas(true);
}

function updatePresetDivergenceMarkers(node) {
    if (!node.widgets) return;
    const presetW = node.widgets.find(w => w.name === "style_preset");
    const presetVal = presetW ? presetW.value : "None (Custom)";
    const config = presetVal === "None (Custom)" ? null : getPresetConfig(presetVal);

    let changed = false;
    for (const w of node.widgets) {
        if (!w || !w.name || !TRACKED_FIELDS.includes(w.name)) continue;
        let marked = false;
        if (config && config[w.name] !== undefined) {
            marked = String(w.value) !== String(config[w.name]);
        }
        if (w._radOrigLabel === undefined && !marked) continue;
        if (w._radOrigLabel === undefined) w._radOrigLabel = w.label ?? w.name;
        const wanted = marked ? w._radOrigLabel + PRESET_MARKER : w._radOrigLabel;
        if (w.label !== wanted) {
            w.label = wanted;
            changed = true;
        }
    }
    if (changed) node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "FXTD.RadianceCinematicPromptEncoder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceCinematicPromptEncoder") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const self = this;
            const presetWidget = this.widgets?.find(w => w.name === "style_preset");
            if (!presetWidget) return;

            let lastPresetValue = presetWidget.value;

            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(presetWidget, value);
                if (window.app && window.app.configuringGraph) return;
                if (value !== lastPresetValue) {
                    if (value === "None (Custom)") {
                        resetToCustomDefaults(this);
                    } else {
                        applyPreset(this, value);
                    }
                }
                lastPresetValue = value;
            };

            // Poll for manual widget edits, undo/redo — same state-based
            // approach as js/radiance_sampler.js's marker refresh.
            this._presetMarkerInterval = setInterval(() => updatePresetDivergenceMarkers(self), 250);
            const origOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (self._presetMarkerInterval) {
                    clearInterval(self._presetMarkerInterval);
                    self._presetMarkerInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };
        };

        // Flag pre-existing divergence in a loaded workflow (e.g. a widget
        // edited manually before saving) — mirrors js/radiance_sampler.js.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            setTimeout(() => updatePresetDivergenceMarkers(self), 150);
            setTimeout(() => updatePresetDivergenceMarkers(self), 600);
        };
    }
});

console.log("[Radiance Cinematic Prompt Encoder] Extension loaded");
