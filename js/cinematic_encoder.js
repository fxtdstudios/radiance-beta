/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                    FXTD CINEMATIC ENCODER WIDGET
 *              Prompt Preview + Randomize Button for 10/10 UX
 *                        FXTD Studios © 2024-2026
 * ═══════════════════════════════════════════════════════════════════════════════
 */

import { app } from "../../scripts/app.js";

// Style presets list
const STYLE_PRESETS = [
    "None (Custom)",
    "→ Classic Hollywood",
    "→ Film Noir",
    "→ Sci-Fi Cinematic",
    "→ Cyberpunk",
    "→ Drama / Emotional",
    "→ Epic Landscape",
    "→ Portrait",
    "→ Documentary",
    "→ Artistic / Painterly",
    "→ Retro VHS",
    "→ Golden Hour Magic",
    "→ Moody Night",
    "→ Action / Dynamic",
    "→ Wes Anderson",
    // v2.1 presets — synced with nodes_prompt.py
    "→ 1970s New Hollywood",
    "→ 1980s Retro Action",
    "→ 1990s Music Video",
    "→ 2000s Digital Look",
    "→ Horror / Thriller",
    "→ Romance / Soft Focus",
    "→ Christopher Nolan",
    "→ Denis Villeneuve",
    "→ Quentin Tarantino",
    "→ Prestige Drama (HBO)",
    "→ True Crime Documentary",
    "→ Western",
    "→ Mystery / Detective",
    "→ Fantasy / Magical",
];

// Preset configurations (Synced with nodes_prompt.py)
const PRESET_CONFIGS = {
    "\u2192 Classic Hollywood": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Panavision Panaflex Gold II (35mm)",
        "lens_focal": "50mm Standard Prime",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Paramount Lighting",
        "style_aesthetic": "Cinematic Movie Still",
        "film_stock": "Kodak Vision3 500T",
        "color_grading": "Technicolor (Vintage)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "\u2192 Film Noir": {
        "framing": "Low Angle (Hero Shot)",
        "camera_type": "ARRI Alexa 35",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Film Noir Lighting",
        "style_aesthetic": "Monochrome Noir",
        "film_stock": "Kodak Tri-X 400 (B&W)",
        "color_grading": "Bleach Bypass (Gritty)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "\u2192 Sci-Fi Cinematic": {
        "framing": "Extreme Wide Shot (EWS)",
        "camera_type": "ARRI Alexa 65 (IMAX)",
        "lens_focal": "ARRI Master Anamorphic",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Cinematic Haze / Volumetric Fog",
        "style_aesthetic": "Blade Runner Atmosphere",
        "film_stock": "None",
        "color_grading": "Teal and Orange (Blockbuster)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "\u2192 Cyberpunk": {
        "framing": "Dutch Angle (Canted)",
        "camera_type": "Sony Venice 2",
        "lens_focal": "Anamorphic Lens",
        "aperture_dof": "f/1.8 (Soft Background)",
        "lighting": "Neon Cyberpunk Lighting",
        "style_aesthetic": "Cyberpunk 2077 Aesthetic",
        "film_stock": "Cinestill 800T",
        "color_grading": "Cyberpunk Neon Grading",
        "aspect_ratio": "21:9 (Ultrawide)",
    },
    "\u2192 Drama / Emotional": {
        "framing": "Close-Up (CU)",
        "camera_type": "ARRI Alexa Mini LF",
        "lens_focal": "85mm Portrait Prime",
        "aperture_dof": "f/1.2 (Dreamy Bokeh)",
        "lighting": "Rembrandt Lighting",
        "style_aesthetic": "Cinematic Movie Still",
        "film_stock": "Kodak Portra 400",
        "color_grading": "Desaturated (Muted)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "\u2192 Epic Landscape": {
        "framing": "Extreme Wide Shot (EWS)",
        "camera_type": "ARRI Alexa 65 (IMAX)",
        "lens_focal": "14mm Ultra-Wide Angle",
        "aperture_dof": "f/11 (Landscape Sharpness)",
        "lighting": "Golden Hour (Magic Hour)",
        "style_aesthetic": "National Geographic Style",
        "film_stock": "Fujifilm Velvia 50",
        "color_grading": "Vibrant High Contrast",
        "aspect_ratio": "21:9 (Ultrawide)",
    },
    "\u2192 Portrait": {
        "framing": "Medium Close-Up (MCU)",
        "camera_type": "Sony A7S III",
        "lens_focal": "85mm Portrait Prime",
        "aperture_dof": "f/1.2 (Dreamy Bokeh)",
        "lighting": "Soft Window Light",
        "style_aesthetic": "Editorial Photography",
        "film_stock": "Kodak Portra 400",
        "color_grading": "Pastel Soft Tones",
        "aspect_ratio": "4:3 (Academy Ratio)",
    },
    "\u2192 Documentary": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Canon C700 FF",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Practical Lighting",
        "style_aesthetic": "Documentary Texture",
        "film_stock": "None",
        "color_grading": "Desaturated (Muted)",
        "aspect_ratio": "16:9 (Widescreen)",
    },
    "\u2192 Artistic / Painterly": {
        "framing": "Medium Shot (MS)",
        "camera_type": "None",
        "lens_focal": "Petzval 85mm (Classic Swirl)",
        "aperture_dof": "f/1.8 (Soft Background)",
        "lighting": "Soft Window Light",
        "style_aesthetic": "Oil Painting (Classic)",
        "film_stock": "None",
        "color_grading": "Pastel Soft Tones",
        "aspect_ratio": "4:3 (Academy Ratio)",
    },
    "\u2192 Retro VHS": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Super 8mm Camera",
        "lens_focal": "50mm Standard Prime",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Practical Lighting",
        "style_aesthetic": "Vintage 1990s VHS",
        "film_stock": "Polaroid 600",
        "color_grading": "Cross Processed",
        "aspect_ratio": "4:3 (Academy Ratio)",
    },
    "\u2192 Golden Hour Magic": {
        "framing": "Full Body Shot (Wide)",
        "camera_type": "Sony Venice 2",
        "lens_focal": "85mm Portrait Prime",
        "aperture_dof": "f/1.8 (Soft Background)",
        "lighting": "Golden Hour (Magic Hour)",
        "style_aesthetic": "Photorealistic (Raw)",
        "film_stock": "Kodak Ektar 100",
        "color_grading": "Vibrant High Contrast",
        "aspect_ratio": "16:9 (Widescreen)",
    },
    "\u2192 Moody Night": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Sony A7S III",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/1.2 (Dreamy Bokeh)",
        "lighting": "Moonlight",
        "style_aesthetic": "Cinematic Movie Still",
        "film_stock": "Cinestill 800T",
        "color_grading": "Teal and Orange (Blockbuster)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "\u2192 Action / Dynamic": {
        "framing": "Low Angle (Hero Shot)",
        "camera_type": "RED V-Raptor XL",
        "lens_focal": "24mm Wide Angle",
        "aperture_dof": "f/5.6 (Sharp Subject)",
        "lighting": "Harsh Sunlight",
        "style_aesthetic": "Hyper-Realism",
        "film_stock": "None",
        "shutter_speed": "1/1000th sec (Frozen Action)",
        "color_grading": "Teal and Orange (Blockbuster)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "→ Wes Anderson": {
        "framing": "Symmetrical Composition",
        "camera_type": "ARRI Alexa 35",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/8.0 (Deep Focus)",
        "lighting": "Soft Window Light",
        "style_aesthetic": "Wes Anderson Symmetric",
        "film_stock": "Kodak Portra 400",
        "color_grading": "Pastel Soft Tones",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    // ── v2.1 Presets — synced with nodes_prompt.py CinematicDatasets ──────
    "→ 1970s New Hollywood": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Panavision Panaflex Gold II (35mm)",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Practical Lighting",
        "style_aesthetic": "Cinematic Movie Still",
        "film_stock": "Kodak Vision3 500T",
        "color_grading": "Technicolor (Vintage)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "→ 1980s Retro Action": {
        "framing": "Low Angle (Hero Shot)",
        "camera_type": "ARRI Alexa 35",
        "lens_focal": "Anamorphic Lens",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Cinematic Haze / Volumetric Fog",
        "style_aesthetic": "Hyper-Realism",
        "film_stock": "None",
        "color_grading": "Teal and Orange (Blockbuster)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "→ 1990s Music Video": {
        "framing": "Extreme Close-Up (ECU)",
        "camera_type": "Super 8mm Camera",
        "lens_focal": "Fish-Eye Lens",
        "aperture_dof": "f/1.8 (Soft Background)",
        "lighting": "Neon Cyberpunk Lighting",
        "style_aesthetic": "Vintage 1990s VHS",
        "film_stock": "Polaroid 600",
        "color_grading": "Cross Processed",
        "aspect_ratio": "4:3 (Academy Ratio)",
    },
    "→ 2000s Digital Look": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Sony A7S III",
        "lens_focal": "24mm Wide Angle",
        "aperture_dof": "f/5.6 (Sharp Subject)",
        "lighting": "Harsh Sunlight",
        "style_aesthetic": "Editorial Photography",
        "film_stock": "None",
        "color_grading": "Vibrant High Contrast",
        "aspect_ratio": "16:9 (Widescreen)",
    },
    "→ Horror / Thriller": {
        "framing": "Low Angle (Hero Shot)",
        "camera_type": "ARRI Alexa Mini LF",
        "lens_focal": "16mm Ultra-Wide Angle",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Chiaroscuro (High Contrast)",
        "style_aesthetic": "Film Noir Aesthetic",
        "film_stock": "Kodak Vision3 500T",
        "color_grading": "Bleach Bypass (Gritty)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "→ Romance / Soft Focus": {
        "framing": "Medium Close-Up (MCU)",
        "camera_type": "Sony A7S III",
        "lens_focal": "50mm Standard Prime",
        "aperture_dof": "f/1.2 (Dreamy Bokeh)",
        "lighting": "Soft Window Light",
        "style_aesthetic": "Dreamy Soft Focus",
        "film_stock": "Kodak Portra 400",
        "color_grading": "Pastel Soft Tones",
        "aspect_ratio": "1.85:1 (Standard Widescreen)",
    },
    "→ Christopher Nolan": {
        "framing": "Wide Shot (WS)",
        "camera_type": "IMAX 15/70mm Film Camera",
        "lens_focal": "28mm Wide Angle",
        "aperture_dof": "f/8.0 (Deep Focus)",
        "lighting": "Natural Ambient Light",
        "style_aesthetic": "Photorealistic (Raw)",
        "film_stock": "IMAX 15/70mm Film Stock",
        "color_grading": "Neutral ACES Workflow",
        "aspect_ratio": "1.43:1 (IMAX)",
    },
    "→ Denis Villeneuve": {
        "framing": "Extreme Wide Shot (EWS)",
        "camera_type": "ARRI Alexa 65 (IMAX)",
        "lens_focal": "24mm Wide Angle",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Volumetric Fog / Atmospheric Haze",
        "style_aesthetic": "Atmospheric Cinematic",
        "film_stock": "None",
        "color_grading": "Desaturated Cool Tones",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "→ Quentin Tarantino": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Panavision Panaflex Gold II (35mm)",
        "lens_focal": "40mm Semi-Wide",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "High-Key Lighting",
        "style_aesthetic": "Vintage Kodachrome Look",
        "film_stock": "Kodak Vision3 500T",
        "color_grading": "Vibrant High Contrast",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "→ Prestige Drama (HBO)": {
        "framing": "Medium Shot (MS)",
        "camera_type": "ARRI Alexa Mini LF",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/2.0 (Shallow Cinematic)",
        "lighting": "Naturalistic Interior Light",
        "style_aesthetic": "Cinematic TV Drama",
        "film_stock": "None",
        "color_grading": "Moody Shadows",
        "aspect_ratio": "2.00:1 (DCI Flat)",
    },
    "→ True Crime Documentary": {
        "framing": "Medium Close-Up (MCU)",
        "camera_type": "Canon C300 Mark III",
        "lens_focal": "50mm Standard Prime",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Interview 3-Point Setup",
        "style_aesthetic": "Documentary Realism",
        "film_stock": "None",
        "color_grading": "Desaturated Cool Tones",
        "aspect_ratio": "16:9 (Widescreen)",
    },
    "→ Western": {
        "framing": "Medium Full Shot (MFS)",
        "camera_type": "Panavision Panaflex Gold II (35mm)",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/5.6 (Sharp Subject)",
        "lighting": "Harsh Sunlight",
        "style_aesthetic": "Vintage Kodachrome Look",
        "film_stock": "Eastman Kodak 5254 (Vintage)",
        "color_grading": "Orange and Teal (Hollywood)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "→ Mystery / Detective": {
        "framing": "Over-The-Shoulder (OTS)",
        "camera_type": "ARRI Alexa 35",
        "lens_focal": "50mm Standard Prime",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Film Noir Lighting",
        "style_aesthetic": "Neo-Noir Modern",
        "film_stock": "Kodak Vision3 500T",
        "color_grading": "Desaturated Cool Tones",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "→ Fantasy / Magical": {
        "framing": "Medium Wide (MW)",
        "camera_type": "ARRI Alexa 65 (IMAX)",
        "lens_focal": "Anamorphic Lens",
        "aperture_dof": "f/2.0 (Shallow Cinematic)",
        "lighting": "God Rays (Crepuscular Rays)",
        "style_aesthetic": "Magical Realism",
        "film_stock": "Kodak Vision3 250D",
        "color_grading": "Technicolor (Vintage)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// All arrays are kept in FULL SYNC with CinematicDatasets in nodes_prompt.py
// ─────────────────────────────────────────────────────────────────────────────

const FRAMING = [
    "None",
    "Extreme Close-Up (ECU)", "Close-Up (CU)", "Medium Close-Up (MCU)",
    "Medium Shot (MS)", "Medium Wide (MW)", "Medium Full Shot (MFS)",
    "Cowboy Shot (American Shot)", "Wide Shot (WS)", "Full Body Shot (Wide)",
    "Extreme Wide Shot (EWS)", "Establishing Shot", "Over-The-Shoulder (OTS)",
    "Point of View (POV)", "Low Angle (Hero Shot)", "High Angle (Vulnerability)",
    "Bird's Eye View (Overhead)", "Worm's Eye View", "Dutch Angle (Canted)",
    "Symmetrical Composition", "Rule of Thirds"
];

const CAMERAS = [
    "None",
    "ARRI Alexa 65 (IMAX)", "ARRI Alexa Mini LF", "ARRI Alexa 35",
    "Sony Venice 2", "Sony FX9", "Sony A7S III",
    "RED V-Raptor XL", "RED Komodo", "RED Monstro 8K VV",
    "Panavision Millennium DXL2", "Panavision Panaflex Gold II (35mm)",
    "IMAX 15/70mm Film Camera", "Bolex H16 (16mm Film)", "Super 8mm Camera",
    "Canon C700 FF", "Canon C300 Mark III", "Blackmagic URSA Mini Pro 12K",
    "GoPro Hero 12", "iPhone 15 Pro Max", "Polaroid SX-70",
    "Vintage Daguerreotype Camera"
];

const LENSES = [
    "None",
    "14mm Ultra-Wide Angle", "16mm Ultra-Wide Angle", "24mm Wide Angle",
    "28mm Wide Angle", "35mm Classic Wide", "40mm Semi-Wide",
    "50mm Standard Prime", "85mm Portrait Prime", "105mm Macro",
    "135mm Medium Telephoto", "200mm Telephoto Compression", "600mm Super Telephoto",
    "Anamorphic Lens", "Fish-Eye Lens", "Tilt-Shift Lens",
    "Cooke S7/i Full Frame", "Cooke Speed Panchro Vintage",
    "Zeiss Master Prime", "Zeiss Supreme Prime",
    "ARRI/Zeiss Signature Prime", "ARRI Master Anamorphic",
    "Panavision Primo 70", "Panavision C-Series Anamorphic",
    "Canon K35 Vintage", "Canon FD 50mm L",
    "Helios 44-2 58mm (Swirly Bokeh)", "Lensbaby Velvet (Soft Focus)",
    "Petzval 85mm (Classic Swirl)",
    "Leica Summilux 50mm f/1.4", "Leica Summicron 35mm",
    "Sigma Art 35mm f/1.4", "Sigma Art 85mm f/1.4",
    "Sony G Master 24-70mm", "Sony G Master 85mm",
    "Laowa Probe Lens (Macro)", "Freefly Wave (High-Speed)",
    "Angenieux Optimo Zoom", "Fujinon Premista Zoom"
];

const APERTURES = [
    "None",
    "f/0.95 (Razor Thin DoF)", "f/1.2 (Dreamy Bokeh)", "f/1.8 (Soft Background)",
    "f/2.0 (Shallow Cinematic)", "f/2.8 (Cinematic Separation)", "f/4.0 (Balanced)",
    "f/5.6 (Sharp Subject)", "f/8.0 (Deep Focus)", "f/11 (Landscape Sharpness)",
    "f/16 (Everything in Focus)", "f/22 (Diffraction Starbursts)"
];

const LIGHTING = [
    "None",
    "Rembrandt Lighting", "Chiaroscuro (High Contrast)", "Film Noir Lighting",
    "Split Lighting", "Butterfly Lighting", "Paramount Lighting",
    "Soft Window Light", "Golden Hour (Magic Hour)", "Blue Hour",
    "Cinematic Haze / Volumetric Fog", "God Rays (Crepuscular Rays)",
    "Neon Cyberpunk Lighting", "Practical Lighting", "Bioluminescence",
    "Studio Strobe 3-Point Setup", "Ring Light", "Candlelight",
    "Moonlight", "Overcast Soft Light", "Harsh Sunlight",
    "Natural Ambient Light", "High-Key Lighting",
    "Naturalistic Interior Light", "Interview 3-Point Setup",
    "Volumetric Fog / Atmospheric Haze"
];

const STYLES = [
    "None",
    "Photorealistic (Raw)", "Cinematic Movie Still", "Hyper-Realism",
    "Editorial Photography", "National Geographic Style", "Documentary Texture",
    "Vintage 1990s VHS", "Analog Film (Kodak Portra 400)", "Fujifilm Velvia 50",
    "Black and White (Ilford HP5)", "Monochrome Noir",
    "CGI 3D Render (Octane)", "Unreal Engine 5", "Pixar Animation Style",
    "Anime (Makoto Shinkai)", "Oil Painting (Classic)", "Concept Art",
    "Cyberpunk 2077 Aesthetic", "Wes Anderson Symmetric", "Tarantino Violence",
    "Kubrick One-Point Perspective", "Blade Runner Atmosphere",
    "Film Noir Aesthetic", "Dreamy Soft Focus", "Atmospheric Cinematic",
    "Vintage Kodachrome Look", "Cinematic TV Drama", "Documentary Realism",
    "Neo-Noir Modern", "Magical Realism"
];

const FILM_STOCKS = [
    "None",
    "Kodak Vision3 500T", "Kodak Vision3 250D", "Kodak Portra 400", "Kodak Ektar 100",
    "Kodak Tri-X 400 (B&W)", "Fujifilm Pro 400H", "Cinestill 800T",
    "Fujifilm Velvia 50", "Ilford Delta 3200", "Polaroid 600", "Wet Plate Collodion",
    "IMAX 15/70mm Film Stock", "Eastman Kodak 5254 (Vintage)"
];

const SHUTTER_SPEEDS = [
    "None",
    "1/50th sec (Standard Motion Blur)", "1/1000th sec (Frozen Action)",
    "Long Exposure (Light Trails)", "Slow Shutter (Dreamy Blur)"
];

const COLOR_GRADING = [
    "None",
    "Teal and Orange (Blockbuster)", "Bleach Bypass (Gritty)",
    "Technicolor (Vintage)", "Cross Processed", "Desaturated (Muted)",
    "Vibrant High Contrast", "Sepia Tone", "Monochrome High Key",
    "Cyberpunk Neon Grading", "Pastel Soft Tones",
    "Neutral ACES Workflow", "Desaturated Cool Tones",
    "Moody Shadows", "Orange and Teal (Hollywood)"
];

const ASPECT_RATIOS = [
    "None",
    "16:9 (Widescreen)", "2.39:1 (Anamorphic Scope)",
    "4:3 (Academy Ratio)", "1:1 (Square)", "9:16 (Social Vertical)",
    "21:9 (Ultrawide)",
    "1.43:1 (IMAX)", "1.85:1 (Standard Widescreen)", "2.00:1 (DCI Flat)"
];

app.registerExtension({
    name: "FXTD.CinematicEncoder",

    async nodeCreated(node) {
        if (node.comfyClass !== "RadianceCinematicPromptEncoder") return;

        // Add custom widget for prompt preview
        const previewWidget = node.addWidget("text", "prompt_preview", "", () => { }, {
            multiline: true,
            inputEl: null,
        });
        previewWidget.computeSize = () => [node.size[0] - 20, 60];
        previewWidget.serializeValue = () => undefined; // Don't save this

        // Style the preview
        if (previewWidget.inputEl) {
            previewWidget.inputEl.readOnly = true;
            previewWidget.inputEl.style.background = "rgba(0, 168, 255, 0.1)";
            previewWidget.inputEl.style.border = "1px solid rgba(0, 168, 255, 0.3)";
            previewWidget.inputEl.style.color = "#aaccff";
            previewWidget.inputEl.style.fontSize = "10px";
            previewWidget.inputEl.style.fontFamily = "monospace";
        }

        // Randomize button
        node.addWidget("button", "\u203a Randomize Style", null, () => {
            randomizeSettings(node);
        });

        // Export Preset button
        node.addWidget("button", "\u203a Export Preset", null, () => {
            exportPreset(node);
        });

        // Import Preset button (uses hidden file input)
        node.addWidget("button", "\u203a Import Preset", null, () => {
            importPreset(node, previewWidget);
        });

        // Update preview when inputs change
        const originalOnPropertyChanged = node.onPropertyChanged;
        node.onPropertyChanged = function (property, value) {
            if (originalOnPropertyChanged) originalOnPropertyChanged.call(this, property, value);
            updatePreview(node, previewWidget);
        };

        // Also update on widget change
        const originalOnWidgetChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value, old_value, widget) {
            if (originalOnWidgetChanged) originalOnWidgetChanged.call(this, name, value, old_value, widget);

            if (name === "style_preset") {
                // Apply preset values
                applyPreset(node, value);
            } else if (name !== "prompt_preview" && name !== "style_preset") {
                // Check if user manually changed something
                checkCustomOverride(node, name, value);
            }

            updatePreview(node, previewWidget);
        };

        // Initial update
        setTimeout(() => updatePreview(node, previewWidget), 100);
    }
});

// Global flag to prevent recursive updates
let IS_APPLYING_PRESET = false;

function applyPreset(node, presetName) {
    if (IS_APPLYING_PRESET) return;
    if (presetName === "None (Custom)" || !PRESET_CONFIGS[presetName]) return;

    IS_APPLYING_PRESET = true;

    const config = PRESET_CONFIGS[presetName];
    const widgets = node.widgets;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
        }
    }

    IS_APPLYING_PRESET = false;
    node.setDirtyCanvas(true);
}

function checkCustomOverride(node, changedWidgetName, newValue) {
    if (IS_APPLYING_PRESET) return;

    const presetWidget = node.widgets.find(w => w.name === "style_preset");
    if (!presetWidget || presetWidget.value === "None (Custom)") return;

    // Check if the new value contradicts the current preset
    const currentPreset = presetWidget.value;
    const config = PRESET_CONFIGS[currentPreset];

    if (config && config[changedWidgetName] !== undefined) {
        if (config[changedWidgetName] !== newValue) {
            // User manually changed something that contradicts the preset
            // Switch preset to Custom
            presetWidget.value = "None (Custom)";
            node.setDirtyCanvas(true);
        }
    }
}

function randomizeSettings(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    // Find and randomize each widget
    // Set preset to None so individual randomized values aren't overwritten by applyPreset.
    // "Pure random" — each param randomized independently.
    const presetWidget = widgets.find(w => w.name === "style_preset");
    if (presetWidget) presetWidget.value = "None (Custom)";

    for (const widget of widgets) {
        // Helper: pick a random non-None entry from an array
        const pick = (arr) => arr[Math.floor(Math.random() * (arr.length - 1)) + 1];

        switch (widget.name) {
            case "framing":
                widget.value = pick(FRAMING);
                break;
            case "camera_type":
                widget.value = pick(CAMERAS);
                break;
            case "lens_focal":
                widget.value = pick(LENSES);
                break;
            case "aperture_dof":
                widget.value = pick(APERTURES);
                break;
            case "lighting":
                widget.value = pick(LIGHTING);
                break;
            case "style_aesthetic":
                widget.value = pick(STYLES);
                break;
            case "film_stock":
                // 40% chance of "None" (many styles look better without a stock override)
                widget.value = Math.random() < 0.4 ? "None" : pick(FILM_STOCKS);
                break;
            case "shutter_speed":
                // 60% chance of "None"
                widget.value = Math.random() < 0.6 ? "None" : pick(SHUTTER_SPEEDS);
                break;
            case "color_grading":
                widget.value = pick(COLOR_GRADING);
                break;
            case "aspect_ratio":
                widget.value = pick(ASPECT_RATIOS);
                break;
            case "year_era":
                widget.value = 1950 + Math.floor(Math.random() * 80); // 1950–2030
                break;
        }
    }

    // Trigger update
    node.setDirtyCanvas(true);

    // Find and update preview
    const previewWidget = widgets.find(w => w.name === "prompt_preview");
    if (previewWidget) {
        updatePreview(node, previewWidget);
    }
}

function updatePreview(node, previewWidget) {
    if (!previewWidget) return;

    const widgets = node.widgets;
    if (!widgets) return;

    // Get current values
    let basePrompt = "", framing = "", camera = "", lighting = "", style = "";
    let lens = "", aperture = "", film = "", grade = "", ratio = "", shutter = "";
    let yearEra = 0;

    for (const widget of widgets) {
        switch (widget.name) {
            case "base_prompt": basePrompt = widget.value || "A cinematic scene..."; break;
            case "framing": framing = widget.value; break;
            case "camera_type": camera = widget.value; break;
            case "lighting": lighting = widget.value; break;
            case "style_aesthetic": style = widget.value; break;
            case "lens_focal": lens = widget.value; break;
            case "aperture_dof": aperture = widget.value; break;
            case "film_stock": film = widget.value; break;
            case "color_grading": grade = widget.value; break;
            case "aspect_ratio": ratio = widget.value; break;
            case "shutter_speed": shutter = widget.value; break;
            case "year_era": yearEra = widget.value; break;
        }
    }

    // Build preview (mirrors build_cinematic_prompt logic from nodes_prompt.py)
    let parts = [];

    // 1. Framing + Subject
    if (framing && framing !== "None") {
        parts.push(`${framing} of ${basePrompt}.`);
    } else {
        parts.push(`${basePrompt}.`);
    }

    // 2. Camera Tech stack
    let tech = [];
    if (camera && camera !== "None") tech.push(`Shot on ${camera}`);
    if (lens && lens !== "None") tech.push(`with ${lens}`);
    if (aperture && aperture !== "None") tech.push(`at ${aperture}`);
    if (shutter && shutter !== "None") tech.push(`shutter speed ${shutter}`);
    if (tech.length > 0) parts.push(tech.join(" ") + ".");

    // 3. Lighting
    if (lighting && lighting !== "None") parts.push(`Lighting is ${lighting}.`);

    // 4. Color grading
    if (grade && grade !== "None") parts.push(`Color graded in ${grade}.`);

    // 5. Style + Film + Era
    let finish = [];
    if (style && style !== "None") finish.push(style);
    if (film && film !== "None") finish.push(`on ${film}`);
    if (yearEra && yearEra !== 2024) finish.push(`Est. Year ${yearEra}.`);
    if (finish.length > 0) parts.push(finish.join(", ") + ".");

    // 6. Aspect ratio
    if (ratio && ratio !== "None") parts.push(`${ratio} format.`);

    const preview = parts.join(". ").substring(0, 300) + (parts.join(". ").length > 300 ? "..." : "");

    previewWidget.value = preview;

    // Update DOM if exists
    if (previewWidget.inputEl) {
        previewWidget.inputEl.value = preview;
    }

    node.setDirtyCanvas(true);
}

// ═══════════════════════════════════════════════════════════════════════════
//                          PRESET EXPORT/IMPORT
// ═══════════════════════════════════════════════════════════════════════════

function exportPreset(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    // Gather all current settings
    const preset = {
        name: "Custom Preset",
        created: new Date().toISOString(),
        version: "1.0",
        settings: {}
    };

    // Export all relevant settings
    const exportFields = [
        "base_prompt", "framing", "camera_type", "lens_focal", "aperture_dof",
        "lighting", "style_aesthetic", "film_stock", "shutter_speed",
        "color_grading", "aspect_ratio", "custom_details", "year_era"
    ];

    for (const widget of widgets) {
        if (exportFields.includes(widget.name)) {
            preset.settings[widget.name] = widget.value;
        }
    }

    // Prompt for preset name
    const presetName = prompt("Enter preset name:", "My Custom Preset");
    if (!presetName) return;

    preset.name = presetName;

    // Download as JSON
    const jsonStr = JSON.stringify(preset, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${presetName.replace(/[^a-z0-9_\-]/gi, "_")}_preset.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[Radiance] Preset "${presetName}" exported successfully!`);
}

function importPreset(node, previewWidget) {
    // Create hidden file input
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

                // Validate preset structure
                if (!preset.settings || typeof preset.settings !== 'object') {
                    alert('Invalid preset file: Missing settings object');
                    return;
                }

                // Apply settings to widgets
                const widgets = node.widgets;
                let appliedCount = 0;

                for (const widget of widgets) {
                    if (preset.settings[widget.name] !== undefined) {
                        widget.value = preset.settings[widget.name];
                        appliedCount++;
                    }
                }

                // Switch preset selector to "Custom"
                const presetWidget = widgets.find(w => w.name === "style_preset");
                if (presetWidget) {
                    presetWidget.value = "None (Custom)";
                }

                // Update preview
                if (previewWidget) {
                    updatePreview(node, previewWidget);
                }

                node.setDirtyCanvas(true);

                console.log(`[Radiance] Preset "${preset.name || 'Unnamed'}" imported successfully! (${appliedCount} settings applied)`);
                alert(`Preset "${preset.name || 'Unnamed'}" imported!\n${appliedCount} settings applied.`);

            } catch (error) {
                console.error('[Radiance] Failed to import preset:', error);
                alert('Failed to import preset: ' + error.message);
            }
        };

        reader.readAsText(file);
    };

    document.body.appendChild(fileInput);
    fileInput.click();
    document.body.removeChild(fileInput);
}
