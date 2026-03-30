import { app } from "../../scripts/app.js";

// Elegant Color Palette
const COLORS = {
    HEADER: "#232330",       // Dark Slate Blue
    BODY: "#0f0f14",         // Deep Black/Blue
    BORDER: "#4a4a6a",       // Muted Blue-Grey
    TEXT_HEADER: "#ffffff",  // White
    TEXT_BODY: "#a0a0b0",    // Dim White
    ACCENT: "#00a8ff"        // Radiance Blue
};

(function patchLiteGraphShape() {
    if (!window.LiteGraph) return; // LiteGraph not yet loaded; skip
    if (Object.getOwnPropertyDescriptor(LiteGraph, '_shape')) return; // already patched

    const _mapShape = (v) => {
        if (typeof v === 'number' && [1, 2, 4].includes(v)) return v;
        const s = String(v).toLowerCase();
        if (s === 'box') return 1; // LiteGraph.BOX_SHAPE
        if (s === 'round') return 2; // LiteGraph.ROUND_SHAPE
        if (s === 'circle') return 2;
        if (s === 'card') return 4; // LiteGraph.CARD_SHAPE
        return typeof v === 'number' ? Math.max(1, v) : 2; // fallback: ROUND
    };

    let _current = _mapShape(LiteGraph.NODE_DEFAULT_SHAPE);
    Object.defineProperty(LiteGraph, 'NODE_DEFAULT_SHAPE', {
        get() { return _current; },
        set(v) { _current = _mapShape(v); },
        configurable: true,
        enumerable: true,
    });
})();

app.registerExtension({
    name: "FXTD.Radiance.Style",

    async init() {
        // Shape shim now applied at module load time (top-level IIFE above).
    },

    async setup() {


        // Fix #2: VHS Nodes crash on load when force_size is null
        if (window.LiteGraph && window.LGraphNode) {
            const origConfigure = window.LGraphNode.prototype.configure;
            window.LGraphNode.prototype.configure = function (info) {
                if (this.type && this.type.includes("VHS_") && info && info.widgets_values) {
                    if (typeof info.widgets_values === 'object' && !Array.isArray(info.widgets_values)) {
                        if (info.widgets_values.force_size === null) {
                            info.widgets_values.force_size = "Disabled";
                        }
                    }
                }
                return origConfigure.apply(this, arguments);
            };
        }
        // ---------------------------------------------

        // v3.0 #15: High Contrast Mode & Premium Glass Styles
        const style = document.createElement("style");
        style.innerHTML = `
            .radiance-glass-dock {
                transition: background 0.2s, border 0.2s;
            }

            /* High Contrast Mode Overrides — Force pure black/white/yellow for legibility */
            .radiance-glass-dock.high-contrast {
                background: #000000 !important;
                backdrop-filter: none !important;
                border: 2px solid #ffffff !important;
                box-shadow: 0 0 20px rgba(255,255,255,0.2) !important;
            }

            .radiance-glass-dock.high-contrast * {
                color: #ffffff !important;
                text-shadow: none !important;
            }

            .radiance-glass-dock.high-contrast input[type="range"] {
                accent-color: #ffff00 !important;
                background: #222 !important;
                border: 1px solid #555;
            }

            .radiance-glass-dock.high-contrast .radiance-knob-track {
                background: #111 !important;
                border: 1.5px solid #fff !important;
            }
            
            .radiance-glass-dock.high-contrast .radiance-knob-fill {
                background: #ffff00 !important;
            }

            .radiance-glass-dock.high-contrast .radiance-tab-btn.active {
                border-bottom: 3px solid #ffff00 !important;
                color: #ffff00 !important;
            }
        `;
        document.head.appendChild(style);
    },

    async nodeCreated(node, app) {
        // Identify Radiance nodes by their class name prefix
        const isRadiance = node.comfyClass && (
            node.comfyClass.startsWith("FXTD") ||
            node.comfyClass.startsWith("Radiance") ||
            node.comfyClass.includes("Radiance")
        );

        if (isRadiance) {
            // Apply Colors
            node.color = COLORS.HEADER;
            node.bgcolor = COLORS.BODY;

            // Optional: Set shape to box if not set (standard is 1)
            // node.shape = 1; 

            // If the node has a widget resizing method, we can hook it here if needed, 
            // but for now, color is the main request.

            // Force redraw to apply immediate changes if needed
            if (node.graph) {
                node.setDirtyCanvas(true, true);
            }
        }
    }
});
