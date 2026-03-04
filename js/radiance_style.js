import { app } from "../../scripts/app.js";

/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                         RADIANCE UI THEME
 *                    Premium Styling for Custom Nodes
 * ═══════════════════════════════════════════════════════════════════════════════
 */

// Elegant Color Palette
const COLORS = {
    HEADER: "#232330",       // Dark Slate Blue
    BODY: "#0f0f14",         // Deep Black/Blue
    BORDER: "#4a4a6a",       // Muted Blue-Grey
    TEXT_HEADER: "#ffffff",  // White
    TEXT_BODY: "#a0a0b0",    // Dim White
    ACCENT: "#00a8ff"        // Radiance Blue
};

app.registerExtension({
    name: "FXTD.Radiance.Style",

    async setup() {
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
