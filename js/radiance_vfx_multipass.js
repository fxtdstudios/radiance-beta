/**
 * ◎ Radiance VFX Multipass — Frontend Extension
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Provides ComfyUI UX enhancements for the RadianceVFXMultipass node:
 *
 *   1. Colour-codes output slots by VFX pass type so the graph reads
 *      clearly (beauty=white, diffuse=orange, specular=yellow, masks=green,
 *      depth=blue, ao=teal, edge=pink).
 *
 *   2. Hides the EXR path + frame_index inputs when export_exr=false,
 *      keeping the node compact by default.
 *
 *   3. Shows a warning badge on the node title when depth_map is not
 *      connected (depth and AO outputs will be black).
 *
 *   4. Tooltip on the node header reminding users to place it after
 *      ◎ Radiance VAE Decode.
 */

import { app } from "../../scripts/app.js";

// ── Output slot colours (LiteGraph slot colour string) ──────────────────────
const PASS_COLORS = {
    beauty:         "#e8e8e8",  // near-white  — full image
    diffuse:        "#f0a070",  // warm orange — base colour
    specular:       "#f0e060",  // yellow      — highlights/detail
    shadow_mask:    "#5090c0",  // blue        — shadow region
    highlight_mask: "#c0e060",  // lime        — highlight region
    midtone_mask:   "#70c080",  // green       — midtone region
    depth:          "#6090d0",  // blue        — depth/Z
    ao:             "#50b0a0",  // teal        — occlusion
    edge:           "#d070b0",  // pink        — outline/edge
};

// ── Widget names that should be hidden when export_exr = false ──────────────
const EXR_DEPENDENT_WIDGETS = ["exr_output_path", "frame_index"];

app.registerExtension({
    name: "Radiance.VFXMultipass",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RadianceVFXMultipass") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // ── 1. Colour output slots ─────────────────────────────────────
            if (this.outputs) {
                const passOrder = [
                    "beauty", "diffuse", "specular",
                    "shadow_mask", "highlight_mask", "midtone_mask",
                    "depth", "ao", "edge",
                    // last output is STRING (pass_info) — leave default
                ];
                this.outputs.forEach((slot, i) => {
                    const passName = passOrder[i];
                    if (passName && PASS_COLORS[passName]) {
                        slot.color_on  = PASS_COLORS[passName];
                        slot.color_off = PASS_COLORS[passName] + "88"; // 53% opacity when disconnected
                    }
                });
            }

            // ── 2. EXR widget visibility ───────────────────────────────────
            const exportWidget = this.widgets
                ? this.widgets.find(w => w.name === "export_exr")
                : null;

            EXR_DEPENDENT_WIDGETS.forEach(name => {
                const w = this.widgets ? this.widgets.find(x => x.name === name) : null;
                if (w) w.origType = w.type;
            });

            const setExrWidgetsVisible = (visible) => {
                EXR_DEPENDENT_WIDGETS.forEach(name => {
                    const w = this.widgets ? this.widgets.find(x => x.name === name) : null;
                    if (!w) return;
                    w.type = visible ? (w.origType || "text") : "hidden";
                });
                // Resize node to fit after hiding/showing
                this.setSize(this.computeSize());
                if (this.graph) this.setDirtyCanvas(true, true);
            };

            if (exportWidget) {
                const origCb = exportWidget.callback;
                exportWidget.callback = function (value) {
                    if (origCb) origCb.call(this, value);
                    setExrWidgetsVisible(!!value);
                };
                // Apply on init (deferred so all widgets are ready)
                setTimeout(() => setExrWidgetsVisible(!!exportWidget.value), 30);
            }

            // ── 3. Depth warning ───────────────────────────────────────────
            // Store original title and update it based on depth connection state.
            this._vfx_base_title = this.title;

            const updateDepthWarning = () => {
                const depthInput = this.inputs
                    ? this.inputs.find(i => i.name === "depth_map")
                    : null;
                const connected = depthInput && depthInput.link != null;
                this.title = connected
                    ? this._vfx_base_title
                    : this._vfx_base_title + "  ⚠ depth not connected";
                if (this.graph) this.setDirtyCanvas(true, false);
            };

            // Re-run on every connection change
            const origConnect    = this.onConnectionsChange;
            this.onConnectionsChange = function (...args) {
                if (origConnect) origConnect.apply(this, args);
                updateDepthWarning();
            };

            setTimeout(updateDepthWarning, 50);

            return r;
        };

        // ── 4. Tooltip on node title bar ────────────────────────────────────
        nodeType.prototype.onMouseEnter = function () {
            // LiteGraph doesn't have a native tooltip API; we set the
            // canvas title attribute which browsers show on hover.
            if (this.graph && this.graph.canvas && this.graph.canvas.canvas) {
                this.graph.canvas.canvas.title =
                    "◎ Radiance VFX Multipass — place after ◎ Radiance VAE Decode. "
                    + "Connect ◎ Radiance Depth Map for depth and AO passes.";
            }
        };

        nodeType.prototype.onMouseLeave = function () {
            if (this.graph && this.graph.canvas && this.graph.canvas.canvas) {
                this.graph.canvas.canvas.title = "";
            }
        };
    },
});
