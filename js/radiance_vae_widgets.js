/**
 * radiance_vae_widgets.js
 * v1.0 — Radiance VAE Decode / Roundtrip widget sync
 *
 * WHAT THIS FIXES:
 *
 *   Caveat 2 (Compress Log-only controls):
 *     display_tonemap only applies to Compress(Log) decode output. This
 *     extension grays HDR display/export widgets out for other HDR modes so
 *     their scope is visually clear.
 *
 *   Additional: hdr_output=True shows an amber warning badge on the node
 *   so the user knows the ComfyUI preview will look overexposed (intentional
 *   for VFX pipelines — the tensor carries raw scene-linear values).
 *
 * INSTALL:
 *   Place this file in the same folder as radiance_bootstrap.js (the custom
 *   node's web/js directory). ComfyUI auto-loads all .js files from that dir.
 *   No other changes needed.
 */

import { app } from "../../scripts/app.js";

// Nodes this extension manages
const HDR_DECODE_NODE  = "◎ Radiance HDR VAE Decode";

// Widget name constants
const W_HDR_OUTPUT       = "hdr_output";
const W_DISPLAY_TONEMAP  = "display_tonemap";
const W_HDR_MODE         = "hdr_mode";
const W_EXPORT_RHDR      = "export_rhdr";

/** Return widget by name from a node, or null. */
function getWidget(node, name) {
    return node.widgets?.find(w => w.name === name) ?? null;
}

/**
 * Apply or remove the "disabled" visual state on a widget.
 * ComfyUI doesn't have a built-in disabled state, so we:
 *  - Store the original draw function
 *  - Override it to draw grayed-out text
 *  - Set a .disabled flag that the node can check
 */
function setWidgetDisabled(widget, disabled) {
    if (!widget) return;
    widget.disabled = disabled;

    if (disabled) {
        // Save original draw if not already saved
        if (!widget._origDraw) widget._origDraw = widget.draw;
        widget.draw = function(ctx, node, width, y) {
            ctx.save();
            ctx.globalAlpha = 0.35;
            if (widget._origDraw) widget._origDraw.call(this, ctx, node, width, y);
            ctx.restore();
        };
    } else {
        // Restore original draw
        if (widget._origDraw) {
            widget.draw = widget._origDraw;
            delete widget._origDraw;
        }
    }
}

/**
 * Show or hide the amber "HDR OUTPUT: ComfyUI preview will be overexposed"
 * badge on the node title bar.
 */
function setHDROutputBadge(node, show) {
    // Use node.badges array if available (ComfyUI 0.3+), else use title suffix
    if (!node._radianceHDRBadge) {
        const badge = document.createElement("div");
        badge.textContent = "⚠ HDR OUT";
        badge.title = (
            "display_tonemap='None' + hdr_output=True + Compress(Log): " +
            "raw scene-linear values far above 1.0 pass through — guaranteed overexposure " +
            "in ComfyUI preview. " +
            "Set display_tonemap='ACES Filmic' or 'Reinhard' to fix. " +
            "Use 'None' only when feeding an OCIO-aware viewer (Nuke, Resolve)."
        );
        badge.style.cssText = [
            "display:none",
            "position:absolute",
            "top:2px",
            "right:4px",
            "background:rgba(251,146,22,0.15)",
            "border:1px solid rgba(251,146,22,0.5)",
            "border-radius:3px",
            "font-size:8px",
            "font-weight:800",
            "letter-spacing:0.8px",
            "color:#fb9216",
            "padding:1px 5px",
            "pointer-events:auto",
            "z-index:10",
        ].join(";");
        node._radianceHDRBadge = badge;
        // Attach to node DOM element if available
        node.element?.appendChild(badge);
    }
    node._radianceHDRBadge.style.display = show ? "inline-block" : "none";
}

/**
 * Sync all dependent widget states based on current hdr_output + hdr_mode.
 */
function syncWidgets(node) {
    const hdrOutputW    = getWidget(node, W_HDR_OUTPUT);
    const displayTmW    = getWidget(node, W_DISPLAY_TONEMAP);
    const hdrModeW      = getWidget(node, W_HDR_MODE);
    const exportRhdrW   = getWidget(node, W_EXPORT_RHDR);

    if (!hdrOutputW) return;

    const hdrOut  = !!hdrOutputW.value;
    const hdrMode = hdrModeW?.value ?? "";
    const isCompressLog = hdrMode === "Compress (Log)";

    // HDR-only widgets are only relevant when Compress(Log) is active.
    // When the user switches to Clip(SDR) / Soft Clip / Passthrough, these
    // settings have no effect — gray them out to reduce confusion.
    setWidgetDisabled(hdrOutputW,     !isCompressLog);
    setWidgetDisabled(displayTmW,     !isCompressLog);
    setWidgetDisabled(exportRhdrW,    !isCompressLog);

    // Amber badge when hdr_output=True AND display_tonemap=None (guaranteed blown)
    const displayTmVal = displayTmW?.value ?? "ACES Filmic";
    const willBlowOut = hdrOut && displayTmVal === "None" && isCompressLog;
    setHDROutputBadge(node, willBlowOut);

    // Force canvas redraw so graying appears immediately
    app.graph.setDirtyCanvas(true, false);
}

app.registerExtension({
    name: "Radiance.VAEWidgetSync",

    nodeCreated(node) {
        // ComfyUI uses node.type in newer versions (Vue-based frontend).
        // Older versions used node.comfyClass. Check both for compatibility.
        const nodeId = node.type ?? node.comfyClass ?? "";
        if (nodeId !== HDR_DECODE_NODE) {
            return;
        }

        const hdrOutputW    = getWidget(node, W_HDR_OUTPUT);
        const displayTmW    = getWidget(node, W_DISPLAY_TONEMAP);
        const hdrModeW      = getWidget(node, W_HDR_MODE);

        if (!hdrOutputW) return;

        // Hook hdr_output changes
        const origHDRCallback = hdrOutputW.callback;
        hdrOutputW.callback = function(value) {
            syncWidgets(node);
            if (origHDRCallback) origHDRCallback.call(this, value);
        };

        // Hook hdr_mode changes (display_tonemap also depends on this)
        if (hdrModeW) {
            const origModeCallback = hdrModeW.callback;
            hdrModeW.callback = function(value) {
                syncWidgets(node);
                if (origModeCallback) origModeCallback.call(this, value);
            };
        }

        // Apply initial state (respects saved workflow values)
        // Use setTimeout to let ComfyUI finish constructing the node first
        setTimeout(() => syncWidgets(node), 0);
    },

    /**
     * After a workflow is loaded, re-sync all Radiance VAE nodes.
     * Required because widget values are restored before nodeCreated fires.
     */
    loadedGraphNode(node) {
        const nodeId = node.type ?? node.comfyClass ?? "";
        if (nodeId !== HDR_DECODE_NODE) {
            return;
        }
        setTimeout(() => syncWidgets(node), 50);
    },
});
