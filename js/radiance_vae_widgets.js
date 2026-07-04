/**
 * radiance_vae_widgets.js
 * v2.0 — Radiance HDR VAE Decode widget sync (Compress(Log)-only controls)
 *
 * WHAT THIS FIXES:
 *
 *   Caveat 2 (Compress Log-only controls):
 *     display_tonemap only applies to Compress(Log) decode output. This
 *     extension flags HDR display/export widgets for other HDR modes so
 *     their scope is visually clear.
 *
 *   Additional: hdr_output=True + display_tonemap=None + Compress(Log) shows
 *   a warning on hdr_output's label so the user knows the ComfyUI preview
 *   will look overexposed (intentional for VFX pipelines — the tensor
 *   carries raw scene-linear values).
 *
 * INSTALL:
 *   Place this file in the same folder as radiance_bootstrap.js (the custom
 *   node's web/js directory). ComfyUI auto-loads all .js files from that dir.
 *   No other changes needed.
 */

import { app } from "../../scripts/app.js";

// ALBABIT-FIX: v1.0 matched node.type/comfyClass against the display-name
// string "◎ Radiance HDR VAE Decode" (renamed since to "VAE Decode (HDR)"
// via nodes/branding.py). node.type is always the class registration key
// ("RadianceHDRVAEDecode"), never the display name -- confirmed live via
// browser console this session (app.graph._nodes.find(n=>n.type===...)
// matches class keys, not display titles). This condition never matched,
// so the whole extension was dead code since v3.1 (never grayed the
// widgets, never showed the amber badge). Also replaced the ctx.globalAlpha
// draw-override dimming and the raw node.element DOM badge (both
// LiteGraph-canvas-only techniques that don't render on the Vue "Nodes 2.0"
// frontend) with the widget.label marker convention already validated on
// Sampler/Loader/Prompt/Tonemap/Resolution this session.
const TARGET_NODE = "RadianceHDRVAEDecode";

// Widget name constants
const W_HDR_OUTPUT       = "hdr_output";
const W_DISPLAY_TONEMAP  = "display_tonemap";
const W_HDR_MODE         = "hdr_mode";
const W_EXPORT_RHDR      = "export_rhdr";

const NOT_APPLICABLE_MARKER = " (n/a: not Compress-Log)";
const BLOWOUT_MARKER        = " ⚠ PREVIEW WILL BLOW OUT";

/** Return widget by name from a node, or null. */
function getWidget(node, name) {
    return node.widgets?.find(w => w.name === name) ?? null;
}

// Same convention as radiance_resolution.js's _setLabelMarker: cache the
// original label once, then swap between origLabel and origLabel+marker.
// Renders identically on the legacy LiteGraph canvas and the Vue frontend.
function _setLabelMarker(widget, marker) {
    if (!widget) return;
    if (widget._radOrigLabel === undefined && !marker) return;
    if (widget._radOrigLabel === undefined) widget._radOrigLabel = widget.label ?? widget.name;
    const wanted = marker ? widget._radOrigLabel + marker : widget._radOrigLabel;
    if (widget.label !== wanted) widget.label = wanted;
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

    // Guaranteed-overexposure warning takes priority over the plain
    // not-applicable marker on hdr_output (mutually exclusive: blowout
    // only fires when isCompressLog is true).
    const displayTmVal = displayTmW?.value ?? "ACES Filmic";
    const willBlowOut = hdrOut && displayTmVal === "None" && isCompressLog;

    _setLabelMarker(hdrOutputW, willBlowOut ? BLOWOUT_MARKER : (!isCompressLog ? NOT_APPLICABLE_MARKER : null));
    _setLabelMarker(displayTmW, !isCompressLog ? NOT_APPLICABLE_MARKER : null);
    _setLabelMarker(exportRhdrW, !isCompressLog ? NOT_APPLICABLE_MARKER : null);

    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "Radiance.VAEWidgetSync",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            const self = this;

            const hdrOutputW = getWidget(this, W_HDR_OUTPUT);
            if (!hdrOutputW) return;

            // Wire callbacks for instant feedback on the driver widgets.
            [hdrOutputW, getWidget(this, W_HDR_MODE), getWidget(this, W_DISPLAY_TONEMAP)]
                .forEach(w => {
                    if (!w) return;
                    const origCallback = w.callback;
                    w.callback = function (...args) {
                        const res = origCallback ? origCallback.apply(this, args) : undefined;
                        syncWidgets(self);
                        return res;
                    };
                });

            // ALBABIT-FIX: poll as a state-based fallback (undo/redo, preset
            // import, workflow load) — same pattern as radiance_sampler.js's
            // preset divergence markers, covers mutation paths that bypass
            // the wrapped callbacks above.
            this._vaeSyncInterval = setInterval(() => syncWidgets(self), 250);
            const origOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (self._vaeSyncInterval) {
                    clearInterval(self._vaeSyncInterval);
                    self._vaeSyncInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            setTimeout(() => syncWidgets(self), 150);
        };

        // Re-apply after a saved workflow restores this node — onNodeCreated
        // runs before ComfyUI deserializes widget values.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            setTimeout(() => syncWidgets(self), 150);
            setTimeout(() => syncWidgets(self), 600);
        };
    },
});
