/**
 * radiance_vae_widgets.js
 * v3.0 — Radiance HDR VAE Decode mode-aware widget sync
 *
 * WHAT THIS FIXES:
 *
 *   decode_mode owns the color contract. Sampler mode hides log-only controls
 *   and RHDR export. Direct HDR / RUDRA exposes the log profile, decode noise,
 *   HDR scale, and opt-in RHDR precision while the backend fixes output to
 *   scene-linear Linear with no display tonemap.
 *
 *   temporal_overlap is hidden unless temporal_size > 0 (its own tooltip:
 *   "Only active when temporal_size > 0").
 *
 *   Post-execution (read from engine.py's "ui" channel via onExecuted, since
 *   only knowable once decode actually runs):
 *     - rudra_decoder gets a warning when RUDRA fell all the way back to the
 *       standard VAE (no compatible checkpoint at all), or a shorter note
 *       when a cross-architecture checkpoint was silently substituted (e.g.
 *       wan -> flux, fast_vae.py's _DECODER_TYPE_FALLBACKS).
 *     - hdr_output gets a warning when Compress(Log) ran with no
 *       radiance_meta on the latent (no genuine HDR-encoded source upstream,
 *       stripped by a sampler in between): the log-decompression curve then
 *       inverts values that were never log-encoded, producing genuinely
 *       wrong output. display_tonemap can't fix this, since hdr/vae.py
 *       applies it AFTER the decompression. target_space alone is NOT
 *       flagged: that's a ComfyUI-preview-only quirk, real output is
 *       unaffected. Both confirmed against real decoded pixel values.
 *
 * INSTALL:
 *   Place this file in the same folder as radiance_bootstrap.js (the custom
 *   node's web/js directory). ComfyUI auto-loads all .js files from that dir.
 *   No other changes needed.
 */

import { app } from "../../scripts/app.js";

import {
    forceWidgetReinsert as _forceWidgetReinsert,
    setWidgetVisible as _setWidgetVisible,
    getWidget,
} from "./radiance_widget_utils.js";

// Widget helpers now live in radiance_widget_utils.js; this module's only
// local difference was the "number" fallback type, which is passed through.
function setWidgetVisible(widget, visible, node) {
    return _setWidgetVisible(widget, visible, node, { fallbackType: "number" });
}

// ALBABIT-FIX: v1.0 matched node.type against the display-name string
// "◎ Radiance HDR VAE Decode" instead of the class key ("RadianceHDRVAEDecode")
// -- confirmed via live browser console that node.type is always the class
// key, so this never matched and the extension was dead code since v3.1.
// Also replaced the canvas ctx.globalAlpha dimming + raw node.element DOM
// badge (LiteGraph-only, inert on Vue) with the widget.label marker
// convention already used on Sampler/Loader/Prompt/Tonemap/Resolution.
const TARGET_NODE = "RadianceHDRVAEDecode";

// Widget name constants
const W_HDR_OUTPUT       = "hdr_output";
const W_DECODE_MODE      = "decode_mode";
const W_DISPLAY_TONEMAP  = "display_tonemap";
const W_HDR_MODE         = "hdr_mode";
const W_EXPORT_RHDR      = "export_rhdr";
const W_INVERSE_TONEMAP  = "inverse_tonemap";
const W_TARGET_STOPS     = "target_stops";
const W_RHDR_PRECISION   = "rhdr_precision";
const W_RUDRA_DECODER    = "rudra_decoder";
const W_DECODER_SIZE     = "decoder_size";
const W_TARGET_SPACE     = "target_space";
const W_SOURCE_SPACE     = "source_space";
const W_DECODE_NOISE_SCALE = "decode_noise_scale";
const W_HDR_SCALE_FACTOR   = "hdr_scale_factor";
const W_TEMPORAL_SIZE      = "temporal_size";
const W_TEMPORAL_OVERLAP   = "temporal_overlap";

// ALBABIT-FIX: mirrors nodes/generate/engine.py's _SCENE_REFERRED complement --
// only these 2 of the 12 target_space options are display-ready [0,1] sRGB;
// the other 10 (Linear, ACEScg, ACES 2065-1, Rec.2020 Linear, and the 6 log
// spaces) are scene-referred and will look wrong in ComfyUI's native preview.
const DISPLAY_READY_SPACES = new Set(["sRGB", "Raw"]);

// ALBABIT-FIX: kept short -- a long label suffix widens the whole node and
// squeezes every other widget's value column (Vue sizes the label/value
// split off the widest label in the node). display_tonemap's own tooltip
// already explains the Compress(Log) dependency on hover.
const BLOWOUT_MARKER = " ⚠ overexp risk";

// ALBABIT-FIX: post-execution only -- set from onExecuted's "ui.rudra_fallback"
// (engine.py), since whether a compatible RUDRA checkpoint exists is only
// known once decode actually runs, not from any widget's static value.
const RUDRA_FALLBACK_MARKER = " ⚠ VAE fallback";

/** Return widget by name from a node, or null. */
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


function refreshNodeSize(node) {
    if (!node.computeSize) return;
    const sz = node.computeSize();
    const newWidth = Math.max(node.size[0], sz[0]);
    const newHeight = sz[1];
    if (node.size[0] === newWidth && node.size[1] === newHeight) return;
    node.setSize([newWidth, newHeight]);
    node.setDirtyCanvas(true, true);
}

/**
 * Sync all dependent widget states based on current hdr_output + hdr_mode.
 */
function syncWidgets(node) {
    const decodeModeW     = getWidget(node, W_DECODE_MODE);
    const hdrOutputW      = getWidget(node, W_HDR_OUTPUT);
    const displayTmW      = getWidget(node, W_DISPLAY_TONEMAP);
    const hdrModeW        = getWidget(node, W_HDR_MODE);
    const exportRhdrW     = getWidget(node, W_EXPORT_RHDR);
    const inverseTmW      = getWidget(node, W_INVERSE_TONEMAP);
    const targetStopsW    = getWidget(node, W_TARGET_STOPS);
    const rhdrPrecisionW  = getWidget(node, W_RHDR_PRECISION);
    const rudraDecoderW   = getWidget(node, W_RUDRA_DECODER);
    const decoderSizeW    = getWidget(node, W_DECODER_SIZE);
    const targetSpaceW    = getWidget(node, W_TARGET_SPACE);
    const sourceSpaceW    = getWidget(node, W_SOURCE_SPACE);
    const decodeNoiseW    = getWidget(node, W_DECODE_NOISE_SCALE);
    const hdrScaleW       = getWidget(node, W_HDR_SCALE_FACTOR);
    const temporalSizeW   = getWidget(node, W_TEMPORAL_SIZE);
    const temporalOverlapW = getWidget(node, W_TEMPORAL_OVERLAP);

    if (!decodeModeW) return;

    const hdrMode = hdrModeW?.value ?? "";
    if (rudraDecoderW?.value === "Enabled" && decodeModeW.value !== "Direct HDR / RUDRA") {
        decodeModeW.value = "Direct HDR / RUDRA";
    }
    const directHDR = decodeModeW.value === "Direct HDR / RUDRA";
    const isCompressLog = directHDR || hdrMode === "Compress (Log)";
    // Overexposure-risk marking happens post-execution in onExecuted below
    // (merged from beta/main): risk depends on radiance_meta, only known once
    // the decode has actually run — not derivable from widget values here.
    const displayTmVal = displayTmW?.value ?? "ACES Filmic";
    const targetSpaceVal = targetSpaceW?.value ?? "sRGB";

    // decode_mode owns these values at execution: sampler mode is Clip/sRGB,
    // direct mode is Compress(Log)/Linear with no display tonemap.
    // MERGE-PORT (beta/main 9a5ac88): gate the node resize on actual
    // visibility transitions so the 250 ms poll stays a no-op at rest.
    let changed = false;
    if (setWidgetVisible(hdrModeW, false, node)) changed = true;
    if (setWidgetVisible(hdrOutputW, false, node)) changed = true;
    if (setWidgetVisible(displayTmW, false, node)) changed = true;
    if (setWidgetVisible(targetSpaceW, !directHDR, node)) changed = true;
    if (setWidgetVisible(exportRhdrW, directHDR, node)) changed = true;
    if (!directHDR && exportRhdrW?.value) exportRhdrW.value = false;

    if (setWidgetVisible(targetStopsW, !!inverseTmW?.value && !directHDR, node)) changed = true;
    if (setWidgetVisible(rhdrPrecisionW, !!exportRhdrW?.value, node)) changed = true;
    if (setWidgetVisible(decoderSizeW, rudraDecoderW?.value === "Enabled", node)) changed = true;
    if (setWidgetVisible(sourceSpaceW, directHDR, node)) changed = true;
    if (setWidgetVisible(decodeNoiseW, directHDR, node)) changed = true;
    if (setWidgetVisible(inverseTmW, !directHDR, node)) changed = true;
    if (setWidgetVisible(hdrScaleW, directHDR, node)) changed = true;

    // ALBABIT-FIX: temporal_overlap is only read inside 'if latent.ndim == 5
    // and temporal_size > 0:' (hdr/vae.py:2763) -- matches its own tooltip
    // ("Only active when temporal_size > 0").
    if (setWidgetVisible(temporalOverlapW, (parseInt(temporalSizeW?.value, 10) || 0) > 0, node)) changed = true;

    if (changed) refreshNodeSize(node);
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
            const decodeModeW = getWidget(this, W_DECODE_MODE);
            if (!decodeModeW) return;

            // Wire callbacks for instant feedback on the driver widgets.
            [
                hdrOutputW,
                decodeModeW,
                getWidget(this, W_HDR_MODE),
                getWidget(this, W_DISPLAY_TONEMAP),
                getWidget(this, W_INVERSE_TONEMAP),
                getWidget(this, W_EXPORT_RHDR),
                getWidget(this, W_RUDRA_DECODER),
                getWidget(this, W_DECODER_SIZE),
                getWidget(this, W_TARGET_SPACE),
                getWidget(this, W_TEMPORAL_SIZE),
            ]
                .forEach(w => {
                    if (!w) return;
                    const origCallback = w.callback;
                    w.callback = function (...args) {
                        const res = origCallback ? origCallback.apply(this, args) : undefined;
                        syncWidgets(self);
                        // ALBABIT-FIX: the RUDRA fallback marker only reflects
                        // the last completed execution -- stale as soon as
                        // either setting that affects checkpoint lookup changes.
                        if (w.name === W_RUDRA_DECODER || w.name === W_DECODER_SIZE) {
                            _setLabelMarker(getWidget(self, W_RUDRA_DECODER), null);
                        }
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

        // ALBABIT-FIX: flag a silent RUDRA→standard-VAE fallback, or a
        // cross-architecture checkpoint substitution (fast_vae.py's
        // _DECODER_TYPE_FALLBACKS, e.g. wan -> flux when no wan-specific
        // checkpoint exists), post-execution. Both are only knowable once
        // decode actually runs (engine.py's "ui.rudra_fallback"/
        // "ui.rudra_substituted_type", same "ui" side-channel convention as
        // resolution.js's computed_width) -- a console-only log is easy to
        // miss, this puts it on the node itself.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            const fellBack = !!message?.rudra_fallback?.[0];
            const substitutedType = message?.rudra_substituted_type?.[0] || "";
            let marker = null;
            if (fellBack) marker = RUDRA_FALLBACK_MARKER;
            else if (substitutedType) marker = ` ⚠ ${substitutedType} ckpt`;
            _setLabelMarker(getWidget(this, W_RUDRA_DECODER), marker);

            // ALBABIT-FIX: same "ui" side-channel convention, for hdr_output.
            // engine.py computes log_overexposure_risk from radiance_meta
            // (only known at decode time), so this only reflects the run that
            // just finished, not the current widget values.
            const overexpRisk = !!message?.log_overexposure_risk?.[0];
            _setLabelMarker(getWidget(this, W_HDR_OUTPUT), overexpRisk ? BLOWOUT_MARKER : null);

            this.setDirtyCanvas?.(true, true);
        };
    },
});
