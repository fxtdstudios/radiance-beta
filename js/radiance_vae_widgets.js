/**
 * radiance_vae_widgets.js
 * v2.0 — Radiance HDR VAE Decode widget sync (Compress(Log)-only controls)
 *
 * WHAT THIS FIXES:
 *
 *   Caveat 2 (Compress Log-only controls):
 *     display_tonemap/source_space/decode_noise_scale only apply to
 *     Compress(Log) decode output -- hidden otherwise. inverse_tonemap is the
 *     reverse (hidden IN Compress(Log), where its own block never runs).
 *     hdr_scale_factor is hidden unless target_space is scene-referred (else
 *     silently ignored). hdr_output remains active in every hdr_mode
 *     (verified in hdr/vae.py) so it stays visible and unmarked. export_rhdr
 *     only captures genuinely extra (pre-tonemap) data in Compress(Log) + an
 *     active tonemap curve -- outside that it would just duplicate the image
 *     output, so it's forced off and hidden there too.
 *
 *   Additional: hdr_output=True shows a warning on hdr_output's label when
 *   either (a) hdr_mode=Compress(Log) + display_tonemap=None (no tonemap
 *   applied, unbounded values) or (b) target_space is scene-referred (not
 *   sRGB/Raw) -- both make the ComfyUI preview look overexposed/wrong
 *   (intentional for VFX pipelines that consume the raw tensor downstream).
 *
 *   temporal_overlap is hidden unless temporal_size > 0 (its own tooltip:
 *   "Only active when temporal_size > 0").
 *
 *   Post-execution: rudra_decoder gets a warning when RUDRA fell all the way
 *   back to the standard VAE (no compatible checkpoint at all), or a shorter
 *   note when a cross-architecture checkpoint was silently substituted (e.g.
 *   wan -> flux, fast_vae.py's _DECODER_TYPE_FALLBACKS) -- both read from
 *   engine.py's "ui" channel via onExecuted, since only knowable once decode
 *   actually runs.
 *
 * INSTALL:
 *   Place this file in the same folder as radiance_bootstrap.js (the custom
 *   node's web/js directory). ComfyUI auto-loads all .js files from that dir.
 *   No other changes needed.
 */

import { app } from "../../scripts/app.js";

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

// ── Widget visibility helpers (same pattern as radiance_sampler.js) ──
// ALBABIT-FIX: hidden unless their condition holds (verified against
// hdr/vae.py): target_stops needs `inverse_tonemap and hdr_mode !=
// "Compress (Log)"`; rhdr_precision needs `export_rhdr`; decoder_size needs
// `rudra_decoder=="Enabled"`; display_tonemap/source_space/decode_noise_scale
// all need `hdr_mode == "Compress (Log)"` (tonemap block, log decompression
// curve, and noise-injection gate are each individually conditioned on it);
// inverse_tonemap needs the OPPOSITE (`hdr_mode != "Compress (Log)"` --
// its own block never runs otherwise); hdr_scale_factor needs target_space to
// be scene-referred (silently ignored-with-warning otherwise). hdr_output is
// NOT hidden -- it's active in every hdr_mode, unlike v1.0's incorrect
// "Compress-Log only" grouping. export_rhdr's condition is below (separate,
// since it's also forced off, not just hidden).
function _forceWidgetReinsert(widget, node) {
    if (!node?.widgets) return;
    const idx = node.widgets.indexOf(widget);
    if (idx === -1) return;
    node.widgets.splice(idx, 1);
    node.widgets.splice(idx, 0, widget);
}

function setWidgetVisible(widget, visible, node) {
    if (!widget) return;

    if (!widget.options) widget.options = {};
    widget.options.hidden = !visible;
    widget.hidden = !visible;

    if (visible) {
        if (widget.type === "hidden") {
            widget.type = widget._origType || "number";
            if (widget._origComputeSize !== undefined) {
                widget.computeSize = widget._origComputeSize;
            } else {
                delete widget.computeSize;
            }
            delete widget._origComputeSize;
            widget.computedHeight = widget._origComputedHeight ?? 32;
            delete widget._origComputedHeight;
        }
    } else {
        if (widget.type !== "hidden") {
            widget._origType = widget.type;
            widget._origComputeSize = widget.computeSize;
            widget._origComputedHeight = widget.computedHeight;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
            widget.computedHeight = 4;
        }
    }

    _forceWidgetReinsert(widget, node);
}

function refreshNodeSize(node) {
    if (!node.computeSize) return;
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas(true, true);
}

/**
 * Sync all dependent widget states based on current hdr_output + hdr_mode.
 */
function syncWidgets(node) {
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

    if (!hdrOutputW) return;

    const hdrOut  = !!hdrOutputW.value;
    const hdrMode = hdrModeW?.value ?? "";
    const isCompressLog = hdrMode === "Compress (Log)";

    // Guaranteed-overexposure warning: hdr_output=True skips the final [0,1]
    // clamp in every hdr_mode, so the preview looks wrong for two independent
    // reasons -- (a) hdr_mode=Compress(Log) + display_tonemap=None applies no
    // tonemap at all, so raw decompressed HDR values pass through unbounded;
    // (b) target_space is scene-referred (anything but sRGB/Raw), which
    // ComfyUI's native preview always misinterprets as sRGB regardless of
    // tonemap -- confirmed by target_space's own tooltip ("Requires
    // hdr_output=True for true linear passthrough... Log/ACEScg/Rec.2020
    // spaces are scene-referred — connect to a tonemap node before SaveImage").
    const displayTmVal = displayTmW?.value ?? "ACES Filmic";
    const targetSpaceVal = targetSpaceW?.value ?? "sRGB";
    const willBlowOut = hdrOut && (
        (isCompressLog && displayTmVal === "None") ||
        !DISPLAY_READY_SPACES.has(targetSpaceVal)
    );
    _setLabelMarker(hdrOutputW, willBlowOut ? BLOWOUT_MARKER : null);

    setWidgetVisible(displayTmW, isCompressLog, node);

    // ALBABIT-FIX: force export_rhdr off before reading its value below, so
    // rhdr_precision's visibility (which depends on it) reflects the forced
    // state in the same pass instead of lagging one sync behind.
    const rhdrRedundant = !isCompressLog || displayTmVal === "None";
    if (rhdrRedundant && exportRhdrW?.value) {
        exportRhdrW.value = false;
    }
    setWidgetVisible(exportRhdrW, !rhdrRedundant, node);

    setWidgetVisible(targetStopsW, !!inverseTmW?.value && !isCompressLog, node);
    setWidgetVisible(rhdrPrecisionW, !!exportRhdrW?.value, node);
    setWidgetVisible(decoderSizeW, rudraDecoderW?.value === "Enabled", node);
    setWidgetVisible(sourceSpaceW, isCompressLog, node);
    setWidgetVisible(decodeNoiseW, isCompressLog, node);
    setWidgetVisible(inverseTmW, !isCompressLog, node);
    setWidgetVisible(hdrScaleW, !DISPLAY_READY_SPACES.has(targetSpaceVal), node);

    // ALBABIT-FIX: temporal_overlap is only read inside `if latent.ndim == 5
    // and temporal_size > 0:` (hdr/vae.py:2763) -- matches its own tooltip
    // ("Only active when temporal_size > 0").
    setWidgetVisible(temporalOverlapW, (parseInt(temporalSizeW?.value, 10) || 0) > 0, node);

    refreshNodeSize(node);
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
            [
                hdrOutputW,
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
            this.setDirtyCanvas?.(true, true);
        };
    },
});
