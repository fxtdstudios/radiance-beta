import { app } from "../../../scripts/app.js";
// FIX 5: was "../../scripts/app.js" — extensions in custom_nodes/Radiance/web/
// need three ../ to reach ComfyUI's scripts/ directory.

/**
 * Radiance Resolution — Widget Management (v2.4)
 *
 * FIX 5: Import path corrected (see above).
 * FIX 6: Switched from nodeCreated hook + nested setTimeout to beforeRegisterNodeDef
 *         which is the standard Radiance pattern. nodeCreated fires for every node
 *         in the graph including unrelated ones; beforeRegisterNodeDef targets
 *         exactly our node type before registration, avoiding the fragile 10ms
 *         race condition on widget availability.
 *
 * ALBABIT-FIX v2.5: Updated fix for Nodes 2.0 empty-space bug:
 *   1. setWidgetVisible now takes (widget, visible, node) and unconditionally
 *      forces a remove+reinsert of the widget in node.widgets via
 *      _forceWidgetReinsert — a splice(0,0) no-op alone stops working once a
 *      widget's Vue component has been (re)mounted.
 *   2. computedHeight = 4 (not -4) for hidden widgets — Vue maps this to 0px CSS height.
 *   3. refreshNodeSize uses node.setSize(...) — this is the API Vue's resize
 *      handling actually observes; raw node.size[i] mutation has zero visual effect.
 *   4. Initial toggleFields deferred 100ms — Vue must complete its first layout pass
 *      BEFORE any widget is hidden, otherwise widget.computedHeight is undefined and
 *      the restore path falls back to 32 (wrong height, causes ghost space on re-show).
 *
 * FEATURE: Shows/hides mp_target and mp_aspect_ratio widgets based on whether
 *          mp_target > 0, alongside the existing video/batch toggle.
 */

// Force Vue to destroy and recreate a widget's component instance by doing a real
// remove+re-insert in the reactive array. A splice(0,0) no-op only notifies Vue that
// the array changed but Vue's vdom differ may reuse the existing component instance
// (same object reference) and skip re-reading changed properties like `type`.
// A true remove+insert forces Vue to treat it as a new item → fresh component mount.
function _forceWidgetReinsert(widget, node) {
    if (!node?.widgets) return;
    const idx = node.widgets.indexOf(widget);
    if (idx === -1) return;
    node.widgets.splice(idx, 1);          // remove → Vue destroys component instance
    node.widgets.splice(idx, 0, widget);  // re-insert → Vue creates fresh instance
}

// ALBABIT-FIX: node param required so we can force a remove+reinsert in Vue's
// reactive widgets array. Nodes 2.0 uses widget.options.hidden to filter widgets
// from the Vue render list.
// (confirmed in ComfyUI frontend source: t.filter(e=>!(e.options?.hidden||...)))
function setWidgetVisible(widget, visible, node) {
    if (!widget) return;

    if (!widget.options) widget.options = {};
    widget.options.hidden = !visible;
    widget.hidden = !visible;

    if (visible) {
        if (widget.type === "hidden") {
            widget.type = widget._origType || "INT";
            // ALBABIT-FIX: delete override so LiteGraph prototype recalculates correctly;
            // a fallback closure gave wrong heights for toggles/combos.
            if (widget._origComputeSize !== undefined) {
                widget.computeSize = widget._origComputeSize;
            } else {
                delete widget.computeSize;
            }
            delete widget._origComputeSize;
            // ALBABIT-FIX: restore saved computedHeight for Nodes 2.0 Vue layout.
            if (widget._origComputedHeight !== undefined) {
                widget.computedHeight = widget._origComputedHeight;
                delete widget._origComputedHeight;
            } else {
                widget.computedHeight = 32;
            }
        }
    } else {
        if (widget.type !== "hidden") {
            widget._origType = widget.type;
            widget._origComputeSize = widget.computeSize;
            widget._origComputedHeight = widget.computedHeight;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
            // ALBABIT-FIX: 4 not -4 — Vue uses computedHeight for CSS; 4px collapses the row.
            widget.computedHeight = 4;
        }
    }

    // ALBABIT-FIX: always force a remove+reinsert, even if type/hidden didn't
    // change this call. Once a widget's Vue component has been (re)mounted, it
    // stops reacting to later type/hidden changes via a no-op splice(0,0) alone
    // -- it keeps rendering its previous state until reinserted again. Reinserting
    // unconditionally guarantees every widget's component reflects its current
    // state regardless of how many times it toggled before.
    _forceWidgetReinsert(widget, node);
}

// ALBABIT-FIX: Mirrors the temporal stride logic in resolution.py's generate()
// (n*stride + 1 frame counts) so video_frames <-> duration_seconds stay in sync
// when the user toggles frame_computation.
function _frameStride(modelType) {
    const m = (modelType || "").toLowerCase();
    if (m.includes("ltx")) return 8;
    if (m.includes("wan") || m.includes("hunyuan")) return 4;
    return 4;
}

function refreshNodeSize(node) {
    if (!node.computeSize) return;

    const sz = node.computeSize();
    // ALBABIT-FIX: node.setSize(...) is the API Vue's resize handling actually
    // observes; raw node.size[i] mutation has zero visual effect.
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "FXTD.Radiance.Resolution",
    // FIX 6: beforeRegisterNodeDef is the correct hook — targets our node only,
    // widgets exist immediately in onNodeCreated, no timing races.
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceResolution") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const find = (name) => this.widgets?.find(w => w.name === name);

            const enableVideoW  = find("enable_video");
            const videoFramesW  = find("video_frames");
            const frameRateW    = find("frame_rate");
            const batchSizeW    = find("batch_size");
            const mpTargetW     = find("mp_target");
            const mpAspectW     = find("mp_aspect_ratio");
            // ALBABIT-FIX: Restored from previous radiance version — frame computation
            // mode (Manual frames vs Auto seconds) toggle.
            const frameModeW    = find("frame_computation");
            const durSecW       = find("duration_seconds");
            // ALBABIT-FIX: widgets needed for preset auto-fill
            const presetW       = find("preset");
            const widthW        = find("width");
            const heightW       = find("height");
            const modelTypeW    = find("model_type");
            const scaleFactorW  = find("scale_factor");

            const toggleFields = () => {
                if (!enableVideoW) return;

                // ALBABIT-FIX: include integer 1/0 — Nodes 2.0 may store toggle values as 0/1
                const isVideo = enableVideoW.value === true
                    || enableVideoW.value === 1
                    || enableVideoW.value === "true"
                    || enableVideoW.value === "True";

                // ALBABIT-FIX: Auto (Seconds) hides video_frames in favor of duration_seconds
                const isAutoSec = frameModeW && frameModeW.value === "Auto (Seconds)";

                // ALBABIT-FIX: pass node (this) so setWidgetVisible can splice the widgets array
                setWidgetVisible(frameModeW,   isVideo, this);
                setWidgetVisible(videoFramesW, isVideo && !isAutoSec, this);
                setWidgetVisible(durSecW,      isVideo && isAutoSec, this);
                setWidgetVisible(frameRateW,   isVideo, this);
                setWidgetVisible(batchSizeW,   !isVideo, this);

                // FEATURE: mp_aspect_ratio only visible when mp_target > 0
                if (mpTargetW) {
                    const mpActive = parseFloat(mpTargetW.value) > 0;
                    setWidgetVisible(mpAspectW, mpActive, this);
                }

                refreshNodeSize(this);
            };

            // Wire callbacks
            if (enableVideoW) {
                const orig = enableVideoW.callback;
                enableVideoW.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    toggleFields();
                };
            }

            // ALBABIT-FIX: Restored from previous radiance version. Also syncs
            // video_frames <-> duration_seconds when switching modes, so the
            // hidden widget's value reflects what the other mode last computed
            // instead of going stale.
            if (frameModeW) {
                const orig = frameModeW.callback;
                frameModeW.callback = function () {
                    if (orig) orig.apply(this, arguments);

                    const fps = frameRateW ? parseFloat(frameRateW.value) || 24.0 : 24.0;
                    const stride = _frameStride(modelTypeW?.value);

                    if (frameModeW.value === "Manual (Frames)" && durSecW && videoFramesW) {
                        const raw = parseFloat(durSecW.value) * fps;
                        videoFramesW.value = Math.max(1, Math.round(raw / stride) * stride + 1);
                    } else if (frameModeW.value === "Auto (Seconds)" && durSecW && videoFramesW) {
                        durSecW.value = Math.round((videoFramesW.value / fps) * 10) / 10;
                    }

                    toggleFields();
                };
            }

            if (mpTargetW) {
                const orig = mpTargetW.callback;
                mpTargetW.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    toggleFields();
                };
            }

            // ALBABIT-FIX (étape 4): presets are now plain Cinema/Social resolutions,
            // model-agnostic. Set width/height as an immediate visual baseline; the
            // final model_type-aligned values are synced back via onExecuted below.
            if (presetW && widthW && heightW) {
                const orig = presetW.callback;

                presetW.callback = function () {
                    if (orig) orig.apply(this, arguments);

                    if (presetW.value === "Custom") return;

                    // Extract dimensions from "... (WxH)"
                    const match = presetW.value.match(/\((\d+)[x×](\d+)\)/);
                    if (match) {
                        widthW.value  = parseInt(match[1], 10);
                        heightW.value = parseInt(match[2], 10);
                        if (widthW.inputEl)  widthW.inputEl.value  = widthW.value;
                        if (heightW.inputEl) heightW.inputEl.value = heightW.value;
                    }

                    // Reset scale_factor to 1.0 — per-model spatial scale is now
                    // handled by SPATIAL_SCALE in resolution.py, not by this widget.
                    if (scaleFactorW) {
                        scaleFactorW.value = 1.0;
                    }

                    toggleFields();
                };
            }

            // ALBABIT-FIX: Defer initial toggleFields 100ms so Vue completes its first layout
            // pass before any widget is hidden. If we hide immediately, widget.computedHeight is
            // undefined — the restore path falls back to 32, creating ghost space on re-show.
            setTimeout(() => toggleFields(), 100);

            return r;
        };

        // ALBABIT-FIX: Re-apply toggle state after a saved workflow restores widget values.
        // onNodeCreated fires BEFORE ComfyUI deserialises the JSON, so a node saved with
        // enable_video=true loads stuck-collapsed without this hook.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            const reapply = () => {
                const enableVideoW = self.widgets?.find(w => w.name === "enable_video");
                const mpTargetW    = self.widgets?.find(w => w.name === "mp_target");
                const videoFramesW = self.widgets?.find(w => w.name === "video_frames");
                const frameRateW   = self.widgets?.find(w => w.name === "frame_rate");
                const batchSizeW   = self.widgets?.find(w => w.name === "batch_size");
                const mpAspectW    = self.widgets?.find(w => w.name === "mp_aspect_ratio");
                const frameModeW   = self.widgets?.find(w => w.name === "frame_computation");
                const durSecW      = self.widgets?.find(w => w.name === "duration_seconds");
                if (!enableVideoW) return;

                const isVideo = enableVideoW.value === true || enableVideoW.value === 1;
                const mpActive = mpTargetW ? parseFloat(mpTargetW.value) > 0 : false;
                const isAutoSec = frameModeW && frameModeW.value === "Auto (Seconds)";

                setWidgetVisible(frameModeW,   isVideo, self);
                setWidgetVisible(videoFramesW, isVideo && !isAutoSec, self);
                setWidgetVisible(durSecW,      isVideo && isAutoSec, self);
                setWidgetVisible(frameRateW,   isVideo, self);
                setWidgetVisible(batchSizeW,   !isVideo, self);
                setWidgetVisible(mpAspectW,    mpActive, self);
                refreshNodeSize(self);
            };
            requestAnimationFrame(reapply);
            setTimeout(reapply, 250);
        };

        // ALBABIT-FIX (étape 4): sync width/height widgets to the final
        // model_type-aligned values computed by generate() (always >= preset
        // values, rounded up — see resolution.py _align_up).
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            const newW = message?.computed_width?.[0];
            const newH = message?.computed_height?.[0];
            const widthW  = this.widgets?.find((wg) => wg.name === "width");
            const heightW = this.widgets?.find((wg) => wg.name === "height");

            if (newW != null && widthW) {
                widthW.value = newW;
                if (widthW.inputEl) widthW.inputEl.value = newW;
            }
            if (newH != null && heightW) {
                heightW.value = newH;
                if (heightW.inputEl) heightW.inputEl.value = newH;
            }
            this.setDirtyCanvas?.(true, true);
        };
    }
});
