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
 * ALBABIT-FIX v2.4: Four-part fix for Nodes 2.0 empty-space bug:
 *   1. setWidgetVisible now takes (widget, visible, node) — node.widgets.splice(0,0)
 *      triggers Vue reactive proxy re-evaluation of options.hidden.
 *   2. computedHeight = 4 (not -4) for hidden widgets — Vue maps this to 0px CSS height.
 *   3. refreshNodeSize uses in-place mutation (node.size[1] = sz[1]) — array replacement
 *      breaks Vue reactivity tracking.
 *   4. Initial toggleFields deferred 100ms — Vue must complete its first layout pass
 *      BEFORE any widget is hidden, otherwise widget.computedHeight is undefined and
 *      the restore path falls back to 32 (wrong height, causes ghost space on re-show).
 *
 * FEATURE: Shows/hides mp_target and mp_aspect_ratio widgets based on whether
 *          mp_target > 0, alongside the existing video/batch toggle.
 */

// ALBABIT-FIX: node param required so splice can trigger Vue reactive re-evaluation.
// Nodes 2.0 uses widget.options.hidden to filter widgets from the Vue render list.
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
    // ALBABIT-FIX: splice triggers Vue reactive proxy to re-evaluate options.hidden,
    // even when widget.type was already "hidden" (e.g. showing a widget on fresh load).
    if (node?.widgets) node.widgets.splice(0, 0);
}

function refreshNodeSize(node) {
    // ALBABIT-FIX: in-place element mutation required — replacing node.size with a new
    // array breaks Vue's reactive tracking of the existing array reference.
    if (node.computeSize) {
        const sz = node.computeSize();
        node.size[0] = Math.max(node.size[0], sz[0]);
        node.size[1] = sz[1];
        app.graph.setDirtyCanvas(true, true);
    }
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

            const toggleFields = () => {
                if (!enableVideoW) return;

                // ALBABIT-FIX: include integer 1/0 — Nodes 2.0 may store toggle values as 0/1
                const isVideo = enableVideoW.value === true
                    || enableVideoW.value === 1
                    || enableVideoW.value === "true"
                    || enableVideoW.value === "True";

                // ALBABIT-FIX: pass node (this) so setWidgetVisible can splice the widgets array
                setWidgetVisible(videoFramesW, isVideo, this);
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

            if (mpTargetW) {
                const orig = mpTargetW.callback;
                mpTargetW.callback = function () {
                    if (orig) orig.apply(this, arguments);
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
                if (!enableVideoW) return;

                const isVideo = enableVideoW.value === true || enableVideoW.value === 1;
                const mpActive = mpTargetW ? parseFloat(mpTargetW.value) > 0 : false;

                setWidgetVisible(videoFramesW, isVideo, self);
                setWidgetVisible(frameRateW,   isVideo, self);
                setWidgetVisible(batchSizeW,   !isVideo, self);
                setWidgetVisible(mpAspectW,    mpActive, self);
                refreshNodeSize(self);
            };
            requestAnimationFrame(reapply);
            setTimeout(reapply, 250);
        };
    }
});
