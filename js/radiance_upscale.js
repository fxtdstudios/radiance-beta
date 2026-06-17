import { app } from "../../../scripts/app.js";

/**
 * Radiance AI Upscale Widget Visibility (v1.0)
 * Hides SUPIR-specific widgets when a non-SUPIR model is selected.
 *
 * SUPIR-only widgets: sdxl_model_name, supir_prompt, vae, clip
 * These are shown only when model_name is one of the SUPIR model identifiers.
 *
 * Uses the same three-mechanism visibility pattern as radiance_io.js:
 *   1. widget.options.hidden      — Nodes 2.0 Vue reactive filter
 *   2. widget.hidden              — LiteGraph getLayoutWidgets() exclusion
 *   3. type="hidden"+computeSize  — physical height collapse (all widget types)
 *      + draw=()=>{}              — prevents text bleeding on STRING widgets
 *      + inputEl/element display  — hides DOM node for customtext widgets
 */

// ALBABIT-FIX: SUPIR model name identifiers — must match _SUPIR_MODELS in upscale.py.
const SUPIR_MODEL_NAMES = ["SUPIR-v0F_fp16", "SUPIR-v0Q_fp16"];

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

function setWidgetVisible(widget, visible, node) {
	if (!widget) return;

	if (!widget.options) widget.options = {};
	widget.options.hidden = !visible;

	widget.hidden = !visible;
	if (visible) {
		if (widget.type === "hidden") {
			widget.type = widget._origType || "text";
			delete widget.computeSize;
			delete widget._origComputeSize;
			if (widget._origDraw !== undefined) {
				widget.draw = widget._origDraw;
				delete widget._origDraw;
			} else {
				delete widget.draw;
			}
			if (widget.inputEl) widget.inputEl.style.display = "";
			if (widget.element)  widget.element.style.display  = "";
			if (widget._origComputedHeight !== undefined) {
				widget.computedHeight = widget._origComputedHeight;
				delete widget._origComputedHeight;
			} else {
				widget.computedHeight = 32;
			}
		}
	} else {
		if (widget.type !== "hidden") {
			widget._origType        = widget.type;
			widget._origComputeSize = widget.computeSize;
			widget._origComputedHeight = widget.computedHeight;
			widget.type = "hidden";
			widget.computeSize = () => [0, -4];
			if (widget.draw) widget._origDraw = widget.draw;
			widget.draw = function() {};
			if (widget.inputEl) widget.inputEl.style.display = "none";
			if (widget.element)  widget.element.style.display  = "none";
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

function refreshNodeSize(node) {
	if (!node.computeSize) return;

	const sz = node.computeSize();
	// ALBABIT-FIX: node.setSize(...) is the API Vue's resize handling actually
	// observes; raw node.size[i] mutation has zero visual effect.
	node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
	app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
	name: "Radiance.AIUpscale",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name !== "RadianceAIUpscale") return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
			const node = this;

			const modelWidget       = this.widgets.find(w => w.name === "model_name");
			const sdxlModelWidget   = this.widgets.find(w => w.name === "sdxl_model_name");
			const supirPromptWidget = this.widgets.find(w => w.name === "supir_prompt");
			const vaeWidget         = this.widgets.find(w => w.name === "vae");
			const clipWidget        = this.widgets.find(w => w.name === "clip");

			const updateSupirWidgets = () => {
				const isSupir = modelWidget
					? SUPIR_MODEL_NAMES.includes(modelWidget.value)
					: false;

				setWidgetVisible(sdxlModelWidget,   isSupir, node);
				setWidgetVisible(supirPromptWidget, isSupir, node);
				setWidgetVisible(vaeWidget,         isSupir, node);
				setWidgetVisible(clipWidget,        isSupir, node);

				refreshNodeSize(node);
			};

			if (modelWidget) {
				const origCb = modelWidget.callback;
				modelWidget.callback = function () {
					if (origCb) origCb.apply(this, arguments);
					updateSupirWidgets();
				};
			}

			// 100ms delay — matches radiance_io.js pattern to ensure all widgets
			// are fully initialised before the first hide/show pass.
			setTimeout(updateSupirWidgets, 100);

			return r;
		};
	}
});
