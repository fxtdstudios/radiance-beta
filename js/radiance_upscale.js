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

// Copied verbatim from radiance_io.js — shared pattern across all Radiance JS extensions.
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
	if (node?.widgets) node.widgets.splice(0, 0);
}

function refreshNodeSize(node) {
	if (node.computeSize) {
		const sz = node.computeSize();
		node.size[0] = Math.max(node.size[0], sz[0]);
		node.size[1] = sz[1];
		app.graph.setDirtyCanvas(true, true);
	}
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
