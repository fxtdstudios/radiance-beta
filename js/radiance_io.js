import { app } from "../../scripts/app.js";

/**
 * Radiance Universal I/O Widget Management (v2.3)
 * Handles dynamic visibility for Digital Cinema Read and Write nodes.
 *
 * Fixes applied:
 *  FIX 1: Node names updated to match nodes_io.py after ◎ was removed from
 *          NODE_CLASS_MAPPINGS keys. Extension was completely dead before.
 *  FIX 2: Widget hiding uses w.type = "hidden" (valid ComfyUI API) instead of
 *          "converted-widget" which is not a recognised hide token — all
 *          show/hide logic was a no-op.
 *  FIX 3: origType forEach now guards against undefined widgets with `if (w)`.
 *  FIX 4: Read node calls sourceWidget.callback() at init so label is correct
 *          when a saved workflow is restored.
 *  FIX 5: Read node isVideo check extended to include .avi .mkv .webm,
 *          matching the Python reader's accepted extensions exactly.
 *
 *  2026-07: RadianceWrite per-format widget visibility rewritten from scratch.
 *          The previous block targeted "RadianceDigitalCinemaWrite" and its
 *          pre-v3 write_mode/bit_depth/alpha_mode widgets, none of which exist
 *          on that class in the Beta (reduced to a 4-parameter shim) -- dead
 *          code that never matched anything. The full widget set now lives on
 *          "RadianceWrite" behind a single flat `format` dropdown, so the new
 *          version derives the group from the format's prefix directly.
 */

// FIX 5: Single source-of-truth for video extensions — mirrors Python read().
const VIDEO_EXTENSIONS = [".mp4", ".mov", ".gif", ".webp", ".avi", ".mkv", ".webm"];

// FIX 6: Intercept browser file inputs to allow selecting images, videos, and high-fidelity sequences (EXR/DPX/HDR)
try {
	const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "accept");
	if (descriptor && descriptor.set) {
		const originalSet = descriptor.set;
		Object.defineProperty(HTMLInputElement.prototype, "accept", {
			configurable: true,
			enumerable: true,
			get: descriptor.get,
			set: function (value) {
				if (value === "image/*" || value === "video/*") {
					value = "image/*,video/*,.exr,.dpx,.hdr";
				}
				originalSet.call(this, value);
			}
		});
	}
} catch (e) {
	console.warn("[Radiance.IO] Failed to patch HTMLInputElement.accept", e);
}

const originalClick = HTMLInputElement.prototype.click;
HTMLInputElement.prototype.click = function () {
	if (this.type === "file" && (this.accept === "image/*" || this.accept === "video/*")) {
		this.accept = "image/*,video/*,.exr,.dpx,.hdr";
	}
	return originalClick.apply(this, arguments);
};

// FIX 7: Sleek Obsidian & Neon SVG placeholders for unsupported formats (Videos, EXR, DPX, HDR Sequences)
const VIDEO_PLACEHOLDER_SVG = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="%230a0f1d"/>
      <stop offset="100%" stop-color="%2307070a"/>
    </linearGradient>
    <linearGradient id="glowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="%2300f2ff"/>
      <stop offset="100%" stop-color="%23ff007f"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="15" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <rect width="512" height="512" rx="24" fill="url(%23bgGrad)" stroke="%231a2035" stroke-width="2"/>
  <g opacity="0.05">
    <path d="M0 64h512M0 128h512M0 192h512M0 256h512M0 320h512M0 384h512M0 448h512" stroke="%23ffffff" stroke-width="1"/>
    <path d="M64 0v512M128 0v512M192 0v512M256 0v512M320 0v512M384 0v512M448 0v512" stroke="%23ffffff" stroke-width="1"/>
  </g>
  <circle cx="256" cy="220" r="80" fill="url(%23glowGrad)" opacity="0.15" filter="url(%23glow)"/>
  <g filter="url(%23glow)" transform="translate(186, 150)">
    <rect x="-10" y="-10" width="160" height="140" rx="16" fill="none" stroke="url(%23glowGrad)" stroke-width="6"/>
    <rect x="5" y="5" width="20" height="20" rx="4" fill="%2300f2ff" opacity="0.8"/>
    <rect x="115" y="5" width="20" height="20" rx="4" fill="%2300f2ff" opacity="0.8"/>
    <rect x="5" y="95" width="20" height="20" rx="4" fill="%23ff007f" opacity="0.8"/>
    <rect x="115" y="95" width="20" height="20" rx="4" fill="%23ff007f" opacity="0.8"/>
    <path d="M60 35 L100 60 L60 85 Z" fill="url(%23glowGrad)" stroke="%23ffffff" stroke-width="2" stroke-linejoin="round"/>
  </g>
  <text x="256" y="370" text-anchor="middle" fill="%23ffffff" font-family="'Inter', -apple-system, sans-serif" font-weight="800" font-size="28" letter-spacing="4">VIDEO SOURCE</text>
  <text x="256" y="405" text-anchor="middle" fill="%2300f2ff" font-family="'Courier New', monospace" font-weight="700" font-size="16" opacity="0.8">LOADED IN RADIANCE</text>
  <rect x="186" y="435" width="140" height="30" rx="15" fill="%23141a2e" stroke="%2300f2ff" stroke-width="1.5" opacity="0.9"/>
  <text x="256" y="455" text-anchor="middle" fill="%23ffffff" font-family="'Inter', -apple-system, sans-serif" font-weight="700" font-size="12" letter-spacing="1">MP4 / MOV / MKV</text>
</svg>`;

const SEQUENCE_PLACEHOLDER_SVG = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="%23110d05"/>
      <stop offset="100%" stop-color="%23070502"/>
    </linearGradient>
    <linearGradient id="glowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="%23eab308"/>
      <stop offset="100%" stop-color="%23ef4444"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="15" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <rect width="512" height="512" rx="24" fill="url(%23bgGrad)" stroke="%233a2a0a" stroke-width="2"/>
  <g opacity="0.05">
    <path d="M0 64h512M0 128h512M0 192h512M0 256h512M0 320h512M0 384h512M0 448h512" stroke="%23ffffff" stroke-width="1"/>
    <path d="M64 0v512M128 0v512M192 0v512M256 0v512M320 0v512M384 0v512M448 0v512" stroke="%23ffffff" stroke-width="1"/>
  </g>
  <circle cx="256" cy="220" r="80" fill="url(%23glowGrad)" opacity="0.15" filter="url(%23glow)"/>
  <g filter="url(%23glow)" transform="translate(196, 140)">
    <rect x="20" y="-10" width="100" height="110" rx="12" fill="%23141a2e" stroke="%23ef4444" stroke-width="3" opacity="0.5"/>
    <rect x="10" y="5" width="100" height="110" rx="12" fill="%23141a2e" stroke="%23eab308" stroke-width="3" opacity="0.8"/>
    <rect x="0" y="20" width="100" height="110" rx="12" fill="%23141a2e" stroke="url(%23glowGrad)" stroke-width="4"/>
    <circle cx="50" cy="65" r="18" fill="url(%23glowGrad)" opacity="0.8"/>
    <path d="M20 100 L45 75 L60 90 L80 65 L100 95 Z" fill="url(%23bgGrad)" opacity="0.9"/>
  </g>
  <text x="256" y="370" text-anchor="middle" fill="%23ffffff" font-family="'Inter', -apple-system, sans-serif" font-weight="800" font-size="28" letter-spacing="4">IMAGE SEQUENCE</text>
  <text x="256" y="405" text-anchor="middle" fill="%23eab308" font-family="'Courier New', monospace" font-weight="700" font-size="16" opacity="0.8">LOADED IN RADIANCE</text>
  <rect x="186" y="435" width="140" height="30" rx="15" fill="%23241c0e" stroke="%23eab308" stroke-width="1.5" opacity="0.9"/>
  <text x="256" y="455" text-anchor="middle" fill="%23ffffff" font-family="'Inter', -apple-system, sans-serif" font-weight="700" font-size="12" letter-spacing="1">EXR / DPX / HDR</text>
</svg>`;

// Helper function to resolve placeholders dynamically
function resolvePlaceholder(srcVal) {
	if (typeof srcVal !== "string") return srcVal;
	if (srcVal.includes("/api/view") || srcVal.includes("/view")) {
		try {
			const url = new URL(srcVal, window.location.origin);
			const filename = url.searchParams.get("filename") || "";
			const lowerFile = filename.toLowerCase();
			if (lowerFile.endsWith(".mp4") || lowerFile.endsWith(".mov") || lowerFile.endsWith(".mkv") || lowerFile.endsWith(".webm") || lowerFile.endsWith(".avi")) {
				return VIDEO_PLACEHOLDER_SVG;
			} else if (lowerFile.endsWith(".exr") || lowerFile.endsWith(".dpx") || lowerFile.endsWith(".hdr")) {
				return SEQUENCE_PLACEHOLDER_SVG;
			}
		} catch (e) {}
	}
	return srcVal;
}

// Hook HTMLImageElement src property to dynamically replace non-image preview assets
try {
	const srcDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
	if (srcDescriptor && srcDescriptor.set) {
		const originalSrcSet = srcDescriptor.set;
		Object.defineProperty(HTMLImageElement.prototype, "src", {
			configurable: true,
			enumerable: true,
			get: srcDescriptor.get,
			set: function (value) {
				const resolved = resolvePlaceholder(value);
				originalSrcSet.call(this, resolved);
			}
		});
	}
} catch (e) {
	console.warn("[Radiance.IO] Failed to patch HTMLImageElement.src", e);
}

// Hook Element.prototype.setAttribute as a fail-safe uploader fallback
try {
	const originalSetAttribute = Element.prototype.setAttribute;
	Element.prototype.setAttribute = function (name, value) {
		let resolved = value;
		if (name === "src" && this.tagName === "IMG") {
			resolved = resolvePlaceholder(value);
		}
		return originalSetAttribute.call(this, name, resolved);
	};
} catch (e) {
	console.warn("[Radiance.IO] Failed to patch Element.prototype.setAttribute", e);
}

// Widget visibility helpers — same pattern as radiance_vae_widgets.js /
// radiance_sampler.js (three-mechanism: options.hidden for Nodes 2.0 Vue
// filtering, type="hidden"+computeSize for legacy LiteGraph canvas, splice
// reinsert to force Vue to re-evaluate options.hidden on show).
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

app.registerExtension({
	name: "Radiance.IO",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		// ── RadianceRead: Reload button ─────────────────────────────
		if (nodeData.name === "RadianceRead") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				// Find the hidden reload widget
				const reloadWidget = this.widgets.find(w => w.name === "reload");
				if (!reloadWidget) return r;

				this.addWidget("button", "RELOAD", "reload", () => {
					reloadWidget.value = (reloadWidget.value || 0) + 1;
					this.setDirtyCanvas(true);
				});

				return r;
			};
		}

		// 1. Digital Cinema Read — label intelligence
		// FIX 1: was "◎ RadianceDigitalCinemaRead" — ◎ removed from Python mapping key.
		if (nodeData.name === "RadianceDigitalCinemaRead") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const sourceWidget = this.widgets.find(w => w.name === "source_path");

				if (sourceWidget) {
					// FIX 5: use shared VIDEO_EXTENSIONS list
					const updateLabel = () => {
						const val = (sourceWidget.value || "").toLowerCase();
						const isVideo = VIDEO_EXTENSIONS.some(ext => val.endsWith(ext));
						sourceWidget.label = isVideo ? "SOURCE (VIDEO)" : "SOURCE (SEQUENCE/IMAGE)";
					};

					sourceWidget.callback = updateLabel;

					// FIX 4: run immediately so restored workflows show the right label
					setTimeout(updateLabel, 20);
				}

				return r;
			};
		}

		// 2. RadianceWrite — per-format widget visibility
		//
		// ALBABIT-FIX: this block previously targeted "RadianceDigitalCinemaWrite"
		// and looked for widgets (write_mode, bit_depth, alpha_mode...) that only
		// existed on that class in the pre-v3 fork. In the Beta, that class was
		// reduced to a 4-parameter shim, and all the real widgets live on
		// "RadianceWrite" itself, behind a single flat `format` dropdown (prefix
		// "IMG │"/"SEQ │"/"VID │" — no separate write_mode widget). The old code
		// was dead: it never matched a class that actually has these widgets.
		// Rewritten from scratch against the current widget set, deriving the
		// group directly from the format prefix rather than a second widget.
		if (nodeData.name === "RadianceWrite") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				const node = this;

				const formatWidget       = node.widgets.find(w => w.name === "format");
				const fpsWidget          = node.widgets.find(w => w.name === "fps");
				const qualityWidget      = node.widgets.find(w => w.name === "quality");
				const exrCompWidget      = node.widgets.find(w => w.name === "exr_compression");
				const startFrameWidget   = node.widgets.find(w => w.name === "start_frame");
				const framePaddingWidget = node.widgets.find(w => w.name === "frame_padding");
				const audioSourceWidget  = node.widgets.find(w => w.name === "audio_source");

				const updateWidgets = () => {
					const fmt = formatWidget ? formatWidget.value : "";
					const isImg = fmt.startsWith("IMG");
					const isSeq = fmt.startsWith("SEQ");
					const isVid = fmt.startsWith("VID");
					const isExr = fmt.includes("EXR");
					// quality only actually does something for video CRF and
					// JPEG/WEBP images (nodes_io.py::_save_pil_image) -- every
					// other image/sequence format ignores it entirely.
					const isJpgWebp = isImg && (fmt.includes("JPEG") || fmt.includes("WEBP"));

					if (qualityWidget) {
						qualityWidget.label = isVid ? "quality (CRF)"
							: isJpgWebp ? "quality (JPEG/WEBP)"
							: "quality";
					}

					setWidgetVisible(fpsWidget,          isVid, node);
					setWidgetVisible(qualityWidget,       isVid || isJpgWebp, node);
					setWidgetVisible(exrCompWidget,       isExr, node);
					setWidgetVisible(startFrameWidget,    isSeq, node);
					setWidgetVisible(framePaddingWidget,  isSeq, node);
					setWidgetVisible(audioSourceWidget,   isVid, node);

					refreshNodeSize(node);
				};

				if (formatWidget) {
					const origCallback = formatWidget.callback;
					formatWidget.callback = function (...args) {
						const res = origCallback ? origCallback.apply(this, args) : undefined;
						updateWidgets();
						return res;
					};
				}

				// Poll as a state-based fallback (undo/redo, workflow load,
				// upstream mute/bypass) -- same pattern as radiance_vae_widgets.js.
				node._radianceWriteSyncInterval = setInterval(updateWidgets, 250);
				const origOnRemoved = node.onRemoved;
				node.onRemoved = function () {
					if (node._radianceWriteSyncInterval) {
						clearInterval(node._radianceWriteSyncInterval);
						node._radianceWriteSyncInterval = null;
					}
					if (origOnRemoved) origOnRemoved.apply(this, arguments);
				};

				setTimeout(updateWidgets, 150);
				return r;
			};

			// Re-apply after a saved workflow restores this node — onNodeCreated
			// runs before ComfyUI deserializes widget values.
			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (info) {
				const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
				const node = this;
				const formatWidget = node.widgets?.find(w => w.name === "format");
				if (formatWidget && typeof formatWidget.callback === "function") {
					setTimeout(() => formatWidget.callback(formatWidget.value), 150);
					setTimeout(() => formatWidget.callback(formatWidget.value), 600);
				}
				return r;
			};
		}
	}
});
