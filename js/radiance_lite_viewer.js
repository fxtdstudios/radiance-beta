import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

class RadianceLiteViewerUI {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.frames = [];
        this.compareFrames = [];
        this.currentFrame = 0;
        this.mode = "single";
        this.showClip = false;
        this.showAlpha = true;
        this.wipe = 0.5;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.dragging = false;
        this.lastPointer = null;
        this.pixelText = "RGB --";
        this.diffCache = new Map();

        this._build();
        this._wire();
        this.resize();
    }

    _build() {
        this.container.className = "radiance-lite-viewer";
        this.container.tabIndex = 0;
        this.container.innerHTML = "";

        if (!document.getElementById("radiance-lite-viewer-style")) {
            const style = document.createElement("style");
            style.id = "radiance-lite-viewer-style";
            style.textContent = `
                .radiance-lite-viewer {
                    position: relative;
                    width: 100%;
                    height: 100%;
                    min-height: 320px;
                    overflow: hidden;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 8px;
                    background: #07080b;
                    color: #e7edf5;
                    font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    user-select: none;
                }
                .radiance-lite-viewer canvas {
                    display: block;
                    width: 100%;
                    height: 100%;
                    cursor: grab;
                }
                .radiance-lite-viewer.is-dragging canvas { cursor: grabbing; }
                .radiance-lite-toolbar {
                    position: absolute;
                    left: 10px;
                    right: 10px;
                    top: 10px;
                    z-index: 2;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    min-width: 0;
                    padding: 6px;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 7px;
                    background: rgba(9, 11, 16, 0.82);
                    backdrop-filter: blur(10px);
                }
                .radiance-lite-toolbar button,
                .radiance-lite-toolbar select {
                    height: 24px;
                    border: 1px solid rgba(255,255,255,0.14);
                    border-radius: 5px;
                    background: rgba(255,255,255,0.06);
                    color: #dbe7f3;
                    font: inherit;
                }
                .radiance-lite-toolbar button {
                    min-width: 28px;
                    padding: 0 8px;
                    cursor: pointer;
                }
                .radiance-lite-toolbar button.active {
                    border-color: rgba(0,189,255,0.55);
                    background: rgba(0,189,255,0.18);
                    color: #9ee8ff;
                }
                .radiance-lite-toolbar input[type="range"] {
                    width: 110px;
                    accent-color: #00bdff;
                }
                .radiance-lite-spacer { flex: 1 1 auto; min-width: 12px; }
                .radiance-lite-status {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    color: #aab7c4;
                    font-family: "SF Mono", Consolas, monospace;
                }
                .radiance-lite-readout {
                    position: absolute;
                    left: 10px;
                    bottom: 10px;
                    z-index: 2;
                    padding: 5px 7px;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 6px;
                    background: rgba(9, 11, 16, 0.78);
                    color: #dbe7f3;
                    font-family: "SF Mono", Consolas, monospace;
                }
                .radiance-lite-empty {
                    position: absolute;
                    inset: 0;
                    display: grid;
                    place-items: center;
                    color: #738091;
                    pointer-events: none;
                }
            `;
            document.head.appendChild(style);
        }

        this.canvas = document.createElement("canvas");
        this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
        this.container.appendChild(this.canvas);

        this.toolbar = document.createElement("div");
        this.toolbar.className = "radiance-lite-toolbar";
        this.toolbar.innerHTML = `
            <button data-action="prev" title="Previous frame">‹</button>
            <button data-action="next" title="Next frame">›</button>
            <button data-action="fit" title="Fit">Fit</button>
            <button data-action="one" title="1:1">1:1</button>
            <select data-action="mode" title="Compare mode">
                <option value="single">Single</option>
                <option value="wipe">Wipe</option>
                <option value="split">Split</option>
                <option value="diff">Diff</option>
                <option value="onion">Onion</option>
            </select>
            <input data-action="wipe" type="range" min="0" max="1" step="0.005" value="0.5" title="Wipe / onion amount">
            <button data-action="clip" title="Toggle clipping check">Clip</button>
            <button data-action="alpha" class="active" title="Toggle checkerboard alpha">Alpha</button>
            <span class="radiance-lite-spacer"></span>
            <span class="radiance-lite-status">No image</span>
        `;
        this.container.appendChild(this.toolbar);

        this.readout = document.createElement("div");
        this.readout.className = "radiance-lite-readout";
        this.readout.textContent = this.pixelText;
        this.container.appendChild(this.readout);

        this.empty = document.createElement("div");
        this.empty.className = "radiance-lite-empty";
        this.empty.textContent = "Run the node to view";
        this.container.appendChild(this.empty);

        this.statusEl = this.toolbar.querySelector(".radiance-lite-status");
        this.wipeEl = this.toolbar.querySelector('[data-action="wipe"]');
        this.modeEl = this.toolbar.querySelector('[data-action="mode"]');
        this.clipBtn = this.toolbar.querySelector('[data-action="clip"]');
        this.alphaBtn = this.toolbar.querySelector('[data-action="alpha"]');
    }

    _wire() {
        this.toolbar.addEventListener("click", (event) => {
            const action = event.target?.dataset?.action;
            if (!action) return;
            if (action === "prev") this.setFrame(this.currentFrame - 1);
            if (action === "next") this.setFrame(this.currentFrame + 1);
            if (action === "fit") this.fit();
            if (action === "one") this.oneToOne();
            if (action === "clip") {
                this.showClip = !this.showClip;
                this.clipBtn.classList.toggle("active", this.showClip);
                this.render();
            }
            if (action === "alpha") {
                this.showAlpha = !this.showAlpha;
                this.alphaBtn.classList.toggle("active", this.showAlpha);
                this.render();
            }
        });
        this.modeEl.addEventListener("change", () => {
            this.mode = this.modeEl.value;
            this.render();
        });
        this.wipeEl.addEventListener("input", () => {
            this.wipe = Number(this.wipeEl.value);
            this.render();
        });

        this.canvas.addEventListener("pointerdown", (event) => {
            this.container.focus();
            this.dragging = true;
            this.container.classList.add("is-dragging");
            this.lastPointer = [event.clientX, event.clientY];
            this.canvas.setPointerCapture(event.pointerId);
        });
        this.canvas.addEventListener("pointermove", (event) => {
            if (this.dragging && this.lastPointer) {
                const dx = event.clientX - this.lastPointer[0];
                const dy = event.clientY - this.lastPointer[1];
                this.panX += dx;
                this.panY += dy;
                this.lastPointer = [event.clientX, event.clientY];
                this.render();
            }
            this._updatePixelReadout(event);
        });
        this.canvas.addEventListener("pointerup", (event) => {
            this.dragging = false;
            this.lastPointer = null;
            this.container.classList.remove("is-dragging");
            this.canvas.releasePointerCapture(event.pointerId);
        });
        this.canvas.addEventListener("wheel", (event) => {
            event.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            const before = this._screenToImage(x, y);
            const factor = Math.exp(-event.deltaY * 0.0015);
            this.zoom = Math.max(0.05, Math.min(32, this.zoom * factor));
            this.panX = x - before.x * this.zoom;
            this.panY = y - before.y * this.zoom;
            this.render();
        }, { passive: false });
        this.container.addEventListener("keydown", (event) => {
            if (event.key === "f" || event.key === "F") this.fit();
            if (event.key === "1") this.oneToOne();
            if (event.key === "a" || event.key === "A") this._cycleMode();
            if (event.key === "c" || event.key === "C") this.clipBtn.click();
            if (event.key === "ArrowLeft") this.setFrame(this.currentFrame - 1);
            if (event.key === "ArrowRight") this.setFrame(this.currentFrame + 1);
        });

        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.container);
    }

    destroy() {
        this.resizeObserver?.disconnect();
    }

    async load(payload) {
        const items = payload?.radiance_lite_images || [];
        this.frames = await this._loadImages(items.filter((item) => !item.is_compare));
        this.compareFrames = await this._loadImages(items.filter((item) => item.is_compare));
        this.currentFrame = 0;
        this.diffCache.clear();
        this.empty.style.display = this.frames.length ? "none" : "grid";
        this.fit();
    }

    async _loadImages(items) {
        return Promise.all(items.map((item) => new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => resolve({ ...item, image: img });
            img.onerror = () => resolve({ ...item, image: null, error: true });
            img.src = api.apiURL(`/view?filename=${encodeURIComponent(item.filename)}&subfolder=${encodeURIComponent(item.subfolder || "")}&type=${item.type || "temp"}`);
        }))).then((loaded) => loaded.filter((item) => item.image));
    }

    resize() {
        const rect = this.container.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width));
        const h = Math.max(1, Math.floor(rect.height));
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
            this.fit();
        } else {
            this.render();
        }
    }

    setFrame(index) {
        if (!this.frames.length) return;
        this.currentFrame = Math.max(0, Math.min(this.frames.length - 1, index));
        this.render();
    }

    fit() {
        const frame = this.frames[this.currentFrame];
        if (!frame?.image) {
            this.render();
            return;
        }
        const rect = this.canvas.getBoundingClientRect();
        const topPad = 54;
        const bottomPad = 28;
        this.zoom = Math.min((rect.width - 20) / frame.image.naturalWidth, (rect.height - topPad - bottomPad) / frame.image.naturalHeight);
        this.zoom = Math.max(0.01, this.zoom);
        this.panX = (rect.width - frame.image.naturalWidth * this.zoom) * 0.5;
        this.panY = topPad + (rect.height - topPad - bottomPad - frame.image.naturalHeight * this.zoom) * 0.5;
        this.render();
    }

    oneToOne() {
        const frame = this.frames[this.currentFrame];
        if (!frame?.image) return;
        const rect = this.canvas.getBoundingClientRect();
        this.zoom = 1;
        this.panX = (rect.width - frame.image.naturalWidth) * 0.5;
        this.panY = (rect.height - frame.image.naturalHeight) * 0.5;
        this.render();
    }

    render() {
        const rect = this.canvas.getBoundingClientRect();
        const ctx = this.ctx;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fillStyle = "#07080b";
        ctx.fillRect(0, 0, rect.width, rect.height);

        const frame = this.frames[this.currentFrame];
        const compare = this.compareFrames[Math.min(this.currentFrame, this.compareFrames.length - 1)];
        if (!frame?.image) {
            this.statusEl.textContent = "No image";
            return;
        }

        if (this.showAlpha) this._drawChecker(rect.width, rect.height);

        const hasCompare = !!compare?.image;
        const mode = hasCompare ? this.mode : "single";
        if (mode === "diff") {
            const diff = this._getDiffCanvas(frame.image, compare.image);
            this._drawImage(diff);
        } else if (mode === "wipe") {
            this._drawImage(frame.image);
            ctx.save();
            ctx.beginPath();
            ctx.rect(rect.width * this.wipe, 0, rect.width, rect.height);
            ctx.clip();
            this._drawImage(compare.image);
            ctx.restore();
            ctx.strokeStyle = "#00bdff";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(rect.width * this.wipe, 0);
            ctx.lineTo(rect.width * this.wipe, rect.height);
            ctx.stroke();
        } else if (mode === "split") {
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, 0, rect.width * 0.5, rect.height);
            ctx.clip();
            this._drawImage(frame.image);
            ctx.restore();
            ctx.save();
            ctx.beginPath();
            ctx.rect(rect.width * 0.5, 0, rect.width * 0.5, rect.height);
            ctx.clip();
            this._drawImage(compare.image);
            ctx.restore();
        } else if (mode === "onion") {
            this._drawImage(frame.image);
            this._drawImage(compare.image, this.wipe);
        } else {
            this._drawImage(frame.image);
        }

        if (this.showClip) this._drawClipOverlay();

        const range = frame.data_range || [0, 1];
        const compareText = hasCompare ? ` | ${mode.toUpperCase()}` : "";
        this.statusEl.textContent = `${this.currentFrame + 1}/${this.frames.length} | ${frame.width || frame.image.naturalWidth}x${frame.height || frame.image.naturalHeight} | range ${Number(range[0]).toFixed(3)}..${Number(range[1]).toFixed(3)}${compareText}`;
    }

    _drawImage(image, alpha = 1) {
        const ctx = this.ctx;
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.imageSmoothingEnabled = this.zoom < 1;
        ctx.setTransform(this.zoom, 0, 0, this.zoom, this.panX, this.panY);
        ctx.drawImage(image, 0, 0);
        ctx.restore();
    }

    _drawChecker(width, height) {
        const ctx = this.ctx;
        const size = 16;
        for (let y = 0; y < height; y += size) {
            for (let x = 0; x < width; x += size) {
                ctx.fillStyle = ((x / size + y / size) & 1) ? "#20242b" : "#11151b";
                ctx.fillRect(x, y, size, size);
            }
        }
    }

    _drawClipOverlay() {
        const rect = this.canvas.getBoundingClientRect();
        const ctx = this.ctx;
        const w = Math.max(1, Math.floor(rect.width));
        const h = Math.max(1, Math.floor(rect.height));
        const source = ctx.getImageData(0, 0, w, h);
        const overlay = ctx.createImageData(w, h);
        const data = overlay.data;
        const src = source.data;
        for (let i = 0; i < data.length; i += 4) {
            const r = src[i], g = src[i + 1], b = src[i + 2];
            if (r > 250 || g > 250 || b > 250) {
                data[i] = 255; data[i + 1] = 40; data[i + 2] = 40; data[i + 3] = 210;
            } else if (r < 2 && g < 2 && b < 2) {
                data[i] = 40; data[i + 1] = 90; data[i + 2] = 255; data[i + 3] = 190;
            }
        }
        const overlayCanvas = document.createElement("canvas");
        overlayCanvas.width = w;
        overlayCanvas.height = h;
        overlayCanvas.getContext("2d").putImageData(overlay, 0, 0);
        ctx.drawImage(overlayCanvas, 0, 0);
    }

    _getDiffCanvas(a, b) {
        const key = `${a.src}|${b.src}`;
        if (this.diffCache.has(key)) return this.diffCache.get(key);
        const w = Math.min(a.naturalWidth, b.naturalWidth);
        const h = Math.min(a.naturalHeight, b.naturalHeight);
        const ca = document.createElement("canvas");
        const cb = document.createElement("canvas");
        const out = document.createElement("canvas");
        ca.width = cb.width = out.width = w;
        ca.height = cb.height = out.height = h;
        const xa = ca.getContext("2d", { willReadFrequently: true });
        const xb = cb.getContext("2d", { willReadFrequently: true });
        const xo = out.getContext("2d");
        xa.drawImage(a, 0, 0, w, h);
        xb.drawImage(b, 0, 0, w, h);
        const da = xa.getImageData(0, 0, w, h);
        const db = xb.getImageData(0, 0, w, h);
        for (let i = 0; i < da.data.length; i += 4) {
            da.data[i] = Math.abs(da.data[i] - db.data[i]);
            da.data[i + 1] = Math.abs(da.data[i + 1] - db.data[i + 1]);
            da.data[i + 2] = Math.abs(da.data[i + 2] - db.data[i + 2]);
            da.data[i + 3] = 255;
        }
        xo.putImageData(da, 0, 0);
        this.diffCache.set(key, out);
        return out;
    }

    _screenToImage(x, y) {
        return {
            x: (x - this.panX) / this.zoom,
            y: (y - this.panY) / this.zoom,
        };
    }

    _updatePixelReadout(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = Math.floor(event.clientX - rect.left);
        const y = Math.floor(event.clientY - rect.top);
        if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return;
        const px = this.ctx.getImageData(x, y, 1, 1).data;
        const img = this._screenToImage(x, y);
        this.pixelText = `XY ${Math.floor(img.x)}, ${Math.floor(img.y)} | RGB ${px[0]} ${px[1]} ${px[2]} | A ${px[3]}`;
        this.readout.textContent = this.pixelText;
    }

    _cycleMode() {
        const modes = ["single", "wipe", "split", "diff", "onion"];
        const next = modes[(modes.indexOf(this.mode) + 1) % modes.length];
        this.mode = next;
        this.modeEl.value = next;
        this.render();
    }
}

app.registerExtension({
    name: "FXTD.RadianceLiteViewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RadianceLiteViewer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            if ((this.size?.[0] || 0) < 720 || (this.size?.[1] || 0) < 460) {
                this.size = [720, 460];
            }

            const container = document.createElement("div");
            container.id = `radiance-lite-viewer-${this.id}`;
            container.style.width = "100%";
            container.style.height = "100%";
            const widget = this.addDOMWidget("lite_viewer", "lite_viewer", container, {
                serialize: false,
                hideOnZoom: false,
            });
            widget.computeSize = () => [this.size[0] - 20, this.size[1] - 80];
            this.radianceLiteViewer = new RadianceLiteViewerUI(this, container);
            this.onRemoved = () => this.radianceLiteViewer?.destroy();
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            onResize?.apply(this, arguments);
            this.radianceLiteViewer?.resize();
        };

        nodeType.prototype.onExecuted = function (message) {
            if (message?.error?.length) {
                console.error("[Radiance Lite Viewer]", message.error.join("; "));
                return;
            }
            if (!message?.radiance_lite_images?.length) return;
            this.radianceLiteViewer?.load(message);
        };
    },
});
