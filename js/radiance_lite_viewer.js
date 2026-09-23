import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


// ─────────────────────────────────────────────────────────────────────────────
// 3.5.0 — what changed and why
//
//  * The PNG it shows is display-referred through the same view as the Radiance
//    Viewer (sRGB untouched, linear through OCIO ACES 2.0). It used to apply
//    x/(1+x) to the whole frame, with no display encoding, whenever one pixel
//    passed 1.05.
//  * Pixel readout, clip check and diff read the node's float proxy (fp16 RHDR),
//    so they report source values, not 8-bit display codes. Coordinates are in
//    SOURCE pixels, also on a downscaled proxy.
//  * The canvas is sized in device pixels: 1:1 means one source pixel per screen
//    pixel on a scaled Windows / Retina display too.
//  * B is scaled to A when the resolutions differ; diff has a gain.
//  * Play / loop at the source fps; frames load progressively.
// ─────────────────────────────────────────────────────────────────────────────

let _H2F = null;
function halfTable() {
    if (_H2F) return _H2F;
    _H2F = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
        const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
        let v;
        if (e === 0) v = m * 2 ** -24;
        else if (e === 31) v = m ? NaN : Infinity;
        else v = (1 + m / 1024) * 2 ** (e - 15);
        _H2F[h] = s * v;
    }
    return _H2F;
}

/** Parse the node's float proxy: 12-byte RHDR header + zlib fp16/fp32. */
export async function parseFloatProxy(buffer) {
    const dv = new DataView(buffer);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== 'RHDR') throw new Error('not an RHDR proxy');
    const w = dv.getUint16(4, true), h = dv.getUint16(6, true), c = dv.getUint16(8, true), flags = dv.getUint16(10, true);
    const ds = new DecompressionStream('deflate');
    const raw = await new Response(new Blob([buffer.slice(12)]).stream().pipeThrough(ds)).arrayBuffer();
    let data;
    if (flags === 1) {
        data = new Float32Array(raw);
    } else {
        const u16 = new Uint16Array(raw);
        const t = halfTable();
        data = new Float32Array(u16.length);
        for (let i = 0; i < u16.length; i++) data[i] = t[u16[i]];
    }
    return { w, h, c, data };
}

const srgbOETF = (x) => (x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
const MODES = ['single', 'wipe', 'split', 'diff', 'onion'];

class RadianceLiteViewerUI {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.frames = [];
        this.compareFrames = [];
        this.currentFrame = 0;
        this.mode = 'single';
        this.showClip = false;
        this.showAlpha = true;
        this.wipe = 0.5;
        this.diffGain = 1;
        this.zoom = 1;          // device pixels per PREVIEW pixel
        this.panX = 0;          // device pixels
        this.panY = 0;
        this.dpr = 1;
        this.fps = 24;
        this.playing = false;
        this.encoding = 'srgb';
        this.dragging = false;
        this.lastPointer = null;
        this.generation = 0;
        this._floatCache = new Map();   // url -> Promise<proxy>
        this._derived = new Map();      // clip / diff canvases
        this._build();
        this._wire();
        this.resize();
    }

    _build() {
        this.container.className = 'radiance-lite-viewer';
        this.container.tabIndex = 0;
        this.container.innerHTML = '';
        if (!document.getElementById('radiance-lite-viewer-style')) {
            const style = document.createElement('style');
            style.id = 'radiance-lite-viewer-style';
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
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        this.container.appendChild(this.canvas);

        this.toolbar = document.createElement('div');
        this.toolbar.className = 'radiance-lite-toolbar';
        this.toolbar.innerHTML = `
            <button data-action="prev" title="Previous frame (Left)">‹</button>
            <button data-action="play" title="Play / pause (Space)">▶</button>
            <button data-action="next" title="Next frame (Right)">›</button>
            <button data-action="fit" title="Fit (F)">Fit</button>
            <button data-action="one" title="1:1 source pixels (1)">1:1</button>
            <select data-action="mode" title="Compare mode (M)">
                <option value="single">Single</option>
                <option value="wipe">Wipe</option>
                <option value="split">Split</option>
                <option value="diff">Diff</option>
                <option value="onion">Onion</option>
            </select>
            <input data-action="wipe" type="range" min="0" max="1" step="0.005" value="0.5" title="Wipe position / onion opacity">
            <select data-action="gain" title="Diff gain">
                <option value="1">Diff ×1</option>
                <option value="10">Diff ×10</option>
                <option value="100">Diff ×100</option>
            </select>
            <button data-action="clip" title="Out-of-range check on source values (C): red above, blue below">Clip</button>
            <button data-action="alpha" class="active" title="Checkerboard behind alpha (A)">Alpha</button>
            <span class="radiance-lite-spacer"></span>
            <span class="radiance-lite-status">No image</span>
        `;
        this.container.appendChild(this.toolbar);

        this.readout = document.createElement('div');
        this.readout.className = 'radiance-lite-readout';
        this.readout.textContent = 'XY -- | --';
        this.container.appendChild(this.readout);

        this.empty = document.createElement('div');
        this.empty.className = 'radiance-lite-empty';
        this.empty.textContent = 'Run the node to view';
        this.container.appendChild(this.empty);

        const q = (a) => this.toolbar.querySelector(`[data-action="${a}"]`);
        this.statusEl = this.toolbar.querySelector('.radiance-lite-status');
        this.wipeEl = q('wipe');
        this.modeEl = q('mode');
        this.gainEl = q('gain');
        this.clipBtn = q('clip');
        this.alphaBtn = q('alpha');
        this.playBtn = q('play');
    }

    _wire() {
        this.toolbar.addEventListener('click', (event) => {
            const action = event.target?.dataset?.action;
            if (!action) return;
            if (action === 'prev') this.step(-1);
            if (action === 'next') this.step(1);
            if (action === 'play') this.togglePlay();
            if (action === 'fit') this.fit();
            if (action === 'one') this.oneToOne();
            if (action === 'clip') this.toggleClip();
            if (action === 'alpha') this.toggleAlpha();
        });
        this.modeEl.addEventListener('change', () => { this.mode = this.modeEl.value; this.render(); });
        this.gainEl.addEventListener('change', () => { this.diffGain = Number(this.gainEl.value) || 1; this.render(); });
        this.wipeEl.addEventListener('input', () => { this.wipe = Number(this.wipeEl.value); this.render(); });

        this.canvas.addEventListener('pointerdown', (event) => {
            this.container.focus();
            this.dragging = true;
            this.container.classList.add('is-dragging');
            this.lastPointer = [event.clientX, event.clientY];
            this.canvas.setPointerCapture(event.pointerId);
        });
        this.canvas.addEventListener('pointermove', (event) => {
            if (this.dragging && this.lastPointer) {
                this.panX += (event.clientX - this.lastPointer[0]) * this.dpr;
                this.panY += (event.clientY - this.lastPointer[1]) * this.dpr;
                this.lastPointer = [event.clientX, event.clientY];
                this.render();
            }
            this._updatePixelReadout(event);
        });
        this.canvas.addEventListener('pointerup', (event) => {
            this.dragging = false;
            this.lastPointer = null;
            this.container.classList.remove('is-dragging');
            this.canvas.releasePointerCapture(event.pointerId);
        });
        this.canvas.addEventListener('wheel', (event) => {
            event.preventDefault();
            const [x, y] = this._devicePoint(event);
            const before = this._screenToPreview(x, y);
            const factor = Math.exp(-event.deltaY * 0.0015);
            this.zoom = Math.max(0.02, Math.min(64, this.zoom * factor));
            this.panX = x - before.x * this.zoom;
            this.panY = y - before.y * this.zoom;
            this.render();
        }, { passive: false });
        this.container.addEventListener('keydown', (event) => {
            const k = event.key;
            if (k === 'f' || k === 'F') this.fit();
            else if (k === '1') this.oneToOne();
            else if (k === ' ') { event.preventDefault(); this.togglePlay(); }
            else if (k === 'm' || k === 'M') this._cycleMode();
            else if (k === 'c' || k === 'C') this.toggleClip();
            else if (k === 'a' || k === 'A') this.toggleAlpha();
            else if (k === 'ArrowLeft') this.step(-1);
            else if (k === 'ArrowRight') this.step(1);
            else if (k === 'Home') this.setFrame(0);
            else if (k === 'End') this.setFrame(this.frames.length - 1);
            else return;
            event.stopPropagation();   // keys belong to this viewer only
        });
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.container);
    }

    destroy() {
        this.playing = false;
        if (this._raf) cancelAnimationFrame(this._raf);
        this.resizeObserver?.disconnect();
    }

    // ── loading ─────────────────────────────────────────────────────────────

    _url(name, item) {
        return api.apiURL(`/view?filename=${encodeURIComponent(name)}&subfolder=${encodeURIComponent(item.subfolder || '')}&type=${item.type || 'temp'}`);
    }

    _loadImage(item) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null);
            img.src = this._url(item.filename, item);
        });
    }

    /** Float proxy of an item (cached, at most 8). Resolves to null when absent. */
    _float(item) {
        if (!item?.float_filename) return Promise.resolve(null);
        const url = this._url(item.float_filename, item);
        if (!this._floatCache.has(url)) {
            if (this._floatCache.size >= 8) this._floatCache.delete(this._floatCache.keys().next().value);
            this._floatCache.set(url, fetch(url).then((r) => (r.ok ? r.arrayBuffer() : null))
                .then((b) => (b ? parseFloatProxy(b) : null)).catch(() => null));
        }
        return this._floatCache.get(url);
    }

    async load(payload) {
        const gen = ++this.generation;
        this.stop();
        const items = payload?.radiance_lite_images || [];
        const main = items.filter((i) => !i.is_compare);
        const cmp = items.filter((i) => i.is_compare);
        this.frames = main.map((meta) => ({ meta, image: null }));
        this.compareFrames = cmp.map((meta) => ({ meta, image: null }));
        this.currentFrame = 0;
        this._derived.clear();
        this._floatCache.clear();
        const fps = Number(payload?.fps?.[0]);
        this.fps = Number.isFinite(fps) && fps > 0 ? fps : 24;
        this.encoding = main[0]?.source_encoding || payload?.source_encoding?.[0] || 'srgb';
        this.empty.style.display = main.length ? 'none' : 'grid';

        // First frame (and its B) first, so something is on screen at once.
        if (this.frames[0]) this.frames[0].image = await this._loadImage(this.frames[0].meta);
        if (this.compareFrames[0]) this.compareFrames[0].image = await this._loadImage(this.compareFrames[0].meta);
        if (gen !== this.generation) return;
        this.fit();

        const queue = [...this.frames.slice(1), ...this.compareFrames.slice(1)];
        const worker = async () => {
            while (queue.length && gen === this.generation) {
                const f = queue.shift();
                f.image = await this._loadImage(f.meta);
            }
        };
        await Promise.all(Array.from({ length: 6 }, worker));
        if (gen === this.generation) this.render();
    }

    // ── geometry (device pixels) ────────────────────────────────────────────

    _devicePoint(event) {
        const rect = this.canvas.getBoundingClientRect();
        return [(event.clientX - rect.left) * this.dpr, (event.clientY - rect.top) * this.dpr];
    }

    _screenToPreview(x, y) {
        return { x: (x - this.panX) / this.zoom, y: (y - this.panY) / this.zoom };
    }

    resize() {
        // The canvas's own box, not the container's: the container has a 1px
        // border, and sizing from it scaled every pixel by ~0.3%.
        let rect = this.canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) rect = this.container.getBoundingClientRect();
        this.dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.round(rect.width * this.dpr));
        const h = Math.max(1, Math.round(rect.height * this.dpr));
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
            this.fit();
        } else {
            this.render();
        }
    }

    _current() { return this.frames[this.currentFrame]; }
    _compare() {
        if (!this.compareFrames.length) return null;
        return this.compareFrames[Math.min(this.currentFrame, this.compareFrames.length - 1)];
    }

    fit() {
        const f = this._current();
        if (!f?.image) { this.render(); return; }
        const W = this.canvas.width, H = this.canvas.height;
        const top = 54 * this.dpr, bottom = 28 * this.dpr, side = 10 * this.dpr;
        const iw = f.image.naturalWidth, ih = f.image.naturalHeight;
        this.zoom = Math.max(0.01, Math.min((W - 2 * side) / iw, (H - top - bottom) / ih));
        this.panX = (W - iw * this.zoom) * 0.5;
        this.panY = top + (H - top - bottom - ih * this.zoom) * 0.5;
        this.render();
    }

    /** One SOURCE pixel per device pixel (a proxy is magnified accordingly). */
    oneToOne() {
        const f = this._current();
        if (!f?.image) return;
        const W = this.canvas.width, H = this.canvas.height;
        const iw = f.image.naturalWidth, ih = f.image.naturalHeight;
        this.zoom = (f.meta.width || iw) / iw;
        this.panX = (W - iw * this.zoom) * 0.5;
        this.panY = (H - ih * this.zoom) * 0.5;
        this.render();
    }

    // ── playback ────────────────────────────────────────────────────────────

    setFrame(i) {
        if (!this.frames.length) return;
        this.currentFrame = Math.max(0, Math.min(this.frames.length - 1, i));
        this.render();
    }

    step(d) {
        if (!this.frames.length) return;
        const n = this.frames.length;
        this.setFrame((this.currentFrame + d + n) % n);
    }

    togglePlay() { if (this.playing) this.stop(); else this.play(); }

    play() {
        if (this.frames.length < 2) return;
        this.playing = true;
        this.playBtn.textContent = 'Ⅱ';
        let last = performance.now();
        const tick = (now) => {
            if (!this.playing) return;
            const interval = 1000 / this.fps;
            if (now - last >= interval) {
                last = now - ((now - last) % interval);
                this.step(1);
            }
            this._raf = requestAnimationFrame(tick);
        };
        this._raf = requestAnimationFrame(tick);
    }

    stop() {
        this.playing = false;
        if (this._raf) cancelAnimationFrame(this._raf);
        if (this.playBtn) this.playBtn.textContent = '▶';
    }

    toggleClip() { this.showClip = !this.showClip; this.clipBtn.classList.toggle('active', this.showClip); this.render(); }
    toggleAlpha() { this.showAlpha = !this.showAlpha; this.alphaBtn.classList.toggle('active', this.showAlpha); this.render(); }

    _cycleMode() {
        this.mode = MODES[(MODES.indexOf(this.mode) + 1) % MODES.length];
        this.modeEl.value = this.mode;
        this.render();
    }

    // ── drawing ─────────────────────────────────────────────────────────────

    /** Draw an image into A's preview rectangle (B is scaled to A). */
    _draw(img, alpha = 1, aw = null, ah = null) {
        const ctx = this.ctx;
        ctx.save();
        ctx.globalAlpha = alpha;
        // Smooth when zoomed out; nearest from 1:1 up, for pixel inspection.
        ctx.imageSmoothingEnabled = this.zoom < 1;
        ctx.setTransform(this.zoom, 0, 0, this.zoom, this.panX, this.panY);
        const w = aw ?? img.naturalWidth ?? img.width, h = ah ?? img.naturalHeight ?? img.height;
        ctx.drawImage(img, 0, 0, w, h);
        ctx.restore();
    }

    render() {
        const ctx = this.ctx;
        const W = this.canvas.width, H = this.canvas.height;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = '#07080b';
        ctx.fillRect(0, 0, W, H);
        const f = this._current();
        if (!f?.image) { this.statusEl.textContent = this.frames.length ? 'Loading…' : 'No image'; return; }
        const aw = f.image.naturalWidth, ah = f.image.naturalHeight;
        if (this.showAlpha) this._drawChecker(W, H);

        const c = this._compare();
        const hasB = !!c?.image;
        const mode = hasB ? this.mode : 'single';
        if (mode === 'diff') {
            const diff = this._derivedCanvas('diff', f, c);
            if (diff) this._draw(diff, 1, aw, ah);
            else this._draw(f.image);                     // until the floats land
        } else if (mode === 'wipe' || mode === 'split') {
            const x = W * (mode === 'split' ? 0.5 : this.wipe);
            this._draw(f.image);
            ctx.save();
            ctx.beginPath();
            ctx.rect(x, 0, W - x, H);
            ctx.clip();
            if (mode === 'split') { ctx.fillStyle = '#07080b'; ctx.fillRect(x, 0, W - x, H); }
            this._draw(c.image, 1, aw, ah);
            ctx.restore();
            ctx.strokeStyle = '#00bdff';
            ctx.lineWidth = this.dpr;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        } else if (mode === 'onion') {
            this._draw(f.image);
            this._draw(c.image, this.wipe, aw, ah);
        } else {
            this._draw(f.image);
        }

        if (this.showClip) {
            const clip = this._derivedCanvas('clip', f);
            if (clip) this._draw(clip, 0.85, aw, ah);
        }

        const m = f.meta;
        const proxy = m.preview_width && m.width && m.preview_width !== m.width ? ` | PROXY ${m.preview_width}x${m.preview_height}` : '';
        const cmpTxt = hasB ? ` | ${mode.toUpperCase()}${mode === 'diff' ? ` ×${this.diffGain}` : ''}` : '';
        const enc = this.encoding === 'linear' ? 'linear → ACES 2.0' : 'sRGB';
        const range = m.data_range || [0, 1];
        this.statusEl.textContent = `${this.currentFrame + 1}/${this.frames.length} | ${m.width}x${m.height}${proxy} | ${enc} | ${+this.fps.toFixed(3)} fps | range ${Number(range[0]).toFixed(3)}..${Number(range[1]).toFixed(3)}${cmpTxt}`;
    }

    _drawChecker(width, height) {
        const ctx = this.ctx;
        const size = 16 * this.dpr;
        for (let y = 0; y < height; y += size) {
            for (let x = 0; x < width; x += size) {
                ctx.fillStyle = ((Math.floor(x / size) + Math.floor(y / size)) & 1) ? '#20242b' : '#11151b';
                ctx.fillRect(x, y, size, size);
            }
        }
    }

    /**
     * Clip overlay or diff image, built from the float proxies at preview
     * resolution, cached per frame. Returns null (and schedules a render) while
     * the floats are still loading.
     */
    _derivedCanvas(kind, a, b = null) {
        const key = `${kind}|${a.meta.float_filename}|${b?.meta?.float_filename || ''}|${this.diffGain}`;
        const hit = this._derived.get(key);
        if (hit === 'pending') return null;
        if (hit) return hit;
        this._derived.set(key, 'pending');
        Promise.all([this._float(a.meta), b ? this._float(b.meta) : null]).then(([fa, fb]) => {
            if (!fa || (kind === 'diff' && !fb)) { this._derived.delete(key); return; }
            const canvas = kind === 'clip' ? this._clipCanvas(fa) : this._diffCanvas(fa, fb);
            if (this._derived.size > 16) this._derived.delete(this._derived.keys().next().value);
            this._derived.set(key, canvas);
            this.render();
        });
        return null;
    }

    _clipCanvas(fa) {
        const { w, h, c, data } = fa;
        const out = document.createElement('canvas');
        out.width = w; out.height = h;
        const img = out.getContext('2d').createImageData(w, h);
        const d = img.data;
        // sRGB sources clip at code 1.0 and crush at 0; linear sources are
        // flagged above diffuse white (1.0) and below zero.
        const srgb = this.encoding === 'srgb';
        const hi = srgb ? 0.999 : 1.0, lo = srgb ? 0.001 : 0.0;
        for (let i = 0, p = 0; i < w * h; i++, p += c) {
            const r = data[p], g = data[p + 1], bl = data[p + 2];
            const o = i * 4;
            if (r > hi || g > hi || bl > hi) { d[o] = 255; d[o + 1] = 40; d[o + 2] = 40; d[o + 3] = 230; }
            else if ((srgb ? (r <= lo && g <= lo && bl <= lo) : (r < lo || g < lo || bl < lo))) {
                d[o] = 40; d[o + 1] = 90; d[o + 2] = 255; d[o + 3] = 210;
            }
        }
        out.getContext('2d').putImageData(img, 0, 0);
        return out;
    }

    _diffCanvas(fa, fb) {
        const { w, h, c, data } = fa;
        const out = document.createElement('canvas');
        out.width = w; out.height = h;
        const img = out.getContext('2d').createImageData(w, h);
        const d = img.data;
        const sx = fb.w / w, sy = fb.h / h, cb = fb.c;
        const g = this.diffGain;
        const lin = this.encoding === 'linear';
        for (let y = 0; y < h; y++) {
            const by = Math.min(fb.h - 1, Math.floor(y * sy));
            for (let x = 0; x < w; x++) {
                const bx = Math.min(fb.w - 1, Math.floor(x * sx));
                const pa = (y * w + x) * c, pb = (by * fb.w + bx) * cb, o = (y * w + x) * 4;
                for (let k = 0; k < 3; k++) {
                    let v = Math.abs(data[pa + k] - fb.data[pb + k]) * g;
                    v = Math.min(1, lin ? srgbOETF(v) : v);
                    d[o + k] = Math.round(v * 255);
                }
                d[o + 3] = 255;
            }
        }
        out.getContext('2d').putImageData(img, 0, 0);
        return out;
    }

    async _updatePixelReadout(event) {
        const f = this._current();
        if (!f?.image) return;
        const [x, y] = this._devicePoint(event);
        const p = this._screenToPreview(x, y);
        const iw = f.image.naturalWidth, ih = f.image.naturalHeight;
        const px = Math.floor(p.x), py = Math.floor(p.y);
        if (px < 0 || py < 0 || px >= iw || py >= ih) { this.readout.textContent = 'XY -- | outside image'; return; }
        const sx = Math.floor(px * (f.meta.width || iw) / iw);
        const sy = Math.floor(py * (f.meta.height || ih) / ih);
        const fa = await this._float(f.meta);
        if (fa && px < fa.w && py < fa.h) {
            const o = (py * fa.w + px) * fa.c;
            const v = (k) => fa.data[o + k].toFixed(4);
            const a = fa.c > 3 ? ` | A ${v(3)}` : '';
            this.readout.textContent = `XY ${sx}, ${sy} | R ${v(0)} G ${v(1)} B ${v(2)}${a} | ${this.encoding}`;
        } else {
            this.readout.textContent = `XY ${sx}, ${sy} | (float values loading)`;
        }
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
