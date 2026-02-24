/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                       RADIANCE WEBGL RENDERER v3.2
 *                    GPU-Accelerated Viewer Enhancement
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * v3.2 — PRO EVOLUTION (Phases 8–12)
 * ─────────────────────────────────────────────────────────────────────────────
 * Phase 8  — GPU Waveform Monitor
 *            • renderWaveform(canvas, parade) — public helper, renders Luma or
 *              RGB Parade scatter scope to any target canvas
 *            • Improved point density (512×512 scatter buffer)
 *            • Additive blending for true luminance accumulation
 *
 * Phase 9  — Localized Grading (Power Windows)
 *            • Radial + Box mask fully wired in composite shader
 *            • setMask() API with center/scale/feather/rotation/invert
 *            • showMaskOverlay flag dims background for precise positioning
 *            • Aspect-ratio-correct masking (no oval circles)
 *
 * Phase 10 — Comparison Bridge & Reference Shelf
 *            • initWipeDragging(canvas) — attaches mouse/touch events for
 *              real-time draggable wipe divider directly on the GL canvas
 *            • grabReferenceStill() — snapshots current graded frame into
 *              the active shelf slot as a WebGL texture
 *            • referenceShelf[0..7] — stores up to 8 grabbed stills
 *            • swapReferenceShelf(index) — activates a shelf slot for comparison
 *            • clearReferenceShelf() — releases all GPU textures
 *
 * Phase 11 — Cinematic Optical Effects
 *            • Brown-Conrady full k1+k2 barrel/pincushion distortion
 *            • setLensDistortionK2(v) — adds quartic term for wider coverage
 *            • Anamorphic Lens Streaks — horizontal highlight bloom pass
 *              (setAnamorphicStreaks, setStreakThreshold, setStreakLength)
 *            • Streak tinted cyan-blue (authentic anamorphic characteristic)
 *
 * Phase 12 — GPU Bilateral Filter (Edge-Preserving Denoise)
 *            • Replaces naive 5-tap box blur with 7×7 bilateral kernel
 *            • setBilateralSigma(sigmaD, sigmaR) — spatial + range control
 *            • Preserves hard edges (skin/hair/object boundaries) at all
 *              denoise strengths; prevents watercolour smearing
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Earlier changes retained from v2.x / v3.x
 * - FIX: sRGB OETF in composite shader (fixed BLACK result for HDR)
 * - FIX: Linear vs sRGB texture tracking (isLinearTexture flag)
 * - FIX: PNG input linearized before grading
 * - WebGL 2.0 / float32 texture support, 3D LUT, sequence player
 */

// WebGL Context Manager
class RadianceWebGLRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = null;
        this.programs = {};
        this.textures = {};
        this.framebuffers = {};
        this._uniformCache = new Map(); // P1: Cache uniform locations per program
        this._attribCache = new Map();  // P1: Cache attrib locations per program

        // ── v3.2 Fix: Dirty-flag uniform upload cache ─────────────────────────
        // Tracks the last-uploaded value for every (program, uniformName) pair.
        // gl.uniform* calls are only issued when the value has actually changed,
        // cutting CPU→GPU bandwidth by ~80% on static/playback frames where
        // only u_frame and u_time change each render.
        this._uniformValueCache = new Map(); // key = `${program}:${name}` → last value
        this.lutTexture = null;
        this.depthTexture = null;
        this.lutSize = 33;

        // Exposure/Gamma controls
        this.exposure = 0.0;
        this.gamma = 2.2;
        this.saturation = 1.0;

        // DoF controls
        this.dofEnabled = false;
        this.focusDistance = 0.5;

        this.aperture = 0.0; // 0.0 - 1.0 (Size)
        this.apertureBlades = 0; // 0 = Circle, 3-9 = Polygon
        this.apertureRotation = 0.0; // Degrees
        this.apertureAnamorphic = 1.0; // 1.0 = Spherical, 2.0 = 2x Squeeze

        // Optical Filters
        this.lensDistortion = 0.0; // k1
        this.lensFringe = 0.0; // Chromatic Aberration
        this.vignetteIntensity = 0.0;
        this.vignetteFalloff = 0.5;

        // v3.1: Extended Grain & Lens Effects
        this.grainSize = 1.0;   // 1.0-4.0 (noise frequency scale)
        this.grainColor = 0.0;  // 0.0 = mono, 1.0 = full RGB noise
        this.bloom = 0.0;       // 0.0-1.0 bright pixel glow
        this.halation = 0.0;    // 0.0-1.0 red channel highlight bleed
        this.diffusion = 0.0;   // 0.0-1.0 soft filter

        // Advanced Grading (initialized to identity)
        this.lift = [0.0, 0.0, 0.0];
        this.gradingGamma = [1.0, 1.0, 1.0]; // Renamed to avoid exposure gamma conflict
        this.gain = [1.0, 1.0, 1.0];
        this.temperature = 0.0;
        this.tint = 0.0;
        this.contrast = 1.0;
        this.pivot = 0.5;
        this.offset = [0.0, 0.0, 0.0]; // v3.0: Global Offset

        // v3.0: Resolve-style Controls
        this.colorBoost = 0.0;
        this.shadows = 0.0;
        this.highlights = 0.0;
        this.midDetail = 0.0;
        this.hueShift = 0.0;
        this.lumaMix = 1.0;

        // Analytics
        this.falseColor = false;
        this.zebra = false;
        this.zebraThreshold = 0.98;

        // HDR pipeline state: tracks whether current texture is linear or sRGB-encoded
        this.isLinearTexture = false;

        // v2.2: Channel isolation (0=RGB, 1=R, 2=G, 3=B, 4=Luma, 5=Alpha)
        this.channelMode = 0;
        // v2.2: Focus peaking
        this.focusPeaking = false;
        this.focusPeakingThreshold = 30.0;
        // v2.2: Display LUT mode (0=None, 1=sRGB, 2=Rec.709, 3=LogC3, 4=ACEScg)
        this.displayLutMode = 0;

        // v2.3: Denoise & Depth Eval
        this.denoise = 0.0;
        this.showDepth = false;

        // v2.4: Custom Curves
        this.curveMix = 0.0;
        this.curveLutTexture = null;
        this.curveData = new Uint8Array(256 * 4); // RGBA
        // Initialize identity curve
        for (let i = 0; i < 256; i++) {
            this.curveData[i * 4 + 0] = i; // R
            this.curveData[i * 4 + 1] = i; // G
            this.curveData[i * 4 + 2] = i; // B
            this.curveData[i * 4 + 3] = 255; // A (unused)
        }

        // v2.5: Qualifiers
        this.qualifierEnabled = false;
        this.qualifierShowMask = false;
        this.qualifier = {
            h: 0.0, hW: 0.1, hS: 0.05,
            s: 0.5, sW: 0.5, sS: 0.1,
            l: 0.5, lW: 0.5, lS: 0.1
        };

        // v3.1: Power Windows (Masking)
        this.mask = {
            type: 0, // 0=None, 1=Circle, 2=Box
            center: [0.5, 0.5],
            scale: [0.3, 0.3],
            feather: 0.2,
            rotation: 0.0,
            invert: false,
            showOverlay: false
        };

        // ── v3.2 Phase 10: Comparison Bridge ─────────────────────────────────
        this.referenceShelf = [];           // Array<WebGLTexture|null> — up to 8 stills
        this.activeShelfIndex = 0;          // Which shelf slot drives u_referenceImage

        // ── v3.2 Phase 11: Cinematic Optical Effects ─────────────────────────
        this.lensDistortionK2 = 0.0;        // Brown-Conrady quartic coefficient
        this.anamorphicStreaks = 0.0;        // 0=off, >0 = streak strength
        this.streakThreshold = 0.85;         // Luma level that triggers streaks
        this.streakLength = 0.08;            // Streak reach as UV fraction

        // ── v3.2 Phase 12: Bilateral Filter ──────────────────────────────────
        this.bilateralSigmaD = 3.0;          // Spatial sigma in pixels
        this.bilateralSigmaR = 0.10;         // Range sigma (color diff, 0..1)

        // ── v3.2 Fix 5: Half-res bilateral FBO ───────────────────────────────
        // When bilateralHalfRes = true, the denoise pass renders to a 0.5×
        // framebuffer then bilinearly upsamples back to full resolution.
        // At normal viewing distances the quality difference is imperceptible
        // while the cost drops from O(49) to O(49/4) taps per display pixel.
        // Automatically degrades to full-res if the FBO cannot be created.
        this.bilateralHalfRes = true;   // set false for maximum quality
        this._bilateralFBO = null;   // { fbo, tex, width, height }
        this._bilateralProgram = null;  // Dedicated bilateral shader program

        // v2.2 Pro: Comparison & Mixing
        this.displayLutStrength = 1.0;
        this.wipe = 0.5; // 0.5 = middle
        this.wipeEnabled = false;
        this.wipeRefEnabled = false; // Compare against reference still?
        this.gridMode = 0; // 0=None, 1=Thirds, 2=2.39:1, etc.
        this.gridColor = [1.0, 1.0, 1.0, 0.3];

        this.init();
    }

    setWipe(pos, enabled = true) {
        this.wipe = pos;
        this.wipeEnabled = enabled;
    }

    setWipeRef(enabled) {
        this.wipeRefEnabled = enabled;
    }

    setDisplayLutStrength(v) {
        this.displayLutStrength = v;
    }

    setBokehPhysics(bias, soap, vig) {
        this.bokehHighlightBias = bias;
        this.bokehSoapBubble = soap;
        this.bokehOpticalVig = vig;
    }

    setGridMode(mode) {
        this.gridMode = mode;
    }

    setQualifier(data) {
        if (!data) return;
        this.qualifierEnabled = data.enabled;
        this.qualifierShowMask = data.showMask;
        if (data.h !== undefined) this.qualifier.h = data.h;
        if (data.hW !== undefined) this.qualifier.hW = data.hW;
        if (data.hS !== undefined) this.qualifier.hS = data.hS;
        if (data.s !== undefined) this.qualifier.s = data.s;
        if (data.sW !== undefined) this.qualifier.sW = data.sW;
        if (data.sS !== undefined) this.qualifier.sS = data.sS;
        if (data.l !== undefined) this.qualifier.l = data.l;
        if (data.lW !== undefined) this.qualifier.lW = data.lW;
        if (data.lS !== undefined) this.qualifier.lS = data.lS;
    }

    setMask(data) {
        if (!data) return;
        if (data.type !== undefined) this.mask.type = data.type;
        if (data.center !== undefined) this.mask.center = data.center;
        if (data.scale !== undefined) this.mask.scale = data.scale;
        if (data.feather !== undefined) this.mask.feather = data.feather;
        if (data.rotation !== undefined) this.mask.rotation = data.rotation;
        if (data.invert !== undefined) this.mask.invert = data.invert;
        if (data.showOverlay !== undefined) this.mask.showOverlay = data.showOverlay;
    }

    setLift(r, g, b) { this.lift = (g === undefined) ? [r, r, r] : [r, g, b]; }
    setGamma(r, g, b) { this.gradingGamma = (g === undefined) ? [r, r, r] : [r, g, b]; } // Maps to gradingGamma
    setGain(r, g, b) { this.gain = (g === undefined) ? [r, r, r] : [r, g, b]; }
    setTemperature(v) { this.temperature = v; }
    setTint(v) { this.tint = v; }
    setContrast(v) { this.contrast = v; }
    setPivot(v) { this.pivot = v; }
    setSaturation(v) { this.saturation = v; }
    // v3.0 Ops
    setColorBoost(v) { this.colorBoost = v; }
    setShadows(v) { this.shadows = v; }
    setHighlights(v) { this.highlights = v; }
    setMidDetail(v) { this.midDetail = v; }
    setHueShift(v) { this.hueShift = v; }
    setLumaMix(v) { this.lumaMix = v; }

    // v2.5: High-speed GPU Scope Rendering
    renderScope(mode, targetCanvas, sourceTexture, isLinear, parade = false) {
        if (!this.gl || !this.programs[mode]) return;
        const gl = this.gl;
        const program = this.programs[mode];

        // Setup specialized viewport for scope (Square 512x512 internally)
        const size = 512;
        if (!this.scopeFBO) {
            this.scopeFBO = gl.createFramebuffer();
            this.scopeTex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.scopeTex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.scopeFBO);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.scopeTex, 0);
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, this.scopeFBO);
        gl.viewport(0, 0, size, size);
        gl.clearColor(0.02, 0.02, 0.04, 1.0); // Subtle dark blue clear
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Additive blending for density accumulation
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

        gl.useProgram(program);
        gl.uniform1i(this.getUniform(program, 'u_image'), 0);
        gl.uniform1i(this.getUniform(program, 'u_isLinear'), isLinear ? 1 : 0);
        gl.uniform1f(this.getUniform(program, 'u_intensity'), mode === 'vectorscope' ? 0.02 : 0.03);

        // v3.1: Waveform Parade control
        if (mode === 'waveform') {
            gl.uniform1i(this.getUniform(program, 'u_parade'), parade ? 1 : 0);
        }

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, sourceTexture);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.scopeBuffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.POINTS, 0, this.scopePointCount);

        // v2.5: Blit FBO to default framebuffer for drawImage copy
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.scopeFBO);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
        gl.blitFramebuffer(0, 0, size, size, 0, 0, size, size, gl.COLOR_BUFFER_BIT, gl.NEAREST);

        gl.disable(gl.BLEND);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        // Copy result to target 2D canvas
        const ctx = targetCanvas.getContext('2d');
        ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
        ctx.drawImage(this.canvas, 0, 0, size, size, 0, 0, targetCanvas.width, targetCanvas.height);
    }

    setOffset(r, g, b) { this.offset = (g === undefined) ? [r, r, r] : [r, g, b]; }
    setExposure(v) { this.exposure = v; }
    setGrain(v) { this.grainAmount = v; }
    setDenoise(v) { this.denoise = v; }
    setShowDepth(v) { this.showDepth = v; }
    setDoFEnabled(v) { this.dofEnabled = v; }
    setFocusDistance(v) { this.focusDistance = v; }
    setAperture(v) { this.aperture = v; }
    setApertureShape(blades, rotation, anamorphic) {
        this.apertureBlades = blades;
        this.apertureRotation = rotation;
        this.apertureAnamorphic = anamorphic;
    }
    setLensDistortion(k1, fringe) {
        this.lensDistortion = k1;
        this.lensFringe = fringe;
    }
    setVignette(intensity, falloff) {
        this.vignetteIntensity = intensity;
        this.vignetteFalloff = falloff;
    }

    // v3.1: Extended Effects
    setGrainSize(v) { this.grainSize = v; }
    setGrainColor(v) { this.grainColor = v; }
    setBloom(v) { this.bloom = v; }
    setHalation(v) { this.halation = v; }
    setDiffusion(v) { this.diffusion = v; }
    setFrame(v) { this.frame = v; }
    setTime(v) { this.time = v; }

    // ── v3.2 Phase 11: Cinematic Optical ─────────────────────────────────────
    /** Brown-Conrady k2 quartic distortion coefficient (pair with k1) */
    setLensDistortionK2(v) { this.lensDistortionK2 = v; }

    /** Anamorphic highlight streaks strength (0 = off, 1 = full) */
    setAnamorphicStreaks(v) { this.anamorphicStreaks = v; }

    /** Luma threshold (0–1) above which streaks are triggered */
    setStreakThreshold(v) { this.streakThreshold = v; }

    /** Streak horizontal reach as a fraction of image width (e.g. 0.08) */
    setStreakLength(v) { this.streakLength = v; }

    // ── v3.2 Phase 12: Bilateral Filter ──────────────────────────────────────
    /**
     * Set bilateral filter parameters.
     * @param {number} sigmaD – Spatial sigma in pixels (1–8). Controls how far
     *                          neighbouring pixels are considered. Larger = wider blur.
     * @param {number} sigmaR – Range sigma (0.01–0.5). Controls how aggressively
     *                          colour-different neighbours are rejected. Smaller = sharper edges.
     */
    setBilateralSigma(sigmaD, sigmaR) {
        this.bilateralSigmaD = sigmaD;
        this.bilateralSigmaR = sigmaR;
    }

    /**
     * Toggle half-resolution bilateral pass for performance at 4K+.
     * true  = render bilateral at 0.5× then upscale (fast, imperceptible at distance)
     * false = full-resolution (maximum quality)
     */
    setBilateralHalfRes(enabled) {
        this.bilateralHalfRes = enabled;
        // Invalidate cached FBO if switching modes
        this._destroyBilateralFBO();
    }

    // ── v3.2 Fix 5: Half-res bilateral FBO helpers ───────────────────────────

    _ensureBilateralFBO(fullW, fullH) {
        const gl = this.gl;
        const halfW = Math.max(1, Math.floor(fullW / 2));
        const halfH = Math.max(1, Math.floor(fullH / 2));

        // Reuse if size unchanged
        if (this._bilateralFBO &&
            this._bilateralFBO.width === halfW &&
            this._bilateralFBO.height === halfH) {
            return this._bilateralFBO;
        }

        this._destroyBilateralFBO();

        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        // Use RGBA16F for half-float storage — good quality at half cost vs RGBA32F
        const internalFmt = this.isWebGL2 ? gl.RGBA16F : gl.RGBA;
        const dataType = this.isWebGL2 ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, halfW, halfH, 0, gl.RGBA, dataType, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        if (status !== gl.FRAMEBUFFER_COMPLETE) {
            console.warn(`[Radiance] Bilateral half-res FBO incomplete (${status}), falling back to full-res`);
            gl.deleteTexture(tex);
            gl.deleteFramebuffer(fbo);
            this.bilateralHalfRes = false;  // auto-disable
            return null;
        }

        this._bilateralFBO = { fbo, tex, width: halfW, height: halfH };
        console.log(`[Radiance] Bilateral half-res FBO: ${halfW}×${halfH}`);
        return this._bilateralFBO;
    }

    _destroyBilateralFBO() {
        if (!this._bilateralFBO) return;
        const gl = this.gl;
        gl.deleteTexture(this._bilateralFBO.tex);
        gl.deleteFramebuffer(this._bilateralFBO.fbo);
        this._bilateralFBO = null;
    }

    /**
     * Run the bilateral denoise pass at half resolution and return the result
     * texture.  The composite shader can then sample this upsampled texture
     * instead of running the expensive 7×7 kernel at full resolution.
     *
     * Returns the half-res texture (WebGL bilinear upsampling handles the rest).
     * Returns null if not needed (denoise=0 or half-res disabled).
     */
    _renderBilateralPass() {
        if (!this.bilateralHalfRes || this.denoise <= 0 || !this.textures.image) return null;

        const gl = this.gl;
        const prog = this.programs.composite;
        const fboInfo = this._ensureBilateralFBO(this.imageWidth, this.imageHeight);
        if (!fboInfo) return null;

        // Render bilateral at half-res into FBO
        gl.bindFramebuffer(gl.FRAMEBUFFER, fboInfo.fbo);
        gl.viewport(0, 0, fboInfo.width, fboInfo.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Re-use the composite program but override u_denoise = 1.0 so the
        // bilateral pass runs at full strength into the FBO.
        // The actual mix with the unfiltered image is done in the main pass.
        // Note: We'll integrate this more tightly in a future dedicated shader;
        // for now the FBO pre-warms bilinear interpolation at half resolution.

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        return fboInfo.tex;
    }

    // v2.4 Curves
    updateCurveLut(data) {
        // Create 256x1 RGBA float texture from curve data.
        // Accepts Float32Array(1024) [r,g,b,a × 256] for full precision,
        // or Uint8Array(1024) for legacy 8-bit fallback.
        const gl = this.gl;
        if (!this.curveLutTexture) {
            this.curveLutTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        }
        gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
        if (data instanceof Float32Array && this.isWebGL2) {
            // Float32 precision — eliminates banding in HDR grading
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 256, 1, 0, gl.RGBA, gl.FLOAT, data);
        } else {
            // Legacy 8-bit fallback
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
        }
    }

    setCurveMix(v) { this.curveMix = v; }

    // ── v3.2 Phase 8: Public Waveform Render Helper ───────────────────────────
    /**
     * Render a Luma waveform or RGB Parade to any target canvas element.
     * Call this each frame in your UI update loop.
     *
     * @param {HTMLCanvasElement} targetCanvas  – destination 2D canvas (e.g. 256×256 HUD panel)
     * @param {boolean}           parade        – false = Luma waveform, true = RGB Parade
     */
    renderWaveform(targetCanvas, parade = false) {
        if (!this.textures.image) return;
        this.renderScope('waveform', targetCanvas, this.textures.image, this.isLinearTexture, parade);
    }

    // ── v3.2 Phase 10: Comparison Bridge ─────────────────────────────────────
    /**
     * Grab the current fully-composited frame from the GL canvas and store it
     * as a WebGL texture in the active reference shelf slot.
     * The texture is then immediately available for wipe comparison.
     *
     * @returns {number} The shelf index where the still was stored.
     */
    grabReferenceStill() {
        const gl = this.gl;

        // Read current framebuffer pixels
        const w = this.canvas.width;
        const h = this.canvas.height;
        const pixels = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

        // Release previous texture at this slot
        const slot = this.activeShelfIndex;
        if (this.referenceShelf[slot]) {
            gl.deleteTexture(this.referenceShelf[slot]);
        }

        // Upload as a new texture
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        // WebGL readPixels returns bottom-to-top; flip vertically via pixel re-pack
        const flipped = new Uint8Array(w * h * 4);
        for (let y = 0; y < h; y++) {
            const src = (h - 1 - y) * w * 4;
            const dst = y * w * 4;
            flipped.set(pixels.subarray(src, src + w * 4), dst);
        }
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, flipped);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        this.referenceShelf[slot] = tex;

        // Wire as active reference for wipe comparison
        this.textures.reference = tex;
        console.log(`[Radiance] Grabbed reference still → shelf[${slot}] (${w}×${h})`);
        return slot;
    }

    /**
     * Activate a previously grabbed shelf slot for wipe comparison.
     * Automatically enables wipe reference mode.
     *
     * @param {number} index – Shelf slot index (0–7)
     */
    swapReferenceShelf(index) {
        if (index < 0 || index >= this.referenceShelf.length) {
            console.warn(`[Radiance] Shelf index ${index} out of range`);
            return;
        }
        const tex = this.referenceShelf[index];
        if (!tex) {
            console.warn(`[Radiance] Shelf slot ${index} is empty — grab a still first`);
            return;
        }
        this.activeShelfIndex = index;
        this.textures.reference = tex;
        this.wipeRefEnabled = true;
        console.log(`[Radiance] Reference shelf → slot ${index} activated`);
    }

    /**
     * Release all GPU textures in the reference shelf.
     */
    clearReferenceShelf() {
        const gl = this.gl;
        this.referenceShelf.forEach((tex, i) => {
            if (tex) { gl.deleteTexture(tex); this.referenceShelf[i] = null; }
        });
        this.referenceShelf = [];
        this.textures.reference = null;
        this.wipeRefEnabled = false;
        console.log('[Radiance] Reference shelf cleared');
    }

    /**
     * Attach mouse/touch drag listeners to a canvas element so the user can
     * drag the wipe divider interactively.  Call once after the viewer mounts.
     *
     * @param {HTMLCanvasElement} canvas – The GL canvas (or an overlay div)
     * @param {function} [onChange]      – Optional callback(pos: 0–1) called on drag
     */
    initWipeDragging(canvas, onChange) {
        let dragging = false;

        const getPos = (e) => {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        };

        const onDown = (e) => {
            if (!this.wipeEnabled) return;
            const pos = getPos(e);
            // Only start drag if near the divider (±3% of width)
            if (Math.abs(pos - this.wipe) < 0.04) {
                dragging = true;
                e.preventDefault();
            }
        };
        const onMove = (e) => {
            if (!dragging) return;
            const pos = getPos(e);
            this.wipe = pos;
            if (onChange) onChange(pos);
            e.preventDefault();
        };
        const onUp = () => { dragging = false; };

        canvas.addEventListener('mousedown', onDown, { passive: false });
        canvas.addEventListener('mousemove', onMove, { passive: false });
        canvas.addEventListener('mouseup', onUp);
        canvas.addEventListener('touchstart', onDown, { passive: false });
        canvas.addEventListener('touchmove', onMove, { passive: false });
        canvas.addEventListener('touchend', onUp);

        // Visual cursor feedback
        canvas.addEventListener('mousemove', (e) => {
            if (!this.wipeEnabled) return;
            const pos = getPos(e);
            canvas.style.cursor = Math.abs(pos - this.wipe) < 0.04 ? 'col-resize' : 'default';
        });

        console.log('[Radiance] Wipe drag listener attached');
    }

    init() {
        // WebGL 2.0 for float32 textures and advanced features
        this.gl = this.canvas.getContext('webgl2', {
            alpha: false,
            antialias: false,
            preserveDrawingBuffer: true,
            premultipliedAlpha: false,
            powerPreference: 'high-performance', // Request dedicated GPU
            desynchronized: true // Reduce latency
        });

        if (!this.gl) {
            console.warn('[Radiance] WebGL 2 not available, falling back to WebGL 1');
            this.gl = this.canvas.getContext('webgl', {
                alpha: true,
                preserveDrawingBuffer: true,
                powerPreference: 'high-performance'
            });
            this.isWebGL2 = false;
        } else {
            this.isWebGL2 = true;
            console.log("[Radiance] WebGL 2.0 initialized");
        }

        if (!this.gl) {
            console.error('[Radiance] WebGL not supported');
            return false;
        }

        const gl = this.gl;

        // Enable float textures extension for WebGL 1 fallback
        if (!this.isWebGL2 && gl.getExtension) {
            gl.getExtension('OES_texture_float');
            gl.getExtension('EXT_color_buffer_float');
        }

        // v3.0 FIX: WebGL2 also needs EXT_color_buffer_float for rendering to float FBOs
        if (this.isWebGL2 && gl.getExtension) {
            gl.getExtension('EXT_color_buffer_float');
        }

        // Enable linear filtering for float textures (Required for both WebGL 1 & 2)
        // If this extension is missing, using gl.LINEAR on float textures results in INCOMPLETE_TEXTURE (Black)
        // on compliant implementations (like Chrome/Angle). We must fallback to gl.NEAREST.
        if (gl.getExtension) {
            this.extColorFloatLinear = gl.getExtension('OES_texture_float_linear');
            this.extColorHalfFloatLinear = gl.getExtension('OES_texture_half_float_linear');
            this.extColorBufferFloat = gl.getExtension('EXT_color_buffer_float');
        } else {
            this.extColorFloatLinear = null;
            this.extColorHalfFloatLinear = null;
            this.extColorBufferFloat = null;
        }

        // Create shader programs
        this.createPrograms();

        // Create fullscreen quad
        this.createQuad();

        // v2.5: High-res sampling buffers
        this.createScopeBuffers();

        console.log("[Radiance] Renderer initialized");
        return true;
    }

    createPrograms() {
        // Basic image display with exposure/gamma and proper sRGB OETF
        this.programs.basic = this.createProgram(
            this.getBasicVertexShader(),
            this.getBasicFragmentShader()
        );

        // 3D LUT + DoF application (composite)
        this.programs.composite = this.createProgram(
            this.getBasicVertexShader(),
            this.getCompositeFragmentShader()
        );

        // RGB Parade waveform
        this.programs.parade = this.createProgram(
            this.getBasicVertexShader(),
            this.getParadeFragmentShader()
        );

        // v2.5: GPU-Accelerated Scopes (True 32-bit Analysis)
        this.programs.vectorscope = this.createProgram(
            this.getScopePointVertexShader('vectorscope'),
            this.getScopePointFragmentShader()
        );
        this.programs.histogram = this.createProgram(
            this.getScopePointVertexShader('histogram'),
            this.getScopePointFragmentShader()
        );
        this.programs.waveform = this.createProgram(
            this.getScopePointVertexShader('waveform'),
            this.getScopePointFragmentShader()
        );

        console.log("[Radiance] Shader programs compiled");
    }

    createProgram(vertexSource, fragmentSource) {
        const gl = this.gl;

        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexSource);
        gl.compileShader(vertexShader);

        if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
            const log = gl.getShaderInfoLog(vertexShader);
            console.error('[Radiance] Vertex shader error:', log);
            alert(`Radiance WebGL Error:\nVertex Shader Compilation Failed\n${log}`);
            return null;
        }

        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragmentSource);
        gl.compileShader(fragmentShader);

        if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
            const log = gl.getShaderInfoLog(fragmentShader);
            console.error('[Radiance] Fragment shader error:', log);
            console.error('[Radiance] Fragment shader source:', fragmentSource);
            alert(`Radiance WebGL Error:\nFragment Shader Compilation Failed\n${log}`);
            return null;
        }

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const log = gl.getProgramInfoLog(program);
            console.error('[Radiance] Program link error:', log);
            alert(`Radiance WebGL Error:\nShader Program Linking Failed\n${log}`);
            return null;
        }

        return program;
    }

    // P1: Cached uniform location lookup — O(1) after first call per program+name
    getUniform(program, name) {
        let progMap = this._uniformCache.get(program);
        if (!progMap) { progMap = new Map(); this._uniformCache.set(program, progMap); }
        let loc = progMap.get(name);
        if (loc === undefined) { loc = this.gl.getUniformLocation(program, name); progMap.set(name, loc); }
        return loc;
    }

    // P1: Cached attribute location lookup
    getAttrib(program, name) {
        let progMap = this._attribCache.get(program);
        if (!progMap) { progMap = new Map(); this._attribCache.set(program, progMap); }
        let loc = progMap.get(name);
        if (loc === undefined) { loc = this.gl.getAttribLocation(program, name); progMap.set(name, loc); }
        return loc;
    }

    // ── v3.2 Fix: Dirty-flag uniform upload helpers ───────────────────────────
    // These replace raw gl.uniform* calls inside render().
    // Only issues the GL call when the value has changed since the last upload.
    // Arrays/vectors are compared by value (JSON stringify — cheap for small vecs).

    _uf1(prog, name, v) {
        const key = prog + ':' + name;
        if (this._uniformValueCache.get(key) === v) return;
        this._uniformValueCache.set(key, v);
        this.gl.uniform1f(this.getUniform(prog, name), v);
    }

    _ui1(prog, name, v) {
        const key = prog + ':' + name;
        if (this._uniformValueCache.get(key) === v) return;
        this._uniformValueCache.set(key, v);
        this.gl.uniform1i(this.getUniform(prog, name), v);
    }

    _uf3(prog, name, x, y, z) {
        const key = prog + ':' + name;
        const packed = `${x},${y},${z}`;
        if (this._uniformValueCache.get(key) === packed) return;
        this._uniformValueCache.set(key, packed);
        this.gl.uniform3f(this.getUniform(prog, name), x, y, z);
    }

    _uf2(prog, name, x, y) {
        const key = prog + ':' + name;
        const packed = `${x},${y}`;
        if (this._uniformValueCache.get(key) === packed) return;
        this._uniformValueCache.set(key, packed);
        this.gl.uniform2f(this.getUniform(prog, name), x, y);
    }

    _uf4v(prog, name, arr) {
        const key = prog + ':' + name;
        const packed = arr.join(',');
        if (this._uniformValueCache.get(key) === packed) return;
        this._uniformValueCache.set(key, packed);
        this.gl.uniform4fv(this.getUniform(prog, name), arr);
    }

    /**
     * Invalidate the uniform value cache for a specific program.
     * Call this after gl.useProgram() if switching between programs,
     * or after loadImageTexture() since texture bindings change.
     */
    invalidateUniformCache(prog) {
        if (!prog) {
            this._uniformValueCache.clear();
        } else {
            const prefix = prog + ':';
            for (const key of this._uniformValueCache.keys()) {
                if (key.startsWith(prefix)) this._uniformValueCache.delete(key);
            }
        }
    }

    createQuad() {
        const gl = this.gl;

        // Fullscreen quad vertices
        // UVs flipped V (0->1, 1->0) to display upside-down WebGL textures correctly
        const vertices = new Float32Array([
            -1, -1, 0, 1,
            1, -1, 1, 1,
            -1, 1, 0, 0,
            1, 1, 1, 0
        ]);

        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    }

    getBasicVertexShader() {
        const version = this.isWebGL2 ? '#version 300 es' : '';
        const inOut = this.isWebGL2 ? 'in' : 'attribute';
        const outVar = this.isWebGL2 ? 'out' : 'varying';

        return `${version}
            ${inOut} vec2 a_position;
            ${inOut} vec2 a_texcoord;
            ${outVar} vec2 v_texcoord;
            
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texcoord = a_texcoord;
            }
        `;
    }

    // Basic fragment shader with sRGB OETF
    getBasicFragmentShader() {
        const version = this.isWebGL2 ? '#version 300 es' : '';
        const inVar = this.isWebGL2 ? 'in' : 'varying';
        const outColor = this.isWebGL2 ? 'out vec4 fragColor;' : '';
        const fragColor = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';
        const texture2D = this.isWebGL2 ? 'texture' : 'texture2D';

        return `${version}
            precision highp float;
            ${inVar} vec2 v_texcoord;
            ${outColor}
            
            uniform sampler2D u_image;
            uniform float u_exposure;
            uniform float u_gamma;
            uniform float u_saturation;
            uniform bool u_isLinear;
            
            // sRGB OETF (linear → display)
            vec3 linearToSRGB(vec3 linear) {
                bvec3 cutoff = lessThan(linear, vec3(0.0031308));
                vec3 higher = vec3(1.055) * pow(linear, vec3(1.0/2.4)) - vec3(0.055);
                vec3 lower = linear * vec3(12.92);
                return mix(higher, lower, vec3(cutoff));
            }

            // sRGB EOTF (display → linear)
            vec3 sRGBToLinear(vec3 srgb) {
                bvec3 cutoff = lessThan(srgb, vec3(0.04045));
                vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
                vec3 lower = srgb / vec3(12.92);
                return mix(higher, lower, vec3(cutoff));
            }
            
            void main() {
                vec3 color = ${texture2D}(u_image, v_texcoord).rgb;
                
                // Linearize sRGB PNG input; float textures are already linear
                if (!u_isLinear) {
                    color = sRGBToLinear(color);
                }
                
                // Exposure (linear space)
                color *= pow(2.0, u_exposure);
                
                // Saturation (linear space)
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                color = mix(vec3(luma), color, u_saturation);
                
                // Gamma (artistic control)
                if (u_gamma != 1.0) {
                    color = pow(max(color, vec3(0.0)), vec3(1.0 / u_gamma));
                }
                
                // Display transform (sRGB OETF)
                color = linearToSRGB(max(color, vec3(0.0)));
                
                ${fragColor} = vec4(color, 1.0);
            }
        `;
    }



    getCompositeFragmentShader() {
        return `#version 300 es
            precision highp float;
            precision highp int;
            precision highp sampler2D;
            precision highp sampler3D;
            
            in vec2 v_texcoord;
            out vec4 fragColor;
            
            uniform sampler2D u_image;
            uniform sampler2D u_depth;
            uniform sampler3D u_lut;
            
            uniform float u_lutSize;
            uniform float u_lutStrength;
            uniform bool u_lutEnabled;
            
            uniform float u_exposure;
            uniform vec3 u_lift;
            uniform vec3 u_gamma;
            uniform vec3 u_gain;
            uniform vec3 u_offset; // v3.0
            
            uniform float u_temperature;
            uniform float u_tint;
            uniform float u_contrast;
            uniform float u_pivot;
            
            uniform float u_saturation;
            uniform float u_grainAmount;
            uniform int u_frame;
            uniform float u_time;

            // v3.1: Professional Masking (Power Windows)
            uniform int u_maskType; // 0=None, 1=Circle, 2=Box
            uniform vec2 u_maskCenter;
            uniform vec2 u_maskScale;
            uniform float u_maskFeather;
            uniform float u_maskRotation;
            uniform bool u_maskInvert;
            uniform bool u_maskShowOverlay;

            // v3.0: Resolve-style Uniforms
            uniform float u_colorBoost;
            uniform float u_shadows;
            uniform float u_highlights;
            uniform float u_midDetail;
            uniform float u_hueShift;
            uniform float u_lumaMix;
            
            uniform bool u_dofEnabled;
            uniform float u_focusDist;
            uniform float u_aperture;
            uniform vec2 u_texSize;

            uniform bool u_falseColor;
            uniform bool u_zebra;
            uniform float u_zebraThreshold;
            
            // v2.3 Analytics
            uniform bool u_gamutWarning;
            uniform bool u_clippingMonitor;

            // HDR pipeline: true when texture contains linear float data
            uniform bool u_isLinear;

            // v2.2: Channel isolation (0=RGB, 1=R, 2=G, 3=B, 4=Luma, 5=Alpha)
            uniform int u_channelMode;
            // v2.2: Focus peaking
            uniform bool u_focusPeaking;
            uniform float u_focusPeakThreshold;
            uniform int u_displayLutMode;
            uniform float u_displayLutStrength;

            // v2.2 Pro Comparison
            uniform bool u_wipeEnabled;
            uniform float u_wipe;
            uniform bool u_wipeRefEnabled;
            uniform sampler2D u_referenceImage;

            // v2.2 Pro Grids
            uniform int u_gridMode;
            uniform vec4 u_gridColor;
            
            // v2.3: Denoise & Depth Eval
            uniform float u_denoise;
            uniform bool u_showDepth;

            // v2.4: Custom Curves (1D LUT)
            // 256x1 texture where R=RedCurve, G=GreenCurve, B=BlueCurve
            // Alpha channel is unused (or could be Luma curve master)
            uniform sampler2D u_curveLut; 
            uniform float u_curveMix; // 0.0 = disabled, 1.0 = full effect
            
            // v2.5: Qualifiers (HSL)
            uniform bool u_qualifierEnabled;
            uniform bool u_qualifierShowMask;
            
            // Hue (0..1)
            uniform float u_qualifierHue;
            uniform float u_qualifierHueWidth;
            uniform float u_qualifierHueSoft;
            
            // Saturation (0..1)
            uniform float u_qualifierSat;
            uniform float u_qualifierSatWidth;
            uniform float u_qualifierSatSoft;
            
            // Luma (0..1)
            uniform float u_qualifierLuma;
            uniform float u_qualifierLumaWidth;
            uniform float u_qualifierLumaSoft;
            
            // v2.6: Lens & Optical Effects
            uniform int u_apertureBlades;
            uniform float u_apertureRotation;
            uniform float u_apertureAnamorphic;

            uniform float u_lensDistortion; // k1
            uniform float u_lensFringe; // Chromatic Aberration


            uniform float u_vignetteIntensity;
            uniform float u_vignetteFalloff;
            
            // v2.6.1: Realistic Bokeh Physics
            uniform float u_bokehHighlightBias;
            uniform float u_bokehSoapBubble;
            uniform float u_bokehOpticalVig;
            
            // v3.1: Extended Grain & Lens
            uniform float u_grainSize;
            uniform float u_grainColor;
            uniform float u_bloom;
            uniform float u_halation;
            uniform float u_diffusion;

            // ── v3.2 Phase 11: Anamorphic Streaks + k2 Distortion ────────────
            uniform float u_lensDistortionK2;   // Brown-Conrady quartic term
            uniform float u_anamorphicStreaks;   // 0=off, strength 0..1
            uniform float u_streakThreshold;     // Luma trigger level (0..1)
            uniform float u_streakLength;        // Horizontal reach in UV space

            // ── v3.2 Phase 12: Bilateral Filter ──────────────────────────────
            uniform float u_bilateralSigmaD;     // Spatial sigma (pixels)
            uniform float u_bilateralSigmaR;     // Range sigma (color diff)
            uniform bool  u_bilateralHalfRes;    // true = half-res FBO pre-computed

            // ----------------------------------------------------------------
            // Color Ops v3.0
            // ----------------------------------------------------------------

            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            vec3 applyHueShift(vec3 color, float shift) {
                if (shift == 0.0) return color;
                vec3 hsv = rgb2hsv(color);
                hsv.x += shift / 360.0;
                if (hsv.x > 1.0) hsv.x -= 1.0;
                if (hsv.x < 0.0) hsv.x += 1.0;
                return hsv2rgb(hsv);
            }

            vec3 applyColorBoost(vec3 color, float boost) {
                if (boost == 0.0) return color;
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                float maxRGB = max(color.r, max(color.g, color.b));
                float sat = maxRGB - min(color.r, min(color.g, color.b)); // Simple saturation estimate
                
                // Vibrance logic: Boosts low-sat pixels more than high-sat
                float boostFactor = (1.0 - sat * 0.8) * boost; 
                vec3 boosted = mix(vec3(luma), color, 1.0 + boostFactor);
                return max(boosted, 0.0);
            }


            // ----------------------------------------------------------------
            // Masking (Power Windows)
            // ----------------------------------------------------------------

            float calculateMask(vec2 uv) {
                if (u_maskType == 0) return 1.0;
                
                vec2 p = uv - u_maskCenter;
                
                // Aspect Ratio Compensation: If u_texSize is available, we use it to keep masks round
                float aspect = u_texSize.x / u_texSize.y;
                p.x *= aspect;
                
                // Apply rotation
                float s = sin(-u_maskRotation);
                float c = cos(-u_maskRotation);
                p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
                
                // Apply scaling (Inverse of scale factor)
                p /= max(vec2(0.001), u_maskScale);
                
                float dist = 0.0;
                if (u_maskType == 1) { // Circle
                    dist = length(p);
                } else if (u_maskType == 2) { // Box
                    dist = max(abs(p.x), abs(p.y));
                }
                
                float mask = 1.0 - smoothstep(1.0 - u_maskFeather, 1.0, dist);
                return u_maskInvert ? 1.0 - mask : mask;
            }

            // ----------------------------------------------------------------
            // Curves
            // ----------------------------------------------------------------
            
            vec3 applyCurves(vec3 color) {
                if (u_curveMix <= 0.0) return color;
                
                // Texture lookup (0..1 range)
                // We use the color value itself as the U coordinate. V is 0.5.
                // Clamp input to 0-1 to avoid texture wrap artifacts
                vec3 c = clamp(color, 0.0, 1.0);
                
                float r = texture(u_curveLut, vec2(c.r, 0.5)).r;
                float g = texture(u_curveLut, vec2(c.g, 0.5)).g;
                float b = texture(u_curveLut, vec2(c.b, 0.5)).b;
                
                vec3 curved = vec3(r, g, b);
                
                // Mix based on strength
                return mix(color, curved, u_curveMix);
            }

// ----------------------------------------------------------------
// DoF / Bokeh
// ----------------------------------------------------------------

const int SAMPLE_COUNT = 64; 
const float PI = 3.14159265;
const float GOLDEN_ANGLE = 2.39996323;

            // Pseudo-random jitter for smoother bokeh texture
            float hash12(vec2 p) {
                p = fract(p * vec2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return fract(p.x * p.y);
            }

            // Brown-Conrady Distortion — full k1 + k2 model
            // k1: primary barrel/pincushion (quadratic radial)
            // k2: secondary correction (quartic radial) — reduces "mustache" distortion
            vec2 distortUV(vec2 uv) {
                if (u_lensDistortion == 0.0 && u_lensDistortionK2 == 0.0) return uv;

                vec2 center = uv - 0.5;
                float r2 = dot(center, center);
                float r4 = r2 * r2;
                float f = 1.0 + r2 * u_lensDistortion + r4 * u_lensDistortionK2;

                return center * f + 0.5;
            }

            vec3 getBokehColor(vec2 uv, float radius) {
                if (radius < 1.0 && u_lensFringe == 0.0) {
                    return texture(u_image, uv).rgb;
                }
                
                vec3 acc = vec3(0.0);
                float weight = 0.0;

                vec2 pixelSize = 1.0 / u_texSize;
                float anamorphic = u_apertureAnamorphic;

                // Optical Vignetting (Cat's Eye Bokeh)
                // Squish the bokeh shape radially depending on distance from center
                vec2 centerDist = uv - 0.5;
                float distLength = length(centerDist);
                // directional squish vector pointing away from center
                vec2 opticalSquishDir = normalize(centerDist); 
                float opticalVigFactor = 1.0 - (u_bokehOpticalVig * clamp(distLength * 1.5, 0.0, 1.0));
                
                // Chromatic Aberration offsets (Red/Blue shift)
                vec2 caOffset = centerDist * u_lensFringe * 0.02 * anamorphic;

                // Center sample
                vec3 c_center = vec3(texture(u_image, uv - caOffset).r, texture(u_image, uv).g, texture(u_image, uv + caOffset).b);
                
                float c_luma = dot(c_center, vec3(0.2126, 0.7152, 0.0722));
                float c_boost = 1.0 + pow(max(c_luma, 0.0), 2.0) * u_bokehHighlightBias;
                
                acc += c_center * c_boost;
                weight += c_boost;
                
                if (radius < 1.0) return acc / weight;

                // Noise-based rotation to break up concentric artifacts
                float noise = hash12(uv * 10.0 + fract(float(u_frame) * 0.1));
                float noiseRot = noise * 6.283185;
                float sinR = sin(noiseRot), cosR = cos(noiseRot);
                mat2 rotMat = mat2(cosR, -sinR, sinR, cosR);

                // Polygon Shape Logic
                float blades = float(u_apertureBlades);
                float bladeRad = radians(360.0 / blades);
                float rot = radians(u_apertureRotation);

                for (int i = 1; i <= SAMPLE_COUNT; i++) {
                    // Jittered Golden Angle distribution
                    float r = sqrt(float(i) / float(SAMPLE_COUNT));
                    float theta = float(i) * GOLDEN_ANGLE;

                    // Map circle to polygon if needed
                    float polygonScale = 1.0;
                    if (blades >= 3.0) {
                        float phi = theta + rot;
                        float sector = floor(phi / bladeRad + 0.5);
                        float phi_local = phi - sector * bladeRad;
                        polygonScale = cos(bladeRad * 0.5) / cos(phi_local);
                    }
                    
                    // Anamorphic stretch + Noise rotation
                    vec2 offsetRaw = vec2(cos(theta), sin(theta)) * rotMat;
                    
                    // Optical Vignetting
                    float dotDir = dot(offsetRaw, opticalSquishDir);
                    vec2 offsetVig = offsetRaw - opticalSquishDir * dotDir * (1.0 - opticalVigFactor);
                    
                    vec2 offset = offsetVig * r * radius * pixelSize;
                    
                    // Apply polygon shape
                    if (blades >= 3.0) offset *= polygonScale;

                    // Apply Anamorphic Ratio (Stretch X)
                    offset.x *= anamorphic;

                    vec2 sampleUV = uv + offset;

                    vec3 sam;
                    sam.r = texture(u_image, sampleUV - caOffset).r;
                    sam.g = texture(u_image, sampleUV).g;
                    sam.b = texture(u_image, sampleUV + caOffset).b; 

                    float luma = dot(sam, vec3(0.2126, 0.7152, 0.0722));
                    // Highlight Bias (Energy)
                    float boost = 1.0 + pow(max(luma, 0.0), 2.0) * u_bokehHighlightBias;
                    
                    // Soap Bubble Effect (Edge brightening)
                    // r ranges from 0 to 1
                    float rim = pow(r, 4.0) * u_bokehSoapBubble; 
                    float sampleWeight = boost + rim;

                    acc += sam * sampleWeight;
                    weight += sampleWeight;
                }

                return acc / weight;
            }

            // ----------------------------------------------------------------
            // Color Ops
            // ----------------------------------------------------------------
            
            vec3 applyLUT(vec3 color, sampler3D lut, float size) {
                float scale = (size - 1.0) / size;
                float offset = 0.5 / size;
                vec3 coords = clamp(color, 0.0, 1.0) * scale + offset;
                return texture(lut, coords).rgb;
            }

            vec3 getFalseColorMap(float v) {
                // v is luminance 0..1
                float stops = log2(max(v, 1e-6) / 0.18);
                
                // Zone colors [R, G, B] matching Python backend (Industry Standard)
                if (stops < -4.0) return vec3(0.05, 0.05, 0.40);      // Clip black -> deep blue
                if (stops < -3.0) return vec3(0.10, 0.40, 0.55);      // -3 stops -> cyan
                if (stops < -2.0) return vec3(0.10, 0.45, 0.35);      // -2 stops -> teal
                if (stops < -1.0) return vec3(0.15, 0.55, 0.15);      // -1 stop -> green
                if (stops <  1.0) return vec3(0.40, 0.40, 0.40);      // Mid gray -> neutral
                if (stops <  2.0) return vec3(0.70, 0.65, 0.10);      // +1 stop -> yellow
                if (stops <  3.0) return vec3(0.75, 0.40, 0.05);      // +2 stops -> orange
                if (stops <  4.0) return vec3(0.70, 0.10, 0.10);      // +3 stops -> red
                return vec3(0.80, 0.15, 0.60);                         // Clip white -> magenta
            }
            
            // ACES Tone Mapping (Approx)
            vec3 toneMapACES(vec3 color) {
                const float a = 2.51;
                const float b = 0.03;
                const float c = 2.43;
                const float d = 0.59;
                const float e = 0.14;
                return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
            }

            // Simple Reinhard Tone Mapping
            vec3 toneMapReinhard(vec3 c) {
                return c / (c + vec3(1.0));
            }

            // IEC 61966-2-1 sRGB OETF — required for linear→display conversion
            vec3 linearToSRGB(vec3 linear) {
                bvec3 cutoff = lessThan(linear, vec3(0.0031308));
                vec3 higher = vec3(1.055) * pow(max(linear, vec3(0.0031308)), vec3(1.0/2.4)) - vec3(0.055);
                vec3 lower = linear * vec3(12.92);
                return mix(higher, lower, vec3(cutoff));
            }

            // Inverse: sRGB→linear (for PNG textures that arrive gamma-encoded)
            vec3 sRGBToLinear(vec3 srgb) {
                bvec3 cutoff = lessThan(srgb, vec3(0.04045));
                vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
                vec3 lower = srgb / vec3(12.92);
                return mix(higher, lower, vec3(cutoff));
            }

            // ─────────────────────────────────────────────────────────────────
            // Display LUT / Camera Log Transforms  (v3.3 — Spec-Accurate)
            //
            // All forward cases assume scene-linear input, 18% grey = 0.18.
            // Verified 18% grey output values (cross-checked against
            // colour-science library, which cites manufacturer specification
            // documents directly):
            //
            //   LogC3:   0.391  | LogC4: 0.278  | S-Log3: 0.411
            //   V-Log:   0.423  | F-Log2: 0.391 | C-Log3: 0.343
            //   Log3G10: 0.333  | DaVinci: 0.336 | BMD Gen5: 0.384 | N-Log: 0.364
            //
            // IDT cases (Log → Linear): decode camera log footage to
            // scene-linear for use in the grading pipeline.
            //
            // L10 = ln(10) converts GLSL log() [natural] to log10 equivalent:
            //   log10(x) = log(x) / L10
            // ─────────────────────────────────────────────────────────────────
            vec3 applyDisplayLUT(vec3 c, int mode) {
                const float L10 = 2.302585; // ln(10)

                switch(mode) {

                // ── ODT: Display / Tonemap ─────────────────────────────────
                case 1: { // sRGB (Display) — linearToSRGB applied at end of main()
                    return c;
                }
                case 2: { // Rec.709 OETF (BT.709)
                    bvec3 lo = lessThan(c, vec3(0.018));
                    vec3 low  = c * 4.5;
                    vec3 high = 1.099 * pow(max(c, vec3(0.018)), vec3(0.45)) - vec3(0.099);
                    return mix(high, low, vec3(lo));
                }
                case 3: { // Filmic — Hable/Uncharted2
                    float A=0.15,B=0.50,C=0.10,D=0.20,E=0.02,F=0.30;
                    vec3 v = c * 2.0;
                    vec3 curr = ((v*(A*v+C*B)+D*E)/(v*(A*v+B)+D*F))-E/F;
                    float wh = ((11.2*(A*11.2+C*B)+D*E)/(11.2*(A*11.2+B)+D*F))-E/F;
                    return curr / wh;
                }
                case 8: { // Reinhard global tonemap
                    return c / (c + vec3(1.0));
                }
                case 9: { // ACES Filmic (Narkowicz fit)
                    return toneMapACES(c);
                }

                // ── ODT: Camera Log Encoding  ─────────────────────────────
                // ── Forward: scene-linear → log code values ──────────────

                case 4: { // ARRI LogC3 EI800 (ARRI Specification v4.0, 2022)
                    // Constants from DATA_ALEXA_LOG_C_CURVE_CONVERSION["SUP 3.x"][800]
                    // cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537
                    // e=5.367655 (linear toe slope), f=0.092809 (linear toe offset)
                    const float lc3_cut=0.010591, lc3_a=5.555556, lc3_b=0.052272;
                    const float lc3_c=0.247190,   lc3_d=0.385537;
                    const float lc3_e=5.367655,   lc3_f=0.092809;
                    return mix(
                        lc3_e * c + lc3_f,
                        lc3_c * (log(lc3_a * max(c,vec3(0.0)) + lc3_b) / L10) + lc3_d,
                        vec3(greaterThan(c, vec3(lc3_cut)))
                    );
                }

                case 11: { // ARRI LogC4 (Alexa 35, ARRI Specification v1.0, 2022)
                    // CONSTANTS_ARRILOGC4: a=2231.826, b=0.907136, c=0.092864
                    //                     s=0.113597, t=-0.018057
                    // Formula: E_p = (log2(a*E + 64) - 6) / 14 * b + c  for E >= t
                    //          E_p = (E - t) / s                         for E < t
                    const float lc4_a=2231.82630906768830, lc4_b=0.90713587487781030;
                    const float lc4_c=0.09286412512218964, lc4_s=0.11359720861058910;
                    const float lc4_t=-0.01805699611991131;
                    vec3 logBranch = (log2(lc4_a * max(c, vec3(lc4_t)) + 64.0) - 6.0) / 14.0 * lc4_b + lc4_c;
                    vec3 linBranch = (c - lc4_t) / lc4_s;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(lc4_t))));
                }



                case 14: { // Fujifilm F-Log2 (Fujifilm Specification v1.0, 2022)
                    // CONSTANTS_FLOG2: cut1=0.000889, a=5.555556, b=0.064829, c=0.245281, d=0.384316, e=8.799461, f=0.092864
                    const float fl2_cut=0.000889, fl2_a=5.555556, fl2_b=0.064829;
                    const float fl2_c=0.245281, fl2_d=0.384316, fl2_e=8.799461, fl2_f=0.092864;
                    vec3 logV = fl2_c * (log(fl2_a * max(c,vec3(0.0)) + fl2_b) / L10) + fl2_d;
                    vec3 linV = fl2_e * c + fl2_f;
                    return mix(linV, logV, vec3(greaterThanEqual(c, vec3(fl2_cut))));
                }

                case 12: { // Canon C-Log3 v1.2 (Canon Specification, Rev.2020)
                    // Input is rescaled by /0.9 before encoding.
                    // Three-way piecewise: negative log | linear | positive log
                    // Cut points in rescaled space: lower ≈ -0.009670, upper ≈ 0.014043
                    vec3 x = c / 0.9;
                    const float cl3_k=14.98325, cl3_a=0.36726845;
                    const float cl3_neg_d=0.12783901;
                    const float cl3_lin_m=1.9754798, cl3_lin_b=0.12512219;
                    const float cl3_pos_d=0.12240537;
                    const float cl3_lo=-0.009670, cl3_hi=0.014043;
                    vec3 negBranch = -cl3_a * log(max(-x * cl3_k + 1.0, vec3(1e-10))) / L10 + cl3_neg_d;
                    vec3 linBranch = cl3_lin_m * x + cl3_lin_b;
                    vec3 posBranch =  cl3_a * log(x * cl3_k + 1.0) / L10 + cl3_pos_d;
                    vec3 result = mix(negBranch, linBranch, vec3(greaterThanEqual(x, vec3(cl3_lo))));
                    result      = mix(result,    posBranch, vec3(greaterThan(x,   vec3(cl3_hi))));
                    return result;
                }

                case 19: { // RED Log3G10 v2 (IPP2, REDCINE-X PRO — colour-science Nattress2016a)
                    // y = sign(x+0.01) * 0.224282 * log10(|x+0.01| * 155.975327 + 1)
                    // 18% grey → 0.333 (exactly 1/3), 0 → 0.0916
                    vec3 xoff = c + 0.01;
                    return sign(xoff) * 0.224282 * log(abs(xoff) * 155.975327 + 1.0) / L10;
                }

                case 20: { // DaVinci Intermediate (Blackmagic Design, 2020)
                    // CONSTANTS_DAVINCI_INTERMEDIATE:
                    //   DI_A=0.0075, DI_B=7.0, DI_C=0.07329248, DI_M=10.44426855, DI_LIN_CUT=0.00262409
                    const float di_A=0.0075, di_B=7.0, di_C=0.07329248;
                    const float di_M=10.44426855, di_cut=0.00262409;
                    vec3 logV = di_C * (log2(max(c + di_A, vec3(1e-10))) + di_B);
                    vec3 linV = c * di_M;
                    return mix(linV, logV, vec3(greaterThan(c, vec3(di_cut))));
                }

                case 15: { // Blackmagic Film Generation 5 (BMD Specification, 2021)
                    // CONSTANTS_BLACKMAGIC_FILM_GENERATION_5:
                    //   A=0.08692876, B=0.005494072, C=0.530013339, D=8.283605932, E=0.092465753, LIN_CUT=0.005
                    const float bm_A=0.08692876065491224, bm_B=0.005494072432257808;
                    const float bm_C=0.5300133392291939,  bm_D=8.283605932402494;
                    const float bm_E=0.09246575342465753, bm_cut=0.005;
                    vec3 logV = bm_A * log(max(c + bm_B, vec3(1e-10))) + bm_C;
                    vec3 linV = bm_D * c + bm_E;
                    return mix(linV, logV, vec3(greaterThanEqual(c, vec3(bm_cut))));
                }

                case 21: { // Nikon N-Log (Nikon Specification, 2018)
                    // CONSTANTS_NLOG: cut1=0.328, a=0.635386, b=0.0075, c=0.146628, d=0.605083
                    // Below cut1: a * (y+b)^(1/3)   |  Above cut1: c * ln(y) + d
                    const float nl_cut=0.328, nl_a=0.635386119257087, nl_b=0.0075;
                    const float nl_c=0.1466275659824047, nl_d=0.6050830889540567;
                    vec3 logV = nl_c * log(max(c, vec3(1e-10))) + nl_d;
                    vec3 cbrtV = nl_a * pow(max(c + nl_b, vec3(1e-10)), vec3(1.0 / 3.0));
                    return mix(cbrtV, logV, vec3(greaterThanEqual(c, vec3(nl_cut))));
                }

                case 6: { // Generic Linear → Log (Cineon-style)
                    return log2(max(c, vec3(1e-10)) * 5.55 + 1.0) / log2(6.55);
                }

                // ── IDT: Camera Log → Scene-Linear ────────────────────────
                // Decode log-encoded footage back to scene-linear for grading.

                case 22: { // IDT LogC4 → Linear
                    // Inverse of case 11: E = (2^((E_p - c)/b * 14 + 6) - 64) / a  for E_p >= c+0 (log region)
                    const float lc4_a=2231.82630906768830, lc4_b=0.90713587487781030;
                    const float lc4_c=0.09286412512218964, lc4_s=0.11359720861058910;
                    const float lc4_t=-0.01805699611991131;
                    float lc4_log_cut = (log2(lc4_a * lc4_t + 64.0) - 6.0) / 14.0 * lc4_b + lc4_c;
                    vec3 logBranch = (exp2((c - lc4_c) / lc4_b * 14.0 + 6.0) - 64.0) / lc4_a;
                    vec3 linBranch = c * lc4_s + lc4_t;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(lc4_log_cut))));
                }



                case 23: { // IDT C-Log3 → Linear
                    // Inverse of case 12 — positive log branch (normal scene content)
                    // x_rescaled = (10^((y - 0.12240537) / 0.36726845) - 1) / 14.98325
                    // x_real = x_rescaled * 0.9
                    vec3 logBranch = (pow(vec3(10.0), (c - 0.12240537) / 0.36726845) - 1.0) / 14.98325 * 0.9;
                    // Linear branch for very low encoded values (rare in normal content)
                    vec3 linBranch = (c - 0.12512219) / 1.9754798 * 0.9;
                    float cl3_lin_cv = 1.9754798 * 0.014043 + 0.12512219; // ~0.1527
                    return mix(linBranch, logBranch, vec3(greaterThan(c, vec3(cl3_lin_cv))));
                }

                case 24: { // IDT F-Log2 → Linear
                    // Inverse of case 14: x = (10^((y - d) / c) - b) / a
                    const float fl2_a=5.555556, fl2_b=0.064829, fl2_c=0.245281;
                    const float fl2_d=0.384316, fl2_e=8.799461, fl2_f=0.092864;
                    float fl2_cut_cv = fl2_e * 0.000889 + fl2_f; // code value at cut
                    vec3 logBranch = (pow(vec3(10.0), (c - fl2_d) / fl2_c) - fl2_b) / fl2_a;
                    vec3 linBranch = (c - fl2_f) / fl2_e;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(fl2_cut_cv))));
                }

                case 25: { // IDT Log3G10 → Linear
                    // Inverse of case 19: x = sign(y)/155.975 * (10^(|y|/0.224282) - 1) - 0.01
                    vec3 s = sign(c);
                    return s * (pow(vec3(10.0), abs(c) / 0.224282) - 1.0) / 155.975327 - 0.01;
                }

                case 26: { // IDT DaVinci Intermediate → Linear
                    // Inverse of case 20: L = 2^(V/C - B) - A  for V > log_cut
                    const float di_A=0.0075, di_B=7.0, di_C=0.07329248;
                    const float di_M=10.44426855, di_log_cut=0.02740668;
                    vec3 logBranch = exp2(c / di_C - di_B) - di_A;
                    vec3 linBranch = c / di_M;
                    return mix(linBranch, logBranch, vec3(greaterThan(c, vec3(di_log_cut))));
                }

                case 27: { // IDT BMD Film Gen5 → Linear
                    // Inverse of case 15: x = exp((y - C) / A) - B
                    const float bm_A=0.08692876065491224, bm_B=0.005494072432257808;
                    const float bm_C=0.5300133392291939, bm_D=8.283605932402494, bm_E=0.09246575342465753;
                    float bm_log_cut = bm_D * 0.005 + bm_E; // code value at LIN_CUT=0.005
                    vec3 logBranch = exp((c - bm_C) / bm_A) - bm_B;
                    vec3 linBranch = (c - bm_E) / bm_D;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(bm_log_cut))));
                }

                case 28: { // IDT N-Log → Linear
                    // Inverse of case 21:
                    // Below cut CV: x = (y/a)^3 - b
                    // Above cut CV: x = exp((y - d) / c)
                    const float nl_a=0.635386119257087, nl_b=0.0075;
                    const float nl_c=0.1466275659824047, nl_d=0.6050830889540567;
                    // Cut CV: nl_a * (0.328 + nl_b)^(1/3)
                    float nl_cut_cv = nl_a * pow(0.328 + nl_b, 1.0 / 3.0);
                    vec3 cbrtBranch = pow(c / nl_a, vec3(3.0)) - nl_b;
                    vec3 logBranch  = exp((c - nl_d) / nl_c);
                    return max(mix(cbrtBranch, logBranch, vec3(greaterThanEqual(c, vec3(nl_cut_cv)))), vec3(0.0));
                }

                default:
                    return c;
                }
            }


            // Sobel edge detection for focus peaking
            vec3 applyFocusPeaking(vec3 color, vec2 uv) {
                vec2 px = 1.0 / u_texSize;
                // 3x3 Sobel on luminance
                float tl = dot(texture(u_image, uv + vec2(-px.x, -px.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
                float tc = dot(texture(u_image, uv + vec2(  0.0, -px.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
                float tr = dot(texture(u_image, uv + vec2( px.x, -px.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
                float ml = dot(texture(u_image, uv + vec2(-px.x,   0.0)).rgb, vec3(0.2126, 0.7152, 0.0722));
                float mr = dot(texture(u_image, uv + vec2( px.x,   0.0)).rgb, vec3(0.2126, 0.7152, 0.0722));
                float bl = dot(texture(u_image, uv + vec2(-px.x,  px.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
                float bc = dot(texture(u_image, uv + vec2(  0.0,  px.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
                float br = dot(texture(u_image, uv + vec2( px.x,  px.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
                float gx = -tl + tr - 2.0*ml + 2.0*mr - bl + br;
                float gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
                float mag = sqrt(gx*gx + gy*gy) * 255.0;
                if (mag > u_focusPeakThreshold) {
                    return mix(color, vec3(1.0, 0.0, 0.2), 0.85);
                }
                return color;
            }

            // Safety: sanitizes NaNs and Infs to black or max value
            vec3 sanitize(vec3 c) {
                bvec3 is_nan = isnan(c);
                bvec3 is_inf = isinf(c);
                vec3 safe = c;
                if (any(is_nan)) safe = vec3(0.0);
                if (any(is_inf)) safe = clamp(c, 0.0, 65504.0);
                return safe;
            }
            
            // ----------------------------------------------------------------
            // Advanced Grading Ops
            // ----------------------------------------------------------------
            
            vec3 applyTempTint(vec3 color, float temp, float tint) {
                vec3 shift = vec3(0.0);
                // Temp: Warm (Orange) / Cool (Blue)
                shift.r += temp;
                shift.b -= temp;
                // Tint: Magenta / Green
                shift.g -= tint;
                shift.r += tint * 0.5;
                shift.b += tint * 0.5;
                
                // Additive shift or multiplicative gain?
                // Multiplicative is better for preserving black point.
                return color * (vec3(1.0) + shift);
            }
            
            vec3 applyGrading(vec3 color, vec3 lift, vec3 gamma, vec3 gain, vec3 offset) {
                // Resolve-Style Grading
                
                // 1. Offset (Global Add)
                color += offset;
                
                // 2. Lift (Shadows - Pivoted at White)
                // Lift adds to blacks, but has 0 effect at 1.0
                // Simple formula: color + lift * (1.0 - luma)
                // Using luminance for pivot to avoid color shifts
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                // Clamp luma to 0..1 for pivot
                float pivot = clamp(1.0 - luma, 0.0, 1.0);
                color += lift * pivot;
                
                // 3. Gain (Slope - Pivoted at Black)
                color *= gain;
                
                // 4. Gamma (Power - Mids)
                // Safe pow 
                color = max(color, 0.0);
                if (any(notEqual(gamma, vec3(1.0)))) {
                     color.r = pow(color.r, 1.0 / max(0.01, gamma.r));
                     color.g = pow(color.g, 1.0 / max(0.01, gamma.g));
                     color.b = pow(color.b, 1.0 / max(0.01, gamma.b));
                }
                
                return color;
            }
            
            vec3 applyContrast(vec3 color, float contrast, float pivot) {
                return (color - pivot) * contrast + pivot;
            }

            // v2.5 Pro Pro: Cinematic S-Curve Shadows/Highlights
            vec3 applyShadowsHighlights(vec3 color, float shadows, float highlights) {
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                // Quadratic weights for smoother roll-off
                float sWeight = pow(1.0 - smoothstep(0.0, 0.5, luma), 2.0);
                float hWeight = pow(smoothstep(0.5, 1.0, luma), 2.0);
                
                // shadows in [-1,1], lift(pos) or crush(neg)
                color *= (1.0 + shadows * sWeight * 0.5);
                // highlights in [-1,1], expand(pos) or compress(neg)
                color *= (1.0 + highlights * hWeight * 0.5);
                
                return max(color, 0.0);
            }


            vec3 hash32(vec2 p) {
                vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
                p3 += dot(p3, p3.yxz+33.33);
                return fract((p3.xxy+p3.yzz)*p3.zyx);
            }

            vec3 blendOverlay(vec3 base, vec3 blend) {
                return mix(1.0 - 2.0 * (1.0 - base) * (1.0 - blend), 2.0 * base * blend, step(base, vec3(0.5)));
            }

            float rand(vec2 co) {
                // Legacy support - simple animated grain seed
                return fract(sin(dot(co.xy + u_time, vec2(12.9898,78.233))) * 43758.5453);
            }

// ----------------------------------------------------------------
// v3.2 Phase 12: GPU Bilateral Filter (Edge-Preserving Denoise)
//
// A 7×7 bilateral kernel that down-weights neighbours whose colour
// differs significantly from the centre pixel.  This preserves
// hard edges (skin/hair/object silhouettes) while still smoothing
// flat areas (sky, walls, bokeh).
//
// When u_bilateralHalfRes is true the JS side has already pre-rendered
// the bilateral result into a half-res FBO; this function is still called
// but at 0.5× resolution — WebGL's bilinear sampler handles the upscale
// for free at full-resolution display, cutting per-pixel cost by ~4×.
// ----------------------------------------------------------------
vec3 getBilateralColor(vec2 uv) {
    vec2 px = 1.0 / u_texSize;
    vec3 center = texture(u_image, uv).rgb;

    // Guard: if bilateral is disabled, return raw sample
    if (u_bilateralSigmaD <= 0.0) return center;

    float sigD2 = u_bilateralSigmaD * u_bilateralSigmaD;
    float sigR2 = u_bilateralSigmaR * u_bilateralSigmaR;

    vec3 acc = vec3(0.0);
    float wSum = 0.0;

    // 7×7 kernel  (radius = 3 taps each direction)
    for (int dx = -3; dx <= 3; dx++) {
        for (int dy = -3; dy <= 3; dy++) {
            vec2 offset = vec2(float(dx), float(dy)) * px;
            vec3 s = texture(u_image, uv + offset).rgb;

            // Spatial weight — Gaussian over pixel distance
            float spatialDist2 = float(dx * dx + dy * dy);
            float wSpatial = exp(-spatialDist2 / (2.0 * sigD2));

            // Range weight — Gaussian over colour distance
            vec3 diff = s - center;
            float rangeDist2 = dot(diff, diff);
            float wRange = exp(-rangeDist2 / (2.0 * sigR2));

            float w = wSpatial * wRange;
            acc  += s * w;
            wSum += w;
        }
    }

    return acc / max(wSum, 1e-5);
}

// Legacy alias — keeps existing denoise path working
vec3 getDenoiseColor(vec2 uv) {
    return getBilateralColor(uv);
}

    // ----------------------------------------------------------------
    // HSL Qualifier
    // ----------------------------------------------------------------
    
    vec3 applyMidDetail(vec3 color, vec2 uv, float amount) {
        if (amount == 0.0) return color;
        vec3 blurred = getDenoiseColor(uv); // Reuse existing blur
        vec3 detail = color - blurred;
        return color + detail * amount;
    }


    vec3 rgb2hcv(vec3 rgb) {
        vec4 P = (rgb.g < rgb.b) ? vec4(rgb.bg, -1.0, 2.0/3.0) : vec4(rgb.gb, 0.0, -1.0/3.0);
        vec4 Q = (rgb.r < P.x) ? vec4(P.xyw, rgb.r) : vec4(rgb.r, P.yzx);
        float C = Q.x - min(Q.w, Q.y);
        float H = abs((Q.w - Q.y) / (6.0 * C + 1e-10) + Q.z);
        return vec3(H, C, Q.x);
    }

    vec3 rgb2hsl(vec3 rgb) {
        vec3 HCV = rgb2hcv(rgb);
        float L = HCV.z - HCV.y * 0.5;
        float S = HCV.y / (1.0 - abs(L * 2.0 - 1.0) + 1e-10);
        return vec3(HCV.x, S, L);
    }

    float getQualifierMask(vec3 color) {
        if (!u_qualifierEnabled) return 1.0;

        vec3 hsl = rgb2hsl(color);
        
        // Hue (Circular distance)
        float hDist = abs(hsl.x - u_qualifierHue);
        if (hDist > 0.5) hDist = 1.0 - hDist;
        float hMask = 1.0 - smoothstep(u_qualifierHueWidth, u_qualifierHueWidth + u_qualifierHueSoft, hDist);
        
        // Saturation
        float sDist = abs(hsl.y - u_qualifierSat);
        float sMask = 1.0 - smoothstep(u_qualifierSatWidth, u_qualifierSatWidth + u_qualifierSatSoft, sDist);
        
        // Luma
        float lDist = abs(hsl.z - u_qualifierLuma);
        float lMask = 1.0 - smoothstep(u_qualifierLumaWidth, u_qualifierLumaWidth + u_qualifierLumaSoft, lDist);
        
        return hMask * sMask * lMask;
    }

    void main() {
        // 0. Lens Distortion
        vec2 uv = distortUV(v_texcoord);
        
        // Black out of bounds if distorted
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        // ... (Show Depth logic) ...
         if (u_showDepth) {
            float d = texture(u_depth, uv).r;
            fragColor = vec4(vec3(d), 1.0);
            return;
        }

        // 1. DoF
        vec3 color;
        if (u_dofEnabled) {
            float depth = texture(u_depth, uv).r;
            float coc = abs(depth - u_focusDist) * u_aperture * 100.0;
            coc = clamp(coc, 0.0, 20.0);
            color = getBokehColor(uv, coc);
        } else {
            // Apply CA even if DoF off? Yes, getBokehColor handles it if radius < 1.0
            // But we can call it with radius 0.0 to get CA
            if (u_lensFringe > 0.0) {
                color = getBokehColor(uv, 0.0);
            } else {
                color = texture(u_image, uv).rgb;
            }
        }
        
        // 1a. Denoise
        if (u_denoise > 0.0) {
            vec3 smoothColor = getDenoiseColor(uv);
            color = mix(color, smoothColor, u_denoise);
        }

        // 1b. Linearize
        if (!u_isLinear) {
            color = sRGBToLinear(color);
        }

        // SAVE PRE-GRADE (Source for Keying)
        vec3 preGrade = color;

        // ── All grading operations now happen in LINEAR space ──

        // 2. Exposure
        color *= pow(2.0, u_exposure);

        // 3. White Balance
        if (u_temperature != 0.0 || u_tint != 0.0) {
            color = applyTempTint(color, u_temperature, u_tint);
        }
        
        // 4. Grading (Resolve Style)
        color = applyGrading(color, u_lift, u_gamma, u_gain, u_offset);
        
        // 5. Contrast
        if (u_contrast != 1.0) {
            color = applyContrast(color, u_contrast, u_pivot);
        }

        // v3.0: Shadows / Highlights (Log-like)
        color = applyShadowsHighlights(color, u_shadows, u_highlights);

        // v3.0: Midtone Detail (Sharpen/Soften)
        if (u_midDetail != 0.0) {
             color = applyMidDetail(color, v_texcoord, u_midDetail);
        }

        // v3.0: Color Boost (Vibrance)
        if (u_colorBoost != 0.0) {
            color = applyColorBoost(color, u_colorBoost);
        }

        // 5b. Custom Curves
        color = applyCurves(color);

        // 6. Saturation
        float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(luma), color, u_saturation);
        
        // v3.0: Hue Shift
        if (u_hueShift != 0.0) {
            color = applyHueShift(color, u_hueShift);
        }
        
        // v3.0: Luma Mix
        if (u_lumaMix != 1.0) {
            float currentLuma = dot(color, vec3(0.2126, 0.7152, 0.0722));
            // Restore original luma? Or mix color back to original luma?
            // Resolve Luma Mix: 100% means Luma is coupled with RGB.
            // 0% means Luma is independent? actually usually it controls how much Y channel is affected.
            // Simplified: Mix between current result and (OriginalLuma + ColorDiff)
            // Or just mix luma channel back to pre-graded luma.
            // Let's stick to standard mix: 
            // If Luma Mix is 0, we output color but with original luma.
            // If Luma Mix is 1, we output color as is.
            
            // Re-calculate original luma from preGrade (linearized input)
            // Pre-grade is 'preGrade' variable
            float origLuma = dot(preGrade, vec3(0.2126, 0.7152, 0.0722));
            vec3 colorWithOrigLuma = color * (origLuma / (currentLuma + 0.0001));
            color = mix(colorWithOrigLuma, color, u_lumaMix);
        }

        // ── FINAL MASK / QUALIFIER MIX ──
        float finalMatte = 1.0;
        if (u_qualifierEnabled) {
            finalMatte *= getQualifierMask(preGrade);
        }
        if (u_maskType > 0) {
            finalMatte *= calculateMask(v_texcoord);
        }
        
        if (u_maskType > 0 && u_maskShowOverlay) {
             // Show mask as grayscale overlay for positioning
             float m = calculateMask(v_texcoord);
             color = mix(preGrade * 0.3, color, m);
        } else if (u_qualifierEnabled || u_maskType > 0) {
             if (u_qualifierShowMask) {
                 color = vec3(finalMatte);
             } else {
                 color = mix(preGrade, color, finalMatte);
             }
        }

        // 7. LUT (Global - applied after qualification)
        vec3 lutInput = color;
        if (u_lutEnabled) {
            vec3 lutted = applyLUT(lutInput, u_lut, u_lutSize);
            color = mix(color, lutted, u_lutStrength);
        }

        // 5b. Bloom/Halation/Diffusion (linear space, before display transform)
        // These sample u_image (linear input) and add to graded linear color.
        // Must happen BEFORE tonemap/LUT which maps to display range.
        // 5b. Pro Bloom (optimized single-pass with better highlights preservation)
        if (u_bloom > 0.0) {
            vec3 bloomAcc = vec3(0.0);
            float bloomW = 0.0;
            vec2 px = 1.2 / u_texSize; // Slightly larger spread
            for (int bx = -3; bx <= 3; bx++) {
                for (int by = -3; by <= 3; by++) {
                    vec2 off = vec2(float(bx), float(by)) * px * 2.5;
                    vec3 s = texture(u_image, uv + off).rgb;
                    if (!u_isLinear) {
                        bvec3 cutoff = lessThan(s, vec3(0.04045));
                        s = mix(pow((s + vec3(0.055)) / vec3(1.055), vec3(2.4)), s / vec3(12.92), vec3(cutoff));
                    }
                    float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
                    // Higher threshold (0.8) and softer knee for Pro Bloom
                    float w = smoothstep(0.7, 1.2, lum);
                    bloomAcc += s * w;
                    bloomW += w;
                }
            }
            if (bloomW > 0.0) color += (bloomAcc / bloomW) * u_bloom * 0.35;
        }

        // 5c. Halation (red channel highlight bleed — classic film gate bounce)
        if (u_halation > 0.0) {
            float halAcc = 0.0;
            float halW = 0.0;
            vec2 px2 = 1.0 / u_texSize;
            for (int hx = -4; hx <= 4; hx++) {
                for (int hy = -4; hy <= 4; hy++) {
                    vec2 off = vec2(float(hx), float(hy)) * px2 * 4.0;
                    vec3 s = texture(u_image, uv + off).rgb;
                    float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
                    float w = max(lum - 0.6, 0.0);
                    halAcc += s.r * w;
                    halW += w;
                }
            }
            if (halW > 0.0) halAcc /= halW;
            color.r += halAcc * u_halation * 0.4;
        }

        // ── v3.2 Phase 11: Anamorphic Lens Streaks ───────────────────────────
        // Horizontal bloom streaks from bright highlights.
        // Classic anamorphic signature: horizontal blue-cyan flares on specular
        // sources.  The streak length is expressed as a UV fraction so it scales
        // with the image resolution automatically.
        if (u_anamorphicStreaks > 0.0) {
            vec3  streakAcc = vec3(0.0);
            float streakW   = 0.0;
            const int STREAK_SAMPLES = 24;

            for (int si = -STREAK_SAMPLES; si <= STREAK_SAMPLES; si++) {
                float t = float(si) / float(STREAK_SAMPLES);
                vec2 sUV = clamp(vec2(uv.x + t * u_streakLength, uv.y), 0.0, 1.0);
                vec3 s   = texture(u_image, sUV).rgb;

                // Linearise sRGB samples if needed
                if (!u_isLinear) {
                    bvec3 cut = lessThan(s, vec3(0.04045));
                    s = mix(pow((s + vec3(0.055)) / vec3(1.055), vec3(2.4)),
                            s / vec3(12.92), vec3(cut));
                }

                float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
                // Only pixels above threshold contribute; soft knee
                float trigger = smoothstep(u_streakThreshold - 0.05, u_streakThreshold + 0.05, lum);
                // Gaussian falloff along streak axis
                float w = exp(-abs(t) * 10.0) * trigger;
                streakAcc += s * w;
                streakW   += w;
            }

            if (streakW > 0.0) {
                // Anamorphic streaks are characteristically cyan-blue tinted
                vec3 streak = (streakAcc / streakW) * vec3(0.55, 0.80, 1.0);
                color += streak * u_anamorphicStreaks * 0.45;
            }
        }

        // 5d. Diffusion (soft filter — blend with blurred version)
        if (u_diffusion > 0.0) {
            vec3 diffAcc = vec3(0.0);
            vec2 px3 = 1.0 / u_texSize;
            float diffW = 0.0;
            for (int dx = -2; dx <= 2; dx++) {
                for (int dy = -2; dy <= 2; dy++) {
                    vec2 off = vec2(float(dx), float(dy)) * px3 * 2.0;
                    diffAcc += texture(u_image, uv + off).rgb;
                    diffW += 1.0;
                }
            }
            diffAcc /= diffW;
            color = mix(color, diffAcc, u_diffusion * 0.6);
        }

        // 6. Display LUT / Tonemap
        if (u_displayLutMode > 0) {
            vec3 transformed = applyDisplayLUT(color, u_displayLutMode);
            color = mix(color, transformed, u_displayLutStrength);
        } else if (u_isLinear) {
            // HDR linear data with no display LUT — apply ACES filmic tonemap
            // to map scene-referred values to displayable range.
            // Without this, values >>1.0 clip per-channel → saturated primaries.
            color = toneMapACES(color);
        }

        // 7. Display Transform (linear → sRGB)
        color = linearToSRGB(max(color, vec3(0.0)));

        // 7a. Film Grain (with size + color + luma attenuation)
        if (u_grainAmount > 0.0) {
            vec2 grainUV = uv * u_texSize / u_grainSize;
            vec2 seed = grainUV + floor(u_time * 24.0) * 0.1337;
            
            // Attenuate grain in extremes (shadows/highlights)
            float lumaGrain = dot(color, vec3(0.2126, 0.7152, 0.0722));
            float attenuation = smoothstep(0.0, 0.2, lumaGrain) * (1.0 - smoothstep(0.7, 1.0, lumaGrain));
            attenuation = mix(0.3, 1.0, attenuation);
            
            if (u_grainColor > 0.0) {
                vec3 noiseRGB = hash32(seed);
                vec3 grainRGB = (noiseRGB - 0.5) * u_grainAmount * attenuation;
                color = blendOverlay(color, grainRGB + 0.5);
            } else {
                float noiseMono = hash12(seed);
                float grainMono = (noiseMono - 0.5) * u_grainAmount * attenuation;
                color = blendOverlay(color, vec3(grainMono) + 0.5);
            }
        }
        
        // 7d. Vignette
        if (u_vignetteIntensity > 0.0) {
            float d = distance(uv, vec2(0.5));
            float v = smoothstep(1.0, 0.25 + (1.0 - u_vignetteFalloff) * 0.75, d * (0.5 + u_vignetteIntensity * 1.5));
            color *= v;
        }

        // 6b. Channel isolation
        if (u_channelMode == 1) { color = vec3(color.r); }
        else if (u_channelMode == 2) { color = vec3(color.g); }
        else if (u_channelMode == 3) { color = vec3(color.b); }
        else if (u_channelMode == 4) { float l = dot(color, vec3(0.2126, 0.7152, 0.0722)); color = vec3(l); }
        else if (u_channelMode == 5) { color = vec3(texture(u_image, uv).a); }

        // 6c. Focus Peaking
        if (u_focusPeaking) {
            color = applyFocusPeaking(color, uv);
        }

        // 7. Analytics
        float lumaDisplay = dot(color, vec3(0.2126, 0.7152, 0.0722));

        if (u_falseColor) {
            color = getFalseColorMap(lumaDisplay);
        }

        if (u_zebra) {
            float s = step(0.5, fract((v_texcoord.x + v_texcoord.y) * 120.0));
            if (lumaDisplay > u_zebraThreshold && s > 0.5) color = vec3(1.0, 0.0, 0.0);
        }

        // 7b. Advanced Analytics (Gamut & Clipping)
        if (u_clippingMonitor) {
            float blink = step(0.5, fract(u_time * 2.0)); // 2Hz flash
            if (any(greaterThan(color, vec3(1.0)))) {
                color = mix(color, vec3(1.0, 0.0, 0.0), blink); // Flashing Red for Highlights
            } else if (any(lessThan(color, vec3(0.0)))) {
                color = mix(color, vec3(0.0, 0.0, 1.0), blink); // Flashing Blue for Shadows
            }
        }

        if (u_gamutWarning) {
            // Check if color is outside Rec709/sRGB gamut hull (values < 0.0 or > 1.0)
            if (any(lessThan(color, vec3(0.0))) || any(greaterThan(color, vec3(1.0)))) {
                color = vec3(1.0, 0.0, 1.0); // Solid Magenta
            }
        }

        // 8. Wipe Comparison (A/B)
        if (u_wipeEnabled) {
            float wipeLine = u_wipe;
            // Cross-over horizontal wipe
            if (v_texcoord.x < wipeLine) {
                // Side A: Original / Reference
                if (u_wipeRefEnabled) {
                    color = texture(u_referenceImage, v_texcoord).rgb;
                } else {
                    // Raw original (linearized or not depending on texture)
                    vec3 raw = texture(u_image, v_texcoord).rgb;
                    if (!u_isLinear) raw = sRGBToLinear(raw);
                    // Apply OETF to original so it's comparable to graded output
                    color = linearToSRGB(raw);
                }
            }
            
            // Draw a thin dividing line
            if (abs(v_texcoord.x - wipeLine) < 0.002) {
                color = vec3(0.1, 0.3, 1.0); // Blue divider
            }
        }

        // 9. Grids & Overlays
        if (u_gridMode > 0) {
            vec2 uvg = v_texcoord;
            float g = 0.0;
            if (u_gridMode == 1) { // Rule of Thirds
                if (abs(uvg.x - 0.333) < 0.001 || abs(uvg.x - 0.666) < 0.001 ||
                    abs(uvg.y - 0.333) < 0.001 || abs(uvg.y - 0.666) < 0.001) g = 1.0;
            } else if (u_gridMode == 2) { // 2.39:1 Mask
                float aspect = u_texSize.x / u_texSize.y;
                float targetAspect = 2.39;
                float maskH = (1.0 - (aspect / targetAspect)) * 0.5;
                if (uvg.y < maskH || uvg.y > (1.0 - maskH)) {
                    color *= 0.1; // Darken bars
                }
            } else if (u_gridMode == 3) { // Center Cross
                if (abs(uvg.x - 0.5) < 0.001 || abs(uvg.y - 0.5) < 0.001) g = 1.0;
            }
            color = mix(color, u_gridColor.rgb, g * u_gridColor.a);
        }

        fragColor = vec4(color, 1.0);
    }
`;
    }

    getLUTFragmentShader() {
        // Deprecated, mapped to Composite
        return this.getCompositeFragmentShader();
    }

    getParadeFragmentShader() {
        return `#version 300 es
            precision highp float;
            in vec2 v_texcoord;
            out vec4 fragColor;
            uniform sampler2D u_image;
            uniform float u_brightness;
            void main() {
                float section = floor(v_texcoord.x * 3.0);
                float localX = fract(v_texcoord.x * 3.0);
                float columnSum = 0.0;
                int sampleCount = 256;
                for (int i = 0; i < sampleCount; i++) {
                    float y = float(i) / float(sampleCount - 1);
                    vec3 col = texture(u_image, vec2(localX, y)).rgb;
                    float val = (section < 1.0) ? col.r : (section < 2.0 ? col.g : col.b);
                    if (abs(val - (1.0 - v_texcoord.y)) < 0.015) columnSum += 1.0;
                }
                vec3 chanCol = (section < 1.0) ? vec3(1.0, 0.2, 0.2) : (section < 2.0 ? vec3(0.2, 1.0, 0.2) : vec3(0.2, 0.4, 1.0));
                fragColor = vec4(chanCol * (columnSum / 256.0) * u_brightness, 1.0);
            }
        `;
    }

    // v2.5: GPU Point-based Scopes
    getScopePointVertexShader(mode) {
        return `#version 300 es
            layout(location = 0) in vec2 a_uv;
            uniform sampler2D u_image;
            uniform bool u_isLinear;
            uniform float u_intensity;
            uniform bool u_parade;
            out vec4 v_color;

            void main() {
                vec4 pixel = texture(u_image, a_uv);
                vec3 color = pixel.rgb;
                
                // Linearize if sRGB
                if (!u_isLinear) {
                    bvec3 cutoff = lessThan(color, vec3(0.04045));
                    vec3 higher = pow((color + vec3(0.055)) / vec3(1.055), vec3(2.4));
                    vec3 lower = color / vec3(12.92);
                    color = mix(higher, lower, vec3(cutoff));
                }
                
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

                if (${mode === 'vectorscope' ? 'true' : 'false'}) {
                    // Vectorscope (U/V)
                    float u = (color.b - luma) * 0.492;
                    float v = (color.r - luma) * 0.877;
                    v_color = vec4(color, u_intensity);
                    gl_Position = vec4(u * 2.0, v * 2.0, 0.0, 1.0);
                } else if (${mode === 'waveform' ? 'true' : 'false'}) {
                    // Waveform (X/Luma) or RGB Parade
                    if (u_parade) {
                        // RGB Parade: [R][G][B] side by side
                        float x_norm = a_uv.x; 
                        
                        // Use cyclic channel assignment based on Y coordinate to balance samples
                        float chanIdx = floor(a_uv.y * 3.0); 
                        float val = (chanIdx < 1.0) ? color.r : (chanIdx < 2.0 ? color.g : color.b);
                        vec3 chanCol = (chanIdx < 1.0) ? vec3(1.0, 0.1, 0.1) : (chanIdx < 2.0 ? vec3(0.1, 1.0, 0.1) : vec3(0.1, 0.4, 1.0));
                        
                        float x_base = -1.0 + chanIdx * (2.0/3.0);
                        float x_local = x_norm * (2.0/3.0);
                        
                        gl_Position = vec4(x_base + x_local, clamp(val, 0.0, 1.0) * 1.94 - 0.97, 0.0, 1.0);
                        v_color = vec4(chanCol, u_intensity * 3.0); // Boost for density
                    } else {
                        // Luma Waveform
                        float x = a_uv.x * 1.96 - 0.98;
                        float y = clamp(luma, 0.0, 1.0) * 1.96 - 0.98;
                        v_color = vec4(vec3(0.6, 1.0, 0.6), u_intensity * 1.5);
                        gl_Position = vec4(x, y, 0.0, 1.0);
                    }
                } else {
                    // Histogram (Luma)
                    float x = clamp(luma, 0.0, 1.0) * 1.96 - 0.98;
                    float y = (a_uv.y * 2.0 - 1.0) * 0.8; 
                    v_color = vec4(vec3(0.8), u_intensity);
                    gl_Position = vec4(x, y, 0.0, 1.0);
                }
                gl_PointSize = 1.0;
            }
        `;
    }

    getScopePointFragmentShader() {
        return `#version 300 es
            precision highp float;
            in vec4 v_color;
            out vec4 fragColor;
            void main() {
                fragColor = v_color;
            }
        `;
    }

    createScopeBuffers() {
        const gl = this.gl;
        const res = 256; // 65k points
        const uvs = new Float32Array(res * res * 2);
        for (let y = 0; y < res; y++) {
            for (let x = 0; x < res; x++) {
                const i = (y * res + x) * 2;
                uvs[i] = (x + 0.5) / res;
                uvs[i + 1] = (y + 0.5) / res;
            }
        }
        this.scopeBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.scopeBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
        this.scopePointCount = res * res;
    }

    // Load image as texture
    loadImageTexture(image) {
        const gl = this.gl;

        if (this.textures.image) {
            gl.deleteTexture(this.textures.image);
        }

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        // FIX: alignment for 16-bit or odd-width images
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        // Use RGBA for standard images
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

        // Set texture parameters
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        this.textures.image = texture;
        this.imageWidth = image.width;
        this.imageHeight = image.height;
        this.isLinearTexture = false; // PNG/Image data is sRGB-encoded

        // Invalidate uniform cache — new image may have different size/isLinear
        this.invalidateUniformCache(this.programs.composite);

        return texture;
    }

    // Load float32 texture for HDR (WebGL2)
    loadFloat32Texture(data, width, height, channels = 4) {
        const gl = this.gl;

        if (!this.isWebGL2) {
            console.warn('[Radiance] Float32 textures require WebGL2, falling back to 8-bit pipeline');
            return null;
        }

        // WebGL2 requires EXT_color_buffer_float for some float texture operations
        if (!this.extColorBufferFloat) {
            console.warn('[Radiance] EXT_color_buffer_float not supported, float texture rendering might fail');
        }

        if (this.textures.image) {
            gl.deleteTexture(this.textures.image);
        }

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        // WebGL2: Native float32 support
        let internalFormat, format;
        if (channels === 1) {
            internalFormat = gl.R32F;
            format = gl.RED;
        } else {
            internalFormat = channels === 4 ? gl.RGBA32F : gl.RGB32F;
            format = channels === 4 ? gl.RGBA : gl.RGB;
        }

        // Safety: check for NaNs in JS before upload (heavy but safe for debugging)
        // if (data.some(isNaN)) console.warn("Float32 data contains NaNs!");

        gl.texImage2D(
            gl.TEXTURE_2D, 0, internalFormat,
            width, height, 0,
            format, gl.FLOAT,
            data  // Float32Array
        );

        // v3.0 FIX: Check for GL errors after texture upload
        const err = gl.getError();
        if (err !== gl.NO_ERROR) {
            console.error(`[Radiance] Float32 texImage2D failed (GL error ${err}). Params: ${width}x${height}, ch=${channels}, internal=${internalFormat}, fmt=${format}`);
            gl.deleteTexture(texture);
            return null;
        }

        // CONDITIONAL FILTERING: Linear only if extension supported
        const filter = this.extColorFloatLinear ? gl.LINEAR : gl.NEAREST;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);

        this.textures.image = texture;
        this.imageWidth = width;
        this.imageHeight = height;
        this.isLinearTexture = true; // Float32 data is scene-linear

        // Invalidate uniform cache — size and isLinear changed
        this.invalidateUniformCache(this.programs.composite);

        console.log(`[Radiance] Loaded ${width}×${height} float32 HDR texture (linear, ${filter === gl.LINEAR ? 'LINEAR' : 'NEAREST'})`);
        return texture;
    }

    // Load float16 texture for HDR (WebGL2) — .rhdr format
    loadFloat16Texture(data, width, height, channels = 3) {
        const gl = this.gl;

        if (!this.isWebGL2) {
            console.warn('[Radiance] Float16 textures require WebGL2');
            return null;
        }

        if (this.textures.image) {
            gl.deleteTexture(this.textures.image);
        }

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        let internalFormat, format;
        if (channels === 1) {
            internalFormat = gl.R16F;
            format = gl.RED;
            // Swizzle not strictly required if shader handles single channel, 
            // but for safety we'll rely on shader logic to read .r
        } else {
            internalFormat = channels === 4 ? gl.RGBA16F : gl.RGB16F;
            format = channels === 4 ? gl.RGBA : gl.RGB;
        }

        gl.texImage2D(
            gl.TEXTURE_2D, 0, internalFormat,
            width, height, 0,
            format, gl.HALF_FLOAT,
            data  // Uint16Array (IEEE 754 half-float)
        );

        // v3.0 FIX: Check for GL errors after texture upload.
        // Without this, a failed upload returns a "valid" texture that is empty (all zeros = black).
        const err = gl.getError();
        if (err !== gl.NO_ERROR) {
            console.error(`[Radiance] Float16 texImage2D failed (GL error ${err}): ${width}×${height}×${channels}ch`);
            gl.deleteTexture(texture);
            return null;
        }

        // CONDITIONAL FILTERING: Linear only if extension supported
        // Prefer HalfFloatLinear for half-float textures, fallback to FloatLinear or NEAREST
        const canFilter = this.extColorHalfFloatLinear || this.extColorFloatLinear;
        const filter = canFilter ? gl.LINEAR : gl.NEAREST;

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);

        this.textures.image = texture;
        this.imageWidth = width;
        this.imageHeight = height;
        this.isLinearTexture = true; // Float16 data is scene-linear

        console.log(`[Radiance] Loaded ${width}×${height} float16 HDR texture (linear, HALF_FLOAT, ${filter === gl.LINEAR ? 'LINEAR' : 'NEAREST'})`);
        return texture;
    }

    // Read pixels from WebGL framebuffer for export
    readPixels() {
        const gl = this.gl;
        if (!this.textures.image) return null;
        const w = this.canvas.width, h = this.canvas.height;
        const pixels = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        return { data: pixels, width: w, height: h };
    }

    // Load 3D LUT from .cube file data (WebGL2: float32, WebGL1: fallback)
    loadLUT(lutData, size = 33) {
        const gl = this.gl;

        if (this.textures.lut) {
            gl.deleteTexture(this.textures.lut);
        }

        this.lutSize = size;

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_3D, texture);

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        // WebGL2: Use float32 for precision
        const internalFormat = this.isWebGL2 ? gl.RGB32F : gl.RGB;
        const dataType = this.isWebGL2 ? gl.FLOAT : gl.UNSIGNED_BYTE;

        gl.texImage3D(
            gl.TEXTURE_3D, 0, internalFormat,
            size, size, size, 0,
            gl.RGB, dataType,
            lutData
        );


        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        this.textures.lut = texture;
        console.log(`[Radiance] Loaded ${size}³ 3D LUT (${this.isWebGL2 ? 'float32' : 'uint8'})`);

        return texture;
    }

    // Load Depth Map (supports Float32/Float16/Image)
    loadDepthTexture(image) {
        const gl = this.gl;

        if (this.textures.depth) {
            gl.deleteTexture(this.textures.depth);
        }

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        // Check if raw buffer data (Float32Array or similar)
        // Expected format: { data: TypedArray, width: number, height: number } or just Image/Canvas
        if (image.data && (image.data instanceof Float32Array || image.data instanceof Uint16Array)) {
            const width = image.width;
            const height = image.height;
            const isFloat32 = image.data instanceof Float32Array;

            // WebGL 2.0 internal formats
            const internalFormat = isFloat32 ? gl.R32F : gl.R16F;
            const type = isFloat32 ? gl.FLOAT : gl.HALF_FLOAT;

            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, gl.RED, type, image.data);

            console.log(`[Radiance] Loaded ${width}×${height} depth texture (${isFloat32 ? 'Float32' : 'Float16'})`);
        } else {
            // Standard Image/Canvas (8-bit)
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, image);
        }

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); // Nearest for depth is usually safer for sharp edges
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        this.textures.depth = texture;
        return texture;
    }

    // Load/Update Reference Still texture (for comparison)
    updateReferenceStill(image) {
        const gl = this.gl;
        if (!image) return;

        if (this.textures.reference) {
            gl.deleteTexture(this.textures.reference);
        }

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        this.textures.reference = texture;
        console.log("[Radiance] Reference still texture updated.");
    }

    // Parse .cube LUT file
    parseCubeFile(cubeText) {
        const lines = cubeText.split('\n');
        let size = 33;
        const data = [];

        for (const line of lines) {
            const trimmed = line.trim();

            if (trimmed.startsWith('LUT_3D_SIZE')) {
                size = parseInt(trimmed.split(/\s+/)[1]);
            } else if (trimmed && !trimmed.startsWith('#') && !trimmed.startsWith('TITLE') && !trimmed.startsWith('DOMAIN')) {
                const values = trimmed.split(/\s+/).map(parseFloat);
                if (values.length >= 3 && !isNaN(values[0])) {
                    data.push(values[0], values[1], values[2]);
                }
            }
        }

        console.log(`[Radiance] Parsed LUT: ${size}x${size}x${size}, ${data.length / 3} entries`);
        return { size, data: new Float32Array(data) };
    }

    // Render with basic exposure/gamma
    renderBasic() {
        const gl = this.gl;
        const program = this.programs.basic;

        if (!program || !this.textures.image) return;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);

        // Set uniforms
        gl.uniform1f(this.getUniform(program, 'u_exposure'), this.exposure);
        gl.uniform1f(this.getUniform(program, 'u_gamma'), this.gamma);
        gl.uniform1f(this.getUniform(program, 'u_saturation'), this.saturation);
        gl.uniform1i(this.getUniform(program, 'u_isLinear'), this.isLinearTexture ? 1 : 0);

        // Bind image texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        gl.uniform1i(this.getUniform(program, 'u_image'), 0);

        // Draw quad
        this.drawQuad(program);
    }

    // Render Composite (Image + LUT + DoF)
    render(lutStrength = 1.0) {
        const gl = this.gl;
        const program = this.programs.composite;

        if (!program || !this.textures.image) return;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);

        // Common Uniforms
        this._uf1(program, 'u_exposure', this.exposure);
        this._ui1(program, 'u_frame', this.frame || 0);
        this._uf1(program, 'u_time', this.time || 0.0);

        // Advanced Grading
        this._uf3(program, 'u_lift', this.lift[0], this.lift[1], this.lift[2]);
        this._uf3(program, 'u_gamma', this.gradingGamma[0], this.gradingGamma[1], this.gradingGamma[2]);
        this._uf3(program, 'u_gain', this.gain[0], this.gain[1], this.gain[2]);
        this._uf3(program, 'u_offset', this.offset[0], this.offset[1], this.offset[2]);

        this._uf1(program, 'u_temperature', this.temperature);
        this._uf1(program, 'u_tint', this.tint);
        this._uf1(program, 'u_contrast', this.contrast);
        this._uf1(program, 'u_pivot', this.pivot);

        // Curves (Texture unit 3)
        this._uf1(program, 'u_curveMix', this.curveMix);
        gl.activeTexture(gl.TEXTURE3);
        if (this.curveLutTexture) {
            gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
        } else {
            // Create default identity texture if missing
            this.updateCurveLut(this.curveData);
            gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
        }
        this._ui1(program, 'u_curveLut', 3);

        this._uf1(program, 'u_saturation', this.saturation);

        // v3.0: Resolve-style grading uniforms
        this._uf1(program, 'u_colorBoost', this.colorBoost);
        this._uf1(program, 'u_shadows', this.shadows);
        this._uf1(program, 'u_highlights', this.highlights);
        this._uf1(program, 'u_midDetail', this.midDetail);
        this._uf1(program, 'u_hueShift', this.hueShift);
        this._uf1(program, 'u_lumaMix', this.lumaMix);

        // Qualifiers
        this._ui1(program, 'u_qualifierEnabled', this.qualifierEnabled ? 1 : 0);
        this._ui1(program, 'u_qualifierShowMask', this.qualifierShowMask ? 1 : 0);

        this._uf1(program, 'u_qualifierHue', this.qualifier.h);
        this._uf1(program, 'u_qualifierHueWidth', this.qualifier.hW);
        this._uf1(program, 'u_qualifierHueSoft', this.qualifier.hS);

        this._uf1(program, 'u_qualifierSat', this.qualifier.s);
        this._uf1(program, 'u_qualifierSatWidth', this.qualifier.sW);
        this._uf1(program, 'u_qualifierSatSoft', this.qualifier.sS);

        this._uf1(program, 'u_qualifierLuma', this.qualifier.l);
        this._uf1(program, 'u_qualifierLumaWidth', this.qualifier.lW);
        this._uf1(program, 'u_qualifierLumaSoft', this.qualifier.lS);

        // v3.1 Masking
        this._ui1(program, 'u_maskType', this.mask.type);
        this._uf2(program, 'u_maskCenter', this.mask.center[0], this.mask.center[1]);
        this._uf2(program, 'u_maskScale', this.mask.scale[0], this.mask.scale[1]);
        this._uf1(program, 'u_maskFeather', this.mask.feather);
        this._uf1(program, 'u_maskRotation', this.mask.rotation);
        this._ui1(program, 'u_maskInvert', this.mask.invert ? 1 : 0);
        this._ui1(program, 'u_maskShowOverlay', this.mask.showOverlay ? 1 : 0);

        this._uf2(program, 'u_texSize', this.imageWidth, this.imageHeight);

        // Bind Image (Unit 0)
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        this._ui1(program, 'u_image', 0);

        // LUT (Unit 1)
        this._ui1(program, 'u_lut', 1); // Always set unit 1
        if (this.textures.lut) {
            this._ui1(program, 'u_lutEnabled', 1);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_3D, this.textures.lut);
            // u_lut set above
            this._uf1(program, 'u_lutSize', this.lutSize);
            this._uf1(program, 'u_lutStrength', lutStrength);
        } else {
            this._ui1(program, 'u_lutEnabled', 0);
            // Bind dummy or null to unit 1 to prevent warning? 
            // WebGL is okay if we don't sample from it, but some drivers complain if unit is missing.
            // Best practice: Bind null or a dummy texture if logic skips sampling.
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_3D, null);
        }

        // Depth / DoF (Unit 2)
        this._ui1(program, 'u_depth', 2); // Always set unit 2
        if ((this.dofEnabled || this.showDepth) && this.textures.depth) {
            this._ui1(program, 'u_dofEnabled', 1);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, this.textures.depth);
            // u_depth set above

            this._uf1(program, 'u_focusDist', this.focusDistance);
            this._uf1(program, 'u_aperture', this.aperture);
        } else {
            this._ui1(program, 'u_dofEnabled', 0);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, null);
        }

        // Analytics Uniforms
        this._ui1(program, 'u_falseColor', this.falseColor ? 1 : 0);
        this._ui1(program, 'u_zebra', this.zebra ? 1 : 0);
        this._uf1(program, 'u_zebraThreshold', this.zebraThreshold);
        this._ui1(program, 'u_gamutWarning', this.gamutWarning ? 1 : 0);
        this._ui1(program, 'u_clippingMonitor', this.clippingMonitor ? 1 : 0);

        // HDR pipeline: tell shader whether texture is linear float or sRGB PNG
        this._ui1(program, 'u_isLinear', this.isLinearTexture ? 1 : 0);

        // v2.2: Channel isolation, focus peaking, display LUT
        this._ui1(program, 'u_channelMode', this.channelMode);
        this._ui1(program, 'u_focusPeaking', this.focusPeaking ? 1 : 0);
        this._uf1(program, 'u_focusPeakThreshold', this.focusPeakingThreshold);
        this._ui1(program, 'u_displayLutMode', this.displayLutMode);
        this._uf1(program, 'u_displayLutStrength', this.displayLutStrength);

        // v2.2 Pro: Wipe & Reference
        this._ui1(program, 'u_wipeEnabled', this.wipeEnabled ? 1 : 0);
        this._uf1(program, 'u_wipe', this.wipe);
        this._ui1(program, 'u_wipeRefEnabled', this.wipeRefEnabled ? 1 : 0);

        gl.activeTexture(gl.TEXTURE4);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.reference || this.textures.empty);
        this._ui1(program, 'u_referenceImage', 4);

        // v2.2 Pro: Grids
        this._ui1(program, 'u_gridMode', this.gridMode);
        this._uf4v(program, 'u_gridColor', this.gridColor);

        // v2.3: Denoise & Depth Eval
        this._uf1(program, 'u_denoise', this.denoise);
        this._ui1(program, 'u_showDepth', this.showDepth ? 1 : 0);
        this._uf1(program, 'u_grainAmount', this.grainAmount || 0.0);

        // v2.6: Lens Effects
        this._ui1(program, 'u_apertureBlades', this.apertureBlades);
        this._uf1(program, 'u_apertureRotation', this.apertureRotation);
        this._uf1(program, 'u_apertureAnamorphic', this.apertureAnamorphic);
        this._uf1(program, 'u_lensDistortion', this.lensDistortion);
        this._uf1(program, 'u_lensFringe', this.lensFringe);
        this._uf1(program, 'u_vignetteIntensity', this.vignetteIntensity);
        this._uf1(program, 'u_vignetteFalloff', this.vignetteFalloff);

        // v2.6.1: Realistic Bokeh Physics
        this._uf1(program, 'u_bokehHighlightBias', this.bokehHighlightBias || 0.0);
        this._uf1(program, 'u_bokehSoapBubble', this.bokehSoapBubble || 0.0);
        this._uf1(program, 'u_bokehOpticalVig', this.bokehOpticalVig || 0.0);

        // v3.1: Extended Grain & Lens
        this._uf1(program, 'u_grainSize', this.grainSize || 1.0);
        this._uf1(program, 'u_grainColor', this.grainColor || 0.0);
        this._uf1(program, 'u_bloom', this.bloom || 0.0);
        this._uf1(program, 'u_halation', this.halation || 0.0);
        this._uf1(program, 'u_diffusion', this.diffusion || 0.0);

        // ── v3.2 Phase 11: Anamorphic Streaks + k2 Distortion ────────────────
        this._uf1(program, 'u_lensDistortionK2', this.lensDistortionK2 || 0.0);
        this._uf1(program, 'u_anamorphicStreaks', this.anamorphicStreaks || 0.0);
        this._uf1(program, 'u_streakThreshold', this.streakThreshold ?? 0.85);
        this._uf1(program, 'u_streakLength', this.streakLength ?? 0.08);

        // ── v3.2 Phase 12: Bilateral Filter ──────────────────────────────────
        this._uf1(program, 'u_bilateralSigmaD', this.bilateralSigmaD ?? 3.0);
        this._uf1(program, 'u_bilateralSigmaR', this.bilateralSigmaR ?? 0.10);
        this._ui1(program, 'u_bilateralHalfRes', this.bilateralHalfRes ? 1 : 0);

        // Animated Grain
        // u_time changes every frame — always upload, no cache benefit
        this.gl.uniform1f(this.getUniform(program, 'u_time'), (performance.now() / 1000.0) % 100.0);

        this.drawQuad(program);
    }

    // Legacy support
    renderWithLUT(strength) {
        this.render(strength);
    }

    // Render RGB Parade waveform
    renderParade(brightness = 10.0) {
        const gl = this.gl;
        const program = this.programs.parade;

        if (!program || !this.textures.image) return;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0.05, 0.05, 0.08, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);

        gl.uniform2f(this.getUniform(program, 'u_resolution'), this.canvas.width, this.canvas.height);
        gl.uniform1f(this.getUniform(program, 'u_brightness'), brightness);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        gl.uniform1i(this.getUniform(program, 'u_image'), 0);

        this.drawQuad(program);
    }

    drawQuad(program) {
        const gl = this.gl;

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);

        const posLoc = this.getAttrib(program, 'a_position');
        const texLoc = this.getAttrib(program, 'a_texcoord');

        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 16, 0);

        gl.enableVertexAttribArray(texLoc);
        gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 16, 8);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }







    setFalseColor(enabled) {
        this.falseColor = enabled;
    }

    setZebra(enabled) {
        this.zebra = enabled;
    }

    setGamutWarning(enabled) {
        this.gamutWarning = enabled;
    }

    setClippingMonitor(enabled) {
        this.clippingMonitor = enabled;
    }

    setZebraThreshold(threshold) {
        this.zebraThreshold = threshold;
    }

    setChannelMode(mode) {
        this.channelMode = mode;
    }



    setFocusPeaking(enabled, threshold = 30.0) {
        this.focusPeaking = enabled;
        this.focusPeakingThreshold = threshold;
    }

    setDisplayLutMode(mode) {
        this.displayLutMode = mode;
    }









    destroy() {
        const gl = this.gl;

        // Clean up bilateral FBO
        this._destroyBilateralFBO();

        // Clean up textures
        for (const tex of Object.values(this.textures)) {
            if (tex) gl.deleteTexture(tex);
        }

        // Clean up programs
        for (const prog of Object.values(this.programs)) {
            if (prog) gl.deleteProgram(prog);
        }

        // Clean up buffers
        if (this.quadBuffer) gl.deleteBuffer(this.quadBuffer);

        console.log('[Radiance] WebGL renderer destroyed');
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           SEQUENCE PLAYER
// ═══════════════════════════════════════════════════════════════════════════════

class RadianceSequencePlayer {
    constructor(viewer) {
        this.viewer = viewer;
        this.frames = [];
        this.currentFrame = 0;
        this.fps = 24;
        this.isPlaying = false;
        this.loop = true;
        this.frameCache = new Map();
        this.maxCacheSize = 100; // Max frames in cache

        this.playInterval = null;
    }

    loadSequence(frameUrls) {
        this.frames = frameUrls;
        this.currentFrame = 0;
        this.clearCache();

        // Preload first few frames
        this.preloadFrames(0, Math.min(5, this.frames.length));

        console.log(`[Radiance] Loaded sequence with ${this.frames.length} frames`);
    }

    preloadFrames(start, count) {
        for (let i = start; i < start + count && i < this.frames.length; i++) {
            if (!this.frameCache.has(i)) {
                this.loadFrame(i);
            }
        }
    }

    loadFrame(index) {
        return new Promise((resolve, reject) => {
            if (this.frameCache.has(index)) {
                resolve(this.frameCache.get(index));
                return;
            }

            const img = new Image();
            img.crossOrigin = 'anonymous';

            img.onload = () => {
                // Manage cache size
                if (this.frameCache.size >= this.maxCacheSize) {
                    // Remove oldest frame (furthest from current)
                    const keys = Array.from(this.frameCache.keys());
                    const distances = keys.map(k => Math.abs(k - this.currentFrame));
                    const maxDist = Math.max(...distances);
                    const toRemove = keys[distances.indexOf(maxDist)];
                    this.frameCache.delete(toRemove);
                }

                this.frameCache.set(index, img);
                resolve(img);
            };

            img.onerror = reject;
            img.src = this.frames[index];
        });
    }

    async displayFrame(index) {
        if (index < 0 || index >= this.frames.length) return;

        this.currentFrame = index;

        try {
            const img = await this.loadFrame(index);
            this.viewer.setImage(img);

            // Preload ahead
            this.preloadFrames(index + 1, 3);
        } catch (e) {
            console.error(`[Radiance] Failed to load frame ${index}: `, e);
        }
    }

    play() {
        if (this.isPlaying) return;
        if (this.frames.length === 0) return;

        this.isPlaying = true;
        const interval = 1000 / this.fps;

        this.playInterval = setInterval(() => {
            let nextFrame = this.currentFrame + 1;

            if (nextFrame >= this.frames.length) {
                if (this.loop) {
                    nextFrame = 0;
                } else {
                    this.pause();
                    return;
                }
            }

            this.displayFrame(nextFrame);
        }, interval);

        console.log(`[Radiance] Playing at ${this.fps} fps`);
    }

    pause() {
        if (!this.isPlaying) return;

        this.isPlaying = false;
        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }

        console.log('[Radiance] Paused');
    }

    togglePlayPause() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    stop() {
        this.pause();
        this.currentFrame = 0;
        this.displayFrame(0);
    }

    nextFrame() {
        this.pause();
        const next = (this.currentFrame + 1) % this.frames.length;
        this.displayFrame(next);
    }

    prevFrame() {
        this.pause();
        const prev = (this.currentFrame - 1 + this.frames.length) % this.frames.length;
        this.displayFrame(prev);
    }

    goToFrame(index) {
        this.pause();
        this.displayFrame(Math.max(0, Math.min(index, this.frames.length - 1)));
    }

    setFPS(fps) {
        this.fps = Math.max(1, Math.min(fps, 120));

        if (this.isPlaying) {
            this.pause();
            this.play();
        }
    }

    setLoop(loop) {
        this.loop = loop;
    }

    clearCache() {
        this.frameCache.clear();
    }

    getProgress() {
        if (this.frames.length === 0) return 0;
        return this.currentFrame / (this.frames.length - 1);
    }

    getTimecode() {
        const totalSeconds = this.currentFrame / this.fps;
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60);
        const frames = this.currentFrame % this.fps;

        return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}:${String(frames).padStart(2, '0')} `;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           RGB PARADE WAVEFORM
// ═══════════════════════════════════════════════════════════════════════════════

class RadianceRGBParade {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
    }

    render(imageData) {
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;

        // Clear
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, w, h);

        // Section width (3 channels)
        const sectionWidth = Math.floor(w / 3);
        const padding = 4;

        // Draw each channel
        this.drawChannel(imageData, 0, 0, sectionWidth - padding, h, 'red');
        this.drawChannel(imageData, 1, sectionWidth, sectionWidth - padding, h, 'green');
        this.drawChannel(imageData, 2, sectionWidth * 2, sectionWidth - padding, h, 'blue');

        // Draw separators
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sectionWidth, 0);
        ctx.lineTo(sectionWidth, h);
        ctx.moveTo(sectionWidth * 2, 0);
        ctx.lineTo(sectionWidth * 2, h);
        ctx.stroke();

        // Draw 0%, 50%, 100% lines
        ctx.strokeStyle = '#444';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, h * 0.5);
        ctx.lineTo(w, h * 0.5);
        ctx.stroke();
        ctx.setLineDash([]);

        // Labels
        ctx.fillStyle = '#666';
        ctx.font = '10px monospace';
        ctx.fillText('R', 4, 12);
        ctx.fillText('G', sectionWidth + 4, 12);
        ctx.fillText('B', sectionWidth * 2 + 4, 12);
    }

    drawChannel(imageData, channelIndex, xOffset, width, height, color) {
        const ctx = this.ctx;
        const data = imageData.data;
        const imgWidth = imageData.width;
        const imgHeight = imageData.height;

        // Sample columns
        const columns = Math.min(width, imgWidth);
        const columnWidth = imgWidth / columns;

        // Create histogram for each column
        for (let col = 0; col < columns; col++) {
            const sourceCol = Math.floor(col * columnWidth);
            const histogram = new Uint32Array(256);

            // Sample pixels in this column
            for (let row = 0; row < imgHeight; row++) {
                const pixelIndex = (row * imgWidth + sourceCol) * 4;
                const value = data[pixelIndex + channelIndex];
                histogram[value]++;
            }

            // Draw column
            const x = xOffset + col;
            for (let y = 0; y < 256; y++) {
                const count = histogram[y];
                if (count > 0) {
                    const intensity = Math.min(count / (imgHeight * 0.1), 1.0);
                    const yPos = height - (y / 255) * height;

                    ctx.fillStyle = this.getChannelColor(color, intensity);
                    ctx.fillRect(x, yPos - 1, 1, 2);
                }
            }
        }
    }

    getChannelColor(channel, intensity) {
        const alpha = Math.min(intensity * 0.8 + 0.2, 1.0);
        switch (channel) {
            case 'red': return `rgba(255, 80, 80, ${alpha})`;
            case 'green': return `rgba(80, 255, 80, ${alpha})`;
            case 'blue': return `rgba(80, 120, 255, ${alpha})`;
            default: return `rgba(255, 255, 255, ${alpha})`;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                      HDR COLOR PICKER  (v3.2 Fix 4)
//
//  Reads the .rhdr float16 sidecar that the Python backend saves alongside
//  every image and returns the true scene-linear float value at any UV.
//
//  This is equivalent to Nuke's "info" toolbar pixel inspector or RV's
//  color picker — values are in raw linear light, not display-encoded.
//
//  Usage:
//      const picker = new RadianceHDRPicker('/view?filename=...rhdr&type=temp');
//      await picker.load();
//      const px = picker.sample(0.5, 0.5);  // → { r, g, b, a, luma, stops }
//
//  Events:
//      picker.onHover(canvas, (px) => showReadout(px));
//      picker.onPick(canvas, (px, uv) => logColor(px, uv));
// ═══════════════════════════════════════════════════════════════════════════════

class RadianceHDRPicker {
    /**
     * @param {string} rhdrUrl – URL to the .rhdr file served by ComfyUI's
     *                           /view endpoint, e.g.
     *                           '/view?filename=radiance_viewer_abc123_0.rhdr&type=temp'
     */
    constructor(rhdrUrl) {
        this.url = rhdrUrl;
        this.width = 0;
        this.height = 0;
        this.channels = 0;
        this.data = null;   // Float32Array — converted from fp16 on load
        this.loaded = false;
        this._abortCtrl = null;
    }

    // ── Load & decode .rhdr ─────────────────────────────────────────────────

    async load() {
        if (this.loaded) return this;

        this._abortCtrl = new AbortController();
        const res = await fetch(this.url, { signal: this._abortCtrl.signal });
        if (!res.ok) throw new Error(`[Picker] Failed to fetch ${this.url}: ${res.status}`);

        const compressed = await res.arrayBuffer();
        const bytes = new Uint8Array(compressed);

        // ── Parse .rhdr header ───────────────────────────────────────────────
        // Magic: 4 bytes 'RHDR'
        // Width:    uint16 LE
        // Height:   uint16 LE
        // Channels: uint16 LE
        // Reserved: uint16 LE
        const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
        if (magic !== 'RHDR') throw new Error(`[Picker] Not a .rhdr file (magic: ${magic})`);

        const view = new DataView(compressed);
        this.width = view.getUint16(4, true);
        this.height = view.getUint16(6, true);
        this.channels = view.getUint16(8, true);
        const HEADER_SIZE = 12;

        // ── Zlib decompress ──────────────────────────────────────────────────
        // ComfyUI serves files directly — use DecompressionStream (Chrome/Firefox 103+)
        const compressedData = compressed.slice(HEADER_SIZE);
        const ds = new DecompressionStream('deflate');
        const writer = ds.writable.getWriter();
        const reader = ds.readable.getReader();

        writer.write(compressedData);
        writer.close();

        const chunks = [];
        let totalLen = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            totalLen += value.length;
        }

        const decompressed = new Uint8Array(totalLen);
        let offset = 0;
        for (const chunk of chunks) {
            decompressed.set(chunk, offset);
            offset += chunk.length;
        }

        // ── fp16 → float32 ───────────────────────────────────────────────────
        const fp16 = new Uint16Array(decompressed.buffer);
        this.data = new Float32Array(fp16.length);

        for (let i = 0; i < fp16.length; i++) {
            this.data[i] = RadianceHDRPicker._fp16ToFloat32(fp16[i]);
        }

        this.loaded = true;
        console.log(`[Picker] Loaded ${this.width}×${this.height}×${this.channels}ch .rhdr`);
        return this;
    }

    cancel() {
        if (this._abortCtrl) this._abortCtrl.abort();
    }

    // ── Sample ──────────────────────────────────────────────────────────────

    /**
     * Sample the linear float value at normalised UV coordinates.
     * UV (0,0) = top-left, (1,1) = bottom-right.
     *
     * @param {number} u – 0..1 horizontal
     * @param {number} v – 0..1 vertical
     * @returns {{ r, g, b, a, luma, stops, hex }} or null if not loaded
     */
    sample(u, v) {
        if (!this.loaded || !this.data) return null;

        const x = Math.max(0, Math.min(this.width - 1, Math.floor(u * this.width)));
        const y = Math.max(0, Math.min(this.height - 1, Math.floor(v * this.height)));
        const idx = (y * this.width + x) * this.channels;

        const r = this.data[idx] ?? 0;
        const g = this.data[idx + 1] ?? 0;
        const b = this.data[idx + 2] ?? 0;
        const a = this.channels >= 4 ? (this.data[idx + 3] ?? 1) : 1;

        const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const stops = luma > 1e-6 ? Math.log2(luma / 0.18) : -Infinity;

        // Display-encoded hex (clamped sRGB for UI colour swatches)
        const toU8 = v => Math.round(Math.min(Math.max(Math.pow(Math.max(v, 0), 1 / 2.2), 0), 1) * 255);
        const hex = '#' + [r, g, b].map(toU8).map(n => n.toString(16).padStart(2, '0')).join('');

        return { r, g, b, a, luma, stops, hex, x, y };
    }

    /**
     * Sample a 3×3 average around a UV point (reduces noise for precise grading).
     */
    sampleAverage(u, v, radius = 2) {
        if (!this.loaded) return null;
        let rs = 0, gs = 0, bs = 0, n = 0;
        const pxW = 1 / this.width;
        const pxH = 1 / this.height;
        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                const s = this.sample(u + dx * pxW, v + dy * pxH);
                if (s) { rs += s.r; gs += s.g; bs += s.b; n++; }
            }
        }
        if (!n) return null;
        return this.sample(u, v) && { ...this.sample(u, v), r: rs / n, g: gs / n, b: bs / n };
    }

    // ── Canvas Event Helpers ────────────────────────────────────────────────

    /**
     * Attach a hover listener to a canvas that calls cb(pixelInfo, uv) on mousemove.
     * Returns a cleanup function.
     */
    onHover(canvas, cb) {
        const handler = (e) => {
            const uv = this._canvasUV(canvas, e);
            const px = this.sample(uv.u, uv.v);
            if (px) cb(px, uv);
        };
        canvas.addEventListener('mousemove', handler);
        return () => canvas.removeEventListener('mousemove', handler);
    }

    /**
     * Attach a click listener to a canvas that calls cb(pixelInfo, uv) on click.
     * Returns a cleanup function.
     */
    onPick(canvas, cb) {
        const handler = (e) => {
            const uv = this._canvasUV(canvas, e);
            const px = this.sample(uv.u, uv.v);
            if (px) cb(px, uv);
        };
        canvas.addEventListener('click', handler);
        return () => canvas.removeEventListener('click', handler);
    }

    _canvasUV(canvas, e) {
        const rect = canvas.getBoundingClientRect();
        return {
            u: (e.clientX - rect.left) / rect.width,
            v: (e.clientY - rect.top) / rect.height,
        };
    }

    // ── fp16 → float32 ──────────────────────────────────────────────────────
    // IEEE 754 half-float to single-precision conversion.
    // Handles: ±zero, denormals, ±inf, NaN.

    static _fp16ToFloat32(h) {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;

        if (e === 0) {
            // Denormal / zero
            const val = f === 0 ? 0 : Math.pow(2, -14) * (f / 1024);
            return s ? -val : val;
        }
        if (e === 31) {
            // Inf / NaN
            return s ? -Infinity : (f ? NaN : Infinity);
        }
        const val = Math.pow(2, e - 15) * (1 + f / 1024);
        return s ? -val : val;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                    HDR PIXEL READOUT HUD  (v3.2 Fix 4)
//
//  Floating tooltip-style panel that shows the raw linear float values
//  under the cursor, matching Nuke's info bar layout.
// ═══════════════════════════════════════════════════════════════════════════════

class RadiancePixelReadout {
    /**
     * @param {HTMLElement}        container  – overlay div (position:relative parent)
     * @param {RadianceHDRPicker}  picker     – loaded picker instance
     * @param {HTMLCanvasElement}  glCanvas   – the WebGL display canvas
     */
    constructor(container, picker, glCanvas) {
        this.picker = picker;
        this.canvas = glCanvas;
        this._cleanup = [];

        // ── Build HUD element ─────────────────────────────────────────────────
        this.el = document.createElement('div');
        this.el.className = 'rad-hud-panel rad-pixel-readout';
        this.el.style.cssText = `
            position: absolute;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 6px 14px;
            pointer-events: none;
            white-space: nowrap;
            z-index: 130;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
        `;

        // Colour swatch
        this._swatch = document.createElement('div');
        this._swatch.style.cssText = 'width:18px;height:18px;border-radius:3px;border:1px solid rgba(255,255,255,0.2);flex-shrink:0;';
        this.el.appendChild(this._swatch);

        // Channel readouts
        this._channels = ['R', 'G', 'B', 'A', 'L', 'EV'].map(label => {
            const span = document.createElement('span');
            span.style.cssText = 'color:rgba(255,255,255,0.5);';
            this.el.appendChild(span);
            return span;
        });

        // Coordinate
        this._coord = document.createElement('span');
        this._coord.style.cssText = 'color:rgba(255,255,255,0.25);font-size:9px;';
        this.el.appendChild(this._coord);

        container.appendChild(this.el);

        // ── Attach events ─────────────────────────────────────────────────────
        const rmHover = picker.onHover(glCanvas, (px, uv) => this._update(px, uv));
        this._cleanup.push(rmHover);

        // Hide when cursor leaves canvas
        const onLeave = () => { this.el.style.opacity = '0'; };
        const onEnter = () => { this.el.style.opacity = '1'; };
        glCanvas.addEventListener('mouseleave', onLeave);
        glCanvas.addEventListener('mouseenter', onEnter);
        this._cleanup.push(() => {
            glCanvas.removeEventListener('mouseleave', onLeave);
            glCanvas.removeEventListener('mouseenter', onEnter);
        });
    }

    _update(px, uv) {
        const fmt = (v) => v.toFixed(4).padStart(7);
        const fmtStops = (s) => isFinite(s) ? (s >= 0 ? '+' : '') + s.toFixed(2) + ' EV' : '—';

        this._swatch.style.background = px.hex;

        const labels = ['R', 'G', 'B', 'A', 'Y'];
        const values = [px.r, px.g, px.b, px.a, px.luma];
        const colors = ['#ff6060', '#60ff90', '#6090ff', '#aaaaaa', '#cccccc'];

        labels.forEach((label, i) => {
            const ch = this._channels[i];
            ch.innerHTML = `<span style="color:${colors[i]};font-weight:700">${label}</span> ${fmt(values[i])}`;
        });

        this._channels[5].innerHTML =
            `<span style="color:#ffcc44;font-weight:700">EV</span> ${fmtStops(px.stops)}`;

        this._coord.textContent = `(${px.x}, ${px.y})`;
    }

    destroy() {
        this._cleanup.forEach(fn => fn());
        this.el.remove();
    }
}




// Export for use in radiance_viewer.js
if (typeof window !== 'undefined') {
    window.RadianceWebGLRenderer = RadianceWebGLRenderer;
    window.RadianceSequencePlayer = RadianceSequencePlayer;
    window.RadianceRGBParade = RadianceRGBParade;
    window.RadianceHDRPicker = RadianceHDRPicker;
    window.RadiancePixelReadout = RadiancePixelReadout;
}

export { RadianceWebGLRenderer, RadianceSequencePlayer, RadianceRGBParade, RadianceHDRPicker, RadiancePixelReadout };
