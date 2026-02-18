/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                       RADIANCE WEBGL RENDERER v2.2
 *                    GPU-Accelerated Viewer Enhancement
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * v2.1 Changes:
 * - FIX: Added sRGB OETF to composite shader (fixes BLACK result for HDR)
 * - FIX: Linear vs sRGB texture tracking (isLinearTexture flag)
 * - FIX: PNG input now linearized before grading (correct color math)
 * - FIX: All grading operations now happen in linear space
 * - FIX: sRGB OETF always applied as final display transform
 * 
 * Phase 5 Features:
 * - WebGL 2.0 rendering for GPU acceleration
 * - Real-time 3D LUT application in viewer
 * - 32-bit float texture support for HDR
 * - Sequence playback with frame caching
 * - RGB Parade waveform display
 */

// WebGL Context Manager
class RadianceWebGLRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = null;
        this.programs = {};
        this.textures = {};
        this.framebuffers = {};
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

        this.init();
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

    // v2.4 Curves
    updateCurveLut(data) {
        // Create 256x1 texture from data
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
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    }

    setCurveMix(v) { this.curveMix = v; }

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
            console.log('[Radiance] WebGL 2.0 initialized - HDR pipeline enabled');
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

        console.log(`[Radiance] WebGL initialized (WebGL2=${this.isWebGL2}, FloatLinear=${!!this.extColorFloatLinear}, HalfFloatLinear=${!!this.extColorHalfFloatLinear})`);
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

        console.log('[Radiance] Shader programs compiled');
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

            // HDR pipeline: true when texture contains linear float data
            uniform bool u_isLinear;

            // v2.2: Channel isolation (0=RGB, 1=R, 2=G, 3=B, 4=Luma, 5=Alpha)
            uniform int u_channelMode;
            // v2.2: Focus peaking
            uniform bool u_focusPeaking;
            uniform float u_focusPeakThreshold;
            // v2.2: Display LUT mode (0=None, 1=sRGB, 2=Rec.709, 3=LogC3, 4=ACEScg)
            uniform int u_displayLutMode;
            
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
            
            // v3.1: Extended Grain & Lens
            uniform float u_grainSize;
            uniform float u_grainColor;
            uniform float u_bloom;
            uniform float u_halation;
            uniform float u_diffusion;

            uniform float u_time; // For animated grain

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

            vec3 applyShadowsHighlights(vec3 color, float shadows, float highlights) {
                if (shadows == 0.0 && highlights == 0.0) return color;
                
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                
                // Shadows: affect dark areas (< 0.33)
                // Highlights: affect bright areas (> 0.66)
                // Clean implementation using smoothstep masks
                
                float shadowMask = 1.0 - smoothstep(0.0, 0.5, luma);
                float highlightMask = smoothstep(0.5, 1.0, luma);
                
                // Shadow recovery (brighten darks) or crush
                // We use exposure-like adjustment tailored to mask
                
                // Shadows: Resolve style (brings up details)
                // Simple approach: Gamma correction weighted by mask?
                // Or Offset weighted by mask? Lift weighted by mask constitutes "Shadows"
                
                vec3 sRes = color;
                if (shadows != 0.0) {
                   // Shadows slider: -1.0 to 1.0
                   // +: Brightens shadows (Lift-like)
                   // -: Crushes shadows
                   float fac = pow(2.0, shadows * 2.0); // Exposure-like factor
                   sRes = mix(color, color * fac, shadowMask);
                }
                
                vec3 hRes = sRes;
                if (highlights != 0.0) {
                   // Highlights slider: -1.0 to 1.0
                   // +: Brightens highlights (blow out)
                   // -: Recovers highlights (Gain reduction)
                   float fac = pow(2.0, highlights * 2.0);
                   hRes = mix(sRes, sRes * fac, highlightMask);
                }
                
                return hRes;
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

const int SAMPLE_COUNT = 32; // Increased for better quality
const float PI = 3.14159265;
const float GOLDEN_ANGLE = 2.39996323;

            // Brown-Conrady Distortion (k1 only for now)
            vec2 distortUV(vec2 uv) {
                if (u_lensDistortion == 0.0) return uv;
                
                vec2 center = uv - 0.5;
                float r2 = dot(center, center);
                float f = 1.0 + r2 * u_lensDistortion;
                
                // Anamorphic distortion? (Maybe later)
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
                
                // Chromatic Aberration offsets (Red/Blue shift)
                vec2 caOffset = (uv - 0.5) * u_lensFringe * 0.02 * anamorphic;

                // Center sample
                acc.r += texture(u_image, uv - caOffset).r;
                acc.g += texture(u_image, uv).g;
                acc.b += texture(u_image, uv + caOffset).b;
                weight += 1.0;
                
                if (radius < 1.0) return acc / weight;

                // Polygon Shape Logic
                float blades = float(u_apertureBlades);
                float bladeRad = radians(360.0 / blades);
                float rot = radians(u_apertureRotation);

                for (int i = 1; i <= SAMPLE_COUNT; i++) {
                    float r = sqrt(float(i) / float(SAMPLE_COUNT));
                    float theta = float(i) * GOLDEN_ANGLE; // Golden angle distribution

                    // Map circle to polygon if needed
                    float polygonScale = 1.0;
                    if (blades >= 3.0) {
                        float dt = theta + rot;
                        // Distance to nearest edge
                        float localTheta = atan(sin(dt), cos(dt)); // -PI to PI
                        // Sector index
                        // No, simpler approach:
                        // r = cos(PI/N) / cos( (theta % (2PI/N)) - PI/N )
                        float phi = theta + rot;
                        float sector = floor(phi / bladeRad + 0.5);
                        float phi_local = phi - sector * bladeRad;
                        polygonScale = cos(bladeRad * 0.5) / cos(phi_local);
                    }
                    
                    // Anamorphic stretch (vertical squeeze / horizontal stretch)
                    // We apply squeeze to Y, or stretch to X. 
                    // Let's stretch X.
                    vec2 offset = vec2(cos(theta), sin(theta)) * r * radius * pixelSize;
                    
                    // Apply polygon shape
                    if (blades >= 3.0) offset *= polygonScale;

                    // Apply Anamorphic Ratio (Stretch X)
                    offset.x *= anamorphic;

                    // Sample with CA
                    vec2 sampleUV = uv + offset;

                    // Check bounds (optional, clamp to edge)
                    // if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) continue;

                    acc.r += texture(u_image, sampleUV - caOffset).r;
                    acc.g += texture(u_image, sampleUV).g;
                    acc.b += texture(u_image, sampleUV + caOffset).b; // Simple CA
                    weight += 1.0;
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
                // Thermal/Jet style map
                vec3 c = vec3(1.0);
                if (v < 0.25) {
                    c = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), v * 4.0);
                } else if (v < 0.5) {
                    c = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (v - 0.25) * 4.0);
                } else if (v < 0.75) {
                    c = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (v - 0.5) * 4.0);
                } else {
                    c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (v - 0.75) * 4.0);
                }
                return c;
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

            // Display LUT transforms (scene-linear → display)
            vec3 applyDisplayLUT(vec3 c, int mode) {
                if (mode == 1) {
                    // Already applying sRGB OETF at the end, so this is identity
                    return c;
                } else if (mode == 2) {
                    // Rec.709 OETF
                    vec3 lo = c * 4.5;
                    vec3 hi = 1.099 * pow(max(c, vec3(0.018)), vec3(0.45)) - vec3(0.099);
                    return mix(hi, lo, vec3(lessThan(c, vec3(0.018))));
                } else if (mode == 3) {
                    // LogC3 → Linear (ARRI LogC3 decode)
                    vec3 lin;
                    lin.r = c.r > 0.1496582 ? (pow(10.0, (c.r - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (c.r / 0.1496582) * (pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272);
                    lin.g = c.g > 0.1496582 ? (pow(10.0, (c.g - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (c.g / 0.1496582) * (pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272);
                    lin.b = c.b > 0.1496582 ? (pow(10.0, (c.b - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (c.b / 0.1496582) * (pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272);
                    return max(lin, vec3(0.0));
                } else if (mode == 4) {
                    // ACEScg → Rec.709 (AP1 matrix + Reinhard tonemap)
                    vec3 r709;
                    r709.r =  c.r * 1.70486 - c.g * 0.62172 - c.b * 0.08330;
                    r709.g = -c.r * 0.19645 + c.g * 1.26433 + c.b * 0.03212;
                    r709.b = -c.r * 0.01777 - c.g * 0.00404 + c.b * 1.02180;
                    r709 = r709 / (r709 + vec3(1.0)); // Reinhard tonemap
                    return max(r709, vec3(0.0));
                }
                return c;
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
            
            // Simple Reinhard Tone Mapping
            vec3 toneMapReinhard(vec3 c) {
                return c / (c + vec3(1.0));
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

            // ACES Tone Mapping (Approx)
            vec3 toneMapACES(vec3 color) {
                const float a = 2.51;
                const float b = 0.03;
                const float c = 2.43;
                const float d = 0.59;
                const float e = 0.14;
                return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
            }

            float rand(vec2 co) {
                // Animated grain seed
                return fract(sin(dot(co.xy + u_time, vec2(12.9898,78.233))) * 43758.5453);
            }

// ----------------------------------------------------------------
// Denoise (Simple Spatial Blur)
// ----------------------------------------------------------------
vec3 getDenoiseColor(vec2 uv) {
    vec2 px = 1.0 / u_texSize;
    vec3 center = texture(u_image, uv).rgb;
    
    // 5-tap box blur
    vec3 sum = center;
    sum += texture(u_image, uv + vec2(-px.x, -px.y)).rgb;
    sum += texture(u_image, uv + vec2( px.x, -px.y)).rgb;
    sum += texture(u_image, uv + vec2(-px.x,  px.y)).rgb;
    sum += texture(u_image, uv + vec2( px.x,  px.y)).rgb;
    
    return sum / 5.0;
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

        // ── QUALIFIER MIX ──
        if (u_qualifierEnabled) {
            float matte = getQualifierMask(preGrade);
            
            if (u_qualifierShowMask) {
                // Show matte as B&W
                color = vec3(matte);
                // Skip everything else? 
                // We typically want to see the matte in context of display transforms, 
                // but usually raw matte is best.
                // Let's allow DisplayLUT/OETF to handle it so it looks correct on screen.
            } else {
                 // Mix graded result with original (preGrade) based on matte
                 // matte=1 -> graded, matte=0 -> original
                 color = mix(preGrade, color, matte);
            }
        }

        // 7. LUT (Global - applied after qualification)
        vec3 lutInput = color;
        if (u_lutEnabled) {
            vec3 lutted = applyLUT(lutInput, u_lut, u_lutSize);
            color = mix(color, lutted, u_lutStrength);
        }

        // 5b. Display LUT transforms
        if (u_displayLutMode > 0) {
            color = applyDisplayLUT(color, u_displayLutMode);
        } else if (u_isLinear) {
            color = toneMapACES(color);
        }
        
        // 6. Display Transform
        color = linearToSRGB(max(color, vec3(0.0)));

        // 7. Bloom (bright pixel glow via area sampling)
        if (u_bloom > 0.0) {
            vec3 bloomAcc = vec3(0.0);
            float bloomW = 0.0;
            vec2 px = 1.0 / u_texSize;
            for (int bx = -3; bx <= 3; bx++) {
                for (int by = -3; by <= 3; by++) {
                    vec2 off = vec2(float(bx), float(by)) * px * 3.0;
                    vec3 s = texture(u_image, uv + off).rgb;
                    float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
                    float w = max(lum - 0.7, 0.0); // only bright pixels contribute
                    bloomAcc += s * w;
                    bloomW += w;
                }
            }
            if (bloomW > 0.0) bloomAcc /= bloomW;
            color += bloomAcc * u_bloom * 0.5;
        }

        // 7a. Halation (red channel highlight bleed — classic film gate bounce)
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

        // 7b. Diffusion (soft filter — blend with blurred version)
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

        // 7c. Film Grain (with size + color)
        if (u_grainAmount > 0.0) {
            vec2 grainUV = uv * u_texSize / u_grainSize;
            float noiseMono = rand(grainUV);
            float grainMono = (noiseMono - 0.5) * u_grainAmount;
            
            if (u_grainColor > 0.0) {
                // RGB noise channels (offset seeds for independence)
                float noiseR = rand(grainUV + vec2(1.7, 3.1));
                float noiseG = rand(grainUV + vec2(5.3, 7.9));
                float noiseB = rand(grainUV + vec2(11.1, 13.7));
                vec3 grainRGB = (vec3(noiseR, noiseG, noiseB) - 0.5) * u_grainAmount;
                color += mix(vec3(grainMono), grainRGB, u_grainColor);
            } else {
                color += grainMono;
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
            if (lumaDisplay >= u_zebraThreshold) {
                float stripe = mod(gl_FragCoord.x + gl_FragCoord.y, 20.0);
                if (stripe < 10.0) {
                    color = vec3(1.0, 0.0, 0.0);
                }
            }
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
            uniform vec2 u_resolution;
            uniform float u_brightness;

void main() {
                // RGB Parade: split screen into 3 sections
                float section = floor(v_texcoord.x * 3.0);
                float localX = fract(v_texcoord.x * 3.0);

                // Sample column of pixels
                float columnSum = 0.0;
                int sampleCount = 256;

    for (int i = 0; i < sampleCount; i++) {
                    float y = float(i) / float(sampleCount - 1);
                    vec4 texSample = texture(u_image, vec2(localX, y));
                    
                    float value;
        if (section < 1.0) {
            value = texSample.r;
        } else if (section < 2.0) {
            value = texSample.g;
        } else {
            value = texSample.b;
        }

        // Check if this pixel should light up
        if (abs(value - (1.0 - v_texcoord.y)) < 0.02) {
            columnSum += 1.0;
        }
    }

                // Colorize based on channel
                vec3 channelColor;
    if (section < 1.0) {
        channelColor = vec3(1.0, 0.2, 0.2);
    } else if (section < 2.0) {
        channelColor = vec3(0.2, 1.0, 0.2);
    } else {
        channelColor = vec3(0.2, 0.4, 1.0);
    }
                
                float intensity = columnSum / float(sampleCount) * u_brightness;
    fragColor = vec4(channelColor * intensity, 1.0);
}
`;
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
        gl.uniform1f(gl.getUniformLocation(program, 'u_exposure'), this.exposure);
        gl.uniform1f(gl.getUniformLocation(program, 'u_gamma'), this.gamma);
        gl.uniform1f(gl.getUniformLocation(program, 'u_saturation'), this.saturation);
        gl.uniform1i(gl.getUniformLocation(program, 'u_isLinear'), this.isLinearTexture ? 1 : 0);

        // Bind image texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        gl.uniform1i(gl.getUniformLocation(program, 'u_image'), 0);

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
        gl.uniform1f(gl.getUniformLocation(program, 'u_exposure'), this.exposure);

        // Advanced Grading
        gl.uniform3f(gl.getUniformLocation(program, 'u_lift'), this.lift[0], this.lift[1], this.lift[2]);
        gl.uniform3f(gl.getUniformLocation(program, 'u_gamma'), this.gradingGamma[0], this.gradingGamma[1], this.gradingGamma[2]);
        gl.uniform3f(gl.getUniformLocation(program, 'u_gain'), this.gain[0], this.gain[1], this.gain[2]);
        gl.uniform3f(gl.getUniformLocation(program, 'u_offset'), this.offset[0], this.offset[1], this.offset[2]);

        gl.uniform1f(gl.getUniformLocation(program, 'u_temperature'), this.temperature);
        gl.uniform1f(gl.getUniformLocation(program, 'u_tint'), this.tint);
        gl.uniform1f(gl.getUniformLocation(program, 'u_contrast'), this.contrast);
        gl.uniform1f(gl.getUniformLocation(program, 'u_pivot'), this.pivot);

        // Curves (Texture unit 3)
        gl.uniform1f(gl.getUniformLocation(program, 'u_curveMix'), this.curveMix);
        gl.activeTexture(gl.TEXTURE3);
        if (this.curveLutTexture) {
            gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
        } else {
            // Create default identity texture if missing
            this.updateCurveLut(this.curveData);
            gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
        }
        gl.uniform1i(gl.getUniformLocation(program, 'u_curveLut'), 3);

        gl.uniform1f(gl.getUniformLocation(program, 'u_saturation'), this.saturation);

        // v3.0: Resolve-style grading uniforms
        gl.uniform1f(gl.getUniformLocation(program, 'u_colorBoost'), this.colorBoost);
        gl.uniform1f(gl.getUniformLocation(program, 'u_shadows'), this.shadows);
        gl.uniform1f(gl.getUniformLocation(program, 'u_highlights'), this.highlights);
        gl.uniform1f(gl.getUniformLocation(program, 'u_midDetail'), this.midDetail);
        gl.uniform1f(gl.getUniformLocation(program, 'u_hueShift'), this.hueShift);
        gl.uniform1f(gl.getUniformLocation(program, 'u_lumaMix'), this.lumaMix);

        // Qualifiers
        gl.uniform1i(gl.getUniformLocation(program, 'u_qualifierEnabled'), this.qualifierEnabled ? 1 : 0);
        gl.uniform1i(gl.getUniformLocation(program, 'u_qualifierShowMask'), this.qualifierShowMask ? 1 : 0);

        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierHue'), this.qualifier.h);
        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierHueWidth'), this.qualifier.hW);
        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierHueSoft'), this.qualifier.hS);

        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierSat'), this.qualifier.s);
        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierSatWidth'), this.qualifier.sW);
        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierSatSoft'), this.qualifier.sS);

        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierLuma'), this.qualifier.l);
        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierLumaWidth'), this.qualifier.lW);
        gl.uniform1f(gl.getUniformLocation(program, 'u_qualifierLumaSoft'), this.qualifier.lS);

        gl.uniform2f(gl.getUniformLocation(program, 'u_texSize'), this.imageWidth, this.imageHeight);

        // Bind Image (Unit 0)
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        gl.uniform1i(gl.getUniformLocation(program, 'u_image'), 0);

        // LUT (Unit 1)
        gl.uniform1i(gl.getUniformLocation(program, 'u_lut'), 1); // Always set unit 1
        if (this.textures.lut) {
            gl.uniform1i(gl.getUniformLocation(program, 'u_lutEnabled'), 1);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_3D, this.textures.lut);
            // u_lut set above
            gl.uniform1f(gl.getUniformLocation(program, 'u_lutSize'), this.lutSize);
            gl.uniform1f(gl.getUniformLocation(program, 'u_lutStrength'), lutStrength);
        } else {
            gl.uniform1i(gl.getUniformLocation(program, 'u_lutEnabled'), 0);
            // Bind dummy or null to unit 1 to prevent warning? 
            // WebGL is okay if we don't sample from it, but some drivers complain if unit is missing.
            // Best practice: Bind null or a dummy texture if logic skips sampling.
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_3D, null);
        }

        // Depth / DoF (Unit 2)
        gl.uniform1i(gl.getUniformLocation(program, 'u_depth'), 2); // Always set unit 2
        if (this.dofEnabled && this.textures.depth) {
            gl.uniform1i(gl.getUniformLocation(program, 'u_dofEnabled'), 1);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, this.textures.depth);
            // u_depth set above

            gl.uniform1f(gl.getUniformLocation(program, 'u_focusDist'), this.focusDistance);
            gl.uniform1f(gl.getUniformLocation(program, 'u_aperture'), this.aperture);
        } else {
            gl.uniform1i(gl.getUniformLocation(program, 'u_dofEnabled'), 0);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, null);
        }

        // Analytics Uniforms
        gl.uniform1i(gl.getUniformLocation(program, 'u_falseColor'), this.falseColor ? 1 : 0);
        gl.uniform1i(gl.getUniformLocation(program, 'u_zebra'), this.zebra ? 1 : 0);
        gl.uniform1f(gl.getUniformLocation(program, 'u_zebraThreshold'), this.zebraThreshold);

        // HDR pipeline: tell shader whether texture is linear float or sRGB PNG
        gl.uniform1i(gl.getUniformLocation(program, 'u_isLinear'), this.isLinearTexture ? 1 : 0);

        // v2.2: Channel isolation, focus peaking, display LUT
        gl.uniform1i(gl.getUniformLocation(program, 'u_channelMode'), this.channelMode);
        gl.uniform1i(gl.getUniformLocation(program, 'u_focusPeaking'), this.focusPeaking ? 1 : 0);
        gl.uniform1f(gl.getUniformLocation(program, 'u_focusPeakThreshold'), this.focusPeakingThreshold);
        gl.uniform1i(gl.getUniformLocation(program, 'u_displayLutMode'), this.displayLutMode);

        // v2.3: Denoise & Depth Eval
        gl.uniform1f(gl.getUniformLocation(program, 'u_denoise'), this.denoise);
        gl.uniform1i(gl.getUniformLocation(program, 'u_showDepth'), this.showDepth ? 1 : 0);
        gl.uniform1f(gl.getUniformLocation(program, 'u_grainAmount'), this.grainAmount || 0.0);

        // v2.6: Lens Effects
        gl.uniform1i(gl.getUniformLocation(program, 'u_apertureBlades'), this.apertureBlades);
        gl.uniform1f(gl.getUniformLocation(program, 'u_apertureRotation'), this.apertureRotation);
        gl.uniform1f(gl.getUniformLocation(program, 'u_apertureAnamorphic'), this.apertureAnamorphic);
        gl.uniform1f(gl.getUniformLocation(program, 'u_lensDistortion'), this.lensDistortion);
        gl.uniform1f(gl.getUniformLocation(program, 'u_lensFringe'), this.lensFringe);
        gl.uniform1f(gl.getUniformLocation(program, 'u_vignetteIntensity'), this.vignetteIntensity);
        gl.uniform1f(gl.getUniformLocation(program, 'u_vignetteFalloff'), this.vignetteFalloff);

        // v3.1: Extended Grain & Lens
        gl.uniform1f(gl.getUniformLocation(program, 'u_grainSize'), this.grainSize || 1.0);
        gl.uniform1f(gl.getUniformLocation(program, 'u_grainColor'), this.grainColor || 0.0);
        gl.uniform1f(gl.getUniformLocation(program, 'u_bloom'), this.bloom || 0.0);
        gl.uniform1f(gl.getUniformLocation(program, 'u_halation'), this.halation || 0.0);
        gl.uniform1f(gl.getUniformLocation(program, 'u_diffusion'), this.diffusion || 0.0);

        // Animated Grain
        gl.uniform1f(gl.getUniformLocation(program, 'u_time'), (performance.now() / 1000.0) % 100.0);

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

        gl.uniform2f(gl.getUniformLocation(program, 'u_resolution'), this.canvas.width, this.canvas.height);
        gl.uniform1f(gl.getUniformLocation(program, 'u_brightness'), brightness);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        gl.uniform1i(gl.getUniformLocation(program, 'u_image'), 0);

        this.drawQuad(program);
    }

    drawQuad(program) {
        const gl = this.gl;

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);

        const posLoc = gl.getAttribLocation(program, 'a_position');
        const texLoc = gl.getAttribLocation(program, 'a_texcoord');

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
//                           EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

// Export for use in radiance_viewer.js
if (typeof window !== 'undefined') {
    window.RadianceWebGLRenderer = RadianceWebGLRenderer;
    window.RadianceSequencePlayer = RadianceSequencePlayer;
    window.RadianceRGBParade = RadianceRGBParade;
}

export { RadianceWebGLRenderer, RadianceSequencePlayer, RadianceRGBParade };
