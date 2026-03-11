/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                       RADIANCE NEURAL MONITOR v3.0 (PRO)
 *                  Predictive Latent Manifold & Process Flow
 * ═══════════════════════════════════════════════════════════════════════════════
 */

export class RadianceNeuralMonitor {
    constructor(container) {
        this.container = container;
        this.canvas = document.createElement('canvas');
        this.overlay = document.createElement('canvas');
        this.gl = null;
        this.ctx2d = null;
        this.program = null;
        this.meshProgram = null;
        
        this.pointCount = 2000;
        this.layers = 5;
        this.progress = 0.0;
        this.rotation = Math.PI / 6;
        this.isActive = false;
        this._raf = null;
        this._startTime = performance.now();

        this.init();
    }

    init() {
        const style = 'position: absolute; top: 0; left: 0; width: 100%; height: 250px; display: block; border-radius: 8px;';
        this.container.style.position = 'relative';
        this.container.style.height = '250px';
        this.container.style.background = '#010409';
        this.container.style.overflow = 'hidden';
        
        this.canvas.style.cssText = style;
        this.overlay.style.cssText = style + ' pointer-events: none;';
        
        this.container.appendChild(this.canvas);
        this.container.appendChild(this.overlay);

        const gl = this.canvas.getContext('webgl2', { alpha: true, antialias: true });
        if (!gl) return;
        this.gl = gl;
        this.ctx2d = this.overlay.getContext('2d');

        // ─── SHADERS ──────────────────────────────────────────────────────────

        const vsPoint = `#version 300 es
            layout(location = 0) in vec3 a_pos;
            layout(location = 1) in float a_layer;
            uniform float u_time, u_progress;
            uniform mat4 u_mat;
            out float v_life;
            out float v_layer;

            void main() {
                // Flow logic: particles move through layers
                float flow = fract(u_time * 0.2 + a_pos.z * 0.5);
                vec3 pos = a_pos;
                
                // Add some neural-like oscillation
                float pulse = sin(u_time * 2.0 + a_layer * 3.14) * 0.02 * u_progress;
                pos.xy += vec2(sin(a_pos.z * 10.0 + u_time), cos(a_pos.z * 8.0 + u_time)) * 0.01;
                
                gl_Position = u_mat * vec4(pos, 1.0);
                gl_PointSize = (2.0 + (1.0 - a_layer) * 3.0) * (0.8 / gl_Position.w);
                v_life = flow;
                v_layer = a_layer;
            }
        `;

        const fsPoint = `#version 300 es
            precision highp float;
            in float v_life;
            in float v_layer;
            out vec4 color;
            void main() {
                float d = distance(gl_PointCoord, vec2(0.5));
                if (d > 0.5) discard;
                
                vec3 c1 = vec3(0.0, 1.0, 0.8); // Cyan
                vec3 c2 = vec3(0.6, 0.2, 1.0); // Purple
                vec3 base = mix(c1, c2, v_layer);
                
                float glow = pow(1.0 - d * 2.0, 2.0);
                color = vec4(base * glow * 1.5, glow * (0.3 + 0.7 * v_life));
            }
        `;

        const vsMesh = `#version 300 es
            layout(location = 0) in vec3 a_pos;
            uniform float u_time, u_progress, u_activity_base;
            uniform mat4 u_mat;
            out float v_activity;
            void main() {
                vec3 pos = a_pos;
                // Periodic wave through layers + activity jitter
                float wave = sin(a_pos.z * 6.0 - u_time * 5.0) * 0.5 + 0.5;
                v_activity = wave * (u_activity_base + 0.1);
                gl_Position = u_mat * vec4(pos, 1.0);
            }
        `;

        const fsMesh = `#version 300 es
            precision highp float;
            uniform float u_time, u_progress;
            in float v_activity;
            out vec4 color;
            void main() {
                vec3 base = vec3(0.0, 0.4, 0.3);
                vec3 cActive = vec3(0.0, 1.0, 0.8);
                vec3 finalC = mix(base, cActive, v_activity);
                float alpha = 0.05 + v_activity * 0.3;
                
                // Add a scanline horizontal glitch if activity is high
                if (v_activity > 0.8) {
                    float glitch = step(0.98, sin(gl_FragCoord.y * 0.5 + u_time * 20.0));
                    finalC += glitch * 0.5;
                    alpha += glitch * 0.2;
                }
                
                color = vec4(finalC, alpha);
            }
        `;

        this.program = this._createProg(gl, vsPoint, fsPoint);
        this.meshProgram = this._createProg(gl, vsMesh, fsMesh);
        
        this._initBuffers(gl);
        this.isActive = true;
        this.render();
    }

    _initBuffers(gl) {
        // --- Layered Synapse Point Cloud ---
        const pts = [];
        const layers = [];
        for(let i=0; i<this.pointCount; i++) {
            const layerIdx = Math.floor(Math.random() * this.layers);
            const z = (layerIdx / (this.layers - 1)) - 0.5; // -0.5 to 0.5
            const angle = Math.random() * Math.PI * 2;
            const radius = 0.2 + Math.random() * 0.4;
            
            pts.push(Math.cos(angle) * radius, Math.sin(angle) * radius, z);
            layers.push(layerIdx / (this.layers - 1));
        }

        this.vboPts = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboPts);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STATIC_DRAW);

        this.vboLayers = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboLayers);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(layers), gl.STATIC_DRAW);

        // --- Layer Discs (Topology) ---
        const meshPts = [];
        const meshIndices = [];
        const segs = 32;
        for(let l=0; l<this.layers; l++) {
            const z = (l / (this.layers - 1)) - 0.5;
            const r = 0.6;
            for(let i=0; i<=segs; i++) {
                const a = (i / segs) * Math.PI * 2;
                meshPts.push(Math.cos(a) * r, Math.sin(a) * r, z);
            }
            // Circle lines
            const baseIdx = l * (segs + 1);
            for(let i=0; i<segs; i++) {
                meshIndices.push(baseIdx + i, baseIdx + i + 1);
            }
            // Vertical interconnects to next layer
            if (l < this.layers - 1) {
                for(let i=0; i<=segs; i += 4) {
                    meshIndices.push(baseIdx + i, baseIdx + i + segs + 1);
                }
            }
        }
        
        this.vboMesh = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboMesh);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meshPts), gl.STATIC_DRAW);
        
        this.iboMesh = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.iboMesh);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(meshIndices), gl.STATIC_DRAW);
        this.meshIndexCount = meshIndices.length;
    }

    setProgress(p) { this.progress = p; }

    setPhase(phase) {
        this.phase = phase; // 'init', 'sampling', 'decoding', 'stable'
        this._phaseStartTime = performance.now();
    }

    render() {
        if (!this.isActive || !this.gl) return;
        const gl = this.gl;
        const rect = this.canvas.getBoundingClientRect();
        if (this.canvas.width !== Math.floor(rect.width * devicePixelRatio)) {
            this.canvas.width = this.overlay.width = rect.width * devicePixelRatio;
            this.canvas.height = this.overlay.height = rect.height * devicePixelRatio;
        }

        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

        this.rotation += 0.002 * (this.phase === 'sampling' ? 2.5 : (this.phase === 'decoding' ? 1.5 : 1.0));
        const aspect = gl.canvas.width / gl.canvas.height;
        const mat = this._getProj(aspect);
        const time = (performance.now() - this._startTime) / 1000;
        const activity = this.phase === 'sampling' ? 1.0 : (this.phase === 'decoding' ? 0.6 : 0.2);

        // 1. Draw Topology
        gl.useProgram(this.meshProgram);
        gl.uniform1f(gl.getUniformLocation(this.meshProgram, 'u_time'), time);
        gl.uniform1f(gl.getUniformLocation(this.meshProgram, 'u_progress'), this.progress);
        gl.uniform1f(gl.getUniformLocation(this.meshProgram, 'u_activity_base'), activity);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.meshProgram, 'u_mat'), false, mat);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboMesh);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.iboMesh);
        gl.drawElements(gl.LINES, this.meshIndexCount, gl.UNSIGNED_SHORT, 0);

        // 2. Draw Synapse Flow
        gl.useProgram(this.program);
        gl.uniform1f(gl.getUniformLocation(this.program, 'u_time'), time);
        gl.uniform1f(gl.getUniformLocation(this.program, 'u_progress'), this.progress);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'u_mat'), false, mat);
        
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboPts);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboLayers);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
        
        gl.drawArrays(gl.POINTS, 0, this.pointCount);

        this._drawOverlay(time, activity);
        this._raf = requestAnimationFrame(() => this.render());
    }

    _drawOverlay(time, activity) {
        const ctx = this.ctx2d;
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        const w = ctx.canvas.width, h = ctx.canvas.height;
        const scale = devicePixelRatio;
        
        ctx.font = `${8 * scale}px 'JetBrains Mono', 'Fira Code', monospace`;
        ctx.fillStyle = "rgba(0, 255, 204, 0.8)";
        
        const load = (0.2 + this.progress * 0.8) * 88.42;
        const dens = (0.4 + Math.sin(performance.now()*0.001)*0.1 + this.progress * 0.5) * 100;
        const prec = 0.99998 - (1.0 - this.progress) * 0.1;

        const phaseDisplay = {
            'init': 'NEURAL_INIT',
            'sampling': 'LATENT_SAMPLING',
            'decoding': 'VAE_DECODE',
            'stable': 'CORE_STABLE'
        }[this.phase] || 'READY';
        
        ctx.fillText(`SYNAPSE_LOAD::${load.toFixed(6)}%`, 18*scale, 25*scale);
        ctx.fillText(`ACT_DENSITY::${dens.toFixed(4)}%`, 18*scale, 38*scale);
        ctx.fillText(`LAT_PRECISN::${prec.toFixed(8)}`, 18*scale, 51*scale);
        
        // --- v3.5: Tensor Dynamics (Animated mock chart) ---
        ctx.fillStyle = "rgba(0, 255, 204, 0.15)";
        ctx.fillText("TENSOR_DYNAMICS", 18*scale, 75*scale);
        const barW = 2*scale;
        for(let i=0; i<20; i++) {
            const hVal = (Math.sin(time*5.0 + i*0.5)*0.5 + 0.5) * 15 * scale * (0.8 + activity*0.4);
            ctx.fillRect(18*scale + i*(barW + 1*scale), 95*scale - hVal, barW, hVal);
        }

        ctx.textAlign = "right";
        ctx.fillText(`STATE::${phaseDisplay}`, w - 18*scale, h - 20*scale);
        ctx.fillText(`KERNEL_FPS::${(60 + Math.random()*0.1).toFixed(2)}`, w - 18*scale, h - 33*scale);
        ctx.fillText(`MEM_BUFFER::[${(8192 * load / 100).toFixed(0)}MB]`, w - 18*scale, h - 46*scale);

        // Sidebar Binary Stream (Visual noise - more complex)
        ctx.fillStyle = "rgba(0, 255, 204, 0.08)";
        for(let i=0; i<15; i++) {
            const b = (Math.random() > 0.5 ? "1" : "0") + (Math.random() > 0.8 ? "x" : "");
            ctx.fillText(b, w - 8*scale, 20*scale + i*12*scale);
        }

        // Crosshair center
        ctx.strokeStyle = "rgba(0, 255, 204, 0.1)";
        ctx.beginPath();
        ctx.moveTo(w/2 - 5*scale, h/2); ctx.lineTo(w/2 + 5*scale, h/2);
        ctx.moveTo(w/2, h/2 - 5*scale); ctx.lineTo(w/2, h/2 + 5*scale);
        ctx.stroke();

        // 3D Axis Labels (Mock positions based on rotation)
        const rot = this.rotation;
        const axisColor = "rgba(0, 255, 204, 0.5)";
        ctx.fillStyle = axisColor;
        ctx.font = `${7 * scale}px 'JetBrains Mono'`;
        
        const centerX = w / 2, centerY = h / 2;
        const axisLen = 40 * scale;
        
        // Z Axis (Depth)
        const zx = centerX + Math.sin(rot) * axisLen;
        const zy = centerY + Math.cos(rot) * axisLen * 0.2;
        ctx.fillText("Z_AXIS", zx, zy);
        
        // X Axis (Lateral)
        const xx = centerX + Math.cos(rot) * axisLen;
        const xy = centerY - Math.sin(rot) * axisLen * 0.2;
        ctx.fillText("X_AXIS", xx, xy);
        
        // Y Axis (Vertical)
        ctx.fillText("Y_AXIS", centerX, centerY - axisLen * 0.5);
    }

    _getProj(a) {
        const f=2.2, n=0.1, r=this.rotation;
        return new Float32Array([
            f/a*Math.cos(r), 0, Math.sin(r), 0,
            0.2*Math.sin(r), f, 0, 0,
            -Math.sin(r), 0, f/a*Math.cos(r), -1.8,
            0, 0, 0, 1
        ]);
    }

    _createProg(gl, vs, fs) {
        const s = (t, src) => {
            const sh = gl.createShader(t);
            gl.shaderSource(sh, src); gl.compileShader(sh);
            if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(sh));
            return sh;
        };
        const p = gl.createProgram();
        gl.attachShader(p, s(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, s(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(p); return p;
    }

    dispose() {
        this.isActive = false;
        if (this._raf) cancelAnimationFrame(this._raf);
        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        this.container.innerHTML = '';
    }
}
