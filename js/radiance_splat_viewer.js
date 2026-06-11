// Radiance — Splat Viewer 3D
// Interactive WebGL2 viewer for 3D Gaussian Splats inside the
// RadianceSplatViewer3D node. Reads the .splat file the node writes to the
// ComfyUI temp dir and renders it with EWA splatting (sorted, alpha-blended
// instanced quads). Orbit: drag. Zoom: wheel. Pan: right-drag / shift-drag.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;     // quad corner in [-2,2]
layout(location=1) in vec3 aCenter;
layout(location=2) in vec3 aCovA;    // Sxx Sxy Sxz
layout(location=3) in vec3 aCovB;    // Syy Syz Szz
layout(location=4) in vec4 aColor;   // normalized u8 rgba
uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uViewport;
uniform vec2 uFocal;
out vec4 vColor;
out vec2 vPos;
void main() {
  vec4 cam = uView * vec4(aCenter, 1.0);
  float depth = -cam.z;                       // camera looks down -Z
  vec4 clip = uProj * cam;
  if (depth <= 0.05 || clip.w <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  mat3 Vrk = mat3(aCovA.x, aCovA.y, aCovA.z,
                  aCovA.y, aCovB.x, aCovB.y,
                  aCovA.z, aCovB.y, aCovB.z);
  mat3 J = mat3(uFocal.x / depth, 0.0, 0.0,
                0.0, uFocal.y / depth, 0.0,
                -(uFocal.x * cam.x) / (depth * depth),
                -(uFocal.y * cam.y) / (depth * depth), 0.0);
  mat3 W = transpose(mat3(uView));
  mat3 T = W * J;
  mat3 cov2d = transpose(T) * Vrk * T;
  float a = cov2d[0][0] + 0.3;
  float b = cov2d[0][1];
  float d = cov2d[1][1] + 0.3;
  float mid = 0.5 * (a + d);
  float rad = length(vec2((a - d) * 0.5, b));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.1);
  vec2 dir = normalize(vec2(b, l1 - a));
  vec2 v1 = min(sqrt(2.0 * l1), 1024.0) * dir;
  vec2 v2 = min(sqrt(2.0 * l2), 1024.0) * vec2(dir.y, -dir.x);
  vColor = aColor;
  vPos = aPos;
  vec2 ndc = clip.xy / clip.w;
  gl_Position = vec4(ndc + aPos.x * v1 * 2.0 / uViewport
                         + aPos.y * v2 * 2.0 / uViewport, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vPos;
out vec4 frag;
void main() {
  float A = -dot(vPos, vPos);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  frag = vec4(B * vColor.rgb, B);
}`;

// ── tiny mat4 helpers (column-major) ─────────────────────────────────────────
function perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f;
  out[10] = (far + near) / (near - far); out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}
function lookAt(eye, target, up) {
  const z = norm3(sub3(eye, target));          // camera looks down -z
  const x = norm3(cross3(up, z));
  const y = cross3(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
  ]);
}
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function norm3(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

// ── the viewer ───────────────────────────────────────────────────────────────
class SplatViewer3D {
  constructor(container) {
    this.container = container;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText = "width:100%;height:100%;display:block;border-radius:6px;background:#101014;cursor:grab;";
    container.appendChild(this.canvas);

    this.hud = document.createElement("div");
    this.hud.style.cssText = "position:absolute;top:6px;left:8px;font:11px monospace;color:#9a9aa5;" +
      "background:rgba(16,16,20,.65);padding:2px 8px;border-radius:4px;pointer-events:none;";
    this.hud.textContent = "Splat Viewer — run the graph to load a splat";
    container.appendChild(this.hud);

    this.resetBtn = document.createElement("button");
    this.resetBtn.textContent = "reset view";
    this.resetBtn.style.cssText = "position:absolute;top:6px;right:8px;font:11px monospace;color:#ccc;" +
      "background:#26262e;border:1px solid #3a3a44;border-radius:4px;padding:2px 8px;cursor:pointer;";
    this.resetBtn.onclick = () => this.resetView();
    container.appendChild(this.resetBtn);

    this.gl = this.canvas.getContext("webgl2", { antialias: false, alpha: false });
    this.count = 0;
    this.yaw = 0.6; this.pitch = -0.3; this.dist = 5; this.target = [0, 0, 0];
    this.home = { yaw: 0.6, pitch: -0.3, dist: 5, target: [0, 0, 0] };
    this.needSort = true;
    this.lastSort = 0;
    this._initGL();
    this._bindControls();
    const loop = () => { this._draw(); this._raf = requestAnimationFrame(loop); };
    loop();
    new ResizeObserver(() => this._resize()).observe(container);
  }

  _initGL() {
    const gl = this.gl;
    if (!gl) { this.hud.textContent = "WebGL2 not available"; return; }
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error("[Radiance Splat Viewer]", gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) console.error("[Radiance Splat Viewer]", gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.uni = {
      view: gl.getUniformLocation(prog, "uView"),
      proj: gl.getUniformLocation(prog, "uProj"),
      viewport: gl.getUniformLocation(prog, "uViewport"),
      focal: gl.getUniformLocation(prog, "uFocal"),
    };
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    // quad
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    // instance buffers (filled on load)
    this.geoBuf = gl.createBuffer();   // center+covA+covB, 9 floats
    this.colBuf = gl.createBuffer();   // rgba u8
    gl.bindBuffer(gl.ARRAY_BUFFER, this.geoBuf);
    const stride = 9 * 4;
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(1 + i);
      gl.vertexAttribPointer(1 + i, 3, gl.FLOAT, false, stride, i * 12);
      gl.vertexAttribDivisor(1 + i, 1);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 4, gl.UNSIGNED_BYTE, true, 4, 0);
    gl.vertexAttribDivisor(4, 1);
    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // premultiplied, back-to-front
  }

  async load(url, msg) {
    this.hud.textContent = "loading splat…";
    try {
      const buf = await (await fetch(url)).arrayBuffer();
      this._parse(buf);
      const c = msg?.center?.[0], r = msg?.radius?.[0];
      if (c) this.home.target = [c[0], c[1], c[2]];
      if (r) this.home.dist = r * 2.5;
      this.resetView();
      const total = msg?.splat_total?.[0] ?? this.count;
      this.hud.textContent = `${this.count.toLocaleString()} splats` +
        (total > this.count ? ` (of ${total.toLocaleString()})` : "") +
        " — drag orbit · wheel zoom · right-drag pan";
    } catch (e) {
      console.error("[Radiance Splat Viewer]", e);
      this.hud.textContent = "failed to load splat (see console)";
    }
  }

  _parse(buf) {
    const n = Math.floor(buf.byteLength / 32);
    const f32 = new Float32Array(buf);
    const u8 = new Uint8Array(buf);
    const geo = new Float32Array(n * 9);
    const col = new Uint8Array(n * 4);
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const fo = i * 8, bo = i * 32;
      const x = f32[fo], y = f32[fo + 1], z = f32[fo + 2];
      const sx = f32[fo + 3], sy = f32[fo + 4], sz = f32[fo + 5];
      col[i * 4] = u8[bo + 24]; col[i * 4 + 1] = u8[bo + 25];
      col[i * 4 + 2] = u8[bo + 26]; col[i * 4 + 3] = u8[bo + 27];
      let qw = (u8[bo + 28] - 128) / 128, qx = (u8[bo + 29] - 128) / 128,
          qy = (u8[bo + 30] - 128) / 128, qz = (u8[bo + 31] - 128) / 128;
      const ql = Math.hypot(qw, qx, qy, qz) || 1;
      qw /= ql; qx /= ql; qy /= ql; qz /= ql;
      // R(q) * diag(s) -> M; Sigma = M M^T
      const R = [
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
        2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
        2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy),
      ];
      const M = [R[0] * sx, R[1] * sy, R[2] * sz,
                 R[3] * sx, R[4] * sy, R[5] * sz,
                 R[6] * sx, R[7] * sy, R[8] * sz];
      const g = i * 9;
      geo[g] = x; geo[g + 1] = y; geo[g + 2] = z;
      geo[g + 3] = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];           // Sxx
      geo[g + 4] = M[0] * M[3] + M[1] * M[4] + M[2] * M[5];           // Sxy
      geo[g + 5] = M[0] * M[6] + M[1] * M[7] + M[2] * M[8];           // Sxz
      geo[g + 6] = M[3] * M[3] + M[4] * M[4] + M[5] * M[5];           // Syy
      geo[g + 7] = M[3] * M[6] + M[4] * M[7] + M[5] * M[8];           // Syz
      geo[g + 8] = M[6] * M[6] + M[7] * M[7] + M[8] * M[8];           // Szz
      pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
    }
    this.count = n;
    this.baseGeo = geo; this.baseCol = col; this.positions = pos;
    this.sortedGeo = new Float32Array(n * 9);
    this.sortedCol = new Uint8Array(n * 4);
    this.depths = new Float32Array(n);
    this.order = new Uint32Array(n);
    this.needSort = true;
  }

  _sort(view) {
    const n = this.count;
    if (!n) return;
    const p = this.positions, d = this.depths;
    const vz0 = view[2], vz1 = view[6], vz2 = view[10], vz3 = view[14];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const z = -(vz0 * p[i * 3] + vz1 * p[i * 3 + 1] + vz2 * p[i * 3 + 2] + vz3);
      d[i] = z;
      if (z < mn) mn = z;
      if (z > mx) mx = z;
    }
    const B = 65536, counts = new Uint32Array(B);
    const scale = (B - 1) / Math.max(mx - mn, 1e-9);
    const keys = new Uint16Array(n);
    for (let i = 0; i < n; i++) {
      // invert so far splats get the small keys -> drawn first (back-to-front)
      keys[i] = (B - 1 - ((d[i] - mn) * scale)) | 0;
      counts[keys[i]]++;
    }
    let acc = 0;
    for (let k = 0; k < B; k++) { const c = counts[k]; counts[k] = acc; acc += c; }
    for (let i = 0; i < n; i++) this.order[counts[keys[i]]++] = i;
    const sg = this.sortedGeo, sc = this.sortedCol, bg = this.baseGeo, bc = this.baseCol;
    for (let i = 0; i < n; i++) {
      const s = this.order[i], g9 = i * 9, b9 = s * 9;
      for (let k = 0; k < 9; k++) sg[g9 + k] = bg[b9 + k];
      const c4 = i * 4, d4 = s * 4;
      sc[c4] = bc[d4]; sc[c4 + 1] = bc[d4 + 1]; sc[c4 + 2] = bc[d4 + 2]; sc[c4 + 3] = bc[d4 + 3];
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.geoBuf);
    gl.bufferData(gl.ARRAY_BUFFER, sg, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, sc, gl.DYNAMIC_DRAW);
  }

  _eye() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    return [
      this.target[0] + this.dist * cp * cy,
      this.target[1] + this.dist * sp,
      this.target[2] + this.dist * cp * sy,
    ];
  }

  _draw() {
    const gl = this.gl;
    if (!gl || !this.canvas.width) return;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.055, 0.055, 0.07, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.count) return;
    const w = this.canvas.width, h = this.canvas.height;
    const fov = 50 * Math.PI / 180;
    const proj = perspective(fov, w / h, 0.05, 1e4);
    const view = lookAt(this._eye(), this.target, [0, 1, 0]);
    const now = performance.now();
    if (this.needSort && now - this.lastSort > 150) {
      this._sort(view);
      this.lastSort = now;
      this.needSort = false;
    }
    const focal = 0.5 * h / Math.tan(fov / 2);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uni.view, false, view);
    gl.uniformMatrix4fv(this.uni.proj, false, proj);
    gl.uniform2f(this.uni.viewport, w, h);
    gl.uniform2f(this.uni.focal, focal, focal);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    gl.bindVertexArray(null);
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.container.getBoundingClientRect();
    this.canvas.width = Math.max(1, r.width * dpr) | 0;
    this.canvas.height = Math.max(1, r.height * dpr) | 0;
  }

  resetView() {
    this.yaw = this.home.yaw; this.pitch = this.home.pitch;
    this.dist = this.home.dist; this.target = [...this.home.target];
    this.needSort = true;
  }

  _bindControls() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey };
      c.setPointerCapture(e.pointerId);
      c.style.cursor = "grabbing";
      e.preventDefault(); e.stopPropagation();
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.btn === 2 || drag.shift) {
        // pan in view plane
        const s = this.dist * 0.0016;
        const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
        const right = [-sy, 0, cy];
        const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
        const up = [-sp * cy, cp, -sp * sy];
        this.target[0] -= (right[0] * dx - up[0] * dy) * s;
        this.target[1] -= (right[1] * dx - up[1] * dy) * s;
        this.target[2] -= (right[2] * dx - up[2] * dy) * s;
      } else {
        this.yaw += dx * 0.008;
        this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.008));
      }
      this.needSort = true;
      e.preventDefault(); e.stopPropagation();
    });
    const end = (e) => { drag = null; c.style.cursor = "grab"; };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("wheel", (e) => {
      this.dist *= Math.exp(e.deltaY * 0.0012);
      this.dist = Math.max(0.05, Math.min(this.dist, 1e4));
      this.needSort = true;
      e.preventDefault(); e.stopPropagation();
    }, { passive: false });
    c.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });
  }

  destroy() {
    cancelAnimationFrame(this._raf);
  }
}

app.registerExtension({
  name: "radiance.splat_viewer_3d",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "RadianceSplatViewer3D") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const container = document.createElement("div");
      container.style.cssText = "position:relative;width:100%;height:100%;min-height:300px;";
      this._splat3d = new SplatViewer3D(container);
      this.addDOMWidget("splat3d_view", "SPLAT3D", container, { serialize: false });
      this.size = [Math.max(this.size[0], 520), Math.max(this.size[1], 560)];
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const file = message?.splat_file?.[0];
      if (file && this._splat3d) {
        const url = api.apiURL(
          `/view?filename=${encodeURIComponent(file)}&type=temp&subfolder=&rand=${Math.random()}`);
        this._splat3d.load(url, message);
      }
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this._splat3d?.destroy();
      onRemoved?.apply(this, arguments);
    };
  },
});
