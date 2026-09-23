/**
 * Lite Viewer, 3.5.0: measured in Chromium at devicePixelRatio 2.
 *   - the canvas is in device pixels, and 1:1 is one SOURCE pixel per device pixel
 *   - the readout reports float source values at source coordinates
 *   - diff and clip are computed from the float proxies
 *   - the payload fps drives playback
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, access } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { deflateSync } from 'node:zlib';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const W = 40, H = 30;           // preview size; the "source" is reported as 2x
const STUBS = {
    '/scripts/app.js': `export const app = { registerExtension(e) { (globalThis.__exts ||= []).push(e); } }; globalThis.app = app;`,
    '/scripts/api.js': `export const api = { apiURL(p) { return p; } };`,
};
const MIME = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.png': 'image/png' };

function png(v8) {
    const raw = Buffer.alloc((W * 4 + 1) * H);
    for (let y = 0; y < H; y++) {
        raw[y * (W * 4 + 1)] = 0;
        for (let x = 0; x < W; x++) { const o = y * (W * 4 + 1) + 1 + x * 4; raw[o] = raw[o + 1] = raw[o + 2] = v8; raw[o + 3] = 255; }
    }
    const crc32 = (buf) => { let c, crc = 0xffffffff; for (let n = 0; n < buf.length; n++) { c = (crc ^ buf[n]) & 0xff; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crc = (crc >>> 8) ^ c; } return (crc ^ 0xffffffff) >>> 0; };
    const chunk = (t, d) => { const l = Buffer.alloc(4); l.writeUInt32BE(d.length); const td = Buffer.concat([Buffer.from(t), d]); const c = Buffer.alloc(4); c.writeUInt32BE(crc32(td)); return Buffer.concat([l, td, c]); };
    const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 6;
    return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}
function half(v) {   // float -> fp16 bits (normal range only; enough here)
    const f = new Float32Array([v]); const u = new Uint32Array(f.buffer)[0];
    const s = (u >>> 16) & 0x8000; const e = ((u >>> 23) & 0xff) - 127 + 15; const m = (u >>> 13) & 0x3ff;
    return v === 0 ? 0 : (s | (e << 10) | m);
}
function proxy(v) {
    const u16 = new Uint16Array(W * H * 4);
    for (let i = 0; i < W * H; i++) { u16[i * 4] = u16[i * 4 + 1] = u16[i * 4 + 2] = half(v); u16[i * 4 + 3] = half(1); }
    const hdr = Buffer.alloc(12); hdr.write('RHDR', 0); hdr.writeUInt16LE(W, 4); hdr.writeUInt16LE(H, 6); hdr.writeUInt16LE(4, 8); hdr.writeUInt16LE(0, 10);
    return Buffer.concat([hdr, deflateSync(Buffer.from(u16.buffer))]);
}
function file(name) {
    let m = /^p_(\d+)\.png$/.exec(name); if (m) return png(Number(m[1]));
    m = /^f_([\d.]+)\.rhdr$/.exec(name); if (m) return proxy(Number(m[1]));
    return null;
}
const item = (p8, fv, extra = {}) => ({ filename: `p_${p8}.png`, float_filename: `f_${fv}.rhdr`, subfolder: '', type: 'temp',
    width: W * 2, height: H * 2, preview_width: W, preview_height: H, data_range: [fv, fv], ...extra });

async function findChromium() {
    for (const p of [process.env.RADIANCE_TEST_CHROMIUM, '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].filter(Boolean)) { try { await access(p); return p; } catch { /* next */ } }
    return null;
}
async function loadPlaywright() {
    const require = createRequire(import.meta.url);
    for (const spec of ['playwright', '/home/claude/.npm-global/lib/node_modules/playwright/index.js']) { try { return require(spec); } catch { /* next */ } }
    return null;
}
async function serve() {
    const server = createServer(async (req, res) => {
        const u = new URL(req.url, 'http://x');
        if (STUBS[u.pathname]) { res.writeHead(200, { 'Content-Type': 'text/javascript' }); res.end(STUBS[u.pathname]); return; }
        if (u.pathname === '/view') { const f = file(u.searchParams.get('filename')); if (!f) { res.writeHead(404); res.end(); return; } res.writeHead(200); res.end(f); return; }
        try {
            const p = join(ROOT, normalize(decodeURIComponent(u.pathname)));
            if (!p.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
            const data = await readFile(p); res.writeHead(200, { 'Content-Type': MIME[extname(p)] || 'application/octet-stream' }); res.end(data);
        } catch { res.writeHead(404); res.end(); }
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    return server;
}

const playwright = await loadPlaywright();
const skip = playwright ? false : 'Playwright is not installed';
const R = {}; const errs = [];
if (!skip) {
    const server = await serve();
    const exe = await findChromium();
    const browser = await playwright.chromium.launch({ ...(exe ? { executablePath: exe } : {}), args: ['--no-sandbox'] });
    try {
        const page = await browser.newPage({ viewport: { width: 900, height: 700 }, deviceScaleFactor: 2 });
        page.on('pageerror', (e) => errs.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/liteharness.html`);
        await page.waitForFunction(() => window.__ready === true, null, { timeout: 20000 });
        const run = (m, o) => page.evaluate(([mm, oo]) => window.__lite(mm, oo), [m, o || {}]);
        R.single = await run({ radiance_lite_images: [item(128, 0.25, { source_encoding: 'srgb' })], fps: [25] });
        R.diff = await run({ radiance_lite_images: [item(128, 0.5, { source_encoding: 'srgb' }),
            { ...item(64, 0.25, { source_encoding: 'srgb' }), is_compare: true }] }, { mode: 'diff' });
        R.clip = await run({ radiance_lite_images: [item(200, 1.2, { source_encoding: 'linear' })] }, { clip: true });
    } finally { await browser.close(); server.close(); }
}
const bad = (r) => (r && r.ok ? false : `harness: ${r?.error}`);

test('no page errors', { skip }, () => assert.deepEqual(errs, [], errs.join('\n')));

test('the canvas is in device pixels and 1:1 is one source pixel per device pixel', { skip: skip || bad(R.single) }, () => {
    assert.equal(R.single.dpr, 2);
    assert.equal(R.single.canvasW, Math.round(R.single.cssW * 2));
    assert.equal(R.single.zoom11, 2, 'a 2x proxy must be magnified 2x at 1:1');
});

test('the readout is the float source value at source coordinates', { skip: skip || bad(R.single) }, () => {
    assert.match(R.single.readout, /R 0\.2500 G 0\.2500 B 0\.2500/, R.single.readout);
    const [, x, y] = /XY (\d+), (\d+)/.exec(R.single.readout).map(Number);
    assert.ok(Math.abs(x - W) <= 2 && Math.abs(y - H) <= 2, `centre of a ${W * 2}x${H * 2} source reads ${x},${y}`);
});

test('the source fps drives playback', { skip: skip || bad(R.single) }, () => {
    assert.equal(R.single.fps, 25);
    assert.match(R.single.status, /25 fps/);
});

test('diff is computed from float values', { skip: skip || bad(R.diff) }, () => {
    // |0.5 - 0.25| = 0.25 in sRGB code values -> 64
    assert.ok(Math.abs(R.diff.centre[0] - 64) <= 2, `diff shows ${R.diff.centre}`);
});

test('clip flags a linear value above 1.0 from the float source', { skip: skip || bad(R.clip) }, () => {
    const [r, g] = R.clip.centre;
    assert.ok(r > 200 && g < 100, `clip overlay not red: ${R.clip.centre}`);
});
