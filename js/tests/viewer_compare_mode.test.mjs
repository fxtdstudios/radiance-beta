/**
 * Viewer compare and the Simple / Advanced switch (3.5.0).
 *
 * Compare was four handlers that disagreed: A/B grabbed a still of A over a
 * connected compare_image (A compared with itself), Wipe with no B showed the
 * ungraded source, and Difference and Blink drew nothing on the default WebGL
 * path (Blink started playback). One controller, setCompareMode(), now drives
 * every control, in both modes. The Lite Viewer is retired into Simple mode.
 *
 * The methods are lifted from the source and run on a stand-in viewer with a
 * recording renderer (radiance_viewer.js cannot be imported in Node). The
 * browser run (both modes, every compare mode with a known B, pin, save and
 * reload, an old Lite Viewer graph) is in docs/DEVELOPMENT.md.
 *
 * Run: node --test js/tests/viewer_compare_mode.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');
const gl = readFileSync(join(JS, 'radiance_webgl.js'), 'utf8');

function methodSource(name) {
    const start = src.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    const end = rest.search(/\n    \}\n/);
    return rest.slice(0, end) + '\n    }';
}

const METHODS = ['setCompareMode', '_applyCompareToRenderer', '_hasCompareB', 'pinReference',
    'releaseReference', '_updateCompareForFrame', 'setCompareImage', 'cycleCompareMode'];
const Proto = new Function(`return class { ${METHODS.map(methodSource).join('\n')} }`)();

const timers = [];
globalThis.setInterval = (fn, ms) => { timers.push(fn); return timers.length; };
globalThis.clearInterval = () => {};

function renderer() {
    const r = {
        textures: { image: {}, reference: null }, calls: [], wipe: null, show: 0, stills: 0,
        setWipe(pos, on) { this.wipe = on ? pos : null; },
        setCompareShow(show) { this.show = show; },
        render() {},
        grabReferenceStill() { this.stills++; return 0; },
        swapReferenceShelf() { this.textures.reference = 'still'; },
        clearReferenceShelf() { this.textures.reference = null; },
        loadCompareTexture(img) { this.textures.reference = img; },
    };
    return r;
}

function viewer({ compareFrames = null } = {}) {
    const v = new Proto();
    Object.assign(v, {
        renderer: renderer(), compareMode: 'none', compareSource: null, currentFrame: 3,
        wipePosition: 0.5, diffGain: 4, blinkMs: 500, container: { isConnected: true },
        frameCompareImages: compareFrames, rendered: 0,
        render() { this.rendered++; }, _updateCompareBtn() {}, _syncCompareUI() {}, _termLog() {},
    });
    return v;
}

test('each mode sets what the renderer draws', () => {
    const v = viewer({ compareFrames: ['b0', 'b1', 'b2', 'b3'] });
    v.setCompareMode('b');
    assert.equal(v.renderer.show, 1); assert.equal(v.renderer.wipe, null);
    v.setCompareMode('difference');
    assert.equal(v.renderer.show, 2);
    v.setCompareMode('wipe');
    assert.equal(v.renderer.show, 0); assert.equal(v.renderer.wipe, 0.5);
    v.setCompareMode('none');
    assert.equal(v.renderer.show, 0); assert.equal(v.renderer.wipe, null);
});

test('a connected compare_image is B, never a still of A', () => {
    const v = viewer({ compareFrames: ['b0', 'b1', 'b2', 'b3'] });
    v.cycleCompareMode();                       // the A|B button
    assert.equal(v.compareMode, 'wipe');
    assert.equal(v.renderer.stills, 0, 'A/B grabbed a still of A over the compare input');
    assert.equal(v.renderer.textures.reference, 'b3', 'B must be the input frame under the playhead');
    assert.equal(v.compareSource, 'input');
});

test('with no B, a compare mode pins the current frame first', () => {
    const v = viewer();
    v.setCompareMode('difference');
    assert.equal(v.renderer.stills, 1);
    assert.equal(v.compareSource, 'pinned');
    assert.equal(v.renderer.show, 2);
});

test('blink flips A and B on a timer and does not play', () => {
    timers.length = 0;
    const v = viewer({ compareFrames: ['b0'] });
    v.togglePlayback = () => assert.fail('Blink must not start playback');
    v.setCompareMode('blink');
    assert.equal(timers.length, 1);
    assert.equal(v.renderer.show, 0);
    timers[0](); v._applyCompareToRenderer();
    assert.equal(v.renderer.show, 1);
    timers[0](); v._applyCompareToRenderer();
    assert.equal(v.renderer.show, 0);
});

test('a pinned B stays through frame changes and new runs; release goes back to the input', () => {
    const v = viewer({ compareFrames: ['b0', 'b1', 'b2', 'b3'] });
    v.setCompareMode('wipe');
    v.pinReference();
    assert.equal(v.renderer.textures.reference, 'still');
    v._updateCompareForFrame(1);
    assert.equal(v.renderer.textures.reference, 'still', 'the playhead replaced the pinned B');
    v.setCompareImage('new-run-b0');
    assert.equal(v.renderer.textures.reference, 'still', 'a new run replaced the pinned B');
    v.currentFrame = 2;
    v.releaseReference();
    assert.equal(v.compareSource, 'input');
    assert.equal(v.renderer.textures.reference, 'b2');
    assert.equal(v.compareMode, 'wipe');
});

test('the shader draws B and difference from the reference texture', () => {
    assert.match(gl, /uniform int u_compareShow;/);
    assert.match(gl, /u_compareShow == 1[\s\S]{0,120}texture\(u_referenceImage/);
    assert.match(gl, /u_compareShow == 2[\s\S]{0,160}abs\(color - texture\(u_referenceImage/);
    assert.match(gl, /_ui1\(program, 'u_compareShow'/, 'the uniform is declared but never set');
});

test('every compare control goes through setCompareMode', () => {
    const bar = methodSource('createViewerBar');
    assert.doesNotMatch(bar, /Blink'[^\n]*togglePlayback/, 'Blink starts playback again');
    assert.doesNotMatch(bar, /this\.compareMode = '(wipe|difference)'/, 'a viewer-bar button sets compareMode directly');
    const simple = methodSource('_createSimpleBar');
    for (const m of ['none', 'b', 'wipe', 'difference', 'blink']) assert.match(simple, new RegExp(`\\['${m}',`));
    assert.match(simple, /this\.setCompareMode\(m\)/);
});

test('Simple / Advanced: saved on the node, old graphs open as they looked', () => {
    assert.match(methodSource('setUIMode'), /properties\.radiance_viewer_mode = mode/);
    const reg = src.slice(src.indexOf('async beforeRegisterNodeDef'));
    assert.match(reg, /"RadianceLiteViewer"\]\.includes\(nodeData\.name\)/, 'the retired Lite Viewer must open in this viewer');
    assert.match(reg, /this\.type === 'RadianceLiteViewer' \? 'simple' : 'advanced'/);
    const install = methodSource('_installUIMode');
    for (const el of ['proToolbar', 'proSidebar', 'rightControlPanel', 'viewerBar', 'sequenceDock', 'statusBar']) {
        assert.match(install, new RegExp(`this\\.${el}`), `${el} is not hidden in Simple`);
    }
});

test('the Lite Viewer frontend is gone', () => {
    let gone = false;
    try { readFileSync(join(JS, 'radiance_lite_viewer.js')); } catch { gone = true; }
    assert.ok(gone, 'a second frontend would register over RadianceLiteViewer');
});

test('releasing the context on teardown is not reported as a failure', () => {
    assert.match(gl, /if \(this\._destroyed\) return;[\s\S]{0,80}e\.preventDefault\(\)/);
    assert.match(gl, /this\._destroyed = true;\s*\n\s*try \{ this\.gl\.getExtension\('WEBGL_lose_context'\)/);
});
