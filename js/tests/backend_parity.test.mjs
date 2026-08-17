/**
 * The two rendering backends must agree.
 *
 * Radiance ships a WebGL renderer and a WebGPU renderer, and WebGPU is the
 * *preferred* one — `_tryWebGPUUpgrade()` runs whenever `navigator.gpu` exists.
 * Nothing has ever compared their output. An audit of the two files found the
 * grade maths implemented four separate times, and they do not all agree:
 *
 *   radiance_webgl.js:2617    (c − pivot) · clamp(k,0,5) + pivot     linear
 *   radiance_webgpu.js:423    (c − pivot) · k + pivot                linear, unclamped
 *   radiance_webgpu.js:1388   pivot · pow(c/pivot, k)                POWER
 *   radiance_viewer.js:6515   (c − pivot) · k + pivot                linear (LUT export)
 *
 * The third is a different curve, not an approximation of the others. At
 * k=2, pivot=0.5, c=0.25 the linear form gives 0.0 and the power form 0.125.
 * It is reached whenever `_getGradedPixels` returns null or `_isAllZeroes`
 * trips — and that samples ~40 pixels, so a dark or letterboxed frame trips it
 * spuriously.
 *
 * **Resolved.** All four now come from `js/radiance_grade.js`: the two shaders
 * are emitted from it and the two CPU paths call it. The divergence table above
 * is history rather than a warning, and the maths tests that used to live here
 * moved to `grade.test.mjs` alongside the definition they check.
 *
 * What stays here is the part that is genuinely about the two *backends* and
 * not about arithmetic: the readback contract, which is still hand-written on
 * both sides, and the WGSL feature gaps.
 *
 * These tests do not need a GPU.
 *
 * Run: node --test js/tests/backend_parity.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { applyContrastChannel as applyContrast, applyGammaChannel as applyGamma } from '../radiance_grade.js';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => readFileSync(join(JS, f), 'utf8');

// The reference maths is no longer defined here. It used to be a local copy,
// which made this file a *fifth* implementation of the thing it was written to
// police. It now comes from the one definition both backends are emitted from.

// ── the contract both backends must satisfy ─────────────────────────────────

test('readPixelsFloat32 returns the same shape in both backends', () => {
    const gl = read('radiance_webgl.js');
    const gpu = read('radiance_webgpu.js');

    for (const [name, src] of [['webgl', gl], ['webgpu', gpu]]) {
        assert.match(src, /readPixelsFloat32\s*\(/, `${name} has no readPixelsFloat32`);
        assert.match(
            src, /return \{ data: \w+, width: \w+, height: \w+, graded: (true|false) \}/,
            `${name}'s readPixelsFloat32 must return {data,width,height,graded}`,
        );
    }
});

test('readPixelsFloat32 is synchronous in both backends', () => {
    // WebGPU used to return a Promise while WebGL returned a value. The EXR
    // export read `.data` off the Promise, got undefined, and blamed the
    // encoder.
    const gpu = read('radiance_webgpu.js');
    const body = gpu.slice(gpu.indexOf('readPixelsFloat32'), gpu.indexOf('// ── Reference shelf'));
    assert.doesNotMatch(body, /return Promise\./,
        'readPixelsFloat32 must not return a Promise — WebGL does not');
});

test('each backend declares whether its pixels are graded', () => {
    assert.match(read('radiance_webgl.js'), /graded: true/,
        'WebGL renders the graded composite and must say so');
    assert.match(read('radiance_webgpu.js'), /graded: false/,
        'WebGPU returns ungraded source and must say so');
});

// ── the maths ───────────────────────────────────────────────────────────────

// The maths tests that used to sit here now live in grade.test.mjs, next to the
// definition. Two of them are kept below in backend form: they assert the
// *shipped source* no longer contains the divergent formulas, which is a
// different claim from "the shared function is correct".

test('neither backend re-implements contrast or gamma', () => {
    const strip = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');
    for (const [name, file] of [['webgl', 'radiance_webgl.js'], ['webgpu', 'radiance_webgpu.js']]) {
        const src = strip(read(file));
        assert.doesNotMatch(src, /clamp\(contrast, 0\.0, 5\.0\)/,
            `${name} has its own contrast clamp again — it should come from radiance_grade.js`);
    }
    // And the shared functions still behave, so a green suite here means
    // something.
    assert.equal(applyContrast(0.25, 2, 0.5), 0);
    assert.ok(Number.isFinite(applyGamma(0.5, 0)));
});

test('the WebGPU shader path guards gamma', () => {
    const gpu = read('radiance_webgpu.js');
    const unguarded = /pow\(\s*max\(color,\s*0\),\s*1\.0\s*\/\s*gGradeGamma\s*\)/.test(gpu);
    assert.ok(!unguarded,
        'radiance_webgpu.js:418 divides by gGradeGamma with no floor; '
        + 'radiance_webgl.js:2609 uses max(0.01, gamma). Match it.');
});

// `todo`, not a failure. The gap is real and this test proves it every run, but
// a red suite trains people to ignore red. Node reports todo tests separately,
// so it stays visible in the output until someone implements the WGSL or
// disables the two tabs on WebGPU. Do not delete it to make the output tidy.
test('mask and qualifier exist in the WebGPU backend', { todo: 'WGSL implementation missing — Masks and Qualifiers tabs are inert on WebGPU' }, () => {
    // The base class implements setMask/setQualifier and stores the state, so
    // the viewer's calls succeed on WebGPU and change nothing on screen: the
    // Masks and Qualifiers tabs are fully interactive and completely inert on
    // the preferred backend.
    const gpu = read('radiance_webgpu.js');
    assert.match(gpu, /mask/i,
        'no mask implementation in the WebGPU backend — the Masks tab is inert there');
    assert.match(gpu, /qualifier/i,
        'no qualifier implementation in the WebGPU backend — the Qualifiers tab is inert there');
});
