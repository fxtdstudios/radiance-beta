# Radiance — Full Production Audit, 2026-08-09

Thirty-section industry-grade audit per the GENOME specification, executed from
scratch against `release/cleanup` at `4900e98` (v3.2.0 + Read overhaul + video
decode rewrite). Environment: real torch 2.13 (CPU), OpenEXR, OpenImageIO
3.1.16, PyOpenColorIO 2.5.2, tifffile, ffmpeg 4.4. No GPU, no live ComfyUI
runtime — everything GPU- or browser-dependent is marked UNVERIFIED below.

---

## Executive Summary

**Overall health: good core, one severe colour defect found and fixed.**

The I/O layer is in the best shape it has ever been: EXR round-trips are
bit-exact including negative and >1 scene-linear values, the data window is
honoured, alpha is preserved both directions, the video path holds an exact
N-in/N-out frame contract across H.264, H.265 10-bit and ProRes from 1 to 100
frames, and sequence pattern reads slice by true frame number.

The audit's headline finding: **`RadianceColorSpaceConvert` performed no
conversion at all for 10 of its 16 advertised colour spaces** in a default
install, and its inline LogC3 was wrong in both directions. This shipped
through five previous review passes because the test suite's OCIO mock was a
truthy `MagicMock` whose no-op processor made the node an identity transform
*under test as well* — the tests were validating nothing. Both the defect and
the test blindfold are fixed, with published-reference regression tests.

- Biggest risk (pre-fix): silently wrong colour on any workflow using camera
  log or ACES working spaces through the convert node.
- Biggest risk (remaining): nothing has ever run inside a live ComfyUI with a
  GPU. The viewer JS, CUDA paths, VRAM behaviour and model nodes are
  UNVERIFIED as a class.
- Biggest strength: the layered test suite (2,185 tests, including per-node
  functional execution) and an I/O layer that now refuses to lie.

---

## Architecture Assessment

Score: **8/10** (unchanged from July; re-verified, not re-litigated).

Package: 109 nodes in 10 categories, organised under `nodes/<group>/` with a
registration ratchet (`EXPECTED_MIN_NODE_COUNT`), an Environment Guard that
prints an honest dependency table at startup, and shared single-source colour
math in `color/{transfer,matrices,luts,ops}.py`. Concerns: (1) the group
mapping dicts in `nodes/<group>/__init__.py` are still hand-transcribed — the
mechanism that once lost nine nodes; the snapshot test mitigates. (2) Node
classes re-implementing math that exists in `color/` is exactly how this
audit's P0 happened; `RadianceColorSpaceConvert` is fixed, but the pattern
deserves a lint rule.

---

## Node Audit

All 109 nodes load; all 109 `INPUT_TYPES` are callable; the functional harness
executes 80 and skips 30 with stated reasons (opaque inputs: MODEL/VAE/CLIP or
real model files), 1 newly reclassified (HDRLoRALoader's empty-path rejection
is graceful failure, not error). 0 fail.

| Category (nodes) | Status | Notes |
|---|---|---|
| VFX (29) | Executes clean | Optical flow remains single-scale LK (P3, known) |
| HDR (19) | Executes clean | ACES 2.0 tonescale still approximate (P2, documented) |
| Video (14) | **Verified with real data** | Frame contract exact 1–100 frames, 3 codecs |
| Color (13) | **1 P0 found & fixed** | See Critical Bugs #1 |
| Generate (11) | Executes under stubs | Real sampling UNVERIFIED (needs ComfyUI+GPU) |
| Review (10) | Loads; JS UNVERIFIED | No browser in audit environment |
| Load & Save (5) | **Verified with real data** | 2 P1s found & fixed (TIFF, dir/glob) |
| Upscale (4) | Loads; models UNVERIFIED | Downloads sha256-pinned; auto-download is consent-free (P2) |
| Pipeline (3) | Executes clean | Audio transcription API key via env var — correct |
| Core (1) | Executes clean | |

Scores below 90 that matter: `RadianceColorSpaceConvert` was **15/100**
pre-fix (correctness zero for 10/16 spaces); post-fix, with references pinned:
**92/100**. `RadianceWrite`/`RadianceRead` post-fix: **90/100** (deducting for
UNVERIFIED live-ComfyUI upload routes). Model-dependent nodes cannot honestly
be scored above the 70s from this environment and are marked UNVERIFIED.

---

## VFX Pipeline Audit (measured, not inspected)

- **EXR:** 32f round-trip max error 0.0 (bit-exact) on [-0.5, 19.5] HDR data;
  half float max error 0.0078 on the same range (within half precision).
  Data window ≠ display window: conformed correctly, values placed exactly.
  RGBA: alpha 0.5 survives read and write (channels A,B,G,R written).
- **Colour:** all 16 spaces of the convert node now hit published 18%-grey
  code values — LogC3 0.3910, LogC4 0.2784, ACEScc/cct 0.4136, sRGB 0.4614,
  BT.709 OETF 0.4090, DaVinci 0.3360, Log3G10 0.3333, V-Log 0.4233, C-Log3
  0.3434, F-Log2 0.3910, BMD Gen5 0.3836, N-Log 0.3637 — and round-trip at
  float precision (max error 6e-6). Negative scene-linear survives log spaces.
- **Video:** 40 in → 40 out verified; also 1, 2, 10, 24, 25, 50, 100. No
  drops, no duplicates, no off-by-one, all three codecs.
- **Sequences:** `####`, `%04d`, single-frame-of-sequence, and (post-fix)
  directory and glob inputs all resolve; explicit 1002–1004 window returns
  exactly frames 1002–1004 by number.
- **Temporal consistency of AI nodes: UNVERIFIED** (no models, no GPU).

## Performance Report (CPU, 1080p, 30-run mean)

Grade 16.3 ms/frame · Curves 9.7 · WhiteBalance 11.2 · CDL 39.1. No
optimisation attempted; nothing here is a bottleneck at CPU-preview scale.
GPU benchmarks UNVERIFIED.

## Memory Report

150 consecutive 1080p executions of the worst allocator (WhiteBalance):
RSS plateaus at 589 MB after run ~50 and is flat thereafter — allocator arena,
not a leak. VRAM behaviour UNVERIFIED (no GPU).

## Security Report

No `shell=True`; all checkpoint loads use `weights_only=True` or safetensors;
model downloads are sha256-pinned with atomic rename; CDL XML via defusedxml;
API keys resolved from env-var names, never stored; bridge/path-traversal
covered by its own passing test file. `except: pass` instances (26) are
confined to logging/teardown. **No new vulnerabilities found.** Remaining P2:
upscale models auto-download on first queue without a consent prompt.

## Test Coverage

Before: 2,133 pass / 4 fail / 60 skip. After: **2,185 pass / 0 fail / 61
skip** (+21 subtests), ruff clean. New permanent regression files:
`test_colorspace_convert_regression.py` (identity guard, round-trips, pinned
18%-grey references, OCIO-mock guard), `test_io_audit_regressions.py` (TIFF
depth honesty, sequence windowing). Untested areas: live ComfyUI execution,
viewer front-end, CUDA/VRAM, real model inference, DPX write on all
compressions.

---

## Critical Bugs Found This Pass

**#1 (P0, fixed) — RadianceColorSpaceConvert converted nothing for 10/16
spaces.** Root cause: analytical fallback implemented only 6 curves and fell
through to `return img`; correct implementations already existed in
`color/transfer.py`/`color/luts.py`. Evidence: 0.18 → LogC3 returned 0.18;
encode/decode mutually inconsistent (0.18 → 0.060 round trip). Fix: delegate
every space to the shared modules; add ACEScc (S-2014-003) to `transfer.py`;
true BT.709 OETF; AP1 matrix for ACEScc/cct; unsupported spaces raise.
Risk of fix: colour output changes for any graph that used the broken node —
re-check masters, as with 3.2.0.

**#2 (P0 test-infra, fixed) — the suite could not see #1.** conftest's OCIO
manager mock was truthy, so `_try_ocio` "succeeded" via a no-op MagicMock.
Fix: `is_loaded = False` on the mock + a regression test that fails if anyone
reverts it.

**#3 (P1, fixed) — 32-bit float TIFF writes silently produced 8-bit files**
without tifffile (undeclared dependency); 16-bit warned but downgraded.
Both now raise with an install hint; tifffile declared in the Environment
Guard; reader warns when it cannot probe depth.

**#4 (P1, fixed) — directory/glob sequence reads always empty**
(`files[1001:99999]` index slicing with frame-number defaults). Now parses
frame numbers and reconciles the window against what is on disk.

**Open (pre-existing, unchanged priority):** ACES 2.0 tonescale approximation
(P2, documented); consent-free model auto-download (P2); scene-cut batch-max
normalisation (P2); tiled VAE decode pastes without blending (P2);
`RadianceVideoAssembler` class-global accumulator (P2); optical flow
single-scale LK (P3); Tier-3 diffusion upscale ignores `scale` (P3).

---

## Production Readiness

**READY WITH CONDITIONS.**

For image/EXR/sequence/video I/O, grading, CDL, LUTs, and colour-space
conversion, the pipeline is now verified with real pixel data and holds its
contracts. The conditions:

1. **Run one supervised shot through a live ComfyUI with a GPU before real
   production** — the entire runtime/viewer/model layer is UNVERIFIED from
   this environment, and no amount of unit testing substitutes.
2. Re-check any master that ever passed through `RadianceColorSpaceConvert`
   with a camera-log or ACES space: it was not converted.
3. Do not use the ACES 2.0 output transforms for final delivery until the real
   tonescale is implemented (known, documented limitation).
4. `git commit` of this audit's changes is pending: a stale
   `.git/index.lock` on the host blocked the commit from the sandbox —
   delete it and commit.

*"Can a professional VFX studio safely rely on this software in production?"*
For the verified subsystems, yes. As a whole system: not until condition 1 is
met — one honest GPU shot is worth more than the 2,185 tests now passing.
