# Radiance — Viewer Audit & Local Version Status

**Date:** 2026-07-27 · **Branch:** `release/cleanup` (uncommitted) · **Scope:** viewer backend + frontend + delivery path, plus the cumulative state of everything fixed this session.

---

## 1. Verdict

The viewer is the largest and most defect-dense component in the pack: `nodes/monitor/viewer.py` plus a **19,880-line** `radiance_viewer.js` (46% of the entire frontend). I found **31 issues**, verified every one against the source, and fixed the 12 that were unambiguous and low-risk.

The theme is narrower than the earlier audits: **the delivery path silently produces wrong masters.** Four separate defects — a legal-range clamp on float EXR, a mis-parsed aspect ratio, a `-0:` slice, and a blank output path — each independently ruin an export while reporting `"status": "success"`. That is the single most damaging cluster found anywhere in this codebase, because the artist has no signal that anything went wrong.

---

## 2. Fixed in this pass (12)

All verified numerically or by source inspection; full suite **1621 → 1645 passing, zero regressions**, ruff at parity with baseline.

### Delivery — wrong masters, reported as success

| # | Issue | Evidence |
|---|---|---|
| 1 | **`soft_clip` clamped 32-bit EXR to broadcast legal range.** The checkbox defaults ON and the clamp was ungated on format, so a scene-linear master exported as EXR had every pixel squeezed into `[0.0627, 0.9216]` — highlights destroyed, blacks lifted 6.3%. | Clamp now gated: EXR/HDR/DPX/float never legal-limited, and the ignored request is logged. A 40.0 specular now survives. |
| 2 | **`9:16 (Vertical)` parsed as ratio 9.0.** Only the numerator was read. A 1920×1080 plate took the *pillarbox* branch against a ratio of 9.0 and was blanked to a 214-pixel letterbox slit. The other three presets worked only because their denominator is 1. | Now parses both terms → 0.5625. Verified all four presets. |
| 3 | **`pad == 0` blanked the entire frame.** `t[:, :, -0:, :] = 0` is `t[:, :, 0:, :] = 0`. Reachable on small/proxy plates → a fully black master. | `pad > 0` guards on both branches; measured 0% zeroed at `pad=0` (was 100%). |
| 4 | **Blank Location wrote to ComfyUI's CWD *and* bypassed the sandbox check.** The entire containment check lived inside `if output_path:`, so the default — which the placeholder "Default Output Folder" actively invites — skipped it. `str(Path('') / 'Shot_v0001')` is a *relative* path. | Default resolved *before* the check, so containment always runs. |

### Viewer node

| # | Issue | Evidence |
|---|---|---|
| 5 | **Alpha was run through the Reinhard tonemap.** An opaque matte (A=1.0) came out at **0.5**, so the PNG fallback composited the plate at 50% opacity — looking like a decode glitch. | Tonemaps `[..., :3]` only; RGBA `[4,4,4,1]` → `[0.8, 0.8, 0.8, 1.0]`. `RadianceLiteViewer` already did this correctly. |
| 6 | **O(B²) memory traffic in exposure bracketing.** `image * 0.25` and `image * 4.0` multiplied the **entire batch** inside the per-frame loop, while `_process_frame` reads only `image[frame_idx]`. A 120-frame 1080p RGBA fp32 plate moved ~950 GB per preview; OOM outright on a CUDA tensor. | Slices first. Verified numerically identical for every frame. |
| 7 | **No `IS_CHANGED` → delivery breaks permanently.** Once a viewer's entry is evicted from the 8-slot cache, `/radiance/deliver` answers *"No frames found… Run the workflow first"* forever: re-queuing skips the unchanged node, nothing repopulates the cache, and the only escape is perturbing an upstream widget. | Added `IS_CHANGED → NaN` (always-dirty, correct for a side-effecting OUTPUT_NODE). |

### Frontend

| # | Issue | Evidence |
|---|---|---|
| 8 | **Deleting one viewer blanked every other viewer's controls.** The HUD is a `static singletonHUD` shared by all instances, but `destroy()` detached it unconditionally. Survivors lost exposure, curves, scopes and delivery until a page reload, because `createHUD()` never rebuilds an existing singleton. | Detaches only when it isn't the shared HUD, or when it's the last instance. |
| 9 | **Widget-hiding scanned the whole document** — hiding `bit_depth` / `exposure_bracketing` rows on *other node packs'* nodes, with no visible cause. Two copies; one also ran that O(DOM) scan from 6 retry timers **and** a `document.body` MutationObserver with `subtree: true` for 10s, i.e. on every DOM insertion anywhere in ComfyUI during load. | Both scoped to Radiance nodes via `[data-node-id]`; observer and retry storm removed. |
| 10 | **Unescaped `innerHTML` in the delivery history.** `dvFilename.value` (raw user input) and the server's `path`/`qc` interpolated directly. The same class has a working `escapeHtml()` and uses it correctly elsewhere. | All three values escaped. |
| 11 | **`destroy()` contained zero `cancelAnimationFrame` calls.** `_grainRAF` re-schedules itself *before* its own enable guards, so every deleted viewer left a 60fps closure running forever, holding the renderer, GL resource maps and every `Float32Array` of HDR frame data. Ten add/delete cycles = ten immortal render loops. | Cancels `_grainRAF`, `_seqRAF`, `_referenceScopeRAF`, `_videoRAF`; clears both scope debounce timers; disconnects all three ResizeObservers; nulls the renderer and the frame buffers. |
| 12 | **WebGL `destroy()` left stale handles and never released the context.** `textures`/`programs` kept deleted handles, so a scope debounce firing after teardown passed its `if (!this.programs[mode]) return;` guard and issued `useProgram` on a deleted program. No `loseContext()` — ~16 add/delete cycles hit Chrome's context limit, which then kills the *oldest* context (possibly ComfyUI's own canvas). | Maps cleared; `WEBGL_lose_context` released. |

---

## 3. Found, verified, NOT fixed (needs your decision)

These are real — I confirmed each in the source — but each needs either a design call or a refactor I shouldn't make unilaterally.

### Critical

**A. `/radiance/deliver` runs the entire export synchronously inside `async def`.**
Everything from grading through cv2 FX, the 2× upscale, and the blocking `ffmpeg` call runs on aiohttp's event loop. A 240-frame 1080p export freezes ComfyUI's whole websocket for minutes: progress bar, node highlighting and queue view all stall, and the client's own progress poll can't be answered — so the bar sits at 0% and jumps to 100%. `_progress_set(..., 90, "encoding")` is *unobservable by construction*.
**Fix:** `run_in_executor` + per-frame progress. Straightforward but touches the whole handler.

**B. GPU scopes read `UNSIGNED_BYTE` from a float FBO.**
`pipelinePrecision` defaults to `'f32'`, so the scope FBO is `RGBA32F`, but the readback is hard-coded to `gl.RGBA, gl.UNSIGNED_BYTE` (`radiance_webgl.js:203`). Per WebGL2 that pair is only valid for normalized fixed-point buffers → `INVALID_OPERATION`, buffer untouched. On any GPU with `EXT_color_buffer_float` (i.e. every desktop GPU) waveform, vectorscope, histogram, parade and chromaticity should be permanently black or stale.
**Why I didn't fix it:** the fix is mechanical (`Float32Array` + `gl.FLOAT` + convert), but I cannot run a WebGL context here to confirm the failure or validate the fix. **Please check your browser console for `readPixels: invalid format/type combination` — that one line confirms or refutes it in seconds.**

### High

**C. All scopes are computed on the *ungraded* source texture.**
`renderHistogram`/`renderScope` are passed `this.textures.image` — the raw upload. The scope shader applies only an sRGB decode; it never applies exposure, lift/gamma/gain, contrast, curves, LUT or the display transform, all of which live only in the display shader. Pull exposure up 2 stops and the image brightens while the waveform doesn't move. **Every grading decision made against the scopes is against the wrong signal.**
**Decision needed:** render the graded frame to an FBO and scope that (correct for a grading tool), or label the panels "SOURCE" if reading the source is deliberate.

**D. Viewer cache is keyed by LiteGraph node id, and bounded by count not bytes.**
Two workflows each with a viewer at node id 5 overwrite each other; the delivery export then pulls the *other shot's* plate and versions it under this shot's filename — silently. Separately, 8 entries × a 240-frame 4K RGBA fp32 plate ≈ 8 × 31 GB, tensors stored un-cloned on their original device (so a CUDA tensor pins VRAM permanently), and the passthrough returns the *same object* that's cached, so any downstream in-place op mutates the plate that will later be exported.
**Decision needed:** the key needs a client/prompt-id prefix, which changes the JS↔Python contract.

**E. `bit_depth` and `exposure_bracketing` are unreachable — but deliberately so.**
Both are `view()` parameters absent from `INPUT_TYPES`, so ComfyUI pins them at `'32-bit Float'` and `True`. Initially this reads as a bug; the JS makes the intent explicit — it *hides and pins* both widgets. So the design is "always 32-bit, always bracketed". The consequence is that the node writes **12 files per frame** (3 exposures × {rhdr, exr, png, rpick}) with no way to reduce it. A 240-frame shot = 2,880 files per run.
**Decision needed:** is that the intent? If so, item F below matters a lot more.

**F. Viewer temp files are never deleted.** Filenames use `uuid4()`, so every execution produces a fresh set; there is no `unlink`/`rmtree` for them anywhere in the package. Combined with E: scrub the exposure slider and re-queue ten times → ~30,000 orphaned files and >100 GB in temp, reclaimed only by restarting ComfyUI. `zlib.compress(level=6)` on 32 MB of float32 also costs ~0.6 s/frame for ~2% size benefit over level 1.

### Medium (16 further items)

Confirmed and documented, not fixed: PNG fallback written scene-linear but read as sRGB by the JS (≈2.2γ too dark on the fallback path); RGBA→JPEG thumbnail raises and silently aborts *all* sidecar generation while still reporting success; progress left stuck at 90% after a failure, poisoning later polls; `/radiance/deliver` missing the idempotent route-registration guard its two siblings have; `/radiance/progress` returning HTTP 200 on error (→ `NaN%` in the UI); global `Space`/single-letter key capture stealing ComfyUI's canvas-pan modifier; six anonymous global listeners `destroy()` can't remove; `RadianceCurveEditor` has no `destroy()` at all; direct `widgets_values[index]` mutation that can corrupt an unrelated widget's saved value; `onRemoved`/`onSelected` assigned without chaining; five unguarded `JSON.parse` on localStorage inside render paths; five methods defined twice (≈250 lines of dead code in `radiance_viewer.js` shadowed by `radiance_viewer_export.js`); histogram clamping HDR to [0,1] while the waveform correctly uses PQ; unnamespaced global CSS and duplicated element IDs across instances; auto-executing `startup` terminal macro from unvalidated localStorage; `frame N` terminal command changing the index without loading the frame, off by one.

---

## 4. What the viewer gets right

Worth stating, because the list above is long:

- **`cache.py` locking is correct.** Every access goes through helpers holding the right lock; nothing outside the module touches the dicts. No unlocked read-modify-write, no check-then-act, no mutation during iteration.
- **The grading shader applies each operation exactly once, in a defensible order** — linearize → IDT → exposure → white balance → lift/gamma/gain (with a correct ACEScct round-trip) → contrast → log wheels → printer lights → shadows/highlights → curves → saturation → hue. No double exposure, no `[0,1]` clamp before the display transform.
- **Object URLs are balanced** — 7 created / 7 revoked, plus 3/3 in the export module.
- **No `eval`, no `document.write`.** The two `new Function` sites are behind an explicit `localStorage['radiance.devTools']` opt-in and operate only on user-typed code.
- **`_termLog` escapes its input**, so all backend-sourced terminal text is safe; of 83 `innerHTML` sinks, only one was tainted.
- **A malicious `.rad` workflow is inert** in the prompt tab — node titles go through `textContent`, not `innerHTML`.
- **`generationID` guarding** correctly prevents stale frames from a previous queue run overwriting the current one.
- Frame-range arithmetic, the RHDR header overflow guard, `safe_join`, `_validate_image`, and flicker/cut-detection channel handling are all correct.
- `RadianceLiteViewer` handles alpha correctly — it was the reference for fix #5.

---

## 5. Cumulative state of the local version

| Batch | Focus | Files | Tests added |
|---|---|---|---|
| 0 + 1 | Broken features + RCE paths | 17 | 23 |
| 2 | Consolidation (tiling, transfer functions, matrices, caches, `no_grad`) | 21 | 48 |
| SDR→HDR Universal | 5 bugs incl. the dead direct-pixel backend | 2 | 22 |
| Viewer + delivery | 12 fixes | 6 | 24 |
| **Total** | | **~40 unique files** | **117** |

**Test suite: 1524 → 1645 passing.** The 11 remaining failures are identical to the pristine baseline (missing OpenEXR/aiohttp in the sandbox, flux2 detection, node-keys snapshot) — **zero regressions across every batch**, verified each time by diffing the failure set against an untouched copy of the tree.

### Still open

**Guardrails (was "Batch 3"), highest leverage remaining:**
1. `ci.yml:102-108` — the stale `--ignore` list; CI has been red since 2026-07-10 with 13 failures.
2. `tests/test_node_smoke.py:992` — `test_all_keys_have_class` calls `warnings.warn` instead of asserting, and is currently swallowing 16 broken node imports.
3. Startup logs "successfully loaded N nodes" with no comparison against expected — an 88% shortfall reads as success.

**Then:** the 54 unregistered nodes; the JS widget-toolkit dedup (6 diverged copies) and `escapeHtml` (4 copies); the ~30 silent-failure handlers; `OpenImageIO` missing from all three platform requirements files; and the 3 declared-but-never-imported dependencies.

### Git status — unchanged from my last report

Still **uncommitted** on `release/cleanup`. Your HEAD matches `beta/release/cleanup` exactly, but **`beta/main` is 36 commits ahead**, and almost every bug fixed here is still live there. 32 of 36 files apply cleanly onto `beta/main`; the exceptions are `pixel_sdr2hdr.py` and `temporal_rudra.py` (don't exist upstream), `uplift_universal.py` (300-line local divergence), and a trivial `dcc.py` import hunk.

**The work is not protected by a commit.** That's the most urgent item on this list.
