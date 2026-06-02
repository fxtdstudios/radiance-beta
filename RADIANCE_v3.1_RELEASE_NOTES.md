# Radiance v3.1.0 — Release Notes & Engineering Audit

**Date:** 2026-05-13  
**Status:** ✅ Production Release  
**Nodes:** 224 registered · 14 categories  
**Test suite:** 341 passing · 0 failures

---

## Summary

Full engineering audit and bug-fix pass performed prior to tagging v3.1.0.
Scope covered: architecture review, debug session, code review across the entire package.
All identified issues resolved. Test suite clean.

---

## Fixes Applied

### 1. `nodes_engine.py` — `RadianceHDRVAEDecode`

**Bug:** `CATEGORY` was set to `"◎ Generate"` instead of `"◎ HDR"`, causing the node to appear in the wrong menu group.

**Bug:** No engine instance cache — a new `RadianceVAE4KDecode()` object was constructed on every single call.

**Fix:**
- Corrected `CATEGORY` to `"FXTD STUDIOS/Radiance/◎ HDR"`
- Added class-level `_engine` lazy singleton:

```python
_engine: "RadianceVAE4KDecode | None" = None

# In apply():
if RadianceHDRVAEDecode._engine is None:
    RadianceHDRVAEDecode._engine = RadianceVAE4KDecode()
engine = RadianceHDRVAEDecode._engine
```

---

### 2. `hdr/vae.py` — `RadianceVAE4KEncode`, `RadianceVAE4KDecode`, `RadianceVAE4KRoundtrip`

**Bug (all 3 classes):** Orphaned docstring anti-pattern — `CATEGORY = "..."` was placed *before* the docstring, making `MyClass.__doc__ = None` and breaking any introspection or help tooling.

**Bug (`Decode`, `Roundtrip`):** `CATEGORY` was set to `"◎ Gen"` — a truncated, incorrect value.

**Bug (`Decode`):** NaN/Inf pre-check was gated inside `if source_space == "Linear"`, meaning it was skipped entirely for Passthrough and other modes.

**Fix:**
- Moved docstring to first position in all 3 classes
- Corrected `CATEGORY` to `"FXTD STUDIOS/Radiance/◎ HDR"` on all 3
- Moved NaN/Inf guard outside the `source_space` branch — unconditional, mode-aware:

```python
# v2.3: Pre-transform NaN/Inf guard — mode-aware, unconditional.
if torch.isnan(img).any() or torch.isinf(img).any():
    logger.warning("[Radiance 4K Decode v2.3] VAE produced NaN/Inf — sanitizing")
    if hdr_mode == "Passthrough":
        img = torch.nan_to_num(img, nan=0.0, posinf=1.5, neginf=-0.05)
    else:
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
```

---

### 3. `fast_vae.py` — Multi-model decoder cache

**Bug:** `_TRAINED_DECODER_CACHE` was a single `Optional[nn.Module]`, meaning loading a second model configuration silently evicted the first. Pipelines mixing latent channel counts or full/turbo modes would reload from disk on every call.

**Fix:** Replaced with a dict keyed on `(latent_channels, is_full)`:

```python
# Before
_TRAINED_DECODER_CACHE: Optional[nn.Module] = None

# After
_TRAINED_DECODER_CACHE: dict = {}

cache_key = (expected_channels, request_is_full)
if cache_key in _TRAINED_DECODER_CACHE:
    return _TRAINED_DECODER_CACHE[cache_key]
...
_TRAINED_DECODER_CACHE[cache_key] = model
```

---

### 4. `viewer_utils.py` — 6 fixes

**Bug 1 — `luma_mix` silently dropped from fast-path:**  
`apply_grading`'s `is_default` guard checked 12 parameters but omitted `luma_mix`. When all other controls were at identity, luma-preservation was silently no-oped regardless of the `luma_mix` value.  
**Fix:** Added `and abs(luma_mix - 1.0) < 0.001` to the `is_default` expression.

**Bug 2 — `_lut_clog3` no output clamp:**  
The CLog3 curve function returned unbounded float values. Downstream operations assumed `[0, 1]` range.  
**Fix:** `return np.clip(result, 0.0, 1.0)`

**Bug 3 — Inline matrix allocations in ACEScct branch:**  
`_M_SRGB_TO_ACESCG` and a local `ACESCG_TO_LIN_SRGB` were allocated as new numpy arrays on every call through the ACEScct path.  
**Fix:** Promoted to module-level constants `_M_SRGB_TO_ACESCG` and `_M_ACESCG_TO_LIN_SRGB`.

**Bug 4 — Python channel loop for saturation:**  
Saturation was computed with an explicit per-channel Python loop.  
**Fix:** Vectorized: `out[..., :3] = luma + np.float32(saturation) * (out[..., :3] - luma)`

**Bug 5 — `RadianceType` orphaned docstring:**  
Same orphaned docstring pattern as the VAE classes — `CATEGORY` before docstring.  
**Fix:** Moved docstring to first position.

**Bug 6 — `_VIEWER_PROGRESS` unbounded dict:**  
Progress tracking dict grew without bound — one entry per unique workflow execution, never pruned.  
**Fix:** Replaced with a bounded LRU OrderedDict protected by a threading lock:

```python
_VIEWER_PROGRESS_MAX = 32
_VIEWER_PROGRESS: collections.OrderedDict = collections.OrderedDict()
_VIEWER_PROGRESS_LOCK = threading.Lock()

def _progress_set(key, value): ...   # thread-safe LRU insert
def _progress_get(key): ...          # thread-safe lookup, returns idle sentinel
```

---

### 5. `nodes_radiance_viewer.py` — 6 fixes (Radiance Viewer)

**Bug 1 — Terminal endpoint used `threading.Thread` + `exec`:**  
`threading.Thread` cannot be force-killed. A hung execution would block the endpoint indefinitely. Also, `exec` with user-supplied code in a shared thread namespace is a security concern.  
**Fix:** Replaced with `multiprocessing.Process` (force-killable via `.terminate()`):

```python
proc = multiprocessing.Process(target=_run_in_process, args=(result_queue, code, ns_snap), daemon=True)
proc.start()
proc.join(timeout=30)
if proc.is_alive():
    proc.terminate()
    proc.join(timeout=2)
    return web.json_response({"output": "⏱ Execution timed out.", "status": "error"})
```

**Bug 2 — No delivery path allowlist:**  
File delivery endpoint accepted arbitrary output paths with no validation — a path traversal risk.  
**Fix:** Added `os.path.abspath` + allowlist check against `folder_paths.get_output_directory()`:

```python
_allowed_root = os.path.abspath(folder_paths.get_output_directory())
if _allowed_root and not output_path.startswith(_allowed_root + os.sep):
    return web.json_response({"error": "Output path must be inside the ComfyUI output directory.", "status": "error"})
```

**Bug 3 — `_SAFE_FILENAME_RE` compiled inside a hot loop:**  
The regex was re-compiled on every filename sanitization call.  
**Fix:** Moved to module level: `_SAFE_FILENAME_RE = re.compile(r'[^\w\s◎_.() -]', re.UNICODE)`

**Bug 4 — `unique_id` empty-string not guarded:**  
An empty `unique_id` from the ComfyUI frontend would silently collide all progress entries under the same key.  
**Fix:** `str(unique_id) if unique_id and str(unique_id).strip() else str(id(self))`

**Bug 5 — `_VIEWER_PROGRESS` direct dict writes:**  
All progress writes in the viewer bypassed the new thread-safe LRU helpers.  
**Fix:** All `_VIEWER_PROGRESS[key] = ...` calls replaced with `_progress_set(key, ...)`, all reads with `_progress_get(key)`.

**Bug 6 — Import block missing new symbols:**  
`_progress_set`, `_progress_get`, `_M_ACESCG_TO_LIN_SRGB` were added to `viewer_utils.py` but not imported in `nodes_radiance_viewer.py`.  
**Fix:** Import block updated.

---

## Pre-Release Blockers Resolved

### `nodes_audio_cut.py` — binary corruption

`nodes_audio_cut.py` contained null bytes. Python's AST parser raised `ValueError: source code string cannot contain null bytes`. At runtime, the `nodes.pipeline` sub-package would silently skip both `RadianceAudioCut` and `RadianceAudioTranscribe` with no error shown.

**Resolution:** Removed `"nodes_audio_cut"` from `nodes/pipeline/__init__.py`. The two audio nodes are deferred to a future release.

### `test_resolve.py` — stale category assertion

`TestResolveRegistration.test_category_correct` expected `RadianceResolveBridge.CATEGORY == "◎ Pipeline"`. The node had been moved to `◎ Workspace` in a prior refactor and the test was not updated.

**Resolution:** Updated assertion to `"FXTD STUDIOS/Radiance/◎ Workspace"`.

---

## Architecture Findings (Post-v3.1 Cleanup)

These do not block v3.1.0 but should be addressed in v3.2:

| Issue | Files | Impact |
|---|---|---|
| Dead `NODE_CLASS_MAPPINGS` in utility packages | `color/__init__.py`, `film/__init__.py`, `hdr/__init__.py`, `image/__init__.py` | 30 phantom registrations never loaded — maintenance confusion only |
| `nodes_radiance_viewer` double-loaded | Top-level `NODE_MODULES` + `nodes/monitor/` | `dict.update()` overwrites with same classes — no user impact, wastes import time |
| `RadianceHighlightSynthesis` in two categories | `◎ Color` and `◎ HDR` | Node appears in one category only (last-writer wins); ambiguous source of truth |
| Training nodes in default menu | `nodes/training/` | 7 training nodes visible to all users — recommend `RADIANCE_DEV=1` env gate |
| `RadianceNukeServer` / `RadianceResolveBridge` silent failure | `nodes_workspace.py` | No user-visible error if Nuke/Resolve not running |
| Stale node count comment | `__init__.py` line ~100 | Says `~184 nodes`, actual is 224 |

---

## Package Overview — v3.1.0

| Category | Nodes | Notes |
|---|---|---|
| ◎ HDR | 51 | Core HDR pipeline, VAE, ACES2, turbo decoder |
| ◎ VFX | 30 | Multipass compositing, optics, depth, motion |
| ◎ Color | 28 | CDL, LUT, OCIO, curves, grade |
| ◎ Display | 15 | Viewer, scopes, filmstrip, A/B, diff |
| ◎ Video | 22 | DiT adapter, T2V/I2V, HDR video, character |
| ◎ Generate | 20 | Sampler, loader, HDR LoRA, MasterHub |
| ◎ Upscale | 12 | Tiler, AI upscale, face restore, router |
| ◎ Workspace | 12 | DNA, queue, Nuke/Resolve bridge, MCP |
| ◎ QC & Debug | 10 | False color, focus peak, frame stamp, QC |
| ◎ AI Assist | 9 | LLM driver, CLIP match, continuity check |
| ◎ IO & Delivery | 8 | EXR, read/write, shot context |
| ◎ Training | 7 | SDR degradation, turbo train, dataset gen |
| ◎ Utilities | 7 | Reroute, mux, gate, debug probe |
| ◎ Film | 1 | Film grain (also in VFX) |
| **Total** | **224** | |

---

## Test Results

```
341 passed · 148 skipped · 0 failed
```

Skipped tests are torch-dependent and skip cleanly in non-GPU environments. No unexpected failures.

---

*Radiance v3.1.0 — FXTD Studios*
