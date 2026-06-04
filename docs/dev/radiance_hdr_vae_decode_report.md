# ◎ Radiance HDR VAE Decode — Node Audit Report
**File:** `nodes_engine.py` · **Class:** `RadianceHDRVAEDecode` · **Version:** 3.0.0  
**Date:** 2026-05-23

---

## 1. Prompt

> Read `nodes_engine.py` in full. For the `RadianceHDRVAEDecode` node: map the complete data-flow architecture; audit every code path for bugs, silent failures, type mismatches, and edge cases; determine whether the `alpha` input connection is functionally live or dead. Produce a structured engineer report with verdict, evidence, and recommendations.

---

## 2. Architecture Overview

`RadianceHDRVAEDecode` is a **thin orchestration wrapper** around `RadianceVAE4KDecode` (from `hdr/vae.py`). It does not implement decode logic itself — it delegates everything to the underlying engine and adds three layers on top:

```
INPUTS
──────
  samples (LATENT)         ─┐
  vae (VAE)                  │
  target_space (STRING)      │
  tile_size (STRING)         │
  overlap (INT)              ├──► apply()
  exposure_adjust (FLOAT)    │
  alpha? (IMAGE)  ←★         │
  hdr_mode (STRING)          │
  display_tonemap (STRING)   │
  hdr_scale_factor (FLOAT)   │
  use_fast_decoder (STRING)  │
  fast_decoder_size (STRING) │
  **kwargs (inherited)      ─┘

INTERNAL FLOW
─────────────
  1. Lazy-init _engine (RadianceVAE4KDecode singleton, cached at class level)
  2. Auto-detect model arch (16ch+5D = Wan, 16ch = Flux, else SDXL)
  3. [Optional] Load distilled decoder via load_radiance_decoder_weights()
     └─ Forces hdr_mode = "Compress (Log)" if fast_decoder enabled
  4. Strip duplicate kwargs (safe_kwargs) to avoid TypeError
  5. Delegate to engine.decode() with all params + force_hdr_decode=True
  6. Guard hdr_scale_factor: skip multiply for display-referred target_space
  7. Assemble metadata JSON (settings + timestamp)

OUTPUTS
───────
  image    (IMAGE)   — decoded tensor
  metadata (STRING)  — JSON of all decode settings applied
```

### What the wrapper adds over bare `RadianceVAE4KDecode`

| Feature | Status |
|---|---|
| `display_tonemap` widget | Added (was silent Reinhard before — BUG 1 fix) |
| `hdr_scale_factor` with guard | Added (BUG 2 fix) |
| `target_space` default = sRGB | Corrected (was "Linear" — BUG 3 fix) |
| `metadata` STRING output | Added (BUG 7 fix) |
| `**kwargs` forwarded | Fixed (BUG 8 fix) |
| Fast/Turbo decoder path | New in v3.0.0 |

---

## 3. Bug Audit

### ✅ Bugs Fixed in v3.0.0 (confirmed in code)

| # | Bug | Fix Verified |
|---|---|---|
| BUG 1 | `display_tonemap` not forwarded — Compress(Log) silently used Reinhard | ✅ Now in signature + passed to decode() |
| BUG 2 | `hdr_scale_factor` multiplied on sRGB output → blown highlights | ✅ `_SCENE_REFERRED` guard + warning log |
| BUG 3 | `target_space` default was `"Linear"` (silent display mismatch) | ✅ Forced to `"sRGB"` in INPUT_TYPES override |
| BUG 4 | NDI `apply()` IndexError on empty image batch | ✅ `image.shape[0] == 0` guard added |
| BUG 5 | NDI singleton didn't track `stream_name` → stale sender on rename | ✅ `_ndi_stream_name` tracked, sender recreated |
| BUG 6 | NDI fallback (turbo failed) skipped log encoding | ✅ Encoding gated on `turbo_succeeded` |
| BUG 7 | No metadata output — decode settings invisible to downstream nodes | ✅ JSON metadata string returned |
| BUG 8 | `**kwargs` not forwarded — new vae.py params silently dropped | ✅ `safe_kwargs` pattern implemented |
| BUG 9 | Duplicate `numpy` import in NDI `apply()` | ✅ Removed |

---

### ⚠️ Remaining Issues Found

**Issue 1 — `RadianceLUTApply` triple-defined, never registered in this file**

`RadianceLUTApply` is defined in:
- `nodes_engine.py` (simple path-based .cube loader — NOT in NODE_CLASS_MAPPINGS)
- `color/lut.py` (canonical, full-featured — registered via color/__init__.py)
- `nodes/monitor/cdl.py` (another variant — registered separately)
- `nodes_radiance_viewer.py` (viewer-optimised variant)

The comment says `color/__init__.py` is canonical but the `nodes_engine.py` copy is fully built out and internally referenced in tests (`test_lut.py` imports from `nodes_engine`). **Risk:** implementations silently diverge between files. Recommendation: delete the `nodes_engine.py` copy and update test imports to point at `color/lut.py`.

---

**Issue 2 — `Radiancev3_MasterHub` is defined but unregistered (dead node)**

The class exists in `nodes_engine.py` (line 793) with a full `INPUT_TYPES`, `RETURN_TYPES`, and `initialize()` method but is **absent from `NODE_CLASS_MAPPINGS`**. It will never appear in ComfyUI. Either register it or remove it — dead class definitions cause confusion for future maintainers.

---

**Issue 3 — `_SCENE_REFERRED` uses substring matching**

```python
if any(s in target_space for s in _SCENE_REFERRED):
```

This checks whether any element of `_SCENE_REFERRED` is a *substring* of `target_space`. For all current values this works correctly, but it's fragile. A future colorspace named e.g. `"Log (non-Linear sRGB preview)"` would incorrectly match `"Linear (sRGB)"`. Safer: use an exact set lookup (`target_space in _SCENE_REFERRED`).

---

**Issue 4 — Fast decoder model-type detection is fragile**

```python
_ch = _samples.shape[1]
if _ch == 16:
    model_type = "wan" if _samples.ndim == 5 else "flux"
else:
    model_type = "sdxl"
```

This heuristic only covers Flux/Wan (16ch) and SDXL (4ch). SD1.5 is also 4ch and would be misidentified as SDXL. More importantly, if `load_radiance_decoder_weights()` raises an unhandled exception (wrong model type passed), the node will crash with no user-facing message. Recommend wrapping in a try/except with a clear `logger.error()` and graceful fallback to the standard VAE.

---

**Issue 5 — Metadata timestamp uses local time, not UTC**

```python
"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
```

`datetime.now()` is wall-clock local time — ambiguous across timezones and daylight savings. Use `datetime.datetime.now(datetime.timezone.utc).isoformat()` for unambiguous audit trails.

---

## 4. Alpha Connection Verdict

### ✅ ALIVE — Alpha is functional and important. Do not remove it.

**Evidence trail:**

```
RadianceVAE4KEncode.encode()
  └─ extracts pixels[..., 3:4] → alpha tensor
  └─ RETURN_NAMES = ("samples", "alpha", ...)
       tooltip: "Alpha channel tensor — wire to Radiance VAE 4K Decode alpha input."

RadianceHDRVAEDecode.apply(alpha=None)
  └─ passes alpha=alpha → engine.decode()

RadianceVAE4KDecode.decode(alpha=None)
  └─ line 2915-2931: if alpha is not None:
       alpha_ch = alpha_f[..., :1]           # resize to match decoded H×W
       img = torch.cat([img, alpha_ch], -1)  # RGBA output
  └─ metadata["alpha_restored"] = True
```

**What happens without it:**
- RGBA source images (e.g. masked composites, keyed footage) lose their alpha after encode→decode roundtrip
- The decoded image is always RGB — the alpha that was encoded is permanently discarded
- Any downstream node expecting RGBA (compositing, mask extraction) receives wrong data silently

**What it is NOT:**
- It does not affect latent space or the decode math in any way
- It is purely a passthrough restore at the very end of decode
- It carries no data from the diffusion process — it is literally the alpha channel you put in, resized to match the decoded resolution and re-attached

**Summary:** If you are working with purely RGB images (no transparency), the alpha wire is safely unconnected and defaults to `None` with no side effects. If you are working with RGBA content, it is the only mechanism to preserve transparency through the encode→decode cycle.

---

## 5. Node Registration Check

```
NODE_CLASS_MAPPINGS (nodes_engine.py)
  ✅ RadianceHDRVAEDecode        → registered
  ✅ RadianceColorSpaceTransform → registered
  ✅ RadianceHDRAnalysis         → registered
  ✅ RadianceNDISender           → registered
  ⚠️ RadianceLUTApply           → defined here, NOT registered (canonical in color/__init__.py)
  ❌ Radiancev3_MasterHub        → defined here, NOT registered (dead class)
```

---

## 6. Summary & Recommendations

| Priority | Action |
|---|---|
| 🔴 High | Fix `_SCENE_REFERRED` to use `target_space in _SCENE_REFERRED` (exact match) |
| 🔴 High | Register or delete `Radiancev3_MasterHub` — dead class is misleading |
| 🟡 Medium | Wrap fast decoder `load_radiance_decoder_weights()` in try/except with graceful fallback |
| 🟡 Medium | Delete `RadianceLUTApply` from `nodes_engine.py`; canonicalise to `color/lut.py` |
| 🟢 Low | Switch metadata timestamp to UTC (`datetime.timezone.utc`) |
| ✅ Done | All 9 v2.x bugs confirmed fixed in v3.0.0 |

**Alpha connection:** Keep it. It is the sole mechanism for RGBA roundtrip preservation. RGB-only workflows can leave it unconnected — it has no effect when `None`.
