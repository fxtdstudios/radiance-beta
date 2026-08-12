# Radiance v3.1.2 — pre-release review

**Date:** 2026-07-29 · **Tree:** `release/cleanup` @ `8ae9deb` · **Verdict: do not ship yet.**

Five independent review passes — software engineering, colour pipeline, ML systems,
algorithm correctness, ComfyUI integration — plus a new per-node functional harness.
Six defects fixed and committed. **Twenty-two high-severity findings remain open.**

The package is in good structural health: 109 nodes all load, the node contract is clean
across every one of them, and 1891 tests pass. What the review found is not sloppiness —
it is a specific and consistent failure mode. Almost every defect below is something that
*runs without error and reports success* while producing the wrong result. That is the
hardest class to catch from the inside, and it is why five outside lenses found things the
1891 tests did not.

---

## 1 · What shipped in this commit

The functional harness (`tests/test_node_functional.py`) is new. Until now the suite
checked that each node *declares* itself correctly — `INPUT_TYPES` parses, `RETURN_TYPES`
exists, `FUNCTION` names a real method, the class instantiates — and never called anything.
A node could pass all of that and raise on its first line.

The harness builds synthetic inputs from each node's own `INPUT_TYPES` and calls its
`FUNCTION`, then checks the return arity against `RETURN_TYPES`, the shape and dtype of
IMAGE/MASK/LATENT outputs, and that the input tensors were not mutated in place.

**Result: 79 execute, 30 skip with a stated reason, 0 fail.** It found five defects on its
first run.

| # | Defect | Consequence |
|---|---|---|
| 1 | `nodes_io.py` `_out_path` used `Path.with_suffix()` | **Data loss.** `with_suffix` replaces everything after the last dot. `base` is `<dir>/<filename>_v0001`, so `sh010.comp_v0001` had a "suffix" of `.comp_v0001` and every version wrote to the same `sh010.exr`. Dotted stems are routine in VFX naming and `overwrite` defaults to True, so each render silently destroyed the previously approved one and reported success with the truncated path. An empty filename raised `ValueError: PosixPath('.') has an empty name`. |
| 2 | `RadianceSubpixelStabilizer` returned a 2-channel IMAGE | `RETURN_TYPES` declares `displacements_xy` as IMAGE; ComfyUI IMAGE is `(B,H,W,3\|4)` and every consumer indexes channels 0–2. The existing test asserted the wrong shape, pinning the bug. |
| 3 | `RadianceSceneCutSplit` — `json.loads("")` | `cut_data` is a STRING widget with no default. Drop the node, queue before wiring the detector, get a bare `JSONDecodeError` traceback. |
| 4 | `torch.quantile` 2²⁴ cap in multipass | `reshape(B,-1)` gives H·W·3 per row; UHD is 24,883,200. `quantile() input tensor is too large` took the Multipass Master node down on **every 4K plate**. Verified fixed: 4K now completes. |
| 5 | `_FACE_MODEL_CACHE[key]` on a cache with no `__getitem__` | Survived the dict→`GPUModelCache` refactor. The `TypeError` was swallowed by a broad `except Exception` logging at DEBUG, so RetinaFace detection **silently never ran** — face restore fell through to the Haar cascade and still reported "Faces detected: 0" as success. |
| 6 | Shipped workflow fails ComfyUI validation | `peak_nits` became a string combo; the bundled template still carried the JSON number `1000`. `validate_inputs` does `if val not in type_input`, and `1000 != "1000"` — **the shipped workflow could not be queued at all.** |

The harness also blocks network access during tests, after `RadianceUpscaleImage`
downloaded a 67 MB Real-ESRGAN checkpoint on its first functional run.

---

## 2 · Open — must fix before release

Ordered by what a user loses. Every one carries measured evidence.

### Delivery produces a master that is not what was approved

**`js/radiance_viewer.js:3997` vs `delivery/handler.py:300-305`** — the `/radiance/deliver`
payload sends 11 grading keys. The handler reads **six more that are never sent** and falls
back to identity: `shadows`, `highlights`, `hue_shift`, `lut_name`, `lut_intensity`,
`gamut_compression`. `tint` is sent and never read.

All are live viewer controls driving real shader uniforms. A colourist sets Shadows −0.60,
Highlights +0.35, Hue +12°, presses RENDER — the delivered ProRes has none of it, the
endpoint returns 200, the HUD prints `EXPORT COMPLETE`, and the JS comment two lines above
the payload reads *"must match viewer for what-you-see = what-you-export"*. Invisible until
someone A/Bs the master.

### Colour output is wrong on three delivery paths

| Path | Measured | Should be |
|---|---|---|
| **ACES 2.0 Cinema (DCI)** `hdr/color.py:1349,1474,1502` | Hard-clips at ~0.4 scene; 0.40→0.75403, 0.50→0.75405, 8.0→0.75405. Peak white 0.754 ≈ **27 nits** | 48 nits, no clip |
| **ACES 2.0 HDR (HLG)** `hdr/color.py:1475,1499` | `peak_nits` defaults to 100 for HLG, so 18% grey → **127.5 nits**, diffuse white → **1000 nits** | ~26 and ~203 (BT.2408) — **~5× over** |
| **AgX** `hdr/tonemap.py:15-20,346` | Inset matrix applied untransposed. Neutral 0.18 → R/G/B spread **0.0420**; at 16.0 → **0.1155**. Visible warm cast on every grey | spread ~0.0001 |
| **"Rec.2020" output** `nodes_io.py:274-276` | `arr ** (1/2.4)` with a hard clip — no primaries matrix, not the BT.2020 transfer. 4.0 → 1.0, highlights gone | a real 709→2020 conversion |

### Alpha is corrupted in three places

Tone mapping (`hdr/tonemap.py:397,417,438`), expansion (`:100,159`) and the delivery grade
(`color/grading.py:182,251,262`) all operate on the full array with no `[..., :3]` slice.
Measured: a 50% matte comes back as **0.607** through Reinhard+γ2.2, **0.214** through
expand, and **1.059** (over 1.0) through the delivery grade with exposure+contrast. Full
opaque becomes 73% transparent.

### Three "advanced" features do nothing

- **PAG** `sampler_utils.py:848-850` — guards on `extra_options["block_type"]`, a key
  ComfyUI never sets (it sets `"block"`). The patch returns `q,k,v` unmodified on every
  call while logging *"PAG applied with scale …"*. `pag_scale` is never read, so 0.1 and
  5.0 are bit-identical. The perturbation isn't PAG either.
- **Restart sampling** `nodes_sampler.py:706-729` — runs *after* the schedule finishes,
  passes the noised latent as `noise=` so ComfyUI's internal `σ₀·noise + latent` amplifies
  the signal by `(1+σ₀)`, and the sub-schedule never returns to σ=0. Measured: output left
  at σ=0.79 at `restart_sigma=2.0`.
- **`blend_mode`** `nodes/upscale/upscale.py:486` — the parameter appears **zero times** in
  the function body. The info string still reports the mode the user selected.

Also dead: `chromatic_adaptation` on ColorSpaceConvert (`hdr/color.py:719` —
`"Bradford"` and `"None"` return bit-identical tensors), and the `RELOAD` button on
`RadianceRead` (declared as a hidden input, so ComfyUI never passes it and the JS bails
before adding the button).

### Upscaling is unusable at production sizes

- **All built-in upscalers run on CPU** (`upscale.py:510,1417,1726,1978` use
  `images.device`, and ComfyUI IMAGE tensors are CPU-resident). One 4K plate at 4× ≈ 66
  TFLOP → **~10 min/frame** vs ~5 s on GPU. The sibling `image/upscale.py:2459` does it
  correctly.
- **`overlap_temporal=1` produces black frames.** The sine ramp is exactly 0 at i=0, and
  `step = window−1` makes the ramp-down and ramp-up land on the *same* frame. Measured:
  `B=100, window=16, overlap=1` → frames 15, 30, 45, 60, 75, 90 are **fully black**. 1 is
  the widget minimum and the tooltip recommends it.
- **Tier-3 diffusion ignores `scale`.** The backend is fixed 4×; at `scale=2` the result is
  cropped to the top-left quarter. Measured correlation with the true 2× result: **0.0024**.
  `mode="creative"` force-selects this tier regardless of the scale widget.
- **49 GB preallocation** for 100 frames of 1080p at 4×, always on CPU (`:2011-2013`).

### The sampler mutates the caller's model

`nodes_sampler.py:1026,1073` — the clone guards assume an earlier block already cloned, but
the first clone only runs when `guidance_rescale_phi > 0 **and cfg > 1.0**`. Flux defaults
`cfg=1.0`, so a user following the tooltip's own recommendation patches the loader's cached
`ModelPatcher` in place. The patch persists into every later queue and every other branch
fed from that MODEL.

Separately (`:1031,1078`): `getattr(model, "model_sampler_cfg_function", None)` is always
`None` — ComfyUI stores it in `model_options["sampler_cfg_function"]` — so the second patch
silently **overwrites** the first. With rescale + SDR reference both on, rescale is gone
while the log still says it was applied.

### Scene-cut detection reports cuts in cut-free footage

`nodes/ai/scene_cut.py:104-116` — `scores / scores.max()` forces the largest inter-frame
distance in *any* clip to 1.0, which always exceeds the threshold. Measured on a smooth pan
with no cut: detected `[0, 12, 24, 36]` — exact multiples of `min_shot_frames`, the
signature of thresholded noise. On footage with one real cut at frame 24, the real cut is
indistinguishable from the false ones.

### The event loop blocks on user data

`/radiance/projects/dashboard` walks the entire ComfyUI output tree with a `stat()` per
entry and fully unzips every `.rad` — synchronously, on aiohttp's loop. A studio `output/`
with 200k frames freezes the websocket for tens of seconds: progress bar, node
highlighting, queue view and `/prompt` all stall mid-render. Same shape in
`/radiance/assets/thumb` (decodes 4K EXRs and opens videos inline) and `/radiance/ocio/bake`
(~275k-iteration Python loop, 1–2 s per uncached display/view).

`/radiance/ocio/load` accepts an arbitrary absolute path with no containment — the comment
says "Security: resolve and validate path" but only `abspath` + `isfile` happen. On an
unauthenticated route this is a file-existence oracle for any path on the host.

---

## 3 · Open — should fix, not blocking

- **S-Log3 toe is shifted** (`color/transfer.py:145`). Encode/decode are mutually
  consistent so round-trip tests pass, but black decodes to **−0.010** instead of 0.0, and
  0.130 CV decodes 9.5× crushed. `color/luts.py:194` already has the correct curve — the
  two implementations in-tree disagree.
- **ColorSpaceConvert silently skips the primaries transform** for ACES2065-1, DCI-P3 and
  Display-P3 — 3 of 12 advertised spaces return the Rec.709 result byte-identically.
- **Reach gamut compression omits the `s` normalisation** — a colour at the reach limit
  lands at 1.06–1.12 instead of exactly 1.0, so it is not brought to the gamut boundary.
- **Optical flow claims DIS, is single-scale Lucas–Kanade.** Measured recovery: 1 px →
  104%, 3 px → 17%, 5 px → **1%**. The Fast/Medium/Ultra preset changes the answer by <1%.
  Mask propagation is effectively static above ~2 px of motion.
- **Direct-pixel recovery applies a BT.2020→709 matrix to data that is already 709** —
  measured +24% red on skin tones.
- **EXR data window ignored** (`nodes_io.py:495`); an overscan render comes back shifted and
  at the wrong resolution with no warning. Alpha association is not handled on
  read or write, so round-tripping a matte changes edge pixels.
- **Directory/glob sequence patterns slice by list index using `start_frame`** — with the
  default `1001`, any directory with fewer than 1001 files yields an empty list.
- **`RadianceVideoAssembler` output is not a function of its inputs** — a class-global
  accumulator keyed on a widget defaulting to the same literal for every instance.
- **`_out_path` siblings:** version-backup eviction sorts lexically (`.v10 < .v2`), so it
  deletes the *middle* of the history rather than the oldest.
- **`radiance_sessions.json`** is a lock-free read-modify-write straight to the final path;
  two concurrent deliveries lose one, and a crash mid-write discards the whole history.
- **Unbounded GPU model dict** in `nodes/vfx/depth.py:26` missed the cache sweep — all three
  tiers pinned is ~1.83 GB for the process lifetime.
- **`fast_vae.py:452` tiled decode pastes tiles with no blending** — a hard 1-px grid on
  every `tile_size` boundary.

---

## 4 · What came back clean

Worth as much as the findings, and none of it was assumed — all of it was measured.

**Node contract, all 109:** every node has `CATEGORY`; every `FUNCTION` names a method that
exists; `RETURN_NAMES`, `OUTPUT_TOOLTIPS` and `OUTPUT_IS_LIST` lengths all match
`RETURN_TYPES`; every display name unique. Zero malformed `INPUT_TYPES` — no `min > max`,
no zero steps, no defaults outside range, no empty combos, no key collisions. Every
`FUNCTION` accepts every declared input, bind-tested both with and without optionals.
Slowest `INPUT_TYPES` at menu-build time is 1.6 ms.

**In-place mutation:** an AST scan of all 109 functions plus the runtime check in the new
harness found **zero** mutations of IMAGE/LATENT/CONDITIONING/MODEL inputs.

**Routes:** 37 across 4 modules, all behind idempotency guards, no duplicate `(method, path)`.

**Saved-workflow compatibility:** widget lists reconstructed and diffed positionally against
both shipped workflows — no reorder, no mid-list insertion, no changed combo default. The
one exception is fixed above.

**Transfer functions:** round-trip error over x ∈ [−0.02, 64] — sRGB 7.6e−6, LogC3 1.1e−5,
LogC4 2.7e−5, V-Log 1.1e−5, C-Log3 1.9e−5, Log3G10 3.1e−5, ACEScct 7.2e−7, DaVinci
Intermediate 3.1e−5. All monotonic, all anchors correct. PQ and HLG constants exact.

**Algorithms verified against their papers:** CFG rescale (Lin et al. 2023) max |Δ| 1.9e−6;
Flux/SD3 timestep shift 6.0e−8; Reinhard simple and extended **exact**; Uncharted 2/Hable
1.2e−7; Narkowicz ACES fit **exact**, and its analytic inverse round-trips to 1.1e−6.

**The confidence-gated temporal fallback is sound** — bit-for-bit identity outside the mask,
correct collapse to the deterministic path as confidence → 0, residual bounded and masked.

**Batch 0–3 fixes all hold:** every `torch.load` is `weights_only=True`; every checkpoint
load is `strict=True` with a real fallback; `core/tiling.py` ramps are an exact partition of
unity with correct border handling; `@torch.no_grad` present on every inference entry point;
the Nuke bridge's eval/exec really is gone.

---

## 5 · Ship criteria

Against the brief's acceptance bar:

| Criterion | Status |
|---|---|
| Every node passes functionally or skips with a reason | **Met** — 79 / 30 / 0 |
| No new test failures vs baseline | **Met** — 1891 pass, failure set identical both directions |
| Ruff at parity | **Met** — one warning removed, none added |
| Wheel contains nothing it shouldn't | **Met** — 245 files, no tests/caches/docs/local workflows |
| No unfixed HIGH finding | **NOT MET** — 22 open |

**Recommendation: fix the delivery grade mismatch, the three colour output paths, the alpha
handling, and the shipped-workflow validation before any public release.** Those five are
the ones where a user gets a wrong deliverable and has no way to know. The dead features
(PAG, restart, blend_mode, chromatic_adaptation) should either be fixed or removed from the
UI — shipping a control that does nothing is worse than not shipping it. The CPU upscaler
and the event-loop blocking are quality-of-life but will dominate first impressions.

Nothing here is architectural. Every one of these is a contained fix in a file that already
has a test harness pointed at it.
