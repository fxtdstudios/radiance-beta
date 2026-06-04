# Pre-Release Engineering and VFX Review Report

**Project:** Radiance — Professional VFX / HDR / Color ComfyUI node pack
**Version reviewed:** 3.1.0 (`pyproject.toml`)
**Scope:** ~80,200 lines of Python across 250 files, 23 JS files, 43 test files.
**Method:** Static source review + structural analysis. The package could **not** be imported live (ComfyUI, torch, OpenEXR, OCIO not present in the review sandbox), so runtime claims below are flagged where they could not be executed.

> Review stance: strict, release-oriented, "this ships to client shots next week." Findings are tied to specific files and lines so they can be fixed and re-tested.

---

## 1. Executive Summary

Radiance is an ambitious, broad package with a genuinely clean **node-registration architecture** (declarative catalog + defensive loader + branding normalizer). That part is well engineered. However, the tool is **not safe to release** in its current state, and the reasons sit squarely in its core value proposition — 32-bit / HDR / EXR fidelity.

Three findings alone are release-blocking on their own:

1. **The image-to-tensor path silently destroys HDR data.** `_np_to_tensor` divides any array whose max exceeds 2.0 by 255 (or 65535). The EXR reader routes through this function, so any scene-linear plate with a highlight above 2.0 — i.e. essentially every real VFX plate — is silently crushed ~255×. For a tool sold on "32-bit float / HDR / ACES," this is a correctness failure in the first node an artist touches.
2. **The EXR writer can silently produce a 0-byte file** and report success, and it **drops the alpha channel** entirely.
3. **CI provides false confidence.** Of ~43 test files, the CI and the publish gate run only two, with torch stubbed by `MagicMock`, `--cov-fail-under=0`, and OpenEXR/OCIO/colour-science/transformers never installed. One CI job imports a module that does not exist, so CI is red regardless.

Layered on top are an arbitrary-code-execution risk in checkpoint loading (`torch.load` without `weights_only=True`), a bypassable denylist "sandbox" in the DCC bridge, an unfinished module migration (root `nodes_*.py` vs `nodes/` package), and ~57 silent `except` swallows.

The skeleton is good. The substance that VFX artists depend on — predictable, lossless color/HDR/EXR round-trips and trustworthy delivery — is not proven and, in the EXR path, is provably wrong.

---

## 2. Release Readiness Decision

### Status: **Internal testing only**

Not beta. A beta exposes the EXR/HDR pipeline to real artists, and the first scene-linear plate they load will be silently crushed (Finding C-1), while the first EXR they deliver may be empty or alpha-less (C-2, C-3). Those are exactly the operations a VFX user performs on minute one, and they fail silently — the worst failure mode, because the artist ships corrupted frames without knowing.

It can move to **beta** once the Critical and High blockers in Section 3 are fixed and the real test suite (not the mocked 2-file subset) runs green against actual OpenEXR/torch/OCIO.

It can be considered for **public release** only after that, plus a verified live ComfyUI import on Windows + Linux (the README itself admits this has not been done).

---

## 3. Critical Blocking Issues

### C-1 — EXR / HDR load silently divides scene-linear data by 255
- **Severity:** Critical
- **Area:** VFX Quality / Code
- **Problem:** `_np_to_tensor` (`nodes_io.py:215-221`) contains `if arr.max() > 2.0: arr = arr / (255.0 if arr.max() <= 255 else 65535.0)`. `_read_exr_single` returns `_np_to_tensor(arr)` (`nodes_io.py:362`), and the `.hdr` reader does the same (`:328`). Any HDR/linear image whose brightest pixel exceeds 2.0 — sun, specular, practical lights, exposed highlights — is divided by 255 (or 65535).
- **Why it matters:** This is the headline feature of the product. A Read → Write round-trip of a normal EXR plate is not lossless; it is off by a factor of 255. Grades, ACES transforms, tone-mapping, and delivery downstream all operate on corrupted values. It is silent — no error, no warning.
- **Recommended fix:** Remove the magnitude heuristic entirely. Normalization must be driven by the **source dtype/format**, not pixel statistics: integer formats divide by their type max; float formats (EXR/HDR/TIFF-float) pass through untouched. Decide normalization in the reader that knows the format, not in a shared tensor helper.
- **Test after fix:** Round-trip an EXR with known values up to 50.0 through Read→Write and assert bit-exact (float32) / within-half-ulp (float16) equality. Add a unit test asserting `_read_exr_single` preserves values > 2.0.

### C-2 — EXR writer silently writes a 0-byte file on failure
- **Severity:** Critical
- **Area:** VFX Quality / Code
- **Problem:** `_save_exr` (`nodes_io.py:664-701`) falls back OpenEXR → cv2 → cv2-as-8-bit → `with open(path, "wb") as f: f.write(b"")`. The final fallback creates an **empty file** and returns normally. Intermediate `except Exception: pass` blocks hide the real cause.
- **Why it matters:** A delivery/write node can report success while producing an empty or non-EXR `.exr`. In production this means missing or corrupt frames discovered only at review or at the client — the single most damaging outcome for a delivery tool.
- **Recommended fix:** Remove the empty-file and silent fallbacks. If OpenEXR and the cv2 EXR path both fail, raise a clear exception naming the path and the missing dependency. Never substitute a different bit-depth/format behind the caller's back.
- **Test after fix:** Mock OpenEXR + cv2 failure and assert a raised exception (not a 0-byte file). Assert written files are non-empty and re-readable with matching dimensions/dtype.

### C-3 — EXR writer drops alpha and crashes on single-channel input
- **Severity:** Critical
- **Area:** VFX Quality / Code
- **Problem:** `_save_exr` writes only R/G/B (`nodes_io.py:672-681`). Alpha is never written. It also assumes a 3-channel `(H,W,3)` array: a 2-D mask `(H,W)` would `IndexError` on `arr_f32[...,1]`, and RGBA input loses channel 4.
- **Why it matters:** Alpha is non-negotiable in compositing — holdouts, premult, mattes, despill. EXRs without alpha are unusable for comp handoff. Single-channel matte export crashes.
- **Recommended fix:** Detect channel count; write A when present; support 1-channel (luminance/`Y` or `A`) and RGBA. Define and document the channel contract.
- **Test after fix:** Save/reload RGBA EXR and assert alpha preserved; save a single-channel matte and assert it round-trips.

### C-4 — `torch.load` without `weights_only=True` (arbitrary code execution)
- **Severity:** High
- **Area:** Code / Security
- **Problem:** Checkpoints are loaded with unrestricted unpickling in `model/vae.py:338`, `fast_vae.py:398`, `nodes/vfx/multipass/core.py:598` and `:705`, and `scripts/training/dataset_hdr.py:494`. `nodes/upscale/upscale.py` correctly uses `weights_only=True` (lines 529, 647, 2115) — so the codebase is inconsistent.
- **Why it matters:** Artists routinely download model weights from the internet. A malicious `.ckpt`/`.pth` executes arbitrary code on load. This is a known, exploited class of attack in the diffusion ecosystem.
- **Recommended fix:** Add `weights_only=True` to every `torch.load`. Prefer `safetensors` where possible (the code already supports it for the `.safetensors` branch).
- **Test after fix:** Grep guard in CI: fail if any `torch.load(` lacks `weights_only=True`.

### C-5 — CI / publish gate does not test the real code
- **Severity:** High
- **Area:** Code / Compatibility (release process)
- **Problem:** `.github/workflows/ci.yml` and `publish.yml` run only `tests/test_nodes_registry.py` and `tests/test_node_smoke.py`, with `--cov-fail-under=0`. torch is replaced by `MagicMock` (`tests/conftest.py`, `tests/torch_mock.py`); OpenEXR, opencolorio, colour-science, transformers, and real torch are never installed. The remaining ~40 test files (EXR round-trip, color math, ACES, sampler regression, HDR pipeline) are **never executed in CI**. Additionally the `smoke` job imports `radiance.nodes_temporal_coherence`, **which does not exist**, so that job fails.
- **Why it matters:** The publish gate green-lights releases without exercising any real tensor math, color science, or EXR I/O — exactly where the Critical bugs above live. CI is simultaneously red (missing module) and meaningless (mocked).
- **Recommended fix:** Fix or remove the missing-module import. Install real torch (CPU) + OpenEXR + OCIO + colour-science in at least one CI lane and run the full `tests/` directory. Raise `--cov-fail-under` to a real baseline. Make the publish gate run the full suite.
- **Test after fix:** CI green with `pytest tests/` (full), real deps, on Ubuntu + Windows.

---

## 4. Potential Bugs and Fragile Areas

- **8-bit vs 16-bit mis-detection (`nodes_io.py:336`).** `maxv = 65535.0 if pil.mode.endswith("16") or np.array(pil).max() > 255 else 255.0`. A genuine 16-bit image whose actual max value is ≤ 255 (a dark/low-key plate) is treated as 8-bit and divided by 255 → 256× too dark. It also decodes the PIL image a second time (`np.array(pil)`), doubling load cost. *Repro:* save a 16-bit PNG with all values < 256, load it.
- **`cv2.imread` results not None-checked (`nodes_io.py:326-327`, `369-370`).** On an unreadable/oversized/permission-denied file, `cv2.imread` returns `None`; `cv2.cvtColor(None, …)` throws an opaque error with no path context. *Repro:* point the reader at a truncated EXR/HDR.
- **EXR reader assumes literal R/G/B channels (`nodes_io.py:354-356`).** Luminance-only EXRs, data passes, or layered/AOV EXRs (`diffuse.R`, `N.X`, etc.) have no plain `R/G/B` and will fall through to the cv2 fallback, which may silently load wrong data or fail. The reader also uses `dataWindow` size but ignores the `dataWindow` vs `displayWindow` offset, so **overscan plates load misaligned**. *Repro:* load a multi-part or overscan EXR from any renderer.
- **EXR pixel type forced to FLOAT on read (`nodes_io.py:353`).** Acceptable (OpenEXR converts HALF→FLOAT), but undocumented and worth an explicit comment/test.
- **~57 silent exception swallows.** 7 bare `except:` and ~50 `except Exception: pass`/`pass`-style handlers across the tree. Each is a place where a real failure (missing dep, bad tensor, I/O error) becomes a silent no-op or wrong-but-quiet result. These need an audit pass; at minimum log at WARNING with context.
- **DCC bridge denylist is bypassable (`nodes/pipeline/dcc.py:31-75`).** `_validate` does substring matching on `code.lower()` then `eval`/`exec` with a restricted-builtins dict. Denylist sandboxing of Python is not robust. It is mitigated by the `RADIANCE_DEV=1` gate and `127.0.0.1` bind, but it must not be described or trusted as a "sandbox." `scripts/start_nuke_server.py:74-75,352-362` has the same pattern.
- **Unfinished module migration.** Some root modules are deprecation shims that warn on import (`nodes_cdl.py`, `nodes_grade.py`), but the `nodes/` package still imports the **live** implementations from other root modules — `radiance.nodes_sampler`, `nodes_io`, `nodes_loader`, `nodes_workspace`, `nodes_realtime_preview`, `nodes_gizmo` (`nodes/generate/__init__.py`, `nodes/io/__init__.py`, `nodes/monitor/__init__.py`, `nodes/pipeline/*`). So "root vs package" is half-migrated: some files are dead shims, others are the source of truth. This is a maintenance trap and a source of duplicate-mapping overrides (logged at DEBUG in `registry._merge_module_mappings`).
- **`_save_pil_image` 16-bit RGB path is partly dead (`nodes_io.py:635-645`).** Builds a `pil` object via `fromarray(..., mode="I;16")` that is never used, then relies on `tifffile`; the no-tifffile fallback silently downgrades to 8-bit (precision loss, no warning).
- **Video codecs assume a full ffmpeg build (`nodes_io.py:716-723`).** ProRes (`prores_ks`), DNxHR (`dnxhd`→`.mxf`), x265 10-bit require encoders not present in every ffmpeg. `imageio-ffmpeg` is listed in the core/Windows requirements but **dropped from `requirements_linux.txt` and `requirements_mac_silicon.txt`**, while `_save_video_ffmpeg` needs ffmpeg on disk. Linux/mac video export may fail at runtime.
- **`shutil.copy2` to an arbitrary destination (`nodes_io.py:1360-1362`).** "Send to DCC folder" copies to a user-supplied `remote_path` with no validation. User-driven, but worth a path sanity check and clear errors.

---

## 5. Code Quality Review

- **Architecture:** The registration layer (`nodes/registry.py`, `nodes/catalog.py`, `nodes/branding.py`, `config/*`) is the strongest part — declarative catalog, feature-flagged groups, defensive import with per-module failure capture, and centralized branding/category assignment. Good separation of concerns. Below that line, quality is uneven and the root-vs-package duplication undermines it.
- **Maintainability:** Hurt by the half-finished migration and by very large files (`hdr/vae.py` 3,141 lines; `image/upscale.py` 2,691; `nodes/upscale/upscale.py` 2,613; `nodes_sampler.py` 1,539; `sampler_utils.py` 1,845). These should be decomposed; they are hard to review and to test in isolation.
- **Naming:** Mostly clear and consistent. Branding/menu taxonomy is thoughtful.
- **Error handling:** The weakest dimension. ~57 silent swallows, the empty-EXR fallback (C-2), and the `>2.0` heuristic (C-1) all prefer "don't crash" over "be correct," which is exactly backwards for a delivery tool. Exceptions should be specific and should name paths/inputs.
- **Modularity:** Good at the node-group level; poor inside the large monolith files.
- **Dependency handling:** Runtime dep validation exists (`config/dependencies.py`), and torch is correctly assumed from ComfyUI rather than installed. But `requirements.txt` (no upper bounds, includes `aiohttp`/`tqdm`) drifts from the platform requirement files and from `pyproject` deps; the three platform files disagree on `scipy`, `imageio-ffmpeg`, and `Imath` pinning.
- **Performance / GPU:** No hardcoded `.cuda()`/`device="cuda"` found in node code — good. But device/dtype/VRAM management could not be validated live, and the loaders are memory-eager (see Section 8).
- **Cross-platform safety:** Paths mostly use `pathlib`/`folder_paths`. Main risks are the codec/ffmpeg availability gaps on Linux/mac and the unverified live import on Windows.

---

## 6. ComfyUI Node Review

- **Node registration:** Solid. `__init__.py` bootstraps the package context, validates deps, and merges mappings through a single defensive loader. Optional groups fail soft; the required `.nodes` group fails hard. `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` / `WEB_DIRECTORY` (`./js`) are exported correctly.
- **Input/output types:** ~87 `FUNCTION` declarations and `RETURN_TYPES` across the catalog. These could not be validated against a live ComfyUI runtime in this review (torch/comfy absent), so type correctness and tensor-shape contracts (`(1,H,W,C)` from `_np_to_tensor`) are **unverified** — treat as a risk until a live import test runs.
- **UI parameters:** Branding normalizes display names and categories at load (`branding.py`), giving a coherent `FXTD STUDIOS/Radiance/...` menu. Note `_set_node_category` **mutates `CATEGORY` on the class at import time**; for any class shared with another pack this is a cross-pack side effect (it is wrapped in try/except, but it still reassigns categories of whatever ends up in the mapping).
- **Workflow usability:** Cannot be assessed without running ComfyUI; the docs describe sensible workflows.
- **ComfyUI Manager / Registry compatibility:** `[tool.comfy]` metadata, `requires-comfyui = ">=0.2.2"`, `.comfyignore`, and a publish workflow are present and look correct. The `"100+ nodes"` claim could not be exactly verified without a live import; the README itself notes the full import test has not been run.
- **Failure modes inside ComfyUI:** Most likely failure is at node *execution*, not load: the EXR/HDR bugs (Section 3) will produce wrong output rather than crashes, and the `cv2.imread` None cases (Section 4) will throw opaque errors mid-graph.

---

## 7. VFX Artist and Supervisor Evaluation

From a supervisor's chair, the headline problem is **trust**. The pieces that must be boringly reliable are not:

- **Output quality / color / HDR / EXR correctness:** Fails. The load-time `>2.0` divide (C-1) means HDR/linear values are silently wrong; EXR writes drop alpha (C-3) and can be empty (C-2). A Read→grade→Write round-trip is not predictable or repeatable on real plates. ACES/OCIO modules exist and are extensively unit-tested *in the repo*, but those tests don't run in CI against real libraries, so correctness is asserted, not demonstrated.
- **Edge quality / mattes:** Single-channel matte EXR export crashes (C-3); alpha is not written, so comp holdouts/premult break.
- **Temporal stability:** Not assessable from static review; the sequence path loads frames independently with no temporal handling visible in I/O.
- **Artist control / repeatability:** The silent heuristics (normalize-if-max>2.0, downgrade-to-8-bit-if-no-tifffile, empty-file-on-failure) are the opposite of what a TD wants — results change based on pixel content and installed libraries, with no log.
- **Shot workflow readiness:** Overscan EXRs load misaligned (dataWindow/displayWindow ignored); layered/AOV EXRs aren't handled; Linux/mac video delivery may fail on missing codecs.
- **What an artist would complain about, day one:** "My EXR came back dark / washed out." "My delivered EXR has no alpha." "Resolve got an empty .exr." "It worked on my Windows box but not on the Linux farm." Each maps to a concrete finding above.

The breadth (plate prep, depth, optics, multipass, DCC handoff) is genuinely attractive. But breadth on top of an unreliable color/EXR foundation is a liability, not a selling point.

---

## 8. Performance and Memory Review

- **Load path is memory-eager and double-decodes.** `_read_image` calls `np.array(pil)` twice for the 16-bit heuristic (`nodes_io.py:335-336`). EXR read builds three separate channel buffers then `np.stack` (extra copies). Sequences/batches load every frame fully into RAM with no streaming — large EXR sequences (4K+, deep frame counts) will exhaust host RAM.
- **VRAM:** No hardcoded device placement found (good), but device/dtype handling in the large sampler/VAE/upscale files (1,500–3,100 lines each) could not be exercised here. fp16/fp32 behavior and GPU→CPU fallback are **unverified** and should be a focused test pass.
- **Bottlenecks/optimization:** Remove the double decode; stream sequences; avoid per-frame PNG temp dumps in `_save_video_ffmpeg` (`nodes_io.py:729-737`) for large clips (disk + time cost); reuse contiguous buffers in EXR read/write.

(Performance numbers were not measured — no GPU/real torch in the review environment. These are structural observations, to be confirmed with the stress tests in Section 11.)

---

## 9. Documentation and User Experience Review

- **Strengths:** Extensive — `README.md`, `NODES.md` (42 KB), `docs/` site with quickstart/concepts/workflows/troubleshooting/developer/coverage, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_STYLE.md`, release notes. Install instructions cover Manager/Registry/manual and per-platform requirements. DCC security defaults are documented.
- **Gaps / risks:**
  - The README **Release Status** table effectively admits the product is unverified: "Local full pytest run — Not run," "Full ComfyUI import test — Must be run … before tagging." Shipping against that is shipping blind.
  - **Known limitations are understated.** No mention that EXR write currently omits alpha, that overscan/layered EXRs aren't supported, or that Linux/mac video export needs specific ffmpeg codecs.
  - **External AI egress is undocumented.** The `[full]` extra pulls `anthropic`, `openai`, `google-generativeai`, and several nodes make outbound calls (`nodes/pipeline/audio.py`, `nodes/video/character.py`, etc.). Artists and studios need to know data leaves the machine, which keys are required, and how to disable it. This is both a privacy and a compliance issue for client work.
  - `"100+ nodes"` is a marketing claim not verified by a live import in this review.

---

## 10. Security and Release Safety

- **Arbitrary code execution via `torch.load`** (C-4) — High. Untrusted checkpoints execute code on load in five locations.
- **Dynamic Python over a TCP socket** (`nodes/pipeline/dcc.py`, `scripts/start_nuke_server.py`) — denylist validation + `eval`/`exec`. Mitigated by `RADIANCE_DEV=1` opt-in and `127.0.0.1` binding, but the validation is bypassable and should never be described as a sandbox. Document it as "developer-only, local, insecure-if-enabled."
- **Silent empty-file writes** (C-2) — data-integrity risk for delivery.
- **External network/data egress** via AI provider SDKs — undocumented; license + privacy exposure. `transformers` and `aiohttp` are also hard runtime deps that broaden the attack/maintenance surface.
- **License:** GPL-3.0 (`LICENSE`, `pyproject` classifier). Note the **license compatibility question**: distributing/operating GPL-3.0 alongside optional proprietary AI SDKs (OpenAI/Anthropic/Google) and other deps should be reviewed by whoever owns licensing, especially if any models/weights ship with the pack.
- **Path handling:** `shutil.copy2` to a user-supplied destination (`nodes_io.py:1360`) — validate.
- **`defusedxml` is a dependency** — good, indicates XML inputs are parsed safely (e.g. CDL/OCIO); confirm it's actually used everywhere XML is read.

---

## 11. Test Plan Before Release

**Make CI real first (blocks everything else):**
- Add a CI lane that installs real CPU torch + OpenEXR + Imath + opencolorio + colour-science and runs the **entire** `tests/` directory on Ubuntu and Windows. Fix/remove the `nodes_temporal_coherence` import. Raise `--cov-fail-under` to a meaningful baseline. Make the publish gate run the full suite.

**Unit tests (add/strengthen):**
- `_np_to_tensor` / readers: integer formats normalize correctly; float formats (EXR/HDR/float-TIFF) pass through values > 2.0 unchanged (locks C-1).
- `_save_exr`: RGBA round-trip preserves alpha; single-channel matte round-trips; failure raises (no 0-byte file) (locks C-2, C-3).
- 16-bit-with-low-max PNG loads at correct brightness (locks the Section 4 heuristic bug).
- Grep guard: no `torch.load(` without `weights_only=True` (locks C-4).

**Integration tests:**
- Read → Grade/ACES/OCIO → Write EXR round-trip, assert numerical fidelity within tolerance.
- Overscan EXR (dataWindow ≠ displayWindow) and layered/AOV EXR load correctly or fail loudly.

**ComfyUI node-loading tests:**
- Live ComfyUI import on Windows + Linux; assert every node in `NODE_CLASS_MAPPINGS` instantiates and `INPUT_TYPES()`/`RETURN_TYPES` are valid; assert no unintended `CATEGORY` mutation of foreign nodes.

**Workflow tests:** Run the bundled `workflows/*.json` end-to-end in a real ComfyUI.

**GPU/CPU fallback tests:** Force CPU; force CUDA; assert no device-mismatch errors and correct fp16/fp32 behavior in sampler/VAE/upscale.

**Platform tests:** Windows + Linux + macOS (Apple Silicon) install from each `requirements_*.txt`; verify video export codecs actually exist or degrade with a clear error.

**Artist usability tests:** Have a compositor run a real shot (plate in → grade → deliver EXR with alpha → open in Nuke/Resolve) and confirm visual + numerical match.

**VFX image-quality tests:** Known-value EXR fixtures; assert no color/exposure drift through the pipeline; verify scopes/QC against reference.

**Stress tests:** 8K EXR; 500-frame EXR sequence; large batch — monitor RAM/VRAM for the memory-eager load path (Section 8).

**Regression tests:** Snapshot node list + display names + categories; pin sampler outputs (`test_sampler_regression.py` already exists — wire it into CI).

---

## 12. Recommended Fix Roadmap

**Must fix before release (blockers):**
- C-1 EXR/HDR load divide-by-255 heuristic.
- C-2 empty-file EXR fallback.
- C-3 alpha + single-channel EXR write.
- C-4 `torch.load(weights_only=True)` everywhere.
- C-5 real CI + fix the missing-module CI failure; full suite gates publish.
- Verify a live ComfyUI import on Windows + Linux.

**Should fix for beta:**
- 16-bit/8-bit detection and double-decode.
- `cv2.imread` None checks with path context.
- Overscan + layered/AOV EXR handling.
- Audit the ~57 silent `except` swallows; log with context.
- Document external AI egress + required keys + opt-out; document EXR/codec limitations.
- Reconcile `requirements*.txt` ↔ `pyproject` deps (ffmpeg on Linux/mac).
- Re-label the DCC bridge as developer-only/insecure-when-enabled.

**Can improve after release:**
- Finish the root→package migration; delete dead shims; split the 1.5k–3k-line files.
- Stream sequences; reduce buffer copies; avoid per-frame PNG temp dumps for video.
- Raise coverage baseline progressively.

**Future professional features:**
- Full multi-part / deep EXR support; explicit display/data window handling.
- Live DaVinci Resolve API push (currently folder handoff only).
- Color-managed viewer round-trip validation; OCIO config bundling guarantees.
- Per-node deterministic/seeded modes for shot repeatability.

---

## 13. Final Verdict

The engineering *scaffolding* — node registry, catalog, branding, dependency validation, docs breadth — is above average for a ComfyUI pack and shows real care. But Radiance is marketed and built as a professional HDR/EXR/color tool, and its core image I/O **silently corrupts HDR data on load, can deliver empty or alpha-less EXRs, and is not protected by meaningful CI.** Those are not polish issues; they are correctness and data-integrity failures in the first operations an artist performs. A supervisor putting this on a client shot next week would get burned silently — the worst kind.

**Verdict: Internal testing only.** Fix the five Critical/High blockers, make CI exercise the real libraries, and verify a live ComfyUI import — then it is a credible beta. It should not be advertised as production-ready until a real EXR/HDR round-trip is demonstrably lossless on actual plates.

| Area | Score / 10 | Risk Level | Notes |
| ---------------------- | ---------: | ---------- | ----- |
| Code Quality | 5 | Medium | Clean registration layer; half-finished migration, huge files, ~57 silent excepts |
| Stability | 3 | High | Silent data-corruption fallbacks; opaque errors on bad I/O |
| ComfyUI Compatibility | 6 | Medium | Solid registration; live import unverified; runtime CATEGORY mutation |
| VFX Production Quality | 2 | Critical | HDR load divide-by-255, alpha drop, empty-EXR delivery |
| Performance | 5 | Medium | Memory-eager loads, double decode; GPU/VRAM unverified |
| Documentation | 6 | Medium | Broad and good, but overstates readiness; AI egress + EXR limits undocumented |
| Release Readiness | 2 | Critical | Internal testing only; CI is red and mocked; blockers open |

---

*Findings are based on static review of the source at version 3.1.0. Items marked "unverified" require a live ComfyUI + GPU environment to confirm and should be treated as risks until then.*

---

## Addendum — Post-Review Fixes Applied (this session)

**Critical new finding (not in the original review):** the working tree had **5 Python files saved truncated mid-statement** — `nodes_io.py`, `model/vae.py`, `fast_vae.py`, `nodes/vfx/multipass/core.py`, and `scripts/training/dataset_hdr.py`. These are newly-written, uncommitted files; their tails were cut off, so the modules did not compile and their nodes never loaded in ComfyUI (only ~15 of 100+ nodes registered). This is the dominant release blocker and would not have been visible from a live-import that "mostly worked." The last staged git versions of all four (besides `nodes_io.py`) were complete; they were restored from git, and the truncated tail of `nodes_io.py` was completed with a clearly-marked recovery stub.

**Fixes implemented and verified:**

- **C-1 — HDR load divide-by-255:** `_np_to_tensor` is now a pure dtype/shape converter (no magnitude-based normalization). EXR/HDR/float reads pass through unchanged; integer readers normalize explicitly. *Verified:* a value of 123.4 survives load; a written EXR retains a max of ~12.0.
- **C-2 — empty/silent EXR writes:** `_save_exr` no longer writes a 0-byte file or silently downgrades format; it raises a clear `RuntimeError` when no EXR backend is available. *Verified:* with all backends blocked, it raises and leaves no 0-byte file.
- **C-3 — alpha + single-channel EXR:** new `_exr_channels` helper writes 1→RGB (grayscale), 3→RGB, 4→RGBA (alpha preserved), and rejects unsupported channel counts. *Verified:* RGBA round-trip preserves alpha; single-channel matte writes without crashing.
- **C-4 — `torch.load` hardening:** `weights_only=True` added to all 5 shippable-code call sites (`model/vae.py`, `fast_vae.py`, `nodes/vfx/multipass/core.py` ×2, `scripts/training/dataset_hdr.py`). Two training-resume loads in `scripts/training/train_turbo_decoder.py` were intentionally left (self-produced checkpoints with optimizer state; `weights_only=True` would break resume and the untrusted-download threat model does not apply).
- Added `cv2.imread` None-checks in the `.hdr` and EXR-fallback read paths (clear errors instead of opaque crashes).
- Added `tests/test_io_hdr_regression.py` covering all of the above (runs in an env with torch + an EXR backend).

**Verification performed in review sandbox (no torch/OpenEXR; cv2 EXR backend used):**
- All 251 Python files compile (was 4 broken).
- C-1/C-2/C-3 behavioral checks all pass against the real `nodes_io` code.

**Still required before release (unchanged from main report):**
- Run the **full** `tests/` suite against real torch + OpenEXR + OCIO + colour-science (CI currently runs 2 mocked files with `--cov-fail-under=0`); fix the CI `smoke` job's import of the non-existent `radiance.nodes_temporal_coherence`.
- Verify a live ComfyUI import on Windows + Linux and confirm the full node count.
- Review/replace the `RadianceDigitalCinemaRead.read` recovery stub in `nodes_io.py` with the intended Digital Cinema metadata logic.
- Wire a MASK/alpha input into the EXR write node so the now-capable `_save_exr` actually receives alpha from the graph.
