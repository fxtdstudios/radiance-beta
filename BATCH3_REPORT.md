# Batch 3 — Guardrails

**Date:** 2026-07-28 · **Scope:** the three items carried forward from the audit and repeated in
`AUDIT.md` and `VIEWER_REPORT.md` §5 · **Files changed:** 30 · **New tests:** 22

Every defect in this batch is a *reporting* defect. In all three cases the code did what it did and
then told you it had gone fine. Batch 0–2 fixed things that were broken; this batch fixes the
reasons nobody noticed.

---

## 1 · CI was red for 17 days because of a list nobody could keep current

`.github/workflows/ci.yml:102-108` carried a hand-maintained `--ignore=` list naming the seven test
files that import torch at module scope, so the no-torch lightweight matrix would skip them.

It went stale the moment anyone added a torch-using test file. `tests/test_multipass_contracts.py`
and `tests/test_temporal_rudra.py` were never added, and CI ran red from **2026-07-10** with 13
failures.

### Why every self-skip idiom in the suite failed

`tests/conftest.py` installs a MagicMock torch stub so module-level `import torch` doesn't kill
collection. Three separate "is torch real?" checks were in use, and the stub defeated all of them:

| idiom | why it says "real torch" under the stub |
|---|---|
| `pytest.importorskip("torch")` | the stub **is** importable |
| `isinstance(torch, MagicMock)` | the stub is a genuine `ModuleType` |
| `hasattr(torch.zeros(1), "shape")` | every attribute is a MagicMock; `hasattr` is always True |

`tests/test_audit_regressions.py` had a `requires_torch` marker built on the second and third of
those. It had never once fired.

### The fix

The stub now marks itself (`torch.__radiance_stub__ = True`), and `conftest.py` gained a gate that
cannot go stale:

1. `HAS_REAL_TORCH` — one source of truth, keyed off that marker.
2. `@pytest.mark.real_torch` — a registered marker; those tests skip when the stub is active.
3. An automatic module-level skip: any test file that binds torch at module scope (unguarded
   `import torch`, `from torch import …`, or `importorskip("torch")`) is skipped wholesale, detected
   by AST at collection time. A guarded `try: import torch / except ImportError` is left alone.
   A file that gates itself per test opts out with `RADIANCE_TORCH_GATED = True`.

The gate is inert when real torch is present, so the `test-full` lane still runs every one of these.
`RADIANCE_DISABLE_TORCH_GATE=1` turns it off for debugging.

Twenty test files were then annotated so their stub-safe assertions keep running on the lightweight
matrix rather than disappearing with the module: eleven already gated correctly and only needed the
opt-out; five needed `@pytest.mark.real_torch` on 22 individual tests; four (the audit/viewer suites)
had their broken `_REAL_TORCH` detector replaced.

Three files also came off the ignore list on their own merits — `test_io_functional.py`,
`test_io_hdr_regression.py` and `test_radiance_load_image_mask.py` were being ignored for a
condition that no longer applied.

> One find along the way: `test_math_transforms.py::test_gradual_sigma_blend` passes alone and fails
> in a full run. Another module in the suite swaps the shared torch stub for a numpy-backed shim
> whose indexing yields plain floats. It is marked `real_torch`; the stub-mutation hygiene issue
> underneath is noted below.

**Result:** the `--ignore` list is gone. The lightweight matrix runs the whole `tests/` tree:
**1136 passed, 610 skipped, 0 failed**, coverage 20.5% against the 15% gate.

---

## 2 · A meta-test that warned instead of asserting

`tests/test_node_smoke.py:992` — `test_all_keys_have_class` compares node keys found by AST in the
source against the keys the package actually publishes, and called `warnings.warn` on a mismatch.
It had been swallowing **12** of them, and the warning text blamed "torch/GPU/ComfyUI at runtime"
for what is really a registration gap: every one lives in a legacy flat-file module the v3 entry
point never loads.

It now asserts against an explicit ratchet, `_KNOWN_UNREGISTERED`, with each entry annotated with
its source file:

```
RadianceBitDepthDegrade     nodes_colorscience.py
RadianceControlApply        nodes_loader.py
RadianceControlNetApply     nodes_loader.py
RadianceDigitalCinemaRead   nodes_io.py
RadianceDigitalCinemaWrite  nodes_io.py
RadianceFlipbookGIF         nodes_realtime_preview.py
RadianceImageLoader         nodes_loader.py
RadianceLUTApply            color/lut.py
RadianceLUTBlend            color/lut.py
RadiancePolicyGuard         nodes_qc.py
RadiancePreviewServer       nodes_realtime_preview.py
RadianceWorkspace           nodes_workspace.py
```

It fails in **both** directions: a key that newly stops resolving is a regression, and a key that
starts resolving must be removed from the list — so the allowlist can only shrink. It skips (rather
than drowning in noise) on a machine that is short a runtime dependency, since that has its own
loud error now — see below.

---

## 3 · An 88% shortfall that printed as success

`__init__.py:56` logged `"Radiance: successfully loaded %d nodes"` unconditionally, with nothing to
compare against. In a scrolling ComfyUI console, *successfully loaded 12 nodes* reads exactly like
*successfully loaded 100 nodes*. Underneath it, `nodes/registry.py:133-138` collected a `failures`
tuple that `nodes/__init__.py` then threw away, and `:226` logged group failures at WARNING several
hundred lines earlier.

Three changes:

- `config/constants.py` gains `EXPECTED_MIN_NODE_COUNT = 100` — a floor, so growing the catalog is
  free and shrinking it is loud.
- `radiance/nodes/__init__.py` publishes `NODE_LOAD_FAILURES` instead of discarding them, and the
  entry point folds them into its own result. Without this, a group that dropped 18 nodes on an
  optional-dependency error still looked clean from the top.
- `report_node_load_health()` picks the severity from reality: INFO on a healthy load, ERROR naming
  each failed module and its exception, ERROR again on a shortfall with the count and percentage.

It caught something immediately. On a machine missing `tqdm`:

```
ERROR  Radiance: 5 node module(s) failed to import; every node they export is missing from ComfyUI.
ERROR    - radiance.nodes.color: ModuleNotFoundError: No module named 'folder_paths'
ERROR    - radiance.nodes.io: ModuleNotFoundError: No module named 'folder_paths'
ERROR    - radiance.nodes.pipeline: ModuleNotFoundError: No module named 'folder_paths'
ERROR    - radiance.nodes.monitor: ModuleNotFoundError: No module named 'folder_paths'
ERROR    - radiance.nodes.generate: ModuleNotFoundError: No module named 'tqdm'
ERROR  Radiance: loaded 59 of at least 100 expected nodes (v3.1.2) - 41 missing (41% of the
       catalog). This is a failed start, not a small one: check the import errors above, then
       re-run with RADIANCE_LOG_LEVEL=DEBUG for tracebacks.
```

That is the exact class of failure the old banner hid.

---

## 4 · One thing found while checking lint parity

`scripts/start_nuke_server.py` still built a `_SAFE_BUILTINS` / `safe_globals` restricted-execution
sandbox at the point where Batch 1 had removed the `eval`/`exec` fallback. It was dead — nothing
referenced it. Removed: a sandbox that guards nothing but reads like an active defence is worse than
no sandbox, because the next person to touch that function will assume it is doing something.

---

## Verification

Same method as every previous batch — a pristine copy of the tree extracted alongside the working
one, the identical command run against both, the FAILED sets diffed.

| check | result |
|---|---|
| Real-torch suite | **1686 passed, 11 failed** — identical failure set to the pristine baseline (`comm -13` empty in both directions) |
| No-torch CI matrix (clean venv: pytest, numpy, scipy, Pillow, defusedxml, opencv-headless, tqdm — no torch) | **1136 passed, 610 skipped, 0 failed** — was 93 failed before this batch |
| Coverage gate | 20.52% vs the 15% floor |
| Ruff | byte-for-byte parity with baseline, per file per rule |
| CI `smoke` job | reproduced locally, all 6 modules import, exit 0 |
| CI `lint-config` job | `ci.yml` parses as YAML, both jobs present |

Test count 1665 → **1686** (+21 net; 22 new guardrail tests, one pre-existing test now correctly
skipped when it cannot run).

Two of the new tests initially failed against my own prose — a substring check matched the comment
explaining the bug rather than the bug. Both now compare parsed code with docstrings and comments
stripped, the same lesson as the `strict=False` self-match in Batch 0.

---

## Still open

Nothing from Batch 3 remains. Carried forward from earlier rounds:

- **Nothing is committed.** ~40 files are uncommitted on `release/cleanup`; `beta/main` is 36
  commits ahead of the local tree.
- **54 unregistered nodes** — the 12 in `_KNOWN_UNREGISTERED` are the subset that a
  `NODE_CLASS_MAPPINGS` literal declares. Wiring them up is a product decision, not a bug fix.
- **Stub mutation between test modules** — one module replaces the shared torch stub with a
  numpy-backed shim, which is why `test_gradual_sigma_blend` is order-dependent. Worth making the
  stub immutable or module-scoped.
- **JS duplication** — 6 diverged copies of the widget toolkit, 4 of `escapeHtml`.
- **~30 silent-failure handlers**; `OpenImageIO` missing from 3 platform requirements files; 3
  declared-but-never-imported deps (`colour-science`, `einops`, `torchsde`).
- `bug_report_reply_draft.md` is still on disk (the device bridge cannot delete; it is in both
  ignore files).
