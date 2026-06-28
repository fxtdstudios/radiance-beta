[← Back to Radiance docs](../README.md)

# Radiance — Incremental Refactor Roadmap

A sequenced plan to pay down the cross-author structural debt **without** a risky
big-bang rewrite. Every phase is independently shippable, leaves the ~1,400-test
suite green, and is its own PR. Phases are ordered so that the safest, enabling
work lands first and the riskiest consolidation lands last — on top of guardrails
that make it safe.

## The one invariant (read first)

`NODE_CLASS_MAPPINGS` **keys are public API** — saved ComfyUI workflows store them.
No refactor may rename, add, or drop a node key. Display names, categories, file
locations, and internal symbols are all free to change; the key set is frozen.
Phase 0 locks this with a test so every later phase is provably safe.

---

## Current state (honest assessment)

**Already healthy — do not rewrite:**
- Declarative registration: `nodes/catalog.py` (groups) → `nodes/registry.py` (loader) → `nodes/branding.py` (names/menu).
- Domain logic separated from node wrappers: `color/`, `hdr/`, `image/`, `io/`, `model/`, `delivery/` vs `nodes/`.
- Centralized config in `config/` (constants, env, dependencies) + runtime dependency checker.
- ~1,400 tests; CI with matrix, smoke, packaging, and (advisory) lint.

**The real debt (what this plan targets):**
1. **Duplicated node layer.** ~40 node keys register from *both* a legacy top-level `nodes_*.py` and the organized `nodes/<group>/` package (see `_KNOWN_CROSS_MODULE_DUPLICATES` in `tests/test_nodes_registry.py`). Two parallel structures from different authors.
2. **Monolithic files.** `nodes_io.py`, `hdr/vae.py`, `core/logging.py`, `nodes_sampler.py` each carry several responsibilities.
3. **Heuristic menu classification.** `classify_menu_section` keyword-guesses each node's menu and mis-files edge cases.
4. **Scattered constants.** Some ports/paths/magic numbers live inline rather than in `config/`.

---

## Phase 0 — Guardrails (enables everything else)

**Goal:** make refactoring provably safe before touching structure.

- **Freeze the node-key set.** Add `tests/test_node_keys_snapshot.py`: assert the full sorted list of `NODE_CLASS_MAPPINGS` keys equals a committed golden list. Any accidental add/drop/rename fails CI. This is the safety net for Phases 3–4.
- **Make lint gating** once the 3 real findings are fixed (already done this cycle): drop `continue-on-error` from the ruff job.
- **Confirm the wheel-completeness + packaging tests** are in CI (added this cycle).

**Risk:** none (additive). **Verify:** suite green; snapshot test passes on current tree.

---

## Phase 1 — Split the monolith files (mechanical, low risk)

**Goal:** Single Responsibility per module; smaller, reviewable units. Pure code
movement — no behavior change.

Targets and proposed splits (keep a thin re-export shim at the old path so imports
and saved workflows keep working):

| Monolith | Split into |
| :--- | :--- |
| `nodes_io.py` | `nodes/io/read.py`, `write.py`, `sequence.py`, `multipart.py` (the `nodes/io/` package already exists — finish moving logic there). |
| `hdr/vae.py` | `hdr/vae/decode.py`, `encode.py`, `config.py` (model-VAE resolution). |
| `core/logging.py` | `core/logging/table.py` (print_table/themes), `dedupe.py` (throttle), `setup.py`. |
| `nodes_sampler.py` | `nodes/generate/sampler.py` + `sampler_presets.py` (preset catalog). |

**Per-file procedure:**
1. Create the new submodule(s); move one cohesive responsibility at a time.
2. Replace the old file body with `from .new_location import *` re-exports (temporary).
3. Run the suite after **each** move (green between every step).
4. Update internal imports to the new paths; leave the shim until Phase 4.

**Risk:** low (no logic change). **Verify:** suite + import-smoke green; `git diff` shows pure moves.

---

## Phase 2 — Decouple remaining hardcoded config

**Goal:** one source of truth for tunables; no magic numbers in logic.

- Sweep for inline constants (ports, bind hosts, size caps, compression levels, default paths) and move them to `config/constants.py` (or a typed `config/defaults.py`).
- Keep the existing `config/env.py` as the env-override wrapper; values read config → env override → caller override (the decoder already follows this pattern; generalize it).
- No secrets exist in-repo today; add a `config/secrets.py` wrapper that only ever reads from environment, so future keys have a single sanctioned path.

**Risk:** low. **Verify:** suite green; grep shows no stray literals in the touched modules.

---

## Phase 3 — Explicit menu declaration (kills a bug class)

**Goal:** replace keyword guessing with intent.

- Add an optional `RADIANCE_SECTION` class attribute to each node (or a small per-group map in each `nodes/<group>/__init__.py`).
- `classify_menu_section` becomes: explicit declaration → `SECTION_OVERRIDES` → (last resort) the existing heuristic, which now only ever runs for nodes nobody declared.
- Add `tests/test_menu_sections.py`: every registered node resolves to a section in `MENU_STRUCTURE`, and no node falls through to the heuristic (once all are declared).

**Risk:** medium (touches every node group, but additively). **Verify:** menu snapshot test; the node-key snapshot from Phase 0 guarantees no key drift.

---

## Phase 4 — Consolidate the duplicate node layer (the big one, last)

**Goal:** one canonical structure. The organized `nodes/<group>/` packages become
the single home; legacy top-level `nodes_*.py` files are retired.

**Why last:** highest blast radius. It relies on Phase 0's key snapshot (proves no
workflow breaks), Phase 1's splits (logic already lives in `nodes/`), and Phase 3
(menu no longer inferred from module names).

**Procedure, one legacy module at a time:**
1. Confirm the canonical implementation lives in `nodes/<group>/` (move it there if the legacy file is still the real one).
2. Convert the legacy `nodes_X.py` to a pure deprecation shim: `from radiance.nodes.<group>.X import *` + a one-line `DeprecationWarning`.
3. Remove the legacy entry from `_KNOWN_CROSS_MODULE_DUPLICATES` as each is collapsed to a single namespace.
4. Run the full suite; the node-key snapshot must be unchanged.
5. After all are shims, delete the shims in a final commit and drop the duplicates-allowlist entirely — at which point `test_no_duplicate_node_class_mapping_keys` enforces single-registration permanently.

**Risk:** high — mitigated by doing it module-by-module behind the key snapshot, never in bulk.

---

## Standards applied throughout

- **Naming:** `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE` constants; node keys stay exactly as-is (public API). Replace vague names only at internal call sites, never node keys.
- **Guard clauses:** flatten deep nesting with early returns; no behavior change in the same commit as a structural move.
- **SRP:** one reason to change per module/function; the splits above are the concrete application.
- **Comments:** explain *why* (the workflow-key invariant, the gsplat-style runtime gating, the decode config precedence) — not *what*.
- **One concern per PR:** never mix a split with a behavior fix.

## Suggested order & effort

| Phase | Risk | Rough size | Depends on |
| :--- | :--- | :--- | :--- |
| 0 Guardrails | none | S | — |
| 1 Monolith splits | low | M | 0 |
| 2 Config decouple | low | S | — |
| 3 Menu declaration | medium | M | 0 |
| 4 Layer consolidation | high | L | 0,1,3 |

Land 0 first, then 1 and 2 in any order, then 3, then 4 last. Each is a clean PR
that leaves the suite green and the node-key snapshot unchanged.
