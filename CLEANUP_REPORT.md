# Radiance — Clean-Release Audit (unwanted files & nodes)

**Goal:** a clean v3.1.0 you can publish to the Comfy Registry + GitHub without dev cruft, dead code, or misleading legacy modules.
**Method:** full tree scan (252 .py files, 252 incl. new AOV reader), cross-referenced against `.comfyignore` and `.gitignore`.
**Important:** This is a report only — nothing was deleted. Nothing here changes the live node set unless noted.

---

## TL;DR

- **No phantom/extra nodes ship.** ComfyUI registers nodes only through the declarative catalog (`radiance.nodes.*`). The 40 root `nodes_*.py` shims do **not** add nodes to the menu — they're Python import-compat stubs only.
- **Real cruft that WILL ship to the registry and should be removed/ignored:** `artifacts/` (logs + mock SVG), `PRE_RELEASE_REVIEW.md`, `radiance_hdr_vae_decode_report.md`. None are excluded by the current `.comfyignore`.
- **40 deprecated shim modules** at the repo root are dead weight; 37 are pure orphans (safe to delete), 3 are still self-imported by the package (must repoint first, then delete).
- **6 root modules are NOT shims — they are the live implementation** and must stay until the migration into `nodes/` is finished.

---

## 1. Unwanted FILES

### 1a. Ships to the registry today and shouldn't (FIX before publish)

| File / dir | Size | Status | Action |
| :-- | --: | :-- | :-- |
| `artifacts/docs-server.err.log` | 2.4 KB | dev log, not in `.comfyignore`* | delete from repo + add `artifacts/` to `.comfyignore` |
| `artifacts/docs-server.out.log` | 0 B | empty dev log | delete |
| `artifacts/radiance_lite_viewer_mock.svg` | 5 KB | design mock, ships to registry | delete or move to `docs/` |
| `PRE_RELEASE_REVIEW.md` | 30 KB | **my audit report** — internal | exclude from registry; keep on GitHub only if you want it |
| `radiance_hdr_vae_decode_report.md` | 9 KB | internal dev report | exclude from registry / delete |

\* `.gitignore` ignores `*.log`, so the two logs aren't committed — but the `artifacts/` directory and the `.svg` **are** tracked and not excluded by `.comfyignore`, so they land in the published package.

### 1b. Already handled — no action needed (verify only)

- `__pycache__/` (31 dirs) and `*.pyc` (513 files, ~6.7 MB): excluded by both `.gitignore` and `.comfyignore`. Confirm none were force-added with `git ls-files | grep -E "pyc$|__pycache__"` (should be empty).
- `tests/`, `tools/check_*`, `package.json`, `package-lock.json`, `RADIANCE_v3.1_RELEASE_NOTES.md`, `*.docx`, `*.bak`, `.github/`, `js/docs/`: all excluded from the registry archive by `.comfyignore`. Good. (They stay on GitHub, which is correct.)
- `tests/test_io_hdr_regression.py` (the regression tests I added): excluded from the registry via `tests/`, kept on GitHub. Correct.

### 1c. Keep (referenced assets — do NOT delete)

`RADIANCE.png`, `r_icon.png`, `icon.png`, `viewer.png`, `Viewer_shortcut.png`, `radiance_workspace.png`, `basic_workflow.png` (README/docs + registry icon), `ACES/config.ocio`, `rpacks/SDXL_Standard.rpack`, `workflows/`, `docs/`.

---

## 2. Unwanted / legacy NODES & modules

### 2a. Deprecation shims (40 files) — dead weight, not live nodes

Every small root `nodes_*.py` (≈600 B–1 KB each) is a backward-compat stub: it emits a `DeprecationWarning` and re-exports from the real `radiance.nodes.<group>.<module>`. They exist so old code that did `import radiance.nodes_grade` still works. **They do not register nodes** (the catalog loads `radiance.nodes.*` only), so deleting them removes zero nodes from the UI.

**37 pure orphans — safe to delete now** (nothing in the package imports them; only break external `import radiance.nodes_X`, which workflows never use — workflows reference node *keys*):

```
nodes_3d  nodes_aces2  nodes_audio_cut  nodes_cdl  nodes_character
nodes_colorscience  nodes_curves  nodes_denoise  nodes_depth  nodes_engine
nodes_grade  nodes_hdr_colorspace  nodes_hdr_delivery  nodes_hdr_encoder
nodes_hdr_inception  nodes_hdr_lora  nodes_hdr_patch  nodes_hdr_synthesis
nodes_hdr_uplift  nodes_metadata  nodes_motion  nodes_motion_blur  nodes_ocio
nodes_optics  nodes_overlay  nodes_prompt  nodes_qc  nodes_radiance_mask
nodes_regional  nodes_resolution  nodes_scene_cut  nodes_sdr_degradation
nodes_send_dcc  nodes_studio  nodes_t2v_pipeline  nodes_upscale  nodes_vae_v3
```

**3 shims still self-imported by the package — repoint, THEN delete:**

| Shim | Imported at | Repoint to |
| :-- | :-- | :-- |
| `nodes_hdr_smart` | `nodes/generate/lora.py:66` | `radiance.nodes.hdr.smart` |
| `nodes_dit_adapter` | `nodes/video/t2v.py:69` | `radiance.nodes.video.dit` |
| `nodes_video_hdr` | `nodes/video/t2v.py:1324` | `radiance.nodes.video.hdr` |

Right now the package imports its **own deprecated paths**, which fires `DeprecationWarning`s on every load and adds pointless indirection. Fix the 3 imports to the real targets, then all 40 shims can go.

> Decision point: if you want to *guarantee* old `import radiance.nodes_*` paths keep working for external users for one more version, keep the 37 orphans for v3.1 and delete in v3.2. They're ~25 KB total and harmless. My recommendation for a "clean" release: repoint the 3, delete all 40, and note it in the changelog.

### 2b. Real code still at root — KEEP (migration incomplete, not cruft)

These six are **not** shims — they're the live implementation the `nodes/` package imports from. Do not delete:

```
nodes_io.py (72 KB)  nodes_sampler.py (71 KB)  nodes_workspace.py (59 KB)
nodes_realtime_preview.py (42 KB)  nodes_loader.py (35 KB)  nodes_gizmo.py (17 KB)
```
`nodes_radiance_viewer.py` is also live (loaded via relative import in `__init__.py`). Finishing the migration (moving these into `nodes/…`) is post-release tech-debt, not a release blocker.

### 2c. Dead node class to remove

- `RadianceDigitalCinemaRead` in `nodes_io.py` — the recovery-stub I rebuilt to make the file import. It is **not** registered by the catalog (only `RadianceRead/Write/EXRMultiPart` are imported by `nodes/io/__init__.py`), so it's dead code. Either finish it with real Digital Cinema metadata logic or delete the class + its local `NODE_CLASS_MAPPINGS` entry.

---

## 3. Recommended `.comfyignore` additions

Append these so the published package stays clean:

```gitignore
# Internal reports & dev artifacts (keep on GitHub, exclude from registry)
artifacts/
PRE_RELEASE_REVIEW.md
CLEANUP_REPORT.md
radiance_hdr_vae_decode_report.md
*.svg
```

(Already covered, listed for confidence: `tests/`, `tools/check_*`, `package*.json`, `*.docx`, `*.bak`, `__pycache__/`, `*.log`, `.github/`, `js/docs/`, `RADIANCE_v*_RELEASE_NOTES.md`.)

## 4. Recommended `.gitignore` addition

```gitignore
# Local dev server artifacts
artifacts/
```
Then `git rm -r --cached artifacts` to stop tracking the logs/mock that are currently committed.

---

## 5. Clean-release action checklist

1. Repoint the 3 package imports (2a table) to the real `radiance.nodes.*` targets.
2. Delete the 40 deprecation shims (or just the 37 orphans if keeping import back-compat for v3.1).
3. Delete / un-track `artifacts/`; remove `PRE_RELEASE_REVIEW.md`, `CLEANUP_REPORT.md`, `radiance_hdr_vae_decode_report.md` from the shipped package (via `.comfyignore`).
4. Resolve `RadianceDigitalCinemaRead` (finish or delete).
5. Add the `.comfyignore` / `.gitignore` lines above.
6. Verify nothing committed is junk: `git ls-files | grep -E "\.pyc$|__pycache__|\.log$"` → should be empty.
7. Re-run the live ComfyUI import and confirm the node count is unchanged after deleting shims (it must be — shims don't register nodes).

**Net effect:** removing all of section 1a + the 40 shims strips ~45 KB of dead modules and the dev artifacts from the package, kills the load-time DeprecationWarnings, and leaves the live node set identical.
