# Radiance v4 — Repository Restructure Plan

This is the reference plan for tidying the package layout in the next major
release. The domain packages (`core/`, `color/`, `hdr/`, `image/`, `io/`,
`model/`, `nodes/…`) are already well organized; the goal is to remove
accumulated clutter at the repository root and group supporting files.

## Problems in the current layout

- **47 root `nodes_*.py` files** — 40 are backward-compatible deprecation
  shims; ~6 still contain real, un-migrated code
  (`nodes_io`, `nodes_loader`, `nodes_realtime_preview`, `nodes_sampler`,
  `nodes_workspace`).
- **~15 loose utility modules** at the root (`sampler_utils.py`,
  `loader_utils.py`, `fast_vae.py`, `lut_utils.py`, `gpu_utils.py`,
  `cache.py`, `recovery.py`, `path_utils.py`, `secret_utils.py`,
  `tensor_contract.py`, `exceptions.py`, `radiance_ocio.py`,
  `color_utils.py`, `config.py`).
- **Internal docs/reports** mixed into the root (`CLEANUP_REPORT.md`,
  `CODE_STYLE.md`, `PRE_RELEASE_REVIEW.md`, old release notes, decode report).
- **Bundled data** loose at the root (`ACES/`, `rpacks/`).

## Target structure

```text
radiance/                     repo root = the `radiance` package (ComfyUI entry)
├── __init__.py               NODE_CLASS_MAPPINGS, display names, WEB_DIRECTORY
├── pyproject.toml · requirements*.txt · package.json
├── README.md · CHANGELOG.md · CONTRIBUTING.md · LICENSE
├── nodes/                    ComfyUI node classes, by domain
│   ├── core/ io/ generate/ color/ hdr/ vfx/ video/ upscale/
│   ├── review/   (rename of monitor/)
│   ├── pipeline/
│   └── _dev/     (rename of training/)
├── lib/                      implementation libraries (no node classes)
│   ├── core/ color/ hdr/ image/ model/ sampling/ film/ gpu/ delivery/ util/
├── web/                      frontend (rename of js/), served at /extensions/radiance/
│   ├── dashboards/ nodes/ assets/
├── data/                     bundled read-only resources
│   ├── ocio/ rpacks/ presets/
├── examples/                 shipped demo workflows
├── docs/                     user docs site (+ docs/dev/ for internal docs)
├── tests/ · tools/ · scripts/
└── (gitignored runtime user data) workflows/  gizmos/
```

## Hard constraints

1. The repository root must remain the `radiance` package — ComfyUI loads the
   root `__init__.py`, and the code imports itself as `radiance`.
2. `NODE_CLASS_MAPPINGS` keys must not change — moving files changes import
   paths, not keys, so saved workflows keep loading. Renaming a node *category*
   (e.g. `monitor` → `review`) only changes the menu location.
3. `WEB_DIRECTORY` and the `/extensions/radiance/` URLs must be preserved — a
   `js/` → `web/` rename changes only the source folder, but every internal JS
   and dashboard reference has to be updated together.

## Phases (each independently shippable; run the full test suite after each)

1. **Remove the 40 deprecation shims** and rewrite imports that still point at
   them. (Automated by `tools/restructure_phase1.py`.)
2. **Migrate the ~6 real root modules** into `nodes/` and update importers.
3. **Consolidate loose root utilities** into `lib/`.
4. **Group data and internal docs** into `data/` and `docs/dev/`.
5. **(Optional) rename `js/` → `web/`** with sub-folders.
6. **Bump to v4.0** and write CHANGELOG migration notes.

Do every phase on a branch, run `python -m pytest tests/` between phases, and
validate a real ComfyUI load before tagging.
