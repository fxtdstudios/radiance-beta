[← Back to Radiance docs](README.md)

# Developer Notes

The practical guide to changing Radiance: environment, adding a node, running
the suite, releasing.

---

## Environment

Radiance runs inside ComfyUI, but the test suite does not need it —
`tests/conftest.py` stubs `comfy.*` and `folder_paths`.

### Minimum: run the suite

```bash
pip install -r requirements.txt
pip install pytest pytest-timeout pytest-cov hypothesis
pytest -q
```

Torch-dependent tests skip themselves automatically if torch is absent. You
will see roughly a thousand tests run and the rest skip.

### Full: run everything

The whole suite needs the optional stack. Without it the catalog is incomplete
and several tests skip rather than fail — which is why a "green" run on a thin
environment does not mean much:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install OpenEXR OpenImageIO opencolorio tifffile safetensors \
            tqdm transformers psutil imageio-ffmpeg
pytest -q
```

Expect **2254 passed / 60 skipped**. The remaining skips are GPU and
network-dependent.

`tqdm` is easy to miss and matters more than it looks: without it
`radiance.nodes.generate` fails to import, taking thirteen nodes with it, and
your baseline silently differs from everyone else's.

### Working against a real ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone <repo> radiance
```

The directory should be named `radiance` — the package imports itself by that
name. (`conftest.py` installs a meta-path finder so tests survive a differently
named checkout, but the runtime does not.)

Useful flags:

| Variable | Effect |
| :--- | :--- |
| `RADIANCE_DEV=1` | Publishes the training node group. |
| `RADIANCE_LOG_LEVEL=DEBUG` | Full tracebacks on node import failure. |
| `RADIANCE_FFMPEG` | Path to a specific ffmpeg binary. |

The startup dependency table prints to **stdout**. If you are capturing JSON
from a script, redirect to a file rather than parsing stdout.

---

## Adding a node

The checklist, in order. Steps 1–3 make the node work; 4–7 are what the build
will fail on if you skip them.

### 1. Write the class

In the right group package — `nodes/<group>/<module>.py`. Never in a root-level
`nodes_*.py`; most of those are deprecation shims now.

```python
class RadianceThing:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "One sentence. This becomes the node's tooltip and its docs entry."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "What this does, in the artist's vocabulary.",
                }),
            },
        }

    def apply(self, image, amount):
        ...
        return (result,)          # always a tuple, even for one output
```

Conventions, beyond [CODE_STYLE.md](dev/CODE_STYLE.md):

- **Image/model inputs first, selections second, scalars last.** Artists scan
  top-down.
- **Every input gets a `tooltip`.** It is the only documentation most users
  read, and the generated node reference shows a blank cell without it.
- **`DESCRIPTION` is mandatory in practice.** It is the node's one-line summary
  in the reference.
- **Optional dependencies degrade, they do not crash.** Log what is missing and
  what to `pip install`.
- **Never write to a bare relative path.** Route it through
  `resolve_output_path()` / `resolve_input_path()` in `radiance.path_utils`.
  Relative goes under `output/`, absolute is honoured as typed, `..` raises.
- **`INPUT_TYPES()` may not touch the filesystem.** It runs on every
  `/object_info` request.
- **Handle 5-D latents** if the node touches latent space. Video models produce
  `(B, C, T, H, W)`; `B, C, H, W = x.shape` raises on them.

### 2. Register it

In the group's `__init__.py` — both dicts, by hand:

```python
from radiance.nodes.color.thing import RadianceThing

NODE_CLASS_MAPPINGS = {
    ...
    "RadianceThing": RadianceThing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ...
    "RadianceThing": "◎ Thing",
}
```

Do this in the same commit as the class. Nine complete node classes were
invisible in ComfyUI's menu for months because this step was skipped, and
nothing fails when you forget — the node simply is not there.

### 3. Check which menu it landed in

`nodes/branding.py` classifies the menu section by keyword, first match wins,
and its rules are not intuitive. A node whose name contains "mask" goes to VFX
before "conditioning" is ever considered for Generate.

```bash
python -c "
import radiance
print(radiance.NODE_CLASS_MAPPINGS['RadianceThing'].CATEGORY)"
```

Wrong section? Add a `SECTION_OVERRIDES` entry in `nodes/branding.py` with a
comment saying why. Do not reorder the keyword rules — that moves other nodes
silently.

### 4. Test it

New test file in `tests/`, named for the behaviour rather than the module.

The one rule that matters: **call the real code.** If the logic you want to
test is buried inside a 900-line method, extract it to a module-level function
and test that. A test that re-implements the logic in its own body will pass
forever regardless of what the implementation does — this has happened here
twice, and both times it hid a shipped bug for months.

Mutation-check anything important: break the fix, confirm the test goes red,
restore it.

```bash
pytest tests/test_thing.py -q
```

### 5. Regenerate the docs

The node reference and the coverage ledger are **generated**. Editing them by
hand fails `test_docs_coverage.py`.

```bash
python tools/generate_docs.py
```

This needs a complete environment — it refuses to run when the catalog is
incomplete, rather than silently writing docs with your node missing. If it
refuses, install what it names.

### 6. Update the ratchets

Adding a node changes three committed artifacts:

```bash
# The public node-key snapshot
RADIANCE_UPDATE_SNAPSHOT=1 pytest tests/test_node_keys_snapshot.py

# README node count — the badge, the "successfully loaded N nodes" line,
# and the "Radiance provides N nodes" line
grep -n "nodes-1[0-9][0-9]\|1[0-9][0-9] nodes" README.md
```

`EXPECTED_MIN_NODE_COUNT` in `config/constants.py` is a floor, not a count —
raise it only when you intend to make the missing-node error stricter.

If your node is published from both a group and a legacy flat module, add it to
`_KNOWN_CROSS_MODULE_DUPLICATES` in `tests/test_nodes_registry.py` with a
comment naming which side is canonical.

### 7. Full suite, then changelog

```bash
pytest -q
ruff check <the files you touched>
```

Add a `CHANGELOG.md` entry under `[Unreleased]` describing the user-visible
effect, not the diff.

---

## Changing an existing node

Extra care in two places.

**Node keys are public API.** They live in every saved workflow. Renaming
`RadianceThing` breaks every graph that uses it, and
`test_node_keys_snapshot.py` will stop you. If a rename is genuinely needed,
keep the old key as an alias.

**Widget defaults are also API, softly.** Changing one changes behaviour for
everyone who never touched that widget. Worth doing when the default is wrong —
both relative-path defaults were fixed exactly this way — but say so in the
changelog under a heading people will read.

Removing a node: delete the mapping entries, keep the class importable for a
release, note it in the changelog, then remove.

---

## Testing

```bash
pytest -q                              # everything
pytest tests/test_nodes_registry.py    # registry integrity
pytest tests/test_node_smoke.py        # every node instantiates and runs
pytest -k "colorspace" -q              # by keyword
```

`pyproject.toml` sets a 30-second per-test timeout, which is the guard against
an infinite loop in node code hanging CI.

CI runs eight jobs: matrix `setup`, the lightweight `test` matrix (Python
3.9–3.12 × Ubuntu/Windows on `main`, 3.11–3.12 × Ubuntu otherwise),
`test-full` with torch/OpenEXR/OCIO, an import `smoke` test, `lint-config`,
`build-install`, plus advisory `lint` (ruff) and `security` (dependency and
static scan). The two advisory jobs are `continue-on-error` and will not block
a merge — read them anyway. Coverage floor is 15%.

The guardrail tests — `test_nodes_registry`, `test_node_keys_snapshot`,
`test_node_load_completeness`, `test_docs_coverage`, `test_output_path_anchoring`,
`test_dead_controls`, `test_package_cleanup`, `test_batch3_guardrails` — each
carry a module docstring explaining the bug that motivated them. Read the
docstring before working around one.

---

## Release

```bash
python tools/check_release_ready.py
```

Then, in order:

1. Confirm the version in `pyproject.toml` and `config/constants.py` agree.
2. `python tools/generate_docs.py` — docs match the catalog.
3. `pytest -q` on a full-dependency environment.
4. Move `CHANGELOG.md`'s `[Unreleased]` section under the new version heading.
5. Build a wheel and check what is in it:

   ```bash
   python -m build --wheel
   python -c "
   import zipfile, glob
   z = zipfile.ZipFile(glob.glob('dist/*.whl')[0])
   n = z.namelist()
   print(len(n), 'files')
   print([x for x in n if 'test' in x.lower() or x.endswith('.pyc')])"
   ```

   Expect ~258 files and nothing from `tests/`. A new sub-package that is not
   listed in `[tool.setuptools] packages` will be silently absent — this check
   is how you catch that.
6. Install the wheel into a clean ComfyUI and confirm the startup banner reads
   `successfully loaded 110 nodes`, not a shortfall error.

---

## Dynamic gizmos

Gizmos are user-authored nodes loaded after the static catalog. A gizmo
declares its own node name, display name, category and wiring. Because the
content is per-studio, Radiance documents the loader contract only — document
your own gizmos alongside them. `gizmos/*.gizmo` is gitignored.

---

## Getting unstuck

| Symptom | Look at |
| :--- | :--- |
| Node missing from the menu | The group `__init__.py` mapping dicts. Then the startup log for an import failure. |
| Node in the wrong menu | `SECTION_OVERRIDES` in `nodes/branding.py`. |
| `test_docs_coverage` fails | Run `tools/generate_docs.py`. Do not edit `docs/coverage.md`. |
| `test_node_keys_snapshot` fails | Intentional? Regenerate with `RADIANCE_UPDATE_SNAPSHOT=1`. Unintentional? You renamed a key. |
| `ValueError: too many values to unpack` mid-sample | A 4-way shape unpack meeting a 5-D video latent. |
| Files appearing in the repo root after a test run | A path default that is not going through a resolver. |
| Startup says "loaded N of at least 109" | Read the ERROR lines above it; a group failed to import. |

---

## See also

- [Code Style](dev/CODE_STYLE.md) — naming and formatting.
- [Known Limitations](limitations.md) — what does not do what its name implies.
- `CONTRIBUTING.md` — pull request process and commit conventions.
