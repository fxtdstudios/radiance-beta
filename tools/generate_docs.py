#!/usr/bin/env python3
"""Generate the Radiance node reference from the live catalog.

Run from the repository root, inside an environment where `import radiance`
succeeds with every optional dependency present:

    python tools/generate_docs.py

Why generate rather than hand-write
-----------------------------------
The previous per-group pages were produced once and then drifted. They
documented 96 nodes while ComfyUI loaded 100; four registered nodes had never
been added at all and nine more existed in the source without ever being
registered. Several parameter tables printed raw Python source in the type
column -- `(_get_input_files(), {'image_upload': True, ...})` -- because the
generator read the source text instead of calling `INPUT_TYPES()`.

This version introspects the real `NODE_CLASS_MAPPINGS`, so a node cannot be
missing, a default cannot be stale, and a tooltip cannot be lost. Prose that a
human should write -- what a group is for, what to do before using it -- lives
in `GROUP_INTROS` below and is preserved across regeneration.

`tests/test_docs_coverage.py` fails if the output drifts from the catalog.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

DOCS = ROOT / "docs" / "built-in-nodes"

#: category suffix -> (filename, human title)
GROUPS = {
    "Load & Save": ("io-delivery.md", "IO and Delivery"),
    "Generate": ("generate.md", "Generate, Loaders, and Sampling"),
    "Color": ("color.md", "Color"),
    "HDR": ("hdr-aces.md", "HDR and ACES"),
    "VFX": ("vfx.md", "VFX, Masks, Optics, and Multipass"),
    "Core": ("pipeline.md", "Pipeline and Studio"),
    "Pipeline": ("pipeline.md", "Pipeline and Studio"),
    "Review": ("review.md", "Review, Viewer, and Preview"),
    "Upscale": ("upscale.md", "Upscale"),
    "Video": ("video.md", "Video"),
}

GROUP_INTROS = {
    "IO and Delivery": """
Load, inspect, save, and package production media. These are the entry and exit
points of every Radiance graph, and the place where a colour-space mistake does
the most damage — what you decode wrongly here is wrong for the rest of the
graph, and nothing downstream will tell you.

## Typical graph

```text
Radiance Read → colour transform → work → Radiance Write
```

## Before you use these nodes

- **Use EXR for anything you intend to keep.** PNG and JPEG clip at 1.0 and
  quantise; they are for review proxies, not masters.
- **Set `color_space` on Read deliberately.** "Auto / Linear (pass-through)"
  means *no decode* — correct for an EXR that is already scene-linear, wrong for
  a camera log file, which will look flat and grade badly.
- **Alpha on EXR.** Write has an optional `mask` input. Connected, with an EXR
  format selected, the matte is written as the alpha channel. It is ignored for
  non-EXR formats. Write failures raise — the node turns red — rather than
  leaving an empty file behind.
- **Give Write a filename.** With `filename` blank, `output_path` is treated as
  the full path including the stem. A blank on both raises a clear error rather
  than writing somewhere surprising.
- **DPX needs OpenImageIO.** There is no Pillow plugin for it. The Digital
  Cinema nodes will tell you if the package is missing.

## Reading video

Video decodes through one raw ffmpeg pipe at the source's own bit depth — 10-bit
ProRes 422, 12-bit ProRes 4444, DNxHR HQX and 10-bit HEVC all survive intact.

- **ProRes 4444 alpha reaches the `mask` output.** 422 in any flavour has no
  alpha channel; only 4444 and 4444 XQ do.
- **`start_frame`, `end_frame` and `frame_step` apply to video**, on the clip's
  own zero-based numbering. `start_frame` defaults to 1001 because that is the
  sequence convention, so a start past the end of a clip is treated as unset
  rather than decoding nothing.
- **Colour tags are read.** On Auto, a file tagged `bt709`, `smpte2084` or
  `arib-std-b67` decodes through the matching curve and the node logs which one.
  An *untagged* file passes through unchanged and warns — a Rec.709 delivery is
  not scene-linear, and Radiance will not guess for you.
- **Read raises on failure.** A missing file, a corrupt MOV or a truncated clip
  turns the node red instead of yielding black frames.

## Known limitations

The EXR reader ignores the display window, so an overscan render (a standard
Nuke output) comes back offset and at the data-window resolution with no
warning. Crop to the display window in your comp application before bringing
overscan plates into Radiance.

A clip is decoded to one batched tensor, so RAM is the ceiling — 240 frames of
4K RGBA float32 is about 31 GB. Interlaced sources are reported but not
deinterlaced.
""",
    "Generate, Loaders, and Sampling": """
Model loading, prompt construction, sampling, and HDR-aware latent decoding.

## Typical graph

```text
Radiance Loader → Cinematic Prompt Encoder → Radiance Sampler → HDR VAE Decode → Viewer
```

## Before you use these nodes

- **Radiance Sampler adapts to the model.** Widgets that do not apply to the
  detected architecture are hidden rather than ignored. If a control you expect
  is missing, check `model_meta` — the loader publishes what it detected.
- **`cfg` on Flux.** Flux models use a guidance embedding and want `cfg = 1.0`.
  Several of the sampler's extras (guidance rescale, CFG++) need `cfg > 1.0` to
  do anything, and will say so in the log.
- **HDR VAE decode needs a RUDRA decoder** matching your base model. Without
  one, decode falls back to the standard VAE and the result is display-referred,
  not scene-linear.
- **LoRA stack order matters.** Entries apply in list order; a style LoRA after
  a structural one behaves differently from the reverse.

## Known limitations

`PAG` here is a mid-block self-attention perturbation, not the full method from
Ahn et al. 2024 — there is no separate perturbed forward pass. Restart sampling
follows Xu et al. Alg. 2 as of 3.2.0, but it re-runs the whole tail of the
schedule, so it is not cheap.
""",
    "Color": """
Grading, colour-space conversion, CDL, LUTs, curves, and QC. Everything here
assumes you know which space your image is in — the nodes will not guess, and a
transform applied to the wrong space is the single most common way to lose an
afternoon.

## Typical graph

```text
Read (decode to scene-linear) → Grade / CDL / Curves → Colorspace Convert (to display) → Viewer
```

## Before you use these nodes

- **Grade in scene-linear.** Exposure is a multiply, and a multiply only means
  "stops" in a linear space. Grading a gamma-encoded image gives you contrast
  changes you did not ask for.
- **Colorspace Convert is a primaries + transfer change**, not a look. It will
  not make an image "correct"; it moves it between two spaces you name.
- **CDL is ASC-standard** — slope, offset, power, saturation — and round-trips
  through `.cc`/`.cdl` files for handoff to a grading system.
- **QC before you deliver.** The QC node reports clipping, out-of-gamut pixels
  and banding risk. Policy Guard turns a house standard into a pass/fail.

## Known limitation

`chromatic_adaptation` on Colorspace Convert has no effect — the D60↔D65
adaptation is baked into the precomputed conversion matrices, and every option
on the widget produces identical output. The node logs a warning if you set it
to anything other than the default.
""",
    "HDR and ACES": """
ACES output transforms, OCIO, HDR encode and decode, tone mapping, SDR→HDR
uplift, and the HDR VAE path.

## Typical graph

```text
scene-linear → ACES 2.0 Output Transform → display
scene-linear → HDR Encode (PQ / HLG) → deliverable
SDR plate    → SDR to HDR Universal → scene-linear HDR
```

## Before you use these nodes

- **Know what "peak nits" means in each place.** For PQ it is an absolute
  display luminance and part of the encode. For HLG it is a property of the
  monitor, not of the file. For DCI it is the projector's luminance at code
  value 1.0. The nodes handle these differently because the standards do.
- **Tone mapping is one-way.** Once you have tone-mapped for a display you have
  thrown away highlight information. Keep the scene-linear master.
- **AgX emits display code values directly** and is not gamma-encoded a second
  time. The other operators are.
- **Alpha is never tone-mapped.** From 3.2.0 the tone-map and expansion nodes
  split alpha off and reattach it untouched.

## Known limitation — read this before trusting the ACES label

The ACES 2.0 tone scale here is a log-space contrast of 1.55 with a tanh
shoulder. It is **not** the Daniele Evo curve the ACES 2.0 specification
defines. In practice 18% grey renders about 0.84 stop above the ACES 2.0
reference on the SDR path, and HLG diffuse white lands at signal 0.915 rather
than the 0.75 that BT.2408 specifies.

The normalisation defects around it were fixed in 3.2.0 — Cinema no longer
flat-lines above 0.4 scene-linear, and HLG no longer pins diffuse white to the
display peak — but the curve itself has not been replaced. Grade against a
reference you trust rather than against the label.
""",
    "VFX, Masks, Optics, and Multipass": """
Plate preparation, masks and roto, depth, camera and lens optics, motion,
multipass extraction, real AOV ingestion, and relighting.

## Typical graph

```text
Read plate → Plate Prep → Multipass Master (or AOV Reader) → Relight / Composite → Write
```

## Before you use these nodes

- **Estimated passes are estimates.** Multipass Master derives albedo,
  roughness, ambient occlusion and a segmentation ID from a single image. That
  is genuinely useful for generated footage and 2D plates. It is not a
  substitute for render passes, and the segmentation output is a clustered
  matte, not a Cryptomatte. For ground truth, feed a multilayer EXR through the
  Multipass AOV Reader.
- **Premultiplication.** The compositing nodes unpremultiply before operating
  and re-premultiply afterwards. If you are bringing in a matte from elsewhere,
  know which convention it uses.
- **Depth needs `transformers`.** Depth Anything V2 weights download on first
  use and are cached; the cache holds one tier at a time.

## Known limitation

Optical flow is a single-scale Lucas–Kanade solve despite the DIS reference in
its docstring. It recovers roughly the full displacement at 1 pixel, about 17%
at 3 pixels, and about 1% at 5 — so mask propagation and vector blur are
effectively static above a couple of pixels of motion. Use it for subtle
stabilisation, not for fast action.
""",
    "Pipeline and Studio": """
Project, shot and asset management, and handoff to Nuke and DaVinci Resolve.

## Typical graph

```text
Project Manager (organise) → work → Send to Nuke / Send to Resolve
```

## Before you use these nodes

- **The dashboards open in-canvas**, as an overlay on the ComfyUI graph rather
  than a separate browser tab. Launch them from the Project Manager node.
- **The Nuke bridge is local and structured.** It binds to `127.0.0.1` and
  accepts a fixed set of production actions. It does not evaluate arbitrary
  Python — that path was removed in 3.2.0 after it was found to be exploitable
  through its own blocklist.
- **Resolve handoff is a folder drop.** Radiance writes PNG, TIFF or EXR into a
  directory that Resolve imports; there is no live link.
- **Write first, send second.** Both bridges hand over media that already
  exists on disk.
""",
    "Review, Viewer, and Preview": """
The Viewer, the Lite Viewer, scopes, focus peaking, contact sheets, flipbooks,
frame stamps, QC and the preview server.

## Typical graph

```text
work → Radiance Viewer → (grade in the viewer) → RENDER → delivered master
```

## Before you use these nodes

- **The Viewer is also the delivery panel.** Grading done in the viewer is
  applied to the exported master. Before 3.2.0 six of those controls — shadows,
  highlights, hue, LUT and gamut compression — were silently dropped on export;
  they now reach the file, and a mismatch between the browser and the server is
  reported in the log. If you are upgrading, hard-refresh ComfyUI so the browser
  picks up the new JavaScript.
- **Scopes are computed on the ungraded texture.** They show you the incoming
  image, not your grade. That is deliberate, but it surprises people.
- **The Viewer always re-runs.** Its `IS_CHANGED` returns NaN so the delivery
  frame cache stays populated, which means it re-executes on every queue even
  when nothing upstream changed.
- **Keyboard shortcuts** are listed in the Viewer's own help overlay.
""",
    "Upscale": """
Image and video upscaling with HDR and colour awareness, tiling, and face
restoration.

## Typical graph

```text
Read → Upscale Image (tiled) → Write
```

## Before you use these nodes

- **Backends work in display-referred space.** Feeding scene-linear values
  straight in will not do what you want. Use the node's HDR and colour-encoding
  options so it encodes, upscales and decodes around the model.
- **Models download on first use** and are cached under your ComfyUI models
  directory. The first run of a tier is slow and needs network access.
- **Tiling is memory-safe by design.** Each tile is computed on the GPU and
  accumulated on the CPU, so peak VRAM is one tile regardless of output size.
- **`blend_mode`:** `gaussian_feather` and `linear` are genuinely different
  weightings. `laplacian_pyramid` is not implemented and falls back to the
  Gaussian feather, logging once when it does.

## Known limitations

The Tier 3 diffusion backend is fixed at 4× and ignores the `scale` widget; at
`scale = 2` the result is the top-left quarter of a 4× upscale. `mode =
"creative"` selects that tier regardless of the scale you asked for.
""",
    "Video": """
Video loading, prompt building, sampling, routing, batch decode, temporal
conditioning and export.

## Typical graph

```text
Video Loader → Video Prompt → Video Sampler → Video Batch Decode → Video Export
```

## Before you use these nodes

- **Frame batches are tensors, not streams.** A 240-frame 4K RGBA float batch
  is roughly 31 GB. Use `max_video_frames` and `proxy_scale` on Read while you
  are building the graph.
- **Export needs ffmpeg.** Radiance looks for it on PATH first, then falls back
  to the binary that `imageio-ffmpeg` installs, so a pip install is enough on
  Windows. Set `RADIANCE_FFMPEG` to point at a specific build.
- **Temporal windows overlap.** `overlap_temporal` controls the blend between
  processing windows; 1 is the minimum and produced black frames before 3.2.0.
""",
    "AI Assist": """
Scene-cut detection and shot splitting.

## Before you use these nodes

The detector normalises its scores by the batch maximum, so `threshold` has no
absolute meaning: the largest inter-frame difference in any clip is always 1.0
and therefore always exceeds the threshold. On footage with no cuts it will
still report cuts, spaced at `min_shot_frames`. Treat the output as a starting
point to be checked by eye, not as a decision.
""",
}

_TYPE_LABEL = {
    "IMAGE": "IMAGE", "MASK": "MASK", "LATENT": "LATENT", "MODEL": "MODEL",
    "CLIP": "CLIP", "VAE": "VAE", "CONDITIONING": "CONDITIONING",
    "INT": "INT", "FLOAT": "FLOAT", "STRING": "STRING", "BOOLEAN": "BOOLEAN",
}


def _clean(text) -> str:
    """One-line, table-safe."""
    if text is None:
        return ""
    text = str(text).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


#: Option lists ComfyUI owns rather than Radiance. Their contents depend on the
#: host install (and on which samplers a given ComfyUI build ships), so printing
#: them would make this page describe one machine instead of the node.
_RUNTIME_LISTS = (
    ("comfy.samplers", "SAMPLERS", "sampler names ComfyUI offers"),
    ("comfy.samplers", "SCHEDULERS", "scheduler names ComfyUI offers"),
    ("comfy.samplers", "KSAMPLER_NAMES", "sampler names ComfyUI offers"),
)


def _runtime_supplied(kind):
    """Return a stable description if `kind` is a ComfyUI-owned option list."""
    for mod_name, attr, what in _RUNTIME_LISTS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        try:
            if kind is getattr(mod, attr, None):
                return f"choice of the {what}"
        except Exception:  # pragma: no cover - defensive on exotic objects
            continue
    return None


def _describe_type(kind) -> str:
    runtime = _runtime_supplied(kind)
    if runtime:
        return runtime
    if isinstance(kind, (list, tuple)):
        options = [str(o) for o in kind]
        if not options:
            return "choice (empty)"
        shown = ", ".join(f"`{o}`" for o in options[:8])
        if len(options) > 8:
            shown += f", … (+{len(options) - 8} more)"
        return f"choice of {shown}"
    if not isinstance(kind, str):
        # Neither a type name nor a list: something the host fills in. Printing
        # its repr would embed a memory address and make the page differ on
        # every run.
        return "choice, supplied by the host at runtime"
    return f"`{_TYPE_LABEL.get(str(kind), str(kind))}`"


def _describe_range(config: dict) -> str:
    if not isinstance(config, dict):
        return ""
    bits = []
    if "min" in config and "max" in config:
        bits.append(f"{config['min']} – {config['max']}")
    elif "min" in config:
        bits.append(f"≥ {config['min']}")
    elif "max" in config:
        bits.append(f"≤ {config['max']}")
    if config.get("step") not in (None, 1):
        bits.append(f"step {config['step']}")
    if config.get("multiline"):
        bits.append("multiline")
    return ", ".join(bits)


def _default_of(kind, config: dict):
    if not isinstance(config, dict):
        config = {}
    if "default" in config:
        return config["default"]
    if _runtime_supplied(kind):
        return None
    if isinstance(kind, (list, tuple)) and kind:
        return kind[0]
    return None


def _input_rows(cls):
    try:
        spec = cls.INPUT_TYPES()
    except Exception as exc:  # pragma: no cover - a broken node fails elsewhere
        return [], f"INPUT_TYPES() raised {type(exc).__name__}: {exc}"

    rows = []
    for section, required in (("required", "Yes"), ("optional", "No")):
        for name, entry in (spec.get(section) or {}).items():
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            kind = entry[0]
            config = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            default = _default_of(kind, config)
            rows.append((
                name,
                required,
                _describe_type(kind),
                "" if default is None else f"`{_clean(default)}`",
                _describe_range(config),
                _clean(config.get("tooltip", "")),
            ))
    hidden = sorted((spec.get("hidden") or {}).keys())
    return rows, hidden


def _output_rows(cls):
    types = getattr(cls, "RETURN_TYPES", ()) or ()
    names = getattr(cls, "RETURN_NAMES", None) or types
    tips = getattr(cls, "OUTPUT_TOOLTIPS", None) or ()
    rows = []
    for i, kind in enumerate(types):
        rows.append((
            _clean(names[i]) if i < len(names) else f"out{i}",
            f"`{kind}`",
            _clean(tips[i]) if i < len(tips) else "",
        ))
    return rows


def _anchor(title: str) -> str:
    keep = [c.lower() if c.isalnum() else ("-" if c in " -_" else "") for c in title]
    return "".join(keep).strip("-")


def _source_of(cls) -> str:
    try:
        module = importlib.import_module(cls.__module__)
        path = pathlib.Path(inspect.getfile(module)).resolve()
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except Exception:
        return cls.__module__


def _node_section(key: str, cls, display: str) -> str:
    rows, hidden = _input_rows(cls)
    outs = _output_rows(cls)
    description = inspect.cleandoc(getattr(cls, "DESCRIPTION", "") or "")
    if not description:
        doc = inspect.cleandoc(cls.__doc__ or "")
        description = doc.split("\n\n")[0] if doc else ""

    out = [f"## {display}", ""]
    out.append(f"**Node key:** `{key}`  ")
    out.append(f"**Menu:** `{getattr(cls, 'CATEGORY', '—')}`  ")
    out.append(f"**Source:** `{_source_of(cls)}`  ")
    if getattr(cls, "OUTPUT_NODE", False):
        out.append("**Output node** — runs even with nothing connected downstream.  ")
    out.append("")
    if description:
        out += [description, ""]

    if isinstance(rows, list) and rows:
        out += ["### Inputs", "",
                "| Input | Required | Type | Default | Range | Description |",
                "| :--- | :---: | :--- | :--- | :--- | :--- |"]
        for name, req, kind, default, rng, tip in rows:
            out.append(f"| `{name}` | {req} | {kind} | {default} | {rng} | {tip} |")
        out.append("")
    elif isinstance(rows, str):
        out += [f"> Inputs could not be introspected: {rows}", ""]
    else:
        out += ["### Inputs", "", "This node takes no inputs.", ""]

    if hidden:
        out += [f"Hidden inputs supplied by ComfyUI: {', '.join(f'`{h}`' for h in hidden)}.", ""]

    if outs:
        out += ["### Outputs", "",
                "| Output | Type | Description |", "| :--- | :--- | :--- |"]
        for name, kind, tip in outs:
            out.append(f"| `{name}` | {kind} | {tip} |")
        out.append("")
    else:
        out += ["### Outputs", "",
                "None — this node is a terminal (it writes, sends, or displays).", ""]
    return "\n".join(out)


def _display_names(radiance):
    return radiance.NODE_DISPLAY_NAME_MAPPINGS


def build(radiance) -> dict:
    """Return {relative path: file text}."""
    classes = radiance.NODE_CLASS_MAPPINGS
    display = _display_names(radiance)

    by_file: dict = {}
    for key, cls in classes.items():
        category = getattr(cls, "CATEGORY", "") or ""
        suffix = category.rsplit("/", 1)[-1].strip()
        entry = GROUPS.get(suffix)
        if entry is None:
            entry = ("ai-assist.md", "AI Assist")
        by_file.setdefault(entry, []).append((key, cls))

    pages = {}
    for (filename, title), items in sorted(by_file.items(), key=lambda kv: kv[0][0]):
        items.sort(key=lambda kv: display.get(kv[0], kv[0]))
        intro = textwrap.dedent(GROUP_INTROS.get(title, "")).strip()

        lines = ["[← Back to Radiance docs](../README.md)", "", f"# {title}", ""]
        if intro:
            lines += [intro, ""]
        lines += [f"## Nodes in this section ({len(items)})", "",
                  "| Node | Key | What it does |", "| :--- | :--- | :--- |"]
        for key, cls in items:
            name = display.get(key, key)
            desc = _clean(inspect.cleandoc(getattr(cls, "DESCRIPTION", "") or "").split(". ")[0])
            lines.append(f"| [{name}](#{_anchor(name)}) | `{key}` | {desc} |")
        lines.append("")
        lines.append("---")
        lines.append("")
        for key, cls in items:
            lines.append(_node_section(key, cls, display.get(key, key)))
            lines.append("---")
            lines.append("")
        pages[f"docs/built-in-nodes/{filename}"] = "\n".join(lines).rstrip() + "\n"

    pages["docs/coverage.md"] = _coverage_page(by_file, display, classes)
    return pages


def _documentation_gaps(classes) -> list:
    """Nodes with no DESCRIPTION, or with no tooltip on any input.

    Both show up as blank cells in the generated tables. Naming them here means
    the gap is a visible, countable thing rather than something a reader only
    notices when they need the answer.
    """
    gaps = []
    for key, cls in sorted(classes.items()):
        has_desc = bool((getattr(cls, "DESCRIPTION", "") or "").strip())
        tips = 0
        total = 0
        try:
            spec = cls.INPUT_TYPES()
        except Exception:
            spec = {}
        for section in ("required", "optional"):
            for _name, entry in (spec.get(section) or {}).items():
                total += 1
                if (isinstance(entry, (list, tuple)) and len(entry) > 1
                        and isinstance(entry[1], dict) and entry[1].get("tooltip")):
                    tips += 1
        if not has_desc or (total and tips == 0):
            gaps.append((key, has_desc, tips, total))
    return gaps


def _coverage_page(by_file, display, classes) -> str:
    total = sum(len(v) for v in by_file.values())
    lines = [
        "[← Back to Radiance docs](README.md)", "",
        "# Node Coverage Ledger", "",
        "Generated from the live `NODE_CLASS_MAPPINGS` by `tools/generate_docs.py`.",
        "Dynamic `.gizmo` nodes are created at runtime and are not counted here.",
        "",
        "`tests/test_docs_coverage.py` fails if this drifts from the catalog — it",
        "did drift before, in both directions at once: nine node classes existed",
        "in the source and were never registered, and four registered nodes had",
        "never been added to this ledger.",
        "",
        "## Summary", "",
        "| Group | Count |", "| :--- | ---: |",
    ]
    for (filename, title), items in sorted(by_file.items(), key=lambda kv: kv[0][1]):
        lines.append(f"| [{title}](built-in-nodes/{filename}) | {len(items)} |")
    lines += [f"| **Total** | **{total}** |", ""]

    for (filename, title), items in sorted(by_file.items(), key=lambda kv: kv[0][1]):
        lines += [f"## {title}", ""]
        for key, _cls in sorted(items, key=lambda kv: display.get(kv[0], kv[0])):
            lines.append(f"- `{key}`")
        lines.append("")

    gaps = _documentation_gaps(classes)
    lines += ["## Documentation gaps", ""]
    if not gaps:
        lines += ["Every node has a description and at least one input tooltip.", ""]
    else:
        lines += [
            f"{len(gaps)} of {len(classes)} nodes are missing a `DESCRIPTION`, an",
            "input tooltip, or both. Those appear as blank cells in the reference",
            "tables above and as an empty hover in ComfyUI itself, so this list is",
            "the honest backlog rather than a silent gap.",
            "",
            "| Node | Has description | Tooltips |", "| :--- | :---: | ---: |",
        ]
        for key, has_desc, tips, total in gaps:
            lines.append(f"| `{key}` | {'yes' if has_desc else '**no**'} | "
                         f"{tips} of {total} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the docs are out of date instead of writing")
    parser.add_argument("--stubs", action="store_true",
                        help="load tests/conftest.py first, so this runs outside a "
                             "ComfyUI install (folder_paths, comfy.* and friends are "
                             "stubbed). Node metadata does not depend on the runtime, "
                             "so the generated reference is identical either way.")
    args = parser.parse_args()

    if args.stubs:
        sys.path.insert(0, str(ROOT / "tests"))
        import conftest  # noqa: F401

    import radiance

    if radiance._LOAD_RESULT.failures:
        missing = ", ".join(f.source.label for f in radiance._LOAD_RESULT.failures)
        print(f"refusing to generate: the catalog is incomplete ({missing}). "
              f"Install the missing dependencies first, or the docs will silently "
              f"omit those nodes.", file=sys.stderr)
        return 2

    pages = build(radiance)
    stale = []
    for rel, text in sorted(pages.items()):
        path = ROOT / rel
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            stale.append(rel)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {rel}")

    if args.check and stale:
        print("out of date: " + ", ".join(stale), file=sys.stderr)
        print("run: python tools/generate_docs.py", file=sys.stderr)
        return 1
    if not args.check:
        print(f"{len(pages)} page(s), {len(radiance.NODE_CLASS_MAPPINGS)} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
