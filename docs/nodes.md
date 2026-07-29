[← Back to Radiance docs](README.md)

# Node Reference

Every Radiance node, with every parameter — type, default, range and
description — generated directly from the live `NODE_CLASS_MAPPINGS` by
`tools/generate_docs.py`. It cannot go stale without a test failing.

## By group

| Group | Nodes | Covers |
| :--- | ---: | :--- |
| [IO and Delivery](built-in-nodes/io-delivery.md) | 6 | Read, Write, EXR multi-part, mask loading, DPX |
| [Generate, Loaders, and Sampling](built-in-nodes/generate.md) | 13 | Model loading, prompts, the Radiance Sampler, HDR VAE decode, LoRA |
| [Color](built-in-nodes/color.md) | 15 | Grade, CDL, LUTs, curves, colour-space conversion, QC |
| [HDR and ACES](built-in-nodes/hdr-aces.md) | 19 | ACES output transforms, OCIO, PQ/HLG encode, tone mapping, SDR→HDR |
| [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) | 25 | Plate prep, roto, depth, optics, motion, multipass, relight |
| [Pipeline and Studio](built-in-nodes/pipeline.md) | 4 | Project Manager, Nuke and Resolve handoff |
| [Review, Viewer, and Preview](built-in-nodes/review.md) | 11 | Viewer, Lite Viewer, scopes, contact sheets, flipbook, preview server |
| [Upscale](built-in-nodes/upscale.md) | 4 | Image and video upscaling, tiling, face restoration |
| [Video](built-in-nodes/video.md) | 14 | Video loading, sampling, routing, batch decode, export |

See the [Coverage Ledger](coverage.md) for the full alphabetical list, the group
totals, and the current documentation gaps.

## Reading a node page

Each entry gives:

- **Node key** — the identifier stored in a saved workflow. Stable across
  releases; renaming one would break existing graphs.
- **Menu** — where it sits under `FXTD STUDIOS/Radiance`.
- **Source** — the file it is implemented in.
- **Inputs** — name, required or optional, type, default, valid range, and the
  tooltip text shown in ComfyUI.
- **Outputs** — name and type.

Choice inputs list their options; where there are many, the first eight are shown
with a count of the rest. The **first** option of a choice is what a workflow
saves as its default, so it matters more than the others.

## Regenerating

```bash
python tools/generate_docs.py            # inside a working ComfyUI environment
python tools/generate_docs.py --stubs    # anywhere else
python tools/generate_docs.py --check    # CI: fail if out of date
```

The generator refuses to run if the catalog is incomplete, so a missing
dependency cannot silently delete a group of nodes from the documentation.
