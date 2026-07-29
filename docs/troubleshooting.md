[← Back to Radiance docs](README.md)

# Troubleshooting

Ordered roughly by how often each one happens. Every entry says what you see,
why, and what to do.

---

## Install and startup

### Radiance nodes do not appear in the menu

Check the ComfyUI console at startup. Radiance always logs its result:

```
Radiance: successfully loaded 109 nodes (v3.2.0)
```

If that line is missing entirely, the package did not import at all — usually the
wrong Python environment. Radiance must be installed into the *same* environment
ComfyUI runs in, not a system Python.

### "loaded 59 of at least 109 expected nodes"

Some node groups failed to import. The lines above name each one and its
exception:

```
ERROR  Radiance: 5 node module(s) failed to import; every node they export is
       missing from ComfyUI.
ERROR    - radiance.nodes.color: ModuleNotFoundError: No module named 'folder_paths'
ERROR    - radiance.nodes.generate: ModuleNotFoundError: No module named 'tqdm'
```

Install the named dependency and restart. `RADIANCE_LOG_LEVEL=DEBUG` adds full
tracebacks.

> Before 3.2.0 this printed "successfully loaded 12 nodes" with no comparison, so
> an 88% shortfall read exactly like a clean start. If you are on an older
> version and nodes are missing, that is why nothing told you.

### A specific node is missing but others loaded

Its group failed, or it needs an optional dependency. The startup dependency
table shows what is present:

```
│ OpenEXR     │ Required │ [X] Missing │ EXR file support │ pip install OpenEXR     │
│ OpenImageIO │ Optional │ [!] Missing │ DPX read/write   │ pip install OpenImageIO │
```

### `ModuleNotFoundError: No module named 'radiance'`

The checkout must be importable as `radiance`. If you cloned `radiance-beta`,
rename the folder to `radiance`.

---

## Colour

### The image looks flat, washed out and grey

A log file decoded as pass-through. Set `color_space` on Read to the actual
camera curve — LogC4, S-Log3, V-Log, Log3G10 and so on.

### The image looks crushed and over-contrasty

The opposite: a linear file decoded as if it were log or sRGB. Use
`Auto / Linear (pass-through)` for a render EXR.

### Highlights are flat above a certain point

Something clipped to 1.0:

- An 8-bit or PNG round trip somewhere in the graph.
- A tone map applied earlier than you thought.
- On 3.1.x, legal-range limiting was ungated, so exporting a 32-bit EXR squeezed
  every pixel into [16/255, 235/255]. Fixed in 3.2.0.

### Neutrals have a colour cast

- A primaries conversion applied twice, or without white-point adaptation.
- On 3.1.x, AgX applied its inset matrix untransposed and tinted every neutral —
  measured channel spread 0.042 at 18% grey, growing with exposure. Fixed in
  3.2.0.

### An ACES output does not match another tool

Expected, and documented. The ACES 2.0 tone scale here is not the Daniele Evo
curve — 18% grey sits about 0.84 stop above the reference on SDR. See
[Known Limitations](limitations.md). For certified output, do the output
transform in OCIO with an official config.

### Changing `chromatic_adaptation` does nothing

Correct — the adaptation is baked into the precomputed matrices. The node logs a
warning when you set it to a non-default value.

---

## Alpha and mattes

### Alpha comes back semi-transparent after a tone map

3.1.x tone-mapped and gamma-encoded alpha along with RGB: an opaque pixel came
back at 0.730, a 50% matte at 0.607. Fixed in 3.2.0.

### Matte edges darken after a blur or resize

Premultiplied RGB filtered without unpremultiplying first. Radiance's
compositing nodes handle this; doing it by hand, unpremultiply → operate →
re-premultiply. See [Concepts](concepts.md).

### The EXR I wrote has no alpha

`mask` must be connected **and** the format must be an EXR variant. It is ignored
for PNG, JPEG and video.

---

## Reading and writing

### "RadianceWrite: no output filename"

Both `output_path` and `filename` are blank. Give `filename` a value, or make
`output_path` a full path including the stem.

### An earlier version of my render disappeared

On 3.1.x, versioned writes could destroy the previous version: the path builder
used `Path.with_suffix()`, which replaces everything after the *last dot*, so
`sh010.comp_v0001` and `sh010.comp_v0002` both resolved to `sh010.exr`. Dotted
stems are normal in VFX naming and overwrite defaults to on. Fixed in 3.2.0.
There is no recovery for files already lost.

### An overscan EXR comes back offset and the wrong size

The reader uses the data window and ignores the display window. Crop in your
comp application first. See [Known Limitations](limitations.md).

### A directory or glob sequence pattern reads nothing

`start_frame` is treated as a list index for `/dir/` and `*.exr` patterns, and
the widget defaults to 1001 — fewer than 1001 files gives an empty list. Set
`start_frame` to 0, or use a `%04d` / `####` pattern, where it is a frame number.

### DPX write fails

`pip install OpenImageIO`. There is no Pillow plugin for DPX.

---

## Video and ffmpeg

### Video export fails with `FileNotFoundError`

ffmpeg was not found. Radiance checks `RADIANCE_FFMPEG`, then PATH, then the
binary `imageio-ffmpeg` installs. On Windows, `pip install imageio-ffmpeg` is
usually the fastest fix.

> On 3.1.x every call used the bare string `"ffmpeg"` from PATH and never used
> the bundled binary, so this failed on most Windows installs despite a working
> ffmpeg sitting in site-packages.

### Frames are fully black at regular intervals

`overlap_temporal = 1` on the video upscaler in 3.1.x. The blend ramp was zero at
the shared frame, from both windows. Measured at `B=100, window=16, overlap=1`:
frames 15, 30, 45, 60, 75 and 90. Fixed in 3.2.0.

### Running out of RAM on a long clip

A frame batch is the whole clip in memory — 240 frames of 4K RGBA float32 is
about 31 GB. Use `max_video_frames` and `proxy_scale` on Read while building,
and process in chunks for the final run.

---

## Viewer and delivery

### The export does not match what the Viewer showed

Hard-refresh ComfyUI so the browser picks up the current JavaScript. If the
browser is stale, the server logs:

```
WARNING  The delivery payload is missing 6 grading key(s): shadows, highlights,
         hue_shift, lut_name, lut_intensity, gamut_compression. Those controls
         will export at their default value even if they are set in the viewer.
```

On 3.1.x this happened silently — those six controls were read by the exporter
and never sent by the browser.

### Scopes are blank

Fixed in 3.2.0. The scope buffer is created at pipeline precision but the
read-back was hard-coded to `UNSIGNED_BYTE`, which WebGL2 rejects for a float
attachment.

### ComfyUI freezes during an export

3.1.x ran the whole export on the event loop. Fixed in 3.2.0 — it runs in a
worker thread and the progress bar updates live.

### The Viewer re-runs every time I queue

By design. Its `IS_CHANGED` returns NaN so the delivery frame cache always holds
current frames.

---

## Performance

### Upscaling takes minutes per frame

On 3.1.x every built-in upscaler ran on the CPU, because it selected
`images.device` and a ComfyUI IMAGE is always CPU-resident. One 4K plate at 4× is
roughly 66 TFLOP — about ten minutes against ~5 seconds on a GPU. Fixed in 3.2.0.

### The Multipass Master node fails on 4K

`torch.quantile` refuses inputs above 2²⁴ elements per row, and a UHD frame is
24,883,200. Fixed in 3.2.0.

### VRAM fills and the sampler OOMs for no clear reason

Model caches. Radiance's are bounded and move evicted modules back to the CPU;
`RADIANCE_CACHE_SIZE` shrinks them further, `RADIANCE_VIEWER_CACHE_BYTES` bounds
the viewer frame cache (default 2 GB).

### ComfyUI stalls when I open the Project Manager

3.1.x walked the entire output tree and unzipped every `.rad` inline on the event
loop. Fixed in 3.2.0.

---

## Sampling

### Guidance rescale seems to do nothing

On 3.1.x, if an SDR reference or energy mask was also connected, the second CFG
patch overwrote the first — rescale was silently discarded while the log still
said it had been applied. Fixed in 3.2.0.

### A previous run's settings seem to affect the next one

On 3.1.x the sampler patched the loader's *cached* model in place whenever
`cfg <= 1.0` (the Flux default), so the patch persisted into every later queue
and every other branch fed from that MODEL. Fixed in 3.2.0. Restart ComfyUI to
clear a model already patched by an older version.

### PAG at scale 0.1 and 5.0 look identical

On 3.1.x, `pag_scale` was an on/off gate and the patch was a complete no-op.
Fixed in 3.2.0 — but note it remains a partial implementation, see
[Known Limitations](limitations.md).

---

## DCC handoff

### The Nuke bridge does not respond

The listener binds to `127.0.0.1` and must be started inside Nuke first:

```python
exec(open("/path/to/ComfyUI/custom_nodes/radiance/scripts/start_nuke_server.py").read())
```

It accepts a fixed set of structured production actions. It does **not** evaluate
arbitrary Python — that path was removed in 3.2.0 after it was found to be
exploitable through its own blocklist. If you relied on sending arbitrary
expressions, that is why it stopped working.

### DaVinci Resolve does not see the media

Resolve handoff is a folder drop, not a live link. Write the media first, then
import the folder in Resolve.

---

## Still stuck

Open an issue at
[github.com/fxtdstudios/radiance/issues](https://github.com/fxtdstudios/radiance/issues)
with:

- The Radiance version and the startup line from the console.
- The node, its widget values, what you expected and what you got.
- Any ERROR or WARNING lines from the console.
- Output with `RADIANCE_LOG_LEVEL=DEBUG` if it is an import or silent-failure
  problem.
