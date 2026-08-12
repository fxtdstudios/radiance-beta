[← Back to Radiance docs](README.md)

# Viewer and Delivery

The Radiance Viewer is two things at once: a review tool and the delivery panel.
Understanding that it is both explains most of its behaviour.

---

## The two viewers

| | **Radiance Viewer** | **Radiance Lite Viewer** |
| :--- | :--- | :--- |
| Rendering | WebGL2, float textures | Canvas 2D |
| Scopes | Waveform, vectorscope, histogram, false colour | Histogram |
| Grading | Full panel, exports with the master | None |
| Delivery | Yes | No |
| Cost | Holds frames in a cache for export | Minimal |

Use the Lite Viewer when you just want to see the picture. Use the full Viewer
when you are judging colour or exporting.

---

## Reviewing

**Playback.** Frame-by-frame, loop, and a frame slider. The HUD shows the
resolution, format, bit depth, frame rate and colour space of what you are
looking at.

**Exposure and viewing transform.** These affect the *view only* — not the
export, unless you also set them in the grading panel. Use them to check
shadow and highlight detail without changing the image.

**Scopes.** Waveform for luminance distribution, vectorscope for hue and
saturation, histogram for clipping, false colour for exposure. They are computed
on the incoming texture, before your grade, so they tell you about the source.

**Focus peaking** highlights in-focus edges — useful for checking a render's
depth of field or a plate's critical focus.

**Compare.** A/B against a reference with a wipe.

---

## Grading in the Viewer

The grading panel applies, in order: exposure → white balance (temp/tint) →
lift/gamma/gain/offset → contrast → shadows/highlights → saturation → hue → LUT
→ gamut compression, plus the FX group (grain, bloom, halation, diffusion,
denoise).

**Everything in that panel is exported with the master.** The WebGL shader and
the Python export path implement the same chain in the same order. That is the
whole point of grading here rather than downstream.

> **If you are upgrading from 3.1.x, hard-refresh ComfyUI.** Before 3.2.0 six
> controls — shadows, highlights, hue, LUT name, LUT intensity and gamut
> compression — were read by the exporter and never sent by the browser, so they
> exported at their default while the viewer showed them applied. The fix is in
> both the Python and the JavaScript; a cached browser will send the old payload.
> When that happens the server now logs a warning naming the missing keys.

**Presets** save and recall a grade. **Curves** gives you a per-channel curve
editor.

---

## Delivering

Press **RENDER** in the delivery panel. Radiance then, in a background thread:

1. Applies the grade to every frame in the cached batch.
2. Applies the FX bake, if any.
3. Optionally upscales.
4. Encodes to the format you chose.
5. Writes sidecars — CDL, AMF, a `.json` metadata file, a thumbnail.
6. Appends an entry to `radiance_sessions.json` in your output directory.

The progress bar polls the server, so it moves during the export rather than
jumping from 0 to 100 at the end.

### Settings

| Setting | What it does |
| :--- | :--- |
| `filename` / `path` | Where it goes. Versioning appends `_v0001` and increments. |
| `format` | Video (MP4/ProRes), image sequence, or single frame. |
| `colorSpace` | The display encode applied on the way out. |
| `aspect_ratio` | Letterboxes or pillarboxes to fit. It does not crop. |
| `range_in` / `range_out` | Frame range. `range_out = 0` means "to the end". |
| `fps` | Frame rate for video containers. |
| `quality` | CRF for H.264/H.265; ignored for lossless formats. |
| `upscale_2x` | Runs the upscaler before encoding. |
| `smart_versioning` | Finds the next free version rather than overwriting. |

### What gets written alongside

- **`.cdl`** — the ASC CDL of your grade, for handoff to a grading system.
- **`.amf`** — ACES Metadata File describing the colour pipeline.
- **`.json`** — the full grade, settings, resolution, frame count and QC result.
- **Thumbnail** — a JPEG contact frame.

Each sidecar is written independently, so a failure in one does not silently
abort the others.

---

## Versioning and overwrite

With `smart_versioning` on, Radiance finds the highest existing `_vNNNN` and
writes the next one. With it off, `overwrite` decides whether an existing file
is replaced.

> **A note for anyone who used 3.1.x.** Versioned writes could destroy the
> previous version. The path builder used `Path.with_suffix()`, which replaces
> everything after the *last dot* — so `sh010.comp_v0001` and
> `sh010.comp_v0002` both resolved to `sh010.exr`. Dotted stems are routine in
> VFX naming, and overwrite defaults to on, so each render replaced the last
> approved one and reported success with the truncated path. Fixed in 3.2.0. If
> you have a shot history that looks thinner than it should, this is why.

---

## Performance notes

**The export runs off the event loop.** Grading, filters, the upscale and the
ffmpeg call all happen in a worker thread, so ComfyUI's websocket keeps
responding — the progress bar, node highlighting and queue view stay live. (In
3.1.x they ran inline and froze the UI for the whole export.)

**The frame cache is bounded by bytes, not frames.** Default 2 GB, override with
`RADIANCE_VIEWER_CACHE_BYTES`. Entries are detached CPU copies, so a CUDA image
does not pin VRAM invisibly behind ComfyUI's model manager.

**The Viewer always re-runs.** Its `IS_CHANGED` returns NaN on purpose, so the
delivery cache always holds the current frames. The cost is that a graph
containing a Viewer re-executes it on every queue even when nothing upstream
changed, and rewrites its sidecar files each time.

**ffmpeg.** Radiance looks on PATH first, then falls back to the binary that
`imageio-ffmpeg` installs — so on Windows, `pip install imageio-ffmpeg` is
enough and you do not need a system install. Point `RADIANCE_FFMPEG` at a
specific build if you need a particular codec set.

---

## Troubleshooting

| Symptom | Cause |
| :--- | :--- |
| Export does not match the viewer | Cached browser JavaScript. Hard-refresh; check the log for a missing-grade-key warning. |
| Scopes are blank | Fixed in 3.2.0 — the float read-back did not match the buffer format. Older versions show this on float pipelines. |
| "EXPORT COMPLETE" but no file | Check the log; the write path raises rather than failing silently, so there will be a traceback. |
| Export is very slow | `upscale_2x` on a long batch. The upscaler is the slow part, not the encode. |
| Viewer re-runs constantly | Expected — see above. |
| Progress bar sits at 0 then jumps | You are on 3.1.x. Fixed in 3.2.0. |
| Frames look black at window boundaries | `overlap_temporal = 1` on the video upscaler in 3.1.x. Fixed in 3.2.0. |

---

## Related

- [Colour Management](color-management.md) — what to set for each deliverable.
- [Review nodes reference](built-in-nodes/review.md) — every parameter.
- [Known Limitations](limitations.md).
