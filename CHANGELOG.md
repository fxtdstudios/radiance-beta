# Changelog

All notable changes to FXTD Radiance will be documented in this file.

## [Unreleased]

### Added

- **The emitted WGSL is compiled and compared against the JS it comes from.**
  `radiance_grade.js` emits GLSL and WGSL from the same file as its JS
  functions; only the GLSL half was ever compiled, because headless Chromium
  has no `navigator.gpu` at all -- absent rather than blocked, under every flag
  combination tried, while WebGL2 works in the same browser. Deno ships WebGPU
  and Mesa's lavapipe provides a software Vulkan device, so the WGSL now runs
  the same 60-case matrix as the GLSL, on a runner with no GPU. Worst deviation
  3.1e-7 against a 2e-6 tolerance. Reverting the WGSL to any of the three ways
  the WebGPU path used to disagree with WebGL -- flat lift, unguarded gamma,
  power-form contrast -- turns the suite red.
- **The viewer panels are built and operated in a browser.** The panel code was
  held by source assertions, which cannot tell you a panel builds; the week the
  Viewer node rendered with no UI, every text-level check passed. The panels
  now load against stubbed ComfyUI modules, get built, and get driven --
  selects changed, toggles clicked -- with the instance state, the persisted
  setting and the control label all checked afterwards.

### Fixed

- **The safe-area caption no longer describes the previous preset.** Switching
  to Legacy 480-line redrew the boxes at 90/80 correctly but left the note
  beneath reading "SMPTE ST 2046-1 and EBU R 95 specify the same two boxes".
  The guide was right and the standard named under it was wrong, which in a QC
  overlay is the worse half. Found by driving the panel, not by reading it.

## [3.4.0] - 2026-08-17

A Viewer release. Everything below is in the Viewer unless it says otherwise.

**Upgrade note.** Two defaults change.

**WebGPU is now opt-in.** It used to upgrade automatically wherever the browser
exposed `navigator.gpu`, so nobody chose it — and the backend it switched people
to is the one that does not implement masks, qualifiers, the HDR heatmap or
OpenColorIO, and whose grade maths differed from WebGL on lift, gamma and
contrast. Defaulting to the backend with the missing features, then explaining
the gaps with in-panel banners, is a worse product than defaulting to the one
that works. View → Framing & Guides → Backend turns it back on, and states what
stops working when you do.

**Safe-area boxes move.** The Viewer drew the modern 93% action box against the
legacy 80% title box — a pairing no published standard specifies. The title box
is now 90%. If you have been placing graphics against the old inner box they
were inset by 10% for a delivery that asks for 5%.

### Added

- **OpenColorIO 2.5 config support.** Load a show's config and the Display and
  View menus come from it; picking a view changes the picture, through OCIO's
  own generated GLSL. Bundled ACES 1.3 and 2.0 configs for anyone without a
  config of their own. With nothing loaded the built-in ACES 1.3 pipeline runs
  exactly as before — OCIO is a capability, not a dependency. Real OpenColorIO
  compiled to WebAssembly, vendored under `js/vendor/ocio/` (BSD-3-Clause).
- **Pixel probe** (PROBE tab). Cursor, region and full-frame sampling; source or
  rendered values; RGBA, luminance, EV, cd/m², HSV and hex; min, max, mean and
  median per channel. NaN, Inf and negative counts are excluded from every
  statistic and reported separately.
- **Scope scales.** 10-bit and 12-bit code value, percent, millivolts, and nits
  for ST.2084 and HLG, with Data/Video levels and a switch for whether the
  scopes measure before or after the viewer colour transforms. Every graticule
  line carries its number; the panel states what is being measured. IRE is
  deliberately not offered.
- **HDR heatmap**, banded to ITU-R BT.2408 with a white line at 200–206 nits.
- **Safe areas labelled with their standard**, an **aspect-ratio matte**,
  **nearest-neighbour magnification on `N`**, and **frames / seconds / timecode**.

### Fixed

- The **HDR heatmap did not exist**: "HDR Heatmap" and "False Color" were one
  toggle under two menu labels, and the map read display luma on ARRI stops.
- **The grade was implemented four times** — WebGL GLSL, WebGPU WGSL, the WebGPU
  CPU readback and the `.cube` export — and no two agreed on lift, gamma or
  contrast. The CPU readback used a different contrast curve entirely and
  produced NaN at pivot 0. All four are now emitted from `js/radiance_grade.js`.
- The scopes' **"HDR ruler" placed its nit lines with a Reinhard curve** as a
  stand-in for the display transform, while the pixels had come through the
  actual ACES transform: "203 nit" drew at half height regardless.
- **Safe areas were drawn on the canvas, not the picture** — at any zoom or pan
  other than an exact fit they measured the viewport.
- Scene-linear black rendered as **NaN or ~1e16** through OCIO views
  (`pow(0, y≤0)` is undefined in GLSL), and every HDR view rendered **solid
  black** (`OES_texture_float_linear` must be enabled, not merely supported).
- `load_trained_turbo_decoder()` raised **NameError on every call** — the
  decoder classes were imported in `train()` and nowhere else.

### Testing

240 JavaScript tests, including two headless-WebGL2 harnesses that compile the
real shaders and compare them against the CPU implementations they were
generated from. Both defects in the OCIO list above were found by rendering, not
by reading — each compiled cleanly and reported nothing.

## [3.3.0] - 2026-08-16

**Upgrade note — read before updating mid-project.** Five changes alter what an
unchanged graph does. None is a preference; each corrects something that was
measurably wrong. But if you are partway through a job, finish it on 3.2.1.

1. **ACES 2.0 output moves.** 18% grey now lands where the standard puts it —
   10.000 nits at a 100-nit peak, 14.512 at 1000 — instead of ~18 nits
   everywhere (legacy transform) or 10% of peak (Daniele Evo node, which put
   grey at 400 nits on a 4000-nit master). In signal terms: sRGB 0.3492 rather
   than 0.4610, PQ 0.3298 rather than 0.3478. **Re-check any master graded
   against the old midtone.** HLG is unchanged — it is anchored to BT.2408.
2. **Scene-cut `threshold` is now `cut_confidence`**, on a calibrated 0–1 scale
   that means the same thing for `histogram`, `edge` and `combined`: 0.5 is "as
   different as a hard cut", lower is more sensitive. It was a fraction of the
   clip's own maximum, then briefly an absolute inter-frame distance whose
   useful value differed by method. The widget was renamed each time
   deliberately — a saved value would otherwise have been silently
   reinterpreted. Existing workflows reset to the new default and need
   re-tuning. The `edge` metric itself changed with it; see Fixed.
3. **Model downloads now refuse by default.** Set `RADIANCE_ALLOW_DOWNLOADS=1`
   to restore automatic fetching. Previously a first queue could pull 67 MB to
   2.4 GB with no prompt.
4. **Tier-3 upscale at 2x returns a different image** — the whole frame rather
   than the top-left quarter of a 4x render.
5. **CDL Export and Flipbook GIF write to `output/`** instead of resolving
   their relative defaults against the ComfyUI install directory.
6. **The node menu gains twenty entries.** Twenty finished nodes had never been
   published; nothing you already had moves or changes. See Added.
7. **Chained Energy Mask nodes now stack.** Two of them in series, or two
   branches joined by Conditioning Combine, previously threw one mask away
   silently. They add now. A graph with two Energy Masks will sample
   differently — and, for the first time, the way it looks like it should.


### Added

- **Twenty nodes published that had never reached the menu.** Every node group's
  `__init__.py` hand-copied a selection of its modules' node keys into the
  mapping ComfyUI reads, and nothing compared the two. Whatever nobody
  remembered to copy did not exist as far as ComfyUI was concerned — not
  broken, not disabled, simply invisible. Seventeen complete nodes were in that
  state, plus three compatibility alias keys:

  `RadianceHDRTurboEncoder`, `RadianceHDRPerChannelNorm`,
  `RadianceHDRPerChannelDenorm`, `RadianceACESMetadataFile`,
  `RadianceACES2Compliance`, `RadianceColorSpaceInfo`,
  `RadianceLuminanceGuidance`, `RadianceHDRBlendValidator`,
  `RadianceHDRAnalysis`, `RadianceNDISender`, `RadianceShotGradeRouter`,
  `RadianceAudioCut`, `RadianceAudioTranscribe`, `RadianceLinearCheck`,
  `RadianceCinemaStudio`, `RadianceCameraSync`, `RadianceVideoPromptBuilder`.

  This is the third time the same defect has surfaced — issue #40 was a sampler
  feature no node could reach, and `RadianceGradeApply` was a finished node
  whose only trace was a menu-section override. `nodes/aggregate.py` inverts
  the default: a group now sweeps its own modules, so writing the node is
  enough to ship it, and withholding one takes a named `WITHHELD_NODES` entry
  that a test reads back. Catalog 111 → 131.

- **Compatibility aliases load without cluttering the menu.**
  `RadianceImageLoader`, `RadianceControlApply` and `RadianceWorkspace` are
  older keys for nodes that still exist. They ship as `DEPRECATED` subclasses:
  a workflow saved against the old key opens normally, and node search shows
  one entry per node rather than two. (Subclasses because `DEPRECATED` is read
  off the class — pointing both keys at the same class object would have hidden
  the canonical node too.)

- **`RadianceGradeApply` reaches the menu**, published as **Bake Viewer Grade**.
  It bakes the Viewer's grading maths into the tensor — a complete node with 16
  documented inputs and a valid contract, listed in `viewer.py`'s own mappings
  since v3, but never transcribed into `nodes/monitor/__init__.py`. It was
  found because `nodes/branding.py` carried a menu-section override for it:
  someone had classified a node nobody could place. Catalog 110 → 111.

  Its label also settles the `Grade` / `Grade Apply` / `Apply Grade Info`
  overlap — the menu now reads Grade, Grade Match, Apply Grade Info and Bake
  Viewer Grade.

- **Menu sections are declared, not guessed.** `NODE_SECTIONS` in
  `nodes/branding.py` names the section for all 111 registered nodes.
  `classify_menu_section`'s keyword rules ran in order and returned on the
  first match, so `"mask"` was tested for VFX several rules before
  `"conditioning"` was tested for Generate — which is how `RadianceEnergyMask`,
  a node that feeds the sampler, landed in the VFX menu. The rules remain only
  as a fallback for undeclared nodes, and they now log a warning when they run.
  `tests/test_menu_sections.py` fails if a registered node is missing from the
  table, so the next one is caught at commit time rather than by a user
  hunting for their node.


- **Energy Mask node (`RadianceEnergyMask`).** Energy-Prioritized Sampling has
  been in the sampler since v3.1, but nothing ever wrote the
  `radiance_energy_mask` key it reads, so no graph could reach it (#40). This
  node writes it: connect CONDITIONING and a MASK, set `priority`, and the
  masked region samples at an effective `cfg x (1 + priority)` while the rest
  of the frame keeps the sampler's `cfg`. Negative priority softens a region
  instead. Menu: Generate.

### Fixed

- **EPS crashed on video latents.** The reader unpacked `B, C, H, W` from the
  denoised prediction, which raises `ValueError` on the 5-D `(B, C, T, H, W)`
  latents WAN, Hunyuan and LTX produce. The mask is now aligned rank-agnostically
  and broadcasts across frames. Nobody had hit this because the feature was
  unreachable.
- **EPS crashed on a mask batch that was neither 1 nor the latent batch.**
  `mask.expand(B, ...)` cannot broadcast 3 -> 4; it now falls back to the first
  mask rather than raising mid-sample.
- The resized mask is cached per latent geometry instead of being
  re-interpolated on every sampling step.
- `RadianceEnergyMask` is pinned to the Generate menu section; the keyword
  classifier would otherwise file anything named "...Mask" under VFX, away from
  the sampler it feeds.
- **Chaining two Energy Mask nodes lost one of the masks, silently.** In series,
  the second `attach_energy_mask` overwrote the key on every conditioning entry,
  so the downstream node won; joined by Conditioning Combine, the sampler
  stopped at the first entry carrying the key, so the upstream node won. The
  node's own docstring described only the second case, and got it backwards for
  the first. Layers accumulate now and the sampler sums `priority x mask` across
  all of them, with the combined modifier floored at 0 so a stack of negative
  priorities suppresses guidance instead of inverting it.

- **The scene-cut `edge` metric had no meaningful threshold at all.** It was a
  mean absolute difference of gradient magnitudes, which scales with the
  footage's own contrast and texture rather than with how different two frames
  are. Measured on one synthetic cut, re-graded:

  | contrast | absolute (old) | relative (new) |
  | --- | --- | --- |
  | 100% | 0.0369 | 0.2950 |
  | 50%  | 0.0185 | 0.2950 |
  | 25%  | 0.0092 | 0.2950 |

  The same cut, four times fainter. A threshold tuned on a bright exterior
  found nothing in a dim interior, and grain on a detailed frame (0.039)
  outscored a real cut in soft content (0.003). It is a relative (Bray-Curtis)
  distance between edge maps now, bounded in [0, 1], with a 5 px pre-blur so a
  high-pass metric stops measuring film grain. `combined` was the worst
  casualty — `0.6 * histogram + 0.4 * edge` added two quantities with no shared
  unit, so the nominal 40% edge contribution was a couple of percent; both
  terms are calibrated confidences now and the weights mean what they say.
  `KNOWN_ISSUES.md` records what the edge method still cannot see.

### Fixed

- **Nodes wrote into the ComfyUI install directory.** CDL Export defaulted its
  `file_path` to `grading/shot_01_output.cdl` and Flipbook GIF defaulted
  `save_path` to `preview/flipbook.gif` — bare relative paths resolved with
  `os.path.abspath()`, which anchors to the process working directory. For
  ComfyUI that is the install root, so running either node on its default
  created `grading/` and `preview/` folders inside the ComfyUI installation
  instead of writing to `output/`. Both now route through the new
  `resolve_output_path()`; relative paths land under `output/`, absolute paths
  are honoured as typed, and `..` traversal is rejected. CDL Import gained the
  matching `resolve_input_path()`, which searches `input/` then `output/` so a
  CDL you just exported is found by the stock widget value.
- **`INPUT_TYPES()` created a directory.** `RadianceLUTApply.get_lut_files()`
  called `os.makedirs(models/luts)` from inside the widget enumerator, which
  ComfyUI invokes for every node at startup and on every `/object_info`
  request. It now reports "No LUTs found" and leaves the filesystem alone.
- **`models_dir` was checked for truthiness, not type.** Anywhere
  `folder_paths.models_dir` is joined, a non-string value now falls back to
  `~/.cache/radiance` instead of being coerced into a path. Affects
  `nodes/upscale`, `nodes/vfx/multipass`, `temporal_rudra`, `radiance_ocio`
  and `color/lut`.
- **`RADIANCE_LOG_LEVEL` did nothing.** The startup shortfall error has told
  users to "re-run with RADIANCE_LOG_LEVEL=DEBUG for tracebacks" since 3.2.0,
  but nothing read the variable — `setup_radiance_logging()` was always called
  at INFO. It now resolves the level from the environment (level name or
  number, falling back rather than raising on garbage), so the advice printed
  by the error message is true.

### Fixed

- **Both ACES 2.0 tone scales now hit the published reference.** Radiance
  shipped two of them and they disagreed by up to 24x.
  `RadianceACES2Tonescale` used the real Daniele Evo curve but pinned 18% grey
  to a flat 10% *of peak*, so grey tracked the display: 10 nits at SDR, 100 at
  1000, 400 at 4000. `RadianceACES2OutputTransform` used a log-contrast + tanh
  approximation that held grey at ~18 nits on every peak — right in kind,
  ~0.85 stop bright at SDR.

  There was no judgement call: ACES 2.0 publishes the mapping. An ACES value of
  0.18 lands at 10.000 / 13.193 / 14.512 / 15.747 / 16.824 nits at peaks of
  100 / 500 / 1000 / 2000 / 4000 — the midtone barely moves while the
  highlights extend. `hdr/tonescale.py` carries that table; the Daniele Evo
  parameterisation derives its grey target from it, and the legacy transform
  solves an input gain that puts grey on the same value while keeping its
  highlight roll-off.

  **Upgrade note.** SDR and PQ output changes: 18% grey encodes to sRGB 0.3492
  instead of 0.4610, and to PQ 0.3298 instead of 0.3478. Both new values are
  independently derivable — 0.3492 is the sRGB encode of 10 nits, 0.3298 is the
  PQ signal for 14.512 nits. Re-check any master graded against the old
  midtone. HLG is deliberately unchanged: it is anchored to BT.2408 reference
  grey (signal 0.38, ~26 nits) and passes a synthetic peak into the curve, so
  the ACES table does not apply to it.

- **Scene-cut thresholds were relative to the batch.** `detect_cuts` divided
  every clip's scores by that clip's own maximum before comparing to
  `threshold`, so the largest inter-frame delta in *any* footage was forced to
  1.0. The threshold widget meant something different on every clip, and
  cut-free footage always reported a cut. The comparison is now against the raw
  distance. `_make_plot` was already comparing raw scores, so the rendered plot
  and the returned cut list could contradict each other on the same input; they
  now agree. Note the scales differ by method — histogram runs 0–2, edge runs
  roughly 0–0.5 — which the normalisation had been hiding.
- **Tier-3 upscale ignored `scale`.** The diffusion backends are fixed 4x, and
  nothing conformed their output, so `tiled_upscale` cropped a 4x render down
  to the requested 2x size — the user got the top-left quarter of the frame
  (measured correlation with the input: 0.0024). The output is now resampled to
  the requested ratio.
- **Optical flow was single-scale Lucas–Kanade.** Brightness constancy was
  linearised as `It = f2 - f1`, valid only within a pixel or two, so mask
  propagation was effectively static on real motion: ~100% recovery at 1 px,
  17% at 3 px, 1% at 5 px. It is now coarse-to-fine over an image pyramid,
  warping by the accumulated flow and solving for the residual at each level.
  1–5 px displacements now recover to within 10%, with 84–100% of the field
  inside half a pixel. Above ~8 px is still unreliable.
- **Model weights downloaded without asking.** Real-ESRGAN, HAT-L, SwinIR,
  Depth Anything V2 and DSINE — 67 MB to 2.4 GB — began fetching the moment a
  graph was queued. `nodes/upscale` had an opt-*out*
  (`RADIANCE_UPSCALE_OFFLINE=1`); `nodes/vfx/multipass` had no gate at all. Both
  now go through `radiance.core.consent`, which defaults to refusing with a
  message naming the file, its size and where to put it. Set
  `RADIANCE_ALLOW_DOWNLOADS=1` to permit them; the legacy opt-out is still
  honoured.

### Removed

- **Six internal write-ups.** `AUDIT.md`, `FULL_AUDIT_2026-08.md`,
  `BATCH3_REPORT.md`, `RELEASE_REVIEW.md`, `VIEWER_REPORT.md` and
  `PACKAGE_REVIEW.md` were point-in-time reports that had already drifted from
  the code — several listed as open bugs that had been fixed months earlier,
  which is how the tiled-VAE blend stayed on the backlog after it was
  corrected. What matters lives in the README's Status & to-do list,
  `KNOWN_ISSUES.md` and this changelog; `.gitignore` now keeps their filenames
  out.
- **`gpu_acceptance_report.md` is no longer tracked.** `tools/gpu_acceptance.py`
  writes it on every run — a generated artifact that should never have been
  committed. Now ignored.

- **The legacy `nodes_*.py` layer.** Forty root modules were pure re-export
  shims over `nodes/`; they are deleted. Six were still the real home of their
  code and moved into the package, which also reverses a dependency that ran
  backwards — the organized `nodes/` package had been importing *from* the
  layer it was introduced to replace:

  | was | is |
  | --- | --- |
  | `nodes_io.py` | `nodes/io/write.py` |
  | `nodes_sampler.py` | `nodes/generate/sampler.py` |
  | `nodes_loader.py` | `nodes/generate/loader.py` |
  | `nodes_workspace.py` | `nodes/pipeline/workspace.py` |
  | `nodes_realtime_preview.py` | `nodes/monitor/realtime.py` |
  | `nodes_gizmo.py` | `nodes/gizmo.py` |

  `radiance.nodes_*` imports no longer resolve. Node keys are untouched, so no
  saved workflow is affected — every one of the 111 pre-existing keys was
  compared class-by-class and widget-by-widget against a snapshot taken before
  the move. The package also drops to a single entry point: the separate
  `.nodes_radiance_viewer` spec was publishing `RadianceViewer` alongside
  `radiance.nodes.monitor`, the one genuine double-import the layer had.
- **Dead `.comfyignore` entries.** Thirty-two of its paths no longer existed. A
  packaging exclusion list naming files that are already gone reads as
  protection it is not providing, so the file was rewritten around what is
  actually there plus deliberate glob guards.

- **The `docs/` folder and its tooling.** Documentation now lives at
  [www.fxtdstudios.com](https://www.fxtdstudios.com) rather than in the
  repository. Removed with it: `tests/test_docs_coverage.py`,
  `tools/generate_docs.py`, `scripts/build_website_docs.py`, and the
  `docs/dev/` handling in `tools/clean_release.py`.

  Note what this gives up — `generate_docs.py` built the node reference from
  the live `NODE_CLASS_MAPPINGS` and a test failed when the two drifted. That
  guarantee is gone, so the published node reference is now maintained by hand.
  Node `DESCRIPTION` strings and per-input tooltips still ship inside the
  package and are shown by ComfyUI on hover, so the in-app reference stays
  accurate on its own.

### Changed

- **All links point at [www.fxtdstudios.com](https://www.fxtdstudios.com).**
  Replaces `radiance.fxtd.org` and `www.fxtd.org` in the README, the
  `Documentation` URL in `pyproject.toml`, and the in-product links in the
  Workspace node, the project-manager launcher, the Viewer and the workspace
  dashboard.

### Housekeeping

- `.gitignore` now covers `grading/`, `preview/` and `MagicMock/` — all three
  were untracked but unignored, one `git add -A` away from being committed.

### Tests

- `tests/test_energy_prioritized_sampling.py` replaces the two EPS tests in
  `test_sampler_regression.py`. Those re-implemented the parser and the modifier
  arithmetic inside the test body and asserted on their own copies, so they
  passed without executing any of `nodes_sampler.py` — including through both
  crashes above. The new tests drive the real node and the real CFG patch.
- `tests/test_output_path_anchoring.py` pins the path fixes above, and adds
  two ratchets: no node may default a writable path to a bare relative
  string without routing it through a resolver, and `INPUT_TYPES()` may not
  touch the filesystem.
- `tests/test_log_level.py` covers the environment variable above, including
  a check that the string in the startup error still names a real flag.
- The README gains a **Status & to-do** section: what is blocking a release,
  the correctness backlog, structural debt, and what is verified done.
- `tests/test_backlog_fixes.py`, `tests/test_download_consent.py` and
  `tests/test_optical_flow.py` cover the fixes above. The tiled-VAE blend is
  pinned too — it had already been fixed months earlier but never tested, which
  is why the audit notes still listed it as open.
- **First JavaScript coverage in the project.** 27 tests over the shared
  `escapeHtml` and widget-visibility helpers, on Node's built-in
  `node --test` — no test framework added — wired into CI as a `js-test` job
  and available as `npm test`.

## [3.2.1] - 2026-08-09 ("Full Audit")

**Upgrade note.** Colour output changes for any graph that used
`RadianceColorSpaceConvert` with a camera-log or ACES space — those
conversions previously did nothing (see below). Re-check affected masters.
16/32-bit TIFF writes now require `tifffile` (`pip install tifffile`) instead
of silently writing 8-bit.

### Added

- **Inline video preview on Read.** MP4/MOV/WebM the browser can decode play
  directly on the node (`/radiance/media/preview`, with seeking); ProRes,
  DNxHR and other production codecs fall back to a first-frame poster
  (`/radiance/media/poster`). Same allowed-roots policy as the info routes.
- **Resolved-path readout on Write.** The node now shows exactly what will
  land on disk — output_path + filename + version + format combined by the
  same Python logic that writes, via `/radiance/media/resolve_write` — plus
  a note line ("frame 1 only" for IMG formats with a batch, "DNxHR is written
  into an MXF container", sequence start frame). A contract test writes real
  files and fails if prediction and reality drift.

### HDR family

- **SDR to HDR Expand: `smoothness` did nothing.** The feathering mask was
  computed and then never applied — a dead control since the node shipped.
  It now feathers the expansion onset as its tooltip promises.
- **SDR to HDR Prepare: the feathered inpainting mask drifted up-left** by
  ~2× the feather radius (32 px at the default 16) with dark bands on the
  right/bottom — the blur loop re-padded asymmetrically after passes 2–3.
  The AI was being guided to repair pixels offset from the actual clipped
  highlights. Symmetric per-pass padding; centroid pinned by test.
- **Fast-VAE tiled decode: tiles were butt-joined.** Each tile decoded with
  overlap context but pasted hard against its neighbour; VAE decoders are
  not shift-invariant at their borders, so seams showed on flat gradients.
  Tiles now accumulate under a raised-cosine weight and normalise — a test
  proves tiled output equals untiled output exactly.

### Viewer

- **VRAM frame cache now evicts by bytes, not just frame count.** The LRU
  held 4–24 frames regardless of size — 400 MB at 1080p but ~7 GB at 8K,
  an out-of-memory long before the count limit was reached. Eviction now
  also respects a byte budget (0.5–3 GB, scaled to machine memory).
- **Frames beyond the GPU's texture limit fail loudly, before upload.**
  8K DCI (8192 px) sits exactly on many GPUs' `MAX_TEXTURE_SIZE`; anything
  over it used to produce a black frame and a cryptic GL error code. The
  viewer now reports the frame size, the GPU's limit, and that `proxy_scale`
  is the way to view it — full-resolution data is unaffected.
- Removed stale `MAX_BATCH_SIZE`/`MAX_IMAGE_DIMENSION` duplicates from
  `hdr/io.py`; `viewer_utils.py` (9999 frames / 16384 px) is the single
  source.

### Changed

- **`overwrite` on Write now defaults to OFF.** Destroying an existing file
  must be an explicit choice; when off, a unique suffix is appended instead.
- Read greys out `color_space` when `raw (no transform)` is selected — the
  reader ignores it, and the UI now says so instead of looking live.
- Write hides `broadcast_safe` for float formats (EXR/HDR/32f TIFF/DPX),
  matching the Python side which already refuses to legal-range-clamp them;
  `version` displays as `v0001` alongside the number.

### Fixed

- **Colour Space Convert did real conversions for only 6 of its 16 spaces.**
  Without an OCIO config — the default install — ACEScc, ACEScct, LogC4,
  F-Log2, C-Log3, Log3G10, DaVinci Intermediate, BMD Film Gen5, V-Log and
  N-Log silently returned the input unchanged. The node's own inline LogC3
  was also wrong: 18% grey encoded to 0.417 instead of ARRI's 0.391, and its
  decode disagreed with its own encode, so a LogC3 round trip lost ~1.5 stops.
  All curves now come from `color/transfer.py` / `color/luts.py` (verified
  against published 18%-grey code values, round-trip exact), Rec.709 OETF is
  the real BT.709 camera curve rather than an alias of sRGB, ACEScc/ACEScct
  apply the Rec.709↔AP1 gamut matrix, and a space with no analytical path
  raises instead of passing pixels through untouched.
- **16-bit and 32-bit float TIFF writes refuse to downgrade.** `tifffile` is
  the only writer for these formats but was never declared as a dependency;
  without it a 32-bit float request silently produced an 8-bit clipped file
  (the 16-bit path at least logged a warning). Both now raise with an install
  hint, the reader warns when it cannot probe TIFF depth, and `tifffile` is
  listed by the Environment Guard.
- **Directory and glob sequence reads always came back empty.** The file list
  was sliced by list index with the widget's frame-number defaults
  (`files[1001:99999]`). Frame numbers are now parsed from filenames and the
  window is reconciled against the range on disk, matching the `####` path.
- **The test suite could not see colour-node bugs.** The conftest OCIO mock
  reported `is_loaded` as a truthy `MagicMock`, so nodes "converted" via a
  no-op mock processor and every colour test through them validated an
  identity transform. The mock now reports `is_loaded = False` and a
  regression test pins it.

## [3.2.0] - 2026-07-29 ("Audit Release")

A three-week audit, five independent review passes, and a new test harness that
calls every node. The theme throughout: almost nothing here raised an error. The
code ran, reported success, and produced the wrong result — which is why the
existing 1,500-test suite had nothing to catch.

**Upgrade note.** Colour output changes on several paths. If you have approved
masters made with 3.1.x, re-check them before conforming new work against them —
particularly anything graded through the Viewer's delivery panel, tone-mapped
with AgX, or exported through the ACES 2.0 Cinema or HLG transforms.

### Added

- **109 nodes, up from 100.** Nine complete node classes were written and never
  listed in any mapping dict, so they never reached ComfyUI's menu:
  Bit-Depth Degrade, Policy Guard, LUT Apply, LUT Blend, Digital Cinema Read,
  Digital Cinema Write, Flipbook GIF, Preview Server and ControlNet Apply.
- **Per-node functional tests.** `tests/test_node_functional.py` builds inputs
  from each node's own `INPUT_TYPES` and calls its `FUNCTION`, checking return
  arity, IMAGE/MASK shape and dtype, and that inputs are not mutated in place.
  80 nodes execute, 30 skip with a stated reason, none fail.
- **A delivery payload contract.** `GRADE_PAYLOAD_KEYS` plus a test that parses
  the viewer's JS and the handler's Python and diffs them in both directions.
- **`core/ffmpeg.py`** — resolves ffmpeg from `RADIANCE_FFMPEG`, then PATH, then
  the binary `imageio-ffmpeg` ships.
- **`core/video.py`** — one video decoder for the whole package. Probes the
  container for frame rate, frame count, bit depth, alpha, colour range, matrix,
  primaries, transfer characteristics, rotation, field order and start timecode,
  then decodes through a single raw ffmpeg pipe. `tests/test_video_read.py`
  encodes real ProRes 4444, ProRes 422 HQ and H.264 fixtures and decodes them
  back — 73 tests.
- **Frame ranges on video.** `start_frame`, `end_frame` and `frame_step` now
  apply to clips as well as sequences, selecting on the decoder's own frame
  counter so the range is exact for long-GOP codecs too.
- **Six more input colour spaces on Read** — Rec.709 (BT.1886), Canon Log 3,
  RED Log3G10, PQ (ST.2084), HLG (BT.2100). The inverses were already in
  `color/transfer.py`; only the menu was missing them.
- **`core/tiling.py`** — shared tile blending weights with correct border
  handling.
- **A real-torch gate in the test suite.** The MagicMock torch stub defeated
  every self-skip idiom in use; CI had been red for seventeen days behind a
  stale `--ignore` list.
- **`core/formats.py`** — the extension tables are built from what the installed
  backends actually register, not from a list someone typed. 9 image extensions
  became 64 on a stock install. Installing OpenImageIO adds DPX, Cineon, ARRI
  and camera raw without a code change, and an unsupported file now says which
  package would open it.
- **`core/exr.py`** — multi-part and multi-layer EXR. A Nuke or Arnold render
  with `diffuse`, `specular`, `Z` and `N` reads every layer through a `layer`
  widget, populated from the file itself by `/radiance/media/layers`.
- **Sequence auto-detection.** Picking `sh010.1004.png` reads the whole
  sequence and reports its real range, the way Nuke's Read does.
  `media_type = Image` is the escape hatch.
- **A third output, `info`** — JSON describing what was actually read:
  resolution, frame count and range, bit depth, codec, EXR layers and windows,
  colour tags, timecode. Nuke's metadata tab, as a wire. Appended last, so
  workflows saved against the two-output version keep working.
- **`on_error`, `raw` and `premultiplied` on Read.** Nuke's error policy, raw
  bypass and unpremultiply, with the same defaults Nuke uses.
- **The Read node hides widgets that do not apply** to the detected media type,
  and draws the file's format, range and layers on itself.

### Fixed — colour

- **The delivery panel dropped half the grade.** Shadows, Highlights, Hue, LUT
  and gamut compression were read by the exporter and never sent by the viewer,
  so each exported at its identity default while the viewer showed it applied.
  Tint was sent and read nowhere. Temperature used a Kelvin white-balance
  multiply against the viewer's additive slider — a different curve entirely.
- **ACES 2.0 Cinema** flat-lined above 0.4 scene-linear at a peak white of
  ~27 nits instead of 48. **ACES 2.0 HLG** placed diffuse white at the display
  peak: 18% grey rendered at ~127 nits where BT.2408 specifies ~26.
- **AgX** applied its inset matrix untransposed, tinting every neutral (channel
  spread 0.042 at 18% grey, 0.116 at 16.0), and double transfer-encoded its
  output, raising the black floor to ~12/255 so pure black was unreachable.
- **Alpha was tone-mapped, expanded and graded.** A 50% matte came back at
  0.607, 0.214 or 1.059 depending on the path.
- **Sony S-Log3's toe** used half the specified slope with a spurious offset:
  black encoded to 0.127921 instead of 0.092864 (~36 code values at 10-bit).
- **Colour Space Convert silently skipped the primaries transform** for
  ACES2065-1, DCI-P3 and Display-P3 — three of twelve advertised spaces
  returned the Rec.709 result unchanged.
- DaVinci Intermediate, Canon Log 3 and RED Log3G10 rewritten to spec; the
  DaVinci Wide Gamut and ARRI Wide Gamut 4 matrices had wrong third rows;
  Bradford D65↔D60 adaptation added; the tone-scale shoulder rebuilt.

### Fixed — output and data integrity

- **Writing a versioned file could destroy the previous version.** `_out_path`
  used `Path.with_suffix()`, which replaces everything after the last dot, so
  `sh010.comp_v0001` and `sh010.comp_v0002` both wrote to `sh010.exr`. With
  overwrite on by default, each render silently replaced the last approved one.
- Legal-range limiting was ungated, squeezing 32-bit EXR masters into
  [16/255, 235/255]. Vertical aspect ratios cropped instead of pillarboxing.
- A wheel built on a working machine shipped that developer's own `.rad` shot
  files.
- The session log is now written atomically under a lock; concurrent deliveries
  used to lose entries and a crash mid-write discarded the whole history.

### Fixed — video read

The Read node's video path did not behave the way any other application in a
facility behaves. Every item below is measured against the fixtures in
`tests/test_video_read.py`.

- **ProRes 4444 alpha was decoded and thrown away.** Frames came back through an
  RGB-only reader, so a vendor plate with a matte arrived as three channels and
  the `mask` output was always zeros. The file's alpha now reaches `mask`.
- **The second decoder quantised everything to 8 bits.** `_load_video_to_numpy`
  — used by the DCC handoff paths — tried OpenCV first, and OpenCV returns 8-bit
  BGR regardless of the source. A 12-bit ProRes 4444 came back on an exact 1/255
  grid, losing four bits per component. It now shares one decoder with Read.
- **Decoding wrote a PNG per frame to a temp directory and read them back.** A
  4-second 1080p ProRes 422 HQ clip: 25.0 s and roughly 600 MB of scratch files,
  against 12.9 s and nothing on disk now.
- **A decode that failed part-way through returned a short clip with no error.**
  OpenCV's read loop just stopped. Both a truncated file and a non-zero ffmpeg
  exit now raise, and the message quotes ffmpeg's own stderr.
- **Every failure in Read became an 8×8 black frame.** A missing plate, a
  corrupt MOV and an unrecognised path all logged an error, returned black, and
  left the node green — so the graph carried on and wrote a master out of it.
  Read now raises. A Read with no path set still returns a frame, because a node
  just added to the canvas is not a failure.
- **The container's colour tags were never read.** A Rec.709 delivery was passed
  through as if it were scene-linear, so every downstream exposure, blur and
  blend operated on gamma-encoded values. On Auto, a tagged file now decodes
  through the matching curve and says which. An untagged file still passes
  through, but warns instead of doing it silently.
- **`nb_frames` of 0 was reported as the frame count** for MXF and the many MOVs
  that carry no count. It now falls back to `duration × fps` and marks the
  number estimated.
- **`r_frame_rate` parsing raised on a bare `"30"`** and took the width and
  height down with it, because one `except` covered the whole probe.
- **The recognised-extension list had seven entries**, so `.m2ts`, `.mts`,
  `.r3d`, `.braw` and friends were classified "unknown" and handed to the image
  reader. One list now serves the browser, the detector and the decoder.
- **The fixed 300-second decode timeout** turned any long clip into a spurious
  failure. There is no cap by default.

### Fixed — reading files at all

- **Nine hand-typed image extensions decided what the node would open.** TGA,
  SGI, PPM, PGM, JP2, PCX and ICO were classified "unknown" and refused —
  measured, every one of them decoded correctly through the reader underneath.
  They never reached it.
- **A multi-layer EXR was rejected, and the message blamed the file.** A Nuke or
  Arnold render with AOVs — the normal output of both — raised "which
  RadianceRead does not support". Every layer now reads.
- **A depth-only or data-only EXR raised.** A Z pass is a render output, not a
  malformed file.
- **The EXR reader ignored the display window.** An overscan render came back at
  the data-window resolution and offset with no warning. It is now conformed to
  the display window; `raw` keeps the overscan. This was on the known-limitations
  page.
- **A sequence's alpha was discarded**, exactly like the ProRes 4444 case:
  `img_t, _ = _read_image(p)` for every frame. An RGBA PNG or EXR sequence came
  back with an empty mask.
- **A sequence numbered from anything but 1001 read nothing**, because
  `start_frame` defaults to the VFX convention. A start outside the range that
  exists on disk is now treated as unset, and said out loud.
- **The frontend's video-extension list disagreed with Python in both
  directions** — it listed `.webp`, which is a still, and omitted `.mxf` — so
  the wrong widgets were shown for the files this pack exists to open. One list
  now, with a test that fails if they drift.

### Fixed — performance and stability

- **The built-in upscalers ran on the CPU.** They selected `images.device`,
  which is always CPU for a ComfyUI IMAGE — roughly ten minutes a frame at 4K
  against about five seconds on a GPU.
- **The delivery export blocked ComfyUI's websocket** for its whole duration:
  grading, filters, a model upscale and ffmpeg all ran on the event loop. The
  Project Manager dashboard did the same while walking the output tree.
- `overlap_temporal=1` — the widget minimum, and the value the tooltip
  recommends — produced fully black frames at every window boundary.
- `torch.quantile`'s 2²⁴-element limit made the Multipass Master node fail on
  every 4K plate.
- The Sampler patched the loader's cached ModelPatcher in place whenever
  `cfg <= 1.0` (the Flux default), leaking a stale patch into every later run;
  and the second CFG patch overwrote the first rather than wrapping it.
- Real-ESRGAN loaded with `strict=False` against mismatched layer names: 8 of
  702 tensors matched and the model ran at near-random initialisation while the
  log reported a successful load.
- Tile blending ramped image borders as if they were seams, darkening the frame
  perimeter.
- Four unbounded GPU-resident model dictionaries replaced with bounded LRU
  caches; `RADIANCE_CACHE_SIZE=0` no longer raises.

### Fixed — controls that did nothing

- **PAG** guarded on an `extra_options` key ComfyUI never sets, so the patch was
  a no-op on every call while logging that it had been applied, and `pag_scale`
  was a gate rather than a strength.
- **Restart sampling** ran after the schedule had finished, passed its noised
  latent through the wrong argument, used the wrong noise variance, and stopped
  before returning to σ=0.
- **`blend_mode`** appeared exactly once in the tiling function — in its own
  signature.
- **`chromatic_adaptation`** returned bit-identical output for every option.
- **The Read node's RELOAD button** never rendered: `reload` was declared as a
  hidden input, where ComfyUI only populates magic-string keys.

### Security

- Removed arbitrary code execution from the Nuke bridge. The guard was a
  substring blocklist that a single space defeated (`open (` does not contain
  `open(`), and that module chaining through the permitted `json` global
  defeated outright. Structured commands and literal values only.
- `torch.load(weights_only=True)` on every user-reachable checkpoint path.
- `/radiance/ocio/load` accepted any absolute path on the host; it is now
  contained to the bundled config, `$OCIO`, `RADIANCE_OCIO_ROOTS` and the
  ComfyUI models tree.
- Delivery history in the viewer is HTML-escaped.

### Changed

- Startup reports a shortfall as an ERROR naming each failed module. It used to
  print "successfully loaded N nodes" whether N was 109 or 12.
- `colour-science`, `einops` and `torchsde` removed from the runtime
  dependencies — nothing imported them. `OpenImageIO` added to the three
  platform requirements files, where it was missing despite being required for
  DPX.
- One implementation each of `escapeHtml` and the widget helpers, replacing five
  and six diverged copies.
- 49 exception handlers that silently swallowed failures now log at DEBUG with
  the operation and exception type.

### Known limitations

- The ACES 2.0 tone scale is a log-space contrast of 1.55 with a tanh shoulder,
  not the Daniele Evo curve the specification defines. 18% grey therefore sits
  about 0.84 stop above the ACES 2.0 reference on SDR, and HLG diffuse white
  lands at signal 0.915 rather than 0.75. The normalisation defects around it
  are fixed; the curve itself is not yet the published one.
- `blend_mode="laplacian_pyramid"` falls back to the Gaussian feather and logs
  that it has done so.
- `chromatic_adaptation` has no effect — the adaptation is baked into the
  precomputed conversion matrices. The widget warns when set to a non-default.
- Optical flow is single-scale Lucas–Kanade despite the DIS reference in its
  docstring; it recovers about 1% of a 5-pixel displacement.
- Scene-cut detection normalises scores by the batch maximum, so the threshold
  has no absolute meaning and cut-free footage still reports cuts.

## [3.1.2] - 2026-07-02 ("GPU-First Release Candidate")

GPU-first completion pass for HDR, RUDRA decode, denoise, motion/flow, upscale, and VFX finishing paths, plus the missing HDR tone-map node migration.

### Fixed

- **HDR Tone Map now loads from the organized HDR package.** `RadianceHDRToneMap` and `RadianceHDRExpandDynamicRange` are registered through `nodes/hdr/`, so saved workflows and the HDR menu can resolve the tone-map node again.
- **HDR Tone Map is GPU-first end to end.** The implementation no longer falls back to NumPy or forces CUDA results back to CPU; tone mapping now stays on the selected Torch device.
- **RUDRA compatibility restored on the public decoder API.** `radiance.model.vae` now reuses the maintained fast decoder implementation, including dynamic-range conditioning (`dr_dim` / `dr_proj`), predictor fallback, checkpoint inference, and hardened explicit-checkpoint loading.
- **Version metadata synchronized.** Python package metadata, runtime constants, README badge, and `package.json` now agree on `3.1.2`.

### Changed

- ACES/HDR color operations, denoise, optical-flow motion blur, multipass helpers, and upscale inference paths were tightened to prefer Torch/GPU execution and avoid unnecessary CPU transfers.
- The node-key snapshot was updated for the intentional HDR tone-map registry restoration.

### Testing

- Restored and enabled the RUDRA compatibility tests that were previously skipped.
- Release verification run performed with real Torch, OpenEXR, OpenColorIO, and colour-science dependencies.

## [3.1.1] - 2026-06-02 ("Fidelity and Trust Release")

Correctness, safety, and delivery-trust fixes for the 32-bit / HDR / EXR pipeline, plus real-AOV ingestion and hardened model loading. Closes the critical data-integrity and security findings from the pre-release engineering review.

### Fixed

- **HDR/EXR load no longer crushes scene-linear data.** The image-to-tensor path previously divided any array whose brightest pixel exceeded 2.0 by 255 (or 65535), silently darkening every real HDR/EXR plate ~255x. Normalization is now driven by the source format: integer formats normalize by their type max; float formats (EXR/HDR/float-TIFF) pass through untouched. Read -> Write of an EXR is now lossless.
- **EXR writer can no longer write a silent 0-byte or downgraded file.** It raises a clear error when no EXR backend is available instead of reporting a phantom success.
- **EXR writer preserves alpha and single-channel mattes** (1->RGB grayscale, 3->RGB, 4->RGBA), and no longer crashes on `(H,W)` matte input.
- **`RadianceWrite` surfaces failures** by re-raising on write error (node turns red) instead of swallowing the exception.
- **Sampler non-finite guard.** `RadianceSamplerPro` detects NaN/Inf in the sampled latent, warns, and sanitizes with `nan_to_num` so blown CFG/precision runs are visible instead of shipping black frames.
- **HDR VAE Decode metadata output.** The node now emits its decode-settings JSON as a second `STRING` output (previously built and discarded).
- Removed internal imports of deprecated shim paths, eliminating load-time `DeprecationWarning`s.

### Security

- **`torch.load` hardened.** All shippable checkpoint loads use `weights_only=True` (VAE, fast VAE, multipass depth/normal models, training data prep), preventing arbitrary-code-execution from malicious `.ckpt`/`.pth` files.
- **Model downloads are atomic + integrity-checked.** The loader and multipass downloaders write to a temp file and atomically move into place, with optional SHA-256 verification and `RADIANCE_LOADER_OFFLINE` / `RADIANCE_UPSCALE_OFFLINE` switches for airgapped setups.

### Added

- **Multipass: AOV Reader.** Reads a real multilayer/AOV OpenEXR (Arnold, Redshift, Karma, Cycles, V-Ray) and splits its named layers into the same outputs as the Multipass Master extractor, so ground-truth render passes flow straight into the EXR-passes writer and relight/comp chain. Scene-linear values preserved; missing layers come through black.
- **Alpha output on the EXR write node.** `RadianceWrite` gained an optional `mask` input, written as the EXR alpha channel (RGBA) for EXR formats.
- **Upscaler HDR + color handling.** Image and video upscalers gained `hdr_mode` (Reinhard tonemap round-trip) and `color_encoding` (linear<->sRGB / linear<->LogC3 OETF round-trip).

### Documentation

- Website docs now render Mermaid workflow diagrams as themed graphs (previously shown as raw code) via `scripts/build_website_docs.py`.

### Testing & CI

- Full `pytest tests/` passes against real torch + OpenEXR + OpenColorIO + colour-science (1347 passed, 34 skipped).
- CI installs real runtime dependencies and runs the whole suite; the publish gate runs the full suite. Removed a stale CI import of a non-existent module.
- Added `tests/test_io_hdr_regression.py` covering the load/write/mask fixes.

## [3.1.0] - 2026-05-25 ("Temporal, Viewer, and Registry Release")

### Added

- Viewer timeline upgrades: filmstrip thumbnails, pinned-frame A/B comparison, and OCIO display transform controls.
- Four color-science nodes: Hue Curves, RGB Curves, White Balance using Bradford adaptation, and Color Space Convert with OCIO-first behavior plus analytical fallback.
- Organized package namespace for color, HDR, I/O, monitor, pipeline, training, upscale, video, VFX, and generation nodes while preserving legacy flat module imports.
- GitHub CI and registry publish workflows for the v3 release path.

### Fixed

- Printer Lights viewer listener leak after tab switches and UI rebuilds.
- HDR VAE decode category placement and repeated engine construction.
- VAE encode/decode/roundtrip docstring placement, category labels, and mode-aware NaN/Inf sanitation.
- Fast VAE trained decoder cache now keys by latent channel count and model mode.
- Viewer grading edge cases, including `luma_mix` fast-path handling, CLog3 output clamping, ACES matrix allocation, vectorized saturation, and bounded progress tracking.

### Release

- Updated Registry metadata for `radiance` v3.1.0 under publisher `fxtdstudios`.
- Restored GitHub/Registry image assets referenced by README and `pyproject.toml`.
- Added `.comfyignore` so Registry packages exclude tests, local scratch files, build caches, and review documents.

---

## [3.0.1] - 2026-05-07 ("Color Science Precision & Demo Tooling")

### Added

- **BT.1886 EOTF** (`nodes_hdr_colorspace.py`): Added `"BT.1886 (TV γ2.4)"` to `_EOTF_MAP`
  — ITU-R BT.1886 display EOTF (γ 2.4) for Rec.709 broadcast reference monitors. Distinct
  from `"Rec.709 (OETF)"` (camera signal encoding) and `"Gamma 2.4"` (generic power curve).
- **Rec.709 / BT.1886 IDT** (`nodes_colorscience.py`): Added `"Rec.709 / BT.1886"` to
  `_COLOR_SPACES` — the correct Input Device Transform for display-referred Rec.709 material.
  OCIO mapping: `"Output - Rec.709"`. Analytical fallback: γ 2.4 linearise / γ 1/2.4 encode.
- **◎ Radiance Bit-Depth Degrade** (`nodes_colorscience.py`): Quantize float images to N-bit
  (4–16) precision. Dither modes: none (hard clip), triangular TPDF, Floyd-Steinberg error
  diffusion. Outputs: quantized image, amplified error delta, banding mask, metrics JSON
  (PSNR dB, max error, dynamic range loss in stops). Critical for demo workflows:
  ARRI LogC / Venice RAW → 8-bit → AI HDR reconstruct → EXR.

### Tests

- **`tests/test_colorscience_v3.py`** (32 tests): BT.1886 decode/encode round-trip,
  `_COLOR_SPACES` contents, `RadianceBitDepthDegrade` node registration, quantisation
  correctness (4-bit ≤16 levels, 8-bit ≤256, 16-bit near-lossless), PSNR monotonicity
  with increasing bit depth, TPDF vs hard-clip divergence, Floyd-Steinberg completion,
  banding mask binary property, alpha channel preservation.

### Changed

- `_COLOR_SPACES` in `nodes_colorscience.py` grouped into labelled sections (scene-linear,
  display-referred SDR, camera log/raw) for clarity.
- `_EOTF_MAP` in `nodes_hdr_colorspace.py` restructured with inline comments separating
  SDR EOTFs, HDR EOTFs, and camera log curves.

---

## [3.0.0] - 2026-04-30 ("The Full Pipeline & Intelligence Update")

### Added — ACES 2.0 Full Pipeline (Pillar 03)

- **◎ Radiance ACES 2.0 RRT+ODT** (`nodes_aces2.py`): Analytical ACES 2.0 Reference
  Rendering Transform with four Output Display Transforms (sRGB D65, DCI-P3 D65,
  Rec.2020 PQ HDR10, Rec.2020 HLG). No external CTL toolchain required.
- **◎ Radiance ACES Input Transform** (`nodes_aces2.py`): 25 IDT matrices covering
  ARRI ALEXA, RED, Sony Venice, Canon C-series, Nikon Z, Panasonic Varicam, Blackmagic,
  and GoPro camera systems.
- **◎ Radiance ACES LMT** (`nodes_aces2.py`): Five LMT presets — Linear, Blue Light Fix,
  Golden, Desaturate Highlights, and Kodak-2383 emulation.
- **◎ Radiance ACES CDL** (`nodes_aces2.py`): ASC CDL v1.2 in ACES working space — slope,
  offset, power, saturation with Rec.709 luma weighting and optional clamp.

### Added — Studio Integrations (Pillar 04)

- **◎ Radiance DaVinci Send** (`nodes_studio_integrations.py`): One-click image/sequence
  export to DaVinci Resolve shared folder with configurable bit depth (8/16 bit, EXR).
- **◎ Radiance Nuke Send** (`nodes_studio_integrations.py`): Write a .nk snippet that
  imports the current image into a Nuke session via Read node.
- **◎ Radiance Shot Metadata** (`nodes_studio_integrations.py`): Attach scene/shot/take
  labels, camera info, and lens data as a JSON sidecar and EXIF-compatible comment.
- **◎ Radiance ASC CDL Export** (`nodes_studio_integrations.py`): Write ASC CDL v1.2 XML
  (single `<ColorCorrection>`) or JSON override file from slope/offset/power/saturation
  values — standard interchange with Resolve, SCRATCH, and Nuke.

### Added — Real-Time Preview (Pillar 05)

- **◎ Radiance False Color Monitor** (`nodes_realtime_preview.py`): Cinema-style false
  colour overlay with 10 configurable exposure zones (adjustable luma thresholds and
  zone colours). Toggle between false-colour and source view per node.
- **◎ Radiance Focus Peaking** (`nodes_realtime_preview.py`): Sobel edge highlight overlay
  for critical focus evaluation. Configurable threshold and peak colour.
- **◎ Radiance Split View** (`nodes_realtime_preview.py`): Side-by-side or top-bottom
  wipe comparison between two images, with adjustable split position and guide line.
- **◎ Radiance Contact Sheet** (`nodes_realtime_preview.py`): Auto-tiled contact sheet
  from a batch of images — configurable grid columns and thumbnail padding.
- **◎ Radiance Flipbook GIF** (`nodes_realtime_preview.py`): Assemble batch frames into an
  animated GIF at configurable FPS and scale for quick motion previews.
- **◎ Radiance Frame Stamp** (`nodes_realtime_preview.py`): Burn timecode, frame number,
  and optional label overlay into images. SMPTE 12M timecode — both NDF (24/25/30/48/50 fps)
  and drop-frame (29.97 DF with d=2; 59.94 DF with d=4). `;` separator for DF, `:` for NDF.
- **◎ Radiance Preview Server** (`nodes_realtime_preview.py`): Ephemeral HTTP server
  (daemon thread) serving the last processed frame as JPEG — lets any browser on the LAN
  see the current result without streaming infrastructure.

### Added — AI Assist Layer (Pillar 06)

- **◎ Radiance Auto Grade** (`nodes_ai_assist.py`): Zone-based ASC CDL matching — aligns
  a source image to a reference still by independently correcting shadow offset, midtone
  slope, highlight power, and global saturation. Strength blend 0–1.
- **◎ Radiance CLIP Match** (`nodes_ai_assist.py`): Cosine similarity match between source
  and a pool of reference images using CLIP ViT-B/32 embeddings. Falls back to
  Bhattacharyya histogram similarity when `transformers` is unavailable. Returns the best
  matching reference image plus a similarity score.
- **◎ Radiance Continuity Check** (`nodes_ai_assist.py`): Shot-to-shot continuity analysis
  — flags luma drift, colour cast, saturation drift, contrast drift, and histogram
  dissimilarity between consecutive frames. Outputs a JSON report + clean/dirty flag.
- **◎ Radiance Grade Prompt** (`nodes_ai_assist.py`): Natural language CDL grading — parse
  35 intent rules ("warmer", "crushed blacks", "pull the highlights", etc.) with intensity
  modifiers (very / slightly / a touch / less). Six preset looks: film, bleach, day-for-
  night, instagram, neon, noir.

### Added — Test Infrastructure (Pillar 07)

- `tests/test_cdl.py`: CDL math, XML round-trip, and node registration tests.
- `tests/test_scene_cut.py`: Scene-cut detection — histogram diff, Sobel edge diff,
  combined method, min-shot-frame enforcement.
- `tests/test_temporal_coherence.py`: Flicker removal and temporal smoothing helpers
  (`_luminance_per_frame`, `_rolling_median`, `_gaussian_smooth`, `_smooth_track`).
- `tests/test_colorscience.py`: Bradford matrix algebra, `_xy_to_XYZ`, and illuminant
  chromaticity coverage — 18 tests (Bradford matrix tests torch-gated).
- `tests/test_curves.py`: HSL round-trip for grey, saturated primaries, and random
  images; hue/saturation range checks.
- `tests/test_optics.py`: Lens distortion, chromatic aberration, anamorphic streaks, and
  vignette — registration, input structure, and return-type validation.
- `conftest.py`: Extended torch stub with 20+ additional attributes so module-level
  `torch.*` calls in all nodes succeed at collection time without real torch.

### Added — AI & Pipeline Nodes (Pillar 08)

- **◎ Radiance LLM Driver** (`nodes_llm_driver.py`): Model-agnostic LLM backend
  configuration — Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google), Ollama (local).
  Single driver JSON wire connects to any downstream AI node.
- **◎ Radiance LLM Prompt** (`nodes_llm_driver.py`): Send text prompts to the configured
  backend; returns response string and echoes driver config.
- **◎ Radiance LLM Image Query** (`nodes_llm_driver.py`): Vision-capable multimodal query
  (base-64 PNG encoded) — falls back to text-only for non-vision backends.
- **◎ Radiance Agent Shot Analyst** (`nodes_agent_pipeline.py`): Per-shot technical analysis
  — exposure, contrast, saturation, dynamic range, and scene classification via LLM.
- **◎ Radiance Agent Grade Advisor** (`nodes_agent_pipeline.py`): AI-driven CDL suggestions
  from shot analysis; outputs slope/offset/power/saturation JSON.
- **◎ Radiance Agent QC Check** (`nodes_agent_pipeline.py`): Automated quality gate —
  runs policy rules and returns pass/fail flag + violation report.
- **◎ Radiance Agent Pipeline** (`nodes_agent_pipeline.py`): Shot Analyst → Grade Advisor →
  QC Check orchestration chain in a single node.
- **◎ Radiance Studio Knowledge Base** (`nodes_knowledge_base.py`): CLIP/HSV image vector
  store with semantic query — index reference stills and retrieve by similarity.
- **◎ Radiance MCP Bridge** (`nodes_mcp_bridge.py`): JSON-RPC MCP server exposing Radiance
  as a tool endpoint; includes client stubs for Nuke, Maya, and ShotGrid.
- **◎ Radiance Policy Preset** (`nodes_policy_guard.py`): Six delivery policy presets —
  Broadcast SDR, Cinema DCP, Streaming HDR, Web SDR, Custom. Outputs JSON policy dict.
- **◎ Radiance Policy Guard** (`nodes_policy_guard.py`): Per-frame delivery compliance gate
  — evaluates peak, clipping, black crush, saturation, and metadata presence against policy.
  Returns original image (pass-through), compliance report STRING, and pass/fail BOOLEAN.
- **◎ Radiance Render Dispatch** (`nodes_render_dispatch.py`): Submit render jobs to
  Deadline, Tractor, and OpenCue; poll status; cancel; list queued jobs.
- **◎ Radiance Audio Cut** (`nodes_audio_cut.py`): Beat/onset detection from audio waveform
  — outputs cut timecodes and Whisper speech transcription (when available).

### Added — DiT Video Model Integration (Tier 1)

- **◎ Radiance DiT Model Config** (`nodes_dit_adapter.py`): Unified architecture config for
  SD-VAE, SDXL-VAE, LTX-Video (128ch), HunyuanVideo (16ch), Wan 2.1 (16ch), CogVideoX
  (16ch), and Mochi-1 (12ch). Outputs spec JSON for downstream adapter nodes.
- **◎ Radiance DiT Latent Adapter** (`nodes_dit_adapter.py`): SD-VAE latent ↔ DiT latent
  channel projection with per-architecture mean/std normalisation. Adaptive channel project
  supports expand (tile) and contract (fold) modes.
- **◎ Radiance DiT Latent Norm** (`nodes_dit_adapter.py`): Per-architecture normalisation
  and denormalisation of DiT latents. Forward and inverse pass.
- **◎ Radiance DiT Inspect** (`nodes_dit_adapter.py`): Latent shape / stats inspector
  reporting channels, spatial dimensions, dtype, min/max/mean.
- **◎ Radiance DiT Frame Split / Merge** (`nodes_dit_adapter.py`): Split video latents
  into per-frame chunks and reassemble — enables frame-parallel DiT workflows.
- **◎ Radiance HDR Video Conditioner** (`nodes_video_hdr.py`): HDR conditioning for DiT
  video models — builds latent sequence from EXR frames with scene-linear encoding.
- **◎ Radiance HDR Video Decoder** (`nodes_video_hdr.py`): Decode DiT video latents back
  to HDR image sequence with optional PQ/HLG EOTF application.
- **◎ Radiance HDR Video Prompt Builder** (`nodes_video_hdr.py`): Construct HDR-aware CLIP
  prompt strings from scene metadata (peak nits, colour space, HDR format).
- **◎ Radiance HDR Video Assembler** (`nodes_video_hdr.py`): Assemble decoded frames into
  an EXR sequence or MP4 (ProRes 4444 / H.265 10-bit).

### Added — Character Consistency System

- **◎ Radiance Character Anchor** (`nodes_character.py`): Build a character profile from a
  reference image — CLIP embedding, HSV histogram, face embedding (facexlib), and metadata.
  Saves to JSON sidecar; returns profile STRING for downstream enforcement.
- **◎ Radiance Character Enforce** (`nodes_character.py`): Inject character embedding into
  CONDITIONING via token-append, IP-Adapter strength, or text-blend strategies.
- **◎ Radiance Character Checker** (`nodes_character.py`): Per-frame cosine similarity check
  against a stored profile. Outputs similarity score, pass/fail flag, and report JSON.
- **◎ Radiance Character Blend** (`nodes_character.py`): Weighted blend of two character
  profiles — useful for gradual style evolution across a sequence.
- **◎ Radiance Character Gallery** (`nodes_character.py`): Load all character profiles from
  a folder; display as structured JSON gallery for selection.
- **◎ Radiance Character Score Timeline** (`nodes_character.py`): Per-shot similarity scores
  serialised as JSON timeline for consistency trend analysis.

### Added — T2V & I2V Pipeline

- **◎ Radiance T2V Wrapper** (`nodes_t2v_pipeline.py`): Text-to-video inference wrapper for
  LTX-Video, HunyuanVideo, Wan 2.1, and CogVideoX. Unified prompt/resolution/steps API.
- **◎ Radiance I2V Wrapper** (`nodes_t2v_pipeline.py`): Image-to-video inference with first-
  frame conditioning; supports motion bucket / flow magnitude parameters.
- **◎ Radiance Video Sampler** (`nodes_t2v_pipeline.py`): Low-level DiT sampler with
  sigma schedule, CFG, and optional restart sampling.
- **◎ Radiance Batch Decode** (`nodes_t2v_pipeline.py`): Decode a batch of video latents to
  pixel frames using the architecture-appropriate VAE.
- **◎ Radiance Video Export** (`nodes_t2v_pipeline.py`): Write decoded frames to ProRes
  4444, DNxHR, H.265, or EXR sequence; embeds timecode metadata.

### Added — AI Upscaler (nodes_upscale.py)

- **◎ Radiance Upscale Tiler** (`nodes_upscale.py`): Anti-seam tiling engine with
  Gaussian-weighted overlap (≥20%), 4-level Laplacian pyramid blending, and per-tile
  confidence map. Handles arbitrary resolution with no visible tile boundaries.
- **◎ Radiance Upscale Image** (`nodes_upscale.py`): Three-tier upscaler for images.
  - Tier 1 — Real-ESRGAN (RRDB, no basicsr): `realesrgan_x4plus`, `x4plus_anime`, `x2plus`
  - Tier 2 — Transformer SOTA (spandrel): HAT-L ×4/×2, SwinIR-L ×4
  - Tier 3 — Diffusion creative: SD ×4 upscaler (stabilityai), SeedVR2 one-step video
  - Auto tier: content classifier (noise/sharpness/saturation/ai-likelihood) selects tier
- **◎ Radiance Upscale Video** (`nodes_upscale.py`): Temporal-coherent video upscaler.
  SeedVR2-style 4n+1 overlapping windows, sinusoidal ramp weights at window edges,
  Lucas-Kanade optical-flow warp for cross-window reference alignment, Laplacian pyramid
  blend at seams. Eliminates inter-chunk flicker.
- **◎ Radiance Upscale Router** (`nodes_upscale.py`): Content-aware tier selector — analyse
  any image/video and route to the appropriate upscale node automatically.
- **◎ Radiance Upscale Face Restore** (`nodes_upscale.py`): Post-upscale face enhancement.
  Face detection: facexlib RetinaFace → OpenCV Haar cascade fallback. Restoration: spandrel
  (auto-detects CodeFormer / GFPGAN from checkpoint) → basicsr CodeFormer → GFPGANer →
  identity. Gaussian-feather composite. `fidelity_weight` controls realism vs. fidelity.
- **◎ Radiance Upscale Colour Fix** (`nodes_upscale.py`): Histogram-match colour-drift
  correction post-diffusion. Fast path: `torch.searchsorted` vectorised CDF inversion
  (~100× faster than pixel-loop). Strength 0–1 blend with original.

### Added — Test Coverage (Pillar 07 sweep)

- `tests/test_upscale.py` (120 assertions): Gaussian kernel normalisation/symmetry, weight
  map positivity, bicubic output shapes, `tiled_upscale` shape/confidence, histogram-match
  identity/strength-0 noop, content classifier keys, tier recommendation, quantisation math,
  model registry completeness, node registration (6 nodes).
- `tests/test_llm_driver.py`: `_driver_from_json` parse + fallback, `_resolve_api_key` env
  resolution, `_DISPATCH` routing for all 4 backends (Claude/GPT/Gemini/Ollama), INPUT_TYPES
  field names, 3-node registration.
- `tests/test_policy_guard.py`: `_luma` BT.709 coefficients, saturation range, gamut
  fraction, `_analyse` key coverage, `_evaluate` tuple return (passed, violations, score),
  metadata presence checks.
- `tests/test_dit_adapter.py`: `_get_spec` for 6 architectures, channel projection shape,
  `_apply_norm` round-trip, config/adapter/norm/inspect/split/merge INPUT_TYPES.
- `tests/test_character.py`: `_has`, HSV histogram, `_cosine_sim` identity/orthogonal/
  opposite, all 6 character node INPUT_TYPES and RETURN_TYPES.
- `tests/test_colorscience_v3.py` *(see v3.0.1)*: BT.1886 decode/encode round-trip,
  `_COLOR_SPACES` contents, `RadianceBitDepthDegrade` quantisation, PSNR monotonicity,
  dither divergence, FS dither completion, banding mask binary property.
- `tests/test_timecode.py` *(Pillar 09 — 68 tests)*: Full SMPTE 12M timecode suite.
  NDF 24/25/30 fps boundary frames; DF 29.97 (d=2) covering frames 0, 1799, 1800, 3597–3599,
  17981–17983, 107891–107892; DF 59.94 (d=4) equivalent boundaries; monotonicity within
  short minutes; dropped-frame absence at non-10th-minute starts; `;` vs `:` separator;
  `_DF_RATES` constant validation; `RadianceFrameStamp` INPUT_TYPES / RETURN_TYPES.

### Added — CI / Packaging (Pillar 10)

- **`.github/workflows/ci.yml`**: Full GitHub Actions CI matrix — Python 3.9 · 3.10 · 3.11 · 3.12
  on `ubuntu-latest`. Installs headless test deps (`opencv-python-headless`, `pytest-cov`),
  runs `pytest` with `--cov`, uploads XML coverage to Codecov (3.11 only), and attaches pytest
  logs as artifacts on failure. Concurrency group cancels in-flight runs on new push.
- **`.github/workflows/ci.yml` — smoke job**: Independent import smoke-test that verifies 11 key
  node modules (`nodes_scene_cut`, `nodes_ai_assist`, `nodes_aces2`, `nodes_realtime_preview`,
  `nodes_llm_driver`, `nodes_policy_guard`, `nodes_upscale`, `nodes_dit_adapter`,
  `nodes_character`, …) import cleanly with minimal stubs — catches syntax errors and bad
  top-level imports independently of the pytest run.
- **`.github/workflows/ci.yml` — lint-config job**: Validates `pyproject.toml` (TOML parse +
  required table check) and all `.github/workflows/*.yml` files (YAML parse) on every push.
- **`.github/workflows/publish.yml`**: Publish workflow triggered by version tags (`v*.*.*`) or
  `workflow_dispatch`. Runs full test gate, then calls `Comfy-Org/publish-node-action@v1` with
  `REGISTRY_ACCESS_TOKEN` secret, then creates a GitHub Release with notes extracted from
  `CHANGELOG.md`. Pre-release tags (`-rc*`) set `prerelease: true`.
- **`pyproject.toml`** — expanded and tightened:
  - Python classifiers now enumerate 3.9 · 3.10 · 3.11 · 3.12 explicitly.
  - Runtime `dependencies` list pinned with minimum versions (`Pillow>=9.0`, `numpy>=1.22`, …).
  - `opencv-python` moved from runtime to `[full]` extra; `opencv-python-headless>=4.7` added
    to `[test]` extra (server-safe, no GUI dependency).
  - `[test]` extra adds `pytest-cov>=4.0` and `pytest-timeout>=2.1`.
  - `[full]` extra extended to cover `diffusers`, `accelerate`, `peft`, `anthropic`, `openai`,
    `google-generativeai`.
  - `Bug Tracker` URL corrected to GitHub Issues.
  - `[tool.coverage.run]` and `[tool.coverage.report]` sections added (`fail_under = 0`,
    ready to raise once baseline is measured).
  - Stale `tb = "short"` (unknown pytest option) removed; `integration` marker added.

### Changed

- All display names standardised to `◎ Radiance …` prefix across all 7 new nodes_*.py files.
- `tests/test_coherence_prior.py` skip guard now also checks `HAS_TORCH` so it correctly
  skips when the conftest torch stub is active (prevents false-pass on stub MagicMocks).

## [2.6.0] - 2026-04-17 ("The HDR Science & Fast VAE Update")

### Added — Color Science

- **◎ Radiance ACES 2.0 Transform** (`nodes_colorscience.py`): Full analytical ACES 2.0
  Reference Rendering Transform (RRT) + four Output Display Transforms — no external CTL
  toolchain required.
  - RRT: cubic Bézier rational polynomial approximation matching CTL within ±0.3%
  - ODT **sRGB D65** — standard monitor / SDR web output
  - ODT **DCI-P3 D65** — digital cinema / wide-gamut display
  - ODT **Rec.2020 PQ (HDR10)** — ST 2084 absolute luminance up to 10,000 nits
  - ODT **Rec.2020 HLG** — ARIB STD-B67 broadcast HDR (BBC/NHK)
  - Full gamut pipeline: ACEScg → XYZ (D60) → D65 Bradford → target display primaries
  - `exposure_offset` (stops), `saturation`, `peak_nits` (cd/m²), `grade_info` passthrough

- **◎ Radiance White Balance** (`nodes_colorscience.py`): Bradford chromatic adaptation in
  three modes — Temperature/Tint (Kang 2002 xy approx), Illuminant-to-Illuminant CAT
  matrix, and Manual RGB gain. Full HDR-safe, no clamping above 1.0.

- **◎ Radiance Color Space Convert** (`nodes_colorscience.py`): OCIO-powered color space
  conversion with artist-friendly dropdown. Falls back to analytical transforms (sRGB OETF,
  LogC3, ACEScg gamut) when no OCIO config is present. Forward and Inverse directions.
  Supports: Linear sRGB, ACEScg, ACEScc, ACEScct, sRGB, Rec.709, LogC3, LogC4, F-Log2,
  C-Log3, Log3G10, DaVinci Intermediate, BMD Film Gen5, V-Log, N-Log.

### Added — AgX Tone Mapping (Full Pipeline)

- **AgX Full Pipeline** (`hdr/tonemap.py`): Replaced the previous smoothstep approximation
  with a mathematically correct AgX operator (Blender/Troy Sobotka, BSD-licensed).
  - `_AGX_M_IN` / `_AGX_M_OUT` gamut matrices (sRGB ↔ AgX working space)
  - Log₂ inset over working range: −10 … +6.5 EV
  - Sigmoid contrast curve fitted to the AgX CDL
  - Full GPU path via `torch.einsum` (no CPU fallback needed for AgX)
  - CPU fallback via NumPy using the same three-stage pipeline
  - Fixes previous hue shifts in saturated reds/blues under the agx operator

### Added — HDR Generation & Enhancement

- **◎ Radiance HDR Enhancer** (`nodes_hdr_inception.py`): Pixel-space generative HDR
  expansion for scene-linear images. No sampler required for standalone use.
  - Luminance-weighted soft highlight roll-off via configurable knee function
  - Calibrated Gaussian noise injection into near-clip specular regions (sub-pixel detail)
  - Luminance-weighted blend back to original (`blend_strength`)
  - Outputs: `enhanced_image` (IMAGE) + `headroom_mask` (MASK) + `stats` (JSON)
  - `headroom_pct`, `delta_stops` reported in stats JSON
  - Optional `lora_name` hint for HDR LoRA workflows (informational, load via Load LoRA node)

- **◎ Radiance HDR Latent Init** (`nodes_hdr_inception.py`): Properly seeded Gaussian noise
  latent (N(0, σ²)) for HDR-aware generation — replaces zero-latent which caused degenerate
  sampler output. Configurable `sigma` (1.0 = SDXL/SD3, 14.6 = Flux, 3.5 = LCM).

- **◎ Radiance HDR Latent Blend** (`nodes_hdr_inception.py`): Per-channel blend between a
  reference HDR latent and a noise latent. Steers generation toward a tonal distribution
  without full img2img. Per-channel weight overrides via comma-separated string.

### Added — HDR Analysis

- **HDR Histogram: Headroom Zone Markers** (`hdr/analysis.py`): `hdr_zone_markers` input
  draws calibrated vertical lines at 1× (+0 EV), 2× (+1 EV), 4× (+2 EV), 8× (+3 EV)
  scene-linear on the linear histogram panel. Essential for HDR authoring visibility above
  the SDR clip point.
- **HDR Histogram: `headroom_pct` Output** (`hdr/analysis.py`): New FLOAT output pin
  reporting the percentage of luminance pixels above 1.0 (scene-linear SDR ceiling).
  Connect directly to downstream math nodes or display.

### Added — Tier 4: TurboDecoder Training Pipeline

- **RadianceTurboDecoder Architecture** (`fast_vae.py`, `hdr/fast_vae.py`): ~2M parameter
  convolutional decoder that maps VAE latents directly to log-coded images, replacing the
  engineered `_denoise_log_highlights()` + `_soft_log_shoulder()` combination with a
  learned equivalent. Supports Flux (16ch) and SDXL (4ch) latent formats.

- **HDRLogLoss** (`train_turbo_decoder.py`): Four-component HDR-aware training loss:
  - L1 in log space (perceptually uniform tonal weight)
  - MSE in log space (sharpness / outlier penalty)
  - Highlight penalty above configurable `knee` (extra weight on near-clip regions)
  - Structural gradient loss (L1 on image gradients, preserves edge fidelity)

- **EMA Weight Tracking** (`train_turbo_decoder.py`): Exponential moving average of decoder
  weights (`decay=0.999`). EMA weights are saved separately as `turbo_decoder_ema_stepN.pth`
  for smooth, stable inference.

- **`train_turbo_decoder.py`** — Full training CLI:
  - AdamW + Cosine Annealing LR scheduler
  - Configurable steps (50k useful / 200k production), batch size, grad clip
  - Resume from checkpoint, JSONL training log, eval PSNR-log every 2k steps
  - Checkpoint save every 5k steps (full + EMA-only inference weights)

- **`dataset_hdr.py`** — HDR pair dataset generator: produces `.npz` (latent, log-coded
  target) pairs from raw HDR EXRI sources for supervised TurboDecoder distillation.

- **`nodes_fast_vae.py`** — ComfyUI inference node for loading a trained TurboDecoder
  checkpoint and running fast log-domain VAE decoding in production workflows.

---

## [2.5.1] - 2026-04-12 ("The New Architecture Update")

### Added
- **Node Architecture**: Unified `◎ Radiance HDR VAE Decode` to officially use the Radiance 4K production engine. Replaces placeholder with production-grade tiling and log-domain math.
- **Node: ◎ Radiance NDI Sender** — Upgraded with **High-Fidelity Log-Encoding** (LogC4/S-Log3) ensuring full HDR dynamic range is preserved when streaming to external apps in 8-bit.
- **Optics Refinement**: Added professional Boundary Handling (`zeros`, `reflection`, `border`) and `Invert` math to the Lens Distortion and Chromatic Aberration suite.

## [2.5.0] - 2026-04-12 ("The Optics & Architecture Update")

### Added
- **Node: ◎ Radiance Lens Distortion** — True barrel/pincushion distortion with proper ST-Map output generation. Validated for 32-bit float coordinate mapping and edge-wrap prevention.
- **Node: ◎ Radiance Chromatic Aberration** — Real-world spectral dispersion shifting (R/G/B) simulating glass Index of Refraction (IoR), operating from the specified optical center.
- **Node: ◎ Radiance Anamorphic Streaks** — High-end cinematic flare generation employing cascaded anisotropic blurs to organically bloom clipped highlights within the HDR/32-bit linear signal.
- **Node: ◎ Radiance SDR to HDR Expand** — Inverse OETF logic combined with mathematical power-curve spline expansion, successfully recovering squashed SDR highlights into full Scene-Linear data capable of pushing 4,000+ nits dynamically.
- **Node: ◎ Radiance Relight Engine** — True 32-bit fp geometric re-lighting node. Ingests any RGB normal map and applies physically-based Lambertian diffuse and Blinn-Phong specular passes. Allows additive relighting of 2D generative sequences safely.
- **Node: ◎ Radiance HDR VAE Decode** — Intercepts Latent tensor data during decode and bypasses traditional StableDiffusion/Flux clipping mechanisms, mapping latent energies directly to fp32 values to retain dynamic range.
- **Node: ◎ Radiance NDI Sender** — Native PyTorch-to-NDI streaming bridge. Streams PyTorch tensor surfaces as BGRA video frames directly over local network to OBS, Resolume, or Nuke via NewTek NDI SDK.

## [2.4.2] - 2026-04-10 (The Temporal & Intelligence Update)

### Added
- **Viewer: Precision Scopes Suite** — GPU-accelerated **CIE 1931 xy Chromaticity** scope with spectral locus and gamut target overlays (Rec.709, P3, Rec.2020).
- **Viewer: HDR Waveform Upgrade** — Full linear-to-ST.2084 (PQ) non-linear mapping for waveforms. Provides visibility for details up to 10,000 nits without clamping.
- **Viewer: HDR HUD Graticules** — Calibrated horizontal lines and labels for 100, 400, 1000, and 4000 nits in the waveform panel.
- **Node: ◎ Radiance Chromaticity** — Backend analytical node for CIE 1931 visualization. Supports batch processing and gamut target comparison.
- **Node: ◎ Radiance Curves & Hue Curves** — Full piecewise-linear RGB/Master curves and HSL-based secondary color correction engine for 32-bit float pipelines.
- **Viewer: Filmstrip Timeline** — A scrollable row of 40×28px frame thumbnails above the transport scrubber. Click any thumbnail to jump directly to that frame. Active frame is highlighted in accent blue. Max 60 thumbnails with step sub-sampling for long sequences.
- **Viewer: Pin Frame (A/B Wipe)** — New 📌 button in the transport panel. Freezes the current rendered frame as the B-side reference for instant A/B wipe comparison. Click again to release. Integrated with existing wipe renderer path.
- **Viewer: OCIO Display Transform Panel** — New panel in VIEW tab; 37 ACES transforms available when OCIO config is loaded.
- **Viewer: Filmstrip Flicker Heatmap** — Per-frame colored dot (green→magenta) on each filmstrip thumbnail showing inter-frame luma delta; ✂ marker and red border on detected scene cuts.
- **◎ Radiance Scene Cut Detector** — Multi-signal hard/soft cut detection: luma histogram distance, frame delta, LAB chroma shift, and flash guard. Outputs IMAGE + MASK + JSON report.
- **◎ Radiance Temporal Color Lock** — LAB-space batch colour stabiliser. Anchors every frame’s L*a*b* statistics to a reference frame with per-scene auto-reset via cut_mask.
- **◎ Radiance Deflicker Pro** — Professional luminance deflickering: Gain Only (fast), Percentile (selective), and Luma Match (full CDF histogram remapping).
- **◎ Radiance Frame Interpolation** — Temporal upsampler with Linear, Optical Flow (Farneback via OpenCV), and RIFE (external binary) modes. Factor 2–8×.
- **Sampler: CogVideoX + StepVideo** — Full `MODEL_DEFAULTS` entries, `VIDEO_MODEL_TYPES` membership, `CFG_GUIDED_MODELS`, and `detect_model_type()` class + config detection.
- **Sampler: Noise Library Expansion** — Three new pure-PyTorch noise generators added to `NOISE_TYPES`: Simplex (octaved hash-grid), Voronoi (cellular distance field), Curl (divergence-free from spectral potential).
- **◎ Radiance LoRA Scheduler** — Generate per-step LoRA strength schedules using Catmull-Rom / Linear / Step interpolation between user-defined control points. Outputs `FLOAT_LIST` + JSON.
- **◎ Radiance Regional Prompt** — Add a spatial region with its own conditioning via bbox or MASK. Additive / Replace merge modes. Chainable.
- **◎ Radiance Regional Grid** — Divide image into a columns×rows grid; assign a separate text prompt to each cell; encodes inline via CLIP input.
- **◎ Radiance EXR Multi-Part** — Write a Nuke/Resolve-compatible multi-part EXR v2 file with named AOV parts: beauty (RGBA), depth (Z), normal (NX/NY/NZ), albedo, and 2 custom layers. Fallback to per-layer EXRs if OpenEXR multi-part is unavailable.
- **◎ Radiance Render Queue** — Submit parametric sweep runs to the ComfyUI queue. Define `{node_id: {field: [values]}}` and choose Product / Zip / Chain combination modes. Dry-run preview. Returns JSON job manifest.
- **Radiance Write: Remote Output** — `RadianceDigitalCinemaWrite` gains optional `remote_path` input supporting UNC (`\\server\share`) and S3 (`s3://bucket/key`) output via `boto3`.
- **.rad v3 Archive Format** — New ZIP-based `.rad` v3 container (`RADZ` magic). Stores `workflow.json`, `manifest.json` (asset list + metadata), and optional `assets/` directory. Full backward-compat reader handles v1 (plain text), v2 (binary+SHA256), and v3 (ZIP) automatically via `_unpack_any_rad()`.

### Fixed
- **Viewer: Printer Lights Listener Leak (B-14)** — `makePrinterStrip()` was attaching permanent `document.addEventListener('mousemove'/'mouseup')` calls every time the grade panel was rebuilt. Fixed by switching to `AbortController`-scoped listeners.
- **Viewer: Printer Lights UX** — Added EV stop readout per channel (e.g. `+0.40EV`), combined RGB status badge, and a RST button to reset all channels simultaneously.

## [2.3.3] - 2026-04-01


### Fixed
- **A/B Split-Wipe Fix**: Repaired the broken A/B comparison functionality in the Radiance Viewer. The wipe mode now correctly uses the GPU shader and features a draggable split-line with A/B labels.
- **Node Cleanup**: Removed the redundant `RadianceSaveEXR` node in favor of the new `◎ Radiance Write` node.
- **Package Hardening**: Resolved JSON serialization errors in QC reports and standardized environment variables.

## [2.3.2] - 2026-03-30

### Added
- **Smart Overwrite Protection**: `Radiance Write` and `EXR Save` now feature automated index detection to prevent accidental file destruction.
- **Universal Digital Cinema I/O**: Consolidated Video, Image Sequence, and Single Image handling into a high-performance unified pipeline.
- **Terminal HUD & Live REPL**: Nuke-style Python interaction directly inside the viewer for real-time data inspection.
- **Interactive Mask Editor**: Non-destructive brush masking in `◎ Radiance Load Image`.

### Fixed
- **Robust Pipeline Validation**: Enhanced null-safety and input checking for high-load production environments.
- **◎ Radiance Grade Match**: Optimized shot-to-shot color statistics transfer using CIE L*a*b* mean/std math.

## [2.3] - 2026-03-29


## [2.2.2] - 2026-03-29

### Fixed
- **Comfy Registry Metadata**: Removed restrictive `Environment :: GPU :: NVIDIA CUDA` classifier from `pyproject.toml` to allow installation on CPU and macOS (MPS) systems.
- **Registry Icon**: Renamed `RADIANCE_ICON.png` to `flogo.png` and updated `pyproject.toml` to ensure the logo appears correctly in the Comfy Registry.

## [2.2.1] - 2026-03-28

### Fixed
- **Save Overwrite Protection** (`hdr/io.py`, `nodes_io.py`): `Radiance Save EXR/HDR` and `Radiance Write` now default to `start_frame=0`, enabling automatic index detection to prevent accidental file overwrites.
- **Null Safety**: Added robust input validation to sequence loading and saving methods to prevent crashes when passed empty paths or invalid indices from third-party nodes.

### Changed
- **Branding Consistency**: `RadianceApplyGradeInfo` display name updated to `◎ Radiance Apply Grade Info` to match the suite's naming convention.
- **Repository Cleanup**: Removed legacy test scripts and temporary debug directories (`MagicMock`) to streamline the package for GitHub and Comfy Registry.

## [2.2] - 2026-03-18

### Changed
- Major version update to 2.2.
- Refined terminal and viewer UI.
- Improved console noise filtering and error stabilization.

## [2.1.1] - 2026-03-17

### Changed
- Maintenance update to version 2.1.1.

## [2.1.0] - 2026-03-10

### Added — Radiance Viewer v2.2 (Terminal & UX Overhaul)
- **TERMINAL HUD Tab** (`radiance_viewer.js`): Live Python REPL embedded directly in the viewer.
  - Persistent namespace (`_TERMINAL_NS`) — variables survive between executions.
  - 30-second timeout guard via `threading.Thread` to prevent ComfyUI event-loop freeze.
  - Pre-injected context: `math`, `os`, `torch`, `np`, `json`, `time`, `folder_paths`.
  - `Reset Namespace` button wipes state on demand.
  - Snippet dropdown with common helper presets.
- **Documentation Links** in Radiance Workspace node (two new buttons: `📖 Docs — radiance.fxtd.org` and `🌐 FXTD Studios — www.fxtd.org`) and in the Terminal tab status bar.

### Changed — Radiance Viewer UX (5 Design Improvements)
- **Toolbar Group Labels**: 9 labeled clusters (`FILE · GRADE · VIEW · CH · NAV · ANALYSIS · COMPARE · SCOPES · ANNOTATE · MEASURE`) with hairline separator rules.
- **Panel Label Typography**: All HUD sub-labels bumped to `11px` + `letter-spacing: 0.06em` for improved legibility.
- **Bottom Dock Colors**: `TERMINAL` tab now renders in `#00a8ff` (brand blue); `SCRIPT EDITOR` in `#7a92b0` (muted slate). `RUN AUTOMATION` button color unified to match.
- **Film Stock Preset Active State**: Clicking a film stock pill now shows a blue-glow border indicator (`rgba(0,168,255,0.45)`) — selected state is always visible.
- **Right Panel Height**: `tabContentContainer` uses `flex:1 + overflow-y:auto` — grading controls fill the full panel height instead of leaving a generic black void below.

### Changed — Nodes
- **◎ Radiance Depth Map** (renamed from `◎ Depth Map Generator`): Display name updated in `nodes_depth.py` — existing workflows unaffected as the internal class name `RadianceDepthMapGenerator` is unchanged.
- **Output Path Consistency**: Widget parameter name unified to `output_path` across:
  - `RadianceWrite` node (`nodes_io.py`) — was `subfolder`.
  - `RadianceSaveEXR` node (`hdr/io.py`) — was `custom_path`.

### Fixed
- Added missing `import json` to `nodes_radiance_viewer.py`.
- Restored accidentally dropped `exportToCDL()` function declaration after Terminal injection.

### Added — Radiance v2.1.0 (The Professional Suite)
- **◎ Radiance 32-bit Denoise** (`nodes_denoise.py`): Edge-preserving bilateral filter for 32-bit float images.
- **◎ Radiance Reroute / Reroute+** (`nodes_layout.py`): Compact visual reroute nodes with auto-type detection and custom labels.
- **◎ Radiance Load Image** (`nodes_radiance_mask.py`): Enhanced image loader with integrated soft-brush mask editor and non-destructive companion mask storage.
- **◎ Show Text (Radiance)** (`nodes_text.py`): Utility node for displaying any data type (string, JSON, etc.) directly on the node UI.
- **Prompt Enhancer Integration**: `Cinematic Prompt Machine` now supports grammar-aware prompt enhancement (Natural, Descriptive, Cinematic styles).

### Improved
- **Metadata Management**: Improved EXR/PNG metadata handling across I/O nodes.
- **Dependency Validation**: enhanced `check_dependencies` in `__init__.py` for clearer installation guidance.

### Added — Radiance Viewer v2.1 (Viewer Overhaul)
- **fp32 Pick Buffer Sidecar** (`#5`): Each frame now saves a zlib-compressed fp32 `.rpick` sidecar (max 256px) for accurate scene-linear HDR color picking — true EV readout at cursor
- **OES_texture_half_float** (`#3`): Explicitly enables the WebGL `OES_texture_half_float` extension; falls back gracefully if unavailable. Half-float upload path now active by default
- **GPU Histogram** (`#6`): Press **H** in the viewer for a GPU-rendered 256-bin per-channel histogram with log scale for HDR images and a dotted white line at `x=1.0` to mark the SDR ceiling
- **LRU Frame Cache** (`#8`): Up to 8 GPU textures cached by frame ID. Scrubbing through frames no longer re-uploads data that is already in VRAM
- **Linear False Color / Zebra** (`#9`): `linearFalseColor` flag evaluates false-color and zebra thresholds in scene-linear space (pre-OETF) for accurate stop-level analysis
- **Display-P3 / HDR Monitor Detection** (`#10`): `RadianceWebGLRenderer.initDisplayP3()` detects P3 and Rec.2020 displays via CSS media queries and configures canvas `colorSpace` accordingly
- **CDL Export + Import** (`#7`): 💾 menu gains **Export CDL (Grade)** and **Import CDL (Grade)** — bidirectional ASC CDL v1.2 XML roundtrip recognized by Nuke, DaVinci Resolve, and OCIO pipelines

### Added — New Nodes (9 nodes)

#### Color
- **◎ Radiance Grade Match** (`nodes_grade.py`): Dedicated shot-to-shot LAB mean/std match grading. Connect `source` + `reference` → matched image + grade_info JSON
- **◎ Apply Grade Info** (`nodes_grade.py`): Replay a `grade_info` JSON string (from Radiance Grade) onto any new image with a `strength` blend slider. Closes the grade roundtrip
- **◎ Radiance LUT Bake** (`nodes_lut.py`): Generate a 33³ `.cube` LUT file from any Radiance Grade parameter set. Compatible with DaVinci Resolve, Nuke, Premiere Pro, and OCIO
- **◎ Radiance LUT Apply** (`nodes_lut.py`): Load and apply any external `.cube` LUT file via trilinear interpolation with `strength` blend control

#### Video / Temporal
- **◎ Radiance Temporal Smooth** (`nodes_temporal.py`): Per-pixel exponential moving average (EMA) across batch frames to remove inter-frame flicker and AI grain. Motion-aware masking preserves sharp moving areas
- **◎ Radiance Flicker Analyze** (`nodes_temporal.py`): Measures frame-to-frame luma delta and outputs a JSON flicker report (flicker index, max delta, per-frame means). Use before/after Temporal Smooth to benchmark improvement

#### Scopes / Analysis
- **◎ Radiance False Color** (`nodes_scopes.py`): Bake a 7-zone calibrated false-color exposure visualization as an IMAGE for headless/batch pipeline use. Zones: Crushed / Under / Dim / Correct / Over / Hot / Clipped. Configurable `is_linear`, `blend`, and `exposure_offset`

#### Compositing
- **◎ Radiance Blend Composite** (`nodes_overlay.py`): Two-layer compositor with 8 blend modes — Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, Divide. Optional `MASK` pin for per-pixel coverage. HDR-safe (no mid-pipeline clamping)

### Improved — Existing Nodes
- **◎ Radiance Grade** (`nodes_grade.py`): Added `reference_image` + `match_strength` optional inputs for auto LAB match grading. Added `preset_file` for loading external JSON preset libraries. `grade_info` output is now a full JSON dump of all 12 grade parameters (was plain text)
- **Sampler Pro v3.6**: Fixed BUG-28 (sigma schedule off-by-one causing negative step indices), BUG-33 (AYS Flux anchors), BUG-20 (deep-copy in guidance), and several noise-injection double-application bugs

### Fixed
- **Viewer Pick Buffer**: HDR color picker now reads true scene-linear fp32 values from `.rpick` sidecar instead of tonemapped 8-bit pixel values
- **Viewer Histogram**: Histogram scope now GPU-rendered in scene-linear space instead of display-encoded space
- **Pixel Loupe**: Enhanced HDR support and magnification controls
- **QC Analysis**: Tonemapping applied before defect detection for more accurate results

---

## [1.2.1] - 2026-02-10

### Changed
- **Code Organization** - Moved `defects.py` to `radiance/image/defects.py` for better modularity.
- **Cleanup** - Removed temporary dev files (`fix_indent.py`, `fix_indent_2.py`).
- **Dependencies** - Updated `nodes_qc.py` to import defects from the new location.

## [1.2.0] - 2026-02-09

### Added - Radiance Studio
- **RadianceQC Node (`RadianceQC`)**
  - Automated technical quality control for VFX workflows.
  - Analyzes Levels (Crushed Blacks/Clipped Whites), Gamut violations, and Noise floor.
  - Generates visual overlay (Red=Clipped, Blue=Crushed) and detailed text report.
- **Project Management System**
  - **Project Settings Node (`RadianceProjectSettings`)**: Define root, sequence, shot, and version.
  - **Smart EXR Saver**: `RadianceSaveEXR` now accepts `RADIANCE_PROJECT` input to auto-generate paths (`Root/Seq/Shot/vXX`).
- **Nuke-Style Backdrops**
  - Right-click context menu "Radiance Studio > Create Backdrop".
  - Wraps selected nodes in a professional, color-coded group using industry standard colors.
  - "Auto-Align Nodes" command for quick organization.

## [1.1.1] - 2026-02-07

### Critical Fixes
- **HDR Clamping Fixes** - Removed unintended 8-bit clamping in CPU-based effects (`apply_bloom`, `apply_chromatic_aberration`, `RadianceProFilmEffects` sharpening).
- **EXR Saver Stability** - Fixed generic `SystemError` crash in OpenEXR writer; fallback mechanism now robustly handles writer failures.
- **EXR Absolute Paths** - `RadianceSaveEXR` now supports absolute paths in `subfolder` input (e.g. for external drive export).

## [1.1.0] - 2026-02-03

### Added
- **Temporal Grain Engine v2** - Frame-coherent grain for video with smooth transitions
  - `generate_temporal_grain()` function with prime decorrelation seed system
  - Per-channel R/G/B intensity controls in `FXTDTemporalGrain` node
  - Temporal smoothness blending between frames
  
- **LUT Engine Upgrade**
  - Tetrahedral interpolation for more accurate 3D LUT lookups
  - `interpolation` parameter (Trilinear/Tetrahedral) in `RadianceLUTApply`
  
- **Comprehensive Test Suite**
  - `test_comprehensive.py` with 18 tests covering 32-bit precision and sampler validation
  - 100% pass rate on all tests

### Fixed
- Phase-Shift sampler override now correctly uses selected sampler
- Dynamic Guidance math corrected for Low → High → Low profile
- **Canon Log3** - Fixed coefficients to match Canon specification (18% gray → 0.343)
- **Panasonic V-Log** - Fixed decode coefficient typo for proper roundtrip precision

### Security & Production Hardening
- **Thread-safe caching** - Added `threading.RLock()` to LUT, depth model, and processor caches
- **Exception handling** - Replaced 14 bare `except:` blocks with specific exception types + logging
- **Logging infrastructure** - Added `logging.getLogger()` to all modules for proper debugging
- **LUT cache eviction** - Bounded cache size (32 max) with FIFO eviction to prevent memory leaks

---

## [1.0.0] - 2026-01-15

### Initial Release
- 55 professional nodes for HDR processing
- Film grain with 30+ camera sensors, 20+ film stocks
- Industry-standard scopes (Histogram, Waveform, Vectorscope)
- GPU-accelerated processing
- EXR/HDR file support
- ACES 2.0 Output Transform
- LUT support with caching
- Radiance Pro Viewer with Nuke/Flame-style shortcuts
