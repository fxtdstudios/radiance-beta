[← Back to Radiance docs](README.md)

# Glossary

Terms used throughout Radiance and its documentation, defined the way they are
used here.

---

## Colour and imaging

**ACES** — Academy Colour Encoding System. A set of colour spaces and transforms
designed so footage from different sources can be combined predictably.

**ACES2065-1 (AP0)** — the ACES interchange space. Very wide primaries, D60
white, linear. Used for archival and exchange, not for working.

**ACEScg (AP1)** — the ACES *working* space. Narrower than AP0, still wider than
Rec.709, linear, D60. This is what you render and composite in.

**ACEScct** — ACES AP1 primaries with a log transfer function and a toe. Used for
grading, because a log curve gives grading controls a sensible response.

**AMF** — ACES Metadata File. An XML sidecar describing the colour pipeline
applied to a deliverable.

**AOV** — Arbitrary Output Variable. A render pass: albedo, normal, depth,
motion vectors, an object ID, and so on.

**CDL** — ASC Colour Decision List. A tiny, portable grade: slope, offset and
power per channel plus a global saturation. Round-trips as `.cc` or `.cdl`.

**Chromatic adaptation** — converting between white points (D60 and D65, say) so
a neutral stays neutral. Bradford is the usual method.

**Cryptomatte** — a standard for per-object mattes carried in an EXR. Radiance's
segmentation output is *not* one.

**Display-referred** — pixel values describe what a display should emit. Bounded,
gamma-encoded.

**DPX** — Digital Picture Exchange. A film and digital-cinema still format.
Requires OpenImageIO in Radiance; there is no Pillow plugin.

**EOTF / OETF** — Electro-Optical and Opto-Electronic Transfer Function. The
curve from code value to light (EOTF, the display side) or from light to code
value (OETF, the camera side).

**EXR / OpenEXR** — the high-dynamic-range image format used throughout VFX.
Float, unbounded, multi-channel, optionally multi-part.

**Gamut** — the range of colours a set of primaries can represent. "Out of
gamut" means a colour that cannot be expressed without a negative channel.

**Gamut compression** — bringing out-of-gamut colours back inside the boundary
in a controlled way rather than clipping them.

**HLG** — Hybrid Log-Gamma, BT.2100. A *relative* HDR encoding: the display
supplies its own system gamma, so one file reads correctly across displays of
different peak brightness. Reference white sits at signal 0.75.

**LUT** — Look-Up Table. A baked colour transform, 1D (per channel) or 3D
(full colour). `.cube` is the common format.

**Log (camera log)** — a transfer function that packs many stops into a limited
bit depth. LogC3, LogC4, S-Log3, V-Log, Canon Log 3, Log3G10, DaVinci
Intermediate.

**Nit** — candela per square metre, the unit of display luminance. SDR reference
white is ~100 nits; HDR peaks run 1000–4000.

**OCIO / OpenColorIO** — the industry colour-management library. If your facility
has a config, it is the authority.

**Premultiplied (associated) alpha** — RGB already multiplied by alpha. EXR
convention. Must be unpremultiplied before filtering or colour correction.

**Primaries** — the specific red, green and blue a colour space is built on.

**PQ (ST.2084)** — Perceptual Quantiser. An *absolute* HDR encoding: a code value
means a fixed nit level regardless of display.

**Scene-linear / scene-referred** — pixel values are proportional to scene light.
Unbounded. 0.18 is middle grey. This is the working space for everything.

**Straight (unassociated) alpha** — RGB holds full colour independent of alpha.
ComfyUI's `MASK` convention.

**Tone mapping** — reducing dynamic range to fit a display. One-way; keep the
scene-linear master.

**Transfer function** — the mapping between a stored number and a light level.

---

## Diffusion and generation

**CFG (classifier-free guidance)** — the scale by which the conditional
prediction is pushed away from the unconditional one. Flux models use a guidance
embedding instead and want `cfg = 1.0`.

**CFG rescale** — Lin et al. 2023. Rescales the guided prediction to match the
conditional prediction's standard deviation, reducing over-saturation at high
CFG.

**Latent** — the compressed representation a diffusion model works in, typically
1/8 the spatial resolution of the image.

**LoRA** — Low-Rank Adaptation. A small trained delta applied on top of a base
model.

**PAG** — Perturbed Attention Guidance. See
[Known Limitations](limitations.md) — Radiance's is a partial implementation.

**RUDRA** — Radiance's trained HDR VAE decoders, published on Hugging Face.
Turbo and Full variants per base model. Without one, HDR VAE decode falls back to
the standard VAE and the output is display-referred.

**Restart sampling** — Xu et al. 2023. Jumps back up the noise schedule and
re-solves, trading time for detail.

**Sigma / schedule** — the noise levels a sampler steps through. Karras,
exponential and others are different distributions of those levels.

**Timestep shift** — the Flux/SD3 reparameterisation `σ' = s·σ / (1 + (s−1)σ)`,
which biases sampling toward higher or lower noise.

**VAE** — the encoder/decoder between image and latent space.

---

## Radiance-specific

**Delivery panel** — the export section of the Radiance Viewer. What you grade
there is what gets written.

**Gizmo** — a group of nodes collapsed into one reusable custom node, created at
runtime from the canvas.

**Lite Viewer** — the lightweight Canvas 2D viewer. No grading, no delivery.

**Multipass Master** — derives estimated render passes from a single image.
Estimation, not ground truth.

**Node key** — the internal identifier saved in a workflow, e.g.
`RadianceGrade`. Distinct from the display name shown in the menu. Renaming a key
breaks saved workflows; Radiance has a snapshot test to prevent that.

**Policy Guard** — turns a house delivery standard into a pass/fail check.

**Project Manager** — the in-canvas dashboard for shows, sequences, shots and
versions.

**Proxy scale** — a downscale factor applied at Read, for building graphs
quickly without carrying full-resolution data.

**Session log** — `radiance_sessions.json` in your ComfyUI output directory. One
entry per delivery, with the grade, settings and QC result.

**Tier (upscale)** — which upscaling backend runs. Tier 1 is Real-ESRGAN, Tier 2
external models, Tier 3 an SD ×4 diffusion upscaler.

---

## Environment variables

| Variable | Effect |
| :--- | :--- |
| `RADIANCE_FFMPEG` | Absolute path to an ffmpeg binary. Overrides PATH and the bundled build. |
| `RADIANCE_FFPROBE` | Same, for ffprobe. |
| `RADIANCE_OCIO_ROOTS` | Extra directories the OCIO load route may read from, OS path separator between entries. |
| `OCIO` | Standard OCIO config path. Radiance honours it. |
| `RADIANCE_VIEWER_CACHE_BYTES` | Byte budget for the viewer frame cache. Default 2 GB. |
| `RADIANCE_CACHE_SIZE` | Entry count for the model caches. |
| `RADIANCE_LOG_LEVEL` | `DEBUG` for tracebacks on import failures and swallowed errors. |
| `RADIANCE_DEV` | Enables the training node group. |
