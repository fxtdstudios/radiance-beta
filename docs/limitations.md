[← Back to Radiance docs](README.md)

# Known Limitations

Everything on this page is a real, current, reproducible gap between what a
control is named and what it does. It is here rather than in a private tracker
because a control that quietly does nothing is worse than one that says so — you
make a decision, believe it landed, and find out later.

Current as of **3.2.0**. Each entry says what happens, what you should do
instead, and — where it is short — the measurement.

---

## Colour

### The ACES 2.0 tone scale is not the Daniele Evo curve

**What happens.** The tone scale is a log-space contrast of 1.55 around 0.18
with a `x²/(x²+0.01²)` toe and a tanh shoulder. The ACES 2.0 specification
defines the Daniele Evo curve, which this is not.

**Measured.** ACEScg 18% grey through the SDR (sRGB/Rec.709) transform renders
at about 18.0 nits where the ACES 2.0 reference gives about 10.1 — roughly 0.84
stop hot. On HLG, diffuse white lands at signal 0.915 rather than the 0.75
BT.2408 specifies.

**What was fixed in 3.2.0.** The *normalisation* defects around the curve. The
Cinema transform used to flat-line above 0.4 scene-linear at a peak white of
about 27 nits instead of 48; it now ramps to full white. HLG used to place
diffuse white at the display peak — 18% grey at ~127 nits — and now puts it at
signal 0.374, against BT.2408's ~0.38.

**What to do.** Use the transform, but grade against a reference you trust
rather than assuming the numbers match a reference ACES 2.0 implementation. If
you need certified ACES 2.0 output, do the output transform in OCIO with an
official config.

### `chromatic_adaptation` has no effect

**What happens.** The Colorspace Convert widget offers Bradford, Von Kries, XYZ
Scaling and None. All four produce bit-identical output, because the D60↔D65
adaptation is already baked into the precomputed conversion matrices and the
selector is never consulted.

**What to do.** Leave it on Bradford. Selecting anything else logs a warning
once per session. If you need a different CAT, do the conversion in OCIO.

### The EXR reader ignores the display window

**What happens.** Radiance reads the data window and reshapes to it. An overscan
render — standard Nuke output, where the data window is larger than the display
window — comes back at the data-window resolution and offset, with no warning.

**What to do.** Crop to the display window in your comp application before
bringing overscan plates into Radiance.

### Alpha association is not tracked on EXR read or write

**What happens.** EXR conventionally holds premultiplied (associated) alpha;
ComfyUI's `MASK` type is straight by convention. Radiance moves between them
without converting, so round-tripping a matte through Radiance changes edge
pixels.

**What to do.** For mattes where edge quality matters, keep the alpha in your
comp application rather than round-tripping it.

---

## Video

### An untagged video is passed through, not decoded

**What happens.** With `color_space` on Auto, Radiance reads the container's
transfer characteristics and decodes accordingly. A file with no colour tags —
which is most files an editor hands you — has nothing to read, so the values
pass through as if they were already scene-linear. They are not: a Rec.709
delivery is gamma encoded, and every exposure, blur and blend downstream is
then operating on the wrong numbers.

Radiance logs a warning naming the file when this happens. It does not guess,
because guessing wrong is silent and guessing right is luck.

**What to do.** Set `color_space` explicitly for untagged sources —
`Rec.709 (BT.1886)` for a graded delivery, the camera curve for a log MOV.

### Video decode is capped by RAM, not streamed

**What happens.** A clip is decoded to a single batched tensor. 240 frames of
4K RGBA float32 is about 31 GB. There is no windowed or streaming mode.

**What to do.** Use `max_video_frames` and `proxy_scale` while building a
graph, and process long clips in chunks with `start_frame` / `end_frame`.

### Interlaced sources are not deinterlaced

**What happens.** `field_order` is read and reported in the node's metadata, but
no deinterlace is applied. An interlaced broadcast source arrives with combing.

**What to do.** Deinterlace upstream (`ffmpeg -vf yadif`) or in your NLE.

### `start_frame` means something different for video

**What happens.** For a sequence it is the frame number in the filename, which
is why it defaults to 1001. For a video it is a zero-based offset into the
clip. Taken literally, the 1001 default would skip the first 1001 frames of
every clip.

Radiance treats a start past the end of the clip as "not set" and reads the
whole thing, logging what it did. A start *inside* the clip is honoured as a
deliberate trim.

**What to do.** Nothing, unless you want a trim — in which case set
`start_frame` to a frame number that exists in the clip.

---

## Sampling

### PAG is not the published method

**What happens.** It is a mid-block self-attention perturbation blended by
`pag_scale`. The method in Ahn et al. 2024 requires a third, separately
perturbed forward pass combined as `ε_u + s(ε_c − ε_u) + s_pag(ε_c − ε_p)`;
ComfyUI's patch API cannot express that here.

**What was fixed in 3.2.0.** Before this it was a complete no-op — it guarded on
an `extra_options` key ComfyUI never sets, so the patch returned its inputs
unchanged on every call while logging "PAG applied with scale …". And
`pag_scale` was an on/off gate, so 0.1 and 5.0 were bit-identical.

**What to do.** Treat it as an attention guidance term with an empirical scale,
not as PAG.

### Restart sampling is expensive

**What happens.** It now follows Xu et al. 2023 Alg. 2 correctly, but because it
runs after the main schedule has completed, each restart re-solves the entire
tail of the schedule down to σ=0.

**What was fixed in 3.2.0.** The noise variance (it used a standard deviation of
`restart_sigma` instead of `√(σ_max² − σ_min²)`), the argument it passed the
noised latent through (ComfyUI's internal `σ₀·noise + latent` amplified the
signal by `1+σ₀`), and the sub-schedule, which stopped a few steps in and left
the latent partially noised — measured at σ=0.79 for `restart_sigma=2.0`.

**What to do.** Budget for the extra time. One restart roughly doubles the
sampling cost from the restart point down.

---

## Upscale

### Tier 3 diffusion ignores `scale`

**What happens.** The Tier 3 (SD ×4 diffusion) backend is fixed at 4×. At
`scale = 2` the result is the top-left quarter of a 4× upscale — measured
correlation with a true 2× result: 0.0024. `mode = "creative"` selects that tier
regardless of the scale you asked for.

**What to do.** Use Tier 3 only at 4×. For 2×, use Tier 1 or 2, or upscale 4×
and downsample.

### `blend_mode = "laplacian_pyramid"` is not implemented

**What happens.** It falls back to the Gaussian feather and logs that it has
done so, once per session. A true Laplacian blend needs the tiles decomposed and
recombined per frequency band, not a per-pixel weight.

**What to do.** `gaussian_feather` and `linear` are genuinely different; pick
between those.

---

## VFX

### Optical flow is Lucas–Kanade, not DIS

**What happens.** Despite the DIS reference in its docstring, the implementation
is a single-scale, non-iterative windowed least-squares solve. It saturates at
about one pixel.

**Measured** on synthetic pure translation of band-limited texture:

| True displacement | Recovered |
| ---: | ---: |
| 1 px | ~104% |
| 2 px | ~51% |
| 3 px | ~17% |
| 5 px | ~1% |
| 10 px | ~0% |

The Fast / Medium / Ultra preset changes the answer by less than 1%, so "Ultra"
buys nothing.

**What to do.** Use it for sub-pixel stabilisation and very slow motion. Mask
propagation and vector blur built on it are effectively static above about two
pixels of motion — do those in a dedicated tracker.

### Multipass extraction is estimation, not render passes

**What happens.** Multipass Master derives albedo, roughness, ambient occlusion
and a segmentation ID from a single image. That is genuinely useful on generated
footage and 2D plates, but it is inference, not ground truth. The segmentation
output is a clustered matte, not a Cryptomatte.

**What to do.** For real passes, render them and read the multilayer EXR through
the Multipass AOV Reader.

---

## AI Assist

### Scene-cut detection reports cuts in cut-free footage

**What happens.** Scores are normalised by the batch maximum, so the largest
inter-frame difference in *any* clip becomes exactly 1.0 and therefore always
exceeds the threshold. `threshold` has no absolute meaning despite the tooltip
describing typical values.

**Measured** on a smooth horizontal pan with no cut (48 frames, default
`threshold=0.35`, `min_shot_frames=12`): detected cuts at frames 0, 12, 24 and
36 — exact multiples of `min_shot_frames`, the signature of thresholded noise.
On the same pan with one real hard cut at frame 24, the real cut is
indistinguishable from the false ones.

**What to do.** Treat the output as a starting point to check by eye. Do not
wire it into an unattended pipeline.

---

## Other

### `RadianceVideoAssembler` output is not a function of its inputs

It accumulates into a class-global store keyed on a `session_key` widget whose
default is the same literal for every instance. Two Assembler nodes left at the
default in one graph interleave into one bucket. Give each one a distinct
`session_key`.

### The Viewer re-runs on every queue

`IS_CHANGED` returns NaN by design, so the delivery frame cache always holds
current frames. The cost is re-execution and sidecar rewrites even when nothing
upstream changed.

### `fast_vae` tiled decode has visible tile seams

The tiled path pastes tiles without blending, leaving a one-pixel grid on every
`tile_size` boundary. Use a larger tile size, or the non-tiled path, where memory
allows.

---

## Reporting something not on this list

Open an issue at
[github.com/fxtdstudios/radiance/issues](https://github.com/fxtdstudios/radiance/issues).
What helps most: the node, the widget values, what you expected, what you got,
and the relevant lines from the ComfyUI console. Radiance logs an ERROR at
startup if it loaded fewer nodes than expected, and names each module that
failed — if you see that, include it.
