# Radiance — Full Code Audit

**Repo:** `D:\A.I\ComfyUI\custom_nodes\radiance` · v3.1.2
**Scope:** 212 Python modules (71,364 lines excl. tests) · 59 test files (16,349 lines) · 29 JS files (43,237 lines)
**Date:** 2026-07-27
**Method:** five parallel deep-read passes (security, correctness/concurrency, performance/GPU, architecture/packaging/tests), then a separate **adversarial verification pass** whose job was to *refute* every finding. Numeric claims were proven by extracting the maths and running it. Sandbox escapes were proven by executing the payload. Roughly a dozen claims from the first pass were corrected or downgraded by verification and are marked accordingly; nothing unverified is published here.

---

## Verdict

**Request changes.** The engineering underneath this is strong — the colour-science test suite, the `.rad` zip hardening, the path-containment helper, the loopback-by-default network config and the zero Python-side monkeypatching of ComfyUI are all better than most node packs manage. But three classes of problem are live on default paths:

1. **The default upscaler produces noise.** Real-ESRGAN weights are silently not loaded — 8 of 702 tensors match, 0.46% of parameters — and the run reports success.
2. **Two command sandboxes are escapable to full RCE**, proven by execution, plus a pickle-RCE `torch.load`.
3. **A third of the node surface never registers** (54 of 154), and when torch is missing the pack loads 12 of 100 nodes while printing "successfully loaded".

Beneath those, a recurring theme: **failures are converted into plausible-looking successes**. Blanket `except` handlers return the input unchanged, a black frame, or the raw noise, so the graph completes and the artist ships the wrong pixels. That pattern — not any single bug — is the thing most worth fixing.

### Severity counts (verified only)

| | Security | Correctness | Performance | Architecture |
|---|---:|---:|---:|---:|
| Critical | 3 | 12 | 8 | 1 |
| High | 3 | 9 | 12 | 7 |
| Medium | 4 | ~30 | 14 | 9 |

---

## Part 1 — Security

Threat model: (a) a shared workflow JSON supplying node inputs, (b) a request hitting an exposed `/radiance/*` route, (c) a downloaded model or asset.

### S1 · Critical — Pickle RCE via `torch.load(weights_only=False)`

`pixel_sdr2hdr.py:299`

```python
checkpoint = torch.load(str(resolved), map_location="cpu", weights_only=False)
```

Reachable from the `pixel_checkpoint` STRING widget on `RadianceSDRToHDRUniversal` (`nodes/hdr/uplift_universal.py:464`), via `convert()` → `_pixel_reconstruct` (`:567`) → `predict_pixel_sdr2hdr` → `load_pixel_sdr2hdr_weights` → `resolve_pixel_checkpoint`. Backend defaults to `Auto`, and a non-empty string activates the path (`:555-557`).

**Verification nuance:** `resolve_pixel_checkpoint` requires `candidate.is_file()`, so this is a two-artifact attack — a workflow JSON *plus* a malicious `.pt` at a predictable local path (bundled alongside, or a "required model" the workflow's README tells the artist to download). Still critical under ComfyUI's model-sharing norms, but not workflow-JSON-alone.

**Also:** `scripts/training/train_turbo_decoder.py:356` and `:614` call `torch.load` with **no** `weights_only` argument at all, which defaults to unsafe on torch < 2.6. Lower reachability (CLI training scripts, path from argv) but the same class of bug. All 11 other `torch.load` sites correctly pass `weights_only=True`.

**Fix:** `weights_only=True` — the loader only needs `checkpoint["config"]` (ints) and `checkpoint["model"]` (a state dict), both of which survive. Constrain the path via `folder_paths.get_full_path`.

### S2 · High — DCC bridge sandbox escape → RCE (proven by execution)

`nodes/pipeline/dcc.py:68-93` (`_validate`), `:96-121` (`_exec_sandbox`)

`_ALLOWED_AST_NODES` (`:63`) includes `ast.Assign`, and `_validate` checks attribute access against the **AST name** `json`, not the runtime object. Rebinding walks straight out:

```python
json = json.codecs        # the real codecs module
json = json.builtins      # the real builtins module
json.exec("...", json.vars(json))
```

Confirmed end to end: `_validate` returns `(True, 'ok')` and the payload wrote a file to disk and executed `os.system`. Even bare `json.open(path, 'w')` validates and truncates an arbitrary file.

**Preconditions:** `RADIANCE_DEV=1` (`:92-93` — without it only `ast.literal_eval` runs) and a loopback peer with a loopback bind (`:146`). A workflow JSON *can* start the listener (`RadianceMCP`, mode `Bridge Server`, `:410`) but cannot itself send the exec command.

**Fix:** delete the eval/exec fallback entirely; keep `ast.literal_eval` plus the already-implemented `_ALLOWED_STRUCTURED_ACTIONS` dispatch, which covers the real use cases. Blocklist-and-allowlist evaluation of attacker-supplied source is not salvageable.

### S3 · High — Nuke bridge: substring blocklist bypass → RCE, unauthenticated by default

`scripts/start_nuke_server.py:83-114` (`_validate_command`), `:19` (auth), `:405` (bind)

Substring matching on the lowercased command. Confirmed bypasses, run against the real `safe_globals`:

- `open ('/etc/passwd')` — a space defeats the `open(` pattern
- `json.codecs.open ('/etc/passwd').read()` — module chaining
- `json.codecs.builtins.exec (...)` — **wrote `/tmp/PWNED_C3`**

`DCC_AUTH_TOKEN = os.environ.get("RADIANCE_DCC_AUTH_TOKEN", "")` defaults empty, so the `if DCC_AUTH_TOKEN:` gate at `:270` is skipped entirely — **no authentication by default**. `RADIANCE_NUKE_BIND_HOST` can be set to `0.0.0.0` (default is loopback).

Separately, the auth signature is `SHA256(token ‖ command)` (`:279-283`) — no nonce (replayable forever) and length-extendable. Use `hmac.new(...)` with `compare_digest`.

Mitigating, and worth stating alongside the finding: eval/exec here is also gated behind `RADIANCE_DEV=1` (`:20`, `:347`).

**Fix:** serve only structured actions; make the token mandatory; refuse to start on a non-loopback bind with no token.

### S4 · Medium — 31 `/radiance/*` routes with no authentication of any kind

`nodes_workspace.py` (~25 via the `_route` helper at `:588`), `nodes_gizmo.py` (2), `radiance_ocio.py` (4), `delivery/handler.py:116`.

Grepping every handler for auth/token/CSRF/origin returns only false positives (`author`, graph-node `origin`). Reachable side effects include:

- `delete_workflow` (`nodes_workspace.py:1335`) — `unlink()` the file (`:1350`), the sidecar (`:1355`), and **every backup in `.versions/`** (`:1358-1362`)
- `/radiance/assets/upload` (`:1807`) — writes into ComfyUI's input dir
- `/radiance/deliver` (`delivery/handler.py:116`) — triggers a full render + encode to a caller-influenced path

The *containment* logic is genuinely good (see "verified clean" below); the gap is that there is no gate in front of it.

**Fix:** at minimum an `Origin` check on mutating routes; ideally a shared-secret header. Document that the ComfyUI port must not be exposed.

### S5 · Medium — `/radiance/ocio/load` accepts any absolute path and echoes it

`radiance_ocio.py:585-600`. Only guard is `abspath` + `isfile`; the error body is `f"File not found: {config_path}"` — a file-existence oracle. On a hit, `OCIO.Config.CreateFromFile()` parses an arbitrary file, and OCIO configs carry `search_path`/`FileTransform`, so an attacker-controlled config reads further files during `/radiance/ocio/bake`.

### S6 · Medium — Arbitrary filesystem write from workflow inputs

`nodes_io.py:1624-1629` — `_out_path` builds the destination straight from the `output_path` widget with **zero** containment:

```python
p = Path(base).with_suffix(ext)
if not overwrite:
    p = _unique_path(p)
p.parent.mkdir(parents=True, exist_ok=True)
```

Four other sites call the safe helper but disable it — `allow_absolute=True` hardcoded at `nodes_io.py:1934`, `nodes/vfx/multipass/master.py:539` and `:624`, `nodes/color/qc.py:343`. `get_safe_output_dir` (`core/system/path_utils.py:43-58`) returns the path unchanged in that mode.

Extension is forced by `format`, so this clobbers `.exr/.png/.mp4/.dpx/.json` targets rather than dropping a `.py` — that is what holds it at Medium.

**Fix:** default `allow_absolute=False`; require an explicit per-node opt-in.

### S7 · Medium — ffmpeg/ffprobe SSRF via workflow-supplied path

`nodes_io.py:606-607`, `:638`, `:722`; `nodes/pipeline/audio.py:142`, `:236`. All use argv lists (no shell injection), but hand the user string to ffmpeg as a **URL**, with no `-protocol_whitelist`.

The enabler is `_path_kind` (`nodes_io.py:302-327`), which classifies **purely by file extension with no `isfile()` check**. So `path = "http://169.254.169.254/latest/meta-data/x.mp4"` classifies as video and reaches `ffmpeg -i http://...` — confirmed SSRF. `concat:` file-disclosure is only partially reachable (needs extension crafting). In `_read_video` the path is also positional, so a value starting with `-` is parsed as an option.

**Fix:** `os.path.isfile()` before invoking; `-protocol_whitelist file`; prefix positional paths with `--`.

### S8 · Low — assorted

| Finding | Location |
|---|---|
| Gizmo executor bypasses `INPUT_TYPES` validation — calls any installed node with unvalidated kwargs; paired with an unauthenticated create route | `nodes_gizmo.py:131-180`, `:320-366` |
| `restore` route omits the `_validate_extension` check its three siblings apply | `nodes_workspace.py:1443-1448` |
| `filename_prefix` unvalidated (`master.py:542` guards the same field; this one doesn't) | `nodes/color/qc.py:348` |
| `np.load(allow_pickle=True)` — currently unreachable (module's mappings are empty) but live code | `nodes/video/character.py:185` |
| `openai_api_key` STRING widget persists a secret into saved workflows; the correct env-var pattern already exists beside it at `:614-617` | `nodes/pipeline/audio.py:563-565` |
| `settings.reveal_folder` on the unauthenticated `/radiance/deliver` triggers `os.startfile`/`xdg-open` on the server's desktop | `delivery/handler.py:494-504` |
| MD5 used for cache keys (not security-critical, but bandit-flagged) | `radiance_ocio.py:431`, `:533` |

### Verified clean

Independently re-checked and confirmed — worth knowing what *isn't* a problem:

- **No `yaml`** anywhere. **No `shell=True` / `os.system` / `os.popen`.** Every `subprocess` call is an argv list with a constant program name.
- **No `tarfile` / `extractall`.** `zipfile` *is* used, and `_unpack_rad_v3` (`nodes_workspace.py:455-500`) is properly hardened: entry-count cap, uncompressed-size cap (zip-bomb), and explicit `Path(name).name != name` rejection of path components.
- **`_resolve_safe_path`** (`nodes_workspace.py:213-231`) does `.resolve()` **before** `.relative_to()` — the ordering most implementations get backwards — plus length and control-byte checks, and is applied on all eight filename-bearing routes.
- **No hardcoded secrets.** `core/system/secret_utils.py` resolves credentials by env-var *name* with a strict identifier regex and never logs values.
- **No `verify=False`**, no unverified SSL context. All download URLs come from `config/model_map.py` constants — no user-controlled URL reaches `urlopen`. (Though **no entry pins a checksum**, so the verification code at `loader_utils.py:78-93` never runs — worth pinning.)
- **All servers default to `127.0.0.1`** (`config/env.py:89, 97, 105, 113`); `dcc.py:199-205` actively refuses a non-loopback bind without an explicit opt-in.
- **JS XSS is mostly handled** — `escapeHtml` applied consistently in the dashboards; both `new Function` sites are behind a `localStorage` dev flag. One genuine hole: `js/radiance_viewer.js:3875-3879` interpolates `result.path` and `dvFilename.value` into `innerHTML` unescaped, while the *same file* escapes correctly at `:1708`.
- **DoS:** all INT widgets carry min/max; LUT sizes clamped; workflow payloads capped at 50 MB with both a Content-Length fast path and post-read enforcement.

---

## Part 2 — Correctness

Every numeric claim below was reproduced by extracting the maths and running it.

### C1 · Critical — The default upscaler runs a near-randomly-initialised network

`nodes/upscale/upscale.py:632-634`

```python
state = {k.replace("module.", ""): v for k, v in state.items()}
net.load_state_dict(state, strict=False)
```

`_RRDBNet` names its convs `c1…c5` / `upsample` (`:552-556`, `:583`, `:592`). Official Real-ESRGAN checkpoints use `body.N.rdbM.conv1…conv5` / `conv_up1` / `conv_up2`. `strict=False` turns the mismatch into silence.

**Measured:**

```
model params (nb=23, scale=4)  : 702
checkpoint params_ema          : 702
MATCHED (loaded)               : 8    (conv_first, conv_body, conv_hr, conv_last — weight+bias)
silently dropped               : 694
parameters actually loaded     : 77,379 / 16,697,987 = 0.463%
```

`tier1_fast` is the default at `:1247`, `:1281`, `:1312`, `:1783`, `:1863`. Because `load_state_dict` raises nothing, the `except` at `:1147` never fires, no bicubic fallback occurs, and the log prints `Loaded realesrgan_x4plus (nb=23, scale=4x)`. **Output is coloured noise, reported as success.**

The 2× path (`realesrgan_x2plus`) *does* hit a genuine size mismatch on `conv_first` and safely falls back to bicubic — so only the 4× and 8× default paths are affected.

**Fix:** `strict=True` (which would have caught this on day one), and rename the modules to match basicsr — or route Tier 1 through spandrel like Tier 2.

### C2 · Critical — Every delivery fails, and returns HTTP 200

`delivery/handler.py:441-450` calls `writer.write(filename_prefix=..., output_format=..., output_color_space=..., ...)`. `RadianceWrite.write` (`nodes_io.py:1633-1654`) has none of those three parameters — it has `format` (required, no default), `filename`, `color_space`. Three unexpected kwargs plus one missing required argument → `TypeError` before the body runs. And the only success return is `nodes_io.py:1715 return ()`, so even a corrected call fails the 3-tuple unpack at `:441`.

Swallowed at `handler.py:562-564` into `web.json_response({"error": ..., "status": "error"})` — which **defaults to HTTP 200**. No file is ever written by `POST /radiance/deliver`.

### C3 · Critical — Sequence reads are broken at the node's own defaults

`nodes_io.py:551`

```python
missing.append(p)
if missing_frames in ("Black", "Skip"):
    paths.append(p)   # keep slot for black-frame insertion
```

"Skip" doesn't skip. Defaults are `start_frame=1001`, `end_frame=0` (`:1248-1255`), and `:1356-1359` expands `end_frame=0` to 99999 → **98,999 slots**. Missing slots become `torch.zeros(1, 8, 8, 3)` (`:585`), real frames are full-res, so `torch.cat` at `:588` raises — and the blanket handler at `:1371-1374` returns an 8×8 black image with only a log line.

**Reading a numbered sequence never works at defaults.**

### C4 · Critical — Four independent blend ramps start at exactly 0 → black frames

The single most-repeated bug in the codebase. A cosine or linear ramp whose first element is 0, applied to *all* tile edges including image borders, followed by division by accumulated weight.

| Site | Ramp | Result |
|---|---|---|
| `sampler_utils.py:1642` | `(1-cos(linspace(0,π,fade)))/2` | Outer latent row/col = exactly 0 → 8-px black band on all four edges after VAE |
| `image/upscale.py:732` | `np.linspace(0, 1, overlap)` | 1-px black frame around every tiled output |
| `image/upscale.py:2489` | same | same, torch path |
| `nodes/upscale/upscale.py:1943` | `sin(π·i/(2·half))`, `i=0` | **Frame 0 of every video render is pure black** |

Reproduced numerically for the first (1024², tile 128, overlap 16): `weight row0 min/max: 0.0 0.0`, result row 0 `[0. 0. 0. 0. 0.]`, interior `0.5`. Index 1 gets weight `1.19e-4` → an ~8400× noise-amplification ring beside the black line.

**The correct pattern already exists in this repo.** `hdr/vae.py:2011-2014`:

```python
ov_top   = pix_overlap if yi > 0 else 0
ov_bot   = pix_overlap if yi < len(tiles_y) - 1 else 0
ov_left  = pix_overlap if xi > 0 else 0
ov_right = pix_overlap if xi < len(tiles_x) - 1 else 0
```

Four missing lines, four times.

**Gating:** sites 1–3 require tiling to be enabled (`tile_mode` defaults to `False`). Site 4 is **ungated** — `overlap_temporal` has `min: 1` and defaults to 4, so frame 0 is black on every multi-frame upscale with no setting that avoids it.

### C5 · Critical — DaVinci Intermediate constants wrong in two of three implementations

`color/transfer.py:334-357` and `hdr/color.py:886-910` are both wrong, in different ways. `color/luts.py:251-259` is correct.

```
                       f(0.18)      f(0.0)     jump at cut    decode monotonic?
color/transfer.py      0.874523    +0.345557    +0.064226     No  (min out −0.109909)
hdr/color.py           0.522996     0.000000    +0.186953     No  (min out −0.005772)
SPEC / color/luts.py   0.336043     0.000000     0.000000     Yes
```

Round-tripping real DI-encoded footage through `transfer.py`'s decoder: linear `[0, 0.001, 0.005, 0.02, 0.09, 0.18, 0.5, 1.0, 4.0]` comes back as `[−0.10991, −0.10659, −0.0941, −0.06758, −0.02502, −0.00303, 0.00326, 0.0064, 0.02475]` — negative blacks and a ~40× crush.

`hdr/color.py:897` uses `b * 0.1` (0.7) where the spec requires `b * c` (0.513047).

The broken copies are the ones wired into the VAE (`hdr/vae.py:1064`), `fast_vae.py:333`, `nodes_io.py:289` and `color/pipeline.py`. **Adopt `color/luts.py`'s formulation everywhere and delete the other two.**

*(First-pass correction: `f(0.0)=0.345557` belongs to `transfer.py` only; `hdr/color.py` returns 0 there and is wrong differently.)*

### C6 · Critical — AgX "sigmoid" flattens everything above mid-grey

`hdr/tonemap.py:351`

```python
x_sig = x_log / (1.0 + torch.abs(x_log - 0.5) * 2.0)
```

For `x_log > 0.5` the denominator is `2·x_log`, so this is `x/(2x) = 0.5` **exactly**. Verified: `x_log ∈ [0.5, 0.5625, 0.625, …, 1.0] → all 0.5`. Since `x_log = (log2(x)+10)/16.5`, the threshold is `2^(0.5·16.5−10) = 0.297302` scene-linear.

```
scene-linear: [0.05    0.18    0.2973  0.5   1.0   4.0   16.0  100.0]
output      : [0.14351 0.37899 0.49999 0.5   0.5   0.5   0.5   0.5  ]
```

A sky at 5.0 and a specular at 100.0 render identically. The shadow side is crushed too — after the CDL step at `:352`, everything below ~0.0126 scene-linear clamps to 0.

**Fix:** a monotonic sigmoid, e.g. `0.5 + (x−0.5)/(1+(2|x−0.5|)^p)^(1/p)`.

### C7 · Critical — "8× (tile cascade)" produces a 16× image

`nodes/upscale/upscale.py:1318`, copied verbatim at `:1631` and `:1883`:

```python
scale_int = {"2×": 2, "4×": 4, "8× (tile cascade)": 4}[scale]
...
if do_double:
    upscaled, conf2 = tiled_upscale(upscaled, _fn, scale=4, ...)
```

4 × 4 = 16. A 1024² input becomes 16384² — the accumulators alone are 3.2 GB fp32. The Tiler's info string papers over it by deriving `eff_scale = oH // H` (`:1355`), so it reports "16×" while the widget says "8×". Second pass must be `scale=2`, in all three copies.

### C8 · Critical — Voronoi and Simplex noise are identical across batch and channels, and not unit-variance

`sampler_utils.py:1313`, `:1381`

```python
return noise_2d.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
```

`generate_noise` (`:1470-1473`) returns these with no renormalisation. Measured:

```
VORONOI   128²: mean −0.4084  std 0.3203     (should be 0.0 / 1.0)
           64²: mean −0.2854  std 0.3718
SIMPLEX   128²: mean +0.0014  std 0.0640     (≈16× under unit variance)
expanded tensor: max|ch0−ch3| = 0.0   max|b0−b1| = 0.0
```

Every image in a batch is pixel-identical and every latent channel carries the same field. The sampler receives ~0.3× the required noise with a large DC offset → washed-out grey. Simplex's octave frequencies (1,2,4,8 over the whole image) resolve only a handful of distinct cells, making it a near-constant gradient rather than noise.

**Fix:** generate per-(B,C) fields; normalise to zero-mean/unit-std as `_perlin_noise_2d` and `_spectral_noise_2d` already do.

### C9 · Critical — DWG and AWG4 → XYZ matrices have wrong third rows

`hdr/color.py:862-869`, `:983-990`. A colour matrix's row sums are its implied white point:

```
DWG_TO_XYZ   row sums = [0.95070, 1.00000, 0.90030]   white xy = (0.33346, 0.35075)
AWG4_TO_XYZ  row sums = [0.95050, 1.00000, 0.85346]   white xy = (0.33899, 0.35664)
D65 required           = [0.95047, 1.00000, 1.08883]   white xy = (0.31273, 0.32902)
_SRGB_TO_XYZ row sums = [0.95047, 1.00000, 1.08883]   ✓ PASS
```

Rows 1–2 of DWG match a from-primaries derivation to 4 dp, so this is a third-row transcription error, not a wrong matrix. Blue is ~17–22% short → every neutral picks up a green/yellow cast. Correct third rows: DWG `[−0.098963, −0.137895, 1.325916]`, AWG4 `[0, 0, 1.089058]`.

A one-line test — `M @ [1,1,1] ≈ white_XYZ` — catches both, and does not exist.

*(`_AP1_TO_XYZ` at `:826` sums to D60, not D65, which is **correct** for ACEScg — not a fourth bug.)*

### C10 · Critical — The HDR video node clamps its input to [0,1] as its first pixel op

`nodes/video/hdr.py:344`

```python
x_lin = x.clamp(0, 1).pow(2.2)   # "Convert sRGB [0,1] → approximate scene-linear"
```

Feed a linear EXR with a 6.0 specular, or the output of `RadianceVideoBatchDecode(output_linear=True)` which deliberately doesn't clamp, and every legitimate highlight becomes 1.0 — inside the node whose entire purpose is preserving headroom.

**And the SDR preview is ~8× too dark.** `:376-378` computes `sdr_scale = sdr_preview_nits/10000` then Reinhards to that peak and never renormalises. Reinhard asymptotes at `peak`, so the result lives in `[0, sdr_scale]`:

```
white (1.0), peak_nits=1000, sdr_preview_nits=100:
  x_nit = 0.1  →  reinhard(0.1, peak=0.01) = 0.00909  →  ^(1/2.2) = 0.11805768
```

The preview can never exceed ~0.128 for any input. The HDR branch has the same defect: diffuse white encodes at 500 nits on a 1000-nit target.

### C11 · Critical — OCIO context mutates a process-global singleton, and its output wire is decorative

`nodes/color/ocio.py:44` calls `mgr.load_config(path)` on the module singleton `radiance_ocio._ocio_manager` (`:549-557`), which swaps `self.config` and clears `_lut_cache` (`:165-167`) for every other node and the HTTP viewer. `nodes/color/colorspace.py:198-203` re-reads `mgr.config` at execute time, so branch A's plate gets graded through branch B's config — silently, and across executions since the context node is a cache hit on re-run.

**Worse than first reported:** grepping the whole repo, `"RADIANCE_OCIO"` appears exactly once — as the `RETURN_TYPES` at `nodes/color/ocio.py:26`. **No node accepts it as an input.** `colorspace.py` has no `ocio_context` parameter at all. The wire is purely decorative; last-executed-wins is the only behaviour there is, not an edge case.

### C12 · Critical — 3D VAE decode unpacks axes in the wrong order

`nodes/video/t2v.py:889-891` and `:1254-1256`

```python
B, C, T, H, W = decoded.shape
decoded = decoded.permute(0, 2, 3, 4, 1).reshape(B * T, H, W, C)
```

The repo documents its own convention twice (`hdr/vae.py:1968`, `:2966`: *"3D VAEs return (B, F, H, W, C)"*) and it matches upstream `comfy/sd.py:1169` (`movedim(1,-1)`). An LTX decode of `(1, 16, 512, 512, 3)` binds `C=16, T=512, H=512, W=3` and reshapes to `(512, 512, 3, 16)` instead of `(16, 512, 512, 3)`. Element counts match, so **nothing raises**.

**Severity qualifier:** both sites are `_decode_preview`, the second return value of the T2V/I2V nodes. The latent output is unaffected — this corrupts the preview, not the generation.

### C13 · Critical (opt-in) — Restart sampling double-counts the latent

`nodes_sampler.py:710-724`

```python
noise = torch.randn_like(result) * r_val
noisy = result + noise
result = _sample_custom_progress_safe(..., noise=noisy, latent_image=result, sigmas=sub_sigmas, ...)
```

Upstream (`comfy/model_sampling.py:39-47`, EPS) computes `x = σ₀·noise + latent_image`. Substituting gives `x = (1+σ₀)·result + σ₀·r·ε` — the finished latent is scaled by (1+σ₀) and the injected noise is squared in amplitude (r² instead of r). On CONST/flow-matching models (Flux, SD3, WAN) the latent isn't scaled but the noise is still r², making restart a near-no-op.

The same file proves the author's model is correct elsewhere — `:1469` passes `torch.zeros_like(noise)` for continuation stages.

**Gating:** `restart_count` defaults to 0.

### C14 · High — Midtones darken as you raise the HDR peak

`hdr/color.py:1307` — `result[..., c] = channel_shoulder / peak_scale` divides *everything below the shoulder* by `peak_scale`:

```
peak_nits=1000: 18% grey → 1.80 nits   (should be ~10–26)
peak_nits=4000: 18% grey → 0.45 nits
```

Highlights are correct; midtones and shadows get darker the higher you set the peak.

*(This finding replaces a first-pass claim that `_pq_encode` ignoring its `peak_nits` argument makes all PQ output peak at 100 nits. **That was refuted** — `peak_nits` is genuinely redundant there because `_apply_tonescale_drt` already returns values scaled to `peak_luminance/100`, and PQ output does reach 1000/2000 nits. The unused parameter at `hdr/color.py:1334` is cosmetic; the real defect is 27 lines above it.)*

Separately, `hdr/color.py:1297` — the ACES 2.0 tonescale shoulder has a hard **10% step** at `0.9 × white_scale` (`tanh(0)=0` makes the branches disagree): `0.8999·ws → 0.89990`, `0.9001·ws → 0.99999`. A visible contour ring around every highlight, non-monotonic thereafter.

### C15 · High — Noise generation is device-dependent

`sampler_utils.py:1458-1460` — `torch.manual_seed(seed)` then `torch.randn(shape, device=device)`. The same seed gives different noise on CPU vs CUDA, so a workflow is not reproducible across machines. `nodes_sampler.py:572-575` routes Gaussian through comfy's device-independent CPU path but **every other noise type** through this one.

*(First-pass correction: the "resets the process-global RNG, correlating downstream nodes" part is **not a Radiance defect** — comfy's own `prepare_noise` calls `torch.manual_seed` identically at `comfy/sample.py:27`. What comfy does differently is `device="cpu"`. Only the device-dependence is a real deviation.)*

`nodes/vfx/optics.py:512` has the same device issue, plus: `seed=0` skips seeding while `IS_CHANGED` (`:453`) hashes `seed`, so the documented "0 = random each run" instead **freezes one cached result forever**.

### C16 · High — `broadcast_safe` clamps EXR and 32-bit TIFF output

`nodes_io.py:1688` applies `np.clip(fr, 16/255, 235/255)` unconditionally *before* format dispatch. Delivering "EXR (32-bit float)" with `broadcast_safe=True` yields a scene-referred file with no value below 0.0627 or above 0.9216 — and `delivery/handler.py:180` makes this the **default** (`soft_clip=True`).

### C17 · High — Silent failure patterns (systemic)

**~30 sites** convert a failure into a plausible-looking success. The worst:

| Location | Failure → what the user sees |
|---|---|
| `nodes_io.py:1371` | Missing/corrupt file → 8×8 black IMAGE, graph completes |
| `nodes/video/t2v.py:213` | Sampling OOM → **the raw noise returned as a finished latent**, reported as `Sampled: [1,128,7,64,64]` |
| `sampler_utils.py:1627` | One OOM tile → that region is undenoised latent, blended in, warning only |
| `image/upscale.py:2420, 2435` | Model scale guessed from filename; RGBA into a 3-channel net → **silent bicubic** labelled "AI model not available" |
| `nodes/upscale/upscale.py:2385` | No face-restore backend → 512² identity round-trip, counted as `restored_faces += 1` |
| `nodes_workspace.py:786, 1560` | Truncated `.shot_status.json` → `{}` → next write **erases every other shot's status** |
| `color/lut.py:421, 434` | LUT node silently becomes a pass-through |
| `nodes_realtime_preview.py:900` | Failed port bind swallowed; node still returns a working-looking URL |

Of 66 literal `except…: pass` sites, ~6 hide genuine data loss; the rest are legitimately best-effort. Ruff counts **353 blind excepts** (`BLE001`) overall.

### C18 · High — HDR clamped inside the HDR pipeline (systemic)

**23 sites** judged genuinely harmful out of ~70 examined, concentrated in `nodes/upscale/upscale.py` (1368, 1585, 2019, 2560, 2764), `nodes_io.py` (260, 1688), `nodes/video/hdr.py` (344, 377), `delivery/handler.py` (276, 289), `color/lut.py:263`.

The codebase knows the rule — `film/grain.py` documents "ZERO clamping", `hdr/vae.py` carries a changelog of clamp removals — but the discipline stops at module boundaries. Two halves actively fight: `nodes/upscale/upscale.py:1697` applies an *inverse* Reinhard (`y/(1−y)`) to output every backend has already clamped to [0,1], exploding clipped pixels to **999999**; and `:1613` clips HDR in `_pre_denoise` before the auto-HDR detector reads `_hdr_in_max`, so `hdr_mode="auto"` never triggers when `denoise_pre > 0`.

### C19 · Medium — Selected further findings

| Finding | Location |
|---|---|
| Version-backup eviction sorts filenames lexicographically → `shot.v10.rad` deleted before `shot.v2.rad` (the adjacent "FIX 4" comment fixed exactly this for `next_num`, not for eviction) | `nodes_workspace.py:342` |
| `_create_version_backup` is a cross-thread read-modify-write with no lock — two racing saves both write `.v12` and one commit is lost | `nodes_workspace.py:333` |
| `RadianceProjectManager` has no `IS_CHANGED`; edit the graph, hit Queue → cache hit, nothing written, no error | `nodes_workspace.py:141` |
| `SigmaCache` fallback key is `id(model)` — CPython reuses freed addresses → wrong sigmas, silently | `sampler_utils.py:304` |
| Viewer cache keyed by LiteGraph node id, which collides across workflows → export pulls the wrong plate | `nodes/monitor/viewer.py:312` |
| Gaussian tile weight normalised to sum=1, so a 40-row edge tile is **45× heavier** than a full tile → hard horizontal seam whenever height isn't a multiple of step | `nodes/upscale/upscale.py:371` |
| `overlap ≥ tile_size` unvalidated at 4 of 5 call sites → `step=1` (effectively a hang) or a broadcast error | `nodes/upscale/upscale.py:462`, `image/upscale.py:647` |
| Alpha treated as a colour channel — transfer functions and tone-maps applied to `[..., :]` instead of `[..., :3]` (**8 sites**) | `nodes_io.py:1686`, `delivery/handler.py:406`, `nodes/video/hdr.py:339`, `film/camera.py:730`, `color/grading.py:182/251/262`, `nodes/vfx/plate.py:65` |
| Depth Anything reads `out["depth"]` (a uint8 PIL preview) instead of `out["predicted_depth"]` → 256-level depth, terraced world-position and staircased SSAO | `nodes/vfx/multipass/core.py:674` |
| Aspect-ratio blanking parses only the numerator — `float("16:9".split(":")[0]) = 16.0` → a 1920×1080 "16:9" delivery blacks out rows 0–479 and 600–1079 | `delivery/handler.py:335` |
| `avg_pool2d` zero-pads in log2 space, where 0.0 means linear 1.0 → darkened frame borders (corners at 0.29× target) | `nodes/vfx/plate.py:55` |
| `alpha_invert` applied *after* unpremultiply → solid black over the entire object | `nodes/vfx/multipass/relight_comp.py:382` |
| `ColorSpaceConvert` silently skips the gamut matrix for DCI-P3, Display_P3, ACES2065-1 (key `"Rec709_to_DCI-P3"` isn't in `FAST_MATRICES`) → Rec.709 data mislabelled as P3 | `hdr/color.py:769` |
| `torch.quantile` hard-fails above 2²⁴ elements; one 3840×2160 beauty is 24.9M → RuntimeError on exactly the film-res case the node targets | `nodes/vfx/multipass/core.py:955` |
| Block-quantisation reshape crashes on any dimension not divisible by `block_size` (1920×1080, block 16) | `film/camera.py:886` |
| Halation blur is a non-separable 2-D conv up to 181×181 (~8×10¹¹ MACs on 4K — appears to hang) | `film/grain.py:584` |
| Image formats write `frames[0]` only — a 24-frame batch loses 23 and logs "saved 1 frame(s)" | `nodes_io.py:1767` |
| `Path(base).with_suffix(ext)` truncates dotted stems: `/out/take.001` → `/out/take.exr`, silently overwriting renders | `nodes_io.py:1625` |
| `proxy_scale` downscales the IMAGE but not the MASK | `nodes_io.py:1328` |
| `cv2.imwrite` and `_copy_to_remote_path` return values discarded — a failed write or NAS delivery reports success | `nodes_io.py:1025`, `:1996` |
| Heavy blocking I/O inside `async` handlers (rglob over the output tree, sync cv2 EXR decode, 65³ LUT bake) freezes ComfyUI's websocket mid-render | `nodes_workspace.py:915`, `:1654`, `radiance_ocio.py:646` |
| `RADIANCE_CACHE_SIZE=0` → `popitem` on an empty OrderedDict → `KeyError` fails the whole render | `model/cache.py:31` |
| sqlite connections opened without `with`/`finally`; a `json.loads` failure leaks the fd and holds the write lock → "database is locked" | `core/param_memory.py:70` |
| Dead widgets that do nothing: `blend_mode` (`upscale.py:430`), `exr_compression` (`nodes_io.py:1709`, threaded two levels deep and never read), `shape_type` (`roto.py:58`), compression/bit-depth in `master.py:582` (unreachable behind an always-true guard) | various |

### C20 · Systemic patterns

1. **Ramps starting at 0 → division by accumulated weight** — 4 implementations, 4 bugs, 1 correct counter-example (`hdr/vae.py:2011`). No test covers any of them.
2. **`except Exception` returning the input unchanged** — ~30 sites, ~6 causing genuine data loss.
3. **Duplicated transfer functions and matrices that disagree** — DaVinci Intermediate exists in 3 places (1 correct); S-Log3 and Canon Log 3 each have a correct copy in `color/luts.py` and a broken one in `color/transfer.py`. **Five piecewise curves are discontinuous or non-monotonic at their own cut point.** Each is caught by a two-line test evaluating at `cut ± ε`. None exists.
4. **Duplicated-and-diverged tiling** — 5 implementations, 3 last-tile strategies, 3 weight schemes, 3 different bugs. The `overlap ≤ tile_size/2` clamp exists at exactly 1 of 5 call sites. The 8×→16× bug is copy-pasted 3×; the inverse-Reinhard blowup 2×.
5. **Unlocked global mutable singletons reachable from two threads** — 7 sites (`radiance_ocio._ocio_manager` + `_lut_cache`, `_PREVIEW_BUFFER`, `dcc._SERVER*`, `core/logging`'s dedupe filter, `model/cache.py`, `sampler_utils._freq_grid_cache`, `fast_vae._TRAINED_DECODER_CACHE`). ComfyUI runs aiohttp on the main thread and execution on a worker. `cache.py`, `color/lut.py`, `nodes/vfx/depth.py` and `image/upscale.py` **do** use `threading.RLock` correctly — the pattern is understood, just applied inconsistently.
6. **Caches keyed by unstable identity** — 3 sites (`id(model)`, `id(self)`, per-workflow node id as a global key).
7. **`IS_CHANGED` coverage: 8 definitions across 212 modules**, and every side-effecting node lacks one.

---

## Part 3 — Performance & VRAM

### P1 · Critical — Autograd graphs retained on inference paths

`nodes/hdr/uplift_universal.py:362` — `RadianceSDRToHDRRecover.recover` has no gradient guard, while its sibling `convert` (`:474`) is decorated `@torch.no_grad()`. The unguarded operations are `vae.encode` (`:198`) and the temporal model forward.

`temporal_rudra.py:214-226` — a per-frame forward loop with **zero** `no_grad`/`inference_mode` in the entire file, accumulating three lists of graph-attached tensors. A 48-frame clip retains 48 full activation graphs simultaneously → linear VRAM growth to OOM.

**Important correction to the first pass.** The initial audit argued this was mostly a false alarm because "ComfyUI IMAGE tensors arrive with `requires_grad=False`". **That reasoning is wrong and is retracted.** A forward pass builds a graph if the input **or any parameter** requires grad. `requires_grad_` appears nowhere in product code (only `scripts/training/train_hdr_lora.py`); loaders call `.eval()` (16 sites, e.g. `temporal_rudra.py:182`) which changes BatchNorm/Dropout but leaves `requires_grad=True` on every parameter.

Recounted properly: **40 NN-forward call sites, 13 lexically guarded, ≥13 genuinely unguarded** —

`temporal_rudra.py:223` · `nodes/hdr/uplift_universal.py:198` · `nodes/hdr/encoder.py:188, 460` · `nodes/generate/regional.py:407` · `nodes/generate/prompt.py:1537, 1539` · `nodes/video/t2v.py:233, 247, 846, 1129` · `nodes_sampler.py:739` · `nodes/upscale/upscale.py:1074` (the external-`UPSCALE_MODEL` closure — notably the only tier closure without a guard; compare `:652`, `:1130`, `:832`)

**This is the single highest-value performance fix in the report.**

### P2 · Critical — Twelve module-level model caches, twelve with no eviction

| File:line | Cache |
|---|---|
| `nodes/upscale/upscale.py:606` | `_MODEL_CACHE` |
| `nodes/upscale/upscale.py:676` | `_SPANDREL_CACHE` |
| `nodes/upscale/upscale.py:769` | `_DIFFUSION_PIPE_CACHE` |
| `nodes/upscale/upscale.py:2184` | `_FACE_MODEL_CACHE` |
| `image/upscale.py:25` | `_MODEL_CACHE` |
| `nodes/vfx/depth.py:26, 27` | `_model_cache`, `_processor_cache` |
| `nodes/vfx/multipass/core.py:509, 627` | `_DSINE_HUB_CACHE`, `_DA_PIPELINE_CACHE` |
| `fast_vae.py:52` | `_TRAINED_DECODER_CACHE` |
| `pixel_sdr2hdr.py:22` | `_MODEL_CACHE` |
| `temporal_rudra.py:19` | `_TEMPORAL_MODEL_CACHE` |

`grep -n "\.pop(\|\.clear()\|^\s*del "` over `nodes/upscale/upscale.py`: **zero matches of any kind**. The only `.clear()` on any of these lives in `tests/test_rudra_compatibility.py:179`. A graph touching tiers 1–3 plus face restore pins ~3 GB permanently, and ComfyUI's `model_management` cannot evict them because it doesn't own them.

**The `unload_model` toggle frees nothing.** `image/upscale.py:2221` inserts the module into `_MODEL_CACHE`; `:2418` does `self.model = self.model.to(device)`, which mutates the cached object in place; `:2405-2406` and `:2524-2525` set `self.model = None` and call `empty_cache()` while the dict still holds a strong reference.

**The repo already has the fix.** `model/cache.py` implements a correct LRU (max 2), used by `nodes_loader.py:50` for UNet/CLIP/VAE. Routing these twelve through it — or wrapping them in `comfy.model_patcher.ModelPatcher` — eliminates most cross-execution VRAM growth. (`sampler_utils.py:1146 _freq_grid_cache` is the sole global that *does* evict, at `:1163`.)

### P3 · Critical — Allocation blowups

| Site | Measured |
|---|---|
| `film/camera.py:535-551` — motion blur `expand`→`reshape` (forces a real copy) materialises `batch×samples` images *plus* an equally large `affine_grid` | **4.24 GB** at 4K/samples=16; **17.0 GB** at samples=64 (slider max, `:449`). The `except RuntimeError` handler then reruns the same 17 GB on CPU |
| `nodes/upscale/upscale.py:1996-2004` — a full-output-resolution **constant-valued** mask through a 3-level Laplacian pyramid, per seam frame | 531 MB per mask at 4K/4×, ~3.5 GB transient. Since the pyramid is linear and the mask is a scalar, this is *exactly* `warped*(1−α) + curr*α` |
| `nodes/monitor/viewer.py:230, 249` — `image * 0.25` and `image * 4.0` multiply the **entire batch** inside a per-frame loop that only reads `image[frame_idx]` | 2·B² frame-copies. A 48-frame 1080p fp32 batch churns ~114 GB |
| `hdr/vae.py:2361` — `self._scene_linear_for_rhdr = img.detach().clone()` guarded only by `hdr_mode`, cleared only inside `if export_rhdr:` (`:3088-3118`) | ~100 MB VRAM pinned per decode node for the server lifetime. Worse at `:2729-2732`, where a *list* of per-frame clones accumulates with no `export_rhdr` guard at all: 81 frames × 99.5 MB ≈ **8 GB** |
| `nodes/monitor/viewer.py:313` + `cache.py:10` — module-global cache bounded by **count (8)**, not bytes, holding entire un-cloned input tensors on whatever device they arrived on | 8 viewers × a 100-frame 1080p batch ≈ **20 GB** held after execution |

### P4 · Critical — `exposure_bracketing` is hard-wired on and cannot be disabled

`nodes/monitor/viewer.py:144` declares `exposure_bracketing: bool = True` as a **Python default absent from `INPUT_TYPES`** (`:102-120` declares only `image`, `compare_image`, `zdepth` + hidden). ComfyUI never passes it, so it is unconditionally on with no widget to turn it off. (`bit_depth` at `:140` is undeclared the same way.)

Each `_process_frame` does `zlib.compress(level=6)` on a full-res fp32 RGBA payload (4K = 141 MB raw, 2–4 s single-threaded) plus an EXR, PNG and rpick write — **tripled**. ~10 s and ~400 MB of temp files per 4K frame, with fresh `uuid4()` names, never unlinked. A 200-frame 2K sequence writes 2400 files.

### P5 · Critical — A per-pixel Python dither loop

`nodes/color/colorspace.py:439-462` — a genuine `for y in range(H): for x in range(W)` Floyd–Steinberg, nested inside `for b: for c:`. A single 4K RGB frame is 24.9 M Python iterations (~60–150 s).

**Severity downgraded by verification:** the owning class `RadianceBitDepthDegrade` (`:410`) is **not registered** — it appears in no active `NODE_CLASS_MAPPINGS`, only in the unloaded shim `nodes_colorscience.py:25`. Not user-reachable in a default install, but a landmine the moment someone wires it up. (`image/upscale.py:1657-1688` is *row*-vectorised — an O(H) Python loop remains — and its own comment concedes it is an **approximation** of FS, not a drop-in replacement.)

### P6 · High — Work repeated per tile / per frame that has ≤9 distinct variants

| Site | Cost |
|---|---|
| `nodes/upscale/upscale.py:507` — Gaussian weight map + confidence meshgrid rebuilt every tile, byte-identical for every interior tile | 4K/tile 512/overlap 128/4× → 60 tiles × ~118 MB ≈ **7 GB of allocator churn per image** |
| `image/upscale.py:2482` — feather mask rebuilt inside the nested tile loop | 40 tiles × 16.8 MB outer product ≈ **670 MB and 160 redundant launches per frame** |
| `hdr/vae.py:2016` — cosine blend mask rebuilt per tile (at most 9 distinct masks exist) | ~180 ms on an 8K decode |
| `nodes/vfx/plate.py:181` — sampling meshgrid rebuilt per frame when only two scalars change | ~12 GB of allocator traffic over 121 frames |
| `hdr/ocio.py:413, 446` — OCIO config **re-parsed from disk** and the CPU processor recompiled on every execution, no memoisation of any kind | 0.5–3 s per execution for the ACES 2.0 studio config, paid once per frame in video workflows |

### P7 · High — GPU→CPU round trips that buy nothing

`hdr/vae.py:1963, 1966` — `.float().cpu()` per tile, accumulators pinned `device="cpu"` (`:2001-2006`), full-res CPU multiply-accumulate blend, then `return output.to(device)` at `:2040`. The whole image goes back to the GPU anyway. A 4K decode (~15 tiles) pays ~0.5–1 s of pure overhead; 8K costs 4–8 s. On GPU the blend is <1 ms/tile.

`hdr/color.py:745` — `ColorSpaceConvert` declares a `use_gpu` widget (`:556`, default True), accepts it (`:727`), and the identifier appears in the method body **exactly once — the signature line itself**. The whole transform is CPU numpy (`:745-797`), and `_get_gpu_adaptation_matrix` (`:684`) has zero references repo-wide. A 24-frame 4K batch is a 30–70 s node that should take under a second.

`color/pipeline.py:66, 96` — GPU→CPU→numpy→GPU round trip for log colorspaces, despite `radiance.color.transfer` already exporting `tensor_logc3_to_linear`, `tensor_slog3_to_linear`, `tensor_acescct_to_linear` — which this module imports *around*.

### P8 · High — LUT application allocates ~800 MB per 4K frame and syncs 5×

`color/lut.py:295` (`tetrahedral_interpolate`), gathers at `:312-319` — eight full-image advanced-index gathers, each allocating a `(B,H,W,3)` tensor. Then exactly **5 `if mask.any():` device syncs**, each guarding a `torch.where` whose branches are Python arguments and therefore fully evaluated regardless. Peak past 2–3 GB for a 4-frame batch.

**Fix:** `F.grid_sample` on the LUT as a 3D volume — one fused kernel, no syncs, no temporaries.

Also `color/lut.py:440` — an unconditional full-image `.clone()` into `img_proc`, which is only ever *read*. `RadianceLUTBlend` calls `apply_lut` twice, doubling it.

### P9 · Medium — selected

- **Boolean-mask soft-clip** (`hdr/vae.py:453, 1209, 2203`, `fast_vae.py:301`): `above.any()` sync + `img.clone()` + `img[above]` (which runs `nonzero()`, another sync). ~350–500 MB transient at 4K vs ~200 MB for branchless `torch.where`. The `fast_vae.py` one is on the realtime/NDI per-frame path.
- **`_gaussian_blur_gpu`** (`image/upscale.py:771`) — docstring says "separable", code applies a rank-1 kernel as a **dense 2-D conv**. At `sharpen_radius=5.0` (UI max, ks=31) that's **15.5× slower**: 383 GMAC instead of 24.7.
- **Shaped bokeh** (`film/camera.py:360`) — dense non-separable conv up to 101×101 over the *full* image at 5 blur levels ≈ 1.1×10¹² FLOPs on a 4K frame. The Circle path correctly uses a separable blur with a 31-tap cap.
- **List-accumulate-then-`stack`** — 5 sites doubling peak memory at the join: `hdr/vae.py:2956` (GPU), `pixel_sdr2hdr.py:382` (GPU), `image/upscale.py:2513` (**96 GB host peak** on a 121-frame 4K sequence), `hdr/ocio.py:452` (4.8 GB host), `temporal_rudra.py:229`.
- **Non-in-place normalisation** at the tiling exit (`image/upscale.py:2509`, `nodes/upscale/upscale.py:528-530`) — ~4.2 GB peak at 4K/4× at the exact moment model weights are also resident. `weight.add_(1e-8); output.div_(weight)` fixes it.
- **7 GPU syncs per frame** in `nodes/vfx/plate.py:136-162` (`.nonzero()[0]` plus 6 `.item()`); ~850 stalls on a 121-frame plate. One `argmax` replaces all of it.
- **Grain generated on CPU even on CUDA** (`film/grain.py:670`) — the MPS-compatibility comment doesn't apply to the CUDA branch. 25 M Gaussians single-threaded (~150–250 ms) + a 106 MB H2D copy per 4K frame → ~20–25 s over a 100-frame batch.
- **float64 promotion** — only 3 live sites (`color/luts.py:105, 288`, `film/camera.py:900`); `hdr/utils.py:42` enforces float32 elsewhere. `_lut_filmic` and `_lut_lin_to_log` return float64 for float32 input, doubling a 4K frame to 200 MB on the viewer/delivery path.
- **`radiance_ocio.py:756`** — a per-pixel Python loop with `.tolist()`: 8.29 M iterations per 4K frame (~60–200 s vs ~30 ms for the packed-buffer call). No in-repo caller today, but it is a documented public export.
- **`radiance_ocio.py:501, 514`** — LUT bake runs two nested Python loops (33³ or 65³ × 2) **synchronously inside an `async def`**, so the whole ComfyUI web server freezes for 0.5–4 s on every display/view change from the viewer.
- **`nodes/upscale/upscale.py:2489`** — a dead per-pixel loop with a GPU sync per pixel (~25 M synchronising round-trips at 4K, hours per image). Verified **unreachable** — only `_histogram_match_fast` below it is used. Delete lines 2436-2498.

**Correct patterns worth noting:** all 16 `torch.cuda.empty_cache()` calls sit in OOM-fallback branches or explicit unloads — none in a hot loop. The three `threading.Thread` sites are one-time daemon server starts guarded by module locks. `color/matrices.py` and `color/ops.py` keep matrices as module constants. `image/upscale.py:644` correctly hoists its blend weight out of the tile loop — the code at `:2482` just didn't follow suit.

---

## Part 4 — Architecture, packaging & tests

### A1 · The "dual layout" is **not** duplication — this is the good news

The most important structural question, and the evidence clears it. Recounted by AST:

| Comparison | Shared class names |
|---|---:|
| root `nodes_*.py` (18 classes) vs `nodes/` (120 classes) | **0** |
| `color/` vs `nodes/color/`, `hdr/` vs `nodes/hdr/`, `io/`, `image/`, `film/`, `model/`, `delivery/`, `gpu/`, `core/`, `config/` | **0 each** |
| All 205 product classes, any two files | **3** |

Of 46 root `nodes_*.py`: **40 are pure deprecation shims** (≤35 lines, no classes, explicit deprecation marker; all 42 re-export targets verified to exist) and **6 are real un-migrated modules** — `nodes_io.py` (2103 lines), `nodes_workspace.py` (1859), `nodes_sampler.py` (1697), `nodes_realtime_preview.py` (1031), `nodes_loader.py` (930), `nodes_gizmo.py` (398).

The real structure is a clean two-layer split: `nodes/` is the ComfyUI adapter layer, top-level `hdr/ color/ image/ film/ core/ model/` is the engine layer, and `nodes/` imports *from* the root modules. KNOWN_ISSUES.md's "~39 duplicate node keys" describes shim key re-declarations, not duplicated logic.

**Duplicated code: 3 classes, ~250 lines, 0.35% of the codebase.** The bug factory is a *third* axis — engine packages that still declare `NODE_CLASS_MAPPINGS` from a pre-migration era but are not catalog groups.

### A2 · High — 54 of 154 declared nodes never register

Statically merging all eleven group `__init__.py` files (all use static dict literals; `nodes/vfx/__init__.py:54` spreads in `**MP_MAPPINGS`):

- **Declared** in `NODE_CLASS_MAPPINGS` dicts anywhere: **154 unique** (253 with duplicates, across 70 files)
- **Actually registered: 100** — independently corroborated by `tests/node_keys_snapshot.json`
- **Never registered: 54**

Two causes. Four leaf modules under `nodes/` are never imported by their group `__init__.py` — `nodes/hdr/inception.py`, `nodes/pipeline/audio.py` (719 lines), `nodes/pipeline/metadata.py`, `nodes/vfx/camera.py`. And the dominant one: engine packages `hdr/` (17 keys), `image/` (11), `film/` (10), `color/` (4) declare mappings but aren't members of `NODE_GROUPS` in `nodes/catalog.py`, so nothing reads them.

**No real collisions** — 93 keys are declared in more than one module, but all 93 resolve to the same class object (AST comparison: 0 divergent). But **60 keys carry two different display strings** (group `__init__.py` vs leaf module); the `__init__.py` value wins, so the leaf strings are dead — which is how the README came to document names users never see.

This is the root cause of the 9 documented-but-missing features in §A8.

### A3 · High — Without torch, 12 of 100 nodes load and startup reports success

Reproduced live. `python3 -c "import radiance"` with torch absent:

```
Radiance node catalog: skipping optional module radiance.nodes.color (No module named 'torch')
... (9 more groups) ...
[Radiance] [INFO] Radiance: successfully loaded 12 nodes (v3.1.2)
```

Ten of eleven groups drop. Group `__init__.py` files import leaf modules eagerly, `nodes/registry.py:133-138` swallows the failure and logs at WARNING (`:226`), and `__init__.py` reports `len(NODE_CLASS_MAPPINGS)` with no comparison against expected. The `NodeLoadResult.failures` tuple is collected and never consulted. **An 88% shortfall is reported as success.**

Confirmed unguarded module-scope optionals: `nodes/color/cdl.py:6` (`defusedxml` → all 13 colour nodes) and `hdr/panorama.py:5` (`cv2` → hdr + generate groups). Also unguarded: `PIL` (5 sites), `aiohttp` (2), `tqdm` (4), `requests` (2).

**Genuinely well-designed, and worth preserving:** the 12-module fatal closure from `__init__.py` imports zero third-party packages, so the pack never hard-bricks ComfyUI.

**Fix:** compare loaded count against expected and log an ERROR on shortfall; guard the two module-scope optionals.

### A4 · High — `RadianceHighlightSynthesis` is forked, and the fixed copy is the dead one

`recovery.py:18-220` (203 lines) vs `hdr/recovery.py:18-222` (205 lines). Behavioural divergence:

```python
h_small = max(1, int(h / detail_scale))    # recovery.py:166
h_small = int(h / detail_scale)            # hdr/recovery.py:165
```

```python
# recovery.py — Screen blend on [0,1] operands
noise_shifted = np.clip((noise_layer + 1.0) * 0.5, 0.0, 1.0)
final_frame[..., ch] = 1.0 - (1.0 - a) * (1.0 - noise_shifted * mask)

# hdr/recovery.py — same formula fed raw [-1,1] noise
final_frame[..., ch] = final_frame[..., ch] + noise_layer - (final_frame[..., ch] * noise_layer)
```

Soft Light diverges too. `recovery.py` carries a `v3.2 Fixes:` docstring block (`:28-38`) enumerating exactly these three bugs — strong evidence the fix was applied to the wrong file.

The **only** import of either module anywhere is `hdr/__init__.py:23: from .recovery import RadianceHighlightSynthesis` — the buggy copy. Root `recovery.py` is imported by nothing and sits at 0% coverage.

*(Two first-pass corrections: the line counts were reported as 65/64, off by ~3×. And "the buggy copy ships" is wrong — `radiance.hdr` is not a catalog group, so **neither copy registers**. The class is imported and instantiable but never reaches ComfyUI's registry. The `int(h/detail_scale)` zero-size failure also needs an image under 5 px, since `detail_scale` caps at 5.0 — real but degenerate.)*

`RadianceFilmGrain` (`film/grain.py:794` vs `nodes/vfx/optics.py:426`) and `RadianceMotionBlur` (`film/camera.py:408` vs `nodes/vfx/motion_blur.py:7`) are unrelated implementations colliding on a name; only the `nodes/vfx` versions register.

### A5 · Critical — JS patches shared prototypes, breaking other extensions

`js/radiance_io.js:52-59` replaces `HTMLInputElement.prototype.click` globally and unconditionally at module scope, affecting every file input on the page including other packs'. (`:36` also redefines the `accept` property descriptor on the same prototype.)

`js/radiance_io.js:167-179` replaces `Element.prototype.setAttribute` — one of the hottest DOM methods — routing every `setAttribute` call in ComfyUI and every other extension through Radiance's `resolvePlaceholder()`.

`js/radiance_backdrop.js:311` replaces `LGraphGroup.prototype.draw` **without saving the original**. Grepping the file: the only occurrences of that string are the comment at `:309` and the assignment at `:311`. Any other extension's group rendering is silently destroyed. The *same file* does it correctly elsewhere — `getGroupAt` saves `origGetGroupAt` at `:416`, `onDoubleClick` saves `origOnDoubleClick` at `:438` — so this reads as an oversight, not a design choice.

`js/radiance_style.js:159-177` also overwrites 9 `LiteGraph.NODE_*` colour constants globally, restyling every pack's nodes, with `!important` on every property and no opt-out.

**By contrast, Python-side monkeypatching of `comfy.*` is genuinely zero** — searching for `comfy.<mod>.<attr> =`, `setattr(comfy`, `folder_paths.<attr> =`, `sys.modules[...] =` and the literal `monkeypatch` returns nothing outside `tests/`. The asymmetry is stark: the Python side is disciplined, the JS side is not.

### A6 · High — `postMessage` handler with no origin check

`js/radiance_workspace.js:15` handles **three** message types with no `event.origin` validation:

- `radiance_load_workflow` (`:16`) → `app.loadGraphData(graphData, false)` at `:19`
- `radiance_append_workflow` (`:24`) → `loadGraphData(..., true)` at `:27`
- `radiance_save_project_version` (`:32`) → serialises the user's graph and replies via `event.source?.postMessage(..., "*")` at `:66` and `:70` — a **wildcard target origin to an unvalidated sender**

The sibling at `js/project_manager_dashboard.mjs:648-649` does it correctly. One-line fix.

### A7 · High — CI has been red since 2026-07-10

`.github/workflows/ci.yml:102-108` hardcodes an `--ignore` list of torch-importing test files. It is stale — two newer test files were never added.

Ran the exact CI command: **`13 failed, 969 passed, 497 skipped`**, all 13 in `tests/test_multipass_contracts.py` (7) and `tests/test_temporal_rudra.py` (6). Both exist and both need real torch. `ci.yml` was last touched 2026-07-10 13:00; `test_temporal_rudra.py` landed the same day at 20:14 — so red since **2026-07-10**.

The more interesting failure: `tests/test_temporal_rudra.py:6` uses `pytest.importorskip("torch")`, which *should* skip cleanly — but `tests/conftest.py:60` installs a MagicMock torch stub into `sys.modules`, so `importorskip` succeeds and hands the test a mock. **The project's own conftest defeats its own skip guard.**

Three ignore-list entries (`test_io_functional.py`, `test_io_hdr_regression.py`, `test_radiance_load_image_mask.py`) have no unconditional torch import and appear unnecessary.

Also: the primary matrix (`ci.yml:56`, py3.9–3.12 × ubuntu/windows) **installs no torch**, so 497 of 1546 tests (32%) silently skip and the rest run against MagicMock tensors. The only real-torch lane (`test-full`, `:144`) is fenced by `if: github.repository == 'fxtdstudios/radiance'` (`:146`) — forks get zero tensor coverage. `.pylintrc` exists but nothing runs pylint. Ruff is advisory (`continue-on-error: true`, `:386`). Security scanning is non-blocking twice over (`:420-429`). No type checking.

### A8 · Tests

| Metric | Measured |
|---|---|
| Test files | 59 `.py` under `tests/` (57 match `test_*.py`) |
| Lines | **16,349** |
| Test functions | **1,061** → 1,546 collected |
| Assertions | 1,519 → **1.43 per test** (healthy is 2–4) |
| Test:source ratio | 0.25 : 1 |
| Coverage | **19.83%** statement, branch enabled |
| CI gate | `--cov-fail-under=15` (`ci.yml:113`) — overriding the project's own `fail_under = 40` (`pyproject.toml:194`) |
| Modules at 0% | **28** |

**Quality is bimodal.** The colour-science tests are genuinely excellent — `tests/test_color_matrices.py` checks 6 invariants × 7 log curves against published spec values, with a docstring recording that `AWG4_TO_ACESCG` was wrong for months (`:6-8`); `tests/test_color_math.py:196-283` ties named regressions to specific bugs. Everything else is smoke: **23 of 57 files contain zero tolerance-checked assertions**, and only 23.4% of behavioural assertions carry an explicit tolerance.

**The most damning single line**, `tests/test_node_smoke.py:992-1004`:

```python
def test_all_keys_have_class(self):
    """Every key discovered by AST must resolve to an actual class."""
    all_keys = set(_discover_node_keys_from_source().keys())
    missing = all_keys - set(_ALL_NODES.keys())
    # Nodes that failed to import are acceptable during a CI run without
    # torch/GPU — record them but do not fail the suite.
    if missing:
        import warnings
        warnings.warn(...)
```

No assertion. The one test whose job is to catch broken registration **cannot fail** — and it is firing right now: my run emitted `UserWarning: 16 nodes could not be imported`. Likewise `tests/test_node_keys_snapshot.py:53-59` writes its own golden file on first run and `pytest.skip`s.

**`tests/conftest.py` mocks production modules, not just external deps** — `:187-190` replaces `radiance.radiance_ocio` (769 lines) with a `MagicMock`; `:200-216` replaces `radiance.image.defects` with canned returns. Both therefore measure 0% coverage while appearing test-referenced.

**Top 10 highest-risk untested modules** (size × complexity):

| # | Module | Lines | Branches | Coverage | Test refs |
|---:|---|---:|---:|---:|---|
| 1 | `hdr/vae.py` | 3,320 | 234 | 10% | indirect |
| 2 | `image/upscale.py` | 2,692 | 223 | **0%** | **none** |
| 3 | `nodes_workspace.py` | 1,860 | 290 | 12% | **zero** |
| 4 | `nodes_io.py` | 2,104 | 266 | 28% | 5 files |
| 5 | `sampler_utils.py` | 1,722 | 185 | 15% | 5 files |
| 6 | `nodes/video/t2v.py` | 1,509 | 184 | 15% | **zero** |
| 7 | `nodes_sampler.py` | 1,698 | 172 | **6%** — lines 828-1686 never execute | 3 files |
| 8 | `hdr/color.py` | 1,464 | 78 | 28% | indirect |
| 9 | `core/logging.py` | 1,004 | 78 | 34% | indirect |
| 10 | `radiance_ocio.py` | 769 | 99 | **0%** — mocked out | 4 files, all stubbed |

Runner-up: `loader_utils.py` (645 lines, 8%, zero references).

Note how well this table predicts Part 2: the Critical correctness findings cluster in exactly these modules.

### A9 · Packaging

**Good:** version 3.1.2 consistent across `pyproject.toml:8`, `package.json:3`, `config/constants.py:5`, `README.md:7` (only `package-lock.json:3` lags at 3.1.0). **Zero `==` pins.** `torch`/`torchvision`/`torchaudio` appear in **no** requirements file — `pyproject.toml:43` documents this and `config/dependencies.py:36` validates at runtime, so the classic "pinned torch fights ComfyUI" failure mode does not exist. Licence is plain GPL-3.0, correctly declared, compatible with ComfyUI and every dependency. `js/` ships correctly (all 34 files, confirmed in `SOURCES.txt`), and `ci.yml:320-348` diffs the built wheel against the tree. The `package-dir` layout (`pyproject.toml:106`) picks up the flat root modules automatically.

**Problems:**

- **Three mutually inconsistent dependency specs.** `requirements.txt` (16 packages) dropped the `<X.0.0` caps that the three platform files carry — **13 of 16 differ**. `pyproject.toml` is a third variant, omitting the caps on `tqdm` and `aiohttp`. So `pip install radiance` and `pip install -r requirements_linux.txt` resolve differently.
- **`OpenImageIO` is missing from all three platform files** (present in `requirements.txt:7` and `pyproject.toml`). It's genuinely required — `nodes_io.py:64` imports it, `:390` raises `ImportError("Reading DPX requires OpenImageIO")`. README sends every user to the platform files (`:53, 73, 78, 90`), so **anyone following the documented install has silently broken DPX I/O**.
- **Three hard deps are never imported at all.** Exhaustive grep including tests: `colour-science` (import name `colour`) — zero matches for `import colour`/`from colour`; the only occurrence anywhere is a *data row* in `config/dependencies.py:45`'s probe table. `einops` — zero matches. `torchsde` — zero matches. All three are unconditional installs in all five manifests. `colour-science` is additionally a **hard** dep in pyproject while the runtime guard reports it Optional. Plus 4 dead entries in the `[full]` extra (`accelerate`, `peft`, `anthropic`, `google-generativeai`).
- **23 third-party modules imported but undeclared** (18 in runtime node code — `tifffile`, `safetensors`, `librosa`, `soundfile`, `whisper`, `basicsr`, `gfpgan`, `NDIlib` …). All guarded, so this is a documentation gap rather than a crash.
- **`workflows/` ships only 2 files** — `.gitignore:112-113` ignores `workflows/*.rad` except `workflows/official/**`, so the template library is effectively empty for clean clones.
- **Six PNGs declared at `pyproject.toml:140-145`** (including the registry `Icon` at `:102`) are **absent from the working tree**. Verify on a clean clone or the registry listing 404s.
- **`delivery/handler.py:116`** registers an aiohttp route at **module scope** with **zero** idempotency guards, while all three siblings have them (`nodes/monitor/viewer.py:1017-1020`, `nodes_gizmo.py:35-38`, `radiance_ocio.py:575-577`). Any reload raises a duplicate-route error; `:11-12` imports `aiohttp`/`server` unguarded, killing the whole monitor group in an aiohttp-less environment.

### A10 · Repo hygiene — files that should not be committed

| Path | Size | In `.gitignore`? | In `.comfyignore`? |
|---|---|---|---|
| `.codex-backup-20260719/` | 56 K | **No** | **No** — ships to the registry |
| `bug_report_reply_draft.md` | 4.2 K | **No** | **No** — an unsent email to a **named external user**, disclosing unreleased defects (silent preset override, Flux.2 crash) that appear in no changelog |
| `PACKAGE_REVIEW.md` | 2.9 K | **No** | **No** — internal scorecard publishing "19% coverage", "no dependency scanning" |
| `.agents/` | empty | **No** | **No** |

`.comfyignore` already excludes `PRE_RELEASE_REVIEW.md`, `CLEANUP_REPORT.md`, `RELEASE_NOTES_*.md` under a comment about dev artifacts — these four were simply missed. `bug_report_reply_draft.md` is the one to remove first.

**Dead code:** 94 of 212 modules (16,119 lines, 23%) are unreachable from `__init__.py`. `scripts/` (6,659 lines) is legitimate. Genuinely dead: `film/` (2,647 lines — not a catalog group, its 3 nodes never register) and the 4 orphan `nodes/` leaves (2,585 lines). Four `tools/` scripts have **broken imports** — `tools/build_rudra_cache.py:13`, `validate_rudra_dataset.py:15`, `compute_descriptor_stats.py:16` import a `rudra` package that doesn't exist; `tools/validate_hdr_pipeline.py:54` imports non-existent `radiance_color`.

`.gitignore` (120 lines) covers all standard build/venv/cache dirs correctly. One latent bug: `:95-106` is a self-cancelling `docs/` / `!docs/` negation block whose six following rules are dead.

### A11 · JS frontend — other findings

29 files / 43,237 lines. **`js/radiance_viewer.js` alone is 19,880 lines — 46% of the frontend**; the top 3 files are 70%.

- **66 undocumented-internal call sites** vs 23 safe `registerExtension` — 39 `widget.computeSize = () => [0,-4]` (a no-op under Vue Nodes 2.0), 12 `app.graph._nodes`, 7 direct `widgets_values` index mutation, 3 `app.graph._groups`. Highest risk: `radiance_viewer.js:18423-18481` scrapes Vue frontend CSS classes (`.lg-node-widget`, `.truncate`) with a retry storm at `[0,100,500,1000,2500,5000]` ms plus a `document.body` MutationObserver with `subtree:true`.
- **Fetch:** 38 call sites, **7 fully unguarded** (`radiance_viewer.js:8431, 18702`; `radiance_mask_editor.js:1231`; `radiance_workspace.js:916, 953, 994, 1321`). Only 2 sites anywhere handle a non-JSON error body.
- **Listener leaks:** 180 `addEventListener` vs 44 `removeEventListener`. **Credit where due — all 8 per-node `setInterval` polls are correctly cleared in a chained `onRemoved`**, which is the classic ComfyUI leak and this pack avoids it. Confirmed leaks: 6 anonymous global listeners in `radiance_viewer.js` (`:2450, 5437, 5450, 19270, 19340, 19354`) that `destroy()` cannot remove, and 3 self-rescheduling rAF loops (`_grainRAF`, `_seqRAF`, `_referenceScopeRAF`) — `destroy()` contains **zero** `cancelAnimationFrame` calls.
- **Duplication:** the widget-visibility toolkit (`setWidgetVisible`, `refreshNodeSize`, `_forceWidgetReinsert`) is copy-pasted **6×** across `radiance_loader/sampler/resolution/vae_widgets/io/upscale.js`, **and the copies have already diverged** (differing MD5 per body). `escapeHtml` has 4 divergent copies — same problem, with security consequences (see the one unescaped `innerHTML` in §S8).

### A12 · Docs accuracy

**Node count: five different numbers ship.** README says **96** (`:9, 95, 202`); `PACKAGE_REVIEW.md:9` says 98; `docs/index.html:266` says 103; `docs/faq.html:185` says 78 (a v2.2.1-era console transcript). **Reality is 100.**

`README.md:95` is actively harmful — it tells users to verify their install by looking for `successfully loaded 96 nodes`, but `__init__.py:56` prints 100. The 4-node gap traces exactly to `docs/coverage.md` omitting `RadianceHDRToneMap`, `RadianceHDRExpandDynamicRange`, `RadianceSDRToHDRRecover`, `RadianceSDRToHDRUniversal` — ironically the very nodes CHANGELOG 3.1.2 exists to restore.

**Documented but missing: 9 of 24 spot-checked.** `Roto` (`README.md:205`) **does not exist anywhere**. `LUTs` (`:213`), `Defocus` (`:205`), `flipbook` (`:220`), `preview server` (`:220`), `Workspace` (`:211`), `video prompt builder` (`:218`) all exist as classes but are among the 54 never registered. `scopes` (`:220`) — `nodes/monitor/scopes.py:17` is literally `NODE_CLASS_MAPPINGS = {}`.

**Undocumented:** the entire `/radiance/*` HTTP surface (12 route families), `nodes_gizmo.py`'s server route, and the `RADIANCE_DEV` training group.

**KNOWN_ISSUES.md is unusually honest** — 6 of 7 claims verified. The one inaccuracy understates. **CHANGELOG** is clean: latest entry matches `constants.py` exactly, 22 headers, no phantom versions, no broken links (gaps: no 2.0.0 entry, and 2.3.0/2.3.1/2.4.0/2.4.1 missing).

---

## Part 5 — Static analysis summary

`ruff check --select ALL` (excluding docstring/annotation/TODO families) over the tree. Top signal:

| Rule | Count | Comment |
|---|---:|---|
| `BLE001` blind-except | **353** | The systemic pattern in §C17 |
| `PLC0415` import-outside-top-level | 386 | Mostly deliberate optional-dep guarding |
| `S101` assert | 613 | Concentrated in tests |
| `F401` unused-import | 573 | Auto-fixable |
| `C901` complex-structure | 132 | |
| `PLR0912/0915` too-many-branches/statements | 84 / 72 | |
| `PLW0603` global-statement | 17 | Matches the 7 unlocked singletons in §C20 |
| `S110` try-except-pass | 36 | |
| `NPY002` numpy-legacy-random | 30 | Legacy `np.random.*` — relevant to the determinism findings |
| `RUF012` mutable-class-default | 86 | |
| `S324` insecure-hash | 13 | MD5 cache keys, non-security |
| `S603` subprocess-without-shell | 13 | All argv-list, verified safe |
| `E501` line-too-long | 2,634 | Cosmetic |

**bandit** (`-ll`, excluding tests/scripts): 11 High, 19 Medium. The only substantive High/Medium hits are already covered above — `pixel_sdr2hdr.py:299` (B614 unsafe torch load), `radiance_ocio.py:88` (B310 urlopen), and two B324 MD5 cache keys.

`python -m compileall`: **clean**, no syntax errors anywhere.

---

## Recommended order of work

**Before the next release**

1. `nodes/upscale/upscale.py:632-634` — fix the Real-ESRGAN key mapping, set `strict=True`. *The default upscaler currently emits noise.*
2. `nodes/upscale/upscale.py:1943` — `if wi > 0:` guard. *Frame 0 of every video render is black, ungated.*
3. `nodes_io.py:551` — `if missing_frames == "Black"`; default `end_frame` to the highest existing frame. *Sequence reads are broken at defaults.*
4. `delivery/handler.py:441` — fix the call signature; have `write()` return the path. *No delivery has ever written a file.*
5. `pixel_sdr2hdr.py:299` — `weights_only=True`. Same for `train_turbo_decoder.py:356, 614`.
6. `nodes/pipeline/dcc.py` and `scripts/start_nuke_server.py` — delete the eval/exec fallbacks; keep only structured-action dispatch. Make the Nuke auth token mandatory.
7. `js/radiance_io.js:52, 167` and `js/radiance_backdrop.js:311` — remove the prototype patches; restore delegation.
8. `js/radiance_workspace.js:15` — add `if (event.origin !== window.location.origin) return;`.
9. `nodes/upscale/upscale.py:1318, 1631, 1883` — second cascade pass is `scale=2`.
10. Delete `bug_report_reply_draft.md`, `.codex-backup-20260719/`, `PACKAGE_REVIEW.md`, `.agents/`; add all four to both ignore files.

**Next**

11. Add `@torch.no_grad()` at the 13 unguarded inference sites (§P1) — start with `uplift_universal.py:362` and `temporal_rudra.py:223`.
12. Route the 12 model caches through the existing `model/cache.py` LRU or `ModelPatcher` (§P2); fix `unload_model` to actually pop.
13. Adopt `color/luts.py`'s DaVinci Intermediate everywhere; delete the two broken copies. Fix S-Log3 and Canon Log 3 the same way. **Add the `cut ± ε` continuity test** — two lines, catches all five.
14. Fix the DWG/AWG4 third rows. **Add `M @ [1,1,1] ≈ white_XYZ`** — one line, catches both.
15. Extract the border-suppression logic from `hdr/vae.py:2011` into one shared tiling helper and use it at all five sites (§C4).
16. Fix the AgX sigmoid (`hdr/tonemap.py:351`) and the `/peak_scale` midtone crush (`hdr/color.py:1307`).
17. Reconcile the 54 unregistered nodes against the README: wire them up or delete them. Strip `NODE_CLASS_MAPPINGS` from the non-catalog engine packages.
18. Resolve the `RadianceHighlightSynthesis` fork — keep `recovery.py`, delete `hdr/recovery.py`, re-point the import.
19. Refresh `ci.yml:102-108`; get CI green. Turn `test_node_smoke.py:992` into a real assertion.
20. Add `OpenImageIO` to the three platform requirements files; drop the 7 declared-but-unimported deps; reconcile the three dependency specs.
21. Guard `nodes/color/cdl.py:6` and `hdr/panorama.py:5`; add an idempotency guard to `delivery/handler.py:116`; make startup log an ERROR when the loaded count falls short.
22. Fix `README.md:9, 95, 202` (96 → 100).

**Then**

23. Audit the ~30 silent-failure handlers (§C17). Adopt a rule: a node may return a fallback **only** if it also surfaces the failure to the user, not just the log.
24. Raise the coverage floor toward `pyproject.toml`'s own 40%, starting with the top-10 table in §A8 — those modules contain most of Part 2.
25. Extract the 6× duplicated JS widget toolkit and the 4× `escapeHtml`; fix the unescaped `innerHTML` at `radiance_viewer.js:3875`.
26. Add `Origin` checks to the mutating `/radiance/*` routes; default `allow_absolute=False`.

---

## What's good

Worth stating plainly, because an audit this long distorts the picture:

- **The dual layout is a clean migration, not a mess.** Zero class overlap between the two trees; 0.35% duplicated code. Whoever did that migration did it properly.
- **The colour-science test suite is excellent** — spec-value invariants, named regression tests, a docstring recording a bug that hid for months. It is the model the rest of the suite should follow.
- **Security fundamentals are mostly right where they matter:** `_resolve_safe_path` gets resolve-before-relative_to correct (most implementations don't), the `.rad` unpacker is hardened against zip-slip *and* zip-bombs with explicit caps, secrets go through env-var-name indirection, servers bind loopback by default and actively refuse otherwise, and there is no shell injection anywhere.
- **Zero Python-side monkeypatching of ComfyUI internals** across 212 modules. That is rare and valuable.
- **The fatal import closure imports no third-party packages**, so the pack can never hard-brick a user's ComfyUI.
- **All 8 per-node `setInterval` polls are correctly cleared in `onRemoved`** — the classic ComfyUI frontend leak, avoided.
- **Every `empty_cache()` is in a fallback branch**, never a hot loop. All INT widgets carry min/max. Workflow payloads are size-capped with correct Content-Length handling.
- **KNOWN_ISSUES.md is honest** — 6 of 7 claims verified, and the one error understates the problem.

The recurring shape of every finding here is the same: **the codebase already contains the correct version of the thing it gets wrong.** Correct tiling in `hdr/vae.py`, correct transfer functions in `color/luts.py`, a correct LRU in `model/cache.py`, correct locking in `cache.py`, correct origin checking in `project_manager_dashboard.mjs`, correct prototype delegation 100 lines from the broken one. The fix is rarely invention — it is consolidation. One tiling helper, one transfer-function module, one model cache, one widget toolkit.
