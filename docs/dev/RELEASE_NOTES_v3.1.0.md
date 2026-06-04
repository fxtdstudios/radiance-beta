# Radiance v3.1.0 — Release Notes

**Focus:** correctness and trust in the 32-bit / HDR / EXR pipeline, plus real-AOV ingestion and hardened model loading. This release closes the critical data-integrity and security issues found in the pre-release engineering review and brings the full test suite green against real torch + OpenEXR.

---

## Critical fixes (data integrity)

- **HDR/EXR load no longer crushes scene-linear data.** The image-to-tensor path previously divided any array whose brightest pixel exceeded 2.0 by 255 (or 65535), silently darkening every real HDR/EXR plate ~255×. Normalization is now driven by the source format: integer formats normalize by their type max; float formats (EXR/HDR/float-TIFF) pass through untouched. *Read → Write of an EXR is now lossless.*
- **EXR writer can no longer produce a silent 0-byte or downgraded file.** The writer raises a clear error when no EXR backend is available instead of writing an empty/placeholder `.exr` and reporting success.
- **EXR writer preserves alpha and single-channel mattes.** It now writes 1→RGB (grayscale), 3→RGB, and 4→RGBA (alpha preserved), and no longer crashes on `(H,W)` matte input.
- **Write node surfaces failures.** `RadianceWrite` re-raises on write failure (turns the node red) instead of swallowing the exception and reporting a phantom success.

## Security

- **`torch.load` hardened.** All shippable checkpoint loads use `weights_only=True` (VAE, fast VAE, multipass depth/normal models, training data prep), preventing arbitrary-code-execution from malicious `.ckpt`/`.pth` files.
- **Model downloads are now atomic + integrity-checked.** The loader and multipass downloaders write to a temporary file and atomically move into place (no truncated checkpoints left at the model path), with optional SHA-256 verification and a `RADIANCE_LOADER_OFFLINE` / `RADIANCE_UPSCALE_OFFLINE` switch to disable auto-download in airgapped setups.

## New

- **◎ Multipass: AOV Reader.** Reads a real multilayer/AOV OpenEXR (Arnold, Redshift, Karma, Cycles, V-Ray) and splits its named layers into the same outputs as the Multipass Master extractor — so ground-truth render passes flow straight into the EXR-passes writer and relight/comp chain. Scene-linear values are preserved; missing layers come through black so the Master node can gap-fill.
- **Alpha output on the EXR write node.** `RadianceWrite` gained an optional `mask` input, written as the EXR alpha channel (RGBA) for EXR formats.

## Improvements

- **Upscaler HDR + color handling.** Image and video upscalers gained `hdr_mode` (`auto`/`preserve`/`clamp`, Reinhard tonemap round-trip so highlights above 1.0 survive SR) and `color_encoding` (`passthrough` / `linear↔sRGB` / `linear↔LogC3` OETF round-trip so the LDR-trained networks see display-referred input).
- **Sampler non-finite guard.** `RadianceSamplerPro` detects NaN/Inf in the sampled latent, warns, and sanitizes — so a blown CFG/precision run is visible in the log instead of shipping black frames.
- **HDR VAE Decode metadata output.** The node now actually emits its decode-settings JSON as a second `STRING` output (previously built and discarded).
- **Cleaner loading.** The package no longer imports its own deprecated shim paths; load-time `DeprecationWarning`s from internal imports are gone.

## Testing & CI

- Full `pytest tests/` passes against **real** torch + OpenEXR + OpenColorIO + colour-science: **1347 passed, 34 skipped**.
- CI now installs the real runtime dependencies and runs the whole suite (previously only two files with torch mocked); the publish gate runs the full suite. The stale CI import of a non-existent module was removed.
- Added `tests/test_io_hdr_regression.py` covering the C-1/C-2/C-3 fixes and the mask→alpha write path.

## Known limitations

- **RUDRA dynamic-range decoder is not in this release.** The HDR VAE Decode node ships the baseline decoder; the `dr_dim`-conditioned RUDRA decoder and its tests are deferred.
- **Multipass *Master* passes are estimates**, not physically-accurate render AOVs. Use the new AOV Reader for ground-truth passes. The segmentation-ID output is a clustered matte, not a spec Cryptomatte.
- **Legacy `nodes_*.py` shims remain** for backward-compatible imports (they register no nodes); removal is planned for a later release.

## Upgrade notes

- No workflow breakage expected: node type keys are unchanged, and added outputs (e.g. HDR VAE Decode `metadata`, EXR write `mask` input) are backward compatible.
- Requires the package's runtime dependencies in the ComfyUI environment (OpenEXR/Imath recommended for full-precision EXR).

## Verify before tagging

1. `pytest tests/` green in the ComfyUI venv (real deps).
2. Live ComfyUI import on Windows + Linux — confirm the startup log shows `Radiance: successfully loaded N nodes (v3.1.0)` with no tracebacks.
3. One real-plate round-trip: Read EXR → grade → Write EXR (with mask) → open in Nuke/Resolve and confirm a numerically lossless, alpha-intact result.
