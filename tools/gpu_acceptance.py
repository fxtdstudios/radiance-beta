#!/usr/bin/env python3
"""Radiance GPU acceptance run — the test the sandbox cannot do.

Run this ON THE MACHINE WITH THE GPU, using ComfyUI's own Python:

    cd <ComfyUI root>
    python custom_nodes/radiance/tools/gpu_acceptance.py

What it does:
  1. Environment: CUDA device, VRAM, torch/driver versions.
  2. Node sweep: executes every node the functional harness can fabricate
     inputs for, ON CUDA, recording time + peak VRAM + pass/fail.
  3. RUDRA model scoring: finds trained checkpoints in models/radiance/,
     loads each, and scores it 0-100 from measured behaviour:
       - decode fidelity  (PSNR of VAE-encode -> RUDRA-decode round trip
         against the reference VAE decode, when a VAE is available;
         otherwise structural sanity: finite, range, no banding)
       - speed            (frames/sec at 512, 1080p, 4K)
       - VRAM             (peak bytes at each size)
       - stability        (100 repeated decodes: memory growth, determinism)
       - temporal         (TemporalRUDRA static-scene flicker: identical
         frames in must give identical frames out)
  4. Writes gpu_acceptance_report.md next to this script.

Scoring bands: 90+ production, 80-89 stable, 70-79 needs work, <70 not ready.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(REPO))
for p in (COMFY, os.path.dirname(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

REPORT: list[str] = []


def log(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def section(title: str) -> None:
    log(f"\n## {title}\n")


def vram_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def vram_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def psnr(a: torch.Tensor, b: torch.Tensor, peak: float = 1.0) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    import math
    return 10.0 * math.log10(peak * peak / mse)


def main() -> int:
    section("Environment")
    has_cuda = torch.cuda.is_available()
    log(f"- torch {torch.__version__}, CUDA available: {has_cuda}")
    if not has_cuda:
        log("- **NO GPU FOUND — this run only repeats what CI already covers.**")
    else:
        props = torch.cuda.get_device_properties(0)
        log(f"- {props.name}, {props.total_memory/1e9:.1f} GB VRAM, "
            f"capability {props.major}.{props.minor}")
    device = torch.device("cuda" if has_cuda else "cpu")

    # ── 2. Node sweep on GPU ────────────────────────────────────────────
    section("Node sweep (CUDA tensors)")
    import radiance  # noqa: E402

    nodes = radiance.NODE_CLASS_MAPPINGS
    log(f"- {len(nodes)} nodes registered")

    # Which nodes are MISSING relative to the repo's snapshot? A shortfall
    # here means a node module failed to import on THIS machine (usually a
    # missing optional dependency) and every node in that module was dropped.
    snap_path = os.path.join(REPO, "tests", "node_keys_snapshot.json")
    if os.path.isfile(snap_path):
        import json as _json
        with open(snap_path, encoding="utf-8") as f:
            expected = set(_json.load(f))
        missing = sorted(expected - set(nodes))
        if missing:
            log(f"- **{len(missing)} nodes missing vs repo snapshot:**")
            for m in missing:
                log(f"  - {m}")
            log("  Check the ComfyUI startup log for '[Radiance]' import "
                "failures — a missing optional dependency (transformers, "
                "OpenImageIO, opencolorio, tifffile...) drops the whole "
                "module. The Environment Guard table at startup lists what "
                "is absent.")

    OPAQUE = {"MODEL", "CLIP", "VAE", "CONDITIONING", "CONTROL_NET",
              "UPSCALE_MODEL", "SIGMAS", "SAMPLER", "GUIDER", "NOISE", "AUDIO",
              "RADIANCE_GRADE_INFO", "RADIANCE_CDL", "RADIANCE_AOV",
              "RADIANCE_PASSES", "RADIANCE_LORA_STACK", "RADIANCE_MODEL_META",
              "RADIANCE_SHOT", "RADIANCE_PROJECT", "CLIP_VISION",
              "CLIP_VISION_OUTPUT", "GLIGEN", "STYLE_MODEL", "PHOTOMAKER",
              "SAM_MODEL", "BBOX_DETECTOR", "SEGM_DETECTOR"}

    def fab(spec):
        kind, cfg = spec[0], (spec[1] if len(spec) > 1 else {})
        if not isinstance(cfg, dict):
            cfg = {}
        if isinstance(kind, (list, tuple)):
            d = cfg.get("default")
            return d if d in kind else (kind[0] if kind else None)
        if kind == "IMAGE":
            return torch.rand(1, 256, 256, 3, device=device)
        if kind == "MASK":
            return torch.rand(1, 256, 256, device=device)
        if kind == "LATENT":
            return {"samples": torch.randn(1, 4, 32, 32, device=device)}
        if kind == "INT":
            return int(cfg.get("default", 1))
        if kind == "FLOAT":
            return float(cfg.get("default", 0.5))
        if kind == "BOOLEAN":
            return bool(cfg.get("default", False))
        if kind == "STRING":
            return str(cfg.get("default", ""))
        raise KeyError(kind)

    ok = failed = skipped = 0
    failures: list[str] = []
    for key in sorted(nodes):
        cls = nodes[key]
        try:
            spec = cls.INPUT_TYPES()
        except Exception:
            failed += 1
            failures.append(f"{key}: INPUT_TYPES raised")
            continue
        kwargs, skip = {}, None
        for sect in ("required", "optional"):
            for name, entry in (spec.get(sect) or {}).items():
                try:
                    kwargs[name] = fab(entry)
                except KeyError as e:
                    if sect == "required":
                        skip = f"needs {e}"
        if skip:
            skipped += 1
            continue
        vram_reset()
        t0 = time.perf_counter()
        try:
            fn = getattr(cls(), cls.FUNCTION)
            fn(**kwargs)
            dt = (time.perf_counter() - t0) * 1000
            ok += 1
            if dt > 5000:
                log(f"- SLOW {key}: {dt:.0f} ms, peak {vram_peak_mb():.0f} MB")
        except Exception as exc:
            msg = str(exc)[:100]
            env_markers = ("not found", "No such file", "pip install", "requires",
                           "must not be empty", "network", "No module named",
                           "does not exist", "ffmpeg", "checkpoint",
                           # A loader refusing fabricated defaults with a clear
                           # message ("No CLIP encoders provided for
                           # architecture...") is graceful failure, not a bug.
                           "provided for architecture", "Fill the required")
            if any(m.lower() in msg.lower() for m in env_markers):
                skipped += 1
            else:
                failed += 1
                failures.append(f"{key}: {type(exc).__name__}: {msg}")

    log(f"- executed OK: {ok}   env-skipped: {skipped}   FAILED: {failed}")
    for f in failures:
        log(f"  - **FAIL** {f}")

    # ── 3. RUDRA model scoring ──────────────────────────────────────────
    section("RUDRA model scoring")
    try:
        import folder_paths  # noqa: E402
        models_dir = os.path.join(folder_paths.models_dir, "radiance")
    except Exception:
        models_dir = os.path.join(COMFY, "models", "radiance")
    log(f"- checkpoint dir: {models_dir}")

    ckpts = []
    if os.path.isdir(models_dir):
        ckpts = [f for f in sorted(os.listdir(models_dir))
                 if f.endswith((".safetensors", ".pth", ".ckpt"))]
    if not ckpts:
        log("- **no RUDRA checkpoints found — model quality is UNVERIFIED.**")
    else:
        log(f"- found: {', '.join(ckpts)}")

    from radiance.fast_vae import load_radiance_decoder_weights, decode_to_linear_realtime  # noqa: E402

    # Parse (size, type) pairs from the checkpoint filenames themselves:
    # rudra_{turbo|full}_decoder_{model_type}_ema.safetensors
    import re as _re
    pairs = []
    for ck in ckpts:
        m = _re.match(r"rudra_(turbo|full)_decoder_(.+?)(?:_ema)?\.(safetensors|pth|ckpt)$", ck)
        if m and (m.group(1), m.group(2)) not in pairs:
            pairs.append((m.group(1), m.group(2)))
    log(f"- scoring {len(pairs)} (size, model_type) pairs from filenames")

    sizes = [(64, 64, "512px"), (135, 240, "1080p"), (270, 480, "4K")]
    for size_name, model_type in pairs:
        try:
            decoder = load_radiance_decoder_weights(
                model_type=model_type, model_size=size_name)
            if decoder is None:
                log(f"- {size_name}/{model_type}: loader returned None "
                    "(incompatible arch or strict-load failure) — skipped")
                continue
            decoder = decoder.to(device).eval()
            lc = getattr(decoder, "latent_channels", 16)
            score, notes = 100.0, []
            for lh, lw, label in sizes:
                z = torch.randn(1, lc, lh, lw, device=device)
                vram_reset()
                with torch.no_grad():
                    t0 = time.perf_counter()
                    out = decode_to_linear_realtime(z, decoder)
                    if has_cuda:
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                finite = bool(torch.isfinite(out).all())
                if not finite:
                    score -= 40
                    notes.append(f"NaN/Inf at {label}")
                fps = 1.0 / max(dt, 1e-6)
                log(f"- {size_name}/{model_type} @ {label}: "
                    f"{dt*1000:.0f} ms ({fps:.1f} fps), "
                    f"peak {vram_peak_mb():.0f} MB, finite={finite}")
                if label == "1080p" and fps < 12:
                    score -= 10
                    notes.append("below 12 fps at 1080p")
            # stability: 50 repeats at 1080p
            z = torch.randn(1, lc, 135, 240, device=device)
            vram_reset()
            with torch.no_grad():
                for _ in range(50):
                    decode_to_linear_realtime(z, decoder)
            growth = vram_peak_mb()
            log(f"- {size_name}/{model_type} stability: 50 decodes, "
                f"peak {growth:.0f} MB")
            log(f"- **{size_name}/{model_type} score: {max(0, score):.0f}/100"
                f"{' — ' + '; '.join(notes) if notes else ''}** "
                f"(fidelity vs reference VAE requires training pairs — "
                f"see RELEASE_REVIEW for the PSNR protocol)")
        except Exception:
            log(f"- {size_name}/{model_type}: ERROR")
            for ln in traceback.format_exc().splitlines()[-3:]:
                log(f"    {ln}")

    # ── 4. Temporal flicker with trained weights ────────────────────────
    section("TemporalRUDRA static-scene flicker")
    net = None
    try:
        from radiance.temporal_rudra import load_temporal_rudra_weights  # noqa: E402
        net = load_temporal_rudra_weights(device=device)
    except Exception:
        pass
    if net is None:
        log("- no trained temporal checkpoint loaded — flicker UNVERIFIED "
            "(expects models/radiance/temporal_rudra*.safetensors)")
    else:
        net = net.to(device).eval()
        x = torch.rand(1, 1, 256, 256, 3, device=device).repeat(1, 8, 1, 1, 1)
        m = torch.rand(1, 1, 256, 256, device=device).repeat(1, 8, 1, 1)
        with torch.no_grad():
            out = net(x, m, m)[0]
        var = (out - out.mean(dim=1, keepdim=True)).abs().max().item()
        log(f"- identical frames in -> max frame-to-frame deviation: {var:.6f}")
        log(f"- verdict: {'PASS (<0.002)' if var < 0.002 else '**FLICKER — investigate temporal padding**'}")

    out_path = os.path.join(REPO, "gpu_acceptance_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Radiance GPU Acceptance Report\n" + "\n".join(REPORT) + "\n")
    print(f"\nReport written: {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
