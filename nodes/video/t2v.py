# ============================================================
# FXTD STUDIOS — Radiance v3.0.0
# t2v.py  —  Text-to-Video & Image-to-Video Wrappers
# ============================================================
# Unified wrapper layer over any DiT video model loaded in ComfyUI.
# Supports: LTX-Video 2.x, HunyuanVideo, Wan2.1, CogVideoX, Mochi-1.
#
# Node overview
# -------------
#   RadianceVideoModelInfo    — inspect a loaded MODEL's latent spec
#   RadianceVideoLatentNoise  — generate correctly-shaped noise for any DiT
#   RadianceT2VPipeline       — text  → video latents (full sampling loop)
#   RadianceI2VPipeline       — image → video latents (image-conditioned)
#   RadianceVideoSampler      — low-level sampler shared by T2V and I2V
#   RadianceVideoBatchDecode   — decode video latents → IMAGE frame batch
#   RadianceVideoCondMerge    — merge character + HDR + text conditionings
#   RadianceVideoExport       — route frame batch to video writer / EXR
#
# ComfyUI integration notes
# -------------------------
# All sampling is done through ComfyUI's comfy.sample / comfy.samplers
# infrastructure — no custom CUDA kernels or model weights are shipped.
# When running in test environments without ComfyUI, every node falls
# back gracefully and returns zero tensors + descriptive error text.
#
# I2V conditioning strategy (per model)
# --------------------------------------
#   LTX-Video   — first frame injected via img2vid conditioning API
#                  (model.apply_model receives image_start tensor)
#   HunyuanVideo — first-frame encoded latent concatenated to noise
#   Wan2.1       — CLIP vision features injected into cross-attention
#   CogVideoX    — reference latent prepended to the latent sequence
#   Generic      — encode image, inject as first frame, pad remaining
# ============================================================

__version__ = "3.1.0"

import logging
logger = logging.getLogger("radiance.video.t2v")
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ComfyUI runtime imports (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import comfy.samplers
    import comfy.sample
    import comfy.utils
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Import DiT specs from our adapter module
try:
    from radiance.nodes.video.dit import _MODEL_SPECS, MODEL_NAMES, _get_spec, _wrap, _to_tensor
    _DIT_IMPORT = True
except ImportError:
        _DIT_IMPORT = False
        MODEL_NAMES = ["LTX-Video (128ch)", "HunyuanVideo (16ch)",
                       "Wan2.1 (16ch)", "CogVideoX (16ch)", "SD-VAE (4ch)"]
        def _get_spec(n): return {"channels": 16, "spatial_compression": 8,
                                   "temporal_compression": 4, "latent_scale": 1.0,
                                   "temporal": True}
        def _wrap(t): return {"samples": t}
        def _to_tensor(l): return l["samples"] if isinstance(l, dict) else l

# ---------------------------------------------------------------------------
# ComfyUI sampler / scheduler name lists
# ---------------------------------------------------------------------------

_SAMPLERS = [
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
    "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
    "deis", "ddim", "uni_pc", "uni_pc_bh2",
]
_SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform", "simple",
    "ddim_uniform", "beta",
]

# Per-model recommended defaults
_MODEL_DEFAULTS: Dict[str, Dict] = {
    "LTX-Video (128ch)":    {"sampler": "euler", "scheduler": "normal",   "steps": 25, "cfg": 3.5},
    "HunyuanVideo (16ch)": {"sampler": "euler", "scheduler": "simple",   "steps": 50, "cfg": 7.0},
    "Wan2.1 (16ch)":        {"sampler": "dpmpp_2m", "scheduler": "karras","steps": 30, "cfg": 5.0},
    "CogVideoX (16ch)":     {"sampler": "ddim", "scheduler": "ddim_uniform","steps": 50,"cfg": 6.0},
    "Mochi-1 (12ch)":       {"sampler": "euler", "scheduler": "normal",   "steps": 64, "cfg": 4.5},
    "SD-VAE (4ch)":         {"sampler": "euler", "scheduler": "karras",   "steps": 20, "cfg": 7.0},
}


# ===========================================================================
# ALBABIT-FIX: shared dit_config / noise-shape helpers (were triplicated
# across RadianceVideoLatentNoise, RadianceT2VPipeline, RadianceI2VPipeline)
# ===========================================================================

def _resolve_dit_config(dit_config, steps=0, cfg=0.0,
                        sampler_name="euler", scheduler="normal"):
    """Parse dit_config JSON and resolve model spec + effective sampling params."""
    try:
        cfg_dict = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
    except Exception:
        cfg_dict = {}
    spec        = _get_spec(cfg_dict.get("model_name", "")) if cfg_dict else _get_spec("LTX-Video (128ch)")
    spec.update(cfg_dict)
    model_name  = spec.get("model_name", "LTX-Video (128ch)")
    defaults    = _MODEL_DEFAULTS.get(model_name, {})
    eff_steps   = steps        if steps   > 0   else defaults.get("steps",    25)
    eff_cfg     = cfg          if cfg     > 0.0  else defaults.get("cfg",     7.0)
    eff_sampler = sampler_name or defaults.get("sampler",    "euler")
    eff_sched   = scheduler    or defaults.get("scheduler",  "normal")
    return spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched


def _build_noise_shape(spec, width, height, frames, batch_size=1):
    """Compute latent noise tensor shape from model spec and pixel dimensions."""
    sc   = spec.get("spatial_compression", 8)
    tc   = spec.get("temporal_compression", 4)
    ch   = spec.get("channels", 16)
    temp = spec.get("temporal", True)
    lh   = max(1, height // sc)
    lw   = max(1, width  // sc)
    lt   = max(1, math.ceil(frames / tc)) if temp else 1
    if temp and lt > 1:
        return (batch_size, ch, lt, lh, lw)
    return (batch_size, ch, lh, lw)


# ===========================================================================
# Shared sampling helper
# ===========================================================================

def _comfy_sample(model, noise, steps, cfg, sampler_name, scheduler,
                   positive, negative, latent_image, denoise=1.0, seed=0):
    """
    Thin wrapper around comfy.sample.sample_custom (modern ComfyUI API).
    Falls back to returning the noise unchanged when ComfyUI unavailable.
    """
    if not HAS_COMFY or not HAS_TORCH:
        return noise   # test / no-ComfyUI path

    # ALBABIT-FIX: migrate from legacy KSampler to sample_custom, matching
    # RadianceSamplerPro. ComfyUI v0.26.0 CFGGuider.sample() calls .is_nested
    # directly on the latent — LATENT dicts lack this attribute, raw tensors do not.
    if isinstance(latent_image, dict):
        latent_image = latent_image.get("samples", noise)

    try:
        # Sigma computation replicates KSampler.set_steps() / calculate_sigmas(),
        # including penultimate-sigma discard and partial-denoise slicing.
        _DISCARD_PENULTIMATE = {"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"}
        sampler_obj = comfy.samplers.sampler_object(sampler_name)
        model_sampling = model.get_model_object("model_sampling")

        def _calc_sigmas(n):
            _n = n + (1 if sampler_name in _DISCARD_PENULTIMATE else 0)
            sigs = comfy.samplers.calculate_sigmas(model_sampling, scheduler, _n)
            if sampler_name in _DISCARD_PENULTIMATE:
                sigs = torch.cat([sigs[:-2], sigs[-1:]])
            return sigs

        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        elif denoise >= 0.9999:
            sigmas = _calc_sigmas(steps)
        else:
            sigmas = _calc_sigmas(int(steps / denoise))[-(steps + 1):]
        sigmas = sigmas.to(noise.device)

        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler_obj, sigmas,
            positive, negative, latent_image, seed=seed,
        )
        # ALBABIT-FIX: guard against NaN/Inf that silently produce black/corrupt frames.
        # Mirrors the same guard in RadianceSamplerPro.sample().
        try:
            if not torch.isfinite(samples).all():
                _bad = int((~torch.isfinite(samples)).sum().item())
                logger.warning(
                    "[RadianceVideoSampler] %d non-finite value(s) (NaN/Inf) in sampled "
                    "latent — sanitizing via nan_to_num. Check CFG, scheduler, and precision.",
                    _bad,
                )
                samples = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as _nf_err:
            logger.debug("[RadianceVideoSampler] finite-check skipped: %s", _nf_err)
        return samples
    except Exception as exc:
        logger.warning("[RadianceT2V] sampling failed: %s", exc)
        return noise


def _make_noise(shape, seed: int = 0) -> "torch.Tensor":
    """Reproducible gaussian noise."""
    if not HAS_TORCH:
        return None
    gen = torch.Generator()
    gen.manual_seed(seed % (2**32))
    return torch.randn(shape, generator=gen)


def _vae_encode(vae, image_tensor) -> "torch.Tensor":
    """Encode an IMAGE tensor [B,H,W,3] through VAE → latent samples."""
    if not HAS_COMFY or not HAS_TORCH:
        B, H, W, C = image_tensor.shape
        return torch.zeros(B, 4, H // 8, W // 8)
    try:
        # ComfyUI VAE encodes images as [B,H,W,C] → LATENT dict
        encoded = vae.encode(image_tensor[:, :, :, :3])
        return encoded
    except Exception as exc:
        logger.warning(f"[RadianceT2V] VAE encode failed: {exc}")
        B, H, W, C = image_tensor.shape
        return torch.zeros(B, 4, H // 8, W // 8)


def _vae_decode(vae, latent) -> "torch.Tensor":
    """Decode latents → IMAGE tensor [B,H,W,3]."""
    if not HAS_COMFY or not HAS_TORCH:
        return latent
    try:
        samples = _to_tensor(latent)
        return vae.decode(samples)
    except Exception as exc:
        logger.warning(f"[RadianceT2V] VAE decode failed: {exc}")
        return torch.zeros(1, 64, 64, 3)


# ===========================================================================
# Node: RadianceVideoModelInfo
# ===========================================================================

class RadianceVideoModelInfo:
    """
    Inspect a ComfyUI MODEL object and produce a DiT config JSON describing
    its latent space.  Attempts to auto-detect the model type from its class
    name / config attributes; falls back to the user-selected preset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_preset": (MODEL_NAMES, {"default": "LTX-Video (128ch)"}),
            },
            "optional": {
                "override_channels": ("INT", {"default": 0, "min": 0, "max": 512}),
                "override_latent_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0}),
                "print_info": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "dit_config", "info_report")
    FUNCTION = "inspect"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Display configuration and parameter info for a loaded video model."

    def inspect(self, model, model_preset: str,
                override_channels: int = 0,
                override_latent_scale: float = 0.0,
                print_info: bool = False):

        spec = dict(_get_spec(model_preset))
        spec["model_name"] = model_preset

        # Auto-detect from model class name
        cls_name = type(model).__name__.lower() if model is not None else ""
        model_inner = getattr(model, "model", None)
        inner_name  = type(model_inner).__name__.lower() if model_inner else ""
        combined    = cls_name + " " + inner_name

        detected = None
        if "ltx" in combined:
            detected = "LTX-Video (128ch)"
        elif "hunyuan" in combined:
            detected = "HunyuanVideo (16ch)"
        elif "wan" in combined:
            detected = "Wan2.1 (16ch)"
        elif "cogvideo" in combined:
            detected = "CogVideoX (16ch)"
        elif "mochi" in combined:
            detected = "Mochi-1 (12ch)"

        if detected and detected != model_preset:
            spec = dict(_get_spec(detected))
            spec["model_name"] = detected

        if override_channels > 0:
            spec["channels"] = override_channels
        if override_latent_scale > 0:
            spec["latent_scale"] = override_latent_scale

        info = [
            f"=== RadianceVideoModelInfo v{__version__} ===",
            f"Preset      : {model_preset}",
            f"Auto-detect : {detected or '(no match)'}",
            f"Final spec  :",
            f"  channels           = {spec['channels']}",
            f"  spatial_compression= {spec['spatial_compression']}",
            f"  temporal_compression= {spec.get('temporal_compression', 1)}",
            f"  latent_scale       = {spec['latent_scale']}",
            f"  has_temporal       = {spec.get('temporal', False)}",
        ]
        if print_info:
            logger.info("\n".join(info))

        defaults = _MODEL_DEFAULTS.get(spec["model_name"], {})
        spec["defaults"] = defaults

        return (model, json.dumps(spec), "\n".join(info))


# ===========================================================================
# Node: RadianceVideoLatentNoise
# ===========================================================================

class RadianceVideoLatentNoise:
    """
    Generate correctly-shaped Gaussian noise for a given DiT video model
    and target resolution / frame count.

    For temporal models the output shape is [B, C, T, H//sc, W//sc].
    For non-temporal (image) models it is [B, C, H//sc, W//sc].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit_config": ("STRING", {
                    "default": "{}",
                    "tooltip": "JSON from RadianceVideoModelInfo",
                }),
                "width":  ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 8}),
                "frames": ("INT", {"default": 25,   "min": 1,  "max": 512}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "seed":   ("INT", {"default": 0,   "min": 0,  "max": 2**31}),
            },
            "optional": {
                "noise_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01,
                    "tooltip": "Multiply noise standard deviation (1.0 = unit Gaussian)",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("noise_latent", "shape_report")
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Generate temporally-coherent latent noise for video initialisation."

    def generate(self, dit_config: str, width: int, height: int, frames: int,
                 batch_size: int, seed: int, noise_scale: float = 1.0):

        try:
            cfg = json.loads(dit_config)
        except Exception:
            cfg = {}

        spec = _get_spec(cfg.get("model_name", "")) if cfg else _get_spec("LTX-Video (128ch)")
        if cfg:
            spec.update(cfg)

        shape = _build_noise_shape(spec, width, height, frames, batch_size)
        sc = spec.get("spatial_compression", 8)  # for report
        tc = spec.get("temporal_compression", 4)  # for report

        noise = _make_noise(shape, seed)
        if noise is not None:
            noise = noise * noise_scale

        report = [
            f"=== RadianceVideoLatentNoise ===",
            f"Model     : {cfg.get('model_name', 'unknown')}",
            f"Input     : {width}×{height}  {frames}f",
            f"Latent    : {list(shape)}",
            f"SC×TC     : {sc}×{tc}",
            f"Seed      : {seed}",
        ]

        if noise is None:
            import numpy as np_
            noise_np = np_.random.default_rng(seed).standard_normal(shape).astype("float32")
            noise = noise_np   # numpy path for non-torch environments
            return ({"samples": noise}, "\n".join(report))

        return (_wrap(noise), "\n".join(report))


# ===========================================================================
# Node: RadianceVideoCondMerge
# ===========================================================================

class RadianceVideoCondMerge:
    """
    Merge up to three conditioning inputs (text, character, HDR) into a
    single positive conditioning tensor.

    Merge modes
    -----------
    concat    — append tokens from each conditioning in order
    weighted  — scale each conditioning by its weight then sum / average
    priority  — use the first non-empty conditioning as base, patch in others
    """

    MODES = ["concat", "weighted", "priority"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_conditioning": ("CONDITIONING",),
                "merge_mode": (cls.MODES, {"default": "concat"}),
            },
            "optional": {
                "character_conditioning": ("CONDITIONING",),
                "hdr_conditioning": ("CONDITIONING",),
                "text_weight":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "character_weight": ("FLOAT", {"default": 0.75,"min": 0.0, "max": 2.0, "step": 0.05}),
                "hdr_weight":       ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("merged_conditioning", "merge_report")
    FUNCTION = "merge"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Merge multiple video conditioning signals into a unified tensor."

    def merge(self, text_conditioning, merge_mode,
              character_conditioning=None, hdr_conditioning=None,
              text_weight=1.0, character_weight=0.75, hdr_weight=0.5):

        sources = [
            (text_conditioning,      text_weight,      "text"),
            (character_conditioning, character_weight, "character"),
            (hdr_conditioning,       hdr_weight,       "hdr"),
        ]
        sources = [(c, w, n) for c, w, n in sources if c is not None]

        report = [f"=== RadianceVideoCondMerge === mode={merge_mode}",
                  f"Sources: {[n for _,_,n in sources]}"]

        if not sources:
            return (text_conditioning, "\n".join(report))

        if len(sources) == 1 or not HAS_TORCH:
            return (sources[0][0], "\n".join(report))

        if merge_mode == "concat":
            merged = self._concat(sources)
        elif merge_mode == "weighted":
            merged = self._weighted(sources)
        else:
            merged = self._priority(sources)

        report.append(f"Output pairs: {len(merged)}")
        return (merged, "\n".join(report))

    def _concat(self, sources):
        """Concatenate token sequences from all sources."""
        result = []
        # Use the text conditioning as base pairs
        base_cond, base_weight, _ = sources[0]
        for i, (base_t, base_d) in enumerate(base_cond):
            new_d = dict(base_d)
            tokens = base_t
            for extra_cond, w, name in sources[1:]:
                if i < len(extra_cond):
                    ext_t, ext_d = extra_cond[i]
                    # Project extra tokens to same token_dim if needed
                    if ext_t.shape[-1] != tokens.shape[-1]:
                        try:
                            ext_t = F.interpolate(
                                ext_t.float().unsqueeze(0),
                                size=(ext_t.shape[-2], tokens.shape[-1]),
                                mode="bilinear", align_corners=False,
                            ).squeeze(0).to(tokens.dtype)
                        except Exception:
                            continue
                    tokens = torch.cat([tokens, ext_t * w], dim=1)
                    # Merge extra dicts
                    for k, v in ext_d.items():
                        if k not in new_d:
                            new_d[k] = v
            result.append([tokens, new_d])
        return result

    def _weighted(self, sources):
        """Weighted sum of token tensors (mean-pooled to same length)."""
        result = []
        base_cond, base_weight, _ = sources[0]
        for i, (base_t, base_d) in enumerate(base_cond):
            acc = base_t * base_weight
            new_d = dict(base_d)
            total_w = base_weight
            for extra_cond, w, name in sources[1:]:
                if i < len(extra_cond):
                    ext_t, ext_d = extra_cond[i]
                    # Pool or pad to match base seq_len
                    tgt_len = acc.shape[1]
                    if ext_t.shape[1] != tgt_len:
                        ext_t = F.adaptive_avg_pool1d(
                            ext_t.float().permute(0, 2, 1),
                            tgt_len,
                        ).permute(0, 2, 1).to(acc.dtype)
                    acc = acc + ext_t * w
                    total_w += w
            result.append([acc / max(total_w, 1e-8), new_d])
        return result

    def _priority(self, sources):
        """Use first source as base, patch extra dict keys from others."""
        result = []
        base_cond, _, _ = sources[0]
        for i, (base_t, base_d) in enumerate(base_cond):
            new_d = dict(base_d)
            for extra_cond, _, name in sources[1:]:
                if i < len(extra_cond):
                    _, ext_d = extra_cond[i]
                    for k, v in ext_d.items():
                        if k not in new_d:
                            new_d[k] = v
            result.append([base_t, new_d])
        return result


# ===========================================================================
# Node: RadianceVideoSampler
# ===========================================================================

class RadianceVideoSampler:
    """
    Low-level video sampler — shared engine behind T2V and I2V pipelines.

    Accepts pre-built noise + conditioning and runs the ComfyUI sampling
    loop via _comfy_sample(). An optional cfg_schedule_json (from
    RadianceAudioCFGSchedule) overrides the static CFG value with the
    first frame's schedule value. Tiling is automatically restored after
    sampling.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Run the diffusion sampler to generate video latents from pre-built noise and conditioning."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_noise": ("LATENT",),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (_SAMPLERS, {"default": "euler"}),
                "scheduler": (_SCHEDULERS, {"default": "normal"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31}),
            },
            "optional": {
                # ALBABIT-FIX: dit_config promoted from required to optional.
                # When connected (RadianceVideoModelInfo), model-specific defaults
                # (steps/cfg/sampler/scheduler) override the manual widgets.
                # When absent, widget values are used as-is.
                "dit_config": ("STRING", {
                    "default": "{}",
                    "tooltip": "JSON from RadianceVideoModelInfo — when connected, overrides steps/cfg/sampler/scheduler with model-specific defaults.",
                }),
                "cfg_schedule_json": ("STRING", {
                    "default": "",
                    "tooltip": "JSON float array from RadianceAudioCFGSchedule — first value overrides CFG",
                }),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled sampling for large resolutions (reduces VRAM)",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "sampler_report")
    FUNCTION = "sample"

    def sample(self, model, positive, negative, latent_noise,
               steps, cfg, sampler_name, scheduler, seed,
               dit_config="{}", cfg_schedule_json="", denoise=1.0, tiling=False):

        # ALBABIT-FIX: resolve effective sampling params — dit_config wins when connected.
        # A real dit_config always carries a "model_name" key from RadianceVideoModelInfo.
        # An absent/empty dit_config ("{}") means the node is not connected → use widgets.
        try:
            _dc = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
        except Exception:
            _dc = {}

        if _dc.get("model_name"):
            _, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, 0, 0.0, "", "",
            )
            _from_dit_config = True
        else:
            eff_steps, eff_cfg, eff_sampler, eff_sched = steps, cfg, sampler_name, scheduler
            model_name = "manual"
            _from_dit_config = False

        # Parse CFG schedule — applied on top of whichever source won above
        cfg_eff = eff_cfg
        _cfg_from_schedule = False  # ALBABIT-FIX: track actual parse success for the report label
        if cfg_schedule_json.strip():
            try:
                parsed = json.loads(cfg_schedule_json)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    cfg_eff = float(parsed[0])
                    _cfg_from_schedule = True
            except Exception as _cfg_err:
                # ALBABIT-FIX: warn instead of silently falling back to static CFG
                logger.warning(
                    "[RadianceVideoSampler] cfg_schedule_json parse failed (%s) — "
                    "using static CFG value %.2f.", _cfg_err, eff_cfg,
                )

        _cfg_src = " (schedule)" if _cfg_from_schedule else (" (dit_config)" if _from_dit_config else "")
        report = [
            f"=== RadianceVideoSampler v{__version__} ===",
            f"Model    : {model_name}",
            f"Sampler  : {eff_sampler}",
            f"Scheduler: {eff_sched}",
            f"Steps    : {eff_steps}",
            f"CFG      : {cfg_eff}{_cfg_src}",
            f"Denoise  : {denoise}",
            f"Seed     : {seed}",
        ]

        if not HAS_TORCH:
            report.append("torch unavailable — returning noise unchanged")
            return (latent_noise, "\n".join(report))

        # Enable tiling if requested; always restore after sampling
        try:
            if tiling and hasattr(model, "model") and hasattr(model.model, "set_tiling"):
                model.model.set_tiling(True)
                report.append("Tiling: enabled")

            noise = _to_tensor(latent_noise)
            samples = _comfy_sample(
                model, noise, eff_steps, cfg_eff, eff_sampler, eff_sched,
                positive, negative, latent_noise, denoise=denoise, seed=seed,
            )
            report.append(f"Output shape: {list(samples.shape) if HAS_TORCH else 'N/A'}")
            return (_wrap(samples), "\n".join(report))

        finally:
            # Restore tiling so it doesn't leak to subsequent nodes
            if tiling and hasattr(model, "model") and hasattr(model.model, "set_tiling"):
                try:
                    model.model.set_tiling(False)
                except Exception:
                    pass


# ===========================================================================
# Node: RadianceT2VPipeline
# ===========================================================================

class RadianceT2VPipeline:
    """
    Full Text-to-Video pipeline — one node from prompt to video IMAGE batch.

    Sampling is performed via _comfy_sample() — the same module-level helper
    used by RadianceVideoSampler. Per-model defaults (sampler, scheduler,
    steps, CFG) are resolved from dit_config via _resolve_dit_config().
    You can override every parameter explicitly.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "End-to-end text-to-video generation pipeline with HDR support."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "cinematic HDR video, stunning visuals, 4K, film grain",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "watermark, blurry, low quality, sdr, flickering",
                }),
                "width":  ("INT", {"default": 768,  "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 8}),
                "frames": ("INT", {"default": 25,   "min": 1,  "max": 512}),
                "seed":   ("INT", {"default": 0,    "min": 0,  "max": 2**31}),
            },
            "optional": {
                "dit_config": ("STRING", {"default": "{}",
                    "tooltip": "JSON from RadianceVideoModelInfo — sets model-specific defaults"}),
                "character_conditioning": ("CONDITIONING",),
                "cfg_schedule_json": ("STRING", {"default": "",
                    "tooltip": "JSON float array from RadianceAudioCFGSchedule"}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "0 = use model default"}),
                "cfg":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "0 = use model default"}),
                "sampler_name": (_SAMPLERS, {"default": "euler"}),
                "scheduler":    (_SCHEDULERS, {"default": "normal"}),
                "peak_nits":    ([str(n) for n in [100,203,400,600,1000,4000,10000]],
                                  {"default": "1000"}),
                "target_gamut": (["BT.2020","P3-D65","P3-DCI","BT.709","ACEScg"],
                                  {"default": "BT.2020"}),
                "hdr_eotf":     (["PQ (ST.2084)","HLG (BT.2100)","Linear","sRGB / BT.1886"],
                                  {"default": "PQ (ST.2084)"}),
                "hdr_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "CONDITIONING", "STRING")
    RETURN_NAMES = ("video_latent", "preview_frames", "positive_cond", "pipeline_report")
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"

    def generate(self, model, clip, vae, positive_prompt, negative_prompt,
                 width, height, frames, seed,
                 dit_config="{}", character_conditioning=None,
                 cfg_schedule_json="", steps=0, cfg=0.0,
                 sampler_name="euler", scheduler="normal",
                 peak_nits="1000", target_gamut="BT.2020",
                 hdr_eotf="PQ (ST.2084)", hdr_strength=0.5):

        report = [f"=== RadianceT2VPipeline v{__version__} ===",
                  f"Prompt : {positive_prompt[:80]}...",
                  f"Size   : {width}×{height}  {frames}f  seed={seed}"]

        # ALBABIT-FIX: dit_config wins when connected (model_name key present),
        # mirroring RadianceVideoSampler.sample() — JS greys out manual widgets.
        try:
            _dc = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
        except Exception:
            _dc = {}
        if _dc.get("model_name"):
            spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, 0, 0.0, "", "")
            _from_dit_config = True
        else:
            spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, steps, cfg, sampler_name, scheduler)
            _from_dit_config = False

        # ALBABIT-FIX: cfg_schedule_json was accepted but silently ignored — mirrors
        # RadianceVideoSampler.sample() parsing logic.
        cfg_eff = eff_cfg
        _cfg_from_schedule = False
        if cfg_schedule_json.strip():
            try:
                _parsed = json.loads(cfg_schedule_json)
                if isinstance(_parsed, (list, tuple)) and len(_parsed) > 0:
                    cfg_eff = float(_parsed[0])
                    _cfg_from_schedule = True
            except Exception as _cfg_err:
                logger.warning(
                    "[RadianceT2VPipeline] cfg_schedule_json parse failed (%s) — "
                    "using static CFG %.2f.", _cfg_err, eff_cfg,
                )

        _cfg_src = " (schedule)" if _cfg_from_schedule else (" (dit_config)" if _from_dit_config else "")
        report.append(f"Model  : {model_name}")
        report.append(f"Steps={eff_steps}  CFG={cfg_eff}{_cfg_src}  {eff_sampler}/{eff_sched}")

        # --- Build positive conditioning ---
        hdr_tokens = self._hdr_tokens(peak_nits, target_gamut, hdr_eotf, hdr_strength)
        full_prompt = positive_prompt.strip()
        if hdr_tokens:
            full_prompt = full_prompt.rstrip(", ") + ", " + hdr_tokens

        pos_cond = self._encode_text(clip, full_prompt)
        neg_cond = self._encode_text(clip, negative_prompt)

        # Merge character conditioning if provided
        if character_conditioning is not None and pos_cond is not None:
            pos_cond = self._merge_cond(pos_cond, character_conditioning, 0.75)

        # --- Generate noise ---
        noise_shape = _build_noise_shape(spec, width, height, frames)

        noise = _make_noise(noise_shape, seed)
        if noise is None:
            report.append("ERROR: torch unavailable")
            return (_wrap(self._zero_latent(noise_shape)),
                    self._zero_image(height, width, frames),
                    pos_cond or [], "\n".join(report))

        noise_latent = _wrap(noise)
        report.append(f"Noise shape: {list(noise_shape)}")

        # --- Sample ---
        samples = _comfy_sample(
            model, noise, eff_steps, cfg_eff, eff_sampler, eff_sched,
            pos_cond or [], neg_cond or [], noise_latent,
            denoise=1.0, seed=seed,
        )
        video_latent = _wrap(samples)
        report.append(f"Sampled: {list(samples.shape)}")

        # --- Decode preview ---
        preview = self._decode_preview(vae, video_latent, frames)
        report.append(f"Preview frames: {preview.shape[0]}")

        return (video_latent, preview, pos_cond or [], "\n".join(report))

    # ------------------------------------------------------------------
    def _encode_text(self, clip, text):
        if clip is None or not HAS_TORCH:
            return []
        try:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]
        except Exception as exc:
            logger.warning(f"[T2V] text encode failed: {exc}")
            return []

    def _hdr_tokens(self, peak_nits, gamut, eotf, strength):
        if strength < 0.01:
            return ""
        _G = {"BT.2020": "wide color gamut rec2020 vivid",
               "P3-D65": "DCI-P3 cinema color",
               "BT.709": "standard dynamic range",
               "ACEScg": "aces linear light vfx reference"}
        _E = {"PQ (ST.2084)": "HDR10 PQ specular highlights",
               "HLG (BT.2100)": "HLG broadcast HDR",
               "Linear": "linear EXR 32bit"}
        parts = [_G.get(gamut, ""), _E.get(eotf, ""),
                 f"{peak_nits} nits HDR" if int(peak_nits) > 100 else ""]
        return ", ".join(p for p in parts if p)

    def _merge_cond(self, base, extra, weight):
        if not HAS_TORCH:
            return base
        merged = []
        for i, (bt, bd) in enumerate(base):
            new_d = dict(bd)
            if i < len(extra):
                et, _ = extra[i]
                if et.shape[-1] == bt.shape[-1]:
                    tok = torch.cat([bt, et * weight], dim=1)
                else:
                    tok = bt
            else:
                tok = bt
            merged.append([tok, new_d])
        return merged

    def _decode_preview(self, vae, latent, target_frames):
        if not HAS_TORCH:
            return self._zero_image(64, 64, target_frames)
        try:
            decoded = _vae_decode(vae, latent)
            # Ensure [B,H,W,3]
            if decoded.dim() == 5:
                B, C, T, H, W = decoded.shape
                decoded = decoded.permute(0, 2, 3, 4, 1).reshape(B * T, H, W, C)
            return decoded.clamp(0, 1)
        except Exception:
            return self._zero_image(64, 64, target_frames)

    def _zero_latent(self, shape):
        if HAS_TORCH:
            return torch.zeros(shape)
        return shape

    def _zero_image(self, h, w, n):
        if HAS_TORCH:
            return torch.zeros(n, h, w, 3)
        return None


# ===========================================================================
# Node: RadianceI2VPipeline
# ===========================================================================

class RadianceI2VPipeline:
    """
    Full Image-to-Video pipeline — one reference image → video IMAGE batch.

    Sampling is performed via _comfy_sample() — the same module-level helper
    used by RadianceVideoSampler and RadianceT2VPipeline.

    I2V conditioning strategy (applied automatically by model type)
    ---------------------------------------------------------------
    LTX-Video      — reference image injected via model's img2vid API;
                     first latent frame is clamped to the image encoding
    HunyuanVideo   — encoded image latent concatenated channel-wise to noise
    Wan2.1         — CLIP vision embedding injected into cross-attention
    CogVideoX      — reference latent prepended to the video latent sequence
    Generic        — image encoded and blended into the first latent frame

    Outputs
    -------
    video_latent   — raw DiT latent (route to RadianceVideoHDRDecode)
    preview_frames — quick VAE-decoded IMAGE batch
    pipeline_report — diagnostic text
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "End-to-end image-to-video generation pipeline with motion control."

    I2V_STRATEGIES = [
        "auto", "first_frame_lock", "concat_channels", "clip_vision_inject",
        "prepend_latent",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "smooth camera motion, cinematic HDR, 4K",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "watermark, blurry, flickering, sdr",
                }),
                "frames": ("INT", {"default": 25, "min": 1, "max": 512}),
                "seed":   ("INT", {"default": 0,  "min": 0, "max": 2**31}),
            },
            "optional": {
                "dit_config": ("STRING", {"default": "{}"}),
                "character_conditioning": ("CONDITIONING",),
                "cfg_schedule_json": ("STRING", {"default": ""}),
                "i2v_strategy": (cls.I2V_STRATEGIES, {"default": "auto"}),
                "image_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How strongly the reference image anchors the generation",
                }),
                "motion_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Amount of motion / temporal variation (0=nearly static)",
                }),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200}),
                "cfg":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (_SAMPLERS, {"default": "euler"}),
                "scheduler":    (_SCHEDULERS, {"default": "normal"}),
                "peak_nits":    ([str(n) for n in [100,203,400,600,1000,4000,10000]],
                                  {"default": "1000"}),
                "target_gamut": (["BT.2020","P3-D65","P3-DCI","BT.709","ACEScg"],
                                  {"default": "BT.2020"}),
                "hdr_eotf":     (["PQ (ST.2084)","HLG (BT.2100)","Linear","sRGB / BT.1886"],
                                  {"default": "PQ (ST.2084)"}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("video_latent", "preview_frames", "pipeline_report")
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"

    def generate(self, model, clip, vae, reference_image,
                 positive_prompt, negative_prompt, frames, seed,
                 dit_config="{}", character_conditioning=None,
                 cfg_schedule_json="", i2v_strategy="auto",
                 image_strength=0.85, motion_strength=0.5,
                 steps=0, cfg=0.0, sampler_name="euler", scheduler="normal",
                 peak_nits="1000", target_gamut="BT.2020", hdr_eotf="PQ (ST.2084)"):

        report = [f"=== RadianceI2VPipeline v{__version__} ===",
                  f"Strategy       : {i2v_strategy}",
                  f"Image strength : {image_strength}",
                  f"Motion strength: {motion_strength}",
                  f"Frames         : {frames}  seed={seed}"]

        # ALBABIT-FIX: dit_config wins when connected (model_name key present),
        # mirroring RadianceVideoSampler.sample() — JS greys out manual widgets.
        try:
            _dc = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
        except Exception:
            _dc = {}
        if _dc.get("model_name"):
            spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, 0, 0.0, "", "")
            _from_dit_config = True
        else:
            spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, steps, cfg, sampler_name, scheduler)
            _from_dit_config = False

        # ALBABIT-FIX: cfg_schedule_json was accepted but silently ignored — mirrors
        # RadianceVideoSampler.sample() parsing logic.
        cfg_eff = eff_cfg
        _cfg_from_schedule = False
        if cfg_schedule_json.strip():
            try:
                _parsed = json.loads(cfg_schedule_json)
                if isinstance(_parsed, (list, tuple)) and len(_parsed) > 0:
                    cfg_eff = float(_parsed[0])
                    _cfg_from_schedule = True
            except Exception as _cfg_err:
                logger.warning(
                    "[RadianceI2VPipeline] cfg_schedule_json parse failed (%s) — "
                    "using static CFG %.2f.", _cfg_err, eff_cfg,
                )

        _cfg_src = " (schedule)" if _cfg_from_schedule else (" (dit_config)" if _from_dit_config else "")
        report.append(f"Model  : {model_name}")
        report.append(f"Steps={eff_steps}  CFG={cfg_eff}{_cfg_src}  {eff_sampler}/{eff_sched}")

        # Resolve I2V strategy
        strategy = i2v_strategy
        if strategy == "auto":
            if   "LTX"      in model_name: strategy = "first_frame_lock"
            elif "Hunyuan"  in model_name: strategy = "concat_channels"
            elif "Wan"      in model_name: strategy = "clip_vision_inject"
            elif "CogVideo" in model_name: strategy = "prepend_latent"
            else:                          strategy = "first_frame_lock"
        report.append(f"Resolved strategy: {strategy}")

        # --- Encode reference image ---
        img_latent = None
        if HAS_TORCH and vae is not None:
            img_latent_raw = _vae_encode(vae, reference_image)
            if HAS_TORCH and isinstance(img_latent_raw, torch.Tensor):
                img_latent = img_latent_raw
            elif isinstance(img_latent_raw, dict):
                img_latent = _to_tensor(img_latent_raw)
            report.append(f"Image latent: {list(img_latent.shape) if img_latent is not None else 'N/A'}")

        # --- Build noise ---
        _, img_h, img_w, _ = reference_image.shape
        noise_shape = _build_noise_shape(spec, img_w, img_h, frames)
        ch = spec.get("channels", 16)  # needed by _first_frame_lock / _concat_channels

        noise = _make_noise(noise_shape, seed)
        if noise is None:
            report.append("ERROR: torch unavailable")
            return (_wrap(self._zero(noise_shape)),
                    self._zero_image(img_h, img_w, frames),
                    "\n".join(report))

        # Scale noise by motion_strength (less noise = less motion)
        motion_noise = noise * (0.2 + motion_strength * 0.8)

        # --- Apply I2V strategy to conditioning ---
        pos_cond = self._encode_text(clip, positive_prompt, peak_nits, target_gamut, hdr_eotf)
        neg_cond = self._encode_text(clip, negative_prompt)

        if character_conditioning is not None and pos_cond:
            pos_cond = self._merge_cond(pos_cond, character_conditioning, 0.75)

        start_latent = _wrap(motion_noise)

        # Strategy-specific noise / conditioning preparation
        if strategy == "first_frame_lock" and img_latent is not None:
            motion_noise = self._first_frame_lock(
                motion_noise, img_latent, image_strength, noise_shape, ch)

        elif strategy == "concat_channels" and img_latent is not None:
            motion_noise, pos_cond = self._concat_channels(
                motion_noise, img_latent, pos_cond, image_strength, ch)

        elif strategy == "clip_vision_inject" and img_latent is not None:
            pos_cond = self._clip_vision_inject(pos_cond, reference_image, image_strength)

        elif strategy == "prepend_latent" and img_latent is not None:
            motion_noise = self._prepend_latent(motion_noise, img_latent, image_strength)

        start_latent = _wrap(motion_noise)
        report.append(f"Final noise shape: {list(motion_noise.shape)}")

        # --- Sample ---
        denoise = 1.0 - (image_strength * 0.4)   # stronger image → less denoising
        samples = _comfy_sample(
            model, motion_noise, eff_steps, cfg_eff, eff_sampler, eff_sched,
            pos_cond, neg_cond, start_latent,
            denoise=denoise, seed=seed,
        )
        video_latent = _wrap(samples)
        report.append(f"Sampled: {list(samples.shape)}")

        # --- Decode preview ---
        preview = self._decode_preview(vae, video_latent, frames)
        report.append(f"Preview frames: {preview.shape[0]}")

        return (video_latent, preview, "\n".join(report))

    # ------------------------------------------------------------------
    def _encode_text(self, clip, text, peak_nits=None, gamut=None, eotf=None):
        if clip is None or not HAS_TORCH:
            return []
        hdr_sfx = ""
        if peak_nits:
            n = int(peak_nits)
            if n > 100:
                hdr_sfx = f", {n} nits HDR, {gamut or 'BT.2020'}"
        try:
            tokens = clip.tokenize(text + hdr_sfx)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]
        except Exception as exc:
            logger.warning("[nodes_t2v_pipeline] _encode_text: %s", exc)
            return []

    def _merge_cond(self, base, extra, weight):
        if not HAS_TORCH:
            return base
        merged = []
        for i, (bt, bd) in enumerate(base):
            new_d = dict(bd)
            if i < len(extra):
                et, _ = extra[i]
                if et.shape[-1] == bt.shape[-1]:
                    tok = torch.cat([bt, et * weight], dim=1)
                else:
                    tok = bt
            else:
                tok = bt
            merged.append([tok, new_d])
        return merged

    # --- Strategy implementations ---

    def _first_frame_lock(self, noise, img_latent, strength, shape, ch):
        """Lock the first frame of the video latent to the image encoding."""
        if img_latent is None or not HAS_TORCH:
            return noise
        try:
            if noise.dim() == 5:
                # Resize img_latent to match spatial dims
                il = img_latent.float()
                if il.shape[1] != ch:
                    rep = math.ceil(ch / il.shape[1])
                    il = il.repeat(1, rep, 1, 1)[:, :ch]
                lh, lw = noise.shape[-2], noise.shape[-1]
                if il.shape[-2] != lh or il.shape[-1] != lw:
                    il = F.interpolate(il, size=(lh, lw), mode="bilinear", align_corners=False)
                # Lock first temporal frame
                locked = noise.clone()
                locked[:, :, 0] = locked[:, :, 0] * (1 - strength) + il * strength
                return locked
            else:
                lh, lw = noise.shape[-2], noise.shape[-1]
                il = F.interpolate(img_latent.float(), size=(lh, lw),
                                    mode="bilinear", align_corners=False)
                if il.shape[1] != ch:
                    rep = math.ceil(ch / il.shape[1])
                    il = il.repeat(1, rep, 1, 1)[:, :ch]
                return noise * (1 - strength) + il * strength
        except Exception as exc:
            logger.warning(f"[I2V first_frame_lock] {exc}")
            return noise

    def _concat_channels(self, noise, img_latent, pos_cond, strength, ch):
        """Concatenate image encoding as extra channels on the noise."""
        if img_latent is None or not HAS_TORCH:
            return noise, pos_cond
        try:
            lh, lw = noise.shape[-2], noise.shape[-1]
            il = F.interpolate(img_latent.float(), size=(lh, lw),
                                mode="bilinear", align_corners=False) * strength
            if noise.dim() == 5:
                # Expand to temporal: [B,C,T,H,W]
                il_t = il.unsqueeze(2).expand(-1, -1, noise.shape[2], -1, -1)
                new_noise = torch.cat([noise, il_t], dim=1)
            else:
                new_noise = torch.cat([noise, il], dim=1)
            # Store extra channels count in conditioning dict so model can use it
            for idx, (ct, cd) in enumerate(pos_cond):
                pos_cond[idx][1]["image_concat_channels"] = il.shape[1]
            return new_noise, pos_cond
        except Exception as exc:
            logger.warning(f"[I2V concat_channels] {exc}")
            return noise, pos_cond

    def _clip_vision_inject(self, pos_cond, image, strength):
        """Inject CLIP vision embedding of image into conditioning dict."""
        if not HAS_TORCH:
            return pos_cond
        try:
            # Resize to 224×224 for CLIP input
            img_224 = F.interpolate(
                image.permute(0, 3, 1, 2).float(),
                size=(224, 224), mode="bilinear", align_corners=False,
            )
            # Normalise (CLIP preprocessing)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            img_224 = (img_224 - mean) / std
            for idx, (ct, cd) in enumerate(pos_cond):
                pos_cond[idx][1]["clip_vision_input"]   = img_224
                pos_cond[idx][1]["clip_vision_strength"] = strength
            return pos_cond
        except Exception as exc:
            logger.warning(f"[I2V clip_vision] {exc}")
            return pos_cond

    def _prepend_latent(self, noise, img_latent, strength):
        """Prepend image latent as first frame(s) of video latent."""
        if img_latent is None or noise.dim() != 5 or not HAS_TORCH:
            return noise
        try:
            lh, lw = noise.shape[-2], noise.shape[-1]
            ch     = noise.shape[1]
            il = F.interpolate(img_latent.float(), size=(lh, lw),
                                mode="bilinear", align_corners=False)
            if il.shape[1] != ch:
                rep = math.ceil(ch / il.shape[1])
                il = il.repeat(1, rep, 1, 1)[:, :ch]
            il_t = il.unsqueeze(2) * strength   # [B,C,1,H,W]
            # Replace first temporal frame
            result = noise.clone()
            result[:, :, :1] = result[:, :, :1] * (1 - strength) + il_t
            return result
        except Exception as exc:
            logger.warning(f"[I2V prepend_latent] {exc}")
            return noise

    def _decode_preview(self, vae, latent, target_frames):
        if not HAS_TORCH:
            return torch.zeros(target_frames, 64, 64, 3)
        try:
            decoded = _vae_decode(vae, latent)
            if decoded.dim() == 5:
                B, C, T, H, W = decoded.shape
                decoded = decoded.permute(0, 2, 3, 4, 1).reshape(B * T, H, W, C)
            return decoded.clamp(0, 1)
        except Exception:
            return torch.zeros(target_frames, 64, 64, 3) if HAS_TORCH else None

    def _zero(self, shape):
        return torch.zeros(shape) if HAS_TORCH else shape

    def _zero_image(self, h, w, n):
        return torch.zeros(n, h, w, 3) if HAS_TORCH else None


# ===========================================================================
# Node: RadianceVideoBatchDecode
# ===========================================================================

class RadianceVideoBatchDecode:
    """
    Decode a DiT video LATENT tensor into an IMAGE frame batch.

    Handles both temporal (5-D) and spatial (4-D) latents.
    Applies optional per-model latent scale correction before decoding.
    Outputs IMAGE [N_frames, H, W, 3] ready for RadianceVideoHDRDecode,
    RadianceVideoFrameRouter, or any standard ComfyUI image node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
            },
            "optional": {
                "dit_config": ("STRING", {"default": "{}"}),
                "tile_decode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Tile the VAE decode to reduce VRAM on large videos",
                }),
                "tile_overlap": ("INT", {
                    "default": 64, "min": 0, "max": 256,
                    "tooltip": "Pixel overlap between tiles (higher = smoother seams)",
                }),
                "output_linear": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip gamma correction — output linear-light frames for HDR pipeline",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "decode_report")
    FUNCTION = "decode"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Batch-decode video latents to pixel frames with memory management."

    def decode(self, vae, latent, dit_config="{}", tile_decode=False,
               tile_overlap=64, output_linear=False):

        report = [f"=== RadianceVideoBatchDecode v{__version__} ==="]

        try:
            cfg_dict = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
        except Exception:
            cfg_dict = {}

        samples = _to_tensor(latent)
        report.append(f"Latent shape : {list(samples.shape)}")

        # Apply latent scale correction
        scale = cfg_dict.get("latent_scale", 1.0)
        if abs(scale - 1.0) > 1e-5 and HAS_TORCH:
            samples = samples / scale
            report.append(f"Scale correct: ÷{scale}")

        # Enable tile mode on VAE if requested
        if tile_decode and vae is not None:
            for method in ("enable_tiling", "set_tiling"):
                if hasattr(vae, method):
                    try:
                        getattr(vae, method)(True)
                        report.append("Tile decode: enabled")
                        break
                    except Exception as exc:
                        logger.warning("[nodes_t2v_pipeline] decode: %s", exc)

        # Handle 5-D temporal latents: decode per-frame or as batch
        if HAS_TORCH and samples.dim() == 5:
            B, C, T, H, W = samples.shape
            # Flatten temporal into batch: [B*T, C, H, W]
            samples_flat = samples.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            frames = _vae_decode(vae, _wrap(samples_flat))
            report.append(f"Decoded {T} frames from 5D latent")
        else:
            frames = _vae_decode(vae, _wrap(samples))

        if HAS_TORCH and isinstance(frames, torch.Tensor):
            if frames.dim() == 3:
                frames = frames.unsqueeze(0)
            if not output_linear:
                frames = frames.clamp(0, 1)
            n = frames.shape[0]
            report.append(f"Output : {n} frames  {list(frames.shape[1:])}")
        else:
            n = 0
            report.append("Decode returned non-tensor")

        return (frames, n, "\n".join(report))


# ===========================================================================
# Node: RadianceVideoExport
# ===========================================================================

class RadianceVideoExport:
    """
    Route a decoded video IMAGE batch to the appropriate output format.

    Modes
    -----
    passthrough  — return frames unchanged (connect to video writer nodes)
    hdr_decode   — apply PQ/HLG decode via RadianceVideoHDRDecode inline
    exr_sequence — save each frame as a 16-bit EXR to a folder
    preview_gif  — save a small GIF preview and return frame batch

    The node is designed as the final stage in the pipeline:
      T2V / I2V → VideoBatchDecode → VideoExport → (VideoWriter / EXR IO)
    """

    MODES = ["passthrough", "hdr_decode", "exr_sequence", "preview_gif"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "mode": (cls.MODES, {"default": "passthrough"}),
            },
            "optional": {
                "hdr_metadata_json": ("STRING", {
                    "default": '{"peak_nits":1000,"eotf":"PQ (ST.2084)"}',
                }),
                "output_folder": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "radiance_video"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "frame_offset": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "export_report")
    FUNCTION = "export"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Export generated video frames to a file with format and codec options."
    OUTPUT_NODE = True

    def export(self, frames, mode, hdr_metadata_json="{}",
               output_folder="", filename_prefix="radiance_video",
               fps=24.0, frame_offset=0):

        report = [f"=== RadianceVideoExport v{__version__} ===",
                  f"Mode   : {mode}",
                  f"Frames : {frames.shape[0] if HAS_TORCH else '?'}"]

        if mode == "passthrough":
            n = frames.shape[0] if HAS_TORCH else 0
            return (frames, n, "\n".join(report))

        if mode == "hdr_decode" and HAS_TORCH:
            try:
                from radiance.nodes.video.hdr import RadianceVideoHDRDecode
            except ImportError:
                RadianceVideoHDRDecode = None
            if RadianceVideoHDRDecode is not None:
                decoder = RadianceVideoHDRDecode()
                hdr_frames, sdr_frames, _ = decoder.decode(
                    frames, hdr_metadata_json, "Reinhard")
                report.append("HDR decode applied")
                return (hdr_frames, frames.shape[0], "\n".join(report))

        if mode == "exr_sequence" and output_folder:
            saved = self._save_exr_sequence(frames, output_folder,
                                              filename_prefix, frame_offset, report)
            report.append(f"Saved {saved} EXR frames to {output_folder}")

        if mode == "preview_gif" and output_folder and HAS_PIL:
            gif_path = self._save_gif(frames, output_folder, filename_prefix, fps, report)
            report.append(f"GIF saved: {gif_path}")

        n = frames.shape[0] if HAS_TORCH else 0
        return (frames, n, "\n".join(report))

    def _save_exr_sequence(self, frames, folder, prefix, offset, report):
        os.makedirs(folder, exist_ok=True)
        saved = 0
        try:
            import OpenEXR, Imath, numpy as np  # type: ignore
            for i in range(frames.shape[0]):
                arr = frames[i].cpu().float().numpy()
                h, w, _ = arr.shape
                header = OpenEXR.Header(w, h)
                header["channels"] = {
                    c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                    for c in "RGB"
                }
                out_path = os.path.join(folder, f"{prefix}_{offset+i:06d}.exr")
                exr = OpenEXR.OutputFile(out_path, header)
                exr.writePixels({
                    "R": arr[:,:,0].tobytes(),
                    "G": arr[:,:,1].tobytes(),
                    "B": arr[:,:,2].tobytes(),
                })
                exr.close()
                saved += 1
        except ImportError:
            # Fallback: save PNG
            try:
                from PIL import Image as PILImage
                import numpy as np
                for i in range(frames.shape[0]):
                    arr = (frames[i].cpu().float().clamp(0,1).numpy() * 65535).astype("uint16")
                    img = PILImage.fromarray(arr[:,:,:3].astype("uint16"), mode="I;16")
                    out_path = os.path.join(folder, f"{prefix}_{offset+i:06d}.png")
                    img.save(out_path)
                    saved += 1
            except Exception as exc:
                report.append(f"EXR/PNG save failed: {exc}")
        return saved

    def _save_gif(self, frames, folder, prefix, fps, report):
        os.makedirs(folder, exist_ok=True)
        from PIL import Image as PILImage
        import numpy as np
        duration_ms = int(1000 / fps)
        pil_frames = []
        scale = min(1.0, 320 / frames.shape[2])  # max 320px wide
        for i in range(frames.shape[0]):
            arr = (frames[i].cpu().clamp(0,1).float().numpy() * 255).astype("uint8")
            img = PILImage.fromarray(arr[:,:,:3])
            if scale < 1.0:
                nw = int(arr.shape[1] * scale)
                nh = int(arr.shape[0] * scale)
                img = img.resize((nw, nh), PILImage.LANCZOS)
            pil_frames.append(img.convert("P", palette=PILImage.ADAPTIVE))
        gif_path = os.path.join(folder, f"{prefix}_preview.gif")
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           loop=0, duration=duration_ms)
        return gif_path


# ALBABIT-FIX: NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS removed.
# Live registry is nodes/video/__init__.py — this file's mappings were never
# read by ComfyUI and had drifted (stale display names, missing 4 HDR nodes).
