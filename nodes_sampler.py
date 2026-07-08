import torch
import time
import math
import logging
import gc
import errno
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
try:
    from .tensor_contract import ensure_4d, ensure_5d
except (ImportError, ValueError):
    from tensor_contract import ensure_4d, ensure_5d


try:
    from comfy.nested_tensor import NestedTensor as _NestedTensor
    _HAS_NESTED_TENSOR = True
except ImportError:
    _NestedTensor = None
    _HAS_NESTED_TENSOR = False

try:
    from .sampler_utils import (
        DYNAMIC_GUIDANCE_EARLY_MULTIPLIER, DYNAMIC_GUIDANCE_LATE_MULTIPLIER, 
        DYNAMIC_GUIDANCE_EARLY_THRESHOLD, DYNAMIC_GUIDANCE_LATE_THRESHOLD, 
        DYNAMIC_GUIDANCE_RAMP_WIDTH, GUIDANCE_RESCALE_PHI, 
        SIGMA_DISCONTINUITY_THRESHOLD, PAG_DEFAULT_SCALE, PAG_LAYER_NAMES, 
        CFG_PLUS_PLUS_DEFAULT_SCALE, CFG_GUIDANCE_MODELS, 
        DYNAMIC_CFG_EARLY_MULTIPLIER, DYNAMIC_CFG_LATE_MULTIPLIER, 
        DYNAMIC_CFG_EARLY_THRESHOLD, DYNAMIC_CFG_LATE_THRESHOLD, MODEL_TYPES, 
        VIDEO_MODEL_TYPES, GUIDANCE_EMBED_MODELS, CFG_GUIDED_MODELS, 
        PREVIEW_METHODS, NOISE_TYPES, CLIP_TARGETS,
        TILE_BLEND_MODES, SamplerMode, SigmaCache, _sigma_cache,
        RadianceModelRegistry, detect_by_config, detect_by_architecture,
        detect_by_sampling, detect_model_type, get_model_defaults,
        parse_model_meta, refine_distillation_from_meta,
        gradual_sigma_blend, log_tensor, SigmaIndexer, SamplingStage,
        apply_flux_guidance, compute_dynamic_guidance, compute_dynamic_cfg,
        compute_base_sigmas, WORKFLOW_PRESETS, flux_shift_sigmas, get_flux_sigmas,
        validate_step_range, apply_pag_to_model, AYS_ANCHORS, get_ays_sigmas,
        guidance_rescale_cfg, correct_sigma_end, apply_cfg_plus_plus,
        build_sigma_report, _temporally_correlate, _perlin_noise, _perlin_noise_2d,
        _spectral_noise, _get_freq_grid, _spectral_noise_2d, _brownian_noise,
        _simplex_noise, _voronoi_noise, _curl_noise, generate_noise,
        route_conditioning, tile_sample,
        MODEL_DEFAULTS,
    )
except (ImportError, ValueError):
    from sampler_utils import (
        DYNAMIC_GUIDANCE_EARLY_MULTIPLIER, DYNAMIC_GUIDANCE_LATE_MULTIPLIER,
        DYNAMIC_GUIDANCE_EARLY_THRESHOLD, DYNAMIC_GUIDANCE_LATE_THRESHOLD,
        DYNAMIC_GUIDANCE_RAMP_WIDTH, GUIDANCE_RESCALE_PHI,
        SIGMA_DISCONTINUITY_THRESHOLD, PAG_DEFAULT_SCALE, PAG_LAYER_NAMES,
        CFG_PLUS_PLUS_DEFAULT_SCALE, CFG_GUIDANCE_MODELS,
        DYNAMIC_CFG_EARLY_MULTIPLIER, DYNAMIC_CFG_LATE_MULTIPLIER,
        DYNAMIC_CFG_EARLY_THRESHOLD, DYNAMIC_CFG_LATE_THRESHOLD, MODEL_TYPES,
        VIDEO_MODEL_TYPES, GUIDANCE_EMBED_MODELS, CFG_GUIDED_MODELS,
        PREVIEW_METHODS, NOISE_TYPES, CLIP_TARGETS,
        TILE_BLEND_MODES, SamplerMode, SigmaCache, _sigma_cache,
        RadianceModelRegistry, detect_by_config, detect_by_architecture,
        detect_by_sampling, detect_model_type, get_model_defaults,
        parse_model_meta, refine_distillation_from_meta,
        gradual_sigma_blend, log_tensor, SigmaIndexer, SamplingStage,
        apply_flux_guidance, compute_dynamic_guidance, compute_dynamic_cfg,
        compute_base_sigmas, WORKFLOW_PRESETS, flux_shift_sigmas, get_flux_sigmas,
        validate_step_range, apply_pag_to_model, AYS_ANCHORS, get_ays_sigmas,
        guidance_rescale_cfg, correct_sigma_end, apply_cfg_plus_plus,
        build_sigma_report, _temporally_correlate, _perlin_noise, _perlin_noise_2d,
        _spectral_noise, _get_freq_grid, _spectral_noise_2d, _brownian_noise,
        _simplex_noise, _voronoi_noise, _curl_noise, generate_noise,
        route_conditioning, tile_sample,
        MODEL_DEFAULTS,
    )

logger = logging.getLogger("radiance.sampler")


def _is_invalid_progress_stream_error(exc: BaseException) -> bool:
    """Detect Windows/tqdm/wandb progress-stream failures without hiding model errors."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError):
            err_no = getattr(current, "errno", None)
            msg = str(current)
            if err_no == errno.EINVAL or "Errno 22" in msg or "Invalid argument" in msg:
                return True
        current = current.__cause__ or current.__context__
    return False


def _sample_custom_progress_safe(context: str, **kwargs):
    """Run Comfy sampling, retrying once with progress output disabled if stderr is broken."""

    try:
        return comfy.sample.sample_custom(**kwargs)
    except Exception as exc:
        if not _is_invalid_progress_stream_error(exc) or kwargs.get("disable_pbar"):
            raise

        retry_kwargs = dict(kwargs)
        retry_kwargs["disable_pbar"] = True
        retry_kwargs["callback"] = None
        logger.warning(
            "[%s] Comfy progress output failed with OSError(22). "
            "Retrying with progress display disabled; sampling settings are unchanged.",
            context,
        )
        return comfy.sample.sample_custom(**retry_kwargs)

class RadianceSamplerPro:
    """
    Universal diffusion sampler (v3.0.0) — Flux, SD3, SDXL, WAN, LTX,
    HunyuanVideo, Lumina2, Chroma.  Supports phase-shift sampling, AYS
    schedules, PAG, tiled sampling, multi-conditioning, noise types,
    refiner chaining, and restart (IRES) sampling.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Universal diffusion sampler with phase-shift, PAG, AYS, tiling, restart, and multi-model support."

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "preset": (WORKFLOW_PRESETS, {"default": "None"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Total denoising steps. More steps = higher quality but slower. 20–30 is typical for most samplers."
                }),
                "start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Start step (0 = beginning)",
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "End step (0 = use total steps)",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "sampler_mode": (SamplerMode.ALL, {"default": SamplerMode.STANDARD}),
                "phase_split": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "scheduler_mode": (
                    ["Manual", "Auto (Match Steps)"],
                    {"default": "Manual"},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "flux_shift": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.1},
                ),
                "flux_guidance": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "flux_guidance_profile": (
                    ["Static", "Dynamic (Creative Start/End)"],
                    {"default": "Static"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for reproducible results. Use the control below it (randomize / increment / fixed) to vary the seed between runs."
                }),

                "pag_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "PAG strength (0=off). Perturbs attention for better prompt adherence.",
                    },
                ),

                "model_type": (MODEL_TYPES, {"default": "auto"}),
                "sigma_blend_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Smooth sigma transition steps at phase-shift boundary",
                    },
                ),

                "guidance_rescale_phi": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Guidance rescale (Imagen). 0=off, 0.7=recommended for SDXL. Prevents oversaturation at high CFG.",
                    },
                ),

                "preview_method": (PREVIEW_METHODS, {"default": "None"}),

                "noise_type": (
                    NOISE_TYPES,
                    {"default": "Gaussian",
                     "tooltip": ("Noise generation algorithm. Perlin=coherent structure, "
                                 "Spectral=pink/1f noise, Brownian=video-correlated, Uniform=flat distribution.")},
                ),

                "conditioning_clip_target": (
                    CLIP_TARGETS,
                    {"default": "Auto",
                     "tooltip": "Route conditioning to a specific encoder slot (clip_l, clip_g, t5xxl). Auto = no routing."},
                ),

                "add_noise": ("BOOLEAN", {"default": True,
                    "tooltip": "Inject fresh noise at the start of sampling. Disable for img2img-style passes that should preserve structure."
                }),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False,
                    "tooltip": "Return the latent with residual noise un-removed. Useful for multi-pass workflows."
                }),
                "ays_schedule": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use AYS (Align Your Steps) research-optimized sigma schedule. Best at 8-15 steps.",
                    },
                ),

                "tile_mode": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Enable tiled sampling for memory-efficient high-resolution generation."},
                ),
                "tile_size": (
                    "INT",
                    {"default": 128, "min": 32, "max": 1024, "step": 32,
                     "tooltip": "Tile size in latent pixels (128 latent ≈ 1024px output with VAE factor 8)."},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 16, "min": 0, "max": 256, "step": 8,
                     "tooltip": "Overlap between adjacent tiles to reduce seam artifacts."},
                ),
                "tile_blend": (
                    TILE_BLEND_MODES,
                    {"default": "feather",
                     "tooltip": "Seam blending method. feather=cosine fade, gaussian=bell curve, average=uniform."},
                ),

                "terminal_sigma_to_zero": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Ensure terminal step reaches zero noise even on truncated image-to-image runs. Vital for Flow Matching models."},
                ),
                "force_exact_steps": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Ensure precise step count in image-to-image runs, adjusting calculations rather than purely truncating stages."},
                ),

                # ALBABIT-FIX: Removed UI normalization factors entirely. Native LTX handles recombination cleanly.
            },
            "optional": {
                "refiner_model": ("MODEL",),
                "refiner_start_step": (
                    "INT",
                    {"default": 20, "min": 0, "max": 200, "step": 1},
                ),
                "noise_override": ("LATENT",),

                "sigmas_override": (
                    "SIGMAS",
                    {"tooltip": "Inject a pre-computed sigma schedule. Bypasses all internal sigma computation."},
                ),

                # ── Workflow-compat absorbers (positions 38/39/40 in old saves) ─
                # Old workflow JSONs serialised two JS button widgets (null, null)
                # and the preset_info text widget at these positions. STRING type
                # absorbs null/any-string without a validation error, so the new
                # feature inputs below land at 41/42/43 where they are absent
                # from old JSON → ComfyUI falls back to their defaults.
                "_js_export_btn": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "JS serialization placeholder (not user-editable)."}),
                "_js_import_btn": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "JS serialization placeholder (not user-editable)."}),
                "_js_preset_info": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "JS serialization placeholder (not user-editable)."}),

                # ── Restart sampling ─────────────────────────────────────────
                "restart_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4,
                        "step": 1,
                        "tooltip": (
                            "Number of restart iterations at each restart_schedule sigma. "
                            "0 = disabled. 1–2 restarts add ~5% extra steps but measurably "
                            "improve high-frequency detail on Flux and WAN."
                        ),
                    },
                ),

                # ── Noise alpha schedule ──────────────────────────────────────
                "noise_alpha_start": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Blend weight of the selected noise_type at step 0. "
                            "1.0 = pure noise_type. Cosine-interpolates to noise_alpha_end "
                            "across the denoising trajectory. Set <1 to blend structured "
                            "noise with Gaussian (e.g. 0.8 Perlin → 0.0 Gaussian for video)."
                        ),
                    },
                ),
                "noise_alpha_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Blend weight of the selected noise_type at the final step. "
                            "Set lower than noise_alpha_start to fade structured noise "
                            "into pure Gaussian in late denoising steps."
                        ),
                    },
                ),

                # ── Custom AYS anchors ────────────────────────────────────────
                "custom_ays_anchors": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional model-specific AYS anchor schedule. "
                            "Overrides the built-in AYS tables when ays_schedule=True. "
                            "Must be a monotonically decreasing tensor ending at 0."
                        ),
                    },
                ),

                # ── Restart sigma schedule (socket-only — link a SIGMAS node) ──
                "restart_schedule": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional list of sigma levels at which to re-inject noise "
                            "and re-denoise (Restart / IRES style). Improves fine detail "
                            "at fixed step count. Requires restart_count > 0."
                        ),
                    },
                ),
                # ── SDR conditioning ─────────────────────────
                "sdr_reference": ("IMAGE",),
                "sdr_vae": ("VAE",),
                "sdr_blend": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "sdr_inject_steps": (
                    "INT",
                    {"default": 6, "min": 0, "max": 100, "step": 1},
                ),
                "sdr_decay": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                # ── Model-aware auto defaults ─────────────────────────────────
                "model_meta": (
                    "STRING",
                    {
                        "default": "", "forceInput": True,
                        "tooltip": "Optional: connect RadianceUnifiedLoader's model_meta output. "
                                   "Only used when preset='None'/'Custom' and model_type='auto'. "
                                   "Refines cfg/guidance/steps beyond what the loaded model's "
                                   "architecture alone can tell -- e.g. distinguishing Flux.2 Klein "
                                   "Base from Klein distilled, which are architecturally identical.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "SIGMAS", "SIGMAS", "IMAGE")
    RETURN_NAMES = ("latent", "sigmas", "sigmas_remaining", "sigma_plot")
    OUTPUT_TOOLTIPS = (
        "Denoised latent ready for VAE decode.",
        "The full sigma schedule used — chain to another sampler or inspect.",
        "Unused tail of the sigma schedule after end_step. Chain directly to a refiner or upscaler sampler.",
        "Visual plot of the sigma schedule as an IMAGE. Blank if matplotlib is unavailable.",
    )
    FUNCTION = "sample"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = (
        "v3.0.0 — Universal diffusion sampler. Auto-detects model type (Flux, SD3, SDXL, "
        "WAN, LTX, HunyuanVideo, Lumina2, Chroma). Phase-shift sampling, AYS schedules, "
        "PAG, dynamic guidance, tiled sampling, multi-conditioning, noise types, refiner chain. "
        "Restart sampling (IRES style), noise alpha schedule, sigma plot output, "
        "custom AYS anchors, sigmas_remaining for multi-node chains. "
        "AVControl-style SDR reference conditioning: encode an SDR reference through the VAE "
        "and blend it into the initial latent + per-step post-CFG anchor for HDR structure preservation."
    )

    def _apply_presets(self, preset, **kwargs):
        if preset in ("None", "Custom"):
            return kwargs

        if preset not in WORKFLOW_PRESETS:
            logger.warning(f"[Radiance] Preset '{preset}' not found.")
            return kwargs

        # ALBABIT-FIX: presets no longer overwrite UI widget values (parity
        # with old Radiance "relying strictly on UI parameters"). The JS
        # applyPreset fills the widgets when a preset is selected and flags
        # any user-edited widget with a divergence marker. Forcing preset
        # values here silently overrode user edits (e.g. cfg 1.0 -> 3.0 on
        # the LTX HighRes pass: negative prompt evaluated -> 2 forward
        # passes per step instead of 1 -> x2.5 slower + different render).
        logger.info(f"[Radiance] Preset '{preset}' active — relying strictly on UI parameters.")
        return kwargs

    @staticmethod
    def _resolved_values_ui(cfg, flux_guidance, flux_shift, sampler, steps):
        # ALBABIT-FIX: cfg/flux_guidance/flux_shift/sampler/steps can be
        # silently adjusted by _configure_model_and_defaults() (MODEL_DEFAULTS
        # auto-adapt, or model_meta-driven Flux.2 Klein refinement) without the
        # widget on screen changing -- mirrors RadianceResolution's
        # computed_width/height "ui" write-back so onExecuted (radiance_sampler.js)
        # can sync the widgets to what was actually used, post-run.
        return {
            "resolved_cfg": [cfg], "resolved_flux_guidance": [flux_guidance],
            "resolved_flux_shift": [flux_shift], "resolved_sampler": [sampler],
            "resolved_steps": [steps],
        }

    def _configure_model_and_defaults(self, model, model_type, preset, model_meta="", **kwargs):
        detected_type = detect_model_type(model) if model_type == "auto" else model_type

        # model_meta's arch is the Loader's own (already-correct) resolution --
        # prefer it over re-detecting from the loaded MODEL object, which can't
        # distinguish Flux.2 Dev from Flux.2 Klein either. Only applies in auto
        # mode, never overriding an explicit manual model_type.
        meta_arch, meta_unet_file = parse_model_meta(model_meta)
        if model_type == "auto" and meta_arch:
            detected_type = meta_arch

        # ALBABIT-FIX: "and model_type == auto" used to also gate this block --
        # redundant on top of the per-field "still at generic default" checks
        # below, which already protect any field the user set by hand,
        # regardless of how model_type itself was determined.
        if preset in ("None", "Custom"):
            defaults = get_model_defaults(detected_type)
            distilled = refine_distillation_from_meta(detected_type, meta_unet_file)

            # Apply defaults if user has them at "default" values
            if kwargs.get('cfg') == 1.0 and detected_type != "flux":
                kwargs['cfg'] = defaults.get("cfg", kwargs['cfg'])

            # ALBABIT-FIX: "detected_type != flux" used to gate this whole block
            # off for Flux.1 -- harmless when guidance always matched (3.5 ==
            # the widget's own generic default), but it silently blocked the
            # Schnell override (0.0) once distillation_refined started covering
            # Flux.1 too. Removed -- a plain "flux" with no override still
            # resolves guidance=3.5, an unchanged no-op.
            if kwargs.get('flux_guidance') == 3.5:
                model_default_guidance = (distilled or {}).get(
                    "guidance", defaults.get("guidance", kwargs['flux_guidance']))
                if model_default_guidance > 0 or defaults.get("guidance_type") != "embedding":
                    kwargs['flux_guidance'] = model_default_guidance

            if kwargs.get('steps') == 20 and distilled:
                kwargs['steps'] = distilled["steps"]
                logger.info(f"Auto-applied steps={kwargs['steps']} for {detected_type} (from model_meta)")

            default_shift = defaults.get("shift", 1.0)
            if kwargs.get('flux_shift') == 1.0 and default_shift != 1.0:
                kwargs['flux_shift'] = default_shift
                logger.info(f"Auto-applied shift={kwargs['flux_shift']} for {detected_type}")

            default_sampler = defaults.get("sampler", kwargs['sampler'])
            if kwargs.get('sampler') == "euler" and default_sampler != "euler":
                kwargs['sampler'] = default_sampler
                logger.info(f"Auto-applied sampler={kwargs['sampler']} for {detected_type}")

        # Auto-match scheduler
        if kwargs.get('scheduler_mode') == "Auto (Match Steps)":
            defaults = get_model_defaults(detected_type)
            auto_scheduler = defaults.get("scheduler", kwargs['scheduler'])
            if auto_scheduler != kwargs['scheduler']:
                logger.info(f"Auto scheduler: {kwargs['scheduler']} → {auto_scheduler}")
                kwargs['scheduler'] = auto_scheduler

        return detected_type, kwargs

    def _prepare_noise(self, latent_samples, seed, noise_type, noise_override, device, frames):
        if noise_override is not None:
            noise = noise_override["samples"]
            if noise.shape != latent_samples.shape:
                raise ValueError(f"noise_override shape {noise.shape} mismatch with latent {latent_samples.shape}")
        else:
            if noise_type == "Gaussian":
                noise = comfy.sample.prepare_noise(latent_samples, seed, None)
            else:
                noise = generate_noise(latent_samples, seed, noise_type, frames=frames)
        
        return noise.to(device)

    def _prepare_sigmas(self, model, detected_type, steps, denoise, flux_shift,
                        scheduler, ays_schedule, terminal_sigma_to_zero,
                        force_exact_steps, sigmas_override, device,
                        custom_ays_anchors=None):
        if sigmas_override is not None:
            # ALBABIT-FIX: restore informational log when sigmas_override is active
            logger.info(
                "[Radiance] sigmas_override active — bypassing internal sigma computation. "
                "steps / denoise / scheduler / flux_shift / AYS and related settings are ignored."
            )
            sigmas = sigmas_override.to(device)
            sigmas = correct_sigma_end(sigmas)
            target_steps = max(1, len(sigmas) - 1)
            return sigmas, target_steps

        try:
            if ays_schedule:
                # Custom AYS anchors take priority over the built-in tables
                if custom_ays_anchors is not None:
                    sigmas = custom_ays_anchors.to(device)
                    logger.info(f"Using custom AYS anchors ({len(sigmas)} values)")
                else:
                    sigmas = get_ays_sigmas(detected_type, steps)

                if sigmas is not None:
                    if denoise < 1.0:
                        total_s = len(sigmas) - 1
                        if total_s > 0:
                            start_s = max(0, int(total_s * (1.0 - denoise)))
                            sigmas = sigmas[start_s:]
                    if custom_ays_anchors is None:
                        logger.info(f"Using AYS schedule for {detected_type} ({steps} steps)")
                else:
                    sigmas = get_flux_sigmas(
                        model, scheduler, steps, denoise, flux_shift,
                        force_full=terminal_sigma_to_zero, force_exact=force_exact_steps,
                    )
            else:
                sigmas = get_flux_sigmas(
                    model, scheduler, steps, denoise, flux_shift,
                    force_full=terminal_sigma_to_zero, force_exact=force_exact_steps,
                )

            sigmas = correct_sigma_end(sigmas)
            return sigmas, steps
        except ValueError as e:
            logger.error(f"Failed to calculate sigmas: {e}")
            raise

    # ── Sigma plot ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sigma_plot(sigmas: torch.Tensor, detected_type: str) -> torch.Tensor:
        """Render sigma schedule as a small (256×128) IMAGE tensor (B,H,W,3).
        Falls back to a blank IMAGE if matplotlib is unavailable."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io
            import numpy as np

            vals = sigmas.cpu().float().numpy()
            steps = list(range(len(vals)))

            fig, ax = plt.subplots(figsize=(3.2, 1.6), dpi=80)
            ax.plot(steps, vals, color="#3266ad", linewidth=1.5)
            ax.fill_between(steps, vals, alpha=0.12, color="#3266ad")
            ax.set_xlabel("step", fontsize=7)
            ax.set_ylabel("sigma", fontsize=7)
            ax.set_title(f"sigma schedule  [{detected_type}]", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.tight_layout(pad=0.4)

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)

            from PIL import Image as PILImage
            img = PILImage.open(buf).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)
        except Exception as e:
            logger.debug(f"sigma_plot: matplotlib unavailable or failed ({e}) — returning blank")
            return torch.zeros(1, 64, 128, 3)

    # ── Restart sampling ──────────────────────────────────────────────────────

    @staticmethod
    def _apply_restarts(
        latent: torch.Tensor,
        model,
        positive,
        negative,
        sampler_obj,
        sigmas: torch.Tensor,
        restart_schedule: torch.Tensor,
        restart_count: int,
        cfg: float,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        IRES / Restart sampling: at each sigma level in restart_schedule,
        re-inject noise scaled to that sigma and run restart_count extra
        denoising steps from that level down to the next sigma in the
        main schedule.
        """
        if restart_count <= 0 or len(restart_schedule) == 0 or len(sigmas) < 2:
            return latent

        result = latent
        s_vals = sigmas.to(device)

        for r_sigma in restart_schedule.to(device):
            r_val = float(r_sigma)
            if r_val <= 0.0:
                continue

            # Find where this sigma sits in the main schedule
            diffs = (s_vals - r_val).abs()
            idx = int(diffs.argmin().item())
            if idx >= len(s_vals) - 1:
                continue

            # Subsection of schedule from r_sigma down to the next waypoint
            sub_sigmas = s_vals[idx:min(idx + restart_count + 2, len(s_vals))]
            if len(sub_sigmas) < 2:
                continue

            for _ in range(restart_count):
                # Re-inject noise at r_sigma level
                noise = torch.randn_like(result) * r_val
                noisy = result + noise
                try:
                    result = _sample_custom_progress_safe(
                        "RadianceSamplerPro restart",
                        model=model,
                        noise=noisy,
                        cfg=cfg,
                        sampler=sampler_obj,
                        sigmas=sub_sigmas,
                        positive=positive,
                        negative=negative,
                        latent_image=result,
                        noise_mask=None,
                        callback=None,
                        disable_pbar=True,
                        seed=seed,
                    )
                except Exception as e:
                    logger.warning(f"Restart sampling failed at sigma={r_val:.4f}: {e}")
                    break

        return result

    # ── AVControl — SDR Reference Conditioning ───────────────────────────────

    def _encode_sdr_reference(self, ref, vae, work):
        res = vae.encode(ref)
        ref_latent = res["samples"]
        B = work.shape[0]
        if ref_latent.shape[0] == 1 and B > 1:
            ref_latent = ref_latent.expand(B, -1, -1, -1)
            
        target_H, target_W = work.shape[-2], work.shape[-1]
        if ref_latent.shape[-2] != target_H or ref_latent.shape[-1] != target_W:
            import torch.nn.functional as F
            ref_latent = F.interpolate(ref_latent, size=(target_H, target_W), mode="nearest")
            
        target_C = work.shape[1]
        current_C = ref_latent.shape[1]
        if current_C > target_C:
            ref_latent = ref_latent[:, :target_C, :, :]
        elif current_C < target_C:
            import torch
            pad = torch.zeros(ref_latent.shape[0], target_C - current_C, ref_latent.shape[2], ref_latent.shape[3],
                              device=ref_latent.device, dtype=ref_latent.dtype)
            ref_latent = torch.cat([ref_latent, pad], dim=1)
            
        if work.ndim == 5:
            T = work.shape[2]
            ref_latent = ref_latent.unsqueeze(2).expand(-1, -1, T, -1, -1)
            
        return ref_latent.to(dtype=work.dtype, device=work.device)



    def sample(
        self,
        model,
        positive: List,
        negative: List,
        latent_image: Dict[str, torch.Tensor],
        preset: str,
        steps: int,
        start_step: int,
        end_step: int,
        cfg: float,
        sampler: str,
        sampler_mode: str,
        phase_split: float,
        scheduler: str,
        scheduler_mode: str,
        denoise: float,
        flux_shift: float,
        flux_guidance: float,
        flux_guidance_profile: str,
        add_noise: bool,
        return_with_leftover_noise: bool,
        seed: int,
        pag_scale: float = 0.0,
        model_type: str = "auto",
        sigma_blend_steps: int = 0,
        ays_schedule: bool = False,
        guidance_rescale_phi: float = 0.0,
        preview_method: str = "None",
        noise_type: str = "Gaussian",
        conditioning_clip_target: str = "Auto",
        tile_mode: bool = False,
        tile_size: int = 128,
        tile_overlap: int = 16,
        tile_blend: str = "feather",
        refiner_model=None,
        refiner_start_step: int = 20,
        noise_override: Optional[Dict[str, torch.Tensor]] = None,
        sigmas_override: Optional[torch.Tensor] = None,
        latent_format: str = "",
        terminal_sigma_to_zero: bool = False,
        force_exact_steps: bool = False,
        # ── Workflow-compat absorbers (swallow old JS-serialised widget values) ─
        _js_export_btn=None,
        _js_import_btn=None,
        _js_preset_info=None,
        # ── Custom AYS anchors, restart sampling, noise alpha schedule ─────────
        custom_ays_anchors: Optional[torch.Tensor] = None,
        restart_schedule: Optional[torch.Tensor] = None,
        restart_count: int = 0,
        noise_alpha_start: float = 1.0,
        noise_alpha_end: float = 1.0,
        sdr_reference: Optional[torch.Tensor] = None,
        sdr_vae: Optional[Any] = None,
        sdr_blend: float = 0.0,
        sdr_inject_steps: int = 0,
        sdr_decay: float = 0.65,
        model_meta: str = "",
    ) -> Tuple:

        t_start = time.time()
        timings: Dict[str, float] = {}

        # 1. Apply Presets
        params = self._apply_presets(preset, 
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, 
            denoise=denoise, flux_shift=flux_shift, flux_guidance=flux_guidance,
            sampler_mode=sampler_mode, ays_schedule=ays_schedule, model_type=model_type)
        
        # Unpack params
        steps, cfg, sampler, scheduler = params['steps'], params['cfg'], params['sampler'], params['scheduler']
        denoise, flux_shift, flux_guidance = params['denoise'], params['flux_shift'], params['flux_guidance']
        sampler_mode, ays_schedule, model_type = params['sampler_mode'], params['ays_schedule'], params['model_type']

        # 2. Model Detection & Default Calibration
        detected_type, params = self._configure_model_and_defaults(model, model_type, preset,
            model_meta=model_meta, cfg=cfg, flux_guidance=flux_guidance, flux_shift=flux_shift,
            sampler=sampler, scheduler=scheduler, scheduler_mode=scheduler_mode, steps=steps)

        cfg, flux_guidance, flux_shift = params['cfg'], params['flux_guidance'], params['flux_shift']
        sampler, scheduler, steps = params['sampler'], params['scheduler'], params['steps']

        # 3. Latent Preparation
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            _got = type(latent_image).__name__
            raise RuntimeError(
                "This node needs a LATENT, but the input is an IMAGE.\n\n"
                "Make sure you connect a VAE Encode or a Sampler node before this one."
            )
        t0 = time.time()
        latent_samples = latent_image["samples"]
        noise_mask = latent_image.get("noise_mask")  # Optional inpainting mask
        
        # Standardize to 5D for processing if it's a video model, otherwise keep 4D
        is_video = detected_type in VIDEO_MODEL_TYPES
        if is_video:
            latent_samples = ensure_5d(latent_samples, "RadianceSamplerPro")
            frames = latent_samples.shape[2]
            logger.warning(
                "[RadianceSamplerPro] Video model '%s' detected. "
                "For native video-model parity, especially LTX/LTX 2.3, prefer "
                "RadianceVideoSampler. SamplerPro remains a universal/experimental "
                "sampler and may differ because guidance, noise, callbacks, and "
                "staged sampling are still Radiance-controlled.",
                detected_type,
            )
            if sigmas_override is not None:
                logger.warning(
                    "[RadianceSamplerPro] sigmas_override only replaces the sigma "
                    "schedule. It does not make SamplerPro identical to the native "
                    "video sampler path."
                )
        else:
            latent_samples = ensure_4d(latent_samples, "RadianceSamplerPro")
            frames = None
        timings["latent_prep"] = time.time() - t0

        device = comfy.model_management.get_torch_device()
        work_latent = latent_samples.to(device)

        # Blend SDR reference into initial latent if provided
        sdr_latent = None
        if sdr_reference is not None and sdr_vae is not None and sdr_blend > 0.0:
            sdr_latent = self._encode_sdr_reference(sdr_reference, sdr_vae, work_latent)
            work_latent = (1.0 - sdr_blend) * work_latent + sdr_blend * sdr_latent
            logger.info(f"[SDR Conditioning] Blended SDR reference into initial latent with strength {sdr_blend:.2f}")

        # 4. Noise & Sigmas Preparation
        t0 = time.time()
        noise = self._prepare_noise(latent_samples, seed, noise_type, noise_override, device, frames)

        # FEAT-NOISE-ALPHA: blend structured noise with Gaussian across steps.
        # noise_alpha=1.0 = pure noise_type; 0.0 = pure Gaussian.
        # When both ends differ, blending is deferred per-step via a stored ramp.
        # For noise injection we apply the start-alpha to the initial noise tensor.
        _noise_alpha_ramp_active = (
            abs(noise_alpha_start - 1.0) > 1e-4
            or abs(noise_alpha_end - 1.0) > 1e-4
            or abs(noise_alpha_start - noise_alpha_end) > 1e-4
        ) and noise_type != "Gaussian"
        if _noise_alpha_ramp_active and noise_alpha_start < 1.0 - 1e-4:
            gaussian_noise = torch.randn_like(noise)
            noise = noise * noise_alpha_start + gaussian_noise * (1.0 - noise_alpha_start)
            logger.info(
                f"Noise alpha schedule: {noise_type} × {noise_alpha_start:.2f} → "
                f"Gaussian × {noise_alpha_end:.2f}"
            )

        timings["prepare_noise"] = time.time() - t0

        t0 = time.time()
        sigmas, target_total_steps = self._prepare_sigmas(
            model, detected_type, steps, denoise,
            flux_shift, scheduler, ays_schedule, terminal_sigma_to_zero,
            force_exact_steps, sigmas_override, device,
            custom_ays_anchors=custom_ays_anchors,
        )
        timings["prepare_sigmas"] = time.time() - t0

        # 5. LTX-AV Detection (Specific logic for audio/video separation)
        is_ltx_av = False
        ltxav_obj = None
        try:
            inner_model = model.get_model_object("diffusion_model")
            if hasattr(inner_model, "separate_audio_and_video_latents") and \
               hasattr(inner_model, "recombine_audio_and_video_latents"):
                is_ltx_av = True
                ltxav_obj = inner_model
                logger.info("[Radiance] LTX-AV detected. Latents will be processed natively.")
        except Exception as exc:
            logger.warning("[nodes_sampler]: %s", exc)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_tensor("Sigmas", sigmas)

        if len(sigmas) <= 1:
            logger.warning("[Radiance] Sigma schedule is trivial, returning input unchanged")
            report = build_sigma_report(
                detected_type,
                steps,
                scheduler,
                flux_shift,
                denoise,
                sigmas,
                sampler_mode,
                [],
                [],
                0.0,
                frames=frames,
            )
            blank_plot = torch.zeros(1, 64, 128, 3)
            return {
                "ui": self._resolved_values_ui(cfg, flux_guidance, flux_shift, sampler, steps),
                "result": (latent_image.copy(), sigmas, torch.tensor([0.0]), blank_plot),
            }
        log_tensor("Work Latent (Start)", work_latent)



        if conditioning_clip_target != "Auto":
            positive = route_conditioning(positive, conditioning_clip_target)
            logger.info(f"[v3.0.0] Conditioning routed to {conditioning_clip_target}")

        primary_sampler = sampler
        secondary_sampler: Optional[str] = None

        secondary_scheduler: Optional[str] = None
        split_step = -1

        effective_end = end_step if end_step > 0 else target_total_steps
        effective_end = min(effective_end, target_total_steps)
        effective_start = min(start_step, effective_end)

        splits = {effective_start, effective_end}

        is_cfg_plus_plus = SamplerMode.is_cfg_plus_plus(sampler_mode)
        if is_cfg_plus_plus:
            logger.info(f"CFG++ (Perpendicular) mode active with base CFG {cfg}")

            if detected_type == "flux" and cfg <= 1.05:
                logger.warning(
                    f"CFG++ enabled but CFG is {cfg}. For Flux, CFG++ requires CFG > 1.0"
                )

        if pag_scale > 0:
            model = apply_pag_to_model(model, pag_scale)

        if guidance_rescale_phi > 0.0 and cfg > 1.0:
            phi = guidance_rescale_phi
            model = (
                model.clone() if pag_scale <= 0 else model
            )                                

            def guidance_rescale_patch(args):

                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                cfg_val = args["cond_scale"]

                guided = uncond + cfg_val * (cond - uncond)

                dims = list(range(1, guided.ndim))
                guided_std = guided.std(dim=dims, keepdim=True).clamp(min=1e-6)
                cond_std = cond.std(dim=dims, keepdim=True).clamp(min=1e-6)

                rescaled = guided * (cond_std / guided_std)
                return rescaled * phi + guided * (1.0 - phi)

            model.set_model_sampler_cfg_function(guidance_rescale_patch)
            logger.info(f"Guidance Rescale applied (phi={phi:.2f})")

        if sdr_latent is not None and sdr_inject_steps > 0:
            model = model.clone() if (pag_scale <= 0 and guidance_rescale_phi <= 0) else model
            _step_counter = [0]
            _sdr_ref_lat = sdr_latent.detach()
            
            # Retrieve existing cfg patch if any
            existing_cfg_fn = getattr(model, "model_sampler_cfg_function", None)

            def _sdr_post_cfg_patch(args):
                denoised = existing_cfg_fn(args) if existing_cfg_fn is not None else args["denoised"]
                step = _step_counter[0]
                _step_counter[0] += 1
                if step >= sdr_inject_steps:
                    return denoised
                blend = sdr_blend * (sdr_decay ** step)
                if blend < 1e-5:
                    return denoised
                
                ref = _sdr_ref_lat
                if hasattr(ref, "to"):
                    ref = ref.to(device=denoised.device, dtype=denoised.dtype)
                
                ref_shape = tuple(ref.shape)
                denoised_shape = tuple(denoised.shape)
                if ref_shape != denoised_shape:
                    if ref.ndim == denoised.ndim:
                        if ref_shape[0] == 1 and denoised_shape[0] > 1:
                            ref = ref.expand_as(denoised)
                        else:
                            return denoised
                    else:
                        return denoised
                return (1.0 - blend) * denoised + blend * ref

            model.set_model_sampler_cfg_function(_sdr_post_cfg_patch)
            logger.info(f"[SDR Conditioning] Registered post-CFG SDR anchor patch (steps={sdr_inject_steps}, blend={sdr_blend:.2f}, decay={sdr_decay:.2f})")

        # ── Energy-Prioritized Sampling (EPS) Detection ──────────────────────
        energy_mask = None
        energy_priority = 1.0
        if isinstance(positive, list):
            for _, cond_dict in positive:
                if isinstance(cond_dict, dict) and "radiance_energy_mask" in cond_dict:
                    energy_mask = cond_dict["radiance_energy_mask"]
                    energy_priority = cond_dict.get("radiance_energy_priority", 1.0)
                    break

        if energy_mask is not None:
            model = model.clone() if (pag_scale <= 0 and guidance_rescale_phi <= 0 and sdr_reference is None) else model
            _eps_mask = energy_mask.detach()
            _eps_priority = float(energy_priority)
            
            # Retrieve existing CFG function (e.g., guidance_rescale or base)
            existing_cfg_fn = getattr(model, "model_sampler_cfg_function", None)

            def _energy_prioritized_cfg_patch(args):
                import torch.nn.functional as F
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                cfg_val = args["cond_scale"]
                
                # Unify mask shape to (B, 1, H_l, W_l) to match the latent space
                mask = _eps_mask.to(device=cond.device, dtype=cond.dtype)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                
                # Align batch size
                B, C, H_l, W_l = cond.shape
                if mask.shape[0] != B:
                    mask = mask.expand(B, -1, -1, -1)
                
                # Align spatial resolution of the mask to match the latent space
                if mask.shape[-2] != H_l or mask.shape[-1] != W_l:
                    mask = F.interpolate(mask, size=(H_l, W_l), mode="bilinear", align_corners=False)
                
                # Apply energy-priority CFG scaling specifically in the mask regions
                eps_modifier = 1.0 + _eps_priority * mask
                
                # Compute EPS-boosted positive conditioning projection
                cond_eps = uncond + (cond - uncond) * eps_modifier
                
                # Update args dict so downstream CFG functions (e.g. rescale) inherit it
                new_args = args.copy()
                new_args["cond_denoised"] = cond_eps
                new_args["denoised"] = uncond + cfg_val * (cond_eps - uncond)
                
                if existing_cfg_fn is not None:
                    return existing_cfg_fn(new_args)
                return new_args["denoised"]

            model.set_model_sampler_cfg_function(_energy_prioritized_cfg_patch)
            logger.info(f"[Energy Guidance] Registered Energy-Prioritized Sampling (EPS) patch (priority={_eps_priority:.2f})")
        if SamplerMode.is_phase_shift(sampler_mode) and detected_type in VIDEO_MODEL_TYPES:
            logger.warning(
                f"[v3.0.0] Phase-Shift mode is not supported for video model '{detected_type}' — "
                f"sampler switching mid-schedule causes temporal discontinuities. "
                f"Falling back to Standard mode."
            )
            sampler_mode = SamplerMode.STANDARD

        if SamplerMode.is_phase_shift(sampler_mode):

            if sigma_blend_steps == 0:
                sigma_blend_steps = 3
                logger.info("Phase-Shift: auto-enabled 3 sigma blend steps (was 0)")

            if sampler_mode == SamplerMode.PHASE_SHIFT_DPM:
                secondary_sampler = "dpmpp_2m"
                secondary_scheduler = None                       

                flow_match_types = GUIDANCE_EMBED_MODELS | {"wan", "hunyuan_video"}
                if detected_type in flow_match_types:
                    logger.warning(
                        f"WARNING: DPM solvers are generally incompatible with {detected_type}'s "
                        f"Flow Matching and may produce severe noise/grain. "
                        f"Recommended to use Euler or Phase-Shift SGM."
                    )
            elif sampler_mode == SamplerMode.PHASE_SHIFT_SGM:

                secondary_sampler = sampler                     
                secondary_scheduler = "sgm_uniform"                              
            else:
                secondary_sampler = sampler

            split_step = int(target_total_steps * max(0.0, min(1.0, phase_split)))

            if effective_start < split_step < effective_end:
                splits.add(split_step)
                phase2_label = secondary_sampler or sampler
                if secondary_scheduler:
                    phase2_label = f"{phase2_label}+{secondary_scheduler}"
                logger.info(
                    f"Phase-Shift: {primary_sampler} (0-{split_step}) → "
                    f"{phase2_label} ({split_step}-{target_total_steps})"
                )

                if sigma_blend_steps > 0:
                    logger.info(f"Sigma blend: {sigma_blend_steps} steps at transition")

        if refiner_model is not None:
            refiner_step = max(0, min(refiner_start_step, target_total_steps))
            if effective_start < refiner_step < effective_end:
                splits.add(refiner_step)
                logger.info(f"Refiner starts at step {refiner_step}")

        is_dynamic = "Dynamic" in flux_guidance_profile and detected_type in GUIDANCE_EMBED_MODELS

        is_dynamic_cfg = "Dynamic" in flux_guidance_profile and detected_type in CFG_GUIDED_MODELS

        if is_dynamic:

            denoising_steps = (
                int(target_total_steps * denoise) if denoise < 1.0 else target_total_steps
            )
            denoising_start = target_total_steps - denoising_steps

            idx_20 = denoising_start + int(
                denoising_steps * DYNAMIC_GUIDANCE_EARLY_THRESHOLD
            )
            idx_90 = denoising_start + int(
                denoising_steps * DYNAMIC_GUIDANCE_LATE_THRESHOLD
            )

            if effective_start < idx_20 < effective_end:
                splits.add(idx_20)
            if effective_start < idx_90 < effective_end:
                splits.add(idx_90)
            logger.info(
                f"Dynamic Guidance Active (effective range: steps {denoising_start}-{target_total_steps}, "
                f"early={idx_20}, late={idx_90})"
            )

        elif is_dynamic_cfg:

            denoising_steps = (
                int(target_total_steps * denoise) if denoise < 1.0 else target_total_steps
            )
            denoising_start = target_total_steps - denoising_steps

            idx_15 = denoising_start + int(denoising_steps * DYNAMIC_CFG_EARLY_THRESHOLD)
            idx_85 = denoising_start + int(denoising_steps * DYNAMIC_CFG_LATE_THRESHOLD)

            if effective_start < idx_15 < effective_end:
                splits.add(idx_15)
            if effective_start < idx_85 < effective_end:
                splits.add(idx_85)
            logger.info(
                f"Dynamic CFG Active for {detected_type} (effective range: steps "
                f"{denoising_start}-{target_total_steps}, boost→{idx_15}, taper→{idx_85})"
            )

        sorted_splits = sorted(
            s for s in splits if effective_start <= s <= effective_end
        )

        pbar_ref = None
        use_custom_preview = False
        if preview_method != "None":

            if preview_method == "TAESD":
                try:

                    from comfy.taesd.taesd import TAESDDecoder              
                except (ImportError, AttributeError):
                    logger.warning(
                        "[Radiance] TAESD unavailable, fallback to Latent2RGB"
                    )
                    preview_method = "Latent2RGB"

            try:
                # ALBABIT-FIX: ProgressBar respects the actual iterations being run
                actual_iterations = len(sigmas) - 1
                pbar_ref = comfy.utils.ProgressBar(actual_iterations)
                use_custom_preview = True
                logger.debug(f"Preview callback active: {preview_method}")
            except (AttributeError, TypeError) as e:
                logger.warning(f"Failed to create preview callback: {e}")

                preview_method = "None"

        def create_phase_callback(phase_start_step):
            def callback(step, x0, x1, total_steps):
                global_step = step + phase_start_step
                if pbar_ref:
                    pbar_ref.update_absolute(global_step + 1, total_steps, (x0,))
            return callback

        current_latent = work_latent
        prev_stage_sigmas: Optional[torch.Tensor] = None
        _ltxav_mr_base = None
        _ltxav_mr_orig = None

        # RADIANCE-AUDIT v2.6.0 [IMPORTANT]: removed dead pre-seed cache
        # block. It wrote under (primary_scheduler, target_total_steps)
        # but the cache is only consulted for *secondary* schedulers via
        # compute_base_sigmas, which uses a different calc_steps key when
        # force_exact_steps is on. The write never produced a hit.

        def _get_base_sigmas(mdl, sched: str = scheduler) -> torch.Tensor:
            if sched == scheduler and mdl is model:
                return sigmas
            return compute_base_sigmas(
                mdl, sched, target_total_steps, scheduler, flux_shift, denoise, _sigma_cache,
                force_full_denoise=terminal_sigma_to_zero,
                force_exact_steps=force_exact_steps
            )

        t0 = time.time()

        stage_timings: List[Tuple[int, int, int, str, float]] = []

        planned_stages: List[SamplingStage] = []
        for si in range(len(sorted_splits) - 1):
            gs = sorted_splits[si]
            ge = sorted_splits[si + 1]
            if gs >= ge:
                continue

            stage_model = model
            if refiner_model is not None and gs >= refiner_start_step:
                stage_model = refiner_model

            stage_sampler = primary_sampler
            stage_scheduler = scheduler
            is_shifted = False
            is_blend = False

            if SamplerMode.is_phase_shift(sampler_mode) and gs >= split_step:
                is_shifted = True
                if secondary_sampler:
                    stage_sampler = secondary_sampler
                if secondary_scheduler:
                    stage_scheduler = secondary_scheduler
                if gs == split_step and sigma_blend_steps > 0:
                    is_blend = True

            planned_stages.append(
                SamplingStage(
                    index=si,
                    global_start=gs,
                    global_end=ge,
                    model=stage_model,
                    sampler_name=stage_sampler,
                    scheduler_name=stage_scheduler,
                    is_phase_shifted=is_shifted,
                    is_blend_point=is_blend,
                )
            )

        for ps in planned_stages:
            label = f"{ps.sampler_name}+{ps.scheduler_name}"
            if ps.is_blend_point:
                label += " [blend]"
            
            # ALBABIT-FIX: Display logical 1-based steps for clarity in logs instead of 0-based
            logger.info(
                f"Plan Stage {ps.index + 1}: Steps {ps.global_start + 1}→{ps.global_end} [{label}]"
            )

        try:
            if tile_mode and latent_samples.ndim == 4:
                logger.info(
                    f"[v3.0.0] Tile sampling: size={tile_size}, "
                    f"overlap={tile_overlap}, blend={tile_blend}"
                )
                t_tile = time.time()
                _tile_sampler_obj = comfy.samplers.sampler_object(primary_sampler)
                effective_cfg = cfg
                if is_cfg_plus_plus:
                    sigma_max = sigmas[0].item() if len(sigmas) > 0 else 1.0
                    effective_cfg = apply_cfg_plus_plus(cfg, sigmas[0], sigma_max)

                current_latent = tile_sample(
                    model=model,
                    noise=noise,
                    latent_samples=work_latent,
                    positive=positive,
                    negative=negative,
                    sigmas=sigmas,
                    sampler_obj=_tile_sampler_obj,
                    seed=seed,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    tile_blend=tile_blend,
                    noise_mask=noise_mask,
                    cfg=effective_cfg,
                )
                
                timings["tile_sampling"] = time.time() - t_tile
                timings["sampling"] = timings["tile_sampling"]
                logger.info(
                    f"[v3.0.0] Tile sampling done in {timings['tile_sampling']:.2f}s"
                )

            # ALBABIT-FIX: LTX-AV NestedTensor is packed to (B, 1, flat_N) by
            # KSampler.sample() before reaching _calc_cond_batch. The memory_required
            # formula computes area = B * flat_N (~16M) instead of B * true_spatial
            # (~63K), inflating the estimate 128× and causing the memory check to
            # exceed free VRAM → sequential conditioning evaluation (2 forward passes
            # per step) instead of one batched pass → ×2.5 slowdown at HighRes.
            # Patch the BaseModel instance to correct the area estimate.
            if is_ltx_av and hasattr(model, "model"):
                _ltxav_mr_base = model.model
                _ltxav_mr_orig = _ltxav_mr_base.memory_required
                def _ltxav_memory_required(input_shape, cond_shapes={}):
                    if len(input_shape) == 3 and input_shape[1] == 1 and input_shape[2] > 500_000:
                        return _ltxav_mr_orig([input_shape[0], 1, input_shape[2] // 128], cond_shapes)
                    return _ltxav_mr_orig(input_shape, cond_shapes)
                _ltxav_mr_base.memory_required = _ltxav_memory_required

            for plan_idx, stage in enumerate(planned_stages):
                if tile_mode and latent_samples.ndim == 4:
                    break
                t_stage = time.time()
                i = stage.index
                s_start = stage.global_start
                s_end = stage.global_end
                current_model = stage.model
                current_sampler = stage.sampler_name
                current_scheduler = stage.scheduler_name

                stage_positive = positive
                if is_dynamic:
                    effective_guidance = compute_dynamic_guidance(
                        flux_guidance, s_start, target_total_steps, denoise
                    )
                    stage_positive = apply_flux_guidance(positive, effective_guidance)
                    logger.debug(
                        f"Dynamic Guidance @ step {s_start}: {effective_guidance:.2f}"
                    )

                elif detected_type in GUIDANCE_EMBED_MODELS:

                    stage_positive = apply_flux_guidance(positive, flux_guidance)

                logger.info(
                    f"Stage {i+1}: Steps {s_start + 1}-{s_end} | "
                    f"Sampler: {current_sampler} | Scheduler: {current_scheduler}"
                )

                try:

                    base_sigmas = _get_base_sigmas(current_model, current_scheduler)

                    indexer = SigmaIndexer(target_total_steps, base_sigmas)
                    stage_sigmas = indexer.get_stage_sigmas(s_start, s_end)

                    if stage_sigmas is None:

                        continue

                    if (
                        stage.is_blend_point
                        and prev_stage_sigmas is not None
                    ):
                        stage_sigmas = gradual_sigma_blend(
                            prev_stage_sigmas, stage_sigmas, sigma_blend_steps
                        )

                    if len(stage_sigmas) < 2:
                        logger.warning(
                            f"Stage {i+1}: Insufficient sigmas ({len(stage_sigmas)}), skipping"
                        )
                        continue

                    is_last_stage = plan_idx == len(planned_stages) - 1
                    if (
                        return_with_leftover_noise
                        and is_last_stage
                        and len(stage_sigmas) >= 3                                   
                        and stage_sigmas[-1].item() == 0.0
                    ):
                        stage_sigmas = stage_sigmas[:-1]
                        logger.debug("Stripped terminal sigma for leftover noise")

                    sampler_obj = comfy.samplers.sampler_object(current_sampler)

                    if is_cfg_plus_plus and len(stage_sigmas) > 0:
                        sigma_max = (
                            base_sigmas[0].item() if len(base_sigmas) > 0 else 1.0
                        )
                        effective_cfg = apply_cfg_plus_plus(
                            cfg, stage_sigmas[0], sigma_max
                        )
                        logger.debug(
                            f"CFG++: {cfg:.2f} → {effective_cfg:.2f} at sigma {stage_sigmas[0].item():.4f}"
                        )
                    elif is_dynamic_cfg:

                        effective_cfg = compute_dynamic_cfg(
                            cfg, s_start, target_total_steps, denoise
                        )
                        logger.debug(
                            f"Dynamic CFG @ step {s_start}: {cfg:.2f} → {effective_cfg:.2f}"
                        )
                    else:
                        effective_cfg = cfg

                    stage_latent = current_latent

                    is_first_stage = plan_idx == 0
                    if is_first_stage and add_noise:
                        stage_noise = noise
                        logger.debug("Stage 1: sampler will add initial noise")
                    else:
                        stage_noise = torch.zeros_like(noise)
                        if is_first_stage:
                            logger.debug("Stage 1: add_noise=False, no noise")
                        else:
                            logger.debug(f"Stage {i+1}: continuation, no extra noise")

                    # RADIANCE-AUDIT v2.6.0 [IMPORTANT]: this block used to
                    # require noise_type != "gaussian", so a Gaussian
                    # NestedTensor multi-stage path fed a plain Tensor as
                    # stage_noise against a NestedTensor latent and
                    # crashed inside sample_custom. Rebuild whenever the
                    # latent is a NestedTensor, regardless of noise_type.
                    if (
                        _HAS_NESTED_TENSOR
                        and isinstance(current_latent, _NestedTensor)
                    ):
                        try:
                            new_noises = []
                            for idx, sub_t in enumerate(current_latent.tensors):
                                if is_first_stage and add_noise:
                                    # ALBABIT-FIX: use pre-seeded noise sub-tensor from
                                    # _prepare_noise instead of torch.randn_like, which
                                    # ignores the seed and produces non-deterministic noise
                                    # mismatched from native RandomNoise — root cause of
                                    # black bars in LTX-AV NestedTensor inference.
                                    if (
                                        isinstance(noise, _NestedTensor)
                                        and idx < len(noise.tensors)
                                    ):
                                        sub_noise = noise.tensors[idx]
                                    elif noise_type.lower() == "gaussian":
                                        sub_noise = torch.randn_like(sub_t)
                                    else:
                                        sub_noise = generate_noise(
                                            sub_t, seed, noise_type, frames=frames
                                        )
                                else:
                                    sub_noise = torch.zeros_like(sub_t)
                                new_noises.append(sub_noise)
                            stage_noise = _NestedTensor(tuple(new_noises))
                        except Exception as _ne:
                            logger.warning(
                                f"[Radiance] LTX-AV NestedTensor noise rebuild failed: {_ne}. "
                            )

                    comfy.model_management.load_model_gpu(current_model)

                    result = _sample_custom_progress_safe(
                        f"RadianceSamplerPro stage {i+1}",
                        model=current_model,
                        noise=stage_noise,
                        cfg=effective_cfg,
                        sampler=sampler_obj,
                        sigmas=stage_sigmas,
                        positive=stage_positive,
                        negative=negative,
                        latent_image=stage_latent,
                        noise_mask=noise_mask,
                        callback=create_phase_callback(s_start),
                        disable_pbar=use_custom_preview,
                        seed=seed,
                    )
                    current_latent = result

                    prev_stage_sigmas = indexer.get_stage_sigmas(s_start, s_end)
                    log_tensor(f"Stage {i+1} Output", current_latent)

                    if plan_idx < len(planned_stages) - 1:
                        next_stage = planned_stages[plan_idx + 1]
                        next_base = _get_base_sigmas(
                            next_stage.model, next_stage.scheduler_name
                        )
                        next_indexer = SigmaIndexer(target_total_steps, next_base)
                        next_sigma = next_indexer.get_sigma_at(next_stage.global_start)

                        if (
                            next_sigma is not None
                            and len(stage_sigmas) > 0
                        ):
                            expected_sigma = stage_sigmas[-1].item()
                            sigma_diff = abs(expected_sigma - next_sigma)
                            if sigma_diff > SIGMA_DISCONTINUITY_THRESHOLD:
                                logger.warning(
                                    f"Sigma discontinuity: {expected_sigma:.4f} → {next_sigma:.4f} "
                                    f"(Δ={sigma_diff:.4f})"
                                )

                    stage_time = time.time() - t_stage
                    stage_timings.append(
                        (i + 1, s_start, s_end, current_sampler, stage_time)
                    )

                except (RuntimeError, ValueError) as e:
                    logger.error(f"Error in Stage {i+1}: {e}")
                    raise

            if hasattr(current_latent, "is_cuda"):
                samples = current_latent.cpu() if current_latent.is_cuda else current_latent
            else:
                samples = current_latent

            if not (tile_mode and latent_samples.ndim == 4):
                timings["sampling"] = time.time() - t0

            if tile_mode and latent_samples.ndim == 5:
                logger.warning(
                    "[Radiance] Tile sampling ignored for 5D video latents."
                )

        finally:

            if _ltxav_mr_base is not None and _ltxav_mr_orig is not None:
                _ltxav_mr_base.memory_required = _ltxav_mr_orig

            if "noise" in locals():
                try: del noise
                except UnboundLocalError: pass
            if "work_latent" in locals():
                try: del work_latent
                except UnboundLocalError: pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        if is_ltx_av and ltxav_obj is not None and _HAS_NESTED_TENSOR:
            try:
                final_t = samples["samples"] if isinstance(samples, dict) else samples

                if isinstance(final_t, _NestedTensor):
                    final_t_raw = final_t.tensors
                else:
                    final_t_raw = final_t

                # ALBABIT-FIX: Removed internal normalization multipliers entirely. LTX natively recombines.
                v_s, a_s = ltxav_obj.separate_audio_and_video_latents(final_t_raw, None)
                v_s = v_s.detach()
                a_s = a_s.detach()

                recombined = _NestedTensor(
                    ltxav_obj.recombine_audio_and_video_latents(v_s, a_s)
                )
                if isinstance(samples, dict):
                    samples["samples"] = recombined
                else:
                    samples = recombined

                del v_s, a_s, final_t_raw
                logger.info("[Radiance] LTX-AV recombination applied (native scaling).")
            except Exception as _av_err:
                logger.warning(
                    f"[Radiance] LTX-AV recombination failed: {_av_err}. "
                )

        t0 = time.time()
        out = latent_image.copy()
        out["samples"] = samples
        timings["output_prep"] = time.time() - t0

        total_time = time.time() - t_start

        logger.info(
            f"Sampling complete: {target_total_steps} total target steps, {total_time:.2f}s total, "
            f"{timings['sampling']:.2f}s sampling"
        )
        for stage_num, s_start_t, s_end_t, samp, t in stage_timings:
            logger.info(
                f"  Stage {stage_num}: steps {s_start_t + 1}→{s_end_t} [{samp}] = {t:.3f}s"
            )

        # ── Restart sampling (IRES style) ─────────────────────────────────────
        if restart_count > 0 and restart_schedule is not None and len(restart_schedule) > 0:
            try:
                _rs_sampler = comfy.samplers.sampler_object(primary_sampler)
                raw_samples = out["samples"].to(device)
                raw_samples = self._apply_restarts(
                    raw_samples, model, positive, negative,
                    _rs_sampler, sigmas, restart_schedule,
                    restart_count, cfg, seed, device,
                )
                out["samples"] = raw_samples.cpu()
                logger.info(
                    f"Restart sampling applied: {restart_count} restart(s) at "
                    f"{len(restart_schedule)} sigma level(s)"
                )
            except Exception as _rs_err:
                logger.warning(f"Restart sampling failed: {_rs_err} — using unmodified output")

        output_sigmas = (
            sigmas if sigmas is not None and len(sigmas) > 0 else torch.tensor([0.0])
        )

        # ── sigmas_remaining: the unused schedule tail after effective_end ───
        _eff_end_idx = min(effective_end, len(output_sigmas))
        sigmas_remaining = output_sigmas[_eff_end_idx:] if _eff_end_idx < len(output_sigmas) else torch.tensor([0.0])

        # ── Non-finite guard ───────────────────────────────────────────────────
        # CFG blowups, fp16/bf16 overflow, or a degenerate schedule can produce
        # NaN/Inf in the sampled latent, which silently decode to black or garbage
        # frames with no error. Detect, warn loudly, and sanitize so a bad run is
        # visible in the log rather than shipped as a corrupt plate.
        try:
            _s = out.get("samples") if isinstance(out, dict) else None
            if _s is not None and not torch.isfinite(_s).all():
                _bad = int((~torch.isfinite(_s)).sum().item())
                logger.warning(
                    "[RadianceSamplerPro] %d non-finite value(s) (NaN/Inf) in sampled "
                    "latent — sanitizing via nan_to_num. Check CFG, scheduler, and precision.",
                    _bad,
                )
                out = {**out, "samples": torch.nan_to_num(_s, nan=0.0, posinf=0.0, neginf=0.0)}
        except Exception as _nf_err:  # never let the guard itself break a valid run
            logger.debug("[RadianceSamplerPro] finite-check skipped: %s", _nf_err)

        # ── sigma_plot IMAGE ──────────────────────────────────────────────────
        sigma_plot = self._build_sigma_plot(output_sigmas, detected_type)

        return {
            "ui": self._resolved_values_ui(cfg, flux_guidance, flux_shift, sampler, steps),
            "result": (out, output_sigmas, sigmas_remaining, sigma_plot),
        }

NODE_CLASS_MAPPINGS = {
    "RadianceSamplerPro": RadianceSamplerPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSamplerPro": "◎ Radiance Sampler",
}
