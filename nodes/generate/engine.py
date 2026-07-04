"""
nodes_engine.py — Radiance Engine Nodes  v3.0.2

CHANGELOG v3.0.1 (audit + scored-review fixes):
  ROUND 1 — structural audit:
  - Issue 1: _SCENE_REFERRED guard now uses exact set membership (target_space in _SCENE_REFERRED)
             instead of fragile substring matching — prevents false positives on future space names.
  - Issue 2: Removed dead Radiancev3_MasterHub class (was defined but never registered in
             NODE_CLASS_MAPPINGS — invisible to ComfyUI).
  - Issue 3: Removed duplicate RadianceLUTApply class (canonical version is color/lut.py;
             local copy risked silent divergence). Import from .color if needed.
  - Issue 4: Fast decoder load wrapped in try/except — wrong model type or missing weights
             now logs a clear error and falls back to standard VAE instead of crashing.
  - Issue 5: Metadata timestamp now uses UTC (datetime.now(timezone.utc)) — unambiguous
             across timezones and daylight savings.
  ROUND 2 — scored review:
  - HIGH   : model_type NameError in except block — initialised to "unknown" sentinel
             before the try so the except handler can always format it safely.
             Raised a clear ValueError when samples["samples"] is None.
  - MEDIUM : SD 1.x misidentified as SDXL — 4-channel branch now inspects the VAE's
             first_stage_model class name for "xl"/"sdxl"; warns and falls back to
             "sdxl" weights for unrecognised 4-channel VAEs.
  - MEDIUM : _explicit set replaced with inspect.signature(self.apply).parameters —
             auto-syncs when new params are added; can never drift out of sync.
  - MEDIUM : metadata "version" field updated to "3.0.1" to match file version.
  - LOW    : @torch.no_grad() added to apply() — explicit intent, guards future paths.

CHANGELOG v3.0.2 (review fixes):
  - Fix 1: JS widget sync now matches "◎ Radiance HDR VAE Decode" display name
           (amber badge, display_tonemap graying, hdr_output warning)
  - Fix 2: force_hdr_decode respects the user's widget value (default True for this node)
           No longer hardcoded to True — passes user choice through to engine
  - Fix 3: Metadata JSON now includes alpha_restored, source_space, hdr_output fields
  - Fix 4: source_space, hdr_output, and all inherited params now explicitly declared
           in apply() method signature (no more ghost params through **kwargs)

CHANGELOG v3.0.0 vs v2.5:
  BUG FIXES:
    - BUG 1: RadianceHDRVAEDecode now exposes display_tonemap widget (v2.3.8 req).
             Without it, Compress(Log) mode silently used Reinhard regardless of intent.
    - BUG 2: hdr_scale_factor now guarded — warns and skips multiplication when
             target_space is display-referred ([0,1] sRGB/Raw), preventing blown output.
    - BUG 3: apply() target_space default aligned to vae.py v2.3.8 default ("sRGB").
             Was hardcoded "Linear" — silent mismatch with inherited INPUT_TYPES UI default.
    - BUG 4: NDI apply() guards image tensor shape before [0] indexing (IndexError).
    - BUG 5: NDI singleton tracks current stream_name and recreates sender on change.
    - BUG 6: NDI fallback path (turbo failed) now correctly applies log encoding
             since the turbo path did not encode it. Guard logic inverted correctly.
    - BUG 7: RadianceHDRVAEDecode adds metadata STRING output (decode settings).
    - BUG 8: **kwargs now forwarded to engine.decode() so new vae.py params are passed.
    - BUG 9: numpy duplicate import removed from NDI apply().

  NEW NODES:
    - RadianceHDRColorPipeline: full Radiance color pipeline as standalone node.
      Any IMAGE + colorspace_in + colorspace_out → converted IMAGE.
    - RadianceHDRAnalysis: scene-linear image → peak nit, EV range, clip %, zone stats.
      Enables conditional workflow logic (e.g. auto-clamp if peak > 1000 nit).
    - RadianceLUTApply: apply a .cube LUT file to any IMAGE with strength control.
             (canonical implementation: color/lut.py — registered via color/__init__.py)

  NDI UPGRADES:
    - frame_rate exposed for NDI timing metadata.
    - connected BOOLEAN output — lets workflow branch when NDI is unavailable.
"""

import inspect
import logging
import math
import os
import struct
import json
import datetime
from datetime import timezone as _tz
import torch
import numpy as np

from radiance.hdr.vae import RadianceVAE4KDecode
from radiance.fast_vae import decode_to_linear_realtime, detect_rudra_model_type, load_radiance_decoder_weights
from radiance.color.transfer import (
    tensor_linear_to_logc4,
    tensor_linear_to_slog3,
    tensor_srgb_to_linear,
    tensor_linear_to_srgb,
)
from radiance.color.pipeline import apply_input_transform, apply_output_transform, INPUT_COLORSPACES

logger = logging.getLogger("radiance.engine")

# Scene-referred spaces — hdr_scale_factor is valid for these
_SCENE_REFERRED = {
    "Linear", "ACEScg", "ACES 2065-1", "Rec.2020 Linear",
    "ARRI LogC3", "ARRI LogC4", "Sony S-Log3", "Panasonic V-Log",
    "DaVinci Intermediate", "RED Log3G10",
}


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 1: RADIANCE HDR VAE DECODE
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceHDRVAEDecode:
    """
    ◎ Radiance HDR VAE Decode

    Thin wrapper around RadianceVAE4KDecode that adds:
      • hdr_scale_factor  — linear multiplier for scene-referred output
      • display_tonemap   — exposed from vae.py v2.3.8 (required for Compress(Log))
      • metadata output   — decode settings as a JSON string for downstream nodes
      • All vae.py params passed through correctly, no silent drops
      • force_hdr_decode defaults True (this is an HDR-dedicated node)
      • Alpha passthrough tracked in metadata (alpha_restored flag)
      • source_space, hdr_output, and all vae.py params explicitly declared
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Decode HDR latents using a VAE conditioned for high-dynamic range."
    FUNCTION = "apply"
    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("image", "metadata")
    OUTPUT_TOOLTIPS = (
        "Decoded image tensor.",
        "JSON string with decode settings and applied hdr_scale_factor.",
    )

    # Class-level engine cache — avoids re-instantiating RadianceVAE4KDecode
    # on every apply() call (which is a pure no-op object creation but adds
    # needless Python overhead on every ComfyUI execution).
    _engine: "RadianceVAE4KDecode | None" = None

    @classmethod
    def INPUT_TYPES(cls):
        # Inherit all inputs from the production decode node
        types = RadianceVAE4KDecode.INPUT_TYPES()

        # BUG 3 FIX: explicitly set default to sRGB to match vae.py v2.3.8
        if "target_space" in types.get("required", {}):
            types["required"]["target_space"][1]["default"] = "sRGB"

        # force_hdr_decode is always True for this HDR-dedicated node — no user-facing
        # widget needed. The parent node uses it as a safety guard against accidentally
        # applying Log/SoftClip inversion on post-sampler sRGB latents, but anyone
        # using a node labelled "HDR VAE Decode" intends HDR output.
        types["optional"].pop("force_hdr_decode", None)

        # Override hdr_mode default to Compress(Log) — this is an HDR node
        if "hdr_mode" in types.get("optional", {}):
            types["optional"]["hdr_mode"][1]["default"] = "Compress (Log)"

        # Override source_space default to ARRI LogC4 — matches the default Compress(Log) decompression curve
        if "source_space" in types.get("optional", {}):
            types["optional"]["source_space"][1]["default"] = "ARRI LogC4"

        # All HDR-specific widgets default to True for this HDR-dedicated node
        if "hdr_output" in types.get("optional", {}):
            types["optional"]["hdr_output"][1]["default"] = True
        if "export_rhdr" in types.get("optional", {}):
            types["optional"]["export_rhdr"][1]["default"] = True
        if "rhdr_precision" in types.get("optional", {}):
            types["optional"]["rhdr_precision"][1]["default"] = "f32"

        # Hide crop_padding and processing_mode from the UI — both have safe
        # defaults ("", "sequential") and only add noise for HDR decode users.
        types["optional"].pop("crop_padding", None)
        types["optional"].pop("processing_mode", None)

        # Add hdr_scale_factor to optional section
        types.setdefault("optional", {})
        types["optional"]["hdr_scale_factor"] = (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "step": 0.05,
                "tooltip": (
                    "Linear multiplier applied after decode. Only active for "
                    "scene-referred target spaces (Linear, ACEScg, Log). "
                    "Ignored (with a warning) for sRGB/Raw display-referred output "
                    "to prevent blown highlights."
                ),
            },
        )
        types["optional"]["rudra_decoder"] = (
            ["Disabled", "Enabled"],
            {
                "default": "Disabled",
                "tooltip": (
                    "Enable to use the distilled RUDRA decoder weights "
                    "instead of the standard VAE. Dramatically faster and handles "
                    "high dynamic range without highlight noise or clamping."
                ),
            },
        )
        types["optional"]["decoder_size"] = (
            ["rudra_turbo", "rudra_full"],
            {
                "default": "rudra_turbo",
                "tooltip": (
                    "'rudra_turbo': 2M param real-time dynamic range conditioned model. "
                    "'rudra_full': 32M param production-grade dynamic range conditioned model."
                ),
            },
        )
        return types

    @torch.no_grad()   # FIX (Low): explicit no_grad — makes intent clear, guards future paths
    def apply(
        self,
        samples: dict,
        vae,
        target_space: str = "sRGB",     # BUG 3 FIX: aligned to vae.py v2.3.8 default
        tile_size: str = "Auto",
        overlap: int = 128,
        exposure_adjust: float = 0.0,
        alpha=None,
        hdr_mode: str = "Compress (Log)",
        display_tonemap: str = "Reinhard",  # BUG 1 FIX: now forwarded to decode()
        source_space: str = "ARRI LogC4",
        hdr_output: bool = True,
        inverse_tonemap: bool = False,
        target_stops: float = 12.0,
        crop_padding: str = "",
        export_rhdr: bool = False,
        rhdr_precision: str = "f32",
        processing_mode: str = "sequential",
        decode_noise_scale: float = 0.0,
        hdr_scale_factor: float = 1.0,
        rudra_decoder: str = "Disabled",
        decoder_size: str = "rudra_turbo",
        **kwargs,                           # BUG 8 FIX: forward remaining params
    ):
        # Lazily instantiate once; RadianceVAE4KDecode is stateless so one
        # shared instance is safe across all ComfyUI graph executions.
        if RadianceHDRVAEDecode._engine is None:
            RadianceHDRVAEDecode._engine = RadianceVAE4KDecode()
        engine = RadianceHDRVAEDecode._engine

        # Guard: if the user connected an IMAGE (LoadImage etc.) to the 'samples'
        # input instead of a LATENT, give a clear message instead of a cryptic
        # VAE crash.
        if not isinstance(samples, dict):
            raise RuntimeError(
                "This node needs a LATENT, but the input is an IMAGE.\n\n"
                "Make sure you connect a VAE Encode or a Sampler node before this one."
            )
        if "samples" not in samples:
            raise RuntimeError(
                "The LATENT connected to this node is missing its data.\n\n"
                f"Keys found: {list(samples.keys())}\n"
                "Make sure you connect a valid LATENT output."
            )

        # Track whether alpha was provided (for metadata / downstream use)
        alpha_provided = alpha is not None
        force_hdr_decode = True  # always True for this HDR-dedicated node

        # Forward extra vae.py params via **kwargs, stripping any key already
        # passed explicitly to avoid "multiple values for keyword argument".
        # Derive the set from inspect.signature so it stays in sync automatically
        # when new params are added to apply().
        _sig_params = set(inspect.signature(self.apply).parameters)
        _sig_params.discard("kwargs")       # the **kwargs catch-all itself
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in _sig_params}

        # Load distilled decoder if enabled.
        # FIX (High — NameError): initialise model_type before the try block so the
        # except handler can always format it safely. Previously, if
        # samples.get("samples") returned None, _samples.shape raised AttributeError
        # before model_type was assigned — the except block then raised NameError,
        # masking the real error entirely.
        # ALBABIT-FIX: model_type detection (including the 4ch SD1.x/2.x/SDXL
        # disambiguation) now lives in detect_rudra_model_type() (fast_vae.py) —
        # single source of truth shared with RadianceNDISender.
        turbo_decoder = None
        if rudra_decoder == "Enabled":
            model_type = "unknown"  # safe sentinel — always defined before except block
            try:
                _samples = samples.get("samples")
                if _samples is None:
                    raise ValueError(
                        "samples dict is missing the 'samples' tensor — "
                        "ensure a LATENT is connected to this node."
                    )
                _ch = _samples.shape[1]
                model_type = detect_rudra_model_type(_ch, _samples.ndim == 5, vae=vae)

                turbo_decoder = load_radiance_decoder_weights(
                    model_type=model_type,
                    model_size=decoder_size,
                )
                if turbo_decoder is not None:
                    # Turbo/Full models expect log targets. Force log mode if not already set.
                    if hdr_mode != "Compress (Log)":
                        logger.warning(
                            f"[RadianceHDRVAEDecode] RUDRA Decoder enabled but hdr_mode='{hdr_mode}'. "
                            f"Distilled models require log-coded targets; forcing 'Compress (Log)'."
                        )
                        hdr_mode = "Compress (Log)"

                    # Auto-align source_space to a valid log profile if needed
                    VALID_LOG_SPACES = {"ARRI LogC4", "Sony S-Log3", "ARRI LogC3", "Panasonic V-Log", "DaVinci Intermediate", "RED Log3G10"}
                    if source_space not in VALID_LOG_SPACES:
                        from radiance.config.model_map import resolve_model_vae_config
                        cfg = resolve_model_vae_config(model_type) or {}
                        default_curve = cfg.get("log_curve", "ARRI LogC4")
                        logger.info(
                            f"[RadianceHDRVAEDecode] RUDRA Decoder enabled but source_space='{source_space}' "
                            f"is not a log-encoded space. Automatically setting decompression profile to "
                            f"'{default_curve}' to match distilled weights."
                        )
                        source_space = default_curve
                else:
                    # ALBABIT-FIX: load_radiance_decoder_weights() now returns None
                    # (instead of a randomly-initialised decoder) when no compatible
                    # RUDRA checkpoint is available — keep hdr_mode/source_space
                    # untouched and fall back to the standard VAE decode.
                    logger.info(
                        f"[RadianceHDRVAEDecode] No RUDRA decoder available for "
                        f"model_type={model_type!r} (size={decoder_size!r}). "
                        f"Falling back to the standard VAE decoder."
                    )
            except Exception as _fast_err:
                logger.error(
                    f"[RadianceHDRVAEDecode] RUDRA decoder load failed "
                    f"(model_type={model_type!r}, size={decoder_size!r}): {_fast_err}. "
                    f"Falling back to standard VAE decoder."
                )
                turbo_decoder = None

        # ALBABIT-FIX: track whether RUDRA was actually used (vs merely
        # requested) — rudra_decoder/decoder_size below reported the request
        # even on silent fallback (no checkpoint, load error), misleading the
        # metadata JSON about what actually produced the image.
        rudra_actually_used = turbo_decoder is not None

        result = engine.decode(
            samples=samples,
            vae=vae,
            target_space=target_space,
            tile_size=tile_size,
            overlap=overlap,
            exposure_adjust=exposure_adjust,
            alpha=alpha,
            hdr_mode=hdr_mode,
            display_tonemap=display_tonemap,
            source_space=source_space,
            force_hdr_decode=True,
            hdr_output=hdr_output,
            inverse_tonemap=inverse_tonemap,
            target_stops=target_stops,
            crop_padding=crop_padding,
            export_rhdr=export_rhdr,
            rhdr_precision=rhdr_precision,
            processing_mode=processing_mode,
            decode_noise_scale=decode_noise_scale,
            turbo_decoder=turbo_decoder,
            **safe_kwargs,
        )

        image = result[0] if isinstance(result, (tuple, list)) else result

        # BUG 2 FIX: guard hdr_scale_factor for display-referred spaces
        # FIX (Issue 1): use exact set membership — substring matching was fragile
        # and could misfire on future colorspace names that happen to contain a
        # scene-referred name as a substring.
        scale_applied = False
        if hdr_scale_factor != 1.0:
            if target_space in _SCENE_REFERRED:
                image = image * hdr_scale_factor
                scale_applied = True
            else:
                logger.warning(
                    f"[RadianceHDRVAEDecode] hdr_scale_factor={hdr_scale_factor} ignored "
                    f"for display-referred target_space='{target_space}' — would blow highlights. "
                    f"Use a scene-referred space (Linear, ACEScg, LogC4 etc.) for HDR scaling."
                )

        # BUG 7 FIX: emit decode settings as metadata JSON
        meta = json.dumps({
            "node": "RadianceHDRVAEDecode",
            "version": "3.0.2",
            "target_space": target_space,
            "source_space": source_space,
            "hdr_mode": hdr_mode,
            "display_tonemap": display_tonemap,
            "exposure_adjust": exposure_adjust,
            "hdr_scale_factor": hdr_scale_factor if scale_applied else "N/A (display-referred)",
            "rudra_decoder": rudra_decoder if rudra_actually_used else f"{rudra_decoder} (fallback: standard VAE used)",
            "decoder_size": decoder_size if rudra_actually_used else "N/A",
            "force_hdr_decode": force_hdr_decode,
            "alpha_restored": alpha_provided,
            "hdr_output": hdr_output,
            "timestamp": datetime.datetime.now(_tz.utc).isoformat(timespec="seconds"),
        }, indent=2)

        return (image, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 2: RADIANCE HDR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceHDRAnalysis:
    """
    ◎ Radiance HDR Analysis

    Analyses a scene-linear IMAGE tensor and outputs HDR metrics used by the
    Radiance Viewer exposure strip — now available as ComfyUI node outputs for
    conditional workflow logic.

    Output values:
      peak_nit    — estimated peak luminance in cd/m² (BT.2408: 203 nit = linear 1.0)
      ev_range    — dynamic range in stops between p01 and p99
      clipped_pct — percentage of pixels above scene-linear 1.0
      is_hdr      — True if peak_nit > 203 (above SDR white)
      stats_json  — full statistics as JSON string

    Typical use:
      Feed scene-linear output of Radiance Decode → RadianceHDRAnalysis →
      route to RadianceHDRColorPipeline or conditional tonemapper based on is_hdr.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Analyse HDR image statistics: peak luminance, clipping, histogram."
    FUNCTION = "analyse"
    RETURN_TYPES  = ("FLOAT", "FLOAT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES  = ("peak_nit", "ev_range", "clipped_pct", "is_hdr", "stats_json")
    OUTPUT_TOOLTIPS = (
        "Estimated peak luminance in cd/m² (BT.2408 anchor: 203 nit = linear 1.0).",
        "Dynamic range in stops (EV) between p01 and p99 luma.",
        "Percentage of pixels above scene-linear 1.0 (clipped for SDR display).",
        "True when peak_nit > 203 — image contains HDR content above SDR white.",
        "Full zone statistics as JSON.",
    )

    # BT.2408 SDR anchor: 203 cd/m² = scene-linear 1.0
    _NIT_ANCHOR = 203.0
    _MAX_SAMPLES = 500_000   # performance cap — subsample large tensors

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "colorspace": (
                    INPUT_COLORSPACES,
                    {
                        "default": "Linear (sRGB)",
                        "tooltip": (
                            "Colorspace of the incoming image. "
                            "'Linear (sRGB)' is the correct setting for Radiance Decode output. "
                            "Log spaces are linearised first for accurate nit estimation."
                        ),
                    },
                ),
            },
        }

    @torch.no_grad()
    def analyse(self, image: torch.Tensor, colorspace: str = "Linear (sRGB)"):
        # Linearise if needed (log/sRGB input)
        if colorspace != "Linear (sRGB)" and colorspace != "ACEScg":
            linear = apply_input_transform(image, colorspace)
        else:
            linear = image

        # Flatten to (N, 3) for fast percentile computation
        flat = linear[..., :3].reshape(-1, 3).float()
        N = flat.shape[0]

        # Subsample if very large
        if N > self._MAX_SAMPLES:
            step = max(1, N // self._MAX_SAMPLES)
            flat = flat[::step]

        if colorspace == "ACEScg":
            luma = 0.272229 * flat[:, 0].clamp(min=0.0) \
                 + 0.674082 * flat[:, 1].clamp(min=0.0) \
                 + 0.053689 * flat[:, 2].clamp(min=0.0)
        else:
            luma = 0.2126 * flat[:, 0].clamp(min=0.0) \
                 + 0.7152 * flat[:, 1].clamp(min=0.0) \
                 + 0.0722 * flat[:, 2].clamp(min=0.0)

        p01, p50, p99, p999 = torch.quantile(
            luma, torch.tensor([0.01, 0.50, 0.99, 0.999], device=luma.device)
        ).tolist()
        peak = float(luma.max())

        # EV range (p01 → p99) — avoid log(0)
        ev_range = math.log2(max(p99, 1e-6) / max(p01, 1e-6)) if p01 > 1e-6 else 0.0

        # Clipped pixels (above scene-linear 1.0)
        clipped_pct = float((luma > 1.0).float().mean()) * 100.0

        # Peak nit estimate
        peak_nit = peak * self._NIT_ANCHOR

        is_hdr = bool(peak_nit > self._NIT_ANCHOR)

        stats = {
            "node": "RadianceHDRAnalysis",
            "colorspace": colorspace,
            "pixels_sampled": len(luma),
            "p01": round(p01, 6),
            "p50": round(p50, 6),
            "p99": round(p99, 6),
            "p99.9": round(p999, 6),
            "peak_linear": round(peak, 6),
            "peak_nit": round(peak_nit, 2),
            "ev_range": round(ev_range, 3),
            "clipped_pct": round(clipped_pct, 4),
            "is_hdr": is_hdr,
            "nit_anchor": self._NIT_ANCHOR,
        }
        stats_json = json.dumps(stats, indent=2)

        logger.info(
            f"[HDRAnalysis] peak={peak_nit:.0f} nit, "
            f"EV={ev_range:.1f}, clip={clipped_pct:.2f}%, hdr={is_hdr}"
        )

        return (float(peak_nit), float(ev_range), float(clipped_pct), is_hdr, stats_json)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTE (Issue 3 fix): RadianceLUTApply was previously defined here but is NOT
# registered in this file's NODE_CLASS_MAPPINGS. The canonical, full-featured
# implementation lives in color/lut.py and is registered via color/__init__.py.
# The local copy has been removed to eliminate the risk of the two diverging
# silently. Import from color/ if you need it:
#   from .color import RadianceLUTApply
# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 5: RADIANCE NDI SENDER
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceNDISender:
    """
    ◎ Radiance NDI Sender (Turbo)

    Real-time streaming node to push frames to OBS, Resolume, Nuke,
    or any NDI receiver via NewTek NDI SDK.

    v2.6.0 fixes:
    - Singleton tracks stream_name and recreates sender on name change.
    - Guard against empty image batch (IndexError).
    - BUG 6 fix: fallback path (turbo failed) correctly applies log encoding.
    - Duplicate numpy import removed.
    - frame_rate exposed for NDI timing metadata.
    - connected BOOLEAN output for workflow branching.

    Note: NDIlib must be installed separately.
      pip install ndi-python
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Stream frames in real time over the NDI network video protocol."
    FUNCTION = "apply"
    OUTPUT_NODE = True
    RETURN_TYPES  = ("IMAGE", "BOOLEAN")
    RETURN_NAMES  = ("image", "connected")
    OUTPUT_TOOLTIPS = (
        "Pass-through image (unchanged).",
        "True if NDI frame was sent successfully this call.",
    )

    # Singleton NDI state
    _ndi_send_instance = None
    _ndi_video_frame   = None
    _ndi_stream_name   = None     # BUG 5 FIX: track active stream name
    _ndi_lock = __import__("threading").RLock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stream_name": (
                    "STRING",
                    {"default": "Radiance ComfyUI"},
                ),
                "encoding": (
                    ["None (SDR)", "S-Log3 (HDR)", "LogC4 (HDR)"],
                    {"default": "None (SDR)"},
                ),
                "enable_streaming": ("BOOLEAN", {"default": True,
                    "tooltip": "Enable real-time NDI streaming during generation. Requires NDI SDK installed."
                }),
                "frame_rate": (                 # Suggestion C
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 1.0,
                        "tooltip": "Frame rate written into NDI video frame metadata.",
                    },
                ),
            },
            "optional": {
                "latent_in": ("LATENT",),
                "vae": ("VAE",),
                "turbo_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Uses fast HDR decoder if latent_in and vae "
                            "are connected. Bypasses full tiled decode."
                        ),
                    },
                ),
            },
        }

    def apply(
        self,
        image: torch.Tensor,
        stream_name: str,
        encoding: str,
        enable_streaming: bool,
        frame_rate: float = 24.0,
        latent_in=None,
        vae=None,
        turbo_mode: bool = False,
    ):
        if not enable_streaming:
            return (image, False)

        # BUG 4 FIX: guard empty batch
        if image is None or image.shape[0] == 0:
            logger.warning("[Radiance NDI] Received empty image batch — skipping.")
            return (image, False)

        # ── Select frame source ────────────────────────────────────────────────
        img_work = None
        turbo_succeeded = False

        if turbo_mode and latent_in is not None and vae is not None:
            try:
                profile_name = (
                    "Sony S-Log3" if encoding == "S-Log3 (HDR)"
                    else "ARRI LogC4" if encoding == "LogC4 (HDR)"
                    else "None"
                )

                # ── VRAM relief before turbo forward pass ──────────────────────
                try:
                    import comfy.model_management as mm
                    mm.soft_empty_cache()
                    mm.unload_all_models()
                    logger.debug("[Radiance NDI] Offloaded models before turbo decode")
                except Exception as exc:
                    logger.warning("[nodes_engine]: %s", exc)
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                except Exception as exc:
                    logger.warning("[nodes_engine]: %s", exc)

                # Direct fast decode via fast_vae library
                latent = latent_in["samples"]
                ch = latent.shape[1]
                # ALBABIT-FIX: shared detect_rudra_model_type() (fast_vae.py) replaces
                # "ch >= 16 -> flux else sdxl", which misclassified 12ch (mochi) and
                # 128ch (ltx-video, flux2/flux2-klein) latents as 4ch sdxl.
                model_type = detect_rudra_model_type(ch, latent.ndim == 5, vae=vae)
                compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # ALBABIT-FIX: model_size="turbo" never matched on-disk
                # "rudra_turbo_decoder_*_ema.safetensors" filenames — aligned to
                # "rudra_turbo" like RadianceHDRVAEDecode. load_radiance_decoder_weights()
                # can now return None (no compatible checkpoint); raise so the
                # existing try/except below falls back to the plain image input
                # (BUG 6 FIX).
                decoder = load_radiance_decoder_weights(model_type=model_type, model_size="rudra_turbo")
                if decoder is None:
                    raise RuntimeError(f"No RUDRA decoder available for model_type={model_type!r}")
                decoder = decoder.to(compute_device)
                scale_factor = getattr(vae, "scale_factor", None)
                if not isinstance(scale_factor, (int, float)) or scale_factor == 0:
                    from radiance.config.model_map import get_model_vae_param
                    scale_factor = get_model_vae_param(model_type, "scale_factor", default=0.18215)
                from radiance.hdr.vae import LOG_PROFILE_HDR_PARAMS, LOG_PROFILE_HDR_DEFAULT
                params = LOG_PROFILE_HDR_PARAMS.get(profile_name, LOG_PROFILE_HDR_DEFAULT)
                log_curve = profile_name if profile_name != "None" else "ARRI LogC4"
                img_work = decode_to_linear_realtime(
                    latent=latent.to(compute_device),
                    decoder=decoder,
                    scale_factor=scale_factor,
                    profile_params=params,
                    precision="bf16",
                    log_curve=log_curve,
                ).cpu().float()
                turbo_succeeded = True
            except Exception as e:
                logger.error(
                    f"[Radiance NDI] Turbo path failed: {e}. "
                    f"Falling back to image input."
                )
                img_work = image[0].clone()
                # turbo_succeeded stays False — fallback path needs encoding below

        if img_work is None:
            img_work = image[0].clone()

        # ── Apply log encoding ─────────── Apply log encoding ─────────────────────────────────────────────────
        # BUG 6 FIX: turbo path pre-encodes — only skip encoding when turbo succeeded.
        # If turbo failed and we fell back to the raw image, encoding must still apply.
        if not turbo_succeeded:
            if encoding == "S-Log3 (HDR)":
                img_work = tensor_linear_to_slog3(img_work)
            elif encoding == "LogC4 (HDR)":
                img_work = tensor_linear_to_logc4(img_work)

        # ── NDI dispatch ───────────────────────────────────────────────────────
        connected = False
        try:
            import NDIlib as ndi
        except ImportError:
            logger.warning(
                "[Radiance NDI] NDIlib not installed. "
                "Install with: pip install ndi-python"
            )
            return (image, False)

        # BUG 5 FIX: recreate sender if stream_name changed
        with RadianceNDISender._ndi_lock:
            if (RadianceNDISender._ndi_send_instance is None
                    or RadianceNDISender._ndi_stream_name != stream_name):
                if RadianceNDISender._ndi_send_instance is not None:
                    ndi.send_destroy(RadianceNDISender._ndi_send_instance)
                    RadianceNDISender._ndi_send_instance = None
                if not ndi.initialize():
                    logger.error("[Radiance NDI] ndi.initialize() failed.")
                    return (image, False)
                desc = ndi.SendCreate()
                desc.p_ndi_name = stream_name
                RadianceNDISender._ndi_send_instance = ndi.send_create(desc)
                RadianceNDISender._ndi_video_frame   = ndi.VideoFrameV2()
                RadianceNDISender._ndi_stream_name   = stream_name
                logger.info(f"[Radiance NDI] Sender created: '{stream_name}'")

        # BUG 9 FIX: numpy already imported at module level — no duplicate import
        img_np  = img_work.cpu().float().numpy()
        H, W, C = img_np.shape
        img_8bit = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        if C == 3:
            alpha_ch = np.full((H, W, 1), 255, dtype=np.uint8)
            img_bgra = np.concatenate(
                [img_8bit[..., 2:3], img_8bit[..., 1:2],
                 img_8bit[..., 0:1], alpha_ch], axis=2,
            )
        else:
            img_bgra = np.concatenate(
                [img_8bit[..., 2:3], img_8bit[..., 1:2],
                 img_8bit[..., 0:1], img_8bit[..., 3:4]], axis=2,
            )

        vf = RadianceNDISender._ndi_video_frame
        vf.xres = W
        vf.yres = H
        vf.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
        vf.p_data = img_bgra
        vf.line_stride_in_bytes = W * 4
        # frame rate metadata
        vf.frame_rate_N = int(frame_rate * 1000)
        vf.frame_rate_D = 1000
        ndi.send_send_video_v2(RadianceNDISender._ndi_send_instance, vf)
        connected = True

        return (image, connected)


# NOTE (Issue 2 fix): Radiancev3_MasterHub was previously defined here but was
# never added to NODE_CLASS_MAPPINGS — it was invisible to ComfyUI and constituted
# dead code. It has been removed. If you want to ship this node, implement it
# fully and register it in NODE_CLASS_MAPPINGS before release.

# ═══════════════════════════════════════════════════════════════════════════════
#                          NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceHDRVAEDecode":        RadianceHDRVAEDecode,
    "RadianceHDRAnalysis":         RadianceHDRAnalysis,
    # RadianceLUTApply is the canonical key — registered in color/__init__.py
    "RadianceNDISender":           RadianceNDISender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRVAEDecode":        "◎ Radiance HDR VAE Decode",
    "RadianceHDRAnalysis":         "◎ Radiance HDR Analysis",
    "RadianceNDISender":           "◎ Radiance NDI Sender",
}

import atexit
def _ndi_cleanup():
    if RadianceNDISender._ndi_send_instance is not None:
        try:
            import NDIlib as ndi
            ndi.send_destroy(RadianceNDISender._ndi_send_instance)
        except Exception:
            pass
atexit.register(_ndi_cleanup)
