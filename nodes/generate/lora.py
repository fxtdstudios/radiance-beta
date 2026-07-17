"""
nodes_hdr_lora.py — Radiance HDR LoRA ComfyUI nodes.
════════════════════════════════════════════════════════════════════════════════

Two nodes that wire the Radiance HDR LoRA workflow into any ComfyUI pipeline:

  1. RadianceHDRLoRALoader
     • Loads a .safetensors LoRA file trained by train_hdr_lora.py.
     • Reads Radiance metadata (model_name, compression_ratio, rank, alpha)
       embedded at export time.
     • Passes the raw MODEL through so downstream samplers see the patched
       weights.  Outputs compression_ratio so it can be wired directly into
       RadianceHDREncoder without manual entry.

  2. RadianceHDRLoRAApply
     • Applies a LoRA dict (from the Loader, or any other source) to a MODEL
       with a user-adjustable strength multiplier.
     • If model_hint is wired, it cross-checks against the preset table and
       warns when the LoRA was trained on a different model family.
     • Outputs the patched MODEL and the effective compression_ratio (from
       LoRA metadata, or RADIANCE_MODEL_PRESETS if not found).

Pipeline wiring example
────────────────────────
  [CheckpointLoader] ──MODEL──▶ [RadianceHDRLoRAApply] ──MODEL──▶ [KSampler]
                                        ▲                              │
  [RadianceHDRLoRALoader] ─lora_dict──┘                              │
         └── compression_ratio ──▶ [RadianceHDREncoder] ◀── image ──┘

The MODEL output from RadianceHDRLoRAApply replaces the base checkpoint MODEL
everywhere in the graph so no other nodes need changing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from radiance.path_utils import strip_path_quotes

logger = logging.getLogger("radiance.hdr_lora")
diag_logger = logging.getLogger("radiance.diagnostics")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy safetensors import — optional at import time, required when loading
# ─────────────────────────────────────────────────────────────────────────────

def _require_safetensors():
    try:
        from safetensors.torch import load_file, save_file  # noqa: F401
        return load_file
    except ImportError:
        raise ImportError(
            "safetensors is required for RadianceHDRLoRALoader.  "
            "Install it with: pip install safetensors"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Import shared presets from nodes_hdr_smart (or fall back gracefully)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from radiance.nodes.hdr.smart import RADIANCE_MODEL_PRESETS, _resolve_model
    _HAS_PRESETS = True
except ImportError:
        logger.warning(
            "nodes_hdr_smart not available — model preset lookup disabled in LoRA nodes."
        )
        RADIANCE_MODEL_PRESETS: dict = {}
        _HAS_PRESETS = False

        def _resolve_model(hint: str) -> str | None:  # type: ignore[misc]
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_LORA_METADATA_KEYS = {
    "radiance_model_name",
    "radiance_compression_ratio",
    "radiance_rank",
    "radiance_alpha",
    "radiance_version",
}


def _parse_lora_metadata(raw_meta: dict) -> dict:
    """Extract and coerce Radiance metadata from a safetensors metadata dict.

    safetensors stores ALL metadata values as strings, so we cast carefully.
    Returns a dict with Python-typed values (float, int, str).
    """
    out: dict[str, Any] = {}
    for key in _LORA_METADATA_KEYS:
        val = raw_meta.get(key)
        if val is None:
            continue
        if key in ("radiance_compression_ratio", "radiance_alpha"):
            try:
                out[key] = float(val)
            except (ValueError, TypeError):
                pass
        elif key in ("radiance_rank",):
            try:
                out[key] = int(val)
            except (ValueError, TypeError):
                pass
        else:
            out[key] = str(val)
    return out


def _apply_lora_to_model(model, lora_tensors: dict, strength: float) -> None:
    """Apply lora_down/up weight pairs to matching Linear layers in *model*.

    Key format (kohya_ss compatible):
        lora_unet_<dotted.path>.lora_down.weight  → A  (rank × in_features)
        lora_unet_<dotted.path>.lora_up.weight    → B  (out_features × rank)
        lora_unet_<dotted.path>.alpha             → scalar alpha (optional)

    The update is:   W += (strength * alpha / rank) * B @ A

    Works on raw state-dicts (plain nn.Module) as well as ComfyUI model
    wrappers that expose .model or .unet.
    """
    # Unwrap ComfyUI ModelPatcher → diffusion_model
    target = model
    for attr in ("model", "unet", "diffusion_model"):
        if hasattr(target, attr):
            target = getattr(target, attr)

    named_modules = dict(target.named_modules())

    # Group keys by dotted path prefix
    prefixes: set[str] = set()
    for k in lora_tensors:
        if k.endswith(".lora_down.weight"):
            # strip "lora_unet_" prefix and ".lora_down.weight" suffix
            inner = k[len("lora_unet_"):-len(".lora_down.weight")]
            prefixes.add(inner)

    applied = 0
    for prefix in prefixes:
        down_key = f"lora_unet_{prefix}.lora_down.weight"
        up_key   = f"lora_unet_{prefix}.lora_up.weight"
        alpha_key = f"lora_unet_{prefix}.alpha"

        A = lora_tensors.get(down_key)
        B = lora_tensors.get(up_key)
        if A is None or B is None:
            continue

        alpha_val = float(lora_tensors[alpha_key]) if alpha_key in lora_tensors else float(A.shape[0])
        rank = A.shape[0]
        scale = strength * alpha_val / rank

        # Convert dotted path "down_blocks_0_attentions_0_…" back to Python attr chain
        attr_path = prefix.replace("_", ".")
        # Walk the module tree — handle both _ and . separations
        module = _find_module_by_dotted_path(target, named_modules, prefix)
        if module is None:
            logger.debug("LoRA prefix not found in model: %s", prefix)
            continue

        if not hasattr(module, "weight"):
            continue

        device = module.weight.device
        dtype  = module.weight.dtype

        delta = scale * (B.to(device=device, dtype=torch.float32) @
                         A.to(device=device, dtype=torch.float32))
        with torch.no_grad():
            module.weight.add_(delta.to(dtype))
        applied += 1

    logger.info("RadianceHDRLoRAApply: applied %d LoRA delta(s) at strength=%.3f", applied, strength)
    diag_logger.info(
        "HDR_LORA_APPLY applied=%d strength=%.3f",
        applied, strength
    )


def _find_module_by_dotted_path(root, named_modules: dict, kohya_path: str):
    """Resolve a kohya_ss '_'-joined path to an nn.Module.

    kohya_ss replaces '.' with '_' in module paths, so we must try both
    forms.  We prefer the longest matching key in named_modules to handle
    ambiguous _ vs . cases (e.g. 'ff_net_0_proj' vs 'ff.net.0.proj').
    """
    # Try direct dotted replacement
    dotted = kohya_path.replace("_", ".")
    if dotted in named_modules:
        return named_modules[dotted]

    # Brute-force: try all keys where replacing '.' with '_' matches
    for name, mod in named_modules.items():
        if name.replace(".", "_") == kohya_path:
            return mod

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: RadianceHDRLoRALoader
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRLoRALoader:
    """Load a Radiance HDR LoRA .safetensors file.

    Reads the Radiance metadata embedded at training time so the
    compression_ratio flows automatically into RadianceHDREncoder.

    Outputs
    -------
    lora_dict         — opaque dict of tensors + metadata (pass to LoRAApply)
    compression_ratio — float extracted from LoRA metadata (or 0.5 default)
    model_name        — string model family the LoRA was trained on
    metadata_json     — full metadata as a JSON string for display/debugging
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "/path/to/radiance_hdr_lora.safetensors",
                    "tooltip": "Path to a Radiance HDR LoRA checkpoint (.safetensors or .pt). Leave blank to use the RADIANCE_HDR_LORA env var."
                }),
            },
            "optional": {
                "fallback_compression_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Used when LoRA metadata does not contain compression_ratio.",
                }),
            },
        }

    RETURN_TYPES    = ("LORA_DICT", "FLOAT",              "STRING",     "STRING")
    RETURN_NAMES    = ("lora_dict", "compression_ratio",  "model_name", "metadata_json")
    FUNCTION        = "load"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION     = (
        "Load a Radiance HDR LoRA .safetensors and extract embedded training metadata. "
        "Wire compression_ratio → RadianceHDREncoder to guarantee consistent HDR response."
    )

    def load(
        self,
        lora_path: str,
        fallback_compression_ratio: float = 0.5,
    ):
        load_file = _require_safetensors()

        lora_path = strip_path_quotes(lora_path)
        if not lora_path:
            raise ValueError("RadianceHDRLoRALoader: lora_path must not be empty.")
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(
                f"RadianceHDRLoRALoader: file not found — {lora_path}"
            )

        # Load tensors + raw string metadata
        from safetensors import safe_open  # type: ignore
        tensors: dict[str, torch.Tensor] = {}
        raw_meta: dict[str, str] = {}

        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
            raw_meta = dict(f.metadata()) if f.metadata() else {}

        meta = _parse_lora_metadata(raw_meta)

        compression_ratio = float(meta.get("radiance_compression_ratio", fallback_compression_ratio))
        model_name        = str(meta.get("radiance_model_name", "unknown"))

        # Build rich metadata JSON for display node / console
        display_meta = {
            "lora_path":          lora_path,
            "model_name":         model_name,
            "compression_ratio":  compression_ratio,
            "rank":               meta.get("radiance_rank", "?"),
            "alpha":              meta.get("radiance_alpha", "?"),
            "radiance_version":   meta.get("radiance_version", "?"),
            "tensor_count":       len(tensors),
            "raw_metadata":       raw_meta,
        }
        metadata_json = json.dumps(display_meta, indent=2)

        logger.info(
            "Loaded Radiance HDR LoRA: model=%s  ratio=%.3f  tensors=%d  path=%s",
            model_name, compression_ratio, len(tensors), lora_path,
        )
        diag_logger.info(
            "HDR_LORA_LOAD model=%s compression_ratio=%.3f rank=%s tensors=%d",
            model_name, compression_ratio, meta.get("radiance_rank", "?"), len(tensors),
        )

        lora_dict = {
            "tensors":  tensors,
            "metadata": meta,
            "path":     lora_path,
        }

        return (lora_dict, compression_ratio, model_name, metadata_json)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: RadianceHDRLoRAApply
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRLoRAApply:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """Apply a Radiance HDR LoRA to a diffusion MODEL.

    This node modifies the MODEL weights in-place by adding the low-rank
    delta (strength × alpha/rank × B@A) to each matching Linear projection.

    It is safe to chain multiple LoRA applies — each one accumulates deltas.

    Inputs
    ------
    model             — ComfyUI MODEL (from CheckpointLoader or prior LoRAApply)
    lora_dict         — LORA_DICT from RadianceHDRLoRALoader
    strength          — scaling multiplier (0 = skip, 1 = full, 1.5 = amplified)
    model_hint        — optional string to cross-check the LoRA's intended model
                        family.  Emits a WARNING if mismatched — does NOT block.

    Outputs
    -------
    MODEL             — patched MODEL (replaces the input MODEL in the graph)
    compression_ratio — from LoRA metadata (wire to RadianceHDREncoder)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("MODEL",),
                "lora_dict": ("LORA_DICT",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "LoRA strength multiplier. 1.0 = trained weight. 0 = no effect.",
                }),
            },
            "optional": {
                "model_hint": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "ltx-video / flux / wan / sdxl / …",
                    "tooltip": "Cross-checks the LoRA's trained model against this hint and warns if mismatched.",
                }),
            },
        }

    RETURN_TYPES  = ("MODEL",  "FLOAT")
    RETURN_NAMES  = ("MODEL",  "compression_ratio")
    FUNCTION      = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION   = (
        "Apply a Radiance HDR LoRA to a diffusion model. "
        "Outputs the patched MODEL and the matching compression_ratio for RadianceHDREncoder."
    )

    def apply(
        self,
        model,
        lora_dict: dict,
        strength: float = 1.0,
        model_hint: str = "",
    ):
        import copy

        tensors = lora_dict.get("tensors", {})
        meta    = lora_dict.get("metadata", {})

        lora_model_name     = str(meta.get("radiance_model_name", "unknown"))
        compression_ratio   = float(meta.get("radiance_compression_ratio", 0.5))

        # ── Cross-check model hint ─────────────────────────────────────────
        if model_hint and model_hint.strip():
            resolved = _resolve_model(model_hint.strip())
            if resolved and resolved != lora_model_name:
                logger.warning(
                    "RadianceHDRLoRAApply: LoRA was trained on '%s' but model_hint='%s' "
                    "resolved to '%s'.  Applying anyway — results may be inconsistent.",
                    lora_model_name, model_hint, resolved,
                )
            elif _HAS_PRESETS and resolved is None:
                logger.info(
                    "RadianceHDRLoRAApply: model_hint='%s' not found in preset table; "
                    "using compression_ratio=%.3f from LoRA metadata.",
                    model_hint, compression_ratio,
                )
            # If hint resolved to lora_model_name → all good, no warning needed

        # ── Skip if strength == 0 ─────────────────────────────────────────
        if abs(strength) < 1e-6:
            logger.info("RadianceHDRLoRAApply: strength=0, skipping apply.")
            return (model, compression_ratio)

        # ── Clone the ComfyUI model wrapper so we don't mutate the original ──
        try:
            patched_model = model.clone()
        except AttributeError:
            # Fallback for non-ComfyUI model objects (tests, custom pipelines)
            patched_model = copy.deepcopy(model)

        # ── Apply LoRA deltas ──────────────────────────────────────────────
        _apply_lora_to_model(patched_model, tensors, strength)

        diag_logger.info(
            "HDR_LORA_APPLY_DONE lora_model=%s hint=%s strength=%.3f compression_ratio=%.3f",
            lora_model_name, model_hint or "none", strength, compression_ratio,
        )

        return (patched_model, compression_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceHDRLoRALoader": RadianceHDRLoRALoader,
    "RadianceHDRLoRAApply":  RadianceHDRLoRAApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRLoRALoader": "◎ Radiance HDR LoRA Loader",
    "RadianceHDRLoRAApply":  "◎ Radiance HDR LoRA Apply",
}
