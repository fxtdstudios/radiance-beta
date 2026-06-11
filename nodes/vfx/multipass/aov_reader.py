"""
◎ Radiance Multipass AOV Reader

Reads a *real* multilayer OpenEXR (renderer AOVs) and splits its named layers
into the same outputs as RadianceMultipassMaster — so genuine render passes
(diffuse/albedo, N, P, Z, specular, AO, cryptomatte, …) flow straight into the
same EXR-passes writer and relight/comp chain.

This is the "make it real" counterpart to the Master extractor: instead of
*estimating* passes from a 2D beauty, it ingests ground-truth layers produced by
Arnold / Redshift / Karma / Cycles / V-Ray, preserving scene-linear values
(no normalization). Layers that are absent in the EXR come through as black, so
the Master node can still be used to gap-fill the missing ones.

Channel/layer mapping is alias-based and case-insensitive (e.g. "diffuse",
"diffuse_color", "albedo", "basecolor" all map to the albedo slot). A per-read
report of which layers were found and how they mapped is logged.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("radiance.vfx_multipass.aov_reader")

# Output slot -> accepted layer-name aliases (all lowercased, dots/spaces/dashes
# stripped to underscores before matching).
_LAYER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "beauty":         ("", "rgba", "rgb", "beauty", "composite", "combined", "final"),
    "albedo":         ("albedo", "diffuse", "diffuse_color", "diffusecolor", "basecolor", "base_color", "diff", "diffuse_albedo"),
    "normal":         ("n", "normal", "normals", "normal_world", "worldnormal", "world_normal"),
    "depth":          ("z", "depth", "zdepth", "z_depth"),
    "roughness":      ("roughness", "rough"),
    "specular":       ("specular", "spec", "reflect", "reflection_color"),
    "metallic":       ("metallic", "metalness", "metal"),
    "ao":             ("ao", "occlusion", "ambient_occlusion", "ambocc"),
    "emission":       ("emission", "emit", "emissive", "incandescence", "self_illumination"),
    "transmission":   ("transmission", "refraction", "transmissive", "refract"),
    "highpass":       ("highpass", "high_pass"),
    "world_position": ("p", "position", "worldposition", "world_position", "point", "world_point"),
    "curvature":      ("curvature", "curv"),
    "shadow_mask":    ("shadow", "shadow_mask", "shadows"),
    "midtone_mask":   ("midtone", "midtone_mask", "midtones"),
    "highlight_mask": ("highlight", "highlight_mask", "highlights"),
    "reflection_mask": ("reflection_mask", "reflectionmask"),
    "motion_vector":  ("mv", "motion", "motionvector", "motion_vector", "velocity", "vector", "motionvectors"),
    "object_id":      ("id", "crypto", "cryptomatte", "objectid", "object_id", "segmentation", "matteid"),
}

# Output order MUST match RadianceMultipassMaster so this node is a drop-in.
_OUTPUT_ORDER: Tuple[str, ...] = (
    "beauty", "albedo", "normal", "depth", "roughness", "specular", "metallic",
    "ao", "emission", "transmission", "highpass", "world_position", "curvature",
    "shadow_mask", "midtone_mask", "highlight_mask", "reflection_mask",
    "motion_vector", "object_id",
)


def _norm_layer_name(name: str) -> str:
    return name.strip().lower().replace(".", "_").replace(" ", "_").replace("-", "_")


def _read_multilayer_exr(path: str) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Return ({layer_name: (H,W,C) float32}, H, W) for a multilayer EXR.

    Channels are grouped by the text before the final '.'; the default layer
    (channels with no '.') is keyed as "". Channel suffixes are ordered
    canonically (R,G,B,A then X,Y,Z then alphabetical) so colour and vector
    layers assemble predictably. Scene-linear values are preserved exactly.
    """
    try:
        import OpenEXR  # type: ignore
        import Imath    # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading multilayer EXR requires the OpenEXR + Imath packages "
            "(pip install OpenEXR Imath)."
        ) from exc

    if not os.path.isfile(path):
        raise RuntimeError(f"EXR not found: {path}")

    f = OpenEXR.InputFile(path)
    try:
        hdr = f.header()
        dw = hdr["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = list(hdr["channels"].keys())

        # Group channels by layer prefix.
        groups: Dict[str, Dict[str, str]] = {}
        for ch in channels:
            if "." in ch:
                layer, suffix = ch.rsplit(".", 1)
            else:
                layer, suffix = "", ch
            groups.setdefault(layer, {})[suffix] = ch

        def _suffix_rank(suf: str) -> Tuple[int, str]:
            order = {"R": 0, "G": 1, "B": 2, "A": 3, "X": 0, "Y": 1, "Z": 2}
            s = suf.upper()
            return (order.get(s, 99), suf)

        layers: Dict[str, np.ndarray] = {}
        for layer, suffices in groups.items():
            ordered = sorted(suffices.items(), key=lambda kv: _suffix_rank(kv[0]))
            planes = []
            for _suf, chname in ordered:
                buf = f.channel(chname, pt)
                planes.append(np.frombuffer(buf, dtype=np.float32).reshape(h, w))
            arr = np.stack(planes, axis=-1)  # (H, W, C)
            layers[layer] = arr
        return layers, h, w
    finally:
        f.close()


def _to_image_tensor(arr: Optional[np.ndarray], h: int, w: int) -> torch.Tensor:
    """Coerce a layer to a (1, H, W, 3) float32 IMAGE tensor (scene-linear preserved)."""
    if arr is None:
        return torch.zeros(1, h, w, 3, dtype=torch.float32)
    a = np.ascontiguousarray(arr.astype(np.float32))
    if a.ndim == 2:
        a = a[..., None]
    c = a.shape[2]
    if c == 1:
        a = np.repeat(a, 3, axis=2)          # scalar -> grey RGB
    elif c == 2:
        a = np.concatenate([a, np.zeros_like(a[..., :1])], axis=2)  # XY -> XY0
    elif c >= 4:
        a = a[..., :3]                        # drop alpha/extra for IMAGE slot
    return torch.from_numpy(a).unsqueeze(0)


class RadianceMultipassAOVReader:
    """◎ Multipass: AOV Reader — split a real multilayer EXR into Radiance passes."""

    CATEGORY = "FXTD STUDIOS/Radiance/VFX"
    DESCRIPTION = (
        "Read a real multilayer/AOV OpenEXR and split its named layers into the "
        "same passes as the Master extractor. Ground-truth renderer passes — not "
        "estimates. Missing layers come through black."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a multilayer/AOV EXR (Arnold/Redshift/Karma/Cycles/V-Ray).",
                }),
            },
            "optional": {
                "beauty_layer": ("STRING", {"default": "auto",
                    "tooltip": "Override the beauty layer name, or 'auto' to detect (RGBA / 'beauty')."}),
                "albedo_layer": ("STRING", {"default": "auto",
                    "tooltip": "Override the albedo/diffuse layer name, or 'auto'."}),
                "normal_layer": ("STRING", {"default": "auto",
                    "tooltip": "Override the normal layer name, or 'auto'."}),
                "depth_layer": ("STRING", {"default": "auto",
                    "tooltip": "Override the depth (Z) layer name, or 'auto'."}),
            },
        }

    # Mirrors RadianceMultipassMaster exactly so the two are interchangeable.
    RETURN_TYPES = (
        "RADIANCE_PASSES", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
    )
    RETURN_NAMES = (
        "passes", "beauty", "albedo", "normal_map", "depth", "roughness", "specular",
        "metallic", "ao", "emission", "transmission", "highpass", "world_position",
        "curvature", "shadow_mask", "midtone_mask", "highlight_mask", "reflection_mask",
        "motion_vector", "segmentation_id",
    )
    FUNCTION = "read_passes"

    def read_passes(
        self,
        exr_path: str,
        beauty_layer: str = "auto",
        albedo_layer: str = "auto",
        normal_layer: str = "auto",
        depth_layer: str = "auto",
    ) -> Tuple:
        layers, h, w = _read_multilayer_exr(exr_path.strip())

        # Normalized lookup of available layers.
        norm_to_raw = {_norm_layer_name(k): k for k in layers}

        manual = {
            "beauty": beauty_layer, "albedo": albedo_layer,
            "normal": normal_layer, "depth": depth_layer,
        }

        resolved: Dict[str, Optional[np.ndarray]] = {}
        report: List[str] = []
        used_layers = set()

        for slot in _OUTPUT_ORDER:
            chosen_raw: Optional[str] = None

            # 1. explicit manual override (exact or normalized match)
            ov = manual.get(slot, "auto")
            if ov and ov.strip().lower() not in ("", "auto"):
                key = _norm_layer_name(ov)
                chosen_raw = norm_to_raw.get(key) or (ov if ov in layers else None)

            # 2. alias auto-match: exact first, then prefix for multi-char aliases
            #    so 'crypto' matches 'crypto00'/'crypto01', 'diffuse' matches
            #    'diffuse_indirect', etc. Short aliases (n, z, p, id, mv) stay exact.
            if chosen_raw is None:
                avail_sorted = sorted(norm_to_raw)
                for alias in _LAYER_ALIASES.get(slot, ()):  # ordered by preference
                    if alias in norm_to_raw:
                        chosen_raw = norm_to_raw[alias]
                        break
                    if len(alias) >= 4:
                        pref = next((n for n in avail_sorted if n.startswith(alias)), None)
                        if pref is not None:
                            chosen_raw = norm_to_raw[pref]
                            break

            if chosen_raw is not None and chosen_raw in layers:
                resolved[slot] = layers[chosen_raw]
                used_layers.add(chosen_raw)
                report.append(f"{slot} <- '{chosen_raw or 'RGBA'}'")
            else:
                resolved[slot] = None
                report.append(f"{slot} <- (none, black)")

        # Beauty fallback: if no beauty layer matched, use the first colour layer.
        if resolved["beauty"] is None:
            for raw, arr in layers.items():
                if arr.shape[-1] >= 3:
                    resolved["beauty"] = arr
                    used_layers.add(raw)
                    report[0] = f"beauty <- '{raw or 'RGBA'}' (fallback)"
                    break

        unmapped = sorted(set(layers) - used_layers)
        logger.info(
            "[Radiance AOV Reader] %s (%dx%d): %d layers, mapped: %s%s",
            os.path.basename(exr_path), w, h, len(layers), "; ".join(report),
            f"  | unmapped: {', '.join(repr(u) for u in unmapped)}" if unmapped else "",
        )

        images = {slot: _to_image_tensor(resolved[slot], h, w) for slot in _OUTPUT_ORDER}

        passes_dict = dict(images)
        # Keep the internal dict key as object_id for compatibility with the
        # EXR-passes writer / relight nodes, while the socket label reads
        # 'segmentation_id' to avoid implying real Cryptomatte support.
        passes_dict["object_id"] = images["object_id"]

        return (passes_dict,) + tuple(images[slot] for slot in _OUTPUT_ORDER)


NODE_CLASS_MAPPINGS = {
    "RadianceMultipassAOVReader": RadianceMultipassAOVReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMultipassAOVReader": "Multipass AOV Reader",
}
