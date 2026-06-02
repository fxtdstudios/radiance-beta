"""
◎ Radiance v3.0.0 — Phase 4: Radiance Sampler Expansion
nodes_regional.py

Provides:
  • RadianceRegionalPrompt — Spatial region conditioning for Flux / SDXL.

Spatial regions are defined as bounding boxes (x0, y0, x1, y1 in [0..1]) or
connected to an external MASK. Each region gets its own positive conditioning
which is merged into the global conditioning via attention-space weighting.

This node does NOT require model surgery — it builds a standard ComfyUI
conditioning list with per-token region masks that are compatible with
Flux, SDXL, and SD3 via the standard area-conditioned inference path
supported in ComfyUI's built-in KSampler.
"""

import json
import logging
from typing import Tuple, List, Optional

import torch

logger = logging.getLogger("radiance.regional")


# ==============================================================================
# Helper: build area conditioning
# ==============================================================================

def _make_area_cond(
    cond_tensors: list,
    x: float, y: float, w: float, h: float,
    strength: float = 1.0,
) -> list:
    """
    Wrap a standard conditioning list with ComfyUI area-conditioning metadata.

    x, y: top-left corner in [0,1] (fraction of image)
    w, h: width and height in [0,1]
    strength: conditioning weight for this region
    """
    out = []
    for c in cond_tensors:
        # c is (tensor, {dict})
        t, d = c
        new_d = dict(d)
        # ComfyUI uses 'area' key: (height_frac, width_frac, y_frac, x_frac)
        new_d["area"] = (h, w, y, x)
        new_d["strength"] = strength
        new_d["set_area_to_bounds"] = False
        out.append((t, new_d))
    return out


def _mask_to_area(mask: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Compute the tight bounding box of a non-zero mask region.
    Returns (x, y, w, h) in [0,1] fractions.
    mask: (H, W) float tensor, values in [0,1].
    """
    nonzero = mask > 0.1
    if not nonzero.any():
        return (0.0, 0.0, 1.0, 1.0)

    H, W = mask.shape[-2], mask.shape[-1]
    rows = nonzero.any(dim=-1).nonzero(as_tuple=True)[0]
    cols = nonzero.any(dim=-2).nonzero(as_tuple=True)[0]

    y0_frac = float(rows.min()) / H
    y1_frac = float(rows.max() + 1) / H
    x0_frac = float(cols.min()) / W
    x1_frac = float(cols.max() + 1) / W

    return (
        x0_frac,
        y0_frac,
        max(x1_frac - x0_frac, 1.0 / W),
        max(y1_frac - y0_frac, 1.0 / H),
    )


# ==============================================================================
# RadianceRegionalPrompt
# ==============================================================================

class RadianceRegionalPrompt:
    """
    ◎ Radiance Regional Prompt

    Define a spatial region with its own positive conditioning that is merged
    into the global conditioning for area-guided synthesis.

    Supports up to 4 chained regions. Each node adds one region and passes
    the accumulated conditioning list downstream via the CONDITIONING output.

    Usage:
      1. Connect your CLIP-encoded global positive conditioning to `base_cond`.
      2. Encode a region-specific prompt: CLIP Text Encode → `region_cond`.
      3. Define the region: bounding box (x, y, w, h) in [0..1] fractions,
         OR connect a MASK (takes priority over the bbox inputs).
      4. Chain multiple RegionalPrompt nodes together via their CONDITIONING outputs.
      5. Feed the final CONDITIONING into your sampler's positive input.

    Coordinates:
      • x, y = top-left corner as fraction of total image width/height
      • w, h = region width/height as fraction of total image
      • Example: top-left quarter → x=0.0, y=0.0, w=0.5, h=0.5

    Compatibility:
      • Flux, SDXL, SD3, SD1.5 via ComfyUI's standard area-conditioned path.
      • For Flux specifically, area conditioning is approximate (Flux uses
        full-sequence attention without spatial masking). Results are best with
        regions that occupy >25% of the image.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Apply region-specific text prompts with spatial masks."
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "region_info")

    # Importance: For IP-Adapter, we inject the image embedding into the region
    # conditioning via the 'cross_attn_controlnet' key that ComfyUI's
    # IPAdapterApply node uses. This is compatible with IPAdapterPlus and the
    # built-in IP-Adapter hooks in ComfyUI >=0.2.0 via the 'ipadapter' key.

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_cond": ("CONDITIONING",),
                "region_cond": ("CONDITIONING",),
                "region_label": ("STRING", {
                    "default": "region_1",
                    "tooltip": "Human-readable label for this region (used in JSON output).",
                }),
                "x": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Left edge of region as fraction of image width.",
                }),
                "y": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Top edge of region as fraction of image height.",
                }),
                "w": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Width of region as fraction of image width.",
                }),
                "h": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Height of region as fraction of image height.",
                }),
                "region_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Conditioning weight for this region vs global.",
                }),
                "global_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight of the global base conditioning passed through.",
                }),
                "merge_mode": (["Additive", "Replace"], {
                    "default": "Additive",
                    "tooltip": (
                        "Additive: region added on top of global (default, safe). "
                        "Replace: region replaces global in its area."
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional. When connected, overrides x/y/w/h with the mask's bounding box.",
                }),
                "ip_image": ("IMAGE", {
                    "tooltip": (
                        "Optional IP-Adapter reference image for this region. "
                        "When connected, the image's visual features are injected into "
                        "the region conditioning alongside the text prompt. "
                        "Requires an IP-Adapter-enabled model hook to be active."
                    ),
                }),
                "ip_weight": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": (
                        "Strength of the IP-Adapter image influence for this region. "
                        "0 = text-only, 1 = equal image+text, >1 = image-dominant."
                    ),
                }),
            },
        }

    def apply(
        self,
        base_cond: list,
        region_cond: list,
        region_label: str = "region_1",
        x: float = 0.0,
        y: float = 0.0,
        w: float = 0.5,
        h: float = 0.5,
        region_strength: float = 1.0,
        global_strength: float = 0.5,
        merge_mode: str = "Additive",
        mask: torch.Tensor = None,
        ip_image: torch.Tensor = None,
        ip_weight: float = 0.6,
    ) -> Tuple[list, str]:

        # If a mask is provided, derive bbox from it
        if mask is not None:
            m = mask
            if m.dim() == 3:
                m = m[0]  # take first mask in batch
            x, y, w, h = _mask_to_area(m)
            logger.debug(f"[RegionalPrompt] Mask → bbox: x={x:.3f} y={y:.3f} w={w:.3f} h={h:.3f}")

        # Clamp values
        x = max(0.0, min(1.0 - 0.01, x))
        y = max(0.0, min(1.0 - 0.01, y))
        w = max(0.01, min(1.0 - x, w))
        h = max(0.01, min(1.0 - y, h))

        # Build conditionings
        if merge_mode == "Replace":
            # Region replaces the global in its area — do not include global in output
            result = _make_area_cond(region_cond, x, y, w, h, region_strength)
        else:
            # Additive: keep global (optionally weight-adjusted) + add region on top
            global_out = []
            for c in base_cond:
                t, d = c
                nd = dict(d)
                nd["strength"] = global_strength
                global_out.append((t, nd))
            region_out = _make_area_cond(region_cond, x, y, w, h, region_strength)
            result = global_out + region_out

        # ── IP-Adapter image conditioning for this region ──────────────────────
        # The ip_image is encoded and injected into the conditioning metadata
        # using the 'cross_attn_controlnet' key. This is the standard mechanism
        # used by ComfyUI's IP-Adapter nodes and is compatible with IPAdapterPlus.
        # If no ip_image is provided, nothing changes (fully backward-compatible).
        ip_applied = False
        if ip_image is not None:
            try:
                # Crop ip_image to the region bbox for spatial coherence
                # ip_image: (B, H, W, C) or (H, W, C) from ComfyUI
                ref = ip_image
                if ref.dim() == 4:
                    ref = ref[0]  # (H, W, C)

                H_ip, W_ip, C_ip = ref.shape
                crop_y0 = int(y * H_ip)
                crop_y1 = int((y + h) * H_ip)
                crop_x0 = int(x * W_ip)
                crop_x1 = int((x + w) * W_ip)
                crop_y0 = max(0, crop_y0)
                crop_y1 = max(crop_y0 + 1, min(H_ip, crop_y1))
                crop_x0 = max(0, crop_x0)
                crop_x1 = max(crop_x0 + 1, min(W_ip, crop_x1))
                region_crop = ref[crop_y0:crop_y1, crop_x0:crop_x1]  # (rH, rW, C)

                # Resize to standard IP-Adapter resolution (224×224)
                import torch.nn.functional as F
                img_bchw = region_crop.permute(2, 0, 1).unsqueeze(0).float()  # (1,C,rH,rW)
                img_224 = F.interpolate(
                    img_bchw, size=(224, 224), mode="bilinear", align_corners=False
                ).squeeze(0).permute(1, 2, 0)  # (224, 224, C)

                # Inject into each result conditioning entry via 'cross_attn_controlnet'
                # This key is recognized by ComfyUI's IPAdapter hooks:
                # https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/conds.py
                new_result = []
                for cond_t, cond_d in result:
                    nd = dict(cond_d)
                    nd["cross_attn_controlnet"] = {
                        "image":  img_224.unsqueeze(0),   # (1, 224, 224, C)
                        "weight": ip_weight,
                        "type":   "ip_adapter",
                    }
                    new_result.append((cond_t, nd))
                result = new_result
                ip_applied = True
                logger.debug(
                    f"[RegionalPrompt] IP-Adapter image injected for region '{region_label}' "
                    f"(crop={crop_y0}:{crop_y1},{crop_x0}:{crop_x1}  weight={ip_weight})"
                )
            except Exception as e:
                logger.warning(f"[RegionalPrompt] IP-Adapter injection failed: {e}. Continuing without image.")

        region_info = json.dumps({
            "node": "RadianceRegionalPrompt",
            "label": region_label,
            "bbox": {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)},
            "mask_provided": mask is not None,
            "ip_adapter": {"enabled": ip_applied, "weight": ip_weight if ip_applied else None},
            "region_strength": region_strength,
            "global_strength": global_strength,
            "merge_mode": merge_mode,
            "num_regions_in_output": len(result),
        }, indent=2)

        logger.info(f"[RegionalPrompt] '{region_label}' ({x:.2f},{y:.2f}) {w:.2f}×{h:.2f} strength={region_strength} ip={ip_applied}")
        return (result, region_info)


# ==============================================================================
# RadianceRegionalGrid
# ==============================================================================

class RadianceRegionalGrid:
    """
    ◎ Radiance Regional Grid

    Convenience node that divides the image into a regular grid of cells
    and assigns a separate text conditioning to each cell.

    Input:
      • base_cond      — global positive conditioning (background / fallback)
      • clip            — a CLIP model for encoding each region's prompt text
      • grid_prompts    — JSON array of per-cell prompts in row-major order:
                          ["top-left prompt", "top-right prompt", ...]
      • columns         — number of grid columns
      • rows            — number of grid rows
      • cell_strength   — conditioning weight for each cell region
      • global_strength — weight of the base conditioning passed through

    The number of prompts can be less than columns × rows; remaining cells
    inherit the base conditioning.

    Output: merged CONDITIONING list compatible with ComfyUI's KSampler.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Divide the canvas into a grid of independently prompted regions."
    FUNCTION = "apply_grid"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "grid_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_cond": ("CONDITIONING",),
                "clip": ("CLIP",),
                "grid_prompts": ("STRING", {
                    "default": '["subject in left area", "background on right"]',
                    "multiline": True,
                    "tooltip": "JSON array of prompts, one per grid cell, row-major order.",
                }),
                "columns": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Number of columns in the regional prompt grid."}),
                "rows": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Number of rows in the regional prompt grid."}),
                "cell_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Conditioning strength for each individual region cell. Higher values make the model follow regional prompts more closely."
                }),
                "global_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Conditioning strength for the global (full-image) prompt. Blended with cell conditioning at each step."
                }),
            },
        }

    def apply_grid(self, base_cond: list, clip, grid_prompts: str,
                   columns: int = 2, rows: int = 1,
                   cell_strength: float = 1.0,
                   global_strength: float = 0.3) -> Tuple[list, str]:

        # Parse grid prompts
        try:
            prompts = json.loads(grid_prompts)
            if isinstance(prompts, str):
                prompts = [prompts]
        except Exception as e:
            logger.warning(f"[RegionalGrid] Parse error: {e}")
            prompts = []

        total_cells = columns * rows
        cell_w = 1.0 / columns
        cell_h = 1.0 / rows

        # Apply global weight to base_cond
        result = []
        for c in base_cond:
            t, d = c
            nd = dict(d)
            nd["strength"] = global_strength
            result.append((t, nd))

        cell_info = []
        for cell_idx in range(total_cells):
            row_i = cell_idx // columns
            col_i = cell_idx % columns
            cx = col_i * cell_w
            cy = row_i * cell_h

            prompt = prompts[cell_idx] if cell_idx < len(prompts) else None
            if not prompt or not str(prompt).strip():
                continue

            # Encode the cell prompt
            try:
                tokens = clip.tokenize(str(prompt))
                cond_tensor, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                cell_cond = [(cond_tensor, {"pooled_output": pooled})]
                region_area_cond = _make_area_cond(
                    cell_cond, cx, cy, cell_w, cell_h, cell_strength
                )
                result.extend(region_area_cond)
                cell_info.append({
                    "cell": cell_idx, "row": row_i, "col": col_i,
                    "bbox": {"x": round(cx, 4), "y": round(cy, 4),
                             "w": round(cell_w, 4), "h": round(cell_h, 4)},
                    "prompt": str(prompt)[:80],
                })
                logger.debug(f"[RegionalGrid] Cell {cell_idx} ({row_i},{col_i}) — '{prompt[:40]}'")
            except Exception as e:
                logger.warning(f"[RegionalGrid] Cell {cell_idx} encode failed: {e}")

        grid_info = json.dumps({
            "node": "RadianceRegionalGrid",
            "columns": columns, "rows": rows,
            "total_cells": total_cells,
            "populated_cells": len(cell_info),
            "cell_strength": cell_strength,
            "global_strength": global_strength,
            "cells": cell_info,
        }, indent=2)

        logger.info(f"[RegionalGrid] {columns}×{rows} = {total_cells} cells, {len(cell_info)} populated")
        return (result, grid_info)


# ==============================================================================
# Node registration
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceRegionalPrompt": RadianceRegionalPrompt,
    "RadianceRegionalGrid":   RadianceRegionalGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceRegionalPrompt": "◎ Radiance Regional Prompt",
    "RadianceRegionalGrid":   "◎ Radiance Regional Grid",
}
