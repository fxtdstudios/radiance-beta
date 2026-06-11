"""Interactive 3D splat viewer node — WebGL rendering in the browser (no GPU node-side)."""
from __future__ import annotations

import os
import uuid

import numpy as np

from radiance.splatting.data import Splat
from radiance.splatting.splat_format import save_splat

_CATEGORY = "FXTD STUDIOS/Radiance/Gaussian Splatting"


def _temp_dir() -> str:
    try:
        import folder_paths
        return folder_paths.get_temp_directory()
    except Exception:  # outside ComfyUI (tests)
        import tempfile
        return tempfile.gettempdir()


def _subsample(splat: Splat, max_points: int) -> Splat:
    if splat.count <= max_points:
        return splat
    idx = np.random.default_rng(0).choice(splat.count, int(max_points), replace=False)
    idx.sort()
    return Splat(splat.means[idx], splat.scales[idx], splat.quats[idx],
                 splat.opacities[idx], splat.sh[idx], splat.sh_degree,
                 meta={**splat.meta, "subsampled": int(max_points)})


class RadianceSplatViewer3D:
    """Orbit/pan/zoom a SPLAT interactively, rendered in the browser via WebGL."""

    CATEGORY = _CATEGORY
    DESCRIPTION = ("Interactive 3D viewer for a SPLAT — orbit, pan, and zoom in the node. "
                   "Renders in the browser (WebGL); works without gsplat or a CUDA GPU.")
    FUNCTION = "show"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "splat": ("SPLAT",),
                "max_points": ("INT", {
                    "default": 500000, "min": 1000, "max": 4000000, "step": 1000,
                    "tooltip": "Splats above this count are randomly subsampled to keep the viewer responsive.",
                }),
            }
        }

    def show(self, splat, max_points):
        sub = _subsample(splat, int(max_points))
        out_dir = _temp_dir()
        os.makedirs(out_dir, exist_ok=True)
        name = f"radiance_splat_{uuid.uuid4().hex[:10]}.splat"
        save_splat(sub, os.path.join(out_dir, name))
        center = sub.means.mean(axis=0)
        radius = float(np.linalg.norm(sub.means - center, axis=1).mean()) or 1.0
        return {"ui": {
            "splat_file": [name],
            "splat_count": [int(sub.count)],
            "splat_total": [int(splat.count)],
            "center": [[float(center[0]), float(center[1]), float(center[2])]],
            "radius": [radius],
        }}


NODE_CLASS_MAPPINGS = {"RadianceSplatViewer3D": RadianceSplatViewer3D}
NODE_DISPLAY_NAME_MAPPINGS = {"RadianceSplatViewer3D": "Splat Viewer 3D"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
