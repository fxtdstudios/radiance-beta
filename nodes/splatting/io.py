"""Gaussian Splatting IO nodes — load, inspect, and export .ply splats.

These are CPU-only (numpy) and have no GPU/gsplat dependency; rendering and
training nodes are added in later phases.
"""
from __future__ import annotations

import os

from radiance.splatting.ply import load_ply, save_ply

_CATEGORY = "FXTD STUDIOS/Radiance/Gaussian Splatting"


class RadianceSplatLoad:
    """Load a 3D Gaussian Splatting .ply file into a SPLAT."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Load a Gaussian Splatting .ply file into a SPLAT (CPU, no GPU required)."
    FUNCTION = "load"
    RETURN_TYPES = ("SPLAT", "STRING")
    RETURN_NAMES = ("splat", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute or ComfyUI-relative path to a Gaussian Splatting .ply file.",
                }),
            }
        }

    def load(self, ply_path: str):
        path = ply_path.strip().strip('"')
        if not path:
            raise ValueError("RadianceSplatLoad: ply_path is empty.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"RadianceSplatLoad: file not found: {path}")
        splat = load_ply(path)
        return (splat, splat.info())


class RadianceSplatInfo:
    """Report point count, SH degree, bounds, and opacity range for a SPLAT."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Inspect a SPLAT — point count, SH degree, bounding box, opacity range."
    FUNCTION = "describe"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"splat": ("SPLAT",)}}

    def describe(self, splat):
        text = splat.info()
        return {"ui": {"text": [text]}, "result": (text,)}


class RadianceSplatExport:
    """Write a SPLAT back out as a binary 3DGS .ply file."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Export a SPLAT to a binary Gaussian Splatting .ply file."
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "splat": ("SPLAT",),
                "output_path": ("STRING", {
                    "default": "output/radiance_splat.ply",
                    "tooltip": "Destination .ply path (folders are created as needed).",
                }),
            }
        }

    def export(self, splat, output_path: str):
        path = save_ply(splat, output_path.strip().strip('"'))
        return {"ui": {"text": [f"Saved {splat.count:,} gaussians -> {path}"]}, "result": (path,)}


NODE_CLASS_MAPPINGS = {
    "RadianceSplatLoad": RadianceSplatLoad,
    "RadianceSplatInfo": RadianceSplatInfo,
    "RadianceSplatExport": RadianceSplatExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSplatLoad": "Splat Load",
    "RadianceSplatInfo": "Splat Info",
    "RadianceSplatExport": "Splat Export",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
