"""Gaussian Splatting edit nodes — transform, crop, merge. CPU-only (numpy)."""
from __future__ import annotations

from radiance.splatting.edit import crop, merge, transform

_CATEGORY = "FXTD STUDIOS/Radiance/Gaussian Splatting"


class RadianceSplatTransform:
    """Translate, rotate (XYZ Euler degrees), and uniformly scale a SPLAT."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Translate, rotate, and uniformly scale a SPLAT (gaussian orientations follow)."
    FUNCTION = "apply"
    RETURN_TYPES = ("SPLAT",)
    RETURN_NAMES = ("splat",)

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda: ("FLOAT", {"default": 0.0, "min": -1e6, "max": 1e6, "step": 0.01})
        r = lambda: ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0})
        return {
            "required": {
                "splat": ("SPLAT",),
                "translate_x": f(), "translate_y": f(), "translate_z": f(),
                "rotate_x": r(), "rotate_y": r(), "rotate_z": r(),
                "scale": ("FLOAT", {"default": 1.0, "min": 1e-4, "max": 1e4, "step": 0.01}),
            }
        }

    def apply(self, splat, translate_x, translate_y, translate_z,
              rotate_x, rotate_y, rotate_z, scale):
        out = transform(
            splat,
            translate=(translate_x, translate_y, translate_z),
            rotate_euler=(rotate_x, rotate_y, rotate_z),
            scale=scale,
        )
        return (out,)


class RadianceSplatCrop:
    """Keep only gaussians whose centres lie inside an axis-aligned box."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Crop a SPLAT to an axis-aligned bounding box."
    FUNCTION = "apply"
    RETURN_TYPES = ("SPLAT", "STRING")
    RETURN_NAMES = ("splat", "info")

    @classmethod
    def INPUT_TYPES(cls):
        lo = lambda d: ("FLOAT", {"default": d, "min": -1e6, "max": 1e6, "step": 0.01})
        return {
            "required": {
                "splat": ("SPLAT",),
                "min_x": lo(-1.0), "min_y": lo(-1.0), "min_z": lo(-1.0),
                "max_x": lo(1.0), "max_y": lo(1.0), "max_z": lo(1.0),
            }
        }

    def apply(self, splat, min_x, min_y, min_z, max_x, max_y, max_z):
        out = crop(splat, (min_x, min_y, min_z), (max_x, max_y, max_z))
        info = f"Cropped {splat.count:,} -> {out.count:,} gaussians"
        return (out, info)


class RadianceSplatMerge:
    """Concatenate two SPLATs (SH padded to the higher degree)."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Merge two SPLATs into one."
    FUNCTION = "apply"
    RETURN_TYPES = ("SPLAT", "STRING")
    RETURN_NAMES = ("splat", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"splat_a": ("SPLAT",), "splat_b": ("SPLAT",)}}

    def apply(self, splat_a, splat_b):
        out = merge(splat_a, splat_b)
        info = f"Merged {splat_a.count:,} + {splat_b.count:,} -> {out.count:,} gaussians"
        return (out, info)


NODE_CLASS_MAPPINGS = {
    "RadianceSplatTransform": RadianceSplatTransform,
    "RadianceSplatCrop": RadianceSplatCrop,
    "RadianceSplatMerge": RadianceSplatMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSplatTransform": "Splat Transform",
    "RadianceSplatCrop": "Splat Crop",
    "RadianceSplatMerge": "Splat Merge",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
