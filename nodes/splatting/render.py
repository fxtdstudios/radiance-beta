"""Gaussian Splatting render nodes — camera generation and gsplat rendering.

Camera generation is CPU/numpy. Rendering needs gsplat + a CUDA GPU; the node
imports the backend lazily and raises a clear error if either is missing, so the
node always registers and the rest of the pack is unaffected.
"""
from __future__ import annotations

from radiance.splatting.cameras import orbit

_CATEGORY = "FXTD STUDIOS/Radiance/Gaussian Splatting"


class RadianceCameraOrbit:
    """Generate a horizontal orbit of cameras looking at a center point."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Build an orbit camera rig (RAD_CAMERAS) for rendering a splat."
    FUNCTION = "build"
    RETURN_TYPES = ("RAD_CAMERAS",)
    RETURN_NAMES = ("cameras",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_frames": ("INT", {"default": 60, "min": 1, "max": 2048}),
                "radius": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 1000.0, "step": 0.05}),
                "elevation": ("FLOAT", {"default": 15.0, "min": -89.0, "max": 89.0, "step": 1.0}),
                "fov": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 170.0, "step": 1.0}),
                "width": ("INT", {"default": 960, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 540, "min": 16, "max": 8192, "step": 8}),
                "center_x": ("FLOAT", {"default": 0.0, "step": 0.05}),
                "center_y": ("FLOAT", {"default": 0.0, "step": 0.05}),
                "center_z": ("FLOAT", {"default": 0.0, "step": 0.05}),
            }
        }

    def build(self, num_frames, radius, elevation, fov, width, height,
              center_x, center_y, center_z):
        cams = orbit(
            num_frames=num_frames, radius=radius, elevation_deg=elevation,
            center=(center_x, center_y, center_z), fov_deg=fov,
            width=width, height=height,
        )
        return (cams,)


class RadianceSplatRender:
    """Render a SPLAT from a camera rig into image, depth, and alpha (needs gsplat + CUDA)."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Render a SPLAT through a camera rig (gsplat/CUDA) -> IMAGE + depth + alpha."
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "depth", "alpha")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "splat": ("SPLAT",),
                "cameras": ("RAD_CAMERAS",),
                "bg_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bg_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bg_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def render(self, splat, cameras, bg_r, bg_g, bg_b):
        import torch
        from radiance.splatting.backend import render as _render

        image, depth, alpha = _render(splat, cameras, background=(bg_r, bg_g, bg_b))
        # Normalize depth to a viewable grayscale IMAGE (B,H,W,3).
        dmin = depth.amin(dim=(1, 2, 3), keepdim=True)
        dmax = depth.amax(dim=(1, 2, 3), keepdim=True)
        depth_img = ((depth - dmin) / (dmax - dmin + 1e-8)).repeat(1, 1, 1, 3)
        mask = alpha.squeeze(-1)  # (B,H,W) -> MASK
        return (image, depth_img, mask)


NODE_CLASS_MAPPINGS = {
    "RadianceCameraOrbit": RadianceCameraOrbit,
    "RadianceSplatRender": RadianceSplatRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceCameraOrbit": "Camera Orbit",
    "RadianceSplatRender": "Splat Render",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
