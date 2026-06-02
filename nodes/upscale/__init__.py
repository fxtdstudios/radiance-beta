"""Upscaling node group."""
from __future__ import annotations

import logging

from radiance.nodes.upscale.upscale import (
    RadianceUpscaleTiler,
    RadianceUpscaleImage,
    RadianceUpscaleVideo,
    RadianceUpscaleFaceRestore,
)

logger = logging.getLogger("radiance.nodes.upscale")

NODE_CLASS_MAPPINGS = {
    "RadianceUpscaleTiler": RadianceUpscaleTiler,
    "RadianceUpscaleImage": RadianceUpscaleImage,
    "RadianceUpscaleVideo": RadianceUpscaleVideo,
    "RadianceUpscaleFaceRestore": RadianceUpscaleFaceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceUpscaleTiler": "◎ Upscale Tiler",
    "RadianceUpscaleImage": "◎ Upscale Image",
    "RadianceUpscaleVideo": "◎ Upscale Video",
    "RadianceUpscaleFaceRestore": "◎ Upscale Face Restore",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
