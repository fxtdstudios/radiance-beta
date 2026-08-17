"""Upscaling node group."""
from __future__ import annotations

import logging

from radiance.nodes.upscale.upscale import (
    RadianceUpscaleTiler,
    RadianceUpscaleImage,
    RadianceUpscaleVideo,
    RadianceUpscaleFaceRestore,
)
from radiance.nodes.aggregate import fold_in_module_nodes

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

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
