"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE
              Professional HDR Image Processing Suite
                     Radiance © 2024-2026

GPU-accelerated nodes for HDR, color grading, film effects, and upscaling.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import glob
import importlib
import logging

# Configure module logger
logger = logging.getLogger("radiance")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Enable OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ═══════════════════════════════════════════════════════════════════════════════
#                       DEPENDENCY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def check_dependencies():
    """Check for optional dependencies and print helpful messages."""
    missing = []

    # Core dependencies (should always be available)
    try:
        import torch  # pylint: disable=unused-import
        import numpy  # pylint: disable=unused-import
        from PIL import Image  # pylint: disable=unused-import
    except ImportError as e:
        logger.error(f"CRITICAL: Missing core dependency: {e}")
        return

    # Optional: OpenEXR for EXR I/O
    try:
        import OpenEXR  # pylint: disable=unused-import
    except ImportError:
        missing.append(("OpenEXR", "EXR file support", "pip install OpenEXR"))

    # Optional: transformers for Depth Anything V2
    try:
        import transformers  # pylint: disable=unused-import
    except ImportError:
        missing.append(
            ("transformers", "Depth Map Generator", "pip install transformers")
        )

    # Optional: colour-science for advanced color
    try:
        import colour  # pylint: disable=unused-import
    except ImportError:
        missing.append(
            ("colour-science", "Advanced OCIO/color", "pip install colour-science")
        )

    # Print optional dependency status
    if missing:
        logger.info("Optional dependencies not installed:")
        for name, feature, cmd in missing:
            logger.info(f"  • {name}: {feature} (Install: {cmd})")
    else:
        logger.debug("All optional dependencies available")


check_dependencies()

# ═══════════════════════════════════════════════════════════════════════════════
#                       DYNAMIC NODE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

from . import (
    nodes_camera,
    nodes_color,
    nodes_denoise,
    nodes_depth,
    nodes_dna,
    nodes_exr,
    nodes_filmgrain,
    nodes_grade,
    nodes_hdr,
    nodes_io,
    nodes_layout,
    nodes_loader,
    nodes_lut,
    nodes_nuke,
    nodes_overlay,
    nodes_prompt,
    nodes_qc,
    nodes_radiance_mask,
    nodes_radiance_viewer,
    nodes_resolution,
    nodes_sampler,
    nodes_scopes,
    nodes_studio,
    nodes_temporal,
    nodes_text,
    nodes_upscale,
    nodes_workspace
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

modules = [
    nodes_camera,
    nodes_color,
    nodes_denoise,
    nodes_depth,
    nodes_dna,
    nodes_exr,
    nodes_filmgrain,
    nodes_grade,
    nodes_hdr,
    nodes_io,
    nodes_layout,
    nodes_loader,
    nodes_lut,
    nodes_nuke,
    nodes_overlay,
    nodes_prompt,
    nodes_qc,
    nodes_radiance_mask,
    nodes_radiance_viewer,
    nodes_resolution,
    nodes_sampler,
    nodes_scopes,
    nodes_studio,
    nodes_temporal,
    nodes_text,
    nodes_upscale,
    nodes_workspace
]

for module in modules:
    try:
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        logger.debug(f"Loaded {module.__name__}")
    except Exception as e:
        logger.error(f"FAILED to load {module.__name__}: {e}")

# Package info
__version__ = "2.2"
__author__ = "Radiance"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

logger.info(
    f"Radiance: Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes (v{__version__})"
)
logger.debug("Radiance Viewer JavaScript extension enabled")
