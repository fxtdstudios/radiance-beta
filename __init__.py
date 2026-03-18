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

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

# Find all files starting with "nodes_"
node_files = glob.glob(os.path.join(os.path.dirname(__file__), "nodes_*.py"))

for file_path in node_files:
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        module = importlib.import_module(f".{module_name}", package=__name__)

        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        logger.debug(f"Loaded {module_name}")

    except Exception as e:
        logger.error(f"FAILED to load {module_name}: {e}")

# Package info
__version__ = "2.1.1"
__author__ = "Radiance"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

logger.info(
    f"Radiance: Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes (v{__version__})"
)
logger.debug("Radiance Viewer JavaScript extension enabled")
