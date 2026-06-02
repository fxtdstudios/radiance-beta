"""Radiance — HDR/VFX/Color pipeline for ComfyUI."""
from __future__ import annotations

import os
import sys


def _bootstrap_package_context() -> None:
    """Make relative imports reliable when ComfyUI loads this file directly."""

    global __package__, __path__

    if not __package__:
        __package__ = "radiance"
        __path__ = [os.path.dirname(os.path.abspath(__file__))]

    sys.modules.setdefault("radiance", sys.modules.get(__name__, type(sys)(__name__)))


_bootstrap_package_context()

from .config.constants import AUTHOR, VERSION, WEB_DIRECTORY
from .config.dependencies import validate_runtime_dependencies
from .config.env import configure_runtime_environment
from .core.logging import setup_radiance_logging
from .nodes.registry import NodeModuleSpec, load_node_mappings

logger = setup_radiance_logging()


def _load_comfyui_nodes() -> tuple[dict, dict]:
    """Load the organized node catalog and optional viewer extension."""

    entrypoint_modules = (
        NodeModuleSpec(".nodes", package=__name__, required=True),
        NodeModuleSpec(".nodes_radiance_viewer", package=__name__),
    )
    load_result = load_node_mappings(
        entrypoint_modules,
        logger=logger,
        context="Radiance entry point",
    )
    return load_result.class_mappings, load_result.display_name_mappings


configure_runtime_environment()
validate_runtime_dependencies(logger)

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = _load_comfyui_nodes()

__version__ = VERSION
__author__ = AUTHOR
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

logger.info(
    "Radiance: successfully loaded %d nodes (v%s)",
    len(NODE_CLASS_MAPPINGS),
    __version__,
)
logger.debug("Radiance Viewer JavaScript extension enabled")
