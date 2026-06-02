"""
nodes_scene_cut.py — BACKWARD-COMPATIBILITY SHIM
================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_scene_cut`` continue to work without modification.

New code should import from::

    radiance.nodes.ai.scene_cut

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_scene_cut is deprecated; "
    "import from radiance.nodes.ai.scene_cut instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.ai.scene_cut import *  # noqa: F401, F403
# Explicitly import private variables needed by unit tests
from radiance.nodes.ai.scene_cut import (
    _luminance,
    _histogram_diff,
    _edge_diff,
)
