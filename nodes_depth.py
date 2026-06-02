"""
nodes_depth.py — BACKWARD-COMPATIBILITY SHIM
============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_depth`` continue to work without modification.

New code should import from::

    radiance.nodes.vfx.depth

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_depth is deprecated; "
    "import from radiance.nodes.vfx.depth instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.vfx.depth import *  # noqa: F401, F403
