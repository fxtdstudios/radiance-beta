"""
nodes_3d.py — BACKWARD-COMPATIBILITY SHIM
=========================================
This file is retained so existing workflows importing directly from
``radiance.nodes_3d`` continue to work without modification.

New code should import from::

    radiance.nodes.vfx.camera

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_3d is deprecated; "
    "import from radiance.nodes.vfx.camera instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.vfx.camera import *  # noqa: F401, F403
