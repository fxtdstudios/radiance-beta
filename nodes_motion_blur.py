"""
nodes_motion_blur.py — BACKWARD-COMPATIBILITY SHIM
==================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_motion_blur`` continue to work without modification.

New code should import from::

    radiance.nodes.vfx.motion_blur

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_motion_blur is deprecated; "
    "import from radiance.nodes.vfx.motion_blur instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.vfx.motion_blur import *  # noqa: F401, F403
