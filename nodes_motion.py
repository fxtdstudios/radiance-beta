"""
nodes_motion.py — BACKWARD-COMPATIBILITY SHIM
=============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_motion`` continue to work without modification.

New code should import from::

    radiance.nodes.vfx.motion

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_motion is deprecated; "
    "import from radiance.nodes.vfx.motion instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.vfx.motion import *  # noqa: F401, F403
