"""
nodes_optics.py — BACKWARD-COMPATIBILITY SHIM
=============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_optics`` continue to work without modification.

New code should import from::

    radiance.nodes.vfx.optics

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_optics is deprecated; "
    "import from radiance.nodes.vfx.optics instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.vfx.optics import *  # noqa: F401, F403
