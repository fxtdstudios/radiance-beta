"""
nodes_overlay.py — BACKWARD-COMPATIBILITY SHIM
==============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_overlay`` continue to work without modification.

New code should import from::

    radiance.nodes.pipeline.overlay

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_overlay is deprecated; "
    "import from radiance.nodes.pipeline.overlay instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.pipeline.overlay import *  # noqa: F401, F403
