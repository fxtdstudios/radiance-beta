"""
nodes_resolution.py — BACKWARD-COMPATIBILITY SHIM
=================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_resolution`` continue to work without modification.

New code should import from::

    radiance.nodes.generate.resolution

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_resolution is deprecated; "
    "import from radiance.nodes.generate.resolution instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.generate.resolution import *  # noqa: F401, F403
