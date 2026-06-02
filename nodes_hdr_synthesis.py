"""
nodes_hdr_synthesis.py — BACKWARD-COMPATIBILITY SHIM
====================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_synthesis`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.synthesis

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_synthesis is deprecated; "
    "import from radiance.nodes.hdr.synthesis instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.synthesis import *  # noqa: F401, F403
