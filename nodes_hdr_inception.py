"""
nodes_hdr_inception.py — BACKWARD-COMPATIBILITY SHIM
====================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_inception`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.inception

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_inception is deprecated; "
    "import from radiance.nodes.hdr.inception instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.inception import *  # noqa: F401, F403
