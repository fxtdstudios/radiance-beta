"""
nodes_hdr_patch.py — BACKWARD-COMPATIBILITY SHIM
================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_patch`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.patch

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_patch is deprecated; "
    "import from radiance.nodes.hdr.patch instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.patch import *  # noqa: F401, F403
