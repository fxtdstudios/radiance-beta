"""
nodes_radiance_mask.py — BACKWARD-COMPATIBILITY SHIM
====================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_radiance_mask`` continue to work without modification.

New code should import from::

    radiance.nodes.io.mask

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_radiance_mask is deprecated; "
    "import from radiance.nodes.io.mask instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.io.mask import *  # noqa: F401, F403
