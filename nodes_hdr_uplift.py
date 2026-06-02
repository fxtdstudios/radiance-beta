"""
nodes_hdr_uplift.py — BACKWARD-COMPATIBILITY SHIM
=================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_uplift`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.uplift

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_uplift is deprecated; "
    "import from radiance.nodes.hdr.uplift instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.uplift import *  # noqa: F401, F403
