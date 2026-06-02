"""
nodes_metadata.py — BACKWARD-COMPATIBILITY SHIM
===============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_metadata`` continue to work without modification.

New code should import from::

    radiance.nodes.pipeline.metadata

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_metadata is deprecated; "
    "import from radiance.nodes.pipeline.metadata instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.pipeline.metadata import *  # noqa: F401, F403
