"""
nodes_t2v_pipeline.py — BACKWARD-COMPATIBILITY SHIM
===================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_t2v_pipeline`` continue to work without modification.

New code should import from::

    radiance.nodes.video.t2v

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_t2v_pipeline is deprecated; "
    "import from radiance.nodes.video.t2v instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.video.t2v import *  # noqa: F401, F403
