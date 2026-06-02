"""
nodes_video_hdr.py — BACKWARD-COMPATIBILITY SHIM
================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_video_hdr`` continue to work without modification.

New code should import from::

    radiance.nodes.video.hdr

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_video_hdr is deprecated; "
    "import from radiance.nodes.video.hdr instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.video.hdr import *  # noqa: F401, F403
