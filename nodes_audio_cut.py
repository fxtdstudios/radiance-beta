"""
nodes_audio_cut.py — BACKWARD-COMPATIBILITY SHIM
================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_audio_cut`` continue to work without modification.

New code should import from::

    radiance.nodes.pipeline.audio

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_audio_cut is deprecated; "
    "import from radiance.nodes.pipeline.audio instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.pipeline.audio import *  # noqa: F401, F403
