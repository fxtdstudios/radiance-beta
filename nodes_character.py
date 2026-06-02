"""
nodes_character.py — BACKWARD-COMPATIBILITY SHIM
================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_character`` continue to work without modification.

New code should import from::

    radiance.nodes.video.character

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_character is deprecated; "
    "import from radiance.nodes.video.character instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.video.character import *  # noqa: F401, F403
# Explicitly import private variables needed by unit tests
from radiance.nodes.video.character import (
    _has,
    _cosine_sim,
    _hsv_histogram,
)
