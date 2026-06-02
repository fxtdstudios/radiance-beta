"""
nodes_dit_adapter.py — BACKWARD-COMPATIBILITY SHIM
==================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_dit_adapter`` continue to work without modification.

New code should import from::

    radiance.nodes.video.dit

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_dit_adapter is deprecated; "
    "import from radiance.nodes.video.dit instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.video.dit import *  # noqa: F401, F403
# Explicitly import private variables needed by unit tests
from radiance.nodes.video.dit import (
    _get_spec,
    _MODEL_SPECS,
    MODEL_NAMES,
)
