"""
nodes_aces2.py — BACKWARD-COMPATIBILITY SHIM
============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_aces2`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.aces2

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_aces2 is deprecated; "
    "import from radiance.nodes.hdr.aces2 instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.aces2 import *  # noqa: F401, F403
# Explicitly import private variables needed by unit tests
from radiance.nodes.hdr.aces2 import (
    _DanieleEvoParams,
    _daniele_evo_fwd,
    _daniele_evo_luma_preserving,
    _reach_compress_channel,
    _reach_gamut_compress,
    _build_amf,
    _parse_amf,
    _s2126_check,
    _pq_encode,
    _hlg_encode,
    _srgb_encode,
    LUMA_AP1,
)
