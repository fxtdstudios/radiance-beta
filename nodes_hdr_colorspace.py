"""
nodes_hdr_colorspace.py — BACKWARD-COMPATIBILITY SHIM
=====================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_colorspace`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.colorspace

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_colorspace is deprecated; "
    "import from radiance.nodes.hdr.colorspace instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.colorspace import *  # noqa: F401, F403
# Explicitly import private variables/methods needed by unit tests
from radiance.nodes.hdr.colorspace import (
    _apply_matrix,
    _eotf_srgb,
    _eotf_rec709,
    _eotf_gamma22,
    _eotf_gamma24,
    _eotf_bt1886,
    _eotf_pq,
    _eotf_hlg,
    _BRADFORD_CAT,
    _PRIMARIES_MATRICES,
)
