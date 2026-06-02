"""
nodes_sdr_degradation.py — BACKWARD-COMPATIBILITY SHIM
======================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_sdr_degradation`` continue to work without modification.

New code should import from::

    radiance.nodes.training.sdr_degradation

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_sdr_degradation is deprecated; "
    "import from radiance.nodes.training.sdr_degradation instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.training.sdr_degradation import *  # noqa: F401, F403
# Explicitly import private variables needed by unit tests
from radiance.nodes.training.sdr_degradation import (
    _tonemap,
    _highlight_clip,
    _oetf_srgb,
    _color_shift,
    _sensor_noise,
    _quantize,
    _vignette,
    _gaussian_blur,
    _jpeg_compress,
    _TONEMAP_OPERATORS,
)
