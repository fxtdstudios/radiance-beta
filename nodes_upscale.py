"""
nodes_upscale.py — BACKWARD-COMPATIBILITY SHIM
==============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_upscale`` continue to work without modification.

New code should import from::

    radiance.nodes.upscale.upscale

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_upscale is deprecated; "
    "import from radiance.nodes.upscale.upscale instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.upscale.upscale import *  # noqa: F401, F403
# Explicitly import private variables/methods needed by unit tests
from radiance.nodes.upscale.upscale import (
    _UPSCALE_MODEL_REGISTRY,
    _TIER_CHOICES,
    _gaussian_kernel_1d,
    _build_gaussian_weight_map,
    _bicubic_upscale,
    _histogram_match_fast,
    _classify_content,
    _recommend_tier,
)
