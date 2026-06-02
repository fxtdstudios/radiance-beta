"""
nodes_hdr_smart.py — BACKWARD-COMPATIBILITY SHIM
================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_smart`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.smart

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_smart is deprecated; "
    "import from radiance.nodes.hdr.smart instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.smart import (
    RADIANCE_MODEL_PRESETS,
    _resolve_model,
    RadianceHDRAutoLogSelect,
    RadianceHDRDiagnostics,
)  # noqa: F401
