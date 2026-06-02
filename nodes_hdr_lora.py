"""
nodes_hdr_lora.py — BACKWARD-COMPATIBILITY SHIM
===============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_lora`` continue to work without modification.

New code should import from::

    radiance.nodes.generate.lora

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_lora is deprecated; "
    "import from radiance.nodes.generate.lora instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.generate.lora import *  # noqa: F401, F403
