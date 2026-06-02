"""
nodes_hdr_encoder.py — BACKWARD-COMPATIBILITY SHIM
==================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_encoder`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.encoder

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_encoder is deprecated; "
    "import from radiance.nodes.hdr.encoder instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.encoder import *  # noqa: F401, F403
# Explicitly import private variables needed by unit tests
from radiance.nodes.hdr.encoder import (
    _hdr_soft_compress,
    _hdr_soft_decompress,
)
