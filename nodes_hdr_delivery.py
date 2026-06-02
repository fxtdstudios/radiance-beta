"""
nodes_hdr_delivery.py — BACKWARD-COMPATIBILITY SHIM
===================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_hdr_delivery`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.delivery

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_hdr_delivery is deprecated; "
    "import from radiance.nodes.hdr.delivery instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.delivery import *  # noqa: F401, F403
# Explicitly import private variables/methods needed by unit tests
from radiance.nodes.hdr.delivery import (
    _linear_to_pq,
    _linear_to_hlg,
)
