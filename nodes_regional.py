"""
nodes_regional.py — BACKWARD-COMPATIBILITY SHIM
===============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_regional`` continue to work without modification.

New code should import from::

    radiance.nodes.generate.regional

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_regional is deprecated; "
    "import from radiance.nodes.generate.regional instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.generate.regional import *  # noqa: F401, F403
