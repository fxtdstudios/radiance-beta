"""
nodes_denoise.py — BACKWARD-COMPATIBILITY SHIM
==============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_denoise`` continue to work without modification.

New code should import from::

    radiance.nodes.generate.denoise

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_denoise is deprecated; "
    "import from radiance.nodes.generate.denoise instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.generate.denoise import *  # noqa: F401, F403
