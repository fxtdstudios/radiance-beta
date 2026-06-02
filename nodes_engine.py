"""
nodes_engine.py — BACKWARD-COMPATIBILITY SHIM
=============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_engine`` continue to work without modification.

New code should import from::

    radiance.nodes.generate.engine

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_engine is deprecated; "
    "import from radiance.nodes.generate.engine instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.generate.engine import *  # noqa: F401, F403
try:
    from radiance.color.lut import RadianceLUTApply  # noqa: F401
except ImportError:
    pass
