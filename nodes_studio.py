"""
nodes_studio.py — BACKWARD-COMPATIBILITY SHIM
=============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_studio`` continue to work without modification.

New code should import from::

    radiance.nodes.pipeline.studio

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_studio is deprecated; "
    "import from radiance.nodes.pipeline.studio instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.pipeline.studio import *  # noqa: F401, F403
