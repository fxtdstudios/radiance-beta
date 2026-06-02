"""
nodes_prompt.py — BACKWARD-COMPATIBILITY SHIM
=============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_prompt`` continue to work without modification.

New code should import from::

    radiance.nodes.generate.prompt

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_prompt is deprecated; "
    "import from radiance.nodes.generate.prompt instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.generate.prompt import *  # noqa: F401, F403
