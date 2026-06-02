"""
nodes_send_dcc.py — BACKWARD-COMPATIBILITY SHIM
===============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_send_dcc`` continue to work without modification.

New code should import from::

    radiance.nodes.pipeline.dcc

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_send_dcc is deprecated; "
    "import from radiance.nodes.pipeline.dcc instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.pipeline.dcc import *  # noqa: F401, F403
