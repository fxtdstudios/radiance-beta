"""
nodes_vae_v3.py — BACKWARD-COMPATIBILITY SHIM
=============================================
This file is retained so existing workflows importing directly from
``radiance.nodes_vae_v3`` continue to work without modification.

New code should import from::

    radiance.nodes.hdr.vae

This shim will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "radiance.nodes_vae_v3 is deprecated; "
    "import from radiance.nodes.hdr.vae instead.",
    DeprecationWarning,
    stacklevel=2,
)
from radiance.nodes.hdr.vae import *  # noqa: F401, F403
