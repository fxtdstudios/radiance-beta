"""
nodes_radiance_viewer.py — BACKWARD-COMPATIBILITY SHIM
======================================================
This file is retained so existing workflows importing directly from
``radiance.nodes_radiance_viewer`` continue to work without modification.

New code should import from::

    radiance.nodes.monitor.viewer

This shim will be removed in a future release.
"""
import logging as _logging

_LOGGER = _logging.getLogger("radiance.nodes_radiance_viewer")

try:
    from radiance.nodes.monitor.viewer import RadianceViewer
    from radiance.nodes.monitor.viewer import _lut_clip_check

    NODE_CLASS_MAPPINGS = {"RadianceViewer": RadianceViewer}
    NODE_DISPLAY_NAME_MAPPINGS = {"RadianceViewer": "◎ Radiance Viewer"}
except Exception as _exc:
    _LOGGER.warning(
        "Could not import radiance.nodes.monitor.viewer: %s. "
        "Viewer node will not be available.", _exc,
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]