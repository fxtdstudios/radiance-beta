"""Monitor, scopes, and preview node group."""
from __future__ import annotations

import logging

from radiance.nodes.monitor.lite_viewer import RadianceLiteViewer
from radiance.nodes.monitor.viewer import RadianceViewer
from radiance.nodes_realtime_preview import (
    RadianceFocusPeaking,
    RadianceContactSheet,
    RadianceFrameStamp,
)

logger = logging.getLogger("radiance.nodes.monitor")

NODE_CLASS_MAPPINGS = {
    "RadianceLiteViewer": RadianceLiteViewer,
    "RadianceViewer": RadianceViewer,
    "RadianceFocusPeaking": RadianceFocusPeaking,
    "RadianceContactSheet": RadianceContactSheet,
    "RadianceFrameStamp": RadianceFrameStamp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLiteViewer": "◎ Radiance Lite Viewer",
    "RadianceViewer": "◎ Radiance Viewer",
    "RadianceFocusPeaking": "◎ Focus Peaking",
    "RadianceContactSheet": "◎ Contact Sheet",
    "RadianceFrameStamp": "◎ Frame Stamp",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
