"""Monitor, scopes, and preview node group."""
from __future__ import annotations

import logging

from radiance.nodes.monitor.lite_viewer import RadianceLiteViewer
from radiance.nodes.monitor.viewer import RadianceViewer
from radiance.nodes_realtime_preview import (
    RadianceFocusPeaking,
    RadianceContactSheet,
    RadianceFlipbookGIF,
    RadianceFrameStamp,
    RadiancePreviewServer,
)

logger = logging.getLogger("radiance.nodes.monitor")

NODE_CLASS_MAPPINGS = {
    "RadianceLiteViewer": RadianceLiteViewer,
    "RadianceViewer": RadianceViewer,
    "RadianceFocusPeaking": RadianceFocusPeaking,
    "RadianceContactSheet": RadianceContactSheet,
    "RadianceFlipbookGIF": RadianceFlipbookGIF,
    "RadianceFrameStamp": RadianceFrameStamp,
    "RadiancePreviewServer": RadiancePreviewServer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLiteViewer": "◎ Radiance Lite Viewer",
    "RadianceViewer": "◎ Radiance Viewer",
    "RadianceFocusPeaking": "◎ Focus Peaking",
    "RadianceContactSheet": "◎ Contact Sheet",
    "RadianceFlipbookGIF": "◎ Flipbook GIF",
    "RadianceFrameStamp": "◎ Frame Stamp",
    "RadiancePreviewServer": "◎ Preview Server",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
