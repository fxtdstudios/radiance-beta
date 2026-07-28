"""Monitor, scopes, and preview node group."""
from __future__ import annotations

import logging

from radiance.nodes.monitor.lite_viewer import RadianceLiteViewer
from radiance.nodes.monitor.viewer import RadianceViewer
from radiance.nodes_realtime_preview import (
    RadianceFocusPeaking,
    RadianceContactSheet,
    RadianceFrameStamp,
    RadianceFlipbookGIF,
    RadiancePreviewServer,
)

logger = logging.getLogger("radiance.nodes.monitor")

NODE_CLASS_MAPPINGS = {
    "RadianceLiteViewer": RadianceLiteViewer,
    "RadianceViewer": RadianceViewer,
    "RadianceFocusPeaking": RadianceFocusPeaking,
    "RadianceContactSheet": RadianceContactSheet,
    "RadianceFrameStamp": RadianceFrameStamp,
    # The v3 reorganisation transcribed these mapping dicts by hand and dropped
    # entries on the way. The classes were written, complete and importable the
    # whole time -- they simply never appeared in ComfyUI's node menu.
    "RadianceFlipbookGIF": RadianceFlipbookGIF,
    "RadiancePreviewServer": RadiancePreviewServer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLiteViewer": "◎ Radiance Lite Viewer",
    "RadianceViewer": "◎ Radiance Viewer",
    "RadianceFocusPeaking": "◎ Focus Peaking",
    "RadianceContactSheet": "◎ Contact Sheet",
    "RadianceFrameStamp": "◎ Frame Stamp",
    "RadianceFlipbookGIF": "◎ Radiance Flipbook GIF",
    "RadiancePreviewServer": "◎ Radiance Preview Server",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
