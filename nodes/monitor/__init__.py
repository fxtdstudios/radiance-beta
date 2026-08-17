"""Monitor, scopes, and preview node group."""
from __future__ import annotations

import logging

from radiance.nodes.monitor.lite_viewer import RadianceLiteViewer
from radiance.nodes.monitor.viewer import RadianceGradeApply, RadianceViewer
from radiance.nodes.monitor.realtime import (
    RadianceFocusPeaking,
    RadianceContactSheet,
    RadianceFrameStamp,
    RadianceFlipbookGIF,
    RadiancePreviewServer,
)
from radiance.nodes.aggregate import fold_in_module_nodes

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
    # Same story again, found 2026-08-16: viewer.py has always published this
    # in its own NODE_CLASS_MAPPINGS, and nodes/branding.py even carried a
    # SECTION_OVERRIDES entry classifying it — but this dict never listed it,
    # so it was invisible. Complete node: 16 documented inputs, valid contract.
    "RadianceGradeApply": RadianceGradeApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceGradeApply": "◎ Radiance Bake Viewer Grade",
    "RadianceLiteViewer": "◎ Radiance Lite Viewer",
    "RadianceViewer": "◎ Radiance Viewer",
    "RadianceFocusPeaking": "◎ Focus Peaking",
    "RadianceContactSheet": "◎ Contact Sheet",
    "RadianceFrameStamp": "◎ Frame Stamp",
    "RadianceFlipbookGIF": "◎ Radiance Flipbook GIF",
    "RadiancePreviewServer": "◎ Radiance Preview Server",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
