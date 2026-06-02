"""AI assist and knowledge node group."""
from __future__ import annotations

import logging

from radiance.nodes.ai.scene_cut import (
    RadianceSceneCutDetect,
    RadianceSceneCutSplit,
)

logger = logging.getLogger("radiance.nodes.ai")

NODE_CLASS_MAPPINGS = {
    "RadianceSceneCutDetect": RadianceSceneCutDetect,
    "RadianceSceneCutSplit": RadianceSceneCutSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSceneCutDetect": "◎ Scene Cut Detect",
    "RadianceSceneCutSplit": "◎ Scene Cut Split",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
