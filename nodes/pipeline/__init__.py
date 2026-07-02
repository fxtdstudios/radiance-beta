"""Pipeline orchestration node group."""
from __future__ import annotations

import logging

from radiance.nodes_workspace import RadianceProjectManager
# nodes_layout.py is currently empty — no Layout node classes defined yet.
from radiance.nodes.pipeline.overlay import RadianceBlendComposite
from radiance.nodes.pipeline.dcc import RadianceMCP
from radiance.nodes.pipeline.studio_integrations import RadianceDaVinciSend, RadianceNukeSend

logger = logging.getLogger("radiance.nodes.pipeline")

NODE_CLASS_MAPPINGS = {
    "RadianceProjectManager": RadianceProjectManager,
    "RadianceBlendComposite": RadianceBlendComposite,
    "RadianceMCP": RadianceMCP,
    "RadianceNukeSend": RadianceNukeSend,
    "RadianceDaVinciSend": RadianceDaVinciSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceProjectManager": "◎ Project Manager",
    "RadianceBlendComposite": "◎ Blend Composite",
    "RadianceMCP": "◎ Radiance MCP Bridge",
    "RadianceNukeSend": "◎ Radiance Send to Nuke",
    "RadianceDaVinciSend": "◎ Radiance Send to DaVinci Resolve",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
