"""Pipeline orchestration node group."""
from __future__ import annotations

import logging

from radiance.nodes.pipeline.audio import RadianceAudioCut, RadianceAudioTranscribe
from radiance.nodes_workspace import RadianceProjectManager
# nodes_layout.py is currently empty — no Layout node classes defined yet.
from radiance.nodes.pipeline.studio import RadianceCinemaStudio
from radiance.nodes.pipeline.overlay import RadianceBlendComposite
from radiance.nodes.pipeline.dcc import RadianceMCP
from radiance.nodes.pipeline.studio_integrations import RadianceDaVinciSend, RadianceNukeSend

# Import custom parameter history tracker
from radiance.core.param_memory import RadianceParamHistoryTracker

logger = logging.getLogger("radiance.nodes.pipeline")

NODE_CLASS_MAPPINGS = {
    "RadianceAudioCut": RadianceAudioCut,
    "RadianceProjectManager": RadianceProjectManager,
    "RadianceBlendComposite": RadianceBlendComposite,
    "RadianceCinemaStudio": RadianceCinemaStudio,
    "RadianceMCP": RadianceMCP,
    "RadianceNukeSend": RadianceNukeSend,
    "RadianceDaVinciSend": RadianceDaVinciSend,
    "RadianceParamHistoryTracker": RadianceParamHistoryTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceAudioCut": "◎ Audio Cut",
    "RadianceProjectManager": "◎ Project Manager",
    "RadianceBlendComposite": "◎ Blend Composite",
    "RadianceCinemaStudio": "◎ Cinema Studio",
    "RadianceMCP": "◎ Radiance MCP Bridge",
    "RadianceNukeSend": "◎ Radiance Send to Nuke",
    "RadianceDaVinciSend": "◎ Radiance Send to DaVinci Resolve",
    "RadianceParamHistoryTracker": "◎ Parameter History Tracker",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
