"""Generation, sampling, and model-loading node group."""
from __future__ import annotations

import logging

from radiance.nodes_sampler import RadianceSamplerPro
from radiance.nodes.generate.engine import (
    RadianceHDRVAEDecode,
)
from radiance.nodes_loader import (
    RadianceLoraStack,
    RadianceUnifiedLoader,
    RadianceVideoLoader,
    RadianceControlNetApply,
)
from radiance.nodes.generate.lora import RadianceHDRLoRALoader, RadianceHDRLoRAApply
from radiance.nodes.generate.prompt import (
    RadianceCinematicPromptEncoder,
)
from radiance.nodes.generate.regional import RadianceRegionalPrompt, RadianceRegionalGrid
from radiance.nodes.generate.resolution import RadianceResolution
from radiance.nodes.generate.denoise import RadianceDenoise

logger = logging.getLogger("radiance.nodes.generate")

NODE_CLASS_MAPPINGS = {
    "RadianceSamplerPro": RadianceSamplerPro,
    "RadianceHDRVAEDecode": RadianceHDRVAEDecode,
    "RadianceLoraStack": RadianceLoraStack,
    "RadianceUnifiedLoader": RadianceUnifiedLoader,
    "RadianceVideoLoader": RadianceVideoLoader,
    "RadianceControlNetApply": RadianceControlNetApply,
    "RadianceHDRLoRALoader": RadianceHDRLoRALoader,
    "RadianceHDRLoRAApply": RadianceHDRLoRAApply,
    "RadianceCinematicPromptEncoder": RadianceCinematicPromptEncoder,
    "RadianceRegionalPrompt": RadianceRegionalPrompt,
    "RadianceRegionalGrid": RadianceRegionalGrid,
    "RadianceResolution": RadianceResolution,
    "RadianceDenoise": RadianceDenoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSamplerPro": "◎ Radiance Sampler Pro",
    "RadianceHDRVAEDecode": "◎ HDR VAE Decode",
    "RadianceLoraStack": "◎ LoRA Stack",
    "RadianceUnifiedLoader": "◎ Radiance Read Models",
    "RadianceVideoLoader": "◎ Video Loader",
    "RadianceControlNetApply": "◎ ControlNet Apply",
    "RadianceHDRLoRALoader": "◎ HDR LoRA Loader",
    "RadianceHDRLoRAApply": "◎ HDR LoRA Apply",
    "RadianceCinematicPromptEncoder": "◎ Cinematic Prompt Encoder",
    "RadianceRegionalPrompt": "◎ Regional Prompt",
    "RadianceRegionalGrid": "◎ Regional Grid",
    "RadianceResolution": "◎ Resolution",
    "RadianceDenoise": "◎ Denoise",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
