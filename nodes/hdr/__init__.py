"""HDR pipeline node group."""
from __future__ import annotations

import logging

from radiance.nodes.hdr.aces2 import (
    RadianceACES2Tonescale,
    RadianceACES2ReachGamutCompress,
    RadianceACES2OutputTransformFull,
)
from radiance.nodes.hdr.colorspace import (
    RadianceHDRColorPipeline,
)
from radiance.nodes.hdr.delivery import (
    RadianceHDREncode,
    RadianceHDRMonitor,
)
from radiance.nodes.hdr.encoder import (
    RadianceHDRLatentEncoder,
)
from radiance.nodes.hdr.smart import (
    RadianceHDRAutoLogSelect,
    RadianceHDRDiagnostics,
)
from radiance.nodes.hdr.uplift import (
    RadianceClipDetector,
    RadianceSDRToHDRPrepare,
    RadianceHDRHighlightComposite,
)
from radiance.nodes.hdr.synthesis import (
    RadianceSDRtoHDRExpand,
    RadianceHDRSynthesisEngine,
    RadianceRelightEngine,
)
from radiance.nodes.hdr.vae import (
    RadianceNativeHDREncoder,
)

logger = logging.getLogger("radiance.nodes.hdr")

NODE_CLASS_MAPPINGS = {
    "RadianceACES2Tonescale": RadianceACES2Tonescale,
    "RadianceACES2ReachGamutCompress": RadianceACES2ReachGamutCompress,
    "RadianceACES2OutputTransformFull": RadianceACES2OutputTransformFull,
    "RadianceHDRColorPipeline": RadianceHDRColorPipeline,
    "RadianceHDREncode": RadianceHDREncode,
    "RadianceHDRMonitor": RadianceHDRMonitor,
    "RadianceHDRAutoLogSelect": RadianceHDRAutoLogSelect,
    "RadianceHDRDiagnostics": RadianceHDRDiagnostics,
    "RadianceClipDetector": RadianceClipDetector,
    "RadianceSDRToHDRPrepare": RadianceSDRToHDRPrepare,
    "RadianceHDRHighlightComposite": RadianceHDRHighlightComposite,
    "RadianceSDRtoHDRExpand": RadianceSDRtoHDRExpand,
    "RadianceHDRSynthesisEngine": RadianceHDRSynthesisEngine,
    "RadianceRelightEngine": RadianceRelightEngine,
    "RadianceHDRLatentEncoder": RadianceHDRLatentEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceACES2Tonescale": "◎ ACES 2.0 Tonescale",
    "RadianceACES2ReachGamutCompress": "◎ ACES 2.0 Gamut Compress",
    "RadianceACES2OutputTransformFull": "◎ ACES 2.0 Output Transform",
    "RadianceHDRColorPipeline": "◎ HDR Color Pipeline",
    "RadianceHDREncode": "◎ HDR Encode",
    "RadianceHDRMonitor": "◎ HDR Monitor",
    "RadianceHDRAutoLogSelect": "◎ HDR Auto Log Select",
    "RadianceHDRDiagnostics": "◎ HDR Diagnostics",
    "RadianceClipDetector": "◎ Clip Detector",
    "RadianceSDRToHDRPrepare": "◎ SDR to HDR Prepare",
    "RadianceHDRHighlightComposite": "◎ HDR Highlight Composite",
    "RadianceSDRtoHDRExpand": "◎ SDR to HDR Expand",
    "RadianceHDRSynthesisEngine": "◎ HDR Synthesis Engine",
    "RadianceRelightEngine": "◎ Relight Engine",
    "RadianceHDRLatentEncoder": "◎ HDR Latent Encoder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
