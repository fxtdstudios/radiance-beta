"""Color-science node group — organized internal modules."""
from __future__ import annotations

import logging

from radiance.nodes.color.cdl import (
    RadianceCDLTransform,
    RadianceCDLImport,
    RadianceCDLExport,
)
from radiance.nodes.color.colorspace import (
    RadianceWhiteBalance,
    RadianceColorSpaceConvert,
    RadianceACESTransform,
    RadianceBitDepthDegrade,
)
from radiance.nodes.color.curves import (
    RadianceHueCurves,
    RadianceCurves,
)
from radiance.nodes.color.grade import (
    RadianceGrade,
    RadianceApplyGradeInfo,
    RadianceGradeMatch,
)
from radiance.nodes.color.ocio import (
    RadianceOCIOContext,
)
from radiance.nodes.color.qc import (
    RadianceQC,
    RadiancePolicyGuard,
    RadianceQCExport,
)

logger = logging.getLogger("radiance.nodes.color")

NODE_CLASS_MAPPINGS = {
    "RadianceCDLTransform": RadianceCDLTransform,
    "RadianceCDLImport": RadianceCDLImport,
    "RadianceCDLExport": RadianceCDLExport,
    "RadianceWhiteBalance": RadianceWhiteBalance,
    "RadianceColorSpaceConvert": RadianceColorSpaceConvert,
    "RadianceACESTransform": RadianceACESTransform,
    "RadianceBitDepthDegrade": RadianceBitDepthDegrade,
    "RadianceHueCurves": RadianceHueCurves,
    "RadianceCurves": RadianceCurves,
    "RadianceGrade": RadianceGrade,
    "RadianceApplyGradeInfo": RadianceApplyGradeInfo,
    "RadianceGradeMatch": RadianceGradeMatch,
    "RadianceOCIOContext": RadianceOCIOContext,
    "RadianceQC": RadianceQC,
    "RadiancePolicyGuard": RadiancePolicyGuard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceCDLTransform": "◎ Radiance CDL Transform",
    "RadianceCDLImport": "◎ Radiance CDL Import",
    "RadianceCDLExport": "◎ Radiance CDL Export",
    "RadianceWhiteBalance": "◎ Radiance White Balance",
    "RadianceColorSpaceConvert": "◎ Radiance Colorspace Convert",
    "RadianceACESTransform": "◎ Radiance ACES Transform",
    "RadianceBitDepthDegrade": "◎ Radiance Bit Depth Degrade",
    "RadianceHueCurves": "◎ Radiance Hue Curves",
    "RadianceCurves": "◎ Radiance Curves",
    "RadianceGrade": "◎ Radiance Grade",
    "RadianceApplyGradeInfo": "◎ Radiance Apply Grade Info",
    "RadianceGradeMatch": "◎ Radiance Grade Match",
    "RadianceOCIOContext": "◎ Radiance OCIO Context",
    "RadianceQC": "◎ Radiance QC",
    "RadiancePolicyGuard": "◎ Policy Guard",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
