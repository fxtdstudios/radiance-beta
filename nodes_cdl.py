"""Backward-compatible re-exports. Class definitions moved to radiance.nodes.color.cdl."""
import warnings
warnings.warn(
    "Import from radiance.nodes_cdl is deprecated; use radiance.nodes.color.cdl directly.",
    DeprecationWarning, stacklevel=2,
)

from radiance.nodes.color.cdl import (
    RadianceCDLTransform,
    RadianceCDLImport,
    RadianceCDLExport,
)

NODE_CLASS_MAPPINGS = {
    "RadianceCDLTransform": RadianceCDLTransform,
    "RadianceCDLImport": RadianceCDLImport,
    "RadianceCDLExport": RadianceCDLExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceCDLTransform": "◎ Radiance CDL Transform",
    "RadianceCDLImport": "◎ Radiance CDL Import",
    "RadianceCDLExport": "◎ Radiance CDL Export",
}
