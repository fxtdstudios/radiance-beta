"""Backward-compatible re-exports. Class definitions moved to radiance.nodes.color.ocio."""
import warnings
warnings.warn(
    "Import from radiance.nodes_ocio is deprecated; use radiance.nodes.color.ocio directly.",
    DeprecationWarning, stacklevel=2,
)

from radiance.nodes.color.ocio import RadianceOCIOContext

NODE_CLASS_MAPPINGS = {
    "RadianceOCIOContext": RadianceOCIOContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceOCIOContext": "◎ Radiance OCIO Context",
}
