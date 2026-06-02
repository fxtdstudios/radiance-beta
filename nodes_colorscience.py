"""Backward-compatible re-exports. Class definitions moved to radiance.nodes.color.colorspace."""
import warnings
warnings.warn(
    "Import from radiance.nodes_colorscience is deprecated; use radiance.nodes.color.colorspace directly.",
    DeprecationWarning, stacklevel=2,
)

from radiance.nodes.color.colorspace import (
    RadianceWhiteBalance,
    RadianceColorSpaceConvert,
    RadianceACESTransform,
    RadianceBitDepthDegrade,
    _xy_to_XYZ,
    _build_bradford_matrix,
    _temperature_to_xy,
    _ILLUMINANT_XY,
)

_COLOR_SPACES = RadianceColorSpaceConvert._COLOR_SPACES

NODE_CLASS_MAPPINGS = {
    "RadianceWhiteBalance": RadianceWhiteBalance,
    "RadianceColorSpaceConvert": RadianceColorSpaceConvert,
    "RadianceACESTransform": RadianceACESTransform,
    "RadianceBitDepthDegrade": RadianceBitDepthDegrade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceWhiteBalance": "◎ Radiance White Balance",
    "RadianceColorSpaceConvert": "◎ Radiance Color Space Convert",
    "RadianceACESTransform": "◎ Radiance ACES 2.0 Transform",
    "RadianceBitDepthDegrade": "◎ Radiance Bit-Depth Degrade",
}
