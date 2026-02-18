
from .film import (
    RadianceWhiteBalance,
    RadianceDepthOfField,
    RadianceMotionBlur,
    RadianceRollingShutter,
    RadianceCompressionArtifacts
)

NODE_CLASS_MAPPINGS = {
    "RadianceWhiteBalance": RadianceWhiteBalance,
    "RadianceDepthOfField": RadianceDepthOfField,
    "RadianceMotionBlur": RadianceMotionBlur,
    "RadianceRollingShutter": RadianceRollingShutter,
    "RadianceCompressionArtifacts": RadianceCompressionArtifacts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceWhiteBalance": "◎ Radiance White Balance",
    "RadianceDepthOfField": "◎ Radiance Depth of Field",
    "RadianceMotionBlur": "◎ Radiance Motion Blur",
    "RadianceRollingShutter": "◎ Radiance Rolling Shutter",
    "RadianceCompressionArtifacts": "◎ Radiance Compression Artifacts"
}
