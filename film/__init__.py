from .grain import RadianceFilmGrain
from .camera import (
    RadianceWhiteBalance,
    RadianceDepthOfField,
    RadianceMotionBlur,
    RadianceRollingShutter,
    RadianceCompressionArtifacts,
)

__all__ = [
    "RadianceFilmGrain",
    "RadianceWhiteBalance",
    "RadianceDepthOfField",
    "RadianceMotionBlur",
    "RadianceRollingShutter",
    "RadianceCompressionArtifacts",
]

NODE_CLASS_MAPPINGS = {
    "RadianceFilmGrain": RadianceFilmGrain,
    "RadianceWhiteBalance": RadianceWhiteBalance,
    "RadianceDepthOfField": RadianceDepthOfField,
    "RadianceMotionBlur": RadianceMotionBlur,
    "RadianceRollingShutter": RadianceRollingShutter,
    "RadianceCompressionArtifacts": RadianceCompressionArtifacts,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceFilmGrain": "◎ Radiance Film Grain",
    "RadianceWhiteBalance": "◎ Radiance White Balance",
    "RadianceDepthOfField": "◎ Radiance Depth of Field",
    "RadianceMotionBlur": "◎ Radiance Motion Blur",
    "RadianceRollingShutter": "◎ Radiance Rolling Shutter",
    "RadianceCompressionArtifacts": "◎ Radiance Compression Artifacts",
    
}
