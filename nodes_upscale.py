
"""
Radiance Upscale - Legacy Facade
"""
from .image.upscale import (
    RadianceProUpscale, 
    RadianceUpscaleBySize,
    RadianceDownscale32bit,
    RadianceBitDepthConvert,
    RadianceAIUpscale
)

NODE_CLASS_MAPPINGS = {
    "RadianceProUpscale": RadianceProUpscale,
    "RadianceUpscaleBySize": RadianceUpscaleBySize,
    "RadianceDownscale32bit": RadianceDownscale32bit,
    "RadianceBitDepthConvert": RadianceBitDepthConvert,
    "RadianceAIUpscale": RadianceAIUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceProUpscale": "◎ Radiance Pro Upscale",
    "RadianceUpscaleBySize": "◎ Radiance Upscale By Size",
    "RadianceDownscale32bit": "◎ Radiance Downscale 32-bit",
    "RadianceBitDepthConvert": "◎ Radiance Bit Depth Convert",
    "RadianceAIUpscale": "◎ Radiance AI Upscale",
}
