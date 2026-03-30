from .utils import (
    tensor_to_numpy_float32,
    numpy_to_tensor_float32,
)

from .color import (
    ImageToFloat32,
    Float32ColorCorrect,
    ColorSpaceConvert,
    DaVinciWideGamut,
    ARRIWideGamut4,
    ACES2OutputTransform,
)

from .tonemap import HDRExpandDynamicRange, HDRToneMap


from .io import RadianceSaveEXR

from .analysis import HDRHistogram

from .processing import HDRExposureBlend, HDRShadowHighlightRecovery, GPUTensorOps

from .panorama import HDR360Generate

from .recovery import RadianceHighlightSynthesis

from .ocio import ACESConfigManager, ACESConfigManager, OCIOListColorspaces

from .vae import (
    RadianceVAE4KEncode as RadianceVAEEncode,
    RadianceVAE4KDecode as RadianceVAEDecode,
)

NODE_CLASS_MAPPINGS = {
    "ImageToFloat32": ImageToFloat32,
    "Float32ColorCorrect": Float32ColorCorrect,
    "ColorSpaceConvert": ColorSpaceConvert,
    "DaVinciWideGamut": DaVinciWideGamut,
    "ARRIWideGamut4": ARRIWideGamut4,
    "ACES2OutputTransform": ACES2OutputTransform,
    "HDRExpandDynamicRange": HDRExpandDynamicRange,
    "HDRToneMap": HDRToneMap,
    # Legacy I/O nodes removed in favor of Digital Cinema Universal I/O
    "RadianceSaveEXR": RadianceSaveEXR,
    "HDRHistogram": HDRHistogram,
    "HDRExposureBlend": HDRExposureBlend,
    "HDRShadowHighlightRecovery": HDRShadowHighlightRecovery,
    "GPUTensorOps": GPUTensorOps,
    "RadianceHighlightSynthesis": RadianceHighlightSynthesis,
    "HDR360Generate": HDR360Generate,
    "RadianceVAEEncode": RadianceVAEEncode,
    "RadianceVAEDecode": RadianceVAEDecode,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToFloat32": "◎ Radiance Image To Float32",
    "Float32ColorCorrect": "◎ Radiance Float32 Color Correct",
    "ColorSpaceConvert": "◎ Radiance Color Space Convert",
    "DaVinciWideGamut": "◎ Radiance DaVinci Wide Gamut",
    "ARRIWideGamut4": "◎ Radiance ARRI Wide Gamut 4",
    "ACES2OutputTransform": "◎ Radiance ACES 2.0 Output Transform",
    "HDRExpandDynamicRange": "◎ Radiance Expand Dynamic Range",
    "HDRToneMap": "◎ Radiance HDR Tone Map",
    "RadianceSaveEXR": "◎ Radiance Save EXR",
    "HDRHistogram": "◎ Radiance HDR Histogram",
    "HDRExposureBlend": "◎ Radiance HDR Exposure Blend",
    "HDRShadowHighlightRecovery": "◎ Radiance HDR Shadow/Highlight Recovery",
    "GPUTensorOps": "◎ Radiance GPU Tensor Ops",
    "RadianceHighlightSynthesis": "◎ Radiance Highlight Synthesis",
    "HDR360Generate": "◎ Radiance HDR 360 Generate",
    "RadianceVAEEncode": "◎ Radiance VAE Encode",
    "RadianceVAEDecode": "◎ Radiance VAE Decode",
    
}
