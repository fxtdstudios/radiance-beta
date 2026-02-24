"""
Radiance HDR Nodes - Facade
Redirects to the modular implementation in radiance.hdr package.
"""

import logging

# Import from the modular package
try:
    from .hdr import (
        ImageToFloat32,
        Float32ColorCorrect,
        ColorSpaceConvert,
        DaVinciWideGamut,
        ARRIWideGamut4,
        ACES2OutputTransform,
        HDRExpandDynamicRange,
        HDRToneMap,
        LoadImageEXR,
        LoadImageEXRSequence,
        SaveImage16bit,
        RadianceSaveEXR,
        HDRHistogram,
        HDRExposureBlend,
        HDRShadowHighlightRecovery,
        GPUTensorOps,
        HDR360Generate,
        ACESConfigManager,
        OCIOListColorspaces,
        RadianceVAEEncode,
        RadianceVAEDecode,
        RadianceHighlightSynthesis,
    )
except ImportError as e:
    logging.error(f"Failed to import Radiance HDR modules: {e}")

    # Define dummy classes to avoid complete crash if something is wrong
    class ImageToFloat32:
        pass

    """
    Convert images to 32-bit float precision for HDR processing. Preserves full dynamic range without clamping.
    """

    class Float32ColorCorrect:
        pass

    """
    Professional 32-bit color correction with exposure, contrast, saturation, gamma, and per-channel lift/gain controls.
    """

    class ColorSpaceConvert:
        pass

    """
    GPU-accelerated color space conversion (sRGB, ACEScg, ACEScct, Rec.2020, DCI-P3). Uses industry-standard matrices.
    """

    class DaVinciWideGamut:
        pass

    """
    Convert to/from DaVinci Wide Gamut and DaVinci Intermediate.
    """

    class ARRIWideGamut4:
        pass

    """
    Convert to/from ARRI Wide Gamut 4 (AWG4) for Alexa 35.
    """

    class ACES2OutputTransform:
        pass

    """
    Apply ACES 2.0 Output Transform with proper gamut mapping for SDR, HDR, or Cinema output.
    """

    class HDRExpandDynamicRange:
        pass

    """
    Expand SDR images to HDR dynamic range by recovering highlights and extending stops of exposure latitude.
    """

    class HDRToneMap:
        pass

    """
    Professional HDR tone mapping with presets and advanced controls. GPU-accelerated.
    """

    class LoadImageEXR:
        pass

    """
    Load EXR/HDR files with full HDR dynamic range. Enter file path directly or select from input folder.
    """

    class LoadImageEXRSequence:
        pass

    """
    Load EXR/HDR image sequence from a folder. Returns a batch of images.
    """

    class SaveImage16bit:
        pass

    """
    Save images as 16-bit PNG or TIFF for wider software compatibility while preserving extended range.
    """

    class HDRHistogram:
        pass

    """
    Analyze HDR image histogram with dynamic range statistics, clipping indicators, and stops visualization.
    """

    class HDRExposureBlend:
        pass

    """
    Blend multiple exposures (bracketing) for extended dynamic range. Takes highlights from low exposure and shadows from high exposure for optimal color grading.
    """

    class HDRShadowHighlightRecovery:
        pass

    """
    Recover shadow and highlight detail from a single HDR image for better color grading flexibility.
    """

    class GPUTensorOps:
        pass

    """
    GPU-accelerated HDR operations. Fast exposure, gamma, and normalization.
    """

    class HDR360Generate:
        pass

    """
    Generate 360° equirectangular panoramas for HDRI environment mapping in 3D applications.
    """

    class ACESConfigManager:
        pass

    """
    Detect, download, and manage ACES OCIO configurations for professional color workflows.
    """

    class OCIOListColorspaces:
        pass

    """
    List all colorspaces available in an OCIO configuration file.
    """

    class RadianceVAEEncode:
        pass

    """
    Encode 32-bit Linear/ACEScg images to VAE Latents with correct color handling.
    """

    class RadianceVAEDecode:
        pass

    """
    Decode VAE Latents directly to 32-bit Linear/ACEScg images.
    """

    class RadianceHighlightSynthesis:
        pass

    """
    Synthesize high dynamic range details in clipped highlights to simulate film scan quality.
    """

    class RadianceSaveEXR:
        pass

    """
    Save images as EXR files with full HDR and metadata support.
    """


# ═══════════════════════════════════════════════════════════════════════════════
#                          NODE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "ImageToFloat32": ImageToFloat32,
    "Float32ColorCorrect": Float32ColorCorrect,
    "HDRExpandDynamicRange": HDRExpandDynamicRange,
    "HDRToneMap": HDRToneMap,
    "ColorSpaceConvert": ColorSpaceConvert,
    "LoadImageEXR": LoadImageEXR,
    "LoadImageEXRSequence": LoadImageEXRSequence,
    "SaveImage16bit": SaveImage16bit,
    "RadianceSaveEXR": RadianceSaveEXR,
    "HDRHistogram": HDRHistogram,
    "HDRExposureBlend": HDRExposureBlend,
    "HDRShadowHighlightRecovery": HDRShadowHighlightRecovery,
    "OCIOListColorspaces": OCIOListColorspaces,
    "GPUTensorOps": GPUTensorOps,
    "HDR360Generate": HDR360Generate,
    "ACES2OutputTransform": ACES2OutputTransform,
    "ACESConfigManager": ACESConfigManager,
    "DaVinciWideGamut": DaVinciWideGamut,
    "ARRIWideGamut4": ARRIWideGamut4,
    "RadianceVAEEncode": RadianceVAEEncode,
    "RadianceVAEDecode": RadianceVAEDecode,
    "RadianceHighlightSynthesis": RadianceHighlightSynthesis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToFloat32": "◎ Radiance Image to Float32",
    "Float32ColorCorrect": "◎ Radiance Float32 Color Correct",
    "HDRExpandDynamicRange": "◎ Radiance HDR Expand Dynamic Range",
    "HDRToneMap": "◎ Radiance HDR Tone Map",
    "ColorSpaceConvert": "◎ Radiance Color Space Convert",
    "LoadImageEXR": "◎ Radiance Load EXR",
    "LoadImageEXRSequence": "◎ Radiance Load EXR Sequence",
    "SaveImage16bit": "◎ Radiance Save 16-bit PNG/TIFF",
    "RadianceSaveEXR": "◎ Radiance Save EXR/HDR",
    "HDRHistogram": "◎ Radiance HDR Histogram",
    "HDRExposureBlend": "◎ Radiance HDR Exposure Blend",
    "HDRShadowHighlightRecovery": "◎ Radiance HDR Shadow/Highlight Recovery",
    "OCIOListColorspaces": "◎ Radiance OCIO List Colorspaces",
    "GPUTensorOps": "◎ Radiance GPU Tensor Ops",
    "HDR360Generate": "◎ Radiance HDR 360 Generate",
    "ACES2OutputTransform": "◎ Radiance ACES 2.0 Output Transform",
    "ACESConfigManager": "◎ Radiance ACES Config Manager",
    "DaVinciWideGamut": "◎ Radiance DaVinci Wide Gamut",
    "ARRIWideGamut4": "◎ Radiance ARRI Wide Gamut 4",
    "RadianceVAEEncode": "◎ Radiance VAE Encode Pro",
    "RadianceVAEDecode": "◎ Radiance VAE Decode Pro",
    "RadianceHighlightSynthesis": "◎ Radiance Highlight Synthesis",
}

if __name__ == "__main__":
    logging.info("Radiance HDR Nodes loaded via Facade.")
