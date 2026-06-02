"""VFX, compositing, and optical-effects node group."""
from __future__ import annotations

import logging

from radiance.nodes.vfx.depth import RadianceDepthMapGenerator
from radiance.nodes.vfx.motion import RadianceOpticalFlow
from radiance.nodes.vfx.motion_blur import RadianceMotionBlur
from radiance.nodes.vfx.optics import (
    RadianceLensDistortion,
    RadianceChromaticAberration,
    RadianceAnamorphicStreaks,
    RadianceFilmGrain,
    RadianceVignette,
)
from radiance.nodes.vfx.multipass import (
    NODE_CLASS_MAPPINGS as MP_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MP_DISPLAY_NAMES,
)

# ── Phase 1, 2, 3 and 5 new high-precision linear VFX nodes ───────────
from radiance.nodes.vfx.masking import (
    RadianceSAMModelLoader,
    RadianceSAMGenerator,
    RadianceMultiMaskVisualPicker,
    RadianceLinearMatting,
)
from radiance.nodes.vfx.plate import (
    RadianceHDRGrainMatcher,
    RadianceSubpixelStabilizer,
)
from radiance.nodes.vfx.inpaint import (
    RadianceHDRCrop,
    RadianceHDRStitch,
    RadianceTemporalStitchStabilizer,
)
from radiance.nodes.vfx.roto import (
    RadianceVectorMaskDraw,
    RadianceVideoMaskPropagator,
)

logger = logging.getLogger("radiance.nodes.vfx")

NODE_CLASS_MAPPINGS = {
    # Existing VFX nodes
    "RadianceDepthMapGenerator": RadianceDepthMapGenerator,
    "RadianceOpticalFlow": RadianceOpticalFlow,
    "RadianceMotionBlur": RadianceMotionBlur,
    "RadianceLensDistortion": RadianceLensDistortion,
    "RadianceChromaticAberration": RadianceChromaticAberration,
    "RadianceAnamorphicStreaks": RadianceAnamorphicStreaks,
    "RadianceFilmGrain": RadianceFilmGrain,
    "RadianceVignette": RadianceVignette,
    **MP_MAPPINGS,
    
    # Phase 1: Masking & Matting
    "RadianceSAMModelLoader": RadianceSAMModelLoader,
    "RadianceSAMGenerator": RadianceSAMGenerator,
    "RadianceMultiMaskVisualPicker": RadianceMultiMaskVisualPicker,
    "RadianceLinearMatting": RadianceLinearMatting,
    
    # Phase 2: Plate Prep
    "RadianceHDRGrainMatcher": RadianceHDRGrainMatcher,
    "RadianceSubpixelStabilizer": RadianceSubpixelStabilizer,
    
    # Phase 3: Regional Inpainting Crop & Stitch
    "RadianceHDRCrop": RadianceHDRCrop,
    "RadianceHDRStitch": RadianceHDRStitch,
    "RadianceTemporalStitchStabilizer": RadianceTemporalStitchStabilizer,
    
    # Phase 5: Advanced Rotoscoping & Propagation
    "RadianceVectorMaskDraw": RadianceVectorMaskDraw,
    "RadianceVideoMaskPropagator": RadianceVideoMaskPropagator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Existing display names
    "RadianceDepthMapGenerator": "◎ Depth Map Generator",
    "RadianceOpticalFlow": "◎ Optical Flow",
    "RadianceMotionBlur": "◎ Physical Motion Blur",
    "RadianceLensDistortion": "◎ Lens Distortion",
    "RadianceChromaticAberration": "◎ Chromatic Aberration",
    "RadianceAnamorphicStreaks": "◎ Anamorphic Streaks",
    "RadianceFilmGrain": "◎ Film Grain (Simple)",
    "RadianceVignette": "◎ Vignette",
    **MP_DISPLAY_NAMES,
    
    # Phase 1: Masking & Matting display names
    "RadianceSAMModelLoader": "◎ SAM Model Loader",
    "RadianceSAMGenerator": "◎ SAM Mask Generator",
    "RadianceMultiMaskVisualPicker": "◎ SAM Multi-Mask Picker",
    "RadianceLinearMatting": "◎ Linear Alpha Matting",
    
    # Phase 2: Plate Prep display names
    "RadianceHDRGrainMatcher": "◎ HDR Grain Matcher",
    "RadianceSubpixelStabilizer": "◎ Subpixel Plate Stabilizer",
    
    # Phase 3: Regional Inpainting display names
    "RadianceHDRCrop": "◎ HDR Inpaint Crop",
    "RadianceHDRStitch": "◎ HDR Inpaint Stitch",
    "RadianceTemporalStitchStabilizer": "◎ Temporal Stitch Stabilizer",
    
    # Phase 5: Advanced Rotoscoping display names
    "RadianceVectorMaskDraw": "◎ Vector Mask Draw (Roto)",
    "RadianceVideoMaskPropagator": "◎ Video Mask Propagator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
