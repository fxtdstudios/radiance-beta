from .lut import RadianceLUTApply, RadianceLUTBlend, LUTCache
from .transform import (
    RadianceGPUColorMatrix,
    RadianceOCIOColorTransform,
    RadianceSceneLinearWorkflow,
    RadianceLogCurveDecode,
    RadianceLogCurveEncode,
)
from .ocio_view import RadianceOCIODisplayView, RadianceOCIOCDL

NODE_CLASS_MAPPINGS = {
    "RadianceLUTApply": RadianceLUTApply,
    "RadianceLUTBlend": RadianceLUTBlend,
    "RadianceGPUColorMatrix": RadianceGPUColorMatrix,
    "RadianceOCIOColorTransform": RadianceOCIOColorTransform,
    "RadianceSceneLinearWorkflow": RadianceSceneLinearWorkflow,
    "RadianceLogCurveDecode": RadianceLogCurveDecode,
    "RadianceLogCurveEncode": RadianceLogCurveEncode,
    "RadianceOCIODisplayView": RadianceOCIODisplayView,
    "RadianceOCIOCDL": RadianceOCIOCDL,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLUTApply": "◎ Radiance LUT Apply",
    "RadianceLUTBlend": "◎ Radiance LUT Blend",
    "RadianceGPUColorMatrix": "◎ Radiance Color Matrix",
    "RadianceOCIOColorTransform": "◎ Radiance OCIO Color Transform",
    "RadianceSceneLinearWorkflow": "◎ Radiance Workflow Presets",
    "RadianceLogCurveDecode": "◎ Radiance Log Curve Decode",
    "RadianceLogCurveEncode": "◎ Radiance Log Curve Encode",
    "RadianceOCIODisplayView": "◎ Radiance OCIO Display/View",
    "RadianceOCIOCDL": "◎ Radiance CDL Transform",
    
}
