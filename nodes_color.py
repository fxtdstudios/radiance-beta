
from .color import (
    RadianceLUTApply,
    RadianceLUTBlend,
    RadianceGPUColorMatrix,
    RadianceOCIOColorTransform,
    RadianceSceneLinearWorkflow,
    RadianceLogCurveDecode,
    RadianceLogCurveEncode,
    RadianceACES2OutputTransform,
    RadianceOCIODisplayView,
    RadianceOCIOCDL
)

NODE_CLASS_MAPPINGS = {
    "RadianceLUTApply": RadianceLUTApply,
    "RadianceGPUColorMatrix": RadianceGPUColorMatrix,
    "RadianceOCIOColorTransform": RadianceOCIOColorTransform,
    "RadianceOCIODisplayView": RadianceOCIODisplayView,
    "RadianceOCIOCDL": RadianceOCIOCDL,
    "RadianceSceneLinearWorkflow": RadianceSceneLinearWorkflow,
    "RadianceLogCurveDecode": RadianceLogCurveDecode,
    "RadianceLogCurveEncode": RadianceLogCurveEncode,
    "RadianceACES2OutputTransform": RadianceACES2OutputTransform,
    "RadianceLUTBlend": RadianceLUTBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLUTApply": "◎ Radiance LUT Apply",
    "RadianceGPUColorMatrix": "◎ Radiance Color Matrix",
    "RadianceOCIOColorTransform": "◎ Radiance OCIO Color Transform",
    "RadianceOCIODisplayView": "◎ Radiance OCIO Display/View",
    "RadianceOCIOCDL": "◎ Radiance CDL Transform",
    "RadianceSceneLinearWorkflow": "◎ Radiance Workflow Presets",
    "RadianceLogCurveDecode": "◎ Radiance Log Curve Decode",
    "RadianceLogCurveEncode": "◎ Radiance Log Curve Encode",
    "RadianceACES2OutputTransform": "◎ Radiance ACES 2.0 Output",
    "RadianceLUTBlend": "◎ Radiance LUT Blend",
}
