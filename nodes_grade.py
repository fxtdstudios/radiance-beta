"""Backward-compatible re-exports. Class definitions moved to radiance.nodes.color.grade."""
import warnings
warnings.warn(
    "Import from radiance.nodes_grade is deprecated; use radiance.nodes.color.grade directly.",
    DeprecationWarning, stacklevel=2,
)

from radiance.nodes.color.grade import (
    RadianceGrade,
    RadianceApplyGradeInfo,
    RadianceGradeMatch,
    GRADE_PRESETS,
    _apply_grade,
    _match_grade_params,
    _rgb_to_lab,
)

NODE_CLASS_MAPPINGS = {
    "RadianceGrade": RadianceGrade,
    "RadianceApplyGradeInfo": RadianceApplyGradeInfo,
    "RadianceGradeMatch": RadianceGradeMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceGrade": "◎ Radiance Grade",
    "RadianceApplyGradeInfo": "◎ Radiance Apply Grade Info",
    "RadianceGradeMatch": "◎ Radiance Grade Match",
}
