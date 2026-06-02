"""Backward-compatible re-exports. Class definitions moved to radiance.nodes.color.qc."""
import warnings
warnings.warn(
    "Import from radiance.nodes_qc is deprecated; use radiance.nodes.color.qc directly.",
    DeprecationWarning, stacklevel=2,
)

from radiance.nodes.color.qc import (
    RadianceQC,
    RadiancePolicyGuard,
    RadianceQCExport,
    _luma,
    _mean_saturation,
    _gamut_out_of_p3,
    _policy_analyse,
    _evaluate_policy,
)

NODE_CLASS_MAPPINGS = {
    "RadianceQC": RadianceQC,
    "RadiancePolicyGuard": RadiancePolicyGuard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceQC": "◎ Radiance QC / Export",
    "RadiancePolicyGuard": "◎ Radiance Policy Preset / Guard",
}
