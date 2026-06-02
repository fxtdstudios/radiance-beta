"""Backward-compatible re-exports. Class definitions moved to radiance.nodes.color.curves."""
import warnings
warnings.warn(
    "Import from radiance.nodes_curves is deprecated; use radiance.nodes.color.curves directly.",
    DeprecationWarning, stacklevel=2,
)

from radiance.nodes.color.curves import (
    RadianceHueCurves,
    RadianceCurves,
    _rgb_to_hsl,
    _hsl_to_rgb_fast as _hsl_to_rgb,
)

NODE_CLASS_MAPPINGS = {
    "RadianceHueCurves": RadianceHueCurves,
    "RadianceCurves": RadianceCurves,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHueCurves": "◎ Radiance Hue Curves",
    "RadianceCurves": "◎ Radiance Curves",
}
