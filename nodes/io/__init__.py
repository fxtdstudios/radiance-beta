"""Input/output node group."""
from __future__ import annotations

import logging

from radiance.nodes_io import (
    RadianceRead,
    RadianceWrite,
    RadianceEXRMultiPart,
)
from radiance.nodes.io.mask import (
    RadianceLoadImageMask,
)

logger = logging.getLogger("radiance.nodes.io")

NODE_CLASS_MAPPINGS = {
    "RadianceRead": RadianceRead,
    "RadianceWrite": RadianceWrite,
    "RadianceEXRMultiPart": RadianceEXRMultiPart,
    "RadianceLoadImageMask": RadianceLoadImageMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceRead": "◎ Radiance Read",
    "RadianceWrite": "◎ Radiance Write",
    "RadianceEXRMultiPart": "◎ Radiance EXR Multi-Part",
    "RadianceLoadImageMask": "◎ Radiance Load Image Mask",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
