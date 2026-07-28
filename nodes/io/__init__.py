"""Input/output node group."""
from __future__ import annotations

import logging

from radiance.nodes_io import (
    RadianceRead,
    RadianceWrite,
    RadianceEXRMultiPart,
    RadianceDigitalCinemaRead,
    RadianceDigitalCinemaWrite,
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
    # The v3 reorganisation transcribed these mapping dicts by hand and dropped
    # entries on the way. The classes were written, complete and importable the
    # whole time -- they simply never appeared in ComfyUI's node menu.
    "RadianceDigitalCinemaRead": RadianceDigitalCinemaRead,
    "RadianceDigitalCinemaWrite": RadianceDigitalCinemaWrite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceRead": "◎ Radiance Read",
    "RadianceWrite": "◎ Radiance Write",
    "RadianceEXRMultiPart": "◎ Radiance EXR Multi-Part",
    "RadianceLoadImageMask": "◎ Radiance Load Image Mask",
    "RadianceDigitalCinemaRead": "◎ Radiance Digital Cinema Read",
    "RadianceDigitalCinemaWrite": "◎ Radiance Digital Cinema Write",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
