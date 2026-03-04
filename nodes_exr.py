"""
Radiance EXR Saver - Facade
Redirects to the modular implementation in radiance.hdr.io.
"""

import logging

logger = logging.getLogger("radiance.exr")

_IMPORT_OK = False
RadianceSaveEXR = None

try:
    # 1. Package relative import (Standard for ComfyUI)
    from .hdr.io import RadianceSaveEXR
    _IMPORT_OK = True
except ImportError:
    try:
        # 2. Absolute package import (If custom_nodes in sys.path)
        from radiance.hdr.io import RadianceSaveEXR
        _IMPORT_OK = True
    except ImportError:
        try:
            # 3. Direct module import (Dev/Test env where CWD is package root)
            from hdr.io import RadianceSaveEXR
            _IMPORT_OK = True
        except ImportError:
            logger.error(
                "Radiance: Failed to import RadianceSaveEXR from hdr.io. "
                "The EXR node will not be available. "
                "Install OpenEXR: pip install OpenEXR"
            )

# Only register the node if import succeeded — never register a broken class
if _IMPORT_OK and RadianceSaveEXR is not None:
    NODE_CLASS_MAPPINGS = {
        "RadianceSaveEXR": RadianceSaveEXR,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "RadianceSaveEXR": "◎ Radiance Save EXR/HDR",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
