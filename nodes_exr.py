"""
Radiance EXR Saver - Facade
Redirects to the modular implementation in radiance.hdr.io.
"""
import logging

try:
    # 1. Package relative import (Standard for ComfyUI)
    from .hdr.io import RadianceSaveEXR
except ImportError:
    try:
        # 2. Absolute package import (If custom_nodes in sys.path)
        from radiance.hdr.io import RadianceSaveEXR
    except ImportError:
        try:
             # 3. Direct module import (Dev/Test env where CWD is package root)
            from hdr.io import RadianceSaveEXR
        except ImportError:
            # Fallback/Dummy if import into fails
            logging.error("Radiance: Failed to import RadianceSaveEXR from hdr.io")
            class RadianceSaveEXR: pass
            """
            Save images as EXR files with full HDR and metadata support.
            """

NODE_CLASS_MAPPINGS = {
    "RadianceSaveEXR": RadianceSaveEXR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSaveEXR": "◎ Radiance Save EXR/HDR",
}
