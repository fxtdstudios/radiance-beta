import os
import logging
import folder_paths
from radiance.radiance_ocio import get_ocio_manager, HAS_OCIO

logger = logging.getLogger("radiance.ocio")


class RadianceOCIOContext:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Set OpenColorIO context variables for environment-aware transforms."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_path": ("STRING", {
                    "default": "C:/ACES/config.ocio", "multiline": False,
                }),
                "working_space": ("STRING", {
                    "default": "ACES - ACEScg",
                }),
            },
        }

    RETURN_TYPES = ("RADIANCE_OCIO",)
    RETURN_NAMES = ("ocio_context",)
    FUNCTION = "set_context"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"

    def set_context(self, config_path, working_space):
        mgr = get_ocio_manager()
        if not HAS_OCIO:
            logger.warning("[OCIO Context] PyOpenColorIO not installed.")
            return ({"status": "inactive", "error": "OCIO not installed"},)

        if not os.path.isabs(config_path):
            for search_dir in [folder_paths.get_input_directory(), folder_paths.get_output_directory()]:
                candidate = os.path.join(search_dir, config_path)
                if os.path.exists(candidate):
                    config_path = candidate
                    break

        success = mgr.load_config(config_path)
        if not success:
            logger.error(f"[OCIO Context] Failed to load config: {config_path}")
            return ({"status": "error", "path": config_path},)

        context = {
            "status": "active", "path": config_path,
            "working_space": working_space, "config_name": mgr.config_name,
        }
        return (context,)
