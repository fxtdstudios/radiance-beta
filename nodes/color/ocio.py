import os
import logging
import folder_paths
from radiance.radiance_ocio import get_ocio_manager, HAS_OCIO
from radiance.path_utils import strip_path_quotes

logger = logging.getLogger("radiance.ocio")


class RadianceOCIOContext:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Load an OpenColorIO config into Radiance's shared OCIO manager for this session (used, for example, by Radiance Color Space Convert as its OCIO path). It does not set OCIO context variables."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_path": ("STRING", {
                    "default": "C:/ACES/config.ocio", "multiline": False,
                    "tooltip": "Path to a config.ocio file. A relative path is looked for in ComfyUI's input/ then output/ folder. The loaded config replaces the session-wide one until another is loaded.",
                }),
                "working_space": ("STRING", {
                    "default": "ACES - ACEScg",
                    "tooltip": "Working colour space name, stored in the ocio_context output only. It is not validated against the config and no node currently reads it.",
                }),
            },
        }

    RETURN_TYPES = ("RADIANCE_OCIO",)
    RETURN_NAMES = ("ocio_context",)
    FUNCTION = "set_context"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"

    def set_context(self, config_path, working_space):
        config_path = strip_path_quotes(config_path)
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
