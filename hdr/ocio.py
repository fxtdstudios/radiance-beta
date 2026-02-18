import logging
logger = logging.getLogger("radiance.hdr.ocio")

import os
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch

from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32
try:
    # Try relative package import
    from ..color_utils import (
        SRGB_TO_ACESCG, ACESCG_TO_SRGB,
        linear_to_logc4, logc4_to_linear,
        linear_to_srgb, srgb_to_linear
    )
    HAS_COLOR_UTILS = True
except ImportError:
    try:
        # Fallback to local import if running as script from package root
        # This handles cases where 'radiance' is in sys.path but not treated as package
        from radiance.color_utils import (
            SRGB_TO_ACESCG, ACESCG_TO_SRGB,
            linear_to_logc4, logc4_to_linear,
            linear_to_srgb, srgb_to_linear
        )
        HAS_COLOR_UTILS = True
    except ImportError:
        try:
            # Last resort: try importing directly from sys.path
            from color_utils import (
                SRGB_TO_ACESCG, ACESCG_TO_SRGB,
                linear_to_logc4, logc4_to_linear,
                linear_to_srgb, srgb_to_linear
            )
            HAS_COLOR_UTILS = True
        except ImportError:
            HAS_COLOR_UTILS = False
            logger.warning("Radiance OCIO: color_utils not found. Fallback transforms disabled.")

# Check for PyOpenColorIO
try:
    import PyOpenColorIO as OCIO
    HAS_OCIO = True
except ImportError:
    HAS_OCIO = False
    logger.info("PyOpenColorIO not found. Install with: pip install opencolorio")

# ═══════════════════════════════════════════════════════════════════════════════
#                          OCIO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class ACESConfigManager:
    """
    Detect, download, and manage ACES OCIO configurations for professional color workflows.
    """
    
    # Known ACES config locations
    COMMON_PATHS = [
        # Windows
        r"C:\ACES\config.ocio",
        r"C:\Program Files\ACES\config.ocio",
        r"C:\Users\Public\ACES\config.ocio",
        # Linux/Mac
        "/opt/ACES/config.ocio",
        "/usr/share/ACES/config.ocio",
        "~/ACES/config.ocio",
    ]
    
    # Official ACES 2.0 config download URL (ACES 2.0 / OCIO 2.5) - v4.0.0
    ACES2_CONFIG_URL = "https://github.com/AcademySoftwareFoundation/OpenColorIO-Config-ACES/releases/download/v4.0.0/cg-config-v4.0.0_aces-v2.0_ocio-v2.5.ocio"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "action": (["Detect Config", "Download ACES 2.0", "List Colorspaces", "Get Config Info"], {
                    "default": "Detect Config",
                    "tooltip": "Action to perform. Detect = find existing config. Download = get official ACES 2.0 v4.0.0."
                }),
            },
            "optional": {
                "custom_config_path": ("STRING", {
                    "default": "",
                    "tooltip": "Custom path to OCIO config file. Leave empty to use auto-detection."
                }),
                "install_path": ("STRING", {
                    "default": "",
                    "tooltip": "Where to install downloaded ACES config. Default: 'ACES' folder inside Radiance node directory."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("config_path", "colorspaces_list", "status_info")
    FUNCTION = "manage_config"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Detect, download, and manage ACES OCIO configurations for professional color workflows."
    OUTPUT_NODE = True  # Force node to execute even if outputs aren't connected
    
    def _find_existing_config(self) -> Tuple[str, str]:
        """Find existing ACES config on the system."""
        try:
            # Check environment variable first
            env_config = os.environ.get("OCIO", "")
            if env_config and os.path.exists(env_config):
                return env_config, f"Found config from OCIO environment: {env_config}"
            
            # Check common paths
            for path in self.COMMON_PATHS:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    return expanded_path, f"Found config at: {expanded_path}"
            
            # Check ComfyUI models folder (safely)
            try:
                import folder_paths
                models_dir = folder_paths.models_dir
                aces_path = os.path.join(models_dir, "ACES", "config.ocio")
                if os.path.exists(aces_path):
                    return aces_path, f"Found config in ComfyUI models: {aces_path}"
            except ImportError:
                pass  # folder_paths not available, skip this check
            
            # Check local radiance folder
            current_dir = os.path.dirname(os.path.realpath(__file__))
            # Go up one level from 'hdr' to 'radiance'
            radiance_dir = os.path.dirname(current_dir)
            local_aces = os.path.join(radiance_dir, "ACES", "config.ocio")
            if os.path.exists(local_aces):
                return local_aces, f"Found config in Radiance folder: {local_aces}"
            
            return "", "No ACES config found. Use 'Download ACES 2.0' to install."
        except Exception as e:
            return "", f"Error detecting config: {str(e)}"
    
    def _list_colorspaces(self, config_path: str) -> str:
        """List available colorspaces in the config."""
        if not HAS_OCIO:
            return "PyOpenColorIO not installed. Run: pip install opencolorio"
        
        if not config_path or not os.path.exists(config_path):
            return "No valid config path provided."
        
        try:
            config = OCIO.Config.CreateFromFile(config_path)
            colorspaces = []
            for i in range(config.getNumColorSpaces()):
                name = config.getColorSpaceNameByIndex(i)
                colorspaces.append(name)
            
            # Group by category for readability
            result = f"Config: {os.path.basename(config_path)}\n"
            result += f"Total Colorspaces: {len(colorspaces)}\n"
            result += "-" * 40 + "\n"
            result += "\n".join(colorspaces[:50])  # Limit output
            if len(colorspaces) > 50:
                result += f"\n... and {len(colorspaces) - 50} more"
            
            return result
        except Exception as e:
            return f"Error reading config: {str(e)}"
    
    def _download_aces_config(self, install_path: str) -> Tuple[str, str]:
        """Download the official ACES 2.0 config."""
        import urllib.request
        
        # Default install path: "ACES" folder inside radiance package
        if not install_path:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            radiance_dir = os.path.dirname(current_dir)
            install_path = os.path.join(radiance_dir, "ACES")
        
        os.makedirs(install_path, exist_ok=True)
        config_file = os.path.join(install_path, "config.ocio")
        
        # Check if already exists
        if os.path.exists(config_file):
            return config_file, f"ACES config already exists at: {config_file}"
        
        try:
            # Download the config
            logger.info(f"Downloading ACES 2.0 config to {config_file}...")
            # Use a custom user agent to avoid 403 errors on some systems
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(self.ACES2_CONFIG_URL, config_file)
            
            # Set environment variable (optional, for this session)
            os.environ["OCIO"] = config_file
            
            return config_file, f"Successfully downloaded ACES 2.0 config to: {config_file}\nOCIO environment variable set."
            
        except Exception as e:
            return "", f"Download failed: {str(e)}\n\nManual download URL:\n{self.ACES2_CONFIG_URL}"
    
    def _get_config_info(self, config_path: str) -> str:
        """Get detailed info about the config."""
        if not HAS_OCIO:
            return "PyOpenColorIO not installed. Run: pip install opencolorio"
        
        if not config_path or not os.path.exists(config_path):
            return "No valid config path provided."
        
        try:
            config = OCIO.Config.CreateFromFile(config_path)
            
            info = []
            info.append(f"Config Path: {config_path}")
            info.append(f"Description: {config.getDescription()}")
            info.append(f"Search Paths: {config.getSearchPath()}")
            info.append(f"Colorspaces: {config.getNumColorSpaces()}")
            info.append(f"Displays: {', '.join([config.getDisplay(i) for i in range(config.getNumDisplays())])}")
            info.append(f"Looks: {config.getNumLooks()}")
            info.append(f"")
            info.append("Common ACES Colorspaces:")
            info.append("  - ACES - ACEScg (working space)")
            info.append("  - ACES - ACES2065-1 (archival)")
            info.append("  - Output - sRGB (web/SDR)")
            info.append("  - Output - Rec.2100-PQ (HDR)")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"Error reading config: {str(e)}"
    
    def manage_config(self, action: str, custom_config_path: str = "",
                     install_path: str = "") -> Tuple[str, str, str]:
        
        # Wrap everything in try-except to ensure we ALWAYS return a valid tuple
        try:
            config_path = custom_config_path if custom_config_path else ""
            colorspaces_list = ""
            status_info = ""
            
            if action == "Detect Config":
                if not custom_config_path:
                    config_path, status_info = self._find_existing_config()
                else:
                    config_path = custom_config_path
                    status_info = f"Using custom config: {config_path}" if os.path.exists(config_path) else "Custom config not found!"
                
                if config_path and os.path.exists(config_path):
                    colorspaces_list = self._list_colorspaces(config_path)
            
            elif action == "Download ACES 2.0":
                config_path, status_info = self._download_aces_config(install_path)
                if config_path:
                    colorspaces_list = self._list_colorspaces(config_path)
            
            elif action == "List Colorspaces":
                if not custom_config_path:
                    config_path, _ = self._find_existing_config()
                else:
                    config_path = custom_config_path
                colorspaces_list = self._list_colorspaces(config_path)
                status_info = f"Listed colorspaces from: {config_path}"
            
            elif action == "Get Config Info":
                if not custom_config_path:
                    config_path, _ = self._find_existing_config()
                else:
                    config_path = custom_config_path
                status_info = self._get_config_info(config_path)
            
            return (config_path, colorspaces_list, status_info)
            
        except Exception as e:
            # Ensure we ALWAYS return a valid tuple, even on catastrophic failure
            error_msg = f"ACESConfigManager error: {str(e)}"
            logger.error(f"ACESConfigManager: {error_msg}")
            return ("", "", error_msg)


class OCIOListColorspaces:
    """
    List all colorspaces available in an OCIO configuration file.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "ocio_config_path": ("STRING", {"default": "", "multiline": False}),
                "filter_role": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("colorspaces_list",)
    FUNCTION = "list_spaces"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "List all colorspaces available in an OCIO configuration file."
    OUTPUT_NODE = True  # Force execution even when outputs aren't connected

    def list_spaces(self, ocio_config_path: str = "", 
                    filter_role: str = "") -> Tuple[str]:
        
        if not HAS_OCIO:
            return ("PyOpenColorIO not installed. pip install opencolorio",)

        try:
            # 1. Try provided path
            if ocio_config_path and os.path.exists(ocio_config_path):
                config = OCIO.Config.CreateFromFile(ocio_config_path)
            else:
                # 2. Try environment variable
                try:
                    config = OCIO.GetCurrentConfig()
                except Exception as e:
                    logger.debug(f"OCIO GetCurrentConfig failed: {e}")
                    # 3. Try local radiance ACES folder
                    current_dir = os.path.dirname(os.path.realpath(__file__))
                    # Go up one level from 'hdr' to 'radiance'
                    radiance_dir = os.path.dirname(current_dir)
                    local_aces = os.path.join(radiance_dir, "ACES", "config.ocio")
                    if os.path.exists(local_aces):
                        config = OCIO.Config.CreateFromFile(local_aces)
                    else:
                        return ("No OCIO config found. Use ACESConfigManager to download or set OCIO environment variable.",)

            # Get colorspaces
            spaces = []
            for i in range(config.getNumColorSpaces()):
                name = config.getColorSpaceNameByIndex(i)
                cs = config.getColorSpace(name)
                family = cs.getFamily() if cs else ""
                spaces.append(f"{name} [{family}]" if family else name)

            # Get looks
            looks = [config.getLookNameByIndex(i) for i in range(config.getNumLooks())]

            # Get displays and views
            displays = []
            for d in range(config.getNumDisplays()):
                display = config.getDisplay(d)
                views = [config.getView(display, v) for v in range(config.getNumViews(display))]
                displays.append(f"{display}: {', '.join(views)}")

            result = "═══ OCIO COLORSPACES ═══\n"
            result += "\n".join(spaces[:50])  # Limit to 50
            if len(spaces) > 50:
                result += f"\n... and {len(spaces) - 50} more"
            
            if looks:
                result += "\n\n═══ LOOKS ═══\n"
                result += "\n".join(looks)
            
            if displays:
                result += "\n\n═══ DISPLAYS ═══\n"
                result += "\n".join(displays)

            return (result,)

        except Exception as e:
            return (f"Error: {str(e)}",)




NODE_CLASS_MAPPINGS = {
    "ACESConfigManager": ACESConfigManager,
    "OCIOListColorspaces": OCIOListColorspaces,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACESConfigManager": "ACES Config Manager",
    "OCIOListColorspaces": "List OCIO Colorspaces",
}
