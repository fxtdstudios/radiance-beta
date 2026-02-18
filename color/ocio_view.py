
import os
import logging
import numpy as np

# Module logger
logger = logging.getLogger("radiance.color.ocio_view")

# Import shared color utilities
try:
    from radiance.color_utils import numpy_to_tensor_float32
except ImportError:
    try:
        from ...color_utils import numpy_to_tensor_float32
    except ImportError:
         pass

# Optional PyOpenColorIO
try:
    import PyOpenColorIO as OCIO
    HAS_OCIO = True
except ImportError:
    OCIO = None
    HAS_OCIO = False

# OCIO Processor Cache for performance
_OCIO_PROCESSOR_CACHE = {}


class RadianceOCIODisplayView:
    """
    OCIO Display/View Transform with Look support. Professional color pipeline node for final output transforms: - Selects display device (sRGB, Rec.709, P3-DCI, etc.) - Selects view transform (ACES, Raw, Log, etc.) - Optional look application (Film looks, CDLs) - Processor caching for performance v1.1: Full OCIO 2.x support with context variables
    """
    
    # Common displays (fallbacks if config not available)
    DEFAULT_DISPLAYS = ["sRGB", "Rec.709", "P3-DCI", "P3-D65", "Rec.2020"]
    DEFAULT_VIEWS = ["ACES 1.0 - SDR Video", "Raw", "Log", "Un-tone-mapped"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "config_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to OCIO config. Leave empty to use OCIO env variable or auto-detect ACES."
                }),
                "display": ("STRING", {
                    "default": "sRGB",
                    "tooltip": "Display device (e.g., sRGB, Rec.709, P3-DCI). Use OCIOListColorspaces to see options."
                }),
                "view": ("STRING", {
                    "default": "ACES 1.0 - SDR Video",
                    "tooltip": "View transform (e.g., ACES 1.0 - SDR Video, Raw, Log)."
                }),
            },
            "optional": {
                "look": ("STRING", {
                    "default": "",
                    "tooltip": "Optional look to apply (film emulation, creative LUT). Leave empty for none."
                }),
                "exposure_adjust": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Exposure adjustment in stops (applied before view transform)."
                }),
                "context_key": ("STRING", {
                    "default": "",
                    "tooltip": "OCIO context variable key (e.g., 'shot', 'sequence')."
                }),
                "context_value": ("STRING", {
                    "default": "",
                    "tooltip": "OCIO context variable value."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_display_view"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "OCIO Display/View transform for professional output. Supports ACES configs with Look and context variables."
    
    def _get_config(self, config_path: str):
        """Get or create OCIO config with caching."""
        if not HAS_OCIO:
            return None
        
        # Check cache
        if config_path in _OCIO_PROCESSOR_CACHE:
            cached = _OCIO_PROCESSOR_CACHE[config_path]
            if cached.get("config"):
                return cached["config"]
        
        try:
            if config_path and os.path.exists(config_path):
                config = OCIO.Config.CreateFromFile(config_path)
            else:
                # Try environment, then local ACES
                try:
                    config = OCIO.Config.CreateFromEnv()
                except Exception:
                    # Check for local ACES config
                    # Assumes structure: radiance/color/ocio_view.py -> radiance/ACES/config.ocio
                    # Current file: radiance/color/ocio_view.py
                    # Parent: radiance/color
                    # Grandparent: radiance
                    # ACES is in radiance/ACES
                    
                    current_dir = os.path.dirname(os.path.realpath(__file__)) # radiance/color
                    radiance_dir = os.path.dirname(current_dir) # radiance
                    local_aces = os.path.join(radiance_dir, "ACES", "config.ocio")
                    
                    if os.path.exists(local_aces):
                        config = OCIO.Config.CreateFromFile(local_aces)
                    else:
                        logger.warning("No OCIO config found. Using built-in.")
                        config = OCIO.Config.CreateBuiltinConfig("aces_cg")
            
            # Cache the config
            _OCIO_PROCESSOR_CACHE[config_path] = {"config": config}
            return config
            
        except Exception as e:
            logger.error(f"Failed to load OCIO config: {e}")
            return None
    
    def _get_processor(self, config, display, view, look, context_key, context_value):
        """Get cached processor or create new one."""
        cache_key = f"{display}|{view}|{look}|{context_key}:{context_value}"
        
        cached = _OCIO_PROCESSOR_CACHE.get("processors", {})
        if cache_key in cached:
            return cached[cache_key]
        
        try:
            # Create context if needed
            context = None
            if context_key and context_value:
                context = config.getCurrentContext().createEditableCopy()
                context.setStringVar(context_key, context_value)
            
            # Create DisplayViewTransform
            transform = OCIO.DisplayViewTransform()
            transform.setSrc(OCIO.ROLE_SCENE_LINEAR)
            transform.setDisplay(display)
            transform.setView(view)
            
            # Add look if specified
            if look:
                transform.setLooksBypass(False)
                # Note: Looks are typically applied via the view in ACES configs
                # For explicit look control, use LookTransform separately
            
            # Get processor
            if context:
                processor = config.getProcessor(context, transform, OCIO.TRANSFORM_DIR_FORWARD)
            else:
                processor = config.getProcessor(transform)
            
            cpu_processor = processor.getDefaultCPUProcessor()
            
            # Cache processor
            if "processors" not in _OCIO_PROCESSOR_CACHE:
                _OCIO_PROCESSOR_CACHE["processors"] = {}
            _OCIO_PROCESSOR_CACHE["processors"][cache_key] = cpu_processor
            
            return cpu_processor
            
        except Exception as e:
            logger.error(f"Failed to create OCIO processor: {e}")
            return None
    
    def apply_display_view(self, image, config_path, display, view, 
                           look="", exposure_adjust=0.0,
                           context_key="", context_value=""):
        
        if not HAS_OCIO:
            logger.warning("PyOpenColorIO not installed. Install with: pip install opencolorio")
            return (image,)
        
        config = self._get_config(config_path)
        if not config:
            return (image,)
        
        processor = self._get_processor(config, display, view, look, context_key, context_value)
        if not processor:
            return (image,)
        
        try:
            img_np = image.cpu().numpy().astype(np.float32)
            batch, h, w, c = img_np.shape
            
            # Apply exposure adjustment if needed
            if exposure_adjust != 0.0:
                img_np = img_np * (2.0 ** exposure_adjust)
            
            # Flatten for OCIO processing
            flat = img_np.reshape(-1, c)
            
            # Process RGB channels
            if c >= 3:
                rgb = flat[:, :3].copy()
                processor.applyRGB(rgb)
                flat[:, :3] = rgb
            
            result = flat.reshape(batch, h, w, c)
            return (numpy_to_tensor_float32(result),)
            
        except Exception as e:
            logger.error(f"OCIO Display/View error: {e}")
            return (image,)


class RadianceOCIOCDL:
    """
    OCIO CDL (Color Decision List) Transform. Applies ASC CDL (American Society of Cinematographers Color Decision List) operations for on-set grading and shot matching: - Slope: Multiply (gain per channel) - Offset: Add (lift per channel) - Power: Exponent (gamma per channel) - Saturation: Global saturation control Can also load CDL values from .cc or .ccc files.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # CDL file input
                "cdl_file": ("STRING", {
                    "default": "",
                    "tooltip": "Path to .cc or .ccc CDL file. Overrides manual values if specified."
                }),
                # Manual CDL values
                "slope_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "slope_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "slope_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "offset_r": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "offset_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "offset_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "power_r": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "power_g": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "power_b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_cdl"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "ASC CDL transform for shot matching. Supports .cc/.ccc files or manual SOP values."
    
    def _parse_cdl_file(self, cdl_file: str):
        """Parse CDL values from .cc or .ccc XML file."""
        if not cdl_file or not os.path.exists(cdl_file):
            return None
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(cdl_file)
            root = tree.getroot()
            
            # Find ColorCorrection or first child
            cc = root.find('.//ColorCorrection') or root.find('.//ColorDecision//ColorCorrection')
            if cc is None:
                cc = root[0] if len(root) > 0 else None
            
            if cc is None:
                return None
            
            # Parse SOP node
            sop = cc.find('SOPNode')
            sat = cc.find('SatNode')
            
            cdl = {
                'slope': [1.0, 1.0, 1.0],
                'offset': [0.0, 0.0, 0.0],
                'power': [1.0, 1.0, 1.0],
                'saturation': 1.0
            }
            
            if sop is not None:
                slope_elem = sop.find('Slope')
                if slope_elem is not None and slope_elem.text:
                    cdl['slope'] = [float(x) for x in slope_elem.text.split()]
                    
                offset_elem = sop.find('Offset')
                if offset_elem is not None and offset_elem.text:
                    cdl['offset'] = [float(x) for x in offset_elem.text.split()]
                    
                power_elem = sop.find('Power')
                if power_elem is not None and power_elem.text:
                    cdl['power'] = [float(x) for x in power_elem.text.split()]
            
            if sat is not None:
                sat_elem = sat.find('Saturation')
                if sat_elem is not None and sat_elem.text:
                    cdl['saturation'] = float(sat_elem.text)
            
            return cdl
            
        except Exception as e:
            logger.warning(f"Failed to parse CDL file: {e}")
            return None
    
    def apply_cdl(self, image, cdl_file="",
                  slope_r=1.0, slope_g=1.0, slope_b=1.0,
                  offset_r=0.0, offset_g=0.0, offset_b=0.0,
                  power_r=1.0, power_g=1.0, power_b=1.0,
                  saturation=1.0):
        
        # Try loading from file first
        cdl = self._parse_cdl_file(cdl_file)
        if cdl:
            slope = cdl['slope']
            offset = cdl['offset']
            power = cdl['power']
            sat = cdl['saturation']
        else:
            slope = [slope_r, slope_g, slope_b]
            offset = [offset_r, offset_g, offset_b]
            power = [power_r, power_g, power_b]
            sat = saturation
        
        # Apply CDL transform: out = (in * slope + offset) ^ power
        img_np = image.cpu().numpy().astype(np.float32)
        
        # SOP per channel
        result = np.empty_like(img_np)
        for c in range(3):
            result[..., c] = np.power(
                np.maximum(img_np[..., c] * slope[c] + offset[c], 0.0),
                power[c]
            )
        
        # Preserve alpha if present
        if img_np.shape[-1] > 3:
            result[..., 3:] = img_np[..., 3:]
        
        # Apply saturation
        if sat != 1.0:
            luma = 0.2126 * result[..., 0] + 0.7152 * result[..., 1] + 0.0722 * result[..., 2]
            luma = luma[..., np.newaxis]
            result[..., :3] = luma + sat * (result[..., :3] - luma)
        
        return (numpy_to_tensor_float32(result),)
