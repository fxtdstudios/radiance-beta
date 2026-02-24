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
        # FIX #7: previous code silently swallowed this with bare 'pass'.
        # Both apply_display_view() and apply_cdl() call numpy_to_tensor_float32()
        # — a NameError crash at runtime with no startup warning.
        # Provide a lightweight torch fallback so the module loads cleanly.
        import torch as _torch

        def numpy_to_tensor_float32(x):
            return _torch.from_numpy(x.copy() if not x.flags["C_CONTIGUOUS"] else x)


# Optional PyOpenColorIO
try:
    import PyOpenColorIO as OCIO

    HAS_OCIO = True
except ImportError:
    OCIO = None
    HAS_OCIO = False

# FIX #14: use two separate dicts instead of one shared dict with ad-hoc sub-keys.
# Previously _get_config() stored {config_path: {"config": obj}} and
# _get_processor() stored {"processors": {cache_key: cpu_proc}} in the SAME dict.
# If a user's config_path was literally the string "processors", _get_config()
# overwrote the processor sub-dict — silently breaking the processor cache.
_OCIO_CONFIG_CACHE: dict = {}  # keyed by config_path string
_OCIO_PROCESSOR_CACHE: dict = {}  # keyed by (display|view|look|ctx) string


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
                "config_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to OCIO config. Leave empty to use OCIO env variable or auto-detect ACES.",
                    },
                ),
                "display": (
                    "STRING",
                    {
                        "default": "sRGB",
                        "tooltip": "Display device (e.g., sRGB, Rec.709, P3-DCI). Use OCIOListColorspaces to see options.",
                    },
                ),
                "view": (
                    "STRING",
                    {
                        "default": "ACES 1.0 - SDR Video",
                        "tooltip": "View transform (e.g., ACES 1.0 - SDR Video, Raw, Log).",
                    },
                ),
            },
            "optional": {
                "look": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional look to apply (film emulation, creative LUT). Leave empty for none.",
                    },
                ),
                "exposure_adjust": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Exposure adjustment in stops (applied before view transform).",
                    },
                ),
                "context_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "OCIO context variable key (e.g., 'shot', 'sequence').",
                    },
                ),
                "context_value": (
                    "STRING",
                    {"default": "", "tooltip": "OCIO context variable value."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_display_view"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "OCIO Display/View transform for professional output. Supports ACES configs with Look and context variables."

    def _get_config(self, config_path: str):
        """Get or create OCIO config with caching."""
        if not HAS_OCIO:
            return None

        # FIX #14: use the dedicated config cache (not the shared processor dict)
        if config_path in _OCIO_CONFIG_CACHE:
            return _OCIO_CONFIG_CACHE[config_path]

        try:
            if config_path and os.path.exists(config_path):
                config = OCIO.Config.CreateFromFile(config_path)
            else:
                # Try environment, then local ACES
                try:
                    config = OCIO.Config.CreateFromEnv()
                except Exception:
                    current_dir = os.path.dirname(os.path.realpath(__file__))
                    radiance_dir = os.path.dirname(current_dir)
                    local_aces = os.path.join(radiance_dir, "ACES", "config.ocio")

                    if os.path.exists(local_aces):
                        config = OCIO.Config.CreateFromFile(local_aces)
                    else:
                        logger.warning("No OCIO config found. Using built-in.")
                        config = OCIO.Config.CreateBuiltinConfig("aces_cg")

            _OCIO_CONFIG_CACHE[config_path] = config
            return config

        except Exception as e:
            logger.error(f"Failed to load OCIO config: {e}")
            return None

    def _get_processor(self, config, display, view, look, context_key, context_value):
        """Get cached processor or create new one."""
        cache_key = f"{display}|{view}|{look}|{context_key}:{context_value}"

        # FIX #14: use the dedicated processor cache
        if cache_key in _OCIO_PROCESSOR_CACHE:
            return _OCIO_PROCESSOR_CACHE[cache_key]

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

            # FIX #6: apply named look via the correct OCIO 2.x API.
            # The original code called transform.setLooksBypass(False) which is
            # the DEFAULT state and adds nothing — it only avoids bypassing looks
            # already embedded in the View definition, but it does NOT apply a
            # named look. setLooks(name) is the correct call.
            if look:
                try:
                    transform.setLooks(look)
                except AttributeError:
                    # OCIO < 2.x fallback: setLooks not available
                    logger.warning(
                        f"OCIO version does not support setLooks(); "
                        f"look '{look}' will not be applied. Upgrade to OCIO 2.x."
                    )

            # Get processor
            if context:
                processor = config.getProcessor(
                    context, transform, OCIO.TRANSFORM_DIR_FORWARD
                )
            else:
                processor = config.getProcessor(transform)

            cpu_processor = processor.getDefaultCPUProcessor()

            # FIX #14: store in the dedicated processor cache
            _OCIO_PROCESSOR_CACHE[cache_key] = cpu_processor
            return cpu_processor

        except Exception as e:
            logger.error(f"Failed to create OCIO processor: {e}")
            return None

    def apply_display_view(
        self,
        image,
        config_path,
        display,
        view,
        look="",
        exposure_adjust=0.0,
        context_key="",
        context_value="",
    ):

        if not HAS_OCIO:
            logger.warning(
                "PyOpenColorIO not installed. Install with: pip install opencolorio"
            )
            return (image,)

        config = self._get_config(config_path)
        if not config:
            return (image,)

        processor = self._get_processor(
            config, display, view, look, context_key, context_value
        )
        if not processor:
            return (image,)

        try:
            img_np = image.cpu().numpy().astype(np.float32)
            batch, h, w, c = img_np.shape

            # Apply exposure adjustment if needed
            # FIX #12: apply EV gain to RGB only — alpha encodes transparency,
            # not luminance, and must not be scaled.
            if exposure_adjust != 0.0:
                ev_scale = 2.0**exposure_adjust
                if c >= 3:
                    img_np[..., :3] = img_np[..., :3] * ev_scale
                else:
                    img_np = img_np * ev_scale

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
                "cdl_file": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to .cc or .ccc CDL file. Overrides manual values if specified.",
                    },
                ),
                # Manual CDL values
                "slope_r": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "slope_g": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "slope_b": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "offset_r": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001},
                ),
                "offset_g": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001},
                ),
                "offset_b": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001},
                ),
                "power_r": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01},
                ),
                "power_g": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01},
                ),
                "power_b": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
            },
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
            import defusedxml.ElementTree as ET

            tree = ET.parse(cdl_file)
            root = tree.getroot()

            # FIX #13: some .ccc files include a namespace declaration:
            #   <ColorCorrectionCollection xmlns='urn:ASC:CDL:v1.01'>
            # ElementTree then treats the tag as '{urn:...}ColorCorrection', so
            # find('.//ColorCorrection') returns None. Use the Python 3.8+ {*}
            # wildcard to match any namespace (or no namespace).
            cc = root.find(".//{*}ColorCorrection") or root.find(
                ".//{*}ColorDecision//{*}ColorCorrection"
            )
            if cc is None:
                cc = root[0] if len(root) > 0 else None

            if cc is None:
                return None

            # Parse SOP and Sat nodes (also with namespace-agnostic lookup)
            sop = cc.find("{*}SOPNode") if "}" in (cc.tag or "") else cc.find("SOPNode")
            sat = cc.find("{*}SatNode") if "}" in (cc.tag or "") else cc.find("SatNode")
            # Simpler unified approach using {*} wildcard for both
            sop = (
                cc.find(".//{*}SOPNode") or cc.find(".//SOPNode") or cc.find("SOPNode")
            )
            sat = (
                cc.find(".//{*}SatNode") or cc.find(".//SatNode") or cc.find("SatNode")
            )

            cdl = {
                "slope": [1.0, 1.0, 1.0],
                "offset": [0.0, 0.0, 0.0],
                "power": [1.0, 1.0, 1.0],
                "saturation": 1.0,
            }

            if sop is not None:
                slope_elem = sop.find(".//{*}Slope") or sop.find("Slope")
                if slope_elem is not None and slope_elem.text:
                    cdl["slope"] = [float(x) for x in slope_elem.text.split()]

                offset_elem = sop.find(".//{*}Offset") or sop.find("Offset")
                if offset_elem is not None and offset_elem.text:
                    cdl["offset"] = [float(x) for x in offset_elem.text.split()]

                power_elem = sop.find(".//{*}Power") or sop.find("Power")
                if power_elem is not None and power_elem.text:
                    cdl["power"] = [float(x) for x in power_elem.text.split()]

            if sat is not None:
                sat_elem = sat.find(".//{*}Saturation") or sat.find("Saturation")
                if sat_elem is not None and sat_elem.text:
                    cdl["saturation"] = float(sat_elem.text)

            return cdl

        except Exception as e:
            logger.warning(f"Failed to parse CDL file: {e}")
            return None

    def apply_cdl(
        self,
        image,
        cdl_file="",
        slope_r=1.0,
        slope_g=1.0,
        slope_b=1.0,
        offset_r=0.0,
        offset_g=0.0,
        offset_b=0.0,
        power_r=1.0,
        power_g=1.0,
        power_b=1.0,
        saturation=1.0,
    ):

        # Try loading from file first
        cdl = self._parse_cdl_file(cdl_file)
        if cdl:
            slope = cdl["slope"]
            offset = cdl["offset"]
            power = cdl["power"]
            sat = cdl["saturation"]
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
                np.maximum(img_np[..., c] * slope[c] + offset[c], 0.0), power[c]
            )

        # Preserve alpha if present
        if img_np.shape[-1] > 3:
            result[..., 3:] = img_np[..., 3:]

        # Apply saturation
        if sat != 1.0:
            luma = (
                0.2126 * result[..., 0]
                + 0.7152 * result[..., 1]
                + 0.0722 * result[..., 2]
            )
            luma = luma[..., np.newaxis]
            result[..., :3] = luma + sat * (result[..., :3] - luma)

        return (numpy_to_tensor_float32(result),)


# =============================================================================
# NODE MAPPINGS
# FIX #2: NODE_CLASS_MAPPINGS was absent — both nodes were invisible to ComfyUI.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceOCIODisplayView": RadianceOCIODisplayView,
    "RadianceOCIOCDL": RadianceOCIOCDL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceOCIODisplayView": "◎ OCIO Display/View",
    "RadianceOCIOCDL": "◎ OCIO CDL",
}
