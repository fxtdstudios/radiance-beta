"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE LAYOUT NODES
              Advanced Reroute & Workflow Organization
═══════════════════════════════════════════════════════════════════════════════
"""

import math


class RadianceReroute:
    """Basic pass-through reroute for backward compatibility."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"data": ("*",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "route"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    DESCRIPTION = "A pass-through node for organizing workflows."

    def route(self, data):
        return (data,)


class RadianceAdvancedReroute:
    """
    Advanced Reroute with label, color presets, and auto-type detection.
    Renders as a compact pill in the frontend via radiance_layout.js.
    """

    COLOR_OPTIONS = [
        "Auto",      # Picks color based on connected type
        "Gray",
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Cyan",
        "Blue",
        "Purple",
        "Magenta",
        "White",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*",),
            },
            "optional": {
                "label": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Label..."
                }),
                "color": (s.COLOR_OPTIONS, {"default": "Auto"}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("data",)
    FUNCTION = "route"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    DESCRIPTION = "Advanced reroute with labels and color coding. Use 'Auto' color to match connected data type."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always pass through — never cache
        return math.nan

    def route(self, data, label="", color="Auto"):
        return (data,)


NODE_CLASS_MAPPINGS = {
    "RadianceReroute": RadianceReroute,
    "RadianceAdvancedReroute": RadianceAdvancedReroute,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceReroute": "◎ Radiance Reroute",
    "RadianceAdvancedReroute": "◎ Radiance Reroute+",
}
