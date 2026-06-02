import logging

logger = logging.getLogger("radiance.metadata")

class RadianceLinearCheck:
    """
    ◎ Radiance Linear Check
    
    Validates that the incoming image is tagged as 'Linear'.
    If not, it can optionally raise a warning or error.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shot_metadata": ("RADIANCE_SHOT",),
                "action": (["Log Warning", "Strict Error", "Ignore"], {"default": "Log Warning"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "RADIANCE_SHOT")
    RETURN_NAMES = ("image", "shot_metadata")
    FUNCTION = "check"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = "Verify that an image tensor is in linear scene-referred colour space."

    def check(self, image, shot_metadata, action):
        cs = shot_metadata.get("colorspace", "Unknown")
        if cs != "Linear" and cs != "ACEScg" and "Linear" not in cs:
            msg = f"[Radiance Pipeline] WARNING: Non-linear colorspace '{cs}' detected in a linear-only node. This will cause incorrect grading results."
            if action == "Strict Error":
                raise ValueError(msg)
            elif action == "Log Warning":
                logger.warning(msg)
        
        return (image, shot_metadata)

NODE_CLASS_MAPPINGS = {
    "RadianceLinearCheck": RadianceLinearCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLinearCheck": "◎ Radiance Linear Check",
}
