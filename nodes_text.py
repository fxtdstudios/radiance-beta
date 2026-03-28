import json
import logging

logger = logging.getLogger("◎ Radiance.text")


class AnyType(str):
    """
    A special class that is always equal in not equal checks.
    This allows you to connect any type of node to an input socket.
    """

    def __ne__(self, __value: object) -> bool:
        return False


# Global AnyType instance for the wildcard input
any_type = AnyType("*")


class RadianceShowText:
    """A utility node for displaying text or data output from other nodes."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_input": (
                    any_type,
                    {
                        "forceInput": True,
                        "tooltip": "Connect any string, number, dictionary, or list to display it.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/Utilities"
    DESCRIPTION = "Displays any input data as text directly on the node UI."

    def notify(self, any_input, unique_id=None, extra_pnginfo=None):
        out_text = []

        # Process the incoming list of inputs
        for item in any_input:
            if isinstance(item, str):
                out_text.append(item)
            elif isinstance(item, (int, float, bool)):
                out_text.append(str(item))
            elif isinstance(item, (dict, list)):
                try:
                    out_text.append(json.dumps(item, indent=2))
                except Exception:
                    out_text.append(str(item))
            else:
                out_text.append(str(item))

        # The 'ui' dictionary sends data to the frontend JavaScript node via onExecuted
        return {"ui": {"text": out_text}, "result": (out_text,)}


NODE_CLASS_MAPPINGS = {
    "◎ RadianceShowText": RadianceShowText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "◎ RadianceShowText": "◎ Radiance Show Text",
}
