"""Retired: the Lite Viewer (3.5.0).

`◎ Radiance Viewer` has a Simple mode now (picture, compare and transport,
nothing else), which is what the Lite Viewer was for. Keeping two viewers
meant two frontends, two preview paths and two sets of compare bugs, so the
Lite Viewer is retired the way this pack retires nodes: hidden from the menu,
still loaded so saved graphs open. A saved Lite Viewer runs through the
Radiance Viewer's code and opens in the Radiance Viewer's frontend, in Simple
mode.

Its inputs are unchanged, in the same order, so the widget values a saved graph
restores by position still land on `input_space` and `fps`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from radiance.viewer_utils import image_video_type
from radiance.nodes.monitor.viewer import VIEWER_INPUT_SPACES, RadianceViewer


class RadianceLiteViewer:
    """Retired alias of the Radiance Viewer in Simple mode. Loads, does not list."""

    DEPRECATED = True   # hidden from the menu; saved graphs still open

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": (image_video_type, {
                    "tooltip": "IMAGE batch or VIDEO to check. Returned unchanged as an IMAGE."}),
            },
            "optional": {
                "compare_image": (image_video_type, {
                    "tooltip": "Optional B side for the viewer's compare modes."}),
                "input_space": (VIEWER_INPUT_SPACES, {
                    "default": "Auto",
                    "tooltip": "What the incoming pixels are, as on the Radiance Viewer."}),
                "fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Playback rate. 0 = the source's rate (VIDEO input), 24 for an image batch."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Review"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "view"
    OUTPUT_NODE = True
    DESCRIPTION = ("Retired in 3.5.0: use the Radiance Viewer, which opens in Simple mode. "
                   "Saved graphs still open and run through the Radiance Viewer.")

    @classmethod
    def IS_CHANGED(cls, image=None, unique_id=None, **kwargs):
        return RadianceViewer.IS_CHANGED(image=image, unique_id=unique_id, **kwargs)

    def view(
        self,
        image: Any,
        compare_image: Optional[Any] = None,
        input_space: str = "Auto",
        fps: float = 0.0,
        unique_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return RadianceViewer().view(
            image=image,
            compare_image=compare_image,
            input_space=input_space,
            fps=fps,
            unique_id=unique_id,
        )
