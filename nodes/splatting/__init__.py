"""Radiance Gaussian Splatting node group (IO + render + training)."""
from __future__ import annotations

from radiance.nodes.splatting import io as _io
from radiance.nodes.splatting import render as _render
from radiance.nodes.splatting import train as _train

NODE_CLASS_MAPPINGS = {
    **_io.NODE_CLASS_MAPPINGS,
    **_render.NODE_CLASS_MAPPINGS,
    **_train.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_io.NODE_DISPLAY_NAME_MAPPINGS,
    **_render.NODE_DISPLAY_NAME_MAPPINGS,
    **_train.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
