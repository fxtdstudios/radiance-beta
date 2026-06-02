"""
radiance/nodes/monitor/scopes.py
──────────────────────────────────────────────────────────────────────────────
Video scope and comparison nodes extracted from nodes_radiance_viewer.py:
  • RadianceScopesNode  — Waveform, RGB Parade, Histogram, Vectorscope

Scope computation functions (waveform, histogram, vectorscope) are defined in
viewer_utils.py and imported here.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("radiance.nodes.monitor.scopes")


NODE_CLASS_MAPPINGS = {}

NODE_DISPLAY_NAME_MAPPINGS = {}
