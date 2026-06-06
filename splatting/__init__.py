"""Radiance Gaussian Splatting library (CPU IO + data types).

GPU rendering/training (gsplat) is added in later phases; this package's data
model and .ply IO are dependency-light (numpy only) and import without CUDA.
"""
from __future__ import annotations

from radiance.splatting.data import Splat
from radiance.splatting.ply import load_ply, save_ply

__all__ = ["Splat", "load_ply", "save_ply"]
