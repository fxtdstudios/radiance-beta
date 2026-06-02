"""Lightweight compare/check viewer node for Radiance."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image as PILImage

import folder_paths

from radiance.path_utils import safe_join
from radiance.viewer_utils import compute_data_range, image_video_type, safe_tensor_to_numpy

logger = logging.getLogger("radiance.lite_viewer")


class RadianceLiteViewer:
    """
    Fast viewer for compare/check workflows.

    This intentionally avoids the full Radiance Viewer HDR sidecars, scopes,
    grading state, and export helpers. It writes compact temp PNG previews and
    lets the frontend handle quick canvas compare modes.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": (image_video_type,),
            },
            "optional": {
                "compare_image": (
                    image_video_type,
                    {"tooltip": "Optional B image for wipe, split, diff, and onion checks."},
                ),
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
    DESCRIPTION = (
        "Fast lightweight viewer for compare/check workflows. "
        "Provides fit, 1:1, pan/zoom, wipe, split, diff, onion, clipping, alpha, and pixel inspect."
    )

    def view(
        self,
        image: Any,
        compare_image: Optional[Any] = None,
        unique_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        image = self._extract_tensor(image)
        compare_image = self._extract_tensor(compare_image) if compare_image is not None else None

        validation_error = self._validate_image(image, "image")
        if validation_error:
            logger.error(validation_error)
            return {"ui": {"radiance_lite_images": [], "error": [validation_error]}, "result": (image,)}

        try:
            output_dir = folder_paths.get_temp_directory()
            batch_size = image.shape[0] if image.dim() == 4 else 1
            run_id = uuid.uuid4().hex[:12]
            images: List[Dict[str, Any]] = []

            for frame_idx in range(batch_size):
                frame = self._write_frame(image, frame_idx, output_dir, run_id, is_compare=False)
                if frame:
                    frame["frame"] = frame_idx
                    frame["total_frames"] = batch_size
                    images.append(frame)

            if compare_image is not None:
                compare_error = self._validate_image(compare_image, "compare_image")
                if compare_error:
                    logger.warning(compare_error)
                else:
                    compare_count = compare_image.shape[0] if compare_image.dim() == 4 else 1
                    for frame_idx in range(compare_count):
                        frame = self._write_frame(compare_image, frame_idx, output_dir, run_id, is_compare=True)
                        if frame:
                            frame["frame"] = frame_idx
                            frame["total_frames"] = compare_count
                            frame["is_compare"] = True
                            images.append(frame)

            return {
                "ui": {
                    "radiance_lite_images": images,
                    "instance_id": [str(unique_id) if unique_id else run_id],
                    "batch_size": [batch_size],
                },
                "result": (image,),
            }
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.exception("Radiance Lite Viewer failed")
            return {"ui": {"radiance_lite_images": [], "error": [str(exc)]}, "result": (image,)}

    def _extract_tensor(self, value: Any) -> Any:
        if hasattr(value, "get_components"):
            try:
                components = value.get_components()
                if hasattr(components, "images"):
                    return components.images
            except Exception as exc:
                logger.debug("Could not extract video components: %s", exc)
        if isinstance(value, dict) and "samples" in value:
            return value["samples"]
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
            return value[0]
        return value

    def _validate_image(self, image: Any, name: str) -> Optional[str]:
        if not isinstance(image, torch.Tensor):
            return f"{name} must be a torch.Tensor, got {type(image)}"
        if image.dim() not in (3, 4):
            return f"{name} must be 3D or 4D tensor, got {image.dim()}D"
        shape = image.shape[1:] if image.dim() == 4 else image.shape
        if len(shape) != 3:
            return f"{name} must have HWC image layout"
        h, w, c = shape
        if h <= 0 or w <= 0:
            return f"{name} has invalid dimensions: {w}x{h}"
        if c not in (1, 3, 4):
            return f"{name} has {c} channels, expected 1, 3, or 4"
        return None

    def _write_frame(
        self,
        image: torch.Tensor,
        frame_idx: int,
        output_dir: str,
        run_id: str,
        is_compare: bool,
    ) -> Optional[Dict[str, Any]]:
        frame = safe_tensor_to_numpy(image[frame_idx] if image.dim() == 4 else image)
        if not np.isfinite(frame).all():
            frame = np.nan_to_num(frame, nan=0.0, posinf=65504.0, neginf=0.0)

        d_min, d_max, has_hdr, hdr_stats = compute_data_range(frame)
        preview = self._to_preview_rgba(frame, has_hdr=has_hdr, d_max=d_max)

        filename = f"radiance_lite_{'b' if is_compare else 'a'}_{run_id}_{frame_idx}.png"
        filepath = safe_join(output_dir, filename)
        PILImage.fromarray(preview, mode="RGBA").save(filepath, compress_level=1)

        h, w = frame.shape[:2]
        return {
            "filename": filename,
            "subfolder": "",
            "type": "temp",
            "width": int(w),
            "height": int(h),
            "data_range": [float(d_min), float(d_max)],
            "hdr_stats": hdr_stats,
            "has_hdr": bool(has_hdr),
            "is_compare": bool(is_compare),
        }

    def _to_preview_rgba(self, frame: np.ndarray, has_hdr: bool, d_max: float) -> np.ndarray:
        arr = frame.astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            alpha = np.ones(arr.shape[:2] + (1,), dtype=np.float32)
            rgb = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            rgb = arr[..., :3]
            alpha = np.clip(arr[..., 3:4], 0.0, 1.0)
        else:
            rgb = arr[..., :3]
            alpha = np.ones(arr.shape[:2] + (1,), dtype=np.float32)

        rgb = np.maximum(rgb, 0.0)
        if has_hdr or d_max > 1.05:
            rgb = rgb / (1.0 + rgb)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgba = np.concatenate([rgb, alpha], axis=-1)
        return (rgba * 255.0 + 0.5).astype(np.uint8)


NODE_CLASS_MAPPINGS = {
    "RadianceLiteViewer": RadianceLiteViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLiteViewer": "◎ Radiance Lite Viewer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "RadianceLiteViewer"]
