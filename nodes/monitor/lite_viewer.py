"""Lightweight compare/check viewer node for Radiance."""
from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import struct
import zlib

import numpy as np
import torch
from PIL import Image as PILImage

import folder_paths

from radiance.path_utils import safe_join
from radiance.viewer_utils import compute_data_range, image_video_type, safe_tensor_to_numpy
from radiance.color.display_preview import display_preview
from radiance.nodes.monitor.viewer import (
    VIEWER_INPUT_SPACES,
    _video_fps,
    resolve_viewer_input_space,
)

logger = logging.getLogger("radiance.lite_viewer")

# ── Preview sizing ─────────────────────────────────────────────────────────────
# The docstring below has always said "compact temp PNG previews", but
# _to_preview_rgba did no downscaling at all: this node wrote one FULL
# RESOLUTION RGBA PNG per frame. At 4K that is ~30 MB a frame before
# compression, and the frontend draws the result into a canvas it fits to the
# panel, so nothing consumed the extra pixels. This is the size the preview is
# actually looked at, with headroom for a 1:1 zoom on a 2K panel.
LITE_PREVIEW_MAX_DIM = 2048

# ── Temp-file bookkeeping ──────────────────────────────────────────────────────
# This node had no purge path whatsoever: `grep -r radiance_lite --include=*.py`
# matched only this file. Every execution wrote a fresh uuid-named PNG per
# frame, so nothing collided and nothing overwrote, and scrubbing a 240-frame
# shot and re-queueing ten times left 2,400 full-resolution PNGs in temp/ until
# ComfyUI restarted. Mirrors the tracking RadianceViewer already does.
_LITE_TEMP_FILES: Dict[str, List[str]] = {}
_LITE_TEMP_LOCK = threading.Lock()


def _lite_track_temp(instance_key: str, paths: List[str]) -> None:
    """Record the files written for this instance's current execution."""
    if not instance_key:
        return
    with _LITE_TEMP_LOCK:
        _LITE_TEMP_FILES[instance_key] = list(paths)


def _lite_purge_temp(instance_key: str) -> int:
    """Delete the previous execution's previews for this instance."""
    if not instance_key:
        return 0
    with _LITE_TEMP_LOCK:
        stale = _LITE_TEMP_FILES.pop(instance_key, [])
    removed = 0
    for path in stale:
        try:
            os.unlink(path)
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Could not remove stale lite viewer temp %s: %s", path, exc)
    if removed:
        logger.debug("Purged %d stale lite viewer preview(s) for %s", removed, instance_key)
    return removed


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
                "image": (image_video_type, {
                    "tooltip": "IMAGE batch or VIDEO to check. Returned unchanged as an IMAGE; "
                               "input_space decides how it is displayed."}),
            },
            "optional": {
                "compare_image": (
                    image_video_type,
                    {"tooltip": "Optional B image for wipe, split, diff, and onion checks."},
                ),
                # 3.5.0, appended so saved widget values keep their positions.
                "input_space": (
                    VIEWER_INPUT_SPACES,
                    {
                        "default": "Auto",
                        "tooltip": (
                            "What the incoming pixels are. sRGB (a ComfyUI IMAGE) is shown "
                            "untouched; linear sources are shown through OpenColorIO ACES 2.0 "
                            "SDR, the same view as the Radiance Viewer. Auto: linear when any "
                            "value is above 1.0 or below 0."
                        ),
                    },
                ),
                "fps": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                     "tooltip": "Playback rate. 0 = the source's rate (VIDEO input), 24 for an image batch."},
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
        input_space: str = "Auto",
        fps: float = 0.0,
        unique_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_fps = _video_fps(image)
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

            # Remove the previous execution's previews before writing new ones.
            purge_key = str(unique_id).strip() if unique_id and str(unique_id).strip() else str(id(self))
            _lite_purge_temp(purge_key)

            view_space = resolve_viewer_input_space(input_space, image)
            for frame_idx in range(batch_size):
                frame = self._write_frame(image, frame_idx, output_dir, run_id, is_compare=False,
                                          view_space=view_space)
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
                    cmp_space = resolve_viewer_input_space(input_space, compare_image)
                    for frame_idx in range(compare_count):
                        frame = self._write_frame(compare_image, frame_idx, output_dir, run_id, is_compare=True,
                                                  view_space=cmp_space)
                        if frame:
                            frame["frame"] = frame_idx
                            frame["total_frames"] = compare_count
                            frame["is_compare"] = True
                            images.append(frame)

            _lite_track_temp(
                purge_key,
                [safe_join(output_dir, entry[k]) for entry in images
                 for k in ("filename", "float_filename") if entry.get(k)],
            )

            play_fps = float(fps) if fps and fps > 0 else (source_fps or 24.0)
            return {
                "ui": {
                    "radiance_lite_images": images,
                    "fps": [play_fps],
                    "source_encoding": [view_space[0]],
                    "source_colorspace": [view_space[1]],
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
        view_space: tuple = ("srgb", "sRGB Encoded Rec.709 (sRGB)"),
    ) -> Optional[Dict[str, Any]]:
        frame = safe_tensor_to_numpy(image[frame_idx] if image.dim() == 4 else image)
        if not np.isfinite(frame).all():
            frame = np.nan_to_num(frame, nan=0.0, posinf=65504.0, neginf=0.0)

        d_min, d_max, has_hdr, hdr_stats = compute_data_range(frame)
        h, w = frame.shape[:2]

        # 3.5.0: downscale the FLOAT frame first (area average, the proxy a
        # review tool shows), then put it through the view. The probe, clip
        # check and diff read this same float proxy, so the numbers on screen
        # are source values, not 8-bit display codes.
        proxy = self._float_proxy(frame)
        preview = self._to_preview_rgba(proxy, view_space)

        filename = f"radiance_lite_{'b' if is_compare else 'a'}_{run_id}_{frame_idx}.png"
        filepath = safe_join(output_dir, filename)
        pil = PILImage.fromarray(preview, mode="RGBA")
        pil.save(filepath, compress_level=1)

        float_name = f"radiance_lite_{'b' if is_compare else 'a'}_{run_id}_{frame_idx}.rhdr"
        float_saved = self._write_float_proxy(proxy, safe_join(output_dir, float_name))

        h, w = frame.shape[:2]
        return {
            "filename": filename,
            "subfolder": "",
            "type": "temp",
            "width": int(w),
            "height": int(h),
            # The preview is a proxy when these differ from width/height.
            "preview_width": int(pil.width),
            "preview_height": int(pil.height),
            "float_filename": float_name if float_saved else None,
            "source_encoding": view_space[0],
            "source_colorspace": view_space[1],
            "data_range": [float(d_min), float(d_max)],
            "hdr_stats": hdr_stats,
            "has_hdr": bool(has_hdr),
            "is_compare": bool(is_compare),
        }

    @staticmethod
    def _float_proxy(frame: np.ndarray) -> np.ndarray:
        """Float RGBA proxy, longest side <= LITE_PREVIEW_MAX_DIM (area average)."""
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = np.concatenate([np.repeat(arr, 3, axis=-1), np.ones_like(arr)], axis=-1)
        elif arr.shape[-1] == 3:
            arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,), np.float32)], axis=-1)
        h, w = arr.shape[:2]
        if max(h, w) <= LITE_PREVIEW_MAX_DIM:
            return arr
        scale = LITE_PREVIEW_MAX_DIM / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        try:
            import cv2  # type: ignore
            return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001 - PIL float resize per channel
            chans = [np.asarray(PILImage.fromarray(arr[..., c], mode="F").resize((nw, nh), PILImage.BOX))
                     for c in range(arr.shape[-1])]
            return np.stack(chans, axis=-1).astype(np.float32)

    @staticmethod
    def _write_float_proxy(proxy: np.ndarray, path: str) -> bool:
        """fp16 RHDR (the Viewer's sidecar format) of the proxy, for the probe."""
        try:
            h, w, c = proxy.shape
            data = np.clip(proxy, -65504.0, 65504.0).astype(np.float16).tobytes()
            with open(path, "wb") as f:
                f.write(struct.pack("<4sHHHH", b"RHDR", w, h, c, 0))
                f.write(zlib.compress(data, level=1))
            return True
        except (OSError, ValueError, struct.error) as exc:
            logger.warning("Lite Viewer float proxy not written: %s", exc)
            return False

    def _to_preview_rgba(self, frame: np.ndarray, view_space: tuple = ("srgb", "")) -> np.ndarray:
        """Display-referred 8-bit RGBA through the same view as the Viewer.

        3.5.0: this used to switch the whole frame to x/(1+x) with no display
        encoding as soon as one pixel passed 1.05 (0.5 became 0.33, linear
        sources looked dark), and write everything else as if it were sRGB.
        """
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = np.concatenate([np.repeat(arr, 3, axis=-1), np.ones_like(arr)], axis=-1)
        elif arr.shape[-1] == 3:
            arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,), np.float32)], axis=-1)
        encoding, colorspace = view_space
        out = display_preview(arr, encoding, colorspace)
        return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


NODE_CLASS_MAPPINGS = {
    "RadianceLiteViewer": RadianceLiteViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLiteViewer": "◎ Radiance Lite Viewer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "RadianceLiteViewer"]
