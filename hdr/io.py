import torch
import numpy as np
import os
import json
import logging
import folder_paths
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Local imports
from .utils import numpy_to_tensor_float32

import datetime
import struct
import zlib

# Imports for RadianceSaveEXR
try:
    from ..color_utils import (
        linear_to_logc3,
        linear_to_logc4,
        linear_to_slog3,
        srgb_to_linear,
        linear_to_srgb,
        apply_matrix_transform,
        AWG3_TO_ACESCG,
        AWG4_TO_ACESCG,
        SGAMUT3_CINE_TO_ACESCG,
        ACESCG_TO_SRGB,
    )
    from ..path_utils import safe_join, get_safe_output_dir
except ImportError:
    try:
        from radiance.color_utils import (
            linear_to_logc3,
            linear_to_logc4,
            linear_to_slog3,
            srgb_to_linear,
            linear_to_srgb,
            apply_matrix_transform,
            AWG3_TO_ACESCG,
            AWG4_TO_ACESCG,
            SGAMUT3_CINE_TO_ACESCG,
            ACESCG_TO_SRGB,
        )
        from radiance.path_utils import safe_join, get_safe_output_dir
    except ImportError:
        # Last resort fallback if needed, or logging
        pass

logger = logging.getLogger("radiance.hdr.io")

# ═══════════════════════════════════════════════════════════════════════════════
#                           EXR INPUT NODES
# ═══════════════════════════════════════════════════════════════════════════════


class LoadImageEXR:
    """
    Load EXR/HDR files with full HDR dynamic range. Enter file path directly or select from input folder.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        input_dir = folder_paths.get_input_directory()
        files = []
        try:
            for f in os.listdir(input_dir):
                if f.lower().endswith((".exr", ".hdr")):
                    files.append(f)
        except Exception:  # nosec B110
            pass

        # Build file list with "none" as first option
        file_options = (
            ["-- Select from input folder --"] + sorted(files)
            if files
            else ["-- No EXR files in input folder --"]
        )

        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Full path to EXR/HDR file (e.g., C:/images/my_image.exr). Leave empty to use dropdown.",
                    },
                ),
            },
            "optional": {
                "input_folder_file": (
                    file_options,
                    {
                        "tooltip": "Select EXR/HDR from ComfyUI's input folder. Only used if file_path is empty."
                    },
                ),
                "exposure_adjust": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Adjust exposure in stops (EV). Positive = brighter, negative = darker.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "alpha_mask", "depth_mask", "metadata")
    FUNCTION = "load"
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = "Load EXR/HDR files with full HDR dynamic range. Enter file path directly or select from input folder."

    @classmethod
    def IS_CHANGED(cls, file_path, **kwargs):
        # Determine which path to use
        actual_path = cls._get_actual_path(
            file_path, kwargs.get("input_folder_file", "")
        )
        if not actual_path:
            return float("nan")
        try:
            if os.path.exists(actual_path):
                return os.path.getmtime(actual_path)
        except Exception:  # nosec B110
            pass
        return float("nan")

    @classmethod
    def _get_actual_path(cls, file_path: str, input_folder_file: str) -> str:
        """Determine the actual file path to load."""

        # Priority 1: Direct file path if provided
        if file_path and file_path.strip():
            # Strip whitespace and quotes (users often paste paths with quotes)
            cleaned_path = file_path.strip().strip('"').strip("'").strip()
            if cleaned_path:
                return cleaned_path

        # Priority 2: Input folder selection
        if input_folder_file and not input_folder_file.startswith("--"):
            return folder_paths.get_annotated_filepath(input_folder_file)

        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, file_path, **kwargs):
        input_folder_file = kwargs.get("input_folder_file", "")

        actual_path = cls._get_actual_path(file_path, input_folder_file)

        if not actual_path:
            return "No file specified. Either enter a file path OR select a file from the input folder dropdown."

        if not os.path.exists(actual_path):
            return f"File not found: '{actual_path}'"

        if not actual_path.lower().endswith((".exr", ".hdr")):
            return f"Invalid file type. Only .exr and .hdr files are supported. Got: '{actual_path}'"

        return True

    def load(
        self, file_path: str, input_folder_file: str = "", exposure_adjust: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Load EXR/HDR file using multiple backends for maximum compatibility."""

        # Get the actual path to load
        actual_path = self._get_actual_path(file_path, input_folder_file)

        if not actual_path:
            raise ValueError(
                "No file path specified. Enter a file path or select from dropdown."
            )

        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"File not found: {actual_path}")

        img = None
        alpha_img = None
        depth_img = None
        load_method = "unknown"

        # Method 0: Try OpenEXR library (most reliable, just installed)
        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(actual_path)
            header = exr_file.header()

            dw = header["dataWindow"]
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Get channel names
            channels = list(header["channels"].keys())

            # Read pixel data
            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            if "R" in channels and "G" in channels and "B" in channels:
                # RGB or RGBA
                r_str = exr_file.channel("R", pt)
                g_str = exr_file.channel("G", pt)
                b_str = exr_file.channel("B", pt)

                r = np.frombuffer(r_str, dtype=np.float32).copy().reshape(height, width)
                g = np.frombuffer(g_str, dtype=np.float32).copy().reshape(height, width)
                b = np.frombuffer(b_str, dtype=np.float32).copy().reshape(height, width)

                if "A" in channels:
                    a_str = exr_file.channel("A", pt)
                    alpha_img = (
                        np.frombuffer(a_str, dtype=np.float32)
                        .copy()
                        .reshape(height, width)
                    )
                else:
                    alpha_img = np.ones((height, width), dtype=np.float32)

                # RGB only
                img = np.stack([r, g, b], axis=-1)
            elif "Y" in channels:
                # Grayscale -> RGB
                y_str = exr_file.channel("Y", pt)
                y = np.frombuffer(y_str, dtype=np.float32).copy().reshape(height, width)
                img = np.stack([y, y, y], axis=-1)

                if "A" in channels:
                    a_str = exr_file.channel("A", pt)
                    alpha_img = (
                        np.frombuffer(a_str, dtype=np.float32)
                        .copy()
                        .reshape(height, width)
                    )
                else:
                    alpha_img = np.ones((height, width), dtype=np.float32)
            else:
                # Use first available channel -> RGB
                ch_name = channels[0]
                ch_str = exr_file.channel(ch_name, pt)
                c = (
                    np.frombuffer(ch_str, dtype=np.float32)
                    .copy()
                    .reshape(height, width)
                )
                img = np.stack([c, c, c], axis=-1)
                alpha_img = np.ones((height, width), dtype=np.float32)

            # Extract Depth/Z if present
            # Common names: 'Z', 'Depth', 'depth', 'z'
            depth_ch = None
            for dname in ["Z", "Depth", "depth", "z"]:
                if dname in channels:
                    depth_ch = dname
                    break

            if depth_ch:
                z_str = exr_file.channel(depth_ch, pt)
                depth_img = (
                    np.frombuffer(z_str, dtype=np.float32).copy().reshape(height, width)
                )
            else:
                depth_img = np.zeros((height, width), dtype=np.float32)

            exr_file.close()
            load_method = "OpenEXR"
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"OpenEXR library failed: {e}")

        # Method 1: Try imageio (good compatibility)
        if img is None:
            try:
                import imageio.v3 as iio

                img = iio.imread(actual_path)
                load_method = "imageio"
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"imageio failed: {e}")

        # Method 2: Try OpenImageIO (industry standard)
        if img is None:
            try:
                import OpenImageIO as oiio

                inp = oiio.ImageInput.open(actual_path)
                if inp:
                    spec = inp.spec()
                    img = np.zeros(
                        (spec.height, spec.width, spec.nchannels), dtype=np.float32
                    )
                    inp.read_image(0, 0, oiio.FLOAT, img)
                    inp.close()
                    load_method = "OpenImageIO"
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"OpenImageIO failed: {e}")

        # Method 3: Try OpenCV (may not have EXR support)
        if img is None:
            try:
                import cv2

                # Enable OpenEXR if available
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                img = cv2.imread(actual_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    # Convert BGR to RGB
                    if len(img.shape) == 3:
                        if img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                        elif img.shape[2] >= 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    load_method = "OpenCV"
            except Exception as e:
                logger.warning(f"OpenCV failed: {e}")

        # Method 4: Try pyexr
        if img is None:
            try:
                import pyexr

                img = pyexr.read(actual_path)
                load_method = "pyexr"
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"pyexr failed: {e}")

        if img is None:
            raise RuntimeError(
                f"Failed to load EXR file: {actual_path}\n\n"
                "Please install one of these packages:\n"
                "  pip install imageio[pyav]\n"
                "  pip install imageio-ffmpeg\n"
                "  pip install pyexr\n"
                "  pip install OpenImageIO\n"
            )

        if img is None:
            raise RuntimeError(f"Failed to load EXR: {actual_path}")

        # Normalize shapes
        # Imageio/OpenCV fallbacks might have returned (H, W, 4) or (H, W) or (H, W, 3)
        img = np.asarray(img, dtype=np.float32)  # Ensure float32

        if len(img.shape) == 2:
            # Grayscale -> RGB
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                # RGBA -> RGB + Alpha
                if alpha_img is None:
                    alpha_img = img[:, :, 3]
                img = img[:, :, :3]
            elif img.shape[2] == 1:
                # Grayscale (H, W, 1) -> RGB
                img = np.repeat(img, 3, axis=2)

        # Ensure alpha/depth are defined
        if alpha_img is None:
            alpha_img = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)

        if depth_img is None:
            depth_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        # Determine depth range for metadata
        d_min = float(depth_img.min())
        d_max = float(depth_img.max())

        # Apply exposure adjustment
        if exposure_adjust != 0:
            img = img * (2.0**exposure_adjust)

        # Build metadata
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        # Calculate dynamic range safely
        min_val = float(img.min())
        max_val = float(img.max())
        if min_val > 0 and max_val > 0:
            dynamic_range = float(np.log2(max_val / min_val))
        else:
            dynamic_range = 0.0

        metadata = {
            "file": os.path.basename(actual_path),
            "path": actual_path,
            "width": w,
            "height": h,
            "channels": channels,
            "dtype": str(img.dtype),
            "min": min_val,
            "max": max_val,
            "dynamic_range_stops": dynamic_range,
            "dynamic_range_stops": dynamic_range,
            "depth_range": f"{d_min:.2f} - {d_max:.2f}",
            "load_method": load_method,
        }

        metadata_str = json.dumps(metadata, indent=2)

        # Return:
        # - Image (batch, H, W, 3) because numpy_to_tensor_float32 adds batch if missing
        # - Alpha (batch, H, W)
        # - Depth (batch, H, W)

        alpha_tensor = torch.from_numpy(alpha_img).unsqueeze(0)  # (1, H, W)
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0)  # (1, H, W)

        return (numpy_to_tensor_float32(img), alpha_tensor, depth_tensor, metadata_str)


class LoadImageEXRSequence:
    """
    Load EXR/HDR image sequence from a folder. Returns a batch of images.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Full path to folder containing EXR sequence (e.g., C:/renders/sequence/)",
                    },
                ),
            },
            "optional": {
                "file_pattern": (
                    "STRING",
                    {
                        "default": "*.exr",
                        "multiline": False,
                        "tooltip": "File pattern to match (e.g., *.exr, render.*.exr, frame_????.exr)",
                    },
                ),
                "start_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "tooltip": "First frame to load (0 = from beginning)",
                    },
                ),
                "end_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "tooltip": "Last frame to load (0 = to end)",
                    },
                ),
                "frame_step": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Load every Nth frame",
                    },
                ),
                "max_frames": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Maximum frames to load (0 = no limit, use with caution!). Default 1000 prevents OOM.",
                    },
                ),
                "exposure_adjust": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Adjust exposure in stops (EV)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "INT")
    RETURN_NAMES = ("images", "alpha_masks", "depth_masks", "metadata", "frame_count")
    FUNCTION = "load_sequence"
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = (
        "Load EXR/HDR image sequence from a folder. Returns a batch of images."
    )

    @classmethod
    def IS_CHANGED(cls, folder_path, **kwargs):
        folder_path = folder_path.strip().strip('"').strip("'")
        if not folder_path or not os.path.isdir(folder_path):
            return float("nan")
        try:
            return os.path.getmtime(folder_path)
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, folder_path, **kwargs):
        folder_path = folder_path.strip().strip('"').strip("'")
        if not folder_path:
            return "No folder path specified."
        if not os.path.isdir(folder_path):
            return f"Folder not found: '{folder_path}'"
        return True

    def _load_single_exr(
        self, file_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single EXR file using the best available method. Returns (RGB, Alpha, Depth)."""
        img = None
        alpha_img = None
        depth_img = None

        # Method 1: OpenEXR library
        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(file_path)
            header = exr_file.header()

            dw = header["dataWindow"]
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            channels = list(header["channels"].keys())
            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            if "R" in channels and "G" in channels and "B" in channels:
                r_str = exr_file.channel("R", pt)
                g_str = exr_file.channel("G", pt)
                b_str = exr_file.channel("B", pt)

                r = np.frombuffer(r_str, dtype=np.float32).copy().reshape(height, width)
                g = np.frombuffer(g_str, dtype=np.float32).copy().reshape(height, width)
                b = np.frombuffer(b_str, dtype=np.float32).copy().reshape(height, width)

                if "A" in channels:
                    a_str = exr_file.channel("A", pt)
                    alpha_img = (
                        np.frombuffer(a_str, dtype=np.float32)
                        .copy()
                        .reshape(height, width)
                    )
                else:
                    alpha_img = np.ones((height, width), dtype=np.float32)

                # RGB only
                img = np.stack([r, g, b], axis=-1)
            elif "Y" in channels:
                # Grayscale -> RGB
                y_str = exr_file.channel("Y", pt)
                y = np.frombuffer(y_str, dtype=np.float32).copy().reshape(height, width)
                img = np.stack([y, y, y], axis=-1)

                if "A" in channels:
                    a_str = exr_file.channel("A", pt)
                    alpha_img = (
                        np.frombuffer(a_str, dtype=np.float32)
                        .copy()
                        .reshape(height, width)
                    )
                else:
                    alpha_img = np.ones((height, width), dtype=np.float32)
            else:
                # Use first available channel -> RGB
                ch_name = channels[0] if channels else "Y"
                ch_str = exr_file.channel(ch_name, pt)
                c = (
                    np.frombuffer(ch_str, dtype=np.float32)
                    .copy()
                    .reshape(height, width)
                )
                img = np.stack([c, c, c], axis=-1)
                alpha_img = np.ones((height, width), dtype=np.float32)

            # Extract Depth/Z if present
            depth_img = None
            depth_ch = None
            for dname in ["Z", "Depth", "depth", "z"]:
                if dname in channels:
                    depth_ch = dname
                    break

            if depth_ch:
                z_str = exr_file.channel(depth_ch, pt)
                depth_img = (
                    np.frombuffer(z_str, dtype=np.float32).copy().reshape(height, width)
                )
            else:
                depth_img = np.zeros((height, width), dtype=np.float32)

            exr_file.close()
            return img, alpha_img, depth_img
        except Exception as e:
            logger.debug(f"OpenEXR load failed for {file_path}: {e}")

        # Method 2: imageio
        if img is None:
            try:
                import imageio.v3 as iio

                img = iio.imread(file_path)
                # Imageio basic read likely doesn't separate depth easily or it's part of channels if multi-channel
                # For safety, return None for depth if not using OpenEXR
                return np.asarray(img, dtype=np.float32), None, None
            except Exception as e:
                logger.debug(f"imageio load failed for {file_path}: {e}")

        # Method 3: OpenCV
        if img is None:
            try:
                import cv2

                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    if len(img.shape) == 3 and img.shape[2] >= 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img.astype(np.float32), None, None
            except Exception as e:
                logger.debug(f"OpenCV load failed for {file_path}: {e}")

        return None, None, None

    def load_sequence(
        self,
        folder_path: str,
        file_pattern: str = "*.exr",
        start_frame: int = 0,
        end_frame: int = 0,
        frame_step: int = 1,
        max_frames: int = 0,
        exposure_adjust: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        """Load EXR sequence from folder with memory optimization."""
        import glob
        import re

        # Clean path
        folder_path = folder_path.strip().strip('"').strip("'")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        # Find matching files
        pattern = os.path.join(folder_path, file_pattern)
        files = glob.glob(pattern)

        if not files:
            raise ValueError(
                f"No files matching pattern '{file_pattern}' in folder: {folder_path}"
            )

        # Sort files naturally (handle frame numbers correctly)
        def natural_sort_key(s):
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split("([0-9]+)", s)
            ]

        files = sorted(files, key=natural_sort_key)

        # Apply frame range
        if start_frame > 0:
            files = files[start_frame:]
        if end_frame > 0 and end_frame < len(files) + start_frame:
            files = files[: end_frame - start_frame + 1]

        # Apply frame step
        if frame_step > 1:
            files = files[::frame_step]

        # Apply max_frames limit with warnings
        num_files = len(files)
        if max_frames > 0 and num_files > max_frames:
            logger.warning(
                f"Limiting sequence load: {num_files} files found, loading first {max_frames} "
                f"(set max_frames=0 in node to load all)"
            )
            files = files[:max_frames]
        elif max_frames == 0 and num_files > 1000:
            # Warn if user is loading a very large sequence without limits
            estimated_gb = (num_files * 8 * 1024 * 1024 * 4) / (
                1024**3
            )  # Rough estimate for 4K float32
            logger.warning(
                f"Loading {num_files} frames with max_frames=0 (unlimited). "
                f"Estimated memory: ~{estimated_gb:.1f}GB for 4K. Set max_frames to limit."
            )

        if not files:
            raise ValueError("No frames to load after applying range/step filters.")

        logger.info(f"Loading {len(files)} frames from {folder_path}")

        # 1. Load first frame to determine dimensions
        try:
            first_img, first_alpha, first_depth = self._load_single_exr(files[0])
            if first_img is None:
                raise RuntimeError(f"Failed to load first frame: {files[0]}")

            # Normalize first frame shape
            # Imageio/OpenCV fallbacks might have returned (H, W, 4) or (H, W) or (H, W, 3)
            first_img = np.asarray(first_img, dtype=np.float32)  # Ensure float32

            if len(first_img.shape) == 2:
                # Grayscale -> RGB
                first_img = np.stack([first_img, first_img, first_img], axis=-1)
            elif len(first_img.shape) == 3:
                if first_img.shape[2] == 4:
                    # RGBA -> RGB + Alpha
                    if first_alpha is None:
                        first_alpha = first_img[:, :, 3]
                    first_img = first_img[:, :, :3]
                elif first_img.shape[2] == 1:
                    # Grayscale (H, W, 1) -> RGB
                    first_img = np.repeat(first_img, 3, axis=2)

            if first_alpha is None:
                first_alpha = np.ones(
                    (first_img.shape[0], first_img.shape[1]), dtype=np.float32
                )

            if first_depth is None:
                first_depth = np.zeros(
                    (first_img.shape[0], first_img.shape[1]), dtype=np.float32
                )

            h, w, c = first_img.shape
            dtype = torch.float32  # ComfyUI uses float32 tensors

            logger.info(f"Sequence definition: {w}x{h} ({c} channels)")

        except Exception as e:
            raise RuntimeError(
                f"Could not determine sequence format from first frame: {e}"
            )

        # 2. Pre-allocate tensor memory (B, H, W, C)
        # We process on CPU to avoid allocating huge VRAM chunks immediately
        num_frames = len(files)
        batch_tensor = torch.zeros((num_frames, h, w, c), dtype=dtype)
        batch_alpha = torch.zeros((num_frames, h, w), dtype=dtype)
        batch_depth = torch.zeros((num_frames, h, w), dtype=dtype)

        loaded_count = 0
        loaded_files = []

        # 3. Load frames directly into pre-allocated tensor
        for i, file_path in enumerate(files):
            try:
                # If it's the first frame, we already loaded it
                if i == 0:
                    img, alpha_img, depth_img = first_img, first_alpha, first_depth
                else:
                    img, alpha_img, depth_img = self._load_single_exr(file_path)

                    if img is None:
                        logger.warning(
                            f"Failed to load {file_path}, frame will be black."
                        )
                        continue

                    # Normalize shape (same logic as above)
                    img = np.asarray(img, dtype=np.float32)
                    if len(img.shape) == 2:
                        img = np.stack([img, img, img], axis=-1)
                    elif len(img.shape) == 3:
                        if img.shape[2] == 4:
                            if alpha_img is None:
                                alpha_img = img[:, :, 3]
                            img = img[:, :, :3]
                        elif img.shape[2] == 1:
                            img = np.repeat(img, 3, axis=2)

                    if alpha_img is None:
                        alpha_img = np.ones(
                            (img.shape[0], img.shape[1]), dtype=np.float32
                        )

                    if depth_img is None:
                        depth_img = np.zeros(
                            (img.shape[0], img.shape[1]), dtype=np.float32
                        )

                # Check dimensions
                if img.shape[:2] != (h, w):
                    logger.warning(
                        f"Skipping frame {i}: Dimension mismatch {img.shape[:2]} vs {(h, w)}"
                    )
                    continue

                # Handle channel mismatch
                if img.shape[2] != c:
                    if img.shape[2] < c:
                        # Pad
                        padding = np.ones((h, w, c - img.shape[2]), dtype=np.float32)
                        img = np.concatenate([img, padding], axis=-1)
                    else:
                        # Crop
                        img = img[:, :, :c]

                # Apply exposure
                if exposure_adjust != 0:
                    img = img * (2.0**exposure_adjust)

                # Write to tensor (avoiding extra copy if possible)
                batch_tensor[i] = torch.from_numpy(img)
                batch_alpha[i] = torch.from_numpy(alpha_img)
                batch_depth[i] = torch.from_numpy(depth_img)

                loaded_files.append(os.path.basename(file_path))
                loaded_count += 1

                if (i + 1) % 10 == 0:
                    logger.debug(f"Loaded {i + 1}/{num_frames} frames")

            except Exception as e:
                logger.error(f"Error loading frame {file_path}: {e}")

        if loaded_count == 0:
            raise RuntimeError("Failed to load any valid frames.")

        # Build metadata
        metadata = {
            "folder": folder_path,
            "pattern": file_pattern,
            "frame_count": num_frames,
            "files": loaded_files[:10] + (["..."] if len(loaded_files) > 10 else []),
            "width": w,
            "height": h,
            "channels": c,
            "total_found": len(files),
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

        metadata_str = json.dumps(metadata, indent=2)
        logger.info(f"Done. Loaded {loaded_count}/{num_frames} frames.")

        # Convert to tensor: (B, H, W, C)
        # It's already a tensor, just return it
        # Convert to tensor: (B, H, W, C)
        # It's already a tensor, just return it
        return (batch_tensor, batch_alpha, batch_depth, metadata_str, num_frames)


class SaveImage16bit:
    """
    Save images as 16-bit PNG or TIFF for wider software compatibility while preserving extended range.
    """

    FORMATS = ["PNG", "TIFF"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "output/16bit_"}),
                "format": (cls.FORMATS, {"default": "PNG"}),
                "tonemap": (
                    ["Clip (Standard)", "Compress (LogC4)", "Compress (Reinhard)"],
                    {"default": "Clip (Standard)"},
                ),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = "Save images as 16-bit PNG or TIFF for wider software compatibility while preserving extended range."

    def save(
        self,
        images: torch.Tensor,
        filename_prefix: str = "output/16bit_",
        format: str = "PNG",
        tonemap: str = "Clip (Standard)",
    ) -> Dict:

        import cv2
        import folder_paths

        # Get ComfyUI output directory (use absolute path for Windows compatibility)
        output_dir = Path(folder_paths.get_output_directory())

        # Parse the filename prefix
        clean_prefix = filename_prefix.replace("\\", "/")
        if clean_prefix.startswith("output/"):
            clean_prefix = clean_prefix[7:]

        # Split into directory and base name
        if "/" in clean_prefix:
            subdir, base_name = clean_prefix.rsplit("/", 1)
            full_dir = output_dir / subdir
        else:
            full_dir = output_dir
            base_name = clean_prefix

        # Create directory
        full_dir.mkdir(parents=True, exist_ok=True)

        results = []
        ext = ".png" if format == "PNG" else ".tiff"

        for i, img in enumerate(images):
            np_img = img.cpu().numpy()

            # Apply Tone mapping / Compression
            if tonemap == "Compress (LogC4)":
                # HDR preservation: Linear -> LogC4 (0-1)
                # Ensure we have the function
                if "linear_to_logc4" in globals():
                    np_img = linear_to_logc4(np_img)
                else:
                    logger.warning("LogC4 requested but not found. Clipping.")
                    np_img = np.clip(np_img, 0, 1)
            elif tonemap == "Compress (Reinhard)":
                # Simple SDR compression
                np_img = np_img / (1.0 + np_img)
            else:
                # Clip (Standard) - hard clamp for SDR 16-bit
                np_img = np.clip(np_img, 0, 1)

            # Final safety clip to ensure standard [0,1] range for conversion
            np_img = np.nan_to_num(np_img)
            np_img = np.clip(np_img, 0, 1)

            # Convert to 16-bit integer
            np_img = (np_img * 65535).astype(np.uint16)

            # Convert RGB to BGR for OpenCV
            if np_img.shape[-1] == 3:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            elif np_img.shape[-1] == 4:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)

            filename = f"{base_name}{i:05d}{ext}"
            filepath = full_dir / filename

            cv2.imwrite(str(filepath), np_img)
            results.append({"filename": str(filepath), "type": "output"})
            logger.info(f"Saved 16-bit: {filepath}")

        return {"ui": {"images": results}}


# ═══════════════════════════════════════════════════════════════════════════════
#                           EXR CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_BATCH_SIZE = 1000
MAX_IMAGE_DIMENSION = 32768

EXR_COMPRESSION = {
    "None": 0,
    "RLE": 1,
    "ZIPS": 2,
    "ZIP": 3,
    "PIZ": 4,
    "PXR24": 5,
    "B44": 6,
    "B44A": 7,
    "DWAA": 8,
    "DWAB": 9,
}
EXR_PIXEL_TYPE = {"UINT": 0, "HALF": 1, "FLOAT": 2}
EXR_LINE_ORDER = {"INCREASING_Y": 0, "DECREASING_Y": 1, "RANDOM_Y": 2}


# Cached inverse matrices
def _safe_inverse(matrix: np.ndarray, name: str) -> Optional[np.ndarray]:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        logger.warning(f"Failed to invert {name} matrix: {e}")
        return None


try:
    SRGB_TO_ACESCG = _safe_inverse(ACESCG_TO_SRGB, "ACESCG_TO_SRGB")
    ACESCG_TO_AWG3 = _safe_inverse(AWG3_TO_ACESCG, "AWG3_TO_ACESCG")
    ACESCG_TO_AWG4 = _safe_inverse(AWG4_TO_ACESCG, "AWG4_TO_ACESCG")
    ACESCG_TO_SGAMUT = _safe_inverse(SGAMUT3_CINE_TO_ACESCG, "SGAMUT3_CINE_TO_ACESCG")
except NameError:
    SRGB_TO_ACESCG = None
    ACESCG_TO_AWG3 = None
    ACESCG_TO_AWG4 = None
    ACESCG_TO_SGAMUT = None


def float32_to_float16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float16)


def float16_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float16).tobytes()


def float32_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


# ═══════════════════════════════════════════════════════════════════════════════
#                           EXR WRITERS
# ═══════════════════════════════════════════════════════════════════════════════


def write_exr_cv2(
    filepath: str,
    image: np.ndarray,
    bit_depth: str = "float32",
    compression: str = "ZIP",
) -> bool:
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not available for EXR writing")
        return False

    dir_path = os.path.dirname(os.path.abspath(filepath))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    img = image.astype(np.float32)
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    if "16" in str(bit_depth) or "half" in str(bit_depth).lower():
        exr_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
    else:
        exr_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]

    try:
        success = cv2.imwrite(filepath, img, exr_flags)
        if not success:
            logger.warning(f"OpenCV failed to write: {filepath}")
        return success
    except Exception as e:
        logger.error(f"OpenCV EXR write error: {e}")
        return False


def check_openexr_available() -> bool:
    try:
        import OpenEXR  # noqa: F401

        return True
    except ImportError:
        return False


def write_exr_openexr(
    filepath: str,
    channels: Dict[str, np.ndarray],
    compression: str = "ZIP",
    pixel_type: str = "HALF",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    import OpenEXR
    import Imath

    if not channels:
        raise ValueError("No channels provided")

    first_channel = list(channels.values())[0]
    height, width = int(first_channel.shape[0]), int(first_channel.shape[1])

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    header = OpenEXR.Header(width, height)
    header["dataWindow"] = Imath.Box2i(
        Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1)
    )
    header["displayWindow"] = Imath.Box2i(
        Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1)
    )
    header["pixelAspectRatio"] = 1.0

    comp_map = {
        "None": Imath.Compression(Imath.Compression.NO_COMPRESSION),
        "RLE": Imath.Compression(Imath.Compression.RLE_COMPRESSION),
        "ZIPS": Imath.Compression(Imath.Compression.ZIPS_COMPRESSION),
        "ZIP": Imath.Compression(Imath.Compression.ZIP_COMPRESSION),
        "PIZ": Imath.Compression(Imath.Compression.PIZ_COMPRESSION),
        "PXR24": Imath.Compression(Imath.Compression.PXR24_COMPRESSION),
        "B44": Imath.Compression(Imath.Compression.B44_COMPRESSION),
        "B44A": Imath.Compression(Imath.Compression.B44A_COMPRESSION),
        "DWAA": Imath.Compression(Imath.Compression.DWAA_COMPRESSION),
        "DWAB": Imath.Compression(Imath.Compression.DWAB_COMPRESSION),
    }
    header["compression"] = comp_map.get(
        compression, Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    )

    ptype = (
        Imath.PixelType(Imath.PixelType.HALF)
        if pixel_type == "HALF"
        else Imath.PixelType(Imath.PixelType.FLOAT)
    )

    channel_defs = {}
    for name in sorted(channels.keys()):
        channel_defs[name] = Imath.Channel(ptype)
    header["channels"] = channel_defs

    if metadata:
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)):
                header[k] = v

    out = OpenEXR.OutputFile(filepath, header)

    channel_data = {}
    for name, data in channels.items():
        if pixel_type == "HALF":
            channel_data[name] = data.astype(np.float16).tobytes()
        else:
            # Try passing bytes explicitly to avoid binding confusion
            channel_data[name] = data.astype(np.float32).tobytes()

    try:
        out.writePixels(channel_data)
        out.close()
    except Exception as e:
        # Detailed error logging
        import traceback

        logger.error(f"OpenEXR writePixels failed: {e}")
        logger.debug(traceback.format_exc())
        out.close()
        raise e


def write_exr_imageio(
    filepath: str, image: np.ndarray, pixel_type: str = "HALF"
) -> bool:
    try:
        import imageio.v3 as iio

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        img = image.astype(np.float32)
        iio.imwrite(filepath, img, plugin="pyav" if filepath.endswith(".exr") else None)
        return True
    except Exception as e:
        logger.warning(f"imageio EXR write failed: {e}")
        return False


class SimpleEXRWriter:
    MAGIC = 20000630
    VERSION = 2

    def write(
        self,
        filepath: str,
        channels: Dict[str, np.ndarray],
        compression: str = "ZIP",
        pixel_type: str = "HALF",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not channels:
            raise ValueError("No channels provided")

        first_channel = list(channels.values())[0]
        height, width = first_channel.shape[:2]
        channel_list = sorted(channels.keys())

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(struct.pack("<I", self.MAGIC))
            f.write(struct.pack("<I", self.VERSION))
            self._write_header(
                f, width, height, channel_list, compression, pixel_type, metadata
            )
            self._write_scanlines(
                f, channels, channel_list, width, height, compression, pixel_type
            )

    def _write_header(
        self,
        f,
        width: int,
        height: int,
        channel_list: List[str],
        compression: str,
        pixel_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._write_attribute(
            f, "channels", "chlist", self._encode_channel_list(channel_list, pixel_type)
        )
        comp_value = EXR_COMPRESSION.get(compression, 3)
        if comp_value == 3:
            comp_value = 2  # ZIPS fallback
        self._write_attribute(
            f, "compression", "compression", struct.pack("<B", comp_value)
        )
        self._write_attribute(
            f, "dataWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1)
        )
        self._write_attribute(
            f,
            "displayWindow",
            "box2i",
            struct.pack("<iiii", 0, 0, width - 1, height - 1),
        )
        self._write_attribute(
            f,
            "lineOrder",
            "lineOrder",
            struct.pack("<B", EXR_LINE_ORDER["INCREASING_Y"]),
        )
        self._write_attribute(f, "pixelAspectRatio", "float", struct.pack("<f", 1.0))
        self._write_attribute(
            f, "screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)
        )
        self._write_attribute(f, "screenWindowWidth", "float", struct.pack("<f", 1.0))

        # Standard EXR attributes that we handle explicitly
        reserved_attrs = {
            "channels",
            "compression",
            "dataWindow",
            "displayWindow",
            "lineOrder",
            "pixelAspectRatio",
            "screenWindowCenter",
            "screenWindowWidth",
            "tiles",
            "version",
            "chunkCount",
        }

        if metadata:
            for key, value in metadata.items():
                if key in reserved_attrs:
                    continue

                if isinstance(value, str):
                    self._write_attribute(f, key, "string", self._encode_string(value))
                elif isinstance(value, float):
                    self._write_attribute(f, key, "float", struct.pack("<f", value))
                elif isinstance(value, int):
                    self._write_attribute(f, key, "int", struct.pack("<i", value))
        f.write(b"\x00")

    def _write_attribute(self, f, name: str, attr_type: str, data: bytes) -> None:
        f.write(name.encode("ascii") + b"\x00")
        f.write(attr_type.encode("ascii") + b"\x00")
        f.write(struct.pack("<I", len(data)))
        f.write(data)

    def _encode_channel_list(self, channel_list: List[str], pixel_type: str) -> bytes:
        data = b""
        ptype = EXR_PIXEL_TYPE.get(pixel_type, 1)
        for name in channel_list:
            data += name.encode("ascii") + b"\x00"
            data += struct.pack("<I", ptype)
            data += struct.pack("<B", 0)
            data += b"\x00\x00\x00"
            data += struct.pack("<i", 1)
            data += struct.pack("<i", 1)
        data += b"\x00"
        return data

    def _encode_string(self, s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded

    def _write_scanlines(
        self,
        f,
        channels: Dict[str, np.ndarray],
        channel_list: List[str],
        width: int,
        height: int,
        compression: str,
        pixel_type: str,
    ) -> None:
        offset_table_pos = f.tell()
        offsets = []
        for _ in range(height):
            f.write(struct.pack("<Q", 0))
        comp_type = EXR_COMPRESSION.get(compression, 3)

        for y in range(height):
            offsets.append(f.tell())
            f.write(struct.pack("<i", y))
            scanline_data = b""
            for channel_name in channel_list:
                channel = channels[channel_name]
                row = channel[y, :]
                row_bytes = (
                    float16_to_bytes(row)
                    if pixel_type == "HALF"
                    else float32_to_bytes(row)
                )
                scanline_data += row_bytes

            if comp_type in [2, 3]:
                # Always compress, even if size increases, as format expects zlib stream
                # OpenEXR typically uses zlib level 4
                scanline_data = zlib.compress(scanline_data, level=4)

            f.write(struct.pack("<I", len(scanline_data)))
            f.write(scanline_data)

        current_pos = f.tell()
        f.seek(offset_table_pos)
        for offset in offsets:
            f.write(struct.pack("<Q", offset))
        f.seek(current_pos)


def rgb_to_rgbe(rgb: np.ndarray) -> np.ndarray:
    max_val = np.max(rgb, axis=-1, keepdims=True)
    max_val = np.maximum(max_val, 1e-32)
    exponent = np.floor(np.log2(max_val)) + 128
    exponent = np.clip(exponent, 0, 255).astype(np.uint8)
    divisor = 2.0 ** (exponent.astype(np.float32) - 128)
    mantissa = (rgb / divisor) * 256.0
    mantissa = np.clip(mantissa, 0, 255).astype(np.uint8)
    rgbe = np.concatenate([mantissa, exponent], axis=-1)
    return rgbe


def write_hdr_rgbe(filepath: str, image: np.ndarray) -> bool:
    dir_path = os.path.dirname(os.path.abspath(filepath))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    height, width = image.shape[:2]
    try:
        with open(filepath, "wb") as f:
            f.write(b"#?RADIANCE\n# Created by Radiance\nFORMAT=32-bit_rle_rgbe\n\n")
            f.write(f"-Y {height} +X {width}\n".encode())
            for y in range(height):
                rgbe_line = rgb_to_rgbe(image[y])
                f.write(rgbe_line.tobytes())
        return True
    except Exception as e:
        logger.error(f"Failed to write HDR: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#                           COMFYUI NODE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceSaveEXR:
    """
    Save images as EXR files with full HDR and metadata support.
    """

    def __init__(self):
        # Default to ComfyUI standard output directory
        import folder_paths
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Radiance"}),
                "format": (["EXR", "HDR"], {"default": "EXR"}),
                "bit_depth": (["16-bit Half Float", "32-bit Float"],),
                "compression": (
                    [
                        "ZIP",
                        "ZIPS",
                        "PIZ",
                        "RLE",
                        "None",
                        "PXR24",
                        "B44",
                        "B44A",
                        "DWAA",
                        "DWAB",
                    ],
                ),
            },
            "optional": {
                "input_color_space": (
                    ["Linear (sRGB)", "sRGB (Gamma)", "ACEScg", "Raw"],
                ),
                "output_color_space": (
                    [
                        "Linear (sRGB)",
                        "Linear (ACEScg)",
                        "sRGB (Display)",
                        "ARRI LogC3 (AWG3)",
                        "ARRI LogC4 (AWG4)",
                        "Sony S-Log3 (S-Gamut3)",
                        "Same as Input",
                    ],
                ),
                "alpha_mode": (["None", "From Image", "Solid White", "Solid Black"],),
                "premultiply_alpha": ("BOOLEAN", {"default": False}),
                "output_path": ("STRING", {"default": "", "tooltip": "Absolute or relative output path. Example: D:\\saved or my_subfolder"}),

                "start_frame": ("INT", {"default": 1, "min": 0, "max": 999999}),
                "frame_padding": ("INT", {"default": 4, "min": 1, "max": 8}),
                "add_metadata": ("BOOLEAN", {"default": True}),
                "custom_metadata": ("STRING", {"default": "", "multiline": True}),
                "channel_format": (["RGB", "BGR", "RGBA", "BGRA", "ACEScg"],),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("file_paths", "folder_path", "frame_count")
    FUNCTION = "save_exr"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = "Save images as EXR files with full HDR and metadata support."

    def save_exr(
        self,
        images: torch.Tensor,
        filename_prefix: str,
        format: str = "EXR",
        bit_depth: str = "32-bit Float",
        compression: str = "ZIP",
        input_color_space: str = "sRGB",
        output_color_space: str = "Linear",
        alpha_mode: str = "None",
        premultiply_alpha: bool = False,
        output_path: str = "",
        start_frame: int = 1,
        frame_padding: int = 4,
        add_metadata: bool = True,
        custom_metadata: str = "",
        channel_format: str = "RGB",
        prompt: Optional[Any] = None,
        extra_pnginfo: Optional[Any] = None,
    ) -> Dict[str, Any]:

        try:
            if output_path and os.path.isabs(output_path):
                output_dir = output_path
                os.makedirs(output_dir, exist_ok=True)
            else:
                # Sanitize: Remove 'output/' prefix if it was added by automation scripts
                clean_path = output_path
                if clean_path:
                    clean_path = clean_path.replace("\\", "/")
                    if clean_path.startswith("output/"):
                        clean_path = clean_path[7:]
                output_dir = get_safe_output_dir(self.output_dir, clean_path)
        except ValueError as e:
            logger.error(f"Invalid output path: {e}")
            raise ValueError(f"Security error: {e}")

        if not isinstance(images, torch.Tensor):
            raise ValueError(f"images must be a torch.Tensor, got {type(images)}")

        if images.dim() not in (3, 4):
            raise ValueError(f"images must be 3D or 4D tensor, got {images.dim()}D")

        pixel_type = "HALF" if "16" in bit_depth else "FLOAT"

        metadata: Dict[str, Any] = {}
        if add_metadata:
            metadata["software"] = "Radiance - ComfyUI"
            metadata["created"] = datetime.datetime.now().isoformat()
            metadata["compression"] = compression
            metadata["bitDepth"] = bit_depth
            metadata["colorSpace"] = output_color_space

        if custom_metadata:
            for line in custom_metadata.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    metadata[key.strip()] = value.strip()

        batch_size = images.shape[0] if images.dim() == 4 else 1

        if batch_size > MAX_BATCH_SIZE:
            logger.warning(f"Batch size {batch_size} exceeds limit {MAX_BATCH_SIZE}")
            batch_size = MAX_BATCH_SIZE

        logger.info(f"Radiance: Saving {batch_size} image(s) to: {output_dir}")

        saved_paths: List[str] = []
        use_openexr = check_openexr_available()
        writer = SimpleEXRWriter()

        for i in range(batch_size):
            try:
                if images.dim() == 4:
                    img = images[i].cpu().numpy().astype(np.float32)
                else:
                    img = images.cpu().numpy().astype(np.float32)

                img_out = self._convert_color_space(
                    img, input_color_space, output_color_space
                )
                alpha = self._get_alpha(img, alpha_mode)

                if premultiply_alpha and alpha is not None:
                    for c in range(3):
                        img_out[..., c] *= alpha

                channels = self._build_channels(
                    img_out, alpha, channel_format, alpha_mode
                )

                frame_num = start_frame + i
                frame_str = str(frame_num).zfill(frame_padding)
                ext = ".hdr" if format == "HDR" else ".exr"
                filename = f"{filename_prefix}_{frame_str}{ext}"
                
                # Support absolute paths in prefix for VFX pipelines
                if os.path.isabs(filename):
                    filepath = os.path.normpath(filename)
                    # Ensure directory exists for direct absolute paths
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                else:
                    filepath = safe_join(output_dir, filename)

                logger.info(f"Radiance: Writing EXR [{i+1}/{batch_size}] -> {filepath}")

                if format == "HDR":
                    if write_hdr_rgbe(filepath, img_out):
                        saved_paths.append(filepath)
                    continue

                frame_metadata = {**metadata, "frame": frame_num}

                success = self._write_exr_with_fallback(
                    filepath,
                    channels,
                    img_out,
                    alpha,
                    alpha_mode,
                    compression,
                    pixel_type,
                    bit_depth,
                    frame_metadata,
                    use_openexr,
                    writer,
                )

                if success:
                    saved_paths.append(filepath)
                else:
                    logger.error(f"Radiance Error: All EXR writers failed for {filepath}")

            except Exception as e:
                logger.error(f"Radiance Error: Processing frame {i}: {e}")
                continue

        # Prepare UI response (filenames for preview)
        # Browsers cannot display EXR/HDR files, so we skip the 'file_paths' list
        # if the format is not compatible OR if it's an absolute path (which breaks local server URL mapping).
        ui_files = []
        is_preview_compatible = format in ["PNG", "JPG", "JPEG"]
        
        # In ComfyUI, returned 'ui'['images'] or 'ui'['file_paths'] triggers preview.
        # For RadianceSaveEXR, we strictly avoid this for non-visual formats to prevent the "empty box" error.
        
        # Emergency Fix: Ensure name 'paths_str' exists just in case of weird scoping/caching
        paths_str = ",".join(saved_paths)
        logger.info(f"!!! RADIANCE DEBUG: save_exr executing line 1596 - saved_paths count: {len(saved_paths)} !!!")

        return {
            "ui": {"file_paths": []}, 
            "result": (paths_str, str(output_dir), len(saved_paths)),
        }

    def _convert_color_space(
        self, img: np.ndarray, input_cs: str, output_cs: str
    ) -> np.ndarray:
        img_rgb = img[..., :3].copy()
        if input_cs == "Raw" or output_cs == "Same as Input":
            return img_rgb

        if input_cs == "sRGB (Gamma)":
            img_linear = srgb_to_linear(img_rgb)
            img_aces = (
                apply_matrix_transform(img_linear, SRGB_TO_ACESCG)
                if SRGB_TO_ACESCG is not None
                else img_linear
            )
        elif input_cs == "ACEScg":
            img_aces = img_rgb
        else:
            img_aces = (
                apply_matrix_transform(img_rgb, SRGB_TO_ACESCG)
                if SRGB_TO_ACESCG is not None
                else img_rgb
            )

        if output_cs == "Linear (ACEScg)":
            return img_aces
        elif output_cs == "Linear (sRGB)":
            return apply_matrix_transform(img_aces, ACESCG_TO_SRGB)
        elif output_cs == "sRGB (Display)":
            img_lin = apply_matrix_transform(img_aces, ACESCG_TO_SRGB)
            return linear_to_srgb(img_lin)
        elif output_cs == "ARRI LogC4 (AWG4)" and ACESCG_TO_AWG4 is not None:
            img_awg4 = apply_matrix_transform(img_aces, ACESCG_TO_AWG4)
            return linear_to_logc4(img_awg4)
        elif output_cs == "ARRI LogC3 (AWG3)" and ACESCG_TO_AWG3 is not None:
            img_awg3 = apply_matrix_transform(img_aces, ACESCG_TO_AWG3)
            return linear_to_logc3(img_awg3)
        elif output_cs == "Sony S-Log3 (S-Gamut3)" and ACESCG_TO_SGAMUT is not None:
            img_sgamut = apply_matrix_transform(img_aces, ACESCG_TO_SGAMUT)
            return linear_to_slog3(img_sgamut)
        else:
            return img_aces

    def _get_alpha(self, img: np.ndarray, alpha_mode: str) -> Optional[np.ndarray]:
        if alpha_mode == "From Image" and img.shape[-1] == 4:
            return img[..., 3]
        elif alpha_mode == "Solid White":
            return np.ones(img.shape[:2], dtype=np.float32)
        elif alpha_mode == "Solid Black":
            return np.zeros(img.shape[:2], dtype=np.float32)
        return None

    def _build_channels(
        self,
        img_rgb: np.ndarray,
        alpha: Optional[np.ndarray],
        channel_format: str,
        alpha_mode: str,
    ) -> Dict[str, np.ndarray]:
        channels = {}
        if channel_format in ["RGB", "RGBA", "ACEScg"]:
            channels["R"], channels["G"], channels["B"] = (
                img_rgb[..., 0],
                img_rgb[..., 1],
                img_rgb[..., 2],
            )
        elif channel_format in ["BGR", "BGRA"]:
            channels["B"], channels["G"], channels["R"] = (
                img_rgb[..., 0],
                img_rgb[..., 1],
                img_rgb[..., 2],
            )

        if alpha_mode != "None" and alpha is not None:
            channels["A"] = alpha
        return channels

    def _write_exr_with_fallback(
        self,
        filepath,
        channels,
        img_rgb,
        alpha,
        alpha_mode,
        compression,
        pixel_type,
        bit_depth,
        metadata,
        use_openexr,
        writer,
    ) -> bool:
        if use_openexr:
            try:
                write_exr_openexr(filepath, channels, compression, pixel_type, metadata)
                return True
            except Exception as e:
                logger.warning(f"OpenEXR failed: {e}, trying OpenCV")

        cv_img = img_rgb.copy()
        if alpha is not None and alpha_mode != "None":
            cv_img = np.concatenate([cv_img, alpha[..., np.newaxis]], axis=-1)

        if write_exr_cv2(filepath, cv_img, bit_depth, compression):
            return True

        try:
            writer.write(filepath, channels, compression, pixel_type, metadata)
            return True
        except Exception as e:
            logger.warning(f"SimpleEXRWriter failed: {e}, trying imageio")

        return write_exr_imageio(filepath, cv_img, pixel_type)
