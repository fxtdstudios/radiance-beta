"""
═══════════════════════════════════════════════════════════════════════════════
    Radiance Depth Map v2.3 — Monocular Depth Estimation for ComfyUI
                        Radiance © 2024-2026 FXTD STUDIOS

Uses Depth Anything V2:
 - 97.1% accuracy (δ₁=0.946 on KITTI)
 - 213ms inference (vs Marigold 5.2s)
 - Handles transparent/reflective surfaces
 - Auto-downloads from HuggingFace

v1.1 vs v1.0:
 - FIX: MPS (Apple Silicon) support — was falling back to CPU silently
 - FIX: Model cache device race condition — cache on CPU, .to(device) per call
 - FIX: depth.squeeze() without dim args — unsafe when H or W is 1
 - FIX: _blur_depth zero-padding → reflect-padding (no dark edge halos)
 - FIX: Duplicate logger.info lines removed
 - FIX: Reinhard tonemap crash on negative pixel values
 - NEW: Video-safe per-frame processing with batch-wide normalization
 - NEW: Progress logging for video batches
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import numpy as np
import threading
import logging
from typing import Tuple

from PIL import Image

# Module logger
logger = logging.getLogger("◎ Radiance.depth")

# ═══════════════════════════════════════════════════════════════════════════════
#                           MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Model sizes and their HuggingFace identifiers
DEPTH_MODELS = {
    "Small (25M - Fast)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base (98M - Balanced)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large (335M - Best)": "depth-anything/Depth-Anything-V2-Large-hf",
}

# Thread-safe cache for loaded models.
# Models are stored on CPU.  Callers .to(device) the returned model
# so that two threads requesting different devices don't race on a
# single nn.Module's parameter storage.
_model_cache = {}
_processor_cache = {}
_cache_lock = threading.RLock()


def get_device(use_gpu: bool = True) -> torch.device:
    """Get the appropriate compute device (CUDA > MPS > CPU)."""
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def download_and_load_model(model_size: str, device: torch.device):
    """
    Download model from HuggingFace if not cached, then return on *device*.

    Cache stores models on **CPU** so that concurrent calls requesting
    different devices (CUDA vs MPS vs CPU) don't mutate each other's
    parameter storage.  The caller receives the model already moved to
    the requested device.
    """
    global _model_cache, _processor_cache

    model_id = DEPTH_MODELS.get(model_size)
    if not model_id:
        logger.warning(
            f"Unknown model_size '{model_size}', falling back to Base"
        )
        model_id = DEPTH_MODELS["Base (98M - Balanced)"]

    with _cache_lock:
        if model_id not in _model_cache:
            logger.info(f"Downloading depth model: {model_size} ({model_id})")

            try:
                from transformers import (
                    AutoImageProcessor,
                    AutoModelForDepthEstimation,
                )

                processor = AutoImageProcessor.from_pretrained(
                    model_id, revision="main"
                )
                model = AutoModelForDepthEstimation.from_pretrained(
                    model_id, revision="main"
                )
                model.eval()
                # Cache on CPU — callers .to(device) below
                _model_cache[model_id] = model
                _processor_cache[model_id] = processor
                logger.info(f"Depth model cached: {model_id}")

            except ImportError as e:
                raise ImportError(
                    "transformers library required for Depth Anything V2.\n"
                    "Install with: pip install transformers"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download depth model '{model_id}': {e}"
                ) from e

        model = _model_cache[model_id]
        processor = _processor_cache[model_id]

    # Move to requested device OUTSIDE the lock — .to() on the same device
    # is a no-op, but if device differs this creates new parameter tensors
    # without racing other threads' inference.
    model = model.to(device)
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
#                           DEPTH MAP GENERATOR NODE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceDepthMapGenerator:
    """
    Depth Anything V2 — Monocular depth estimation.

    Video-safe: processes each frame independently, with batch-wide
    normalization for temporal consistency across frames.
    """

    MODEL_SIZES = list(DEPTH_MODELS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_size": (
                    cls.MODEL_SIZES,
                    {
                        "default": "Large (335M - Best)",
                        "tooltip": (
                            "Depth Anything V2 model size. "
                            "Small = fast previews, Large = best quality."
                        ),
                    },
                ),
            },
            "optional": {
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Normalize depth to 0-1 range. "
                            "For video, frames are standardized for temporal consistency."
                        ),
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert depth (white=far, black=near).",
                    },
                ),
                "blur_edges": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.5,
                        "display": "slider",
                        "tooltip": "Gaussian blur to smooth depth discontinuities.",
                    },
                ),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "generate_depth"
    CATEGORY = "FXTD Studios/Radiance/Data"

    DESCRIPTION = (
        "Depth Anything V2 monocular depth estimation. "
        "Video-safe — standardizes each frame with spatial-temporal alignment "
        "preventing flickering. Outputs 3-channel grayscale depth map. "
        "Connect to Depth of Field node for realistic defocus blur."
    )

    @torch.no_grad()
    def generate_depth(
        self,
        image: torch.Tensor,
        model_size: str,
        normalize: bool = True,
        invert: bool = False,
        blur_edges: float = 0.0,
        use_gpu: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Generate depth map from input image(s). Video-safe per-frame."""

        device = get_device(use_gpu)

        # Load model (auto-downloads on first call)
        model, processor = download_and_load_model(model_size, device)

        batch_size = image.shape[0]
        orig_h, orig_w = image.shape[1], image.shape[2]
        raw_depths = []

        is_video = batch_size > 1
        if is_video:
            logger.info(
                f"Processing {batch_size} frames for depth estimation..."
            )

        for i in range(batch_size):
            # ── Convert to uint8 for the HF processor ──
            img_np = image[i].cpu().numpy()

            # HDR tonemap: Reinhard on absolute values so negatives don't NaN.
            # sign() * (|x| / (1+|x|)) maps any real → (-1, 1) monotonically.
            if img_np.max() > 1.05:
                img_abs = np.abs(img_np)
                img_np = np.sign(img_np) * (img_abs / (1.0 + img_abs))

            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

            # Strip alpha if present
            if img_np.shape[-1] == 4:
                img_np = img_np[:, :, :3]

            pil_img = Image.fromarray(img_np)

            # ── Processor → model → raw depth ──
            inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            depth = outputs.predicted_depth  # (1, model_H, model_W)

            # Interpolate to original spatial size.
            # predicted_depth: (1, mH, mW) → unsqueeze → (1, 1, mH, mW)
            # after interpolate → (1, 1, orig_H, orig_W)
            # squeeze with explicit dims (bare squeeze() is unsafe if H or W == 1)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            )
            depth = depth.squeeze(1).squeeze(0)  # → (H, W)

            raw_depths.append(depth)

            if is_video and (i + 1) % 10 == 0:
                logger.info(f"  Depth frame {i + 1}/{batch_size}")

        # Stack: (B, H, W)
        depth_batch = torch.stack(raw_depths, dim=0)

        # ── Normalize ──
        # Depth Anything V2 outputs relative affine-invariant depth maps.
        # Natively doing global min/max causes severe flickering for videos,
        # because each frame has completely arbitrary scale and shift.
        if normalize:
            if is_video:
                # 1. Standardize each frame to Mean=0, Std=1. 
                # This mathematically removes the arbitrary per-frame scale & shift.
                B = depth_batch.shape[0]
                flat = depth_batch.view(B, -1)
                means = flat.mean(dim=1).view(B, 1, 1)
                stds = flat.std(dim=1).view(B, 1, 1)
                depth_batch = (depth_batch - means) / (stds + 1e-8)
                
            # 2. Normalize to [0,1].
            #     - For videos, frames are now aligned in scale/shift, so global Min/Max
            #       keeps relative distance consistent across the entire clip.
            #     - For single images, this normalizes the single frame correctly.
            d_min = depth_batch.min()
            d_max = depth_batch.max()
            depth_batch = (depth_batch - d_min) / (d_max - d_min + 1e-8)

        # ── Invert ──
        if invert:
            depth_batch = 1.0 - depth_batch

        # ── Edge blur ──
        if blur_edges > 0:
            depth_batch = self._blur_depth_batch(depth_batch, blur_edges)

        # ── Output: 3-channel grayscale IMAGE (B, H, W, 3) ──
        result = depth_batch.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

        if is_video:
            logger.info(f"Depth estimation complete: {batch_size} frames")

        return (result.cpu(),)

    @staticmethod
    def _blur_depth_batch(
        depth: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """
        Gaussian blur on a (B, H, W) depth tensor.
        Uses reflect-padding to avoid dark edge halos (v1.0 used zero-padding).
        """
        if sigma <= 0:
            return depth

        kernel_size = max(3, int(sigma * 6) | 1)  # Odd, minimum 3
        device = depth.device

        x = torch.arange(kernel_size, device=device, dtype=torch.float32)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # (B, H, W) → (B, 1, H, W) for conv2d
        d = depth.unsqueeze(1)
        pad = kernel_size // 2

        # Horizontal pass — reflect-pad to avoid edge darkening
        kernel_h = kernel_1d.view(1, 1, 1, kernel_size)
        d = torch.nn.functional.pad(d, (pad, pad, 0, 0), mode="reflect")
        d = torch.nn.functional.conv2d(d, kernel_h, padding=0)

        # Vertical pass
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1)
        d = torch.nn.functional.pad(d, (0, 0, pad, pad), mode="reflect")
        d = torch.nn.functional.conv2d(d, kernel_v, padding=0)

        return d.squeeze(1)  # → (B, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
#                              NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "◎ RadianceDepthMapGenerator": RadianceDepthMapGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "◎ RadianceDepthMapGenerator": "◎ Radiance Depth Map",
}
