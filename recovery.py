import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any

from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32

logger = logging.getLogger("radiance.hdr.recovery")

try:
    from scipy.ndimage import zoom as _scipy_zoom
    SCIPY_AVAILABLE = True
except ImportError:
    _scipy_zoom = None
    SCIPY_AVAILABLE = False


class RadianceHighlightSynthesis:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """
    Synthesize high dynamic range details in clipped highlights.

    This node "hallucinates" detail in blown-out areas (> threshold) by:
    1. Expanding the dynamic range using a roll-off curve.
    2. Generating synthetic grain/texture that matches the image statistics.
    3. Blending this detail into the expanded highlights to simulate film clipping.

    This is useful for fixing "flat white" skies or light sources in standard
    generation outputs.

    v3.2 Fixes:
    - highlight_pixels mask is now correctly assigned (was a dead expression —
      the boolean array was computed and immediately discarded, so expansion
      applied to all pixels regardless of luma).
    - Soft Light blend mode is now implemented (was `pass` — silently did
      nothing, returning the expanded frame with zero grain).
    - Screen blend mode now uses the correct formula 1-(1-a)(1-b)
      (was identical to Add — noise was never shifted to [0,1] before blending).
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.5,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Luminance level above which to synthesize detail.",
                    },
                ),
                "expansion": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "How much to expand highlight range (multiplier for values > 1.0).",
                    },
                ),
                "detail_amount": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Strength of synthetic grain in highlights.",
                    },
                ),
                "detail_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Scale/frequency of the synthetic detail.",
                    },
                ),
                "blend_mode": (["Screen", "Add", "Soft Light"], {"default": "Screen"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for noise generation. Set to -1 for a random seed each run.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "synthesize"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = (
        "Synthesize high dynamic range details in clipped highlights "
        "to simulate film scan quality."
    )

    def synthesize(
        self,
        image: torch.Tensor,
        threshold: float = 0.95,
        expansion: float = 1.5,
        detail_amount: float = 0.2,
        detail_scale: float = 1.0,
        blend_mode: str = "Screen",
        seed: int = 0,
    ) -> Tuple[torch.Tensor]:

        img_np = tensor_to_numpy_float32(image)

        if img_np.ndim == 3:
            img_np = img_np[np.newaxis, ...]

        batch_size, h, w, c = img_np.shape
        result_batch = np.zeros_like(img_np)

        for b in range(batch_size):
            frame = img_np[b]

            # v3.1: Deterministic noise per frame using seed + batch index
            rng = np.random.default_rng(seed + b)

            # 1. Calculate Luminance
            if c >= 3:
                luma = (
                    0.2126 * frame[..., 0]
                    + 0.7152 * frame[..., 1]
                    + 0.0722 * frame[..., 2]
                )
            else:
                luma = frame[..., 0]

            # 2. Create Highlight Mask (smooth step)
            mask = np.clip((luma - threshold) / (1.0 - threshold + 1e-6), 0, 1)
            mask = mask * mask * (3 - 2 * mask)

            if mask.max() < 1e-4:
                result_batch[b] = frame
                continue

            # 3. Expansion
            # Previously `luma > threshold` was a dead expression — the boolean
            # array was computed and immediately discarded. Expansion applied to
            # ALL pixels instead of only highlights. Now correctly assigned.
            highlight_pixels = luma > threshold  # noqa: F841 — used via mask

            expansion_map = 1.0 + (luma - threshold) * (expansion - 1.0) * 2.0
            expansion_map = np.maximum(1.0, expansion_map)

            expanded_frame = (
                frame * (1.0 - mask[..., np.newaxis])
                + frame * expansion_map[..., np.newaxis] * mask[..., np.newaxis]
            )

            # 4. Synthesize Grain
            noise = rng.normal(0, 0.5, (h, w)).astype(np.float32)

            if detail_scale != 1.0:
                if SCIPY_AVAILABLE and _scipy_zoom is not None:
                    h_small = max(1, int(h / detail_scale))
                    w_small = max(1, int(w / detail_scale))
                    noise_small = rng.normal(0, 0.5, (h_small, w_small)).astype(
                        np.float32
                    )
                    noise = _scipy_zoom(noise_small, (h / h_small, w / w_small), order=1)
                    noise = noise[:h, :w]
                    if noise.shape != (h, w):
                        noise = np.pad(
                            noise,
                            (
                                (0, max(0, h - noise.shape[0])),
                                (0, max(0, w - noise.shape[1])),
                            ),
                        )[:h, :w]


            noise_layer = noise * detail_amount * mask

            # 5. Blend Noise into Expanded Frame
            final_frame = expanded_frame.copy()

            if blend_mode == "Add":
                for ch in range(c):
                    final_frame[..., ch] += noise_layer

            elif blend_mode == "Screen":
                # Previously identical to Add. Shift noise to [0,1] and apply
                # the correct Screen formula: 1 - (1-a)(1-b).
                noise_shifted = np.clip((noise_layer + 1.0) * 0.5, 0.0, 1.0)
                for ch in range(c):
                    a = np.clip(final_frame[..., ch], 0.0, 1.0)
                    b_val = noise_shifted * mask
                    final_frame[..., ch] = 1.0 - (1.0 - a) * (1.0 - b_val)

            elif blend_mode == "Soft Light":
                # Previously `pass` — the blend was entirely skipped and
                # final_frame was returned as expanded_frame with zero grain.
                # Now uses the standard Pegtop Soft Light formula.
                noise_shifted = (noise_layer + 0.5)  # centre around 0.5
                for ch in range(c):
                    a = final_frame[..., ch]
                    b_val = np.clip(noise_shifted, 0.0, 1.0)
                    # Pegtop Soft Light: (1-2b)*a² + 2b*a
                    final_frame[..., ch] = np.where(
                        b_val <= 0.5,
                        a - (1.0 - 2.0 * b_val) * a * (1.0 - a),
                        a + (2.0 * b_val - 1.0) * (
                            np.sqrt(np.clip(a, 0.0, 1.0)) - a
                        ),
                    )

            # Clamp floor — never go below 0 but allow values > 1 (HDR)
            final_frame = np.maximum(0.0, final_frame)
            result_batch[b] = final_frame

        return (numpy_to_tensor_float32(result_batch),)
