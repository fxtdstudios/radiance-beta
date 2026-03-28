"""
═══════════════════════════════════════════════════════════════════════════════
                 RADIANCE TEMPORAL NODES v2.2.1
          Temporal Video Processing for ComfyUI Pipelines
                      Radiance © 2024-2026

 Nodes:
   RadianceTemporalSmooth — Per-pixel EMA across batch frames (flicker/noise reduction)
   RadianceFlickerAnalyze — Frame-to-frame luma consistency metric + QC output
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Tuple, List

import torch
import numpy as np

logger = logging.getLogger("◎ Radiance.temporal")


# ─────────────────────────────────────────────────────────────────────────────
#                    RadianceTemporalSmooth
# ─────────────────────────────────────────────────────────────────────────────

class RadianceTemporalSmooth:
    """
    Reduce inter-frame flicker and AI noise in generated video batches using
    per-pixel exponential moving average (EMA).

    EMA formula:  out[t] = α × in[t]  +  (1-α) × out[t-1]

    α = 1.0 → no smoothing (passthrough)
    α = 0.2 → heavy smoothing (good for fine grain / high-frequency flicker)

    Motion-aware mode: α is boosted per-pixel when inter-frame delta exceeds
    `motion_threshold`, preserving sharp motion while smoothing static areas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Batch of frames (B, H, W, C). Must be in temporal order.",
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "EMA weight for current frame. "
                            "1.0 = no smoothing. 0.2 = heaviest smoothing."
                        ),
                    },
                ),
                "motion_aware": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Boost α to 1.0 (no blend) on pixels that change more than "
                            "motion_threshold — preserves sharp motion while smoothing grain."
                        ),
                    },
                ),
                "motion_threshold": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 1.0,
                        "step": 0.005,
                        "tooltip": "Per-pixel delta magnitude above which motion is detected.",
                    },
                ),
            },
            "optional": {
                "warmup_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16,
                        "tooltip": (
                            "Number of initial frames to pass through unchanged "
                            "while the EMA 'warms up'. Prevents faded first frames."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "stats")
    OUTPUT_TOOLTIPS = (
        "Temporally smoothed image batch.",
        "JSON stats: avg delta before/after smoothing per frame.",
    )
    FUNCTION = "smooth"
    CATEGORY = "FXTD Studios/Radiance/Video"
    DESCRIPTION = "Reduce inter-frame flicker in video batches via per-pixel EMA with optional motion-aware masking."

    def smooth(
        self,
        images: torch.Tensor,
        alpha: float = 0.5,
        motion_aware: bool = True,
        motion_threshold: float = 0.05,
        warmup_frames: int = 0,
    ) -> Tuple[torch.Tensor, str]:
        B, H, W, C = images.shape

        if B == 1:
            logger.debug("[TemporalSmooth] Single frame — passthrough")
            return (images, '{"frames": 1, "smoothed": false}')

        out = torch.empty_like(images)
        ema = images[0].clone()
        out[0] = ema

        deltas_before: List[float] = []
        deltas_after:  List[float] = []

        for t in range(1, B):
            curr = images[t]
            delta = curr - ema  # (H, W, C)
            deltas_before.append(delta.abs().mean().item())

            if t < warmup_frames:
                ema = curr.clone()
                out[t] = ema
                continue

            if motion_aware:
                # Per-pixel motion magnitude (averaged over channels)
                motion_mag = delta.abs().mean(dim=-1, keepdim=True)  # (H, W, 1)
                # Where motion is large → use current frame directly (α → 1.0)
                motion_mask = (motion_mag > motion_threshold).float()
                eff_alpha = alpha * (1.0 - motion_mask) + 1.0 * motion_mask
                ema = eff_alpha * curr + (1.0 - eff_alpha) * ema
            else:
                ema = alpha * curr + (1.0 - alpha) * ema

            out[t] = ema
            deltas_after.append((curr - ema).abs().mean().item())

        avg_before = float(np.mean(deltas_before)) if deltas_before else 0.0
        avg_after  = float(np.mean(deltas_after))  if deltas_after  else 0.0
        reduction  = (1.0 - avg_after / (avg_before + 1e-8)) * 100.0

        import json
        stats = json.dumps({
            "frames": B,
            "alpha": alpha,
            "motion_aware": motion_aware,
            "avg_delta_before": round(avg_before, 5),
            "avg_delta_after":  round(avg_after, 5),
            "flicker_reduction_pct": round(reduction, 1),
        })
        logger.info(f"[TemporalSmooth] {B} frames | flicker ↓ {reduction:.1f}%")
        return (out, stats)


# ─────────────────────────────────────────────────────────────────────────────
#                    RadianceFlickerAnalyze
# ─────────────────────────────────────────────────────────────────────────────

class RadianceFlickerAnalyze:
    """
    Measure frame-to-frame luma consistency (flicker index) for QC purposes.
    Outputs the original image batch unchanged + a JSON metrics string.
    Use before RadianceTemporalSmooth to benchmark improvement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel": (
                    ["Luma (Y)", "Red", "Green", "Blue", "All (max)"],
                    {"default": "Luma (Y)"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "flicker_report")
    FUNCTION = "analyze"
    CATEGORY = "FXTD Studios/Radiance/Video"
    DESCRIPTION = "Compute per-frame flicker index (luma delta %) for video QC. Passthrough image, outputs JSON report."

    def analyze(self, images: torch.Tensor, channel: str = "Luma (Y)") -> Tuple[torch.Tensor, str]:
        import json
        B, H, W, C = images.shape

        if B < 2:
            return (images, json.dumps({"error": "Need at least 2 frames"}))

        # Extract channel signal
        if channel == "Luma (Y)" and C >= 3:
            signal = 0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2]
        elif channel == "Red":
            signal = images[..., 0]
        elif channel == "Green":
            signal = images[..., 1] if C > 1 else images[..., 0]
        elif channel == "Blue":
            signal = images[..., 2] if C > 2 else images[..., 0]
        else:  # All (max)
            signal = images.max(dim=-1).values

        # Per-frame mean
        frame_means = signal.mean(dim=[-2, -1]).cpu().numpy()  # (B,)
        deltas = np.abs(np.diff(frame_means))

        overall_mean = float(np.mean(frame_means))
        flicker_index = float(np.std(frame_means) / (overall_mean + 1e-8))
        max_delta = float(np.max(deltas)) if len(deltas) > 0 else 0.0

        frame_stats = [
            {"frame": int(i), "mean": round(float(frame_means[i]), 5)}
            for i in range(B)
        ]

        report = json.dumps({
            "frames": B,
            "channel": channel,
            "flicker_index": round(flicker_index, 5),
            "max_frame_delta": round(max_delta, 5),
            "overall_mean": round(overall_mean, 5),
            "per_frame": frame_stats,
        }, indent=2)

        logger.info(f"[FlickerAnalyze] {B} frames | flicker_index={flicker_index:.4f}")
        return (images, report)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "◎ RadianceTemporalSmooth":  RadianceTemporalSmooth,
    "◎ RadianceFlickerAnalyze":  RadianceFlickerAnalyze,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "◎ RadianceTemporalSmooth": "◎ Radiance Temporal Smooth",
    "◎ RadianceFlickerAnalyze": "◎ Radiance Flicker Analyze",
}
