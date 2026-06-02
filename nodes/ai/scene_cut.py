"""
nodes_scene_cut.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Radiance v3 — Video-First Pipeline · Scene-Cut Detection

Nodes
─────
  RadianceSceneCutDetect    Detect shot boundaries in an IMAGE batch
  RadianceSceneCutSplit     Split a batch into per-shot sub-batches
  RadianceShotGradeRouter   Apply per-shot grade parameters via index lookup
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
import torch

log = logging.getLogger("radiance.scene_cut")


# ─────────────────────────────────────────────────────────────────────────────
# Core detection algorithm
# ─────────────────────────────────────────────────────────────────────────────

def _luminance(frames: np.ndarray) -> np.ndarray:
    """(B,H,W,3) → (B,) perceptual luminance per frame."""
    return (0.2126 * frames[:, :, :, 0] +
            0.7152 * frames[:, :, :, 1] +
            0.0722 * frames[:, :, :, 2]).mean(axis=(1, 2))


def _histogram_diff(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Histogram intersection distance between two frames (0=identical, 2=opposite)."""
    score = 0.0
    for c in range(3):
        ha, _ = np.histogram(a[:, :, c].ravel(), bins=bins, range=(0.0, 1.0))
        hb, _ = np.histogram(b[:, :, c].ravel(), bins=bins, range=(0.0, 1.0))
        ha = ha.astype(np.float32) / (ha.sum() + 1e-8)
        hb = hb.astype(np.float32) / (hb.sum() + 1e-8)
        score += float(np.abs(ha - hb).sum())
    return score / 3.0


def _edge_diff(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean absolute difference of Sobel edge maps between two frames.
    Catches hard cuts that don't change average colour (e.g. same palette, different content).
    """
    def sobel_mag(img):
        # Luma only for speed
        gray = (0.2126 * img[:, :, 0] +
                0.7152 * img[:, :, 1] +
                0.0722 * img[:, :, 2])
        # Simple finite-difference approximation
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        return np.sqrt(gx ** 2 + gy ** 2)

    mag_a = sobel_mag(a)
    mag_b = sobel_mag(b)
    return float(np.abs(mag_a - mag_b).mean())


def detect_cuts(
    frames: np.ndarray,
    threshold: float = 0.30,
    min_shot_frames: int = 8,
    method: str = "histogram",
) -> tuple[list[int], np.ndarray]:
    """
    Detect hard cuts in a frame sequence.

    Parameters
    ──────────
    frames          (B, H, W, 3) float32 [0, 1]
    threshold       inter-frame distance above which a cut is declared
    min_shot_frames minimum frames between two cut points
    method          "histogram" | "edge" | "combined"

    Returns
    ───────
    cut_frames      list of frame indices where a new shot begins (always includes 0)
    scores          (B-1,) per-frame inter-frame distance array
    """
    B = frames.shape[0]
    if B < 2:
        return [0], np.zeros(max(B - 1, 1), dtype=np.float32)

    scores = np.zeros(B - 1, dtype=np.float32)
    for i in range(B - 1):
        a, b = frames[i], frames[i + 1]
        if method == "histogram":
            scores[i] = _histogram_diff(a, b)
        elif method == "edge":
            scores[i] = _edge_diff(a, b)
        else:  # combined
            scores[i] = 0.6 * _histogram_diff(a, b) + 0.4 * _edge_diff(a, b)

    # Normalise scores to [0, 1]
    max_s = scores.max()
    if max_s > 1e-6:
        scores_norm = scores / max_s
    else:
        scores_norm = scores.copy()

    # Find cuts above threshold, enforcing min_shot_frames gap
    raw_cuts = [i + 1 for i in range(B - 1) if scores_norm[i] >= threshold]
    cuts = [0]
    for c in raw_cuts:
        if c - cuts[-1] >= min_shot_frames:
            cuts.append(c)

    return cuts, scores


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — RadianceSceneCutDetect
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSceneCutDetect:
    """
    Detect hard shot cuts in a video sequence batch.

    Outputs a JSON list of cut frame indices and a diagnostic plot tensor.
    The cut_data STRING connects directly to RadianceSceneCutSplit.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Automatically detect scene cuts in a video using visual change metrics."
    FUNCTION     = "detect"
    RETURN_TYPES = ("STRING", "INT",   "IMAGE")
    RETURN_NAMES = ("cut_data", "shot_count", "score_plot")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Full video sequence as IMAGE batch.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Cut sensitivity. Lower = detect more cuts. "
                        "Typical range 0.25–0.45 for most footage."
                    ),
                }),
                "min_shot_frames": ("INT", {
                    "default": 12, "min": 1, "max": 500,
                    "tooltip": "Minimum frames between detected cuts.",
                }),
                "method": (["histogram", "edge", "combined"], {
                    "default": "combined",
                    "tooltip": (
                        "histogram: colour distribution diff (fast). "
                        "edge: Sobel edge map diff (catches content cuts). "
                        "combined: weighted blend of both (recommended)."
                    ),
                }),
            },
        }

    # ── score plot ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_plot(scores: np.ndarray, cuts: list[int], threshold: float) -> torch.Tensor:
        """Render a 512×128 plot of inter-frame scores as an IMAGE tensor."""
        PW, PH = 512, 128
        plot = np.ones((PH, PW, 3), dtype=np.float32) * 0.12  # dark background

        B = len(scores)
        if B == 0:
            return torch.from_numpy(plot[None])

        # Score bars
        for i, s in enumerate(scores):
            x = int(i / B * PW)
            bar_h = int(min(s, 1.0) * (PH - 16))
            col = [0.2, 0.6, 1.0] if s < threshold else [1.0, 0.3, 0.2]
            plot[PH - bar_h - 1: PH - 1, x: x + max(1, PW // B), :] = col

        # Threshold line
        ty = PH - int(threshold * (PH - 16)) - 1
        plot[ty: ty + 1, :, :] = [1.0, 0.8, 0.0]

        # Cut markers
        for c in cuts:
            if c > 0:
                cx = int((c - 1) / B * PW)
                plot[:, cx: cx + 2, :] = [0.0, 1.0, 0.4]

        return torch.from_numpy(plot[None])  # (1, H, W, 3)

    def detect(
        self,
        images: torch.Tensor,
        threshold: float,
        min_shot_frames: int,
        method: str,
    ):
        frames = images.detach().cpu().float().numpy()  # (B,H,W,3)
        cuts, scores = detect_cuts(frames, threshold, min_shot_frames, method)
        shot_count = len(cuts)

        # Build shot table: [{shot_idx, start_frame, end_frame, length}, ...]
        shots = []
        for i, start in enumerate(cuts):
            end = cuts[i + 1] - 1 if i + 1 < len(cuts) else frames.shape[0] - 1
            shots.append({
                "shot": i,
                "start": start,
                "end":   end,
                "length": end - start + 1,
            })

        cut_data = json.dumps({
            "cuts":       cuts,
            "shots":      shots,
            "shot_count": shot_count,
            "threshold":  threshold,
            "method":     method,
            "total_frames": frames.shape[0],
        })

        log.info("SceneCutDetect: %d shots detected in %d frames",
                 shot_count, frames.shape[0])
        for s in shots:
            log.debug("  Shot %02d: frames %d–%d (%d frames)",
                      s["shot"], s["start"], s["end"], s["length"])

        plot = self._make_plot(scores, cuts, threshold)
        return (cut_data, shot_count, plot)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — RadianceSceneCutSplit
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSceneCutSplit:
    """
    Split an IMAGE batch into per-shot sub-batches using cut_data from
    RadianceSceneCutDetect.

    OUTPUT_IS_LIST mode: each output slot carries one shot's frames.
    Use shot_index to select a specific shot instead.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Split a video at detected scene cut points into discrete segments."
    FUNCTION     = "split"
    RETURN_TYPES = ("IMAGE",  "INT",  "INT",   "INT",   "STRING")
    RETURN_NAMES = ("frames", "shot_index", "start_frame", "end_frame", "shot_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":    ("IMAGE",),
                "cut_data":  ("STRING", {
                    "tooltip": "JSON from RadianceSceneCutDetect.",
                }),
                "shot_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Which shot to extract (0-based). "
                               "Connect shot_count output to know the range.",
                }),
            },
        }

    def split(
        self,
        images: torch.Tensor,
        cut_data: str,
        shot_index: int,
    ):
        data  = json.loads(cut_data)
        shots = data["shots"]

        if not shots:
            return (images, 0, 0, images.shape[0] - 1,
                    json.dumps({"error": "no shots in cut_data"}))

        idx   = max(0, min(shot_index, len(shots) - 1))
        shot  = shots[idx]
        start = shot["start"]
        end   = shot["end"] + 1   # exclusive for slice

        shot_frames = images[start:end]
        info = json.dumps({
            "shot_index":  idx,
            "start_frame": shot["start"],
            "end_frame":   shot["end"],
            "length":      shot["length"],
            "total_shots": len(shots),
        })

        log.info("SceneCutSplit: shot %d — frames %d–%d (%d frames)",
                 idx, shot["start"], shot["end"], shot["length"])
        return (shot_frames, idx, shot["start"], shot["end"], info)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — RadianceShotGradeRouter
# ─────────────────────────────────────────────────────────────────────────────

class RadianceShotGradeRouter:
    """
    Per-shot grade parameter lookup table.

    Store a JSON array of grade entries keyed by shot index.
    Outputs the correct exposure / temperature / saturation floats
    for the current shot_index so downstream grade nodes receive
    shot-specific parameters without manual intervention.

    Grade table format (JSON array, one entry per shot):
    [
      {"exposure": 0.0, "temperature": 0, "saturation": 1.0, "contrast": 1.0},
      {"exposure": 0.3, "temperature": -200, "saturation": 0.9, "contrast": 1.1},
      ...
    ]
    Missing entries fall back to the last defined entry.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Route frames to per-shot grade nodes based on scene cut metadata."
    FUNCTION     = "route"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("exposure", "temperature", "saturation", "contrast", "shot_index")

    # Defaults applied when a key is absent in the table entry
    _DEFAULTS = {
        "exposure":    0.0,
        "temperature": 0.0,
        "saturation":  1.0,
        "contrast":    1.0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        default_table = json.dumps([
            {"exposure": 0.0, "temperature": 0, "saturation": 1.0, "contrast": 1.0},
            {"exposure": 0.3, "temperature": -150, "saturation": 0.9, "contrast": 1.1},
        ], indent=2)
        return {
            "required": {
                "shot_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Current shot index from SceneCutSplit.",
                }),
                "grade_table": ("STRING", {
                    "default": default_table,
                    "multiline": True,
                    "tooltip": "JSON array of per-shot grade parameters.",
                }),
            },
        }

    def route(self, shot_index: int, grade_table: str):
        try:
            table = json.loads(grade_table)
        except json.JSONDecodeError as exc:
            log.warning("ShotGradeRouter: invalid JSON (%s) — using defaults", exc)
            table = []

        # Clamp to last entry if index out of range
        idx   = min(shot_index, len(table) - 1) if table else -1
        entry = table[idx] if idx >= 0 else {}

        exposure    = float(entry.get("exposure",    self._DEFAULTS["exposure"]))
        temperature = float(entry.get("temperature", self._DEFAULTS["temperature"]))
        saturation  = float(entry.get("saturation",  self._DEFAULTS["saturation"]))
        contrast    = float(entry.get("contrast",    self._DEFAULTS["contrast"]))

        log.debug("ShotGradeRouter: shot %d → EV%.2f  temp%.0f  sat%.2f  con%.2f",
                  shot_index, exposure, temperature, saturation, contrast)
        return (exposure, temperature, saturation, contrast, shot_index)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceSceneCutDetect":  RadianceSceneCutDetect,
    "RadianceSceneCutSplit":   RadianceSceneCutSplit,
    "RadianceShotGradeRouter": RadianceShotGradeRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSceneCutDetect":  "◎ Radiance Scene Cut Detect",
    "RadianceSceneCutSplit":   "◎ Radiance Scene Cut Split",
    "RadianceShotGradeRouter": "◎ Radiance Shot Grade Router",
}
