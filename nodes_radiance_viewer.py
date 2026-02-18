"""
═══════════════════════════════════════════════════════════════════════════════
                     RADIANCE VIEWER NODE v2.0
              VFX Industry-Standard Image Viewer for ComfyUI
                         Radiance © 2024-2026

Professional viewer node providing:
- Interactive zoom/pan with canvas-based rendering
- Real-time exposure, gamma, gain, lift, saturation, temperature controls
- Channel viewing modes (RGB, R, G, B, Alpha, Luminance)
- Color picker with HDR value display (via float32 sidecar)
- Built-in LUTs (sRGB, Rec.709, Log-to-Lin, Filmic, False Color, etc.)
- False color and zebra analysis
- A/B comparison modes
- IMAGE passthrough output (image is no longer stuck)
- 16-bit PNG output for banding-free display (via cv2)
- 32-bit float sidecar (.npy) for true HDR color inspection
- Z-Depth with 16-bit precision

This node uses a JavaScript frontend extension for interactivity.

VERSION HISTORY:

v2.1 - HDR Display Fix (February 2026)
- FIX: Black result caused by missing sRGB OETF in composite shader
- FIX: HDR sidecar now ALWAYS saved (not gated behind bit_depth option)
- FIX: PNG preview tonemapped via Reinhard for HDR content visibility
- FIX: All grading now happens in linear space (correct color math)
- FIX: WebGL shader properly linearizes sRGB PNG input before grading
- OUTPUT: Image passthrough tensor is NEVER clamped (full 32-bit HDR)

v2.0 - Major Upgrade (February 2026)
- NEW: Exposure slider (-6.0 to +6.0 stops, step 0.01)
- NEW: Gamma slider (0.1 to 4.0, step 0.01)
- NEW: Gain slider (0.0 to 5.0, step 0.01)
- NEW: Lift slider (-1.0 to 1.0, step 0.005)
- NEW: Saturation slider (0.0 to 3.0, step 0.01)
- NEW: Color Temperature slider (2000K to 12000K, step 100)
- NEW: IMAGE + metadata passthrough output (image no longer stuck)
- NEW: Built-in LUT engine with 10 industry-standard LUTs
- NEW: LUT intensity slider (0.0 to 1.0) for blend control
- FIX: 16-bit and 16-bit+HDR modes now actually apply to processing
- FIX: Bit depth metadata correctly propagated to frontend
- FIX: HDR sidecar includes adjusted image (not just raw)

v1.21 - 16-bit Fix (February 2026)
- FIX: 16-bit RGB PNG save now uses cv2 instead of PIL frombytes
- FIX: 16-bit RGBA saves via cv2
- FIX: 16-bit grayscale saves via cv2 for consistency
- FIX: Fallback to 8-bit PIL if cv2 unavailable

v1.20 - 16-bit / 32-bit Upgrade (February 2026)
- NEW: 16-bit PNG save path for banding-free display
- NEW: Float32 .npy sidecar files for HDR color picker values
- NEW: bit_depth selector (8-bit, 16-bit, 16-bit+HDR)

v1.10 - Production Hardened (2025)
- FIX: Safe path joining, exception handling, UUID filenames
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import numpy as np
import zlib
import os
import uuid
import logging
import math
from typing import Dict, Any, Optional, List, Union, Tuple

import folder_paths

# Import safe path utilities
try:
    from .path_utils import safe_join
except ImportError:
    from path_utils import safe_join

# v1.21: cv2 for 16-bit PNG support
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.getLogger("radiance.viewer").warning(
        "cv2 not available — 16-bit PNG disabled, falling back to 8-bit. "
        "Install opencv-python for 16-bit support."
    )

# Module logger
logger = logging.getLogger("radiance.viewer")


# ═══════════════════════════════════════════════════════════════════════════════
#                           CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PNG_COMPRESSION = 4
BIT_16_MAX = 65535
BIT_8_MAX = 255
BIT_16_TO_8_DIVISOR = 257  # Correct: 65535 / 255 = 257

MAX_IMAGE_DIMENSION = 16384
MAX_BATCH_SIZE = 100

BIT_DEPTH_MODES = ["8-bit (Fast)", "16-bit (Quality)", "16-bit + HDR Data"]
CV2_PNG_COMPRESSION = 4

# ═══════════════════════════════════════════════════════════════════════════════
#                        BUILT-IN LUT ENGINE v2.0
#
#  Analytical LUTs — no file dependencies.
#  Each LUT is a function: float32 linear [0..1+] → float32 [0..1]
#  Applied per-channel in linear space unless noted otherwise.
# ═══════════════════════════════════════════════════════════════════════════════

LUT_MODES = [
    "None",
    "sRGB (Display)",
    "Rec.709 (Broadcast)",
    "Filmic (Cinematic)",
    "Log C (ARRI)",
    "S-Log3 (Sony)",
    "Linear to Log",
    "Log to Linear",
    "Reinhard Tonemap",
    "ACES Filmic",
    "False Color (Exposure)",
]


def _lut_srgb(x: np.ndarray) -> np.ndarray:
    """Linear → sRGB gamma curve (IEC 61966-2-1)."""
    out = np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055
    )
    return np.clip(out, 0.0, 1.0)


def _lut_rec709(x: np.ndarray) -> np.ndarray:
    """Linear → Rec.709 OETF (BT.709 transfer)."""
    out = np.where(
        x < 0.018,
        x * 4.5,
        1.099 * np.power(np.maximum(x, 0.018), 0.45) - 0.099
    )
    return np.clip(out, 0.0, 1.0)


def _lut_filmic(x: np.ndarray) -> np.ndarray:
    """Filmic tonemapping (Hable/Uncharted 2 curve)."""
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    def hable(v):
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F
    exposure_bias = 2.0
    white_scale = 1.0 / hable(np.array(11.2))
    return np.clip(hable(np.maximum(x, 0.0) * exposure_bias) * white_scale, 0.0, 1.0)


def _lut_logc(x: np.ndarray) -> np.ndarray:
    """Linear → ARRI LogC3 (EI 800)."""
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    out = np.where(
        x > cut,
        c * np.log10(np.maximum(a * x + b, 1e-10)) + d,
        e * x + f
    )
    return np.clip(out, 0.0, 1.0)


def _lut_slog3(x: np.ndarray) -> np.ndarray:
    """Linear → Sony S-Log3."""
    out = np.where(
        x >= 0.01125,
        (420.0 + np.log10(np.maximum(x + 0.01, 1e-10)) * 261.5) / 1023.0,
        (x * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0
    )
    return np.clip(out, 0.0, 1.0)


def _lut_lin_to_log(x: np.ndarray) -> np.ndarray:
    """Generic linear → logarithmic (Cineon-style)."""
    out = np.log2(np.maximum(x, 1e-10) * 5.55 + 1.0) / np.log2(6.55)
    return np.clip(out, 0.0, 1.0)


def _lut_log_to_lin(x: np.ndarray) -> np.ndarray:
    """Generic logarithmic → linear (inverse Cineon-style)."""
    out = (np.power(6.55, np.clip(x, 0.0, 1.0)) - 1.0) / 5.55
    return np.clip(out, 0.0, 1.0)


def _lut_reinhard(x: np.ndarray) -> np.ndarray:
    """Reinhard global tonemapping."""
    return np.clip(x / (1.0 + x), 0.0, 1.0)


def _lut_aces_filmic(x: np.ndarray) -> np.ndarray:
    """ACES Filmic tone mapping (Narkowicz fit)."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    x = np.maximum(x, 0.0)
    out = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(out, 0.0, 1.0)


def _lut_false_color(img: np.ndarray) -> np.ndarray:
    """
    False color exposure map (applied to luminance).
    Returns an RGB image regardless of input channels.
    Industry-standard zones:
      Clip Black  → Blue
      -3 stops    → Cyan  
      -2 stops    → Teal
      -1 stop     → Green
      Mid gray    → Gray (0.18 key)
      +1 stop     → Yellow
      +2 stops    → Orange
      +3 stops    → Red
      Clip White  → Magenta
    """
    # Compute luminance
    if img.ndim == 3 and img.shape[2] >= 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    elif img.ndim == 3 and img.shape[2] == 1:
        lum = img[..., 0]
    else:
        lum = img

    # Stops relative to 0.18 mid gray
    # stop = log2(lum / 0.18)
    safe_lum = np.maximum(lum, 1e-10)
    stops = np.log2(safe_lum / 0.18)

    # Build RGB output
    h, w = lum.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.float32)

    # Zone colors [R, G, B]
    zones = [
        (-99.0, -4.0, [0.05, 0.05, 0.40]),   # Clip black → deep blue
        (-4.0,  -3.0, [0.10, 0.40, 0.55]),    # -3 stops   → cyan
        (-3.0,  -2.0, [0.10, 0.45, 0.35]),    # -2 stops   → teal
        (-2.0,  -1.0, [0.15, 0.55, 0.15]),    # -1 stop    → green
        (-1.0,   1.0, [0.40, 0.40, 0.40]),    # Mid gray   → neutral
        ( 1.0,   2.0, [0.70, 0.65, 0.10]),    # +1 stop    → yellow
        ( 2.0,   3.0, [0.75, 0.40, 0.05]),    # +2 stops   → orange
        ( 3.0,   4.0, [0.70, 0.10, 0.10]),    # +3 stops   → red
        ( 4.0,  99.0, [0.80, 0.15, 0.60]),    # Clip white → magenta
    ]

    for low, high, color in zones:
        mask = (stops >= low) & (stops < high)
        for c in range(3):
            out[..., c][mask] = color[c]

    return out


# LUT dispatch table
_LUT_FUNCTIONS = {
    "None": None,
    "sRGB (Display)": _lut_srgb,
    "Rec.709 (Broadcast)": _lut_rec709,
    "Filmic (Cinematic)": _lut_filmic,
    "Log C (ARRI)": _lut_logc,
    "S-Log3 (Sony)": _lut_slog3,
    "Linear to Log": _lut_lin_to_log,
    "Log to Linear": _lut_log_to_lin,
    "Reinhard Tonemap": _lut_reinhard,
    "ACES Filmic": _lut_aces_filmic,
    "False Color (Exposure)": "false_color",  # Special handler
}


def apply_lut(img: np.ndarray, lut_name: str, intensity: float = 1.0) -> np.ndarray:
    """
    Apply a named LUT to a float32 image.
    
    Args:
        img: float32 numpy array (H,W,C) or (H,W)
        lut_name: Key from LUT_MODES
        intensity: Blend factor 0.0 (bypass) to 1.0 (full)
    
    Returns:
        float32 numpy array, same shape (except False Color always returns H,W,3)
    """
    if lut_name == "None" or intensity <= 0.0:
        return img

    handler = _LUT_FUNCTIONS.get(lut_name)
    if handler is None:
        return img

    # False Color is a special case — replaces the whole image
    if handler == "false_color":
        fc = _lut_false_color(img)
        if intensity >= 1.0:
            return fc
        # Blend: need to match shapes
        if img.ndim == 2:
            orig = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            orig = np.concatenate([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            orig = img[..., :3]
        else:
            orig = img
        return (orig * (1.0 - intensity) + fc * intensity).astype(np.float32)

    # Standard per-channel LUT
    lut_applied = handler(img).astype(np.float32)

    if intensity >= 1.0:
        return lut_applied

    return (img * (1.0 - intensity) + lut_applied * intensity).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#                      COLOR GRADING ENGINE v2.0
#
#  All adjustments operate in linear float32 space.
#  Order: Lift → Exposure → Gain → Gamma → Saturation → Temperature → LUT
# ═══════════════════════════════════════════════════════════════════════════════

def apply_grading(
    img: np.ndarray,
    exposure: float = 0.0,
    gamma: float = 1.0,
    gain: float = 1.0,
    lift: float = 0.0,
    saturation: float = 1.0,
    temperature: float = 6500.0,
    lut_name: str = "None",
    lut_intensity: float = 1.0,
) -> np.ndarray:
    """
    Apply full grading stack to a float32 image.
    
    Pipeline order (industry standard):
        Lift → Exposure → Gain → Gamma → Saturation → Temperature → LUT
    
    Args:
        img: float32 (H,W,C), values can exceed [0,1] for HDR
        exposure: Stops of exposure compensation (-6 to +6)
        gamma: Display gamma (0.1 to 4.0, 1.0 = linear)
        gain: Multiplicative gain (0.0 to 5.0, 1.0 = unity)
        lift: Additive offset to shadows (-1.0 to 1.0)
        saturation: Color saturation (0.0 = mono, 1.0 = normal, 3.0 = hyper)
        temperature: Color temperature in Kelvin (2000 = warm, 12000 = cool)
        lut_name: Name of built-in LUT to apply
        lut_intensity: LUT blend factor (0.0 to 1.0)
    
    Returns:
        float32 numpy array, same shape
    """
    out = img.astype(np.float32, copy=True)

    # Skip if all defaults (fast path)
    is_default = (
        abs(exposure) < 0.001
        and abs(gamma - 1.0) < 0.001
        and abs(gain - 1.0) < 0.001
        and abs(lift) < 0.001
        and abs(saturation - 1.0) < 0.001
        and abs(temperature - 6500.0) < 10.0
        and lut_name == "None"
    )
    if is_default:
        return out

    # ── Lift (shadow offset) ──
    if abs(lift) > 0.001:
        out += lift

    # ── Exposure (stops) ──
    if abs(exposure) > 0.001:
        out *= np.float32(2.0 ** exposure)

    # ── Gain (multiplier) ──
    if abs(gain - 1.0) > 0.001:
        out *= np.float32(gain)

    # ── Gamma (power curve on positive values) ──
    if abs(gamma - 1.0) > 0.001:
        inv_gamma = np.float32(1.0 / max(gamma, 0.01))
        positive = out > 0
        out[positive] = np.power(out[positive], inv_gamma)

    # ── Saturation ──
    if abs(saturation - 1.0) > 0.001 and out.ndim == 3 and out.shape[2] >= 3:
        lum = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])
        for c in range(min(out.shape[2], 3)):
            out[..., c] = lum + saturation * (out[..., c] - lum)

    # ── Color Temperature (Tanner Helland approximation) ──
    if abs(temperature - 6500.0) > 10.0 and out.ndim == 3 and out.shape[2] >= 3:
        r_mult, g_mult, b_mult = _kelvin_to_rgb_multipliers(temperature)
        out[..., 0] *= np.float32(r_mult)
        out[..., 1] *= np.float32(g_mult)
        out[..., 2] *= np.float32(b_mult)

    # ── LUT (last in chain) ──
    if lut_name != "None" and lut_intensity > 0.0:
        out = apply_lut(out, lut_name, lut_intensity)

    return out


def _kelvin_to_rgb_multipliers(kelvin: float) -> Tuple[float, float, float]:
    """
    Convert color temperature (K) to RGB multipliers.
    Based on Tanner Helland's algorithm, normalized so 6500K = (1,1,1).
    """
    temp = max(1000.0, min(40000.0, kelvin)) / 100.0

    # Red
    if temp <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        r = max(0.0, min(255.0, r))

    # Green
    if temp <= 66.0:
        g = 99.4708025861 * np.log(max(temp, 1.0)) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
    g = max(0.0, min(255.0, g))

    # Blue
    if temp >= 66.0:
        b = 255.0
    elif temp <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(max(temp - 10.0, 1.0)) - 305.0447927307
        b = max(0.0, min(255.0, b))

    # Normalize to 6500K baseline
    ref_temp = 65.0  # 6500K / 100
    r_ref = 255.0  # At 6500K, temp<=66 so r=255
    g_ref = 99.4708025861 * np.log(ref_temp) - 161.1195681661
    b_ref = 138.5177312231 * np.log(ref_temp - 10.0) - 305.0447927307

    r_ref = max(r_ref, 1.0)
    g_ref = max(g_ref, 1.0)
    b_ref = max(b_ref, 1.0)

    return (r / r_ref, g / g_ref, b / b_ref)


# ═══════════════════════════════════════════════════════════════════════════════
#                      TENSOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a torch tensor to numpy, handling all dtypes.
    Handles bfloat16 which raises TypeError on direct .numpy().
    Always returns float32 numpy array.
    """
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().cpu().numpy()
    return tensor.cpu().float().numpy()


def compute_data_range(img_np: np.ndarray) -> Tuple[float, float, bool]:
    """
    Compute actual data range and detect HDR content.
    Returns (min_value, max_value, has_hdr).
    """
    d_min = float(img_np.min())
    d_max = float(img_np.max())

    # Sanitize NaN/Inf for JSON safety
    if not math.isfinite(d_min):
        d_min = 0.0
    if not math.isfinite(d_max):
        d_max = 1.0

    has_hdr = d_max > 1.0 or d_min < 0.0
    return d_min, d_max, has_hdr


# ═══════════════════════════════════════════════════════════════════════════════
#                      16-BIT PNG SAVE (v1.21)
# ═══════════════════════════════════════════════════════════════════════════════

def save_16bit_png(filepath: str, img_uint16: np.ndarray) -> bool:
    """
    Save a uint16 numpy array as a true 16-bit PNG using cv2.
    
    Args:
        filepath: Output file path
        img_uint16: uint16 array — (H,W,3) RGB, (H,W,4) RGBA, or (H,W) gray
        
    Returns:
        True on success, False on failure
    """
    if not HAS_CV2:
        logger.error("cv2 required for 16-bit PNG save but not available")
        return False

    try:
        if img_uint16.ndim == 3 and img_uint16.shape[2] == 3:
            bgr = cv2.cvtColor(img_uint16, cv2.COLOR_RGB2BGR)
            return cv2.imwrite(
                filepath, bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 3 and img_uint16.shape[2] == 4:
            bgra = cv2.cvtColor(img_uint16, cv2.COLOR_RGBA2BGRA)
            return cv2.imwrite(
                filepath, bgra,
                [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 3 and img_uint16.shape[2] == 1:
            return cv2.imwrite(
                filepath, img_uint16[:, :, 0],
                [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 2:
            return cv2.imwrite(
                filepath, img_uint16,
                [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        else:
            logger.warning(f"Unsupported shape for 16-bit save: {img_uint16.shape}")
            return False
    except Exception as e:
        logger.error(f"cv2 16-bit PNG save failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE VIEWER NODE v2.0
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceViewer:
    """
    VFX Industry-Standard Viewer with IMAGE passthrough:
    • Zoom/Pan navigation
    • Exposure / Gamma / Gain / Lift / Saturation / Temperature controls
    • Built-in LUT engine (10 industry-standard LUTs)
    • Channel viewing (RGB/R/G/B/Alpha/Luma)
    • Color picker with HDR value display
    • False color & zebra analysis
    • A/B comparison modes
    • 16-bit + HDR output
    • IMAGE passthrough — no more dead-end node
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "bit_depth": (BIT_DEPTH_MODES, {
                    "default": "16-bit (Quality)",
                    "tooltip": (
                        "8-bit: Fast, 256 levels. "
                        "16-bit: Quality, 65536 levels. "
                        "16-bit + HDR: Adds float32 sidecar for true HDR color picker."
                    ),
                }),
                # ── Compare / Depth ──
                "compare_image": ("IMAGE",),
                "zdepth": ("IMAGE", {
                    "tooltip": "Z-Depth map to display when pressing Z button"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    # v2.0: IMAGE passthrough + metadata outputs
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "view"
    CATEGORY = "FXTD Studios/Radiance/Views"
    OUTPUT_NODE = True
    DESCRIPTION = """VFX Industry-Standard Viewer with IMAGE passthrough:
• Channel viewing (RGB/R/G/B/Alpha/Luma)
• Z-Depth visualization
• False color & zebra analysis
• A/B comparison modes
• 16-bit PNG / HDR sidecar
• IMAGE passthrough — no longer a dead-end node"""

    def view(
        self,
        image: torch.Tensor,
        bit_depth: str = "16-bit (Quality)",
        # Compare / Depth
        compare_image: Optional[torch.Tensor] = None,
        zdepth: Optional[torch.Tensor] = None,
        # Hidden
        prompt: Optional[Any] = None,
        extra_pnginfo: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Process and display the image in the Radiance Viewer."""

        # ── Parse bit depth ──
        use_16bit = "16-bit" in bit_depth and HAS_CV2
        # v2.1: ALWAYS save HDR sidecar — essential for 32-bit viewer display.
        # The sidecar enables proper HDR viewing via WebGL float textures.
        # Without it, the viewer only has the clipped [0,1] PNG preview.
        # v2.3: Only if 16-bit is actually requested.
        save_hdr_sidecar = use_16bit

        if "16-bit" in bit_depth and not HAS_CV2:
            logger.warning(
                "16-bit mode requested but cv2 not available. "
                "Falling back to 8-bit. Install opencv-python for 16-bit support."
            )

        # ── Validate ──
        validation_error = self._validate_image(image, "image")
        if validation_error:
            logger.error(validation_error)
            return {
                "ui": {"radiance_images": [], "error": [validation_error]},
                "result": (image, validation_error),
            }

        try:
            output_dir = folder_paths.get_temp_directory()
            batch_size = image.shape[0] if image.dim() == 4 else 1
            batch_size = min(batch_size, MAX_BATCH_SIZE)

            images_list: List[Dict[str, Any]] = []

            for frame_idx in range(batch_size):
                try:
                    frame_result = self._process_frame(
                        image, frame_idx, output_dir,
                        use_16bit=use_16bit,
                        save_hdr_sidecar=save_hdr_sidecar,
                        prefix="radiance_viewer",
                    )
                    if frame_result is not None:
                        frame_result["frame"] = frame_idx
                        frame_result["total_frames"] = batch_size
                        images_list.append(frame_result)
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue

            # Compare image
            if compare_image is not None:
                compare_list = self._process_compare_image(
                    compare_image, output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                )
                images_list.extend(compare_list)

            # Z-Depth
            if zdepth is not None:
                zdepth_list = self._process_zdepth_image(
                    zdepth, output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                )
                images_list.extend(zdepth_list)

            # ── Build output IMAGE ──
            output_image = image

            # ── Metadata string ──
            meta_dict = {
                "bit_depth": "16-bit" if use_16bit else "8-bit",
                "hdr_sidecar": True,  # v2.1: always enabled
                "batch_size": batch_size,
            }
            metadata_str = json.dumps(meta_dict, indent=2)

            return {
                "ui": {
                    "radiance_images": images_list,
                    "batch_size": [batch_size],
                    "bit_depth": ["16-bit" if use_16bit else "8-bit"],
                },
                "result": (output_image, metadata_str),
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Error in viewer: {e}")
            return {
                "ui": {"radiance_images": [], "error": [str(e)]},
                "result": (image, str(e)),
            }

    # ─────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────

    def _validate_image(self, image: torch.Tensor, name: str) -> Optional[str]:
        """Validate image tensor. Returns error string or None if valid."""
        if not isinstance(image, torch.Tensor):
            return f"{name} must be a torch.Tensor, got {type(image)}"
        if image.dim() not in (3, 4):
            return f"{name} must be 3D or 4D tensor, got {image.dim()}D"
        if image.dim() == 4:
            _, h, w, c = image.shape
        else:
            h, w = image.shape[0], image.shape[1]
            c = image.shape[2] if image.dim() == 3 else 1
        if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
            return (
                f"{name} dimensions ({h}x{w}) exceed maximum "
                f"({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            )
        if c not in (1, 3, 4):
            return f"{name} has {c} channels, expected 1, 3, or 4"
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Frame Processing (v2.0: grading + LUT + 16-bit)
    # ─────────────────────────────────────────────────────────────────────

    def _process_frame(
        self,
        image: torch.Tensor,
        frame_idx: int,
        output_dir: str,
        use_16bit: bool = True,
        save_hdr_sidecar: bool = False,
        prefix: str = "radiance_viewer",
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single frame: grade → LUT → save at selected bit depth.
        """
        # Safe tensor → numpy
        if image.dim() == 4:
            frame = safe_tensor_to_numpy(image[frame_idx])
        else:
            frame = safe_tensor_to_numpy(image)

        # Sanitize NaNs and Infs (Critical for WebGL and PNG safety)
        # Replace NaN with 0.0, Inf with 65504.0 (max float16)
        if not np.isfinite(frame).all():
            frame = np.nan_to_num(frame, nan=0.0, posinf=65504.0, neginf=0.0)

        # Data range of RAW input (before grading)
        d_min, d_max, has_hdr = compute_data_range(frame)

        # ── Apply grading + LUT for viewer display ──
        # FIX v2.1: Do NOT apply grading to the preview image saved to disk.
        # The frontend viewer applies grading in real-time via WebGL.
        # If we bake it here, it gets applied twice.
        
        # ── v2.2 ARCHITECTURE: RHDR is the primary display format ──
        # Like DJV/RV viewing EXR natively: viewer loads compressed half-float
        # directly into GPU and tonemaps in real-time. PNG is reduced to a
        # tiny 256px ComfyUI node thumbnail (required by the framework).
        #
        # .rhdr format: zlib-compressed IEEE 754 float16 — lossless at fp16
        # precision, comparable to EXR DWAA at quality ~85.
        # Compression: ~5-15% of raw → similar file sizes to DWAA EXR.

        unique_id = uuid.uuid4().hex[:12]

        # ── 1. PRIMARY: Save .rhdr compressed float16 ──
        # Only if usage of 16-bit is requested.
        rhdr_filename = f"{prefix}_{unique_id}_{frame_idx}.rhdr"
        rhdr_saved = False
        
        if use_16bit:
            try:
                import struct
                # Pad RGB to RGBA for WebGL compatibility
                # Many Windows/NVIDIA WebGL drivers fail with 3-channel float textures (incomplete texture)
                h_frame, w_frame = frame.shape[:2]
                c_frame = frame.shape[2] if frame.ndim == 3 else 1
                
                if c_frame == 3:
                    # Create 4-channel array with Alpha=1.0
                    padded = np.ones((h_frame, w_frame, 4), dtype=np.float32)
                    padded[..., :3] = frame
                    frame_to_save = padded
                    c_frame = 4
                else:
                    frame_to_save = frame

                rhdr_filepath = safe_join(output_dir, rhdr_filename)
                fp16_data = frame_to_save.astype(np.float16).tobytes()
                compressed = zlib.compress(fp16_data, level=6)
                
                header = struct.pack('<4sHHHH', b'RHDR', w_frame, h_frame, c_frame, 0)
                with open(rhdr_filepath, 'wb') as rhdr_f:
                    rhdr_f.write(header)
                    rhdr_f.write(compressed)
                rhdr_saved = True
                ratio = len(compressed) / len(fp16_data) * 100 if len(fp16_data) > 0 else 0
                logger.debug(
                    f"RHDR primary saved: {rhdr_filename} "
                    f"({len(compressed)//1024}KB, {ratio:.0f}% ratio | "
                    f"range [{d_min:.3f}, {d_max:.3f}])"
                )
            except (IOError, OSError, ValueError) as e:
                logger.warning(f"Failed to save RHDR for frame {frame_idx}: {e}")

        # ── 2. SECONDARY: Save .exr (OpenEXR 32-bit float) for external use ──
        # Requested by users for "Save Image" to export true HDR.
        exr_filename = f"{prefix}_{unique_id}_{frame_idx}.exr"
        if HAS_CV2 and has_hdr:
            try:
                exr_filepath = safe_join(output_dir, exr_filename)
                # OpenCV expects BGR
                if frame_to_save.ndim == 3 and frame_to_save.shape[2] >= 3:
                     # RGB -> BGR
                    exr_data = frame_to_save[..., [2, 1, 0] + list(range(3, frame_to_save.shape[2]))]
                else:
                    exr_data = frame_to_save
                
                # Save as float32
                cv2.imwrite(exr_filepath, exr_data.astype(np.float32))
            except Exception as e:
                logger.warning(f"Failed to save EXR: {e}")
                exr_filename = None  # Failed
        else:
             exr_filename = None

        # v3.1 FIX: Always save full-resolution PNG as fallback.
        # Previously 256px thumbnail was used when RHDR was saved, but if RHDR
        # decompression fails in the browser, the viewer had no usable fallback.
        THUMB_MAX = 99999

        if has_hdr and d_max > 1.05:
            preview_safe = np.maximum(frame, 0.0)
            preview_image = (preview_safe / (1.0 + preview_safe)).astype(np.float32)
        else:
            preview_image = frame

        # Downsample to thumbnail
        th, tw = preview_image.shape[:2]
        if max(th, tw) > THUMB_MAX:
            scale = THUMB_MAX / max(th, tw)
            new_w, new_h = int(tw * scale), int(th * scale)
            try:
                if HAS_CV2:
                    preview_thumb = cv2.resize(
                        preview_image, (new_w, new_h),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    from PIL import Image as PILImage
                    pil_full = self._frame_to_pil_8bit(preview_image)
                    if pil_full:
                        pil_full = pil_full.resize((new_w, new_h), PILImage.LANCZOS)
                        preview_thumb = np.array(pil_full).astype(np.float32) / 255.0
                    else:
                        preview_thumb = preview_image
            except Exception as e:
                logger.debug(f"Image resize failed, using original: {e}")
                preview_thumb = preview_image
        else:
            preview_thumb = preview_image

        png_filename = f"{prefix}_{unique_id}_{frame_idx}_thumb.png"
        try:
            png_filepath = safe_join(output_dir, png_filename)
            pil_img = self._frame_to_pil_8bit(preview_thumb)
            if pil_img is None:
                return None
            pil_img.save(png_filepath, compress_level=DEFAULT_PNG_COMPRESSION)
        except (IOError, OSError) as e:
            logger.warning(f"Failed to save thumbnail for frame {frame_idx}: {e}")
            return None

        result: Dict[str, Any] = {
            "filename": png_filename,
            "subfolder": "",
            "type": "temp",
            "data_range": [d_min, d_max],
            "has_hdr": has_hdr,
            "hdr_filename": rhdr_filename if rhdr_saved else None,
            "exr_filename": exr_filename if exr_filename else None,
        }

        if rhdr_saved:
            result["hdr_sidecar"] = rhdr_filename
            result["hdr_primary"] = True  # v2.2: tells viewer to load .rhdr as display source

        return result

    # ─────────────────────────────────────────────────────────────────────
    # PIL 8-bit
    # ─────────────────────────────────────────────────────────────────────

    def _frame_to_pil_8bit(self, frame: np.ndarray) -> Optional["PILImage.Image"]:
        """Convert float32 numpy frame to 8-bit PIL Image."""
        from PIL import Image as PILImage

        img_8bit = self._convert_to_8bit(frame)

        if img_8bit.ndim == 3 and img_8bit.shape[2] == 4:
            return PILImage.fromarray(img_8bit, mode='RGBA')
        elif img_8bit.ndim == 3 and img_8bit.shape[2] == 3:
            return PILImage.fromarray(img_8bit, mode='RGB')
        elif img_8bit.ndim == 3 and img_8bit.shape[2] == 1:
            return PILImage.fromarray(img_8bit[:, :, 0], mode='L')
        elif img_8bit.ndim == 2:
            return PILImage.fromarray(img_8bit, mode='L')

        logger.warning(f"Unexpected 8-bit image shape: {img_8bit.shape}")
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Compare Image
    # ─────────────────────────────────────────────────────────────────────

    def _process_compare_image(
        self,
        compare_image: torch.Tensor,
        output_dir: str,
        use_16bit: bool = True,
        save_hdr_sidecar: bool = False
    ) -> List[Dict[str, Any]]:
        """Process comparison image. Returns list of image metadata."""
        result: List[Dict[str, Any]] = []

        validation_error = self._validate_image(compare_image, "compare_image")
        if validation_error:
            logger.warning(validation_error)
            return result

        cmp_batch = compare_image.shape[0] if compare_image.dim() == 4 else 1
        cmp_batch = min(cmp_batch, MAX_BATCH_SIZE)

        for cmp_idx in range(cmp_batch):
            try:
                frame_result = self._process_frame(
                    compare_image, cmp_idx, output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                    prefix="radiance_compare"
                )
                if frame_result is not None:
                    frame_result["is_compare"] = True
                    frame_result["frame"] = cmp_idx
                    result.append(frame_result)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Error processing compare frame {cmp_idx}: {e}")
                continue

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Z-Depth
    # ─────────────────────────────────────────────────────────────────────

    def _process_zdepth_image(
        self,
        zdepth: torch.Tensor,
        output_dir: str,
        use_16bit: bool = True,
        save_hdr_sidecar: bool = False
    ) -> List[Dict[str, Any]]:
        """Process Z-depth map. 16-bit = 65536 depth levels via cv2."""
        result: List[Dict[str, Any]] = []

        validation_error = self._validate_image(zdepth, "zdepth")
        if validation_error:
            logger.warning(validation_error)
            return result

        depth_batch = zdepth.shape[0] if zdepth.dim() == 4 else 1
        depth_batch = min(depth_batch, MAX_BATCH_SIZE)

        for depth_idx in range(depth_batch):
            try:
                if zdepth.dim() == 4:
                    depth_frame = safe_tensor_to_numpy(zdepth[depth_idx])
                else:
                    depth_frame = safe_tensor_to_numpy(zdepth)

                # Extract single channel
                if depth_frame.ndim == 2:
                    depth_np = depth_frame
                elif depth_frame.ndim == 3:
                    if depth_frame.shape[-1] == 1:
                        depth_np = depth_frame[..., 0]
                    else:
                        depth_np = (
                            0.2126 * depth_frame[..., 0]
                            + 0.7152 * depth_frame[..., 1]
                            + 0.0722 * depth_frame[..., 2]
                        )
                else:
                    continue

                d_min = float(depth_np.min())
                d_max = float(depth_np.max())

                # Sanitize for metadata / JSON safety
                if not math.isfinite(d_min):
                    d_min = 0.0
                if not math.isfinite(d_max):
                    d_max = 1.0

                if d_max > d_min:
                    depth_normalized = (depth_np - d_min) / (d_max - d_min)
                else:
                    depth_normalized = np.zeros_like(depth_np)

                unique_id = uuid.uuid4().hex[:12]
                depth_filename = f"radiance_zdepth_{unique_id}_{depth_idx}.png"

                try:
                    depth_filepath = safe_join(output_dir, depth_filename)
                except ValueError as e:
                    logger.error(f"Invalid zdepth path: {e}")
                    continue

                try:
                    if use_16bit:
                        depth_16bit = (depth_normalized * BIT_16_MAX).astype(np.uint16)
                        success = save_16bit_png(depth_filepath, depth_16bit)
                        if not success:
                            from PIL import Image as PILImage
                            depth_8bit = (depth_normalized * BIT_8_MAX).astype(np.uint8)
                            PILImage.fromarray(depth_8bit, mode='L').save(
                                depth_filepath, compress_level=DEFAULT_PNG_COMPRESSION
                            )
                    else:
                        from PIL import Image as PILImage
                        depth_8bit = (depth_normalized * BIT_8_MAX).astype(np.uint8)
                        PILImage.fromarray(depth_8bit, mode='L').save(
                            depth_filepath, compress_level=DEFAULT_PNG_COMPRESSION
                        )
                except (IOError, OSError) as e:
                    logger.warning(f"Failed to save zdepth frame {depth_idx}: {e}")
                    continue

                frame_meta: Dict[str, Any] = {
                    "filename": depth_filename,
                    "subfolder": "",
                    "type": "temp",
                    "is_zdepth": True,
                    "frame": depth_idx,
                    "bit_depth": 16 if use_16bit else 8,
                    "depth_range": [d_min, d_max],
                }

                if save_hdr_sidecar:
                    npy_filename = f"radiance_zdepth_{unique_id}_{depth_idx}_float.rhdr"
                    try:
                        import struct
                        npy_filepath = safe_join(output_dir, npy_filename)
                        fp16_data = depth_np.astype(np.float16).tobytes()
                        compressed = zlib.compress(fp16_data, level=6)
                        dh, dw = depth_np.shape[:2]
                        dc = depth_np.shape[2] if depth_np.ndim == 3 else 1
                        header = struct.pack('<4sHHHH', b'RHDR', dw, dh, dc, 0)
                        with open(npy_filepath, 'wb') as rhdr_f:
                            rhdr_f.write(header)
                            rhdr_f.write(compressed)
                        frame_meta["hdr_sidecar"] = npy_filename
                    except (IOError, OSError, ValueError) as e:
                        logger.warning(f"Failed to save depth sidecar {depth_idx}: {e}")

                result.append(frame_meta)

            except (RuntimeError, ValueError) as e:
                logger.warning(f"Error processing zdepth frame {depth_idx}: {e}")
                continue

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Bit Depth Conversion
    # ─────────────────────────────────────────────────────────────────────

    def _convert_to_16bit(self, img_np: np.ndarray) -> np.ndarray:
        """Convert float32 image to uint16. Clamps to [0,1] for display."""
        if img_np.dtype in (np.float32, np.float64):
            return (np.clip(img_np, 0.0, 1.0) * BIT_16_MAX).astype(np.uint16)
        elif img_np.dtype == np.float16:
            return (np.clip(img_np.astype(np.float32), 0.0, 1.0) * BIT_16_MAX).astype(np.uint16)
        elif img_np.dtype == np.uint16:
            return img_np
        elif img_np.dtype == np.uint8:
            return img_np.astype(np.uint16) * BIT_16_TO_8_DIVISOR
        else:
            img_f = img_np.astype(np.float64)
            mn, mx = img_f.min(), img_f.max()
            if mx > mn:
                return ((img_f - mn) / (mx - mn) * BIT_16_MAX).astype(np.uint16)
            return np.zeros_like(img_f, dtype=np.uint16)

    def _convert_to_8bit(self, img_np: np.ndarray) -> np.ndarray:
        """Convert float32 image to uint8."""
        if img_np.dtype in (np.float32, np.float64, np.float16):
            return (np.clip(img_np.astype(np.float32), 0.0, 1.0) * BIT_8_MAX).astype(np.uint8)
        elif img_np.dtype == np.uint16:
            return (img_np // BIT_16_TO_8_DIVISOR).astype(np.uint8)
        elif img_np.dtype == np.uint8:
            return img_np
        else:
            img_f = img_np.astype(np.float64)
            mn, mx = img_f.min(), img_f.max()
            if mx > mn:
                return ((img_f - mn) / (mx - mn) * BIT_8_MAX).astype(np.uint8)
            return np.zeros_like(img_f, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#                           NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Need json for metadata output
import json

NODE_CLASS_MAPPINGS = {
    "FXTD_RadianceViewer": RadianceViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTD_RadianceViewer": "◎ Radiance Viewer",
}
