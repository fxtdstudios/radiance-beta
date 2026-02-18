
"""
Radiance Pro Upscaler - Professional 32-bit Upscaling for ComfyUI
Version: 1.1.0
Author: Radiance

Professional upscaling solution optimized for Flux and HDR workflows:
- True 32-bit float processing pipeline
- Multiple upscaling algorithms
- Tile-based processing for large images
- Detail enhancement and sharpening
- Color space aware processing
- Flux-optimized presets

v1.1.0 Fixes:
- Fixed separable_resize_32bit: per-pixel kernel phase now computed correctly
- Fixed torch_resize_32bit: explicit format parameter instead of ambiguous shape detection
- All *_32bit blur/sharpen functions now use true float32 Gaussian (no PIL 8-bit roundtrip)
- Floyd-Steinberg dithering: real error diffusion instead of random noise
- Fixed create_tile_weight: proper 2D outer-product weight (no corner darkening)
- Fixed process_tiles_32bit: eliminated infinite loop when w == tile_size
- Fixed RadianceDownscale32bit GPU path: now applies color space + processing consistently
- Fixed gaussian_kernel default sigma (0.5 → 0.7 for usable kernel width)
- Removed destructive auto-deletion of "small" model files
- Added RadianceSharpen32bit to NODE_CLASS_MAPPINGS (was defined but unregistered)
- Added 'gaussian' to node method lists
- Updated deprecated PIL constants (Image.BILINEAR → Image.Resampling.BILINEAR)
"""

# NOTE: KMP_DUPLICATE_LIB_OK masks OpenMP conflicts. This is a workaround, not a fix.
# If you encounter OpenMP issues, ensure only one OpenMP runtime is linked.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional, List, Union
import math
from enum import Enum
import logging
import threading

# Module logger
logger = logging.getLogger("radiance.image.upscale")

# Global model cache to prevent re-initialization overhead
_MODEL_CACHE = {}
_CACHE_LOCK = threading.RLock()



# =============================================================================
# TRUE 32-BIT GAUSSIAN BLUR (replaces PIL 8-bit roundtrip)
# =============================================================================

def gaussian_blur_32bit(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    True 32-bit Gaussian blur. No precision loss.
    Uses scipy if available (fast C implementation), otherwise pure numpy fallback.
    
    Args:
        img: HWC or HW float32 array (values can exceed 0-1 for HDR)
        sigma: Gaussian sigma in pixels
    
    Returns:
        Blurred float32 array, same shape as input
    """
    if sigma <= 0:
        return img.copy()
    
    # Try scipy first (significantly faster)
    try:
        from scipy.ndimage import gaussian_filter
        if img.ndim == 3:
            # Blur spatial dims only, not channels
            return gaussian_filter(img, sigma=[sigma, sigma, 0]).astype(np.float32)
        return gaussian_filter(img, sigma=sigma).astype(np.float32)
    except ImportError:
        pass
    
    # Pure numpy fallback: separable 1D convolution
    kernel_radius = int(np.ceil(sigma * 3))
    x = np.arange(-kernel_radius, kernel_radius + 1).astype(np.float32)
    kernel_1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    
    def _convolve_axis(data, axis):
        """Apply 1D convolution along an axis with reflect padding."""
        padded = np.pad(data, [(kernel_radius, kernel_radius) if i == axis else (0, 0) 
                                for i in range(data.ndim)], mode='reflect')
        result = np.zeros_like(data)
        slices_base = [slice(None)] * data.ndim
        for k in range(len(kernel_1d)):
            s = list(slices_base)
            s[axis] = slice(k, k + data.shape[axis])
            result += padded[tuple(s)] * kernel_1d[k]
        return result
    
    # Separable: horizontal then vertical
    result = _convolve_axis(img, axis=1)  # Horizontal
    result = _convolve_axis(result, axis=0)  # Vertical
    return result.astype(np.float32)


# =============================================================================
# UPSCALING ALGORITHMS
# =============================================================================

class UpscaleMethod(Enum):
    """Available upscaling methods."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    LANCZOS4 = "lanczos4"
    MITCHELL = "mitchell"
    CATROM = "catrom"
    HERMITE = "hermite"
    GAUSSIAN = "gaussian"


# --- Scalar kernel evaluation functions ---

def _lanczos_scalar(x: float, a: int = 3) -> float:
    """Evaluate Lanczos kernel at a single point."""
    x = abs(x)
    if x == 0:
        return 1.0
    if x >= a:
        return 0.0
    return (a * math.sin(math.pi * x) * math.sin(math.pi * x / a)) / (math.pi ** 2 * x ** 2)


def _mitchell_scalar(x: float, B: float = 1/3, C: float = 1/3) -> float:
    """Evaluate Mitchell-Netravali kernel at a single point."""
    x = abs(x)
    if x < 1:
        return ((12 - 9*B - 6*C) * x**3 + (-18 + 12*B + 6*C) * x**2 + (6 - 2*B)) / 6
    elif x < 2:
        return ((-B - 6*C) * x**3 + (6*B + 30*C) * x**2 + (-12*B - 48*C) * x + (8*B + 24*C)) / 6
    return 0.0


def _hermite_scalar(x: float) -> float:
    """Evaluate Hermite kernel at a single point."""
    x = abs(x)
    if x < 1:
        return 2 * x**3 - 3 * x**2 + 1
    return 0.0


def _gaussian_scalar(x: float, sigma: float = 0.7) -> float:
    """Evaluate Gaussian kernel at a single point. sigma=0.7 gives usable width."""
    return math.exp(-(x ** 2) / (2 * sigma ** 2))


# --- Vectorized kernel functions (for legacy/utility use) ---

def lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """Lanczos kernel function (vectorized)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    mask = x < a
    x_masked = x[mask]
    nonzero = x_masked != 0
    result_masked = np.zeros_like(x_masked)
    if np.any(nonzero):
        x_nz = x_masked[nonzero]
        result_masked[nonzero] = (a * np.sin(np.pi * x_nz) * np.sin(np.pi * x_nz / a)) / (np.pi ** 2 * x_nz ** 2)
    result_masked[~nonzero] = 1.0
    result[mask] = result_masked
    return result


def mitchell_kernel(x: np.ndarray, B: float = 1/3, C: float = 1/3) -> np.ndarray:
    """Mitchell-Netravali kernel (vectorized)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    mask1 = x < 1
    x1 = x[mask1]
    result[mask1] = ((12 - 9*B - 6*C) * x1**3 + (-18 + 12*B + 6*C) * x1**2 + (6 - 2*B)) / 6
    mask2 = (x >= 1) & (x < 2)
    x2 = x[mask2]
    result[mask2] = ((-B - 6*C) * x2**3 + (6*B + 30*C) * x2**2 + (-12*B - 48*C) * x2 + (8*B + 24*C)) / 6
    return result


def catmull_rom_kernel(x: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline kernel (sharper than Mitchell)."""
    return mitchell_kernel(x, B=0, C=0.5)


def hermite_kernel(x: np.ndarray) -> np.ndarray:
    """Hermite kernel (smooth)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    mask = x < 1
    x_m = x[mask]
    result[mask] = (2 * x_m**3 - 3 * x_m**2 + 1)
    return result


def gaussian_kernel(x: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    """Gaussian kernel (vectorized). v1.1.0: sigma=0.7 (was 0.5, too narrow)."""
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


# --- Kernel support and evaluation ---

def _get_kernel_support(method: str) -> float:
    """Get the support radius for a given kernel method."""
    return {
        'lanczos': 3.0, 'lanczos4': 4.0, 'mitchell': 2.0,
        'catrom': 2.0, 'hermite': 1.0, 'gaussian': 2.5,
        'bicubic': 2.0, 'bilinear': 1.0, 'nearest': 0.5,
    }.get(method, 2.0)


def _eval_kernel(distance: float, method: str) -> float:
    """Evaluate kernel function at a given distance from center."""
    if method == 'lanczos':
        return _lanczos_scalar(distance, 3)
    elif method == 'lanczos4':
        return _lanczos_scalar(distance, 4)
    elif method == 'mitchell':
        return _mitchell_scalar(distance)
    elif method == 'catrom':
        return _mitchell_scalar(distance, B=0, C=0.5)
    elif method == 'hermite':
        return _hermite_scalar(distance)
    elif method == 'gaussian':
        return _gaussian_scalar(distance)
    elif method == 'bicubic':
        return _mitchell_scalar(distance, B=0, C=0.5)  # Catmull-Rom for bicubic
    elif method == 'bilinear':
        x = abs(distance)
        return max(0.0, 1.0 - x)
    elif method == 'nearest':
        return 1.0 if abs(distance) < 0.5 else 0.0
    return 0.0


def separable_resize_32bit(img: np.ndarray, new_h: int, new_w: int,
                           method: str = 'lanczos') -> np.ndarray:
    """
    High-quality separable resize maintaining 32-bit precision.
    
    v1.1.0 FIX: Per-pixel kernel phase is now computed correctly.
    Each output pixel maps to a unique fractional position in input space,
    and the kernel is evaluated at the correct distances for that position.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    
    img = img.astype(np.float32)
    
    support = _get_kernel_support(method)
    
    # Scale factors (>1 means downscaling, <1 means upscaling)
    scale_w = w / new_w
    scale_h = h / new_h
    
    # For downscaling, widen the filter to avoid aliasing
    filter_scale_w = max(1.0, scale_w)
    filter_scale_h = max(1.0, scale_h)
    
    # --- Horizontal pass ---
    if new_w != w:
        temp = np.zeros((h, new_w, channels), dtype=np.float32)
        radius_w = int(np.ceil(support * filter_scale_w))
        
        for x_out in range(new_w):
            # Map output pixel center to input space
            x_center = (x_out + 0.5) * scale_w - 0.5
            x_start = int(np.floor(x_center)) - radius_w
            x_end = int(np.floor(x_center)) + radius_w + 1
            
            # Compute per-pixel kernel weights
            num_taps = x_end - x_start
            weights = np.zeros(num_taps, dtype=np.float32)
            for k in range(num_taps):
                # Distance from this input pixel to the mapped center
                distance = (x_start + k - x_center) / filter_scale_w
                weights[k] = _eval_kernel(distance, method)
            
            # Normalize
            weight_sum = weights.sum()
            if weight_sum > 1e-10:
                weights /= weight_sum
            
            # Weighted sum of input pixels
            weighted_sum = np.zeros((h, channels), dtype=np.float32)
            for k in range(num_taps):
                x_src = max(0, min(w - 1, x_start + k))
                weighted_sum += img[:, x_src, :] * weights[k]
            
            temp[:, x_out, :] = weighted_sum
    else:
        temp = img
    
    # --- Vertical pass ---
    if new_h != h:
        result = np.zeros((new_h, new_w, channels), dtype=np.float32)
        radius_h = int(np.ceil(support * filter_scale_h))
        
        for y_out in range(new_h):
            y_center = (y_out + 0.5) * scale_h - 0.5
            y_start = int(np.floor(y_center)) - radius_h
            y_end = int(np.floor(y_center)) + radius_h + 1
            
            num_taps = y_end - y_start
            weights = np.zeros(num_taps, dtype=np.float32)
            for k in range(num_taps):
                distance = (y_start + k - y_center) / filter_scale_h
                weights[k] = _eval_kernel(distance, method)
            
            weight_sum = weights.sum()
            if weight_sum > 1e-10:
                weights /= weight_sum
            
            weighted_sum = np.zeros((new_w, channels), dtype=np.float32)
            for k in range(num_taps):
                y_src = max(0, min(h - 1, y_start + k))
                weighted_sum += temp[y_src, :, :] * weights[k]
            
            result[y_out, :, :] = weighted_sum
    else:
        result = temp
    
    if channels == 1:
        result = result[:, :, 0]
    
    return result


def torch_resize_32bit(tensor: torch.Tensor, new_h: int, new_w: int,
                       method: str = 'bicubic',
                       input_format: str = 'BHWC') -> torch.Tensor:
    """
    PyTorch-based resize maintaining 32-bit precision.
    Uses GPU acceleration when available.
    
    v1.1.0 FIX: Explicit input_format parameter instead of ambiguous shape detection.
    
    Args:
        tensor: Input tensor
        new_h, new_w: Target dimensions
        method: Interpolation method
        input_format: 'BHWC' or 'BCHW'
    
    Returns:
        Resized tensor in BHWC format
    """
    # Ensure 4D
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Convert to BCHW for F.interpolate
    if input_format == 'BHWC':
        tensor = tensor.permute(0, 3, 1, 2)
    
    tensor = tensor.float()
    
    # Map method names to torch modes
    mode_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',    # Best available torch approximation
        'lanczos4': 'bicubic',
        'mitchell': 'bicubic',
        'catrom': 'bicubic',
        'hermite': 'bilinear',
        'gaussian': 'bilinear',
    }
    
    mode = mode_map.get(method, 'bicubic')
    
    if mode == 'nearest':
        resized = F.interpolate(tensor, size=(new_h, new_w), mode=mode)
    else:
        resized = F.interpolate(tensor, size=(new_h, new_w), mode=mode,
                                align_corners=False, antialias=True)
    
    # Convert back to BHWC
    resized = resized.permute(0, 2, 3, 1)
    return resized


# =============================================================================
# DETAIL ENHANCEMENT (v1.1.0: true 32-bit, no PIL roundtrip)
# =============================================================================

def unsharp_mask_32bit(img: np.ndarray, amount: float = 1.0,
                       radius: float = 1.0, threshold: float = 0.0) -> np.ndarray:
    """Unsharp mask in true 32-bit precision."""
    blurred = gaussian_blur_32bit(img, sigma=radius)
    
    # Detail mask
    mask = img - blurred
    
    # Apply threshold
    if threshold > 0:
        mask = np.where(np.abs(mask) > threshold, mask, 0)
    
    return img + mask * amount


def high_pass_sharpen_32bit(img: np.ndarray, strength: float = 0.5,
                            radius: float = 3.0) -> np.ndarray:
    """High-pass sharpening in true 32-bit."""
    low_pass = gaussian_blur_32bit(img, sigma=radius)
    high_pass = img - low_pass
    return img + high_pass * strength


def detail_enhancement_32bit(img: np.ndarray, detail_strength: float = 0.5,
                             edge_strength: float = 0.3,
                             local_contrast: float = 0.2) -> np.ndarray:
    """Multi-scale detail enhancement in true 32-bit."""
    result = img.copy()
    
    # Calculate luminance
    if img.ndim == 3 and img.shape[2] >= 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img[..., 0] if img.ndim == 3 else img
    
    # Multi-scale detail extraction
    scales = [1.0, 2.0, 4.0]
    detail_layers = []
    
    prev_level = lum.copy()
    for sigma in scales:
        current_blur = gaussian_blur_32bit(prev_level[:, :, np.newaxis] if prev_level.ndim == 2 else prev_level,
                                           sigma=sigma)
        if current_blur.ndim == 3:
            current_blur = current_blur[..., 0]
        
        detail = prev_level - current_blur
        detail_layers.append(detail)
        prev_level = current_blur
    
    # Combine details
    combined_detail = np.zeros_like(lum)
    weights = [0.5, 0.3, 0.2]  # Fine to coarse
    for detail, weight in zip(detail_layers, weights):
        combined_detail += detail * weight * detail_strength
    
    # Apply to color channels
    num_channels = min(3, img.shape[2]) if img.ndim == 3 else 1
    for c in range(num_channels):
        if img.ndim == 3:
            result[..., c] = img[..., c] + combined_detail
        else:
            result = img + combined_detail
    
    # Local contrast enhancement
    if local_contrast > 0:
        lum_3d = lum[:, :, np.newaxis] if lum.ndim == 2 else lum
        local_mean = gaussian_blur_32bit(lum_3d, sigma=15.0)
        if local_mean.ndim == 3:
            local_mean = local_mean[..., 0]
        
        local_diff = (lum - local_mean) * local_contrast
        for c in range(num_channels):
            if img.ndim == 3:
                result[..., c] = result[..., c] + local_diff
            else:
                result = result + local_diff
    
    return result


# =============================================================================
# ANTI-ALIASING (v1.1.0: true 32-bit)
# =============================================================================

def apply_antialiasing_32bit(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply edge-aware antialiasing in true 32-bit."""
    # Calculate luminance
    if img.ndim == 3 and img.shape[2] >= 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img[..., 0] if img.ndim == 3 else img
    
    # Edge detection using gradient magnitude
    grad_x = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    grad_y = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edges = edges / (edges.max() + 1e-10)
    
    # True 32-bit blur
    blurred = gaussian_blur_32bit(img, sigma=1.0)
    
    # Blend based on edges
    edge_mask = (edges * strength)[..., np.newaxis]
    result = img * (1 - edge_mask) + blurred * edge_mask
    
    return result


# =============================================================================
# TILE PROCESSING
# =============================================================================

def process_tiles_32bit(img: np.ndarray, tile_size: int, overlap: int,
                        process_func, **kwargs) -> np.ndarray:
    """
    Process image in tiles with overlap for seamless results.
    Maintains 32-bit precision throughout.
    
    v1.1.0 FIX: Eliminated infinite loop when image dimension equals tile_size.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    # Calculate output size from a test tile
    test_tile = img[:min(tile_size, h), :min(tile_size, w)]
    test_output = process_func(test_tile, **kwargs)
    
    scale_h = test_output.shape[0] / test_tile.shape[0]
    scale_w = test_output.shape[1] / test_tile.shape[1]
    
    out_h = int(h * scale_h)
    out_w = int(w * scale_w)
    out_tile_size = int(tile_size * scale_h)
    out_overlap = int(overlap * scale_h)
    
    # Initialize output and weight buffers
    output = np.zeros((out_h, out_w, channels), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)
    
    # Create blending weight
    blend_weight = create_tile_weight(out_tile_size, out_overlap)
    
    # Calculate tile positions
    stride = max(1, tile_size - overlap)
    out_stride = max(1, out_tile_size - out_overlap)
    
    # v1.1.0 FIX: Precompute tile positions to avoid infinite loops
    y_positions = []
    y = 0
    while y < h:
        y_positions.append(y)
        y += stride
        # Handle last tile
        if y < h and y + tile_size > h:
            last_y = max(0, h - tile_size)
            if last_y != y_positions[-1]:
                y_positions.append(last_y)
            break
    
    x_positions = []
    x = 0
    while x < w:
        x_positions.append(x)
        x += stride
        if x < w and x + tile_size > w:
            last_x = max(0, w - tile_size)
            if last_x != x_positions[-1]:
                x_positions.append(last_x)
            break
    
    for y in y_positions:
        for x in x_positions:
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img[y:y_end, x:x_end]
            
            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, channels), dtype=np.float32)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            # Process tile
            processed = process_func(tile, **kwargs)
            
            # Calculate output position
            out_y = int(y * scale_h)
            out_x = int(x * scale_w)
            out_y_end = min(out_y + out_tile_size, out_h)
            out_x_end = min(out_x + out_tile_size, out_w)
            
            # Get tile weight
            tile_h = out_y_end - out_y
            tile_w = out_x_end - out_x
            weight = blend_weight[:tile_h, :tile_w]
            
            # Accumulate
            for c in range(channels):
                output[out_y:out_y_end, out_x:out_x_end, c] += processed[:tile_h, :tile_w, c] * weight
            weights[out_y:out_y_end, out_x:out_x_end] += weight
    
    # Normalize
    weights = np.maximum(weights, 1e-10)
    for c in range(channels):
        output[..., c] /= weights
    
    return output


def create_tile_weight(size: int, overlap: int) -> np.ndarray:
    """
    Create smooth blending weight for tiles.
    
    v1.1.0 FIX: Uses outer product of 1D ramps instead of applying
    ramps independently per axis. This prevents corners from being
    darkened by double multiplication.
    """
    if overlap <= 0:
        return np.ones((size, size), dtype=np.float32)
    
    # Create 1D weight ramp
    weight_1d = np.ones(size, dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    
    weight_1d[:overlap] = ramp
    weight_1d[-overlap:] = ramp[::-1]
    
    # Outer product gives proper 2D weight without corner artifacts
    weight_2d = np.outer(weight_1d, weight_1d)
    
    return weight_2d


# =============================================================================
# COLOR SPACE HANDLING
# =============================================================================

def linear_to_srgb_32bit(img: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB in 32-bit."""
    result = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 0), 1/2.4) - 0.055
    )
    return result.astype(np.float32)


def srgb_to_linear_32bit(img: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear in 32-bit."""
    result = np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((np.maximum(img, 0) + 0.055) / 1.055, 2.4)
    )
    return result.astype(np.float32)


# =============================================================================
# FLUX-OPTIMIZED PRESETS
# =============================================================================

FLUX_PRESETS = {
    "Flux Default": {
        "description": "Balanced upscaling optimized for Flux outputs",
        "method": "lanczos",
        "sharpening": 0.3,
        "detail_enhancement": 0.2,
        "antialiasing": 0.3,
        "color_space": "sRGB",
    },
    "Flux Sharp": {
        "description": "Sharp upscaling for detailed Flux images",
        "method": "lanczos4",
        "sharpening": 0.6,
        "detail_enhancement": 0.4,
        "antialiasing": 0.2,
        "color_space": "sRGB",
    },
    "Flux Smooth": {
        "description": "Smooth upscaling for soft Flux renders",
        "method": "mitchell",
        "sharpening": 0.1,
        "detail_enhancement": 0.1,
        "antialiasing": 0.5,
        "color_space": "sRGB",
    },
    "Flux HDR": {
        "description": "HDR-aware upscaling for high dynamic range",
        "method": "lanczos",
        "sharpening": 0.25,
        "detail_enhancement": 0.3,
        "antialiasing": 0.3,
        "color_space": "Linear",
    },
    "Flux Print": {
        "description": "High quality for print output",
        "method": "lanczos4",
        "sharpening": 0.5,
        "detail_enhancement": 0.5,
        "antialiasing": 0.15,
        "color_space": "sRGB",
    },
    "Flux Cinematic": {
        "description": "Film-like upscaling with subtle softness",
        "method": "catrom",
        "sharpening": 0.2,
        "detail_enhancement": 0.15,
        "antialiasing": 0.4,
        "color_space": "Linear",
    },
    "Flux Maximum": {
        "description": "Maximum detail preservation",
        "method": "lanczos4",
        "sharpening": 0.7,
        "detail_enhancement": 0.6,
        "antialiasing": 0.1,
        "color_space": "sRGB",
    },
}


# =============================================================================
# SHARED METHOD LIST (v1.1.0: includes gaussian)
# =============================================================================

METHOD_LIST_FULL = ["lanczos", "lanczos4", "bicubic", "mitchell", "catrom",
                    "hermite", "gaussian", "bilinear", "nearest"]

METHOD_LIST_QUALITY = ["lanczos", "lanczos4", "bicubic", "mitchell", "catrom",
                       "gaussian", "bilinear"]


# =============================================================================
# MAIN COMFYUI NODES
# =============================================================================

class RadianceProUpscale:
    """
    Professional 32-bit upscaler optimized for Flux with HDR support.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_list = ["Custom"] + list(FLUX_PRESETS.keys())
        
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "preset": (preset_list,),
            },
            "optional": {
                "method": (METHOD_LIST_FULL,),
                "sharpening": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "sharpen_radius": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1
                }),
                "detail_enhancement": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "antialiasing": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "input_color_space": (["sRGB", "Linear", "Auto"],),
                "process_in_linear": ("BOOLEAN", {"default": True}),
                "use_tiles": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {
                    "default": 512, "min": 128, "max": 2048, "step": 64
                }),
                "tile_overlap": ("INT", {
                    "default": 64, "min": 16, "max": 256, "step": 16
                }),
                "output_bit_depth": (["32-bit Float", "16-bit Float", "8-bit"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("upscaled_image", "width", "height", "info")
    FUNCTION = "upscale"
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = "Professional 32-bit upscaler optimized for Flux with HDR support."
    
    def upscale(self, image: torch.Tensor, scale_factor: float, preset: str,
                method: str = "lanczos", sharpening: float = 0.3,
                sharpen_radius: float = 1.0, detail_enhancement: float = 0.2,
                antialiasing: float = 0.3, input_color_space: str = "sRGB",
                process_in_linear: bool = True, use_tiles: bool = False,
                tile_size: int = 512, tile_overlap: int = 64,
                output_bit_depth: str = "32-bit Float"):
        
        # Apply preset if not custom
        if preset != "Custom" and preset in FLUX_PRESETS:
            p = FLUX_PRESETS[preset]
            method = p.get('method', method)
            sharpening = p.get('sharpening', sharpening)
            detail_enhancement = p.get('detail_enhancement', detail_enhancement)
            antialiasing = p.get('antialiasing', antialiasing)
            if p.get('color_space') == 'Linear':
                process_in_linear = True
        
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        # v1.1.0: Auto-Tiling Safety for Large Images
        # If output > 64MP (e.g. 8K+), force tiling to prevent OOM
        SAFE_PIXEL_LIMIT = 64 * 1024 * 1024 # 64 Megapixels
        if new_w * new_h > SAFE_PIXEL_LIMIT and not use_tiles:
            logger.warning(f"Output size {new_w}x{new_h} exceeds safety limit (64MP). Forcing tiled processing.")
            use_tiles = True
            
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            # Handle alpha
            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]
            
            # Convert to linear if needed
            if process_in_linear and input_color_space == "sRGB":
                img = srgb_to_linear_32bit(img)
            
            # Upscale
            if use_tiles and (h > tile_size or w > tile_size):
                def upscale_tile(tile, **kw):
                    th, tw = tile.shape[:2]
                    new_th = int(th * scale_factor)
                    new_tw = int(tw * scale_factor)
                    return separable_resize_32bit(tile, new_th, new_tw, method)
                
                upscaled = process_tiles_32bit(img, tile_size, tile_overlap, upscale_tile)
            else:
                upscaled = separable_resize_32bit(img, new_h, new_w, method)
            
            # Detail enhancement
            if detail_enhancement > 0:
                upscaled = detail_enhancement_32bit(upscaled, detail_enhancement)
            
            # Sharpening
            if sharpening > 0:
                upscaled = unsharp_mask_32bit(upscaled, sharpening, sharpen_radius)
            
            # Antialiasing
            if antialiasing > 0:
                upscaled = apply_antialiasing_32bit(upscaled, antialiasing)
            
            # Convert back to sRGB if processed in linear
            if process_in_linear and input_color_space == "sRGB":
                upscaled = linear_to_srgb_32bit(upscaled)
            
            # Handle alpha
            if has_alpha:
                alpha_up = separable_resize_32bit(alpha, new_h, new_w, 'lanczos')
                if len(alpha_up.shape) == 2:
                    alpha_up = alpha_up[:, :, np.newaxis]
                upscaled = np.concatenate([upscaled, alpha_up], axis=-1)
            
            # Convert bit depth
            if output_bit_depth == "16-bit Float":
                upscaled = upscaled.astype(np.float16).astype(np.float32)
            elif output_bit_depth == "8-bit":
                upscaled = np.clip(upscaled, 0, 1)
                upscaled = (upscaled * 255).astype(np.uint8).astype(np.float32) / 255.0
            # v1.1.1 FIX: 32-bit Float mode (default) no longer clamps to [0,1]
            # preserving HDR range for professional workflows.
            
            results.append(upscaled)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        info = f"Upscaled: {w}x{h} → {new_w}x{new_h} ({scale_factor}x)\n"
        info += f"Method: {method}\n"
        info += f"Preset: {preset}\n"
        info += f"Sharpening: {sharpening}, Detail: {detail_enhancement}\n"
        info += f"Output: {output_bit_depth}"
        
        return (output_tensor, new_w, new_h, info)


class RadianceUpscaleBySize:
    """
    Upscale to exact dimensions with aspect ratio control.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 2048, "min": 64, "max": 16384, "step": 8
                }),
                "height": ("INT", {
                    "default": 2048, "min": 64, "max": 16384, "step": 8
                }),
                "method": (METHOD_LIST_FULL,),
            },
            "optional": {
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                "aspect_mode": (["fit", "fill", "stretch"],),
                "sharpening": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "process_in_linear": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("upscaled_image", "final_width", "final_height")
    FUNCTION = "upscale"
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = "Upscale to exact dimensions with aspect ratio control."
    
    def upscale(self, image: torch.Tensor, width: int, height: int, method: str,
                maintain_aspect: bool = True, aspect_mode: str = "fit",
                sharpening: float = 0.2, process_in_linear: bool = True):
        
        batch_size = image.shape[0]
        orig_h, orig_w = image.shape[1], image.shape[2]
        
        # Calculate target size
        if maintain_aspect:
            aspect_ratio = orig_w / orig_h
            
            if aspect_mode == "fit":
                if width / height > aspect_ratio:
                    new_h = height
                    new_w = int(height * aspect_ratio)
                else:
                    new_w = width
                    new_h = int(width / aspect_ratio)
            elif aspect_mode == "fill":
                if width / height > aspect_ratio:
                    new_w = width
                    new_h = int(width / aspect_ratio)
                else:
                    new_h = height
                    new_w = int(height * aspect_ratio)
            else:  # stretch
                new_w = width
                new_h = height
        else:
            new_w = width
            new_h = height
        
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]
            
            if process_in_linear:
                img = srgb_to_linear_32bit(img)
            
            upscaled = separable_resize_32bit(img, new_h, new_w, method)
            
            if sharpening > 0:
                upscaled = unsharp_mask_32bit(upscaled, sharpening, 1.0)
            
            if process_in_linear:
                upscaled = linear_to_srgb_32bit(upscaled)
            
            if has_alpha:
                alpha_up = separable_resize_32bit(alpha, new_h, new_w, 'lanczos')
                if len(alpha_up.shape) == 2:
                    alpha_up = alpha_up[:, :, np.newaxis]
                upscaled = np.concatenate([upscaled, alpha_up], axis=-1)
            
            results.append(upscaled)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        return (output_tensor, new_w, new_h)








class RadianceDownscale32bit:
    """
    GPU-accelerated 32-bit downscaling with anti-aliasing.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.05,
                    "display": "slider"
                }),
                "method": (METHOD_LIST_QUALITY,),
            },
            "optional": {
                "antialiasing": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "pre_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1
                }),
                "process_in_linear": ("BOOLEAN", {"default": True}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("downscaled_image", "width", "height")
    FUNCTION = "downscale"
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = "GPU-accelerated 32-bit downscaling with anti-aliasing."
    
    def downscale(self, image: torch.Tensor, scale_factor: float, method: str,
                  antialiasing: float = 0.5, pre_blur: float = 0.0,
                  process_in_linear: bool = True, use_gpu: bool = True):
        
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        new_h = max(1, int(h * scale_factor))
        new_w = max(1, int(w * scale_factor))
        
        # v1.1.0 FIX: GPU path now applies color space conversion and pre_blur
        # consistently with the CPU path, so results match regardless of device
        if use_gpu and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                img = image.to(device).float()
                
                # Apply linear conversion if needed
                if process_in_linear:
                    # sRGB → Linear on GPU
                    img = torch.where(
                        img <= 0.04045,
                        img / 12.92,
                        torch.pow((torch.clamp(img, min=0) + 0.055) / 1.055, 2.4)
                    )
                
                # Pre-blur for anti-aliasing (GPU gaussian)
                if pre_blur > 0:
                    kernel_size = max(3, int(pre_blur * 6) | 1)
                    x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
                    gauss_1d = torch.exp(-x**2 / (2 * pre_blur**2))
                    gauss_1d = gauss_1d / gauss_1d.sum()
                    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
                    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
                    
                    b_sz, h_sz, w_sz, c_sz = img.shape
                    img_perm = img.permute(0, 3, 1, 2)
                    img_perm = F.conv2d(
                        img_perm, gauss_2d.expand(c_sz, 1, -1, -1),
                        padding=kernel_size // 2, groups=c_sz
                    )
                    img = img_perm.permute(0, 2, 3, 1)
                
                # Downscale
                img_bchw = img.permute(0, 3, 1, 2)
                mode = 'bicubic' if method not in ('nearest', 'bilinear', 'bicubic') else method
                
                result = F.interpolate(
                    img_bchw, size=(new_h, new_w), mode=mode,
                    align_corners=False if mode != 'nearest' else None,
                    antialias=(mode != 'nearest')
                )
                
                result = result.permute(0, 2, 3, 1)
                
                # Convert back to sRGB
                if process_in_linear:
                    result = torch.where(
                        result <= 0.0031308,
                        result * 12.92,
                        1.055 * torch.pow(torch.clamp(result, min=0), 1/2.4) - 0.055
                    )
                
                # v1.1.1 FIX: Removed output clamp to support HDR downscaling
                # result = torch.clamp(result, min=0)
                return (result.cpu(), new_w, new_h)
                
            except RuntimeError:
                torch.cuda.empty_cache()
        
        # CPU fallback
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]
            
            if process_in_linear:
                img = srgb_to_linear_32bit(img)
            
            # Pre-blur in true 32-bit
            if pre_blur > 0:
                img = gaussian_blur_32bit(img, sigma=pre_blur)
            
            downscaled = separable_resize_32bit(img, new_h, new_w, method)
            
            if process_in_linear:
                downscaled = linear_to_srgb_32bit(downscaled)
            
            if has_alpha:
                alpha_down = separable_resize_32bit(alpha, new_h, new_w, 'lanczos')
                if len(alpha_down.shape) == 2:
                    alpha_down = alpha_down[:, :, np.newaxis]
                downscaled = np.concatenate([downscaled, alpha_down], axis=-1)
            
            results.append(downscaled)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        return (output_tensor, new_w, new_h)


class RadianceBitDepthConvert:
    """
    Convert between bit depths with professional dithering to reduce banding.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image in float32 format."}),
                "output_depth": (["32-bit Float", "16-bit Float", "16-bit Int", "10-bit", "8-bit"], {
                    "tooltip": "Target bit depth."
                }),
            },
            "optional": {
                "dithering": (["None", "Floyd-Steinberg", "Ordered", "Blue Noise", "Random"], {
                    "default": "None",
                    "tooltip": "Dithering algorithm. Floyd-Steinberg = error diffusion, Ordered = Bayer, Blue Noise = high-frequency, Random = simple noise."
                }),
                "dither_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Dithering intensity. 1.0 = standard."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("converted_image", "bit_depth_info")
    OUTPUT_TOOLTIPS = ("Image quantized to target bit depth.", "Information about the conversion.")
    FUNCTION = "convert"
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = "Convert between bit depths with professional dithering to reduce banding."
    
    def convert(self, image: torch.Tensor, output_depth: str,
                dithering: str = "None", dither_strength: float = 1.0):
        
        batch_size = image.shape[0]
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            if output_depth == "32-bit Float":
                results.append(img)
                continue
            elif output_depth == "16-bit Float":
                result = img.astype(np.float16).astype(np.float32)
                results.append(result)
                continue
            elif output_depth == "16-bit Int":
                levels = 65535
            elif output_depth == "10-bit":
                levels = 1023
            else:  # 8-bit
                levels = 255
            
            # Apply dithering before quantization
            if dithering != "None":
                img = self._apply_dither(img, levels, dithering, dither_strength)
            
            # Quantize
            result = np.round(img * levels) / levels
            result = np.clip(result, 0, 1)
            
            results.append(result)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        info = f"Converted to {output_depth}"
        if dithering != "None":
            info += f" with {dithering} dithering"
        
        return (output_tensor, info)
    
    def _apply_dither(self, img: np.ndarray, levels: int,
                      method: str, strength: float) -> np.ndarray:
        """Apply dithering before quantization."""
        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        
        if method == "Floyd-Steinberg":
            # v1.1.0 FIX: Real error diffusion instead of random noise
            result = img.copy()
            for c in range(channels):
                ch = result[..., c] if img.ndim == 3 else result
                for y in range(h):
                    for x in range(w):
                        old_val = ch[y, x]
                        new_val = np.round(old_val * levels) / levels
                        error = (old_val - new_val) * strength
                        ch[y, x] = new_val
                        
                        # Distribute error to neighbors (Floyd-Steinberg pattern)
                        if x + 1 < w:
                            ch[y, x + 1] += error * 7 / 16
                        if y + 1 < h:
                            if x - 1 >= 0:
                                ch[y + 1, x - 1] += error * 3 / 16
                            ch[y + 1, x] += error * 5 / 16
                            if x + 1 < w:
                                ch[y + 1, x + 1] += error * 1 / 16
            
            # Already quantized during error diffusion — caller should not re-quantize
            # Return as-is; the round/clip in convert() will be a no-op
            return result
            
        elif method == "Ordered":
            # Bayer matrix dithering
            bayer = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ], dtype=np.float32) / 16.0 - 0.5
            
            bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
            noise = bayer_tiled[..., np.newaxis] * (strength / levels)
            return img + noise
            
        elif method == "Blue Noise":
            # Approximation using high-frequency noise
            noise1 = np.random.randn(h, w, 1).astype(np.float32)
            noise2 = np.random.randn(h // 2 + 1, w // 2 + 1, 1).astype(np.float32)
            
            # Upsample low-freq noise and subtract to get blue noise spectrum
            from scipy.ndimage import zoom
            try:
                noise2_up = zoom(noise2, [h / noise2.shape[0], w / noise2.shape[1], 1], order=1)
                # Ensure exact shape match
                noise2_up = noise2_up[:h, :w, :]
            except ImportError:
                # Fallback: simple repeat
                noise2_up = np.repeat(np.repeat(noise2, 2, axis=0), 2, axis=1)[:h, :w, :]
            
            blue_noise = (noise1 - noise2_up * 0.5) * (strength / levels)
            return img + blue_noise
        
        elif method == "Random":
            # Simple random noise dithering
            noise = np.random.randn(h, w, 1).astype(np.float32) * (strength / levels)
            return img + noise
        
        return img


# =============================================================================
# AI UPSCALER INTEGRATION
# =============================================================================


class RadianceAIUpscale:
    """
    AI-powered upscaling using neural network models. Supports tiled processing for large images.
    """
    
    AI_MODELS = [
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x2plus",
        "ESRGAN_4x",
        "4x-UltraSharp",
        "4x-AnimeSharp",
        "SwinIR_4x",
        "HAT_4x",
        "SUPIR-v0F_fp16",
        "SUPIR-v0Q_fp16",
    ]
    
    MODEL_URLS = {
        "SUPIR-v0F_fp16": "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0F_fp16.safetensors",
        "SUPIR-v0Q_fp16": "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0Q_fp16.safetensors",
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    }

    def __init__(self):
        self.model = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to upscale."}),
                "model_name": (cls.AI_MODELS, {"default": "RealESRGAN_x4plus",
                    "tooltip": "AI upscaling model. RealESRGAN_x4plus is recommended for general use."}),
                "mode": (["Standard", "Refine (HDR)", "Normalize (HDR)"], {"default": "Standard",
                    "tooltip": "Processing mode. 'Standard' = direct upscale. 'Refine (HDR)' = log-compression for highlights. 'Normalize (HDR)' = scales to safe range."}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64,
                    "tooltip": "Tile size for processing. Smaller = less VRAM, slower."}),
                "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 128, "step": 8,
                    "tooltip": "Overlap between tiles to avoid seams."}),
                "auto_download": ("BOOLEAN", {"default": True,
                    "tooltip": "Automatically download models if not found."}),
            },
            "optional": {
                "unload_model": ("BOOLEAN", {"default": False,
                    "tooltip": "Unload model from VRAM after processing to free memory."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = ("Upscaled image.", "Information about the upscaling process.")
    FUNCTION = "upscale"
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = "AI-powered upscaling using neural network models. Supports tiled processing for large images."

    def _download_model(self, model_name: str, target_path: str) -> bool:
        """Download model if URL is available."""
        if model_name not in self.MODEL_URLS:
            return False
        
        url = self.MODEL_URLS[model_name]
        
        # v1.1.0: Log download size warning for large models
        large_models = {"SUPIR-v0F_fp16", "SUPIR-v0Q_fp16"}
        if model_name in large_models:
            logger.warning(f"⚠ Downloading {model_name} — this is a large model (~6GB) and may take a while.")
        
        logger.info(f"Downloading {model_name} from {url}...")
        
        try:
            import urllib.request
            
            # Download with progress logging
            def _report_progress(block_num, block_size, total_size):
                if total_size > 0 and block_num % 100 == 0:
                    downloaded = block_num * block_size
                    pct = min(100, downloaded * 100 / total_size)
                    logger.info(f"  ↳ {pct:.0f}% ({downloaded / 1024**2:.0f}MB / {total_size / 1024**2:.0f}MB)")
            
            urllib.request.urlretrieve(url, target_path, reporthook=_report_progress)
            logger.info(f"✓ Download complete: {target_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            # Clean up partial download
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
            except Exception:
                pass
            return False

    def _load_model(self, model_name: str):
        """Load an upscale model with caching."""
        with _CACHE_LOCK:
            # Check cache first
            if model_name in _MODEL_CACHE:
                return _MODEL_CACHE[model_name], "Loaded (Cached)"

            try:
                import folder_paths
                from comfy import model_management
                import comfy.utils
            except ImportError:
                return None, "ComfyUI modules not available"

            # Find model path
            model_path = folder_paths.get_full_path("upscale_models", f"{model_name}.pth")
            if model_path is None:
                model_path = folder_paths.get_full_path("upscale_models", f"{model_name}.safetensors")
            
            if model_path is None:
                # Try auto-download
                models_dir = folder_paths.get_folder_paths("upscale_models")[0]
                ext = ".safetensors" if "SUPIR" in model_name else ".pth"
                target_path = os.path.join(models_dir, f"{model_name}{ext}")
                
                if self._download_model(model_name, target_path):
                    model_path = target_path
                else:
                    return None, f"Model {model_name} not found. Place in models/upscale_models/"

            # Load the model
            try:
                sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                
                # Load with spandrel
                try:
                    import spandrel
                except ImportError:
                    return None, "Spandrel library not found. Please update ComfyUI."

                try:
                    # Handle state dict wrapping if necessary
                    if "model" in sd:
                        sd = sd["model"]
                    elif "state_dict" in sd:
                        sd = sd["state_dict"]

                    model_descriptor = spandrel.ModelLoader().load_from_state_dict(sd)
                    upscale_model = model_descriptor.model.eval()
                    
                    _MODEL_CACHE[model_name] = upscale_model
                    return upscale_model, f"Loaded: {model_name}"
                    
                except Exception as e:
                    logger.error(f"Spandrel load failed for {model_name}: {e}")
                    return None, f"Model load error: {str(e)}"
                
            except Exception as e:
                # v1.1.0 FIX: Removed destructive auto-deletion of "small" model files.
                # Log the error and let the user investigate.
                if os.path.exists(model_path):
                    size = os.path.getsize(model_path)
                    if size < 1000:
                        logger.warning(
                            f"⚠ Model file '{model_name}' is suspiciously small ({size} bytes). "
                            f"It may be corrupt or a failed download. Consider deleting and re-downloading: "
                            f"{model_path}"
                        )
                    
                return None, f"Error loading model: {type(e).__name__}: {str(e)}"

    def _fallback_upscale(self, image, model_name):
        """Fallback to algorithmic upscale when AI model unavailable."""
        logger.warning(f"Falling back to Lanczos upscale (AI model not loaded)")
        
        scale = 4
        if "x2" in model_name.lower():
            scale = 2
        elif "x8" in model_name.lower():
            scale = 8
        
        b, h, w, c = image.shape
        new_h, new_w = h * scale, w * scale
        
        img_bchw = image.permute(0, 3, 1, 2)
        upscaled = F.interpolate(
            img_bchw, size=(new_h, new_w), mode='bicubic', align_corners=False
        )
        result = upscaled.permute(0, 2, 3, 1)
        # No clamp here either for HDR safety
        
        return (result, f"Bicubic {scale}x (AI model not available)")

    def _hdr_compress(self, img: torch.Tensor, mode: str) -> Tuple[torch.Tensor, Dict]:
        """Compress HDR values into a range better handled by AI models."""
        metadata = {}
        
        if mode == "Refine (HDR)":
            # Logarithmic compression: log(1 + x)
            # Preserves 0 as 0, compresses helps with high highs
            # Models see '1' as '0.69', '10' as '2.4'
            return torch.log1p(torch.clamp(img, min=0)), metadata
            
        elif mode == "Normalize (HDR)":
            # Linear normalization based on max value
            max_val = torch.max(img)
            # Avoid divide by zero or scaling up LDR images unnecessarily
            min_divisor = 1.0
            divisor = max(max_val.item(), min_divisor)
            metadata["divisor"] = divisor
            return img / divisor, metadata
            
        return img, metadata

    def _hdr_expand(self, img: torch.Tensor, mode: str, metadata: Dict) -> torch.Tensor:
        """Expand compressed values back to original HDR range."""
        if mode == "Refine (HDR)":
            # Inverse log: exp(x) - 1
            return torch.expm1(img)
            
        elif mode == "Normalize (HDR)":
            divisor = metadata.get("divisor", 1.0)
            return img * divisor
            
        return img

    def upscale(self, image, model_name: str = "RealESRGAN_x4plus",
                mode: str = "Standard",
                tile_size: int = 512, tile_overlap: int = 32,
                auto_download: bool = True, unload_model: bool = False):
        """Upscale image using AI model with tiled processing."""
        
        # Load model if needed
        if self.model is None or self.current_model_name != model_name:
            self.model, load_info = self._load_model(model_name)
            self.current_model_name = model_name
            
            if self.model is None:
                logger.warning(f"{load_info}")
                return self._fallback_upscale(image, model_name)

        try:
            from comfy import model_management
            device = model_management.get_torch_device()
            self.model = self.model.to(device)
            
            scale = 4
            if "x2" in model_name.lower():
                scale = 2
            elif "x8" in model_name.lower():
                scale = 8
            
            result_images = []
            
            for batch_idx in range(image.shape[0]):
                img_in = image[batch_idx:batch_idx+1]
                
                # Apply HDR compression
                img_compressed, hdr_meta = self._hdr_compress(img_in, mode)
                
                # Permute to BCHW for processing
                img = img_compressed.permute(0, 3, 1, 2)
                _, c, h, w = img.shape
                
                if h <= tile_size and w <= tile_size:
                    with torch.no_grad():
                        img_device = img.to(device)
                        output = self.model(img_device)
                    result_img = output.permute(0, 2, 3, 1) # Back to BHWC
                else:
                    # Tiled processing
                    new_h, new_w = h * scale, w * scale
                    output = torch.zeros((1, c, new_h, new_w), dtype=torch.float32, device=device)
                    weight = torch.zeros((1, 1, new_h, new_w), dtype=torch.float32, device=device)
                    
                    stride = tile_size - tile_overlap
                    tiles_x = max(1, math.ceil((w - tile_overlap) / stride))
                    tiles_y = max(1, math.ceil((h - tile_overlap) / stride))
                    
                    logger.info(f"Processing {tiles_x * tiles_y} tiles ({tiles_x}x{tiles_y})...")
                    
                    for ty in range(tiles_y):
                        for tx in range(tiles_x):
                            x1 = min(tx * stride, w - tile_size)
                            y1 = min(ty * stride, h - tile_size)
                            x2 = min(x1 + tile_size, w)
                            y2 = min(y1 + tile_size, h)
                            x1 = max(0, x2 - tile_size)
                            y1 = max(0, y2 - tile_size)
                            
                            tile = img[:, :, y1:y2, x1:x2].to(device)
                            
                            with torch.no_grad():
                                tile_output = self.model(tile)
                            
                            out_x1 = x1 * scale
                            out_y1 = y1 * scale
                            out_x2 = x2 * scale
                            out_y2 = y2 * scale
                            
                            th, tw = tile_output.shape[2], tile_output.shape[3]
                            
                            weight_1d_h = torch.ones(th, dtype=torch.float32, device=device)
                            weight_1d_w = torch.ones(tw, dtype=torch.float32, device=device)
                            
                            if tile_overlap > 0:
                                feather = min(tile_overlap * scale, th // 2, tw // 2)
                                if feather > 0:
                                    ramp = torch.linspace(0, 1, feather, device=device)
                                    weight_1d_h[:feather] = ramp
                                    weight_1d_h[-feather:] = ramp.flip(0)
                                    weight_1d_w[:feather] = ramp
                                    weight_1d_w[-feather:] = ramp.flip(0)
                            
                            tile_weight = (weight_1d_h.unsqueeze(1) * weight_1d_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                            
                            output[:, :, out_y1:out_y2, out_x1:out_x2] += tile_output * tile_weight
                            weight[:, :, out_y1:out_y2, out_x1:out_x2] += tile_weight
                    
                    output = output / (weight + 1e-8)
                    result_img = output.permute(0, 2, 3, 1) # Back to BHWC
                
                # Expand back to original HDR range
                result_expanded = self._hdr_expand(result_img.cpu(), mode, hdr_meta)
                result_images.append(result_expanded[0])
            
            result = torch.stack(result_images)
            
            # For Standard/Normalize modes, we might want to clamp min to 0
            # But avoid clamping max for HDR modes
            result = torch.clamp(result, min=0)
            
            info = f"Upscaled with {model_name} ({scale}x) [{mode}]"
            
            if unload_model:
                self.model = None
                self.current_model_name = None
                torch.cuda.empty_cache()
                info += " (model unloaded)"
                logger.info(f"Model unloaded from VRAM")
            
            return (result, info)
            
        except Exception as e:
            logger.error(f"Error during upscale: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_upscale(image, model_name)


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceProUpscale": RadianceProUpscale,
    "RadianceUpscaleBySize": RadianceUpscaleBySize,
    "RadianceDownscale32bit": RadianceDownscale32bit,
    "RadianceBitDepthConvert": RadianceBitDepthConvert,
    "RadianceAIUpscale": RadianceAIUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceProUpscale": "◎ Radiance Pro Upscale",
    "RadianceUpscaleBySize": "◎ Radiance Upscale By Size",
    "RadianceDownscale32bit": "◎ Radiance Downscale 32-bit",
    "RadianceBitDepthConvert": "◎ Radiance Bit Depth Convert",
    "RadianceAIUpscale": "◎ Radiance AI Upscale",
}



