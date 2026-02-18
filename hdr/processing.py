
import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional

# Local imports
from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32

logger = logging.getLogger("radiance.hdr.processing")

GPU_UTILS_AVAILABLE = False
try:
    # Try to import GPU utils if available in the package
    from ..gpu_utils import gpu_laplacian_pyramid_blend, gpu_local_contrast
    GPU_UTILS_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
#                          HDR EXPOSURE BLENDING
# ═══════════════════════════════════════════════════════════════════════════════

class HDRExposureBlend:
    """
    Blend multiple exposures (bracketing) for extended dynamic range. Takes highlights from low exposure and shadows from high exposure for optimal color grading.
    """

    BLEND_METHODS = [
        "Mertens Fusion",
        "Luminance Weighted",
        "Shadow/Highlight Mask",
        "Exposure Weighted",
        "Laplacian Pyramid",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "low_exposure": ("IMAGE",),   # Darker image - good highlights
                "high_exposure": ("IMAGE",),  # Brighter image - good shadows
                "blend_method": (cls.BLEND_METHODS, {"default": "Mertens Fusion"}),
            },
            "optional": {
                "mid_exposure": ("IMAGE",),   # Optional middle exposure
                "shadow_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "highlight_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "transition_smoothness": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05}),
                "exposure_offset_low": ("FLOAT", {"default": -2.0, "min": -6.0, "max": 0.0, "step": 0.5}),
                "exposure_offset_high": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 6.0, "step": 0.5}),
                "ghost_removal": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("blended_hdr", "blend_mask", "blend_info")
    FUNCTION = "blend_exposures"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Blend multiple exposures (bracketing) for extended dynamic range. Takes highlights from low exposure and shadows from high exposure for optimal color grading."

    def _calculate_luminance(self, img: np.ndarray) -> np.ndarray:
        """Calculate Rec.709 luminance."""
        return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    def _mertens_weights(self, img: np.ndarray, contrast_weight: float = 1.0,
                        saturation_weight: float = 1.0, exposure_weight: float = 1.0) -> np.ndarray:
        """Calculate Mertens exposure fusion weights."""
        # Contrast measure (Laplacian)
        gray = self._calculate_luminance(img)
        laplacian = np.abs(np.gradient(np.gradient(gray, axis=0), axis=0) + 
                          np.gradient(np.gradient(gray, axis=1), axis=1))
        contrast = laplacian ** contrast_weight
        
        # Saturation measure
        mean_rgb = np.mean(img, axis=-1)
        saturation = np.sqrt(np.mean((img - mean_rgb[..., np.newaxis]) ** 2, axis=-1))
        saturation = saturation ** saturation_weight
        
        # Well-exposedness measure (Gaussian centered at 0.5)
        sigma = 0.2
        exposedness = np.exp(-0.5 * ((img - 0.5) / sigma) ** 2)
        exposedness = np.prod(exposedness, axis=-1) ** exposure_weight
        
        # Combined weight
        weight = contrast * saturation * exposedness + 1e-10
        return weight

    def _mertens_fusion(self, images: list, weights: list = None) -> np.ndarray:
        """Mertens exposure fusion algorithm."""
        if weights is None:
            weights = [self._mertens_weights(img) for img in images]
        
        # Normalize weights
        weight_sum = sum(weights)
        weight_sum = np.maximum(weight_sum, 1e-10)
        normalized_weights = [w / weight_sum for w in weights]
        
        # Weighted blend
        result = np.zeros_like(images[0])
        for img, w in zip(images, normalized_weights):
            result += img * w[..., np.newaxis]
        
        return result

    def _luminance_weighted_blend(self, low_exp: np.ndarray, high_exp: np.ndarray,
                                  transition: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Luminance-weighted blending for natural HDR merge."""
        # Calculate luminance of both images
        lum_low = self._calculate_luminance(low_exp)
        lum_high = self._calculate_luminance(high_exp)
        
        # Create smooth transition mask based on luminance
        # Low exposure: prefer it for bright areas (highlights preserved)
        # High exposure: prefer it for dark areas (shadows preserved)
        
        # Sigmoid-based transition
        mid_point = 0.5
        x = (lum_low - mid_point) / transition
        low_weight = 1.0 / (1.0 + np.exp(-x))  # High weight for bright areas
        high_weight = 1.0 - low_weight
        
        # Blend
        result = (low_exp * low_weight[..., np.newaxis] + 
                  high_exp * high_weight[..., np.newaxis])
        
        mask = low_weight[..., np.newaxis].repeat(3, axis=-1)
        return result, mask

    def _shadow_highlight_blend(self, low_exp: np.ndarray, high_exp: np.ndarray,
                                shadow_weight: float, highlight_weight: float,
                                transition: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract shadows from high exposure, highlights from low exposure."""
        lum_low = self._calculate_luminance(low_exp)
        
        # Shadow mask (where low exposure is dark)
        shadow_threshold = 0.25
        shadow_mask = np.clip((shadow_threshold - lum_low) / transition + 0.5, 0, 1)
        
        # Highlight mask (where low exposure is bright)
        highlight_threshold = 0.75
        highlight_mask = np.clip((lum_low - highlight_threshold) / transition + 0.5, 0, 1)
        
        # Midtone mask (remainder)
        midtone_mask = 1.0 - shadow_mask - highlight_mask
        midtone_mask = np.maximum(midtone_mask, 0)
        
        # Blend: shadows from high_exp, highlights from low_exp, midtones averaged
        result = (high_exp * shadow_mask[..., np.newaxis] * shadow_weight +
                  low_exp * highlight_mask[..., np.newaxis] * highlight_weight +
                  (low_exp + high_exp) / 2 * midtone_mask[..., np.newaxis])
        
        # Normalize
        total_weight = (shadow_mask * shadow_weight + 
                       highlight_mask * highlight_weight + 
                       midtone_mask)
        total_weight = np.maximum(total_weight, 1e-10)
        result = result / total_weight[..., np.newaxis]
        
        # Visualization mask
        mask = np.stack([shadow_mask, midtone_mask, highlight_mask], axis=-1)
        return result, mask

    def _laplacian_pyramid_blend(self, low_exp: np.ndarray, high_exp: np.ndarray,
                                 levels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-scale Laplacian pyramid blending for seamless HDR merge."""
        
        # Use GPU-accelerated version if available
        if GPU_UTILS_AVAILABLE and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                low_tensor = torch.from_numpy(low_exp).to(device)
                high_tensor = torch.from_numpy(high_exp).to(device)
                
                # Create mask based on luminance
                lum = 0.2126 * low_tensor[..., 0] + 0.7152 * low_tensor[..., 1] + 0.0722 * low_tensor[..., 2]
                mask = (lum > 0.5).float()
                
                result_tensor = gpu_laplacian_pyramid_blend(low_tensor, high_tensor, mask, levels)
                result = result_tensor.cpu().numpy()
                mask_np = mask.cpu().numpy()[..., np.newaxis].repeat(3, axis=-1)
                return result, mask_np
            except Exception as e:
                logger.debug(f"GPU Laplacian blend failed, falling back to CPU: {e}")
        
        # Fallback to scipy
        from scipy.ndimage import gaussian_filter
        
        def build_gaussian_pyramid(img, levels):
            pyramid = [img]
            for _ in range(levels - 1):
                blurred = gaussian_filter(pyramid[-1], sigma=2)
                downsampled = blurred[::2, ::2]
                pyramid.append(downsampled)
            return pyramid
        
        def build_laplacian_pyramid(gaussian_pyr):
            laplacian = []
            for i in range(len(gaussian_pyr) - 1):
                upsampled = np.repeat(np.repeat(gaussian_pyr[i+1], 2, axis=0), 2, axis=1)
                # Handle size mismatch
                h, w = gaussian_pyr[i].shape[:2]
                upsampled = upsampled[:h, :w]
                laplacian.append(gaussian_pyr[i] - upsampled)
            laplacian.append(gaussian_pyr[-1])
            return laplacian
        
        # Build pyramids
        g_low = build_gaussian_pyramid(low_exp, levels)
        g_high = build_gaussian_pyramid(high_exp, levels)
        l_low = build_laplacian_pyramid(g_low)
        l_high = build_laplacian_pyramid(g_high)
        
        # Create mask based on luminance
        lum = self._calculate_luminance(low_exp)
        mask = (lum > 0.5).astype(np.float32)
        mask_pyr = build_gaussian_pyramid(mask[..., np.newaxis].repeat(3, axis=-1), levels)
        
        # Blend pyramids
        blended_laplacian = []
        for l_l, l_h, m in zip(l_low, l_high, mask_pyr):
            blended = l_l * m + l_h * (1 - m)
            blended_laplacian.append(blended)
        
        # Reconstruct from blended Laplacian pyramid
        result = blended_laplacian[-1]
        for i in range(len(blended_laplacian) - 2, -1, -1):
            upsampled = np.repeat(np.repeat(result, 2, axis=0), 2, axis=1)
            h, w = blended_laplacian[i].shape[:2]
            upsampled = upsampled[:h, :w]
            result = upsampled + blended_laplacian[i]
        
        return result, mask[..., np.newaxis].repeat(3, axis=-1)

    def blend_exposures(self, low_exposure: torch.Tensor, high_exposure: torch.Tensor,
                       blend_method: str = "Mertens Fusion",
                       mid_exposure: torch.Tensor = None,
                       shadow_weight: float = 1.0, highlight_weight: float = 1.0,
                       transition_smoothness: float = 0.3,
                       exposure_offset_low: float = -2.0, exposure_offset_high: float = 2.0,
                       ghost_removal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, str]:

        low_np = tensor_to_numpy_float32(low_exposure)
        high_np = tensor_to_numpy_float32(high_exposure)
        
        # Handle batch dimension
        if low_np.ndim == 4:
            low_np = low_np[0]
            high_np = high_np[0]
        
        # Apply exposure compensation to bring to common scale
        low_np = low_np * (2.0 ** (-exposure_offset_low))  # Brighten low exposure
        high_np = high_np * (2.0 ** (-exposure_offset_high))  # Darken high exposure
        
        # Perform blending based on method
        if blend_method == "Mertens Fusion":
            images = [low_np, high_np]
            if mid_exposure is not None:
                mid_np = tensor_to_numpy_float32(mid_exposure)
                if mid_np.ndim == 4:
                    mid_np = mid_np[0]
                images.insert(1, mid_np)
            result = self._mertens_fusion(images)
            mask = np.ones_like(result) * 0.5
            
        elif blend_method == "Luminance Weighted":
            result, mask = self._luminance_weighted_blend(low_np, high_np, transition_smoothness)
            
        elif blend_method == "Shadow/Highlight Mask":
            result, mask = self._shadow_highlight_blend(low_np, high_np, shadow_weight, 
                                                        highlight_weight, transition_smoothness)
            
        elif blend_method == "Exposure Weighted":
            # Simple weighted average based on exposure settings
            total_range = abs(exposure_offset_high - exposure_offset_low)
            low_weight = abs(exposure_offset_high) / total_range
            high_weight = abs(exposure_offset_low) / total_range
            result = low_np * low_weight + high_np * high_weight
            mask = np.ones_like(result) * low_weight
            
        elif blend_method == "Laplacian Pyramid":
            result, mask = self._laplacian_pyramid_blend(low_np, high_np)
        else:
            result = (low_np + high_np) / 2
            mask = np.ones_like(result) * 0.5
        
        # Ensure valid range for 32-bit HDR
        result = np.maximum(result, 0.0)
        
        # Calculate blend statistics
        dynamic_range = np.log2(np.maximum(result.max(), 1e-10) / np.maximum(result[result > 0].min(), 1e-10))
        
        info = f"Method: {blend_method} | DR: {dynamic_range:.1f} stops | Range: [{result.min():.3f}, {result.max():.3f}]"
        
        return (numpy_to_tensor_float32(result), numpy_to_tensor_float32(mask.astype(np.float32)), info)


class HDRShadowHighlightRecovery:
    """
    Recover shadow and highlight detail from a single HDR image for better color grading flexibility.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "highlight_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "shadow_tone": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.01}),
                "highlight_tone": ("FLOAT", {"default": 0.75, "min": 0.5, "max": 1.0, "step": 0.01}),
                "color_correction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "local_contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("recovered_image",)
    FUNCTION = "recover"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Recover shadow and highlight detail from a single HDR image for better color grading flexibility."

    def recover(self, image: torch.Tensor, shadow_amount: float = 0.5,
                highlight_amount: float = 0.5, shadow_tone: float = 0.25,
                highlight_tone: float = 0.75, color_correction: float = 0.5,
                local_contrast: float = 0.0) -> Tuple[torch.Tensor]:
        
        img = tensor_to_numpy_float32(image)
        if img.ndim == 4:
            img = img[0]
        
        # Calculate luminance
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        lum = np.maximum(lum, 1e-10)
        
        # Shadow recovery
        # New HDR-safe mask: 1.0 at 0, smooth falloff to 0.0 at shadow_tone
        shadow_ratio = lum / (shadow_tone + 1e-6)
        shadow_mask = np.exp(-3.0 * shadow_ratio)  # Exponential decay
        
        shadow_boost = (1.0 + shadow_amount * shadow_mask)
        
        # Highlight recovery
        # New HDR-safe mask: 0.0 below tone, smooth rise to 1.0+ above
        
        # Normalized position above threshold
        h_pos = (lum - highlight_tone) / (1.0 - highlight_tone + 1e-6)
        
        # Sigmoid-like activation that works for > 1.0
        # h_pos: 0.0 at highlight_tone
        # rises to 1.0 at h_pos=1.0 (original lum=1.0)
        # continues rising for h_pos > 1.0 (super-whites)
        
        highlight_mask = np.maximum(0.0, h_pos)
        
        # Apply reduction (soft compression, non-clamping)
        highlight_reduce = 1.0 / (1.0 + highlight_amount * highlight_mask * 0.5)
        
        # Apply to image
        result = img * shadow_boost[..., np.newaxis] * highlight_reduce[..., np.newaxis]
        
        # Color correction to reduce oversaturation in lifted shadows
        if color_correction > 0:
            new_lum = 0.2126 * result[..., 0] + 0.7152 * result[..., 1] + 0.0722 * result[..., 2]
            new_lum = np.maximum(new_lum, 1e-10)
            sat_factor = 1.0 - (shadow_mask * color_correction * 0.3)
            result = new_lum[..., np.newaxis] + sat_factor[..., np.newaxis] * (result - new_lum[..., np.newaxis])
        
        # Local contrast enhancement
        if local_contrast != 0:
            # Use GPU-accelerated local contrast if available
            if GPU_UTILS_AVAILABLE and torch.cuda.is_available():
                try:
                    device = torch.device("cuda")
                    result_tensor = torch.from_numpy(result.astype(np.float32)).to(device)
                    result_tensor = gpu_local_contrast(result_tensor, sigma=50.0, amount=local_contrast)
                    result = result_tensor.cpu().numpy().astype(np.float32)
                except Exception as e:
                    logger.debug(f"GPU local contrast failed, falling back to CPU: {e}")
                    from scipy.ndimage import gaussian_filter
                    local_lum = gaussian_filter(lum, sigma=50)
                    detail = lum / (local_lum + 1e-10)
                    contrast_boost = 1.0 + local_contrast * (detail - 1.0)
                    result = result * contrast_boost[..., np.newaxis]
            else:
                # Fallback to scipy
                from scipy.ndimage import gaussian_filter
                local_lum = gaussian_filter(lum, sigma=50)
                detail = lum / (local_lum + 1e-10)
                contrast_boost = 1.0 + local_contrast * (detail - 1.0)
                result = result * contrast_boost[..., np.newaxis]

        return (numpy_to_tensor_float32(result),)


class GPUTensorOps:
    """
    GPU-accelerated HDR operations. Fast exposure, gamma, and normalization.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": (["Exposure", "Gamma", "Lift/Gain", "Normalize", "Clamp"], {"default": "Exposure"}),
            },
            "optional": {
                "value": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "force_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "performance_info")
    FUNCTION = "process"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "GPU-accelerated HDR operations. Fast exposure, gamma, and normalization."

    def process(self, image: torch.Tensor, operation: str,
                value: float = 0.0, force_gpu: bool = True) -> Tuple[torch.Tensor, str]:
        
        import time
        start_time = time.time()

        # Determine device
        if force_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        img = image.to(device)

        # Apply operation
        if operation == "Exposure":
            result = img * (2.0 ** value)
        elif operation == "Gamma":
            gamma = max(0.1, value) if value != 0 else 2.2
            result = torch.pow(torch.clamp(img, 0.001, None), 1.0 / gamma)
        elif operation == "Lift/Gain":
            result = img + value  # Simple lift
        elif operation == "Normalize":
            min_val = img.min()
            max_val = img.max()
            result = (img - min_val) / (max_val - min_val + 1e-10)
        elif operation == "Clamp":
            result = torch.clamp(img, 0.0, 1.0)
        else:
            result = img

        result = result.cpu()
        
        elapsed = (time.time() - start_time) * 1000
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        info = f"Device: {device_name} | Op: {operation} | Time: {elapsed:.2f}ms"

        return (result, info)
