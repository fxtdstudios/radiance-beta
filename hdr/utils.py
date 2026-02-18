
import torch
import numpy as np
import logging

# Module logger
logger = logging.getLogger("radiance.hdr")

# ═══════════════════════════════════════════════════════════════════════════════
#                           UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy float32 array.
    
    CRITICAL: Always detaches and moves to CPU before conversion
    to prevent GPU memory leaks and ensure float32 precision.
    """
    # Ensure we're working with float32 tensor first
    tensor = tensor.detach().float()
    
    if tensor.dim() == 4:
        # Batch dimension: (B, H, W, C)
        return tensor.cpu().numpy().astype(np.float32)
    elif tensor.dim() == 3:
        # Single image: (H, W, C)
        return tensor.cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unexpected tensor dimensions: {tensor.dim()}")


def numpy_to_tensor_float32(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert numpy array to PyTorch float32 tensor.
    
    Args:
        array: Input numpy array
        device: Target device ("cpu" or "cuda")
        
    CRITICAL: Explicitly enforces float32 dtype to prevent
    silent promotion to float64.
    """
    if array.ndim == 3:
        array = array[np.newaxis, ...]  # Add batch dimension
    # Explicit float32 enforcement - never allow float64
    tensor = torch.from_numpy(array.astype(np.float32))
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def ensure_linear(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Convert sRGB to linear color space."""
    
    # Bypass if gamma is 1.0 (Already Linear)
    if abs(gamma - 1.0) < 0.01:
        return image.astype(np.float32)
        
    # Handle negative values
    sign = np.sign(image)
    abs_image = np.abs(image)
    
    # If gamma matches sRGB approx (2.2), use the precise sRGB curve
    if abs(gamma - 2.2) < 0.1:
        linear = np.where(
            abs_image <= 0.04045,
            abs_image / 12.92,
            np.power((abs_image + 0.055) / 1.055, 2.4)
        )
    else:
        # Generic Gamma
        linear = np.power(abs_image, gamma)
        
    return sign * linear


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB color space."""
    sign = np.sign(image)
    abs_image = np.abs(image)
    
    srgb = np.where(
        abs_image <= 0.0031308,
        abs_image * 12.92,
        1.055 * np.power(abs_image, 1.0/2.4) - 0.055
    )
    return sign * srgb


def tensor_srgb_to_linear(tensor: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """
    Convert sRGB tensor to linear color space.
    GPU-compatible and differentiable.
    """
    # Bypass if gamma is 1.0 (Already Linear)
    if abs(gamma - 1.0) < 0.01:
        return tensor.float()
        
    # Handle negative values
    sign = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)
    
    # If gamma matches sRGB approx (2.2), use the precise sRGB curve
    if abs(gamma - 2.2) < 0.1:
        linear = torch.where(
            abs_tensor <= 0.04045,
            abs_tensor / 12.92,
            torch.pow((abs_tensor + 0.055) / 1.055, 2.4)
        )
    else:
        # Generic Gamma
        linear = torch.pow(abs_tensor, gamma)
        
    return sign * linear


def tensor_linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert linear tensor to sRGB color space.
    GPU-compatible and differentiable.
    """
    sign = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)
    
    srgb = torch.where(
        abs_tensor <= 0.0031308,
        abs_tensor * 12.92,
        1.055 * torch.pow(abs_tensor, 1.0/2.4) - 0.055
    )
    return sign * srgb
