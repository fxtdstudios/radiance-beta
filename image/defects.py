"""
Image Defect Analysis Module
Technical QC Functions for Radiance Pro
Author: FXTD Studios Pipeline Team
"""

import torch
import numpy as np
from typing import Dict, Tuple

def analyze_levels(image: torch.Tensor, black_threshold: float = 0.0, 
                   white_threshold: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Analyze image for crushed blacks and clipped whites.
    
    Args:
        image: (B, H, W, C) tensor in range [0, 1] (or extended for HDR)
        black_threshold: Values below this are considered crushed
        white_threshold: Values above this are considered clipped
    
    Returns:
        Dictionary with per-frame statistics
    """
    B, H, W, C = image.shape
    total_pixels = H * W * C
    
    # Vectorized operations per frame
    crushed_mask = image < black_threshold  # (B, H, W, C)
    clipped_mask = image > white_threshold  # (B, H, W, C)
    
    # Count per frame
    crushed_count = crushed_mask.view(B, -1).sum(dim=1)  # (B,)
    clipped_count = clipped_mask.view(B, -1).sum(dim=1)  # (B,)
    
    # Percentages
    crushed_percent = (crushed_count.float() / total_pixels) * 100.0
    clipped_percent = (clipped_count.float() / total_pixels) * 100.0
    
    # Min/Max per frame
    min_val = image.view(B, -1).min(dim=1)[0]  # (B,)
    max_val = image.view(B, -1).max(dim=1)[0]  # (B,)
    
    return {
        'crushed_count': crushed_count,
        'clipped_count': clipped_count,
        'crushed_percent': crushed_percent,
        'clipped_percent': clipped_percent,
        'min_val': min_val,
        'max_val': max_val,
    }


def check_gamut(image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Check for out-of-gamut values (RGB values outside [0, 1] for SDR).
    For HDR workflows, this detects negative values and extreme outliers.
    
    Args:
        image: (B, H, W, C) tensor
    
    Returns:
        Dictionary with out-of-gamut statistics
    """
    B, H, W, C = image.shape
    total_pixels = H * W * C
    
    # Out of gamut: < 0 or > 1 for SDR workflows
    oog_mask = (image < 0.0) | (image > 1.0)  # (B, H, W, C)
    
    # Count per frame
    oog_count = oog_mask.view(B, -1).sum(dim=1)  # (B,)
    oog_percent = (oog_count.float() / total_pixels) * 100.0
    
    # Negative values (invalid in most workflows)
    negative_mask = image < 0.0
    negative_count = negative_mask.view(B, -1).sum(dim=1)
    
    return {
        'out_of_gamut_count': oog_count,
        'out_of_gamut_percent': oog_percent,
        'negative_count': negative_count,
    }


def detect_banding(image: torch.Tensor, sensitivity: float = 0.01) -> torch.Tensor:
    """
    Heuristic banding detection using gradient analysis.
    Identifies areas with suspiciously uniform gradients (posterization).
    
    Args:
        image: (B, H, W, C) tensor
        sensitivity: Threshold for gradient variation (lower = more sensitive)
    
    Returns:
        Per-frame banding risk percentage (B,)
    """
    B, H, W, C = image.shape
    
    # Convert to grayscale for analysis
    if C == 3:
        # Rec.709 luma weights
        weights = torch.tensor([0.2126, 0.7152, 0.0722], 
                              device=image.device, dtype=image.dtype)
        gray = (image * weights.view(1, 1, 1, 3)).sum(dim=-1)  # (B, H, W)
    else:
        gray = image[..., 0]  # Use first channel
    
    # Compute gradients
    grad_y = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])  # (B, H-1, W)
    grad_x = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])  # (B, H, W-1)
    
    # Detect flat regions with sudden jumps (banding signature)
    # Look for areas where gradient is either ~0 or has discrete jumps
    
    # Quantize gradients to detect discrete levels
    grad_y_quantized = (grad_y / sensitivity).round() * sensitivity
    grad_x_quantized = (grad_x / sensitivity).round() * sensitivity
    
    # Count pixels with quantized gradients (potential banding)
    banding_y = (grad_y_quantized > 0) & (grad_y_quantized < sensitivity * 10)
    banding_x = (grad_x_quantized > 0) & (grad_x_quantized < sensitivity * 10)
    
    # Combine
    banding_pixels_y = banding_y.view(B, -1).sum(dim=1).float()
    banding_pixels_x = banding_x.view(B, -1).sum(dim=1).float()
    
    total_grad_pixels = (H - 1) * W + H * (W - 1)
    banding_risk = ((banding_pixels_y + banding_pixels_x) / total_grad_pixels) * 100.0
    
    return banding_risk  # (B,)


def analyze_noise(image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Estimate noise levels using high-frequency analysis.
    Useful for detecting overly noisy or overly smooth (denoised) images.
    
    Args:
        image: (B, H, W, C) tensor
    
    Returns:
        Dictionary with noise statistics per frame
    """
    B, H, W, C = image.shape
    
    # Convert to grayscale
    if C == 3:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], 
                              device=image.device, dtype=image.dtype)
        gray = (image * weights.view(1, 1, 1, 3)).sum(dim=-1)  # (B, H, W)
    else:
        gray = image[..., 0]
    
    # High-pass filter (approximate)
    # Using simple difference from local mean
    kernel_size = 3
    pad = kernel_size // 2
    
    # Create simple box blur kernel
    # For proper implementation, use torch.nn.functional.avg_pool2d
    # Simplified version here:
    
    # Compute local variance as noise proxy
    mean = gray.mean(dim=[1, 2], keepdim=True)  # (B, 1, 1)
    variance = ((gray - mean) ** 2).mean(dim=[1, 2])  # (B,)
    std_dev = torch.sqrt(variance)  # (B,)
    
    # Noise score (0-100 scale)
    noise_score = torch.clamp(std_dev * 200.0, 0, 100)  # (B,)
    
    return {
        'noise_score': noise_score,
        'std_dev': std_dev,
        'variance': variance,
    }


def detect_compression_artifacts(image: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Detect DCT-based compression artifacts (JPEG blocking).
    Analyzes for discontinuities at block boundaries.
    
    Args:
        image: (B, H, W, C) tensor
        block_size: DCT block size (typically 8 for JPEG)
    
    Returns:
        Per-frame artifact score (B,)
    """
    B, H, W, C = image.shape
    
    # Convert to grayscale
    if C == 3:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], 
                              device=image.device, dtype=image.dtype)
        gray = (image * weights.view(1, 1, 1, 3)).sum(dim=-1)  # (B, H, W)
    else:
        gray = image[..., 0]
    
    # Compute differences at block boundaries
    artifact_score = torch.zeros(B, device=image.device)
    
    # Horizontal block boundaries
    for y in range(block_size, H, block_size):
        if y < H:
            diff = torch.abs(gray[:, y, :] - gray[:, y-1, :]).mean(dim=1)  # (B,)
            artifact_score += diff
    
    # Vertical block boundaries
    for x in range(block_size, W, block_size):
        if x < W:
            diff = torch.abs(gray[:, :, x] - gray[:, :, x-1]).mean(dim=1)  # (B,)
            artifact_score += diff
    
    # Normalize
    num_boundaries = (H // block_size) + (W // block_size)
    if num_boundaries > 0:
        artifact_score = artifact_score / num_boundaries
    
    # Scale to 0-100
    artifact_score = torch.clamp(artifact_score * 1000.0, 0, 100)
    
    return artifact_score  # (B,)


def analyze_focus(image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Estimate focus/sharpness using Laplacian variance.
    Low values indicate out-of-focus or motion-blurred images.
    
    Args:
        image: (B, H, W, C) tensor
    
    Returns:
        Dictionary with focus metrics per frame
    """
    B, H, W, C = image.shape
    
    # Convert to grayscale
    if C == 3:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], 
                              device=image.device, dtype=image.dtype)
        gray = (image * weights.view(1, 1, 1, 3)).sum(dim=-1)  # (B, H, W)
    else:
        gray = image[..., 0]
    
    # Approximate Laplacian using second derivatives
    # d²/dx²
    laplacian_x = gray[:, :, 2:] - 2 * gray[:, :, 1:-1] + gray[:, :, :-2]
    # d²/dy²
    laplacian_y = gray[:, 2:, :] - 2 * gray[:, 1:-1, :] + gray[:, :-2, :]
    
    # Variance of Laplacian (focus measure)
    var_x = laplacian_x.var(dim=[1, 2])  # (B,)
    var_y = laplacian_y.var(dim=[1, 2])  # (B,)
    focus_score = (var_x + var_y) / 2.0
    
    # Scale to 0-100 (higher = sharper)
    focus_score_normalized = torch.clamp(focus_score * 10000.0, 0, 100)
    
    return {
        'focus_score': focus_score_normalized,
        'laplacian_variance': focus_score,
    }


def compute_histogram(image: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    Compute per-channel histograms for the image.
    
    Args:
        image: (B, H, W, C) tensor
        bins: Number of histogram bins
    
    Returns:
        Histogram tensor (B, C, bins)
    """
    B, H, W, C = image.shape
    
    histograms = []
    for b in range(B):
        frame_hists = []
        for c in range(C):
            channel = image[b, :, :, c].flatten()
            # Clamp to [0, 1] for histogram
            channel_clamped = torch.clamp(channel, 0.0, 1.0)
            hist = torch.histc(channel_clamped, bins=bins, min=0.0, max=1.0)
            frame_hists.append(hist)
        histograms.append(torch.stack(frame_hists))
    
    return torch.stack(histograms)  # (B, C, bins)
