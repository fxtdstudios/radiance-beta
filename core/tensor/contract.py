"""Dimension contract helpers — safe 4D/5D tensor reshaping for image/video."""
import torch
import logging

logger = logging.getLogger("radiance.core.tensor.contract")


def ensure_5d(tensor: torch.Tensor, context_label: str = "Unknown") -> torch.Tensor:
    if tensor.ndim == 4:
        logger.debug("[%s] Auto-reshaping 4D -> 5D (B, C, 1, H, W)", context_label)
        return tensor.unsqueeze(2)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(
        f"[{context_label}] Expected 4D or 5D tensor, got {tensor.ndim}D (shape={tensor.shape})"
    )


def ensure_4d(tensor: torch.Tensor, context_label: str = "Unknown") -> torch.Tensor:
    if tensor.ndim == 5:
        B, C, F, H, W = tensor.shape
        if F == 1:
            logger.debug("[%s] Auto-reshaping 5D(F=1) -> 4D (B, C, H, W)", context_label)
            return tensor.squeeze(2)
        logger.debug("[%s] Auto-flattening 5D(F=%d) -> 4D (B*F, C, H, W)", context_label, F)
        return tensor.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(
        f"[{context_label}] Expected 4D or 5D tensor, got {tensor.ndim}D (shape={tensor.shape})"
    )
