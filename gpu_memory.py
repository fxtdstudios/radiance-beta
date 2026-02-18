"""
═══════════════════════════════════════════════════════════════════════════════
                         RADIANCE GPU MEMORY UTILITIES
                    Safe GPU memory management for PyTorch
                        Radiance © 2024-2026
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import logging
from typing import Optional, TypeVar, Callable, Any
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger("radiance.gpu_memory")

__all__ = [
    'cleanup_gpu_memory',
    'gpu_memory_guard',
    'safe_gpu_operation',
    'get_gpu_memory_info',
    'ensure_cpu_result'
]


def cleanup_gpu_memory() -> None:
    """
    Force cleanup of GPU memory.
    
    Call this after GPU operations complete or when catching errors
    to prevent VRAM exhaustion during batch processing.
    
    Supports:
    - CUDA (NVIDIA GPUs)
    - MPS (Apple M1/M2/M3/M4 chips)
    """
    # NVIDIA CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Apple MPS cleanup (Metal Performance Shaders)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()


def get_gpu_memory_info() -> dict:
    """
    Get current GPU memory usage statistics.
    
    Returns:
        Dict with allocated, reserved, and total memory in MB
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


@contextmanager
def gpu_memory_guard(cleanup_on_success: bool = True, cleanup_on_error: bool = True):
    """
    Context manager that ensures GPU memory is cleaned up.
    
    Args:
        cleanup_on_success: Whether to cleanup after successful execution
        cleanup_on_error: Whether to cleanup after an error
        
    Example:
        >>> with gpu_memory_guard():
        ...     img = image.to("cuda")
        ...     result = process(img)
        ...     return result.cpu()  # Move back to CPU before exiting
    """
    try:
        yield
    except Exception:
        if cleanup_on_error:
            cleanup_gpu_memory()
        raise
    finally:
        if cleanup_on_success:
            cleanup_gpu_memory()


def ensure_cpu_result(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure a tensor is on CPU before returning from a node.
    
    This prevents VRAM accumulation when results are passed between nodes.
    
    Args:
        tensor: Input tensor (may be on GPU or CPU)
        
    Returns:
        Same tensor data, guaranteed to be on CPU
    """
    if tensor.is_cuda:
        return tensor.cpu()
    return tensor


T = TypeVar('T')


def safe_gpu_operation(
    fallback_to_cpu: bool = True,
    cleanup_after: bool = True
) -> Callable:
    """
    Decorator for GPU operations with automatic fallback and cleanup.
    
    Args:
        fallback_to_cpu: If True, retry on CPU when GPU fails
        cleanup_after: If True, cleanup GPU memory after operation
        
    Example:
        >>> @safe_gpu_operation(fallback_to_cpu=True)
        ... def process(image, use_gpu=True):
        ...     device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        ...     # ... processing ...
        ...     return result
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            use_gpu = kwargs.get('use_gpu', True)
            
            if use_gpu and torch.cuda.is_available():
                try:
                    result = func(*args, **kwargs)
                    if cleanup_after:
                        cleanup_gpu_memory()
                    return result
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "CUDA" in str(e):
                        logger.warning(f"GPU operation failed: {e}. Falling back to CPU.")
                        cleanup_gpu_memory()
                        
                        if fallback_to_cpu:
                            kwargs['use_gpu'] = False
                            return func(*args, **kwargs)
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class GPUMemoryTracker:
    """
    Track GPU memory usage across operations for debugging.
    
    Example:
        >>> tracker = GPUMemoryTracker()
        >>> tracker.checkpoint("before_processing")
        >>> # ... processing ...
        >>> tracker.checkpoint("after_processing")
        >>> tracker.report()
    """
    
    def __init__(self):
        self.checkpoints = []
    
    def checkpoint(self, label: str) -> None:
        """Record current memory state with a label."""
        if torch.cuda.is_available():
            self.checkpoints.append({
                "label": label,
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            })
    
    def report(self) -> str:
        """Generate a report of memory usage across checkpoints."""
        if not self.checkpoints:
            return "No checkpoints recorded"
        
        lines = ["GPU Memory Report:", "-" * 40]
        for cp in self.checkpoints:
            lines.append(
                f"{cp['label']}: {cp['allocated_mb']:.1f}MB allocated, "
                f"{cp['reserved_mb']:.1f}MB reserved"
            )
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all checkpoints."""
        self.checkpoints = []
