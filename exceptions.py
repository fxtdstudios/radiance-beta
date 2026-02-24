"""
═══════════════════════════════════════════════════════════════════════════════
                         RADIANCE EXCEPTION UTILITIES
                    Unified error handling for all Radiance nodes
                        Radiance © 2024-2026
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import functools
from typing import Any, Callable, TypeVar, Optional, Union

logger = logging.getLogger("radiance.exceptions")

__all__ = [
    "RadianceError",
    "RadianceImageError",
    "RadiancePathError",
    "RadianceGPUError",
    "RadianceDependencyError",
    "handle_node_errors",
    "validate_image_input",
]

# ═══════════════════════════════════════════════════════════════════════════════
#                           CUSTOM EXCEPTION CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceError(Exception):
    """Base exception for all Radiance errors."""

    def __init__(self, message: str, node_name: Optional[str] = None):
        self.node_name = node_name
        prefix = f"[{node_name}] " if node_name else "[Radiance] "
        super().__init__(f"{prefix}{message}")


class RadianceImageError(RadianceError):
    """Raised when image input is invalid or corrupted."""



class RadiancePathError(RadianceError):
    """Raised when file path is invalid or inaccessible."""



class RadianceGPUError(RadianceError):
    """Raised when GPU operation fails."""



class RadianceDependencyError(RadianceError):
    """Raised when a required dependency is missing."""



# ═══════════════════════════════════════════════════════════════════════════════
#                           ERROR HANDLING DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")


def handle_node_errors(
    node_name: str, fallback: Optional[Any] = None, reraise: bool = True
) -> Callable:
    """
    Decorator for consistent error handling in ComfyUI nodes.

    Args:
        node_name: Name of the node for error messages
        fallback: Value to return if error occurs and reraise=False
        reraise: Whether to re-raise exceptions after logging

    Example:
        @handle_node_errors("RadianceUpscale", reraise=True)
        def process(self, image, scale):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except RadianceError:
                # Our custom errors are already formatted
                if reraise:
                    raise
                logger.exception(f"[{node_name}] Operation failed")
                return fallback
            except MemoryError as e:
                logger.error(f"[{node_name}] Out of memory: {e}")
                if reraise:
                    raise RadianceGPUError(
                        "Insufficient memory. Try reducing image size or batch count.",
                        node_name=node_name,
                    ) from e
                return fallback
            except FileNotFoundError as e:
                logger.error(f"[{node_name}] File not found: {e}")
                if reraise:
                    raise RadiancePathError(
                        f"File not found: {e.filename}", node_name=node_name
                    ) from e
                return fallback
            except PermissionError as e:
                logger.error(f"[{node_name}] Permission denied: {e}")
                if reraise:
                    raise RadiancePathError(
                        f"Permission denied: {e.filename}", node_name=node_name
                    ) from e
                return fallback
            except Exception as e:
                logger.exception(f"[{node_name}] Unexpected error: {e}")
                if reraise:
                    raise RadianceError(
                        f"Unexpected error: {e}", node_name=node_name
                    ) from e
                return fallback

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
#                           INPUT VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def validate_image_input(
    image: Any,
    node_name: str,
    min_dims: int = 2,
    max_dims: int = 4,
    require_batch: bool = True,
) -> None:
    """
    Validate image tensor input for a node.

    Args:
        image: Input to validate
        node_name: Node name for error messages
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        require_batch: Whether batch dimension is required

    Raises:
        RadianceImageError: If validation fails
    """
    import torch

    if image is None:
        raise RadianceImageError("No image input provided", node_name=node_name)

    if not isinstance(image, torch.Tensor):
        raise RadianceImageError(
            f"Expected torch.Tensor, got {type(image).__name__}", node_name=node_name
        )

    ndim = len(image.shape)
    if ndim < min_dims or ndim > max_dims:
        raise RadianceImageError(
            f"Expected {min_dims}-{max_dims}D tensor, got {ndim}D", node_name=node_name
        )

    if require_batch and ndim == 3:
        raise RadianceImageError(
            "Missing batch dimension. Expected shape (B, H, W, C)", node_name=node_name
        )

    # Check for empty or degenerate images
    if any(s == 0 for s in image.shape):
        raise RadianceImageError(
            f"Image has zero dimension: {image.shape}", node_name=node_name
        )


def validate_positive(
    value: Union[int, float], name: str, node_name: str, allow_zero: bool = False
) -> None:
    """
    Validate that a value is positive (or non-negative if allow_zero=True).

    Args:
        value: Value to validate
        name: Parameter name for error message
        node_name: Node name for error message
        allow_zero: Whether zero is allowed

    Raises:
        RadianceError: If validation fails
    """
    if allow_zero:
        if value < 0:
            raise RadianceError(
                f"{name} must be >= 0, got {value}", node_name=node_name
            )
    else:
        if value <= 0:
            raise RadianceError(f"{name} must be > 0, got {value}", node_name=node_name)


def validate_range(
    value: Union[int, float],
    name: str,
    node_name: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
) -> None:
    """
    Validate that a value is within a specified range.

    Args:
        value: Value to validate
        name: Parameter name for error message
        node_name: Node name for error message
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Raises:
        RadianceError: If validation fails
    """
    if min_val is not None and value < min_val:
        raise RadianceError(
            f"{name} must be >= {min_val}, got {value}", node_name=node_name
        )
    if max_val is not None and value > max_val:
        raise RadianceError(
            f"{name} must be <= {max_val}, got {value}", node_name=node_name
        )
