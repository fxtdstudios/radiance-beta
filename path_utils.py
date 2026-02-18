"""
═══════════════════════════════════════════════════════════════════════════════
                         RADIANCE PATH UTILITIES
                    Secure file path handling utilities
                        Radiance © 2024-2026
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from typing import Union

__all__ = ['safe_join', 'validate_output_path', 'get_safe_output_dir']


def safe_join(base: str, *paths: str) -> str:
    """
    Safely join paths, preventing directory traversal attacks.
    
    Args:
        base: The base directory that all paths must stay within
        *paths: Path components to join
        
    Returns:
        Normalized absolute path that is guaranteed to be within base
        
    Raises:
        ValueError: If the resulting path would be outside the base directory
        
    Example:
        >>> safe_join("/output", "subdir", "file.exr")
        '/output/subdir/file.exr'
        
        >>> safe_join("/output", "../etc/passwd")  # Raises ValueError
    """
    # Normalize the base path
    base = os.path.normpath(os.path.abspath(base))
    
    # Join and normalize the full path
    full_path = os.path.normpath(os.path.abspath(os.path.join(base, *paths)))
    
    # Verify the result is within the base directory
    # Ensure base directory ends with a separator to avoid partial matches
    # e.g. /home/user vs /home/user2
    base_with_sep = base if base.endswith(os.sep) else base + os.sep

    if not (full_path.startswith(base_with_sep) or full_path == base):
        raise ValueError(
            f"Path traversal detected: '{os.path.join(*paths)}' "
            f"escapes base directory '{base}'"
        )
    
    return full_path


def validate_output_path(base_dir: str, subfolder: str, filename: str) -> str:
    """
    Validate and construct a safe output file path.
    
    Args:
        base_dir: The base output directory
        subfolder: Optional subfolder (can be empty string)
        filename: The filename to write
        
    Returns:
        Safe absolute path for the output file
        
    Raises:
        ValueError: If any path component would escape the base directory
    """
    # Reject absolute paths in subfolder - force relative to base
    if subfolder and os.path.isabs(subfolder):
        raise ValueError(
            f"Absolute subfolder paths not allowed for security: '{subfolder}'. "
            f"Use relative paths only."
        )
    
    # Construct safe path
    if subfolder:
        return safe_join(base_dir, subfolder, filename)
    else:
        return safe_join(base_dir, filename)


def get_safe_output_dir(base_dir: str, subfolder: str = "") -> str:
    """
    Get a validated output directory, creating it if necessary.
    
    Args:
        base_dir: The base output directory
        subfolder: Optional subfolder within base_dir
        
    Returns:
        Safe absolute path to the output directory
        
    Raises:
        ValueError: If subfolder would escape base_dir
    """
    if subfolder and os.path.isabs(subfolder):
        raise ValueError(
            f"Absolute subfolder paths not allowed: '{subfolder}'"
        )
    
    output_dir = safe_join(base_dir, subfolder) if subfolder else os.path.abspath(base_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir
