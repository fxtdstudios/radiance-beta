"""Safe filesystem path resolution — prevents directory traversal attacks."""
from __future__ import annotations

import os
import re


def safe_join(base: str, *paths: str) -> str:
    base = os.path.normpath(os.path.abspath(base))
    full_path = os.path.normpath(os.path.abspath(os.path.join(base, *paths)))
    base_with_sep = base if base.endswith(os.sep) else base + os.sep

    if not (full_path.startswith(base_with_sep) or full_path == base):
        raise ValueError(
            f"Path traversal detected: '{os.path.join(*paths)}' "
            f"escapes base directory '{base}'"
        )
    return full_path


def validate_output_path(
    base_dir: str, subfolder: str, filename: str, allow_absolute: bool = False
) -> str:
    if subfolder and os.path.isabs(subfolder):
        if allow_absolute:
            return os.path.normpath(os.path.join(subfolder, filename))
        raise ValueError(
            f"Absolute subfolder paths not allowed for security: '{subfolder}'. "
            f"Use relative paths only."
        )
    if subfolder:
        return safe_join(base_dir, subfolder, filename)
    return safe_join(base_dir, filename)


def get_safe_output_dir(base_dir: str, subfolder: str = "", allow_absolute: bool = False) -> str:
    if subfolder and os.path.isabs(subfolder):
        if allow_absolute:
            output_dir = os.path.normpath(subfolder)
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
        raise ValueError(
            f"Absolute subfolder paths not allowed for security: '{subfolder}'. "
            f"Use relative paths only."
        )
    output_dir = (
        safe_join(base_dir, subfolder) if subfolder else os.path.abspath(base_dir)
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_safe_input_path(base_dir: str, filename: str, allow_absolute: bool = False) -> str:
    if os.path.isabs(filename):
        if allow_absolute:
            return os.path.normpath(filename)
        raise ValueError(
            f"Absolute input paths are not permitted by default: '{filename}'. "
            f"Pass allow_absolute=True to explicitly allow unrestricted paths."
        )
    base_dir = os.path.normpath(os.path.abspath(base_dir))
    return safe_join(base_dir, filename)


def get_next_index(directory: str, prefix: str, extension: str) -> int:
    if not os.path.isdir(directory):
        return 0

    max_idx = -1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(extension)}$")

    try:
        for f in os.listdir(directory):
            match = pattern.match(f)
            if match:
                try:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
    except Exception:
        return 0
    return max_idx + 1
