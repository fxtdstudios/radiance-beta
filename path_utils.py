
import os

__all__ = ["safe_join", "validate_output_path", "get_safe_output_dir", "get_safe_input_path", "get_next_index"]


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
        subfolder: Optional subfolder within base_dir (must be relative)

    Returns:
        Safe absolute path to the output directory

    Raises:
        ValueError: If subfolder is absolute or would escape base_dir

    FIX 1: Previous implementation accepted absolute subfolder paths and returned
    them directly via `os.path.normpath(subfolder)` — completely bypassing
    safe_join and any containment check. The docstring claimed it raised ValueError
    for escaping paths but it never did, contradicting validate_output_path which
    correctly rejects absolute subfolders. Fixed: absolute subfolders now raise
    ValueError, consistent with validate_output_path.
    """
    if subfolder and os.path.isabs(subfolder):
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
    """
    Validate and construct a safe input file path within the input directory.

    Args:
        base_dir:        The base input directory (from ComfyUI)
        filename:        The filename or subpath to read
        allow_absolute:  If True, absolute paths are returned as-is (no containment
                         check). Use only for VFX workflows that explicitly require
                         reading from arbitrary filesystem locations (network drives,
                         project roots, etc.). Default False.

    Returns:
        Safe absolute path for the input file

    Raises:
        ValueError: If filename is absolute and allow_absolute=False, or if the
                    relative path would escape base_dir.

    FIX 2: Previous implementation silently accepted any absolute path with no
    validation, giving callers a false sense of security — the function name and
    docstring imply containment but absolute paths bypassed it entirely.
    Fixed: absolute paths now require an explicit allow_absolute=True opt-in so
    callers must consciously acknowledge the unrestricted access they are granting.
    """
    if os.path.isabs(filename):
        if allow_absolute:
            # Caller has explicitly opted in — normalise but do not constrain.
            return os.path.normpath(filename)
        raise ValueError(
            f"Absolute input paths are not permitted by default: '{filename}'. "
            f"Pass allow_absolute=True to explicitly allow unrestricted paths "
            f"(VFX network drive / project root access only)."
        )

    base_dir = os.path.normpath(os.path.abspath(base_dir))
    return safe_join(base_dir, filename)

def get_next_index(directory: str, prefix: str, extension: str) -> int:
    """
    Find the next available sequence index in a directory for a given prefix.
    Scans the directory for files matching prefix + [digits] + extension and
    returns the highest index found + 1. Returns 0 if no matching files exist.

    Args:
        directory: The directory to scan
        prefix:    Filename prefix
        extension: File extension (e.g., '.png') — include the leading dot

    Returns:
        The next available integer index (0 if no existing files match)

    FIX 3: `padding` parameter was accepted and documented but never used —
    the regex always matched `\\d+` (any digit count) regardless of the value
    passed. Removed to eliminate silent dead-parameter confusion. Callers that
    passed `padding=N` can remove the argument; behavior is unchanged.
    """
    import re
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
