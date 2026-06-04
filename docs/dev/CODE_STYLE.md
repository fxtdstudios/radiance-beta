# Radiance Code Style Guide

## Naming Conventions

### Classes
```python
# ComfyUI Nodes: PascalCase with Radiance prefix
class RadianceLogCurveDecode:  # ✅
class LogCurveDecode:          # ❌ Missing prefix
class radiance_log_curve:      # ❌ Wrong case
```

### Display Names (Nodes)
All nodes specific to Radiance must use the `◎` icon prefix.

```python
NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLogCurveDecode": "◎ Log Curve Decode",
    "RadianceColorGrade": "◎ Color Grade"
}
```

### Menu Category
All nodes must be grouped under `FXTD STUDIOS/Radiance`.

```python
CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"  # ✅
CATEGORY = "Radiance"               # ❌ Old schema
CATEGORY = "utils"                        # ❌ Too generic
```

### Functions
```python
# Utility functions: snake_case, verb_noun pattern
def linear_to_logc3(img):      # ✅ Conversion
def apply_cdl(img, slope):     # ✅ Action
def get_config(path):          # ✅ Getter
def logc3(img):                # ❌ Unclear
```

### Constants
```python
# Module-level: SCREAMING_SNAKE_CASE
LOGC3_EI_PARAMS = {...}        # ✅
AWG3_TO_ACESCG = np.array(...) # ✅
logc3_params = {...}           # ❌
```

### Private/Internal
```python
def _parse_cdl_file(self):     # ✅ Single underscore
_OCIO_PROCESSOR_CACHE = {}     # ✅ Module-private cache
```

## Type Hints

Required for module interface functions:

```python
def linear_to_logc3(img: np.ndarray, ei: int = 800) -> np.ndarray:
    """Convert linear to LogC3."""
    ...
```

Optional but encouraged for node methods.

## Docstrings

### Utility Functions
```python
def linear_to_logc3(img: np.ndarray, ei: int = 800) -> np.ndarray:
    """
    ARRI LogC3 encoding with Exposure Index support.
    
    Args:
        img: Linear image data (H, W, C) or (B, H, W, C)
        ei: Exposure Index (160-3200, default 800)
    
    Returns:
        LogC3 encoded image (same shape)
    """
```

### Node Classes
```python
class RadianceLogCurveDecode:
    """
    Logarithmic to Linear conversion.
    
    v2.0 Features:
    - EI control for LogC3 (160-3200)
    - RED Log3G10 support
    - GPU acceleration option
    """
```

## ComfyUI Node Patterns

### INPUT_TYPES Structure
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            # Always first: image/model inputs
            "image": ("IMAGE",),
            # Then: selection inputs
            "curve": (list(cls.CURVES.keys()),),
            # Then: parameters
            "strength": ("FLOAT", {"default": 1.0}),
        },
        "optional": {
            # Toggle features
            "use_gpu": ("BOOLEAN", {"default": True}),
        }
    }
```

### Widget Tooltips
All widgets should have tooltips for better UX.

```python
"exposure_index": (cls.EI_OPTIONS, {
    "default": "EI 800",
    "tooltip": "Exposure Index for ARRI LogC3 (only affects LogC3 curve)."
}),
```

### Return Tuples
```python
# Always return tuple, even for single output
return (result,)           # ✅
return result              # ❌

# Multiple outputs
RETURN_TYPES = ("IMAGE", "STRING", "INT")
RETURN_NAMES = ("image", "info", "count")
return (image, info, count)
```

## Error Handling

### Graceful Degradation
```python
# Check optional dependencies
if not HAS_OCIO:
    logger.warning("PyOpenColorIO not installed.")
    return (image,)  # Return input unchanged

# Wrap risky operations
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return (image,)
```

## Performance Patterns

### GPU vs CPU Paths
```python
# Check GPU availability
if use_gpu and image.is_cuda:
    try:
        return gpu_function(image)
    except Exception:
        pass  # Fall back to CPU

# CPU fallback
return cpu_function(image.cpu().numpy())
```

## File Organization

Each `nodes_*.py` should have:

```python
# 1. Imports
import logging
import torch
import folder_paths

# 2. Logger setup
logger = logging.getLogger("Radiance")

# 3. Module-level constants

# 4. Helper classes

# 5. Node classes (main content)

# 6. Registration (always at end)
NODE_CLASS_MAPPINGS = {...}
NODE_DISPLAY_NAME_MAPPINGS = {...}
```
