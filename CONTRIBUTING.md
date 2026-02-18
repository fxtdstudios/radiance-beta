# Contributing to Radiance

Thank you for contributing to Radiance! This guide helps maintain consistency across the codebase.

## Quick Start

```bash
# Clone and test
cd custom_nodes/radiance
python -m py_compile *.py  # Syntax check all modules
```

## Project Structure

```
radiance/
├── __init__.py           # Node registration hub
├── color_utils.py        # Pure functions (no UI/ComfyUI deps)
├── nodes/*.py            # ComfyUI node classes (by category)
├── tools/                # Scripts and utilities
├── requirements.txt      # Dependencies
├── CONTRIBUTING.md       # This file
└── CODE_STYLE.md         # Detailed style guide
```

## Adding a New Node

### 1. Choose the right module
- Color operations → `nodes_color.py`
- Model loading → `nodes_loader.py`
- HDR/OCIO → `nodes_hdr.py`
- IO/Video → `nodes_io.py`

### 2. Follow the node template

```python
class RadianceYourNode:
    """
    One-line description for the class registry.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "tooltip": "Descriptions are mandatory"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "FXTD Studios/Radiance/YourCategory"  # ✅ Correct hierarchy
    DESCRIPTION = "Tooltip for the node itself."
    
    def process(self, image, strength=1.0):
        # Implementation
        return (result,)
```

### 3. Register the node

Add to the module's `NODE_CLASS_MAPPINGS`. Ensure the display name uses the `◎` icon.

```python
NODE_CLASS_MAPPINGS = {
    "RadianceYourNode": RadianceYourNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceYourNode": "◎ Your Node Name",  # ✅ Icon prefix
}
```

## Dependencies & Tests

- **Nuke Bridge**: If modifying `start_nuke_server.py`, verify connectivity with Nuke 13+.
- **Video IO**: If touching `nodes_io.py`, test ffmpeg availability.
- **Viewer**: Changes to `radiance_viewer.js` require a hard refresh in browser.

## Pull Request Process

1.  **Branch Naming**:
    -   `feat/new-node`
    -   `fix/bug-description`
    -   `docs/update-readme`
2.  **Checklist**:
    -   [ ] Code follows `CODE_STYLE.md`
    -   [ ] Added tooltips to all widgets
    -   [ ] Used `◎` icon in display name
    -   [ ] Verified generic `image` input/output types (no custom types unless necessary)
3.  **Submit PR**: detailed description of changes.

## Version Numbering

- **v2.0.x**: Patch fixes
- **v2.1.0**: New features (non-breaking)
- **v3.0.0**: Breaking changes

## Questions?

Open an issue or contact the FXTD Studios team.
