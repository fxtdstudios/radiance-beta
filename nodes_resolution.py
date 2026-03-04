"""
═══════════════════════════════════════════════════════════════════════════════
    Radiance Resolution v2.1 — Professional Resolution Selector
                    Radiance © 2024-2026 FXTD STUDIOS

Place this at: radiance/nodes_resolution.py
  (replaces the old facade that imported from .image.resolution)

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  RadianceResolution                                             │
  │                                                                 │
  │  Preset ──┐                                                     │
  │  Custom W ─┤──▶ Resolve (W, H) ──▶ Empty Latent (B,C,H/8,W/8) │
  │  Custom H ─┤         │                                          │
  │  Orient. ──┘         │                                          │
  │                      ▼                                          │
  │              Internal Preview                                   │
  │           (resolution info card)                                │
  │           saved to temp → "ui"                                  │
  │                                                                 │
  │  OUTPUT: LATENT only                                            │
  └─────────────────────────────────────────────────────────────────┘

Features:
  - 30+ cinema/digital/AI resolution presets
  - Custom W×H override with 8-pixel alignment
  - Model-aware latent channels (SD=4, SDXL=4, Flux/SD3=16)
  - Landscape / Portrait / Square orientation
  - Megapixel target mode (auto-calculate dimensions from aspect ratio)
  - Internal preview: resolution info card rendered via PIL
  - OUTPUT_NODE = True for built-in preview display
  - LATENT-only output (no IMAGE output)

v3.0 — Full rewrite:
  - Old facade imported .image.resolution which didn't exist → ImportError
  - Now self-contained, no external dependencies beyond torch/PIL
  - Internal preview instead of ComfyUI default preview
  - Output is LATENT only (was IMAGE in some versions)
  - 8-pixel alignment enforced (VAE requires dimensions divisible by 8)
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import os
import math
import logging
from typing import Dict, Any, Tuple

import folder_paths

logger = logging.getLogger("radiance.resolution")


# ═══════════════════════════════════════════════════════════════════════════════
#                         RESOLUTION DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

# Format: (width, height, category, aspect_ratio_label)
PRESETS: Dict[str, Tuple[int, int, str, str]] = {
    # ── Cinema / Film ──
    "4K DCI (4096×2160)": (4096, 2160, "Cinema", "1.90:1"),
    "4K UHD (3840×2160)": (3840, 2160, "Cinema", "16:9"),
    "2K DCI (2048×1080)": (2048, 1080, "Cinema", "1.90:1"),
    "HD 1080p (1920×1080)": (1920, 1080, "Cinema", "16:9"),
    "HD 720p (1280×720)": (1280, 720, "Cinema", "16:9"),
    "Anamorphic 2.39:1 (2048×856)": (2048, 856, "Cinema", "2.39:1"),
    "Anamorphic 2.39:1 (4096×1712)": (4096, 1712, "Cinema", "2.39:1"),
    "Super 35 (2048×1552)": (2048, 1552, "Cinema", "1.32:1"),
    "IMAX (5616×4096)": (5616, 4096, "Cinema", "1.37:1"),
    "Academy 4:3 (1440×1080)": (1440, 1080, "Cinema", "4:3"),
    "VistaVision (3072×2048)": (3072, 2048, "Cinema", "3:2"),
    "8K UHD (7680×4320)": (7680, 4320, "Cinema", "16:9"),
    # ── Social / Delivery ──
    "Instagram Square (1080×1080)": (1080, 1080, "Social", "1:1"),
    "Instagram Story (1080×1920)": (1080, 1920, "Social", "9:16"),
    "YouTube Thumb (1280×720)": (1280, 720, "Social", "16:9"),
    "TikTok (1080×1920)": (1080, 1920, "Social", "9:16"),
    # ── Flux (1 megapixel target) ──
    "Flux Square (1024×1024)": (1024, 1024, "Flux", "1:1"),
    "Flux 16:9 (1360×768)": (1360, 768, "Flux", "16:9"),
    "Flux 9:16 (768×1360)": (768, 1360, "Flux", "9:16"),
    "Flux 3:2 (1256×832)": (1256, 832, "Flux", "3:2"),
    "Flux 2:3 (832×1256)": (832, 1256, "Flux", "2:3"),
    "Flux 21:9 (1536×656)": (1536, 656, "Flux", "21:9"),
    "Flux 4:3 (1184×888)": (1184, 888, "Flux", "4:3"),
    "Flux 2.39:1 (1568×656)": (1568, 656, "Flux", "2.39:1"),
    # ── SDXL (1 megapixel target) ──
    "SDXL Square (1024×1024)": (1024, 1024, "SDXL", "1:1"),
    "SDXL 16:9 (1216×832)": (1216, 832, "SDXL", "3:2"),
    "SDXL 9:16 (832×1216)": (832, 1216, "SDXL", "2:3"),
    "SDXL 4:3 (1152×896)": (1152, 896, "SDXL", "9:7"),
    "SDXL 3:4 (896×1152)": (896, 1152, "SDXL", "7:9"),
    # ── SD 1.5 ──
    "SD 1.5 Square (512×512)": (512, 512, "SD 1.5", "1:1"),
    "SD 1.5 Wide (768×512)": (768, 512, "SD 1.5", "3:2"),
    "SD 1.5 Tall (512×768)": (512, 768, "SD 1.5", "2:3"),
}

PRESET_NAMES = ["Custom"] + list(PRESETS.keys())

MODEL_TYPES = ["Auto (Flux 16ch)", "Flux / SD3 (16ch)", "SDXL / SD 1.5 (4ch)"]

ORIENTATIONS = ["As Preset", "Landscape", "Portrait", "Square"]

# Latent channels per model type
LATENT_CHANNELS = {
    "Auto (Flux 16ch)": 16,
    "Flux / SD3 (16ch)": 16,
    "SDXL / SD 1.5 (4ch)": 4,
}

LATENT_SCALE = 8  # VAE downscale factor


# ═══════════════════════════════════════════════════════════════════════════════
#                         HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _align8(val: int) -> int:
    """Round to nearest multiple of 8 (VAE requirement)."""
    return max(8, (val + 4) // 8 * 8)


def _apply_orientation(w: int, h: int, orientation: str) -> Tuple[int, int]:
    """Apply orientation override."""
    if orientation == "Landscape":
        return (max(w, h), min(w, h))
    elif orientation == "Portrait":
        return (min(w, h), max(w, h))
    elif orientation == "Square":
        side = max(w, h)
        return (side, side)
    return (w, h)  # "As Preset"


def _gcd_ratio(w: int, h: int) -> str:
    """Calculate simplified aspect ratio string."""
    g = math.gcd(w, h)
    rw, rh = w // g, h // g
    # Simplify common large ratios
    if rw > 50 or rh > 50:
        ratio = w / h
        # Check common cinema ratios
        for name, val in [
            ("1:1", 1.0),
            ("4:3", 4 / 3),
            ("3:2", 3 / 2),
            ("16:9", 16 / 9),
            ("21:9", 21 / 9),
            ("2.39:1", 2.39),
            ("1.85:1", 1.85),
            ("1.90:1", 1.9),
            ("1.37:1", 1.37),
            ("9:16", 9 / 16),
            ("2:3", 2 / 3),
            ("3:4", 3 / 4),
        ]:
            if abs(ratio - val) < 0.02:
                return name
        return f"{ratio:.2f}:1"
    return f"{rw}:{rh}"


def _render_preview_card(
    width: int,
    height: int,
    preset_name: str,
    model_type: str,
    latent_c: int,
    batch_size: int,
) -> "PIL.Image.Image":
    """
    Render a resolution info card as a PIL Image.

    Layout:
    ┌──────────────────────────────────────────────┐
    │                                              │
    │     ┌──────────────────────┐                 │
    │     │                      │                 │
    │     │   aspect ratio box   │                 │
    │     │   with crosshair     │                 │
    │     │                      │                 │
    │     └──────────────────────┘                 │
    │                                              │
    │     RESOLUTION    1920 × 1080                │
    │     ASPECT RATIO  16:9                       │
    │     MEGAPIXELS    2.07 MP                    │
    │     LATENT        240 × 135 × 4ch           │
    │     PRESET        HD 1080p                   │
    │     BATCH         1                          │
    │                                              │
    └──────────────────────────────────────────────┘
    """
    from PIL import Image, ImageDraw

    # Card dimensions (fixed size for consistent display)
    card_w, card_h = 512, 512
    bg_color = (24, 24, 28)  # Flame dark
    box_border = (180, 120, 50)  # Radiance orange
    box_fill = (32, 32, 36)  # Slightly lighter
    text_bright = (220, 220, 220)
    text_dim = (140, 140, 140)
    text_accent = (220, 150, 60)  # Orange accent
    cross_color = (80, 80, 85)  # Crosshair

    img = Image.new("RGB", (card_w, card_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Try to load a monospace font, fall back to default
    font_small = None
    font_label = None
    font_title = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                from PIL import ImageFont as IF

                font_title = IF.truetype(fp, 18)
                IF.truetype(fp, 24)
                font_small = IF.truetype(fp, 13)
                font_label = IF.truetype(fp, 13)
                break
            except Exception:  # nosec B112
                continue

    # ── Aspect ratio box ──
    box_margin = 50
    box_area_w = card_w - box_margin * 2
    box_area_h = 180  # Max height for the AR box

    # Scale to fit
    ar = width / height
    if ar >= 1.0:
        bw = box_area_w
        bh = int(bw / ar)
        if bh > box_area_h:
            bh = box_area_h
            bw = int(bh * ar)
    else:
        bh = box_area_h
        bw = int(bh * ar)
        if bw > box_area_w:
            bw = box_area_w
            bh = int(bw / ar)

    box_x = (card_w - bw) // 2
    box_y = 50

    # Draw box
    draw.rectangle(
        [box_x, box_y, box_x + bw, box_y + bh],
        fill=box_fill,
        outline=box_border,
        width=2,
    )

    # Crosshair
    cx, cy = box_x + bw // 2, box_y + bh // 2
    draw.line([box_x + 4, cy, box_x + bw - 4, cy], fill=cross_color, width=1)
    draw.line([cx, box_y + 4, cx, box_y + bh - 4], fill=cross_color, width=1)

    # Thirds grid
    t3_x1, t3_x2 = box_x + bw // 3, box_x + 2 * bw // 3
    t3_y1, t3_y2 = box_y + bh // 3, box_y + 2 * bh // 3
    for x in [t3_x1, t3_x2]:
        draw.line([x, box_y + 2, x, box_y + bh - 2], fill=(50, 50, 55), width=1)
    for y in [t3_y1, t3_y2]:
        draw.line([box_x + 2, y, box_x + bw - 2, y], fill=(50, 50, 55), width=1)

    # Center dot
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=box_border)

    # Dimension labels — width below box, height to the right
    dim_text = f"{width} × {height}"
    draw.text(
        (cx, box_y + bh + 12), dim_text, fill=text_accent, font=font_small, anchor="mt"
    )

    # ── Info section ──
    info_y = box_y + bh + 35
    left_col = 50  # Label x
    right_col = 220  # Value x
    line_h = 24

    aspect_str = _gcd_ratio(width, height)
    megapixels = (width * height) / 1_000_000
    lat_w = width // LATENT_SCALE
    lat_h = height // LATENT_SCALE

    rows = [
        ("RESOLUTION", f"{width} × {height}"),
        ("ASPECT RATIO", aspect_str),
        ("MEGAPIXELS", f"{megapixels:.2f} MP"),
        ("LATENT", f"{lat_w} × {lat_h} × {latent_c}ch"),
        ("PRESET", preset_name if preset_name != "Custom" else "Custom"),
        ("MODEL", model_type.split("(")[0].strip()),
        ("BATCH", str(batch_size)),
    ]

    for i, (label, value) in enumerate(rows):
        y = info_y + i * line_h
        draw.text((left_col, y), label, fill=text_dim, font=font_label)
        draw.text((right_col, y), value, fill=text_bright, font=font_label)

    # ── Title bar ──
    draw.text(
        (card_w // 2, 16),
        "RADIANCE RESOLUTION",
        fill=text_accent,
        font=font_title,
        anchor="mt",
    )

    # ── Bottom border line ──
    draw.line([20, card_h - 20, card_w - 20, card_h - 20], fill=(50, 50, 55), width=1)

    # ── Footer ──
    footer = f"{width}×{height}  {aspect_str}  {megapixels:.1f}MP  lat:{lat_w}×{lat_h}×{latent_c}"
    draw.text(
        (card_w // 2, card_h - 10), footer, fill=text_dim, font=font_small, anchor="mb"
    )

    return img


# ═══════════════════════════════════════════════════════════════════════════════
#                        NODE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceResolution:
    """
    Professional resolution selector with internal preview.

    Outputs an empty LATENT at the selected resolution with correct
    channel count for your model (4ch for SD/SDXL, 16ch for Flux/SD3).

    The internal preview shows a resolution info card with aspect ratio
    visualization, dimensions, megapixel count, and latent info.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset": (
                    PRESET_NAMES,
                    {
                        "default": "Flux Square (1024×1024)",
                        "tooltip": (
                            "Resolution preset. Cinema, Social, Flux, SDXL, SD 1.5 presets available. "
                            "Select 'Custom' to use manual width/height."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Custom width (only used when preset is 'Custom'). Auto-aligned to 8px.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Custom height (only used when preset is 'Custom'). Auto-aligned to 8px.",
                    },
                ),
                "orientation": (
                    ORIENTATIONS,
                    {
                        "default": "As Preset",
                        "tooltip": "Override orientation. 'As Preset' uses the preset's native orientation.",
                    },
                ),
                "model_type": (
                    MODEL_TYPES,
                    {
                        "default": "Auto (Flux 16ch)",
                        "tooltip": (
                            "Determines latent channel count. "
                            "Flux/SD3 = 16 channels. SDXL/SD 1.5 = 4 channels."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Number of latent frames in batch.",
                    },
                ),
            },
            "optional": {
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 4.0,
                        "step": 0.25,
                        "tooltip": (
                            "Scale the resolution by this factor. "
                            "0.5 = half res, 2.0 = double res. Applied after preset/custom."
                        ),
                    },
                ),
                "latent_channels": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "Override latent channel count. 0 = use model_type default. "
                            "Common: 4 (SD/SDXL), 16 (Flux/SD3). "
                            "Set manually for custom architectures or experimentation."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "channels", "info")
    OUTPUT_TOOLTIPS = (
        "Empty latent tensor at the selected resolution.",
        "Final image width (pixels).",
        "Final image height (pixels).",
        "Latent channel count.",
        "Resolution info string.",
    )
    FUNCTION = "generate"
    CATEGORY = "FXTD Studios/Radiance/Image"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Professional resolution selector with internal preview card. "
        "Outputs empty LATENT for Flux/SDXL/SD with correct channel count. "
        "30+ cinema/social/AI presets. Manual latent_channels override for custom architectures."
    )

    def generate(
        self,
        preset: str,
        width: int,
        height: int,
        orientation: str,
        model_type: str,
        batch_size: int,
        scale_factor: float = 1.0,
        latent_channels: int = 0,
    ) -> Dict[str, Any]:

        # ── Resolve resolution ──
        if preset != "Custom" and preset in PRESETS:
            w, h, category, ar_label = PRESETS[preset]
        else:
            w, h = width, height
            _gcd_ratio(w, h)

        # Apply scale factor
        if scale_factor != 1.0:
            w = int(w * scale_factor)
            h = int(h * scale_factor)

        # Apply orientation
        w, h = _apply_orientation(w, h, orientation)

        # Align to 8px (VAE requirement)
        w = _align8(w)
        h = _align8(h)

        # ── Latent channels ──
        # Manual override takes priority; 0 = use model_type default
        if latent_channels > 0:
            latent_c = latent_channels
        else:
            latent_c = LATENT_CHANNELS.get(model_type, 16)

        # ── Create empty latent ──
        lat_h = h // LATENT_SCALE
        lat_w = w // LATENT_SCALE
        latent = torch.zeros(batch_size, latent_c, lat_h, lat_w, dtype=torch.float32)
        latent_dict = {"samples": latent}

        # ── Build info string ──
        megapixels = (w * h) / 1_000_000
        ar_str = _gcd_ratio(w, h)
        ch_src = "manual" if latent_channels > 0 else model_type.split("(")[0].strip()
        info = (
            f"{w}×{h} ({ar_str}) {megapixels:.2f}MP | "
            f"Latent: {lat_w}×{lat_h}×{latent_c}ch ({ch_src}) | "
            f"Batch: {batch_size}"
        )

        logger.info(f"[RadianceResolution] {info}")

        # ── Render internal preview card ──
        preview_images = []
        try:
            display_model = (
                f"Manual ({latent_c}ch)" if latent_channels > 0 else model_type
            )
            preview_img = _render_preview_card(
                width=w,
                height=h,
                preset_name=preset,
                model_type=display_model,
                latent_c=latent_c,
                batch_size=batch_size,
            )

            # Save to ComfyUI temp directory
            output_dir = folder_paths.get_temp_directory()
            import uuid

            preview_filename = f"radiance_resolution_{uuid.uuid4().hex[:8]}.png"
            preview_path = os.path.join(output_dir, preview_filename)

            preview_img.save(preview_path, "PNG")

            preview_images.append(
                {
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp",
                }
            )

            logger.debug(f"[RadianceResolution] Preview saved: {preview_path}")

        except Exception as e:
            logger.warning(f"[RadianceResolution] Preview render failed: {e}")

        # ── Return with UI preview ──
        return {
            "ui": {
                "images": preview_images,
            },
            "result": (latent_dict, w, h, latent_c, info),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                        NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceResolution": RadianceResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceResolution": "◎ Radiance Resolution",
}
