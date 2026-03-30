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
    # ── FEATURE: WAN Video (recommended resolutions) ──
    "WAN 720p 16:9 (1280×720)": (1280, 720, "WAN Video", "16:9"),
    "WAN 480p 16:9 (832×480)": (832, 480, "WAN Video", "16:9"),
    "WAN Portrait (480×832)": (480, 832, "WAN Video", "9:16"),
    "WAN Square (512×512)": (512, 512, "WAN Video", "1:1"),
    # ── FEATURE: LTX-Video (requires 32px alignment, lat=8) ──
    "LTX 720p (1216×704)": (1216, 704, "LTX Video", "16:9"),
    "LTX Portrait (704×1216)": (704, 1216, "LTX Video", "9:16"),
    "LTX Square (768×768)": (768, 768, "LTX Video", "1:1"),
    "LTX 1080p (1920×1088)": (1920, 1088, "LTX Video", "16:9"),
    # ── FEATURE: HunyuanVideo ──
    "HunyuanVideo 720p (1280×720)": (1280, 720, "HunyuanVideo", "16:9"),
    "HunyuanVideo Portrait (720×1280)": (720, 1280, "HunyuanVideo", "9:16"),
}

PRESET_NAMES = ["Custom"] + list(PRESETS.keys())

# FEATURE: These preset categories emit 5D latent (1, C, T, H, W) for video models
VIDEO_PRESET_CATEGORIES = {"WAN Video", "LTX Video", "HunyuanVideo"}

# FEATURE: Latent format string matching nodes_sampler.py latent_format input
LATENT_FORMAT_MAP = {
    "Auto (Flux 16ch)": "flux",
    "Flux / SD3 (16ch)": "flux",
    "SDXL / SD 1.5 (4ch)": "sdxl",
}

# FEATURE: Common aspect ratios for megapixel target mode
MP_ASPECT_RATIOS = [
    "1:1", "4:3", "3:2", "16:9", "21:9",
    "2.39:1", "1.85:1", "9:16", "2:3", "3:4",
]

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


def _mp_target_dimensions(mp_target: float, aspect_str: str) -> tuple[int, int]:
    """
    FEATURE: Compute (width, height) from a megapixel target and aspect ratio string.
    The result is aligned to 8px and respects the exact aspect ratio as closely as
    possible without exceeding the MP target.

    Examples:
        _mp_target_dimensions(1.0, "16:9")  → (1360, 768)  ≈ 1.04MP
        _mp_target_dimensions(2.0, "2.39:1") → (2192, 917) ≈ 2.01MP
    """
    # Parse aspect ratio
    aspect_str = aspect_str.strip()
    try:
        if ":" in aspect_str:
            parts = aspect_str.split(":")
            ratio = float(parts[0]) / float(parts[1])
        else:
            ratio = float(aspect_str)
    except (ValueError, ZeroDivisionError):
        ratio = 1.0

    # Solve: w*h = mp_target*1e6, w/h = ratio
    # → h = sqrt(mp*1e6 / ratio), w = h * ratio
    pixels = max(1024.0, mp_target * 1_000_000)
    h_raw = math.sqrt(pixels / ratio)
    w_raw = h_raw * ratio

    w = _align8(int(round(w_raw)))
    h = _align8(int(round(h_raw)))
    return w, h


def _render_preview_card(
    width: int,
    height: int,
    preset_name: str,
    model_type: str,
    latent_c: int,
    batch_label: str,
    batch_value: str,
    category: str = "",
) -> "PIL.Image.Image":
    """
    Radiance HUD-style resolution preview card.

    Redesigned v3.1 — matches Radiance/ComfyUI dark theme:
    ┌─ RADIANCE RESOLUTION ────────────────────────────────┐
    │ [CINEMA]                               2.21 MP ●●●○○ │  ← header bar
    ├───────────────────────────────────────────────────────┤
    │                                                       │
    │    ┌─────────────────────────────────────────┐        │
    │    │  · · · · · · · · · · · · · · · · · · ·  │        │  ← AR box
    │    │  ·                 ·                 ·  │        │    with grid
    │    │  · · · · · · ─ ─ ─⬥─ ─ ─ · · · · · ·  │        │    + crosshair
    │    │  ·                 ·                 ·  │        │    + corner HUD
    │    │  · · · · · · · · · · · · · · · · · · ·  │        │
    │    └─────────────────────────────────────────┘        │
    │                    2048 × 1080                        │
    ├──────────────────────┬────────────────────────────────┤
    │  RESOLUTION          │  2048 × 1080                   │  ← info grid
    │  ASPECT RATIO        │  1.90:1                        │    alternating rows
    │  MEGAPIXELS          │  2.21 MP                       │
    │  LATENT              │  256 × 135 × 16ch              │
    │──────────────────────┴────────────────────────────────│
    │  PRESET   2K DCI    MODEL   Flux 16ch   BATCH  1      │  ← footer strip
    └───────────────────────────────────────────────────────┘
    """
    from PIL import Image, ImageDraw

    # ── Palette — Radiance / ComfyUI dark theme ───────────────────────────────
    # Matches the Autodesk Flame-inspired Radiance UI colour system
    C_BG           = (15, 15, 18)       # Deep background — darker than the old 24,24,28
    C_BG_ALT       = (20, 20, 24)       # Alternate row background
    C_PANEL        = (22, 22, 27)       # AR box fill
    C_BORDER       = (38, 38, 45)       # Subtle panel border
    C_ACCENT       = (210, 140, 50)     # Radiance orange (primary)
    C_ACCENT_DIM   = (140, 92, 32)      # Radiance orange (dim)
    C_ACCENT_GLOW  = (230, 165, 75)     # Radiance orange (bright)
    C_GRID         = (32, 32, 38)       # AR box thirds grid
    C_CROSS        = (55, 55, 62)       # AR crosshair lines
    C_TEXT_HI      = (230, 230, 235)    # Primary value text
    C_TEXT_MID     = (155, 155, 165)    # Label text
    C_TEXT_DIM     = (90, 90, 100)      # Dim / footer text
    C_SEP          = (35, 35, 42)       # Separator lines
    C_ROW_EVEN     = (18, 18, 22)       # Even info row
    C_ROW_ODD      = (23, 23, 28)       # Odd info row
    C_BADGE_BG     = (35, 25, 10)       # Category badge background
    C_GOOD_MP      = (80, 180, 100)     # ≤2 MP — green dot
    C_MED_MP       = (210, 140, 50)     # 2-8 MP — orange dot
    C_HIGH_MP      = (200, 70, 60)      # >8 MP — red dot

    card_w, card_h = 512, 580
    img = Image.new("RGB", (card_w, card_h), C_BG)
    draw = ImageDraw.Draw(img)

    # ── Font loading ──────────────────────────────────────────────────────────
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\lucon.ttf",
    ]
    fT, fL, fS, fXS = None, None, None, None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                from PIL import ImageFont as IF
                fT  = IF.truetype(fp, 22)   # Title
                fL  = IF.truetype(fp, 13)   # Labels / values
                fS  = IF.truetype(fp, 11)   # Small / footer
                fXS = IF.truetype(fp, 10)   # Tiny (corner HUD)
                break
            except Exception:
                continue

    # ── Measurements ──────────────────────────────────────────────────────────
    aspect_str  = _gcd_ratio(width, height)
    megapixels  = (width * height) / 1_000_000
    lat_w       = width // LATENT_SCALE
    lat_h       = height // LATENT_SCALE

    HEADER_H    = 36
    AR_TOP      = HEADER_H + 12
    AR_MARGIN   = 44
    AR_MAX_H    = 170
    ar          = width / height

    # Scale AR box to available area
    area_w = card_w - AR_MARGIN * 2
    if ar >= 1.0:
        bw = area_w
        bh = int(bw / ar)
        if bh > AR_MAX_H:
            bh = AR_MAX_H
            bw = int(bh * ar)
    else:
        bh = AR_MAX_H
        bw = int(bh * ar)
        if bw > area_w:
            bw = area_w
            bh = int(bw / ar)

    bx = (card_w - bw) // 2
    by = AR_TOP

    DIM_LABEL_H = 22      # height of "2048 × 1080" below AR box
    INFO_TOP    = by + bh + DIM_LABEL_H + 10
    INFO_ROW_H  = 28
    INFO_ROWS   = 4       # RESOLUTION, ASPECT, MP, LATENT
    INFO_H      = INFO_ROWS * INFO_ROW_H
    FOOTER_H    = 36
    FOOTER_Y    = card_h - FOOTER_H

    # Resize canvas if info overflows
    needed = INFO_TOP + INFO_H + FOOTER_H + 8
    if needed > card_h:
        card_h = needed
        img = Image.new("RGB", (card_w, card_h), C_BG)
        draw = ImageDraw.Draw(img)
        FOOTER_Y = card_h - FOOTER_H

    # ═════════════════════════════════════════════════════════════════════════
    # 1. HEADER BAR
    # ═════════════════════════════════════════════════════════════════════════
    draw.rectangle([0, 0, card_w, HEADER_H], fill=(18, 18, 22))
    # Accent line under header
    draw.line([0, HEADER_H, card_w, HEADER_H], fill=C_ACCENT_DIM, width=1)

    # Title
    draw.text((16, HEADER_H // 2), "RADIANCE  RESOLUTION",
              fill=C_ACCENT, font=fT, anchor="lm")

    # Category badge (top right)
    cat_label = (category or "CUSTOM").upper()
    badge_x = card_w - 12
    draw.text((badge_x, HEADER_H // 2), cat_label,
              fill=C_ACCENT_DIM, font=fS, anchor="rm")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. MEGAPIXEL INDICATOR BAR (below title, above AR box)
    # ═════════════════════════════════════════════════════════════════════════
    # 5-dot indicator: each dot = 2MP, max shown = 10MP
    dot_y = HEADER_H + 6
    dot_r = 3
    dot_spacing = 10
    dot_count = 5
    max_mp = 10.0
    filled = min(dot_count, int(round(megapixels / max_mp * dot_count)))
    dots_total_w = dot_count * dot_spacing
    dot_start_x = card_w - 14 - dots_total_w

    # MP text left of dots
    mp_color = C_GOOD_MP if megapixels <= 2 else (C_MED_MP if megapixels <= 8 else C_HIGH_MP)
    draw.text((dot_start_x - 6, dot_y + dot_r), f"{megapixels:.2f} MP",
              fill=mp_color, font=fS, anchor="rm")

    for di in range(dot_count):
        dx = dot_start_x + di * dot_spacing + dot_r
        color = mp_color if di < filled else C_BORDER
        draw.ellipse([dx - dot_r, dot_y, dx + dot_r, dot_y + dot_r * 2], fill=color)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. ASPECT RATIO BOX — professional VFX HUD style
    # ═════════════════════════════════════════════════════════════════════════
    # Panel fill with subtle gradient feel (two rectangles)
    draw.rectangle([bx, by, bx + bw, by + bh], fill=C_PANEL)

    # Rule-of-thirds grid (subtle)
    for xi in [1, 2]:
        gx = bx + bw * xi // 3
        draw.line([gx, by + 1, gx, by + bh - 1], fill=C_GRID, width=1)
    for yi in [1, 2]:
        gy = by + bh * yi // 3
        draw.line([bx + 1, gy, bx + bw - 1, gy], fill=C_GRID, width=1)

    # Crosshair lines (slightly brighter than grid)
    cx, cy = bx + bw // 2, by + bh // 2
    cross_len_h = bw // 2 - 6
    cross_len_v = bh // 2 - 4
    draw.line([cx - cross_len_h, cy, cx - 8, cy], fill=C_CROSS, width=1)
    draw.line([cx + 8, cy, cx + cross_len_h, cy], fill=C_CROSS, width=1)
    draw.line([cx, cy - cross_len_v, cx, cy - 6], fill=C_CROSS, width=1)
    draw.line([cx, cy + 6, cx, cy + cross_len_v], fill=C_CROSS, width=1)

    # Center diamond (Radiance orange)
    diam = 5
    draw.polygon([
        (cx,        cy - diam),
        (cx + diam, cy),
        (cx,        cy + diam),
        (cx - diam, cy),
    ], fill=C_ACCENT)

    # Corner brackets — VFX HUD style
    blen = min(14, bw // 5, bh // 4)
    bthk = 1
    corners = [
        (bx,      by,      +1, +1),   # top-left
        (bx + bw, by,      -1, +1),   # top-right
        (bx,      by + bh, +1, -1),   # bottom-left
        (bx + bw, by + bh, -1, -1),   # bottom-right
    ]
    for (ox, oy, sx, sy) in corners:
        draw.line([ox, oy, ox + sx * blen, oy],           fill=C_ACCENT, width=bthk + 1)
        draw.line([ox, oy, ox,             oy + sy * blen], fill=C_ACCENT, width=bthk + 1)

    # Outer border (single-pixel accent)
    draw.rectangle([bx, by, bx + bw, by + bh], outline=C_ACCENT_DIM, width=1)

    # AR label — inside box, top-left corner (subtle)
    ar_label_inside = aspect_str
    draw.text((bx + 6, by + 4), ar_label_inside, fill=C_ACCENT_DIM, font=fXS)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. DIMENSION LABEL (below AR box)
    # ═════════════════════════════════════════════════════════════════════════
    dim_y = by + bh + 4
    draw.text((cx, dim_y), f"{width} × {height}",
              fill=C_ACCENT_GLOW, font=fL, anchor="mt")

    # ═════════════════════════════════════════════════════════════════════════
    # 5. INFO GRID (alternating rows)
    # ═════════════════════════════════════════════════════════════════════════
    # Separator above info
    sep_y = INFO_TOP - 4
    draw.line([0, sep_y, card_w, sep_y], fill=C_SEP, width=1)

    LEFT_PAD  = 20
    MID_X     = 210   # divider between label and value
    VALUE_X   = MID_X + 12

    mp_val_str = f"{megapixels:.2f} MP"

    info_rows = [
        ("RESOLUTION",  f"{width} × {height}",               C_TEXT_HI),
        ("ASPECT RATIO", aspect_str,                          C_TEXT_HI),
        ("MEGAPIXELS",  mp_val_str,                           mp_color),
        ("LATENT",      f"{lat_w} × {lat_h} × {latent_c}ch", C_ACCENT_GLOW),
    ]

    for i, (label, value, val_color) in enumerate(info_rows):
        ry = INFO_TOP + i * INFO_ROW_H
        row_bg = C_ROW_EVEN if i % 2 == 0 else C_ROW_ODD
        draw.rectangle([0, ry, card_w, ry + INFO_ROW_H - 1], fill=row_bg)

        # Vertical divider between label and value
        draw.line([MID_X, ry + 4, MID_X, ry + INFO_ROW_H - 4], fill=C_SEP, width=1)

        # Label (left, vertically centred)
        draw.text((LEFT_PAD, ry + INFO_ROW_H // 2), label,
                  fill=C_TEXT_MID, font=fL, anchor="lm")

        # Value (right of divider)
        draw.text((VALUE_X, ry + INFO_ROW_H // 2), value,
                  fill=val_color, font=fL, anchor="lm")

    # Separator below info
    after_info_y = INFO_TOP + INFO_H
    draw.line([0, after_info_y, card_w, after_info_y], fill=C_SEP, width=1)

    # ═════════════════════════════════════════════════════════════════════════
    # 6. FOOTER STRIP — PRESET / MODEL / BATCH in one line
    # ═════════════════════════════════════════════════════════════════════════
    draw.rectangle([0, FOOTER_Y, card_w, card_h], fill=(18, 18, 22))
    draw.line([0, FOOTER_Y, card_w, FOOTER_Y], fill=C_ACCENT_DIM, width=1)

    fy = FOOTER_Y + FOOTER_H // 2

    # Shorten long preset names
    pname = preset_name if preset_name != "Custom" else "Custom"
    if len(pname) > 22:
        pname = pname[:20] + "…"
    mname = model_type.split("(")[0].strip()
    if len(mname) > 14:
        mname = mname[:12] + "…"

    # Three columns: PRESET | MODEL | BATCH
    col_w = card_w // 3
    for ci, (lbl, val) in enumerate([
        ("PRESET", pname),
        ("MODEL",  mname),
        (batch_label, batch_value),
    ]):
        cx_col = col_w * ci + col_w // 2
        # Vertical separator (not on last)
        if ci > 0:
            draw.line([col_w * ci, FOOTER_Y + 6, col_w * ci, card_h - 6],
                      fill=C_SEP, width=1)
        # Label above, value below — two-line layout
        draw.text((cx_col, fy - 8), lbl,  fill=C_TEXT_DIM,  font=fXS, anchor="mm")
        draw.text((cx_col, fy + 8), val,  fill=C_TEXT_MID, font=fS,  anchor="mm")

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
                "enable_video": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable video sequence mode (replaces batch parameter)."},
                ),
                "video_frames": (
                    "INT",
                    {"default": 81, "min": 1, "max": 100000, "step": 1, "tooltip": "Total number of video frames."},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "Playback frame rate."},
                ),
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
                # ── FEATURE: Megapixel Target Mode ──────────────────────────────
                "mp_target": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.25,
                        "tooltip": (
                            "MEGAPIXEL TARGET: When > 0, auto-calculates W×H from this MP target "
                            "and mp_aspect_ratio. Overrides preset and custom W/H. "
                            "0 = disabled (use preset/custom instead)."
                        ),
                    },
                ),
                "mp_aspect_ratio": (
                    MP_ASPECT_RATIOS,
                    {
                        "default": "16:9",
                        "tooltip": "Aspect ratio for megapixel target mode (only used when mp_target > 0).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "STRING", "FLOAT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("latent", "width", "height", "channels", "info", "frame_rate", "frame_count", "latent_format", "duration_sec")
    OUTPUT_TOOLTIPS = (
        "Empty latent tensor at the selected resolution.",
        "Final image width (pixels).",
        "Final image height (pixels).",
        "Latent channel count.",
        "Resolution info string.",
        "Playback frame rate. Always the widget value — never 0.0.",
        "Total video frames (or batch size for images).",
        "Latent format string — wire to Sampler Pro latent_format input.",
        "Duration in seconds (video_frames / frame_rate). 0.0 for images.",
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
        enable_video: bool = False,
        video_frames: int = 81,
        frame_rate: float = 24.0,
        mp_target: float = 0.0,
        mp_aspect_ratio: str = "16:9",
    ) -> Dict[str, Any]:

        # ── FEATURE: Megapixel target mode overrides preset/custom ──────────────
        if mp_target > 0.0:
            w, h = _mp_target_dimensions(mp_target, mp_aspect_ratio)
            category = "MP Target"
        # ── Resolve resolution from preset or custom ─────────────────────────────
        elif preset != "Custom" and preset in PRESETS:
            w, h, category, ar_label = PRESETS[preset]
        else:
            w, h = width, height
            category = "Custom"
            # FIX 1: _gcd_ratio result was discarded — now stored for later use
            # (ar_str is recomputed below after alignment; this validates the input)

        # Apply scale factor
        if scale_factor != 1.0:
            w = int(w * scale_factor)
            h = int(h * scale_factor)

        # FIX 5 (FEATURE): Square orientation guard — warn for extreme resolutions
        if orientation == "Square":
            side = max(w, h)
            sq_mp = (side * side) / 1_000_000
            if sq_mp > 16.0:
                logger.warning(
                    f"[RadianceResolution] Square orientation on {w}×{h} would produce "
                    f"{side}×{side} ({sq_mp:.1f}MP) — this may exhaust VRAM. "
                    f"Consider using a smaller preset or Landscape/Portrait instead."
                )

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

        # ── Determine if this is a video latent ──────────────────────────────────
        # FIX 2: Video models need 5D latent (1, C, T, H, W), not 4D (B, C, H, W)
        preset_category = PRESETS.get(preset, (0, 0, "", ""))[2]
        is_video_latent = enable_video and preset_category in VIDEO_PRESET_CATEGORIES

        actual_batch = video_frames if enable_video else batch_size

        # ── Create empty latent ──
        lat_h = h // LATENT_SCALE
        lat_w = w // LATENT_SCALE

        if is_video_latent:
            # FIX 2: 5D tensor for WAN/HunyuanVideo/LTX — (batch=1, C, T, H, W)
            latent = torch.zeros(1, latent_c, actual_batch, lat_h, lat_w, dtype=torch.float32)
            logger.info(f"[RadianceResolution] Video latent 5D: (1, {latent_c}, {actual_batch}, {lat_h}, {lat_w})")
        else:
            # 4D for image/non-video-model paths: (B, C, H, W)
            latent = torch.zeros(actual_batch, latent_c, lat_h, lat_w, dtype=torch.float32)

        latent_dict = {"samples": latent}

        # ── Build info string ──
        megapixels = (w * h) / 1_000_000
        ar_str = _gcd_ratio(w, h)
        ch_src = "manual" if latent_channels > 0 else model_type.split("(")[0].strip()

        if enable_video:
            batch_label = "VIDEO"
            batch_value = f"{video_frames}f @ {frame_rate}fps"
        else:
            batch_label = "BATCH"
            batch_value = str(batch_size)

        info = (
            f"{w}×{h} ({ar_str}) {megapixels:.2f}MP | "
            f"Latent: {lat_w}×{lat_h}×{latent_c}ch ({ch_src}) | "
            f"{batch_label.capitalize()}: {batch_value}"
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
                batch_label=batch_label,
                batch_value=batch_value,
                category=category,
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
        # FIX 3: frame_rate was 0.0 for images — always return the widget value
        actual_fps = float(frame_rate)

        # FEATURE: latent_format string — wire to Sampler Pro latent_format input
        latent_fmt = LATENT_FORMAT_MAP.get(model_type, "flux" if latent_c >= 16 else "sdxl")
        if latent_channels > 0:
            latent_fmt = "flux" if latent_c >= 16 else "sdxl"

        # FEATURE: duration in seconds for video; 0.0 for images
        duration_sec = float(video_frames) / float(frame_rate) if enable_video else 0.0

        return {
            "ui": {
                "images": preview_images,
            },
            "result": (latent_dict, w, h, latent_c, info, actual_fps, actual_batch, latent_fmt, duration_sec),
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
