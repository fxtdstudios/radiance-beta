import torch
import os
import math
import logging
import uuid
import hashlib
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
    # BUG FIX: renamed misleading presets — actual ratios match names now
    "SDXL Square (1024×1024)": (1024, 1024, "SDXL", "1:1"),
    "SDXL 3:2 (1216×832)": (1216, 832, "SDXL", "3:2"),    # was mislabeled "16:9" — actual=1.46
    "SDXL 2:3 (832×1216)": (832, 1216, "SDXL", "2:3"),
    "SDXL 9:7 (1152×896)": (1152, 896, "SDXL", "9:7"),    # was mislabeled "4:3" — actual=1.29
    "SDXL 7:9 (896×1152)": (896, 1152, "SDXL", "7:9"),
    # ── SD 1.5 ──
    "SD 1.5 Square (512×512)": (512, 512, "SD 1.5", "1:1"),
    "SD 1.5 Wide (768×512)": (768, 512, "SD 1.5", "3:2"),
    "SD 1.5 Tall (512×768)": (512, 768, "SD 1.5", "2:3"),
    # ── WAN Video 1.x (recommended resolutions, 16px-aligned) ──
    "WAN 720p 16:9 (1280×720)": (1280, 720, "WAN Video", "16:9"),
    "WAN 480p 16:9 (832×480)": (832, 480, "WAN Video", "16:9"),
    "WAN Portrait (480×832)": (480, 832, "WAN Video", "9:16"),
    "WAN Square (512×512)": (512, 512, "WAN Video", "1:1"),
    # ── WAN 2.1 (updated recommended sizes) ──
    "WAN 2.1 720p (1280×720)": (1280, 720, "WAN 2.1", "16:9"),
    "WAN 2.1 480p (854×480)": (854, 480, "WAN 2.1", "16:9"),
    "WAN 2.1 Portrait 720p (720×1280)": (720, 1280, "WAN 2.1", "9:16"),
    "WAN 2.1 Portrait 480p (480×854)": (480, 854, "WAN 2.1", "9:16"),
    "WAN 2.1 Square (512×512)": (512, 512, "WAN 2.1", "1:1"),
    "WAN 2.1 1080p (1920×1080)": (1920, 1080, "WAN 2.1", "16:9"),
    # ── LTX-Video (requires 32px alignment, lat=8) ──
    "LTX 720p (1216×704)": (1216, 704, "LTX Video", "16:9"),
    "LTX Portrait (704×1216)": (704, 1216, "LTX Video", "9:16"),
    "LTX Square (768×768)": (768, 768, "LTX Video", "1:1"),
    "LTX 1080p (1920×1088)": (1920, 1088, "LTX Video", "16:9"),
    # ── HunyuanVideo (text-to-video) ──
    "HunyuanVideo 720p (1280×720)": (1280, 720, "HunyuanVideo", "16:9"),
    "HunyuanVideo Portrait (720×1280)": (720, 1280, "HunyuanVideo", "9:16"),
    # ── Hunyuan I2V (image-to-video — different resolution set) ──
    "Hunyuan I2V 720p (1280×720)": (1280, 720, "Hunyuan I2V", "16:9"),
    "Hunyuan I2V 480p (848×480)": (848, 480, "Hunyuan I2V", "16:9"),
    "Hunyuan I2V Portrait (720×1280)": (720, 1280, "Hunyuan I2V", "9:16"),
    # ── CogVideoX ──
    "CogVideoX 720p (720×480)": (720, 480, "CogVideoX", "3:2"),
    "CogVideoX 1080p (1280×720)": (1280, 720, "CogVideoX", "16:9"),
}

PRESET_NAMES = ["Custom"] + list(PRESETS.keys())

# These preset categories emit 5D latent (1, C, T, H, W) for video models
VIDEO_PRESET_CATEGORIES = {"WAN Video", "WAN 2.1", "LTX Video", "HunyuanVideo", "Hunyuan I2V", "CogVideoX"}

# Categories that require WAN's (4k+1) frame count rule
WAN_FRAME_CATEGORIES = {"WAN Video", "WAN 2.1"}

# Categories that require 32px pixel alignment instead of 8px
ALIGN32_CATEGORIES = {"LTX Video"}

# Latent format string matching nodes_sampler.py latent_format input
LATENT_FORMAT_MAP = {
    "Auto (Flux 16ch)": "flux",
    "Flux / SD3 (16ch)": "flux",
    "SDXL / SD 1.5 (4ch)": "sdxl",
    "Cosmos (16ch)": "flux",
    "FLUX.1-Kontext (16ch)": "flux",
    "CogVideoX (16ch)": "flux",
    "Mochi (12ch)": "flux",
}

# Latent format for video preset categories (overrides LATENT_FORMAT_MAP)
VIDEO_LATENT_FORMAT_MAP = {
    "WAN Video": "wan",
    "WAN 2.1": "wan",
    "LTX Video": "ltx",
    "HunyuanVideo": "hunyuan",
    "Hunyuan I2V": "hunyuan",
    "CogVideoX": "flux",
}

# Common aspect ratios for megapixel target mode
MP_ASPECT_RATIOS = [
    "1:1", "4:3", "3:2", "16:9", "21:9",
    "2.39:1", "1.85:1", "9:16", "2:3", "3:4",
]

MODEL_TYPES = [
    "Auto (Flux 16ch)",
    "Flux / SD3 (16ch)",
    "SDXL / SD 1.5 (4ch)",
    "Cosmos (16ch)",
    "FLUX.1-Kontext (16ch)",
    "CogVideoX (16ch)",
    "Mochi (12ch)",
]

ORIENTATIONS = ["As Preset", "Landscape", "Portrait", "Square"]

# Latent channels per model type
LATENT_CHANNELS = {
    "Auto (Flux 16ch)": 16,
    "Flux / SD3 (16ch)": 16,
    "SDXL / SD 1.5 (4ch)": 4,
    "Cosmos (16ch)": 16,
    "FLUX.1-Kontext (16ch)": 16,
    "CogVideoX (16ch)": 16,
    "Mochi (12ch)": 12,
}

# ── VRAM Estimation Metadata ──────────────────────────────────────────────────
# Bytes per latent element (ComfyUI usually uses float32 internally = 4 bytes)
LATENT_ELEMENT_BYTES = 4
# Typical VRAM overhead for a modern diffusion model pass (Geniune rough estimate in GB)
MODEL_BASE_VRAM = {
    "flux": 12.0,  # Flux is heavy
    "sdxl": 4.5,   # SDXL is medium
    "sd15": 2.5,   # SD 1.5 is light
    "wan":  14.0,  # Video models are very heavy
    "ltx":  10.0,
}

LATENT_SCALE = 8  # VAE downscale factor


# ═══════════════════════════════════════════════════════════════════════════════
#                         HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _align8(val: int) -> int:
    """Round to nearest multiple of 8 (VAE requirement)."""
    return max(8, (val + 4) // 8 * 8)


def _align32(val: int) -> int:
    """Round UP to nearest multiple of 32 (LTX Video requirement)."""
    return max(32, math.ceil(val / 32) * 32)


def _estimate_vram(w: int, h: int, c: int, b: int, format_key: str = "flux") -> float:
    """
    Estimate VRAM usage in Gigabytes.
    Includes latent tensor size + estimated model activation overhead.
    """
    # Latent dimensions (1/8th of pixel res)
    lw, lh = w // 8, h // 8
    # Tensor size in bytes
    latent_bytes = b * c * lw * lh * LATENT_ELEMENT_BYTES
    # Convert to GB
    latent_gb = latent_bytes / (1024**3)
    # Model overhead
    base_gb = MODEL_BASE_VRAM.get(format_key.lower(), 4.0)
    # Total
    return latent_gb + base_gb


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
    enable_video: bool = False,
    video_frames: int = 0,
    frame_rate: float = 24.0,
    vram_est: float = 0.0,
    align_label: str = "8px",
) -> "PIL.Image.Image":
    """
    Radiance HUD-style resolution preview card.

    Redesigned v3.6 — Compact HUD Update:
    ┌─ RADIANCE RESOLUTION ────────────────────────────────┐
    │ [CINEMA]                               2.21 MP ●●●○○ │  ← header bar
    ├───────────────────────────────────────────────────────┤
    │  VRAM PRESSURE: [||||||||      ]  12.4 GB    (ALIGNED)│  ← New VRAM bar
    │    ┌─────────────────────────────────────────┐        │
    │    │  · · · · · · · · · · · · · · · · · · ·  │        │  ← AR box
    │    │  · · · · · · ─ ─ ─⬥─ ─ ─ · · · · · ·  │        │
    │    └─────────────────────────────────────────┘        │
    │                    2048 × 1080                        │
    ├──────────────────────┬────────────────────────────────┤
    │  RESOLUTION          │  2048 × 1080  [32px]           │  ← align info
    │  ASPECT RATIO        │  1.90:1                        │
    │  EST. VRAM           │  12.42 GB                      │  ← VRAM info
    │  LATENT              │  256 × 135 × 16ch              │
    ├──────────────────────┴────────────────────────────────┤
    │  PRESET   2K DCI    MODEL   Flux 16ch   BATCH  1      │
    └───────────────────────────────────────────────────────┘
    """
    from PIL import Image, ImageDraw

    # ── Palette — Radiance / ComfyUI dark theme ───────────────────────────────
    # Matches the Autodesk Flame-inspired Radiance UI colour system
    # ── Obsidian Glass & Ocean Cyan Design Overhaul (Apple macOS Style) ──────
    C_BG           = (8, 8, 12)         # Deep Obsidian Black backing
    C_PANEL        = (22, 22, 29)       # Apple dark obsidian glass fill
    C_BORDER       = (45, 45, 55)       # Satin graphite border outline
    C_ACCENT       = (0, 189, 255)      # Premium Ocean Cyan accent
    C_ACCENT_DIM   = (0, 120, 170)      # Muted Ocean Cyan shadow
    C_ACCENT_GLOW  = (72, 183, 255)     # Glowing neon cyan keylights
    C_GRID         = (30, 32, 40)       # Rule-of-thirds grid
    C_CROSS        = (60, 65, 80)       # HUD centering crosshairs
    C_TEXT_HI      = (245, 245, 247)    # Crisp white display value
    C_TEXT_MID     = (160, 160, 168)    # Clean midtone labels
    C_TEXT_DIM     = (124, 124, 132)    # Subdued footers
    C_SEP          = (28, 28, 35)       # Fine graphite separators
    C_ROW_EVEN     = (14, 14, 20)       # Alternating even rows
    C_ROW_ODD      = (20, 20, 26)       # Alternating odd rows
    C_GOOD_MP      = (52, 199, 89)      # macOS green status dot (low density)
    C_MED_MP       = (255, 149, 0)      # macOS orange status dot (medium density)
    C_HIGH_MP      = (255, 59, 48)      # macOS red status dot (high density)
    C_VRAM_BAR     = (30, 30, 38)       # VRAM progress track

    card_w, card_h = 512, 406
    img = Image.new("RGB", (card_w, card_h), C_BG)
    draw = ImageDraw.Draw(img)

    # ── Font loading (cross-platform) ─────────────────────────────────────────
    import platform as _platform
    _plat = _platform.system()
    if _plat == "Windows":
        _wf = os.environ.get("WINDIR", "C:\\Windows")
        font_paths = [
            os.path.join(_wf, "Fonts", "consola.ttf"),
            os.path.join(_wf, "Fonts", "lucon.ttf"),
            os.path.join(_wf, "Fonts", "cour.ttf"),
            os.path.join(_wf, "Fonts", "arial.ttf"),
        ]
    elif _plat == "Darwin":
        font_paths = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/Library/Fonts/Courier New.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    fT, fL, fS, fXS = None, None, None, None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                from PIL import ImageFont as IF
                fT  = IF.truetype(fp, 20)   # Title
                fL  = IF.truetype(fp, 13)   # Labels / values
                fS  = IF.truetype(fp, 11)   # Small / footer
                fXS = IF.truetype(fp, 10)   # Tiny (corner HUD)
                break
            except Exception:
                continue
    if fT is None:
        logger.debug("No system font found — using PIL default.")
        from PIL import ImageFont
        fT = fL = fS = fXS = ImageFont.load_default()

    # ── Measurements ──────────────────────────────────────────────────────────
    aspect_str  = _gcd_ratio(width, height)
    megapixels  = (width * height) / 1_000_000
    lat_w       = width // LATENT_SCALE
    lat_h       = height // LATENT_SCALE

    HEADER_H    = 34
    STATUS_H    = 22
    AR_TOP      = HEADER_H + STATUS_H + 8
    AR_MARGIN   = 44
    AR_MAX_H    = 160
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

    DIM_LABEL_H = 20      # height of "2048 × 1080" below AR box
    INFO_TOP    = by + bh + DIM_LABEL_H + 8
    INFO_ROW_H  = 26
    INFO_ROWS   = 4       # RESOLUTION, ASPECT, MP, LATENT
    INFO_H      = INFO_ROWS * INFO_ROW_H
    FOOTER_H    = 34

    # The preview is content-driven. Older builds pinned the footer to a fixed
    # 580px canvas, which left a large empty lower screen in ComfyUI.
    card_h = INFO_TOP + INFO_H + FOOTER_H + 8
    img = Image.new("RGB", (card_w, card_h), C_BG)
    draw = ImageDraw.Draw(img)
    FOOTER_Y = card_h - FOOTER_H

    # ═════════════════════════════════════════════════════════════════════════
    # 1. HEADER BAR
    # ═════════════════════════════════════════════════════════════════════════
    draw.rectangle([0, 0, card_w, HEADER_H], fill=(22, 22, 29))
    # Elegant divider line under header
    draw.line([0, HEADER_H, card_w, HEADER_H], fill=C_SEP, width=1)

    # Title (brand-aligned ◎ Resolution)
    draw.text((16, HEADER_H // 2), "◎ RADIANCE RESOLUTION",
              fill=C_ACCENT, font=fT, anchor="lm")

    # Category badge (top right)
    cat_label = (category or "CUSTOM").upper()
    badge_x = card_w - 12
    draw.text((badge_x, HEADER_H // 2), cat_label,
              fill=C_ACCENT_DIM, font=fS, anchor="rm")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. MEGAPIXEL / VRAM STATUS STRIP (below title, above AR box)
    # ═════════════════════════════════════════════════════════════════════════
    # 5-dot indicator: each dot = 2MP, max shown = 10MP
    dot_y = HEADER_H + 5
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

    # ── VRAM PRESSURE BAR (New v3.5) ──────────────────────────────────────────
    vram_y = HEADER_H + 15
    vram_bar_w = 112
    vram_bar_h = 4
    vram_bar_x = 16
    
    # Label
    draw.text((vram_bar_x, vram_y - 2), "VRAM", fill=C_TEXT_DIM, font=fXS, anchor="lt")
    
    # Track (rounded ends)
    track_x = vram_bar_x + 42
    draw.rounded_rectangle([track_x, vram_y, track_x + vram_bar_w, vram_y + vram_bar_h], radius=2, fill=C_VRAM_BAR)
    
    # Fill based on 24GB max (RTX 3090/4090 standard)
    max_vram = 24.0
    fill_pct = min(1.0, vram_est / max_vram)
    fill_w = int(vram_bar_w * fill_pct)
    fill_color = C_GOOD_MP if vram_est < 12 else (C_MED_MP if vram_est < 20 else C_HIGH_MP)
    
    if fill_w > 0:
        draw.rounded_rectangle([track_x, vram_y, track_x + fill_w, vram_y + vram_bar_h], radius=2, fill=fill_color)
    
    # Value text
    draw.text((track_x + vram_bar_w + 10, vram_y - 2), f"{vram_est:.1f} GB", fill=fill_color, font=fXS, anchor="lt")
    
    # Alignment Badge (right)
    draw.text((card_w - 14, vram_y - 2), f"[{align_label.upper()} ALIGNED]", fill=C_TEXT_DIM, font=fXS, anchor="rt")

    # ═════════════════════════════════════════════════════════════════════════
    # 3. ASPECT RATIO BOX — professional VFX HUD style (macOS Squircled)
    # ═════════════════════════════════════════════════════════════════════════
    # Panel fill using beautiful rounded corner rectangles
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=8, fill=C_PANEL)

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

    # Center diamond (Ocean Cyan)
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

    # Outer border (graphite satin boundary outline matching macOS squircles)
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=8, outline=C_BORDER, width=1)

    # AR label — inside box, top-left corner (subtle)
    ar_label_inside = aspect_str
    draw.text((bx + 6, by + 4), ar_label_inside, fill=C_ACCENT_DIM, font=fXS)

    # Video badge — inside AR box, top-right corner
    if enable_video:
        duration = float(video_frames) / max(frame_rate, 1.0)
        vid_label = f"▶ {video_frames}f  {duration:.1f}s"
        draw.text((bx + bw - 6, by + 4), vid_label,
                  fill=(100, 200, 255), font=fXS, anchor="ra")

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

    info_rows = [
        ("RESOLUTION",  f"{width} × {height}  ({align_label})",  C_TEXT_HI),
        ("ASPECT RATIO", aspect_str,                          C_TEXT_HI),
        ("EST. VRAM",   f"{vram_est:.2f} GB",                 fill_color),
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
    draw.rectangle([0, FOOTER_Y, card_w, card_h], fill=(22, 22, 29))
    draw.line([0, FOOTER_Y, card_w, FOOTER_Y], fill=C_SEP, width=1)

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
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """
    Professional resolution selector with internal preview.

    Outputs an empty LATENT at the selected resolution with correct
    channel count for your model (4ch for SD/SDXL, 16ch for Flux/SD3).

    The internal preview shows a resolution info card with aspect ratio
    visualization, dimensions, megapixel count, and latent info.
    """

    # Per-node preview file tracking — keyed by unique_id.
    # Overwritten on each run so temp files don't accumulate.
    _preview_paths: Dict[str, str] = {}

    @classmethod
    def IS_CHANGED(
        cls,
        preset, width, height, orientation, model_type, batch_size,
        scale_factor=1.0, latent_channels=0, enable_video=False,
        video_frames=81, frame_rate=24.0, mp_target=0.0,
        mp_aspect_ratio="16:9", unique_id="",
    ):
        """Re-execute only when inputs actually change — avoids redundant renders."""
        state = (
            f"{preset}|{width}|{height}|{orientation}|{model_type}|{batch_size}|"
            f"{scale_factor}|{latent_channels}|{enable_video}|{video_frames}|"
            f"{frame_rate}|{mp_target}|{mp_aspect_ratio}"
        )
        return hashlib.md5(state.encode()).hexdigest()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset": (
                    PRESET_NAMES,
                    {
                        "default": "Flux Square (1024×1024)",
                        "tooltip": (
                            "Resolution preset. Cinema, Social, Flux, SDXL, SD 1.5, "
                            "WAN, WAN 2.1, LTX, HunyuanVideo, Hunyuan I2V, CogVideoX presets. "
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
                        "tooltip": "Custom width (only used when preset is 'Custom'). Auto-aligned to 8px (32px for LTX Video).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Custom height (only used when preset is 'Custom'). Auto-aligned to 8px (32px for LTX Video).",
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
                            "Flux/SD3/Cosmos/Kontext = 16ch. SDXL/SD 1.5 = 4ch. Mochi = 12ch."
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
                            "Scale the resolution by this factor after preset/custom. "
                            "0.5 = half res, 2.0 = double res. Applied before alignment."
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
                            "Common: 4 (SD/SDXL), 12 (Mochi), 16 (Flux/SD3/Cosmos). "
                            "Set manually for custom architectures."
                        ),
                    },
                ),
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
                            "0 = disabled."
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
                "enable_video": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable video sequence mode (replaces batch parameter)."},
                ),
                "video_frames": (
                    "INT",
                    {
                        "default": 81, "min": 1, "max": 100000, "step": 1,
                        "tooltip": (
                            "Total number of video frames. "
                            "WAN/WAN 2.1: must satisfy (4k+1) — e.g. 1, 5, 9, 13, 17, 21, 49, 81. "
                            "A warning is logged if this constraint is violated."
                        ),
                    },
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "Playback frame rate."},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = (
        "Empty latent tensor at the selected resolution.",
    )
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Professional resolution selector with internal preview card. "
        "Outputs empty LATENT for Flux/SDXL/SD/Cosmos/CogVideoX with correct channel count. "
        "40+ cinema/social/AI/video presets. WAN frame count validation. LTX 32px alignment. "
        "Manual latent_channels override for custom architectures."
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
        unique_id: str = "",
    ) -> Dict[str, Any]:

        # ── Megapixel target mode overrides preset/custom ────────────────────────
        if mp_target > 0.0:
            # BUG FIX: warn when mp_target produces a size too small for any model
            min_pixels = 256 * 256  # 65536px — floor below which no model works
            if mp_target * 1_000_000 < min_pixels:
                logger.warning(
                    f"mp_target={mp_target} MP produces < 256×256 pixels. "
                    f"Minimum recommended is 0.065 MP (256×256). Results may be unusable."
                )
            w, h = _mp_target_dimensions(mp_target, mp_aspect_ratio)
            category = "MP Target"
        # ── Resolve resolution from preset or custom ─────────────────────────────
        elif preset != "Custom" and preset in PRESETS:
            w, h, category, ar_label = PRESETS[preset]
        else:
            w, h = width, height
            category = "Custom"

        # Apply scale factor — log if it causes significant alignment correction
        if scale_factor != 1.0:
            w_pre = int(w * scale_factor)
            h_pre = int(h * scale_factor)
            if w_pre != w or h_pre != h:
                logger.debug(
                    f"Scale {scale_factor}×: {w}×{h} → {w_pre}×{h_pre} "
                    f"(before alignment)"
                )
            w, h = w_pre, h_pre

        # ── Step 2: Determine Alignment Rule (Adaptive v3.5) ─────────────────────
        align_val   = 8
        align_label = "8px"
        
        # LTX Video requires 32px
        if category in ALIGN32_CATEGORIES or "ltx" in model_type.lower():
            align_val   = 32
            align_label = "32px"
        # WAN Video often works better with 16px or its specific 4k+1 rule
        elif category in WAN_FRAME_CATEGORIES or "wan" in model_type.lower():
            align_val   = 16
            align_label = "16px"
            
        # Apply alignment
        if align_val == 32:
            w, h = _align32(w), _align32(h)
        else:
            # Standard align to whatever align_val is (usually 8 or 16)
            w = max(align_val, (w + align_val // 2) // align_val * align_val)
            h = max(align_val, (h + align_val // 2) // align_val * align_val)

        # ── Step 3: Latent Format & VRAM Estimation ──────────────────────────────
        latent_format = "sdxl"
        preset_category = PRESETS.get(preset, (0, 0, "", ""))[2]
        if enable_video and preset_category in VIDEO_LATENT_FORMAT_MAP:
            latent_format = VIDEO_LATENT_FORMAT_MAP[preset_category]
        else:
            # Fallback to model_type map
            for key, fmt in LATENT_FORMAT_MAP.items():
                if key == model_type:
                    latent_format = fmt
                    break
        
        # Estimate VRAM
        v_count = video_frames if enable_video else batch_size
        vram_est = _estimate_vram(w, h, latent_channels or LATENT_CHANNELS.get(model_type, 4), v_count, latent_format)

        # Apply orientation
        w, h = _apply_orientation(w, h, orientation)

        # ── Step 5: WAN frame count validation ──────────────────────────────────
        # WAN Video and WAN 2.1 require frame count = (4k + 1): 1, 5, 9, 13, 17...
        if enable_video and preset_category in WAN_FRAME_CATEGORIES:
            if (video_frames - 1) % 4 != 0:
                k_low  = (video_frames - 1) // 4
                v_low  = 4 * k_low + 1
                v_high = v_low + 4
                logger.warning(
                    f"{preset_category} requires frame count = 4k+1 "
                    f"(1, 5, 9, 13, 17, 21, 49, 81, 97...). Got {video_frames}. "
                    f"Nearest valid values: {v_low} or {v_high}. "
                    f"Using {video_frames} may cause sampler errors or incorrect output."
                )

        # ── Latent channels ──────────────────────────────────────────────────────
        if latent_channels > 0:
            latent_c = latent_channels
        else:
            latent_c = LATENT_CHANNELS.get(model_type, 16)

        # ── Determine if this is a video latent ──────────────────────────────────
        is_video_latent = enable_video and preset_category in VIDEO_PRESET_CATEGORIES
        actual_batch = video_frames if enable_video else batch_size

        # ── Create empty latent ──────────────────────────────────────────────────
        lat_h = h // LATENT_SCALE
        lat_w = w // LATENT_SCALE

        if is_video_latent:
            latent = torch.zeros(1, latent_c, actual_batch, lat_h, lat_w, dtype=torch.float32)
            logger.info(
                f"Video latent 5D: (1, {latent_c}, {actual_batch}, {lat_h}, {lat_w})"
            )
        else:
            latent = torch.zeros(actual_batch, latent_c, lat_h, lat_w, dtype=torch.float32)

        latent_dict = {"samples": latent}

        # ── Build info string ────────────────────────────────────────────────────
        megapixels = (w * h) / 1_000_000
        pixel_count = w * h
        ar_str = _gcd_ratio(w, h)
        ch_src = "manual" if latent_channels > 0 else model_type.split("(")[0].strip()

        if enable_video:
            batch_label = "VIDEO"
            batch_value = f"{video_frames}f @ {frame_rate}fps"
        else:
            batch_label = "BATCH"
            batch_value = str(batch_size)

        from radiance.core.logging import supports_unicode
        divider = "│" if supports_unicode() else "|"
        info = (
            f"{w}×{h} ({ar_str}) {megapixels:.2f}MP {divider} "
            f"Latent: {lat_w}×{lat_h}×{latent_c}ch ({ch_src}) {divider} "
            f"{batch_label.capitalize()}: {batch_value}"
        )
        logger.info(info)

        # ── Render internal preview card ─────────────────────────────────────────
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
                enable_video=enable_video,
                video_frames=video_frames,
                frame_rate=frame_rate,
                vram_est=vram_est,
                align_label=align_label,
            )

            output_dir = folder_paths.get_temp_directory()

            # BUG FIX: Use deterministic filename per node instance (unique_id).
            # Overwrites the previous preview for this node on each run —
            # no temp file accumulation. Falls back to uuid if unique_id not set.
            node_key = unique_id if unique_id else uuid.uuid4().hex[:8]
            preview_filename = f"radiance_resolution_{node_key}.png"
            preview_path = os.path.join(output_dir, preview_filename)

            # Remove old preview file if it exists (in case of uuid fallback path)
            _old = RadianceResolution._preview_paths.get(node_key)
            if _old and _old != preview_path and os.path.exists(_old):
                try:
                    os.remove(_old)
                except OSError:
                    pass
            RadianceResolution._preview_paths[node_key] = preview_path

            preview_img.save(preview_path, "PNG")
            preview_images.append({"filename": preview_filename, "subfolder": "", "type": "temp"})
            logger.debug(f"Preview saved: {preview_path}")

        except Exception as e:
            logger.warning(f"Preview render failed: {e}")

        # ── Latent format string ─────────────────────────────────────────────────
        # Video preset categories get their own format map; image models use model_type
        if is_video_latent:
            latent_fmt = VIDEO_LATENT_FORMAT_MAP.get(
                preset_category,
                "flux" if latent_c >= 16 else "sdxl"
            )
        else:
            latent_fmt = LATENT_FORMAT_MAP.get(model_type, "flux" if latent_c >= 16 else "sdxl")
        if latent_channels > 0 and not is_video_latent:
            latent_fmt = "flux" if latent_c >= 16 else "sdxl"

        duration_sec = video_frames / frame_rate if enable_video else 0.0

        return {
            "ui": {"images": preview_images},
            "result": (latent_dict,),
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
