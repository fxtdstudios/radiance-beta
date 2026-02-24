"""
Radiance Resolution - Professional Resolution & Aspect Ratio Node
Version: 1.2.0

v1.2.0 Fixes:
- Added NODE_CLASS_MAPPINGS — node was invisible to ComfyUI without it
- Fixed Flux/SD3 latent spatial factor: //16 not //8 (SDXL stays //8)
- swap_dimensions now inverts aspect_str (e.g. 16:9 → 9:16)
- Separator display strings excluded from selectable preset list
- Megapixel scaling uses candidate-search for minimum AR drift (was up to 5.5%)
- Multi-path font loading: tries DejaVu/Liberation/Helvetica before default
- Completed ASPECT_RATIOS dict (5 entries missing: 3:1, 4:1, 2.63:1, 1.5:1, 1.375:1)
- Preview text positions clamped to stay inside box at extreme aspect ratios
- Info string shows actual latent spatial dimensions and compression factor
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import logging

# Module logger
logger = logging.getLogger("radiance.image.resolution")


class RadianceResolution:
    """
    Professional Resolution Generator with industry presets: • Social Media (Instagram, TikTok, YouTube, Twitter, etc.) • Film & Cinema (4K DCI, 2K, IMAX, Anamorphic scopes) • Television (8K/4K UHD, HD, SD) • AI Optimized (SDXL, FLUX) Outputs empty latent + visual preview.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # RESOLUTION PRESETS
    # ═══════════════════════════════════════════════════════════════════════════

    PRESETS = {
        # ─────────────────────────────────────────────────────────────────────
        # SOCIAL MEDIA
        # ─────────────────────────────────────────────────────────────────────
        "── SOCIAL MEDIA ──": None,
        "Instagram Square (1:1)": (1080, 1080, "1:1"),
        "Instagram Portrait (4:5)": (1080, 1350, "4:5"),
        "Instagram Story/Reel (9:16)": (1080, 1920, "9:16"),
        "Instagram Landscape (1.91:1)": (1080, 566, "1.91:1"),
        "TikTok (9:16)": (1080, 1920, "9:16"),
        "YouTube Thumbnail (16:9)": (1280, 720, "16:9"),
        "YouTube Short (9:16)": (1080, 1920, "9:16"),
        "Twitter/X Post (16:9)": (1200, 675, "16:9"),
        "Twitter/X Header (3:1)": (1500, 500, "3:1"),
        "Facebook Post (1.91:1)": (1200, 628, "1.91:1"),
        "Facebook Cover (2.63:1)": (820, 312, "2.63:1"),
        "LinkedIn Post (1.91:1)": (1200, 627, "1.91:1"),
        "LinkedIn Banner (4:1)": (1584, 396, "4:1"),
        "Pinterest Pin (2:3)": (1000, 1500, "2:3"),
        "Snapchat (9:16)": (1080, 1920, "9:16"),
        # ─────────────────────────────────────────────────────────────────────
        # FILM & CINEMA
        # ─────────────────────────────────────────────────────────────────────
        "── FILM & CINEMA ──": None,
        "4K DCI (1.90:1)": (4096, 2160, "1.90:1"),
        "4K DCI Scope (2.39:1)": (4096, 1716, "2.39:1"),
        "4K DCI Flat (1.85:1)": (4096, 2214, "1.85:1"),
        "2K DCI (1.90:1)": (2048, 1080, "1.90:1"),
        "2K DCI Scope (2.39:1)": (2048, 858, "2.39:1"),
        "2K DCI Flat (1.85:1)": (2048, 1107, "1.85:1"),
        "IMAX (1.43:1)": (5616, 3924, "1.43:1"),
        "IMAX Digital (1.90:1)": (4096, 2160, "1.90:1"),
        "Anamorphic 2.39:1": (2880, 1206, "2.39:1"),
        "Anamorphic 2.76:1 Ultra Panavision": (2880, 1044, "2.76:1"),
        "Academy Ratio (1.375:1)": (2048, 1489, "1.375:1"),
        "VistaVision (1.5:1)": (3072, 2048, "1.5:1"),
        "Super 35 (1.85:1)": (2048, 1107, "1.85:1"),
        # ─────────────────────────────────────────────────────────────────────
        # TELEVISION & BROADCAST
        # ─────────────────────────────────────────────────────────────────────
        "── TELEVISION ──": None,
        "8K UHD (16:9)": (7680, 4320, "16:9"),
        "4K UHD (16:9)": (3840, 2160, "16:9"),
        "1080p Full HD (16:9)": (1920, 1080, "16:9"),
        "1080i HD (16:9)": (1920, 1080, "16:9"),
        "720p HD (16:9)": (1280, 720, "16:9"),
        "576p PAL (16:9)": (1024, 576, "16:9"),
        "480p NTSC (16:9)": (854, 480, "16:9"),
        "4:3 SD (4:3)": (640, 480, "4:3"),
        # ─────────────────────────────────────────────────────────────────────
        # PHOTOGRAPHY & PRINT
        # ─────────────────────────────────────────────────────────────────────
        "── PHOTOGRAPHY ──": None,
        "Full Frame 3:2": (3000, 2000, "3:2"),
        "Medium Format 4:3": (4000, 3000, "4:3"),
        "Square 1:1": (2000, 2000, "1:1"),
        "Panoramic 3:1": (3000, 1000, "3:1"),
        "Ultra-Wide 21:9": (2520, 1080, "21:9"),
        # ─────────────────────────────────────────────────────────────────────
        # AI/FLUX OPTIMIZED
        # ─────────────────────────────────────────────────────────────────────
        "── AI OPTIMIZED ──": None,
        "SDXL Square (1:1)": (1024, 1024, "1:1"),
        "SDXL Portrait (3:4)": (896, 1152, "3:4"),
        "SDXL Landscape (4:3)": (1152, 896, "4:3"),
        "SDXL Wide (16:9)": (1344, 768, "16:9"),
        "SDXL Tall (9:16)": (768, 1344, "9:16"),
        "FLUX 1MP Square": (1024, 1024, "1:1"),
        "FLUX 1MP Wide (16:9)": (1344, 768, "16:9"),
        "FLUX 1MP Portrait (9:16)": (768, 1344, "9:16"),
        "FLUX 2K Square (1:1)": (1440, 1440, "1:1"),
        "FLUX 2K Wide (16:9)": (1920, 1080, "16:9"),
        "FLUX 2K Portrait (9:16)": (1080, 1920, "9:16"),
        "FLUX 2K DCI (1.90:1)": (2048, 1080, "1.90:1"),
    }

    ASPECT_RATIOS = {
        "1:1": 1.0,
        "4:5": 0.8,
        "9:16": 0.5625,
        "16:9": 1.7778,
        "3:2": 1.5,
        "2:3": 0.6667,
        "4:3": 1.3333,
        "3:4": 0.75,
        "21:9": 2.3333,
        "1.85:1": 1.85,
        "2.39:1": 2.39,
        "2.76:1": 2.76,
        "1.43:1": 1.43,
        "1.90:1": 1.90,
        "1.91:1": 1.91,
        # FIX #7: entries that appeared in PRESETS but were missing from this dict
        "3:1": 3.0,
        "4:1": 4.0,
        "2.63:1": 2.63,
        "1.5:1": 1.5,
        "1.375:1": 1.375,
    }

    # FIX #4: separator display strings — excluded from the selectable preset list
    # so users can never accidentally select them as a resolution.
    SEPARATORS = {
        "── SOCIAL MEDIA ──",
        "── FILM & CINEMA ──",
        "── TELEVISION ──",
        "── PHOTOGRAPHY ──",
        "── AI OPTIMIZED ──",
    }

    MEGAPIXEL_TARGETS = ["0.5", "1.0", "1.5", "2.0", "4.0", "8.0", "Custom"]

    # Model types with their latent channel counts
    MODEL_TYPES = {
        "SDXL / SD1.5": 4,
        "Flux / SD3": 16,
    }

    # FIX #2: VAE spatial compression factor per model family.
    # SD1.5 / SDXL downsamples 8× (latent = image // 8).
    # Flux / SD3 downsamples 16× (latent = image // 16).
    # Using 8 for Flux produces a 4× oversized latent tensor that causes a
    # shape mismatch at inference.
    LATENT_SPATIAL = {
        "SDXL / SD1.5": 8,
        "Flux / SD3": 16,
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # FIX #4: exclude separator display strings — they are not real presets
        # and must never appear as selectable options in the UI.
        preset_list = [k for k in cls.PRESETS.keys() if k not in cls.SEPARATORS]
        aspect_list = list(cls.ASPECT_RATIOS.keys())
        model_type_list = list(cls.MODEL_TYPES.keys())

        return {
            "required": {
                "preset": (preset_list, {"default": "FLUX 1MP Square"}),
            },
            "optional": {
                "model_type": (model_type_list, {"default": "Flux / SD3"}),
                "megapixels": (cls.MEGAPIXEL_TARGETS, {"default": "1.0"}),
                "custom_width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 64},
                ),
                "custom_height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 64},
                ),
                "custom_aspect": (aspect_list, {"default": "1:1"}),
                "use_custom": ("BOOLEAN", {"default": False}),
                "divisible_by": ([8, 16, 32, 64], {"default": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "swap_dimensions": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "preview", "width", "height", "info")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/Image"
    DESCRIPTION = """Professional Resolution Generator with industry presets:
• Social Media (Instagram, TikTok, YouTube, Twitter, etc.)
• Film & Cinema (4K DCI, 2K, IMAX, Anamorphic scopes)
• Television (8K/4K UHD, HD, SD)
• AI Optimized (SDXL, FLUX)
Outputs empty latent + visual preview."""

    def _round_to_divisible(self, value: int, divisor: int) -> int:
        """Round value to nearest multiple of divisor."""
        return round(value / divisor) * divisor

    def _best_divisible_pair(
        self, target_w: float, target_h: float, ar: float, mp: float, div: int
    ):
        """
        Find the (width, height) pair that is divisible by `div` and minimises
        aspect-ratio error while staying within ±15% of target megapixels.

        Searches ±4 divisible steps around the rounded target_w, evaluates two
        height candidates (floor and ceil to divisible) for each width, and returns
        the candidate with the smallest AR deviation.  MP proximity breaks ties.

        When no candidate satisfies the ±15% MP tolerance the constraint is relaxed
        and the globally minimum AR-error pair is returned instead.
        """
        base_w = self._round_to_divisible(int(round(target_w)), div)
        candidates = []
        for dw in range(-4, 5):
            cw = base_w + dw * div
            if cw < div:
                continue
            ch_ideal = cw / ar
            ch_lo = max(div, math.floor(ch_ideal / div) * div)
            ch_hi = ch_lo + div
            for ch in (ch_lo, ch_hi):
                mp_actual = cw * ch / 1_000_000
                mp_err = abs(mp_actual - mp) / mp
                ar_err = abs(cw / ch - ar) / ar
                within_mp = mp_err <= 0.15
                candidates.append((within_mp, ar_err, mp_err, cw, ch))

        # Sort: prefer within-MP-tolerance, then lowest AR error, then closest MP
        candidates.sort(key=lambda x: (not x[0], x[1], x[2]))
        _, _, _, best_w, best_h = candidates[0]
        return int(best_w), int(best_h)

    def _create_preview_image(
        self, width: int, height: int, preset_name: str, aspect_str: str
    ) -> torch.Tensor:
        """Create a visual preview image showing the resolution and aspect ratio."""
        # Create preview at reasonable size
        preview_max = 512
        scale = min(preview_max / width, preview_max / height)

        pw = int(width * scale)
        ph = int(height * scale)

        # Create image with dark background
        img = Image.new("RGB", (preview_max, preview_max), color=(20, 20, 28))
        draw = ImageDraw.Draw(img)

        # Calculate centered position for aspect ratio box
        x_offset = (preview_max - pw) // 2
        y_offset = (preview_max - ph) // 2

        # Draw aspect ratio frame
        # Outer glow
        for i in range(3, 0, -1):
            alpha = int(80 / i)
            draw.rectangle(
                [x_offset - i, y_offset - i, x_offset + pw + i, y_offset + ph + i],
                outline=(0, 136 + alpha, 255),
                width=1,
            )

        # Main frame
        draw.rectangle(
            [x_offset, y_offset, x_offset + pw, y_offset + ph],
            outline=(220, 60, 60),
            width=2,
        )

        # Fill with subtle gradient effect
        for y in range(y_offset + 2, y_offset + ph - 2):
            alpha = int(20 + 10 * (y - y_offset) / ph)
            draw.line(
                [(x_offset + 2, y), (x_offset + pw - 2, y)],
                fill=(alpha, alpha, alpha + 10),
            )

        # Draw center crosshair
        cx = x_offset + pw // 2
        cy = y_offset + ph // 2
        draw.line([(cx - 20, cy), (cx + 20, cy)], fill=(100, 100, 120), width=1)
        draw.line([(cx, cy - 20), (cx, cy + 20)], fill=(100, 100, 120), width=1)

        # Rule of thirds guides
        third_w = pw // 3
        third_h = ph // 3
        for i in range(1, 3):
            # Vertical lines
            x = x_offset + i * third_w
            draw.line([(x, y_offset), (x, y_offset + ph)], fill=(60, 60, 80), width=1)
            # Horizontal lines
            y = y_offset + i * third_h
            draw.line([(x_offset, y), (x_offset + pw, y)], fill=(60, 60, 80), width=1)

        # FIX #6: try platform-appropriate font paths before falling back to the
        # Pillow bitmap default.  "arial.ttf" only works on Windows; the majority
        # of ComfyUI production servers run Linux where it is never present.
        _font_candidates = [
            "arial.ttf",  # Windows
            "Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Ubuntu/Debian
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Fedora/RHEL
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/System/Library/Fonts/SFNSText.ttf",
        ]
        title_font = info_font = None
        for _path in _font_candidates:
            try:
                title_font = ImageFont.truetype(_path, 24)
                info_font = ImageFont.truetype(_path, 14)
                break
            except OSError:
                continue
        if title_font is None:
            logger.debug(
                "No TrueType font found on any search path — using Pillow bitmap default"
            )
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Draw resolution text
        res_text = f"{width}x{height}"
        ratio_text = f"({aspect_str})"

        # FIX #8: clamp text positions so they never overflow the preview box,
        # which happens at extreme aspect ratios (e.g. Twitter Header 3:1 or
        # Instagram Story 9:16 where one dimension of the box is very small).
        text_bbox = draw.textbbox((0, 0), res_text, font=title_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        ratio_bbox = draw.textbbox((0, 0), ratio_text, font=info_font)
        ratio_w = ratio_bbox[2] - ratio_bbox[0]

        # Clamp x so text stays inside the box
        res_x = max(x_offset + 4, min(cx - text_w // 2, x_offset + pw - text_w - 4))
        ratio_x = max(x_offset + 4, min(cx - ratio_w // 2, x_offset + pw - ratio_w - 4))

        # Clamp y so text stays inside the box and above the bottom info bar
        res_y = max(y_offset + 4, min(cy - 20, y_offset + ph - text_h * 2 - 16))
        ratio_y = max(res_y + text_h + 4, min(cy + 10, y_offset + ph - text_h - 8))

        draw.text((res_x, res_y), res_text, fill=(220, 60, 60), font=title_font)
        draw.text((ratio_x, ratio_y), ratio_text, fill=(0, 170, 255), font=info_font)

        # Bottom info bar
        info_text = f"Resolution: {width} x {height}"
        draw.text(
            (10, preview_max - 25), info_text, fill=(180, 180, 200), font=info_font
        )

        # Convert to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    def generate(
        self,
        preset: str,
        model_type: str = "Flux / SD3",
        megapixels: str = "1.0",
        custom_width: int = 1024,
        custom_height: int = 1024,
        custom_aspect: str = "1:1",
        use_custom: bool = False,
        divisible_by: int = 64,
        batch_size: int = 1,
        swap_dimensions: bool = False,
    ):

        # FIX #4: separator guard removed — separators are now excluded from the
        # selectable preset list in INPUT_TYPES so this branch can never be reached.

        # Determine dimensions
        if use_custom:
            width = custom_width
            height = custom_height
            aspect_str = custom_aspect
        elif preset in self.PRESETS and self.PRESETS[preset] is not None:
            width, height, aspect_str = self.PRESETS[preset]
        else:
            width, height, aspect_str = 1024, 1024, "1:1"

        # Apply megapixel scaling if not custom.
        # FIX #5 (v1.2.0): exhaustive candidate search finds the globally-optimal
        # (width, height) pair that:
        #   1. minimises aspect-ratio error, subject to both dims being divisible_by-aligned
        #   2. stays within ±15% of the target megapixels (MP accuracy as tiebreaker)
        #
        # Background: independently rounding both scaled dims to divisible_by caused
        # up to 5.5% AR drift (e.g. Instagram Story (9:16) at 0.5MP → 512x960, AR=0.533).
        # A pure AR-preserving formula (derive h from rounded w) still inherits the same
        # snap error when the ideal long-dimension isn't a divisible_by multiple.  The
        # search approach evaluates all ±4-step neighbourhood candidates and picks the
        # pair with the smallest AR deviation.  Residual error is mathematically bounded
        # by the divisibility grid (e.g. 640/0.8=800, not div-by-64 → min error ~3.8%).
        if not use_custom and megapixels != "Custom":
            mp = float(megapixels)
            ar = width / height
            # Ideal (non-integer) dimensions at target MP
            new_h = math.sqrt(mp * 1_000_000 / ar)
            new_w = new_h * ar
            width, height = self._best_divisible_pair(
                new_w, new_h, ar, mp, divisible_by
            )

        # Ensure divisibility
        width = self._round_to_divisible(width, divisible_by)
        height = self._round_to_divisible(height, divisible_by)

        # Swap dimensions if requested
        if swap_dimensions:
            width, height = height, width
            # FIX #3: update aspect_str to reflect the swapped orientation.
            # For simple "N:M" strings flip the tokens; for decimal ratios recompute.
            parts = aspect_str.split(":")
            if len(parts) == 2:
                # e.g. "16:9" → "9:16",  "4:5" → "5:4"
                aspect_str = f"{parts[1]}:{parts[0]}"
            else:
                # Decimal form like "2.39:1" — recompute from swapped dimensions
                ar_new = height / width  # after swap height is the wider dim
                aspect_str = f"{width / height:.2f}:1"

        # Ensure minimum size
        width = max(divisible_by, width)
        height = max(divisible_by, height)

        # Get latent channels based on model type
        # SDXL/SD1.5 = 4 channels, Flux/SD3 = 16 channels
        latent_channels = self.MODEL_TYPES.get(model_type, 16)

        # FIX #2: use the correct VAE spatial compression factor per model family.
        # Flux/SD3 compresses 16× (not 8×); using //8 produces a 4× oversized
        # latent tensor (e.g. 128×128 instead of 64×64 for a 1024×1024 Flux image)
        # which causes a shape mismatch at inference.
        spatial = self.LATENT_SPATIAL.get(model_type, 8)
        latent = torch.zeros(
            [batch_size, latent_channels, height // spatial, width // spatial]
        )

        # Create preview image
        preview = self._create_preview_image(width, height, preset, aspect_str)

        # Build info string
        mp_actual = (width * height) / 1_000_000
        info = f"""═══ RADIANCE RESOLUTION ═══
Preset: {preset}
Model: {model_type}
Resolution: {width} x {height}
Aspect Ratio: {aspect_str}
Megapixels: {mp_actual:.2f} MP
Latent: {latent_channels}ch @ {width // spatial}x{height // spatial} (1/{spatial})
═══════════════════════"""

        return {
            "ui": {"resolution": [f"{width}x{height}"]},
            "result": ({"samples": latent}, preview, width, height, info),
        }


# =============================================================================
# NODE MAPPINGS
# FIX #1: NODE_CLASS_MAPPINGS was missing — RadianceResolution was invisible
# to ComfyUI and could not be used in any workflow.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceResolution": RadianceResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceResolution": "◎ Radiance Resolution",
}
