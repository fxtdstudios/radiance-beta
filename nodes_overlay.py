# ═══════════════════════════════════════════════════════════════════════════════
# RADIANCE: Professional HDR & VFX Suite
# Version: v2.3
# ═══════════════════════════════════════════════════════════════════════════════
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger("◎ Radiance.overlay")


class RadianceMetadataOverlay:
    """
    Burn-in metadata, timecode, project info, and text reports onto images.
    Creates a professional "slate" look for dailies and reviews.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "project": ("STRING", {"default": "◎ Radiance_PRO", "multiline": False}),
                "shot": ("STRING", {"default": "SH_010", "multiline": False}),
                "frame": ("INT", {"default": 1, "min": 0, "max": 999999}),
                "timecode": ("STRING", {"default": "00:00:00:00", "multiline": False}),
                "position": (
                    [
                        "Bottom Bar (Slate)",
                        "Top Left",
                        "Top Right",
                        "Bottom Left",
                        "Bottom Right",
                    ],
                ),
                "font_size": ("INT", {"default": 20, "min": 10, "max": 100}),
                "opacity": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
            },
            "optional": {
                "user_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Additional notes...",
                    },
                ),
                "report_text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "placeholder": "Connect QC Report here...",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "overlay_metadata"
    CATEGORY = "FXTD Studios/Radiance/Draw"
    DESCRIPTION = "Burn-in metadata, timecode, and reports onto images (Slate/Burn-in)."

    def overlay_metadata(
        self,
        image: torch.Tensor,
        enabled: bool,
        project: str,
        shot: str,
        frame: int,
        timecode: str,
        position: str,
        font_size: int,
        opacity: float,
        user_text: str = "",
        report_text: str = "",
    ):

        if not enabled:
            return (image,)

        batch_size, h, w, c = image.shape
        device = image.device

        # Font setup (try to find a monospace font)
        try:
            # Common paths for monospace fonts
            # Windows: consolo.ttf, arial.ttf
            # We try a few standard ones
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.load_default()
            except (OSError, IOError) as e:
                logger.warning(f"Failed to load default font: {e}")

        # We will build the overlay mask and RGB layers
        # Since text generation allows flexible content per frame (timecode, numbers),
        # we process each frame's overlay on CPU via PIL, then compositing on Tensor.

        result_images = []

        for i in range(batch_size):
            # Create transparent overlay layer
            overlay_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_layer)

            # Prepare text content
            metadata_lines = []
            if project:
                metadata_lines.append(f"PRJ: {project}")
            if shot:
                metadata_lines.append(f"SHT: {shot}")
            if frame is not None:
                current_frame = frame + i  # Increment frame number for batch
                metadata_lines.append(f"FRM: {current_frame:04d}")
            if timecode:
                metadata_lines.append(f"TC : {timecode}")  # TODO: Increment TC?

            # Combine all text
            main_text = " | ".join(metadata_lines)

            # Additional text (User notes + Report)
            extra_text = []
            if user_text:
                extra_text.append(user_text)
            if report_text:
                extra_text.append("--- REPORT ---")
                extra_text.append(report_text)

            full_extra_text = "\n".join(extra_text)

            # Layout Logic
            margin = 20

            # Colors
            bg_color = (0, 0, 0, int(255 * opacity))
            text_color = (255, 255, 255, 255)
            sub_text_color = (200, 200, 200, 255)

            if position == "Bottom Bar (Slate)":
                # Background Bar
                bar_height = font_size * 2 + 10
                if full_extra_text:
                    # Calculate extra height needed
                    lines = full_extra_text.count("\n") + 1
                    bar_height += lines * (font_size + 4) + 10

                draw.rectangle([(0, h - bar_height), (w, h)], fill=bg_color)

                # Draw Main Line
                text_y = h - bar_height + 10
                draw.text((margin, text_y), main_text, font=font, fill=text_color)

                # Draw Extra Text
                if full_extra_text:
                    draw.text(
                        (margin, text_y + font_size + 10),
                        full_extra_text,
                        font=font,
                        fill=sub_text_color,
                    )

            else:
                # Floating Box Logic
                text_to_draw = main_text
                if full_extra_text:
                    text_to_draw += "\n" + full_extra_text

                # Estimate box size
                bbox = draw.multiline_textbbox(
                    (0, 0), text_to_draw, font=font, align="left"
                )
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                box_x, box_y = 0, 0

                if "Top" in position:
                    box_y = margin
                else:  # Bottom
                    box_y = h - text_h - margin - 10

                if "Left" in position:
                    box_x = margin
                else:  # Right
                    box_x = w - text_w - margin - 10

                padding = 10
                draw.rectangle(
                    [
                        (box_x - padding, box_y - padding),
                        (box_x + text_w + padding, box_y + text_h + padding),
                    ],
                    fill=bg_color,
                )

                # Draw Text
                draw.multiline_text(
                    (box_x, box_y),
                    text_to_draw,
                    font=font,
                    fill=text_color,
                    align="left",
                )

            # Convert overlay to tensor (H, W, 4)
            # Normalize to 0-1 float
            overlay_np = np.array(overlay_layer).astype(np.float32) / 255.0
            overlay_tensor = torch.from_numpy(overlay_np).to(device)

            # Separate RGB and Alpha
            overlay_rgb = overlay_tensor[..., :3]
            overlay_alpha = overlay_tensor[..., 3:4]

            # Expand overlay alpha to match image channels if needed
            if c == 1:
                # For grayscale image, we rely on broadcasting or converting overlay to gray?
                # Usually overlays are color. If image is gray, we might output color?
                # Let's assume output remains input channels, but overlay is colored -> force output RGB?
                # If input is grayscale, outputting grayscale overlay is safer for pipeline consistency.
                # However, usually slates ARE colored.
                # ComfyUI practice: if input is mask/gray, and we draw color, we should promote to RGB.
                # But here we want to modify 'image'.
                # Let's keep input channels for now. If RGB overlay on Gray, convert overlay to Gray.
                overlay_rgb = (
                    overlay_rgb[..., 0] * 0.299
                    + overlay_rgb[..., 1] * 0.587
                    + overlay_rgb[..., 2] * 0.114
                ).unsqueeze(-1)

            result_frame = (
                image[i] * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha
            )
            result_images.append(result_frame)

        output_tensor = torch.stack(result_images, dim=0)
        return (output_tensor,)


# ─────────────────────────────────────────────────────────────────────────────
#  RadianceBlendComposite  (v1.0)
# ─────────────────────────────────────────────────────────────────────────────

BLEND_MODES = ["Normal", "Add", "Screen", "Multiply", "Overlay", "Soft Light", "Difference", "Divide"]


class RadianceBlendComposite:
    """
    Composite two images using industry-standard blend modes.
    All math is in scene-linear fp32; HDR values survive Add and Screen.
    An optional mask (MASK) controls compositing coverage per-pixel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base":    ("IMAGE", {"tooltip": "Bottom layer (background)."}),
                "blend":   ("IMAGE", {"tooltip": "Top layer (foreground)."}),
                "mode":    (BLEND_MODES, {"default": "Normal"}),
                "opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Overall strength of the blend layer."},
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {"tooltip": "Optional per-pixel mask (grayscale). White = full blend, Black = base only."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "FXTD Studios/Radiance/Draw"
    DESCRIPTION = "Composite two images using Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, or Divide blend modes."

    def composite(
        self,
        base: torch.Tensor,
        blend: torch.Tensor,
        mode: str = "Normal",
        opacity: float = 1.0,
        mask: torch.Tensor = None,
    ) -> tuple:
        A = base.float()
        B = blend.float()

        # Match spatial dimensions — broadcast blend to base if needed
        if B.shape[1:3] != A.shape[1:3]:
            import torch.nn.functional as F
            B = F.interpolate(
                B.permute(0, 3, 1, 2),
                size=(A.shape[1], A.shape[2]),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        # Match batch dimension
        if B.shape[0] == 1 and A.shape[0] > 1:
            B = B.expand_as(A)

        # ── Blend operations ────────────────────────────────────────────────
        if mode == "Normal":
            blended = B
        elif mode == "Add":
            blended = A + B
        elif mode == "Screen":
            blended = 1.0 - (1.0 - A.clamp(0, 1)) * (1.0 - B.clamp(0, 1))
        elif mode == "Multiply":
            blended = A * B
        elif mode == "Overlay":
            low  = 2.0 * A * B
            high = 1.0 - 2.0 * (1.0 - A) * (1.0 - B)
            blended = torch.where(A < 0.5, low, high)
        elif mode == "Soft Light":
            # Pegtop formula (no clamping)
            blended = (1.0 - 2.0 * B) * A * A + 2.0 * B * A
        elif mode == "Difference":
            blended = torch.abs(A - B)
        elif mode == "Divide":
            blended = A / (B + 1e-7)
        else:
            blended = B

        # ── Opacity + mask compositing ───────────────────────────────────────
        alpha = opacity
        if mask is not None:
            # mask shape: (B, H, W) or (H, W) → expand to (B, H, W, 1)
            m = mask.float()
            if m.dim() == 2:
                m = m.unsqueeze(0)
            m = m.unsqueeze(-1)  # (B, H, W, 1)
            if m.shape[0] == 1 and A.shape[0] > 1:
                m = m.expand(A.shape[0], -1, -1, -1)
            alpha_map = m * opacity
            out = A * (1.0 - alpha_map) + blended * alpha_map
        else:
            out = A * (1.0 - alpha) + blended * alpha

        return (out,)



NODE_CLASS_MAPPINGS = {
    "◎ RadianceMetadataOverlay": RadianceMetadataOverlay,
    "◎ RadianceBlendComposite":  RadianceBlendComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "◎ RadianceMetadataOverlay": "◎ Radiance Metadata Overlay",
    "◎ RadianceBlendComposite": "◎ Radiance Blend Composite",
}

