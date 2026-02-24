import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any

# Local imports
from .utils import (
    tensor_srgb_to_linear,
    tensor_to_numpy_float32,
    numpy_to_tensor_float32,
)

logger = logging.getLogger("radiance.hdr.tonemap")

# ═══════════════════════════════════════════════════════════════════════════════
#                          TONE MAPPING NODES
# ═══════════════════════════════════════════════════════════════════════════════


class HDRExpandDynamicRange:
    """
    Expand SDR images to HDR dynamic range by recovering highlights and extending stops of exposure latitude.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "source_gamma": (
                    "FLOAT",
                    {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1},
                ),
                "highlight_recovery": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "black_point": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001},
                ),
                "target_stops": (
                    "FLOAT",
                    {"default": 14.0, "min": 8.0, "max": 20.0, "step": 0.5},
                ),
                "highlight_rolloff": (
                    "FLOAT",
                    {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "expand"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Expand SDR images to HDR dynamic range by recovering highlights and extending stops of exposure latitude."

    def expand(
        self,
        image: torch.Tensor,
        source_gamma: float = 2.2,
        highlight_recovery: float = 1.0,
        black_point: float = 0.0,
        target_stops: float = 14.0,
        highlight_rolloff: float = 1.5,
    ) -> Tuple[torch.Tensor]:

        # GPU-accelerated implementation
        img = image.float()

        # 1. Convert to linear using tensor helper
        linear = tensor_srgb_to_linear(img, source_gamma)

        # 2. Adjust black point
        linear = linear - black_point
        linear = torch.clamp(linear, min=0.0)

        # 3. Calculate Luminance (to preserve color relationships)
        # Using Rec.709 coefficients
        luma = (
            0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
        )

        # 4. Expansion Logic
        # Target Peak: 2^(stops - 8). e.g., 14 stops -> 2^6 = 64.0
        target_peak = 2.0 ** (target_stops - 8.0)

        # Use highlight_rolloff to control the threshold softness
        # Higher rolloff = softer transition, starts expansion earlier
        # Range 1.0-3.0 maps to threshold 0.9-0.5
        threshold = (
            1.0 - (highlight_rolloff - 1.0) * 0.2
        )  # rolloff 1.0->0.9, 1.5->0.8, 3.0->0.5
        threshold = max(0.5, min(0.95, threshold))  # Clamp to safe range

        if target_peak > 1.0 and highlight_recovery > 0:
            t = threshold

            # Calculate 'a' coefficient for quadratic curve
            a = (target_peak - 1.0) / ((1.0 - t) ** 2)

            # Clamp luma for curve calculation
            luma_clamped = torch.clamp(luma, max=1.0)

            # Quadratic expansion: y = t + (x - t) + a * (x - t)^2
            # Only apply to values above threshold
            expanded_luma = torch.where(
                luma > t,
                t + (luma_clamped - t) + a * torch.pow(luma_clamped - t, 2),
                luma,
            )

            # For inputs > 1.0, extrapolate linearly with slope at 1.0
            slope_at_1 = 1.0 + 2 * a * (1.0 - t)
            expanded_luma = torch.where(
                luma > 1.0, target_peak + (luma - 1.0) * slope_at_1, expanded_luma
            )

            # Blend based on recovery strength
            final_luma = expanded_luma * highlight_recovery + luma * (
                1.0 - highlight_recovery
            )

            # Apply new luminance to RGB
            # NewColor = OldColor * (NewLuma / OldLuma)
            ratio = final_luma / (luma + 1e-8)

            # Expand dims for RGB multiply
            ratio = ratio.unsqueeze(-1)

            linear = linear * ratio

        return (linear,)


class HDRToneMap:
    """
    Professional HDR tone mapping with presets and advanced controls. GPU-accelerated.
    """

    TONEMAP_OPERATORS = [
        "filmic_aces",
        "filmic_uncharted2",
        "agx",
        "reinhard",
        "reinhard_extended",
        "reinhard_luminance",
        "linear_clamp",
        "exposure_only",
    ]

    # Tone map look presets
    LOOK_PRESETS = [
        "None (Custom)",
        "🎬 Cinematic Film",
        "📺 HDR Display",
        "🌐 Web / Social",
        "🖨️ Print Ready",
        "🎮 Game Engine",
        "📷 Photography",
        "🌙 Low Key / Dark",
        "☀️ High Key / Bright",
    ]

    PRESET_CONFIGS = {
        "🎬 Cinematic Film": {
            "operator": "filmic_aces",
            "exposure": 0.0,
            "gamma": 2.2,
            "white_point": 1.0,
            "contrast": 1.1,
            "saturation": 0.95,
            "highlight_compression": 0.8,
            "shadow_lift": 0.02,
        },
        "📺 HDR Display": {
            "operator": "agx",
            "exposure": 0.3,
            "gamma": 2.2,
            "white_point": 2.0,
            "contrast": 1.0,
            "saturation": 1.1,
            "highlight_compression": 0.5,
            "shadow_lift": 0.0,
        },
        "🌐 Web / Social": {
            "operator": "filmic_aces",
            "exposure": 0.2,
            "gamma": 2.2,
            "white_point": 1.0,
            "contrast": 1.15,
            "saturation": 1.1,
            "highlight_compression": 0.9,
            "shadow_lift": 0.01,
        },
        "🖨️ Print Ready": {
            "operator": "reinhard_luminance",
            "exposure": -0.2,
            "gamma": 2.2,
            "white_point": 1.0,
            "contrast": 0.95,
            "saturation": 0.9,
            "highlight_compression": 0.7,
            "shadow_lift": 0.03,
        },
        "🎮 Game Engine": {
            "operator": "filmic_uncharted2",
            "exposure": 0.0,
            "gamma": 2.2,
            "white_point": 4.0,
            "contrast": 1.05,
            "saturation": 1.0,
            "highlight_compression": 0.6,
            "shadow_lift": 0.0,
        },
        "📷 Photography": {
            "operator": "reinhard_extended",
            "exposure": 0.0,
            "gamma": 2.2,
            "white_point": 2.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "highlight_compression": 0.5,
            "shadow_lift": 0.0,
        },
        "🌙 Low Key / Dark": {
            "operator": "filmic_aces",
            "exposure": -0.5,
            "gamma": 2.4,
            "white_point": 1.0,
            "contrast": 1.2,
            "saturation": 0.85,
            "highlight_compression": 0.9,
            "shadow_lift": 0.0,
        },
        "☀️ High Key / Bright": {
            "operator": "reinhard",
            "exposure": 0.5,
            "gamma": 2.0,
            "white_point": 1.5,
            "contrast": 0.9,
            "saturation": 1.05,
            "highlight_compression": 0.4,
            "shadow_lift": 0.05,
        },
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input HDR or SDR image to tone map"}),
            },
            "optional": {
                "preset": (
                    cls.LOOK_PRESETS,
                    {
                        "default": "🎬 Cinematic Film",
                        "tooltip": "Quick look preset. Overrides settings below.",
                    },
                ),
                "operator": (
                    cls.TONEMAP_OPERATORS,
                    {
                        "default": "filmic_aces",
                        "tooltip": "Tone mapping algorithm:\n• filmic_aces: Industry-standard film curve\n• filmic_uncharted2: Game industry favorite\n• agx: Modern Blender default\n• reinhard: Classic, preserves color\n• linear_clamp: Simple clip",
                    },
                ),
                "exposure": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -5.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Exposure adjustment in stops. Negative = darker, Positive = brighter.",
                    },
                ),
                "gamma": (
                    "FLOAT",
                    {
                        "default": 2.2,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Display gamma. 2.2 = sRGB standard. Higher = darker midtones.",
                    },
                ),
                "white_point": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Maximum brightness that maps to white. Higher = more headroom for highlights.",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Contrast adjustment around midpoint. >1 = more punch.",
                    },
                ),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Color saturation. 0 = grayscale, 1 = original, >1 = vivid.",
                    },
                ),
                "highlight_compression": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Compress highlights to prevent clipping. 0 = no compression, 1 = full.",
                    },
                ),
                "shadow_lift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.2,
                        "step": 0.01,
                        "tooltip": "Lift shadows to reveal detail. 0 = no lift, 0.1 = subtle.",
                    },
                ),
                "use_gpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use GPU acceleration when available.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Tone-mapped SDR image ready for display or export.",)
    FUNCTION = "tonemap"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Professional HDR tone mapping with presets and advanced controls. GPU-accelerated."

    def _gpu_tonemap(
        self, x: torch.Tensor, operator: str, white_point: float
    ) -> torch.Tensor:
        """GPU-accelerated tone mapping."""
        if operator == "reinhard":
            return x / (1.0 + x)
        elif operator == "reinhard_extended":
            white_sq = white_point * white_point
            return (x * (1.0 + x / white_sq)) / (1.0 + x)
        elif operator == "reinhard_luminance":
            # GPU version of luminance-based Reinhard
            luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
            luma = torch.clamp(luma, min=1e-6)
            white_sq = white_point * white_point
            luma_tm = (luma * (1.0 + luma / white_sq)) / (1.0 + luma)
            scale = (luma_tm / luma).unsqueeze(-1)
            return x * scale
        elif operator == "filmic_aces":
            # FIX #12 (GPU): white_point acts as an input scene-exposure scale.
            # The ACES curve has no built-in white_point parameter; we divide the
            # input by white_point so values at white_point map to the curve's
            # shoulder, matching user expectation (previously ignored silently).
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            x_scaled = x / (white_point + 1e-8)
            return torch.clamp(
                (x_scaled * (a * x_scaled + b)) / (x_scaled * (c * x_scaled + d) + e),
                0,
                1,
            )
        elif operator == "filmic_uncharted2":
            A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

            def curve(v):
                return (
                    (v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)
                ) - E / F

            white_scale = 1.0 / curve(torch.tensor(white_point, device=x.device))
            return curve(x) * white_scale
        elif operator == "agx":
            # FIX #11 (GPU): previous code clamped to 1e-6 then added 1e-6 again inside
            # log2(), doubling the epsilon offset at the minimum value. Use a single guard.
            # NOTE: this is an AgX-inspired approximation (log+smoothstep). A full AgX
            # implementation requires the sRGB→AgX gamut matrix; that is documented in
            # the CPU _agx() method.
            x = torch.clamp(x, min=1e-10)
            x = torch.log2(x) / 16.0 + 0.5
            x = torch.clamp(x, 0, 1)
            return x * x * (3.0 - 2.0 * x)
        elif operator == "linear_clamp":
            return torch.clamp(x / white_point, 0, 1)
        else:  # exposure_only
            return torch.clamp(x, 0, 1)

    def _reinhard(self, x: np.ndarray, white: float = 1.0) -> np.ndarray:
        """Simple Reinhard tone mapping."""
        return x / (1.0 + x)

    def _reinhard_extended(self, x: np.ndarray, white: float = 4.0) -> np.ndarray:
        """Extended Reinhard with white point."""
        numerator = x * (1.0 + x / (white * white))
        return numerator / (1.0 + x)

    def _reinhard_luminance(self, rgb: np.ndarray, white: float = 4.0) -> np.ndarray:
        """Reinhard applied to luminance only."""
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        luma = np.maximum(luma, 1e-6)

        luma_tm = self._reinhard_extended(luma, white)
        scale = luma_tm / luma

        return rgb * scale[..., np.newaxis]

    def _filmic_aces(self, x: np.ndarray, white_point: float = 1.0) -> np.ndarray:
        """ACES filmic tone mapping approximation.
        FIX #12: white_point now used as input pre-scale (was silently ignored).
        """
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        x = x / (white_point + 1e-8)
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

    def _filmic_uncharted2(self, x: np.ndarray) -> np.ndarray:
        """Uncharted 2 filmic curve."""
        A = 0.15  # Shoulder strength
        B = 0.50  # Linear strength
        C = 0.10  # Linear angle
        D = 0.20  # Toe strength
        E = 0.02  # Toe numerator
        F = 0.30  # Toe denominator

        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    def _agx(self, x: np.ndarray) -> np.ndarray:
        """AgX-inspired tone mapping approximation.
        NOTE: This is a log-compressed smoothstep, not full AgX. A complete
        AgX implementation requires the sRGB→AgX gamut matrix:
          [[0.842479, 0.042328, 0.042376],
           [0.078434, 0.878469, 0.078434],
           [0.079224, 0.079166, 0.879142]]
        FIX #11: removed double-epsilon (was maximum(x,0)+1e-6 then log2(x+1e-6)).
        """
        x = np.maximum(x, 1e-10)
        x = np.log2(x) / 16.0 + 0.5
        x = np.clip(x, 0, 1)
        # Smoothstep contrast curve
        x = x * x * (3.0 - 2.0 * x)
        return x

    def tonemap(
        self,
        image: torch.Tensor,
        preset: str = "🎬 Cinematic Film",
        operator: str = "filmic_aces",
        exposure: float = 0.0,
        gamma: float = 2.2,
        white_point: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        highlight_compression: float = 0.5,
        shadow_lift: float = 0.0,
        use_gpu: bool = True,
    ) -> Tuple[torch.Tensor]:

        # Apply preset if selected
        if preset != "None (Custom)" and preset in self.PRESET_CONFIGS:
            config = self.PRESET_CONFIGS[preset]
            operator = config.get("operator", operator)
            exposure = config.get("exposure", exposure)
            gamma = config.get("gamma", gamma)
            white_point = config.get("white_point", white_point)
            contrast = config.get("contrast", contrast)
            saturation = config.get("saturation", saturation)
            highlight_compression = config.get(
                "highlight_compression", highlight_compression
            )
            shadow_lift = config.get("shadow_lift", shadow_lift)

        # Try GPU path
        if use_gpu and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                img = image.to(device).float()

                # Apply exposure
                img = img * (2.0**exposure)

                # Apply highlight compression (before tone mapping)
                if highlight_compression > 0:
                    # Soft knee compression for highlights
                    threshold = 1.0 - highlight_compression * 0.5
                    highlight_mask = img > threshold
                    compressed = threshold + (img - threshold) / (
                        1.0 + (img - threshold) * highlight_compression * 2
                    )
                    img = torch.where(highlight_mask, compressed, img)

                # Apply shadow lift
                if shadow_lift > 0:
                    img = img + shadow_lift * (1.0 - img)

                # Apply tone mapping
                result = self._gpu_tonemap(img, operator, white_point)

                # Apply contrast (linear space, before gamma encoding)
                # FIX #10: contrast was applied after gamma, causing nonlinear colour
                # shifts because the 0.5 midpoint has different physical meaning in
                # gamma-encoded space. Now applied in linear tone-mapped space.
                if contrast != 1.0:
                    result = (result - 0.5) * contrast + 0.5

                # Apply saturation (linear space, before gamma encoding)
                if saturation != 1.0 and result.shape[-1] >= 3:
                    luma = (
                        0.2126 * result[..., 0]
                        + 0.7152 * result[..., 1]
                        + 0.0722 * result[..., 2]
                    )
                    luma = luma.unsqueeze(-1)
                    result = luma + saturation * (result - luma)

                # Apply gamma (display encoding — must be last)
                result = torch.clamp(result, 0, 1)
                result = torch.pow(result, 1.0 / gamma)

                # IMPORTANT: Move result to CPU before returning to prevent VRAM accumulation
                cpu_result = result.cpu()
                return (cpu_result,)

            except RuntimeError as e:
                logger.warning(f"GPU tone mapping failed: {e}. Falling back to CPU.")
            finally:
                # FIX #8: guard empty_cache — unconditional call crashes on MPS / CPU-only.
                # FIX #9: removed non-functional gpu_tensors list; only img was appended to
                #         it, and del on the list never freed the tensors it contained since
                #         Python's reference count kept them alive. empty_cache() is the
                #         real cleanup and is all that is needed here.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # CPU fallback
        img = tensor_to_numpy_float32(image)

        # Apply exposure
        img = img * (2.0**exposure)

        # Apply highlight compression (before tone mapping)
        if highlight_compression > 0:
            threshold = 1.0 - highlight_compression * 0.5
            highlight_mask = img > threshold
            compressed = threshold + (img - threshold) / (
                1.0 + (img - threshold) * highlight_compression * 2
            )
            img = np.where(highlight_mask, compressed, img)

        # Apply shadow lift
        if shadow_lift > 0:
            img = img + shadow_lift * (1.0 - img)

        # Apply tone mapping operator
        if operator == "reinhard":
            result = self._reinhard(img, white_point)
        elif operator == "reinhard_extended":
            result = self._reinhard_extended(img, white_point)
        elif operator == "reinhard_luminance":
            result = self._reinhard_luminance(img, white_point)
        elif operator == "filmic_aces":
            result = self._filmic_aces(img, white_point)
        elif operator == "filmic_uncharted2":
            white_scale = 1.0 / self._filmic_uncharted2(np.array([white_point]))[0]
            result = self._filmic_uncharted2(img) * white_scale
        elif operator == "agx":
            result = self._agx(img)
        elif operator == "linear_clamp":
            result = np.clip(img / white_point, 0, 1)
        else:  # exposure_only
            result = np.clip(img, 0, 1)

        # FIX #10: apply contrast and saturation BEFORE gamma encoding.
        # Previous code applied them after gamma, causing nonlinear colour shifts
        # because the 0.5 midpoint has different physical meaning in gamma space.
        # Apply contrast (linear tone-mapped space)
        if contrast != 1.0:
            result = (result - 0.5) * contrast + 0.5

        # Apply saturation adjustment (linear tone-mapped space)
        if saturation != 1.0 and result.shape[-1] >= 3:
            luma = (
                0.2126 * result[..., 0]
                + 0.7152 * result[..., 1]
                + 0.0722 * result[..., 2]
            )
            luma = luma[..., np.newaxis]
            result = luma + saturation * (result - luma)

        # Apply gamma (display encoding — must be the final step)
        result = np.clip(result, 0, 1)
        result = np.power(result, 1.0 / gamma)

        return (numpy_to_tensor_float32(result),)


# =============================================================================
# NODE MAPPINGS
# FIX #7: NODE_CLASS_MAPPINGS was absent — both nodes were invisible to ComfyUI.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "HDRExpandDynamicRange": HDRExpandDynamicRange,
    "HDRToneMap": HDRToneMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HDRExpandDynamicRange": "◎ HDR Expand Dynamic Range",
    "HDRToneMap": "◎ HDR Tone Map",
}
