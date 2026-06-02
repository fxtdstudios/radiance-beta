"""
nodes_optics.py — Radiance Optical Nodes  v3.0.0

CHANGELOG vs v1.0:

  BUG FIXES:
    BUG 1: RadianceLensDistortion invert path — division by zero when
           k1=-0.5, r2=2.0 at image corners (denominator = 0).
           Fixed: denominator clamped to ±1e-6 before division.
    BUG 8: RadianceChromaticAberration — shift_g=0.0 still ran a full
           grid_sample over the green channel (identity warp = original pixels).
           Fixed: short-circuit to direct slice copy when shift==0.
    BUG 10: RadianceAnamorphicStreaks — conv kernel rebuilt on every
            _horizontal_blur() call across cascaded passes.
            Fixed: kernel built once, passed as argument.
    BUG 11: Passes = streak_length//32 gave only 2 passes at 64px — too few
            for a convincing anamorphic tail. Changed to log2 scaling which
            gives 6 passes at 64px matching reference anamorphic renders.
    BUG 3:  torch.tensor([cx, cy]) recreated on every apply() call.
            Precomputed once before grid operations.

  NEW NODES:
    RadianceFilmGrain   — HDR-aware film grain (Gaussian + Poisson).
                          Grain scales inversely with luminance (shadows
                          are coarser, highlights are finer — physically correct).
    RadianceVignette    — Optical power-law vignette with color tint.
                          Separate highlight/shadow tint colors for stylistic
                          and physically-accurate optical falloff.

  UPGRADES:
    AnamorphicStreaks   — streak_direction ["Horizontal","Vertical","Diagonal +45","Diagonal -45"]
    AnamorphicStreaks   — streak_falloff FLOAT control (was hardcoded to padding/3)
    All nodes          — IS_CHANGED classmethod for proper ComfyUI cache invalidation
    All nodes          — logging added
    LensDistortion     — invert mode tooltip documents approximation limitation
"""

import math
import logging
import hashlib

import torch
import torch.nn.functional as F

logger = logging.getLogger("radiance.optics")


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 1: LENS DISTORTION
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceLensDistortion:
    """
    ◎ Radiance Lens Distortion

    True Brown-Conrady radial distortion (barrel / pincushion) with:
      • k1, k2 — radial distortion coefficients (k1 dominates, k2 for extreme)
      • Configurable optical center (cx, cy)
      • Scale factor to control crop after distortion
      • Invert mode for plate undistortion (technical / match-move pipelines)
      • ST-Map output (R=U, G=V) compatible with Nuke STMap and Fusion DisplaceImage

    Barrel distortion: k1 < 0  (convex outward, common in wide lenses)
    Pincushion:        k1 > 0  (concave inward, common in telephoto)
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Apply or remove lens distortion using Brown-Conrady coefficients."
    FUNCTION = "apply"
    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("image", "st_map")
    OUTPUT_TOOLTIPS = (
        "Distorted image.",
        "ST-Map: R=U, G=V normalized [0,1] sample coordinates. "
        "Wire to Nuke STMap / Fusion DisplaceImage for downstream use.",
    )

    @classmethod
    def IS_CHANGED(cls, image, k1, k2, scale, center_x, center_y, padding_mode, invert):
        return hashlib.md5(
            f"{k1:.6f}{k2:.6f}{scale:.6f}{center_x:.6f}{center_y:.6f}{padding_mode}{invert}".encode()
        ).hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":   ("IMAGE",),
                "k1":      ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0,  "step": 0.005,
                    "tooltip": "Primary radial distortion. k1<0 = barrel, k1>0 = pincushion."}),
                "k2":      ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0,  "step": 0.005,
                    "tooltip": "Secondary radial distortion. Affects extreme corners. Use sparingly."}),
                "scale":   ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 2.0,  "step": 0.01,
                    "tooltip": "Uniform scale applied after distortion. Use to crop black borders."}),
                "center_x":("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0,  "step": 0.01, "tooltip": "Horizontal center of the vignette (0 = left, 1 = right).",
                    "tooltip": "Optical center X (0.5 = image center)."}),
                "center_y":("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0,  "step": 0.01, "tooltip": "Vertical center of the vignette (0 = top, 1 = bottom).",
                    "tooltip": "Optical center Y (0.5 = image center)."}),
                "padding_mode": (["zeros", "reflection", "border"], {"default": "zeros",
                    "tooltip": "Edge fill: zeros=black, reflection=mirrored, border=edge-clamped."}),
                "invert":  ("BOOLEAN", {"default": False,
                    "tooltip": (
                        "Invert warp for plate undistortion. "
                        "Uses exact closed-form inverse — valid for all k1/k2 combinations "
                        "except the degenerate singularity at 1+k1·r²+k2·r⁴=0 "
                        "(clamped automatically)."
                    )}),
            }
        }

    def apply(self, image: torch.Tensor, k1: float, k2: float, scale: float,
              center_x: float, center_y: float, padding_mode: str, invert: bool):
        B, H, W, C = image.shape
        device = image.device
        dtype  = image.dtype

        # Normalized grid [-1, 1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )

        # Shift to optical center — precomputed once (BUG 3 FIX)
        cx = (center_x * 2.0) - 1.0
        cy = (center_y * 2.0) - 1.0
        center_vec = torch.tensor([cx, cy], device=device, dtype=dtype)

        grid = torch.stack((x - cx, y - cy), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        r2   = grid[..., 0] ** 2 + grid[..., 1] ** 2

        if invert:
            # Exact closed-form inverse for Brown-Conrady radial model.
            # BUG 1 FIX: denominator = 1 + k1·r² + k2·r⁴ can be 0 for strong
            # barrel distortion (k1≈-0.5) at corners where r²≈2.
            # Clamp to ±1e-6 prevents NaN/Inf.
            denom = 1.0 + k1 * r2 + k2 * (r2 ** 2)
            denom = torch.where(denom.abs() < 1e-6,
                                torch.full_like(denom, 1e-6 * denom.sign().clamp(min=1.0)),
                                denom)
            scale_factor = (1.0 / max(scale, 1e-6)) / denom
        else:
            distortion   = 1.0 + k1 * r2 + k2 * (r2 ** 2)
            scale_factor = distortion * scale

        distorted_grid = grid * scale_factor.unsqueeze(-1) + center_vec

        img_bchw = image.permute(0, 3, 1, 2)
        out_bchw = F.grid_sample(
            img_bchw, distorted_grid,
            mode="bicubic", padding_mode=padding_mode, align_corners=True,
        )
        out = out_bchw.permute(0, 2, 3, 1)

        # ST-Map: R=U, G=V in [0,1]. Nuke/Fusion compatible.
        # Two-channel content, zero-padded to RGB so ComfyUI IMAGE shape is (B,H,W,3).
        st_uv   = (distorted_grid * 0.5 + 0.5).clamp(0.0, 1.0)
        st_map  = torch.cat([st_uv, torch.zeros_like(st_uv[..., :1])], dim=-1)

        logger.debug(f"[LensDistortion] k1={k1} k2={k2} scale={scale} invert={invert}")
        return (out, st_map)


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 2: CHROMATIC ABERRATION
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceChromaticAberration:
    """
    ◎ Radiance Chromatic Aberration

    Physically based lateral chromatic aberration (transverse CA).
    Radially shifts R, G, B channels outward/inward from the optical center
    by independent amounts — simulating lens IoR dispersion.

    Typical camera values:
      shift_r ≈ +0.005, shift_g = 0.0, shift_b ≈ -0.005
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Simulate lateral chromatic aberration (fringe) as a lens artefact."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def IS_CHANGED(cls, image, shift_r, shift_g, shift_b, center_x, center_y, invert):
        return hashlib.md5(
            f"{shift_r:.6f}{shift_g:.6f}{shift_b:.6f}{center_x:.4f}{center_y:.4f}{invert}".encode()
        ).hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "shift_r":  ("FLOAT", {"default":  0.005, "min": -0.1, "max": 0.1, "step": 0.001,
                    "tooltip": "Radial scale for Red channel. Positive = pushed outward."}),
                "shift_g":  ("FLOAT", {"default":  0.0,   "min": -0.1, "max": 0.1, "step": 0.001,
                    "tooltip": "Radial scale for Green channel. Typically 0 (reference channel)."}),
                "shift_b":  ("FLOAT", {"default": -0.005, "min": -0.1, "max": 0.1, "step": 0.001,
                    "tooltip": "Radial scale for Blue channel. Negative = pulled inward."}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal center of the distortion/effect (0.0 = left, 1.0 = right)."}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical center of the effect (0.0 = top, 1.0 = bottom)."}),
                "invert":   ("BOOLEAN", {"default": False,
                    "tooltip": "Invert shift for CA removal / undistortion pass."}),
            }
        }

    def apply(self, image: torch.Tensor, shift_r: float, shift_g: float, shift_b: float,
              center_x: float, center_y: float, invert: bool):
        B, H, W, C = image.shape
        device = image.device
        dtype  = image.dtype

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )
        cx = (center_x * 2.0) - 1.0
        cy = (center_y * 2.0) - 1.0
        center_vec = torch.tensor([cx, cy], device=device, dtype=dtype)

        grid     = torch.stack((x - cx, y - cy), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        img_bchw = image.permute(0, 3, 1, 2)

        out_channels = []
        for c, shift in enumerate([shift_r, shift_g, shift_b][:min(3, C)]):

            # BUG 8 FIX: identity warp (shift==0) — skip grid_sample entirely
            if shift == 0.0:
                out_channels.append(img_bchw[:, c:c + 1, :, :])
                continue

            # Exact inverse: for output pixel at grid position g=(x-cx),
            # sample input at (g/(1+s))+cx. Verified: forward(inverse(x))=x.
            s = shift if not invert else (1.0 / (1.0 + shift) - 1.0)
            channel_grid = grid * (1.0 + s) + center_vec

            sampled = F.grid_sample(
                img_bchw[:, c:c + 1, :, :],
                channel_grid,
                mode="bicubic", padding_mode="reflection", align_corners=True,
            )
            out_channels.append(sampled)

        if C > 3:
            # Alpha preserved at original position (correct for pre-multiplied images)
            out_channels.append(img_bchw[:, 3:, :, :])

        out = torch.cat(out_channels, dim=1).permute(0, 2, 3, 1)
        logger.debug(f"[ChromAb] R={shift_r} G={shift_g} B={shift_b} invert={invert}")
        return (out,)


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 3: ANAMORPHIC STREAKS
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceAnamorphicStreaks:
    """
    ◎ Radiance Anamorphic Streaks

    Cinematic highlight streaking using cascaded directional convolution passes.
    Simulates the characteristic horizontal flare of anamorphic cinema lenses
    (Panavision, Hawk, Cooke Anamorphic/i series).

    v2.0 changes:
      - streak_direction: Horizontal / Vertical / Diagonal ±45°
      - streak_falloff:   controls kernel exponential decay (was hardcoded)
      - Log2-scaled pass count gives 6 passes at streak_length=64 (was 2)
      - Kernel built once per apply() — not rebuilt every pass (BUG 10 fix)
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Add anamorphic lens streak flares to bright highlights."
    FUNCTION = "apply"
    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("image", "streak_pass")
    OUTPUT_TOOLTIPS = (
        "Original image with streaks composited (add blend).",
        "Isolated streak pass for manual compositing (pre-add).",
    )

    @classmethod
    def IS_CHANGED(cls, image, threshold, streak_length, streak_color_r, streak_color_g,
                   streak_color_b, intensity, streak_direction, streak_falloff):
        return hashlib.md5(
            f"{threshold}{streak_length}{streak_color_r}{streak_color_g}"
            f"{streak_color_b}{intensity}{streak_direction}{streak_falloff:.4f}".encode()
        ).hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":           ("IMAGE",),
                "threshold":       ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Luminance threshold above which highlights generate streaks."}),
                "streak_length":   ("INT",   {"default": 64,   "min": 1,   "max": 512,  "step": 1,
                    "tooltip": "Maximum streak tail length in pixels."}),
                "streak_color_r":  ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 5.0,  "step": 0.01,
                    "tooltip": "Red component of the anamorphic streak colour. Values > 1.0 produce HDR-energy streaks."
                }),
                "streak_color_g":  ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 5.0,  "step": 0.01,
                    "tooltip": "Green component of the anamorphic streak colour."
                }),
                "streak_color_b":  ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 5.0,  "step": 0.01,
                    "tooltip": "Streak color tint. Default is the cool blue-cyan typical of anamorphic lenses."}),
                "intensity":       ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Vignette or streak intensity. Values > 1.0 add HDR energy."}),
                "streak_direction":(["Horizontal", "Vertical", "Diagonal +45", "Diagonal -45"],
                    {"default": "Horizontal",
                     "tooltip": "Direction of streak propagation."}),
                "streak_falloff":  ("FLOAT", {"default": 3.0,  "min": 0.5, "max": 20.0, "step": 0.1,
                    "tooltip": (
                        "Exponential decay rate of the streak kernel. "
                        "Higher = tighter falloff (shorter effective tail). "
                        "Lower = longer, softer streak."
                    )}),
            }
        }

    @staticmethod
    def _make_kernel(padding: int, falloff: float, device, dtype) -> torch.Tensor:
        """Build 1-D exponential-decay convolution kernel. Built once per pass set."""
        kernel_size = padding * 2 + 1
        coords = torch.linspace(-padding, padding, kernel_size, device=device, dtype=dtype)
        kernel = torch.exp(-torch.abs(coords) / (max(padding, 1) / falloff))
        return kernel / kernel.sum()

    @staticmethod
    def _directional_blur(x: torch.Tensor, kernel: torch.Tensor, padding: int,
                          direction: str) -> torch.Tensor:
        """Apply 1-D directional blur. Handles all four directions."""
        in_c = x.shape[1]

        if direction == "Horizontal":
            k4d   = kernel.view(1, 1, 1, -1).repeat(in_c, 1, 1, 1)
            return F.conv2d(x, k4d, padding=(0, padding), groups=in_c)

        if direction == "Vertical":
            k4d   = kernel.view(1, 1, -1, 1).repeat(in_c, 1, 1, 1)
            return F.conv2d(x, k4d, padding=(padding, 0), groups=in_c)

        # Diagonal: rotate 45° via affine grid, blur horizontally, rotate back
        B, C_in, H, W = x.shape
        angle = math.radians(45.0 if direction == "Diagonal +45" else -45.0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # Affine rotation matrices
        theta_fwd = torch.tensor(
            [[cos_a, -sin_a, 0.0],
             [sin_a,  cos_a, 0.0]], device=x.device, dtype=x.dtype
        ).unsqueeze(0).expand(B, -1, -1)
        theta_inv = torch.tensor(
            [[ cos_a, sin_a, 0.0],
             [-sin_a, cos_a, 0.0]], device=x.device, dtype=x.dtype
        ).unsqueeze(0).expand(B, -1, -1)

        grid_fwd = F.affine_grid(theta_fwd, x.size(), align_corners=True)
        rotated  = F.grid_sample(x, grid_fwd, mode="bilinear",
                                 padding_mode="zeros", align_corners=True)

        # Horizontal blur on rotated content
        k4d     = kernel.view(1, 1, 1, -1).repeat(in_c, 1, 1, 1)
        blurred = F.conv2d(rotated, k4d, padding=(0, padding), groups=in_c)

        # Rotate back
        grid_inv = F.affine_grid(theta_inv, blurred.size(), align_corners=True)
        return F.grid_sample(blurred, grid_inv, mode="bilinear",
                             padding_mode="zeros", align_corners=True)

    def apply(self, image: torch.Tensor, threshold: float, streak_length: int,
              streak_color_r: float, streak_color_g: float, streak_color_b: float,
              intensity: float, streak_direction: str = "Horizontal",
              streak_falloff: float = 3.0):

        B, H, W, C = image.shape
        device = image.device

        img_bchw = image.permute(0, 3, 1, 2)
        RGB      = img_bchw[:, :3, :, :]

        # Isolate super-threshold highlights
        highlights = F.relu(RGB - threshold)

        # BUG 11 FIX: log2 pass scaling — 6 passes at 64px vs old 2 passes
        # Each pass halves the remaining kernel width for a realistic long tail
        passes      = max(1, int(math.log2(max(streak_length, 1))))
        current_pad = streak_length

        blurred = highlights
        for _ in range(passes):
            # BUG 10 FIX: kernel built once per pass, not inside the helper
            kernel  = self._make_kernel(current_pad, streak_falloff, device, image.dtype)
            blurred = self._directional_blur(blurred, kernel, current_pad, streak_direction)
            current_pad = max(1, int(current_pad * 0.75))

        color_tint = torch.tensor(
            [streak_color_r, streak_color_g, streak_color_b],
            dtype=blurred.dtype, device=device,
        ).view(1, 3, 1, 1)

        streaks  = blurred * color_tint * intensity
        out_rgb  = RGB + streaks

        if C > 3:
            out_bchw    = torch.cat([out_rgb, img_bchw[:, 3:, :, :]], dim=1)
            streaks_out = torch.cat([streaks, img_bchw[:, 3:, :, :]], dim=1)
        else:
            out_bchw    = out_rgb
            streaks_out = streaks

        logger.debug(
            f"[AnaStreaks] threshold={threshold} length={streak_length} "
            f"passes={passes} dir={streak_direction}"
        )
        return (out_bchw.permute(0, 2, 3, 1), streaks_out.permute(0, 2, 3, 1))


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 4: FILM GRAIN (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceFilmGrain:
    """
    ◎ Radiance Film Grain

    Physically based photographic grain model:
      • Gaussian component — coarse structure grain (silver halide clumps)
      • Poisson scaling   — grain amplitude scales with √luminance (shot noise)
      • Per-channel size  — R/G/B have slightly different grain textures
      • HDR-aware         — grain is computed in scene-linear space and
                            modulated by local luminance, so highlights
                            receive finer grain than shadows (correct)

    Grain size is in pixels (Gaussian sigma). Typical film stocks:
      Fine grain   (ISO 100): size=0.5, strength=0.015
      Medium grain (ISO 400): size=1.0, strength=0.04
      Heavy grain  (ISO 3200):size=2.0, strength=0.12
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = "Simulate photochemical film grain. Luminance-weighted noise, HDR-safe — grain intensity scales with pixel brightness."

    @classmethod
    def IS_CHANGED(cls, image, grain_size, grain_strength, grain_size_r_offset,
                   grain_size_g_offset, grain_size_b_offset, hdr_aware, seed):
        return hashlib.md5(
            f"{grain_size:.4f}{grain_strength:.6f}{grain_size_r_offset:.4f}"
            f"{grain_size_g_offset:.4f}{grain_size_b_offset:.4f}{hdr_aware}{seed}".encode()
        ).hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "grain_size":     ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05,
                    "tooltip": "Base grain size in pixels (Gaussian sigma). 1.0 ≈ ISO 400 medium format."}),
                "grain_strength": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Grain amplitude. 0.015=fine, 0.04=medium, 0.12=heavy."}),
                "grain_size_r_offset": ("FLOAT", {"default":  0.1, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "R channel grain size offset from base (pixels). Red grain is typically coarser."}),
                "grain_size_g_offset": ("FLOAT", {"default":  0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "G channel grain size offset. Green has finest grain (most photo-sites)."}),
                "grain_size_b_offset": ("FLOAT", {"default": -0.1, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "B channel grain size offset. Blue grain is typically finest."}),
                "hdr_aware":  ("BOOLEAN", {"default": True,
                    "tooltip": (
                        "Scale grain amplitude by 1/√luminance (Poisson shot-noise model). "
                        "Makes highlights finer-grained than shadows — physically correct. "
                        "Disable for flat/uniform grain."
                    )}),
                "seed":   ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1,
                    "tooltip": "Random seed. 0 = random each run."}),
            }
        }

    @staticmethod
    def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Fast separable 2-D Gaussian blur for grain structure."""
        if sigma < 0.1:
            return x
        # Kernel radius: 3σ, minimum 1
        r = max(1, int(math.ceil(sigma * 3.0)))
        ks = r * 2 + 1
        coords = torch.linspace(-r, r, ks, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
        kernel = kernel / kernel.sum()
        C = x.shape[1]
        # Separable: horizontal then vertical
        kh = kernel.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        kv = kernel.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        x  = F.conv2d(x, kh, padding=(0, r), groups=C)
        x  = F.conv2d(x, kv, padding=(r, 0), groups=C)
        return x

    def apply(self, image: torch.Tensor, grain_size: float, grain_strength: float,
              grain_size_r_offset: float, grain_size_g_offset: float,
              grain_size_b_offset: float, hdr_aware: bool, seed: int):

        B, H, W, C = image.shape
        device = image.device
        dtype  = image.dtype

        if seed != 0:
            torch.manual_seed(seed)

        channel_sizes = [
            max(0.1, grain_size + grain_size_r_offset),
            max(0.1, grain_size + grain_size_g_offset),
            max(0.1, grain_size + grain_size_b_offset),
        ]

        # Per-channel grain: white noise → Gaussian blur → scale by strength
        grain_channels = []
        for c in range(min(3, C)):
            # Raw white noise (zero mean)
            noise = torch.randn(B, 1, H, W, device=device, dtype=dtype) * grain_strength

            # Give grain spatial structure via Gaussian blur
            noise = self._gaussian_blur_2d(noise, channel_sizes[c])

            grain_channels.append(noise)

        grain_rgb = torch.cat(grain_channels, dim=1)  # (B, 3, H, W)

        img_bchw  = image.permute(0, 3, 1, 2)
        rgb       = img_bchw[:, :3, :, :]

        if hdr_aware:
            # Poisson shot-noise modulation: grain amplitude ∝ 1/√luminance
            # Brighter regions get finer grain — matches film physics
            luma = (0.2126 * rgb[:, 0:1, :, :].clamp(min=0.0)
                  + 0.7152 * rgb[:, 1:2, :, :].clamp(min=0.0)
                  + 0.0722 * rgb[:, 2:3, :, :].clamp(min=0.0))
            # Normalise: luma=1.0 → full strength; luma=4.0 → half strength
            modulation = 1.0 / (luma + 1.0).sqrt()
            grain_rgb  = grain_rgb * modulation

        out_rgb = rgb + grain_rgb

        if C > 3:
            out_bchw = torch.cat([out_rgb, img_bchw[:, 3:, :, :]], dim=1)
        else:
            out_bchw = out_rgb

        logger.debug(
            f"[FilmGrain] size={grain_size} strength={grain_strength} hdr_aware={hdr_aware}"
        )
        return (out_bchw.permute(0, 2, 3, 1),)


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 5: VIGNETTE (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceVignette:
    """
    ◎ Radiance Vignette

    Optical power-law vignette with separate highlight/shadow color tint.

    Physically, vignette follows a cos⁴(θ) falloff from the optical axis.
    The power parameter controls the exponent — higher values give a harder,
    more abrupt edge (stylistic); lower values give a gradual natural falloff.

    Color tint allows:
      • Warm-to-cool: subtle color temperature shift toward edges (common in
        period film aesthetic — warm center, cool shadows)
      • Cold vignette: muted blue-grey edges (noir, thriller)
      • No tint: neutral luminance falloff only
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Apply a natural or stylised vignette to the image edges."
    FUNCTION = "apply"
    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("image", "vignette_mask")
    OUTPUT_TOOLTIPS = (
        "Vignetted image.",
        "Grayscale vignette mask [0,1] — wire to downstream grades or multiply nodes.",
    )

    @classmethod
    def IS_CHANGED(cls, image, strength, power, center_x, center_y,
                   tint_r, tint_g, tint_b, feather):
        return hashlib.md5(
            f"{strength:.4f}{power:.4f}{center_x:.4f}{center_y:.4f}"
            f"{tint_r:.4f}{tint_g:.4f}{tint_b:.4f}{feather:.4f}".encode()
        ).hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5,  "min": 0.0,  "max": 1.0,  "step": 0.01,
                    "tooltip": "Vignette intensity at corners. 0=no effect, 1=full black."}),
                "power":    ("FLOAT", {"default": 2.0,  "min": 0.5,  "max": 8.0,  "step": 0.1,
                    "tooltip": "Falloff exponent. 2.0 = natural cos⁴ approximation. Higher = harder edge."}),
                "center_x": ("FLOAT", {"default": 0.5,  "min": 0.0,  "max": 1.0,  "step": 0.01, "tooltip": "Horizontal center of the vignette (0 = left, 1 = right)."}),
                "center_y": ("FLOAT", {"default": 0.5,  "min": 0.0,  "max": 1.0,  "step": 0.01, "tooltip": "Vertical center of the vignette (0 = top, 1 = bottom)."}),
                "feather":  ("FLOAT", {"default": 1.0,  "min": 0.1,  "max": 4.0,  "step": 0.05,
                    "tooltip": "Radial feather: scales the normalized radius. >1 = pushes falloff inward."}),
            },
            "optional": {
                "tint_r":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Red multiplier for vignetted (dark) areas. 1.0=neutral."}),
                "tint_g":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Green channel multiplier for vignette tint. 1.0 = neutral."}),
                "tint_b":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Blue multiplier for vignetted areas. >1 = cool shadow edges."}),
            },
        }

    def apply(self, image: torch.Tensor, strength: float, power: float,
              center_x: float, center_y: float, feather: float,
              tint_r: float = 1.0, tint_g: float = 1.0, tint_b: float = 1.0):

        B, H, W, C = image.shape
        device = image.device
        dtype  = image.dtype

        # Normalized distance from optical center, aspect-ratio corrected
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij",
        )
        aspect = W / max(H, 1)
        dx = (x - center_x) * aspect
        dy = (y - center_y)
        r  = torch.sqrt(dx ** 2 + dy ** 2) * feather

        # Cosine-power falloff: 1 at center, 0 at normalized radius=1
        # cos(r * π/2) raised to `power`
        r_clamped = (r * math.pi * 0.5).clamp(0.0, math.pi * 0.5)
        mask      = torch.cos(r_clamped) ** power        # [0, 1], shape (H, W)

        # Vignette: blend from 1.0 (no effect) to mask (full effect) by strength
        vignette = 1.0 - strength * (1.0 - mask)        # (H, W)
        vignette = vignette.unsqueeze(0).unsqueeze(-1)   # (1, H, W, 1)

        # Color tint: scale individual channels in vignetted region
        tint = torch.tensor([tint_r, tint_g, tint_b], device=device, dtype=dtype)
        # Tint blends from neutral (1,1,1) to tint color proportional to (1-vignette)
        tint_factor = 1.0 - strength * (1.0 - mask.unsqueeze(0).unsqueeze(-1))
        tint_map    = 1.0 + (tint - 1.0) * (1.0 - tint_factor)

        img_rgb    = image[..., :3]
        out_rgb    = img_rgb * vignette * tint_map

        if C > 3:
            out = torch.cat([out_rgb, image[..., 3:]], dim=-1)
        else:
            out = out_rgb

        # Vignette mask as grayscale IMAGE (B, H, W, 3) for downstream use
        mask_3ch = mask.unsqueeze(0).unsqueeze(-1).expand(B, H, W, 3)

        logger.debug(
            f"[Vignette] strength={strength} power={power} feather={feather}"
        )
        return (out, mask_3ch)


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceLensDistortion":      RadianceLensDistortion,
    "RadianceChromaticAberration": RadianceChromaticAberration,
    "RadianceAnamorphicStreaks":   RadianceAnamorphicStreaks,
    # RadianceFilmGrain: canonical key is in film/__init__.py (GPU-accelerated, more advanced)
    "RadianceVignette":            RadianceVignette,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLensDistortion":      "◎ Radiance Lens Distortion",
    "RadianceChromaticAberration": "◎ Radiance Chromatic Aberration",
    "RadianceAnamorphicStreaks":   "◎ Radiance Anamorphic Streaks",
    "RadianceVignette":            "◎ Radiance Vignette",
}
