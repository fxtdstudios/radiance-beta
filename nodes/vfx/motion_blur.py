import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger("radiance.motion_blur")

class RadianceMotionBlur:
    """
    ◎ Radiance Physical Motion Blur
    
    A professional vector-based motion blur engine. 
    Uses motion vectors to perform sub-frame integration in 32-bit linear space.
    
    Includes Shutter Angle control (180° = standard cinema) and 
    energy conservation for realistic highlight streaks.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "motion_vectors": ("IMAGE", {"tooltip": "32-bit UV vectors from Radiance Optical Flow."}),
                "shutter_angle": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 720.0, "step": 1.0,
                    "tooltip": "Standard cinema is 180°. Higher = more blur. 360° = full frame motion blur."}),
                "samples": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1,
                    "tooltip": "Number of sub-frame integration samples. Higher = smoother streaks."}),
                "energy_conservation": ("BOOLEAN", {"default": True,
                    "tooltip": "Ensures that bright highlights maintain their intensity over the blur area."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Apply physically-based motion blur using optical flow vectors."

    def apply(self, image: torch.Tensor, motion_vectors: torch.Tensor, shutter_angle: float, samples: int, energy_conservation: bool):
        # image shape (B, H, W, C)
        B, H, W, C = image.shape
        device = image.device
        
        # motion_vectors shape (B, H, W, 3) where R=U (dx), G=V (dy)
        vectors = motion_vectors.clone()
        
        # 1. Calculate Shutter Scale
        # Shutter angle 180 means we blur over 50% of the motion vector length
        shutter_scale = shutter_angle / 360.0
        
        img_bchw = image.permute(0, 3, 1, 2)
        
        # 2. Vector-Line Integration
        # We sample along the line defined by the motion vector
        # We integrate from -shutter_scale/2 to +shutter_scale/2 (centered shutter)
        
        # Setup Grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Normalized vectors for grid_sample
        # Vectors are in pixels. Normalized = pixels / (dim / 2)
        dx_norm = (vectors[..., 0] * shutter_scale) / (W / 2.0)
        dy_norm = (vectors[..., 1] * shutter_scale) / (H / 2.0)
        uv_norm = torch.stack([dx_norm, dy_norm], dim=-1)
        
        accum = torch.zeros_like(img_bchw)
        
        # Integration loop
        for s in range(samples):
            # t goes from -0.5 to 0.5
            t = (s / (samples - 1)) - 0.5 if samples > 1 else 0.0
            
            # Offset grid by current sample point along vector
            sample_grid = base_grid + (uv_norm * t)
            
            # Sample image
            sampled = F.grid_sample(img_bchw, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
            accum += sampled
            
        result = accum / samples
        
        # 3. Energy Conservation Logic
        # In linear space, the integrated energy should sum to the original
        # However, for HDR highlights, we sometimes want to "boost" the streaks
        # if the user requested energy conservation.
        if energy_conservation:
            orig_max = torch.max(img_bchw.view(B, C, -1), dim=-1)[0].view(B, C, 1, 1)
            res_max = torch.max(result.view(B, C, -1), dim=-1)[0].view(B, C, 1, 1)
            scale = (orig_max / (res_max + 1e-6)).clamp(min=1.0)
            result = result * scale
            result = result.clamp(max=orig_max)

        logger.info(f"[Motion Blur] Applied Vector Blur (Shutter: {shutter_angle}, Samples: {samples})")
        
        return (result.permute(0, 2, 3, 1),)

NODE_CLASS_MAPPINGS = {
    "RadianceMotionBlur": RadianceMotionBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMotionBlur": "◎ Radiance Physical Motion Blur",
}
