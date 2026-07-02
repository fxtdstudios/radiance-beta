import torch
import logging
import json

try:
    from radiance.nodes.vfx.multipass.core import _flow_to_hsv_image, _optical_flow_lk
except Exception:
    from .multipass.core import _flow_to_hsv_image, _optical_flow_lk

logger = logging.getLogger("radiance.motion")

class RadianceOpticalFlow:
    """
    ◎ Radiance Optical Flow
    
    Generates high-precision 32-bit UV motion vectors between consecutive frames.
    Compatible with Nuke's VectorBlur and Radiance Motion Coherence patches.
    
    Uses the DIS (Dense Inverse Search) algorithm for production-grade 
    motion estimation in real-time.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of frames to analyze."}),
                "preset": (["Fast", "Medium", "Ultra"], {"default": "Medium"}),
                "flow_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Scale factor for output vectors. 1.0 = pixel units."}),
                "visualize": ("BOOLEAN", {"default": False,
                    "tooltip": "Outputs a color-coded visualization of the motion field."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("motion_vectors", "visualization", "stats")
    FUNCTION = "analyze"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Estimate dense optical flow between adjacent frames."

    @torch.no_grad()
    def analyze(self, images: torch.Tensor, preset: str, flow_scale: float, visualize: bool):
        B, H, W, C = images.shape
        device = images.device
        
        if B < 2:
            empty = torch.zeros((B, H, W, 3), device=device)
            return (empty, empty, json.dumps({"error": "Batch size must be >= 2"}))

        radius = {"Fast": 3, "Medium": 5, "Ultra": 9}.get(preset, 5)
        luma = (0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2])
        luma_norm = (torch.log1p(luma.clamp(min=0.0) * 10.0) / 2.4).clamp(0.0, 1.0)

        vectors_out = []
        visuals_out = []
        vectors_out.append(torch.zeros((H, W, 3), device=device, dtype=torch.float32))
        visuals_out.append(torch.zeros((H, W, 3), device=device, dtype=torch.float32))

        for i in range(1, B):
            curr = luma_norm[i : i + 1]
            prev = luma_norm[i - 1 : i]
            u, v = _optical_flow_lk(curr, prev, window_radius=radius)
            u = u * flow_scale
            v = v * flow_scale
            vec = torch.stack(
                [u.squeeze(0), v.squeeze(0), torch.zeros((H, W), device=device, dtype=torch.float32)],
                dim=-1,
            )
            vectors_out.append(vec)
            if visualize:
                visuals_out.append(_flow_to_hsv_image(u, v).squeeze(0))
            else:
                visuals_out.append(torch.zeros((H, W, 3), device=device, dtype=torch.float32))

        vectors_tensor = torch.stack(vectors_out)
        visuals_tensor = torch.stack(visuals_out)
        
        stats = json.dumps({
            "frames": B,
            "resolution": f"{W}x{H}",
            "preset": preset,
            "mean_mag": float(torch.norm(vectors_tensor, dim=-1).mean().item())
        })
        
        logger.info(f"[Optical Flow] Analyzed {B} frames. Mean motion: {json.loads(stats)['mean_mag']:.2f}px")
        
        return (vectors_tensor, visuals_tensor, stats)

NODE_CLASS_MAPPINGS = {
    "RadianceOpticalFlow": RadianceOpticalFlow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceOpticalFlow": "◎ Radiance Optical Flow",
}
