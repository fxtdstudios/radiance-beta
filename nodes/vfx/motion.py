import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import json

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

        # 1. Initialize DIS Flow
        # DIS_MEDIUM is a good balance for 1080p+ content
        inst = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_FAST if preset == "Fast" 
            else cv2.DISOPTICAL_FLOW_PRESET_MEDIUM if preset == "Medium"
            else cv2.DISOPTICAL_FLOW_PRESET_ULTRA
        )
        
        # 2. Pre-process frames (Luma only for flow)
        # We normalize HDR to [0, 1] for the estimator to ensure contrast consistency
        luma = (0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2])
        # Log-like normalization to prevent highlight clamping in the estimator
        luma_norm = torch.log1p(luma * 10.0) / 2.4
        luma_np = (luma_norm.cpu().numpy() * 255).astype(np.uint8)

        vectors_out = []
        visuals_out = []
        
        # Frame 0 has zero motion relative to itself
        vectors_out.append(torch.zeros((H, W, 2), device=device))
        visuals_out.append(torch.zeros((H, W, 3), device=device))

        for i in range(1, B):
            prev = luma_np[i-1]
            curr = luma_np[i]
            
            # 3. Compute Flow (current -> previous)
            # We want vectors that tell us where pixels in the CURRENT frame came from in the PREVIOUS frame.
            flow = inst.calc(curr, prev, None)
            
            # flow is (H, W, 2) [dx, dy]
            # 4. Format UV Vectors (R=dx, G=dy) for Nuke VectorBlur compat
            u = torch.from_numpy(flow[..., 0]) * flow_scale
            v = torch.from_numpy(flow[..., 1]) * flow_scale
            vec = torch.stack([u, v], dim=-1).to(device)
            vectors_out.append(vec)
            
            # 5. Visualization (Color-wheel style)
            if visualize:
                hsv = np.zeros((H, W, 3), dtype=np.uint8)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                visuals_out.append((torch.from_numpy(bgr).float().to(device) / 255.0))
            else:
                visuals_out.append(torch.zeros((H, W, 3), device=device))

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
