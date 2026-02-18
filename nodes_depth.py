"""
Radiance Depth Map Generator - Professional Depth Estimation for ComfyUI
Version: 1.0.0
Author: Radiance

Uses Depth Anything V2 - Best depth estimation model 2024:
- 97.1% accuracy (δ₁=0.946 on KITTI)  
- 213ms inference (vs Marigold 5.2s)
- Handles transparent/reflective surfaces
- Auto-downloads from HuggingFace
"""

import torch
import numpy as np
import os
import folder_paths
import threading
import logging
from typing import Tuple

# Module logger
logger = logging.getLogger("radiance.depth")

# ═══════════════════════════════════════════════════════════════════════════════
#                           MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Model sizes and their HuggingFace identifiers
DEPTH_MODELS = {
    "Small (25M - Fast)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base (98M - Balanced)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large (335M - Best)": "depth-anything/Depth-Anything-V2-Large-hf"
}

# Thread-safe cache for loaded models
_model_cache = {}
_processor_cache = {}
_cache_lock = threading.RLock()


def get_device(use_gpu: bool = True) -> torch.device:
    """Get the appropriate device."""
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def download_and_load_model(model_size: str, device: torch.device):
    """Download model from HuggingFace if not cached, then load (thread-safe)."""
    global _model_cache, _processor_cache
    
    model_id = DEPTH_MODELS.get(model_size)
    if not model_id:
        model_id = DEPTH_MODELS["Base (98M - Balanced)"]
    
    with _cache_lock:
        # Check cache
        if model_id in _model_cache:
            model = _model_cache[model_id]
            processor = _processor_cache[model_id]
            model.to(device)
            return model, processor
        
        logger.info(f"Downloading depth model: {model_size}...")
        logger.info(f"Downloading depth model: {model_size}...")
        logger.debug(f"Model ID: {model_id}")
        
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            # Download and load
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id)
            model.to(device)
            model.eval()
            
            # Cache for reuse
            _model_cache[model_id] = model
            _processor_cache[model_id] = processor
            
            logger.info(f"Depth model loaded successfully: {model_id}")
            logger.info(f"Depth model loaded successfully: {model_id}")
            return model, processor
            
        except ImportError as e:
            logger.error(f"transformers library not installed: {e}")
            raise ImportError(
                "transformers library required for Depth Anything V2.\n"
                "Install with: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to download depth model: {e}")
            raise RuntimeError(f"Failed to download model: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#                           DEPTH MAP GENERATOR NODE
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceDepthMapGenerator:
    """
    🎯 Depth Anything V2 - Best 2024 Depth Estimation • Auto-downloads model from HuggingFace • 97.1% accuracy, 213ms inference • Outputs grayscale depth map (white=near, black=far) • Connect to Depth of Field node for realistic blur Model Sizes: • Small (25M) - Fastest, good for previews • Base (98M) - Balanced speed/quality • Large (335M) - Best quality for final renders
    """
    
    MODEL_SIZES = list(DEPTH_MODELS.keys())
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_size": (cls.MODEL_SIZES, {"default": "Base (98M - Balanced)"}),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": True}),
                "invert": ("BOOLEAN", {"default": False}),
                "blur_edges": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "generate_depth"
    CATEGORY = "FXTD Studios/Radiance/Data"
    
    DESCRIPTION = """🎯 Depth Anything V2 - Best 2024 Depth Estimation
    
• Auto-downloads model from HuggingFace
• 97.1% accuracy, 213ms inference
• Outputs grayscale depth map (white=near, black=far)
• Connect to Depth of Field node for realistic blur

Model Sizes:
• Small (25M) - Fastest, good for previews
• Base (98M) - Balanced speed/quality
• Large (335M) - Best quality for final renders"""

    @torch.no_grad()
    def generate_depth(
        self, 
        image: torch.Tensor,
        model_size: str,
        normalize: bool = True,
        invert: bool = False,
        blur_edges: float = 0.0,
        use_gpu: bool = True
    ) -> Tuple[torch.Tensor]:
        """Generate depth map from input image."""
        
        device = get_device(use_gpu)
        
        # Load model (auto-downloads if needed)
        model, processor = download_and_load_model(model_size, device)
        
        # Process batch
        batch_size = image.shape[0]
        depth_maps = []
        
        for i in range(batch_size):
            # Convert to PIL for processor
            img_np = image[i].cpu().numpy()
            
            # handle HDR input - tonemap to preserve details
            if img_np.max() > 1.05: # Slight tolerance for float error
                # Simple Reinhard tonemap: x / (1 + x)
                # This compresses HDR values into 0-1 range instead of hard clipping
                img_np = img_np / (1.0 + img_np)
                
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            
            # Handle different channel formats
            if img_np.shape[-1] == 4:  # RGBA
                img_np = img_np[:, :, :3]
            
            from PIL import Image
            pil_img = Image.fromarray(img_np)
            
            # Prepare input
            inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run inference
            outputs = model(**inputs)
            depth = outputs.predicted_depth
            
            # Interpolate to original size
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(image.shape[1], image.shape[2]),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            
            # Normalize to 0-1
            if normalize:
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            # Invert if requested (white=far, black=near)
            if invert:
                depth = 1.0 - depth
            
            # Apply edge blur if requested
            if blur_edges > 0:
                depth = self._blur_depth(depth, blur_edges)
            
            # Convert to 3-channel grayscale image
            depth_3ch = depth.unsqueeze(-1).repeat(1, 1, 3)
            depth_maps.append(depth_3ch.cpu())
        
        # Stack batch
        result = torch.stack(depth_maps)
        
        return (result,)
    
    def _blur_depth(self, depth: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to smooth depth edges."""
        if sigma <= 0:
            return depth
        
        # Create Gaussian kernel
        kernel_size = int(sigma * 6) | 1  # Ensure odd
        x = torch.arange(kernel_size, device=depth.device) - kernel_size // 2
        kernel_1d = torch.exp(-x.float() ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Separable blur
        depth = depth.unsqueeze(0).unsqueeze(0)
        
        # Horizontal
        kernel_h = kernel_1d.view(1, 1, 1, -1)
        depth = torch.nn.functional.conv2d(
            depth, kernel_h, padding=(0, kernel_size // 2)
        )
        
        # Vertical  
        kernel_v = kernel_1d.view(1, 1, -1, 1)
        depth = torch.nn.functional.conv2d(
            depth, kernel_v, padding=(kernel_size // 2, 0)
        )
        
        return depth.squeeze(0).squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
#                              NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceDepthMapGenerator": RadianceDepthMapGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDepthMapGenerator": "◎ Depth Map Generator",
}
