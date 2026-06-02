import torch
import torch.nn.functional as F
import numpy as np
import logging
import json

logger = logging.getLogger("radiance.vfx.inpaint")

class RadianceHDRCrop:
    """
    ◎ Radiance HDR Crop
    
    Crops around a mask with custom padding context. Auto-aligns crop dimensions
    to multiples of 16 to guarantee compatibility with Wan / LTX video architectures.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "context_padding": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "force_multiple": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STITCHER_DATA")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "stitcher_data")
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Inpainting"

    def apply(self, image: torch.Tensor, mask: torch.Tensor, context_padding: float, force_multiple: int):
        B, H, W, C = image.shape
        device = image.device
        
        # Ensure mask is [B, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(B, -1, -1)
            
        # 1. Compute union bounding box across frames (for stable video cropping)
        # Find active pixels
        active_pixels = (mask > 0.05).nonzero()
        
        if active_pixels.numel() == 0:
            # Fallback to full image if mask is empty
            ymin, xmin, ymax, xmax = 0, 0, H - 1, W - 1
        else:
            # Active coords (ignoring batch index for union)
            y_coords = active_pixels[:, 1]
            x_coords = active_pixels[:, 2]
            
            ymin, ymax = y_coords.min().item(), y_coords.max().item()
            xmin, xmax = x_coords.min().item(), x_coords.max().item()
            
        # Bounding box center and size
        box_h = ymax - ymin + 1
        box_w = xmax - xmin + 1
        cy = ymin + box_h // 2
        cx = xmin + box_w // 2
        
        # Expand box by context padding
        new_h = int(box_h * context_padding)
        new_w = int(box_w * context_padding)
        
        # 2. Force multiples of 16 (or custom VAE downscale factors)
        if force_multiple > 1:
            new_h = ((new_h + force_multiple - 1) // force_multiple) * force_multiple
            new_w = ((new_w + force_multiple - 1) // force_multiple) * force_multiple
            
        # Clamp dims to original frame boundaries
        new_h = min(new_h, H)
        new_w = min(new_w, W)
        
        # Calculate new crop box coords centered around original bbox center
        ymin_new = max(0, cy - new_h // 2)
        xmin_new = max(0, cx - new_w // 2)
        
        # Adjust if box hits bottom/right boundaries
        if ymin_new + new_h > H:
            ymin_new = H - new_h
        if xmin_new + new_w > W:
            xmin_new = W - new_w
            
        ymax_new = ymin_new + new_h
        xmax_new = xmin_new + new_w
        
        # 3. Crop batch
        cropped_img = image[:, ymin_new:ymax_new, xmin_new:xmax_new, :]
        cropped_mask = mask[:, ymin_new:ymax_new, xmin_new:xmax_new]
        
        # Package stitcher coordinates metadata
        stitcher_data = {
            "ymin": ymin_new,
            "xmin": xmin_new,
            "ymax": ymax_new,
            "xmax": xmax_new,
            "h_orig": H,
            "w_orig": W,
            "h_crop": new_h,
            "w_crop": new_w
        }
        
        logger.info(f"[HDR Crop] Cropped region: [{ymin_new}:{ymax_new}, {xmin_new}:{xmax_new}] (Wan-safe alignment).")
        return (cropped_img, cropped_mask, stitcher_data)


class RadianceHDRStitch:
    """
    ◎ Radiance HDR Stitch
    
    Stitches inpainted crops seamlessly back into the original plate.
    Uses ACEScg Linear Laplacian Pyramids and Gaussian edge-aware blending
    to eliminate color seams without dynamic range clamping.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "cropped_mask": ("MASK",),
                "stitcher_data": ("STITCHER_DATA",),
                "blend_mode": (["Linear_Laplacian", "Linear_Gaussian", "Standard"], {"default": "Linear_Laplacian"}),
                "feather_radius": ("INT", {"default": 16, "min": 0, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("stitched_image", "stitch_blend_mask")
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Inpainting"

    def apply(self, original_image: torch.Tensor, cropped_image: torch.Tensor, cropped_mask: torch.Tensor, stitcher_data: dict, blend_mode: str, feather_radius: int):
        B, H, W, C = original_image.shape
        device = original_image.device
        
        ymin = stitcher_data["ymin"]
        xmin = stitcher_data["xmin"]
        ymax = stitcher_data["ymax"]
        xmax = stitcher_data["xmax"]
        h_crop = stitcher_data["h_crop"]
        w_crop = stitcher_data["w_crop"]
        
        # Bring cropped mask to B, H, W, C shape for multiplying
        if cropped_mask.dim() == 2:
            cropped_mask = cropped_mask.unsqueeze(0)
        
        # Build raw mask in original frame resolution
        orig_mask = torch.zeros((B, H, W), device=device)
        orig_mask[:, ymin:ymax, xmin:xmax] = cropped_mask.clamp(0.0, 1.0)
        
        # Feather/Gaussian blur the blending mask
        if feather_radius > 0:
            k = feather_radius * 2 + 1
            pad = feather_radius
            # Reshape for 2D pooling / blur
            orig_mask_bchw = orig_mask.unsqueeze(1)
            blurred_mask = F.avg_pool2d(orig_mask_bchw, k, stride=1, padding=pad)
            blend_mask = blurred_mask.squeeze(1).clamp(0.0, 1.0)
        else:
            blend_mask = orig_mask
            
        blend_mask_3d = blend_mask.unsqueeze(-1) # B, H, W, 1
        
        # Reconstruct crop into full frame
        full_cropped_img = original_image.clone()
        full_cropped_img[:, ymin:ymax, xmin:xmax, :] = cropped_image
        
        if blend_mode == "Standard":
            # Direct paste
            stitched = original_image * (1.0 - orig_mask.unsqueeze(-1)) + full_cropped_img * orig_mask.unsqueeze(-1)
        elif blend_mode == "Linear_Gaussian":
            # Simple soft blending in linear space
            stitched = original_image * (1.0 - blend_mask_3d) + full_cropped_img * blend_mask_3d
        else:
            # Linear Laplacian Pyramid Blending in pure PyTorch (unclamped scene-linear)
            # Standard pyramid layers
            levels = 3
            
            # Setup image pyrs
            gp_orig = [original_image.permute(0, 3, 1, 2)]
            gp_crop = [full_cropped_img.permute(0, 3, 1, 2)]
            gp_mask = [blend_mask.unsqueeze(1)]
            
            for l in range(levels - 1):
                gp_orig.append(F.avg_pool2d(gp_orig[-1], 3, stride=2, padding=1))
                gp_crop.append(F.avg_pool2d(gp_crop[-1], 3, stride=2, padding=1))
                gp_mask.append(F.avg_pool2d(gp_mask[-1], 3, stride=2, padding=1))
                
            # Build Laplacian Pyramids
            lp_orig = []
            lp_crop = []
            
            for l in range(levels - 1):
                # Upsample next level to subtract from current
                size = gp_orig[l].shape[2:]
                up_orig = F.interpolate(gp_orig[l+1], size=size, mode="bilinear", align_corners=True)
                up_crop = F.interpolate(gp_crop[l+1], size=size, mode="bilinear", align_corners=True)
                
                lp_orig.append(gp_orig[l] - up_orig)
                lp_crop.append(gp_crop[l] - up_crop)
                
            # Last levels are standard Gaussians
            lp_orig.append(gp_orig[-1])
            lp_crop.append(gp_crop[-1])
            
            # Fuse Laplacian pyramids using mask layers
            lp_fused = []
            for l in range(levels):
                mask_layer = gp_mask[l]
                fused = lp_orig[l] * (1.0 - mask_layer) + lp_crop[l] * mask_layer
                lp_fused.append(fused)
                
            # Reconstruct from fused Laplacian pyramid
            recon = lp_fused[-1]
            for l in reversed(range(levels - 1)):
                size = lp_fused[l].shape[2:]
                recon_up = F.interpolate(recon, size=size, mode="bilinear", align_corners=True)
                recon = recon_up + lp_fused[l]
                
            stitched = recon.squeeze(0).permute(0, 2, 3, 1)
            stitched = stitched.clamp(min=0.0) # scene-linear float output (unclamped max!)
            
        logger.info(f"[HDR Stitch] Composited crop back into frame using {blend_mode} (Feather: {feather_radius}).")
        return (stitched, blend_mask)


class RadianceTemporalStitchStabilizer:
    """
    ◎ Radiance Temporal Stitch Stabilizer
    
    Eliminates video mask edge jitter/popping by running a high-precision 1D temporal Gaussian filter
    along the sequence timeline for each mask pixel.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "temporal_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("stabilized_masks",)
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Inpainting"

    def apply(self, masks: torch.Tensor, temporal_sigma: float):
        B, H, W = masks.shape
        device = masks.device
        
        if temporal_sigma <= 0.1 or B <= 1:
            return (masks,)
            
        # 1D Gaussian kernel along the time axis (dimension 0)
        radius = int(temporal_sigma * 3)
        radius = max(1, radius)
        kernel_size = radius * 2 + 1
        
        # Build 1D Gaussian kernel
        t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (t / temporal_sigma)**2)
        kernel = kernel / kernel.sum()
        
        # Reshape masks for 1D convolution
        # Shape becomes [1, H*W, B] where H*W are channels, batch is 1, and B is timeline length
        masks_flat = masks.view(B, H * W).permute(1, 0).unsqueeze(0) # [1, HW, B]
        
        # Pad along timeline boundaries to prevent clipping
        padded = F.pad(masks_flat, (radius, radius), mode="replicate")
        
        # Conv1d
        kernel_1d = kernel.view(1, 1, kernel_size).expand(H * W, 1, -1)
        smoothed = F.conv1d(padded, kernel_1d, groups=H * W) # [1, HW, B]
        
        # Permute back to [B, H, W]
        output = smoothed.squeeze(0).permute(1, 0).view(B, H, W).clamp(0.0, 1.0)
        
        logger.info(f"[Temporal Stitch Stabilizer] Applied temporal smoothing filter (Sigma: {temporal_sigma}).")
        return (output,)
