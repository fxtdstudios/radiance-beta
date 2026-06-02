import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger("radiance.vfx.plate")

class RadianceHDRGrainMatcher:
    """
    ◎ Radiance HDR Grain Matcher
    
    Extracts high-frequency film grain from a reference plate in log2 exposure space
    and maps it matching the target image's exposure range to prevent highlight burnout.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE",),
                "reference": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 15, "step": 2}),
                "r_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "g_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "b_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grained_image",)
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Plate Prep"

    def apply(self, target: torch.Tensor, reference: torch.Tensor, intensity: float, kernel_size: int, r_gain: float, g_gain: float, b_gain: float):
        B, H, W, C = target.shape
        ref_B = reference.shape[0]
        device = target.device
        
        # Bring inputs to BCHW and extract log2 exposure space
        # log2 prevents grain extraction from being influenced by absolute light levels
        eps = 1e-4
        target_log = torch.log2(target.clamp(min=0.0) + eps)
        
        # We loop over batch and extract grain dynamically from reference frames
        # If reference has fewer frames, we wrap around or cycle
        grain_accum = []
        for i in range(B):
            ref_idx = i % ref_B
            ref_frame = reference[ref_idx].permute(2, 0, 1).unsqueeze(0) # 1, C, H, W
            ref_log = torch.log2(ref_frame.clamp(min=0.0) + eps)
            
            # Box-filter smooth
            pad = kernel_size // 2
            ref_smooth = F.avg_pool2d(ref_log, kernel_size, stride=1, padding=pad)
            
            # High-frequency grain
            grain = ref_log - ref_smooth
            grain_accum.append(grain)
            
        grain_tensor = torch.cat(grain_accum, dim=0) # B, C, H, W
        grain_tensor = grain_tensor.permute(0, 2, 3, 1) # B, H, W, C
        
        # Apply channel gains
        gains = torch.tensor([r_gain, g_gain, b_gain], device=device).view(1, 1, 1, 3)
        grained_log = target_log + (grain_tensor * intensity * gains)
        
        # Convert back from log2 space
        grained_img = torch.pow(2.0, grained_log) - eps
        grained_img = grained_img.clamp(min=0.0)
        
        logger.info(f"[HDR Grain Matcher] Extracted and re-applied grain (Intensity: {intensity})")
        return (grained_img,)


class RadianceSubpixelStabilizer:
    """
    ◎ Radiance Subpixel Stabilizer
    
    Stabilizes an image sequence batch to a selected reference anchor frame.
    Calculates displacements using high-precision sub-pixel FFT Phase Correlation in pure PyTorch.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "anchor_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "max_shift": ("INT", {"default": 64, "min": 4, "max": 512, "step": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("stabilized_sequence", "displacements_xy")
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Plate Prep"

    def apply(self, image: torch.Tensor, anchor_frame: int, max_shift: int):
        B, H, W, C = image.shape
        device = image.device
        
        anchor_idx = min(anchor_frame, B - 1)
        # Target reference anchor frame (convert to grayscale for correlation)
        ref_frame = image[anchor_idx].mean(dim=-1) # H, W
        
        # Precompute window function (Hann window) to minimize FFT boundary leakage
        hann_y = torch.hann_window(H, device=device).unsqueeze(1)
        hann_x = torch.hann_window(W, device=device).unsqueeze(0)
        window = hann_y * hann_x
        
        ref_windowed = ref_frame * window
        F_ref = torch.fft.fft2(ref_windowed)
        
        stabilized = torch.zeros_like(image)
        displacements = torch.zeros((B, H, W, 2), device=device) # stores dx, dy maps
        
        for i in range(B):
            if i == anchor_idx:
                stabilized[i] = image[i]
                continue
                
            cur_frame = image[i].mean(dim=-1)
            cur_windowed = cur_frame * window
            F_cur = torch.fft.fft2(cur_windowed)
            
            # Cross-power spectrum
            # R = (F_ref * conj(F_cur)) / |F_ref * conj(F_cur)|
            cross = F_ref * torch.conj(F_cur)
            R = cross / (torch.abs(cross) + 1e-12)
            
            # Inverse FFT to find peak
            r = torch.fft.ifft2(R).real
            
            # Find integer peak
            max_val = torch.max(r)
            idx = (r == max_val).nonzero()[0]
            dy_int, dx_int = idx[0].item(), idx[1].item()
            
            # Wrap around coordinates
            if dy_int > H // 2:
                dy_int -= H
            if dx_int > W // 2:
                dx_int -= W
                
            # Subpixel refinement using 3x3 parabolic interpolation around peak
            # f(x) = A*x^2 + B*x + C
            # Peak center is x_c = -B / 2A
            dy_sub, dx_sub = float(dy_int), float(dx_int)
            
            try:
                # 3x3 neighborhood around peak (safe wrap around index check)
                y_indices = [(dy_int + offset) % H for offset in [-1, 0, 1]]
                x_indices = [(dx_int + offset) % W for offset in [-1, 0, 1]]
                
                # Retrieve neighborhood values
                val_n1_0 = r[y_indices[0], x_indices[1]].item() # y - 1
                val_p1_0 = r[y_indices[2], x_indices[1]].item() # y + 1
                val_0_0  = r[y_indices[1], x_indices[1]].item() # center
                
                val_0_n1 = r[y_indices[1], x_indices[0]].item() # x - 1
                val_0_p1 = r[y_indices[1], x_indices[2]].item() # x + 1
                
                # Parabolic peak estimation
                denom_y = val_n1_0 + val_p1_0 - 2 * val_0_0
                if abs(denom_y) > 1e-5:
                    dy_sub += (val_n1_0 - val_p1_0) / (2 * denom_y)
                    
                denom_x = val_0_n1 + val_0_p1 - 2 * val_0_0
                if abs(denom_x) > 1e-5:
                    dx_sub += (val_0_n1 - val_0_p1) / (2 * denom_x)
            except Exception:
                pass
                
            # Clamp maximum shifts to prevent wild drift on noise
            dx = max(min(dx_sub, float(max_shift)), -float(max_shift))
            dy = max(min(dy_sub, float(max_shift)), -float(max_shift))
            
            # Translate current frame using grid_sample
            # Grid maps [-1, 1] range. Translate delta by pixels / size
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing="ij"
            )
            
            grid_shift_x = grid_x - (dx / (W / 2.0))
            grid_shift_y = grid_y - (dy / (H / 2.0))
            
            grid = torch.stack([grid_shift_x, grid_shift_y], dim=-1).unsqueeze(0)
            
            img_chw = image[i].permute(2, 0, 1).unsqueeze(0) # 1, C, H, W
            warp = F.grid_sample(img_chw, grid, mode="bicubic", padding_mode="border", align_corners=True)
            
            stabilized[i] = warp.squeeze(0).permute(1, 2, 0)
            
            # Store displacement mapping (diagnostic)
            displacements[i, ..., 0] = dx
            displacements[i, ..., 1] = dy
            
        logger.info(f"[Subpixel Stabilizer] Anchored sequence to frame {anchor_idx}.")
        return (stabilized, displacements)
