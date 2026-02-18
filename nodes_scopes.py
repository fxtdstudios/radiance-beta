"""
Radiance - Visual Analysis Scopes
Professional Waveform and Vectorscope monitors for color grading.
"""

import torch
import numpy as np
import cv2
from typing import Tuple

class RadianceWaveform:
    """
    Visualize image exposure and channel balance using a Waveform monitor.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["RGB (Overlay)", "Luma (Y)", "Parade (RGB)"], {"default": "RGB (Overlay)"}),
                "size": ("INT", {"default": 512, "min": 256, "max": 2048}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("waveform_image",)
    FUNCTION = "generate_waveform"
    CATEGORY = "FXTD Studios/Radiance/Analyze"
    DESCRIPTION = "Visualize image exposure and channel balance using a Waveform monitor."

    def generate_waveform(self, image: torch.Tensor, mode: str, size: int, opacity: float):
        # Handle batch
        results = []
        for i in range(image.shape[0]):
            # Convert to numpy (H, W, 3), 0-1 float
            img = image[i].cpu().numpy()
            h, w, c = img.shape
            
            # Resize source width to scope size for performance if too large
            if w > size:
                scale = size / w
                new_h = int(h * scale)
                img = cv2.resize(img, (size, new_h), interpolation=cv2.INTER_AREA)
                h, w, c = img.shape
            
            # Initialize scope image (black)
            scope = np.zeros((256, w, 3), dtype=np.uint8)
            
            # Prepare data
            # Y-coordinate = intensity * 255 (inverted because image 0 is top)
            # X-coordinate = column index
            
            if mode == "Luma (Y)":
                # Calculate Luma: 0.2126 R + 0.7152 G + 0.0722 B
                luma = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722)
                luma = (luma * 255).astype(np.uint8) # [H, W] content is Intensity (0-255)
                
                # We need to map this to the scope:
                # For each column x, for each pixel y in image, plot point at (x, 255-intensity)
                
                # Doing this purely with loops is slow. 
                # Optimization: 2D Histogram-like approach?
                # Or scatter plot.
                
                # Let's try a fast scatter using numpy indexing.
                # X indices: broadcasted
                x_indices = np.tile(np.arange(w), (h, 1))
                y_values = 255 - luma # Invert so 255 (bright) is at top (row 0)
                
                # We can accumulate exposure.
                # However, drawing is easier.
                
                # Optimized Method:
                # 1. Flatten arrays
                x_flat = x_indices.flatten()
                y_flat = y_values.flatten()
                
                # 2. Bin counts (2d histogram)
                # scope shape (256, w)
                heatmap, _, _ = np.histogram2d(y_flat, x_flat, bins=[256, w], range=[[0, 256], [0, w]])
                
                # Normalize heatmap for display intensity
                heatmap = heatmap / (heatmap.max() * 0.1 + 1e-5) # Boost visibility
                heatmap = np.clip(heatmap * 255 * opacity, 0, 255).astype(np.uint8)
                
                # Stack to RGB (Grayscale/Green phosphor look)
                scope = np.stack([heatmap, heatmap, heatmap], axis=-1)
                
            elif mode == "RGB (Overlay)":
                scope_r = np.zeros((256, w), dtype=np.float32)
                scope_g = np.zeros((256, w), dtype=np.float32)
                scope_b = np.zeros((256, w), dtype=np.float32)
                
                x_indices = np.tile(np.arange(w), (h, 1)).flatten()
                
                for chan, scope_c in zip([0, 1, 2], [scope_r, scope_g, scope_b]):
                    val_flat = 255 - (img[..., chan] * 255).astype(np.uint8).flatten()
                    dt, _, _ = np.histogram2d(val_flat, x_indices, bins=[256, w], range=[[0, 256], [0, w]])
                    dt = dt / (dt.max() * 0.1 + 1e-5)
                    np.copyto(scope_c, dt) # store
                
                # Combine
                scope_r = np.clip(scope_r * 255 * opacity, 0, 255).astype(np.uint8)
                scope_g = np.clip(scope_g * 255 * opacity, 0, 255).astype(np.uint8)
                scope_b = np.clip(scope_b * 255 * opacity, 0, 255).astype(np.uint8)
                
                scope = np.stack([scope_r, scope_g, scope_b], axis=-1)
                
            elif mode == "Parade (RGB)":
                # Divide width into 3 sections
                section_w = w // 3
                # We might need to resize scope if w is not divisible or too small
                
                # Instead, simpler approach: Generate 3 narrow scopes and concat
                # Or just put them side by side in the `size` width
                
                sub_w = size // 3
                if sub_w < 1: sub_w = 1
                
                # Resize image width to sub_w * 3? 
                # Let's resize input to (sub_w, h)
                img_small = cv2.resize(image[i].cpu().numpy(), (sub_w, h), interpolation=cv2.INTER_AREA)
                
                scope_final = np.zeros((256, sub_w * 3, 3), dtype=np.uint8)
                
                x_indices = np.tile(np.arange(sub_w), (h, 1)).flatten()
                
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # R, G, B output tint
                
                for idx, (chan, color) in enumerate(zip([0, 1, 2], colors)):
                     # 0=R, 1=G, 2=B input
                     val_flat = 255 - (img_small[..., chan] * 255).astype(np.uint8).flatten()
                     dt, _, _ = np.histogram2d(val_flat, x_indices, bins=[256, sub_w], range=[[0, 256], [0, sub_w]])
                     
                     dt = dt / (dt.max() * 0.1 + 1e-5)
                     dt = np.clip(dt * 255 * opacity, 0, 255).astype(np.uint8)
                     
                     # Tint it
                     # Only put in correct channel? Parade typically shows R in Red, G in Green...
                     # Or white in different slots. Let's do colored.
                     
                     rgb_slice = np.zeros((256, sub_w, 3), dtype=np.uint8)
                     if chan == 0: rgb_slice[..., 0] = dt  # Red
                     elif chan == 1: rgb_slice[..., 1] = dt # Green
                     elif chan == 2: rgb_slice[..., 2] = dt # Blue
                     
                     scope_final[:, idx*sub_w:(idx+1)*sub_w, :] = rgb_slice
                
                scope = scope_final

            # Add grid lines (25%, 50%, 75%)
            # 256 height
            # 0% = 255, 100% = 0
            grid_color = (60, 60, 60)
            cv2.line(scope, (0, 64), (scope.shape[1], 64), grid_color, 1) # 75%
            cv2.line(scope, (0, 128), (scope.shape[1], 128), grid_color, 1) # 50%
            cv2.line(scope, (0, 192), (scope.shape[1], 192), grid_color, 1) # 25%

            # Resize to final requested size if needed (Parade was already handled, others match W)
            if scope.shape[1] != size:
                scope = cv2.resize(scope, (size, 256), interpolation=cv2.INTER_LINEAR)

            # Convert to float tensor
            scope_tensor = torch.from_numpy(scope.astype(np.float32) / 255.0)
            results.append(scope_tensor)
            
        return (torch.stack(results),)


class RadianceVectorscope:
    """
    Visualize color saturation and hue distribution (U/V plot).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "skin_tone_line": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("vectorscope_image",)
    FUNCTION = "generate_vectorscope"
    CATEGORY = "FXTD Studios/Radiance/Analyze"
    DESCRIPTION = "Visualize color saturation and hue distribution (U/V plot)."

    def generate_vectorscope(self, image: torch.Tensor, size: int, opacity: float, skin_tone_line: bool):
        results = []
        for i in range(image.shape[0]):
            img = image[i].cpu().numpy()
            
            # Downscale for performance if insanely huge
            if img.shape[0] * img.shape[1] > 512*512:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

            # 1. Convert to YUV (Bt.601 or 709? OpenCV uses 601 usually)
            # OpenCV RGB -> YCrCb
            # Y (Luma), Cr (R-Y), Cb (B-Y)
            # U = Cb (x-axis), V = Cr (y-axis)
            
            # img is 0-1 float RGB
            img_u8 = (img * 255).astype(np.uint8)
            yuv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb)
            
            # Extract Cb (U) and Cr (V)
            # Y=0, Cr=1, Cb=2
            cr = yuv[..., 1].flatten()
            cb = yuv[..., 2].flatten()
            
            # 2. Plot 2D Histogram
            # Ranges: 0-255. Center at 128,128.
            hist, _, _ = np.histogram2d(cr, cb, bins=256, range=[[0, 256], [0, 256]])
            
            # 3. Process Histogram to Image
            # Hist axes are (row=Cr, col=Cb) -> (y=V, x=U)
            # Standard Vectorscope: U on X, V on Y?
            # Actually standard: B-Y (U/Cb) on X, R-Y (V/Cr) on Y.
            # So X = Cb, Y = Cr.
            # But histogram2d(y, x). So histogram2d(cr, cb) puts cr on rows(Y), cb on cols(X). Correct.
            
            # Flip Y axis (Cr) because 0 is top
            # Cr: 0..255. 240 is Red (Positive). 16 is Cyan.
            # Standard scope: Red/Magenta up top.
            # So we likely need to invert row order.
            hist = np.flipud(hist)
            
            # Normalize
            hist = hist / (hist.max() * 0.1 + 1e-5)
            hist = np.clip(hist * 255 * opacity, 0, 255).astype(np.uint8)
            
            # Colorize? Or typical Green/Cyan phosphor.
            scope = np.zeros((256, 256, 3), dtype=np.uint8)
            scope[..., 1] = hist # Green channel
            scope[..., 2] = (hist * 0.5).astype(np.uint8) # Slight cyan tint
            scope[..., 0] = (hist * 0.2).astype(np.uint8)
            
            # 4. Draw Graticule
            center = (128, 128)
            
            # Circle 75% Saturation
            # (Rough approx for digital video)
            cv2.circle(scope, center, 128, (40, 40, 40), 1) # Outer limit
            cv2.circle(scope, center, 96, (60, 60, 60), 1)  # 75% safe area
            
            # Crosshair
            cv2.line(scope, (0, 128), (255, 128), (40, 40, 40), 1)
            cv2.line(scope, (128, 0), (128, 255), (40, 40, 40), 1)
            
            # Skin Tone Line (I-Line)
            # Typically at angle ~123 degrees? 
            # In CbCr plane: Cb negative (Blue-), Cr positive (Red+). Upper Left quadrant.
            if skin_tone_line:
                # skin tone roughly: Cr= high, Cb= low.
                # Angle approx 105 to 120 degrees from positive X axis? 
                # Nope, standard vectorscope "skin line" acts at [R-Y]+ axis rotated.
                # Let's draw a line towards Top-Left.
                end_pt = (128 - 70, 128 - 90) # approx
                cv2.line(scope, center, end_pt, (100, 80, 60), 1)
            
            # Color Targets (Optional ticks: R, Mg, B, Cy, G, Yl)
            
            # Resize to output size
            if size != 256:
                scope = cv2.resize(scope, (size, size), interpolation=cv2.INTER_LINEAR)
                
            results.append(torch.from_numpy(scope.astype(np.float32) / 255.0))
            
        return (torch.stack(results),)

NODE_CLASS_MAPPINGS = {
    "RadianceWaveform": RadianceWaveform,
    "RadianceVectorscope": RadianceVectorscope,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceWaveform": "◎ Radiance Waveform",
    "RadianceVectorscope": "◎ Radiance Vectorscope",
}
