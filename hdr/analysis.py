
import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any

# Local imports
from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32

logger = logging.getLogger("radiance.hdr.analysis")

# ═══════════════════════════════════════════════════════════════════════════════
#                          ANALYSIS NODES
# ═══════════════════════════════════════════════════════════════════════════════

class HDRHistogram:
    """
    Analyze HDR image histogram with dynamic range statistics, clipping indicators, and stops visualization.
    """
    
    MODES = ["luminance", "rgb", "log_luminance"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "luminance"}),
                "show_clipping": ("BOOLEAN", {"default": True}),
                "stops_range": ("INT", {"default": 14, "min": 8, "max": 24}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("histogram", "stats")
    FUNCTION = "analyze"
    CATEGORY = "FXTD Studios/Radiance/Analyze"
    DESCRIPTION = "Analyze HDR image histogram with dynamic range statistics, clipping indicators, and stops visualization."
    
    def analyze(self, image: torch.Tensor, mode: str = "luminance",
                show_clipping: bool = True, stops_range: int = 14) -> Tuple[torch.Tensor, str]:
        
        logger.debug("HDRHistogram analyzing...")
        try:
            from PIL import Image, ImageDraw
            
            img = tensor_to_numpy_float32(image)
            if img.ndim == 4:
                img = img[0]  # Take first image in batch
            
            # Sanitize Input: Handle NaNs and Infs
            img = np.nan_to_num(img, nan=0.0, posinf=65504.0, neginf=-65504.0)
            
            # Calculate luminance
            if img.shape[-1] >= 3:
                luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            else:
                luma = img[..., 0]
            
            # Statistics
            min_val = float(np.min(img))
            max_val = float(np.max(img))
            mean_val = float(np.mean(img))
            
            # Dynamic range in stops
            eps = 1e-10
            dynamic_range = np.log2(max(max_val, eps) / max(min_val, eps))
            
            # Clipping analysis
            clip_low = float(np.sum(img <= 0) / img.size * 100)
            clip_high = float(np.sum(img >= 1) / img.size * 100)
            
            stats = f"""═══════════════════════════════════
            HDR IMAGE ANALYSIS
    ═══════════════════════════════════
    Min Value:     {min_val:.6f}
    Max Value:     {max_val:.6f}
    Mean Value:    {mean_val:.6f}
    Dynamic Range: {dynamic_range:.1f} stops
    ───────────────────────────────────
    Clipped Low:   {clip_low:.2f}%
    Clipped High:  {clip_high:.2f}%
    ═══════════════════════════════════"""
            
            # ═══════════════════════════════════════════════════════════════
            # CREATE HISTOGRAM WITH PIL (NO MATPLOTLIB)
            # ═══════════════════════════════════════════════════════════════
            logger.debug("HDRHistogram rendering with PIL...")
            
            # Canvas size
            hist_w, hist_h = 1000, 600
            bg_color = (26, 26, 46)  # Dark blue background
            
            # Create image
            hist_img = Image.new('RGB', (hist_w, hist_h), bg_color)
            draw = ImageDraw.Draw(hist_img)
            
            # Graph area
            margin = 60
            graph_w = hist_w - margin * 2
            graph_h = (hist_h - margin * 3) // 2
            graph_top1 = margin
            graph_top2 = margin + graph_h + margin
            
            # ── LINEAR HISTOGRAM ──
            draw.rectangle([margin - 5, graph_top1 - 5, margin + graph_w + 5, graph_top1 + graph_h + 5], 
                          fill=(22, 33, 62))
            
            # Compute histogram bins
            num_bins = 256
            bin_width = graph_w / num_bins
            
            if mode == "rgb" and img.shape[-1] >= 3:
                # RGB mode - draw each channel
                colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]
                for ch_idx, color in enumerate(colors):
                    channel = img[..., ch_idx].flatten()
                    channel = np.clip(channel, 0, max(1, max_val))
                    hist_vals, _ = np.histogram(channel, bins=num_bins, range=(0, max(1, max_val)))
                    max_count = max(hist_vals.max(), 1)
                    
                    for i, count in enumerate(hist_vals):
                        bar_h = int((count / max_count) * graph_h * 0.9)
                        x1 = margin + int(i * bin_width)
                        x2 = margin + int((i + 1) * bin_width)
                        y1 = graph_top1 + graph_h - bar_h
                        y2 = graph_top1 + graph_h
                        # Semi-transparent overlay
                        alpha_color = tuple(c // 2 for c in color)
                        draw.rectangle([x1, y1, x2, y2], fill=alpha_color)
            else:
                # Luminance mode
                hist_vals, _ = np.histogram(luma.flatten(), bins=num_bins, range=(0, max(1, max_val)))
                max_count = max(hist_vals.max(), 1)
                
                for i, count in enumerate(hist_vals):
                    bar_h = int((count / max_count) * graph_h * 0.9)
                    x1 = margin + int(i * bin_width)
                    x2 = margin + int((i + 1) * bin_width)
                    y1 = graph_top1 + graph_h - bar_h
                    y2 = graph_top1 + graph_h
                    draw.rectangle([x1, y1, x2, y2], fill=(233, 69, 96))
            
            # Clipping indicators
            if show_clipping:
                draw.line([(margin, graph_top1), (margin, graph_top1 + graph_h)], fill=(0, 255, 255), width=2)
                # White clipping at normalized 1.0
                white_x = margin + int(graph_w * min(1.0 / max(max_val, 1), 1.0))
                draw.line([(white_x, graph_top1), (white_x, graph_top1 + graph_h)], fill=(255, 255, 0), width=2)
            
            # Title
            draw.text((margin, graph_top1 - 25), "Linear Histogram", fill=(255, 255, 255))
            
            # ── LOG HISTOGRAM (STOPS) ──
            draw.rectangle([margin - 5, graph_top2 - 5, margin + graph_w + 5, graph_top2 + graph_h + 5], 
                          fill=(22, 33, 62))
            
            # Compute log histogram
            log_luma = np.log2(np.maximum(luma, eps))
            min_stop = -8
            max_stop = stops_range - 8
            log_bins = stops_range * 10
            
            log_hist, log_edges = np.histogram(log_luma.flatten(), bins=log_bins, range=(min_stop, max_stop))
            log_max = max(log_hist.max(), 1)
            log_bin_width = graph_w / log_bins
            
            for i, count in enumerate(log_hist):
                bar_h = int((count / log_max) * graph_h * 0.9)
                x1 = margin + int(i * log_bin_width)
                x2 = margin + int((i + 1) * log_bin_width)
                y1 = graph_top2 + graph_h - bar_h
                y2 = graph_top2 + graph_h
                draw.rectangle([x1, y1, x2, y2], fill=(15, 52, 96))
            
            # Middle gray marker (0 stops = 18% gray)
            mid_x = margin + int(graph_w * (-min_stop) / (max_stop - min_stop))
            draw.line([(mid_x, graph_top2), (mid_x, graph_top2 + graph_h)], fill=(233, 69, 96), width=2)
            
            # Title
            draw.text((margin, graph_top2 - 25), f"Log Histogram ({stops_range} stops)", fill=(255, 255, 255))
            
            # Bottom info
            draw.text((margin, hist_h - 30), f"DR: {dynamic_range:.1f} stops | Min: {min_val:.3f} | Max: {max_val:.3f}", 
                     fill=(180, 180, 200))
            
            # Convert to tensor
            hist_np = np.array(hist_img).astype(np.float32) / 255.0
            logger.debug("HDRHistogram done.")
            return (numpy_to_tensor_float32(hist_np), stats)
            
        except Exception as e:
            logger.error(f"HDRHistogram error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return RED ERROR IMAGE so user knows it failed
            if image.dim() == 4:
                B, H, W, C = image.shape
                err_img = torch.zeros((B, H, W, C), dtype=torch.float32)
                err_img[..., 0] = 1.0  # Red
            else:
                H, W, C = image.shape
                err_img = torch.zeros((H, W, C), dtype=torch.float32)
                err_img[..., 0] = 1.0  # Red
                
            return (err_img, f"Error: {str(e)}")
