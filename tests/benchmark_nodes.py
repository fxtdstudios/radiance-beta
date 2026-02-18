import torch
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from radiance.nodes_filmgrain import FXTDFilmGrain
    from radiance.nodes_upscale import FXTDUpscaleBySize
except ImportError:
    # Fallback
    try:
        from nodes_filmgrain import FXTDFilmGrain
        from nodes_upscale import FXTDUpscaleBySize
    except ImportError:
        print("Could not import nodes. Please adjust python path.")
        sys.exit(1)

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # Create large dummy image (4K equivalent)
    # (1, 2160, 3840, 3) float32
    print("Creating 4K dummy input...")
    img = torch.rand((1, 2160, 3840, 3), dtype=torch.float32).to(device)
    
    # Benchmark Film Grain
    grain_node = FXTDFilmGrain()
    start_time = time.time()
    # intensity, scale, temperature, application_mode, use_gpu
    _ = grain_node.apply_grain(img.cpu(), 0.5, 1.0, 1.0, "Overlay", True) 
    # Note: passing CPU tensor to simulate real usage where data comes in. Node handles to(device).
    # Wait for GPU sync if needed
    if torch.cuda.is_available(): torch.cuda.synchronize()
    grain_time = time.time() - start_time
    print(f"FilmGrain (4K): {grain_time:.4f} seconds")
    
    # Benchmark Upscale (2x)
    upscale_node = FXTDUpscaleBySize()
    # scale_by, method, antialias
    # Input image needs to be channel last or channel first? Comfy is usually NHWC
    # but torch functions often expect NCHW. Let's check node implementation later if needed.
    # The node likely handles conversion.
    
    # Let's test a smaller image for upscale (HD -> 4K) to avoid OOM on weak GPUs during dev
    hd_img = torch.rand((1, 1080, 1920, 3), dtype=torch.float32)
    start_time = time.time()
    # upscale(image, width, height, method)
    _ = upscale_node.upscale(hd_img, 3840, 2160, "lanczos")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    upscale_time = time.time() - start_time
    print(f"Upscale (HD->4K): {upscale_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
