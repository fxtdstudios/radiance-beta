import torch
import numpy as np
import sys
import os

# Add parent directory to path to import nodes_denoise
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes_denoise import RadianceDenoise

def test_denoise_rgb():
    node = RadianceDenoise()
    # Create random float32 image [1, 64, 64, 3]
    img = torch.rand((1, 64, 64, 3), dtype=torch.float32)
    
    # Run
    # d=5, sigmaColor=0.1, sigmaSpace=10.0
    out = node.denoise(img, 5, 0.1, 10.0)[0]
    
    assert out.shape == img.shape, f"Shape mismatch: {out.shape} != {img.shape}"
    assert out.dtype == torch.float32
    print("RGB Denoise OK")

def test_denoise_rgba():
    node = RadianceDenoise()
    # Create random float32 image [1, 64, 64, 4]
    img = torch.rand((1, 64, 64, 4), dtype=torch.float32)
    
    # Run
    out = node.denoise(img, 5, 0.1, 10.0)[0]
    
    assert out.shape == img.shape, f"Shape mismatch: {out.shape} != {img.shape}"
    assert out.dtype == torch.float32
    print("RGBA Denoise OK")

def test_denoise_gray():
    node = RadianceDenoise()
    # Create random float32 image [1, 64, 64, 1]
    img = torch.rand((1, 64, 64, 1), dtype=torch.float32)
    
    # Run
    out = node.denoise(img, 5, 0.1, 10.0)[0]
    
    assert out.shape == img.shape, f"Shape mismatch: {out.shape} != {img.shape}"
    assert out.dtype == torch.float32
    print("Gray Denoise OK")

if __name__ == "__main__":
    try:
        test_denoise_rgb()
        test_denoise_rgba()
        test_denoise_gray()
        print("All tests passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
