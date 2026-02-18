import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory to path to allow importing radiance
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) # Adjust to d:\Space\ComfyUI\custom_nodes
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import unittest
import torch
import numpy as np

# Import via package name 'radiance' assuming it's in custom_nodes/radiance
# But since we are inside the package structure, we might need to adjust
try:
    from radiance.hdr.processing import HDRExposureBlend
    from radiance.hdr.vae import RadianceVAEEncode
except ImportError:
    # Fallback for local run
    sys.path.append(os.path.dirname(current_dir)) # Add d:\Space\ComfyUI\custom_nodes\radiance
    from hdr.processing import HDRExposureBlend
    from hdr.vae import RadianceVAEEncode

class TestHDRMath(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        
    def test_mertens_fusion_shapes(self):
        """Verify Mertens fusion returns correct shapes and range."""
        processor = HDRExposureBlend()
        
        # Create dummy bracketed exposures (B, H, W, C)
        # 3 exposures: -2ev, 0ev, +2ev
        low = torch.rand((1, 64, 64, 3)).float() * 0.1  # Dark
        high = torch.rand((1, 64, 64, 3)).float() * 5.0 # Bright
        
        # Test Mertens
        result, mask, info = processor.blend_exposures(
            low, high, 
            blend_method="Mertens Fusion",
            exposure_offset_low=-2.0,
            exposure_offset_high=2.0
        )
        
        self.assertEqual(result.shape, (1, 64, 64, 3))
        self.assertEqual(mask.shape, (1, 64, 64, 3))
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(isinstance(info, str))

    def test_soft_clip_math(self):
        """Verify soft clip tanh behavior."""
        encoder = RadianceVAEEncode()
        
        # Create gradient 0.0 -> 2.0
        data = torch.linspace(0.0, 2.0, 100).reshape(1, 100, 1, 1)
        
        # 1. Hard clip (baseline)
        hard = torch.clamp(data, 0, 1)
        
        # 2. Soft clip
        # knee=0.85, max=1.0
        soft = encoder._soft_clip(data, max_val=1.0, knee=0.85)
        
        # Check values below knee are linear
        below_knee = data < 0.85
        self.assertTrue(torch.allclose(data[below_knee], soft[below_knee], atol=1e-6))
        
        # Check values above knee are compressed but < 1.0
        above_knee = data > 0.85
        self.assertTrue(torch.all(soft[above_knee] < 1.0 + 1e-6))
        
        # Check smoothness (monotonicity)
        # Differences between adjacent elements should be positive
        diffs = soft.flatten()[1:] - soft.flatten()[:-1]
        self.assertTrue(torch.all(diffs >= 0))

    def test_vae_tiling_logic(self):
        """Verify tiling logic calculates correct coordinates."""
        encoder = RadianceVAEEncode()
        
        # Simulate 100x100 image, tile size 50, overlap 10
        # This is a logic test for the internal method _tiled_vae_encode
        # Since _tiled_vae_encode requires a VAE model, we'll mock the VAE
        
        class MockVAE:
            def encode(self, pixels):
                # Mock return: downscale by 8
                b, h, w, c = pixels.shape
                return {"samples": torch.zeros((b, 4, h//8, w//8))}
        
        pixels = torch.zeros((1, 128, 128, 3))
        vae = MockVAE()
        
        # Should run without error
        try:
            result = encoder._tiled_vae_encode(pixels, vae, tile_size=64, overlap=16)
        except Exception as e:
            self.fail(f"_tiled_vae_encode raised exception: {e}")
            
        self.assertEqual(result["samples"].shape, (1, 4, 16, 16)) # 128/8 = 16

if __name__ == '__main__':
    unittest.main()
