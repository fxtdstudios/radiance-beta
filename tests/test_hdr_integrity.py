
import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Mock ComfyUI modules
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].get_output_directory.return_value = "."
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()

# Add custom_nodes directory to path to allow 'radiance' package import
# __file__ = radiance/tests/test_hdr_integrity.py
# .. = radiance
# ../.. = custom_nodes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import nodes to test using full package path
from radiance.image.upscale import RadianceProUpscale
from radiance.nodes_grade import RadianceGrade
from radiance.hdr.vae import RadianceVAEEncode, RadianceVAEDecode
from radiance.hdr.io import SaveImage16bit
try:
    from radiance.color_utils import linear_to_logc4
    HAS_COLOR_UTILS = True
except ImportError:
    HAS_COLOR_UTILS = False

class TestHDRIntegrity(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"

    def test_upscale_no_clamp(self):
        """Test that Upscaler preserves values > 1.0 in float mode."""
        upscaler = RadianceProUpscale()
        # Create HDR input: 64x64 pixel with value 10.0 (B, H, W, C)
        img = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 10.0
        
        # Test upscale 2x
        result = upscaler.upscale(img, scale_factor=2.0, preset="Custom", 
                                  method="nearest", output_bit_depth="32-bit Float")
        
        output_tensor = result[0]
        max_val = output_tensor.max().item()
        
        print(f"Upscale Output Max: {max_val}")
        self.assertGreater(max_val, 1.0, "Upscaler clamped HDR value to 1.0!")
        self.assertAlmostEqual(max_val, 10.0, delta=0.1, msg="Upscale altered magnitude excessively.")

    def test_grade_negative_gamma(self):
        """Test that Grade handles negative values with Gamma != 1.0."""
        grader = RadianceGrade()
        # Input -0.5 (e.g. lifted black / noise)
        img = torch.tensor([[[[-0.5, 0.0, 0.0]]]], dtype=torch.float32)
        
        # Gamma 2.0
        # Expected: sign(-0.5) * abs(-0.5)^0.5 ? No, Gamma 2.0 means power 1/2.0 in simple terms, 
        # but RadianceGrade implements: pow(x, 1/gamma).
        # So pow(0.5, 0.5) = 0.707. Sign preserved -> -0.707.
        
        result, _ = grader.grade(img, gamma_r=2.0, gamma_g=2.0, gamma_b=2.0)
        
        val = result[0, 0, 0, 0].item()
        print(f"Grade Gamma(2.0) on -0.5: {val}")
        
        self.assertLess(val, 0.0, "Grade clamped negative value to 0!")
        self.assertAlmostEqual(val, -0.7071, delta=0.001)

    def test_vae_encode_hdr_mode(self):
        """Test VAE Encode 'Compress (Log)' mode preserves HDR."""
        if not HAS_COLOR_UTILS:
            print("Skipping VAE test (color_utils missing)")
            return

        encoder = RadianceVAEEncode()
        
        # Mock VAE
        mock_vae = MagicMock()
        mock_vae.encode.side_effect = lambda x: x # Pass through
        
        # Input HDR: 10.0 Linear
        img = torch.tensor([[[[10.0, 0.0, 0.0]]]], dtype=torch.float32)
        
        # Mode: Compress (Log)
        # Should convert 10.0 -> LogC4(10.0) -> ~0.5 something
        result_pkg = encoder.encode(img, mock_vae, source_space="Linear",
                                    hdr_mode="Compress (Log)")
        latent = result_pkg[0] # {"samples": ...} ? No, encode returns (latent, alpha, metadata)
        # Check return type: (latent, alpha, str)
        # latent is whatever vae.encode returns, which we mocked as input tensor.
        
        # Wait, encoder.encode converts img -> linear -> logc4 -> vae.encode(img).
        # So latent will be the logc4 image.
        
        val = latent[0, 0, 0, 0].item() # B, H, W, C (Wait, VAE inputs usually BCHW or BHWC? 
        # Check encoder logic: "8. Encode ... if img.dim() == 3: unsqueeze... vae.encode(img)"
        # Standard Comfy VAE encode expects BHWC -> encode -> Latent.
        # But we mocked it to return input.
        # Input to encode is BHWC.
        
        print(f"VAE Encode 'Compress (Log)' Output: {val}")
        
        self.assertLess(val, 1.0, "VAE Encode did not compress HDR value!")
        self.assertGreater(val, 0.3, "VAE Encode compressed too much?")
        
        # Mode: Clip (SDR)
        result_pkg_sdr = encoder.encode(img, mock_vae, hdr_mode="Clip (SDR)", soft_clip=False)
        latent_sdr = result_pkg_sdr[0]
        val_sdr = latent_sdr[0, 0, 0, 0].item()
        
        print(f"VAE Encode 'Clip (SDR)' Output: {val_sdr}")
        self.assertAlmostEqual(val_sdr, 1.0, delta=0.001, msg="SDR mode should hard clamp to 1.0")

    def test_save_16bit_log(self):
        """Test SaveImage16bit logic for LogC4."""
        saver = SaveImage16bit()
        
        # Input 10.0 Linear
        img = torch.tensor([[[[10.0, 0.0, 0.0]]]], dtype=torch.float32)
        
        
        # Manually patch linear_to_logc4 into saver module logic
        saver_module = sys.modules[SaveImage16bit.__module__]
        if HAS_COLOR_UTILS and not hasattr(saver_module, 'linear_to_logc4'):
            saver_module.linear_to_logc4 = linear_to_logc4
            print(f"Patched linear_to_logc4 into {SaveImage16bit.__module__}")

        with patch('cv2.imwrite') as mock_write, patch('folder_paths.get_output_directory', return_value="."):
            saver.save(img, tonemap="Compress (LogC4)")
            
            # Check what was passed to imwrite
            # Args: filename, img
            args, _ = mock_write.call_args
            np_img_saved = args[1]
            
            # np_img_saved is uint16
            val_int = np_img_saved[0, 0, 0] # BGR?
            # 10.0 -> LogC4 -> ~0.57 -> * 65535 -> ~37000
            
            print(f"Save 16bit LogC4 Value: {val_int}")
            self.assertGreater(val_int, 30000)
            self.assertLess(val_int, 65535)

if __name__ == '__main__':
    unittest.main()
