
import torch
import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock folder_paths module
import sys
from unittest.mock import MagicMock
sys.modules["folder_paths"] = MagicMock()

from hdr.vae import RadianceVAEDecode

class MockVAE:
    def decode(self, latent):
        # Return a tensor with HDR values (super-white > 1.0, super-black < 0.0)
        # Latent shape (B, 4, H, W) -> Output (B, H*8, W*8, 3)
        b, c, h, w = latent.shape
        # Create a simple pattern: 
        # - Region 1: Super Black (-0.5)
        # - Region 2: Mid Gray (0.18)
        # - Region 3: Standard White (1.0)
        # - Region 4: Super White (5.0)
        
        # We'll just return a constant tensor for simplicity of checking range, 
        # or a simple gradient.
        # Let's make it 8x8 pixels (from 1x1 latent)
        out = torch.zeros((b, h*8, w*8, 3), dtype=torch.float32)
        out[..., 0] = -0.5 # Red channel: Super Black
        out[..., 1] = 0.5  # Green channel: Mid
        out[..., 2] = 5.0  # Blue channel: Super White
        return out

class TestVAEDecoding(unittest.TestCase):
    def test_vae_decode_no_clamp(self):
        print("\n--- Testing Radiance VAE Decode for Clamping ---")
        
        decoder = RadianceVAEDecode()
        mock_vae = MockVAE()
        
        # Latent input (dummy)
        latent = {"samples": torch.zeros((1, 4, 8, 8))} # 1 tile
        
        # Test 1: Linear Output, SDR Mode (Standard)
        # Should preserve full range if no tone mapping is applied
        out, _ = decoder.decode(
            latent, mock_vae, 
            target_space="Linear", 
            hdr_mode="Clip (SDR)", # Name implies clip, but let's see if it actually clips linear
            inverse_tonemap=False
        )
        print(f"Linear (SDR Mode): Min={out.min().item():.4f}, Max={out.max().item():.4f}")
        
        # Checking: In 'Clip (SDR)' mode, code path:
        # Step 3: target=Linear. 
        # if inverse_tonemap... (False)
        # if exposure... (0.0)
        # It does NOT seem to clamp explicitly in code unless I missed something?
        # WAIT. I saw `hdr_mode` logic in Encode, but in Decode?
        
        # Let's re-read the code for SDR Mode in Decode:
        # if hdr_mode == "Compress (Log)": ... else: tensor_srgb_to_linear(img)
        # Ah! `tensor_srgb_to_linear` assumes input IS sRGB. 
        # If the VAE output is *already* Linear (mock mock), treating it as sRGB and converting to Linear 
        # applies a gamma curve. 
        # sRGB->Linear: x^2.4. 
        # -0.5 -> sign preserved? utils says yes used sign * abs. 
        # 5.0 -> 5.0^2.4 ~ 47. 
        # So it transforms, but does NOT clamp.
        
        self.assertLess(out.min(), 0.0, "Should handle negatives")
        self.assertGreater(out.max(), 1.0, "Should handle super-whites")

        # Test 2: sRGB Output
        out_srgb, _ = decoder.decode(
            latent, mock_vae, 
            target_space="sRGB", 
            hdr_mode="Clip (SDR)"
        )
        print(f"sRGB (SDR Mode): Min={out_srgb.min().item():.4f}, Max={out_srgb.max().item():.4f}")
        # sRGB from Linear (if we assume VAE output was sRGB, then converted to Linear, then back to sRGB?)
        # Wait, if VAE output is sRGB:
        # 1. `tensor_srgb_to_linear` -> Linear
        # 2. `target_space="sRGB"` -> `tensor_linear_to_srgb` -> sRGB
        # Roundtrip should be roughly identity.
        # -0.5 -> Linear -> -0.5 (approx gamma) -> sRGB -> -0.5
        # 5.0 -> Linear -> 47 -> sRGB -> 5.0
        # Should be unclamped.
        self.assertLess(out_srgb.min(), 0.0)
        self.assertGreater(out_srgb.max(), 1.0)
        
        # Test 3: Passthrough (Unclamped) Mode
        out_pass, _ = decoder.decode(
            latent, mock_vae,
            target_space="Linear",
            hdr_mode="Passthrough (Unclamped)"
        )
        print(f"Linear (Passthrough): Min={out_pass.min().item():.4f}, Max={out_pass.max().item():.4f}")
        self.assertLess(out_pass.min(), 0.0)
        self.assertGreater(out_pass.max(), 1.0)

if __name__ == "__main__":
    unittest.main()
