
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import unittest
import torch
from unittest.mock import MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules["folder_paths"] = MagicMock()

from hdr.vae import RadianceVAEEncode, RadianceVAEDecode

class MockFluxVAE:
    """Mocks a Flux VAE which uses 16 channels in Latent space."""
    def encode(self, pixels):
        # Input: (B, H, W, 3)
        # Output: {"samples": (B, 16, H/8, W/8)}
        b, h, w, c = pixels.shape
        latent_h = h // 8
        latent_w = w // 8
        # Mock 16 channels
        return {"samples": torch.randn((b, 16, latent_h, latent_w))}

    def decode(self, latent):
        # Input: (B, 16, H, W)
        # Output: (B, H*8, W*8, 3)
        b, c, h, w = latent.shape
        pixel_h = h * 8
        pixel_w = w * 8
        return torch.zeros((b, pixel_h, pixel_w, 3))

class TestFluxCompatibility(unittest.TestCase):
    def test_flux_encode_decode(self):
        print("\n--- Testing Flux (16-channel) Compatibility ---")
        
        encoder = RadianceVAEEncode()
        decoder = RadianceVAEDecode()
        flux_vae = MockFluxVAE()
        
        # 1. Test Encode
        # Input image 512x512
        pixels = torch.rand((1, 512, 512, 3), dtype=torch.float32)
        
        print("Encoding 512x512 image with Mock Flux VAE...")
        latent, alpha, meta = encoder.encode(pixels, flux_vae, tile_size="None")
        
        self.assertEqual(latent["samples"].shape[1], 16, "Latent should have 16 channels")
        self.assertEqual(latent["samples"].shape[2], 64, "Latent H should be 64 (512/8)")
        self.assertEqual(latent["samples"].shape[3], 64, "Latent W should be 64 (512/8)")
        print("Encode successful. Latent shape:", latent["samples"].shape)
        
        # 2. Test Decode
        print("Decoding 16-channel latent...")
        decoded_img, meta_dec = decoder.decode(latent, flux_vae, tile_size="None")
        
        self.assertEqual(decoded_img.shape[1], 512, "Decoded H should be 512")
        self.assertEqual(decoded_img.shape[2], 512, "Decoded W should be 512")
        self.assertEqual(decoded_img.shape[3], 3, "Decoded image should be RGB (3 channels)")
        print("Decode successful. Image shape:", decoded_img.shape)
        
        # 3. Test Tiled Encode with 16 channels
        print("Testing TILED Encode with Flux...")
        # Make a larger image to force tiling
        large_pixels = torch.rand((1, 1024, 1024, 3), dtype=torch.float32)
        latent_tiled, _, _ = encoder.encode(large_pixels, flux_vae, tile_size="512")
        
        self.assertEqual(latent_tiled["samples"].shape[1], 16, "Tiled Latent should still have 16 channels")
        print("Tiled Encode successful. Latent shape:", latent_tiled["samples"].shape)

        # 4. Test Tiled Decode with 16 channels
        print("Testing TILED Decode with Flux...")
        decoded_tiled, _ = decoder.decode(latent_tiled, flux_vae, tile_size="512")
        self.assertEqual(decoded_tiled.shape, large_pixels.shape)
        print("Tiled Decode successful.")

if __name__ == "__main__":
    unittest.main()
