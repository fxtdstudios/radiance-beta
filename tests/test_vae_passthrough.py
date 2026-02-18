import sys
import os
import unittest
import torch
import types

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Emulate ComfyUI environment
current_file = os.path.abspath(__file__)
radiance_dir = os.path.dirname(os.path.dirname(current_file))
custom_nodes_dir = os.path.dirname(radiance_dir)
comfy_root = os.path.dirname(custom_nodes_dir)

if radiance_dir not in sys.path:
    sys.path.insert(0, radiance_dir)

if comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)

# Mock folder_paths if import fails
try:
    import folder_paths
except ImportError:
    dummy_fp = types.ModuleType("folder_paths")
    dummy_fp.get_input_directory = lambda: "input"
    dummy_fp.get_output_directory = lambda: "output"
    sys.modules["folder_paths"] = dummy_fp

from hdr.vae import RadianceVAEEncode, RadianceVAEDecode

class MockVAE:
    def encode(self, x):
        return {"samples": x * 0.5} # Dummy encoding
        
    def decode(self, x):
        return x * 2.0 # Dummy decoding

class TestVAEPassthrough(unittest.TestCase):
    def test_encode_passthrough_clamping(self):
        """Test that Encode Passthrough does not clip values input to VAE."""
        encoder = RadianceVAEEncode()
        vae = MockVAE()
        
        # Create input with super-whites/blacks
        # Shape: (1, 64, 64, 3)
        pixels = torch.tensor([-0.5, 0.5, 2.0], dtype=torch.float32).repeat(1, 64, 64, 1)
        
        # Inspect what encoder passes to VAE
        # Since we can't easily spy on internal vars, we rely on the logic that 
        # passthrough does Linear->sRGB conversion.
        # sRGB conversion of 2.0 is > 1.0 (approx 1.28).
        # sRGB conversion of -0.5 is -0.5 (sign preserved).
        
        # We'll use a subclass to spy
        class SpyVAEEncode(RadianceVAEEncode):
            def _tiled_vae_encode(self, pixels, vae, tile_size, overlap=64):
                self.last_pixels_to_vae = pixels
                return super()._tiled_vae_encode(pixels, vae, tile_size, overlap)
                
            def encode(self, *args, **kwargs):
                # Ensure we skip tiling for simplicity or mock tiled usage
                return super().encode(*args, **kwargs)

        spy_encoder = SpyVAEEncode()
        
        # Test "Clip (SDR)" - should clamp
        spy_encoder.encode(pixels.clone(), vae, hdr_mode="Clip (SDR)", tile_size="None")
        # We need to access the 'img' variable right before valid.encode calls.
        # But 'encode' calls 'vae.encode(img)'.
        # Our MockVAE returns x*0.5.
        # So latent = img * 0.5.
        # We can deduce img from latent.
        
        # Run Passthrough
        result, _, _ = spy_encoder.encode(pixels.clone(), vae, hdr_mode="Passthrough (Unclamped)", tile_size="None")
        latent = result["samples"]
        
        # Inferred input to VAE
        inferred_vae_input = latent / 0.5
        
        # Check max value
        max_val = inferred_vae_input.max().item()
        min_val = inferred_vae_input.min().item()
        
        print(f"Passthrough Max (converted to sRGB): {max_val}")
        print(f"Passthrough Min (converted to sRGB): {min_val}")
        
        # sRGB(2.0) is approx 1.055 * (2.0)^(1/2.4) - 0.055 ~= 1.35
        # It should be significantly > 1.0
        self.assertTrue(max_val > 1.1, f"Expected max > 1.1, got {max_val}. Values were clamped?")
        
        # sRGB(-0.5) preservation depends on implementation. 
        # tensor_linear_to_srgb preserves sign. 
        # So it should be < -0.2
        self.assertTrue(min_val < -0.1, f"Expected min < -0.1, got {min_val}. Values were clamped?")
        
    def test_decode_passthrough_clamping(self):
        """Test that Decode Passthrough preserves values > 1.0."""
        decoder = RadianceVAEDecode()
        vae = MockVAE()
        
        # Latent that decodes to > 1.0
        # If decode() multiplies latent by 2.0
        # We input 1.0 -> 2.0
        latent_samples = torch.tensor([[-0.5, 0.5, 2.0]], dtype=torch.float32).reshape(1, 3, 1, 1)
        samples = {"samples": latent_samples}
        
        # "Passthrough" mode should treat input as sRGB and convert to Linear.
        # VAE output (from Mock) = 2.0 * input = [-1.0, 1.0, 4.0] (sRGB domain)
        # Linear(-1.0) = -1.0 (approx)
        # Linear(4.0) = ((4.0+0.055)/1.055)^2.4 ~= 23.0
        
        image, _ = decoder.decode(samples, vae, hdr_mode="Passthrough (Unclamped)")
        
        max_val = image.max().item()
        min_val = image.min().item()
        
        print(f"Decode Passthrough Max: {max_val}")
        
        self.assertTrue(max_val > 5.0, f"Expected super-white (>5.0), got {max_val}")
        self.assertTrue(min_val < -0.5, f"Expected super-black (< -0.5), got {min_val}")

if __name__ == '__main__':
    unittest.main()
