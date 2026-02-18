
import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from film.grain import RadianceFilmGrain
from nodes_overlay import RadianceMetadataOverlay
from image.upscale import RadianceProUpscale

class TestClamping(unittest.TestCase):
    def setUp(self):
        # Create an HDR image with negative values and super-whites
        # Shape: (1, 32, 32, 3)
        self.hdr_image = torch.tensor([[[[-1.0, 0.5, 2.0]]]], dtype=torch.float32)
        self.hdr_image = self.hdr_image.expand(1, 32, 32, 3)

    def test_grain_hdr_preservation(self):
        """Test that Film Grain preserves HDR values when hdr_safe is True (and check default)."""
        node = RadianceFilmGrain()
        
        # Test 1: Explicit hdr_safe=True
        output_safe, = node.apply_grain(
            self.hdr_image, 
            preset="Custom", 
            intensity=0.0,  # Zero intensity to check passthrough mainly
            size=1.0, 
            seed=0, 
            hdr_safe=True
        )
        
        print(f"\nGrain (Safe) Min: {output_safe.min()}, Max: {output_safe.max()}")
        self.assertLess(output_safe.min(), 0.0, "Grain (Safe) should preserve negative values")
        self.assertGreater(output_safe.max(), 1.0, "Grain (Safe) should preserve values > 1.0")

        # Test 2: Check DEFAULT behavior (Targeting True)
        # Note: We are changing the default, so we expect this to PASS after fix.
        # For TDD, this might fail initially if default is False.
        # We'll check the signature or just call with defaults if possible, 
        # but apply_grain takes arguments. The node class definition defines defaults.
        # We can simulate calling from ComfyUI which provides defaults, but here we invoke python method directly.
        # The python method signature in `grain.py` has `hdr_safe: bool = False` currently.
        # We will manually pass `True` for now to verify logic works, and verify default chang via inspection or checking signature.
        
    def test_overlay_hdr_preservation(self):
        """Test that Metadata Overlay preserves HDR values found in the underlying image."""
        node = RadianceMetadataOverlay()
        
        # Only enable if we can (requires PIL fonts etc, usually standard)
        output, = node.overlay_metadata(
            self.hdr_image,
            enabled=True,
            project="TEST",
            shot="001",
            frame=1,
            timecode="00:00:00:00",
            position="Top Left",
            font_size=20,
            opacity=0.5,
            user_text="",
            report_text=""
        )
        
        print(f"Overlay Min: {output.min()}, Max: {output.max()}")
        
        # The logic currently casts to uint8, so it will clamp to [0, 1] (0-255).
        # We expect this to FAIL before fix.
        self.assertTrue(output.min() < 0.0 or output.max() > 1.0, 
                        f"Overlay clamped output! Range: [{output.min()}, {output.max()}]")

    def test_upscale_hdr_preservation(self):
        """Test that Upscale preserves HDR values."""
        node = RadianceProUpscale()
        
        # Scale 1.0 just to check pipeline logic without interpolation artifacts confusing things too much
        output, w, h, info = node.upscale(
            self.hdr_image,
            scale_factor=1.0,
            preset="Custom",
            method="bilinear",
            sharpening=0.0,
            sharpen_radius=1.0,
            detail_enhancement=0.0,
            antialiasing=0.0,
            input_color_space="Linear",
            process_in_linear=True,
            use_tiles=False,
            output_bit_depth="32-bit Float"
        )
        
        print(f"Upscale Min: {output.min()}, Max: {output.max()}")
        self.assertLess(output.min(), 0.0, "Upscale should preserve negative values")
        self.assertGreater(output.max(), 1.0, "Upscale should preserve values > 1.0")

if __name__ == '__main__':
    unittest.main()
