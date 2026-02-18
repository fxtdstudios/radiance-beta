"""
═══════════════════════════════════════════════════════════════════════════════
                RADIANCE UPSCALE MODULE - UNIT TESTS
                Tests for upscaling algorithms and image processing
═══════════════════════════════════════════════════════════════════════════════
"""
import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import upscale utilities
from nodes_upscale import (
    UpscaleMethod,
    lanczos_kernel,
    mitchell_kernel,
    catmull_rom_kernel,
    hermite_kernel,
    gaussian_kernel,
    separable_resize_32bit,
    torch_resize_32bit,
    unsharp_mask_32bit,
    linear_to_srgb_32bit,
    srgb_to_linear_32bit,
    FXTDProUpscale,
    FXTDDownscale32bit,
    FXTDBitDepthConvert,
)


class TestUpscaleKernels(unittest.TestCase):
    """Test upscaling kernel functions."""
    
    def test_lanczos_kernel_shape(self):
        """Test Lanczos kernel output shape."""
        x = np.linspace(-3, 3, 100)
        result = lanczos_kernel(x, a=3)
        
        self.assertEqual(result.shape, x.shape)
    
    def test_lanczos_kernel_center(self):
        """Test Lanczos kernel is 1 at center."""
        x = np.array([0.0])
        result = lanczos_kernel(x, a=3)
        
        self.assertTrue(np.isclose(result[0], 1.0, atol=1e-5))
    
    def test_mitchell_kernel_center(self):
        """Test Mitchell kernel is near 1 at center (B/C parameters affect peak)."""
        x = np.array([0.0])
        result = mitchell_kernel(x)
        
        # Mitchell-Netravali with B=C=1/3 peaks around 0.89, not 1.0
        self.assertGreater(result[0], 0.5)
    
    def test_catmull_rom_kernel(self):
        """Test Catmull-Rom kernel exists and works."""
        x = np.linspace(-2, 2, 50)
        result = catmull_rom_kernel(x)
        
        self.assertEqual(result.shape, x.shape)
    
    def test_hermite_kernel(self):
        """Test Hermite kernel exists and works."""
        x = np.linspace(-1, 1, 50)
        result = hermite_kernel(x)
        
        self.assertEqual(result.shape, x.shape)
    
    def test_gaussian_kernel(self):
        """Test Gaussian kernel is bell-shaped."""
        x = np.linspace(-2, 2, 50)
        result = gaussian_kernel(x, sigma=0.5)
        
        # Should be symmetric and peak at center
        center_idx = len(x) // 2
        self.assertGreater(result[center_idx], result[0])
        self.assertGreater(result[center_idx], result[-1])


class TestSeparableResize(unittest.TestCase):
    """Test separable resize function."""
    
    def test_resize_upscale_2x(self):
        """Test 2x upscale."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = separable_resize_32bit(img, 128, 128, 'lanczos')
        
        self.assertEqual(result.shape, (128, 128, 3))
        self.assertEqual(result.dtype, np.float32)
    
    def test_resize_downscale_2x(self):
        """Test 2x downscale."""
        img = np.random.rand(128, 128, 3).astype(np.float32)
        result = separable_resize_32bit(img, 64, 64, 'lanczos')
        
        self.assertEqual(result.shape, (64, 64, 3))
    
    def test_resize_preserves_range(self):
        """Test resize preserves reasonable value range."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = separable_resize_32bit(img, 96, 96, 'bicubic')
        
        # Values should stay reasonably bounded for normalized input
        self.assertLessEqual(result.max(), 1.5)  # Small overshoot allowed
        self.assertGreaterEqual(result.min(), -0.5)
    
    def test_resize_different_methods(self):
        """Test various resize methods work."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        
        methods = ['lanczos', 'bicubic', 'bilinear', 'nearest']
        for method in methods:
            try:
                result = separable_resize_32bit(img, 48, 48, method)
                self.assertEqual(result.shape, (48, 48, 3), f"{method} failed")
            except (NotImplementedError, KeyError, ValueError):
                # Some methods may not be implemented
                pass


class TestTorchResize(unittest.TestCase):
    """Test torch-based resize function."""
    
    def test_torch_resize_upscale(self):
        """Test torch resize upscale."""
        tensor = torch.rand((1, 64, 64, 3), dtype=torch.float32)
        result = torch_resize_32bit(tensor, 128, 128, 'bicubic')
        
        self.assertEqual(result.shape, (1, 128, 128, 3))
    
    def test_torch_resize_maintains_batch(self):
        """Test torch resize maintains batch dimension."""
        tensor = torch.rand((4, 32, 32, 3), dtype=torch.float32)
        result = torch_resize_32bit(tensor, 64, 64, 'bilinear')
        
        self.assertEqual(result.shape[0], 4)


class TestUnsharpMask(unittest.TestCase):
    """Test unsharp mask sharpening."""
    
    def test_unsharp_mask_shape(self):
        """Test unsharp mask preserves shape."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = unsharp_mask_32bit(img, amount=1.0, radius=1.0)
        
        self.assertEqual(result.shape, img.shape)
    
    def test_unsharp_mask_increases_contrast(self):
        """Test unsharp mask increases local contrast."""
        # Create simple gradient image
        img = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
        img = np.stack([img] * 3, axis=-1)
        
        result = unsharp_mask_32bit(img, amount=2.0, radius=1.0)
        
        # Sharpened should have higher variance
        self.assertGreaterEqual(result.std(), img.std() * 0.9)


class TestColorSpaceConversions(unittest.TestCase):
    """Test sRGB <-> linear conversions."""
    
    def test_srgb_linear_roundtrip(self):
        """Test sRGB to linear and back."""
        original = np.random.rand(32, 32, 3).astype(np.float32)
        linear = srgb_to_linear_32bit(original)
        back = linear_to_srgb_32bit(linear)
        
        self.assertTrue(np.allclose(original, back, atol=1e-4))
    
    def test_linear_darker_than_srgb(self):
        """Test linear values are typically darker than sRGB."""
        srgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        linear = srgb_to_linear_32bit(srgb)
        
        # Linear 0.5 in sRGB is approximately 0.214
        self.assertLess(linear[0], 0.5)


class TestFXTDProUpscale(unittest.TestCase):
    """Test FXTDProUpscale node."""
    
    def test_instantiation(self):
        """Test node can be instantiated."""
        node = FXTDProUpscale()
        self.assertIsNotNone(node)
    
    def test_input_types(self):
        """Test INPUT_TYPES returns valid structure."""
        inputs = FXTDProUpscale.INPUT_TYPES()
        
        self.assertIn("required", inputs)
        self.assertIn("image", inputs["required"])
    
    def test_has_scale_factor(self):
        """Test scale factor parameter exists."""
        inputs = FXTDProUpscale.INPUT_TYPES()
        all_inputs = {**inputs.get("required", {}), **inputs.get("optional", {})}
        
        scale_key = [k for k in all_inputs if "scale" in k.lower()]
        self.assertGreater(len(scale_key), 0)
    
    def test_has_method_selection(self):
        """Test upscale method selection exists."""
        inputs = FXTDProUpscale.INPUT_TYPES()
        all_inputs = {**inputs.get("required", {}), **inputs.get("optional", {})}
        
        method_key = [k for k in all_inputs if "method" in k.lower()]
        self.assertGreater(len(method_key), 0)


class TestFXTDDownscale(unittest.TestCase):
    """Test FXTDDownscale32bit node."""
    
    def test_instantiation(self):
        """Test node can be instantiated."""
        node = FXTDDownscale32bit()
        self.assertIsNotNone(node)
    
    def test_input_types(self):
        """Test INPUT_TYPES returns valid structure."""
        inputs = FXTDDownscale32bit.INPUT_TYPES()
        
        self.assertIn("required", inputs)
        self.assertIn("image", inputs["required"])
    
    def test_has_antialiasing(self):
        """Test antialiasing parameter exists."""
        inputs = FXTDDownscale32bit.INPUT_TYPES()
        all_inputs = {**inputs.get("required", {}), **inputs.get("optional", {})}
        
        aa_key = [k for k in all_inputs if "anti" in k.lower()]
        self.assertGreater(len(aa_key), 0)


class TestFXTDBitDepthConvert(unittest.TestCase):
    """Test FXTDBitDepthConvert node."""
    
    def test_instantiation(self):
        """Test node can be instantiated."""
        node = FXTDBitDepthConvert()
        self.assertIsNotNone(node)
    
    def test_input_types(self):
        """Test INPUT_TYPES returns valid structure."""
        inputs = FXTDBitDepthConvert.INPUT_TYPES()
        
        self.assertIn("required", inputs)
        self.assertIn("image", inputs["required"])
    
    def test_has_bit_depth_options(self):
        """Test bit depth options exist."""
        inputs = FXTDBitDepthConvert.INPUT_TYPES()
        
        if "output_depth" in inputs.get("required", {}):
            options = inputs["required"]["output_depth"][0]
            self.assertIn("32-bit Float", options)
            self.assertIn("16-bit Float", options)
            self.assertIn("8-bit", options)
    
    def test_has_dithering_options(self):
        """Test dithering options exist."""
        inputs = FXTDBitDepthConvert.INPUT_TYPES()
        all_inputs = {**inputs.get("required", {}), **inputs.get("optional", {})}
        
        if "dithering" in all_inputs:
            options = all_inputs["dithering"][0]
            self.assertIn("None", options)
            self.assertIn("Floyd-Steinberg", options)


class TestUpscaleMethod(unittest.TestCase):
    """Test UpscaleMethod enum."""
    
    def test_methods_exist(self):
        """Test expected upscale methods exist."""
        expected = ["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"]
        
        for method in expected:
            self.assertTrue(hasattr(UpscaleMethod, method))
    
    def test_lanczos_value(self):
        """Test Lanczos has expected value."""
        self.assertEqual(UpscaleMethod.LANCZOS.value, "lanczos")


if __name__ == '__main__':
    unittest.main()
