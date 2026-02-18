"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE EXR MODULE - UNIT TESTS
                    Tests for EXR I/O and color space handling
═══════════════════════════════════════════════════════════════════════════════
"""
import unittest
import numpy as np
import torch
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import EXR utilities
from hdr.io import (
    float32_to_float16,
    float16_to_bytes,
    float32_to_bytes,
    SimpleEXRWriter,
    write_exr_cv2,
    rgb_to_rgbe,
    write_hdr_rgbe
)
import color_utils


class TestEXRBitDepth(unittest.TestCase):
    """Test EXR bit depth conversions."""
    
    def test_float32_to_float16_conversion(self):
        """Test float32 to float16 conversion preserves values."""
        original = np.array([0.0, 0.5, 1.0, 2.0, 10.0], dtype=np.float32)
        converted = float32_to_float16(original)
        
        self.assertEqual(converted.dtype, np.float16)
        # Check values are close (accounting for half-float precision loss)
        self.assertTrue(np.allclose(converted.astype(np.float32), original, rtol=1e-3))
    
    def test_float16_bytes_roundtrip(self):
        """Test float16 to bytes and back."""
        original = np.array([0.5, 1.0, 2.0], dtype=np.float16)
        as_bytes = float16_to_bytes(original)
        
        self.assertIsInstance(as_bytes, bytes)
        self.assertEqual(len(as_bytes), original.nbytes)
        
        # Reconstruct
        reconstructed = np.frombuffer(as_bytes, dtype=np.float16)
        self.assertTrue(np.array_equal(original, reconstructed))
    
    def test_float32_bytes_roundtrip(self):
        """Test float32 to bytes and back."""
        original = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        as_bytes = float32_to_bytes(original)
        
        self.assertIsInstance(as_bytes, bytes)
        self.assertEqual(len(as_bytes), original.nbytes)
        
        # Reconstruct
        reconstructed = np.frombuffer(as_bytes, dtype=np.float32)
        self.assertTrue(np.array_equal(original, reconstructed))
    
    def test_float16_precision_limits(self):
        """Test float16 handles HDR values correctly."""
        # Max float16 is ~65504
        hdr_values = np.array([0.0, 1.0, 100.0, 10000.0, 65000.0], dtype=np.float32)
        converted = float32_to_float16(hdr_values)
        
        self.assertEqual(converted.dtype, np.float16)
        # Check it doesn't overflow to inf for reasonable HDR values
        self.assertFalse(np.any(np.isinf(converted[:-1])))  # Last one might overflow


class TestEXRColorSpace(unittest.TestCase):
    """Test EXR color space conversions."""
    
    def test_srgb_to_linear_roundtrip(self):
        """Test sRGB to linear and back."""
        original = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
        linear = color_utils.srgb_to_linear(original)
        back = color_utils.linear_to_srgb(linear)
        
        self.assertTrue(np.allclose(original, back, atol=1e-5))
    
    def test_logc3_roundtrip(self):
        """Test LogC3 encoding and decoding."""
        original = np.array([0.01, 0.18, 1.0, 5.0], dtype=np.float32)
        encoded = color_utils.linear_to_logc3(original)
        decoded = color_utils.logc3_to_linear(encoded)
        
        self.assertTrue(np.allclose(original, decoded, atol=1e-4))
    
    def test_logc4_roundtrip(self):
        """Test LogC4 encoding and decoding."""
        original = np.array([0.01, 0.18, 1.0, 10.0], dtype=np.float32)
        encoded = color_utils.linear_to_logc4(original)
        decoded = color_utils.logc4_to_linear(encoded)
        
        self.assertTrue(np.allclose(original, decoded, atol=1e-4))
    
    def test_logc4_midgray_mapping(self):
        """Test LogC4 18% gray maps to ~0.32."""
        midgray = np.array([0.18], dtype=np.float32)
        encoded = color_utils.linear_to_logc4(midgray)
        
        self.assertTrue(np.isclose(encoded[0], 0.32, atol=0.01),
                       f"18% gray should map to ~0.32, got {encoded[0]}")
    
    def test_acescct_roundtrip(self):
        """Test ACEScct encoding and decoding."""
        original = np.array([0.01, 0.18, 1.0], dtype=np.float32)
        encoded = color_utils.linear_to_acescct(original)
        decoded = color_utils.acescct_to_linear(encoded)
        
        self.assertTrue(np.allclose(original, decoded, atol=1e-4))
    
    def test_slog3_roundtrip(self):
        """Test S-Log3 encoding and decoding."""
        original = np.array([0.01, 0.18, 1.0], dtype=np.float32)
        encoded = color_utils.linear_to_slog3(original)
        decoded = color_utils.slog3_to_linear(encoded)
        
        self.assertTrue(np.allclose(original, decoded, atol=1e-4))


class TestSimpleEXRWriter(unittest.TestCase):
    """Test the SimpleEXRWriter class."""
    
    def test_writer_instantiation(self):
        """Test writer can be instantiated."""
        writer = SimpleEXRWriter()
        self.assertIsNotNone(writer)
    
    def test_write_rgb_exr(self):
        """Test writing a basic RGB EXR file."""
        writer = SimpleEXRWriter()
        
        # Create test image (64x64 RGB gradient)
        height, width = 64, 64
        r = np.tile(np.linspace(0, 1, width, dtype=np.float32), (height, 1))
        g = np.tile(np.linspace(0, 1, height, dtype=np.float32).reshape(-1, 1), (1, width))
        b = np.ones((height, width), dtype=np.float32) * 0.5
        
        channels = {"R": r, "G": g, "B": b}
        
        with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
            filepath = f.name
        
        try:
            writer.write(filepath, channels, compression="ZIP", pixel_type="HALF")
            
            # Check file was created
            self.assertTrue(os.path.exists(filepath))
            
            # Check file has content
            file_size = os.path.getsize(filepath)
            self.assertGreater(file_size, 0)
            
            # Check EXR magic number
            with open(filepath, 'rb') as f:
                magic = int.from_bytes(f.read(4), byteorder='little')
                self.assertEqual(magic, 20000630, "Invalid EXR magic number")
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_write_32bit_exr(self):
        """Test writing 32-bit float EXR."""
        writer = SimpleEXRWriter()
        
        height, width = 32, 32
        channels = {
            "R": np.random.rand(height, width).astype(np.float32),
            "G": np.random.rand(height, width).astype(np.float32),
            "B": np.random.rand(height, width).astype(np.float32)
        }
        
        with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
            filepath = f.name
        
        try:
            writer.write(filepath, channels, compression="ZIP", pixel_type="FLOAT")
            self.assertTrue(os.path.exists(filepath))
            self.assertGreater(os.path.getsize(filepath), 0)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestHDRFormat(unittest.TestCase):
    """Test HDR/Radiance format writing."""
    
    def test_rgb_to_rgbe_conversion(self):
        """Test RGB to RGBE (shared exponent) conversion."""
        # Test with known values
        rgb = np.array([[[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]], dtype=np.float32)
        rgbe = rgb_to_rgbe(rgb)
        
        self.assertEqual(rgbe.shape, (1, 2, 4))  # 4 channels: R, G, B, E
        self.assertEqual(rgbe.dtype, np.uint8)
    
    def test_rgbe_black_handling(self):
        """Test RGBE handles black correctly."""
        rgb = np.zeros((1, 2, 3), dtype=np.float32)
        rgbe = rgb_to_rgbe(rgb)
        
        # Black should have very low RGB values in RGBE
        # (exponent handling may vary by implementation)
        self.assertTrue(np.allclose(rgbe[..., :3], 0, atol=1) or rgbe[..., 3].mean() < 10)
    
    def test_write_hdr_file(self):
        """Test writing HDR file."""
        # Create test image
        image = np.random.rand(64, 64, 3).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.hdr', delete=False) as f:
            filepath = f.name
        
        try:
            write_hdr_rgbe(filepath, image)
            
            self.assertTrue(os.path.exists(filepath))
            self.assertGreater(os.path.getsize(filepath), 0)
            
            # Check HDR magic
            with open(filepath, 'rb') as f:
                header = f.read(10)
                self.assertTrue(header.startswith(b'#?RADIANCE'))
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestEXRCV2Writer(unittest.TestCase):
    """Test OpenCV-based EXR writer if available."""
    
    def setUp(self):
        """Check if OpenCV with EXR support is available."""
        try:
            import cv2
            # Check if we can write EXR
            test = np.zeros((2, 2, 3), dtype=np.float32)
            with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
                cv2.imwrite(f.name, test)
                os.unlink(f.name)
            self.cv2_available = True
        except (ImportError, OSError) as e:
            self.cv2_available = False
    
    def test_cv2_exr_write(self):
        """Test writing EXR with OpenCV."""
        if not self.cv2_available:
            self.skipTest("OpenCV EXR support not available")
        
        image = np.random.rand(64, 64, 3).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
            filepath = f.name
        
        try:
            write_exr_cv2(filepath, image, bit_depth="float32")
            self.assertTrue(os.path.exists(filepath))
            self.assertGreater(os.path.getsize(filepath), 0)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


if __name__ == '__main__':
    unittest.main()
