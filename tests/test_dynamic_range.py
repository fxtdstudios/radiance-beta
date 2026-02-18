
import unittest
import numpy as np
import sys
import os
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from color_utils import (
        linear_to_srgb, srgb_to_linear,
        linear_to_logc4, logc4_to_linear,
        linear_to_logc3, logc3_to_linear,
        linear_to_slog3, slog3_to_linear,
        linear_to_acescct, acescct_to_linear,
        apply_matrix_transform,
        ACESCG_TO_SRGB, ACESCG_TO_REC2020
    )
    from nodes_color import RadianceGPUColorMatrix
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class TestDynamicRange(unittest.TestCase):
    """
    Professional VFX Unit Test Suite for Dynamic Range Verification.
    
    Objective:
    - Verify preservation of Super-Whites (Values > 1.0)
    - Verify handling of Super-Blacks (Values < 0.0)
    - Ensure round-trip accuracy for HDR data
    """

    def setUp(self):
        # Test Vectors
        self.super_white = np.array([10.0, 50.0, 100.0], dtype=np.float32)  # High intensity
        self.super_black = np.array([-0.1, -0.5, 0.0], dtype=np.float32)    # Negative/Zero
        self.mid_grey = np.array([0.18, 0.18, 0.18], dtype=np.float32)      # Standard reference
        
        # Tolerance for float32 precision
        self.tolerance = 1e-4

    def test_matrix_transform_hdr(self):
        """Test if Matrix Transformations preserve values > 1.0"""
        print("\n[Test] Matrix Transform HDR Preservation")
        
        # ACEScg to Rec.2020 (Both wide gamut, should preserve intensity)
        input_val = np.array([10.0, 2.0, 5.0], dtype=np.float32)
        transformed = apply_matrix_transform(input_val, ACESCG_TO_REC2020)
        
        # Check if values are still high (not clamped)
        max_val = np.max(transformed)
        self.assertGreater(max_val, 1.0, f"Matrix transform clamped HDR value! Max is {max_val}, expected > 1.0")
        print(f"  ✓ Input Max: {np.max(input_val):.2f} -> Output Max: {max_val:.2f}")

    def test_logc4_roundtrip_hdr(self):
        """Test ARRI LogC4 Roundtrip with Super-Whites"""
        print("\n[Test] LogC4 Roundtrip (Human Vision / HDR)")
        
        # LogC4 is designed for huge dynamic range
        input_hdr = np.array([0.18, 1.0, 10.0, 100.0], dtype=np.float32)
        
        # Encode
        encoded = linear_to_logc4(input_hdr)
        # Check integrity - 100.0 linear should not clip at 1.0 in LogC4 (it might be close to 1.0 but represents 100)
        # Actually LogC4 signal 1.0 is huge.
        
        # Decode
        decoded = logc4_to_linear(encoded)
        
        # Verify
        np.testing.assert_allclose(decoded, input_hdr, rtol=1e-3, atol=1e-5, 
                                   err_msg="LogC4 Roundtrip failed to preserve HDR values")
        print(f"  ✓ LogC4 preserved [0.18, 1.0, 10.0, 100.0] correctly.")

    def test_logc3_negative_handling(self):
        """Test ARRI LogC3 behavior with Negative Values"""
        print("\n[Test] LogC3 Negative Handling")
        
        # LogC3 usually clips or handles negatives via linear slope
        input_neg = np.array([-0.01, 0.0, 0.1], dtype=np.float32)
        
        encoded = linear_to_logc3(input_neg)
        decoded = logc3_to_linear(encoded)
        
        # Arri LogC3 might clamp negatives or linearize them. 
        # We verify that it doesn't return NaNs or explode.
        self.assertFalse(np.any(np.isnan(decoded)), "LogC3 produced NaNs for negative input")
        
        # Check reconstruction of standard negative (within small toe range)
        # Note: LogC3 definition slope at zero handles small negatives.
        np.testing.assert_allclose(decoded[0], input_neg[0], rtol=0.1, atol=1e-3,
                                   err_msg="LogC3 failed to reconstruct small negative value")
        print("  ✓ LogC3 handled small negative values safely.")

    def test_acescct_hdr(self):
        """Test ACEScct with extremely bright values"""
        print("\n[Test] ACEScct HDR Capability")
        
        input_hdr = np.array([200.0], dtype=np.float32) # Very bright
        encoded = linear_to_acescct(input_hdr)
        decoded = acescct_to_linear(encoded)
        
        np.testing.assert_allclose(decoded, input_hdr, rtol=1e-3, 
                                   err_msg="ACEScct failed to roundtrip 200.0 linear")
        print(f"  ✓ ACEScct preserved 200.0 linear -> {encoded[0]:.4f} log -> {decoded[0]:.1f} linear")

    def test_srgb_gamma_unclamped_output(self):
        """Test standard sRGB Gamma EOTF output range"""
        print("\n[Test] sRGB Gamma Unclamped Output")
        
        input_hdr = np.array([2.0, 4.0], dtype=np.float32)
        # Expectation: sRGB curve should continue upwards, not clamp at 1.0
        # y = 1.055 * x^(1/2.4) ...
        
        output = linear_to_srgb(input_hdr)
        
        # 2.0 linear -> ~1.35
        # 4.0 linear -> ~1.81
        
        self.assertTrue(np.all(output > 1.0), "sRGB Gamma function implicitly clamped output > 1.0")
        print(f"  ✓ sRGB Gamma output for input=[2,4]: {output}")

    def test_tensor_gpu_matrix_node(self):
        """Test the RadianceGPUColorMatrix node (simulated) for clamping options"""
        print("\n[Test] RadianceGPUColorMatrix Node Options")
        
        node = RadianceGPUColorMatrix()
        # Create a mock tensor on CPU
        img_tensor = torch.tensor([[[[2.0, 0.0, 0.0]]]], dtype=torch.float32)
        
        # 1. Test Unclamped (clamp_output=False) - Manually specifying expected args
        # Apply identity matrix manual
        # apply_matrix(self, image, preset, matrix_type, r_vector, g_vector, b_vector, ...)
        
        res_unclamped, = node.apply_matrix(
            image=img_tensor,
            preset="Custom",
            matrix_type="RGB (3x3)",
            r_vector="1.0, 0.0, 0.0",
            g_vector="0.0, 1.0, 0.0",
            b_vector="0.0, 0.0, 1.0",
            clamp_output=False
        )
        
        val_unclamped = res_unclamped.max().item()
        self.assertGreater(val_unclamped, 1.0, "Node failed to respect clamp_output=False")
        print(f"  ✓ Node (Clamp=False) preserved max value: {val_unclamped}")
        
        # 2. Test Clamped (clamp_output=True)
        res_clamped, = node.apply_matrix(
            image=img_tensor,
            preset="Custom",
            matrix_type="RGB (3x3)",
            r_vector="1.0, 0.0, 0.0",
            g_vector="0.0, 1.0, 0.0",
            b_vector="0.0, 0.0, 1.0",
            clamp_output=True
        )
        
        val_clamped = res_clamped.max().item()
        self.assertAlmostEqual(val_clamped, 1.0, places=5, msg="Node failed to clamp when requested")
        print(f"  ✓ Node (Clamp=True) correctly clamped to: {val_clamped}")

if __name__ == '__main__':
    unittest.main()
