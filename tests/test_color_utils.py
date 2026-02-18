import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path to import color_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import color_utils

class TestColorUtils(unittest.TestCase):
    def test_srgb_to_linear_numpy(self):
        # Test 0.0 -> 0.0
        self.assertTrue(np.isclose(color_utils.srgb_to_linear(np.array([0.0])), 0.0))
        # Test 1.0 -> 1.0
        self.assertTrue(np.isclose(color_utils.srgb_to_linear(np.array([1.0])), 1.0))
        # Test middle gray (approx 0.5 sRGB -> 0.214 linear)
        val_srgb = np.array([0.5])
        val_linear = color_utils.srgb_to_linear(val_srgb)
        self.assertTrue(np.isclose(val_linear, 0.21404114, atol=1e-5))

    def test_linear_to_srgb_numpy(self):
        # Round trip test
        original = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float32)
        linear = color_utils.srgb_to_linear(original)
        back_to_srgb = color_utils.linear_to_srgb(linear)
        self.assertTrue(np.allclose(original, back_to_srgb, atol=1e-5))

    def test_tensor_srgb_to_linear(self):
        # Test torch tensor version
        tensor_srgb = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        tensor_linear = color_utils.tensor_srgb_to_linear(tensor_srgb)
        
        expected = np.array([0.0, 0.21404114, 1.0], dtype=np.float32)
        self.assertTrue(np.allclose(tensor_linear.numpy(), expected, atol=1e-5))

    def test_tensor_linear_to_srgb(self):
        # Round trip torch
        original = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=torch.float32)
        linear = color_utils.tensor_srgb_to_linear(original)
        back_to_srgb = color_utils.tensor_linear_to_srgb(linear)
        self.assertTrue(torch.allclose(original, back_to_srgb, atol=1e-5))

    def test_logc3_conversions(self):
        # Round trip
        original = np.array([0.02, 0.18, 1.0], dtype=np.float32)
        logc3 = color_utils.linear_to_logc3(original)
        linear = color_utils.logc3_to_linear(logc3)
        self.assertTrue(np.allclose(original, linear, atol=1e-4))

    def test_logc4_conversions(self):
        # Round trip
        original = np.array([0.02, 0.18, 1.0], dtype=np.float32)
        logc4 = color_utils.linear_to_logc4(original)
        linear = color_utils.logc4_to_linear(logc4)
        self.assertTrue(np.allclose(original, linear, atol=1e-4))
        
        # Test standard values (ARRI LogC4 Spec)
        # 18% gray (0.18) should map to ~32% (0.32) per ARRI ALEXA 35 spec
        val_018 = np.array([0.18], dtype=np.float32)
        encoded_018 = color_utils.linear_to_logc4(val_018)
        # Allow some tolerance for float32 precision
        self.assertTrue(np.isclose(encoded_018, 0.32, atol=0.005), f"Expected ~0.32 for 18% gray, got {encoded_018}")

    def test_slog3_conversions(self):
        # Round trip
        original = np.array([0.02, 0.18, 1.0], dtype=np.float32)
        slog3 = color_utils.linear_to_slog3(original)
        linear = color_utils.slog3_to_linear(slog3)
        self.assertTrue(np.allclose(original, linear, atol=1e-4))
    
    def test_vlog_conversions(self):
        # Round trip
        original = np.array([0.02, 0.18, 1.0], dtype=np.float32)
        vlog = color_utils.linear_to_vlog(original)
        linear = color_utils.vlog_to_linear(vlog)
        self.assertTrue(np.allclose(original, linear, atol=1e-4))
        
    def test_canonlog3_conversions(self):
        # Round trip
        original = np.array([0.02, 0.18, 1.0], dtype=np.float32)
        clog3 = color_utils.linear_to_canonlog3(original)
        linear = color_utils.canonlog3_to_linear(clog3)
        self.assertTrue(np.allclose(original, linear, atol=1e-4))

    def test_acescct_conversions(self):
        # Round trip
        original = np.array([0.02, 0.18, 1.0], dtype=np.float32)
        acescct = color_utils.linear_to_acescct(original)
        linear = color_utils.acescct_to_linear(acescct)
        self.assertTrue(np.allclose(original, linear, atol=1e-4))

    def test_davinci_intermediate_conversions(self):
        # Round trip
        original = np.array([0.02, 0.18, 1.0], dtype=np.float32)
        di = color_utils.linear_to_davinci_intermediate(original)
        linear = color_utils.davinci_intermediate_to_linear(di)
        # Note: tolerance might need adjustment if formula is approximate
        self.assertTrue(np.allclose(original, linear, atol=1e-4))

    def test_pq_conversions(self):
        # Round trip PQ
        original = np.array([0.0, 0.1, 1.0], dtype=np.float32)
        pq = color_utils.linear_to_pq(original, peak_nits=1000.0)
        linear = color_utils.pq_to_linear(pq, peak_nits=1000.0)
        self.assertTrue(np.allclose(original, linear, atol=1e-4))

    def test_hlg_conversions(self):
        # Round trip HLG
        original = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float32)
        hlg = color_utils.linear_to_hlg(original)
        linear = color_utils.hlg_to_linear(hlg)
        self.assertTrue(np.allclose(original, linear, atol=1e-4))

    def test_tensor_numpy_conversion(self):
        # Test helper functions
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        n = color_utils.tensor_to_numpy_float32(t)
        self.assertTrue(isinstance(n, np.ndarray))
        self.assertTrue(np.allclose(n, np.array([1.0, 2.0])))

        n2 = np.array([3.0, 4.0], dtype=np.float32)
        t2 = color_utils.numpy_to_tensor_float32(n2)
        self.assertTrue(isinstance(t2, torch.Tensor))
        self.assertTrue(torch.allclose(t2, torch.tensor([3.0, 4.0])))

if __name__ == '__main__':
    unittest.main()
