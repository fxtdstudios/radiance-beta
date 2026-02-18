import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import unittest
import sys
import os

import types

# Mock folder_paths
try:
    import folder_paths
except ImportError:
    dummy_fp = types.ModuleType("folder_paths")
    dummy_fp.get_input_directory = lambda: "input"
    dummy_fp.get_output_directory = lambda: "output"
    sys.modules["folder_paths"] = dummy_fp

# Add parent directory to path to import radiance modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdr.recovery import RadianceHighlightSynthesis

class TestHighlightSynthesis(unittest.TestCase):
    def setUp(self):
        self.node = RadianceHighlightSynthesis()
        
    def test_expansion(self):
        # Create a clamped image (flat white area)
        # Shape: [1, 100, 100, 3]
        img = torch.ones((1, 100, 100, 3), dtype=torch.float32)
        
        # Apply synthesis
        threshold = 0.95
        expansion = 1.5
        result = self.node.synthesize(img, threshold=threshold, expansion=expansion, detail_amount=0.0)[0]
        
        # Verify values > 1.0
        max_val = result.max().item()
        print(f"Max value after expansion: {max_val:.4f}")
        
        self.assertGreater(max_val, 1.0, "Highlights should be expanded above 1.0")
        
        # Expected expansion calculation:
        # At input=1.0, mask=1.0.
        # expansion_map = 1.0 + (1.0 - 0.95) * (1.5 - 1.0) * 2.0 
        #               = 1.0 + 0.05 * 0.5 * 2.0 = 1.05
        # output = 1.0 * 1.05 = 1.05
        # (This is approximate due to the smooth mask ramping)
        
    def test_detail_synthesis(self):
        # Create a clamped image
        img = torch.ones((1, 100, 100, 3), dtype=torch.float32)
        
        # Apply synthesis with noise
        detail_amount = 0.5
        result = self.node.synthesize(img, threshold=0.9, expansion=1.0, detail_amount=detail_amount)[0]
        
        # Calculate standard deviation
        std_dev = torch.std(result).item()
        print(f"Standard deviation with detail: {std_dev:.4f}")
        
        self.assertGreater(std_dev, 0.0, "Result should have variation (detail) added")
        
    def test_threshold_masking(self):
        # Image with gradient 0.0 to 1.0
        x = torch.linspace(0, 1, 100)
        y = torch.linspace(0, 1, 100)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        img = grid_x.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0) # [1, 100, 100, 3]
        
        # Apply synthesis
        threshold = 0.8
        result = self.node.synthesize(img, threshold=threshold, expansion=2.0, detail_amount=0.0)[0]
        
        # Verify values below threshold are unchanged (approximately)
        mask_low = img < 0.7
        diff_low = (result[mask_low] - img[mask_low]).abs().max().item()
        
        self.assertLess(diff_low, 1e-4, "Values well below threshold should be unchanged")
        
        # Verify values above threshold are modified
        mask_high = img > 0.9
        diff_high = (result[mask_high] - img[mask_high]).abs().max().item()
        self.assertGreater(diff_high, 1e-4, "Values above threshold should be modified")

if __name__ == '__main__':
    unittest.main()
