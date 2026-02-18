
import unittest
import torch
import numpy as np
import sys
import os

# Mock import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from radiance.nodes_qc import RadianceQC

class TestRadianceQC(unittest.TestCase):
    def test_qc_analysis_batch(self):
        node = RadianceQC()
        
        # Create a BATCH of test images (2, 10, 10, 3)
        img = torch.zeros((2, 10, 10, 3), dtype=torch.float32)
        
        # Frame 0: Clipped Whites
        img[0, 0:2, 0:2, :] = 2.0
        
        # Frame 1: Crushed Blacks
        img[1, 8:10, 8:10, :] = -0.1
        
        # Run analysis
        overlay, report = node.analyze(img, black_threshold=0.0, white_threshold=1.0, overlay_opacity=0.5)
        
        print("Generated Batch Report:\n", report)
        
        # Check report content for specific frames
        self.assertIn("Frame 1:", report)
        self.assertIn("CLIPPED", report) # Frame 1 has clipped
        
        self.assertIn("Frame 2:", report) 
        self.assertIn("CRUSHED", report) # Frame 2 has crushed
        
        # Check overlay dimensions
        self.assertEqual(overlay.shape, img.shape)
        
        # Check Frame 0 Overlay (Top Left modified)
        self.assertFalse(torch.allclose(overlay[0, 0, 0], img[0, 0, 0]))
        
        # Check Frame 1 Overlay (Bottom Right modified)
        self.assertFalse(torch.allclose(overlay[1, 9, 9], img[1, 9, 9]))

    def test_banding_detection(self):
        node = RadianceQC()
        # Create a gradient that is stepped (simulating banding)
        # 100x100
        img = torch.zeros((1, 100, 100, 3), dtype=torch.float32)
        
        # Create bands: steps of 0.002 (below 0.005 threshold)
        for i in range(100):
            val = (i // 10) * 0.003
            img[0, :, i, :] = val
            
        overlay, report = node.analyze(img, black_threshold=-0.1, white_threshold=1.1, overlay_opacity=0.0)
        print("Banding Report:\n", report)
        
        # Should detect banding
        self.assertIn("BANDING", report)

if __name__ == '__main__':
    unittest.main()
