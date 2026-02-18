
import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to allow importing radiance modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from radiance.nodes_overlay import RadianceMetadataOverlay

class TestRadianceMetadataOverlay(unittest.TestCase):
    def setUp(self):
        self.node = RadianceMetadataOverlay()
        # Create a dummy black image (Batch, Height, Width, Channels)
        # 1 image, 512x512, 3 channels (RGB)
        self.dummy_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    def test_instantiation(self):
        """Test if node can be instantiated."""
        self.assertIsInstance(self.node, RadianceMetadataOverlay)

    def test_overlay_execution(self):
        """Test basic execution of the overlay."""
        result = self.node.overlay_metadata(
            image=self.dummy_image,
            enabled=True,
            project="TEST_PROJECT",
            shot="TEST_SHOT",
            frame=101,
            timecode="01:00:00:00",
            position="Bottom Bar (Slate)",
            font_size=20,
            opacity=0.5,
            user_text="User Note",
            report_text="QC Report Passed"
        )
        
        # Check output type and shape
        out_image = result[0]
        self.assertIsInstance(out_image, torch.Tensor)
        self.assertEqual(out_image.shape, self.dummy_image.shape)
        
        # Check if pixels changed (text was drawn)
        # Since input was all zeros (black), output should have some non-zero values (text/box)
        self.assertTrue(torch.any(out_image > 0), "Output image is still all black, text not drawn?")

    def test_overlay_disabled(self):
        """Test if disabled node returns original image."""
        result = self.node.overlay_metadata(
            image=self.dummy_image,
            enabled=False,
            project="TEST",
            shot="TEST",
            frame=1,
            timecode="00:00:00:00",
            position="Top Left",
            font_size=20,
            opacity=0.5
        )
        
        self.assertTrue(torch.equal(result[0], self.dummy_image))

    def test_floating_box_position(self):
        """Test floating box rendering."""
        result = self.node.overlay_metadata(
            image=self.dummy_image,
            enabled=True,
            project="TEST",
            shot="TEST",
            frame=1,
            timecode="00:00:00:00",
            position="Top Left",
            font_size=20,
            opacity=0.5
        )
        self.assertTrue(torch.any(result[0] > 0))

if __name__ == '__main__':
    unittest.main()
