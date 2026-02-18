
import unittest
import sys
import os
import torch

# Add custom_nodes directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestHDRFacade(unittest.TestCase):
    
    def test_import_hdr_nodes(self):
        """Test that we can import nodes from the facade."""
        try:
            from radiance.nodes_hdr import (
                ImageToFloat32, 
                HDRExpandDynamicRange, 
                HDRToneMap,
                LoadImageEXR,
                HDRExposureBlend,
                HDR360Generate,
                RadianceVAEEncode
            )
            print("\nSuccessfully imported HDR nodes from facade!")
        except ImportError as e:
            self.fail(f"Failed to import HDR nodes: {e}")
            
    def test_instantiate_nodes(self):
        """Test that we can instantiate the nodes."""
        from radiance.nodes_hdr import ImageToFloat32, HDRExpandDynamicRange
        
        node = ImageToFloat32()
        self.assertIsNotNone(node)
        
        node2 = HDRExpandDynamicRange()
        self.assertIsNotNone(node2)
        
    def test_functionality_colorspace(self):
        """Test basic functionality of a simple node."""
        from radiance.nodes_hdr import ImageToFloat32
        node = ImageToFloat32()
        
        # Create a dummy image tensor (1, 64, 64, 3)
        input_tensor = torch.rand((1, 64, 64, 3), dtype=torch.float32)
        
        # Run convert
        result = node.convert(input_tensor, normalize=False)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, input_tensor.shape)

if __name__ == '__main__':
    unittest.main()
