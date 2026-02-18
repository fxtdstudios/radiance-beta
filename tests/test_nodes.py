import unittest
import torch
import sys
import os
import numpy as np

# Add custom_nodes directory to path (parent of current radiance dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import node modules from radiance package
from radiance.nodes_resolution import FXTDResolution
from radiance.nodes_hdr import ImageToFloat32, Float32ColorCorrect, HDRExpandDynamicRange
from radiance.nodes_camera import FXTDPhysicalCamera

class TestNodes(unittest.TestCase):
    
    def test_resolution_node(self):
        node = FXTDResolution()
        # Test default generation
        # generate(self, preset, model_type, megapixels, custom_width, custom_height, ...)
        res = node.generate(preset="Instagram Square (1:1)", model_type="Flux / SD3", megapixels="1.0")
        
        self.assertIsInstance(res, dict)
        self.assertIn("result", res)
        # Result tuple: (latent_dict, preview, width, height, info)
        result_tuple = res["result"]
        self.assertEqual(len(result_tuple), 5)
        
        latent_dict = result_tuple[0]
        self.assertIn("samples", latent_dict)
        latent = latent_dict["samples"]
        self.assertEqual(latent.shape[0], 1) # batch size 1
        # Flux has 16 channels
        self.assertEqual(latent.shape[1], 16)
        
        width = result_tuple[2]
        height = result_tuple[3]
        self.assertEqual(width, 1024)
        self.assertEqual(height, 1024)

    def test_image_to_float32(self):
        node = ImageToFloat32()
        # Create a dummy uint8 tensor (simulating loaded image)
        input_tensor = torch.rand((1, 64, 64, 3), dtype=torch.float32)
        result = node.convert(input_tensor, normalize=False)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].dtype, torch.float32)
        self.assertTrue(torch.allclose(result[0], input_tensor))

    def test_color_correct(self):
        node = Float32ColorCorrect()
        img = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.5
        
        # Test exposure +1stop -> 0.5 * 2 = 1.0
        res = node.correct(img, exposure=1.0)
        self.assertTrue(torch.allclose(res[0], torch.ones_like(img)))
        
        # Test brightness
        res = node.correct(img, brightness=0.1)
        self.assertTrue(torch.allclose(res[0], torch.ones_like(img) * 0.6))
        
        # Test saturation 0 (grayscale)
        color_img = torch.tensor([[[0.1, 0.5, 0.9]]], dtype=torch.float32)
        res = node.correct(color_img, saturation=0.0)
        # Expected: luma. Luma = 0.2126*0.1 + 0.7152*0.5 + 0.0722*0.9 
        # = 0.02126 + 0.3576 + 0.06498 = 0.44384
        self.assertTrue(torch.allclose(res[0][..., 0], res[0][..., 1], atol=1e-4))
        self.assertTrue(torch.allclose(res[0][..., 0], res[0][..., 2], atol=1e-4))

    def test_hdr_expand(self):
        node = HDRExpandDynamicRange()
        img = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.5
        
        # Just run it to ensure no crash
        res = node.expand(img, source_gamma=2.2, target_stops=12.0)
        self.assertIsInstance(res, tuple)
        self.assertEqual(res[0].shape, img.shape)
        
    def test_physical_camera_instantiation(self):
        # Just check we can instantiate and get inputs
        inputs = FXTDPhysicalCamera.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("image", inputs["required"])
        self.assertIn("optional", inputs)
        self.assertIn("aperture", inputs["optional"])

if __name__ == '__main__':
    unittest.main()
