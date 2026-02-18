
import unittest
import sys
import os
import torch
import numpy as np
import shutil
import tempfile
from unittest.mock import MagicMock, patch

# Mock ComfyUI modules
mock_folder_paths = MagicMock()
mock_folder_paths.get_output_directory = MagicMock(return_value=os.getcwd())
sys.modules["folder_paths"] = mock_folder_paths

mock_comfy = MagicMock()
sys.modules["comfy"] = mock_comfy
sys.modules["comfy.utils"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()

# Setup correct import path for radiance package
radiance_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
custom_nodes_dir = os.path.dirname(radiance_dir)
if custom_nodes_dir not in sys.path:
    sys.path.append(custom_nodes_dir)

import logging
logging.basicConfig(level=logging.DEBUG)

from radiance.hdr.io import RadianceSaveEXR, LoadImageEXR, check_openexr_available

class TestRadianceEXRZDepth(unittest.TestCase):
    
    def setUp(self):
        self.loader = LoadImageEXR()
        self.temp_dir = tempfile.mkdtemp()
        print(f"Temp Dir: {self.temp_dir}")
        
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_z_channel_loading(self):
        print("\n--- Testing EXR Z-Channel Loading ---")
        
        # 1. Create Synthetic EXR with Z channel using OpenEXR directly
        try:
            import OpenEXR
            import Imath
        except ImportError:
            print("OpenEXR not installed, skipping test.")
            return

        filename = "test_z_channel.exr"
        filepath = os.path.join(self.temp_dir, filename)
        
        width, height = 64, 64
        header = OpenEXR.Header(width, height)
        # Define channels: R, G, B, Z
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }
        
        # Create data
        # RGB is simple gradient
        # Z is distance gradient (near to far)
        r = np.linspace(0, 1, width * height, dtype=np.float32).reshape(height, width)
        g = np.zeros((height, width), dtype=np.float32)
        b = np.zeros((height, width), dtype=np.float32)
        
        # Z channel: 0.5 to 100.0
        z = np.linspace(0.5, 100.0, width * height, dtype=np.float32).reshape(height, width)
        
        out = OpenEXR.OutputFile(filepath, header)
        out.writePixels({
            'R': r.tobytes(),
            'G': g.tobytes(),
            'B': b.tobytes(),
            'Z': z.tobytes()
        })
        out.close()
        
        print(f"Created EXR with Z channel at: {filepath}")
        
        # 2. Attempt to Load with Radiance Loader
        # We need to verify if it returns Z channel. 
        # Currently, loader returns (image, metadata). 
        # If successfully implemented, it should return (image, depth_mask, metadata) or similar.
        
        # For now, let's see what happens.
        with patch.object(LoadImageEXR, "_get_actual_path", return_value=filepath):
            try:
                result = self.loader.load(filepath)
                
                # Check return structure
                print(f"Loader returned type: {type(result)}")
                if isinstance(result, tuple):
                    print(f"Tuple length: {len(result)}")
                    # Expecting four outputs: image, alpha_mask, depth_mask, metadata
                    if len(result) >= 4:
                        image = result[0]
                        alpha = result[1]
                        depth = result[2]
                        metadata = result[3]
                        
                        print(f"Image shape: {image.shape}")
                        print(f"Alpha shape: {alpha.shape}")
                        print(f"Depth shape: {depth.shape}")
                        
                        print("Found depth output!")
                        
                        # Verify Depth values
                        z_loaded = depth.numpy()
                        print(f"Depth range: {z_loaded.min()} - {z_loaded.max()}")
                        self.assertAlmostEqual(z_loaded.min(), 0.5, delta=0.01)
                        self.assertAlmostEqual(z_loaded.max(), 100.0, delta=1.0)
                        
                        # Verify Alpha is all ones (default for 3-channel EXR)
                        a_loaded = alpha.numpy()
                        print(f"Alpha range: {a_loaded.min()} - {a_loaded.max()}")
                        self.assertTrue(np.allclose(a_loaded, 1.0), "Alpha should be all ones for RGB EXR")
                        
                    else:
                        print(f"Unexpected tuple length: {len(result)}")
                        self.fail("Loader output signature mismatch")
                        
            except Exception as e:
                print(f"Loader failed: {e}")
                raise e

    def test_rgba_loading(self):
        print("\n--- Testing RGBA EXR Loading ---")
        try:
            import OpenEXR, Imath
        except ImportError:
            return

        filename = "test_rgba.exr"
        filepath = os.path.join(self.temp_dir, filename)
        
        width, height = 32, 32
        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'A': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }
        
        # Create data
        r = np.ones((height, width), dtype=np.float32)
        g = np.zeros((height, width), dtype=np.float32)
        b = np.zeros((height, width), dtype=np.float32)
        # Alpha gradient
        a = np.linspace(0, 1, width * height, dtype=np.float32).reshape(height, width)
        
        out = OpenEXR.OutputFile(filepath, header)
        out.writePixels({'R': r.tobytes(), 'G': g.tobytes(), 'B': b.tobytes(), 'A': a.tobytes()})
        out.close()
        
        # Load
        # We need to manually invoke load because we mocked _get_actual_path in the other test
        # Here we can just pass the path if we mock os.path.exists or just use it directly if no validation fails
        # The loader calls _get_actual_path.
        
        with patch.object(LoadImageEXR, "_get_actual_path", return_value=filepath):
            result = self.loader.load(filepath)
            
            self.assertTrue(len(result) >= 4)
            img, alpha, depth, meta = result[:4]
            
            # Verify Image is RGB (3 channels)
            self.assertEqual(img.shape[3], 3) 
            self.assertEqual(img.shape[0], 1) # Batch
            
            # Verify Alpha is separated and matches
            self.assertEqual(alpha.shape, (1, 32, 32))
            a_loaded = alpha.numpy()
            self.assertAlmostEqual(a_loaded.min(), 0.0, delta=0.001)
            self.assertAlmostEqual(a_loaded.max(), 1.0, delta=0.001)
            
            print("RGBA Loading Verified: Alpha extracted separately.")

    def test_sequence_loading(self):
        print("\n--- Testing EXR Sequence Loading ---")
        from radiance.hdr.io import LoadImageEXRSequence
        seq_loader = LoadImageEXRSequence()
        
        # Create 2 frames
        dir_path = os.path.join(self.temp_dir, "seq")
        os.makedirs(dir_path, exist_ok=True)
        
        try:
            import OpenEXR, Imath
        except ImportError:
            return

        for i in range(2):
            filepath = os.path.join(dir_path, f"seq_{i:04d}.exr")
            header = OpenEXR.Header(32, 32)
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            # Frame 0: Z=10, Frame 1: Z=20
            r = np.ones((32, 32), dtype=np.float32) * i
            z = np.ones((32, 32), dtype=np.float32) * (10.0 + i * 10.0)
            
            out = OpenEXR.OutputFile(filepath, header)
            out.writePixels({'R': r.tobytes(), 'Z': z.tobytes()})
            out.close()
            
        # Load sequence
        # load_sequence(folder_path, ...)
        
        # We need to mock folder validation?
        # NO, it uses os.path.isdir/glob which works with real temp files
        
        result = seq_loader.load_sequence(dir_path)
        
        # Return: images, alpha_masks, depth_masks, metadata, frame_count
        self.assertTrue(len(result) == 5)
        images, alphas, depths, meta, count = result
        
        print(f"Sequence loaded: {count} frames")
        print(f"Images shape: {images.shape}")
        print(f"Depths shape: {depths.shape}")
        
        self.assertEqual(count, 2)
        self.assertEqual(images.shape[0], 2)
        self.assertEqual(depths.shape[0], 2)
        
        # Verify Depth values
        z0 = depths[0].numpy()
        z1 = depths[1].numpy()
        
        self.assertAlmostEqual(z0.mean(), 10.0, delta=0.001)
        self.assertAlmostEqual(z1.mean(), 20.0, delta=0.001)
        
        print("Sequence Loading Verified.")

if __name__ == '__main__':
    unittest.main()
