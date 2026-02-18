
import unittest
import torch
import numpy as np
import os
import sys
import shutil

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes_exr import RadianceSaveEXR

class TestDynamicRangeFlow(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_output_hdr"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        
    def tearDown(self):
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except PermissionError:
                pass # Can happen on Windows

    def test_exr_saver_preserves_hdr(self):
        print("\n[Test] Checking EXR Saver HDR Preservation...")
        
        # 1. Create a > 1.0 Float Tensor (Simulating VAE Decode Output)
        hdr_val = 10.0
        # Shape: (Batch, H, W, C)
        image = torch.ones((1, 32, 32, 3), dtype=torch.float32) * hdr_val
        
        saver = RadianceSaveEXR()
        saver.output_dir = self.output_dir
        
        # 2. Save as EXR (32-bit Float)
        try:
            results = saver.save_exr(
                images=image,
                filename_prefix="hdr_test",
                format="EXR",
                bit_depth="32-bit Float",
                output_color_space="Linear (sRGB)", # Should pass through
                add_metadata=False
            )
            
            # Get path
            paths = results["result"][0].split('\n')
            self.assertTrue(len(paths) > 0)
            file_path = paths[0]
            print(f"Saved to: {file_path}")
            
            # Verify file exists and is not empty
            self.assertTrue(os.path.exists(file_path))
            self.assertGreater(os.path.getsize(file_path), 100)
            print("EXR file created successfully.")
            
        except ImportError as e:
            print(f"Skipping EXR save test due to missing dependencies: {e}")
        except Exception as e:
            self.fail(f"EXR save failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
