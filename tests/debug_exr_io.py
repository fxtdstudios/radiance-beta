
import unittest
import sys
import os
import torch
import numpy as np
import shutil
import tempfile
from unittest.mock import MagicMock, patch

# ═══════════════════════════════════════════════════════════════════════════════
#                         MOCKING COMFYUI MODULES
# ═══════════════════════════════════════════════════════════════════════════════
# Mock structure must be set up BEFORE importing
mock_folder_paths = MagicMock()
mock_folder_paths.get_output_directory = MagicMock(return_value=os.getcwd())
sys.modules["folder_paths"] = mock_folder_paths

# Mock global comfy package
mock_comfy = MagicMock()
sys.modules["comfy"] = mock_comfy
sys.modules["comfy.utils"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()

# Add custom_nodes directory to path so we can import 'radiance' as a package
radiance_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
custom_nodes_dir = os.path.dirname(radiance_dir)
if custom_nodes_dir not in sys.path:
    sys.path.append(custom_nodes_dir)

import logging
logging.basicConfig(level=logging.DEBUG)

# Import the module via full package path
from radiance.hdr.io import RadianceSaveEXR, LoadImageEXR, check_openexr_available

class TestRadianceEXRClamping(unittest.TestCase):
    
    def setUp(self):
        self.saver = RadianceSaveEXR()
        self.loader = LoadImageEXR()
        self.temp_dir = tempfile.mkdtemp()
        self.saver.output_dir = self.temp_dir
        print(f"Temp Dir: {self.temp_dir}")
        print(f"OpenEXR Available: {check_openexr_available()}")
        
    def tearDown(self):
        # Cleanup
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_exr_roundtrip_hdr_preservation(self):
        print("\n--- Testing Radiance EXR I/O for Clamping ---")
        
        # 1. Create HDR Tensor with Extreme Values
        # Shape: (1, 32, 32, 3)
        # Values: -100.0 (Super Black), 0.18 (Mid), 1000.0 (Super White), 65000.0 (Near Half Max)
        input_tensor = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
        
        # Set specific pixels
        input_tensor[0, 0, 0, 0] = -100.0  # R: Super Black
        input_tensor[0, 0, 1, 1] = 0.18    # G: Mid Gray
        input_tensor[0, 0, 2, 2] = 1000.0  # B: Super White
        input_tensor[0, 1, 0, 0] = 50000.0 # R: Extreme White (safe for float16)
        
        # 2. Save EXR
        # We patch get_safe_output_dir to return our temp dir
        with patch("radiance.path_utils.get_safe_output_dir", return_value=self.temp_dir):
            result = self.saver.save_exr(
                images=input_tensor,
                filename_prefix="debug_hdr_test",
                format="EXR",
                bit_depth="32-bit Float", # Use full float to test logically
                compression="ZIP",
                input_color_space="Linear (sRGB)", # Passthrough assumptions
                output_color_space="Same as Input", # Passthrough
                add_metadata=False
            )
            
        # Parse result to get file path
        # Result format: {"ui": {}, "result": (paths_str, output_dir, len)}
        paths_str = result["result"][0]
        saved_path = paths_str.split("\n")[0]
        print(f"Saved EXR to: {saved_path}")
        
        if os.path.exists(saved_path):
            size = os.path.getsize(saved_path)
            print(f"File Size: {size} bytes")
        else:
            self.fail("File not created")
        
        self.assertTrue(os.path.exists(saved_path), "EXR file was not created")
        
        # 3. Load EXR back
        # We need to mock folder_paths.get_annotated_filepath if we use the node's helper,
        # but we can call internal load logic or just use imageio/cv2/openexr directly 
        # to verify FILE contents, which is what matters.
        # But let's use the actual Loader node to verify the full loop.
        
        # Patch _get_actual_path to just return the path
        with patch.object(LoadImageEXR, "_get_actual_path", return_value=saved_path):
             loaded_tensor, metadata = self.loader.load(saved_path)
             
        # 4. Verification
        print(f"Loaded Tensor Range: Min={loaded_tensor.min().item()}, Max={loaded_tensor.max().item()}")
        
        # Check specific pixels
        # loaded_tensor is (1, H, W, 3) ??? No, loader returns (B, H, W, C) usually 
        # Check loader source: returns (numpy_to_tensor_float32(img), ...)
        # numpy_to_tensor adds batch dim if needed output is typically 4D?
        # Loader code: 
        # img = ... (H, W, C)
        # numpy_to_tensor -> adds batch -> (1, H, W, C)
        
        # Epsilon for float32 roundtrip
        epsilon = 1e-3 
        
        val_neg = loaded_tensor[0, 0, 0, 0].item()
        val_mid = loaded_tensor[0, 0, 1, 1].item()
        val_high = loaded_tensor[0, 0, 2, 2].item()
        val_huge = loaded_tensor[0, 1, 0, 0].item()
        
        print(f"Values Roundtrip:")
        print(f"  -100.0   -> {val_neg} (Diff: {abs(val_neg - (-100.0))})")
        print(f"  0.18     -> {val_mid} (Diff: {abs(val_mid - 0.18)})")
        print(f"  1000.0   -> {val_high} (Diff: {abs(val_high - 1000.0)})")
        print(f"  50000.0  -> {val_huge} (Diff: {abs(val_huge - 50000.0)})")
        
        self.assertAlmostEqual(val_neg, -100.0, delta=epsilon, msg="-100.0 preservation failed")
        self.assertAlmostEqual(val_high, 1000.0, delta=epsilon, msg="1000.0 preservation failed")
        self.assertAlmostEqual(val_huge, 50000.0, delta=0.1, msg="50000.0 preservation failed") # larger delta for huge #s
        
        print("Confirmed: Radiance EXR I/O preserves full HDR range.")

    def test_openexr_direct_write(self):
        print("\n--- Testing OpenEXR Direct Write ---")
        try:
            import OpenEXR
            import Imath
        except ImportError:
            print("OpenEXR not installed, skipping.")
            return

        filepath = os.path.join(self.temp_dir, "direct_openexr.exr")
        header = OpenEXR.Header(64, 64)
        header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
        
        # Try simple float channel
        header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        
        try:
            out = OpenEXR.OutputFile(filepath, header)
            
            # Create synthetic data
            data = np.zeros((64, 64), dtype=np.float32).tobytes()
            out.writePixels({'R': data})
            out.close()
            print("Direct OpenEXR write SUCCESS")
        except Exception as e:
            print(f"Direct OpenEXR write FAILED: {e}")
            self.fail(f"OpenEXR direct write failed: {e}")

if __name__ == '__main__':
    unittest.main()
