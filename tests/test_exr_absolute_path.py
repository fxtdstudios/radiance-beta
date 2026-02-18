
import unittest
import torch
import os
import sys
import shutil
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes_exr import RadianceSaveEXR

class TestEXRAbsolutePath(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory that is definitely outside the default output path
        self.abs_path = tempfile.mkdtemp()
        
    def tearDown(self):
        if os.path.exists(self.abs_path):
            shutil.rmtree(self.abs_path)

    def test_absolute_subfolder(self):
        print(f"\n[Test] Attempting to save to absolute path: {self.abs_path}")
        
        saver = RadianceSaveEXR()
        # Mocking default output_dir to something else
        saver.output_dir = "test_output" 
        
        image = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
        
        # This should currently fail with ValueError due to path_utils check
        try:
            saver.save_exr(
                images=image,
                filename_prefix="abs_test",
                subfolder=self.abs_path,
                add_metadata=False
            )
            # If we reach here, it succeeded (unexpected for now)
            print("Successfully saved to absolute path (Unexpected before fix)")
            
            # Verify file exists
            expected_file = os.path.join(self.abs_path, "abs_test_0001.exr")
            if os.path.exists(expected_file):
                print(f"File created at: {expected_file}")
            else:
                self.fail("Method returned success but file not found")
                
        except ValueError as e:
            print(f"Caught expected ValueError: {e}")
            # Verify it's the security error
            self.assertIn("Security error", str(e))
        except Exception as e:
            self.fail(f"Caught unexpected exception: {e}")

if __name__ == '__main__':
    unittest.main()
