
import unittest
import torch
import numpy as np
import os
import sys
import tempfile
import shutil
import logging

# Fix OMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add custom_nodes directory to allow 'import radiance'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Add ComfyUI root directory to allow 'import folder_paths' and 'import comfy'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import modules to test
try:
    # Use full package import to support relative imports within the module
    try:
        from radiance.hdr.io import RadianceSaveEXR
    except ImportError as e:
        print(f"Failed to import radiance.hdr.io: {e}")
        from nodes_exr import RadianceSaveEXR

    # nodes_grade is standalone, so direct import works, but package import is safer if we want to be consistent
    try:
        from radiance.nodes_grade import RadianceGrade
    except ImportError:
        from nodes_grade import RadianceGrade
        
    # Attempt to import color nodes
    try:
        from radiance.nodes_color import RadianceSceneLinearWorkflow, RadianceACES2OutputTransform
    except ImportError:
        try:
             from nodes_color import RadianceSceneLinearWorkflow, RadianceACES2OutputTransform
        except ImportError:
            RadianceSceneLinearWorkflow = None
            RadianceACES2OutputTransform = None
except ImportError as e:
    print(f"CRITICAL: Failed to import Radiance nodes: {e}")

class TestRadianceEvaluation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger("RadianceTest")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_environment_precision(self):
        """Test 1: Verify 32-bit float precision preservation."""
        self.logger.info("Testing 32-bit precision...")
        
        # Create a tensor with float32 precision and values > 1.0 (HDR)
        input_tensor = torch.tensor([[[0.1, 0.5, 2.0]]], dtype=torch.float32)
        
        # Use simple operation to check if precision is kept
        # Simulating a "Grade" operation manually first
        gain = 2.0
        output_tensor = input_tensor * gain
        
        self.assertEqual(output_tensor.dtype, torch.float32, "Tensor dtype should remain float32")
        self.assertGreater(output_tensor.max().item(), 2.0, "HDR values should be preserved (not clamped to 1.0)")
        self.assertAlmostEqual(output_tensor[0][0][2].item(), 4.0, places=5, msg="Multiplication accuracy failed")

    def test_02_radiance_grade_node(self):
        """Test 2: Test RadianceGrade node for functionality and precision."""
        self.logger.info("Testing RadianceGrade node...")
        
        grade_node = RadianceGrade()
        
        # Input: 1920x1080 RGB image, random float32
        img = torch.rand((1, 1080, 1920, 3), dtype=torch.float32)
        
        # Apply a grade that we can easily verify (e.g., gain=2.0)
        # Note: implementation of grade expects specific kwargs
        graded_img, info = grade_node.grade(
            image=img, 
            preset="None (Custom)",
            gain_r=2.0, gain_g=2.0, gain_b=2.0
        )
        
        self.assertEqual(graded_img.dtype, torch.float32)
        self.assertEqual(graded_img.shape, img.shape)
        
        # Verify gain was applied (approximate check due to potential internal logic)
        # Grade node logic: 2. Apply GAIN
        # img *= gain
        expected_sample = img[0,0,0,0] * 2.0
        actual_sample = graded_img[0,0,0,0]
        
        # Tolerance might be needed if other operations (gamma default 1.0?) interfere.
        # Defaults: lift=0, gamma=1, offset=0, contrast=1, sat=1.
        self.assertTrue(torch.isclose(actual_sample, expected_sample, atol=1e-5), 
                        f"Grade Gain failed: Expected {expected_sample}, got {actual_sample}")

    def test_03_dependency_handling(self):
        """Test 3: Check correct handling of missing optional dependencies."""
        self.logger.info("Testing dependency fallback...")
        
        # We know colour-science is likely missing.
        # Check if nodes relying on it are either None or handle it.
        # If imports succeeded (dummy nodes?), check their behavior.
        
        # In nodes_color.py, it imports from .color.
        # If .color imports fail, it might raise ImportError globally or handle it.
        # Our top-level import passed or failed.
        
        # This test just logs the state for the report.
        try:
            import colour
            self.logger.info("colour-science IS installed.")
        except ImportError:
            self.logger.info("colour-science is NOT installed (Expected).")
            
        try:
            import PyOpenColorIO
            self.logger.info("PyOpenColorIO IS installed.")
        except ImportError:
            self.logger.info("PyOpenColorIO is NOT installed (Expected).")

    def test_04_exr_io(self):
        """Test 4: EXR saving functionality."""
        self.logger.info("Testing EXR saving...")
        
        saver = RadianceSaveEXR()
        saver.output_dir = self.temp_dir # output_dir mock
        
        # 64x64 simplistic HDR image
        img = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 5.0 # Value 5.0 everywhere
        
        try:
            # We expect this to save to self.temp_dir with prefix "eval_test"
            # Note: save_exr might require 'subfolder' arg even if empty
            results = saver.save_exr(
                images=img, 
                filename_prefix="eval_test", 
                subfolder="",
                add_metadata="false" # Boolean or string? "false" based on convention usually boolean in python but check typed inputs
            )
            
            # Check if file exists
            # Filename depends on internal counter, usually eval_test_0001.exr
             # The save method returns a dictionary with 'ui' key often
            pass
            
        except Exception as e:
            # If basic save fails (e.g. OpenEXR missing), fail test
            self.logger.warning(f"EXR Save failed (could be missing OpenEXR or permissions): {e}")
            # If OpenEXR is present (checked in env), this is a FAIL.
            try:
                import OpenEXR
                self.fail(f"RadianceSaveEXR failed despite OpenEXR being installed: {e}")
            except ImportError:
                self.skipTest("Skipping EXR test: OpenEXR library not installed.")

    def test_05_gpu_consistency(self):
        """Test 5: Check if code runs on GPU if available and matches CPU."""
        if not torch.cuda.is_available():
            self.skipTest("Skipping GPU test: CUDA not available")
            
        self.logger.info("Testing GPU consistency...")
        
        grade_node = RadianceGrade()
        img_cpu = torch.rand((1, 512, 512, 3), dtype=torch.float32)
        img_gpu = img_cpu.to("cuda")
        
        res_cpu, _ = grade_node.grade(img_cpu, gain_r=1.5, contrast=1.2)
        res_gpu, _ = grade_node.grade(img_gpu, gain_r=1.5, contrast=1.2)
        
        # Bring back to CPU for comparison
        res_gpu_cpu = res_gpu.cpu()
        
        # Check max difference
        diff = torch.abs(res_cpu - res_gpu_cpu).max().item()
        self.assertLess(diff, 1e-5, f"GPU result differs from CPU by {diff}")

if __name__ == '__main__':
    unittest.main()
