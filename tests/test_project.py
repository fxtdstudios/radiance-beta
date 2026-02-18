
import unittest
import torch
import numpy as np
import sys
import os
import shutil
import tempfile

# Mock system path
# Add custom_nodes directory to sys.path so we can import 'radiance' as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock ComfyUI folder_paths
import sys
from unittest.mock import MagicMock
mock_folder_paths = MagicMock()
mock_folder_paths.get_output_directory.return_value = tempfile.gettempdir()
mock_folder_paths.get_annotated_filepath.return_value = ""
sys.modules["folder_paths"] = mock_folder_paths

from radiance.nodes_project import RadianceProjectSettings
from radiance.hdr.io import RadianceSaveEXR

class TestRadianceProject(unittest.TestCase):
    def setUp(self):
        self.test_root = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_root)
        
    def test_project_workflow(self):
        # 1. Project Settings
        project_node = RadianceProjectSettings()
        config, info = project_node.create_config(
            project_root=self.test_root,
            sequence="TEST_SEQ",
            shot="SHOT_001",
            version=5,
            use_date_folder=False
        )
        
        expected_path = f"{self.test_root.replace(os.sep, '/')}/TEST_SEQ/SHOT_001/v005"
        self.assertEqual(config['full_path'], expected_path)
        
        # 2. Save Logic
        saver = RadianceSaveEXR()
        
        # Mock Image (1x1 float32)
        img = torch.ones((1, 64, 64, 3), dtype=torch.float32)
        
        # Run Save with Project Config
        results = saver.save_exr(
            images=img,
            filename_prefix="IGNORED", # Should be ignored
            project_config=config,
            format="EXR"
        )
        
        # Verify
        saved_files_str = results['result'][0]
        self.assertTrue(len(saved_files_str) > 0)
        
        saved_path = saved_files_str.split('\n')[0]
        
        # Check logic: Path should contain the project structure
        self.assertIn("TEST_SEQ", saved_path)
        self.assertIn("SHOT_001", saved_path)
        self.assertIn("v005", saved_path)
        
        # Check filename format: SEQ_SHOT_vVER_FRAME
        filename = os.path.basename(saved_path)
        self.assertTrue(filename.startswith("TEST_SEQ_SHOT_001_v005"))

if __name__ == '__main__':
    unittest.main()
