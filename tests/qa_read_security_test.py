
import os
import sys
import unittest
from unittest.mock import MagicMock

# ── PRE-MOCK COMFYUI ──────────────────────────────────────────────────────────
mock_folder_paths = MagicMock()
sys.modules["folder_paths"] = mock_folder_paths

mock_comfy = MagicMock()
sys.modules["comfy"] = mock_comfy
sys.modules["comfy.utils"] = MagicMock()
# ──────────────────────────────────────────────────────────────────────────────

from radiance.nodes_io import RadianceReadVideo, RadianceReadSequence
from radiance.path_utils import get_safe_input_path

class TestReadSecurity(unittest.TestCase):
    def setUp(self):
        # Mock folder_paths.get_input_directory
        self.input_dir = os.path.abspath("tests/mock_input")
        os.makedirs(self.input_dir, exist_ok=True)
        mock_folder_paths.get_input_directory.return_value = self.input_dir
        
        self.video_node = RadianceReadVideo()
        self.seq_node = RadianceReadSequence()

    def test_video_path_traversal(self):
        # Attempt to access something outside input_dir
        evil_path = "../../../windows/system32/drivers/etc/hosts"
        with self.assertRaises(ValueError) as cm:
            self.video_node.read_video(video_path=evil_path, start_frame=0, frame_count=1, 
                                      input_colorspace="sRGB (Standard)", frame_stride=1)
        self.assertIn("escapes base directory", str(cm.exception))

    def test_sequence_path_traversal(self):
        evil_path = "../../../Windows/System32" 
        with self.assertRaises(ValueError) as cm:
            self.seq_node.read_sequence(folder_path=evil_path, pattern="*", 
                                      start_frame=0, frame_limit=1, input_colorspace="sRGB (Standard)")
        self.assertIn("escapes base directory", str(cm.exception))

    def test_absolute_path_allowed(self):
        # We now allow explicit absolute paths
        dummy_file = os.path.join(self.input_dir, "test.mp4")
        with open(dummy_file, "w") as f: f.write("dummy")
        
        abs_path = os.path.abspath(dummy_file)
        # Should not raise ValueError for path traversal
        try:
            self.video_node.read_video(video_path=abs_path, start_frame=0, frame_count=1, 
                                      input_colorspace="sRGB (Standard)", frame_stride=1)
        except (IOError, RuntimeError):
            pass

    def test_valid_input_path(self):
        # Create a dummy file in input_dir
        dummy_file = os.path.join(self.input_dir, "test.mp4")
        with open(dummy_file, "w") as f: f.write("dummy")
        
        # Should not raise ValueError for path traversal (will raise IOError because it's not a real video)
        try:
            self.video_node.read_video(video_path="test.mp4", start_frame=0, frame_count=1, 
                                      input_colorspace="sRGB (Standard)", frame_stride=1)
        except (IOError, RuntimeError):
            pass # We only care about the path check
        except ValueError as e:
            self.fail(f"Valid path raised ValueError: {e}")

if __name__ == "__main__":
    unittest.main()
