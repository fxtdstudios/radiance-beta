"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE PATH UTILITIES - UNIT TESTS
                    Security tests for path handling
═══════════════════════════════════════════════════════════════════════════════
"""
import unittest
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from path_utils import safe_join, validate_output_path, get_safe_output_dir


class TestSafeJoin(unittest.TestCase):
    """Test safe_join prevents path traversal attacks."""
    
    def test_normal_join(self):
        """Test normal path joining works."""
        base = "/output"
        result = safe_join(base, "subdir", "file.exr")
        expected = os.path.abspath(os.path.join(base, "subdir", "file.exr"))
        self.assertEqual(result, expected)

    def test_root_directory_join(self):
        """Test joining from a root directory."""
        # Test specifically for the bug where root dir + sep was rejected
        # Use current drive root on Windows, or / on Unix
        root = os.path.abspath(os.sep)
        result = safe_join(root, "file.exr")
        expected = os.path.join(root, "file.exr")
        self.assertEqual(result, expected)
    
    def test_empty_subfolder(self):
        """Test joining with empty subdirectory."""
        base = "/output"
        result = safe_join(base, "", "file.exr")
        expected = os.path.abspath(os.path.join(base, "file.exr"))
        self.assertEqual(result, expected)
    
    def test_rejects_parent_traversal(self):
        """Test that ../ is rejected."""
        base = "/output/images"
        
        with self.assertRaises(ValueError) as ctx:
            safe_join(base, "../etc/passwd")
        
        self.assertIn("traversal", str(ctx.exception).lower())
    
    def test_rejects_absolute_path(self):
        """Test that absolute paths escape base directory."""
        base = "/output"
        
        # Joining with an absolute path in Python replaces the base
        # Our safe_join should detect this
        with self.assertRaises(ValueError):
            safe_join(base, "/etc/passwd")
    
    def test_rejects_double_dot_encoded(self):
        """Test various traversal patterns."""
        base = "/output"
        
        patterns = [
            "../../../etc/passwd",
            "subdir/../../etc",
            "./../../etc",
        ]
        
        for pattern in patterns:
            with self.assertRaises(ValueError, msg=f"Should reject: {pattern}"):
                safe_join(base, pattern)
    
    def test_allows_subdirectory_dots(self):
        """Test that dots in filenames are allowed."""
        base = "/output"
        
        # These should be fine - dots in filenames, not traversal
        result = safe_join(base, "file.name.exr")
        expected = os.path.abspath(os.path.join(base, "file.name.exr"))
        self.assertEqual(result, expected)
        
        result = safe_join(base, "subdir.v2", "file.exr")
        expected = os.path.abspath(os.path.join(base, "subdir.v2", "file.exr"))
        self.assertEqual(result, expected)


class TestValidateOutputPath(unittest.TestCase):
    """Test validate_output_path function."""
    
    def test_rejects_absolute_subfolder(self):
        """Test that absolute subfolders are rejected."""
        base = "/output"
        
        with self.assertRaises(ValueError) as ctx:
            validate_output_path(base, "/etc", "test.exr")
        
        self.assertIn("absolute", str(ctx.exception).lower())
    
    def test_empty_subfolder(self):
        """Test with empty subfolder."""
        base = "/output"
        result = validate_output_path(base, "", "test.exr")
        expected = os.path.abspath(os.path.join(base, "test.exr"))
        self.assertEqual(result, expected)
    
    def test_normal_subfolder(self):
        """Test normal subfolder path."""
        base = "/output"
        result = validate_output_path(base, "renders/v01", "frame_001.exr")
        expected = os.path.abspath(os.path.join(base, "renders/v01", "frame_001.exr"))
        self.assertEqual(result, expected)


class TestGetSafeOutputDir(unittest.TestCase):
    """Test get_safe_output_dir function."""
    
    def test_creates_directory(self):
        """Test that directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_safe_output_dir(tmpdir, "new_subdir")
            
            self.assertTrue(os.path.isdir(result))
            self.assertEqual(result, os.path.join(tmpdir, "new_subdir"))
    
    def test_rejects_absolute_subfolder(self):
        """Test that absolute subfolders are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                get_safe_output_dir(tmpdir, "/etc")
    
    def test_rejects_traversal(self):
        """Test that traversal is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                get_safe_output_dir(tmpdir, "../other_dir")


if __name__ == '__main__':
    unittest.main()
