import os
import sys
import unittest

# Add radiance to path
sys.path.append(r'd:\A.I\ComfyUI\custom_nodes\radiance')

from path_utils import get_safe_output_dir

class TestPaths(unittest.TestCase):
    def test_absolute_path_allowed(self):
        base = r"C:\fake_output"
        # Since I'm on Windows, I'll use a likely existing or safe drive
        abs_path = r"d:\Radiance_Test_Output"
        try:
            # We don't want to actually create dirs in a unit test if we can help it, 
            # but here it's part of the function. 
            # I'll use a path that is likely safe or just check the return value.
            res = get_safe_output_dir(base, abs_path, allow_absolute=True)
            print(f"Result for absolute: {res}")
            self.assertEqual(os.path.normpath(res), os.path.normpath(abs_path))
            self.assertTrue(os.path.isdir(res))
            # Cleanup
            os.rmdir(res)
        except Exception as e:
            self.fail(f"get_safe_output_dir failed with absolute path: {e}")

    def test_absolute_path_blocked_by_default(self):
        base = r"C:\fake_output"
        abs_path = r"d:\Radiance_Test_Output_Blocked"
        with self.assertRaises(ValueError):
            get_safe_output_dir(base, abs_path, allow_absolute=False)

    def test_relative_path_stays_in_base(self):
        base = os.path.abspath("test_output_base")
        os.makedirs(base, exist_ok=True)
        sub = "shots/shot01"
        res = get_safe_output_dir(base, sub, allow_absolute=True)
        print(f"Result for relative: {res}")
        self.assertTrue(res.startswith(base))
        self.assertTrue(os.path.isdir(res))
        # Cleanup
        import shutil
        shutil.rmtree(base)

if __name__ == "__main__":
    unittest.main()
