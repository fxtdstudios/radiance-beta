import os
import sys
import unittest
import importlib
import re


class TestUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )
        try:
            import unittest.mock as mock

            mock_folder_paths = mock.MagicMock()
            sys.modules["folder_paths"] = mock_folder_paths
            mock_comfy_utils = mock.MagicMock()
            sys.modules["comfy.utils"] = mock_comfy_utils
            mock_model_mgmt = mock.MagicMock()
            sys.modules["comfy.model_management"] = mock_model_mgmt
            sys.modules["comfy.sample"] = mock.MagicMock()
            sys.modules["comfy.sd"] = mock.MagicMock()
            sys.modules["comfy.controlnet"] = mock.MagicMock()
            sys.modules["comfy"] = mock.MagicMock()
            sys.modules["comfy.samplers"] = mock.MagicMock()

            # Fix python package pathing for root-level __init__.py
            spec = importlib.util.spec_from_file_location(
                "radiance", os.path.join(os.path.dirname(__file__), "..", "__init__.py")
            )
            cls.radiance_pkg = importlib.util.module_from_spec(spec)
            sys.modules["radiance"] = cls.radiance_pkg
            spec.loader.exec_module(cls.radiance_pkg)

            cls.mappings = getattr(cls.radiance_pkg, "NODE_CLASS_MAPPINGS", {})
        except Exception as e:
            cls.radiance_pkg = None
            cls.mappings = {}

    def test_viewer_lut_options_match(self):
        """
        Ensures that the LUT options provided in the Python backend of RadianceViewer matched the JS frontend.
        """
        # 1. Get the Python LUT list
        viewer_class = self.mappings.get("RadianceViewer")
        if not viewer_class:
            self.skipTest("RadianceViewer node not found in exports.")

        try:
            inputs = viewer_class.INPUT_TYPES()
            python_luts = inputs["required"]["LUT"][
                0
            ]  # It's a tuple of (list, dict) usually
        except Exception as e:
            self.fail(f"Failed to extract LUT options from Python node: {e}")

        # 2. Get the JS LUT list
        js_path = os.path.join(
            os.path.dirname(__file__), "..", "js", "radiance_viewer.js"
        )
        with open(js_path, "r", encoding="utf-8") as f:
            js_content = f.read()

        # Extract the array from `this.lutOptions = [...]`
        match = re.search(r"this\.lutOptions\s*=\s*\[(.*?)\];", js_content, re.DOTALL)
        self.assertIsNotNone(
            match, "Could not find `this.lutOptions` array in radiance_viewer.js"
        )

        # Parse the strings out of the JS array
        js_luts_raw = match.group(1)
        js_luts = [
            name.strip(" '\r\n") for name in js_luts_raw.split(",") if name.strip()
        ]

        # 3. Compare them
        self.assertCountEqual(
            python_luts,
            js_luts,
            f"Mismatch between python backend LUT lists and javascript frontend.",
        )


if __name__ == "__main__":
    unittest.main()
