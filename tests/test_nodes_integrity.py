import os
import sys
import unittest
import importlib
import inspect


class TestNodesIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )
        try:
            import unittest.mock as mock

            mock_folder_paths = mock.MagicMock()
            mock_folder_paths.get_output_directory.return_value = "/tmp/comfy_output"  # nosec B108
            mock_folder_paths.get_temp_directory.return_value = "/tmp/comfy_temp"  # nosec B108
            mock_folder_paths.get_annotated_filepath.return_value = "/tmp/annotated"  # nosec B108
            sys.modules["folder_paths"] = mock_folder_paths

            mock_comfy_utils = mock.MagicMock()
            mock_comfy_utils.ProgressBar = mock.MagicMock()
            sys.modules["comfy.utils"] = mock_comfy_utils

            mock_model_mgmt = mock.MagicMock()
            mock_model_mgmt.get_torch_device.return_value = "cpu"
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
            print(f"Failed to mock and load: {e}")

    def test_package_import(self):
        self.assertIsNotNone(self.radiance_pkg, "Failed to import Radiance package")
        self.assertGreater(
            len(self.mappings), 0, "No nodes were exported in NODE_CLASS_MAPPINGS"
        )

    def test_all_nodes(self):
        if not self.mappings:
            self.skipTest("No mappings available to test")

        for node_name, node_class in self.mappings.items():
            with self.subTest(node=node_name):
                # 1. The class should exist and be instantiable
                self.assertTrue(
                    inspect.isclass(node_class), f"Node {node_name} is not a class"
                )

                try:
                    instance = node_class()
                except Exception as e:
                    self.fail(f"Failed to instantiate {node_name}: {e}")

                # 2. It must have an INPUT_TYPES classmethod
                self.assertTrue(
                    hasattr(node_class, "INPUT_TYPES"),
                    f"Node {node_name} missing INPUT_TYPES",
                )
                self.assertTrue(
                    callable(getattr(node_class, "INPUT_TYPES")),
                    f"{node_name}.INPUT_TYPES must be callable",
                )

                try:
                    inputs = node_class.INPUT_TYPES()
                except Exception as e:
                    self.fail(f"Failed to call {node_name}.INPUT_TYPES(): {e}")

                self.assertIsInstance(
                    inputs, dict, f"{node_name}.INPUT_TYPES() must return a dict"
                )
                self.assertIn(
                    "required",
                    inputs,
                    f"{node_name}.INPUT_TYPES() must contain 'required' key",
                )
                self.assertIsInstance(
                    inputs["required"],
                    dict,
                    f"{node_name} 'required' inputs must be a dict",
                )

                # 3. It must have a RETURN_TYPES attribute (tuple)
                self.assertTrue(
                    hasattr(node_class, "RETURN_TYPES"),
                    f"Node {node_name} missing RETURN_TYPES",
                )
                self.assertIsInstance(
                    node_class.RETURN_TYPES,
                    tuple,
                    f"{node_name}.RETURN_TYPES must be a tuple",
                )

                # 4. It must have a FUNCTION attribute (string)
                self.assertTrue(
                    hasattr(node_class, "FUNCTION"),
                    f"Node {node_name} missing FUNCTION",
                )
                self.assertIsInstance(
                    node_class.FUNCTION, str, f"{node_name}.FUNCTION must be a string"
                )

                # 5. The FUNCTION must match an actual method on the class
                func_name = node_class.FUNCTION
                self.assertTrue(
                    hasattr(instance, func_name),
                    f"Node {node_name} specifies FUNCTION '{func_name}' but the class does not have this method",
                )
                self.assertTrue(
                    callable(getattr(instance, func_name)),
                    f"Node {node_name}.{func_name} is not callable",
                )


if __name__ == "__main__":
    unittest.main()
