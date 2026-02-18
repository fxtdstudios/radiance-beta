import unittest
import os
import sys
import types
from unittest.mock import MagicMock

# Create dummy modules
comfy = types.ModuleType("comfy")
comfy.samplers = types.ModuleType("comfy.samplers")
comfy.sample = types.ModuleType("comfy.sample")
comfy.utils = types.ModuleType("comfy.utils")
comfy.model_management = types.ModuleType("comfy.model_management")
comfy.sd = types.ModuleType("comfy.sd") # Add this

sys.modules["comfy"] = comfy
sys.modules["comfy.samplers"] = comfy.samplers
sys.modules["comfy.sample"] = comfy.sample
sys.modules["comfy.utils"] = comfy.utils
sys.modules["comfy.model_management"] = comfy.model_management
sys.modules["comfy.sd"] = comfy.sd
sys.modules["folder_paths"] = MagicMock()

# Also mock torch if not present (though it likely is)
# sys.modules["torch"] = MagicMock() 

if __name__ == "__main__":
    # Add parent directory to path so we can import modules as if we are in the package
    # But for relative imports in the package to work, we need to be running from parent
    # OR we hack sys.path to include parent of current dir
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    # Add grandparent to path so 'radiance' is importable
    if grandparent_dir not in sys.path:
        sys.path.insert(0, grandparent_dir)
    
    # Discovery
    loader = unittest.TestLoader()
    suite = loader.discover(current_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(not result.wasSuccessful())
