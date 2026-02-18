
import sys
import os
import types

# Emulate ComfyUI environment
# Add 'custom_nodes' to path so 'import radiance' works
current_file = os.path.abspath(__file__) # custom_nodes/radiance/tests/test_imports.py
radiance_dir = os.path.dirname(os.path.dirname(current_file)) # custom_nodes/radiance
custom_nodes_dir = os.path.dirname(radiance_dir) # custom_nodes
comfy_root = os.path.dirname(custom_nodes_dir) # ComfyUI

if custom_nodes_dir not in sys.path:
    sys.path.append(custom_nodes_dir)

if comfy_root not in sys.path:
    sys.path.append(comfy_root)

# Mock folder_paths (since it might not be importable even if in path if there are other deps)
# But trying real import first is better if comfy is in path.
# However, folder_paths often depends on other things. Let's try to mock it if import fails.
try:
    import folder_paths
except ImportError:
    dummy_fp = types.ModuleType("folder_paths")
    dummy_fp.get_input_directory = lambda: "input"
    dummy_fp.get_output_directory = lambda: "output"
    sys.modules["folder_paths"] = dummy_fp

print(f"Added to path: {custom_nodes_dir}")
print(f"Added to path: {comfy_root}")

try:
    import radiance
    print(f"Radiance package found: {radiance}")
except ImportError as e:
    print(f"Failed to import radiance: {e}")
    sys.exit(1)

failed = False

print("-" * 20)

# Test 1: Import from legacy nodes_exr (Facade)
try:
    from radiance.nodes_exr import RadianceSaveEXR as EXR1
    # Check if it's the class we expect (from hdr.io)
    if "radiance.hdr.io" in str(EXR1):
        print("✅ Imported RadianceSaveEXR from radiance.nodes_exr (Correctly redirected)")
    else:
        print(f"⚠️ Imported RadianceSaveEXR from radiance.nodes_exr but module is {EXR1.__module__}")
        # Depending on how it's imported in facade, module might be radiance.hdr.io
except ImportError as e:
    print(f"❌ Failed to import from nodes_exr: {e}")
    failed = True

# Test 2: Import from new facade nodes_hdr
try:
    from radiance.nodes_hdr import RadianceSaveEXR as EXR2
    print("✅ Imported RadianceSaveEXR from radiance.nodes_hdr")
except ImportError as e:
    print(f"❌ Failed to import from nodes_hdr: {e}")
    failed = True

# Test 3: Import from source hdr.io
try:
    from radiance.hdr.io import RadianceSaveEXR as EXR3
    print("✅ Imported RadianceSaveEXR from radiance.hdr.io")
except ImportError as e:
    print(f"❌ Failed to import from hdr.io: {e}")
    failed = True

if failed:
    sys.exit(1)
