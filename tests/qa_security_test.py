import os
import sys
import torch
import numpy as np

# Add the parent directory to sys.path so we can import radiance
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mocking the folder_paths
class MockFolderPaths:
    @staticmethod
    def get_output_directory():
        return os.path.abspath("tmp_qa_output")

sys.modules['folder_paths'] = MockFolderPaths()

def test_security_path_traversal_fix():
    print("--- Security Test: Path Traversal Fix ---")
    
    from path_utils import get_safe_output_dir
    from folder_paths import get_output_directory
    
    base_output = get_output_directory()
    if not os.path.exists(base_output):
        os.makedirs(base_output, exist_ok=True)
    
    malicious_subfolders = [
        "../escaped",
        "../../../../../../Windows/System32",
        "C:/Pwned" if os.name == 'nt' else "/tmp/pwned"
    ]
    
    for sub in malicious_subfolders:
        print(f"Testing subfolder: {sub}")
        try:
            # This should now raise ValueError due to our fixes
            output_dir = get_safe_output_dir(base_output, sub)
            print(f"FAILED: Path escaped to {output_dir}")
        except ValueError as e:
            print(f"SUCCESS: Caught expected security error: {e}")
        except Exception as e:
            print(f"ERROR: Unexpected exception type: {type(e).__name__}: {e}")
        print("-" * 20)

    # Test valid subfolder
    print("Testing valid subfolder: 'renders/vFX'")
    try:
        output_dir = get_safe_output_dir(base_output, "renders/vFX")
        print(f"SUCCESS: Valid path allowed: {output_dir}")
        if not output_dir.startswith(base_output):
             print(f"FAILED: Valid path escaped bounds unexpectedly!")
    except Exception as e:
        print(f"FAILED: Valid path raised error: {e}")

if __name__ == "__main__":
    test_security_path_traversal_fix()
