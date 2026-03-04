import os
import sys
import torch
import numpy as np
import time
import concurrent.futures
import json
import zlib
import struct
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Mocking the folder_paths
class MockFolderPaths:
    @staticmethod
    def get_temp_directory():
        return "tmp_qa_test"
    @staticmethod
    def get_output_directory():
        return "tmp_qa_output"

sys.modules['folder_paths'] = MockFolderPaths()

# Import the targeted classes
# When running as 'python -m radiance.tests.qa_stress_test', 
# the current directory should be the parent of 'radiance'.
try:
    from radiance.nodes_radiance_viewer import RadianceViewer
    from radiance.nodes_io import RadianceWrite
except ImportError as e:
    print(f"Error importing Radiance modules: {e}")
    # Fallback for standalone run
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from radiance.nodes_radiance_viewer import RadianceViewer
    from radiance.nodes_io import RadianceWrite

# Ensure temp dir exists relative to project root
module_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(module_root, "tmp_qa_test"), exist_ok=True)
os.makedirs(os.path.join(module_root, "tmp_qa_output"), exist_ok=True)

def test_stress_concurrency(num_requests=10):
    print(f"--- Stress Test: {num_requests} Concurrent Frame Processes ---")
    viewer = RadianceViewer()
    dummy_image = torch.rand((1, 1024, 1024, 3))
    start_time = time.time()
    
    def process_request(i):
        try:
            viewer.view(dummy_image, bit_depth="16-bit (Quality)")
            return True
        except Exception as e:
            return f"Request {i} failed: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_request, range(num_requests)))
    
    end_time = time.time()
    failures = [r for r in results if r is not True]
    print(f"Completed {num_requests} requests in {end_time - start_time:.2f}s")
    if failures:
        print(f"Failures: {len(failures)}")
    else:
        print("All requests successful.")

def test_security_path_traversal():
    print("--- Security Test: Path Traversal ---")
    writer = RadianceWrite()
    images = torch.rand((1, 64, 64, 3))
    malicious_subfolder = "../temp_pwn"
    print(f"Testing subfolder traversal with: {malicious_subfolder}")
    
    try:
        writer.write(images, "traversal_test", "Video — MP4 (H.264)", 24.0, 10, "sRGB (Standard)", subfolder=malicious_subfolder)
        base_path = os.path.abspath("tmp_qa_output")
        escaped_path = os.path.abspath(os.path.join(base_path, malicious_subfolder))
        
        if not os.path.normpath(escaped_path).startswith(os.path.normpath(base_path)):
            print(f"VULNERABILITY DETECTED: Path traversal escaped to {escaped_path}")
        else:
            print("Path traversal blocked or within bounds.")
    except Exception as e:
        print(f"Caught error: {e}")

if __name__ == "__main__":
    test_stress_concurrency(10)
    test_security_path_traversal()
