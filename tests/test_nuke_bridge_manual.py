import sys
import os
import time

# Add parent directory to path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.nuke_connector import NukeConnector

def test_connection():
    print("Testing connection to Nuke at 127.0.0.1:1986...")
    
    connector = NukeConnector(host="127.0.0.1", port=1986)
    
    # 1. Simple Command Test
    print("1. Sending simple print command...")
    cmd = "print('Radiance Bridge: Connection Test Successful!')"
    response = connector.send_command(cmd)
    print(f"   Response: {response}")
    
    if "Sent" in response:
        print("   ✅ Basic connection passed!")
    else:
        print("   ❌ Basic connection failed.")
        return

    # 2. Image Sync Test setup (fake file)
    print("\n2. Testing sync command generation...")
    fake_path = "C:/tmp/test_image.exr"
    try:
        response = connector.sync_image(fake_path, "TestNode")
        print(f"   Response: {response}")
        if "Sent" in response:
            print("   ✅ Sync command sent!")
            print("   (Check Nuke Script Editor output window for 'Radiance Bridge' messages)")
        else:
            print("   ❌ Sync command failed.")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    test_connection()
