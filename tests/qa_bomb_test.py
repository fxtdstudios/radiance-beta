
import os
import struct
import zlib
import unittest

class TestRHDRDecompressionBomb(unittest.TestCase):
    def test_generate_rhdr_bomb(self):
        """
        Generates a small RHDR file that claims to be huge.
        Used to manually verify if the JS loader (via server proxy) handles it.
        """
        # Header: Magic(4), Width(2), Height(2), Channels(2), Reserved(2)
        # We claim it's 8192x8192x3 = 192MB raw.
        magic = b'RHDR'
        w, h, c = 8192, 8192, 3
        header = magic + struct.pack('<HHH H', w, h, c, 0)
        
        # Payload: 1KB of zeros compressed.
        # A real bomb would be highly compressed zeros that expand to 192MB+.
        payload = zlib.compress(b'\x00' * 1024)
        
        output_path = "tests/test_bomb.rhdr"
        with open(output_path, "wb") as f:
            f.write(header + payload)
        
        print(f"\n[SEC] Generated test bomb: {output_path} ({len(header + payload)} bytes)")
        print(f"[SEC] Claims to be: {w}x{h}x{c} ({w*h*c*2} bytes raw)")
        
        # Verify sizes
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
