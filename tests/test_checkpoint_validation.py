"""
test_checkpoint_validation.py — RUDRA checkpoint integrity pre-check.

_validate_safetensors_size() parses only the safetensors header (8-byte
length prefix + JSON), so these tests run stub-safe: no torch, no
safetensors package needed. Regression for the ltx-video full-decoder
checkpoint that shipped truncated (23 MB on disk, header declared ~36 MB,
45/124 tensors out of bounds) and failed deep inside deserialization with
a cryptic error instead of a clear diagnostic.
"""
import json
import struct
import unittest
import tempfile
import os

from radiance.fast_vae import _validate_safetensors_size


def _make_safetensors_bytes(tensors: dict) -> bytes:
    """Minimal valid safetensors container: header + zero-filled data blob."""
    header = json.dumps(tensors).encode("utf-8")
    data_len = max((m["data_offsets"][1] for m in tensors.values()
                    if isinstance(m, dict) and "data_offsets" in m), default=0)
    return struct.pack("<Q", len(header)) + header + b"\x00" * data_len


class TestValidateSafetensorsSize(unittest.TestCase):
    def _write(self, blob: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        self.addCleanup(os.unlink, path)
        return path

    def test_valid_file_passes(self):
        blob = _make_safetensors_bytes({
            "layer.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
            "layer.bias":   {"dtype": "F32", "shape": [2],    "data_offsets": [16, 24]},
        })
        self.assertIsNone(_validate_safetensors_size(self._write(blob)))

    def test_truncated_file_reports_missing_bytes_and_oob_tensors(self):
        blob = _make_safetensors_bytes({
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
            "c": {"dtype": "F32", "shape": [4], "data_offsets": [32, 48]},
        })
        err = _validate_safetensors_size(self._write(blob[:-20]))  # cut 20 bytes
        self.assertIsNotNone(err)
        self.assertIn("truncated", err)
        self.assertIn("2 of 3 tensors out of bounds", err)

    def test_tiny_file_rejected(self):
        err = _validate_safetensors_size(self._write(b"\x01\x02"))
        self.assertIsNotNone(err)

    def test_header_longer_than_file_rejected(self):
        blob = struct.pack("<Q", 10_000_000) + b"{}"
        err = _validate_safetensors_size(self._write(blob))
        self.assertIsNotNone(err)
        self.assertIn("corrupt", err)

    def test_garbage_header_rejected(self):
        blob = struct.pack("<Q", 8) + b"notjson!"
        err = _validate_safetensors_size(self._write(blob))
        self.assertIsNotNone(err)
        self.assertIn("unreadable", err)

    def test_metadata_key_ignored(self):
        blob = _make_safetensors_bytes({
            "__metadata__": {"format": "pt"},
            "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        })
        self.assertIsNone(_validate_safetensors_size(self._write(blob)))


if __name__ == "__main__":
    unittest.main()
