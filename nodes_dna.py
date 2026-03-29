import torch
import numpy as np
import json
import zlib
import struct
import time
import logging
from typing import Dict, Any, Tuple, Optional, List

# Module logger
logger = logging.getLogger("◎ Radiance.dna")

# Get version from package
try:
    from . import __version__ as RADIANCE_VERSION
except ImportError:
    RADIANCE_VERSION = "2.3"


class RadianceDigitalDNA:
    """
    Core engine for Radiance Digital DNA (Signature Architecture) v2.3.
    Embeds invisible, lossless metadata into 32-bit floating point images.

    v2.1 Fixes:
        - Special value preservation (NaN/Inf pixels skipped)
        - Explicit error reporting (no silent failures)
        - Full batch decode support (all frames, not just first)
        - Memory-optimized vectorized pipeline
        - Data validation on decode with status messages
    """

    # Magic header to identify Radiance DNA (32 bits)
    # "FXTD" in ASCII binary: 01000110 01011000 01010100 01000100
    MAGIC_HEADER = "01000110010110000101010001000100"
    VERSION = "2.3"

    # ── Bit-level helpers ─────────────────────────────────────────────

    @staticmethod
    def _float_to_int_bits(f_val):
        """Reinterpret float32 bits as uint32."""
        return struct.unpack(">I", struct.pack(">f", f_val))[0]

    @staticmethod
    def _int_bits_to_float(i_val):
        """Reinterpret uint32 bits as float32."""
        return struct.unpack(">f", struct.pack(">I", i_val))[0]

    @staticmethod
    def _build_safe_mask(flat_img: np.ndarray) -> np.ndarray:
        """
        Build a boolean mask of pixels SAFE for LSB modification.
        Skips NaN, Inf, -Inf, and denormalized values to prevent
        corruption of special float32 values in HDR/EXR workflows.

        Returns:
            Boolean array — True = safe to modify, False = skip.
        """
        safe = np.isfinite(flat_img)
        # Also skip denormals (exponent == 0, mantissa != 0)
        int_view = np.frombuffer(flat_img.tobytes(), dtype=np.uint32)
        exponent = (int_view >> 23) & 0xFF
        mantissa = int_view & 0x007FFFFF
        is_denormal = (exponent == 0) & (mantissa != 0)
        safe &= ~is_denormal
        return safe

    # ── Encode ────────────────────────────────────────────────────────

    @classmethod
    def encode(
        cls, image: torch.Tensor, metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, bool, str]:
        """
        Embed metadata into the image tensor's LSBs.

        v2.1 changes:
            - Returns (tensor, success, status_message) instead of silently returning unsigned image
            - Skips special float values (NaN/Inf/denormal) to prevent HDR corruption
            - Memory-optimized: only copies pixels that need modification

        Returns:
            (encoded_image, success_bool, status_message)
        """
        device = image.device
        img_np = image.detach().cpu().numpy().astype(np.float32).copy()

        # ── Prepare payload ──
        payload = {"dna_ver": cls.VERSION, "data": metadata}
        json_str = json.dumps(payload, separators=(",", ":"))  # compact JSON
        compressed = zlib.compress(json_str.encode("utf-8"), level=9)

        # Convert to bit stream
        bits = "".join(f"{byte:08b}" for byte in compressed)
        length_bin = f"{len(bits):032b}"

        # Full stream: Header (32) + Length (32) + Data
        full_stream = cls.MAGIC_HEADER + length_bin + bits
        stream_len = len(full_stream)

        # ── Capacity check with safe-pixel awareness ──
        flat_img = img_np.reshape(-1)
        safe_mask = cls._build_safe_mask(flat_img)
        safe_indices = np.where(safe_mask)[0]
        total_safe = len(safe_indices)

        if stream_len > total_safe:
            msg = (
                f"Insufficient safe pixels for signature: need {stream_len} bits, "
                f"only {total_safe} safe pixels available "
                f"(total={flat_img.size}, skipped={flat_img.size - total_safe} special values)"
            )
            logger.error(msg)
            return (image, False, msg)

        # ── Embed bits — only into safe pixels ──
        target_indices = safe_indices[:stream_len]
        float_vals = flat_img[target_indices]

        int_vals = np.frombuffer(float_vals.tobytes(), dtype=np.uint32).copy()

        # Clear LSB
        clear_mask = np.uint32(0xFFFFFFFE)
        int_vals &= clear_mask

        # Set LSB from stream
        stream_bits = np.array([int(b) for b in full_stream], dtype=np.uint32)
        int_vals |= stream_bits

        # Write back
        new_floats = np.frombuffer(int_vals.tobytes(), dtype=np.float32)
        flat_img[target_indices] = new_floats

        result_np = flat_img.reshape(img_np.shape)

        skipped = flat_img.size - total_safe
        msg = (
            f"✓ DNA signed: {stream_len} bits embedded "
            f"({len(compressed)} bytes compressed, {skipped} special pixels preserved)"
        )
        logger.info(msg)

        return (torch.from_numpy(result_np).to(device), True, msg)

    # ── Decode ────────────────────────────────────────────────────────

    @classmethod
    def decode(cls, image: torch.Tensor) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Attempt to read Radiance DNA from an image.

        v2.1 changes:
            - Returns (is_valid, metadata_dict, status_message) for explicit reporting
            - Safe-pixel-aware decoding (mirrors encode skip logic)
            - Validates payload integrity before returning

        Returns:
            (is_valid, metadata_dict_or_None, status_message)
        """
        img_np = image.detach().cpu().numpy().astype(np.float32)
        flat_img = img_np.reshape(-1)

        # Build same safe mask used during encoding
        safe_mask = cls._build_safe_mask(flat_img)
        safe_indices = np.where(safe_mask)[0]
        total_safe = len(safe_indices)

        header_len = len(cls.MAGIC_HEADER)
        check_len = header_len + 32  # Header + Length field

        if total_safe < check_len:
            return (
                False,
                None,
                f"Not enough safe pixels to read header ({total_safe} < {check_len})",
            )

        # ── Extract header + length from safe pixels ──
        header_pixels = flat_img[safe_indices[:check_len]]
        int_vals = np.frombuffer(header_pixels.tobytes(), dtype=np.uint32)
        lsbs = int_vals & 1
        extracted_bits = "".join(str(b) for b in lsbs)

        # Validate magic header
        extracted_header = extracted_bits[:header_len]
        if extracted_header != cls.MAGIC_HEADER:
            return (False, None, "No valid DNA signature (header mismatch)")

        # Read payload length
        length_bin = extracted_bits[header_len : header_len + 32]
        try:
            payload_len = int(length_bin, 2)
        except ValueError:
            return (False, None, "Corrupted length field")

        # Sanity check payload length
        if payload_len <= 0 or payload_len > 100_000_000:
            return (False, None, f"Invalid payload length: {payload_len}")

        total_needed = check_len + payload_len
        if total_safe < total_needed:
            return (
                False,
                None,
                f"Truncated signature: need {total_needed} safe pixels, only {total_safe} available",
            )

        # ── Extract payload bits from safe pixels ──
        payload_pixels = flat_img[safe_indices[check_len:total_needed]]
        int_payload = np.frombuffer(payload_pixels.tobytes(), dtype=np.uint32)
        payload_lsbs = int_payload & 1
        payload_bits = "".join(str(b) for b in payload_lsbs)

        # ── Reconstruct bytes ──
        try:
            byte_array = bytearray()
            for i in range(0, len(payload_bits), 8):
                byte = payload_bits[i : i + 8]
                if len(byte) == 8:
                    byte_array.append(int(byte, 2))

            json_str = zlib.decompress(bytes(byte_array)).decode("utf-8")
            data = json.loads(json_str)

            # Validate structure
            if not isinstance(data, dict):
                return (False, None, "Decoded payload is not a valid dict")
            if "dna_ver" not in data:
                return (False, None, "Decoded payload missing dna_ver field")

            return (
                True,
                data,
                f"✓ Valid DNA v{data.get('dna_ver', '?')} signature decoded",
            )

        except zlib.error as e:
            return (False, None, f"Decompression failed: {e}")
        except json.JSONDecodeError as e:
            return (False, None, f"JSON parse failed: {e}")
        except Exception as e:
            return (False, None, f"Decode error: {e}")

    # ── Batch helpers ─────────────────────────────────────────────────

    @classmethod
    def decode_batch(
        cls, images: torch.Tensor
    ) -> List[Tuple[bool, Optional[Dict[str, Any]], str]]:
        """
        Decode ALL frames in a batch (not just the first).
        Returns a list of (is_valid, data, status) per frame.
        """
        if images.dim() == 3:
            return [cls.decode(images)]

        results = []
        for i in range(images.shape[0]):
            results.append(cls.decode(images[i]))
        return results


class RadianceSignatureMixin:
    """
    Mixin for ComfyUI Nodes to easily sign their output.
    v2.1: Now reports success/failure status.
    """

    def sign_image(
        self, image: torch.Tensor, extra_metadata: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, bool, str]:
        """
        Sign the image with this node's signature.

        Returns:
            (signed_image, success, status_message)
        """
        node_class = self.__class__.__name__

        metadata = {
            "created_by": "◎ Radiance",
            "node": node_class,
            "timestamp": time.time(),
            "fxtd_ver": RADIANCE_VERSION,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        return RadianceDigitalDNA.encode(image, metadata)


# ═══════════════════════════════════════════════════════════════════════
#                              NODES
# ═══════════════════════════════════════════════════════════════════════


class RadianceDNAReader:
    """
    Reads and reports DNA signature from images.
    v2.1: Full batch decode — checks every frame, not just the first.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_signed", "signature_data")
    FUNCTION = "read_dna"
    CATEGORY = "FXTD Studios/Radiance/Data"
    OUTPUT_NODE = True
    DESCRIPTION = "Reads Radiance Digital DNA signature from all frames in a batch."

    def read_dna(self, image):
        results = RadianceDigitalDNA.decode_batch(image)
        batch_size = len(results)

        # Aggregate results
        all_valid = all(r[0] for r in results)
        # FIX 4: was a bare expression — result computed and immediately discarded.
        any_valid = any(r[0] for r in results)

        if batch_size == 1:
            is_valid, data, status = results[0]
            if is_valid and data:
                info_str = json.dumps(data.get("data", {}), indent=2)
                logger.info(f"DNA Reader: {status}")
            else:
                info_str = f"No valid signature: {status}"
            return (is_valid, info_str)

        # Multi-frame report
        report_lines = [f"Batch DNA Report ({batch_size} frames):"]
        report_lines.append(f"{'─' * 50}")

        for i, (is_valid, data, status) in enumerate(results):
            if is_valid and data:
                meta = data.get("data", {})
                node = meta.get("node", "unknown")
                ver = meta.get("fxtd_ver", "?")
                report_lines.append(
                    f"  Frame {i:>4d}: ✓ Signed by {node} (Radiance {ver})"
                )
            else:
                report_lines.append(f"  Frame {i:>4d}: ✗ {status}")

        report_lines.append(f"{'─' * 50}")
        valid_count = sum(1 for r in results if r[0])
        report_lines.append(f"Summary: {valid_count}/{batch_size} frames signed")

        if not all_valid:
            unsigned = [i for i, r in enumerate(results) if not r[0]]
            if len(unsigned) <= 10:
                report_lines.append(f"Unsigned frames: {unsigned}")
            else:
                report_lines.append(
                    f"Unsigned frames: {unsigned[:10]}... (+{len(unsigned)-10} more)"
                )

        info_str = "\n".join(report_lines)
        logger.info(f"DNA Reader: {valid_count}/{batch_size} frames valid")

        return (all_valid, info_str)


class RadianceDNAWriter:
    """
    Signs images with Radiance Digital DNA metadata.
    Provides explicit success/failure feedback.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "project": ("STRING", {"default": "", "multiline": False}),
                "artist": ("STRING", {"default": "", "multiline": False}),
                "notes": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "success", "status")
    FUNCTION = "write_dna"
    CATEGORY = "FXTD Studios/Radiance/Data"
    DESCRIPTION = (
        "Signs images with Radiance DNA metadata (preserves HDR special values)."
    )

    def write_dna(self, image, project="", artist="", notes=""):
        metadata = {
            "created_by": "◎ Radiance",
            "node": "◎ RadianceDNAWriter",
            "timestamp": time.time(),
            "fxtd_ver": RADIANCE_VERSION,
        }

        # Add optional fields (skip empty)
        if project:
            metadata["project"] = project
        if artist:
            metadata["artist"] = artist
        if notes:
            metadata["notes"] = notes

        # Sign each frame in the batch
        if image.dim() == 4 and image.shape[0] > 1:
            signed_frames = []
            statuses = []
            successes = []  # FIX 2: boolean per frame
            all_success = True

            for i in range(image.shape[0]):
                frame_signed, success, status = RadianceDigitalDNA.encode(
                    image[i], metadata
                )
                if frame_signed.dim() == 3:
                    frame_signed = frame_signed.unsqueeze(0)
                signed_frames.append(frame_signed)
                statuses.append(status)
                successes.append(success)  # FIX 2: track bool, not string
                if not success:
                    all_success = False

            result = torch.cat(signed_frames, dim=0)
            # FIX 2: was sum(1 for s in statuses if "◎" in s) — always 0 because
            # encode() returns "✓ DNA signed:..." or "Insufficient..." (no ◎).
            signed_count = sum(successes)
            total = len(successes)

            if all_success:
                status_msg = f"◎ All {total} frames signed successfully"
            else:
                status_msg = f"◎ {signed_count}/{total} frames signed"
                # FIX 3: was "◎ not in s" — always True for every frame since
                # encode() status messages never contain ◎. Now uses the boolean.
                for i, ok in enumerate(successes):
                    if not ok:
                        status_msg += f"\n  Frame {i}: {statuses[i]}"

            return (result, all_success, status_msg)
        else:
            signed, success, status = RadianceDigitalDNA.encode(image, metadata)
            return (signed, success, status)


class RadianceDNAValidator:
    """
    QC gate: validates DNA signature presence.
    Can optionally block unsigned images from proceeding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "require_signed": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "is_valid", "status")
    FUNCTION = "validate"
    CATEGORY = "FXTD Studios/Radiance/Data"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "QC gate — validates DNA signature (optionally blocks unsigned images)."
    )

    def validate(
        self, image: torch.Tensor, require_signed: bool = True
    ) -> Tuple[torch.Tensor, bool, str]:
        results = RadianceDigitalDNA.decode_batch(image)
        batch_size = len(results)
        valid_count = sum(1 for r in results if r[0])
        all_valid = valid_count == batch_size

        if all_valid:
            status = f"◎ Validated: {valid_count}/{batch_size} frames signed"
            return (image, True, status)

        if require_signed:
            unsigned = [i for i, r in enumerate(results) if not r[0]]
            status = f"◎ VALIDATION FAILED: {valid_count}/{batch_size} frames signed (unsigned: {unsigned[:20]})"
            logger.warning(status)
            return (image, False, status)

        status = f"◎ Partial: {valid_count}/{batch_size} frames signed (require_signed=False, passing through)"
        return (image, False, status)


# ═══════════════════════════════════════════════════════════════════════
#                         NODE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════

# FIX 1: Keys must be plain ASCII — ◎ belongs only in DISPLAY_NAME_MAPPINGS.
NODE_CLASS_MAPPINGS = {
    "RadianceDNAReader":    RadianceDNAReader,
    "RadianceDNAWriter":    RadianceDNAWriter,
    "RadianceDNAValidator": RadianceDNAValidator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDNAReader":    "◎ Radiance DNA Reader",
    "RadianceDNAWriter":    "◎ Radiance DNA Writer",
    "RadianceDNAValidator": "◎ Radiance DNA Validator",
}
