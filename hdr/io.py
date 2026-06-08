import torch
import numpy as np
import os
import json
import logging
try:
    import folder_paths
except ImportError:
    folder_paths = None
import datetime
import struct
import zlib
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Local imports
from .utils import numpy_to_tensor_float32

# Imports
__all__ = [
    "write_exr_robust",
    "write_exr_openexr",
    "write_exr_cv2",
    "write_exr_imageio",
    "check_openexr_available",
    "SimpleEXRWriter",
    "write_hdr_rgbe",
    "build_radiance_hdr_metadata",
]

try:
    from radiance.color.transfer import (
        linear_to_logc3,
        linear_to_logc4,
        linear_to_slog3,
        srgb_to_linear,
        linear_to_srgb,
    )
    from radiance.color.matrices import (
        apply_matrix_transform,
        AWG3_TO_ACESCG,
        AWG4_TO_ACESCG,
        SGAMUT3_CINE_TO_ACESCG,
        ACESCG_TO_SRGB,
    )

    _HAS_COLOR_UTILS = True
except ImportError:
    _HAS_COLOR_UTILS = False

if not _HAS_COLOR_UTILS:
    try:
        from radiance.color.transfer import (
            linear_to_logc3,
            linear_to_logc4,
            linear_to_slog3,
            srgb_to_linear,
            linear_to_srgb,
        )
        from radiance.color.matrices import (
            apply_matrix_transform,
            AWG3_TO_ACESCG,
            AWG4_TO_ACESCG,
            SGAMUT3_CINE_TO_ACESCG,
            ACESCG_TO_SRGB,
        )
        _HAS_COLOR_UTILS = True
    except ImportError:
        pass

logger = logging.getLogger("radiance.hdr.io")

# ═══════════════════════════════════════════════════════════════════════════════
#                           EXR CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_BATCH_SIZE = 1000
MAX_IMAGE_DIMENSION = 32768

EXR_COMPRESSION = {
    "None": 0, "RLE": 1, "ZIPS": 2, "ZIP": 3, "PIZ": 4,
    "PXR24": 5, "B44": 6, "B44A": 7, "DWAA": 8, "DWAB": 9,
}
EXR_PIXEL_TYPE = {"UINT": 0, "HALF": 1, "FLOAT": 2}
EXR_LINE_ORDER = {"INCREASING_Y": 0, "DECREASING_Y": 1, "RANDOM_Y": 2}

def _safe_inverse(matrix: np.ndarray, name: str) -> Optional[np.ndarray]:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        logger.warning(f"Failed to invert {name} matrix: {e}")
        return None

try:
    SRGB_TO_ACESCG = _safe_inverse(ACESCG_TO_SRGB, "ACESCG_TO_SRGB")
    ACESCG_TO_AWG3 = _safe_inverse(AWG3_TO_ACESCG, "AWG3_TO_ACESCG")
    ACESCG_TO_AWG4 = _safe_inverse(AWG4_TO_ACESCG, "AWG4_TO_ACESCG")
    ACESCG_TO_SGAMUT = _safe_inverse(SGAMUT3_CINE_TO_ACESCG, "SGAMUT3_CINE_TO_ACESCG")
except NameError:
    SRGB_TO_ACESCG = ACESCG_TO_AWG3 = ACESCG_TO_AWG4 = ACESCG_TO_SGAMUT = None

def float32_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def float16_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float16).tobytes()

def parse_timecode(tc_str: str, fps: float = 24.0) -> Optional[Any]:
    """Convert HH:MM:SS:FF or HH:MM:SS;FF string to Imath.TimeCode."""
    try:
        import Imath
        parts = tc_str.replace(";", ":").split(":")
        if len(parts) != 4:
            return None
        h, m, s, f = map(int, parts)
        # Drop frame detection (simplified: if ; is used or if FPS is 29.97/59.94)
        is_drop = ";" in tc_str or abs(fps - 29.97) < 0.01 or abs(fps - 59.94) < 0.01
        return Imath.TimeCode(h, m, s, f, rate=int(round(fps)), dropFrame=is_drop)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#                           EXR WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_exr_cv2(filepath: str, image: np.ndarray, bit_depth: str = "float32", compression: str = "ZIP") -> bool:
    try:
        import cv2
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        img = image.astype(np.float32)
        if img.ndim == 3:
            if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if "16" in str(bit_depth) else cv2.IMWRITE_EXR_TYPE_FLOAT]
        return cv2.imwrite(filepath, img, flags)
    except Exception: return False

def check_openexr_available() -> bool:
    try:
        import OpenEXR
        return True
    except Exception: return False

def write_exr_openexr(filepath: str, channels: Dict[str, np.ndarray], compression: str = "ZIP", pixel_type: str = "HALF", metadata: Optional[Dict[str, Any]] = None) -> None:
    import OpenEXR, Imath
    height, width = int(list(channels.values())[0].shape[0]), int(list(channels.values())[0].shape[1])
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    header = OpenEXR.Header(width, height)
    comp_map = {"None": 0, "RLE": 1, "ZIPS": 2, "ZIP": 3, "PIZ": 4, "PXR24": 5, "B44": 6, "B44A": 7, "DWAA": 8, "DWAB": 9}
    header["compression"] = Imath.Compression(comp_map.get(compression, 3))
    ptype = Imath.PixelType(Imath.PixelType.HALF if pixel_type == "HALF" else Imath.PixelType.FLOAT)
    header["channels"] = {n: Imath.Channel(ptype) for n in sorted(channels.keys())}
    if metadata:
        # Standard OpenEXR attributes and their Imath types
        for k, v in metadata.items():
            if k == "timeCode":
                tc = parse_timecode(str(v))
                if tc: header["timeCode"] = tc
            elif k in ["reelName", "capDate", "software", "comments", "owner"]:
                header[k] = v
            elif isinstance(v, (str, int, float)):
                # Prefix Radiance-specific metadata to avoid collisions unless it is standard or Cryptomatte
                key = k if (k.startswith("cryptomatte") or k.startswith("rad_")) else f"rad_{k}"
                header[key] = v
    out = OpenEXR.OutputFile(filepath, header)
    out.writePixels({n: d.astype(np.float16 if pixel_type == "HALF" else np.float32).tobytes() for n, d in channels.items()})
    out.close()

class SimpleEXRWriter:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    def write(self, filepath: str, channels: Dict[str, np.ndarray], compression: str = "ZIP", pixel_type: str = "HALF", metadata: Optional[Dict[str, Any]] = None):
        height, width = list(channels.values())[0].shape[:2]
        ch_list = sorted(channels.keys())
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(struct.pack("<I", 20000630)) # Magic
            f.write(struct.pack("<I", 2)) # Version
            # Simplified header writer
            def write_attr(n, t, d):
                f.write(n.encode() + b"\x00" + t.encode() + b"\x00" + struct.pack("<I", len(d)) + d)
            
            ch_data = b""
            pt = 1 if pixel_type == "HALF" else 2
            for n in ch_list:
                ch_data += n.encode() + b"\x00" + struct.pack("<I", pt) + b"\x00\x00\x00\x00" + struct.pack("<ii", 1, 1)
            ch_data += b"\x00"
            write_attr("channels", "chlist", ch_data)
            write_attr("compression", "compression", struct.pack("<B", EXR_COMPRESSION.get(compression, 3)))
            write_attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, width-1, height-1))
            write_attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, width-1, height-1))
            write_attr("lineOrder", "lineOrder", b"\x00")
            write_attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
            write_attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
            write_attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
            f.write(b"\x00") # End header
            
            offsets_pos = f.tell()
            for _ in range(height): f.write(struct.pack("<Q", 0))
            offsets = []
            for y in range(height):
                offsets.append(f.tell())
                f.write(struct.pack("<iI", y, 0))  # scan-line y + data size (patched below)
                line = b"".join([channels[n][y].astype(np.float16 if pixel_type == "HALF" else np.float32).tobytes() for n in ch_list])
                if compression in ["ZIP", "ZIPS"]: line = zlib.compress(line)
                pos = f.tell()
                f.seek(offsets[-1] + 4)
                f.write(struct.pack("<I", len(line)))
                f.seek(pos)
                f.write(line)
            
            pos = f.tell()
            f.seek(offsets_pos)
            for o in offsets: f.write(struct.pack("<Q", o))
            f.seek(pos)

def rgb_to_rgbe(rgb: np.ndarray) -> np.ndarray:
    max_val = np.maximum(np.max(rgb, axis=-1, keepdims=True), 1e-32)
    exponent = np.floor(np.log2(max_val)) + 128
    exponent = np.clip(exponent, 0, 255).astype(np.uint8)
    mantissa = np.clip((rgb / (2.0**(exponent.astype(np.float32)-128))) * 256.0, 0, 255).astype(np.uint8)
    return np.concatenate([mantissa, exponent], axis=-1)

def write_hdr_rgbe(filepath: str, image: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        h, w = image.shape[:2]
        with open(filepath, "wb") as f:
            f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode())
            for y in range(h): f.write(rgb_to_rgbe(image[y]).tobytes())
        return True
    except Exception: return False

def write_exr_imageio(filepath: str, image: np.ndarray, pixel_type: str = "HALF") -> bool:
    """Fallback EXR writer using imageio."""
    try:
        import imageio.v3 as iio
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Note: imageio usually requires freeimage plugin for EXR
        iio.imwrite(filepath, image.astype(np.float32))
        return True
    except Exception:
        return False

def write_exr_robust(filepath: str, image: np.ndarray, bit_depth: str = "32-bit Float", compression: str = "ZIP", metadata: Optional[Dict[str, Any]] = None) -> bool:
    if write_exr_cv2(filepath, image, bit_depth, compression): return True
    channels = {"R": image[..., 0], "G": image[..., 1], "B": image[..., 2]}
    if image.shape[-1] == 4: channels["A"] = image[..., 3]
    ptype = "HALF" if "16" in str(bit_depth) else "FLOAT"
    if check_openexr_available():
        try:
            write_exr_openexr(filepath, channels, compression, ptype, metadata)
            return True
        except Exception: pass
    try:
        SimpleEXRWriter().write(filepath, channels, compression, ptype, metadata)
        return True
    except Exception: return False


# ═══════════════════════════════════════════════════════════════════════════════
#               HDR METADATA BUILDER  (v2.4)
# ═══════════════════════════════════════════════════════════════════════════════

_RADIANCE_VERSION = "Radiance v3.0.0"

# Map of log-space names to a brief human-readable curve description embedded
# in the color_pipeline summary and the rad_hdr_curve attribute.
_LOG_CURVE_DESC: Dict[str, str] = {
    "ARRI LogC3":            "ARRI LogC3 (AWG3, ~800% EI, γ≈0.25)",
    "ARRI LogC4":            "ARRI LogC4 (AWG4, ~4500%)",
    "Sony S-Log3":           "Sony S-Log3 (S-Gamut3.Cine)",
    "Panasonic V-Log":       "Panasonic V-Log (V-Gamut)",
    "DaVinci Intermediate":  "DaVinci Intermediate",
    "RED Log3G10":           "RED Log3G10 (REDWideGamutRGB)",
    "Linear":                "Scene-Linear",
    "sRGB":                  "sRGB (display-referred)",
    "ACEScg":                "ACEScg (AP1 linear)",
    "ACES 2065-1":           "ACES 2065-1 (AP0 linear)",
    "Rec.2020 Linear":       "Rec.2020 Scene-Linear",
    "Raw":                   "Raw passthrough (no transform)",
}


def build_radiance_hdr_metadata(
    source_space: str = "Linear",
    hdr_mode: str = "Passthrough",
    decode_noise_scale: float = 0.0,
    exposure: float = 0.0,
    target_space: str = "sRGB",
    display_tonemap: str = "None",
    lora_name: str = "",
    lora_type: str = "IC-LoRA",
) -> Dict[str, Any]:
    """
    Build a standardised EXR attribute dictionary capturing the full HDR
    generation pipeline used by Radiance to produce this image.

    All Radiance-specific keys are prefixed with ``rad_`` to avoid collisions
    with standard OpenEXR attributes.  The two standard attributes
    ``software`` and ``comments`` are also set to give generic viewers a
    human-readable summary without any special tooling.

    Args:
        source_space:        Log/linear input space fed into the VAE encoder
                             (e.g. "ARRI LogC3", "Linear").
        hdr_mode:            Encode HDR mode ("Compress (Log)", "Passthrough",
                             "Clip", "Soft Clip").
        decode_noise_scale:  Latent noise injection scale used at decode time
                             (0 = disabled).
        exposure:            EV exposure shift applied before encoding.
        target_space:        Output color space from the decoder
                             (e.g. "ACEScg", "ARRI LogC4", "sRGB").
        display_tonemap:     Display tonemap operator applied ("Reinhard",
                             "ACES", "Filmic", "None").
        lora_name:           Name of the IC-LoRA / LoRA model used for
                             generation (empty string if none).
        lora_type:           LoRA variant type ("IC-LoRA", "LoRA", etc.).

    Returns:
        Dict mapping EXR attribute name → string value, ready to pass as the
        ``metadata`` argument to ``write_exr_openexr`` / ``write_exr_robust``.
    """
    curve_desc = _LOG_CURVE_DESC.get(source_space, source_space)

    # Build a compact human-readable pipeline summary
    pipeline_steps = [curve_desc]
    if hdr_mode != "Passthrough":
        pipeline_steps.append(f"HDR-{hdr_mode}")
    if decode_noise_scale > 0.0:
        pipeline_steps.append(f"noise={decode_noise_scale:.4f}")
    if exposure != 0.0:
        pipeline_steps.append(f"EV{exposure:+.2f}")
    pipeline_steps.append(f"→ {target_space}")
    if display_tonemap and display_tonemap.lower() not in ("none", ""):
        pipeline_steps.append(f"[TM:{display_tonemap}]")
    color_pipeline = " | ".join(pipeline_steps)

    now_iso = datetime.datetime.now().isoformat(timespec="seconds")

    meta: Dict[str, Any] = {
        # ── Standard OpenEXR attributes (readable by any viewer) ─────────────
        "software": _RADIANCE_VERSION,
        "comments": f"Generated by {_RADIANCE_VERSION}. Pipeline: {color_pipeline}",
        # ── Radiance-specific attributes (rad_ prefix) ────────────────────────
        "rad_generator":         _RADIANCE_VERSION,
        "rad_created":           now_iso,
        "rad_hdr_source_space":  source_space,
        "rad_hdr_mode":          hdr_mode,
        "rad_hdr_exposure":      f"{exposure:+.4f}",
        "rad_hdr_noise_scale":   f"{decode_noise_scale:.6f}",
        "rad_hdr_target_space":  target_space,
        "rad_hdr_tonemap":       display_tonemap if display_tonemap else "None",
        "rad_color_pipeline":    color_pipeline,
        "rad_hdr_curve":         curve_desc,
    }

    if lora_name:
        meta["rad_lora_name"] = lora_name
        meta["rad_lora_type"] = lora_type

    return meta


# ═══════════════════════════════════════════════════════════════════════════════
#                    EXR MULTI-PART WRITER (v2.4 Phase 5.4)
# ═══════════════════════════════════════════════════════════════════════════════

def write_exr_multipart(
    filepath: str,
    parts: Dict[str, np.ndarray],
    bit_depth: str = "16-bit Half Float",
    compression: str = "ZIP",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Write a multi-part EXR v2 file with named layers.

    Each entry in `parts` is written as a separate named part:
      • "beauty"  → channels R, G, B (or R, G, B, A if 4-channel)
      • "depth"   → channel Z
      • "normal"  → channels NX, NY, NZ
      • "albedo"  → channels albedo.R, albedo.G, albedo.B
      • any key   → written with layered channel names (key.R, key.G, ...)

    Standard single-channel parts (keys starting with "depth" or "z") are written
    as a single Z channel.

    Compatibility:
      • Nuke 13+: connects automatically via ReadGeo / Read multi-part
      • DaVinci Resolve: requires "Flatten Layers" mode
      • Fusion: multi-part compatible natively

    Args:
        filepath: Output .exr path.
        parts:    Dict of part_name → numpy array (H, W, C) float32.
        bit_depth: "16-bit Half Float" or "32-bit Float".
        compression: EXR compression (ZIP, PIZ, ZIPS, etc.).
        metadata: Optional dict of string metadata embedded in each part header.

    Returns:
        True on success, False on failure.
    """
    if not parts:
        logger.warning("[EXR MultiPart] No parts provided.")
        return False

    ptype_str = "HALF" if "16" in bit_depth else "FLOAT"

    # ── Attempt via OpenEXR (most correct multi-part support) ─────────────────
    if check_openexr_available():
        try:
            import OpenEXR, Imath

            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            comp_code = {
                "None": 0, "RLE": 1, "ZIPS": 2, "ZIP": 3, "PIZ": 4,
                "PXR24": 5, "B44": 6, "B44A": 7, "DWAA": 8, "DWAB": 9
            }.get(compression, 3)
            ptype = Imath.PixelType(
                Imath.PixelType.HALF if ptype_str == "HALF" else Imath.PixelType.FLOAT
            )
            np_dtype = np.float16 if ptype_str == "HALF" else np.float32

            headers = []
            for part_name, img in parts.items():
                if img is None:
                    continue
                arr = np.asarray(img, dtype=np.float32)
                h, w = arr.shape[:2]
                n_ch = arr.shape[2] if arr.ndim == 3 else 1

                header = OpenEXR.Header(w, h)
                header["compression"] = Imath.Compression(comp_code)
                header["name"] = part_name
                header["type"] = b"scanlineimage"

                # Build channel names
                is_depth = part_name.lower() in ("depth", "z", "zdepth", "depth_z")
                if is_depth or n_ch == 1:
                    ch_names = ["Z"]
                elif n_ch == 2:
                    ch_names = ["R", "G"]
                elif n_ch == 3:
                    ch_names = ["R", "G", "B"] if part_name == "beauty" else \
                               ["NX", "NY", "NZ"] if "normal" in part_name.lower() else \
                               [f"{part_name}.R", f"{part_name}.G", f"{part_name}.B"]
                elif n_ch >= 4:
                    if part_name == "beauty":
                        ch_names = ["R", "G", "B", "A"]
                    else:
                        ch_names = [f"{part_name}.R", f"{part_name}.G",
                                    f"{part_name}.B", f"{part_name}.A"]
                else:
                    ch_names = [part_name]

                header["channels"] = {n: Imath.Channel(ptype) for n in ch_names}

                if metadata:
                    for k, v in metadata.items():
                        if k in ["timeCode", "reelName", "capDate", "software", "comments", "owner"]:
                            header[k] = v
                        elif isinstance(v, (str, int, float)):
                            key = k if (k.startswith("cryptomatte") or k.startswith("rad_")) else f"rad_{k}"
                            header[key] = v

                headers.append((part_name, arr, ch_names, header))

            out = OpenEXR.MultiPartOutputFile(filepath, [h for _, _, _, h in headers])
            for i, (part_name, arr, ch_names, _) in enumerate(headers):
                h, w = arr.shape[:2]
                ch_data = {}
                if arr.ndim == 2:
                    arr = arr[..., np.newaxis]
                for ci, cname in enumerate(ch_names):
                    ch_slice = arr[..., ci] if ci < arr.shape[2] else arr[..., 0]
                    ch_data[cname] = ch_slice.astype(np_dtype).tobytes()
                out.writePixels(i, ch_data)
            out.close()

            logger.info(f"[EXR MultiPart] Written {len(headers)} parts → {filepath}")
            return True

        except Exception as e:
            logger.warning(f"[EXR MultiPart] OpenEXR multi-part failed ({e}), falling back to per-layer files.")

    # ── Fallback: write separate EXR files per part ───────────────────────────
    base, _ = os.path.splitext(filepath)
    success_count = 0
    for part_name, img in parts.items():
        if img is None:
            continue
        layer_path = f"{base}.{part_name}.exr"
        arr = np.asarray(img, dtype=np.float32)
        if write_exr_robust(layer_path, arr, bit_depth, compression, metadata):
            success_count += 1
            logger.info(f"[EXR MultiPart] Fallback→ wrote {layer_path}")

    if success_count > 0:
        logger.warning(
            f"[EXR MultiPart] Wrote {success_count} separate EXR files instead of multi-part "
            f"(OpenEXR multi-part unavailable). Load each file individually in Nuke."
        )
    return success_count > 0
