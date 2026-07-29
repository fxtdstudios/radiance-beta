"""
nodes_io_unified.py — Radiance v3.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two nodes replace all previous scattered I/O nodes:

  RadianceRead   — universal reader  (image · EXR · video · sequence)
  RadianceWrite  — universal writer  (image · EXR · video · sequence)

Auto-detects format from file extension / path pattern.

Supported READ
──────────────
  Image     .png .jpg .jpeg .tiff .tif .bmp .webp .dpx .hdr
  EXR       .exr  (single frame)
  Video     .mov .mxf .mp4 .m2ts .mkv .webm .avi … → batch of frames
            Decoded through radiance.core.video: one raw ffmpeg pipe, at the
            source's own bit depth (10/12/16-bit survive), with ProRes 4444
            alpha carried through to the MASK output, exact frame ranges, and
            the container's colour tags read rather than ignored.
  Sequence  any of the above with %04d / #### / * pattern, or a directory

Supported WRITE
───────────────
  Image        PNG (8-bit · 16-bit) · JPEG · TIFF (16 · 32f) · DPX · WEBP · Radiance HDR (.hdr)
  EXR          16-bit half · 32-bit float
  Video        MP4 H.264 · MP4 H.265 10-bit · MOV ProRes 422 / 4444 · MOV DNxHR
  Sequence     PNG / TIFF / EXR / DPX / Radiance HDR numbered sequences
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from .config.constants import VERSION as _RADIANCE_VERSION
except Exception:  # keep the writer importable even if constants move
    _RADIANCE_VERSION = "3.1.1"

try:
    from PIL import Image as _PIL
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import folder_paths as _folder_paths
    _HAS_FOLDER_PATHS = True
except ImportError:
    _HAS_FOLDER_PATHS = False

# ALBABIT-FIX: DPX has no Pillow plugin at all (neither read nor write) --
# "DPX" was offered in the format dropdown since v3.1 but never actually
# worked. OpenImageIO is the VFX-industry-standard library for this format.
try:
    import OpenImageIO as _oiio
    _HAS_OIIO = True
except ImportError:
    _HAS_OIIO = False

from . import color_utils
from .core import formats as _formats
from .core.video import VIDEO_EXTENSIONS as _VIDEO_EXTENSIONS

try:
    from .hdr.io import write_exr_robust, write_exr_multipart
except ImportError:
    try:
        from hdr.io import write_exr_robust, write_exr_multipart  # type: ignore[import]
    except ImportError:
        write_exr_robust = None      # type: ignore[assignment]
        write_exr_multipart = None   # type: ignore[assignment]

try:
    from .path_utils import get_safe_output_dir, strip_path_quotes
except ImportError:
    try:
        from path_utils import get_safe_output_dir, strip_path_quotes  # type: ignore[import]
    except ImportError:
        def get_safe_output_dir(base, override="", **_):  # type: ignore[misc]
            return override.strip() or base
        def strip_path_quotes(path):  # type: ignore[misc]
            return path.strip().strip('"').strip("'")

log = logging.getLogger("radiance.io_unified")


# ═══════════════════════════════════════════════════════════════════════════
# § 0  Constants
# ═══════════════════════════════════════════════════════════════════════════

# ── Output formats (for RadianceWrite) ────────────────────────────────────
_FMT_IMAGE = [
    "IMG │ PNG (8-bit)",
    "IMG │ PNG (16-bit)",
    "IMG │ JPEG",
    "IMG │ TIFF (16-bit)",
    "IMG │ TIFF (32-bit float)",
    "IMG │ DPX",
    "IMG │ WEBP",
    "IMG │ EXR (16-bit half)",
    "IMG │ EXR (32-bit float)",
    "IMG │ Radiance HDR (.hdr)",
]
_FMT_VIDEO = [
    "VID │ MP4 (H.264)",
    "VID │ MP4 (H.265 10-bit)",
    "VID │ MOV (ProRes 422 HQ)",
    "VID │ MOV (ProRes 4444)",
    "VID │ MOV (DNxHR HQ)",
]
_FMT_SEQ = [
    "SEQ │ PNG (8-bit)",
    "SEQ │ PNG (16-bit)",
    "SEQ │ TIFF",
    "SEQ │ EXR (16-bit half)",
    "SEQ │ EXR (32-bit float)",
    "SEQ │ DPX",
    "SEQ │ Radiance HDR (.hdr)",
]

WRITE_FORMATS: List[str] = _FMT_IMAGE + _FMT_SEQ + _FMT_VIDEO


def _fmt_stem(fmt: str) -> str:
    """Strip group prefix (e.g. 'IMG │ PNG (8-bit)' → 'PNG (8-bit)')."""
    return fmt.split(" │ ", 1)[-1] if " │ " in fmt else fmt

# ── Output color spaces ────────────────────────────────────────────────────
OUTPUT_COLOR_SPACES = [
    "Linear (pass-through)",
    "sRGB",
    "Rec.709",
    "Rec.2020",
    "ACEScg",
    "ARRI LogC4",
    "ARRI LogC3",
    "Sony S-Log3",
    "PQ (HDR10 / ST.2084)",
    "HLG (Hybrid Log-Gamma)",
]

# ── Input color spaces (for RadianceRead decode) ───────────────────────────
# The list used to stop at ACEScct, which left no way to decode the three
# things that arrive in a MOV more often than any camera log: a Rec.709
# delivery, an HDR10 master, and an HLG broadcast file. Radiance already had
# all three inverses in color/transfer.py; only the menu was missing them.
INPUT_COLOR_SPACES = [
    "Auto / Linear (pass-through)",
    "Rec.709 (BT.1886)",
    "sRGB",
    "ARRI LogC4",
    "ARRI LogC3",
    "Sony S-Log3",
    "Panasonic V-Log",
    "Canon Log 3",
    "RED Log3G10",
    "DaVinci Intermediate",
    "PQ (ST.2084)",
    "HLG (BT.2100)",
    "ACEScg",
    "ACEScct",
]

# ── EXR compressions ───────────────────────────────────────────────────────
EXR_COMPRESSIONS = ["ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed", "DWAA", "DWAB"]

# ── Image file extensions ──────────────────────────────────────────────────
# These three used to be hand-typed sets, and between them they decided what the
# node would even attempt. Nine image extensions meant TGA, SGI, PPM, JP2 and
# PCX were classified "unknown" and refused -- despite the reader underneath
# opening every one of them correctly, measured. core.formats builds the tables
# from what the installed backends actually register, so the answer tracks the
# install instead of a list someone typed in 2024.
_IMG_EXT  = _formats.image_extensions()
_VID_EXT  = set(_VIDEO_EXTENSIONS)
_EXR_EXT  = set(_formats.EXR_EXTENSIONS)


# ── File browser helpers ──────────────────────────────────────────────────

#: Extensions shown in the browse dropdown
_BROWSEABLE_EXT = _IMG_EXT | _EXR_EXT | _VID_EXT | {".dpx", ".hdr"}

def _get_input_files() -> List[str]:
    """
    Return a sorted list of image / EXR / video files found in ComfyUI's
    input directory.  Called fresh on each INPUT_TYPES() evaluation so new
    uploads appear without restarting the server.

    Falls back to [""] when folder_paths is not available (e.g. test harness).
    """
    if not _HAS_FOLDER_PATHS:
        return [""]
    try:
        input_dir = _folder_paths.get_input_directory()
        files = [
            f for f in sorted(os.listdir(input_dir))
            if os.path.isfile(os.path.join(input_dir, f))
            and Path(f).suffix.lower() in _BROWSEABLE_EXT
        ]
        return [""] + files
    except Exception:
        return [""]


def _resolve_browse(browse: str) -> Optional[str]:
    """
    Convert a browse filename (relative to ComfyUI's input dir) to an
    absolute path.  Returns None if the file cannot be found.
    """
    if not browse or not _HAS_FOLDER_PATHS:
        return None
    try:
        full = os.path.join(_folder_paths.get_input_directory(), browse)
        return full if os.path.isfile(full) else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# § 1  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _np_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) float32 ndarray → (1, H, W, C) float32 tensor.

    This is a pure dtype/shape converter. It performs **no** value
    normalization: scene-linear / HDR / EXR data is preserved exactly,
    including values above 1.0. Callers that read integer formats are
    responsible for dividing by the type maximum (255 / 65535) *before*
    calling this function. Normalizing here based on pixel magnitude
    silently destroyed HDR plates whose highlights exceed 2.0 (the old
    `arr.max() > 2.0` heuristic), so it has been removed.
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # np.ascontiguousarray guards against torch.from_numpy receiving a
    # non-contiguous view (e.g. a channel slice), which would corrupt layout.
    # OpenEXR's np.frombuffer views are contiguous but read-only; Torch warns
    # that wrapping those can lead to undefined behaviour if later mutated.
    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    return torch.from_numpy(arr).unsqueeze(0)


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1, H, W, C) or (H, W, C) tensor → float32 ndarray."""
    arr = t.detach().cpu().float().numpy()
    return arr[0] if arr.ndim == 4 else arr


def _ffmpeg_bin() -> str:
    """ffmpeg path, PATH first then the imageio-ffmpeg bundle."""
    from .core.ffmpeg import require_ffmpeg
    return require_ffmpeg()


def _ffprobe_bin() -> str:
    """ffprobe path. imageio-ffmpeg does not ship ffprobe, so this can be
    absent even when ffmpeg is present; callers fall back to defaults."""
    from .core.ffmpeg import ffprobe_exe
    return ffprobe_exe() or "ffprobe"


def _ffmpeg_ok() -> bool:
    # shutil.which alone missed the ffmpeg that imageio-ffmpeg ships, which
    # is the only one most Windows installs have.
    from .core.ffmpeg import ffmpeg_available
    return ffmpeg_available()


def _apply_output_colorspace(arr: np.ndarray, cs: str) -> np.ndarray:
    """Apply output color space conversion to a float32 (H, W, 3) array."""
    if cs == "Linear (pass-through)":
        return arr
    try:
        if cs == "sRGB":
            return color_utils.linear_to_srgb(arr)
        if cs == "Rec.709":
            return color_utils.linear_to_rec709(arr)
        if cs in ("Rec.2020",):
            # Use PQ gamma as a proxy for Rec.2020 container
            return np.clip(arr ** (1/2.4), 0, 1).astype(np.float32)
        if cs == "ACEScg":
            return color_utils.linear_srgb_to_acescg(arr) if hasattr(color_utils, "linear_srgb_to_acescg") else arr
        if cs == "ARRI LogC4":
            return color_utils.linear_to_logc4(arr)
        if cs == "ARRI LogC3":
            return color_utils.linear_to_logc3(arr)
        if cs == "Sony S-Log3":
            return color_utils.linear_to_slog3(arr)
        if cs == "PQ (HDR10 / ST.2084)":
            return color_utils.linear_to_pq(arr)
        if cs == "HLG (Hybrid Log-Gamma)":
            return color_utils.linear_to_hlg(arr)
    except Exception as e:
        log.warning("Output color space conversion '%s' failed: %s", cs, e)
    return arr


def _rec709_to_linear(arr: np.ndarray) -> np.ndarray:
    """BT.1886 decode — the display transfer function of a Rec.709 delivery.

    A pure 2.4 power law with black at zero, which is what BT.1886 reduces to
    when Lb = 0. This is the right inverse for a graded Rec.709 MOV or MP4;
    it is *not* the BT.709 camera OETF, which nothing is actually encoded with.
    """
    return np.sign(arr) * np.power(np.abs(arr, dtype=np.float32), 2.4, dtype=np.float32)


#: Input colour space name -> the function that takes it back to scene-linear.
_INPUT_DECODERS = {
    "Rec.709 (BT.1886)":     _rec709_to_linear,
    "sRGB":                  lambda a: color_utils.srgb_to_linear(a),
    "ARRI LogC4":            lambda a: color_utils.logc4_to_linear(a),
    "ARRI LogC3":            lambda a: color_utils.logc3_to_linear(a),
    "Sony S-Log3":           lambda a: color_utils.slog3_to_linear(a),
    "Panasonic V-Log":       lambda a: color_utils.vlog_to_linear(a),
    "Canon Log 3":           lambda a: color_utils.canonlog3_to_linear(a),
    "RED Log3G10":           lambda a: color_utils.log3g10_to_linear(a),
    "DaVinci Intermediate":  lambda a: color_utils.davinci_intermediate_to_linear(a),
    "PQ (ST.2084)":          lambda a: color_utils.pq_to_linear(a),
    "HLG (BT.2100)":         lambda a: color_utils.hlg_to_linear(a),
    "ACEScct":               lambda a: color_utils.acescct_to_linear(a),
}


def _apply_input_colorspace(arr: np.ndarray, cs: str) -> np.ndarray:
    """Decode an input color space to scene-linear float32."""
    if cs in ("Auto / Linear (pass-through)", "ACEScg"):
        return arr
    fn = _INPUT_DECODERS.get(cs)
    if fn is None:
        return arr
    try:
        return fn(arr)
    except Exception as e:
        log.warning("Input color space decode '%s' failed: %s", cs, e)
    return arr


# ── Path type detection ────────────────────────────────────────────────────

def _path_kind(path: str) -> str:
    """Classify a path: "image" | "exr" | "video" | "sequence" | "unknown".

    One owner, in core.formats, so the browse filter, this detector and the
    readers cannot disagree about what a given extension is.
    """
    return _formats.classify(path)


def _explain_unknown(path: str) -> str:
    """Why a path was refused, and what would make it work."""
    return _formats.explain_unsupported(path)


# ── Image read ────────────────────────────────────────────────────────────

def _is_16bit_rgb_source(path: str, ext: str) -> bool:
    """Cheaply detect a genuine 16-bit-per-channel RGB(A) PNG/TIFF source,
    without a full pixel decode. Returns False (safe default -- existing
    Pillow path) if detection isn't possible or the source isn't 16-bit RGB(A).
    """
    try:
        if ext == ".png":
            # PNG IHDR chunk has a fixed layout: 8-byte signature + 4-byte
            # length + 4-byte "IHDR" + 4-byte width + 4-byte height + 1-byte
            # bit depth + 1-byte color type (2=RGB, 6=RGBA) -- no decode needed.
            with open(path, "rb") as f:
                header = f.read(26)
            if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
                return False
            bit_depth, color_type = header[24], header[25]
            return bit_depth == 16 and color_type in (2, 6)
        if ext in (".tif", ".tiff"):
            import tifffile  # type: ignore
            with tifffile.TiffFile(path) as tf:
                page = tf.pages[0]
                return page.dtype == np.uint16 and page.samplesperpixel in (3, 4)
    except Exception as _exc:
        log.debug(
            "[Radiance] _is_16bit_rgb_source(): ignoring %s from `if ext == '.png':`: %s",
            type(_exc).__name__, _exc,
        )
    return False


def _is_float32_tiff(path: str) -> bool:
    """Cheaply detect a 32-bit float TIFF via tifffile page metadata (no
    full pixel decode) -- Pillow's standard build cannot open this sample
    format at all."""
    try:
        import tifffile  # type: ignore
        with tifffile.TiffFile(path) as tf:
            return tf.pages[0].dtype == np.float32
    except Exception:
        return False


def _read_image(path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (IMAGE, MASK) tensors from a single image file."""
    ext = Path(path).suffix.lower()

    if ext == ".exr":
        return _read_exr_single(path)

    if ext == ".hdr":
        import cv2  # type: ignore
        arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"Cannot read HDR file '{path}' (unreadable, truncated, or unsupported).")
        # Radiance .hdr is scene-linear float; keep values as-is (may exceed 1.0).
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        return _np_to_tensor(arr), None

    if ext == ".dpx":
        # ALBABIT-FIX: Pillow has no DPX plugin at all; use OpenImageIO, the
        # VFX-industry-standard library for this format.
        if not _HAS_OIIO:
            raise ImportError("Reading DPX requires OpenImageIO (pip install OpenImageIO).")
        inp = _oiio.ImageInput.open(path)
        if inp is None:
            raise RuntimeError(f"Cannot read DPX '{path}': {_oiio.geterror()}")
        try:
            spec = inp.spec()
            pixels = inp.read_image(format=_oiio.FLOAT)
        finally:
            inp.close()
        # read_image() auto-normalises integer DPX samples (e.g. 10-bit) to [0, 1] float.
        arr = np.array(pixels, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels)
        arr = arr[..., :3] if spec.nchannels >= 3 else np.repeat(arr[..., :1], 3, axis=-1)
        return _np_to_tensor(arr), None

    # ALBABIT-FIX: a genuine 16-bit-per-channel RGB(A) PNG/TIFF is silently
    # collapsed to 8-bit by Pillow's .convert("RGB"/"RGBA") below -- Pillow has
    # no internal 16-bit RGB mode, and pil.mode reports "RGB" regardless of
    # source depth. cv2 preserves full precision instead. Grayscale 16-bit is
    # untouched -- Pillow already preserves that losslessly via "I"/"I;16".
    if ext in (".png", ".tif", ".tiff") and _is_16bit_rgb_source(path, ext):
        import cv2  # type: ignore
        arr16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr16 is not None and arr16.dtype == np.uint16 and arr16.ndim == 3 and arr16.shape[-1] in (3, 4):
            has_alpha16 = arr16.shape[-1] == 4
            arr16 = cv2.cvtColor(arr16, cv2.COLOR_BGRA2RGBA if has_alpha16 else cv2.COLOR_BGR2RGB)
            arr16 = arr16.astype(np.float32)
            rgb16  = arr16[..., :3] / 65535.0
            mask16 = arr16[..., 3:4] / 65535.0 if has_alpha16 else None
            img_t16  = _np_to_tensor(rgb16)
            mask_t16 = _np_to_tensor(mask16[..., 0]) if mask16 is not None else None
            return img_t16, mask_t16
        # else: fall through to Pillow below -- defensive, shouldn't normally
        # happen since _is_16bit_rgb_source() already confirmed 16-bit RGB(A).

    # ALBABIT-FIX: a 32-bit float TIFF (RadianceWrite's own "TIFF (32-bit
    # float)" output) can't be opened by Pillow at all -- caught by
    # RadianceRead.read()'s outer handler and surfaced as a silent black
    # image. tifffile (already used to write this format) reads it back correctly.
    if ext in (".tif", ".tiff") and _is_float32_tiff(path):
        import tifffile  # type: ignore
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        rgb32 = arr[..., :3]
        mask32 = arr[..., 3] if arr.shape[-1] == 4 else None
        return _np_to_tensor(rgb32), (_np_to_tensor(mask32) if mask32 is not None else None)

    if not _HAS_PIL:
        raise ImportError("Pillow is required to read image files.")
    pil = _PIL.open(path)
    has_alpha = pil.mode in ("RGBA", "LA", "PA")
    pil_rgb = pil.convert("RGBA") if has_alpha else pil.convert("RGB")
    arr = np.array(pil_rgb, dtype=np.float32)
    # ALBABIT-FIX: maxv used to be chosen from `pil` (pre-convert -- still the
    # source bit depth, e.g. 65535 for a 16-bit grayscale "I;16" source) but
    # applied to `arr`, which comes from `pil_rgb` (post-`.convert("RGB"/
    # "RGBA")`, always 8-bit -- Pillow's RGB/RGBA convert targets are always
    # 8-bit per channel, there's no 16-bit RGB convert mode). Dividing 8-bit
    # data by 65535 crushed 16-bit grayscale sources (mattes, depth passes)
    # to ~1/256th of their real brightness, silently. `arr` is always 8-bit
    # range here, so maxv is always 255.
    rgb  = arr[..., :3] / 255.0
    mask = (arr[..., 3:4] / 255.0) if has_alpha else None
    img_t  = _np_to_tensor(rgb)
    mask_t = _np_to_tensor(mask[..., 0]) if mask is not None else None
    return img_t, mask_t


def _read_exr_single(
    path: str,
    layer: Optional[str] = None,
    raw: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Read one layer of an EXR, honouring the display window.

    Delegated to :mod:`radiance.core.exr`. What used to be here pulled R, G, B
    and A out of part 0 and raised on anything else -- so a multi-layer AOV
    render, which is the normal output of Nuke and Arnold, was rejected with a
    message blaming the file. It also read the data window and reshaped to it,
    so an overscan render came back offset and at the wrong resolution with no
    warning.
    """
    from .core import exr as _exr

    rgb, alpha, _info, _name = _exr.read_layer(path, layer, raw=raw)
    mask = _np_to_tensor(alpha) if alpha is not None else None
    return _np_to_tensor(rgb), mask


def _read_exr_with_info(path: str, layer: Optional[str] = None, raw: bool = False):
    """As above, but also hands back the probe so the node can report it."""
    from .core import exr as _exr

    rgb, alpha, info, name = _exr.read_layer(path, layer, raw=raw)
    mask = _np_to_tensor(alpha) if alpha is not None else None
    return _np_to_tensor(rgb), mask, info, name


# ── Sequence read ─────────────────────────────────────────────────────────

def _resolve_sequence_paths(
    pattern: str,
    start: int,
    end: int,
    step: int = 1,
    missing_frames: str = "Skip",
) -> List[str]:
    """
    Expand a sequence pattern to a list of existing file paths.

    Supported patterns:
      /path/frame.%04d.exr
      /path/frame.####.png
      /path/frame.*.png     → sorted glob
      /path/               → directory: sorted image files
    """
    if os.path.isdir(pattern):
        exts = list(_IMG_EXT | _EXR_EXT)
        files = sorted(
            f for f in Path(pattern).iterdir()
            if f.suffix.lower() in exts
        )
        return [str(f) for f in files][start:end:step] if end > 0 else [str(f) for f in files]

    if "*" in pattern:
        import glob
        return sorted(glob.glob(pattern))[start:end:step]

    # Hash style #### → %04d
    hash_match = re.search(r"(#+)", pattern)
    if hash_match:
        hashes = hash_match.group(1)
        pattern = pattern.replace(hashes, f"%0{len(hashes)}d")

    # Probe the directory once to learn which frame numbers actually exist.
    # Walking the caller's upper bound blindly is not viable: "read all frames"
    # arrives here as end=99999, which expanded a 5-frame sequence into ~99k
    # slots, 98,994 of them phantoms, and blew up the downstream torch.cat.
    available: set = set()
    pad_match = re.search(r"%0(\d+)d", pattern)
    if pad_match:
        import glob as _glob
        token = pad_match.group(0)
        prefix, suffix = pattern.split(token, 1)
        for hit in _glob.glob(pattern.replace(token, "[0-9]" * int(pad_match.group(1)))):
            if hit.startswith(prefix) and hit.endswith(suffix):
                num = hit[len(prefix):len(hit) - len(suffix)] if suffix else hit[len(prefix):]
                if num.isdigit():
                    available.add(int(num))

    if available:
        hi = min(end, max(available)) if end >= start else max(available)
        frames = [f for f in range(start, hi + 1, step)]
        # Requested window misses the sequence entirely (e.g. the default
        # start_frame=1001 against a sequence numbered from 1) -- fall back to
        # everything on disk rather than reporting a range full of holes.
        if not any(f in available for f in frames):
            frames = sorted(available)[::step]
    else:
        frames = list(range(start, end + 1, step)) if end >= start else [start]

    paths  = []
    missing = []
    for f in frames:
        p = pattern % f
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing.append(p)
            if missing_frames == "Black":
                paths.append(p)   # keep slot for black-frame insertion
            else:
                # "Skip" means skip: keeping the slot here inserted an 8x8
                # black tile that then failed to concatenate with real frames.
                log.debug("Frame not found: %s", p)

    if missing_frames == "Error" and missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} frame(s), e.g. {missing[0]}"
        )
    return paths


def _read_sequence(
    pattern: str,
    start: int,
    end: int,
    step: int,
    input_cs: str,
    missing_frames: str = "Skip",
    layer: Optional[str] = None,
    raw: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, int, float, str]:
    """Read a frame sequence into a batched IMAGE tensor, keeping its alpha.

    The alpha used to be discarded here -- ``img_t, _ = _read_image(p)`` -- so a
    sequence of RGBA PNGs or EXRs came back with an empty mask, exactly like the
    ProRes 4444 case. A sequence with a matte is the other half of how plates
    arrive.
    """
    paths = _resolve_sequence_paths(pattern, start, end, step, missing_frames)
    if not paths:
        raise FileNotFoundError(f"No frames found for pattern: {pattern}")

    frames: List[Optional[torch.Tensor]] = []
    masks: List[Optional[torch.Tensor]] = []
    blank_slots: List[int] = []
    missing_paths: List[str] = []
    for p in paths:
        if os.path.isfile(p):
            if os.path.splitext(p)[1].lower() in _EXR_EXT:
                img_t, mask_t = _read_exr_single(p, layer, raw)
            else:
                img_t, mask_t = _read_image(p)
            arr = _tensor_to_np(img_t)
            if not raw:
                arr = _apply_input_colorspace(arr, input_cs)
            frames.append(torch.from_numpy(arr).unsqueeze(0))
            masks.append(mask_t)
        else:
            # Placeholder for a missing frame. Size it from a real frame below
            # -- a fixed 8x8 tile cannot concatenate with the rest of the batch.
            blank_slots.append(len(frames))
            missing_paths.append(p)
            frames.append(None)
            masks.append(None)

    real = next((f for f in frames if f is not None), None)
    if real is None:
        raise FileNotFoundError(f"No readable frames found for pattern: {pattern}")
    for i in blank_slots:
        frames[i] = torch.zeros_like(real)
    if missing_paths:
        log.warning(
            "[Radiance/Read] %d frame(s) of %s are not on disk and were filled "
            "with black: %s%s",
            len(missing_paths), pattern,
            ", ".join(os.path.basename(m) for m in missing_paths[:6]),
            " ..." if len(missing_paths) > 6 else "",
        )

    batch = torch.cat(frames, dim=0)   # (N, H, W, C)
    _, h, w, _ = batch.shape

    # An alpha only survives if every frame that has one agrees on the shape.
    # A sequence where half the frames carry a matte is a mistake worth saying
    # out loud rather than papering over.
    with_alpha = [m for m in masks if m is not None]
    alpha_out = None
    if with_alpha:
        if len(with_alpha) != len(masks):
            log.warning(
                "[Radiance/Read] %d of %d frames in %s carry an alpha channel; "
                "the missing ones are opaque in the mask output.",
                len(with_alpha), len(masks), pattern,
            )
        filled = [
            m if m is not None else torch.ones(1, h, w, dtype=with_alpha[0].dtype)
            for m in masks
        ]
        try:
            alpha_out = torch.cat(filled, dim=0)
        except RuntimeError as exc:
            log.warning("[Radiance/Read] alpha channels in %s do not share a "
                        "shape (%s); mask output left empty.", pattern, exc)
            alpha_out = None

    meta = json.dumps({
        "kind": "sequence",
        "pattern": pattern,
        "frame_count": len(frames),
        "missing": [os.path.basename(m) for m in missing_paths],
        "width": w, "height": h,
        "alpha": alpha_out is not None,
    })
    return batch, alpha_out, w, h, len(frames), 24.0, meta


# ── Video read ────────────────────────────────────────────────────────────

def _read_video(
    path: str,
    max_frames: int,
    input_cs: str,
    start_frame: int = 0,
    end_frame: int = 0,
    frame_step: int = 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float, int, int, int, str]:
    """Decode a video file to a batched IMAGE tensor, plus its alpha if it has one.

    Everything here is delegated to :mod:`radiance.core.video`, which pipes raw
    frames from ffmpeg at the source bit depth. The old implementation in this
    slot wrote every frame to a temporary PNG and read them back: measured on a
    4-second 1080p ProRes 422 HQ clip, 25.0 s and ~600 MB of scratch files
    against 12.9 s and none. It also had no way to express a frame range, and it
    silently dropped the alpha channel of every ProRes 4444.

    Returns
    -------
    (images, alpha, fps, width, height, frame_count, metadata_json)
        ``alpha`` is None unless the file actually carries an alpha channel.
    """
    from .core import video as _video

    info = _video.probe(path)

    # A frame range on the *source* numbering. `end_frame` is inclusive, and 0
    # means "to the end", which is how the sequence reader already behaves.
    start = max(int(start_frame), 0)
    if end_frame and end_frame >= start:
        count = (int(end_frame) - start) // max(int(frame_step), 1) + 1
    else:
        count = 0
    if max_frames > 0:
        count = min(count, max_frames) if count else max_frames

    log.info("[Radiance/Read] %s: %s", os.path.basename(path), info.summary())
    if info.frames_estimated:
        log.debug(
            "%s carries no frame count; %d is estimated from duration x fps.",
            os.path.basename(path), info.frames,
        )

    arr, info = _video.decode(
        path, start=start, count=count, step=max(int(frame_step), 1), info=info,
    )

    # Split alpha off before the transfer decode. A matte is not light: running
    # it through a log or gamma curve is the bug that made a 3.1.x tone map
    # return an opaque pixel at 0.730.
    alpha_np = None
    if arr.shape[-1] == 4:
        alpha_np = np.ascontiguousarray(arr[..., 3])
        arr = np.ascontiguousarray(arr[..., :3])

    resolved_cs = _resolve_video_colorspace(input_cs, info, path)
    arr = _apply_input_colorspace(arr, resolved_cs)

    batch = torch.from_numpy(np.ascontiguousarray(arr))
    alpha_t = torch.from_numpy(alpha_np) if alpha_np is not None else None

    n, h, w, _ = batch.shape
    meta = dict(info.as_dict())
    meta.update({
        "kind": "video",
        "decoded_frames": n,
        "start_frame": start,
        "frame_step": max(int(frame_step), 1),
        "color_space": resolved_cs,
        "color_space_requested": input_cs,
        "alpha": alpha_t is not None,
    })
    return batch, alpha_t, info.fps_float, w, h, n, json.dumps(meta)


#: The sequence-style default for `start_frame`. VFX sequences are numbered
#: from 1001; a clip is numbered from 0, so the same widget means two things.
_SEQUENCE_START_DEFAULT = 1001


def _video_frame_range(path: str, start_frame: int, end_frame: int) -> Tuple[int, int]:
    """Translate the sequence-style frame widgets onto a clip's own numbering.

    `start_frame` defaults to 1001 because that is where a VFX image sequence
    starts. A video's frames are numbered from zero, so taking the widget
    literally would skip the first 1001 frames of every clip -- which, for
    anything under about 40 seconds, means decoding nothing at all and
    reporting an empty file. That is exactly the trap the directory-pattern
    reader fell into.

    So: a start beyond the end of the clip is treated as "the user never
    touched this widget", and said out loud. A start inside the clip is
    honoured, because then it is a deliberate trim.
    """
    from .core import video as _video

    start = max(int(start_frame), 0)
    end = max(int(end_frame), 0)
    if start <= 0:
        return 0, end

    try:
        info = _video.probe(path)
    except Exception:  # the decoder will produce the real error in a moment
        return 0, end

    if info.frames > 0 and start >= info.frames:
        if start == _SEQUENCE_START_DEFAULT:
            log.debug(
                "start_frame is at its sequence default (%d) and %s has only %d "
                "frame(s), so the whole clip is being read.",
                _SEQUENCE_START_DEFAULT, os.path.basename(path), info.frames,
            )
        else:
            log.warning(
                "[Radiance/Read] start_frame=%d is past the end of %s, which has "
                "%d frame(s). Reading the whole clip instead. Video frames are "
                "numbered from 0, unlike an image sequence.",
                start, os.path.basename(path), info.frames,
            )
        return 0, end
    return start, end


def _resolve_video_colorspace(requested: str, info, path: str) -> str:
    """Pick the decode curve for a video, honouring the file's own tags.

    Nuke, Resolve and RV all read the container's transfer characteristics and
    default to them. Radiance did not read them at all, so a Rec.709 delivery
    was passed through as if it were scene-linear -- the values are gamma
    encoded, so every subsequent exposure, blur and blend was operating on the
    wrong numbers.

    An explicit choice always wins. "Auto" now means what its label promises:
    use the tag if there is one, and say so. An untagged file still passes
    through unchanged, but now says that too, because a silent pass-through is
    exactly how the mistake goes unnoticed.
    """
    from .core import video as _video

    if requested != "Auto / Linear (pass-through)":
        tagged = _video.suggest_transfer(info)
        if tagged and tagged != requested:
            log.warning(
                "[Radiance/Read] %s is tagged %s (%s), but color_space is set to "
                "%r. Using your setting. Clear it to Auto to follow the file.",
                os.path.basename(path), info.color_transfer, tagged, requested,
            )
        return requested

    suggestion = _video.suggest_transfer(info)
    if suggestion and suggestion != "Auto / Linear (pass-through)":
        log.info(
            "[Radiance/Read] %s is tagged color_trc=%s; decoding it as %s. "
            "Set color_space explicitly to override.",
            os.path.basename(path), info.color_transfer, suggestion,
        )
        return suggestion

    if info.color_transfer:
        log.warning(
            "[Radiance/Read] %s is tagged color_trc=%s, which Radiance has no "
            "inverse for. Passing the values through unchanged -- they are not "
            "scene-linear. Set color_space by hand.",
            os.path.basename(path), info.color_transfer,
        )
    else:
        log.warning(
            "[Radiance/Read] %s carries no colour tags, so its values are being "
            "passed through as if they were already scene-linear. If it is a "
            "Rec.709 delivery or a camera log file, set color_space -- exposure "
            "and blur are only correct in a linear space.",
            os.path.basename(path),
        )
    return requested


# ── Write input coercion ──────────────────────────────────────────────────

def _find_video_path(data: Any, _depth: int = 0) -> Optional[str]:
    """
    Recursively search dict / list / str structures for a video file path.
    Handles VHS, AnimateDiff, and other video-helper dict formats.
    """
    if _depth > 6:
        return None
    if isinstance(data, str):
        if os.path.isfile(data) and Path(data).suffix.lower() in _VID_EXT:
            return data
        return None
    if isinstance(data, dict):
        # Prefer explicit path keys used by common video helper nodes
        for key in ("video_path", "video", "path", "filename", "file", "filepath"):
            if key in data:
                r = _find_video_path(data[key], _depth + 1)
                if r:
                    return r
        for val in data.values():
            r = _find_video_path(val, _depth + 1)
            if r:
                return r
    if isinstance(data, (list, tuple)):
        for item in data:
            r = _find_video_path(item, _depth + 1)
            if r:
                return r
    return None


def _load_video_to_numpy(
    path: str, max_frames: int = 0, *, alpha: bool = False
) -> np.ndarray:
    """Decode a video file path to (N, H, W, C) float32 at the source bit depth.

    This used to try OpenCV first, which was wrong twice over. OpenCV's
    ``VideoCapture`` hands back 8-bit BGR whatever the source is -- a 12-bit
    ProRes 4444 came out on an exact 1/255 grid, measured -- and when it failed
    part-way through a clip the loop simply stopped, returning a short clip with
    no error. A shot that quietly ends early is the kind of bug that survives
    all the way to a review session.

    Alpha is off by default here because the callers of this helper feed
    DCC handoff paths that expect three channels; pass ``alpha=True`` to keep
    the fourth when the file has one.
    """
    from .core import video as _video

    arr, _info = _video.decode(path, count=max(int(max_frames), 0), alpha=alpha)
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# § 2  Write helpers
# ═══════════════════════════════════════════════════════════════════════════

def _unique_path(path: Path) -> Path:
    """Append _001, _002 ... if path already exists."""
    if not path.exists():
        return path
    stem, ext = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.parent / f"{stem}_{i:03d}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def _workflow_metadata(prompt=None, extra_pnginfo=None) -> dict:
    """
    Build the ComfyUI provenance metadata dict ({"prompt": json, "workflow":
    json, ...}) exactly as core SaveImage embeds into PNG tEXt chunks, so
    files written by Radiance carry the workflow like a saved PNG does.
    Values are JSON strings; storage encoding is handled per format.
    """
    meta = {}
    try:
        if prompt is not None:
            meta["prompt"] = json.dumps(prompt)
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                meta[str(key)] = json.dumps(value)
    except (TypeError, ValueError) as exc:
        log.warning(f"[RadianceWrite] workflow metadata not serialisable: {exc}")
    return meta


def _save_pil_image(arr_f32: np.ndarray, path: Path, fmt: str, quality: int = 18,
                    metadata: dict | None = None) -> None:
    """Save a float32 (H, W, 3) array to an image file via Pillow.

    `quality` is the same widget value used for video CRF (0-51, lower = better).
    JPEG/WEBP use the opposite convention (0-100, higher = better), so it is
    remapped rather than passed through raw -- matching the widget's own
    tooltip ("Also JPEG quality 0-100 (remapped)"), which previously had no
    code behind it at all (see ALBABIT-FIX below).

    PNG output embeds `metadata` (workflow/prompt JSON strings) as tEXt
    chunks — the same convention as ComfyUI core SaveImage, so the file
    can be dragged back into ComfyUI to restore the workflow."""
    if not _HAS_PIL:
        raise ImportError("Pillow required for image writing.")

    # ALBABIT-FIX: tifffile.imwrite() always writes real TIFF bytes -- for
    # "PNG (16-bit)" this silently produced a file with a TIFF magic header
    # under a .png name, rejected by anything that trusts the extension.
    # cv2 writes a genuine 16-bit PNG instead (confirmed via magic-byte +
    # round-trip check) -- already used elsewhere in this module for HDR/DPX.
    if fmt == "PNG (16-bit)":
        import cv2  # type: ignore
        arr_u16 = (np.clip(arr_f32, 0, 1) * 65535).astype(np.uint16)
        if arr_u16.shape[-1] == 4:
            cv2.imwrite(str(path), cv2.cvtColor(arr_u16, cv2.COLOR_RGBA2BGRA))
        else:
            cv2.imwrite(str(path), cv2.cvtColor(arr_u16, cv2.COLOR_RGB2BGR))
        return

    # ALBABIT-FIX: was `"16-bit" in fmt or "TIFF" in fmt`, which also matched
    # "TIFF (32-bit float)" (the substring "TIFF" is in both TIFF formats) --
    # every 32-bit TIFF request was silently written as 16-bit instead, and
    # its own dedicated `"32-bit" in fmt` branch below was never reached.
    # "SEQ │ TIFF" (bare, no depth suffix) is still meant to land here.
    if "16-bit" in fmt or ("TIFF" in fmt and "32-bit" not in fmt):
        arr_u16 = (np.clip(arr_f32, 0, 1) * 65535).astype(np.uint16)
        # ALBABIT-FIX: this used to also do
        # `_PIL.fromarray(arr_u16, mode="I;16" if arr_f32.ndim == 2 else "RGB")`
        # here, whose result was never used (the real write is via tifffile
        # below) -- with Pillow 12.2.0, that dead call raises TypeError for
        # any RGB (ndim==3) array, crashing every 16-bit PNG/TIFF write before
        # tifffile was ever reached. Pillow can't do 16-bit RGB directly
        # anyway; tifffile handles both 2D and 3D arrays natively.
        try:
            import tifffile  # type: ignore
            tifffile.imwrite(str(path), arr_u16)
            return
        except ImportError:
            # Fallback: save 8-bit. Not silent -- a 16-bit request quietly
            # becoming 8-bit is exactly the kind of downgrade this module's
            # EXR writer refuses to do (see _save_exr's docstring).
            log.warning("16-bit write requested for '%s' but tifffile is not installed; "
                        "falling back to 8-bit.", path)
            arr_f32 = np.clip(arr_f32, 0, 1)
    if "32-bit" in fmt:
        try:
            import tifffile  # type: ignore
            tifffile.imwrite(str(path), arr_f32.astype(np.float32))
            return
        except ImportError as _exc:
            log.debug(
                "[Radiance] _save_pil_image(): ignoring %s from `import tifffile`: %s",
                type(_exc).__name__, _exc,
            )

    arr_u8 = (np.clip(arr_f32, 0, 1) * 255).astype(np.uint8)
    pil = _PIL.fromarray(arr_u8)
    pnginfo = None
    if metadata and str(path).lower().endswith(".png"):
        try:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            for _key, _val in metadata.items():
                pnginfo.add_text(_key, _val)
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pnginfo = None
    if "JPEG" in fmt or "WEBP" in fmt:
        # ALBABIT-FIX: quality was hardcoded to 90 here, ignoring the widget
        # entirely -- the tooltip promised JPEG control but nothing in this
        # function ever accepted a quality argument. Remap CRF-scale (0-51,
        # lower = better) to Pillow's JPEG/WEBP scale (0-100, higher = better).
        pil_quality = max(0, min(100, round((1.0 - quality / 51.0) * 100)))
        if "JPEG" in fmt:
            pil.save(str(path), "JPEG", quality=pil_quality)
        else:
            pil.save(str(path), "WEBP", quality=pil_quality, lossless=False)
    elif pnginfo is not None:
        pil.save(str(path), pnginfo=pnginfo)
    else:
        pil.save(str(path))


def _exr_channels(arr_f32: np.ndarray) -> "list[tuple[str, np.ndarray]]":
    """Map an (H,W), (H,W,1), (H,W,3) or (H,W,4) float array to named EXR channels.

    - 1 channel  → R=G=B (grayscale/matte readable in every compositor)
    - 3 channels → R,G,B
    - 4 channels → R,G,B,A (alpha preserved)

    Channels are returned in EXR write order. Alpha is never silently dropped.
    """
    if arr_f32.ndim == 2:
        arr_f32 = arr_f32[..., None]
    if arr_f32.ndim != 3:
        raise ValueError(f"EXR data must be 2-D or 3-D, got shape {arr_f32.shape}")

    c = arr_f32.shape[2]
    if c == 1:
        plane = arr_f32[..., 0]
        return [("R", plane), ("G", plane), ("B", plane)]
    if c == 3:
        return [("R", arr_f32[..., 0]), ("G", arr_f32[..., 1]), ("B", arr_f32[..., 2])]
    if c == 4:
        return [("R", arr_f32[..., 0]), ("G", arr_f32[..., 1]),
                ("B", arr_f32[..., 2]), ("A", arr_f32[..., 3])]
    raise ValueError(f"Unsupported EXR channel count {c}; expected 1, 3, or 4.")


def _save_exr(arr_f32: np.ndarray, path: Path, half: bool,
              metadata: dict | None = None) -> None:
    """Write a float array to OpenEXR, preserving channel count and alpha.

    Accepts (H,W), (H,W,1), (H,W,3) or (H,W,4) float data. Scene-linear
    values above 1.0 are preserved. On unrecoverable failure this raises a
    clear error rather than writing a 0-byte placeholder — silent empty/
    downgraded deliveries are never acceptable in a VFX pipeline.
    """
    arr_f32 = np.asarray(arr_f32, dtype=np.float32)
    channels = _exr_channels(arr_f32)
    np_dtype = np.float16 if half else np.float32

    errors: list[str] = []

    # ── Primary: OpenEXR ──────────────────────────────────────────────────
    try:
        import OpenEXR  # type: ignore
        # ALBABIT-FIX: was OpenEXR.Header()/OutputFile()/writePixels() (the
        # legacy dict-based API). That API only recognises a fixed, small set
        # of attribute names internally -- rad_version/software (and even
        # genuinely standard attributes like "comments") were silently
        # rejected with a C-level "unknown attribute" stderr print, never a
        # Python exception, so they never actually reached disk. Confirmed
        # live: 0 of 12 metadata keys survived a round trip. The modern
        # OpenEXR.File(header_dict, channels_dict) API (same installed
        # version) persists arbitrary string/int/float attributes correctly,
        # and infers HALF vs FLOAT per channel from the numpy dtype directly
        # -- no Imath.Channel/PixelType wiring needed.
        header: Dict[str, Any] = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "type": OpenEXR.scanlineimage,
            "rad_version": _RADIANCE_VERSION,
            "software": f"Radiance v{_RADIANCE_VERSION}",
        }
        if metadata:
            # ComfyUI provenance (workflow/prompt) — stored under these exact
            # unprefixed names so the file can be dragged back into ComfyUI to
            # restore the workflow, same convention as write_exr_openexr() in
            # hdr/io.py. The modern API accepts plain str values directly (the
            # byte-encoding the legacy API needed is no longer necessary).
            for _key, _val in metadata.items():
                try:
                    header[_key] = _val
                except Exception:  # noqa: BLE001 — provenance is best-effort
                    pass
        channels_dict = {
            name: np.ascontiguousarray(plane, dtype=np_dtype)
            for name, plane in channels
        }
        OpenEXR.File(header, channels_dict).write(str(path))
        return
    except ImportError as exc:
        errors.append(f"OpenEXR module unavailable ({exc})")
    except Exception as exc:  # malformed data, write error, etc.
        errors.append(f"OpenEXR write failed ({exc})")

    # ── Fallback: OpenCV with EXR support (loses alpha for >3 channels) ────
    try:
        import cv2  # type: ignore
        # cv2 expects BGR(A); reverse only the colour channels.
        if arr_f32.ndim == 2 or arr_f32.shape[2] == 1:
            cv_arr = arr_f32 if arr_f32.ndim == 3 else arr_f32[..., None]
            cv_arr = np.repeat(cv_arr.reshape(*arr_f32.shape[:2], 1), 3, axis=2)[..., ::-1]
        elif arr_f32.shape[2] == 4:
            rgb = arr_f32[..., :3][..., ::-1]
            cv_arr = np.concatenate([rgb, arr_f32[..., 3:4]], axis=2)  # BGRA
        else:
            cv_arr = arr_f32[..., ::-1]
        ok = cv2.imwrite(
            str(path),
            np.ascontiguousarray(cv_arr.astype(np.float32)),
            [cv2.IMWRITE_EXR_TYPE,
             cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT],
        )
        if ok:
            return
        errors.append("cv2.imwrite returned False (OpenCV built without EXR support?)")
    except ImportError as exc:
        errors.append(f"OpenCV unavailable ({exc})")
    except Exception as exc:
        errors.append(f"OpenCV EXR write failed ({exc})")

    # ── No silent downgrade, no empty file: fail loudly. ──────────────────
    raise RuntimeError(
        f"Failed to write EXR '{path}'. Install OpenEXR (pip install OpenEXR Imath) "
        f"or OpenCV built with EXR support. Details: " + "; ".join(errors)
    )


def _save_dpx(arr_f32: np.ndarray, path: Path) -> None:
    """Write a float32 (H, W, 3) array to 10-bit DPX via OpenImageIO.

    ALBABIT-FIX: DPX has no Pillow plugin at all -- "DPX" was offered in the
    format dropdown since v3.1 but never actually worked (write raised
    "unknown file extension: .dpx"; read fell through to the same Pillow
    path and failed identically). 10-bit is the traditional DPX bit depth
    for film/VFX intermediates (SMPTE 268M). write_image()/read_image()
    auto-convert between float32 [0,1] and the packed integer sample format.
    """
    if not _HAS_OIIO:
        raise RuntimeError(
            f"Failed to write DPX '{path}'. Install OpenImageIO (pip install OpenImageIO)."
        )
    arr = np.ascontiguousarray(np.clip(arr_f32, 0, 1), dtype=np.float32)
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1

    spec = _oiio.ImageSpec(w, h, c, _oiio.UINT16)
    spec.attribute("oiio:BitsPerSample", 10)
    out = _oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"No DPX writer available for '{path}': {_oiio.geterror()}")
    try:
        if not out.open(str(path), spec):
            raise RuntimeError(f"Failed to open DPX '{path}' for writing: {_oiio.geterror()}")
        if not out.write_image(arr):
            raise RuntimeError(f"Failed to write DPX pixels to '{path}': {_oiio.geterror()}")
    finally:
        out.close()


def _save_hdr(arr_f32: np.ndarray, path: Path) -> None:
    """Write a float32 (H, W, 3) array to Radiance HDR (.hdr / RGBE) via cv2.

    ALBABIT-FIX: restores a format lost in the v3 rewrite. The pre-v3
    write_hdr_rgbe() (hdr/io.py) is confirmed broken on round-trip (loses
    ~1 stop of range); cv2's native HDR codec (already used to read .hdr)
    is accurate instead.
    """
    arr = np.ascontiguousarray(arr_f32[..., :3], dtype=np.float32)
    import cv2  # type: ignore
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _coerce_mask_to_alpha(mask: Any, n: int, h: int, w: int) -> Optional[np.ndarray]:
    """Normalize a ComfyUI MASK to an (N, H, W) float32 alpha array.

    Resizes to the frame size and matches the frame count (broadcasting a single
    mask, truncating extras, or padding with the last mask). Returns None when no
    usable mask is given.
    """
    if mask is None:
        return None
    m = mask
    if isinstance(m, (list, tuple)):
        m = m[0] if m else None
        if m is None:
            return None
    try:
        if hasattr(m, "detach"):
            a = m.detach().cpu().float()
        else:
            a = torch.as_tensor(np.asarray(m, dtype=np.float32))
        if a.ndim == 4 and a.shape[-1] == 1:   # (N,H,W,1) -> (N,H,W)
            a = a[..., 0]
        if a.ndim == 2:                         # (H,W) -> (1,H,W)
            a = a.unsqueeze(0)
        if a.ndim != 3:
            return None
        if tuple(a.shape[-2:]) != (h, w):
            a = torch.nn.functional.interpolate(
                a.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
        arr = a.numpy().astype(np.float32)
    except Exception as exc:
        log.warning("RadianceWrite: could not interpret mask as alpha (%s); writing without alpha.", exc)
        return None

    if arr.shape[0] == n:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, n, axis=0)
    if arr.shape[0] > n:
        return arr[:n]
    pad = np.repeat(arr[-1:], n - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _save_video_ffmpeg(
    frames: np.ndarray,       # (N, H, W, 3) float32, range [0, 1]
    output_path: str,
    fmt: str,
    fps: float,
    crf: int,
    audio_source: str,
) -> None:
    """Encode a frame batch to video via ffmpeg, piping raw frames directly
    (no intermediate 8-bit PNG per frame).

    ALBABIT-FIX: frames used to go through an 8-bit PNG intermediate, so no
    "10-bit" format ever carried more than 8 bits of real precision. Now
    piped raw via stdin: ProRes 422 HQ/4444 get a genuine 16-bit source (the
    formats that benefit); H.264/H.265/DNxHR HQ stay 8-bit, since their
    "10-bit" is a codec/container property, not source precision.
    """
    if not _ffmpeg_ok():
        raise RuntimeError("ffmpeg not found.")

    # (codec, src_pix_fmt, dst_pix_fmt, ext, extra ffmpeg args)
    fmt_map = {
        "MP4 (H.264)":        ("libx264",  "rgb24",   "yuv420p",     ".mp4", ["-crf", str(crf), "-preset", "medium"]),
        "MP4 (H.265 10-bit)": ("libx265",  "rgb24",   "yuv420p10le", ".mp4", ["-crf", str(crf), "-preset", "medium", "-tag:v", "hvc1"]),
        "MOV (ProRes 422 HQ)":("prores_ks","rgb48le", "yuv422p10le", ".mov", ["-profile:v", "3", "-qscale:v", "9"]),
        "MOV (ProRes 4444)":  ("prores_ks","rgb48le", "yuva444p10le",".mov", ["-profile:v", "4", "-qscale:v", "9"]),
        "MOV (DNxHR HQ)":     ("dnxhd",    "rgb24",   "yuv422p",     ".mxf", ["-profile:v", "dnxhr_hq"]),
    }
    codec, src_pix_fmt, dst_pix_fmt, ext, extra = fmt_map.get(
        fmt, ("libx264", "rgb24", "yuv420p", ".mp4", ["-crf", str(crf)])
    )

    out_path = str(output_path)
    if not out_path.lower().endswith(ext):
        out_path += ext

    n, h, w = frames.shape[:3]
    if src_pix_fmt == "rgb48le":
        raw = (np.clip(frames, 0, 1) * 65535).astype("<u2").tobytes()
    else:
        raw = (np.clip(frames, 0, 1) * 255).astype(np.uint8).tobytes()

    cmd = [
        _ffmpeg_bin(), "-v", "error", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", src_pix_fmt,
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio_source and os.path.isfile(audio_source):
        cmd += ["-i", audio_source, "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", codec, "-pix_fmt", dst_pix_fmt] + extra + [out_path]
    subprocess.run(cmd, input=raw, check=True, capture_output=True, timeout=600)

    return out_path


def _write_audio_temp_wav(audio: Any) -> Optional[str]:
    """Write a ComfyUI AUDIO dict ({"waveform": tensor, "sample_rate": int}) to a
    temporary 32-bit float PCM WAV file, so it can be fed into the same
    `audio_source` mux path `_save_video_ffmpeg()` already uses for on-disk
    audio files. Returns the temp file path, or None if `audio` isn't a usable
    AUDIO dict. The caller owns the returned file and must delete it.

    # ALBABIT-FIX: the "audio" (AUDIO-type) input was accepted by write()'s
    # signature but never referenced anywhere else in this file -- connecting
    # an AUDIO output here had zero effect. Only the "audio_source" (a STRING
    # path to an existing file) actually worked. No torchaudio/soundfile
    # dependency needed: WAV is simple enough to write by hand (matches the
    # approach radiance.disabled used for the same problem).
    """
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    if waveform is None:
        return None

    import struct

    sr = int(audio.get("sample_rate", 44100))
    wav_np = waveform.detach().cpu().numpy()
    if wav_np.ndim == 3:        # (B, C, T) -> first batch item
        wav_np = wav_np[0]
    if wav_np.ndim == 1:        # (T,) -> (1, T) mono
        wav_np = wav_np[np.newaxis, :]
    n_channels = wav_np.shape[0]
    raw = np.ascontiguousarray(wav_np.T).astype(np.float32).tobytes()  # interleaved (T, C)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    block_align = n_channels * 4
    with open(path, "wb") as f:
        f.write(
            b"RIFF" + struct.pack("<I", 36 + len(raw)) +
            b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, n_channels, sr,
                sr * block_align, block_align, 32) +
            b"data" + struct.pack("<I", len(raw)) + raw
        )
    return path


# ═══════════════════════════════════════════════════════════════════════════
# § 3  RadianceRead
# ═══════════════════════════════════════════════════════════════════════════

class RadianceRead:
    """
    Universal reader — images, EXR, video, numbered sequences.

    File selection
    ──────────────
    Two ways to choose a file:
      1. browse  — Dropdown + upload button.  Click the upload icon to open a
                   native OS file picker, or drag-and-drop.  Uploaded files land
                   in ComfyUI's input/ folder and are selected automatically.
                   Supports images (PNG/JPG/TIFF/EXR/DPX/HDR) and video
                   (MP4/MOV/MXF/AVI/WebM/MKV).

      2. path    — Type or paste any absolute path, UNC network path, or a
                   sequence pattern (/renders/frame.%04d.exr, /frames/####.png).
                   This field is used only when browse is left blank.

    Path auto-detection
    ───────────────────
    The node inspects the extension and path pattern:
      .exr                → EXR (single frame)
      .png/.jpg/.tiff/…   → image (single frame)
      .mp4/.mov/…         → video (all frames decoded to batch)
      /path/%04d.exr      → sequence (printf pattern)
      /path/frame.####    → sequence (hash pattern)
      /path/              → sequence (directory, sorted)

    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = (
        "Read an image, EXR, video or numbered sequence into the pipeline. "
        "Video decodes at the source bit depth through ffmpeg, keeps ProRes "
        "4444 alpha on the mask output, and honours the file's colour tags."
    )
    FUNCTION     = "read"
    # `info` is appended last on purpose: ComfyUI links outputs by index, so
    # every workflow saved against the two-output version keeps working.
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")
    OUTPUT_TOOLTIPS = (
        "Frames as a batch. Scene-linear once color_space has decoded them.",
        "Alpha. For a ProRes 4444, an RGBA EXR or an RGBA sequence this is the "
        "file's own matte; otherwise zeros.",
        "JSON describing what was actually read: resolution, frame count and "
        "range, bit depth, codec, EXR layers and windows, colour tags, "
        "timecode. Nuke's metadata tab, as a wire you can plug in.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "browse": (_get_input_files(), {
                "image_upload": True,
                "tooltip": (
                    "Browse or upload a file from disk.\n"
                    "• Click the upload icon (📎) to open a native file picker.\n"
                    "• Supports images (PNG, JPG, TIFF, EXR, DPX, HDR, WebP) "
                    "and video (MP4, MOV, MXF, AVI, WebM, MKV).\n"
                    "• Uploaded files are copied to ComfyUI's input"
                    "• Leave blank and fill in 'path' below for absolute / "
                    "network / sequence paths."
                ),
            }),
            }, "optional": {
                "media_type": (["Auto", "Image", "Video", "Sequence"], {
                    "default": "Auto",
                    "tooltip": "Override auto-detection. Auto infers from path extension and pattern.",
                }),
                "path": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "/abs/path/file.exr  ·  //nas/share/shot.mov  ·  /seq/frame.%04d.exr",
                "tooltip": (
                    "Optional — used only when 'browse' is left blank.\n"
                    "Accepts any absolute path, UNC network path, or sequence pattern:\n"
                    "  Sequence patterns:  /frames/f.%04d.exr  ·  /frames/f.####.png  ·  /dir/\n"
                    "  Network paths:      /mnt/nas/renders/shot   or   \\\\server\\share\\shot\n"
                    "Format is auto-detected from extension."
                ),
            }),
            "color_space": (INPUT_COLOR_SPACES, {
                "default": "Auto / Linear (pass-through)",
                "tooltip": "Decode the input from this color space to scene-linear before processing.",
            }),
            "start_frame": ("INT", {
                "default": 1001, "min": 0, "max": 99999,
                "tooltip": (
                    "First frame to read.\n"
                    "• Sequence: the frame number in the filename (1001 is the "
                    "usual VFX start).\n"
                    "• Video: a 0-based offset into the clip. The 1001 default "
                    "is a sequence convention, so it is ignored for any clip "
                    "shorter than that rather than reading nothing."
                ),
            }),
            "end_frame": ("INT", {
                "default": 0, "min": 0, "max": 99999,
                "tooltip": "Last frame to read, inclusive. 0 = to the end. Applies to sequences and video.",
            }),
            "frame_step": ("INT", {
                "default": 1, "min": 1, "max": 100,
                "tooltip": "Step size — e.g. 2 reads every other frame. Applies to sequences and video.",
            }),
            "max_video_frames": ("INT", {
                "default": 0, "min": 0, "max": 99999,
                "tooltip": (
                    "Hard cap on decoded video frames (0 = all). Frames are "
                    "float32 RGB in RAM: 240 frames of 4K RGBA is about 31 GB, "
                    "so cap this while building a graph."
                ),
            }),
            "proxy_scale": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Downscale factor for proxy preview (0 = full resolution). 0.5 = half res for faster iteration.",
            }),
            "missing_frames": (["Error", "Black", "Skip"], {
                "default": "Skip",
                "tooltip": (
                    "A frame inside a sequence that is not on disk. Black "
                    "inserts a zero frame, Skip omits it, Error raises.\n"
                    "For a read that fails outright, see on_error."
                ),
            }),
            # ── Nuke-parity controls ────────────────────────────────────
            # A STRING rather than a combo on purpose. The layer list depends
            # on the file, and ComfyUI validates a combo's value against the
            # list it was built with -- so a workflow saved with layer="diffuse"
            # would fail to load on a machine where INPUT_TYPES had not seen
            # that file. js/radiance_io.js turns this into a dropdown populated
            # from /radiance/exr_layers, and a typed name still works with no
            # frontend at all.
            "layer": ("STRING", {
                "default": "auto",
                "multiline": False,
                "placeholder": "auto · rgba · diffuse · specular · Z · N",
                "tooltip": (
                    "Which EXR layer to read. 'auto' takes the beauty, or the "
                    "first colour layer, or — for a data-only file such as a "
                    "Z-depth pass — the first layer of any kind.\n"
                    "With the frontend loaded this is a dropdown listing the "
                    "layers actually in the selected file."
                ),
            }),
            "on_error": (["Error", "Black frame"], {
                "default": "Error",
                "tooltip": (
                    "What to do when the read fails — file missing, corrupt, "
                    "unreadable.\n"
                    "• Error: the node goes red and the queue stops. What Nuke "
                    "and every other application does, and the default.\n"
                    "• Black frame: return black and carry on. This is what "
                    "3.1.x always did, silently, which is how a black master "
                    "got delivered."
                ),
            }),
            "raw": ("BOOLEAN", {
                "default": False,
                "label_on": "raw (no transform)",
                "label_off": "managed",
                "tooltip": (
                    "Hand back exactly what is stored in the file: no colour "
                    "space decode, and no conform to the EXR display window "
                    "(so overscan is preserved). Nuke's 'raw data'."
                ),
            }),
            "premultiplied": ("BOOLEAN", {
                "default": False,
                "label_on": "premultiplied (unpremult on read)",
                "label_off": "straight alpha",
                "tooltip": (
                    "Tick when the file's RGB is already multiplied by its "
                    "alpha — the EXR convention — and you want it divided back "
                    "out on read. Off by default, matching Nuke, because "
                    "turning it on changes pixels."
                ),
            }),
            # `reload` used to live in "hidden". ComfyUI only populates a hidden
            # key when its VALUE is one of the magic strings (PROMPT, UNIQUE_ID,
            # EXTRA_PNGINFO, ...) -- a widget spec there is simply dropped, so
            # it never reached read() or IS_CHANGED. The frontend also builds
            # widgets from required/optional only, so js/radiance_io.js could
            # not find a widget named "reload" and bailed before adding the
            # button. The RELOAD button has therefore never rendered, and a user
            # whose file changed on disk without an mtime or size change (a
            # network share, an atomic replace) had no way to force a re-read.
            "reload": ("INT", {
                "default": 0,
                "min": 0,
                "max": 2 ** 31 - 1,
                "tooltip": "Bump to force a re-read of the file, for when the "
                           "contents changed but the timestamp did not.",
            }),
        }}

    @classmethod
    def IS_CHANGED(cls, browse: str = "", media_type: str = "Auto", path: str = "", reload: int = 0, **_kw):
        """
        Tell ComfyUI to re-execute the node when the selected file changes.
        Returns a hash of the resolved path so caching is file-content aware.
        """
        resolved = _resolve_browse(browse) or path.strip()
        if resolved and os.path.isfile(resolved):
            try:
                stat = os.stat(resolved)
                return f"{resolved}:{stat.st_mtime}:{stat.st_size}:reload{reload}"
            except Exception as _exc:
                log.debug(
                    "[Radiance] IS_CHANGED(): ignoring %s from `stat = os.stat(resolved)`: %s",
                    type(_exc).__name__, _exc,
                )
        return float("nan")

    def read(
        self,
        browse: str = "",
        media_type: str = "Auto",
        path:   str = "",
        color_space: str = "Auto / Linear (pass-through)",
        start_frame: int = 1001,
        end_frame:   int = 0,
        frame_step:  int = 1,
        max_video_frames: int = 0,
        proxy_scale: float = 0.0,
        missing_frames: str = "Skip",
        layer: str = "auto",
        on_error: str = "Error",
        raw: bool = False,
        premultiplied: bool = False,
        reload: int = 0,
    ):
        # ── Resolve path: browse takes priority over manual path ──────────
        path = _resolve_browse(browse) or strip_path_quotes(path)

        if not path:
            # Not a failure: a node just dropped on the canvas has no file yet.
            log.debug("RadianceRead: no path set yet")
            blank = torch.zeros(1, 8, 8, 3)
            return (blank, blank[..., 0], json.dumps({"kind": "empty"}))

        try:
            image, mask, info = self._read_resolved(
                path, media_type, color_space, start_frame, end_frame,
                frame_step, max_video_frames, missing_frames, layer, raw,
            )
        except Exception as exc:
            log.error("RadianceRead: %s: %s", type(exc).__name__, exc)
            if on_error == "Error":
                # 3.1.x caught everything here and handed back an 8x8 black
                # frame, so a missing plate, a corrupt MOV and a wrong layer
                # name all arrived downstream as black with the node still
                # green -- and the graph carried on and wrote a master out of
                # it. Raising is what Nuke, Resolve and RV all do.
                raise RuntimeError(f"RadianceRead failed on {path!r}: {exc}") from exc
            log.warning(
                "[Radiance/Read] on_error is 'Black frame', so a black 8x8 "
                "frame is being substituted for %s. Nothing downstream can "
                "tell this apart from a genuinely black plate.", path,
            )
            blank = torch.zeros(1, 8, 8, 3)
            return (blank, blank[..., 0], json.dumps(
                {"kind": "error", "path": path, "error": str(exc)}))

        # ── Shared post-processing ────────────────────────────────────────
        if premultiplied and mask is not None and float(mask.abs().max()) > 0:
            image = _unpremultiply(image, mask)
            info["unpremultiplied"] = True

        if proxy_scale > 0:
            image = torch.nn.functional.interpolate(
                image.movedim(-1, 1), scale_factor=proxy_scale, mode="bilinear",
            ).movedim(1, -1)
            if mask is not None:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), scale_factor=proxy_scale, mode="bilinear",
                ).squeeze(1)
            info["proxy_scale"] = proxy_scale

        n, h, w, _ = image.shape
        info.update({"frames": n, "width": w, "height": h,
                     "alpha": mask is not None,
                     "color_space": "raw (untransformed)" if raw else color_space})
        if mask is None:
            mask = torch.zeros(n, h, w)

        log.info("[Radiance/Read] %s", _describe_read(info))
        return (image, mask, json.dumps(info, default=str))

    # ── per-kind dispatch ─────────────────────────────────────────────────

    def _read_resolved(
        self, path, media_type, color_space, start_frame, end_frame,
        frame_step, max_video_frames, missing_frames, layer, raw,
    ):
        """Return (image, mask_or_None, info_dict). Raises on any failure."""
        kind = _path_kind(path) if media_type == "Auto" else media_type.lower()

        # Nuke opens shot.1001.exr and offers you the whole sequence. Radiance
        # read one frame and left you to type a %04d pattern by hand. If the
        # picked file has numbered siblings, this is a sequence.
        if kind in ("image", "exr") and media_type == "Auto":
            detected = _formats.detect_sequence(path)
            if detected is not None:
                log.info(
                    "[Radiance/Read] %s is one frame of a sequence: %s. Reading "
                    "the whole range. Set media_type to Image for just this frame.",
                    os.path.basename(path), detected.summary(),
                )
                path, kind = detected.pattern, "sequence"
                start_frame, end_frame = _sequence_frame_range(
                    detected, start_frame, end_frame)

        if kind == "image":
            image, mask = _read_image(path)
            if not raw:
                image = _np_to_tensor(
                    _apply_input_colorspace(_tensor_to_np(image), color_space))
            return image, mask, {"kind": "image", "path": path}

        if kind == "exr":
            image, mask, exr_info, chosen = _read_exr_with_info(path, layer, raw)
            if not raw:
                image = _np_to_tensor(
                    _apply_input_colorspace(_tensor_to_np(image), color_space))
            return image, mask, {
                "kind": "exr", "path": path, "layer": chosen,
                "layers": exr_info.layer_names, "parts": exr_info.parts,
                "compression": exr_info.compression,
                "data_window": [exr_info.width, exr_info.height],
                "display_window": [exr_info.display_width, exr_info.display_height],
                "overscan": exr_info.has_overscan,
                "attributes": exr_info.attributes or {},
            }

        if kind == "video":
            v_start, v_end = _video_frame_range(path, start_frame, end_frame)
            batch, alpha, _fps, _w, _h, _n, meta = _read_video(
                path, max_video_frames, "Auto / Linear (pass-through)" if raw
                else color_space,
                start_frame=v_start, end_frame=v_end, frame_step=frame_step,
            )
            return batch, alpha, json.loads(meta)

        if kind == "sequence":
            detected = _formats.describe_pattern(path)
            if detected is not None:
                start_frame, end_frame = _sequence_frame_range(
                    detected, start_frame, end_frame)
            batch, alpha, _w, _h, _n, _fps, meta = _read_sequence(
                path, start_frame, end_frame if end_frame > 0 else 99999,
                frame_step, color_space, missing_frames, layer, raw,
            )
            info = json.loads(meta)
            if detected is not None:
                info["available_range"] = [detected.first, detected.last]
            return batch, alpha, info

        raise ValueError(f"RadianceRead cannot read {path!r}. {_explain_unknown(path)}")


def _sequence_frame_range(detected, start_frame: int, end_frame: int):
    """Reconcile the widget values with the range that is actually on disk.

    `start_frame` defaults to 1001, which is right for a VFX sequence and wrong
    for everything else. A start outside the range that exists is treated as
    "not set" -- otherwise a sequence numbered from 1 reads nothing, which is
    the documented directory-pattern trap in a new costume.
    """
    start = start_frame
    if not (detected.first <= start <= detected.last):
        if start not in (0, 1001):
            log.warning(
                "[Radiance/Read] start_frame=%d is outside the range on disk "
                "(%d-%d); reading from %d instead.",
                start, detected.first, detected.last, detected.first,
            )
        start = detected.first
    end = end_frame
    if end <= 0 or end > detected.last:
        end = detected.last
    return start, end


def _unpremultiply(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Divide RGB by alpha, leaving fully transparent pixels alone.

    EXR conventionally stores associated (premultiplied) alpha; ComfyUI's MASK
    is straight by convention. Dividing by zero where alpha is zero would turn
    the transparent region into NaN and poison every downstream operation, so
    those pixels keep their stored value.
    """
    alpha = mask.unsqueeze(-1).to(image.dtype)
    safe = torch.where(alpha > 1e-6, alpha, torch.ones_like(alpha))
    out = torch.where(alpha > 1e-6, image / safe, image)
    return out


def _describe_read(info: dict) -> str:
    """One console line, in the order a Nuke Read's info bar reads."""
    bits = [os.path.basename(str(info.get("path", "")) or "?")]
    if info.get("width"):
        bits.append(f"{info['width']}x{info['height']}")
    frames = info.get("frames", 0)
    if frames and frames > 1:
        bits.append(f"{frames} frames")
    for key in ("kind", "codec", "layer"):
        if info.get(key):
            bits.append(str(info[key]))
    if info.get("bit_depth"):
        bits.append(f"{info['bit_depth']}-bit")
    if info.get("alpha"):
        bits.append("alpha")
    if info.get("overscan"):
        bits.append("overscan conformed")
    if info.get("unpremultiplied"):
        bits.append("unpremultiplied")
    cs = info.get("color_space")
    if cs:
        bits.append(str(cs))
    return " · ".join(bits)


# ═══════════════════════════════════════════════════════════════════════════
# § 4  RadianceWrite
# ═══════════════════════════════════════════════════════════════════════════

class RadianceWrite:
    """
    Universal writer — images, EXR, video, numbered sequences.

    Input
    ─────
    Accepts any ComfyUI output: IMAGE tensor, batched video frames, VHS / VideoHelperSuite
    VIDEO dict, video file path string, or any dict carrying a video path.

    Output path
    ───────────
    Any path the host OS can write to.  Network shares work natively — use the
    path as it appears on the machine running ComfyUI:

      Linux / Mac NFS / SMB   →  /mnt/nas/renders/shot_001
      Windows UNC              →  \\\\server\\share\\renders\\shot_001
      Windows mapped drive     →  Z:\\renders\\shot_001
      S3 / object storage      →  mount with s3fs / rclone first, then use mount path

    The output directory is created automatically (mkdir -p).  Write permission
    on the target location is required; an error is raised if the write fails.

    Format is selected from a flat dropdown.  All format-specific settings
    are optional inputs that are ignored when not relevant:

    ┌──────────────┬─────────────────────────────────────────────────────────┐
    │ Format group │ Key settings used                                        │
    ├──────────────┼─────────────────────────────────────────────────────────┤
    │ Image        │ quality, overwrite                                       │
    │ EXR          │ exr_compression, overwrite                               │
    │ Video        │ fps, quality (CRF), audio_source, broadcast_safe         │
    │ Sequence     │ fps, start_frame, frame_padding, overwrite               │
    └──────────────┴─────────────────────────────────────────────────────────┘

    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = "Write images or EXR sequences to disk with configurable format options."
    FUNCTION     = "write"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    # ALBABIT-FIX: without this, ComfyUI never schedules this node -- it has
    # no outputs for anything else to depend on, and OUTPUT_NODE is the only
    # other way the executor knows to run it. Lost when nodes_io.py was
    # recovered from the working-tree truncation (RadianceDigitalCinemaWrite's
    # docstring nearby references the same incident); the old Radiance
    # registered "◎ Radiance Write" under that shim class instead, which does
    # have this flag, masking the gap here.
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("*", {
                "tooltip": (
                    "Accepts any ComfyUI output: IMAGE tensor, batched video frames, "
                    "VHS"
                    "or any dict/list carrying a video path."
                ),
            }),
            "output_path": ("STRING", {
                "default": str(Path.home() / "radiance_output"),
                "multiline": False,
                "placeholder": "/output/render  or  //nas/share/render  or  Z:/renders/shot",
                "tooltip": (
                    "Output directory + filename stem.  Extension is appended automatically based on format.\n"
                    "For sequences: frame number and extension are appended (e.g. /out/frame_0001.exr).\n\n"
                    "Network paths are fully supported — use the path as mounted on this machine:\n"
                    "  Linux / Mac  →  /mnt/nas/renders/shot_001\n"
                    "  Windows UNC  →  \\\\server\\share\\renders\\shot_001\n"
                    "  Windows drive→  Z:\\renders\\shot_001\n"
                    "The directory is created automatically (mkdir -p) if it does not exist.\n"
                    "Write permissions on the share are required."
                ),
            }),
            "format": (WRITE_FORMATS, {
                "default": "IMG │ EXR (16-bit half)",
                "tooltip": "Output format.  Extension is appended automatically.",
            }),
        }, "optional": {
            "filename": ("STRING", {
                "default": "",
                "placeholder": "shot_001",
                "tooltip": "Output filename stem (version appended automatically). Leave empty to use output_path as the full stem.",
            }),
            "version": ("INT", {
                "default": 1,
                "min": 0,
                "max": 9999,
                "tooltip": "Version number appended to filename (e.g. shot_001_v0001).",
            }),
            "color_space": (OUTPUT_COLOR_SPACES, {
                "default": "Linear (pass-through)",
                "tooltip": "Apply this color space transform before saving.",
            }),
            "fps": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                "tooltip": "Frame rate for video and sequence outputs. "
                           "0 = auto-detect from the source video (falls back to 24 if unavailable).",
            }),
            "quality": ("INT", {
                "default": 18, "min": 0, "max": 51,
                "tooltip": "CRF quality for H.264/H.265 (lower = better).  Also JPEG quality 0–100 (remapped).",
            }),
            "exr_compression": (EXR_COMPRESSIONS, {
                "default": "ZIP",
                "tooltip": "EXR compression codec (EXR formats only).",
            }),
            "start_frame": ("INT", {
                "default": 1001, "min": 0, "max": 99999,
                "tooltip": "First frame number for sequences.",
            }),
            "frame_padding": ("INT", {
                "default": 4, "min": 1, "max": 8,
                "tooltip": "Zero-padding width for frame numbers (e.g. 4 → 0001).",
            }),
            "audio_source": ("STRING", {
                "default": "",
                "tooltip": "Path to audio file to mux into video output (optional). "
                           "Takes priority over the 'audio' input when both are set.",
            }),
            "broadcast_safe": ("BOOLEAN", {
                "default": False,
                "tooltip": "Clamp output to broadcast-legal range (16–235 luma) before saving.",
            }),
            "overwrite": ("BOOLEAN", {
                "default": True,
                "tooltip": "Overwrite existing files.  When disabled, a unique suffix is appended.",
            }),
            "proxy_scale": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Downscale output by this factor for proxy preview (0 = full resolution).",
            }),
            "audio": ("AUDIO", {
                "tooltip": "Audio tensor from RadianceVideoLoader (muxed into video output).",
            }),
            "mask": ("MASK", {
                "tooltip": (
                    "Optional alpha/matte. When connected and the format is EXR or PNG, it is "
                    "written as the alpha channel (RGBA). Ignored for other formats."
                ),
            }),
        }, "hidden": {
            "prompt":        "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
        }}

    # ── helpers ───────────────────────────────────────────────────────────

    def _coerce_to_frames(self, data: Any) -> Tuple[np.ndarray, Optional[float]]:
        """
        Normalise any ComfyUI output to (N, H, W, C) float32 numpy array.

        Handled input types
        ───────────────────
        torch.Tensor          Standard IMAGE (B,H,W,C) or (H,W,C)
        list / tuple          List of IMAGE tensors — stacked into batch
        str (file path)       Image or video file read from disk
        dict                  VHS, AnimateDiff, VideoHelperSuite and similar;
                              also recursively searched for embedded video paths.
                              A dict with "samples" key (LATENT) raises a helpful error.

        Returns
        -------
        (frames_nhwc, detected_fps)   detected_fps is None when not available.
        """
        detected_fps: Optional[float] = None

        # ── 1. Standard IMAGE tensor ──────────────────────────────────────
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().float().numpy()
            if arr.ndim == 3:
                arr = arr[np.newaxis]        # (H,W,C) → (1,H,W,C)
            return arr, detected_fps

        # ── 2. List / tuple of tensors ────────────────────────────────────
        if isinstance(data, (list, tuple)):
            # If all elements are tensors, stack them
            if data and isinstance(data[0], torch.Tensor):
                arr = np.stack(
                    [t.detach().cpu().float().numpy() for t in data], axis=0
                )
                return arr, detected_fps
            # Otherwise, recurse on first element that looks like frames
            for item in data:
                try:
                    return self._coerce_to_frames(item)
                except Exception:
                    continue
            raise ValueError(f"Cannot extract frames from list: {type(data[0]).__name__ if data else 'empty'}")

        # ── 3. Dict (VHS, AnimateDiff, custom video nodes …) ─────────────
        if isinstance(data, dict):
            # Grab fps metadata if available
            for fps_key in ("fps", "frame_rate", "framerate"):
                if fps_key in data and isinstance(data[fps_key], (int, float)):
                    detected_fps = float(data[fps_key])
                    break

            # Check for IMAGE tensor stashed inside the dict
            for key in ("frames", "images", "image", "output", "result"):
                if key in data and isinstance(data[key], torch.Tensor):
                    arr, _ = self._coerce_to_frames(data[key])
                    return arr, detected_fps

            # LATENT dict — helpful error
            if "samples" in data and "batch_index" not in data:
                raise ValueError(
                    "RadianceWrite received a LATENT tensor.  "
                    "Decode it with a VAE Decode node first, then connect the IMAGE output."
                )

            # Search for embedded video file path (VHS-style)
            vpath = _find_video_path(data)
            if vpath:
                arr = _load_video_to_numpy(vpath)
                return arr, detected_fps

            raise ValueError(
                f"RadianceWrite: dict input has no recognised IMAGE key or video path.  "
                f"Keys found: {list(data.keys())}"
            )

        # ── 4. String file path ───────────────────────────────────────────
        if isinstance(data, str):
            ext = Path(data).suffix.lower()
            if os.path.isfile(data):
                if ext in _VID_EXT:
                    arr = _load_video_to_numpy(data)
                    return arr, detected_fps
                if ext in (_IMG_EXT | _EXR_EXT):
                    img_t, _ = _read_image(data)
                    arr = img_t.detach().cpu().float().numpy()
                    if arr.ndim == 3:
                        arr = arr[np.newaxis]
                    return arr, detected_fps
            raise ValueError(f"RadianceWrite: path not found or unsupported extension: {data!r}")

        raise ValueError(
            f"RadianceWrite: unsupported input type {type(data).__name__!r}.  "
            "Connect an IMAGE, batched IMAGE, video file path, or video node output."
        )

    def _out_path(self, base: str, ext: str, overwrite: bool) -> Path:
        """Append `ext` to `base`, without Path.with_suffix's truncation.

        `with_suffix` REPLACES everything after the last dot in the final path
        component. VFX filenames are full of dots -- sh010.comp, plate.v2,
        bg.matte -- and `base` here is "<dir>/<filename>_v0001", so
        "sh010.comp_v0001" has a "suffix" of ".comp_v0001" and every version
        collapsed onto the same "sh010.exr". With overwrite defaulting to True,
        each render silently destroyed the previously approved one and reported
        success with the truncated path.

        An empty `base` was worse: Path("") is Path("."), whose name is "", and
        with_suffix raised `ValueError: PosixPath('.') has an empty name` --
        an opaque crash for the ordinary case of leaving `filename` blank.
        """
        p = Path(base)
        stem = p.name
        if not stem or stem in (".", ".."):
            raise ValueError(
                "RadianceWrite: no output filename. Set the `filename` widget, "
                "or give `output_path` a full file path rather than a directory."
            )
        if not stem.lower().endswith(ext.lower()):
            stem += ext
        p = p.with_name(stem)
        if not overwrite:
            p = _unique_path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ── main ──────────────────────────────────────────────────────────────

    def write(
        self,
        image:          torch.Tensor,
        output_path:    str,
        format:         str,
        filename:       str   = "",
        version:        int   = 1,
        color_space:    str  = "Linear (pass-through)",
        fps:            float = 0.0,
        quality:        int   = 18,
        exr_compression: str  = "ZIP",
        start_frame:    int   = 1001,
        frame_padding:  int   = 4,
        audio_source:   str   = "",
        broadcast_safe: bool  = False,
        overwrite:      bool  = True,
        proxy_scale:    float = 0.0,
        audio:          Any   = None,
        mask:           Any   = None,
        prompt:         Any   = None,
        extra_pnginfo:  Any   = None,
    ):
        output_path = strip_path_quotes(output_path)
        frames, detected_fps = self._coerce_to_frames(image)   # (N, H, W, C)
        # ALBABIT-FIX: fps==24.0 used to mean "auto-detect", indistinguishable
        # from a user explicitly choosing 24 -- an explicit 24 on a 23.976
        # source was silently overridden. 0.0 is now the unambiguous "auto"
        # sentinel; any other value (including 24.0) is respected as-is.
        if fps <= 0.0:
            fps = detected_fps if detected_fps is not None else 24.0

        # Proxy downscale for faster preview
        if proxy_scale > 0:
            t = torch.from_numpy(frames).movedim(-1, 1)
            t = torch.nn.functional.interpolate(t, scale_factor=proxy_scale, mode="bilinear")
            frames = t.movedim(1, -1).numpy()

        # Build output path from filename + version when provided
        if filename.strip():
            ver_str = f"v{version:04d}"
            output_path = str(Path(output_path) / f"{filename}_{ver_str}")
        n, h, w, c = frames.shape

        # Optional alpha: written as the alpha channel for EXR and PNG formats.
        # ALBABIT-FIX: was EXR-only; the old radiance.disabled fork also wrote
        # alpha for PNG (8-bit via Pillow RGBA, 16-bit via cv2 BGRA) -- both
        # paths already accept a 4-channel array transparently, so only this
        # gating condition needed to change.
        alpha = _coerce_mask_to_alpha(mask, n, h, w) if mask is not None and ("EXR" in format or "PNG" in format) else None

        # Apply color space + broadcast-safe clamp.
        #
        # The clamp is now gated on the output format. It used to be
        # unconditional, so exporting "EXR (32-bit float)" with broadcast_safe
        # on -- which the delivery panel defaults to -- clamped every pixel of a
        # scene-referred master into [16/255, 235/255]: highlights destroyed,
        # blacks lifted 6.3%, and the .exr extension making it look like a
        # legitimate HDR deliverable. Legal-range limiting is a video/8-bit
        # concept and must never touch a float or DPX deliverable.
        _is_float_format = ("EXR" in format or "HDR" in format
                            or "32-bit float" in format or "DPX" in format)
        apply_legal_range = bool(broadcast_safe) and not _is_float_format
        if broadcast_safe and _is_float_format:
            log.info("RadianceWrite: broadcast_safe ignored for %s "
                     "(legal-range limiting does not apply to float formats)", format)

        out_frames: List[np.ndarray] = []
        for i, fr in enumerate(frames):
            fr = _apply_output_colorspace(fr, color_space)
            if apply_legal_range:
                fr = np.clip(fr, 16/255.0, 235/255.0)
            if alpha is not None and fr.ndim == 3 and fr.shape[-1] == 3:
                fr = np.concatenate([fr, alpha[i][..., None]], axis=-1)   # RGB -> RGBA
            out_frames.append(fr)

        # ALBABIT-FIX: "audio" (AUDIO-type input) was accepted but never used --
        # only "audio_source" (a STRING path to an existing file) actually muxed
        # into video output. When no explicit audio_source is given, fall back
        # to writing the connected AUDIO tensor to a temp WAV so it reaches the
        # same mux path. Only relevant for video output -- image/sequence
        # formats have no audio track.
        effective_audio_source = audio_source
        temp_audio_wav: Optional[str] = None
        if format in _FMT_VIDEO and not audio_source.strip() and audio is not None:
            temp_audio_wav = _write_audio_temp_wav(audio)
            if temp_audio_wav:
                effective_audio_source = temp_audio_wav

        try:
            saved, count = self._dispatch(
                out_frames, output_path, format,
                fps, quality, exr_compression,
                start_frame, frame_padding,
                effective_audio_source, overwrite,
                prompt, extra_pnginfo,
            )
            log.info("RadianceWrite: saved %d frame(s) → %s", count, saved)
            # Return the saved path so programmatic callers (delivery/handler.py)
            # can report it. ComfyUI ignores extra tuple entries for a node whose
            # RETURN_TYPES is empty, so this is safe for graph use.
            return (saved, count)

        except Exception as e:
            # Never silently swallow a delivery failure — surface it so the node
            # turns red in ComfyUI instead of reporting a phantom success.
            log.error("RadianceWrite failed: %s", e)
            raise

        finally:
            if temp_audio_wav and os.path.exists(temp_audio_wav):
                try:
                    os.unlink(temp_audio_wav)
                except OSError as _exc:
                    log.debug(
                        "[Radiance] write(): ignoring %s from `os.unlink(temp_audio_wav)`: %s",
                        type(_exc).__name__, _exc,
                    )

    def _dispatch(
        self,
        frames:         List[np.ndarray],
        output_path:    str,
        format:         str,
        fps:            float,
        quality:        int,
        exr_compression: str,
        start_frame:    int,
        frame_padding:  int,
        audio_source:   str,
        overwrite:      bool,
        prompt:         Any = None,
        extra_pnginfo:  Any = None,
    ) -> Tuple[str, int]:

        n = len(frames)

        # ── Single image / EXR (first frame only) ──────────────────────────
        if format in _FMT_IMAGE:
            stem = _fmt_stem(format)
            ext_map = {
                "PNG (8-bit)":         ".png",
                "PNG (16-bit)":        ".png",
                "JPEG":                ".jpg",
                "TIFF (16-bit)":       ".tiff",
                "TIFF (32-bit float)": ".tiff",
                "DPX":                 ".dpx",
                "WEBP":                ".webp",
                "EXR (16-bit half)":   ".exr",
                "EXR (32-bit float)":  ".exr",
                "Radiance HDR (.hdr)": ".hdr",
            }
            ext  = ext_map[stem]
            path = self._out_path(output_path, ext, overwrite)
            meta = _workflow_metadata(prompt, extra_pnginfo)
            if "EXR" in stem:
                _save_exr(frames[0], path, half="16-bit" in stem, metadata=meta)
            elif "DPX" in stem:
                _save_dpx(frames[0], path)
            elif "HDR" in stem:
                _save_hdr(frames[0], path)
            else:
                _save_pil_image(frames[0], path, stem, quality, metadata=meta)
            return str(path), 1

        # ── Video ─────────────────────────────────────────────────────────
        if format in _FMT_VIDEO:
            out = _save_video_ffmpeg(
                np.stack(frames, axis=0),
                output_path,
                _fmt_stem(format),
                fps,
                quality,
                audio_source,
            )
            return str(out), n

        # ── Sequences ─────────────────────────────────────────────────────
        if format in _FMT_SEQ:
            stem = _fmt_stem(format)
            out_dir  = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            seq_stem = out_dir.name
            pad_fmt  = f"%0{frame_padding}d"

            is_exr   = "EXR" in stem
            is_tiff  = "TIFF" in stem
            is_dpx   = "DPX" in stem
            is_hdr   = "HDR" in stem
            half_exr = "16-bit" in stem

            if is_exr:
                ext = ".exr"
            elif is_tiff:
                ext = ".tiff"
            elif is_dpx:
                ext = ".dpx"
            elif is_hdr:
                ext = ".hdr"
            else:
                ext = ".png"

            saved_paths = []
            seq_meta = _workflow_metadata(prompt, extra_pnginfo)
            for i, fr in enumerate(frames):
                fn   = f"{seq_stem}_{(pad_fmt % (start_frame + i))}{ext}"
                path = out_dir / fn
                if not overwrite:
                    path = _unique_path(path)
                if is_exr:
                    _save_exr(fr, path, half=half_exr, metadata=seq_meta)
                elif is_dpx:
                    _save_dpx(fr, path)
                elif is_hdr:
                    _save_hdr(fr, path)
                else:
                    _save_pil_image(fr, path, stem, quality, metadata=seq_meta)
                saved_paths.append(str(path))

            return str(out_dir), n

        raise ValueError(f"Unknown format: {format!r}")


# ═══════════════════════════════════════════════════════════════════════════
# § 5  RadianceEXRMultiPart — multi-layer AOV EXR writer
#      (migrated from nodes_io.py, Task #141-fix)
# ═══════════════════════════════════════════════════════════════════════════

_EXR_BIT_DEPTHS   = ["16-bit Half Float", "32-bit Float"]
_EXR_COMPRESSIONS = [
    "ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed",
    "PXR24", "B44", "B44A", "DWAA", "DWAB",
]

def _norm_exr_compression(comp: str) -> str:
    """Normalise UI compression label → write_exr_* keyword."""
    return "NO_COMPRESSION" if comp.lower() == "uncompressed" else comp


def _copy_to_remote_path(local_path: str, remote_path: str) -> bool:
    """Best-effort copy to a remote/UNC path using shutil."""
    try:
        dest = Path(remote_path.strip())
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest / Path(local_path).name)
        return True
    except Exception as exc:
        log.warning("[EXRMultiPart] remote copy failed: %s", exc)
        return False


class RadianceEXRMultiPart:
    """Write a named multi-part EXR v2 combining up to 6 AOV layers into one
    file readable by Nuke, DaVinci Resolve (Flatten Layers), and Fusion.

    Standard parts:
      • beauty      → R, G, B (+ A if alpha connected)
      • depth       → Z  (single channel)
      • normal      → NX, NY, NZ
      • albedo      → albedo.R/G/B
      • custom_1/2  → <name>.R/G/B

    Fallback: if OpenEXR v2 multi-part is unavailable the node writes
    separate per-part .exr files instead.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = "Read or write multi-part OpenEXR files with named channel layers."
    FUNCTION    = "write_multipart"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "radiance_multipart"}),
                "beauty":    ("IMAGE",),
                "bit_depth": (_EXR_BIT_DEPTHS,   {"default": "16-bit Half Float"}),
                "compression": (_EXR_COMPRESSIONS, {"default": "ZIP"}),
            },
            "optional": {
                "depth":         ("IMAGE",),
                "normal":        ("IMAGE",),
                "albedo":        ("IMAGE",),
                "custom_1":      ("IMAGE",),
                "custom_1_name": ("STRING", {"default": "emission"}),
                "custom_2":      ("IMAGE",),
                "custom_2_name": ("STRING", {"default": "specular"}),
                "output_path":   ("STRING", {"default": ""}),
                "remote_path":   ("STRING", {"default": ""}),
                "frame_index":   ("INT",    {"default": 1, "min": 1}),
                "custom_metadata": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def write_multipart(
        self,
        filename_prefix: str,
        beauty: torch.Tensor,
        bit_depth: str = "16-bit Half Float",
        compression: str = "ZIP",
        depth: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        albedo: Optional[torch.Tensor] = None,
        custom_1: Optional[torch.Tensor] = None,
        custom_1_name: str = "emission",
        custom_2: Optional[torch.Tensor] = None,
        custom_2_name: str = "specular",
        output_path: str = "",
        remote_path: str = "",
        frame_index: int = 1,
        custom_metadata: str = "",
    ) -> Tuple[str]:
        import datetime, re as _re

        output_path = strip_path_quotes(output_path)
        remote_path = strip_path_quotes(remote_path)

        _fp = _folder_paths if _HAS_FOLDER_PATHS else None
        base_dir = _fp.get_output_directory() if _fp else tempfile.gettempdir()
        out_dir = get_safe_output_dir(base_dir, output_path, allow_absolute=True)
        os.makedirs(out_dir, exist_ok=True)

        frame_num = str(frame_index).zfill(4)
        filepath  = os.path.join(out_dir, f"{filename_prefix}.{frame_num}.exr")
        comp      = _norm_exr_compression(compression)

        # Validate custom part names — spaces / special chars corrupt multi-part EXR
        _SAFE = _re.compile(r'^[A-Za-z0-9_\-\.]+$')
        for name in (custom_1_name, custom_2_name):
            if name and not _SAFE.match(name):
                raise ValueError(
                    f"[EXRMultiPart] Invalid part name '{name}'. "
                    "Use only letters, digits, underscore, hyphen, or dot."
                )

        meta: Dict[str, Any] = {
            "software": f"Radiance v{_RADIANCE_VERSION}",
            "rad_version": _RADIANCE_VERSION,
            "created":  datetime.datetime.now().isoformat(),
        }
        for line in custom_metadata.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()

        def _t(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            if t is None:
                return None
            arr = t.squeeze(0) if t.dim() == 4 and t.shape[0] == 1 else t
            if arr.dim() == 4:
                arr = arr[0]
            return arr.float().cpu().numpy()

        parts: Dict[str, np.ndarray] = {}
        parts["beauty"] = _t(beauty)  # type: ignore[assignment]
        for tensor, name in (
            (depth,    "depth"),
            (normal,   "normal"),
            (albedo,   "albedo"),
            (custom_1, custom_1_name or "custom_1"),
            (custom_2, custom_2_name or "custom_2"),
        ):
            arr = _t(tensor)
            if arr is not None:
                parts[name] = arr

        if write_exr_multipart is not None:
            ok = write_exr_multipart(filepath, parts, bit_depth, comp, meta)
        elif write_exr_robust is not None:
            log.warning("[EXRMultiPart] multi-part unavailable — writing beauty-only EXR")
            ok = write_exr_robust(filepath, parts["beauty"], bit_depth, comp, meta)
        else:
            log.error("[EXRMultiPart] No EXR writer available")
            ok = False

        if not ok:
            raise RuntimeError(f"[EXRMultiPart] Failed to write: {filepath}")

        log.info("[EXRMultiPart] Wrote %d parts → %s", len(parts), filepath)

        if remote_path:
            _copy_to_remote_path(filepath, remote_path)

        return (filepath,)


# ═══════════════════════════════════════════════════════════════════════════
# § 6  Registration
# ═══════════════════════════════════════════════════════════════════════════

# Shims for backwards compatibility
class RadianceDigitalCinemaRead:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": ""}),
                "read_mode": (["Auto", "Video", "Sequence", "EXR"], {"default": "Auto"}),
                "start_frame": ("INT", {"default": 1, "min": 1}),
                "frame_limit": ("INT", {"default": 0, "min": 0}),
                "input_colorspace": (["sRGB (Standard)"], {"default": "sRGB (Standard)"}),
                "fps_override": ("FLOAT", {"default": 0.0, "min": 0.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "RADIANCE_SHOT")
    RETURN_NAMES = ("image", "mask", "shot_metadata")
    FUNCTION = "read"
    CATEGORY = "FXTD Studios/Radiance/IO"

    def read(self, source_path, read_mode, start_frame, frame_limit, input_colorspace, fps_override=0.0):
        media_type = "Auto"
        if read_mode == "Video":
            media_type = "Video"
        elif read_mode in ("Sequence", "EXR"):
            media_type = "Sequence"

        reader = RadianceRead()
        img, mask, read_info = reader.read(
            path=source_path,
            media_type=media_type,
            color_space="sRGB" if "sRGB" in input_colorspace else "Auto / Linear (pass-through)",
            start_frame=start_frame,
            end_frame=(start_frame + frame_limit - 1) if frame_limit > 0 else 0,
        )

        # RECOVERY STUB: the working-tree file was truncated mid-edit at exactly
        # this point (the module would not import). This minimal body restores
        # import-ability and honors the declared (IMAGE, MASK, RADIANCE_SHOT)
        # contract. Review/replace with the intended Digital Cinema metadata logic.
        shot_metadata = {
            "source_path": source_path,
            "read_mode": read_mode,
            "start_frame": start_frame,
            "frame_limit": frame_limit,
            "input_colorspace": input_colorspace,
            "fps_override": fps_override,
        }
        # Fold in what the reader actually found -- resolution, real frame
        # count, codec, bit depth, colour tags, timecode -- rather than echoing
        # back only the widget values it was given.
        try:
            shot_metadata["source"] = json.loads(read_info)
        except (TypeError, ValueError) as _exc:
            log.debug("[Radiance] DigitalCinemaRead: unreadable info payload: %s", _exc)
        return (img, mask, shot_metadata)


class RadianceDigitalCinemaWrite:
    """Backward-compatible Digital Cinema writer shim.

    Delegates to RadianceWrite. Recovered after the working-tree truncation that
    removed this class; provides the OUTPUT_NODE write surface the pipeline and
    tests expect. Returns a STRING status.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # Same default as RadianceWrite, which this node delegates to.
                # It used to default to "", so dropping the node and queueing
                # crashed inside pathlib instead of writing anything.
                "output_path": ("STRING", {
                    "default": str(Path.home() / "radiance_output"),
                    "placeholder": "/output/render  or  Z:/renders/shot",
                }),
            },
            "optional": {
                "format": (WRITE_FORMATS, {"default": "IMG │ EXR (16-bit half)"}),
                "filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "FXTD STUDIOS/Radiance/Pipeline"

    def write(self, images, output_path, format="IMG │ EXR (16-bit half)", filename=""):
        writer = RadianceWrite()
        writer.write(image=images, output_path=output_path, format=format, filename=filename)
        return (f"OK: wrote '{filename or output_path}' as {format}",)


NODE_CLASS_MAPPINGS = {
    "RadianceRead": RadianceRead,
    "RadianceWrite": RadianceWrite,
    "RadianceEXRMultiPart": RadianceEXRMultiPart,
    "RadianceDigitalCinemaRead": RadianceDigitalCinemaRead,
    "RadianceDigitalCinemaWrite": RadianceDigitalCinemaWrite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDigitalCinemaWrite": "◎ Radiance Digital Cinema Write",
    "RadianceRead": "◎ Radiance Read",
    "RadianceWrite": "◎ Radiance Write",
    "RadianceEXRMultiPart": "◎ Radiance EXR Multi-Part",
    "RadianceDigitalCinemaRead": "◎ Radiance Digital Cinema Read",
}


# ═══════════════════════════════════════════════════════════════════════════
# § 6  HTTP routes for the Read node's frontend
# ═══════════════════════════════════════════════════════════════════════════

#: Directories the /radiance/media/* routes may inspect.
#:
#: ComfyUI's own input/output/temp trees are always allowed, because that is
#: where uploads land. A facility reads plates off a NAS, so add those mounts
#: with RADIANCE_READ_ROOTS (os.pathsep-separated). Without it the layer
#: dropdown falls back to a plain text field for paths outside the tree, which
#: still works -- you type the layer name.
_ENV_READ_ROOTS = "RADIANCE_READ_ROOTS"


def _allowed_read_roots() -> List[str]:
    roots: List[str] = []
    try:
        import folder_paths  # type: ignore

        for attr in ("get_input_directory", "get_output_directory",
                     "get_temp_directory"):
            getter = getattr(folder_paths, attr, None)
            if callable(getter):
                roots.append(getter())
        models = getattr(folder_paths, "models_dir", None)
        if models:
            roots.append(models)
    except Exception as _exc:
        log.debug("[Radiance] _allowed_read_roots(): no folder_paths: %s", _exc)
    for part in os.environ.get(_ENV_READ_ROOTS, "").split(os.pathsep):
        part = part.strip().strip('"')
        if part:
            roots.append(part)
    return [os.path.abspath(os.path.expanduser(r)) for r in roots if r]


def _is_inside_allowed_read_root(path: str) -> bool:
    """Is this path somewhere the frontend is allowed to ask us to inspect?

    The Read node itself opens any path the user types -- that is the job. This
    check governs the *unauthenticated HTTP route* only, which would otherwise
    be a file-existence oracle for the whole filesystem.
    """
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    for root in _allowed_read_roots():
        try:
            if os.path.commonpath([resolved, os.path.realpath(root)]) == \
                    os.path.realpath(root):
                return True
        except ValueError:      # different drives on Windows
            continue
    return False


def register_read_routes():
    """Endpoints the Read node's widgets call: EXR layers, and media info."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        log.debug("[Radiance/Read] server not available; routes not registered")
        return

    if getattr(PromptServer.instance, "_radiance_read_routes_registered", False):
        return
    PromptServer.instance._radiance_read_routes_registered = True

    def _resolve_query_path(request) -> Tuple[Optional[str], Optional[Any]]:
        raw = (request.query.get("path") or "").strip()
        if not raw:
            return None, web.json_response({"error": "no path given"}, status=400)
        candidate = _resolve_browse(raw) or strip_path_quotes(raw)
        if not os.path.isfile(candidate):
            return None, web.json_response(
                {"error": "not a file", "path": candidate}, status=404)
        if not _is_inside_allowed_read_root(candidate):
            return None, web.json_response({
                "error": "outside the allowed roots",
                "hint": f"Set {_ENV_READ_ROOTS} to the directories Radiance may "
                        f"inspect, os.pathsep-separated.",
            }, status=403)
        return candidate, None

    @PromptServer.instance.routes.get("/radiance/media/layers")
    async def read_layers_endpoint(request):
        """Layer names inside an EXR, for the `layer` dropdown."""
        path, error = _resolve_query_path(request)
        if error is not None:
            return error
        if os.path.splitext(path)[1].lower() not in _EXR_EXT:
            return web.json_response({"layers": [], "reason": "not an EXR"})
        try:
            from .core import exr as _exr

            return web.json_response({"layers": ["auto"] + _exr.layer_choices(path)})
        except Exception as exc:
            log.debug("[Radiance/Read] layer probe failed for %s: %s", path, exc)
            return web.json_response({"layers": [], "error": str(exc)})

    @PromptServer.instance.routes.get("/radiance/media/info")
    async def read_info_endpoint(request):
        """What the file is, for the info line drawn on the node."""
        path, error = _resolve_query_path(request)
        if error is not None:
            return error
        try:
            return web.json_response(_probe_for_ui(path))
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=200)

    log.debug("[Radiance/Read] HTTP routes registered: /radiance/media/*")


def _probe_for_ui(path: str) -> Dict[str, Any]:
    """A short, cheap description of a file for the node's info line."""
    kind = _path_kind(path)
    out: Dict[str, Any] = {"kind": kind, "name": os.path.basename(path)}

    if kind == "video":
        from .core import video as _video

        info = _video.probe(path)
        out.update(info.as_dict())
        out["summary"] = info.summary()
        return out

    if kind == "exr":
        from .core import exr as _exr

        info = _exr.probe(path)
        out.update({
            "width": info.width, "height": info.height,
            "display_width": info.display_width,
            "display_height": info.display_height,
            "overscan": info.has_overscan, "parts": info.parts,
            "layers": info.layer_names, "compression": info.compression,
            "summary": info.summary(),
        })

    sequence = _formats.detect_sequence(path)
    if sequence is not None:
        out.update({
            "sequence": sequence.pattern,
            "first": sequence.first, "last": sequence.last,
            "frames": sequence.count, "missing": len(sequence.missing),
            "summary": (out.get("summary", "") + " · " + sequence.summary()).strip(" ·"),
        })
    if "summary" not in out:
        try:
            image, mask = _read_image(path)
            _, h, w, _ = image.shape
            out.update({"width": w, "height": h, "alpha": mask is not None,
                        "summary": f"{w}x{h}" + (" · alpha" if mask is not None else "")})
        except Exception as exc:
            out["summary"] = f"unreadable: {exc}"
    return out


# Auto-register on import, the same way the OCIO routes do.
try:
    register_read_routes()
except Exception as _exc:  # pragma: no cover - server not present in tests
    log.debug("[Radiance/Read] route registration deferred: %s", _exc)
