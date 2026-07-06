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
  Video     .mp4 .mov .mxf .avi .webm .mkv  → batch of frames
  Sequence  any of the above with %04d / #### / * pattern, or a directory

Supported WRITE
───────────────
  Image        PNG (8-bit · 16-bit) · JPEG · TIFF (16 · 32f) · DPX · WEBP
  EXR          16-bit half · 32-bit float
  Video        MP4 H.264 · MP4 H.265 10-bit · MOV ProRes 422 / 4444 · MOV DNxHR
  Sequence     PNG / TIFF / EXR / DPX numbered sequences
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

from . import color_utils

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

def _resolve_portable_path(path: str, project_root: str = "") -> str:
    """
    Resolves portable paths (starting with ./) relative to the project root.
    """
    if not path:
        return path
        
    path = path.strip()
    if path.startswith("./") and project_root:
        # Resolve ./ relative to project_root
        root = Path(project_root).resolve()
        relative = path[2:] # Strip ./
        resolved = root / relative
        return str(resolved)
        
    return path


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
INPUT_COLOR_SPACES = [
    "Auto / Linear (pass-through)",
    "sRGB",
    "ARRI LogC4",
    "ARRI LogC3",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "ACEScg",
    "ACEScct",
]

# ── EXR compressions ───────────────────────────────────────────────────────
EXR_COMPRESSIONS = ["ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed", "DWAA", "DWAB"]

# ── Image file extensions ──────────────────────────────────────────────────
_IMG_EXT  = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".dpx", ".hdr"}
_VID_EXT  = {".mp4", ".mov", ".mxf", ".avi", ".webm", ".mkv", ".m4v"}
_EXR_EXT  = {".exr"}


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
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1, H, W, C) or (H, W, C) tensor → float32 ndarray."""
    arr = t.detach().cpu().float().numpy()
    return arr[0] if arr.ndim == 4 else arr


def _ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None


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


def _apply_input_colorspace(arr: np.ndarray, cs: str) -> np.ndarray:
    """Decode an input color space to scene-linear float32."""
    if cs in ("Auto / Linear (pass-through)", "ACEScg"):
        return arr
    try:
        mapping = {
            "sRGB":                  color_utils.srgb_to_linear,
            "ARRI LogC4":            color_utils.logc4_to_linear,
            "ARRI LogC3":            color_utils.logc3_to_linear,
            "Sony S-Log3":           color_utils.slog3_to_linear,
            "Panasonic V-Log":       color_utils.vlog_to_linear,
            "DaVinci Intermediate":  color_utils.davinci_intermediate_to_linear,
            "ACEScct":               color_utils.acescct_to_linear,
        }
        fn = mapping.get(cs)
        if fn:
            return fn(arr)
    except Exception as e:
        log.warning("Input color space decode '%s' failed: %s", cs, e)
    return arr


# ── Path type detection ────────────────────────────────────────────────────

def _path_kind(path: str) -> str:
    """
    Classify a path as: "image", "exr", "video", "sequence", or "unknown".

    Sequence detection:
      - contains %04d / %d style printf format
      - contains #### hash padding
      - contains * glob
      - path is a directory
    """
    p   = path.strip()
    ext = Path(p).suffix.lower()

    # Sequence patterns
    is_seq = ("%0" in p or "##" in p or "*" in p or
              re.search(r"%\d*d", p) or os.path.isdir(p))

    if is_seq:
        return "sequence"
    if ext in _EXR_EXT:
        return "exr"
    if ext in _VID_EXT:
        return "video"
    if ext in _IMG_EXT:
        return "image"
    return "unknown"


# ── Image read ────────────────────────────────────────────────────────────

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

    if not _HAS_PIL:
        raise ImportError("Pillow is required to read image files.")
    pil = _PIL.open(path)
    has_alpha = pil.mode in ("RGBA", "LA", "PA")
    pil_rgb = pil.convert("RGBA") if has_alpha else pil.convert("RGB")
    arr = np.array(pil_rgb, dtype=np.float32)
    maxv = 65535.0 if pil.mode.endswith("16") or np.array(pil).max() > 255 else 255.0
    rgb  = arr[..., :3] / maxv
    mask = (arr[..., 3:4] / maxv) if has_alpha else None
    img_t  = _np_to_tensor(rgb)
    mask_t = _np_to_tensor(mask[..., 0]) if mask is not None else None
    return img_t, mask_t


def _read_exr_single(path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Read a single EXR file using OpenEXR → numpy → tensor."""
    try:
        import OpenEXR, Imath  # type: ignore
        f    = OpenEXR.InputFile(path)
        hdr  = f.header()
        dw   = hdr["dataWindow"]
        w    = dw.max.x - dw.min.x + 1
        h    = dw.max.y - dw.min.y + 1
        pt   = Imath.PixelType(Imath.PixelType.FLOAT)
        r    = np.frombuffer(f.channel("R", pt), dtype=np.float32).reshape(h, w)
        g    = np.frombuffer(f.channel("G", pt), dtype=np.float32).reshape(h, w)
        b    = np.frombuffer(f.channel("B", pt), dtype=np.float32).reshape(h, w)
        arr  = np.stack([r, g, b], axis=-1)
        mask = None
        if "A" in hdr["channels"]:
            a    = np.frombuffer(f.channel("A", pt), dtype=np.float32).reshape(h, w)
            mask = torch.from_numpy(a).unsqueeze(0)
        return _np_to_tensor(arr), mask
    except ImportError:
        pass

    # Fallback: cv2 with OpenEXR flag
    try:
        import cv2  # type: ignore
        arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if arr is not None:
            # EXR is scene-linear float; preserve magnitude (no normalization).
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
            return _np_to_tensor(arr), None
    except Exception:
        pass

    raise RuntimeError(f"Cannot read EXR '{path}': install OpenEXR or OpenCV with EXR support.")


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

    frames = list(range(start, end + 1, step)) if end >= start else [start]
    paths  = []
    missing = []
    for f in frames:
        p = pattern % f
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing.append(p)
            if missing_frames in ("Black", "Skip"):
                paths.append(p)   # keep slot for black-frame insertion
            else:
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
) -> Tuple[torch.Tensor, int, int, int, float, str]:
    """Read a frame sequence → batched IMAGE tensor."""
    paths = _resolve_sequence_paths(pattern, start, end, step, missing_frames)
    if not paths:
        raise FileNotFoundError(f"No frames found for pattern: {pattern}")

    frames: List[torch.Tensor] = []
    for p in paths:
        if os.path.isfile(p):
            img_t, _ = _read_image(p)
            arr = _tensor_to_np(img_t)
            arr = _apply_input_colorspace(arr, input_cs)
            frames.append(torch.from_numpy(arr).unsqueeze(0))
        else:
            # Black frame for missing
            blank = torch.zeros(1, 8, 8, 3)
            frames.append(blank)

    batch  = torch.cat(frames, dim=0)   # (N, H, W, C)
    _, h, w, _ = batch.shape
    return batch, w, h, len(frames), 24.0, json.dumps({"frame_count": len(frames)})


# ── Video read ────────────────────────────────────────────────────────────

def _read_video(
    path: str,
    max_frames: int,
    input_cs: str,
) -> Tuple[torch.Tensor, float, int, int, int, str]:
    """Decode a video file to a batched IMAGE tensor using ffprobe/ffmpeg."""
    if not _ffmpeg_ok():
        raise RuntimeError("ffmpeg not found — install ffmpeg to read video files.")

    # --- probe ---
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", path,
    ]
    try:
        probe_out = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        probe_data = json.loads(probe_out.stdout)
        vstream = next(
            (s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"),
            {},
        )
        fps_str = vstream.get("r_frame_rate", "24/1")
        num, den = (float(x) for x in fps_str.split("/"))
        fps = round(num / max(den, 1), 6)
        w   = int(vstream.get("width", 0))
        h   = int(vstream.get("height", 0))
        nb_frames = int(vstream.get("nb_frames", 0))
        meta = json.dumps({"fps": fps, "width": w, "height": h,
                           "codec": vstream.get("codec_name", ""),
                           "nb_frames": nb_frames})
    except Exception as e:
        log.warning("ffprobe failed (%s); using defaults", e)
        fps, w, h, nb_frames = 24.0, 0, 0, 0
        meta = "{}"

    # --- decode ---
    vf_args = []
    if max_frames > 0:
        vf_args = ["-vframes", str(max_frames)]

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pat = os.path.join(tmpdir, "f%06d.png")
        decode_cmd = [
            "ffmpeg", "-v", "error", "-i", path,
        ] + vf_args + [
            "-vsync", "0", "-f", "image2", frame_pat,
        ]
        subprocess.run(decode_cmd, check=True, capture_output=True, timeout=300)

        frame_files = sorted(Path(tmpdir).glob("*.png"))
        if not frame_files:
            raise RuntimeError("ffmpeg produced no frames.")

        frames: List[torch.Tensor] = []
        for fp in frame_files:
            img_t, _ = _read_image(str(fp))
            arr = _tensor_to_np(img_t)
            arr = _apply_input_colorspace(arr, input_cs)
            frames.append(torch.from_numpy(arr).unsqueeze(0))

    batch = torch.cat(frames, dim=0)
    _, h_, w_, _ = batch.shape
    return batch, fps, w_, h_, len(frames), meta


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


def _load_video_to_numpy(path: str, max_frames: int = 0) -> np.ndarray:
    """
    Decode a video file path to (N, H, W, C) float32 via OpenCV if available,
    otherwise via ffmpeg pipe.
    """
    # OpenCV fast path
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        frames_np: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_np.append(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            if max_frames > 0 and len(frames_np) >= max_frames:
                break
        cap.release()
        if frames_np:
            return np.stack(frames_np, axis=0)
    except Exception as e:
        log.debug("OpenCV video decode failed (%s); falling back to ffmpeg", e)

    # ffmpeg fallback
    if not _ffmpeg_ok():
        raise RuntimeError("Neither OpenCV nor ffmpeg available to decode video.")
    with tempfile.TemporaryDirectory() as tmp:
        frame_pat = os.path.join(tmp, "f%06d.png")
        cmd = ["ffmpeg", "-v", "error", "-i", path]
        if max_frames > 0:
            cmd += ["-vframes", str(max_frames)]
        cmd += ["-vsync", "0", "-f", "image2", frame_pat]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        pngs = sorted(Path(tmp).glob("*.png"))
        if not pngs:
            raise RuntimeError(f"ffmpeg produced no frames from {path!r}")
        frames_np = []
        for fp in pngs:
            if _HAS_PIL:
                arr = np.array(_PIL.open(str(fp)).convert("RGB"), dtype=np.float32) / 255.0
            else:
                import cv2  # type: ignore
                arr = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames_np.append(arr)
    return np.stack(frames_np, axis=0)


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


def _save_pil_image(arr_f32: np.ndarray, path: Path, fmt: str, quality: int = 18) -> None:
    """Save a float32 (H, W, 3) array to an image file via Pillow.

    `quality` is the same widget value used for video CRF (0-51, lower = better).
    JPEG/WEBP use the opposite convention (0-100, higher = better), so it is
    remapped rather than passed through raw -- matching the widget's own
    tooltip ("Also JPEG quality 0-100 (remapped)"), which previously had no
    code behind it at all (see ALBABIT-FIX below).
    """
    if not _HAS_PIL:
        raise ImportError("Pillow required for image writing.")

    if "16-bit" in fmt or "TIFF" in fmt:
        arr_u16 = (np.clip(arr_f32, 0, 1) * 65535).astype(np.uint16)
        pil = _PIL.fromarray(arr_u16, mode="I;16" if arr_f32.ndim == 2 else "RGB")
        # Pillow can't do 16-bit RGB directly; use tifffile if available
        try:
            import tifffile  # type: ignore
            tifffile.imwrite(str(path), arr_u16)
            return
        except ImportError:
            # Fallback: save 8-bit
            arr_f32 = np.clip(arr_f32, 0, 1)
    if "32-bit" in fmt:
        try:
            import tifffile  # type: ignore
            tifffile.imwrite(str(path), arr_f32.astype(np.float32))
            return
        except ImportError:
            pass

    arr_u8 = (np.clip(arr_f32, 0, 1) * 255).astype(np.uint8)
    pil = _PIL.fromarray(arr_u8)
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


def _save_exr(arr_f32: np.ndarray, path: Path, half: bool) -> None:
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
        import OpenEXR, Imath  # type: ignore
        h, w = arr_f32.shape[:2]
        pixel_type = Imath.PixelType(Imath.PixelType.HALF if half else Imath.PixelType.FLOAT)
        hdr = OpenEXR.Header(w, h)
        try:
            hdr["rad_version"] = _RADIANCE_VERSION
            hdr["software"] = f"Radiance v{_RADIANCE_VERSION}"
        except Exception:
            pass  # header stamp is best-effort; never block the EXR write
        hdr["channels"] = {name: Imath.Channel(pixel_type) for name, _ in channels}
        out = OpenEXR.OutputFile(str(path), hdr)
        try:
            out.writePixels({
                name: np.ascontiguousarray(plane).astype(np_dtype).tobytes()
                for name, plane in channels
            })
        finally:
            out.close()
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
    """Encode a frame batch to video via ffmpeg."""
    if not _ffmpeg_ok():
        raise RuntimeError("ffmpeg not found.")

    fmt_map = {
        "MP4 (H.264)":        ("libx264",  "yuv420p",    ".mp4", ["-crf", str(crf), "-preset", "medium"]),
        "MP4 (H.265 10-bit)": ("libx265",  "yuv420p10le",".mp4", ["-crf", str(crf), "-preset", "medium", "-tag:v", "hvc1"]),
        "MOV (ProRes 422 HQ)":("prores_ks","yuv422p10le",".mov", ["-profile:v", "3", "-qscale:v", "9"]),
        "MOV (ProRes 4444)":  ("prores_ks","yuva444p10le",".mov",["-profile:v", "4", "-qscale:v", "9"]),
        "MOV (DNxHR HQ)":     ("dnxhd",   "yuv422p",    ".mxf", ["-profile:v", "dnxhr_hq"]),
    }
    codec, pix_fmt, ext, extra = fmt_map.get(fmt, ("libx264", "yuv420p", ".mp4", ["-crf", str(crf)]))

    out_path = str(output_path)
    if not out_path.lower().endswith(ext):
        out_path += ext

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(frames):
            fpath = os.path.join(tmpdir, f"{i:06d}.png")
            _save_pil_image(frame, Path(fpath), "PNG")

        cmd = [
            "ffmpeg", "-v", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "%06d.png"),
        ]
        if audio_source and os.path.isfile(audio_source):
            cmd += ["-i", audio_source, "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", codec, "-pix_fmt", pix_fmt] + extra + ["-y", out_path]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)

    return out_path


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
    DESCRIPTION = "Read images or EXR sequences from disk into the pipeline."
    FUNCTION     = "read"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

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
                "tooltip": "First frame index (sequences only).",
            }),
            "end_frame": ("INT", {
                "default": 0, "min": 0, "max": 99999,
                "tooltip": "Last frame index (0 = read all frames).",
            }),
            "frame_step": ("INT", {
                "default": 1, "min": 1, "max": 100,
                "tooltip": "Step size — e.g. 2 reads every other frame.",
            }),
            "max_video_frames": ("INT", {
                "default": 0, "min": 0, "max": 99999,
                "tooltip": "Cap on decoded video frames (0 = all frames). Large videos use a lot of RAM.",
            }),
            "proxy_scale": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Downscale factor for proxy preview (0 = full resolution). 0.5 = half res for faster iteration.",
            }),
            "missing_frames": (["Error", "Black", "Skip"], {
                "default": "Skip",
                "tooltip": "How to handle missing sequence frames. Black inserts zero frames, Skip omits them, Error raises.",
            }),
        }, "hidden": {
            "reload": ("INT", {"default": 0, "tooltip": "Bump to force re-read."}),
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
            except Exception:
                pass
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
        reload: int = 0,
    ):
        # ── Resolve path: browse takes priority over manual path ──────────
        resolved = _resolve_browse(browse) or strip_path_quotes(path)
        path = resolved

        if not path:
            log.warning("RadianceRead: no path provided — returning empty frame")
            blank = torch.zeros(1, 8, 8, 3)
            return (blank, blank[..., 0])

        # Auto-detect or use explicit override
        kind = _path_kind(path) if media_type == "Auto" else media_type.lower()
        empty_mask = torch.zeros(1, 8, 8)

        try:
            if kind == "image":
                img_t, mask_t = _read_image(path)
                arr = _tensor_to_np(img_t)
                arr = _apply_input_colorspace(arr, color_space)
                img_t = _np_to_tensor(arr)
                _, h, w, _ = img_t.shape
                meta = json.dumps({"kind": "image", "path": path,
                                   "width": w, "height": h, "color_space": color_space})
                mask_out = mask_t if mask_t is not None else torch.zeros(1, h, w)
                if proxy_scale > 0:
                    img_t = torch.nn.functional.interpolate(
                        img_t.movedim(-1, 1), scale_factor=proxy_scale, mode="bilinear"
                    ).movedim(1, -1)
                return (img_t, mask_out)

            elif kind == "exr":
                img_t, mask_t = _read_exr_single(path)
                arr = _tensor_to_np(img_t)
                arr = _apply_input_colorspace(arr, color_space)
                img_t = _np_to_tensor(arr)
                _, h, w, _ = img_t.shape
                mask_out = mask_t if mask_t is not None else torch.zeros(1, h, w)
                if proxy_scale > 0:
                    img_t = torch.nn.functional.interpolate(
                        img_t.movedim(-1, 1), scale_factor=proxy_scale, mode="bilinear"
                    ).movedim(1, -1)
                return (img_t, mask_out)

            elif kind == "video":
                batch, fps, w, h, n, meta = _read_video(path, max_video_frames, color_space)
                if proxy_scale > 0:
                    batch = torch.nn.functional.interpolate(
                        batch.movedim(-1, 1), scale_factor=proxy_scale, mode="bilinear"
                    ).movedim(1, -1)
                return (batch, torch.zeros(n, h, w))

            elif kind == "sequence":
                batch, w, h, n, fps, meta = _read_sequence(
                    path, start_frame, end_frame if end_frame > 0 else 99999,
                    frame_step, color_space, missing_frames,
                )
                if proxy_scale > 0:
                    batch = torch.nn.functional.interpolate(
                        batch.movedim(-1, 1), scale_factor=proxy_scale, mode="bilinear"
                    ).movedim(1, -1)
                return (batch, torch.zeros(n, h, w))

            else:
                log.warning("RadianceRead: unknown path type for '%s'", path)
                blank = torch.zeros(1, 8, 8, 3)
                return (blank, blank[..., 0])

        except Exception as e:
            log.error("RadianceRead: %s", e)
            blank = torch.zeros(1, 8, 8, 3)
            return (blank, blank[..., 0])


# ═══════════════════════════════════════════════════════════════════════════
# § 4  RadianceWrite
# ═══════════════════════════════════════════════════════════════════════════

class RadianceWrite:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = "Write images or EXR sequences to disk with configurable format options."
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
                "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                "tooltip": "Frame rate for video and sequence outputs.",
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
                "tooltip": "Path to audio file to mux into video output (optional).",
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
                    "Optional alpha/matte. When connected and the format is EXR, it is "
                    "written as the EXR alpha channel (RGBA). Ignored for non-EXR formats."
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
        p = Path(base).with_suffix(ext)
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
        fps:            float = 24.0,
        quality:        int   = 18,
        exr_compression: str  = "ZIP",
        start_frame:    int   = 1,
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
        # Use fps from the incoming data when the user hasn't overridden
        if detected_fps is not None and fps == 24.0:
            fps = detected_fps

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

        # Optional alpha: written as the EXR alpha channel when an EXR format is chosen.
        alpha = _coerce_mask_to_alpha(mask, n, h, w) if (mask is not None and "EXR" in format) else None

        # Apply color space + broadcast-safe clamp
        out_frames: List[np.ndarray] = []
        for i, fr in enumerate(frames):
            fr = _apply_output_colorspace(fr, color_space)
            if broadcast_safe:
                fr = np.clip(fr, 16/255.0, 235/255.0)
            if alpha is not None and fr.ndim == 3 and fr.shape[-1] == 3:
                fr = np.concatenate([fr, alpha[i][..., None]], axis=-1)   # RGB -> RGBA
            out_frames.append(fr)

        try:
            saved, count = self._dispatch(
                out_frames, output_path, format,
                fps, quality, exr_compression,
                start_frame, frame_padding,
                audio_source, overwrite,
            )
            log.info("RadianceWrite: saved %d frame(s) → %s", count, saved)
            return ()

        except Exception as e:
            # Never silently swallow a delivery failure — surface it so the node
            # turns red in ComfyUI instead of reporting a phantom success.
            log.error("RadianceWrite failed: %s", e)
            raise

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
            }
            ext  = ext_map[stem]
            path = self._out_path(output_path, ext, overwrite)
            if "EXR" in stem:
                _save_exr(frames[0], path, half="16-bit" in stem)
            else:
                _save_pil_image(frames[0], path, stem, quality)
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
            half_exr = "16-bit" in stem

            if is_exr:
                ext = ".exr"
            elif is_tiff:
                ext = ".tiff"
            elif is_dpx:
                ext = ".dpx"
            else:
                ext = ".png"

            saved_paths = []
            for i, fr in enumerate(frames):
                fn   = f"{seq_stem}_{(pad_fmt % (start_frame + i))}{ext}"
                path = out_dir / fn
                if not overwrite:
                    path = _unique_path(path)
                if is_exr:
                    _save_exr(fr, path, half=half_exr)
                else:
                    _save_pil_image(fr, path, stem, quality)
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
        img, mask = reader.read(
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
                "output_path": ("STRING", {"default": ""}),
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
