"""
io/writer.py — the write engine, with no node layer above it.

This is `RadianceWrite`'s body, moved down a floor. Nothing here knows what a
ComfyUI node is, and nothing here imports one.

## Why it moved

`delivery/handler.py` used to do `from radiance.nodes.io.write import
RadianceWrite`, instantiate the node, and call `.write()` on it. The delivery
path reaching up into the node layer is backwards, and it had a concrete cost:
the handler could not be exercised without importing ComfyUI's node surface,
so the one thing that decides whether a master lands on disk correctly was the
one thing hardest to test.

The stated reason it could not move was that `RadianceWrite.write` is 118 lines
leaning on module-level helpers in the same 2972-line file. That was true, and
it is what this file is: the helpers came too. `nodes/io/write.py` imports them
straight back under their old private names, so `RadianceEXRMultiPart`,
`RadianceDigitalCinemaWrite` and the reader are untouched, and
`RadianceWrite.write` is now a signature and a delegation.

## The one thing that did not come down

`coerce_to_frames` accepts a file path or a VideoHelperSuite-style dict, and
resolving either means *reading* media -- several hundred lines of decoder that
belong to the reader, a floor up. Rather than drag the reader down or import it
upward, the reader is passed in: `read_media` is a callable the node layer
supplies and the delivery path does not, because the delivery path hands the
writer a tensor. Given a path and no reader, the error says so.

That is the whole of the layering. There is no lazy upward import hiding in a
function body here; `radiance.io.writer` depends on nothing above it, which is
checkable rather than promised — see tests/test_writer_layering.py.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from ..config.constants import VERSION as _RADIANCE_VERSION
except Exception:  # keep the writer importable even if constants move
    _RADIANCE_VERSION = "3.1.1"

try:
    from PIL import Image as _PIL
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# DPX has no Pillow plugin at all, in either direction. OpenImageIO is the
# VFX-standard library for it, and without it the DPX entries in the format
# menu are decorative.
try:
    import OpenImageIO as _oiio
    _HAS_OIIO = True
except ImportError:
    _HAS_OIIO = False

from .. import color_utils

try:
    from ..path_utils import strip_path_quotes
except ImportError:
    try:
        from path_utils import strip_path_quotes  # type: ignore[import]
    except ImportError:
        def strip_path_quotes(path):  # type: ignore[misc]
            return path.strip().strip('"').strip("'")

log = logging.getLogger("radiance.io.writer")


# ═══════════════════════════════════════════════════════════════════════════
# § 0  Formats and colour spaces
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

# ── EXR compressions ───────────────────────────────────────────────────────
EXR_COMPRESSIONS = ["ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed", "DWAA", "DWAB"]


# ═══════════════════════════════════════════════════════════════════════════
# § 1  ffmpeg and colour-space application
# ═══════════════════════════════════════════════════════════════════════════

def _ffmpeg_bin() -> str:
    """ffmpeg path, PATH first then the imageio-ffmpeg bundle."""
    from ..core.ffmpeg import require_ffmpeg
    return require_ffmpeg()


def _ffprobe_bin() -> str:
    """ffprobe path. imageio-ffmpeg does not ship ffprobe, so this can be
    absent even when ffmpeg is present; callers fall back to defaults."""
    from ..core.ffmpeg import ffprobe_exe
    return ffprobe_exe() or "ffprobe"


def _ffmpeg_ok() -> bool:
    # shutil.which alone missed the ffmpeg that imageio-ffmpeg ships, which
    # is the only one most Windows installs have.
    from ..core.ffmpeg import ffmpeg_available
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
        except ImportError as _exc:
            # AUDIT-FIX (2026-08): this used to log a warning and write 8-bit
            # anyway. A "16-bit" master that is actually 8-bit is a silent
            # quality downgrade of a production asset -- the same downgrade
            # this module's EXR writer refuses to do (see _save_exr).
            raise ImportError(
                f"Writing {fmt!r} requires tifffile (pip install tifffile). "
                "Refusing to silently downgrade a 16-bit write to 8-bit."
            ) from _exc
        tifffile.imwrite(str(path), arr_u16)
        return
    if "32-bit" in fmt:
        try:
            import tifffile  # type: ignore
        except ImportError as _exc:
            # AUDIT-FIX (2026-08): this used to swallow the ImportError at
            # DEBUG level and fall through to the 8-bit Pillow path below --
            # a float TIFF master silently written with 8-bit precision and
            # hard-clipped HDR values, discovered by the round-trip test only
            # because the test environment happened to lack tifffile.
            raise ImportError(
                f"Writing {fmt!r} requires tifffile (pip install tifffile). "
                "Refusing to silently downgrade a 32-bit float write to 8-bit."
            ) from _exc
        tifffile.imwrite(str(path), arr_f32.astype(np.float32))
        return

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
# § 3  The writer
# ═══════════════════════════════════════════════════════════════════════════

def coerce_to_frames(
    data: Any,
    *,
    read_media: Optional[Callable[[Any], Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, Optional[float]]:
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
                return coerce_to_frames(item, read_media=read_media)
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
                arr, _ = coerce_to_frames(data[key], read_media=read_media)
                return arr, detected_fps

        # LATENT dict — helpful error
        if "samples" in data and "batch_index" not in data:
            raise ValueError(
                "RadianceWrite received a LATENT tensor.  "
                "Decode it with a VAE Decode node first, then connect the IMAGE output."
            )

        # Search for embedded video file path (VHS-style). Reading media is
        # the reader's job, and the reader lives a layer up in the node module,
        # so it is handed in rather than imported. That is the whole reason
        # this function can live down here at all.
        if read_media is not None:
            arr = read_media(data)
            if arr is not None:
                return arr, detected_fps

        raise ValueError(
            f"RadianceWrite: dict input has no recognised IMAGE key or video path.  "
            f"Keys found: {list(data.keys())}"
        )

    # ── 4. String file path ───────────────────────────────────────────
    if isinstance(data, str):
        if read_media is not None:
            arr = read_media(data)
            if arr is not None:
                return arr, detected_fps
        raise ValueError(f"RadianceWrite: path not found or unsupported extension: {data!r}")

    raise ValueError(
        f"RadianceWrite: unsupported input type {type(data).__name__!r}.  "
        "Connect an IMAGE, batched IMAGE, video file path, or video node output."
    )


def resolve_output_path(base: str, ext: str, overwrite: bool) -> Path:
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


def write_frames(
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
    read_media:     Optional[Callable[[Any], Optional[np.ndarray]]] = None,
):
    output_path = strip_path_quotes(output_path)
    frames, detected_fps = coerce_to_frames(image, read_media=read_media)   # (N, H, W, C)
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
        saved, count = dispatch_write(
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


def dispatch_write(
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
        path = resolve_output_path(output_path, ext, overwrite)
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
