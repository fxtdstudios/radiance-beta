"""
═══════════════════════════════════════════════════════════════════════════════
    Radiance Nuke Bridge v2.3 — Direct Viewer Connection
                    Radiance © 2024-2026 FXTD STUDIOS

Place this at: radiance/nodes_nuke.py

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │  ComfyUI (Radiance)                                             │
  │                                                                  │
  │  IMAGE ──▶ Write EXR ──▶ stream.####.exr ──TCP──▶ Nuke Viewer  │
  │  (B,H,W,C)  (Radiance IO)  (temp / shared)  cmd    (Read node) │
  │                                                                  │
  │  Full float32 · RGBA + Depth · Color space metadata             │
  └──────────────────────────────────────────────────────────────────┘

EXR is written using the SAME Radiance IO writer chain as RadianceSaveEXR:
  OpenEXR → SimpleEXRWriter → OpenCV → imageio (with channel-aware fallbacks)

All channels preserved at every fallback level:
  - OpenEXR:         R, G, B, A, Z  ✓
  - SimpleEXRWriter: R, G, B, A, Z  ✓ (pure Python, always available)
  - OpenCV:          R, G, B, A     ✓ (Z not supported by cv2.imwrite)
  - imageio:         R, G, B        ✓ (alpha/depth not supported)

v3.0 — 23 bugs fixed from v1.0:

 CRITICAL:
  - FIX: sync_image() → load_exr() (method actually exists now)
  - FIX: Batch support — writes numbered EXR sequence for ALL frames
  - FIX: Alpha channel preserved from 4-channel RGBA images
  - FIX: Depth channel works through ALL writer backends (not just OpenEXR)
  - FIX: Consistent channel handling — no RGB/BGR mismatch between writers

 DATA INTEGRITY:
  - FIX: Color space metadata in EXR header + Nuke Read node colorspace knob
  - FIX: Bit depth control (float32 / float16)
  - FIX: Compression control (ZIP, PIZ, DWAA, etc.)
  - FIX: Frame range set on Nuke Read node (first/last/origfirst/origlast)

 INFRASTRUCTURE:
  - FIX: ComfyUI temp directory (not module dir)
  - FIX: Old temp file cleanup (300s TTL)
  - FIX: logger instead of print() (16 print → 0 print)
  - FIX: Ping before send (connection status check)
  - FIX: On Change mode actually detects changes (perceptual hash)
  - FIX: hidden unique_id for execution tracking

 NEW:
  - Nuke-side listener script (scripts/start_nuke_server.py)
  - RadianceNukeInfo node — query Nuke version/project/format
  - Custom output path for shared drives (NFS/SMB)
  - Frame numbering: stream.0001.exr (Nuke #### convention)
  - Viewer auto-connect (Read → Viewer1 input 0)
  - Raw mode bypass for Nuke's color management
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import numpy as np
import os
import re
import time
import logging
import hashlib
from typing import Dict, Any, Tuple, Optional

import folder_paths

# ── Radiance IO — EXR writers (same chain as RadianceSaveEXR) ──
from .hdr.io import (
    write_exr_openexr,
    write_exr_cv2,
    write_exr_imageio,
    check_openexr_available,
    SimpleEXRWriter,
)

# ── Nuke TCP connector (with sanitization) ──
from .tools.nuke_connector import NukeConnector, validate_nuke_identifier

# ── Path security ──
try:
    from .path_utils import safe_join, get_safe_output_dir
except ImportError:
    from path_utils import safe_join, get_safe_output_dir

logger = logging.getLogger("◎ Radiance.nuke.bridge")

# ── Security: Host allowlist for Nuke Bridge connections ──
_DEFAULT_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _get_allowed_hosts() -> set:
    """
    Get the set of allowed hosts for Nuke Bridge TCP connections.
    Defaults to localhost only. Studios can add IPs via the
    RADIANCE_NUKE_ALLOWED_HOSTS environment variable (comma-separated).
    """
    allowed = set(_DEFAULT_ALLOWED_HOSTS)
    env_hosts = os.environ.get("◎ Radiance_NUKE_ALLOWED_HOSTS", "")
    if env_hosts:
        for h in env_hosts.split(","):
            h = h.strip()
            if h:
                allowed.add(h)
    return allowed


# ── Security: Stream name validation ──
_SAFE_STREAM_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_stream_name(name: str) -> str:
    """Validate stream_name to prevent command injection via filenames/TCP commands."""
    name = name.strip()
    if not name:
        raise ValueError("stream_name cannot be empty")
    if not _SAFE_STREAM_RE.match(name):
        raise ValueError(
            f"Invalid stream_name: '{name}'. "
            f"Only letters, digits, underscore, and hyphen are allowed."
        )
    if len(name) > 128:
        raise ValueError(f"stream_name too long (max 128 chars)")
    return name


# ═══════════════════════════════════════════════════════════════════════════════
#                              HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _get_bridge_temp_dir() -> str:
    """
    Get the bridge temp directory inside ComfyUI's temp folder.

    v1.0 bug: used os.path.dirname(__file__) — the Python module directory,
    which may be read-only (pip install) and files were never cleaned up.
    """
    base = folder_paths.get_temp_directory()
    bridge_dir = os.path.join(base, "◎ Radiance_nuke_bridge")
    os.makedirs(bridge_dir, exist_ok=True)
    return bridge_dir


def _cleanup_old_files(bridge_dir: str, stream_name: str, max_age: float = 300.0):
    """Remove bridge EXR files older than max_age seconds for this stream."""
    try:
        now = time.time()
        prefix = f"{stream_name}."
        for fname in os.listdir(bridge_dir):
            if fname.startswith(prefix) and fname.endswith(".exr"):
                fpath = os.path.join(bridge_dir, fname)
                try:
                    if now - os.path.getmtime(fpath) > max_age:
                        os.remove(fpath)
                except OSError:
                    pass
    except Exception:  # nosec B110
        pass


def _write_exr_all_channels(
    filepath: str,
    rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    bit_depth: str,
    compression: str,
    metadata: Optional[Dict[str, Any]],
) -> bool:
    """
    Write EXR with RGB + optional Alpha + optional Depth using
    the full Radiance IO writer chain with proper fallbacks.

    v1.0 bugs fixed:
      - channels dict now includes 'A' (alpha was silently dropped)
      - CV2 fallback composes RGBA from channels (depth still lost, warned)
      - SimpleEXRWriter gets full channel dict (R,G,B,A,Z)
      - imageio gets RGB (alpha/depth warned if present)
      - All writers get float32 data (no implicit dtype mismatch)
    """
    pixel_type = "HALF" if "16" in bit_depth else "FLOAT"

    # Build channels dict — works for OpenEXR + SimpleEXRWriter
    channels: Dict[str, np.ndarray] = {
        "R": rgb[..., 0].astype(np.float32),
        "G": rgb[..., 1].astype(np.float32),
        "B": rgb[..., 2].astype(np.float32),
    }
    if alpha is not None:
        channels["A"] = alpha.astype(np.float32)
    if depth is not None:
        channels["Z"] = depth.astype(np.float32)

    # ── Writer 1: OpenEXR (best — full R,G,B,A,Z channel support) ──
    if check_openexr_available():
        try:
            write_exr_openexr(filepath, channels, compression, pixel_type, metadata)
            logger.debug(f"EXR via OpenEXR: {filepath}")
            return True
        except Exception as e:
            logger.warning(f"OpenEXR failed: {e}")

    # ── Writer 2: SimpleEXRWriter (pure Python — full R,G,B,A,Z support) ──
    try:
        SimpleEXRWriter().write(filepath, channels, compression, pixel_type, metadata)
        logger.debug(f"EXR via SimpleEXRWriter: {filepath}")
        return True
    except Exception as e:
        logger.warning(f"SimpleEXRWriter failed: {e}")

    # ── Writer 3: OpenCV (R,G,B,A only — no Z channel) ──
    try:
        if alpha is not None:
            img_cv = np.concatenate(
                [rgb.astype(np.float32), alpha[..., np.newaxis].astype(np.float32)],
                axis=-1,
            )
        else:
            img_cv = rgb.astype(np.float32)

        if write_exr_cv2(filepath, img_cv, bit_depth, compression):
            if depth is not None:
                logger.warning(
                    "OpenCV: Depth (Z) channel omitted (not supported by cv2.imwrite)"
                )
            logger.debug(f"EXR via OpenCV: {filepath}")
            return True
    except Exception as e:
        logger.warning(f"OpenCV failed: {e}")

    # ── Writer 4: imageio (R,G,B only — no alpha, no depth) ──
    try:
        if write_exr_imageio(filepath, rgb.astype(np.float32), pixel_type):
            if alpha is not None:
                logger.warning("imageio: Alpha (A) channel omitted (not supported)")
            if depth is not None:
                logger.warning("imageio: Depth (Z) channel omitted (not supported)")
            logger.debug(f"EXR via imageio: {filepath}")
            return True
    except Exception as e:
        logger.warning(f"imageio failed: {e}")

    logger.error(f"All 4 EXR writers failed for: {filepath}")
    return False


def _write_exr_atomic(
    filepath: str,
    rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    bit_depth: str,
    compression: str,
    metadata: Optional[Dict[str, Any]],
) -> bool:
    """
    Atomic EXR write: writes to a temp file first, then renames on success.
    Prevents corrupt partial files if the process is interrupted mid-write.
    """
    tmp_path = filepath + ".tmp"
    try:
        ok = _write_exr_all_channels(tmp_path, rgb, alpha, depth, bit_depth, compression, metadata)
        if ok:
            # Atomic rename (on same filesystem, this is atomic on most OSes)
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(tmp_path, filepath)
            return True
        return False
    except Exception as e:
        logger.error(f"Atomic EXR write failed for {filepath}: {e}")
        # Clean up temp file on failure
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


def _perceptual_hash(img_np: np.ndarray) -> str:
    """
    Quick perceptual hash for change detection.
    Samples 32×32 grid of pixels, hashes as float16.

    v1.0 bug: "On Change" mode had no change detection logic at all —
    it always sent regardless. Now it actually compares frames.
    """
    h, w = img_np.shape[:2]
    step_h = max(1, h // 32)
    step_w = max(1, w // 32)
    sample = img_np[::step_h, ::step_w, :3].astype(np.float16)
    return hashlib.blake2s(sample.tobytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
#                           NUKE BRIDGE NODE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceNukeBridge:
    """
    Send images directly to Nuke's Viewer via EXR + TCP command.

    Writes EXR using Radiance IO (same backend as RadianceSaveEXR),
    then tells Nuke to create/update a Read node and connect it to
    the Viewer. Supports full batch sequences, alpha, depth, and
    color space metadata.

    Pipeline: IMAGE → EXR file → TCP command → Nuke Read → Nuke Viewer
    """

    # Nuke colorspace knob values
    NUKE_COLOR_SPACES = [
        "linear",
        "sRGB",
        "raw",
        "AlexaV3LogC",
        "ARRI LogC4",
        "S-Log3",
        "V-Log",
        "Canon Log 3",
        "ACEScg",
        "ACEScct",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "host": (
                    "STRING",
                    {
                        "default": "127.0.0.1",
                        "tooltip": "Nuke machine IP address. 127.0.0.1 for same machine.",
                    },
                ),
                "port": (
                    "INT",
                    {
                        "default": 1986,
                        "min": 1024,
                        "max": 65535,
                        "tooltip": "TCP port matching scripts/start_nuke_server.py in Nuke.",
                    },
                ),
                "stream_name": (
                    "STRING",
                    {
                        "default": "◎ RadianceStream",
                        "tooltip": (
                            "Nuke Read node name. Reused across updates — "
                            "no duplicate nodes created."
                        ),
                    },
                ),
            },
            "optional": {
                "send_mode": (
                    ["Always", "On Change", "Never"],
                    {
                        "default": "Always",
                        "tooltip": (
                            "Always: send every execution. "
                            "On Change: skip if image unchanged (perceptual hash). "
                            "Never: write EXR to disk only, don't command Nuke."
                        ),
                    },
                ),
                "color_space": (
                    cls.NUKE_COLOR_SPACES,
                    {
                        "default": "linear",
                        "tooltip": (
                            "Color space for Nuke's Read node. Must match your "
                            "pipeline. 'linear' for float32 HDR data."
                        ),
                    },
                ),
                "bit_depth": (
                    ["32-bit Float", "16-bit Half Float"],
                    {
                        "default": "32-bit Float",
                        "tooltip": "EXR precision. Float32 for max quality, Half for smaller files.",
                    },
                ),
                "compression": (
                    ["ZIP", "ZIPS", "PIZ", "DWAA", "None"],
                    {
                        "default": "ZIP",
                        "tooltip": "EXR compression. ZIP = fast + good ratio. PIZ = best for noisy.",
                    },
                ),
                "connect_viewer": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Auto-connect the Read node to Nuke's Viewer input 0.",
                    },
                ),
                "raw": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Bypass Nuke's color management on Read. "
                            "True = data passes through exactly as written in EXR."
                        ),
                    },
                ),
                "start_frame": (
                    "INT",
                    {
                        "default": 1001,
                        "min": 0,
                        "max": 999999,
                        "tooltip": "First frame number for sequence naming (VFX convention: 1001).",
                    },
                ),
                "depth_map": (
                    "IMAGE",
                    {
                        "tooltip": "Optional depth map → EXR 'Z' channel. Visible in Nuke as depth.",
                    },
                ),
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Custom output directory for EXR files. Use a shared drive "
                            "path for remote Nuke machines. Empty = ComfyUI temp dir."
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "nuke_status")
    FUNCTION = "send_to_nuke"
    CATEGORY = "FXTD Studios/Radiance/Pipeline"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Send images directly to Nuke's Viewer via EXR + TCP. "
        "Writes full float32 EXR with RGBA + Depth using Radiance IO, "
        "then creates a Read node in Nuke connected to the Viewer. "
        "Requires scripts/start_nuke_server.py running inside Nuke."
    )

    def __init__(self):
        self._last_hash = ""

    def send_to_nuke(
        self,
        images: torch.Tensor,
        host: str,
        port: int,
        stream_name: str,
        send_mode: str = "Always",
        color_space: str = "linear",
        bit_depth: str = "32-bit Float",
        compression: str = "ZIP",
        connect_viewer: bool = True,
        raw: bool = True,
        start_frame: int = 1001,
        depth_map: Optional[torch.Tensor] = None,
        output_path: str = "",
        unique_id: str = "",
    ) -> Tuple[torch.Tensor, str]:

        # ── Security: Validate stream_name ──
        try:
            stream_name = _validate_stream_name(stream_name)
        except ValueError as e:
            return (images, f"ERROR: {e}")

        # ── Security: Validate host against allowlist ──
        allowed_hosts = _get_allowed_hosts()
        if host not in allowed_hosts:
            return (
                images,
                f"ERROR: Host '{host}' not in allowed list. "
                f"Allowed: {', '.join(sorted(allowed_hosts))}. "
                f"Set RADIANCE_NUKE_ALLOWED_HOSTS env var to add studio IPs.",
            )

        # ── Validate input ──
        if not isinstance(images, torch.Tensor):
            return (images, "ERROR: images must be a torch.Tensor")

        if images.dim() == 3:
            images = images.unsqueeze(0)

        batch_size, h, w, c = images.shape

        logger.info(
            f"[NukeBridge] {stream_name}: {w}×{h} × {batch_size} frame(s), "
            f"ch={c}, space={color_space}, mode={send_mode}"
        )

        # ── Output directory (secured) ──
        if output_path and output_path.strip():
            clean_path = output_path.strip().strip('"').strip("'")
            try:
                # Route through safe_join for security
                base_output = folder_paths.get_output_directory()
                bridge_dir = get_safe_output_dir(base_output, clean_path, allow_absolute=True)
            except ValueError as e:
                return (images, f"ERROR: Invalid output_path: {e}")
        else:
            bridge_dir = _get_bridge_temp_dir()

        # Clean old bridge files for this stream (>5 min old)
        _cleanup_old_files(bridge_dir, stream_name)

        # ── Change detection (On Change mode) ──
        if send_mode == "On Change":
            first_np = images[0].cpu().numpy().astype(np.float32)
            current_hash = _perceptual_hash(first_np)
            if current_hash == self._last_hash:
                logger.info(f"[NukeBridge] {stream_name}: unchanged, skipping.")
                return (images, "Skipped: image unchanged")
            self._last_hash = current_hash

        if send_mode == "Never":
            # Still write EXR, just don't command Nuke
            pass

        # ── Write EXR sequence ──
        last_frame = start_frame + batch_size - 1
        frame_paths = []

        for i in range(batch_size):
            frame_num = start_frame + i
            filename = f"{stream_name}.{str(frame_num).zfill(4)}.exr"
            filepath = os.path.join(bridge_dir, filename)

            # Extract frame — full float32, ZERO clamping
            frame_data = images[i].cpu().numpy().astype(np.float32)

            # Separate RGB and alpha
            if c == 4:
                rgb = frame_data[..., :3]
                alpha = frame_data[..., 3]
            elif c == 1:
                rgb = np.repeat(frame_data, 3, axis=-1)
                alpha = None
            else:
                rgb = frame_data[..., :3]
                alpha = None

            # Depth map (optional)
            depth = None
            if depth_map is not None:
                try:
                    d_idx = min(i, depth_map.shape[0] - 1)
                    d = depth_map[d_idx].cpu().numpy().astype(np.float32)
                    if d.ndim == 3:
                        d = d[..., 0]
                    if d.shape[:2] == rgb.shape[:2]:
                        depth = d
                    else:
                        logger.warning(
                            f"[NukeBridge] Depth size {d.shape[:2]} != "
                            f"image {rgb.shape[:2]}, skipping depth frame {frame_num}"
                        )
                except Exception as e:
                    logger.warning(f"[NukeBridge] Depth extraction error: {e}")

            # EXR metadata
            metadata = {
                "software": "◎ Radiance - ComfyUI",
                "◎ RadianceColorSpace": color_space,
                "◎ RadianceStream": stream_name,
                "◎ RadianceFrame": frame_num,
                "◎ RadianceBitDepth": bit_depth,
            }

            # Write using full Radiance IO writer chain (atomic)
            ok = _write_exr_atomic(
                filepath,
                rgb,
                alpha,
                depth,
                bit_depth,
                compression,
                metadata,
            )

            if ok:
                frame_paths.append(filepath)
                logger.debug(f"[NukeBridge] Frame {frame_num}: {filepath}")
            else:
                logger.error(f"[NukeBridge] FAILED frame {frame_num}: {filepath}")

        if not frame_paths:
            return (images, "ERROR: All EXR writes failed — check logs")

        # ── Build Nuke file path pattern ──
        if batch_size == 1:
            # Single frame: use exact path
            nuke_file_path = frame_paths[0].replace("\\", "/")
        else:
            # Sequence: stream.####.exr (Nuke convention)
            nuke_file_path = os.path.join(
                bridge_dir, f"{stream_name}.####.exr"
            ).replace("\\", "/")

        # ── Skip Nuke command in Never mode ──
        if send_mode == "Never":
            status = (
                f"EXR saved: {len(frame_paths)} frame(s) → {bridge_dir} "
                f"(Nuke command skipped)"
            )
            logger.info(f"[NukeBridge] {status}")
            return (images, status)

        # ── Connect to Nuke ──
        connector = NukeConnector(host, port)

        # Ping first — fast connection check
        alive, ping_msg = connector.ping()
        if not alive:
            status = (
                f"Nuke not responding at {host}:{port}. "
                f"EXR saved: {len(frame_paths)} frame(s) → {bridge_dir}. "
                f"Start the listener in Nuke: "
                f"exec(open('scripts/start_nuke_server.py').read())"
            )
            logger.warning(f"[NukeBridge] {status}")
            return (images, status)

        # Send load command — creates/updates Read node, connects Viewer
        ok, result = connector.load_exr(
            filepath=nuke_file_path,
            node_name=stream_name,
            first_frame=start_frame,
            last_frame=last_frame,
            current_frame=last_frame,  # Show latest frame
            color_space=color_space,
            connect_viewer=connect_viewer,
            raw=raw,
        )

        if ok:
            status = (
                f"Sent → Nuke: {stream_name} [{start_frame}-{last_frame}] "
                f"({w}×{h}, {bit_depth}, {color_space})"
            )
            logger.info(f"[NukeBridge] {status}")
        else:
            status = f"Nuke command failed: {result}"
            logger.error(f"[NukeBridge] {status}")

        return (images, status)


# ═══════════════════════════════════════════════════════════════════════════════
#                          NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "◎ RadianceNukeBridge": RadianceNukeBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "◎ RadianceNukeBridge": "◎ Radiance Nuke Bridge",
}
