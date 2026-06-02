import os
import json
import socket
import threading
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import numpy as np

from radiance.config.env import ENV, get_env_bool
from radiance.nodes_io import _save_exr, _save_video_ffmpeg, _load_video_to_numpy, _read_sequence

logger = logging.getLogger("radiance.mcp")

# ── Bridge Protocol ───────────────────────────────────────────────────────────
MCP_EOM = "\n__MCP_EOM__\n"

_SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "print": print, "len": len, "range": range,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple,
    "isinstance": isinstance, "hasattr": hasattr,
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "sorted": sorted, "reversed": reversed,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
}

_BLOCKED_PATTERNS = [
    "import os", "import sys", "import subprocess", "__import__",
    "os.system", "os.popen", "open(", "eval(", "exec(",
    "__builtins__", "__class__", "__subclasses__",
]

_SERVER: Optional[socket.socket] = None
_SERVER_THREAD: Optional[threading.Thread] = None
_SERVER_RUNNING = False


def _validate(code: str) -> Tuple[bool, str]:
    for p in _BLOCKED_PATTERNS:
        if p in code.lower():
            return False, f"blocked: {p}"
    return True, "ok"


def _dynamic_exec_enabled() -> bool:
    return get_env_bool(ENV.RADIANCE_DEV, False)


def _exec_sandbox(code: str) -> str:
    import ast
    g = {"__builtins__": _SAFE_BUILTINS, "json": json}
    try:
        v = ast.literal_eval(code)
        return json.dumps({"ok": True, "result": str(v)})
    except Exception:
        pass
    if not _dynamic_exec_enabled():
        return json.dumps({
            "ok": False,
            "error": "Dynamic bridge execution is disabled. Set RADIANCE_DEV=1 to enable local developer automation.",
        })
    try:
        v = eval(code, g)
        return json.dumps({"ok": True, "result": str(v)})
    except SyntaxError:
        pass
    try:
        exec(code, g)
        return json.dumps({"ok": True, "result": "ok"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def _handle(conn):
    try:
        conn.settimeout(15.0)
        buf = b""
        while True:
            c = conn.recv(1)
            if not c:
                return
            if c == b"\n":
                line = buf.decode("utf-8", errors="replace").strip()
                buf = b""
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    conn.sendall((json.dumps({"ok": False, "error": f"bad json: {e}"}) + "\n").encode())
                    continue
                cmd = msg.get("cmd", "")
                if cmd == "ping":
                    conn.sendall((json.dumps({"ok": True, "result": "pong"}) + "\n").encode())
                elif cmd == "status":
                    conn.sendall((json.dumps({"ok": True, "result": {"mode": "bridge", "running": True}}) + "\n").encode())
                elif cmd == "exec":
                    code = msg.get("code", "")
                    safe, reason = _validate(code)
                    if not safe:
                        conn.sendall((json.dumps({"ok": False, "error": reason}) + "\n").encode())
                    else:
                        result = _exec_sandbox(code)
                        conn.sendall((result + "\n").encode())
                elif cmd == "queue":
                    payload = msg.get("prompt", {})
                    try:
                        import urllib.request
                        from radiance.config.env import get_comfy_url
                        data = json.dumps({"prompt": payload, "client_id": "radiance_mcp"}).encode()
                        comfy_url = get_comfy_url().rstrip("/")
                        req = urllib.request.Request(
                            f"{comfy_url}/prompt",
                            data=data,
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            body = resp.read().decode()
                        conn.sendall((json.dumps({"ok": True, "result": body}) + "\n").encode())
                    except Exception as e:
                        conn.sendall((json.dumps({"ok": False, "error": str(e)}) + "\n").encode())
                else:
                    conn.sendall((json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"}) + "\n").encode())
            else:
                buf += c
    except Exception:
        pass
    finally:
        conn.close()


def start_server(port: int = None, host: str = None) -> str:
    global _SERVER, _SERVER_THREAD, _SERVER_RUNNING
    from radiance.config.env import get_mcp_port, get_mcp_host
    if port is None: port = get_mcp_port()
    if host is None: host = get_mcp_host()
    if _SERVER_THREAD and _SERVER_THREAD.is_alive():
        return f"Bridge already running on {host}:{port}"
    _SERVER_RUNNING = True

    def _run():
        global _SERVER
        _SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _SERVER.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _SERVER.settimeout(0.5)
        try:
            _SERVER.bind((host, port))
            _SERVER.listen(5)
            logger.info(f"MCP Bridge listening on {host}:{port}")
            while _SERVER_RUNNING:
                try:
                    conn, addr = _SERVER.accept()
                    logger.debug(f"MCP connection from {addr}")
                    threading.Thread(target=_handle, args=(conn,), daemon=True).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if _SERVER_RUNNING:
                        logger.warning(f"MCP accept: {e}")
        except OSError as e:
            logger.error(f"MCP socket: {e}")
        finally:
            try:
                _SERVER.close()
            except Exception:
                pass
            _SERVER = None

    _SERVER_THREAD = threading.Thread(target=_run, daemon=True)
    _SERVER_THREAD.start()
    return f"Bridge started on {host}:{port}"


def stop_server() -> str:
    global _SERVER_RUNNING, _SERVER_THREAD, _SERVER
    _SERVER_RUNNING = False
    if _SERVER:
        try:
            _SERVER.close()
        except Exception:
            pass
        _SERVER = None
    if _SERVER_THREAD:
        _SERVER_THREAD.join(timeout=2.0)
        _SERVER_THREAD = None
    return "Bridge stopped."


# ── Export helpers ───────────────────────────────────────────────────────────

_SOURCES = ["Auto", "Images", "Video", "Sequence"]
_DCC_TARGETS = ["Nuke", "Resolve", "Fusion"]
_EXR_FORMATS = [
    "EXR (16-bit half)",
    "EXR (32-bit float)",
    "EXR + H.264 MP4",
    "EXR + ProRes MOV",
]
_MCP_MODES = ["Export Frames", "Bridge Server"]


def _push_to_nuke(
    written: list,
    node_name: str,
    first_frame: int,
    last_frame: int,
    host: str = None,
    port: int = None,
) -> str:
    from radiance.config.env import get_nuke_host, get_nuke_port
    if host is None: host = get_nuke_host()
    if port is None: port = get_nuke_port()
    """Attempt to push the exported EXR to Nuke via the Radiance TCP listener."""
    try:
        from radiance.tools.nuke_connector import NukeConnector
    except ImportError:
        try:
            from radiance.nuke_connector import NukeConnector
        except ImportError:
            return "NukeConnector unavailable"

    if not written:
        return "no frames"

    filepath = written[0].replace("\\", "/")
    if len(written) > 1:
        import re
        filepath = re.sub(r"([_.])\d{4}\.(\w+)$", r"\1####.\2", filepath)

    conn = NukeConnector(host=host, port=port)
    ok, msg = conn.load_exr(
        filepath=filepath,
        node_name=node_name,
        first_frame=first_frame,
        last_frame=last_frame,
        current_frame=first_frame,
        color_space="linear",
        connect_viewer=True,
        raw=True,
    )
    if ok:
        return f"OK ({msg})"
    return f"FAILED ({msg})"


def _push_to_resolve(written: list, filename_prefix: str) -> str:
    """Report the Resolve handoff state for exported frames.

    Resolve's scripting API must run inside the Resolve process, so the MCP node
    can only prepare a folder handoff from ComfyUI.
    """
    if not written:
        return "no frames"
    return "folder handoff ready for manual Resolve import; no live Resolve API push"


class RadianceMCP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (_MCP_MODES, {
                    "default": "Export Frames",
                    "tooltip": "Export Frames = save EXR/video for DCC. Bridge Server = start TCP control server.",
                }),
                "source": (_SOURCES, {
                    "default": "Auto",
                    "tooltip": "Auto = try Images, then Video, then Sequence. Select explicitly to avoid ambiguity.",
                }),
                "target": (_DCC_TARGETS, {
                    "default": "Nuke",
                    "tooltip": "Target DCC application (metadata hint).",
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "placeholder": "/renders/shot_001/",
                    "tooltip": "Output directory for EXR frames (Export mode) or bridge log (Bridge mode).",
                }),
                "format": (_EXR_FORMATS, {
                    "default": "EXR (16-bit half)",
                    "tooltip": "EXR bit depth. +H.264 or +ProRes also generates a video file.",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of frames to export (used when source is Images or Auto).",
                }),
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a video file (.mp4, .mov, etc.) to decode and export (source=Video or Auto).",
                }),
                "sequence_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path/pattern to an image sequence e.g. /frames/frame.%04d.exr (source=Sequence or Auto).",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "Frame rate for video export."}),
                "frame_start": ("INT", {"default": 1001, "min": 0, "max": 999999,
                                        "tooltip": "Starting frame number for EXR sequence export."}),
                "frame_end": ("INT", {"default": 0, "min": 0, "max": 999999,
                                      "tooltip": "Last frame index (0 = read all found frames, for sequences only)."}),
                "filename_prefix": ("STRING", {"default": "frame",
                                               "tooltip": "Prefix for EXR filenames (e.g. frame_1001.exr)."}),
                "bridge_port": ("INT", {"default": 1987, "min": 1024, "max": 65535,
                                        "tooltip": "TCP port for Bridge Server (default 1987)."}),
                "bridge_host": ("STRING", {"default": "127.0.0.1",
                                           "tooltip": "Bind address (127.0.0.1 = loopback only; "
                                                      "0.0.0.0 = all interfaces)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "render_path")
    FUNCTION = "run"
    CATEGORY = "FXTD STUDIOS/Radiance/07 Pipeline & DCC"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "MCP Bridge — Export frames as EXR/video for DCC consumption, "
        "or start a TCP bridge server for command/control between ComfyUI and DCC apps."
    )

    def run(
        self,
        mode:             str   = "Export Frames",
        source:           str   = "Auto",
        target:           str   = "Nuke",
        output_path:      str   = "",
        format:           str   = "EXR (16-bit half)",
        images:           Optional[torch.Tensor] = None,
        video_path:       str   = "",
        sequence_path:    str   = "",
        fps:              float = 24.0,
        frame_start:      int   = 1001,
        frame_end:        int   = 0,
        filename_prefix:  str   = "frame",
        bridge_port:      int   = 1987,
        bridge_host:      str   = "127.0.0.1",
    ) -> Tuple[str, str]:
        from radiance.config.env import get_mcp_port, get_mcp_host
        if bridge_port == 1987:
            bridge_port = get_mcp_port()
        if bridge_host == "127.0.0.1":
            bridge_host = get_mcp_host()

        if mode == "Bridge Server":
            status = start_server(bridge_port, bridge_host)
            logger.info(f"[MCP] {status}")
            return (status, "")

        if not output_path:
            return ("Error: output_path is required in Export Frames mode.", "")

        # ── Resolve source ──────────────────────────────────────────────────
        source = source.lower()
        if source == "auto":
            if images is not None:
                source = "images"
            elif video_path:
                source = "video"
            elif sequence_path:
                source = "sequence"
            else:
                return ("Error: no input provided. Connect images, video_path, or sequence_path.", "")

        # ── Load frames from the selected source ────────────────────────────
        if source == "images":
            if images is None:
                return ("Error: images input is required when source=Images.", "")
            frames = images.detach().cpu().float().numpy()
            if frames.ndim == 3:
                frames = frames[np.newaxis]

        elif source == "video":
            if not video_path:
                return ("Error: video_path is required when source=Video.", "")
            if not os.path.isfile(video_path):
                return (f"Error: video file not found: {video_path}", "")
            try:
                frames = _load_video_to_numpy(video_path)
            except Exception as e:
                return (f"Error: failed to decode video: {e}", "")

        elif source == "sequence":
            if not sequence_path:
                return ("Error: sequence_path is required when source=Sequence.", "")
            end = frame_end if frame_end > 0 else 0
            try:
                batch, w, h, n, seq_fps, _ = _read_sequence(
                    sequence_path, frame_start, end, 1, "Linear (none)", "Skip"
                )
                frames = batch.detach().cpu().float().numpy()
                if frames.ndim == 3:
                    frames = frames[np.newaxis]
            except Exception as e:
                return (f"Error: failed to read sequence: {e}", "")

        else:
            return (f"Error: unknown source: {source}", "")

        # ── Export ──────────────────────────────────────────────────────────
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        half = "16-bit" in format
        make_video = "H.264" in format or "ProRes" in format

        written = []
        for i in range(len(frames)):
            frame_num = frame_start + i
            fname = f"{filename_prefix}_{frame_num:04d}.exr"
            fpath = out_dir / fname
            _save_exr(frames[i], fpath, half=half)
            written.append(str(fpath))

        status = f"{len(written)} EXR frames → {out_dir}"
        render_path = str(out_dir)

        if make_video:
            video_fmt = "MOV (ProRes 422 HQ)" if "ProRes" in format else "MP4 (H.264)"
            video_path = str(out_dir / f"{filename_prefix}_preview")
            try:
                _save_video_ffmpeg(frames, video_path, video_fmt, fps, 18, "")
                ext_map = {"MOV (ProRes 422 HQ)": ".mov", "MP4 (H.264)": ".mp4"}
                ext = ext_map.get(video_fmt, ".mp4")
                final_video = video_path + ext
                if os.path.isfile(final_video):
                    status += f" + video → {final_video}"
                    render_path = final_video
            except Exception as e:
                logger.warning("Video export failed: %s", e)
                status += f" (video failed: {e})"

        # ── Auto-push to DCC ────────────────────────────────────────────────
        if target.lower() == "nuke" and written:
            push_status = _push_to_nuke(
                written, filename_prefix, frame_start, frame_start + len(frames) - 1
            )
            if push_status:
                status += f" | nuke: {push_status}"

        elif target.lower() == "resolve" and written:
            push_status = _push_to_resolve(written, filename_prefix)
            if push_status:
                status += f" | resolve: {push_status}"

        return (status, render_path)


NODE_CLASS_MAPPINGS = {
    "RadianceMCP": RadianceMCP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMCP": "◎ Radiance MCP Bridge",
}
