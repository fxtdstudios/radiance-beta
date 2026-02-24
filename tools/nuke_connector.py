"""
═══════════════════════════════════════════════════════════════════════════════
    Radiance Nuke Connector v3.0 — TCP Client for Nuke Command Server
                        Radiance © 2024-2026 FXTD STUDIOS

This is the NukeConnector class that nodes_nuke.py imports via:
    from .tools.nuke_connector import NukeConnector

Place this at: radiance/tools/nuke_connector.py

Protocol (file-based bridge, command-over-TCP):
  ComfyUI writes EXR to shared/temp path using Radiance IO
  → TCP sends Python command to Nuke
  → Nuke creates/updates Read node → Viewer displays the image

Why file-based, not pixel-over-TCP:
  - 4K RGBA float32 = 127 MB per frame — TCP would be 2+ seconds
  - EXR on disk = Nuke reads natively at memory-mapped speed
  - EXR preserves all channels (RGBA + Z + custom layers)
  - EXR metadata carries color space, frame range, compression
  - Same path works over NFS/SMB for multi-machine setups

Requires radiance_nuke_listener.py running inside Nuke.

v3.0 fixes:
  - NEW: load_exr() — creates Read node, sets frame range, connects Viewer
  - NEW: ping() — connection check before sending
  - NEW: set_frame() — remote frame control
  - NEW: get_info() — query Nuke version/project/format
  - FIX: sync_image() never existed — replaced with load_exr()
  - FIX: Proper binary protocol with magic + length header
  - FIX: Connection refused / timeout returns (False, msg) not exception
  - FIX: All methods return (success: bool, message: str) tuples
═══════════════════════════════════════════════════════════════════════════════
"""

import socket
import struct
import json
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("radiance.nuke.connector")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1986
BUFFER_SIZE = 65536
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 15.0

# Wire protocol
HEADER_MAGIC = b"RCMD"  # 4-byte magic identifier
HEADER_VERSION = 1  # Protocol version
END_MARKER = b"\n__RADIANCE_END__\n"  # Response terminator


class NukeConnector:
    """
    TCP client that sends Python commands to Nuke's Radiance listener.

    Every public method returns (success: bool, result: str|dict).
    No exceptions are raised to the caller — all errors are captured
    and returned as (False, "ERROR: ...") tuples.

    Usage:
        conn = NukeConnector("127.0.0.1", 1986)
        ok, msg = conn.ping()
        if ok:
            conn.load_exr("/renders/stream.0001.exr", "RadianceStream")
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self._last_error = ""

    @property
    def last_error(self) -> str:
        return self._last_error

    # ═════════════════════════════════════════════════════════════════════
    #  LOW-LEVEL: send raw Python command to Nuke
    # ═════════════════════════════════════════════════════════════════════

    def send_command(
        self, command: str, timeout: float = READ_TIMEOUT
    ) -> Tuple[bool, str]:
        """
        Send a Python command string to Nuke and return (success, result).

        The command is executed inside Nuke's script context via
        nuke.executeInMainThread() for thread safety.
        """
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CONNECT_TIMEOUT)
            sock.connect((self.host, self.port))
            sock.settimeout(timeout)

            # Wire frame: MAGIC(4) + VERSION(1) + LENGTH(4 LE) + PAYLOAD
            payload = command.encode("utf-8")
            header = HEADER_MAGIC + struct.pack("<BI", HEADER_VERSION, len(payload))
            sock.sendall(header + payload)

            # Read response chunks until END_MARKER or disconnect
            chunks = []
            accumulated = b""
            while True:
                try:
                    data = sock.recv(BUFFER_SIZE)
                    if not data:
                        break
                    chunks.append(data)
                    accumulated = b"".join(chunks)
                    if END_MARKER in accumulated:
                        break
                except socket.timeout:
                    break

            response = accumulated.decode("utf-8", errors="replace")
            response = response.replace(END_MARKER.decode(), "").strip()

            if response.startswith("ERROR:"):
                self._last_error = response
                return (False, response)

            return (True, response)

        except ConnectionRefusedError:
            msg = (
                f"Nuke not reachable at {self.host}:{self.port}. "
                "Ensure the Radiance listener is running in Nuke:\n"
                "  Script Editor → exec(open('radiance_nuke_listener.py').read())"
            )
            self._last_error = msg
            logger.warning(msg)
            return (False, f"CONNECTION_REFUSED: {msg}")

        except socket.timeout:
            msg = f"Timeout connecting to {self.host}:{self.port}"
            self._last_error = msg
            logger.warning(msg)
            return (False, f"TIMEOUT: {msg}")

        except OSError as e:
            msg = f"Network error: {e}"
            self._last_error = msg
            logger.warning(msg)
            return (False, f"OS_ERROR: {msg}")

        finally:
            if sock:
                try:
                    sock.close()
                except Exception:  # nosec B110
                    pass

    # ═════════════════════════════════════════════════════════════════════
    #  HIGH-LEVEL: Nuke operations
    # ═════════════════════════════════════════════════════════════════════

    def ping(self) -> Tuple[bool, str]:
        """Check if Nuke listener is alive. Fast 3-second timeout."""
        ok, result = self.send_command("'RADIANCE_PONG'", timeout=3.0)
        return (ok and "PONG" in result, result)

    def load_exr(
        self,
        filepath: str,
        node_name: str = "RadianceStream",
        first_frame: int = 1,
        last_frame: int = 1,
        current_frame: int = 1,
        color_space: str = "linear",
        connect_viewer: bool = True,
        raw: bool = True,
    ) -> Tuple[bool, str]:
        """
        Create or update a Read node in Nuke pointing to the EXR,
        and optionally connect it to the Viewer.

        This is the replacement for the broken sync_image() method.

        Args:
            filepath:       EXR path. Use #### for sequences: stream.####.exr
            node_name:      Nuke node name for the Read (reused across updates)
            first_frame:    First frame of sequence
            last_frame:     Last frame of sequence
            current_frame:  Frame to display in Viewer
            color_space:    Nuke colorspace knob value ('linear', 'sRGB', etc.)
            connect_viewer: If True, wire Read → Viewer1 input 0
            raw:            If True, bypass Nuke's internal color management
        """
        safe_path = filepath.replace("\\", "/")

        # Build the Nuke Python script
        cmd_lines = [
            "import nuke",
            "",
            "# Find existing Read or create new",
            "node = None",
            "for n in nuke.allNodes('Read'):",
            f"    if n.name() == '{node_name}':",
            "        node = n",
            "        break",
            "",
            "if node is None:",
            "    node = nuke.createNode('Read', inpanel=False)",
            f"    node.setName('{node_name}')",
            "",
            "# Set file path and frame range",
            f"node['file'].setValue('{safe_path}')",
            f"node['first'].setValue({first_frame})",
            f"node['last'].setValue({last_frame})",
            f"node['origfirst'].setValue({first_frame})",
            f"node['origlast'].setValue({last_frame})",
            f"node['raw'].setValue({raw})",
        ]

        if not raw:
            cmd_lines.append(f"node['colorspace'].setValue('{color_space}')")

        cmd_lines += [
            "",
            "# Force reload — clears Nuke's internal cache for this node",
            "node['reload'].execute()",
        ]

        if connect_viewer:
            cmd_lines += [
                "",
                "# Connect to Viewer",
                "viewer = None",
                "for n in nuke.allNodes('Viewer'):",
                "    viewer = n",
                "    break",
                "if viewer is None:",
                "    viewer = nuke.createNode('Viewer', inpanel=False)",
                "",
                "viewer.setInput(0, node)",
                f"nuke.frame({current_frame})",
            ]

        cmd_lines.append(
            f"\n'{node_name}: loaded {safe_path} [{first_frame}-{last_frame}]'"
        )

        return self.send_command("\n".join(cmd_lines))

    def set_frame(self, frame: int) -> Tuple[bool, str]:
        """Set Nuke's current frame."""
        return self.send_command(f"import nuke; nuke.frame({frame}); 'frame={frame}'")

    def get_info(self) -> Tuple[bool, Dict[str, Any]]:
        """Query Nuke for version, project, format, frame range, fps."""
        ok, result = self.send_command(
            "import nuke, json; json.dumps({"
            "'version': nuke.NUKE_VERSION_STRING, "
            "'project': nuke.root().name(), "
            "'fps': nuke.root()['fps'].value(), "
            "'first': int(nuke.root()['first_frame'].value()), "
            "'last': int(nuke.root()['last_frame'].value()), "
            "'format': str(nuke.root().format().name()), "
            "'width': nuke.root().format().width(), "
            "'height': nuke.root().format().height(), "
            "})"
        )
        if ok:
            try:
                return (True, json.loads(result))
            except (json.JSONDecodeError, TypeError):
                return (True, {"raw": result})
        return (False, {"error": result})
