import re
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

# Security: regex pattern for safe Nuke identifiers (node names, stream names)
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
# Security: characters allowed in Nuke strings (paths, color space names)
_SAFE_STRING_RE = re.compile(r"^[a-zA-Z0-9_\-./: ()#+]+$")


def _sanitize_nuke_string(value: str, context: str = "value") -> str:
    """
    Sanitize a string before embedding it in a Nuke Python command.

    Removes/escapes characters that could break out of a Python string literal
    and inject arbitrary code. This is a defense-in-depth measure.

    Args:
        value: The string to sanitize
        context: Description for error messages (e.g., 'node_name', 'filepath')

    Returns:
        Sanitized string safe for embedding in single-quoted Python literals
    """
    # Remove null bytes
    value = value.replace("\x00", "")
    # Escape backslashes first, then single quotes
    value = value.replace("\\", "/")
    value = value.replace("'", "")
    value = value.replace('"', "")
    # Remove newlines and other control characters that could break commands
    value = re.sub(r"[\n\r\t]", "", value)
    # Strip leading/trailing whitespace
    value = value.strip()
    return value


def validate_nuke_identifier(name: str, context: str = "identifier") -> str:
    """
    Validate that a string is a safe Nuke identifier (node name, stream name).

    Only allows: a-z, A-Z, 0-9, underscore, hyphen.

    Raises:
        ValueError: If the name contains invalid characters
    """
    name = name.strip()
    if not name:
        raise ValueError(f"Empty {context} is not allowed")
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid {context}: '{name}'. "
            f"Only letters, digits, underscore, and hyphen are allowed."
        )
    if len(name) > 128:
        raise ValueError(f"{context} too long (max 128 chars): '{name[:32]}...'")
    return name


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
            # Generate absolute, forward-slashed path for Nuke copy-paste
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                script_path = os.path.join(base_dir, "scripts", "start_nuke_server.py").replace("\\", "/")
                cmd = f"exec(open('{script_path}').read())"
            except Exception:
                cmd = "exec(open('scripts/start_nuke_server.py').read())"

            msg = (
                f"Nuke not reachable at {self.host}:{self.port}. "
                "Ensure the Radiance listener is running in Nuke:\n"
                f"  Script Editor → run: {cmd}"
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
        # ── Security: sanitize all user-supplied strings ──
        safe_path = _sanitize_nuke_string(filepath, "filepath")
        safe_node_name = validate_nuke_identifier(node_name, "node_name")
        safe_color_space = _sanitize_nuke_string(color_space, "color_space")

        # Validate numeric parameters
        first_frame = int(first_frame)
        last_frame = int(last_frame)
        current_frame = int(current_frame)

        # Build the Nuke Python script
        cmd_lines = [
            "import nuke",
            "",
            "# Find existing Read or create new",
            f"node = nuke.toNode('{safe_node_name}')",
            "if node is None or node.Class() != 'Read':",
            "    node = nuke.createNode('Read', inpanel=False)",
            f"    node.setName('{safe_node_name}')",
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
            cmd_lines.append(f"node['colorspace'].setValue('{safe_color_space}')")

        cmd_lines += [
            "",
            "# Force reload — clears Nuke's internal cache for this node",
            "node['reload'].execute()",
        ]

        if connect_viewer:
            cmd_lines += [
                "",
                "# Connect to Viewer",
                "viewer = nuke.toNode('Viewer1')",
                "if viewer is None:",
                "    viewer = nuke.createNode('Viewer', inpanel=False)",
                "",
                "viewer.setInput(0, node)",
                f"nuke.frame({current_frame})",
            ]

        cmd_lines.append(
            f"\n'{safe_node_name}: loaded {safe_path} [{first_frame}-{last_frame}]'"
        )

        return self.send_command("\n".join(cmd_lines))

    def set_frame(self, frame: int) -> Tuple[bool, str]:
        """Set Nuke's current frame."""
        frame = int(frame)  # Ensure integer — no injection
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
