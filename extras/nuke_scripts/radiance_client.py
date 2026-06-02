"""
nuke_scripts/radiance_client.py
================================
Radiance ← Nuke client — companion to the RadianceNukeServer ComfyUI node.

Usage inside Nuke (Script Editor or menu.py):

    import sys
    sys.path.insert(0, "/path/to/radiance/nuke_scripts")
    from radiance_client import RadianceClient

    r = RadianceClient("localhost", 7863)
    r.ping()                     # → {"status": "ok", "version": "3.0.1"}
    r.status()                   # → session info dict
    img = r.get_frame()          # → Nuke node with frame from ComfyUI
    r.colorize("MYNODE", "logc4", "acescg")   # decode log → ACEScg in-place

Quick Panel
-----------
    # Paste in Nuke Script Editor to open a minimal control panel:
    from radiance_client import open_panel
    open_panel()

Requirements
------------
    Nuke 14+ (uses standard library only — no extra pip installs needed)
    Python 3.8+ inside Nuke
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
import urllib.request
import urllib.error
from io import BytesIO
from typing import Any, Dict, Optional


class RadianceClient:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Client for the RadianceNukeServer HTTP endpoint.

    Parameters
    ----------
    host    Hostname running ComfyUI + RadianceNukeServer (default: "localhost")
    port    Port configured in the RadianceNukeServer node (default: 7863)
    timeout HTTP timeout in seconds (default: 10)
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        timeout: int = 10,
        token: str = None,
    ) -> None:
        self.host    = host if host is not None else os.environ.get("RADIANCE_HTTP_HOST", "localhost")
        self.port    = port if port is not None else int(os.environ.get("RADIANCE_HTTP_PORT", "7863"))
        self.timeout = timeout
        self.token   = token if token is not None else os.environ.get("RADIANCE_DCC_AUTH_TOKEN", "")
        self.base    = f"http://{self.host}:{self.port}"

    # ── low-level HTTP helpers ────────────────────────────────────────────────

    def _get(self, path: str) -> Dict[str, Any]:
        url = self.base + path
        headers = {}
        if self.token:
            headers["X-Radiance-Auth"] = self.token
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Radiance server unreachable at {url}: {e}") from e

    def _post(self, path: str, payload: dict) -> Dict[str, Any]:
        url  = self.base + path
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept":       "application/json"
        }
        if self.token:
            headers["X-Radiance-Auth"] = self.token
        req  = urllib.request.Request(
            url, data=body,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Radiance server unreachable at {url}: {e}") from e

    # ── API ───────────────────────────────────────────────────────────────────

    def ping(self) -> Dict[str, Any]:
        """
        Health check.

        Returns
        -------
        dict  {"status": "ok", "version": "3.0.1"}

        Raises
        ------
        ConnectionError  if the server is not reachable
        """
        return self._get("/ping")

    def status(self) -> Dict[str, Any]:
        """
        Query session state.

        Returns
        -------
        dict with keys:
          status        "running"
          version       "3.0.1"
          frame_ready   bool — True if a frame is cached
          frame_width   int
          frame_height  int
          frame_ts      float — Unix timestamp of the last cached frame
          port          int
        """
        return self._get("/status")

    def get_frame(
        self,
        node_name: str = "Radiance_Frame",
        create_read_node: bool = True,
    ) -> Optional[Any]:
        """
        Fetch the last ComfyUI output frame and optionally create a Nuke Read node.

        Parameters
        ----------
        node_name         Name for the created Nuke Read node
        create_read_node  If True (and running inside Nuke), create a Read node.
                          If False, return the raw bytes.

        Returns
        -------
        nuke.Node   if create_read_node is True and nuke is available
        bytes       raw PNG bytes otherwise
        """
        resp = self._get("/get_frame")
        if resp.get("status") != "ok":
            raise RuntimeError(f"get_frame failed: {resp.get('message', resp)}")

        png_bytes = base64.b64decode(resp["image_b64"])

        if create_read_node:
            try:
                import nuke
                tmp = os.path.join(tempfile.gettempdir(), f"{node_name}.png")
                with open(tmp, "wb") as f:
                    f.write(png_bytes)
                node = nuke.createNode("Read")
                node["file"].setValue(tmp)
                node["name"].setValue(node_name)
                node["label"].setValue(
                    f"Radiance frame\n{resp['width']}×{resp['height']}\n"
                    + time.strftime("%H:%M:%S", time.localtime(resp.get("timestamp", 0)))
                )
                print(f"[Radiance] Created Read node '{node_name}' from "
                      f"{resp['width']}×{resp['height']} frame")
                return node
            except ImportError:
                pass  # Not inside Nuke — fall through

        return png_bytes

    def colorize(
        self,
        node_or_b64: Any,
        encode: str = "logc4",
        decode: str = "acescg",
        node_name: str = "Radiance_Colorized",
    ) -> Any:
        """
        Apply a log decode + color-space re-encode to an image.

        Parameters
        ----------
        node_or_b64   Nuke node, file path (str), or raw base64 PNG string
        encode        Source log/camera format: "logc4", "logc3", "slog3", "vlog", "srgb", "linear"
        decode        Target color space:       "acescg", "srgb", "linear"
        node_name     Name for the resulting Nuke Read node (if inside Nuke)

        Returns
        -------
        nuke.Node     if inside Nuke
        bytes         raw PNG bytes otherwise
        """
        b64 = self._to_b64(node_or_b64)
        resp = self._post("/colorize", {
            "image_b64": b64,
            "encode":    encode,
            "decode":    decode,
        })
        if resp.get("status") != "ok":
            raise RuntimeError(f"colorize failed: {resp.get('message', resp)}")

        png_bytes = base64.b64decode(resp["image_b64"])

        try:
            import nuke
            tmp = os.path.join(tempfile.gettempdir(), f"{node_name}.png")
            with open(tmp, "wb") as f:
                f.write(png_bytes)
            node = nuke.createNode("Read")
            node["file"].setValue(tmp)
            node["name"].setValue(node_name)
            node["label"].setValue(f"Radiance: {encode} → {decode}")
            print(f"[Radiance] Colorized node created: {encode} → {decode}")
            return node
        except ImportError:
            return png_bytes

    def run_pipeline(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
    ) -> Dict[str, Any]:
        """
        Request ComfyUI to run a generation pipeline.

        The request is queued asynchronously; poll get_frame() after
        the ComfyUI workflow has executed to retrieve the result.

        Parameters
        ----------
        prompt  Text prompt for generation
        width   Output width in pixels
        height  Output height in pixels
        seed    RNG seed (-1 for random)

        Returns
        -------
        dict  {"status": "queued", "job_id": "...", "prompt": ..., ...}
        """
        return self._post("/run_pipeline", {
            "prompt": prompt,
            "width":  width,
            "height": height,
            "seed":   seed,
        })

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_b64(self, source: Any) -> str:
        """
        Convert various source types to a base64-encoded PNG string.

        Accepts:
          - str starting with "data:image" (data URL)
          - str of a valid file path
          - str that is already raw base64
          - bytes (raw PNG)
          - Nuke node (renders the first frame to a temp file)
        """
        if isinstance(source, bytes):
            return base64.b64encode(source).decode()

        if isinstance(source, str):
            if source.startswith("data:image"):
                # data URL: strip the header
                _, data = source.split(",", 1)
                return data
            if os.path.isfile(source):
                with open(source, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            # Assume raw b64
            return source

        # Try Nuke node
        try:
            import nuke
            tmp = os.path.join(tempfile.gettempdir(), "_radiance_tmp.png")
            nuke.execute(source, nuke.frame(), nuke.frame())
            # Write via Write node
            w = nuke.createNode("Write")
            w["file"].setValue(tmp)
            w["file_type"].setValue("png")
            w.setInput(0, source)
            nuke.execute(w, nuke.frame(), nuke.frame())
            nuke.delete(w)
            with open(tmp, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e:
            raise ValueError(f"Cannot convert source to b64: {e}") from e


# ═══════════════════════════════════════════════════════════════════════════
# § 2  Nuke Panel UI
# ═══════════════════════════════════════════════════════════════════════════

def open_panel(host: str = None, port: int = None) -> None:
    """
    Open a minimal Radiance control panel inside Nuke.

    Call from the Nuke Script Editor or bind to a menu item via menu.py:

        import nuke
        nuke.menu("Nuke").findItem("Edit").addCommand(
            "Radiance/Open Panel",
            "from radiance_client import open_panel; open_panel()",
        )
    """
    if host is None:
        host = os.environ.get("RADIANCE_HTTP_HOST", "localhost")
    if port is None:
        port = int(os.environ.get("RADIANCE_HTTP_PORT", "7863"))

    try:
        import nuke
        import nukescripts
    except ImportError:
        print("[RadianceClient] open_panel() must be called inside Nuke.")
        return

    class _RadiancePanel(nukescripts.PythonPanel):
        def __init__(self) -> None:
            super().__init__("Radiance ← ComfyUI", "uk.co.fxtd.radiance.client")
            self._host = nuke.String_Knob("host",  "ComfyUI host", host)
            self._port = nuke.Int_Knob   ("port",  "Port",          port)
            self._port.clearFlag(nuke.STARTLINE)
            self._status = nuke.Text_Knob("status_lbl", "", "")
            self._ping_btn     = nuke.PyScript_Knob("do_ping",     "Ping",          "")
            self._status_btn   = nuke.PyScript_Knob("do_status",   "Status",        "")
            self._get_btn      = nuke.PyScript_Knob("do_get",      "Get Frame",     "")
            self._colorize_btn = nuke.PyScript_Knob("do_colorize", "Colorize Node", "")
            self._encode = nuke.Enumeration_Knob(
                "encode", "Encode (source)",
                ["logc4", "logc3", "slog3", "vlog", "srgb", "linear"],
            )
            self._decode = nuke.Enumeration_Knob(
                "decode", "Decode (target)",
                ["acescg", "srgb", "linear"],
            )
            for k in (self._host, self._port, self._status,
                      self._ping_btn, self._status_btn, self._get_btn,
                      self._encode, self._decode, self._colorize_btn):
                self.addKnob(k)

        def _client(self) -> RadianceClient:
            return RadianceClient(
                host    = self._host.value(),
                port    = int(self._port.value()),
                timeout = 5,
                token   = os.environ.get("RADIANCE_DCC_AUTH_TOKEN", ""),
            )

        def _set_status(self, msg: str) -> None:
            self._status.setValue(msg)

        def knobChanged(self, knob: Any) -> None:
            c = self._client()
            if knob is self._ping_btn:
                try:
                    r = c.ping()
                    self._set_status(f"✓ {r.get('status')} — v{r.get('version')}")
                except Exception as e:
                    self._set_status(f"✗ {e}")

            elif knob is self._status_btn:
                try:
                    r = c.status()
                    ready = r.get("frame_ready", False)
                    size  = f"{r.get('frame_width')}×{r.get('frame_height')}"
                    self._set_status(f"{'✓ Frame ready' if ready else '– No frame'} | {size}")
                except Exception as e:
                    self._set_status(f"✗ {e}")

            elif knob is self._get_btn:
                try:
                    c.get_frame(node_name="Radiance_Frame", create_read_node=True)
                    self._set_status("✓ Frame created in node graph")
                except Exception as e:
                    self._set_status(f"✗ {e}")

            elif knob is self._colorize_btn:
                try:
                    sel = nuke.selectedNode()
                except Exception:
                    self._set_status("✗ Select a node first")
                    return
                try:
                    c.colorize(
                        sel,
                        encode=self._encode.value(),
                        decode=self._decode.value(),
                        node_name="Radiance_Colorized",
                    )
                    self._set_status("✓ Colorized node created")
                except Exception as e:
                    self._set_status(f"✗ {e}")

    panel = _RadiancePanel()
    panel.show()


# ═══════════════════════════════════════════════════════════════════════════
# § 3  Nuke menu.py snippet (printed when run as __main__)
# ═══════════════════════════════════════════════════════════════════════════

_MENU_SNIPPET = """
# ── Add to your Nuke menu.py ─────────────────────────────────────────────
import sys
sys.path.insert(0, "/path/to/radiance/nuke_scripts")   # ← adjust this path

import nuke
nuke.menu("Nuke").findItem("Edit").addCommand(
    "Radiance/Open Panel",
    "from radiance_client import open_panel; open_panel('localhost', 7863)",
    icon="",
)
# ── Quick access via Python script shortcut ──────────────────────────────
nuke.menu("Nuke").findItem("Edit").addCommand(
    "Radiance/Get Frame",
    "from radiance_client import RadianceClient; "
    "RadianceClient().get_frame(create_read_node=True)",
)
"""


if __name__ == "__main__":
    print(__doc__)
    print()
    print("menu.py snippet:")
    print(_MENU_SNIPPET)

    # Quick CLI test (no Nuke required)
    import argparse
    parser = argparse.ArgumentParser(description="RadianceClient CLI test")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--ping",   action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    c = RadianceClient(args.host, args.port)
    if args.ping:
        print("ping →", c.ping())
    if args.status:
        print("status →", json.dumps(c.status(), indent=2))
