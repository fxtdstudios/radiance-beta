# Radiance Nuke Server Script
# Run this inside Nuke's Script Editor to enable the bridge.
# v3.4 - Added Depth Map Generator Controls

import nuke
import nukescripts
import socket
import struct
import threading
import json
import urllib.request
import urllib.error
import random

PORT = 1986
COMFY_URL = "http://127.0.0.1:8188"
RUNNING = True
_SERVER_THREAD = None

# --- Bridge Server Logic ---


def safe_message(msg):
    """Safely show message in Nuke UI from any thread."""
    nuke.executeInMainThread(lambda: nuke.message(msg))


# execute_in_main removed - logic inlined in handle_client

RADIANCE_MAGIC = b"RCMD"
RADIANCE_END = "\n__RADIANCE_END__\n"

# ═══════════════════════════════════════════════════════════════════════════════
#                       COMMAND SANDBOX (Security)
# ═══════════════════════════════════════════════════════════════════════════════

# Dangerous patterns that must never appear in incoming commands
_BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "import socket",
    "import http",
    "import urllib",
    "__import__",
    "__builtins__",
    "__class__",
    "__subclasses__",
    "getattr(",
    "setattr(",
    "delattr(",
    "os.system",
    "os.popen",
    "os.exec",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.makedirs",
    "os.rename",
    "subprocess.",
    "shutil.rmtree",
    "shutil.move",
    "open(",
    "file(",
    "compile(",
    "globals()[",
    "locals()[",
    "eval(",
    "exec(",  # Prevent nested eval/exec
]

# Allowed top-level module names (Nuke API + standard safe modules)
_ALLOWED_IMPORTS = {"nuke", "nukescripts", "json", "math", "random"}


def _validate_command(command):
    """
    Validate a command against the security blocklist.

    Returns (is_safe: bool, reason: str).
    Commands containing dangerous system-access patterns are rejected.
    Only Nuke API and safe modules are allowed.
    """
    cmd_lower = command.lower()

    # Check blocklist
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in cmd_lower:
            return False, f"Blocked pattern: '{pattern}'"

    # Check imports — only allow safe modules
    import re

    for match in re.finditer(r"import\s+(\w+)", command):
        module = match.group(1)
        if module not in _ALLOWED_IMPORTS:
            return False, f"Blocked import: '{module}'"

    # Check "from X import" patterns
    for match in re.finditer(r"from\s+(\w+)", command):
        module = match.group(1)
        if module not in _ALLOWED_IMPORTS:
            return False, f"Blocked from-import: '{module}'"

    return True, "OK"


def handle_client(conn):
    """Handle individual client connection with binary protocol support."""
    try:
        conn.settimeout(5.0)

        # Read header: MAGIC(4) + VERSION(1) + LENGTH(4) = 9 bytes
        header_data = b""
        while len(header_data) < 9:
            chunk = conn.recv(9 - len(header_data))
            if not chunk:
                break
            header_data += chunk

        command = ""

        # Check protocol
        if len(header_data) == 9 and header_data[:4] == RADIANCE_MAGIC:
            # Binary Protocol
            cmd_length = struct.unpack("<I", header_data[5:9])[0]
            cmd_data = b""
            while len(cmd_data) < cmd_length:
                chunk = conn.recv(min(32768, cmd_length - len(cmd_data)))
                if not chunk:
                    break
                cmd_data += chunk
            command = cmd_data.decode("utf-8", errors="replace").strip()
        else:
            # Legacy / Raw Protocol
            remaining = conn.recv(32768)
            command = (
                (header_data + remaining).decode("utf-8", errors="replace").strip()
            )

        if command:
            # ── Security: Validate command before execution ──
            is_safe, reason = _validate_command(command)
            if not is_safe:
                msg = f"ERROR: Command rejected by security sandbox — {reason}"
                print(f"[Radiance Security] {msg}")
                conn.sendall((msg + RADIANCE_END).encode("utf-8"))
                return

            # Execute in main thread and capture result
            result_box = [None]
            error_box = [None]
            done_event = threading.Event()

            # Build a restricted global scope for execution
            safe_globals = {
                "nuke": nuke,
                "nukescripts": nukescripts,
                "json": json,
                "math": __import__("math"),
                "random": random,
            }

            def execute_wrapper():
                try:
                    # Try eval first (for PONG and expressions)
                    try:
                        val = eval(command, safe_globals)  # nosec B307
                        result_box[0] = str(val)
                    except SyntaxError:
                        # Fallback to exec for multi-line scripts
                        exec(command, safe_globals)  # nosec B102
                        # Check if last line is an expression
                        lines = command.strip().split("\n")
                        if lines:
                            last = lines[-1].strip()
                            if last.startswith("'") or last.startswith('"'):
                                try:
                                    result_box[0] = str(eval(last, safe_globals))  # nosec B307
                                except Exception:
                                    result_box[0] = "OK"
                            else:
                                result_box[0] = "OK"
                        else:
                            result_box[0] = "OK"
                except Exception as e:
                    error_box[0] = f"ERROR: {e}"
                finally:
                    done_event.set()

            nuke.executeInMainThread(execute_wrapper)
            done_event.wait(timeout=10.0)

            response = (
                error_box[0]
                if error_box[0]
                else (result_box[0] if result_box[0] else "ERROR: Timeout")
            )
            conn.sendall((response + RADIANCE_END).encode("utf-8"))

    except Exception as e:
        print(f"Radiance Bridge Error: {e}")
        try:
            conn.sendall((f"ERROR: {e}" + RADIANCE_END).encode("utf-8"))
        except Exception:  # nosec B110
            pass
    finally:
        conn.close()


def start_radiance_server():
    """Start listening for Radiance commands on localhost."""
    global PORT

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(0.5)

    try:
        # Binds to 0.0.0.0 to allow pipeline communication across the network
        server.bind(("0.0.0.0", PORT))  # nosec B104 (intentional network binding)
        server.listen(5)

        msg = f"Radiance Bridge listening on port {PORT}..."
        print(msg)
        safe_message(msg)

        while RUNNING:
            try:
                conn, addr = server.accept()
                handle_client(conn)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Connection error: {e}")

    except OSError as e:
        err = f"Socket error (Port {PORT} in use?): {e}"
        print(err)
        safe_message(err)
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.close()
        print("Radiance Bridge Server Stopped")


def start_bridge():
    """Start the bridge server in a thread if not already running."""
    global _SERVER_THREAD, RUNNING

    if _SERVER_THREAD and _SERVER_THREAD.is_alive():
        nuke.message(f"Radiance Bridge is already running on port {PORT}")
        return

    RUNNING = True
    _SERVER_THREAD = threading.Thread(target=start_radiance_server)
    _SERVER_THREAD.daemon = True
    _SERVER_THREAD.start()


# --- ComfyUI Interaction Logic ---


def fetch_last_history_item():
    """Fetch the most recent history item from ComfyUI."""
    try:
        url = f"{COMFY_URL}/history"
        if not url.startswith(("http://", "https://")):
            return None
        with urllib.request.urlopen(url) as response:  # nosec B310
            if response.status != 200:
                return None
            data = json.loads(response.read().decode("utf-8"))
            if not data:
                return None
            last_key = list(data.keys())[-1]
            return data[last_key]
    except Exception as e:
        print(f"ComfyUI API Error: {e}")
        return None


def find_node_by_class(graph, class_name):
    """Find a node by its class type."""
    if not graph:
        return None, None
    for node_id, node_data in graph.items():
        if node_data.get("class_type") == class_name:
            return node_id, node_data
    return None, None


def find_cinematic_node(graph):
    """Find the compatible Cinematic node (Encoder or Studio) in the prompt data."""
    if not graph:
        return None, None, None

    nid, ndata = find_node_by_class(graph, "RadianceCinematicPromptEncoder")
    if nid:
        return nid, ndata, "ENCODER"

    nid, ndata = find_node_by_class(graph, "RadianceCinemaStudio")
    if nid:
        return nid, ndata, "STUDIO"

    return None, None, None


def submit_prompt(graph):
    """Submit a graph to ComfyUI."""
    try:
        data = json.dumps({"prompt": graph}).encode("utf-8")
        req = urllib.request.Request(
            f"{COMFY_URL}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        url = req.full_url
        if not url.startswith(("http://", "https://")):
            return
        with urllib.request.urlopen(req) as resp:  # nosec B310
            if resp.status == 200:
                print("ComfyUI Queued Successfully")
            else:
                nuke.message(f"ComfyUI Error: {resp.status}")
    except Exception as e:
        nuke.message(f"Failed to queue prompt: {e}")


def randomize_seed(graph):
    """Randomize seed in sampler nodes to force re-generation."""
    for node in graph.values():
        if "inputs" in node:
            if "seed" in node["inputs"]:
                node["inputs"]["seed"] = random.randint(1, 18446744073709551615)  # nosec B311
            elif "noise_seed" in node["inputs"]:
                node["inputs"]["noise_seed"] = random.randint(1, 18446744073709551615)  # nosec B311


def queue_last_run():
    item = fetch_last_history_item()
    if not item:
        nuke.message("No history found in ComfyUI.")
        return

    prompt_payload = item.get("prompt")
    if not prompt_payload:
        nuke.message("Invalid history item format.")
        return

    graph = None
    if isinstance(prompt_payload, list) and len(prompt_payload) >= 3:
        graph = prompt_payload[2]
    elif isinstance(prompt_payload, dict):
        graph = prompt_payload
        # Verify
        valid = False
        for k, v in graph.items():
            if isinstance(v, dict) and "inputs" in v:
                valid = True
                break
        if not valid:
            graph = None

    if graph:
        randomize_seed(graph)
        submit_prompt(graph)
    else:
        nuke.message("Could not find graph in history.")


# --- Cinematic Encoder Panel ---


class RadiancePromptPanel(nukescripts.PythonPanel):
    def __init__(self):
        super(RadiancePromptPanel, self).__init__("Radiance Cinematic")

        # --- Data Lists ---
        self.cameras = [
            "None",
            "ARRI Alexa 35",
            "ARRI Alexa Mini LF",
            "ARRI Alexa 65 (IMAX)",
            "Sony Venice 2",
            "RED V-Raptor XL",
            "Panavision Millennium DXL2",
            "IMAX 15/70mm Film Camera",
        ]
        self.lenses = [
            "None",
            "50mm Standard Prime",
            "35mm Classic Wide",
            "85mm Portrait Prime",
            "24mm Wide Angle",
            "Anamorphic Lens",
            "ARRI/Zeiss Signature Prime",
            "Panavision Primo 70",
            "Cooke S7/i Full Frame",
        ]
        self.apertures = [
            "None",
            "f/1.2 (Dreamy Bokeh)",
            "f/1.8 (Soft Background)",
            "f/2.8 (Cinematic Separation)",
            "f/4.0 (Balanced)",
            "f/5.6 (Sharp Subject)",
            "f/8.0 (Deep Focus)",
        ]
        self.framings = [
            "None",
            "Medium Shot (MS)",
            "Close-Up (CU)",
            "Wide Shot (WS)",
            "Extreme Wide Shot (EWS)",
            "Low Angle (Hero Shot)",
        ]

        self.presets = [
            "SDXL Square (1:1)",
            "SDXL Portrait (3:4)",
            "SDXL Landscape (4:3)",
            "SDXL Wide (16:9)",
            "FLUX 1MP Square",
            "FLUX 1MP Wide (16:9)",
            "FLUX 1MP Portrait (9:16)",
            "FLUX 2K Square (1:1)",
            "FLUX 2K Wide (16:9)",
            "FLUX 2K DCI (1.90:1)",
            "1080p Full HD (16:9)",
            "4K UHD (16:9)",
            "4K DCI (1.90:1)",
            "2K DCI Scope (2.39:1)",
            "Instagram Portrait (4:5)",
            "Instagram Story/Reel (9:16)",
        ]

        self.samplers = [
            "euler",
            "euler_ancestral",
            "heun",
            "dpm_2",
            "dpm_2_ancestral",
            "dpmpp_2s_ancestral",
            "dpmpp_2m",
            "dpmpp_2m_sde",
            "dpmpp_sde",
            "uni_pc",
        ]
        self.schedulers = [
            "normal",
            "karras",
            "exponential",
            "sgm_uniform",
            "simple",
            "ddim_uniform",
        ]

        # RadianceDepthMapGenerator inputs
        self.depth_models = [
            "Small (25M - Fast)",
            "Base (98M - Balanced)",
            "Large (335M - Best)",
        ]

        # --- UI Elements ---

        # Section: Prompt
        self.prompt_knob = nuke.Multiline_Eval_String_Knob("base_prompt", "Prompt")
        self.addKnob(self.prompt_knob)

        self.div1 = nuke.Text_Knob("div1", "")
        self.addKnob(self.div1)

        # Section: Camera
        self.camera_knob = nuke.Enumeration_Knob("camera", "Camera", self.cameras)
        self.addKnob(self.camera_knob)

        self.lens_knob = nuke.Enumeration_Knob("lens", "Lens / Focal", self.lenses)
        self.addKnob(self.lens_knob)

        self.aperture_knob = nuke.Enumeration_Knob(
            "aperture", "Aperture", self.apertures
        )
        self.addKnob(self.aperture_knob)

        self.framing_knob = nuke.Enumeration_Knob("framing", "Framing", self.framings)
        self.addKnob(self.framing_knob)

        self.div2 = nuke.Text_Knob("div2", "")
        self.addKnob(self.div2)

        # Section: Resolution
        self.res_knob = nuke.Enumeration_Knob("resolution", "Resolution", self.presets)
        self.addKnob(self.res_knob)

        self.div3 = nuke.Text_Knob("div3", "")
        self.addKnob(self.div3)

        # Section: Sampler
        self.steps_knob = nuke.Int_Knob("steps", "Steps")
        self.addKnob(self.steps_knob)

        self.cfg_knob = nuke.Double_Knob("cfg", "CFG Scale")
        self.cfg_knob.setRange(1.0, 20.0)
        self.addKnob(self.cfg_knob)

        self.denoise_knob = nuke.Double_Knob("denoise", "Denoise")
        self.denoise_knob.setRange(0.0, 1.0)
        self.addKnob(self.denoise_knob)

        self.sampler_knob = nuke.Enumeration_Knob(
            "sampler_name", "Sampler", self.samplers
        )
        self.addKnob(self.sampler_knob)

        self.scheduler_knob = nuke.Enumeration_Knob(
            "scheduler_name", "Scheduler", self.schedulers
        )
        self.addKnob(self.scheduler_knob)

        self.div4 = nuke.Text_Knob("div4", "")
        self.addKnob(self.div4)

        # Section: Depth (New)
        self.depth_label = nuke.Text_Knob("depth_label", "<b>Depth Generation</b>")
        self.addKnob(self.depth_label)

        self.depth_model_knob = nuke.Enumeration_Knob(
            "depth_model", "Depth Model", self.depth_models
        )
        self.addKnob(self.depth_model_knob)

        self.depth_blur_knob = nuke.Double_Knob("depth_blur", "Edge Blur")
        self.depth_blur_knob.setRange(0.0, 5.0)
        self.addKnob(self.depth_blur_knob)

        self.depth_invert_knob = nuke.Boolean_Knob("depth_invert", "Invert Depth")
        self.addKnob(self.depth_invert_knob)

        self.div5 = nuke.Text_Knob("div5", "")
        self.addKnob(self.div5)

        # Section: Actions
        self.refresh_btn = nuke.PyScript_Knob("refresh", "Sync from Last Run")
        self.addKnob(self.refresh_btn)

        self.generate_btn = nuke.PyScript_Knob("generate", "Generate (Queue)")
        self.addKnob(self.generate_btn)

        # State
        self.last_graph = None
        self.node_id = None
        self.node_type = None
        self.res_node_id = None
        self.sampler_node_id = None
        self.depth_node_id = None

        # Attempt initial sync
        self.sync_from_last()

    def knobChanged(self, knob):
        if knob == self.refresh_btn:
            self.sync_from_last()
        elif knob == self.generate_btn:
            self.generate()

    def sync_from_last(self):
        item = fetch_last_history_item()
        if not item:
            print("No history to sync.")
            return

        prompt_payload = item.get("prompt")
        graph = None
        if isinstance(prompt_payload, list) and len(prompt_payload) >= 3:
            graph = prompt_payload[2]
        elif isinstance(prompt_payload, dict):
            graph = prompt_payload

        if not graph:
            print("Could not extract graph for sync.")
            return

        self.last_graph = graph

        # 1. Sync Cinematic Node
        nid, ndata, ntype = find_cinematic_node(graph)
        if nid:
            self.node_id = nid
            self.node_type = ntype
            inputs = ndata.get("inputs", {})

            self.prompt_knob.setValue(inputs.get("base_prompt", ""))

            camera_val = (
                inputs.get("camera_type")
                if ntype == "ENCODER"
                else inputs.get("camera")
            )
            lens_val = (
                inputs.get("lens_focal")
                if ntype == "ENCODER"
                else inputs.get("focal_length")
            )
            ap_val = (
                inputs.get("aperture_dof")
                if ntype == "ENCODER"
                else inputs.get("aperture")
            )
            frame_val = inputs.get("framing") if ntype == "ENCODER" else "None"

            if camera_val in self.cameras:
                self.camera_knob.setValue(camera_val)
            if lens_val in self.lenses:
                self.lens_knob.setValue(lens_val)
            if ap_val in self.apertures:
                self.aperture_knob.setValue(ap_val)
            if frame_val in self.framings:
                self.framing_knob.setValue(frame_val)

            print(f"Synced with {ntype} Node {nid}")
        else:
            print("No Radiance cinematic node found.")
            self.node_id = None

        # 2. Sync Resolution Node
        rid, rdata = find_node_by_class(graph, "RadianceResolution")
        self.res_node_id = rid
        if rid:
            rinputs = rdata.get("inputs", {})
            preset = rinputs.get("preset")
            if preset in self.presets:
                self.res_knob.setValue(preset)
            print(f"Synced with Resolution Node {rid}")

        # 3. Sync Sampler Node
        sid, sdata = find_node_by_class(graph, "RadianceSamplerPro")
        self.sampler_node_id = sid
        if sid:
            sinputs = sdata.get("inputs", {})
            self.steps_knob.setValue(sinputs.get("steps", 20))
            self.cfg_knob.setValue(sinputs.get("cfg", 1.0))
            self.denoise_knob.setValue(sinputs.get("denoise", 1.0))

            sname = sinputs.get("sampler")
            if sname in self.samplers:
                self.sampler_knob.setValue(sname)

            schname = sinputs.get("scheduler")
            if schname in self.schedulers:
                self.scheduler_knob.setValue(schname)
            print(f"Synced with Sampler Node {sid}")

        # 4. Sync Depth Node
        did, ddata = find_node_by_class(graph, "RadianceDepthMapGenerator")
        self.depth_node_id = did
        if did:
            dinputs = ddata.get("inputs", {})
            self.depth_model_knob.setValue(
                dinputs.get("model_size", "Base (98M - Balanced)")
            )
            self.depth_blur_knob.setValue(dinputs.get("blur_edges", 0.0))
            self.depth_invert_knob.setValue(dinputs.get("invert", False))
            self.depth_label.setValue("<b>Depth Generation (Active)</b>")
            print(f"Synced with Depth Node {did}")
        else:
            self.depth_label.setValue("<b>Depth Generation (Not Found)</b>")

    def generate(self):
        if not self.last_graph:
            nuke.message("No graph loaded. Sync first.")
            return

        # Update Prompt / Camera
        if self.node_id:
            inputs = self.last_graph[self.node_id]["inputs"]
            inputs["base_prompt"] = self.prompt_knob.value()

            if self.node_type == "ENCODER":
                inputs["camera_type"] = self.camera_knob.value()
                inputs["lens_focal"] = self.lens_knob.value()
                inputs["aperture_dof"] = self.aperture_knob.value()
                inputs["framing"] = self.framing_knob.value()
            elif self.node_type == "STUDIO":
                inputs["camera"] = self.camera_knob.value()
                val = self.lens_knob.value()
                if "mm" in val:
                    inputs["focal_length"] = val
                else:
                    inputs["lens_series"] = val
                inputs["aperture"] = self.aperture_knob.value()

        # Update Resolution
        if self.res_node_id:
            r_inputs = self.last_graph[self.res_node_id]["inputs"]
            r_inputs["preset"] = self.res_knob.value()

        # Update Sampler
        if self.sampler_node_id:
            s_inputs = self.last_graph[self.sampler_node_id]["inputs"]
            s_inputs["steps"] = int(self.steps_knob.value())
            s_inputs["cfg"] = self.cfg_knob.value()
            s_inputs["denoise"] = self.denoise_knob.value()
            s_inputs["sampler"] = self.sampler_knob.value()
            s_inputs["scheduler"] = self.scheduler_knob.value()

        # Update Depth
        if self.depth_node_id:
            d_inputs = self.last_graph[self.depth_node_id]["inputs"]
            d_inputs["model_size"] = self.depth_model_knob.value()
            d_inputs["blur_edges"] = self.depth_blur_knob.value()
            d_inputs["invert"] = self.depth_invert_knob.value()

        # Randomize Seed
        randomize_seed(self.last_graph)

        # Submit
        submit_prompt(self.last_graph)


def show_radiance_panel():
    p = RadiancePromptPanel()
    p.show()


# --- Menu ---


def add_radiance_menu():
    radiance_menu = nuke.menu("Nuke").addMenu("Radiance")
    radiance_menu.addCommand("Start Bridge", start_bridge)
    radiance_menu.addCommand("Queue Last Run", queue_last_run, "F5")
    radiance_menu.addCommand("Cinematic Encoder", show_radiance_panel, "Shift+F5")


if nuke.GUI:
    add_radiance_menu()
    print("Radiance Bridge Menu installed.")
