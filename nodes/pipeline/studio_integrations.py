"""
Studio integration nodes for sending renders to DCC applications.

RadianceNukeSend  — Export EXR + .nk Read-node snippet for Nuke.
RadianceDaVinciSend — Export to DaVinci Resolve shared media folder.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from radiance.nodes_io import _save_exr, _save_pil_image

logger = logging.getLogger("radiance.nodes.studio")

_NUKE_NK_TEMPLATE = """\
version {nuke_version}
Read {{
 inputs 0
 file {filepath}
 first {first}
 last {last}
 origfirst {first}
 origlast {last}
 name {node_name}
 xpos 0
 ypos 0
}}
"""


class RadianceNukeSend:
    CATEGORY = "FXTD STUDIOS/Radiance/07 Pipeline & DCC"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "render_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Frame to export. A batch is written as a numbered EXR sequence.",
                }),
                "nuke_folder": ("STRING", {
                    "default": "",
                    "placeholder": "/renders/nuke_exchange/",
                    "tooltip": "Output folder for image + .nk file. Created if missing.",
                }),
                "filename": ("STRING", {
                    "default": "radiance_out",
                    "tooltip": "Base name for the EXR file(s).",
                }),
            },
            "optional": {
                "frame_start": ("INT", {
                    "default": 1001, "min": 0, "max": 999999,
                    "tooltip": "Starting frame number for the EXR sequence.",
                }),
                "push_to_nuke": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True and Nuke listener is running, auto-create a Read node via TCP.",
                }),
                "nuke_host": ("STRING", {
                    "default": "127.0.0.1",
                    "tooltip": "Nuke listener host (used only when push_to_nuke=True).",
                }),
                "nuke_port": ("INT", {
                    "default": 1986, "min": 1024, "max": 65535,
                    "tooltip": "Nuke listener port (used only when push_to_nuke=True).",
                }),
                "half_float": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write 16-bit half EXR (True) or 32-bit float EXR (False).",
                }),
            },
        }

    DESCRIPTION = (
        "Export image as EXR and write a .nk Read-node snippet for direct Nuke import. "
        "Optionally push to a running Nuke instance via the Radiance TCP listener."
    )
    FUNCTION = "run"

    def run(
        self,
        image: torch.Tensor,
        nuke_folder: str,
        filename: str,
        frame_start: int = 1001,
        push_to_nuke: bool = False,
        nuke_host: str = "127.0.0.1",
        nuke_port: int = 1986,
        half_float: bool = True,
    ) -> Tuple[str, str]:
        from radiance.config.env import get_nuke_host, get_nuke_port
        if nuke_host == "127.0.0.1":
            nuke_host = get_nuke_host()
        if nuke_port == 1986:
            nuke_port = get_nuke_port()

        if not nuke_folder:
            return ("Error: nuke_folder is required.", "")

        frames = image.detach().cpu().float().numpy()
        if frames.ndim == 3:
            frames = frames[np.newaxis]

        out_dir = Path(nuke_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        written = []
        for i in range(len(frames)):
            frame_num = frame_start + i
            if len(frames) == 1:
                fname = f"{filename}.exr"
            else:
                fname = f"{filename}.{frame_num:04d}.exr"
            fpath = out_dir / fname
            _save_exr(frames[i], fpath, half=half_float)
            written.append(str(fpath))

        first_frame = frame_start
        last_frame = frame_start + len(frames) - 1

        if len(frames) == 1:
            file_field = str(out_dir / f"{filename}.exr").replace("\\", "/")
        else:
            file_field = str(out_dir / f"{filename}.####.exr").replace("\\", "/")

        nk_content = _NUKE_NK_TEMPLATE.format(
            nuke_version="15.0",
            filepath=file_field,
            first=first_frame,
            last=last_frame,
            node_name=filename,
        )

        nk_path = out_dir / f"{filename}.nk"
        nk_path.write_text(nk_content, encoding="utf-8")

        status = f"{len(written)} EXR frame(s) + .nk → {out_dir}"
        render_path = str(out_dir)

        if push_to_nuke:
            push_status = self._push_to_nuke(
                file_field, filename, first_frame, last_frame, nuke_host, nuke_port
            )
            status += f" | nuke push: {push_status}"

        return (status, render_path)

    @staticmethod
    def _push_to_nuke(
        filepath: str,
        node_name: str,
        first_frame: int,
        last_frame: int,
        host: str,
        port: int,
    ) -> str:
        try:
            from radiance.tools.nuke_connector import NukeConnector
        except ImportError:
            try:
                from radiance.nuke_connector import NukeConnector
            except ImportError:
                return "NukeConnector not available"

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


class RadianceDaVinciSend:
    CATEGORY = "FXTD STUDIOS/Radiance/07 Pipeline & DCC"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "render_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Frame to export. Batches write numbered files.",
                }),
                "resolve_folder": ("STRING", {
                    "default": "",
                    "placeholder": "/tmp/resolve_exchange",
                    "tooltip": "DaVinci Resolve shared media folder. Created if missing.",
                }),
                "filename": ("STRING", {
                    "default": "radiance_out",
                    "tooltip": "Base filename (no extension).",
                }),
                "bit_depth": (["16bit", "8bit", "EXR"], {
                    "default": "16bit",
                    "tooltip": "Output bit depth. EXR writes 16-bit half-float.",
                }),
            },
            "optional": {
                "frame_start": ("INT", {
                    "default": 1001, "min": 0, "max": 999999,
                    "tooltip": "Starting frame number for numbered sequences.",
                }),
            },
        }

    DESCRIPTION = (
        "Export the current image to a DaVinci Resolve shared media folder for manual import. "
        "Supports 8-bit PNG, 16-bit TIFF, and EXR output formats."
    )
    FUNCTION = "run"

    def run(
        self,
        image: torch.Tensor,
        resolve_folder: str,
        filename: str,
        bit_depth: str = "16bit",
        frame_start: int = 1001,
    ) -> Tuple[str, str]:
        if not resolve_folder:
            return ("Error: resolve_folder is required.", "")

        frames = image.detach().cpu().float().numpy()
        if frames.ndim == 3:
            frames = frames[np.newaxis]

        out_dir = Path(resolve_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        written = []
        for i in range(len(frames)):
            frame_num = frame_start + i
            if bit_depth == "EXR":
                if len(frames) == 1:
                    fname = f"{filename}.exr"
                else:
                    fname = f"{filename}.{frame_num:04d}.exr"
                fpath = out_dir / fname
                _save_exr(frames[i], fpath, half=True)
            elif bit_depth == "8bit":
                if len(frames) == 1:
                    fname = f"{filename}.png"
                else:
                    fname = f"{filename}.{frame_num:04d}.png"
                fpath = out_dir / fname
                _save_pil_image(frames[i], fpath, "8-bit PNG")
            else:  # 16bit
                if len(frames) == 1:
                    fname = f"{filename}.tif"
                else:
                    fname = f"{filename}.{frame_num:04d}.tif"
                fpath = out_dir / fname
                _save_pil_image(frames[i], fpath, "16-bit TIFF")

            written.append(str(fpath))

        status = f"{len(written)} frame(s) [{bit_depth}] → {out_dir}"
        return (status, str(out_dir))


NODE_CLASS_MAPPINGS = {
    "RadianceNukeSend": RadianceNukeSend,
    "RadianceDaVinciSend": RadianceDaVinciSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceNukeSend": "◎ Radiance Send to Nuke",
    "RadianceDaVinciSend": "◎ Radiance Send to DaVinci Resolve",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
