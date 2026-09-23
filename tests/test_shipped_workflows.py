"""Shipped workflows load with every widget where it belongs (3.5.0).

The ComfyUI frontend restores a node's widgets from `widgets_values` BY
POSITION. When a node gains an input, every workflow saved before that shifts
by one from that point on. `workflows/start.json` had drifted exactly so: the
Sampler's new `audio_cfg` received "euler", every later widget moved one slot,
the frontend's Sampler extension threw ("samplerMode.includes is not a
function") and ComfyUI refused to queue the graph; HDR VAE Decode also had no
VAE connected and the Flux loader no clip_l.

The file is now re-saved from ComfyUI 0.32 / frontend 1.48 itself. This test
re-derives each Radiance node's widget order from INPUT_TYPES and checks every
saved value against its input's type, so the next added input that breaks a
shipped workflow fails here instead of in a user's first queue.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((ROOT / "workflows").rglob("*.json"))
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")


def _mappings():
    import radiance
    return radiance.NODE_CLASS_MAPPINGS


def _expected_widgets(cls):
    it = cls.INPUT_TYPES()
    out = []
    for section in ("required", "optional"):
        for name, spec in (it.get(section) or {}).items():
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue
            # A combo's type is its option list (a stub object under the
            # test harness); every non-string type is a combo.
            if not isinstance(typ, str) or typ in WIDGET_TYPES:
                out.append((name, typ, opts))
                if typ == "INT" and (name in ("seed", "noise_seed") or opts.get("control_after_generate")):
                    out.append(("control_after_generate", ["fixed", "increment", "decrement", "randomize"], {}))
    return out


def _ok(value, typ):
    if not isinstance(typ, str):
        try:
            choices = list(typ) if isinstance(typ, (list, tuple)) else []
        except TypeError:
            choices = []
        # Stubbed comfy lists (samplers, schedulers) are empty, and model
        # files depend on the machine: a filename is not checked.
        if isinstance(value, str) and value.lower().endswith(MODEL_EXT):
            return True
        return not choices or value in choices
    if typ == "INT":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "FLOAT":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ == "BOOLEAN":
        return isinstance(value, bool)
    if typ == "STRING":
        return isinstance(value, str)
    return True


def test_there_are_shipped_workflows():
    assert WORKFLOWS, "no workflows/*.json found"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_radiance_widget_value_fits_its_input(path):
    mappings = _mappings()
    wf = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for node in wf.get("nodes", []):
        cls = mappings.get(node.get("type"))
        if cls is None:
            continue
        values = node.get("widgets_values") or []
        expected = _expected_widgets(cls)
        # Fewer values than widgets is fine when only the tail is missing (the
        # frontend gives newer trailing inputs their defaults); a gap in the
        # middle shows up below as a value that does not fit its input.
        for (name, typ, _), value in zip(expected, values):
            if not _ok(value, typ):
                problems.append(f"{node['type']} #{node['id']}.{name} = {value!r} does not fit {typ if isinstance(typ, str) else 'its options'}")
    assert not problems, "\n".join(problems)


def test_start_graph_is_wired():
    wf = json.loads((ROOT / "workflows" / "start.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}
    decode = next(n for n in nodes.values() if n["type"] == "RadianceHDRVAEDecode")
    vae_in = next(i for i in decode["inputs"] if i["name"] == "vae")
    assert vae_in.get("link") is not None, "HDR VAE Decode has no VAE connected"
    loader = next(n for n in nodes.values() if n["type"] == "RadianceUnifiedLoader")
    widgets = [e[0] for e in _expected_widgets(_mappings()["RadianceUnifiedLoader"])]
    vals = dict(zip(widgets, loader["widgets_values"]))
    assert vals["clip_l"] != "None" and vals["t5xxl"] != "None", "Flux.1 needs both clip_l and t5xxl"
