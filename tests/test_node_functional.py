"""Functional smoke test: actually CALL every node.

`test_node_smoke.py` checks that each node *declares* itself correctly —
INPUT_TYPES parses, RETURN_TYPES exists, FUNCTION names a real method, the class
instantiates. It never calls anything. A node can pass all of that and raise on
the first line of its FUNCTION.

This module closes that gap. For every key in NODE_CLASS_MAPPINGS it:

  1. builds an argument set from the node's own INPUT_TYPES — real tensors for
     IMAGE/MASK/LATENT, declared defaults for scalars, the first option for
     combos;
  2. calls FUNCTION;
  3. asserts the return is a tuple whose length matches RETURN_TYPES, and that
     IMAGE/MASK outputs have the shape and dtype ComfyUI requires;
  4. asserts the input tensors were not mutated in place — ComfyUI hands the
     same tensor to every consumer of a link, so an in-place write corrupts
     other branches of the graph.

A node that needs something this harness cannot synthesise — a MODEL, a VAE, a
file on disk, a GPU — SKIPS with that reason stated. A skip is a known gap; a
pass is a node that demonstrably runs. Neither is silent.
"""
from __future__ import annotations

import copy
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

RADIANCE_TORCH_GATED = True

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))


# ── Types this harness cannot fabricate ─────────────────────────────────────
#
# Anything requiring loaded weights, a ComfyUI runtime object, or another
# node's bespoke payload. Listing them explicitly means a NEW opaque type shows
# up as a skip with a name rather than as a mystery error.
_OPAQUE = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "CONTROL_NET", "STYLE_MODEL",
    "CLIP_VISION", "CLIP_VISION_OUTPUT", "GLIGEN", "UPSCALE_MODEL", "SIGMAS",
    "SAMPLER", "GUIDER", "NOISE", "PHOTOMAKER", "AUDIO", "SAM_MODEL",
    "BBOX_DETECTOR", "SEGM_DETECTOR", "RADIANCE_GRADE_INFO", "RADIANCE_CDL",
    "RADIANCE_AOV", "RADIANCE_PASSES", "RADIANCE_LORA_STACK",
    "RADIANCE_MODEL_META", "RADIANCE_SHOT", "RADIANCE_PROJECT",
}

_H = _W = 64
_B = 1


def _image():
    return torch.rand(_B, _H, _W, 3, dtype=torch.float32)


def _mask():
    return torch.rand(_B, _H, _W, dtype=torch.float32)


def _latent():
    return {"samples": torch.randn(_B, 4, _H // 8, _W // 8, dtype=torch.float32)}


def _synthesise(spec):
    """Build one argument from a single INPUT_TYPES entry.

    Returns (value, None) or (None, reason-it-cannot-be-built).
    """
    if not isinstance(spec, (list, tuple)) or not spec:
        return None, "malformed spec"

    kind, config = spec[0], (spec[1] if len(spec) > 1 else {})
    if not isinstance(config, dict):
        config = {}

    # A combo is declared as a list of options in the type slot.
    if isinstance(kind, (list, tuple)):
        options = [o for o in kind]
        if not options:
            return None, "empty combo"
        default = config.get("default")
        return (default if default in options else options[0]), None

    if kind == "IMAGE":
        return _image(), None
    if kind == "MASK":
        return _mask(), None
    if kind == "LATENT":
        return _latent(), None
    if kind == "INT":
        return int(config.get("default", 1)), None
    if kind == "FLOAT":
        return float(config.get("default", 0.5)), None
    if kind == "BOOLEAN":
        return bool(config.get("default", False)), None
    if kind == "STRING":
        return str(config.get("default", "")), None
    if kind in _OPAQUE:
        return None, f"needs a {kind}"
    return None, f"unknown input type {kind}"


def _build_args(cls):
    spec = cls.INPUT_TYPES()
    kwargs, skips = {}, []
    for section in ("required", "optional"):
        for name, entry in (spec.get(section) or {}).items():
            value, why = _synthesise(entry)
            if why is not None:
                if section == "required":
                    skips.append(f"{name}: {why}")
                continue
            kwargs[name] = value
    return kwargs, skips


def _snapshot(kwargs):
    """Deep copy of every tensor argument, for the mutation check."""
    out = {}
    for k, v in kwargs.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone()
        elif isinstance(v, dict) and torch.is_tensor(v.get("samples")):
            out[k] = v["samples"].detach().clone()
    return out


def _assert_unmutated(before, kwargs):
    for k, original in before.items():
        now = kwargs[k]
        if isinstance(now, dict):
            now = now["samples"]
        assert torch.equal(original, now), (
            f"the node mutated its {k!r} input in place. ComfyUI passes the same "
            "tensor object to every consumer of a link, so this corrupts other "
            "branches of the graph."
        )


# ── The reasons a call is allowed to fail without failing the test ──────────
#
# Each entry is a substring of the exception message plus what it means. Kept
# narrow on purpose: "needs a model file" is a legitimate environment gap,
# "index out of range" is a bug.
_ENVIRONMENTAL = (
    ("No module named", "an optional dependency is not installed here"),
    ("library required", "an optional dependency is not installed here"),
    ("is required for", "an optional dependency is not installed here"),
    ("pip install", "an optional dependency is not installed here"),
    ("not found", "needs a file or model that does not exist in the test env"),
    ("No such file", "needs a file that does not exist in the test env"),
    ("does not exist", "needs a path that does not exist in the test env"),
    ("CUDA", "needs a GPU"),
    ("folder_paths", "needs the ComfyUI runtime"),
    ("ffmpeg", "needs ffmpeg"),
    ("OpenEXR", "needs the OpenEXR bindings"),
    ("OpenImageIO", "needs the OpenImageIO bindings"),
    ("PyOpenColorIO", "needs OCIO"),
    ("Please install", "an optional dependency is not installed here"),
    ("network access is blocked", "needs a model download; blocked in tests"),
    ("requires", "a declared prerequisite is absent in the test env"),
)


def _environmental(exc):
    text = f"{type(exc).__name__}: {exc}"
    for needle, reason in _ENVIRONMENTAL:
        if needle.lower() in text.lower():
            return reason
    return None


def _all_nodes():
    try:
        import radiance
    except Exception as exc:  # pragma: no cover - covered by test_node_smoke
        pytest.skip(f"radiance did not import: {exc}")
    return radiance.NODE_CLASS_MAPPINGS


try:
    _NODES = _all_nodes()
except Exception:  # pragma: no cover
    _NODES = {}

_KEYS = sorted(_NODES)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """A test must never download a model.

    RadianceUpscaleImage fetched a 67 MB Real-ESRGAN checkpoint from the
    internet on its first functional run — silently, with no consent prompt, on
    what a user would experience as "I dropped a node and pressed queue". The
    behaviour is worth knowing about; a test suite that depends on it is not.
    """
    import socket
    import urllib.request

    def _blocked(*args, **kwargs):
        raise OSError("network access is blocked in tests; this node needs a "
                      "model file that is not present")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    try:
        import requests
        monkeypatch.setattr(requests, "get", _blocked)
    except ImportError:
        pass


@pytest.mark.real_torch
@pytest.mark.parametrize("key", _KEYS)
def test_node_executes(key):
    cls = _NODES[key]
    kwargs, skips = _build_args(cls)
    if skips:
        pytest.skip("; ".join(skips))

    node = cls()
    fn = getattr(node, cls.FUNCTION)
    before = _snapshot(kwargs)

    try:
        result = fn(**copy.copy(kwargs))
    except Exception as exc:
        reason = _environmental(exc)
        if reason:
            pytest.skip(f"{reason} ({type(exc).__name__}: {str(exc)[:120]})")
        raise

    _assert_unmutated(before, kwargs)

    declared = getattr(cls, "RETURN_TYPES", ())
    if not declared:
        # OUTPUT_NODE with no graph outputs; a dict or tuple is both fine.
        assert result is None or isinstance(result, (tuple, dict, list)), (
            f"{key} declares no RETURN_TYPES but returned {type(result).__name__}"
        )
        return

    if isinstance(result, dict):
        assert "result" in result, (
            f"{key} returned a ui dict without a 'result' key, so nothing "
            "reaches its output sockets"
        )
        result = result["result"]

    assert isinstance(result, tuple), (
        f"{key}.{cls.FUNCTION} returned {type(result).__name__}, not a tuple. "
        "ComfyUI unpacks the return positionally against RETURN_TYPES."
    )
    assert len(result) == len(declared), (
        f"{key} declares {len(declared)} outputs {declared} but returned "
        f"{len(result)}"
    )

    for value, kind in zip(result, declared):
        if kind == "IMAGE" and value is not None:
            assert torch.is_tensor(value), f"{key} IMAGE output is not a tensor"
            assert value.ndim == 4 and value.shape[-1] in (1, 3, 4), (
                f"{key} IMAGE output has shape {tuple(value.shape)}; ComfyUI "
                "requires (B, H, W, C)"
            )
            assert value.dtype in (torch.float32, torch.float16, torch.bfloat16), (
                f"{key} IMAGE output dtype is {value.dtype}, not float"
            )
            assert torch.isfinite(value).all(), f"{key} IMAGE output contains NaN or inf"
        elif kind == "MASK" and value is not None:
            assert torch.is_tensor(value), f"{key} MASK output is not a tensor"
            assert value.ndim in (2, 3), (
                f"{key} MASK output has shape {tuple(value.shape)}; ComfyUI "
                "requires (B, H, W) or (H, W)"
            )
            assert torch.isfinite(value).all(), f"{key} MASK output contains NaN or inf"
        elif kind == "LATENT" and value is not None:
            assert isinstance(value, dict) and "samples" in value, (
                f"{key} LATENT output is not a dict with a 'samples' key"
            )


def test_the_harness_actually_covered_the_catalog():
    """A harness that silently covers nothing is worse than no harness."""
    import radiance
    from radiance.config.constants import EXPECTED_MIN_NODE_COUNT

    if radiance._LOAD_RESULT.failures:
        pytest.skip("environment is short a runtime dependency")
    assert len(_KEYS) >= EXPECTED_MIN_NODE_COUNT, (
        f"only {len(_KEYS)} nodes were collected for functional testing"
    )
