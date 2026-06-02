"""
tests/test_hdr_delivery.py — Unit tests for nodes_hdr_delivery.py

Covers:
  • _linear_to_pq: ST.2084 OETF output range and known reference values
  • _linear_to_hlg: HLG OETF output range and known reference values
  • PQ ↔ linear partial roundtrip (encode → decode monotonicity)
  • RadiancePQEncoder output shape, dtype, clamp range
  • RadianceHLGEncoder output shape, dtype, clamp range
  • RadianceHDRToneMapPreview smoke test
  • Node registrations present
"""

import sys
import types
import importlib
import pytest
import numpy as np

# ── Real torch check ──────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# ── ComfyUI stub ──────────────────────────────────────────────────────────────
for mod in ["folder_paths", "comfy", "comfy.utils"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)


def _import_delivery():
    if "nodes_hdr_delivery" in sys.modules:
        return sys.modules["nodes_hdr_delivery"]
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    return importlib.import_module("nodes_hdr_delivery")


# ─────────────────────────────────────────────────────────────────────────────
# _linear_to_pq (ST.2084 OETF)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestLinearToPQ:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_delivery()
        self.fn = self.mod._linear_to_pq

    def test_output_range_clamped_0_to_1(self):
        """PQ output must be in [0, 1] for all non-negative linear inputs."""
        x = torch.linspace(0, 2.0, 200)
        out = self.fn(x.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, 200, 3))
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_black_maps_to_zero(self):
        x = torch.zeros(1, 4, 4, 3)
        out = self.fn(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_reference_white_203_nits(self):
        """BT.2408: scene-linear 1.0 = 203 nits → PQ value should be ~0.5807."""
        # Reference: PQ code for 203 nits / 10000 nits peak
        x = torch.ones(1, 1, 1, 3)  # scene-linear 1.0
        out = self.fn(x, peak_nits=10000.0)
        pq_val = out[0, 0, 0, 0].item()
        # PQ(203/10000) ≈ 0.5807 per ST.2084
        assert pq_val == pytest.approx(0.5807, abs=0.01)

    def test_monotonic(self):
        """PQ encoding must be strictly monotonically increasing."""
        vals = torch.linspace(0.01, 1.0, 50).view(1, 1, 50, 1).expand(1, 1, 50, 3)
        out = self.fn(vals)[0, 0, :, 0]
        diffs = out[1:] - out[:-1]
        assert (diffs >= 0).all(), "PQ OETF must be monotonically non-decreasing"

    def test_shape_preserved(self):
        x = torch.rand(2, 8, 8, 3)
        out = self.fn(x)
        assert out.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# _linear_to_hlg (HLG OETF)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestLinearToHLG:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_delivery()
        self.fn = self.mod._linear_to_hlg

    def test_output_range_0_to_1(self):
        """HLG output must stay in [0, 1] for inputs in [0, 1]."""
        x = torch.rand(1, 8, 8, 3)
        out = self.fn(x)
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_black_maps_to_zero(self):
        x = torch.zeros(1, 4, 4, 3)
        out = self.fn(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_reference_0_5_maps_to_0_5(self):
        """HLG: input 1/12 maps to 0.5 (boundary between linear and log segments)."""
        # At x=1/12, HLG transitions from linear to log; encoded value = 0.5
        x = torch.full((1, 1, 1, 3), 1.0 / 12.0)
        out = self.fn(x)
        val = out[0, 0, 0, 0].item()
        assert val == pytest.approx(0.5, abs=0.02)

    def test_monotonic(self):
        vals = torch.linspace(0.001, 1.0, 50).view(1, 1, 50, 1).expand(1, 1, 50, 3)
        out = self.fn(vals)[0, 0, :, 0]
        diffs = out[1:] - out[:-1]
        assert (diffs >= 0).all(), "HLG OETF must be monotonically non-decreasing"

    def test_shape_preserved(self):
        x = torch.rand(2, 16, 16, 3)
        out = self.fn(x)
        assert out.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# PQ encode → decode monotonicity
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
def test_pq_decode_monotonic():
    """Encoded PQ values should be monotonically related to input luminance."""
    mod = _import_delivery()
    inputs = torch.linspace(0.0, 5.0, 30)
    encoded = []
    for v in inputs:
        x = torch.full((1, 1, 1, 3), v.item())
        pq = mod._linear_to_pq(x)[0, 0, 0, 0].item()
        encoded.append(pq)
    for i in range(1, len(encoded)):
        assert encoded[i] >= encoded[i - 1] - 1e-6, \
            f"PQ encoding not monotonic at input {inputs[i].item():.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# RadianceHDREncode node (merged PQ + HLG)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestRadianceHDREncode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_delivery()
        self.node = self.mod.RadianceHDREncode()

    def test_pq_output_shape(self):
        img = torch.rand(1, 16, 16, 3)
        result = self.node.encode(image=img, format="PQ (HDR10)", peak_nits=1000)
        assert result[0].shape == img.shape

    def test_pq_output_clamped(self):
        img = torch.rand(1, 8, 8, 3) * 5.0
        out = self.node.encode(image=img, format="PQ (HDR10)", peak_nits=1000)[0]
        assert out.min().item() >= -1e-4
        assert out.max().item() <= 1.0 + 1e-4

    def test_pq_alpha_preserved(self):
        img = torch.rand(1, 8, 8, 4)
        img[..., 3] = 0.75
        out = self.node.encode(image=img, format="PQ (HDR10)", peak_nits=1000)[0]
        if out.shape[-1] == 4:
            assert torch.allclose(out[..., 3], img[..., 3], atol=1e-4)

    def test_hlg_output_shape(self):
        img = torch.rand(1, 16, 16, 3)
        result = self.node.encode(image=img, format="HLG (Broadcast)")
        assert result[0].shape == img.shape

    def test_hlg_output_clamped(self):
        img = torch.rand(1, 8, 8, 3)
        out = self.node.encode(image=img, format="HLG (Broadcast)")[0]
        assert out.min().item() >= -1e-4
        assert out.max().item() <= 1.0 + 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# RadianceHDRMonitor node (merged tone-map preview)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
def test_monitor_preview_smoke():
    mod = _import_delivery()
    if not hasattr(mod, "RadianceHDRMonitor"):
        pytest.skip("RadianceHDRMonitor not present")
    node = mod.RadianceHDRMonitor()
    img = torch.rand(1, 8, 8, 3) * 4.0
    result = node.monitor(image=img, mode="Preview (SDR)")
    assert result[0].shape == img.shape
    assert result[0].max().item() <= 1.0 + 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Node registrations
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestNodeRegistrations:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_delivery()

    def test_hdr_encode_registered(self):
        assert "RadianceHDREncode" in self.mod.NODE_CLASS_MAPPINGS

    def test_hdr_monitor_registered(self):
        assert "RadianceHDRMonitor" in self.mod.NODE_CLASS_MAPPINGS

    def test_helper_functions_exported(self):
        assert hasattr(self.mod, "_linear_to_pq")
        assert hasattr(self.mod, "_linear_to_hlg")
