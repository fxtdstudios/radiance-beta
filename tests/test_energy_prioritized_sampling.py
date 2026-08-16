"""
Energy-Prioritized Sampling: producer → conditioning → sampler reader.

Issue #40: RadianceSamplerPro has read `radiance_energy_mask` out of the
positive CONDITIONING since v3.1, but no node ever wrote that key, so the
feature could not be reached from a graph. The two tests that existed
(`TestEnergyPrioritizedSampling` in test_sampler_regression.py) hid this:
one re-implemented the parser loop inside the test body and asserted on its
own copy, the other did the modifier arithmetic inline. Both passed without
executing a single line of nodes_sampler.py.

These tests drive the real producer node and the real reader, and pin the
two crashes the reader was carrying: a 4-way shape unpack that dies on the
5-D latents every video model produces, and an expand() that dies on a mask
whose batch is neither 1 nor B.
"""
import os
import sys

import pytest

try:
    import torch as _t
    _HAS_TORCH = isinstance(getattr(_t, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    pytest.skip("torch not installed", allow_module_level=True)

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.generate.energy import (
    ENERGY_MASK_KEY,
    ENERGY_PRIORITY_KEY,
    RadianceEnergyMask,
)
from radiance.nodes_sampler import _find_energy_mask, _make_energy_cfg_patch


def _conditioning(batch=1):
    return [(torch.zeros(batch, 77, 2048), {"pooled_output": torch.zeros(batch, 1280)})]


def _half_mask(h=64, w=64):
    """Left half black, right half white."""
    m = torch.zeros(1, h, w)
    m[:, :, w // 2:] = 1.0
    return m


def _args(cond, uncond, cfg=7.0):
    return {
        "cond_denoised": cond,
        "uncond_denoised": uncond,
        "cond_scale": cfg,
        "denoised": uncond + cfg * (cond - uncond),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Producer
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyMaskNode:

    def test_the_node_writes_the_keys_the_sampler_reads(self):
        """The whole point of #40: producer output must satisfy the reader."""
        out, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=0.75)

        mask, priority = _find_energy_mask(out)

        assert mask is not None, "the sampler did not find the mask the node attached"
        assert priority == pytest.approx(0.75)

    def test_every_entry_carries_the_keys(self):
        cond = _conditioning() + _conditioning()
        out, _ = RadianceEnergyMask().apply(cond, _half_mask(), priority=0.5)

        assert all(ENERGY_MASK_KEY in meta for _, meta in out)
        assert all(meta[ENERGY_PRIORITY_KEY] == 0.5 for _, meta in out)

    def test_the_input_conditioning_is_not_mutated(self):
        """A conditioning list is routinely fanned out to several branches."""
        cond = _conditioning()
        RadianceEnergyMask().apply(cond, _half_mask(), priority=0.5)

        assert ENERGY_MASK_KEY not in cond[0][1]

    def test_the_returned_mask_is_the_conditioned_one(self):
        out, preview = RadianceEnergyMask().apply(
            _conditioning(), _half_mask(), priority=0.5, invert=True
        )
        assert torch.equal(out[0][1][ENERGY_MASK_KEY], preview)

    def test_invert_swaps_the_region(self):
        _, m = RadianceEnergyMask().apply(
            _conditioning(), _half_mask(), priority=0.5, invert=True
        )
        assert m[0, 0, 0].item() == pytest.approx(1.0)
        assert m[0, 0, -1].item() == pytest.approx(0.0)

    def test_normalize_rescales_an_unnormalised_pass(self):
        """Luminance and depth passes are not guaranteed to sit in [0,1]."""
        raw = torch.full((1, 8, 8), 4.0)
        raw[:, :, 4:] = 12.0

        _, m = RadianceEnergyMask().apply(
            _conditioning(), raw, priority=0.5, normalize=True
        )
        assert m.min().item() == pytest.approx(0.0)
        assert m.max().item() == pytest.approx(1.0)

    def test_out_of_range_masks_are_clamped(self):
        """An unclamped mask would flip the CFG modifier negative."""
        raw = torch.tensor([[[-3.0, 5.0]]])
        _, m = RadianceEnergyMask().apply(_conditioning(), raw, priority=0.5)

        assert m.min().item() >= 0.0
        assert m.max().item() <= 1.0

    def test_gain_hardens_a_soft_mask(self):
        soft = torch.full((1, 4, 4), 0.25)
        _, m = RadianceEnergyMask().apply(_conditioning(), soft, priority=0.5, gain=4.0)
        assert m.mean().item() == pytest.approx(1.0)

    def test_a_non_tensor_mask_is_rejected_at_the_node(self):
        with pytest.raises(ValueError):
            RadianceEnergyMask().apply(_conditioning(), "not a mask", priority=0.5)

    def test_empty_conditioning_is_rejected(self):
        with pytest.raises(ValueError):
            RadianceEnergyMask().apply([], _half_mask(), priority=0.5)

    def test_node_contract(self):
        cls = RadianceEnergyMask
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
        assert hasattr(cls, cls.FUNCTION)
        assert "conditioning" in cls.INPUT_TYPES()["required"]
        assert "mask" in cls.INPUT_TYPES()["required"]

    def test_the_node_is_published_to_comfyui(self):
        """#40 was reader-only code. A producer nobody can place is the same bug."""
        from radiance.nodes.generate import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        assert NODE_CLASS_MAPPINGS.get("RadianceEnergyMask") is RadianceEnergyMask
        assert "RadianceEnergyMask" in NODE_DISPLAY_NAME_MAPPINGS


# ─────────────────────────────────────────────────────────────────────────────
#  Reader
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyMaskParser:

    def test_no_mask_returns_none(self):
        assert _find_energy_mask(_conditioning()) == (None, 0.0)

    def test_non_list_conditioning_is_tolerated(self):
        assert _find_energy_mask(None) == (None, 0.0)

    def test_a_junk_mask_value_is_skipped_not_crashed(self):
        positive = [(torch.zeros(1, 77, 2048), {ENERGY_MASK_KEY: "nonsense"})]
        assert _find_energy_mask(positive) == (None, 0.0)

    def test_a_non_numeric_priority_falls_back(self):
        positive = [(torch.zeros(1, 77, 2048), {
            ENERGY_MASK_KEY: _half_mask(),
            ENERGY_PRIORITY_KEY: "loud",
        })]
        _, priority = _find_energy_mask(positive)
        assert priority == pytest.approx(1.0)

    def test_the_first_carrier_wins(self):
        first, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=0.25)
        second, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=2.0)

        _, priority = _find_energy_mask(first + second)
        assert priority == pytest.approx(0.25)


# ─────────────────────────────────────────────────────────────────────────────
#  CFG patch
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyCfgPatch:

    def test_guidance_is_boosted_inside_the_mask_only(self):
        cond = torch.full((1, 4, 8, 8), 5.0)
        uncond = torch.ones(1, 4, 8, 8)
        mask = torch.zeros(1, 8, 8)
        mask[:, :, 4:] = 1.0

        patch = _make_energy_cfg_patch(mask, 0.5)
        out = patch(_args(cond, uncond, cfg=1.0))

        # cfg=1 → denoised == cond_eps. Unmasked keeps cond (5.0); masked gets
        # uncond + (cond-uncond)*1.5 = 1 + 4*1.5 = 7.0
        assert out[0, 0, 0, 0].item() == pytest.approx(5.0)
        assert out[0, 0, 0, 7].item() == pytest.approx(7.0)

    def test_a_zero_mask_is_a_no_op(self):
        cond = torch.randn(1, 4, 8, 8)
        uncond = torch.randn(1, 4, 8, 8)
        args = _args(cond, uncond)

        patch = _make_energy_cfg_patch(torch.zeros(1, 8, 8), 0.9)
        assert torch.allclose(patch(args), args["denoised"], atol=1e-5)

    def test_negative_priority_suppresses_the_region(self):
        cond = torch.full((1, 4, 4, 4), 5.0)
        uncond = torch.ones(1, 4, 4, 4)

        patch = _make_energy_cfg_patch(torch.ones(1, 4, 4), -0.5)
        out = patch(_args(cond, uncond, cfg=1.0))

        # 1 + 4*0.5 = 3.0, i.e. half the guidance
        assert out[0, 0, 0, 0].item() == pytest.approx(3.0)

    def test_video_latents_do_not_crash(self):
        """The old reader unpacked B, C, H, W and raised ValueError on 5-D."""
        cond = torch.full((1, 16, 5, 8, 8), 5.0)
        uncond = torch.ones(1, 16, 5, 8, 8)
        mask = torch.zeros(1, 8, 8)
        mask[:, :, 4:] = 1.0

        patch = _make_energy_cfg_patch(mask, 0.5)
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out.shape == cond.shape
        # The mask is spatial, so it applies identically on every frame.
        for t in range(5):
            assert out[0, 0, t, 0, 0].item() == pytest.approx(5.0)
            assert out[0, 0, t, 0, 7].item() == pytest.approx(7.0)

    def test_a_mask_at_image_resolution_is_resized_to_the_latent(self):
        cond = torch.full((1, 4, 64, 64), 5.0)
        uncond = torch.ones(1, 4, 64, 64)

        patch = _make_energy_cfg_patch(_half_mask(512, 512), 0.5)
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out[0, 0, 0, 0].item() == pytest.approx(5.0, abs=1e-3)
        assert out[0, 0, 0, 63].item() == pytest.approx(7.0, abs=1e-3)

    def test_a_single_mask_broadcasts_over_a_latent_batch(self):
        cond = torch.full((4, 4, 8, 8), 5.0)
        uncond = torch.ones(4, 4, 8, 8)

        patch = _make_energy_cfg_patch(torch.ones(1, 8, 8), 0.5)
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out.shape == cond.shape
        assert out[3, 0, 0, 0].item() == pytest.approx(7.0)

    def test_a_mismatched_mask_batch_does_not_crash(self):
        """expand() cannot broadcast 3 → 4; the old reader tried anyway."""
        cond = torch.full((4, 4, 8, 8), 5.0)
        uncond = torch.ones(4, 4, 8, 8)

        patch = _make_energy_cfg_patch(torch.ones(3, 8, 8), 0.5)
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out.shape == cond.shape

    def test_an_existing_cfg_function_is_wrapped_not_replaced(self):
        """EPS must compose with guidance rescale and the SDR anchor."""
        seen = {}

        def existing(args):
            seen["cond_denoised"] = args["cond_denoised"]
            return args["denoised"] * 0.0

        cond = torch.full((1, 4, 4, 4), 5.0)
        uncond = torch.ones(1, 4, 4, 4)

        patch = _make_energy_cfg_patch(torch.ones(1, 4, 4), 0.5, existing_cfg_fn=existing)
        out = patch(_args(cond, uncond, cfg=1.0))

        assert torch.allclose(out, torch.zeros_like(out)), "the existing cfg fn did not run"
        assert seen["cond_denoised"][0, 0, 0, 0].item() == pytest.approx(7.0), \
            "the downstream cfg fn did not receive the EPS-boosted prediction"

    def test_mismatched_cond_uncond_shapes_pass_through(self):
        args = {
            "cond_denoised": torch.randn(1, 4, 8, 8),
            "uncond_denoised": torch.randn(1, 4, 4, 4),
            "cond_scale": 7.0,
            "denoised": torch.zeros(1, 4, 8, 8),
        }

        patch = _make_energy_cfg_patch(torch.ones(1, 8, 8), 0.5)
        assert torch.equal(patch(args), args["denoised"])

    def test_the_resized_mask_is_reused_across_steps(self):
        """The mask resize is per-step work that only depends on geometry."""
        cond = torch.full((1, 4, 8, 8), 5.0)
        uncond = torch.ones(1, 4, 8, 8)

        patch = _make_energy_cfg_patch(_half_mask(256, 256), 0.5)
        first = patch(_args(cond, uncond, cfg=1.0))

        with_interpolate_banned = torch.nn.functional.interpolate
        try:
            def _boom(*a, **k):
                raise AssertionError("mask was re-interpolated on a later step")
            torch.nn.functional.interpolate = _boom
            second = patch(_args(cond, uncond, cfg=1.0))
        finally:
            torch.nn.functional.interpolate = with_interpolate_banned

        assert torch.equal(first, second)
