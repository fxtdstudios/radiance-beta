"""
test_sdr_to_hdr_universal.py — ◎ Radiance SDR → HDR Universal.

Contract tests run under the conftest torch stub (no GPU needed).
Math tests require real torch and self-skip on the lightweight CI matrix.
"""
import importlib
import sys
import unittest

import numpy as np


def _real_torch() -> bool:
    try:
        import torch
        return isinstance(getattr(torch, "__version__", None), str)
    except Exception:
        return False


HAS_TORCH = _real_torch()
skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")


# ─────────────────────────────────────────────────────────────────────────────
#  Contract (stub-safe)
# ─────────────────────────────────────────────────────────────────────────────

class TestContract(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("radiance.nodes.hdr.uplift_universal")

    def test_registered_in_group_mappings(self):
        group = importlib.import_module("radiance.nodes.hdr")
        self.assertIn("RadianceSDRToHDRUniversal", group.NODE_CLASS_MAPPINGS)
        self.assertIn("RadianceSDRToHDRUniversal", group.NODE_DISPLAY_NAME_MAPPINGS)

    def test_input_types_contract(self):
        it = self.mod.RadianceSDRToHDRUniversal.INPUT_TYPES()
        req = it["required"]
        for key in ("image", "inverse_oetf", "peak_nits", "knee_mode", "knee",
                    "shoulder_gamma", "temporal_smoothing", "output_encoding"):
            self.assertIn(key, req)
        self.assertIn("PQ (HDR10)", req["output_encoding"][0])
        self.assertIn("HLG", req["output_encoding"][0])
        self.assertIn("Linear", req["output_encoding"][0])

    def test_optional_rudra_inputs(self):
        opt = self.mod.RadianceSDRToHDRUniversal.INPUT_TYPES()["optional"]
        self.assertEqual(opt["vae"][0], "VAE")
        self.assertIn("rudra_turbo", opt["rudra_size"][0])
        self.assertIn("rudra_full", opt["rudra_size"][0])
        self.assertIn("rudra_blend", opt)

    def test_node_metadata(self):
        cls = self.mod.RadianceSDRToHDRUniversal
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "MASK"))
        self.assertEqual(cls.FUNCTION, "convert")
        self.assertTrue(cls.CATEGORY.endswith("HDR"))


# ─────────────────────────────────────────────────────────────────────────────
#  Math (real torch only)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestMath(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.mod = importlib.import_module("radiance.nodes.hdr.uplift_universal")
        self.node = self.mod.RadianceSDRToHDRUniversal()

    def _gradient(self, b=1):
        """[b,4,8,3] linear gradient 0→1 identical across channels."""
        t = self.torch.linspace(0.0, 1.0, 32).reshape(1, 4, 8, 1)
        return t.expand(b, 4, 8, 3).contiguous()

    def test_below_knee_preserved_linear(self):
        img = self._gradient()
        out, _ = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
                                   0.0, "Linear")
        luma_in = img[..., 0]
        below = luma_in <= 0.74
        self.assertTrue(self.torch.allclose(out[..., 0][below],
                                            luma_in[below], atol=1e-4))

    def test_peak_reaches_target(self):
        img = self.torch.ones(1, 4, 4, 3)
        out, _ = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
                                   0.0, "Linear")
        self.assertAlmostEqual(float(out.max()), 10.0, places=3)  # 1000/100 nits

    def test_monotonic(self):
        img = self._gradient()
        out, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 2.0,
                                   0.0, "Linear")
        flat = out[..., 0].flatten()
        self.assertTrue(bool((flat[1:] >= flat[:-1] - 1e-5).all()))

    def test_hue_preserved(self):
        img = self.torch.tensor([[[[0.9, 0.6, 0.3]]]])
        out, _ = self.node.convert(img, "None", 1000.0, "manual", 0.3, 1.0,
                                   0.0, "Linear")
        r, g, b = (float(out[0, 0, 0, i]) for i in range(3))
        self.assertAlmostEqual(r / g, 0.9 / 0.6, places=3)
        self.assertAlmostEqual(g / b, 0.6 / 0.3, places=3)

    def test_pq_encoding_known_value(self):
        # linear 1.0 == 100 nits → PQ ≈ 0.5081 (ST.2084)
        img = self.torch.full((1, 2, 2, 3), 0.75)  # below knee → unchanged luma
        out, _ = self.node.convert(img, "None", 1000.0, "manual", 0.9, 1.6,
                                   0.0, "PQ (HDR10)")
        lin = 0.75
        L = lin * 100.0 / 10000.0
        m1, m2 = 0.1593017578125, 78.84375
        c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
        expected = ((c1 + c2 * L**m1) / (1.0 + c3 * L**m1)) ** m2
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), expected, places=4)

    def test_hlg_bounded(self):
        img = self._gradient()
        out, _ = self.node.convert(img, "sRGB", 1000.0, "adaptive", 0.95, 1.6,
                                   0.0, "HLG")
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_mask_range_and_zero_below_knee(self):
        img = self._gradient()
        _, mask = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
                                    0.0, "Linear")
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)
        below = img[..., 0] <= 0.74
        self.assertAlmostEqual(float(mask[below].abs().max()), 0.0, places=5)

    def test_temporal_smoothing_reduces_knee_variance(self):
        # alternating dark / bright frames → adaptive knee flickers without EMA
        bright = self._gradient()
        dark = bright * 0.3
        video = self.torch.cat([dark, bright] * 4, dim=0)  # 8 frames
        luma = self.mod._luma(video)
        raw = self.mod._adaptive_knees(luma, 0.9, 0.0)
        smooth = self.mod._adaptive_knees(luma, 0.9, 0.9)
        self.assertLess(float(smooth.var()), float(raw.var()))

    def test_single_frame_hwc_accepted(self):
        img = self.torch.rand(4, 4, 3)
        out, mask = self.node.convert(img, "sRGB", 1000.0, "adaptive", 0.9,
                                      1.6, 0.85, "Linear")
        self.assertEqual(tuple(out.shape), (1, 4, 4, 3))
        self.assertEqual(tuple(mask.shape), (1, 4, 4))

    def test_alpha_passthrough(self):
        img = self.torch.rand(1, 4, 4, 4)
        alpha = img[..., 3].clone()
        out, _ = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
                                   0.0, "Linear")
        self.assertTrue(self.torch.allclose(out[..., 3], alpha))


@skip_no_torch
class TestRudraPath(unittest.TestCase):
    """RUDRA integration — VAE/decoder mocked, real torch math."""

    def setUp(self):
        import torch
        import radiance.fast_vae as fv
        self.torch = torch
        self.fv = fv
        self.mod = importlib.import_module("radiance.nodes.hdr.uplift_universal")
        self.node = self.mod.RadianceSDRToHDRUniversal()
        self._orig = (fv.load_radiance_decoder_weights,
                      fv.decode_to_linear_realtime,
                      fv.detect_rudra_model_type)

    def tearDown(self):
        (self.fv.load_radiance_decoder_weights,
         self.fv.decode_to_linear_realtime,
         self.fv.detect_rudra_model_type) = self._orig

    class _FakeVAE:
        scale_factor = 0.18215
        def encode(self, pixels):
            import torch
            b, h, w, _ = pixels.shape
            return torch.randn(b, 16, max(h // 8, 1), max(w // 8, 1))

    def _img(self):
        t = self.torch.linspace(0.0, 1.0, 64).reshape(1, 8, 8, 1)
        return t.expand(1, 8, 8, 3).contiguous()

    def test_fallback_when_no_checkpoint(self):
        """loader returns None → identical to pure-math output, no exception."""
        self.fv.load_radiance_decoder_weights = lambda **kw: None
        self.fv.detect_rudra_model_type = lambda *a, **kw: "flux"
        img = self._img()
        base, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                                    0.0, "Linear")
        out, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                                   0.0, "Linear", vae=self._FakeVAE())
        self.assertTrue(self.torch.allclose(out, base))

    def test_rudra_blended_into_highlights(self):
        """decoder available → highlight region differs from math, shadows don't."""
        torch = self.torch
        rec_value = 7.0
        self.fv.detect_rudra_model_type = lambda *a, **kw: "flux"
        self.fv.load_radiance_decoder_weights = (
            lambda **kw: torch.nn.Identity())
        def fake_decode(latent, decoder, **kw):
            b = latent.shape[0]
            h, w = latent.shape[-2] * 8, latent.shape[-1] * 8
            return torch.full((b, h, w, 3), rec_value)
        self.fv.decode_to_linear_realtime = fake_decode

        img = self._img()
        base, mask = self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                                       0.0, "Linear")
        out, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                                   0.0, "Linear", vae=self._FakeVAE(),
                                   rudra_blend=1.0)
        shadows = mask < 1e-6
        highlights = mask > 0.2
        self.assertTrue(torch.allclose(out[..., 0][shadows],
                                       base[..., 0][shadows], atol=1e-4))
        self.assertTrue(bool((out[..., 0][highlights]
                              != base[..., 0][highlights]).any()))

    def test_blend_zero_disables_rudra(self):
        called = {"n": 0}
        def loader(**kw):
            called["n"] += 1
            return None
        self.fv.load_radiance_decoder_weights = loader
        img = self._img()
        self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                          0.0, "Linear", vae=self._FakeVAE(), rudra_blend=0.0)
        self.assertEqual(called["n"], 0)


if __name__ == "__main__":
    unittest.main()
