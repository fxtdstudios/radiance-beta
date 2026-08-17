"""
test_sdr_to_hdr_universal.py — ◎ Radiance SDR → HDR Universal.

Contract tests run under the conftest torch stub (no GPU needed).
Math tests require real torch and self-skip on the lightweight CI matrix.
"""
import importlib
import unittest


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
        self.assertIn("RadianceSDRToHDRRecover", group.NODE_CLASS_MAPPINGS)
        self.assertIn("RadianceSDRToHDRRecover", group.NODE_DISPLAY_NAME_MAPPINGS)

    def test_input_types_contract(self):
        it = self.mod.RadianceSDRToHDRUniversal.INPUT_TYPES()
        req = it["required"]
        for key in ("image", "inverse_oetf", "peak_nits", "knee_mode", "knee",
                    "shoulder_gamma", "temporal_smoothing", "output_encoding"):
            self.assertIn(key, req)
        self.assertIn("PQ (HDR10)", req["output_encoding"][0])
        self.assertIn("HLG", req["output_encoding"][0])
        self.assertIn("Linear", req["output_encoding"][0])
        self.assertIn("Linear ACES2065-1 (AP0)", req["output_encoding"][0])

    def test_optional_rudra_inputs(self):
        opt = self.mod.RadianceSDRToHDRUniversal.INPUT_TYPES()["optional"]
        self.assertEqual(opt["vae"][0], "VAE")
        self.assertIn("rudra_turbo", opt["rudra_size"][0])
        self.assertIn("rudra_full", opt["rudra_size"][0])
        self.assertIn("rudra_blend", opt)
        self.assertIn("batch_mode", opt)
        self.assertIn("shadow_threshold", opt)
        self.assertEqual(opt["processing_mode"][0], ["Expand", "Recover", "Hybrid"])
        self.assertIn("highlight_threshold", opt)
        self.assertEqual(opt["temporal_window"][0], [5, 7, 9])
        self.assertIn("temporal_checkpoint", opt)

    def test_node_metadata(self):
        cls = self.mod.RadianceSDRToHDRUniversal
        # `report` is appended last so links, which ComfyUI stores by index,
        # survive. It names the path that actually ran — Recover and Hybrid are
        # bit-identical to Expand without a checkpoint, and used to say so only
        # in the console.
        self.assertEqual(cls.RETURN_TYPES,
                         ("IMAGE", "MASK", "MASK", "MASK", "MASK", "STRING"))
        self.assertEqual(cls.RETURN_NAMES, ("image", "highlight_mask", "shadow_mask",
                                            "highlight_confidence", "shadow_confidence",
                                            "report"))
        self.assertEqual(cls.FUNCTION, "convert")
        self.assertTrue(cls.CATEGORY.endswith("HDR"))

    def test_recover_node_contract(self):
        cls = self.mod.RadianceSDRToHDRRecover
        req = cls.INPUT_TYPES()["required"]
        self.assertEqual(cls.INPUT_TYPES()["optional"]["vae"][0], "VAE")
        for key in ("highlight_threshold", "shadow_threshold",
                    "highlight_strength", "shadow_strength"):
            self.assertIn(key, req)
        self.assertEqual(cls.INPUT_TYPES()["optional"]["temporal_window"][0], [5, 7, 9])
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "MASK", "MASK", "MASK", "MASK"))
        self.assertEqual(cls.RETURN_NAMES,
                         ("image", "highlight_mask", "shadow_mask",
                          "highlight_confidence", "shadow_confidence"))
        self.assertEqual(cls.FUNCTION, "recover")


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
        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
                                      0.0, "Linear")
        luma_in = img[..., 0]
        below = luma_in <= 0.74
        self.assertTrue(self.torch.allclose(out[..., 0][below],
                                            luma_in[below], atol=1e-4))

    def test_sdr_white_reaches_reference_white_not_the_display_peak(self):
        """Rewritten in 3.4 — this used to assert the defect.

        It required SDR code 1.0 to come out at `peak_nits`: a white shirt at
        the full mastering peak, 1000 nits, against the 203 that ITU-R BT.2408
        defines as HDR Reference White. `reference_white_nits` separates the
        two, and the old behaviour stays reachable by setting it to peak_nits,
        which the second half asserts.
        """
        img = self.torch.ones(1, 4, 4, 3)

        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
                                            0.0, "Linear")
        self.assertAlmostEqual(float(out.max()), 2.03, places=3)   # 203 nits

        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
                                            0.0, "Linear",
                                            reference_white_nits=1000.0)
        self.assertAlmostEqual(float(out.max()), 10.0, places=3)   # 1000 nits

    def test_monotonic(self):
        img = self._gradient()
        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 2.0,
                                      0.0, "Linear")
        flat = out[..., 0].flatten()
        self.assertTrue(bool((flat[1:] >= flat[:-1] - 1e-5).all()))

    def test_hue_preserved(self):
        img = self.torch.tensor([[[[0.9, 0.6, 0.3]]]])
        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.3, 1.0,
                                      0.0, "Linear")
        r, g, b = (float(out[0, 0, 0, i]) for i in range(3))
        self.assertAlmostEqual(r / g, 0.9 / 0.6, places=3)
        self.assertAlmostEqual(g / b, 0.6 / 0.3, places=3)

    def test_pq_encoding_known_value(self):
        # linear 1.0 == 100 nits → PQ ≈ 0.5081 (ST.2084)
        img = self.torch.full((1, 2, 2, 3), 0.75)  # below knee → unchanged luma
        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.9, 1.6,
                                      0.0, "PQ (HDR10)")
        lin = 0.75
        L = lin * 100.0 / 10000.0
        m1, m2 = 0.1593017578125, 78.84375
        c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
        expected = ((c1 + c2 * L**m1) / (1.0 + c3 * L**m1)) ** m2
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), expected, places=4)

    def test_hlg_bounded(self):
        img = self._gradient()
        out, _, _, _, _, _ = self.node.convert(img, "sRGB", 1000.0, "adaptive", 0.95, 1.6,
                                      0.0, "HLG")
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_mask_range_and_zero_below_knee(self):
        img = self._gradient()
        _, mask, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
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
        out, mask, shadows, h_conf, s_conf, _ = self.node.convert(img, "sRGB", 1000.0, "adaptive", 0.9,
                                               1.6, 0.85, "Linear")
        self.assertEqual(tuple(out.shape), (1, 4, 4, 3))
        self.assertEqual(tuple(mask.shape), (1, 4, 4))
        self.assertEqual(tuple(shadows.shape), (1, 4, 4))
        self.assertEqual(tuple(h_conf.shape), (1, 4, 4))
        self.assertEqual(tuple(s_conf.shape), (1, 4, 4))

    def test_alpha_passthrough(self):
        img = self.torch.rand(1, 4, 4, 4)
        alpha = img[..., 3].clone()
        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.75, 1.6,
                                      0.0, "Linear")
        self.assertTrue(self.torch.allclose(out[..., 3], alpha))

    def test_pq_converts_rec709_primaries_to_rec2020(self):
        from radiance.color.ops import M_REC709_TO_BT2020, apply_matrix_3x3
        img = self.torch.tensor([[[[1.0, 0.0, 0.0]]]])
        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.99, 1.6,
                                      0.0, "PQ (HDR10)")
        rec2020 = apply_matrix_3x3(img, M_REC709_TO_BT2020).clamp(min=0.0)
        expected = self.mod._torch_pq_encode(rec2020)
        self.assertTrue(self.torch.allclose(out, expected, atol=1e-6))
        self.assertGreater(float(out[..., 1]), 1e-3)
        self.assertGreater(float(out[..., 2]), 1e-3)

    def test_aces2065_output_uses_ap0_primaries(self):
        from radiance.color.ops import M_REC709_TO_ACES2065_1, apply_matrix_3x3
        img = self.torch.tensor([[[[1.0, 0.25, 0.0]]]])
        out, _, _, _, _, _ = self.node.convert(
            img, "None", 1000.0, "manual", 0.99, 1.6, 0.0,
            "Linear ACES2065-1 (AP0)",
        )
        expected = apply_matrix_3x3(img, M_REC709_TO_ACES2065_1)
        self.assertTrue(self.torch.allclose(out, expected, atol=1e-6))

    def test_independent_batch_is_order_independent(self):
        bright = self._gradient()
        dark = bright * 0.2
        alone, _, _, _, _, _ = self.node.convert(
            bright, "None", 1000.0, "adaptive", 0.75, 1.6, 0.95, "Linear",
            batch_mode="Independent Images",
        )
        batch, _, _, _, _, _ = self.node.convert(
            self.torch.cat([dark, bright], dim=0), "None", 1000.0,
            "adaptive", 0.75, 1.6, 0.95, "Linear",
            batch_mode="Independent Images",
        )
        self.assertTrue(self.torch.allclose(batch[1], alone[0], atol=1e-6))

    def test_video_batch_uses_temporal_smoothing(self):
        bright = self._gradient()
        dark = bright * 0.2
        independent, _, _, _, _, _ = self.node.convert(
            self.torch.cat([dark, bright], dim=0), "None", 1000.0,
            "adaptive", 0.75, 1.6, 0.95, "Linear",
            batch_mode="Independent Images",
        )
        video, _, _, _, _, _ = self.node.convert(
            self.torch.cat([dark, bright], dim=0), "None", 1000.0,
            "adaptive", 0.75, 1.6, 0.95, "Linear",
            batch_mode="Video Frames",
        )
        self.assertFalse(self.torch.allclose(video[1], independent[1]))

    def test_nan_and_infinity_are_sanitized(self):
        img = self.torch.tensor([[[[float("nan"), float("inf"), float("-inf")]]]])
        out, highlights, shadows, h_conf, s_conf, _ = self.node.convert(
            img, "None", 1000.0, "manual", 0.75, 1.6, 0.0, "Linear",
        )
        self.assertTrue(bool(self.torch.isfinite(out).all()))
        self.assertTrue(bool(self.torch.isfinite(highlights).all()))
        self.assertTrue(bool(self.torch.isfinite(shadows).all()))
        self.assertTrue(bool(self.torch.isfinite(h_conf).all()))
        self.assertTrue(bool(self.torch.isfinite(s_conf).all()))

    def test_saturated_colors_remain_finite_and_peak_bounded(self):
        img = self.torch.tensor([[[[1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0]]]])
        out, _, _, _, _, _ = self.node.convert(
            img, "None", 1000.0, "manual", 0.05, 1.0, 0.0, "Linear",
        )
        self.assertTrue(bool(self.torch.isfinite(out).all()))
        self.assertLessEqual(float(self.mod._luma(out).max()), 10.0 + 1e-5)

    def test_shadow_mask_has_independent_threshold(self):
        img = self.torch.tensor([[[[0.0, 0.0, 0.0],
                                    [0.05, 0.05, 0.05],
                                    [0.2, 0.2, 0.2]]]])
        _, _, shadows, _, _, _ = self.node.convert(
            img, "None", 1000.0, "manual", 0.75, 1.6, 0.0, "Linear",
            shadow_threshold=0.1,
        )
        self.assertAlmostEqual(float(shadows[0, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(shadows[0, 0, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(shadows[0, 0, 2]), 0.0, places=6)


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
        base, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                                       0.0, "Linear")
        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                                      0.0, "Linear", vae=self._FakeVAE())
        self.assertTrue(self.torch.allclose(out, base))

    def test_rudra_blended_only_into_recovery_regions(self):
        """Hybrid changes clipped highlights/shadows, not clean midtones."""
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
        base, mask, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                                          0.0, "Linear")
        out, _, _, _, _, _ = self.node.convert(img, "None", 1000.0, "manual", 0.5, 1.6,
                                      0.0, "Linear", vae=self._FakeVAE(),
                                      rudra_blend=1.0)
        clean_midtones = (img[..., 0] > 0.2) & (img[..., 0] < 0.8)
        recovery_regions = (img[..., 0] < 0.05) | (img[..., 0] > 0.98)
        self.assertTrue(torch.allclose(out[..., 0][clean_midtones],
                                       base[..., 0][clean_midtones], atol=1e-4))
        self.assertTrue(bool((out[..., 0][recovery_regions]
                              != base[..., 0][recovery_regions]).any()))

    def test_expand_mode_never_loads_rudra(self):
        called = {"n": 0}
        def loader(**kw):
            called["n"] += 1
            return self.torch.nn.Identity()
        self.fv.load_radiance_decoder_weights = loader
        self.node.convert(
            self._img(), "None", 1000.0, "manual", 0.5, 1.6, 0.0,
            "Linear", vae=self._FakeVAE(), processing_mode="Expand",
        )
        self.assertEqual(called["n"], 0)

    def test_universal_modes_have_distinct_products(self):
        torch = self.torch
        self.fv.detect_rudra_model_type = lambda *a, **kw: "flux"
        self.fv.load_radiance_decoder_weights = lambda **kw: torch.nn.Identity()

        def fake_decode(latent, decoder, **kw):
            b = latent.shape[0]
            h, w = latent.shape[-2] * 8, latent.shape[-1] * 8
            return torch.full((b, h, w, 3), 2.0)

        self.fv.decode_to_linear_realtime = fake_decode
        args = (self._img(), "None", 1000.0, "manual", 0.5, 1.6, 0.0, "Linear")
        expand, _, _, _, _, _ = self.node.convert(
            *args, vae=self._FakeVAE(), processing_mode="Expand",
        )
        recover, _, _, _, _, _ = self.node.convert(
            *args, vae=self._FakeVAE(), processing_mode="Recover",
        )
        hybrid, _, _, _, _, _ = self.node.convert(
            *args, vae=self._FakeVAE(), processing_mode="Hybrid",
        )
        self.assertFalse(torch.allclose(expand, recover))
        self.assertFalse(torch.allclose(recover, hybrid))

    def test_recover_node_reconstructs_highlights_and_shadows(self):
        torch = self.torch
        self.fv.detect_rudra_model_type = lambda *a, **kw: "flux"
        self.fv.load_radiance_decoder_weights = lambda **kw: torch.nn.Identity()

        def fake_decode(latent, decoder, **kw):
            b = latent.shape[0]
            h, w = latent.shape[-2] * 8, latent.shape[-1] * 8
            return torch.full((b, h, w, 3), 2.0)

        self.fv.decode_to_linear_realtime = fake_decode
        recover = self.mod.RadianceSDRToHDRRecover()
        img = self._img()
        out, highlights, shadows, h_conf, s_conf = recover.recover(
            img, "None", 1000.0, 0.98, 0.05,
            1.0, 1.0, "Linear", "rudra_turbo", vae=self._FakeVAE(),
        )
        clean = (highlights < 1e-6) & (shadows < 1e-6)
        affected = (highlights > 0.0) | (shadows > 0.0)
        self.assertTrue(torch.allclose(out[..., 0][clean], img[..., 0][clean], atol=1e-4))
        self.assertTrue(bool((out[..., 0][affected] != img[..., 0][affected]).any()))
        self.assertTrue(self.torch.allclose(h_conf, highlights))
        self.assertTrue(self.torch.allclose(s_conf, shadows))

    def test_recover_node_requires_checkpoint(self):
        self.fv.load_radiance_decoder_weights = lambda **kw: None
        recover = self.mod.RadianceSDRToHDRRecover()
        with self.assertRaisesRegex(RuntimeError, "compatible trained RUDRA checkpoint"):
            recover.recover(
                self._img(), "None", 1000.0, 0.98,
                0.05, 1.0, 1.0, "Linear", "rudra_turbo", vae=self._FakeVAE(),
            )

    def test_recover_video_requires_temporal_checkpoint(self):
        recover = self.mod.RadianceSDRToHDRRecover()
        with self.assertRaisesRegex(RuntimeError, "Phase 3 temporal checkpoint"):
            recover.recover(
                self._img().expand(5, 8, 8, 3), "None",
                1000.0, 0.98, 0.05, 1.0, 1.0, "Linear", "rudra_turbo",
            )

    def test_rudra_output_respects_peak_nits(self):
        """Extreme learned radiance is soft-limited to the mastering peak."""
        torch = self.torch
        self.fv.detect_rudra_model_type = lambda *a, **kw: "flux"
        self.fv.load_radiance_decoder_weights = lambda **kw: torch.nn.Identity()

        def fake_decode(latent, decoder, **kw):
            b = latent.shape[0]
            h, w = latent.shape[-2] * 8, latent.shape[-1] * 8
            return torch.full((b, h, w, 3), 1000.0)

        self.fv.decode_to_linear_realtime = fake_decode
        out, _, _, _, _, _ = self.node.convert(
            self._img(), "None", 200.0, "manual", 0.5, 1.0, 0.0,
            "Linear", vae=self._FakeVAE(), rudra_blend=1.0,
        )
        self.assertLessEqual(float(self.mod._luma(out).max()), 2.0 + 1e-5)

    def test_multiframe_input_skips_rudra_before_any_compute(self):
        """Video (B>1) + VAE → no encode, no loader call, math output returned.
        Regression: WAN's video VAE compresses 81 frames → 21; the old code ran
        the full encode+decode before discovering the mismatch."""
        encode_calls = {"n": 0}
        loader_calls = {"n": 0}

        class _CountingVAE:
            scale_factor = 0.18215
            def encode(self, pixels):
                encode_calls["n"] += 1
                import torch
                b, h, w, _ = pixels.shape
                return torch.randn(b, 16, max(h // 8, 1), max(w // 8, 1))

        def loader(**kw):
            loader_calls["n"] += 1
            return None
        self.fv.load_radiance_decoder_weights = loader

        video = self._img().expand(5, 8, 8, 3).contiguous()   # 5 frames
        base, _, _, _, _, _ = self.node.convert(video, "None", 1000.0, "manual", 0.5, 1.6,
                                       0.0, "Linear")
        out, _, _, _, _, _ = self.node.convert(video, "None", 1000.0, "manual", 0.5, 1.6,
                                      0.0, "Linear", vae=_CountingVAE())
        self.assertEqual(encode_calls["n"], 0)
        self.assertEqual(loader_calls["n"], 0)
        self.assertTrue(self.torch.allclose(out, base))

    def test_video_mode_dispatches_temporal_recovery_without_vae(self):
        called = {"n": 0}

        def temporal(source, deterministic, highlights, shadows, peak_scale,
                     window, checkpoint, preserve_source):
            called["n"] += 1
            self.assertEqual(window, 7)
            self.assertFalse(preserve_source)
            return deterministic + 0.1, highlights * 0.8, shadows * 0.6

        self.node._temporal_reconstruct = temporal
        video = self._img().expand(5, 8, 8, 3).contiguous()
        out, _, _, h_conf, s_conf, _ = self.node.convert(
            video, "None", 1000.0, "manual", 0.5, 1.6, 0.0,
            "Linear", batch_mode="Video Frames", processing_mode="Hybrid",
            temporal_window=7,
        )
        self.assertEqual(called["n"], 1)
        self.assertTrue(bool((out > 0.0).any()))
        self.assertGreater(float(h_conf.max()), 0.0)
        self.assertGreater(float(s_conf.max()), 0.0)

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

    def test_direct_pixel_backend_runs_without_vae(self):
        called = {"n": 0}

        def pixel(source, base, mask, checkpoint, blend, peak_scale,
                  tile_size, tile_overlap, recovery_mode, strength):
            called["n"] += 1
            self.assertEqual(checkpoint, "pixel.pt")
            self.assertEqual(recovery_mode, "highlights")
            return base + mask.unsqueeze(-1) * 0.05

        self.node._pixel_reconstruct = pixel
        out, _, _, h_conf, s_conf, _ = self.node.convert(
            self._img(), "None", 1000.0, "manual", 0.5, 1.6, 0.0,
            "Linear", learned_backend="Direct Pixel",
            pixel_checkpoint="pixel.pt", pixel_recovery_mode="highlights",
        )
        self.assertEqual(called["n"], 1)
        self.assertTrue(bool((out > 0.0).any()))
        self.assertGreater(float(h_conf.max()), 0.0)
        self.assertAlmostEqual(float(s_conf.max()), 0.0, places=6)

    def test_direct_pixel_backend_accepts_video_frame_batches(self):
        called = {"frames": 0}

        def pixel(source, base, mask, *args):
            called["frames"] = source.shape[0]
            return base

        self.node._pixel_reconstruct = pixel
        video = self._img().expand(5, 8, 8, 3).contiguous()
        self.node.convert(
            video, "None", 1000.0, "manual", 0.5, 1.6, 0.0,
            "Linear", batch_mode="Video Frames",
            learned_backend="Direct Pixel",
        )
        self.assertEqual(called["frames"], 5)

    def test_direct_pixel_failure_uses_deterministic_fallback(self):
        def unavailable(*args, **kwargs):
            raise RuntimeError("missing pixel checkpoint")

        self.node._pixel_reconstruct = unavailable
        img = self._img()
        expected, _, _, _, _, _ = self.node.convert(
            img, "None", 1000.0, "manual", 0.5, 1.6, 0.0,
            "Linear", processing_mode="Expand",
        )
        out, _, _, h_conf, s_conf, _ = self.node.convert(
            img, "None", 1000.0, "manual", 0.5, 1.6, 0.0,
            "Linear", learned_backend="Direct Pixel",
        )
        self.assertTrue(self.torch.allclose(out, expected))
        self.assertAlmostEqual(float(h_conf.max()), 0.0, places=6)
        self.assertAlmostEqual(float(s_conf.max()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
