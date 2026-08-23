"""
tests/test_rudra_conditioning.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A RUDRA decoder with dynamic-range conditioning infers its conditioning
vector from whatever tensor it is handed. Hand it a spatial tile and it
conditions on that tile's statistics; hand it one frame of a clip and it
conditions on that frame. Both produce level differences the blend cannot
hide: neighbouring tiles graded apart, and a clip whose exposure breathes.

These tests pin the fix: the vector is resolved once, from the whole latent,
and passed down to every tile and every chunk.

They are no-ops for decoders without conditioning (dr_dim None), which is
every checkpoint predating dynamic-range conditioning, and the last two
tests pin that those still work.
"""

from __future__ import annotations

import os
import sys
import types
import importlib
import unittest

try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_TORCH = False

skip_no_torch = unittest.skipUnless(HAS_TORCH, "PyTorch not available")

for _m in ["folder_paths", "comfy", "comfy.utils", "comfy.model_management"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
_mm = sys.modules["comfy.model_management"]
if not hasattr(_mm, "get_torch_device"):
    _mm.get_torch_device = lambda: torch.device("cpu") if HAS_TORCH else None
if not hasattr(_mm, "soft_empty_cache"):
    _mm.soft_empty_cache = lambda: None
_cu = sys.modules["comfy.utils"]
if not hasattr(_cu, "ProgressBar"):
    class _FakeProgressBar:
        def __init__(self, *a, **kw): pass
        def update(self, *a): pass
        def update_absolute(self, *a): pass
    _cu.ProgressBar = _FakeProgressBar

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _fast_vae():
    from radiance import fast_vae
    return fast_vae


def _import_vae():
    if "hdr.vae" in sys.modules:
        return sys.modules["hdr.vae"]
    return importlib.import_module("hdr.vae")


def _split_latent(channels=16, h=32, w=64, seed=0):
    """A latent whose left half is far brighter than its right half, so a
    left tile and a right tile see genuinely different statistics."""
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(1, channels, h, w, generator=g) * 0.2
    lat[:, :, :, : w // 2] += 6.0
    lat[:, :, :, w // 2:] -= 2.0
    return lat


@skip_no_torch
class TestConditionResolution(unittest.TestCase):

    def test_no_conditioning_means_no_vector(self):
        fv = _fast_vae()
        dec = fv.RadianceTurboDecoder(latent_channels=16, n_upsample=3, dr_dim=None)
        self.assertIsNone(fv.rudra_condition_for(dec, _split_latent()))

    def test_a_conditioned_decoder_yields_one_vector_per_sample(self):
        fv = _fast_vae()
        dec = fv.RadianceTurboDecoder(latent_channels=16, n_upsample=3, dr_dim=64)
        cond = fv.rudra_condition_for(dec, _split_latent().repeat(3, 1, 1, 1))
        self.assertIsNotNone(cond)
        self.assertEqual(tuple(cond.shape), (3, 64))

    def test_a_tile_and_the_whole_frame_disagree(self):
        """The defect this fix exists for: the decoder's own predictor gives
        a materially different answer for a crop than for the frame."""
        fv = _fast_vae()
        dec = fv.RadianceTurboDecoder(latent_channels=16, n_upsample=3, dr_dim=64)
        lat = _split_latent()
        whole = fv.rudra_condition_for(dec, lat)
        left = fv.rudra_condition_for(dec, lat[:, :, :, :32])
        right = fv.rudra_condition_for(dec, lat[:, :, :, 32:])
        self.assertGreater((left - right).norm().item(), 1.0)
        self.assertGreater((whole - left).norm().item(), 1.0)

    def test_a_clip_shares_one_vector_across_its_frames(self):
        fv = _fast_vae()
        dec = fv.RadianceTurboDecoder(latent_channels=16, n_upsample=3, dr_dim=64)
        frames = torch.cat([_split_latent(seed=s) + s for s in range(4)], dim=0)

        per_frame = fv.rudra_condition_for(dec, frames, video_frames=None)
        self.assertGreater((per_frame[0] - per_frame[3]).norm().item(), 1e-3)

        shared = fv.rudra_condition_for(dec, frames, video_frames=4)
        for i in range(1, 4):
            self.assertTrue(torch.allclose(shared[0], shared[i], atol=1e-6))

    def test_a_batch_of_stills_keeps_its_own_vector_per_image(self):
        """Independent images are not a clip: sharing would be wrong."""
        fv = _fast_vae()
        dec = fv.RadianceTurboDecoder(latent_channels=16, n_upsample=3, dr_dim=64)
        stills = torch.cat([_split_latent(seed=s) + s for s in range(4)], dim=0)
        cond = fv.rudra_condition_for(dec, stills, video_frames=None)
        self.assertGreater((cond[0] - cond[3]).norm().item(), 1e-3)


@skip_no_torch
class TestTiledDecode(unittest.TestCase):
    """A conv decoder is not shift-invariant at a tile border, so tiled and
    untiled never match bit for bit. What must hold is that conditioning
    stops ADDING error on top of that, and that every tile is handed the
    same vector."""

    def _decode(self, dec, lat, **kw):
        return _fast_vae().decode_to_linear_realtime(
            lat, dec, model_type="flux", precision="fp32",
            return_log_coded=True, **kw)

    def test_every_tile_is_handed_the_same_conditioning(self):
        dec = _TileRecordingDecoder(latent_channels=16, dr_dim=64)
        self._decode(dec, _split_latent(), tiled=True, tile_size=128, overlap=64)
        self.assertGreater(len(dec.seen), 1, "expected more than one tile")
        self.assertTrue(all(c is not None for c in dec.seen),
                        "a tile was decoded with no conditioning at all")
        first = dec.seen[0][0]
        for tile in dec.seen:
            for row in tile:
                self.assertTrue(torch.allclose(first, row, atol=1e-6))

    def test_conditioning_does_not_add_tiling_error(self):
        fv = _fast_vae()
        lat = _split_latent()
        kw = dict(tiled=True, tile_size=256, overlap=64)

        plain = fv.RadianceTurboDecoder(latent_channels=16, n_upsample=3, dr_dim=None).eval()
        base = (self._decode(plain, lat) - self._decode(plain, lat, **kw)).abs().mean().item()

        cond = fv.RadianceTurboDecoder(latent_channels=16, n_upsample=3, dr_dim=64).eval()
        got = (self._decode(cond, lat) - self._decode(cond, lat, **kw)).abs().mean().item()

        # Pre-fix this ratio measured about 29x on the same inputs.
        self.assertLess(got, base * 2.0,
                        f"conditioned tiling error {got:.3e} vs unconditioned {base:.3e}")

    def test_a_4d_batch_keeps_a_vector_per_image(self):
        """decode_to_linear_realtime must not treat a batch of unrelated
        stills as a clip: each image keeps its own conditioning."""
        dec = _TileRecordingDecoder(latent_channels=16, dr_dim=64)
        batch = torch.cat([_split_latent(seed=0), _split_latent(seed=1) + 9.0], dim=0)
        self._decode(dec, batch)
        self.assertEqual(len(dec.seen), 1)
        rows = dec.seen[0]
        self.assertEqual(rows.shape[0], 2)
        self.assertGreater((rows[0] - rows[1]).norm().item(), 1e-3)

    def test_a_caller_supplied_vector_is_not_overwritten(self):
        fv = _fast_vae()
        dec = fv.RadianceTurboDecoder(latent_channels=16, n_upsample=3, dr_dim=64).eval()
        lat = _split_latent()
        mine = torch.full((1, 64), 0.37)
        self.assertGreater(
            (self._decode(dec, lat, dr_proj=mine) - self._decode(dec, lat)).abs().max().item(),
            0.0)


if HAS_TORCH:
    class _TileRecordingDecoder(torch.nn.Module):
        """Records the conditioning each TILE receives and upsamples like a
        real decoder, so decode_to_linear_realtime's tiler accepts it."""

        def __init__(self, latent_channels=16, dr_dim=64, n_upsample=3):
            super().__init__()
            self.dr_dim = dr_dim
            self.n_upsample = n_upsample
            self.predictor = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(latent_channels, dr_dim),
            )
            self.seen = []

        def forward(self, x, dr_proj=None):
            self.seen.append(None if dr_proj is None else dr_proj.detach().clone())
            scale = 2 ** self.n_upsample
            return torch.zeros(x.shape[0], 3, x.shape[2] * scale, x.shape[3] * scale)


if HAS_TORCH:
    class _RecordingConditionedDecoder(torch.nn.Module):
        """Records the conditioning each call receives, so a test can prove
        every chunk was graded with the same vector."""

        def __init__(self, dr_dim=64, latent_channels=4):
            super().__init__()
            self.dr_dim = dr_dim
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            self.predictor = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(latent_channels, dr_dim),
            )
            self.seen = []

        def forward(self, x, dr_proj=None):
            self.seen.append(None if dr_proj is None else dr_proj.detach().clone())
            return torch.zeros(x.shape[0], 3, 4, 4)

    class _RecordingPlainDecoder(torch.nn.Module):
        """No dr_dim: must still be called with a single argument."""

        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            self.arg_counts = []

        def forward(self, x):
            self.arg_counts.append(1)
            return torch.zeros(x.shape[0], 3, 4, 4)

    class _FakeVideoVAE:
        def __init__(self):
            self.downscale_ratio = 8
            self.latent_dim = 3

        def temporal_compression_decode(self):
            return 8

        def decode(self, latent):
            return torch.zeros(1, 4, 4, 3)


@skip_no_torch
class TestHdrVaeChunkedDecode(unittest.TestCase):
    """The node path: hdr/vae.py decodes RUDRA in chunks of four frames.
    Before the fix each chunk re-derived its own conditioning, so a clip
    re-graded itself every four frames."""

    def _decode(self, turbo, frames=8):
        node = _import_vae().RadianceVAE4KDecode()
        latent = _split_latent(channels=4, h=32, w=32).unsqueeze(2).repeat(1, 1, frames, 1, 1)
        for t in range(frames):
            latent[:, :, t] += float(t)          # every frame differs
        return node.decode(
            {"samples": latent}, vae=_FakeVideoVAE(),
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="Auto", temporal_overlap=2,
            turbo_decoder=turbo,
        )

    def test_every_chunk_of_a_clip_gets_the_same_conditioning(self):
        turbo = _RecordingConditionedDecoder(latent_channels=4)
        self._decode(turbo, frames=8)
        self.assertGreater(len(turbo.seen), 1, "expected more than one chunk")
        self.assertTrue(all(c is not None for c in turbo.seen),
                        "a chunk was decoded with no conditioning at all")
        first = turbo.seen[0][0]
        for chunk in turbo.seen:
            for row in chunk:
                self.assertTrue(torch.allclose(first, row, atol=1e-6))

    def test_a_decoder_without_conditioning_is_called_with_one_argument(self):
        turbo = _RecordingPlainDecoder()
        self._decode(turbo, frames=8)
        self.assertTrue(turbo.arg_counts)


if __name__ == "__main__":
    unittest.main()


@skip_no_torch
class TestLtxStillsWarning(unittest.TestCase):
    """The LTX decoder warning is true of the shipped stills-trained
    checkpoint and false of a retrained one. It has to read the checkpoint,
    not the model_type."""

    def _write(self, tmpdir, metadata):
        import safetensors.torch as st
        fv = _fast_vae()
        from radiance.config.model_map import resolve_model_vae_config
        cfg = resolve_model_vae_config("ltx-video") or {}
        model = fv.RadianceFullDecoder(
            latent_channels=cfg.get("latent_channels", 128), n_upsample=5, dr_dim=None)
        path = os.path.join(tmpdir, "rudra_full_decoder_ltx-video_ema.safetensors")
        st.save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                     path, metadata=metadata)
        return path

    def _load_and_capture(self, path):
        import logging
        import tempfile
        fv = _fast_vae()
        fv._TRAINED_DECODER_CACHE.clear() if hasattr(fv._TRAINED_DECODER_CACHE, "clear") else None
        records = []

        class _Catch(logging.Handler):
            def emit(self, record):
                records.append((record.levelno, record.getMessage()))

        handler = _Catch()
        log = logging.getLogger("radiance.fast_vae")
        log.addHandler(handler)
        try:
            model = fv.load_radiance_decoder_weights(
                model_type="ltx-video", model_size="rudra_full", checkpoint_path=path)
        finally:
            log.removeHandler(handler)
        return model, records

    def test_an_undeclared_checkpoint_still_warns(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, None)
            model, records = self._load_and_capture(path)
            self.assertIsNotNone(model)
            warnings = [m for lvl, m in records if lvl >= 30 and "isolated still" in m]
            self.assertEqual(len(warnings), 1, records)

    def test_a_single_frame_declaration_still_warns(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"radiance_train_frames": "1"})
            _, records = self._load_and_capture(path)
            self.assertTrue([m for lvl, m in records if lvl >= 30 and "isolated still" in m])

    def test_a_retrained_checkpoint_does_not_carry_a_false_warning(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"radiance_train_frames": "81"})
            model, records = self._load_and_capture(path)
            self.assertIsNotNone(model)
            self.assertFalse([m for lvl, m in records if lvl >= 30 and "isolated still" in m],
                             "a multi-frame checkpoint was warned about as stills-trained")
            self.assertTrue([m for lvl, m in records if "81-frame" in m])

    def test_a_garbage_declaration_is_treated_as_undeclared(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"radiance_train_frames": "many"})
            _, records = self._load_and_capture(path)
            self.assertTrue([m for lvl, m in records if lvl >= 30 and "isolated still" in m])
