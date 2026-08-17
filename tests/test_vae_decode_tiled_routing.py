"""
tests/test_vae_decode_tiled_routing.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests that RadianceVAE4KDecode.decode() routes video latents to the right
place under the VAE tiling integration:

  - turbo_decoder=None: spatial+temporal tiling both go through the single
    vae.decode_tiled() call (comfy.sd.VAE's own integrated tiler), falling
    back to plain vae.decode() when neither is actually needed.
  - turbo_decoder set (RUDRA): vae.decode_tiled() is never touched. RUDRA
    keeps its original recursive-decode chunking, unchanged by this fix.

See project_radiance_vae_tiling_seams memory for the investigation this
integration replaced (Radiance's own stacked spatial-tile-per-temporal-chunk
decode, which re-decoded a full spatial tile's worth of activations for
every temporal chunk independently).
"""

from __future__ import annotations

import sys
import os
import json
import types
import importlib
import unittest

# ── Real torch check ──────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_TORCH = False

skip_no_torch = unittest.skipUnless(HAS_TORCH, "PyTorch not available")

# ── Minimal ComfyUI stubs ─────────────────────────────────────────────────────
for _m in ["folder_paths", "comfy", "comfy.utils", "comfy.model_management"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

_mm = sys.modules["comfy.model_management"]
if not hasattr(_mm, "get_torch_device"):
    _mm.get_torch_device = lambda: torch.device("cpu") if HAS_TORCH else None
if not hasattr(_mm, "soft_empty_cache"):
    _mm.soft_empty_cache = lambda: None  # no-op in tests

_cu = sys.modules["comfy.utils"]
if not hasattr(_cu, "ProgressBar"):
    class _FakeProgressBar:
        def __init__(self, *a, **kw): pass
        def update(self, *a): pass
        def update_absolute(self, *a): pass
    _cu.ProgressBar = _FakeProgressBar

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _import_vae():
    mod_name = "hdr.vae"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    return importlib.import_module(mod_name)


def _make_decoder():
    return _import_vae().RadianceVAE4KDecode()


class _FakeVideoVAE:
    """Stand-in for comfy.sd.VAE exposing only what decode()'s tiling
    decision touches, with call tracking instead of MagicMock's
    auto-generated (and always-truthy) attributes: detect_vae_factor()
    and the routing logic both branch on real attribute *absence*."""

    def __init__(self, frames_per_lat_frame=2, downscale_ratio=8, temporal_compression=8):
        self.downscale_ratio = downscale_ratio
        self.latent_dim = 3
        self._frames_per_lat_frame = frames_per_lat_frame
        self._temporal_compression = temporal_compression
        self.decode_calls = []
        self.decode_tiled_calls = []

    def temporal_compression_decode(self):
        return self._temporal_compression

    def decode(self, latent):
        self.decode_calls.append(tuple(latent.shape))
        T = latent.shape[2] if latent.ndim == 5 else 1
        return torch.zeros(1, T * self._frames_per_lat_frame, 4, 4, 3)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        self.decode_tiled_calls.append(
            dict(tile_x=tile_x, tile_y=tile_y, overlap=overlap, tile_t=tile_t, overlap_t=overlap_t)
        )
        T = samples.shape[2] if samples.ndim == 5 else 1
        return torch.zeros(1, T * self._frames_per_lat_frame, 4, 4, 3)


if HAS_TORCH:
    class _FakeTurboDecoder(torch.nn.Module):
        """Minimal real nn.Module: decode()'s turbo_decoder branch calls
        next(turbo_decoder.parameters()) and .to(device), both real
        nn.Module machinery that a plain callable/MagicMock can't satisfy
        without extra wiring."""

        def __init__(self, frames_per_lat_frame=2):
            super().__init__()
            self._frames_per_lat_frame = frames_per_lat_frame
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            self.call_count = 0

        def forward(self, x):
            self.call_count += 1
            return torch.zeros(x.shape[0], 3, 4, 4)


@skip_no_torch
class TestUnifiedPathFastCase(unittest.TestCase):
    """Small resolution + short clip: neither spatial nor temporal tiling
    is needed, so decode() must skip decode_tiled() and call vae.decode()
    directly, matching pre-fix behavior for the common case."""

    def test_small_video_uses_plain_decode(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        latent = torch.zeros(1, 4, 4, 32, 32)  # pix 256x256, well under max_tile

        img, _, _ = decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="Auto", temporal_overlap=2,
        )

        self.assertEqual(len(vae.decode_calls), 1)
        self.assertEqual(len(vae.decode_tiled_calls), 0)
        self.assertEqual(img.shape[0], 4 * 2)  # 4 latent frames * 2 px/lat


@skip_no_torch
class TestUnifiedPathSpatialOnly(unittest.TestCase):
    """Large resolution, short clip: decode_tiled() must fire for the
    spatial tile, with tile_t/overlap_t omitted (None) so comfy's own
    dispatcher treats the temporal axis as unbounded."""

    def test_large_frame_short_clip_omits_temporal_args(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        # lat 300x300 * factor 8 = pix 2400x2400 -> exceeds max_tile (1536)
        latent = torch.zeros(1, 4, 1, 300, 300)

        decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="Auto", temporal_overlap=2,
        )

        self.assertEqual(len(vae.decode_tiled_calls), 1)
        call = vae.decode_tiled_calls[0]
        self.assertIsNone(call["tile_t"])
        self.assertIsNone(call["overlap_t"])
        self.assertIsNotNone(call["tile_x"])
        self.assertIsNotNone(call["tile_y"])


@skip_no_torch
class TestUnifiedPathTemporalOnly(unittest.TestCase):
    """Small resolution, long clip with a manual temporal_size smaller than
    the clip: decode_tiled() must fire with a real tile_t even though no
    spatial tiling is needed."""

    def test_long_clip_small_frame_sets_temporal_args(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        latent = torch.zeros(1, 4, 20, 4, 4)  # pix 32x32, 20 latent frames

        decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="8", temporal_overlap=2,
        )

        self.assertEqual(len(vae.decode_tiled_calls), 1)
        call = vae.decode_tiled_calls[0]
        self.assertEqual(call["tile_t"], 8)
        self.assertEqual(call["overlap_t"], 2)


@skip_no_torch
class TestTurboDecoderNeverUsesDecodeTiled(unittest.TestCase):
    """RUDRA (turbo_decoder set) must never reach vae.decode_tiled().
    It keeps its own pre-existing recursive-chunking + direct forward-pass
    path, since it's a raw nn.Module, not a comfy.sd.VAE."""

    def test_rudra_with_temporal_chunking_skips_decode_tiled(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        turbo = _FakeTurboDecoder()
        latent = torch.zeros(1, 4, 20, 4, 4)  # long clip, small frame

        _, meta, _ = decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="8", temporal_overlap=2,
            turbo_decoder=turbo,
        )

        self.assertEqual(len(vae.decode_tiled_calls), 0)
        self.assertGreater(turbo.call_count, 0)
        # Confirms the recursive-chunking block (not just "not decode_tiled")
        # is what actually ran.
        self.assertIn("temporal_chunks", json.loads(meta))

    def test_rudra_auto_temporal_size_keeps_pre_fix_default_of_no_chunking(self):
        """RUDRA has no VRAM calibration data of its own, so "Auto" must
        resolve to "disabled" for it (matching the pre-fix default), not
        silently start chunking RUDRA decodes with an untested formula."""
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        turbo = _FakeTurboDecoder()
        latent = torch.zeros(1, 4, 20, 4, 4)

        _, meta, _ = decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="Auto", temporal_overlap=2,
            turbo_decoder=turbo,
        )

        self.assertEqual(len(vae.decode_tiled_calls), 0)
        self.assertGreater(turbo.call_count, 0)
        # "temporal_chunks" only appears in the recursive-chunking block's
        # own metadata dict, so its absence confirms that block never fired:
        # Auto did not invent a chunk size for RUDRA's uncalibrated path.
        self.assertNotIn("temporal_chunks", json.loads(meta))


if __name__ == "__main__":
    unittest.main()
