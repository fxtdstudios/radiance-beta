"""Both ACES 2.0 tone scales must put 18% grey where the standard puts it.

Radiance shipped two of them and they disagreed with each other by up to 24x:

  * `RadianceACES2Tonescale` used the real Daniele Evo curve but pinned grey to
    a fixed 10% *of peak*, so it rendered at 10 nits on SDR and 400 on a
    4000-nit master.
  * `RadianceACES2OutputTransform` used a log-contrast + tanh approximation
    that held grey at ~18 nits on every peak — nearly the right behaviour in
    kind, ~0.85 stop bright at SDR.

There was no judgement call to make in the end. ACES 2.0 publishes the mapping
(Output Transforms → Tone Mapping): an ACES value of 0.18 lands at 10.000 nits
on a 100-nit display, rising gently to 16.824 at 4000. The same table gives
ACES 1.0 → 45.757 nits at 100-nit peak, i.e. the familiar ~48-nit diffuse
white, which is how you can tell those cells are luminances.

https://docs.acescentral.com/system-components/output-transforms/technical-details/tone-mapping/
"""
import os
import sys

import numpy as np
import pytest

try:
    import torch as _t
    _HAS_TORCH = isinstance(getattr(_t, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.hdr.tonescale import (  # noqa: E402
    ACES2_MIDGREY_ANCHORS,
    aces2_midgrey_fraction,
    aces2_midgrey_nits,
)


class TestTheReferenceTable:

    @pytest.mark.parametrize("peak,expected", ACES2_MIDGREY_ANCHORS)
    def test_published_anchors_are_exact(self, peak, expected):
        assert aces2_midgrey_nits(peak) == pytest.approx(expected, abs=1e-6)

    def test_grey_rises_with_peak_but_nowhere_near_proportionally(self):
        """The whole bug in one assertion: 40x the peak, 1.68x the grey."""
        assert aces2_midgrey_nits(4000) / aces2_midgrey_nits(100) == pytest.approx(1.6824, rel=1e-3)

    def test_it_is_monotonic(self):
        vals = [aces2_midgrey_nits(p) for p in range(100, 8000, 50)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_it_interpolates_between_anchors(self):
        assert 10.0 < aces2_midgrey_nits(250) < 13.193
        assert 14.512 < aces2_midgrey_nits(1500) < 15.747

    def test_it_extrapolates_instead_of_clamping(self):
        """A 10000-nit target should get an answer, not the 4000-nit one."""
        assert aces2_midgrey_nits(10000) > aces2_midgrey_nits(4000)
        assert aces2_midgrey_nits(50) < aces2_midgrey_nits(100)

    def test_the_fraction_form_agrees(self):
        for peak, expected in ACES2_MIDGREY_ANCHORS:
            assert aces2_midgrey_fraction(peak) == pytest.approx(expected / peak, rel=1e-9)

    def test_sdr_fraction_is_the_old_hardcoded_value(self):
        """0.10 was right — for 100 nits only, which is why SDR looked fine."""
        assert aces2_midgrey_fraction(100.0) == pytest.approx(0.10, abs=1e-9)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
class TestDanieleEvoNode:

    @pytest.mark.parametrize("peak,expected", ACES2_MIDGREY_ANCHORS)
    def test_it_hits_the_published_grey(self, peak, expected):
        from radiance.nodes.hdr.aces2 import _DanieleEvoParams, _daniele_evo_fwd

        params = _DanieleEvoParams(peak_nits=peak)
        nits = float(_daniele_evo_fwd(np.array([0.18]), params)[0]) * 100.0
        assert nits == pytest.approx(expected, abs=0.01)

    def test_grey_no_longer_tracks_the_display(self):
        """It used to render 400 nits at a 4000-nit peak — 24x reference."""
        from radiance.nodes.hdr.aces2 import _DanieleEvoParams, _daniele_evo_fwd

        nits = float(_daniele_evo_fwd(np.array([0.18]), _DanieleEvoParams(peak_nits=4000.0))[0]) * 100.0
        assert nits < 20.0, f"18% grey at {nits:.1f} nits on a 4000-nit master"

    def test_an_explicit_grey_target_is_still_honoured(self):
        """The parameter stays available for deliberate deviation."""
        from radiance.nodes.hdr.aces2 import _DanieleEvoParams, _daniele_evo_fwd

        params = _DanieleEvoParams(peak_nits=1000.0, grey_target=0.05)
        nits = float(_daniele_evo_fwd(np.array([0.18]), params)[0]) * 100.0
        assert nits == pytest.approx(0.05 * 1000.0, rel=0.01)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
class TestLegacyOutputTransform:

    @staticmethod
    def _grey(peak):
        from radiance.hdr.color import ACES2OutputTransform

        rgb = np.full((1, 1, 1, 3), 0.18, dtype=np.float32)
        out = ACES2OutputTransform()._apply_tonescale_drt(
            rgb, peak_luminance=peak, is_hdr=(peak > 100))
        return float(out[0, 0, 0, 0]) * 100.0

    @pytest.mark.parametrize("peak,expected", ACES2_MIDGREY_ANCHORS)
    def test_it_hits_the_published_grey(self, peak, expected):
        assert self._grey(peak) == pytest.approx(expected, abs=0.05)

    def test_the_two_tone_scales_now_agree(self):
        """They disagreed by up to 24x. Same input, same peak, same answer."""
        from radiance.nodes.hdr.aces2 import _DanieleEvoParams, _daniele_evo_fwd

        for peak, _ in ACES2_MIDGREY_ANCHORS:
            evo = float(_daniele_evo_fwd(np.array([0.18]),
                                         _DanieleEvoParams(peak_nits=peak))[0]) * 100.0
            assert self._grey(peak) == pytest.approx(evo, rel=0.01), \
                f"the two tone scales disagree at {peak} nits"

    def test_a_neutral_stays_neutral(self):
        """The grey solve must not tint anything."""
        from radiance.hdr.color import ACES2OutputTransform

        rgb = np.full((1, 4, 4, 3), 0.18, dtype=np.float32)
        out = ACES2OutputTransform()._apply_tonescale_drt(rgb, peak_luminance=1000.0, is_hdr=True)
        assert out[..., 0] == pytest.approx(out[..., 1], abs=1e-6)
        assert out[..., 1] == pytest.approx(out[..., 2], abs=1e-6)

    def test_it_is_still_monotonic(self):
        from radiance.hdr.color import ACES2OutputTransform

        x = np.linspace(0.0, 40.0, 20001, dtype=np.float32)
        rgb = np.stack([x] * 3, axis=-1)[None, None]
        y = ACES2OutputTransform()._apply_tonescale_drt(rgb, peak_luminance=1000.0, is_hdr=True)[0, 0, :, 0]
        assert bool((np.diff(y) >= -1e-6).all()), "tonescale is not monotonic"

    def test_it_still_reaches_the_requested_peak(self):
        from radiance.hdr.color import ACES2OutputTransform

        rgb = np.full((1, 1, 1, 3), 10000.0, dtype=np.float32)
        out = ACES2OutputTransform()._apply_tonescale_drt(
            rgb, peak_luminance=4000.0, is_hdr=True)
        assert float(out[0, 0, 0, 0]) * 100.0 > 3900.0


@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
def test_the_scalar_probe_matches_the_vectorised_loop():
    """`_tanh_tonescale_scalar` duplicates the per-channel chain deliberately.

    The gain solve needs to evaluate the curve on one value, and the main loop
    is vectorised over an image. This asserts the copy has not drifted — the
    reason for the duplication is speed, not divergence.
    """
    from radiance.hdr.color import ACES2OutputTransform

    t = ACES2OutputTransform()
    for peak in (100.0, 1000.0, 4000.0):
        peak_scale = peak / 100.0
        for x in (0.01, 0.18, 0.5, 1.0, 4.0):
            rgb = np.full((1, 1, 1, 3), x, dtype=np.float32)
            # Undo the grey anchoring so the raw curve is what gets compared.
            gain = t._solve_grey_gain(
                __import__("radiance.hdr.tonescale", fromlist=["x"]).aces2_midgrey_nits(peak) / 100.0,
                peak_scale, 1.0)
            loop = float(t._apply_tonescale_drt(rgb, peak_luminance=peak, is_hdr=(peak > 100))[0, 0, 0, 0])
            scalar = t._tanh_tonescale_scalar(x * gain, peak_scale, 1.55)
            assert loop == pytest.approx(scalar, rel=1e-4, abs=1e-6), \
                f"scalar probe and vectorised loop disagree at x={x}, peak={peak}"
