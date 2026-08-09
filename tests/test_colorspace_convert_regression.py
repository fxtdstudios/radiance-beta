"""Regression tests for RadianceColorSpaceConvert (2026-08 audit).

What this guards against
------------------------
1. The node's inline LogC3 disagreed with the ARRI spec: 18% grey encoded to
   0.417 instead of 0.391, black to 0.271 instead of 0.093 -- and its decode
   disagreed with its own encode, so 0.18 round-tripped to 0.060 (~1.5 stops).
2. Ten of the sixteen advertised spaces had NO analytical implementation and
   silently returned the input unchanged whenever no OCIO config was loaded,
   which is the default install.
3. The conftest OCIO mock reported is_loaded as a truthy MagicMock, so the
   OCIO path "succeeded" with a no-op applyRGB and every colour test through
   this node validated an identity transform.

Every advertised space must (a) actually change the pixels, (b) round-trip to
scene linear within tolerance, and (c) hit the published 18%-grey code value
where one exists.
"""
import importlib

import pytest

torch = pytest.importorskip("torch")

radiance = importlib.import_module("radiance")
CSC = radiance.NODE_CLASS_MAPPINGS["RadianceColorSpaceConvert"]

LINEAR = "Linear sRGB (D65)"
ALL_SPACES = [s for s in CSC._COLOR_SPACES if s != LINEAR]

# Published 18%-grey code values (curve conventions as documented per space).
# ACEScc/ACEScct: (log2(0.18) + 9.72) / 17.52 on near-neutral AP1 grey.
GREY18_REFERENCE = {
    "sRGB (OETF encoded)": 0.4613,
    "Rec.709 (OETF encoded)": 0.4090,   # 1.099 * 0.18^0.45 - 0.099
    "LogC3 (ARRI EI800)": 0.3910,
    "LogC4 (ARRI Alexa 35)": 0.2784,
    "ACEScc": 0.4135,
    "ACEScct": 0.4135,
    "DaVinci Intermediate": 0.3360,
    "F-Log2 (Fujifilm)": 0.3910,
}


def _convert(img, src, dst):
    node = CSC()
    return node.apply(image=img, src_space=src, dst_space=dst,
                      direction="Forward", strength=1.0)[0]


@pytest.fixture()
def hdr_image():
    base = torch.rand(1, 24, 24, 3) * 4.0
    base[0, 0] = 18.0        # HDR highlight
    base[0, 1] = 0.0         # black
    base[0, 2] = 0.18        # 18% grey
    return base


@pytest.mark.real_torch
@pytest.mark.parametrize("space", ALL_SPACES)
def test_forward_is_not_identity(space):
    """A conversion that returns its input unchanged is the P0 this file exists for.

    Saturated input, not neutral grey: a gamut matrix (ACEScg) maps the
    neutral axis to itself, so grey would pass an identity check even when
    the matrix is applied correctly.
    """
    img = torch.tensor([0.6, 0.18, 0.05]).expand(1, 4, 4, 3).contiguous()
    out = _convert(img, LINEAR, space)
    assert not torch.allclose(out, img, atol=1e-4), (
        f"{space}: forward conversion returned the input unchanged")


@pytest.mark.real_torch
@pytest.mark.parametrize("space", ALL_SPACES)
def test_round_trip(space, hdr_image):
    encoded = _convert(hdr_image, LINEAR, space)
    back = _convert(encoded, space, LINEAR)
    # Display-referred and clipping curves cannot represent 18.0; compare on
    # the representable region only.
    lossy_above_1 = space in (
        "sRGB (OETF encoded)", "Rec.709 (OETF encoded)", "Rec.709 / BT.1886",
        "ACEScc",  # negatives pin to the min code value by spec
    )
    mask = (hdr_image <= 1.0) & (hdr_image >= 0.0) if lossy_above_1 \
        else torch.ones_like(hdr_image, dtype=torch.bool)
    err = (back - hdr_image).abs()[mask].max().item()
    assert err < 5e-3, f"{space}: round-trip error {err:.5f}"


@pytest.mark.real_torch
@pytest.mark.parametrize("space,ref", sorted(GREY18_REFERENCE.items()))
def test_grey18_reference(space, ref):
    img = torch.full((1, 2, 2, 3), 0.18)
    out = _convert(img, LINEAR, space)
    got = out[0, 0, 0, 0].item()
    assert abs(got - ref) < 2e-3, (
        f"{space}: 18% grey encoded to {got:.4f}, published value is {ref:.4f}")


@pytest.mark.real_torch
def test_logc3_black_reference():
    """Black must land at f = 0.0928, not the 0.271 the old inline curve gave."""
    img = torch.zeros(1, 2, 2, 3)
    out = _convert(img, LINEAR, "LogC3 (ARRI EI800)")
    assert abs(out[0, 0, 0, 0].item() - 0.092809) < 1e-3


@pytest.mark.real_torch
def test_ocio_mock_cannot_reintroduce_identity():
    """The conftest OCIO manager stub must not report itself as loaded."""
    from radiance.radiance_ocio import get_ocio_manager
    mgr = get_ocio_manager()
    assert not mgr.is_loaded, (
        "conftest's OCIO manager mock is truthy again; RadianceColorSpaceConvert "
        "will 'convert' via a no-op MagicMock processor and every colour test "
        "in this file will silently validate an identity transform.")


@pytest.mark.real_torch
def test_srgb_and_rec709_oetf_differ():
    """Rec.709 OETF was previously aliased to the sRGB curve."""
    img = torch.full((1, 2, 2, 3), 0.18)
    srgb = _convert(img, LINEAR, "sRGB (OETF encoded)")
    r709 = _convert(img, LINEAR, "Rec.709 (OETF encoded)")
    assert abs(srgb[0, 0, 0, 0].item() - r709[0, 0, 0, 0].item()) > 0.02


@pytest.mark.real_torch
def test_negative_scene_linear_survives_log_spaces():
    """Slightly-negative scene-linear (grain, matte edges) must not explode."""
    img = torch.full((1, 2, 2, 3), -0.02)
    for space in ("LogC3 (ARRI EI800)", "LogC4 (ARRI Alexa 35)", "DaVinci Intermediate"):
        out = _convert(img, LINEAR, space)
        assert torch.isfinite(out).all(), f"{space}: non-finite output for negative input"
