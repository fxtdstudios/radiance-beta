"""VFX efficiency fixes (3.5.0): same output, bounded memory.

Six nodes built their work for the whole clip at once or looped in Python per
pixel. Measured on 24 frames of 1024x576 (CPU): Motion Blur ran out of memory
at default settings, Multipass Relight was killed out of memory, Linear Matting
took 107 s, HDR Stitch 24 s, and Floyd-Steinberg dither 8 s per frame. Depth Map
Generator kept every full-resolution depth frame, and a 3-channel copy, on the
GPU. Each now works a few frames at a time (on the GPU when there is one) and
uses separable filters or a wavefront where the maths allows.

These tests pin the outputs to the old maths, written out here as references,
and check that the chunked paths give the same result as a single pass.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not isinstance(getattr(torch, "__version__", None), str):
    pytest.skip("needs real torch", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

pytestmark = pytest.mark.real_torch


@pytest.fixture
def tiny_chunks(monkeypatch):
    """Force one frame per chunk, so every chunk boundary is exercised."""
    import radiance.core.tensor.chunking as ch
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    return ch


# ── chunking helper ──────────────────────────────────────────────────────────

def test_chunks_cover_the_batch_once():
    from radiance.core.tensor.chunking import chunks, frames_per_chunk
    assert list(chunks(7, 3)) == [(0, 3), (3, 6), (6, 7)]
    assert list(chunks(0, 4)) == []
    assert frames_per_chunk(10, 10, 3, 4.0, torch.device("cpu")) >= 1


# ── Motion Blur ──────────────────────────────────────────────────────────────

def _motion_blur_reference(img, blur_type, amount, angle, cx, cy, samples):
    """The pre-3.5.0 batched maths: every sample of every frame at once."""
    b, h, w, c = img.shape
    t = torch.linspace(-0.5, 0.5, samples)
    th = torch.zeros(samples, 2, 3)
    if blur_type == "Directional":
        a = math.radians(angle)
        th[:, 0, 0] = 1; th[:, 1, 1] = 1
        th[:, 0, 2] = t * (math.cos(a) * amount / w) * 2
        th[:, 1, 2] = t * (math.sin(a) * amount / h) * 2
    elif blur_type == "Radial":
        x0, y0 = cx * 2 - 1, cy * 2 - 1
        r = t * amount * 0.02
        cr, sr = torch.cos(r), torch.sin(r)
        th[:, 0, 0] = cr; th[:, 0, 1] = -sr; th[:, 0, 2] = x0 * (1 - cr) + y0 * sr
        th[:, 1, 0] = sr; th[:, 1, 1] = cr; th[:, 1, 2] = y0 * (1 - cr) - x0 * sr
    else:
        x0, y0 = cx * 2 - 1, cy * 2 - 1
        s = 1.0 + (torch.linspace(0, 1, samples) - 0.5) * amount * 0.01
        th[:, 0, 0] = s; th[:, 1, 1] = s; th[:, 0, 2] = x0 * (1 - s); th[:, 1, 2] = y0 * (1 - s)
    th = th.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * samples, 2, 3)
    tiled = img.permute(0, 3, 1, 2).unsqueeze(1).expand(-1, samples, -1, -1, -1).reshape(b * samples, c, h, w)
    grid = F.affine_grid(th, tiled.shape, align_corners=False)
    out = F.grid_sample(tiled, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.reshape(b, samples, c, h, w).mean(1).permute(0, 2, 3, 1).clamp(min=0)


@pytest.mark.parametrize("blur_type", ["Directional", "Radial", "Zoom"])
def test_film_motion_blur_matches_the_batched_maths(blur_type, tiny_chunks):
    from radiance.film.camera import RadianceMotionBlur
    torch.manual_seed(0)
    img = torch.rand(3, 40, 56, 3) * 2
    got = RadianceMotionBlur().apply_motion_blur(img, blur_type, 20.0, 30.0, 0.3, 0.6, 16, use_gpu=False)[0]
    ref = _motion_blur_reference(img, blur_type, 20.0, 30.0, 0.3, 0.6, 16)
    assert got.shape == img.shape
    assert float((got - ref).abs().max()) < 1e-4


def test_shipped_motion_blur_chunked_equals_one_pass(monkeypatch):
    """The Motion Blur in the menu is nodes/vfx/motion_blur.py (vector blur);
    film/camera.py holds a rival class under the same key that is not loaded."""
    import radiance.core.tensor.chunking as ch
    from radiance.nodes.vfx.motion_blur import RadianceMotionBlur
    torch.manual_seed(5)
    img = torch.rand(3, 30, 40, 3) * 2
    vec = torch.randn(3, 30, 40, 3) * 5
    one = RadianceMotionBlur().apply(img, vec, 270.0, 8, False)[0]
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    many = RadianceMotionBlur().apply(img, vec, 270.0, 8, False)[0]
    assert torch.equal(one, many) and one.shape == img.shape


# ── Linear Matting ───────────────────────────────────────────────────────────

def _guided_reference(image, mask, radius, eps):
    r = radius * 2 + 1
    pad = r // 2
    I = image.permute(0, 3, 1, 2)
    p = mask.unsqueeze(1)
    box = lambda x: F.avg_pool2d(x, r, stride=1, padding=pad)  # noqa: E731 - the old 2-D pool
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    q = (box(a) * I + box(b)).mean(1, keepdim=True).clamp(0, 1)
    return q.squeeze(1), image * q.permute(0, 2, 3, 1)


@pytest.mark.parametrize("radius", [0, 3, 12])
def test_matting_separable_filter_equals_the_2d_pool(radius, tiny_chunks):
    from radiance.nodes.vfx.masking import RadianceLinearMatting
    torch.manual_seed(1)
    img = torch.rand(3, 48, 64, 3) * 1.5
    mask = (torch.rand(3, 48, 64) > 0.5).float()
    alpha, fg = RadianceLinearMatting().apply(img, mask, "GuidedFilter", radius, 1e-3)
    ra, rf = _guided_reference(img, mask, radius, 1e-3)
    assert float((alpha - ra).abs().max()) < 1e-4
    assert float((fg - rf).abs().max()) < 1e-4


def test_matting_resizes_a_mask_of_another_size():
    """Load Image gives a 64x64 mask for an image with no alpha; this raised."""
    from radiance.nodes.vfx.masking import RadianceLinearMatting
    alpha, fg = RadianceLinearMatting().apply(torch.rand(1, 90, 120, 3), torch.zeros(1, 64, 64), "GuidedFilter", 4, 1e-3)
    assert alpha.shape == (1, 90, 120) and fg.shape == (1, 90, 120, 3)


def test_matting_accepts_one_mask_for_a_clip():
    from radiance.nodes.vfx.masking import RadianceLinearMatting
    img = torch.rand(4, 32, 32, 3)
    m = torch.zeros(1, 32, 32); m[:, 8:24, 8:24] = 1
    alpha, _ = RadianceLinearMatting().apply(img, m, "GuidedFilter", 4, 1e-3)
    assert alpha.shape == (4, 32, 32)


# ── Multipass Relight ────────────────────────────────────────────────────────

@pytest.mark.parametrize("light_type", ["Directional", "Point"])
def test_relight_chunked_equals_one_pass(light_type, monkeypatch):
    import radiance.core.tensor.chunking as ch
    from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassRelight
    torch.manual_seed(2)
    B, H, W = 3, 24, 32
    kw = dict(albedo=torch.rand(B, H, W, 3), normal_map=torch.rand(B, H, W, 3),
              roughness=torch.rand(B, H, W, 1), ao=torch.rand(1, H, W, 1),
              depth_map=torch.rand(B, H, W, 1), light_type=light_type, output_premultiplied=True,
              alpha=torch.rand(B, H, W, 1))
    one = RadianceMultipassRelight().relight(**kw)
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    many = RadianceMultipassRelight().relight(**kw)
    for a, b in zip(one[:5], many[:5]):
        assert a.shape == (B, H, W, 3)
        assert torch.equal(a, b)


def test_relight_decides_normal_encoding_on_the_whole_clip(tiny_chunks):
    """A clip whose later frame has signed normals is signed throughout, as
    before; a per-chunk decision would decode the frames differently."""
    from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassRelight, _decode_normal_map
    n = torch.rand(2, 8, 8, 3)
    n[1] = n[1] * 2 - 1
    out = RadianceMultipassRelight().relight(torch.ones(2, 8, 8, 3), n, ambient=0.0)[1]
    whole = _decode_normal_map(n, "OpenGL (Y-Up)")
    light = torch.nn.functional.normalize(torch.tensor([-0.35, 0.45, 1.0]), dim=0)
    ndotl = (whole * light).sum(-1).clamp(0, 1)
    assert float((out[..., 0] - ndotl).abs().max()) < 1e-5


# ── HDR Stitch ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["Linear_Laplacian", "Linear_Gaussian", "Standard"])
def test_stitch_separable_feather_equals_the_2d_pool(mode, tiny_chunks):
    from radiance.nodes.vfx.inpaint import RadianceHDRCrop, RadianceHDRStitch
    torch.manual_seed(3)
    img = torch.randn(2, 60, 90, 3) + 0.5
    m = torch.zeros(2, 60, 90); m[:, 20:35, 30:55] = 1
    ci, cm, sd = getattr(RadianceHDRCrop(), RadianceHDRCrop.FUNCTION)(img, m, 1.5, 8)
    stitched, blend = RadianceHDRStitch().apply(img, ci * 2, cm, sd, mode, 6)
    orig_mask = torch.zeros(2, 60, 90)
    orig_mask[:, sd["ymin"]:sd["ymax"], sd["xmin"]:sd["xmax"]] = cm
    ref_blend = F.avg_pool2d(orig_mask.unsqueeze(1), 13, stride=1, padding=6).squeeze(1).clamp(0, 1)
    assert float((blend - ref_blend).abs().max()) < 1e-5
    assert stitched.shape == img.shape
    # Far from the crop the plate is untouched (clamped at 0 for Laplacian).
    far = img[:, :5, :5]
    expect = far.clamp(min=0) if mode == "Linear_Laplacian" else far
    assert float((stitched[:, :5, :5] - expect).abs().max()) < 1e-5


# ── Floyd-Steinberg ──────────────────────────────────────────────────────────

def _fs_scanline(img, levels):
    """The pre-3.5.0 scan-line loop, verbatim arithmetic."""
    step = 1.0 / (levels - 1)
    out = img.float().numpy().copy()
    B, H, W, C = out.shape
    for b in range(B):
        for c in range(C):
            p = out[b, :, :, c]
            for y in range(H):
                for x in range(W):
                    old = p[y, x]
                    new = np.round(old / step) * step
                    err = old - new
                    p[y, x] = new
                    if x + 1 < W:
                        p[y, x + 1] += err * 7 / 16
                    if y + 1 < H:
                        if x > 0:
                            p[y + 1, x - 1] += err * 3 / 16
                        p[y + 1, x] += err * 5 / 16
                        if x + 1 < W:
                            p[y + 1, x + 1] += err * 1 / 16
    return torch.from_numpy(out)


@pytest.mark.parametrize("shape", [(1, 1, 9, 3), (1, 7, 1, 3), (2, 21, 33, 3)])
@pytest.mark.parametrize("bits", [1, 4, 8])
def test_floyd_steinberg_is_bit_identical_to_the_scanline_loop(shape, bits):
    from radiance.nodes.color.colorspace import RadianceBitDepthDegrade
    torch.manual_seed(4)
    img = torch.rand(*shape) * 1.2 - 0.1
    levels = max(2, 2 ** bits - 1)
    assert torch.equal(RadianceBitDepthDegrade._floyd_steinberg(img, levels), _fs_scanline(img, levels))


# ── Depth Map Generator ──────────────────────────────────────────────────────

class _FakeProcessor:
    size = {"height": 28, "width": 28}
    ensure_multiple_of = 14
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.25, 0.25, 0.25)


class _FakeDepthModel(torch.nn.Module):
    """Depth = mean of the normalised input, at the model resolution."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.calls = []

    def forward(self, pixel_values):
        self.calls.append(tuple(pixel_values.shape))
        return type("O", (), {"predicted_depth": pixel_values.mean(1) * self.w})()


def _run_depth(monkeypatch, img, **kw):
    from radiance.nodes.vfx import depth
    model = _FakeDepthModel()
    monkeypatch.setattr(depth, "download_and_load_model", lambda size, dev: (model, _FakeProcessor()))
    out = depth.RadianceDepthMapGenerator().generate_depth(img, "Small (25M - Fast)", use_gpu=False, **kw)[0]
    return out, model


def test_depth_batches_frames_and_returns_a_cpu_image(monkeypatch):
    img = torch.rand(5, 40, 60, 3)
    out, model = _run_depth(monkeypatch, img)
    assert out.shape == (5, 40, 60, 3) and out.device.type == "cpu"
    assert max(c[0] for c in model.calls) > 1, "frames still go through the model one at a time"
    assert float(out.min()) == pytest.approx(0.0, abs=1e-6) and float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_depth_chunked_equals_one_pass(monkeypatch):
    import radiance.core.tensor.chunking as ch
    img = torch.rand(4, 30, 44, 3) * 1.3
    one, _ = _run_depth(monkeypatch, img, blur_edges=1.0)
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    many, _ = _run_depth(monkeypatch, img, blur_edges=1.0)
    assert torch.equal(one, many)


def test_depth_resize_size_matches_the_processor():
    from radiance.nodes.vfx.depth import _dpt_resize_size
    try:
        from transformers.models.dpt.image_processing_dpt import get_resize_output_image_size
    except Exception:  # noqa: BLE001 - transformers absent or a different layout
        pytest.skip("transformers DPT processor not importable")
    for h, w in [(2160, 3840), (1086, 1920), (576, 1024), (1080, 1080), (300, 2000)]:
        ref = get_resize_output_image_size(torch.zeros(3, h, w), (518, 518), True, 14)
        ref = (ref.height, ref.width) if hasattr(ref, "height") else tuple(ref)   # SizeDict (v5) or tuple (v4)
        assert _dpt_resize_size(h, w, 518, 518, 14) == tuple(int(v) for v in ref)
