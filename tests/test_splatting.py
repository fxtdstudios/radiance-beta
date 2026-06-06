"""Tests for the Gaussian Splatting data type, .ply IO, and IO nodes.

All CPU/numpy — no GPU or gsplat required, so these run in CI.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from radiance.splatting.data import Splat
from radiance.splatting.ply import load_ply, save_ply


def _make(n=64, degree=1, seed=0):
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2
    return Splat(
        means=rng.standard_normal((n, 3)).astype(np.float32),
        scales=rng.standard_normal((n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
        opacities=rng.standard_normal(n).astype(np.float32),
        sh=rng.standard_normal((n, k, 3)).astype(np.float32),
        sh_degree=degree,
    )


class TestSplatData:
    def test_validate_ok(self):
        _make().validate()

    def test_count_and_coeffs(self):
        s = _make(n=10, degree=2)
        assert s.count == 10
        assert s.sh_coeffs == 9  # (2+1)^2

    def test_bad_shape_raises(self):
        s = _make(n=10)
        s.scales = s.scales[:5]
        with pytest.raises(ValueError):
            s.validate()

    def test_degree_mismatch_raises(self):
        s = _make(n=10, degree=1)
        s.sh_degree = 2
        with pytest.raises(ValueError):
            s.validate()

    def test_info_text(self):
        t = _make(n=7).info()
        assert "points" in t and "7" in t


class TestPlyRoundtrip:
    @pytest.mark.parametrize("degree", [0, 1, 2])
    def test_roundtrip(self, degree, tmp_path):
        s = _make(n=50, degree=degree, seed=degree)
        path = str(tmp_path / "scene.ply")
        save_ply(s, path)
        r = load_ply(path)
        assert r.count == s.count
        assert r.sh_degree == s.sh_degree
        np.testing.assert_allclose(r.means, s.means)
        np.testing.assert_allclose(r.scales, s.scales)
        np.testing.assert_allclose(r.quats, s.quats)
        np.testing.assert_allclose(r.opacities, s.opacities)
        np.testing.assert_allclose(r.sh, s.sh)

    def test_source_meta(self, tmp_path):
        path = str(tmp_path / "abc.ply")
        save_ply(_make(), path)
        assert load_ply(path).meta["source"] == "abc.ply"

    def test_missing_props_raises(self, tmp_path):
        path = str(tmp_path / "bad.ply")
        with open(path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
                    b"property float x\nend_header\n")
        with pytest.raises(ValueError):
            load_ply(path)


class TestNodes:
    def test_nodes_registered(self):
        from radiance.nodes.splatting import NODE_CLASS_MAPPINGS as M
        assert set(M) == {"RadianceSplatLoad", "RadianceSplatInfo", "RadianceSplatExport",
                          "RadianceCameraOrbit", "RadianceSplatRender",
                          "RadianceColmapLoad", "RadianceSplatTrain"}
        for cls in M.values():
            assert cls.INPUT_TYPES()
            assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)

    def test_load_node(self, tmp_path):
        from radiance.nodes.splatting.io import RadianceSplatLoad, RadianceSplatExport
        path = str(tmp_path / "n.ply")
        save_ply(_make(n=12), path)
        splat, info = RadianceSplatLoad().load(path)
        assert splat.count == 12 and "points" in info
        out = RadianceSplatExport().export(splat, str(tmp_path / "out.ply"))
        assert os.path.isfile(out["result"][0])


from radiance.splatting.cameras import orbit, look_at, Cameras
from radiance.splatting.backend import HAS_GSPLAT


class TestCameras:
    def test_orbit_shapes(self):
        c = orbit(num_frames=8, width=320, height=240)
        assert len(c) == 8
        assert c.viewmats.shape == (8, 4, 4)
        assert c.Ks.shape == (8, 3, 3)
        assert (c.width, c.height) == (320, 240)

    def test_viewmats_orthonormal(self):
        c = orbit(num_frames=5)
        for v in c.viewmats:
            R = v[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)
            assert abs(np.linalg.det(R) - 1.0) < 1e-4

    def test_centers_on_radius(self):
        center = np.array([1.0, 2.0, -1.0], np.float32)
        c = orbit(num_frames=12, radius=4.0, elevation_deg=0.0, center=tuple(center))
        d = np.linalg.norm(c.centers() - center, axis=1)
        np.testing.assert_allclose(d, 4.0, atol=1e-3)

    def test_cameras_look_at_center(self):
        center = np.zeros(3, np.float32)
        c = orbit(num_frames=6, radius=3.0, elevation_deg=10.0)
        centers = c.centers()
        for i, v in enumerate(c.viewmats):
            fwd = v[2, :3]
            to_center = center - centers[i]
            to_center /= np.linalg.norm(to_center)
            assert float(fwd @ to_center) > 0.99


@pytest.mark.skipif(not HAS_GSPLAT, reason="gsplat not installed (CUDA render path)")
def test_render_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    from radiance.splatting.backend import render
    s = _make(n=200, degree=0)
    cams = orbit(num_frames=2, width=64, height=48)
    image, depth, alpha = render(s, cams)
    assert tuple(image.shape) == (2, 48, 64, 3)
    assert tuple(alpha.shape[:3]) == (2, 48, 64)


import struct
from radiance.splatting.init import init_from_points, rgb_to_sh, inverse_sigmoid
from radiance.splatting.colmap import load_colmap
from radiance.splatting.train import TrainConfig, HAS_GSPLAT as TRAIN_HAS_GSPLAT


class TestInit:
    def test_init_shapes(self):
        pts = np.random.default_rng(0).standard_normal((30, 3)).astype(np.float32)
        s = init_from_points(pts, sh_degree=3)
        assert s.count == 30
        assert s.sh_coeffs == 16
        s.validate()

    def test_identity_quats_and_dc(self):
        pts = np.zeros((4, 3), np.float32)
        cols = np.array([[1, 0, 0]] * 4, np.float32)
        s = init_from_points(pts, cols, sh_degree=0)
        np.testing.assert_allclose(s.quats, np.tile([1, 0, 0, 0], (4, 1)), atol=1e-6)
        np.testing.assert_allclose(s.sh[:, 0, :], rgb_to_sh(cols), rtol=1e-5)

    def test_accepts_0_255_colors(self):
        s = init_from_points(np.zeros((2, 3), np.float32),
                             np.array([[255, 255, 255], [0, 0, 0]], np.float32), sh_degree=0)
        s.validate()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            init_from_points(np.zeros((0, 3), np.float32))


def _write_colmap_txt(d):
    with open(d / "cameras.txt", "w") as f:
        f.write("# cam\n1 PINHOLE 64 48 50 50 32 24\n")
    with open(d / "images.txt", "w") as f:
        f.write("1 1 0 0 0 0 0 5 1 a.png\n\n")
        f.write("2 0.9239 0 0.3827 0 1 0 4 1 b.png\n\n")
    with open(d / "points3D.txt", "w") as f:
        f.write("1 0.1 0.2 0.3 255 0 0 0.5\n2 -0.1 0.0 1.0 0 255 0 0.4\n")


def _write_colmap_bin(d):
    with open(d / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, 64, 48))      # id, PINHOLE, w, h
        f.write(struct.pack("<dddd", 50, 50, 32, 24))    # fx fy cx cy
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<idddddddi", 1, 1, 0, 0, 0, 0, 0, 5, 1))
        f.write(b"a.png\x00")
        f.write(struct.pack("<Q", 0))
    with open(d / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<QdddBBBd", 1, 0.1, 0.2, 0.3, 255, 0, 0, 0.5))
        f.write(struct.pack("<Q", 0))


class TestColmap:
    def test_load_text(self, tmp_path):
        _write_colmap_txt(tmp_path)
        cams, pts, cols, names = load_colmap(str(tmp_path))
        assert len(cams) == 2
        assert pts.shape == (2, 3) and cols.shape == (2, 3)
        np.testing.assert_allclose(cams.Ks[0], [[50, 0, 32], [0, 50, 24], [0, 0, 1]], atol=1e-4)
        R = cams.viewmats[0, :3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)

    def test_load_binary(self, tmp_path):
        _write_colmap_bin(tmp_path)
        cams, pts, cols, names = load_colmap(str(tmp_path))
        assert len(cams) == 1 and pts.shape == (1, 3)
        np.testing.assert_allclose(pts[0], [0.1, 0.2, 0.3], atol=1e-5)
        np.testing.assert_allclose(cols[0], [255, 0, 0], atol=1e-3)

    def test_missing_dir_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            load_colmap("/no/such/colmap/dir")

    def test_load_images_matched(self, tmp_path):
        from PIL import Image
        from radiance.splatting.colmap import load_images
        _write_colmap_txt(tmp_path)
        _, _, _, names = load_colmap(str(tmp_path))
        for nm in names:
            Image.new("RGB", (10, 8), (120, 30, 200)).save(tmp_path / nm)
        arr = load_images(str(tmp_path), names, 64, 48)
        assert arr.shape == (len(names), 48, 64, 3)
        assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0


def test_train_config_defaults():
    c = TrainConfig()
    assert c.steps > 0 and 0 <= c.sh_degree <= 4
    assert c.densify is True and c.refine_every > 0 and 0.0 <= c.ssim_lambda <= 1.0


@pytest.mark.skipif(not TRAIN_HAS_GSPLAT, reason="gsplat not installed (CUDA train path)")
def test_train_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    from radiance.splatting.train import train
    pts = np.random.default_rng(0).standard_normal((500, 3)).astype(np.float32)
    init = init_from_points(pts, sh_degree=0)
    cams = orbit(num_frames=2, width=32, height=24)
    imgs = np.zeros((2, 24, 32, 3), np.float32)
    out = train(imgs, cams, init, TrainConfig(steps=3, sh_degree=0))
    assert out.count == init.count
