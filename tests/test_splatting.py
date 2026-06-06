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
        assert set(M) == {"RadianceSplatLoad", "RadianceSplatInfo", "RadianceSplatExport"}
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
