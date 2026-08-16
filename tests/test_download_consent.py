"""Model weights must not be fetched without the operator saying so.

Real-ESRGAN, HAT-L, SwinIR, Depth Anything V2 and DSINE all download on first
use — 67 MB to 2.4 GB between them — and it used to begin the moment a graph
was queued. `nodes/upscale` had an opt-*out* (`RADIANCE_UPSCALE_OFFLINE=1`);
`nodes/vfx/multipass` had no gate at all. The default is now ask-first, in one
place, for both.
"""
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.core.consent import (
    ALLOW_ENV,
    LEGACY_UPSCALE_OFFLINE_ENV,
    downloads_allowed,
    refusal_message,
    require_consent,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    monkeypatch.delenv(LEGACY_UPSCALE_OFFLINE_ENV, raising=False)


class TestTheDefaultIsNo:

    def test_unset_means_do_not_download(self):
        """The whole point: a fresh install does not spend someone's bandwidth."""
        assert downloads_allowed() is False

    def test_explicit_opt_in_allows_it(self, monkeypatch):
        monkeypatch.setenv(ALLOW_ENV, "1")
        assert downloads_allowed() is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_spellings(self, monkeypatch, value):
        monkeypatch.setenv(ALLOW_ENV, value)
        assert downloads_allowed() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off"])
    def test_falsy_spellings(self, monkeypatch, value):
        monkeypatch.setenv(ALLOW_ENV, value)
        assert downloads_allowed() is False

    def test_garbage_falls_back_to_refusing(self, monkeypatch):
        monkeypatch.setenv(ALLOW_ENV, "maybe")
        assert downloads_allowed() is False


class TestTheLegacyOptOutStillCounts:
    """Studios already set RADIANCE_UPSCALE_OFFLINE=1; it must not start
    downloading again because the mechanism changed underneath them."""

    def test_legacy_offline_still_refuses(self, monkeypatch):
        monkeypatch.setenv(LEGACY_UPSCALE_OFFLINE_ENV, "1")
        assert downloads_allowed(legacy_offline_env=LEGACY_UPSCALE_OFFLINE_ENV) is False

    def test_explicit_allow_beats_the_legacy_flag(self, monkeypatch):
        monkeypatch.setenv(LEGACY_UPSCALE_OFFLINE_ENV, "1")
        monkeypatch.setenv(ALLOW_ENV, "1")
        assert downloads_allowed(legacy_offline_env=LEGACY_UPSCALE_OFFLINE_ENV) is True


class TestTheRefusalIsActionable:

    def test_it_names_the_file_size_destination_and_the_way_out(self):
        msg = refusal_message("Real-ESRGAN x4plus", size_mb=67,
                              dest="/models/upscale/x4.pth",
                              url="https://example.invalid/x4.pth")
        assert "Real-ESRGAN x4plus" in msg
        assert "67 MB" in msg
        assert "/models/upscale/x4.pth" in msg
        assert "https://example.invalid/x4.pth" in msg
        assert ALLOW_ENV in msg, "the message must say how to allow it"

    def test_require_consent_logs_the_refusal(self, caplog):
        with caplog.at_level("ERROR"):
            allowed = require_consent("Depth Anything V2 Large", size_mb=1340)
        assert allowed is False
        assert "Depth Anything V2 Large" in caplog.text


class TestBothDownloadersAreGated:
    """One helper, used by both — not two mechanisms with one missing."""

    def test_upscale_downloader_refuses_by_default(self, monkeypatch, tmp_path):
        pytest.importorskip("torch")
        from radiance.nodes.upscale import upscale as up

        def _boom(*a, **k):
            raise AssertionError("a download was attempted without consent")

        monkeypatch.setattr(up, "_get_models_dir", lambda sub: str(tmp_path))
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        monkeypatch.setattr("urllib.request.urlretrieve", _boom, raising=False)

        key = next(iter(up._UPSCALE_MODEL_REGISTRY))
        assert up._download_upscale_model(key) is None

    def test_multipass_downloader_refuses_by_default(self, monkeypatch, tmp_path):
        """This path previously had no gate whatsoever."""
        pytest.importorskip("torch")
        from radiance.nodes.vfx.multipass import core

        def _boom(*a, **k):
            raise AssertionError("a download was attempted without consent")

        monkeypatch.setattr(core, "_get_comfy_models_dir", lambda sub: str(tmp_path))
        monkeypatch.setattr("urllib.request.urlretrieve", _boom, raising=False)

        key = next(iter(core._MODEL_REGISTRY))
        assert core._download_model(key) is None

    def test_the_legacy_upscale_flag_still_blocks(self, monkeypatch, tmp_path):
        pytest.importorskip("torch")
        from radiance.nodes.upscale import upscale as up

        monkeypatch.setenv(LEGACY_UPSCALE_OFFLINE_ENV, "1")
        assert up._offline_mode() is True

    def test_opting_in_reports_downloads_as_permitted(self, monkeypatch):
        pytest.importorskip("torch")
        from radiance.nodes.upscale import upscale as up

        monkeypatch.setenv(ALLOW_ENV, "1")
        assert up._offline_mode() is False
