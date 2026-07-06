"""Functional round-trip tests for RadianceWrite/RadianceRead.

Unlike test_io.py (API-surface only: registration, INPUT_TYPES, RETURN_TYPES),
these tests actually write real pixel data to disk and read it back, checking
the values survive intact. test_io.py's API-only coverage is exactly why the
16-bit PNG/TIFF crash, the 16-bit-precision-crushing read bug, the EXR
metadata loss, and the 8-bit-only video precision bug (2026-07 IO audit) went
undetected for a long time -- none of them affect INPUT_TYPES or RETURN_TYPES.
"""
import importlib
import shutil
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nodes_io = importlib.import_module("radiance.nodes_io")

HAS_FFMPEG = shutil.which("ffmpeg") is not None
skip_no_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


def _write_and_read(tmp_path, image_np, fmt, **write_kwargs):
    writer = nodes_io.RadianceWrite()
    reader = nodes_io.RadianceRead()
    img_t = torch.from_numpy(image_np)
    out_path = str(tmp_path / "test_out")
    writer.write(image=img_t, output_path=out_path, format=fmt, overwrite=True, **write_kwargs)
    produced = list(tmp_path.glob("test_out*"))
    assert produced, f"No file produced for format {fmt!r}"
    img_out, mask_out = reader.read(path=str(produced[0]))
    return img_out[0].numpy(), mask_out, produced[0]


class TestImageRoundTrip:
    """Every RadianceWrite image format, verified with real pixel data."""

    @pytest.mark.parametrize("fmt,atol", [
        ("IMG │ PNG (8-bit)", 1 / 255),
        ("IMG │ TIFF (16-bit)", 1 / 65535 * 2),
        ("IMG │ TIFF (32-bit float)", 1e-6),
        ("IMG │ EXR (16-bit half)", 1e-3),
        ("IMG │ EXR (32-bit float)", 1e-6),
        ("IMG │ DPX", 0.003),                    # 10-bit quantization
        ("IMG │ Radiance HDR (.hdr)", 0.02),
    ])
    def test_round_trip(self, tmp_path, fmt, atol):
        # Random per-pixel noise: a fair precision test for lossless formats.
        img = (np.random.default_rng(0).random((16, 16, 3)) * 0.9 + 0.05).astype(np.float32)
        out, _, _ = _write_and_read(tmp_path, img[None], fmt)
        assert np.abs(out - img).max() < atol, f"{fmt}: precision loss too high"

    @pytest.mark.parametrize("fmt,atol", [
        ("IMG │ JPEG", 0.05),
        ("IMG │ WEBP", 0.05),
    ])
    def test_lossy_round_trip(self, tmp_path, fmt, atol):
        # Smooth gradient, not random noise: chroma-subsampled lossy codecs
        # smear color between adjacent pixels, so per-pixel random noise is
        # an adversarial worst case, not a realistic fidelity check.
        ramp = np.linspace(0.05, 0.95, 32, dtype=np.float32)
        img = np.stack([np.tile(ramp, (32, 1))] * 3, axis=-1)
        out, _, _ = _write_and_read(tmp_path, img[None], fmt)
        assert np.abs(out - img).max() < atol, f"{fmt}: precision loss too high"

    def test_png_16bit_is_genuinely_16bit(self, tmp_path):
        """Regression: PNG (16-bit) used to write real TIFF bytes under a
        .png name (tifffile.imwrite() ignores the target extension)."""
        # 4096 distinct pixel positions -- more than the 256 levels an 8-bit
        # fallback could ever produce, so real 16-bit precision is provable.
        ramp = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64, 1)
        img = np.repeat(ramp, 3, axis=-1)
        out, _, path = _write_and_read(tmp_path, img[None], "IMG │ PNG (16-bit)")
        with open(path, "rb") as f:
            magic = f.read(8)
        assert magic == b"\x89PNG\r\n\x1a\n", "PNG (16-bit) is not a real PNG file!"
        levels = len(np.unique((out[..., 0] * 65535).round().astype(np.uint16)))
        assert levels > 256, "16-bit PNG precision was silently crushed to 8-bit"


class TestAlphaRoundTrip:
    @pytest.mark.parametrize("fmt", [
        "IMG │ PNG (8-bit)", "IMG │ PNG (16-bit)", "IMG │ EXR (32-bit float)",
    ])
    def test_mask_round_trip(self, tmp_path, fmt):
        img = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
        mask = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(1, 16, 16)
        writer = nodes_io.RadianceWrite()
        reader = nodes_io.RadianceRead()
        out_path = str(tmp_path / "alpha_out")
        writer.write(image=torch.from_numpy(img[None]), output_path=out_path, format=fmt,
                     mask=torch.from_numpy(mask), overwrite=True)
        produced = list(tmp_path.glob("alpha_out*"))
        assert produced
        _, mask_out = reader.read(path=str(produced[0]))
        assert mask_out is not None and mask_out.abs().max().item() > 0.5, \
            f"{fmt}: alpha not written/read correctly"


class Test16BitPrecisionPreservation:
    """Guards the RGB 16-bit read fix -- a genuine 16-bit source must not be
    silently collapsed to 8-bit by Pillow (which has no internal 16-bit RGB
    mode and reports pil.mode == 'RGB' regardless of source depth)."""

    def test_genuine_16bit_png_preserves_precision(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        rgb16 = (np.random.default_rng(2).random((16, 16, 3)) * 65535).astype(np.uint16)
        path = tmp_path / "rgb16.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))
        img_out, _ = nodes_io.RadianceRead().read(path=str(path))
        readback = (img_out[0].numpy() * 65535).round().astype(np.int32)
        assert np.abs(readback - rgb16.astype(np.int32)).max() <= 1

    def test_8bit_png_unaffected(self, tmp_path):
        PIL = pytest.importorskip("PIL.Image")
        rgb8 = (np.random.default_rng(3).random((16, 16, 3)) * 255).astype(np.uint8)
        path = tmp_path / "rgb8.png"
        PIL.fromarray(rgb8).save(path)
        img_out, _ = nodes_io.RadianceRead().read(path=str(path))
        readback = (img_out[0].numpy() * 255).round().astype(np.int32)
        assert np.abs(readback - rgb8.astype(np.int32)).max() <= 1


@skip_no_ffmpeg
class TestVideoFormats:
    @pytest.mark.parametrize("fmt", [
        "VID │ MP4 (H.264)",
        "VID │ MP4 (H.265 10-bit)",
        "VID │ MOV (ProRes 422 HQ)",
        "VID │ MOV (ProRes 4444)",
        "VID │ MOV (DNxHR HQ)",
    ])
    def test_writes_without_crash(self, tmp_path, fmt):
        # DNxHR requires >= 256x120 frames; use a comfortably larger size.
        img = np.random.default_rng(4).random((4, 192, 320, 3)).astype(np.float32)
        writer = nodes_io.RadianceWrite()
        out_path = str(tmp_path / "vid_out")
        writer.write(image=torch.from_numpy(img), output_path=out_path, format=fmt,
                     fps=24.0, overwrite=True)
        assert list(tmp_path.glob("vid_out*")), f"No output for {fmt!r}"

    def test_prores_422hq_precision_gain(self, tmp_path):
        """Regression: video frames used to go through an 8-bit PNG
        intermediate, capping every format's real precision at 8 bits."""
        w, h = 1024, 128
        ramp = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        frames = np.stack([np.stack([ramp] * 3, axis=-1)] * 2, axis=0)
        writer = nodes_io.RadianceWrite()
        out_path = str(tmp_path / "prores_out")
        writer.write(image=torch.from_numpy(frames), output_path=out_path,
                     format="VID │ MOV (ProRes 422 HQ)", fps=24.0, overwrite=True)
        produced = list(tmp_path.glob("prores_out*"))
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(produced[0]), "-f", "rawvideo",
               "-pix_fmt", "rgb48le", "-frames:v", "1", "pipe:1"]
        result = subprocess.run(cmd, capture_output=True, check=True)
        arr = np.frombuffer(result.stdout, dtype="<u2").reshape(h, w, 3)
        levels = len(np.unique(arr[0, :, 0]))
        assert levels > 256, "ProRes 422 HQ precision capped at 8-bit again!"

    def test_audio_mux(self, tmp_path):
        sr = 44100
        wave = torch.from_numpy((np.random.default_rng(5).random((1, 2, sr)) * 0.1).astype(np.float32))
        audio = {"waveform": wave, "sample_rate": sr}
        img = np.random.default_rng(6).random((6, 192, 320, 3)).astype(np.float32)
        writer = nodes_io.RadianceWrite()
        out_path = str(tmp_path / "audio_out")
        writer.write(image=torch.from_numpy(img), output_path=out_path,
                     format="VID │ MOV (ProRes 422 HQ)", fps=24.0, audio=audio, overwrite=True)
        produced = list(tmp_path.glob("audio_out*"))
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(produced[0])],
            capture_output=True, text=True,
        )
        assert "audio" in probe.stdout


class TestSequenceRoundTrip:
    @pytest.mark.parametrize("fmt,ext", [
        ("SEQ │ PNG (8-bit)", ".png"),
        ("SEQ │ EXR (32-bit float)", ".exr"),
    ])
    def test_sequence_writes_all_frames(self, tmp_path, fmt, ext):
        n = 3
        img = np.random.default_rng(7).random((n, 16, 16, 3)).astype(np.float32)
        writer = nodes_io.RadianceWrite()
        out_path = str(tmp_path / "seq_out")
        writer.write(image=torch.from_numpy(img), output_path=out_path, format=fmt,
                     start_frame=1, frame_padding=3, overwrite=True)
        produced = sorted((tmp_path / "seq_out").glob(f"*{ext}"))
        assert len(produced) == n


class TestErrorHandling:
    def test_depth_only_exr_raises_clear_error(self, tmp_path):
        """Regression: used to raise a bare TypeError from OpenEXR, caught
        silently by RadianceRead.read()'s outer handler and surfaced only as
        a black image with no indication of the actual cause."""
        OpenEXR = pytest.importorskip("OpenEXR")
        path = tmp_path / "depth_only.exr"
        z = np.random.default_rng(8).random((8, 8)).astype(np.float32)
        OpenEXR.File(
            {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, {"Z": z}
        ).write(str(path))
        with pytest.raises(ValueError, match="R/G/B"):
            nodes_io._read_exr_single(str(path))
