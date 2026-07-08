"""Tests for model/detect.py architecture heuristics."""
import torch
import pytest

from radiance.model.detect import detect_model_type

HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="writes real safetensors files, needs real torch/safetensors."
)


def _flux2_keys(n_single_blocks, n_double_blocks=8):
    keys = ["double_stream_modulation_img.weight"]
    keys += [f"double_blocks.{i}.img_attn.qkv.weight" for i in range(n_double_blocks)]
    keys += [f"single_blocks.{i}.linear1.weight" for i in range(n_single_blocks)]
    return keys


def _write_fake_checkpoint(path, keys):
    from safetensors.torch import save_file
    save_file({k: torch.zeros(1) for k in keys}, path)


# ALBABIT-FIX: Flux.2 Dev and Flux.2 Klein share the double_stream_modulation_img
# key -- only single_blocks depth tells them apart (measured on real checkpoints:
# Dev=48, Klein 9B=24, Klein Base 4B=20).
def test_flux2_dev_detected_by_single_block_count(tmp_path):
    path = str(tmp_path / "dev.safetensors")
    _write_fake_checkpoint(path, _flux2_keys(n_single_blocks=48))
    assert detect_model_type(path) == "flux2"


def test_flux2_klein_9b_detected_by_single_block_count(tmp_path):
    path = str(tmp_path / "klein9b.safetensors")
    _write_fake_checkpoint(path, _flux2_keys(n_single_blocks=24))
    assert detect_model_type(path) == "flux2-klein"


def test_flux2_klein_base_4b_detected_by_single_block_count(tmp_path):
    path = str(tmp_path / "klein_base_4b.safetensors")
    _write_fake_checkpoint(path, _flux2_keys(n_single_blocks=20, n_double_blocks=5))
    assert detect_model_type(path) == "flux2-klein"
