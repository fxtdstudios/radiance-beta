import importlib.util
import json
import os
import sys
import types
import unittest

import torch
import pytest

# Only the tests marked @pytest.mark.real_torch below need genuine tensors; the
# rest run fine against conftest's stub. Opt out of the automatic module-level
# skip so they keep running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_NAMES = (
    "radiance", "radiance.hdr", "radiance.hdr.vae", "radiance.fast_vae",
    "radiance.color", "radiance.color.transfer", "radiance.color.pipeline",
)
PREVIOUS = {name: sys.modules.get(name) for name in MODULE_NAMES}


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


_package("radiance")
_package("radiance.hdr")
_package("radiance.color")


class FakeDecode:
    last_kwargs = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",), "vae": ("VAE",),
                "target_space": (["sRGB", "Linear"], {"default": "sRGB"}),
            },
            "optional": {
                "force_hdr_decode": ("BOOLEAN", {"default": False}),
                "hdr_mode": (["Clip (SDR)", "Compress (Log)"], {"default": "Clip (SDR)"}),
                "source_space": (["sRGB", "ARRI LogC4"], {"default": "sRGB"}),
                "display_tonemap": (["None", "Reinhard"], {"default": "None"}),
                "hdr_output": ("BOOLEAN", {"default": False}),
                "export_rhdr": ("BOOLEAN", {"default": False}),
                "rhdr_precision": (["f16", "f32"], {"default": "f16"}),
                "crop_padding": ("STRING", {"default": ""}),
                "processing_mode": (["sequential"], {"default": "sequential"}),
            },
        }

    def decode(self, **kwargs):
        FakeDecode.last_kwargs = kwargs
        channels = 4 if kwargs.get("alpha") is not None else 3
        image = torch.zeros(1, 2, 2, channels)
        metadata = {
            "resolution": "2x2",
            "latent_format": "engine_fmt",
        }
        if kwargs.get("export_rhdr"):
            metadata["rhdr_export"] = "preserved.rhdr"
        return image, json.dumps(metadata), "returned_fmt"


vae_module = types.ModuleType("radiance.hdr.vae")
vae_module.RadianceVAE4KDecode = FakeDecode
sys.modules["radiance.hdr.vae"] = vae_module

fast_module = types.ModuleType("radiance.fast_vae")
fast_module.decode_to_linear_realtime = lambda *args, **kwargs: None
fast_module.load_radiance_decoder_weights = lambda *args, **kwargs: None
fast_module.resolve_rudra_model_type = lambda *args, **kwargs: "test"
sys.modules["radiance.fast_vae"] = fast_module

transfer_module = types.ModuleType("radiance.color.transfer")
for name in ("tensor_linear_to_logc4", "tensor_linear_to_slog3", "tensor_srgb_to_linear", "tensor_linear_to_srgb"):
    setattr(transfer_module, name, lambda value: value)
sys.modules["radiance.color.transfer"] = transfer_module

pipeline_module = types.ModuleType("radiance.color.pipeline")
pipeline_module.apply_input_transform = lambda value, _space: value
pipeline_module.apply_output_transform = lambda value, _space: value
pipeline_module.INPUT_COLORSPACES = ["Linear (sRGB)"]
sys.modules["radiance.color.pipeline"] = pipeline_module

spec = importlib.util.spec_from_file_location(
    "hdr_vae_engine_contract", os.path.join(ROOT, "nodes", "generate", "engine.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RadianceHDRVAEDecode = module.RadianceHDRVAEDecode

for name, previous in PREVIOUS.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


class TestHDRVAEDecodeContract(unittest.TestCase):
    def setUp(self):
        RadianceHDRVAEDecode._engine = FakeDecode()
        FakeDecode.last_kwargs = None
        self.samples = {"samples": torch.zeros(1, 4, 2, 2)}

    def test_ui_defaults_are_sampler_safe_and_export_is_opt_in(self):
        types_map = RadianceHDRVAEDecode.INPUT_TYPES()
        optional = types_map["optional"]
        self.assertEqual(optional["decode_mode"][1]["default"], "Auto (Recommended)")
        self.assertEqual(optional["hdr_mode"][1]["default"], "Clip (SDR)")
        self.assertEqual(optional["source_space"][1]["default"], "sRGB")
        self.assertFalse(optional["export_rhdr"][1]["default"])

    @pytest.mark.real_torch
    def test_sampler_mode_blocks_log_inversion_and_preserves_engine_metadata(self):
        result = RadianceHDRVAEDecode().apply(
            self.samples, object(), hdr_mode="Compress (Log)",
            source_space="ARRI LogC4", display_tonemap="Reinhard",
            decode_mode="Sampler (SDR-safe)", export_rhdr=True,
        )
        kwargs = FakeDecode.last_kwargs
        self.assertEqual(kwargs["hdr_mode"], "Clip (SDR)")
        self.assertEqual(kwargs["source_space"], "sRGB")
        self.assertFalse(kwargs["force_hdr_decode"])
        self.assertFalse(kwargs["export_rhdr"])
        metadata = json.loads(result["result"][1])
        self.assertEqual(metadata["resolution"], "2x2")
        self.assertNotIn("rhdr_export", metadata)
        self.assertEqual(metadata["rudra_decoder"], "Disabled")

    @pytest.mark.real_torch
    def test_direct_hdr_mode_owns_scene_linear_contract(self):
        alpha = torch.ones(1, 2, 2, 3)
        result = RadianceHDRVAEDecode().apply(
            self.samples, object(), alpha=alpha,
            decode_mode="Direct HDR / RUDRA", export_rhdr=True,
        )
        kwargs = FakeDecode.last_kwargs
        self.assertEqual(kwargs["hdr_mode"], "Compress (Log)")
        self.assertEqual(kwargs["source_space"], "ARRI LogC4")
        self.assertEqual(kwargs["target_space"], "Linear")
        self.assertEqual(kwargs["display_tonemap"], "None")
        self.assertTrue(kwargs["force_hdr_decode"])
        self.assertTrue(kwargs["hdr_output"])
        self.assertTrue(kwargs["export_rhdr"])
        metadata = json.loads(result["result"][1])
        self.assertTrue(metadata["alpha_restored"])
        self.assertEqual(metadata["latent_format"], "returned_fmt")
        self.assertEqual(metadata["rhdr_export"], "preserved.rhdr")

    @pytest.mark.real_torch
    def test_auto_mode_recognizes_direct_hdr_encode_metadata(self):
        samples = {
            "samples": torch.zeros(1, 4, 2, 2),
            "radiance_meta": {"hdr_mode": "Compress (Log)", "source_space": "ARRI LogC4"},
        }
        RadianceHDRVAEDecode().apply(samples, object())
        self.assertTrue(FakeDecode.last_kwargs["force_hdr_decode"])
        self.assertEqual(FakeDecode.last_kwargs["target_space"], "Linear")

    def test_invalid_latent_payload_fails_early(self):
        with self.assertRaisesRegex(RuntimeError, "4D image latent or 5D video latent"):
            RadianceHDRVAEDecode().apply({"samples": "bad"}, object())

    @pytest.mark.real_torch
    def test_av_latent_decodes_only_the_video_stream(self):
        """MiniMax H3 (and any future AV model) hands this node a nested
        video+audio latent. This node only does HDR/tonemap/RUDRA on the
        video stream, same split ComfyUI's own LTXVSeparateAVLatent uses
        (unbind()[0] = video, [1] = audio); audio is a separate node's job."""
        video = torch.zeros(1, 24, 3, 2, 2)
        audio = torch.zeros(1, 32, 2, 5)

        class FakeAVLatent:
            is_nested = True

            def unbind(self):
                return [video, audio]

        RadianceHDRVAEDecode().apply({"samples": FakeAVLatent()}, object())
        self.assertIs(FakeDecode.last_kwargs["samples"]["samples"], video)


if __name__ == "__main__":
    unittest.main()
