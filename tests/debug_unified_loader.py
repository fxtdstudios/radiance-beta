import unittest
import sys
import os
import torch
from unittest.mock import MagicMock, patch

# Mock ComfyUI modules BEFORE importing radiance
sys.modules["folder_paths"] = MagicMock()
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.sd"] = MagicMock()
sys.modules["comfy.utils"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()

import folder_paths
import comfy.sd
import comfy.utils

# Setup mocks
folder_paths.get_filename_list.return_value = ["model.safetensors", "clip.safetensors", "vae.safetensors"]
folder_paths.get_full_path.side_effect = lambda type, name: f"/mock/path/{type}/{name}"
folder_paths.get_folder_paths.return_value = "/mock/embeddings"

# Mock Enum for CLIPType
class MockCLIPType:
    FLUX = 1
    SD3 = 2
    STABLE_DIFFUSION = 3
comfy.sd.CLIPType = MockCLIPType

# Import the module under test
# We need to import it after mocks are set up
# Assuming the file is at d:\Space\ComfyUI\custom_nodes\radiance\nodes_loader.py
# We add the parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes_loader import RadianceUnifiedLoader, _cache, estimate_vram_usage

class TestRadianceUnifiedLoader(unittest.TestCase):
    
    def setUp(self):
        self.loader = RadianceUnifiedLoader()
        _cache.clear() # Clear cache before each test
        
        # Reset mocks
        comfy.sd.load_diffusion_model.reset_mock()
        comfy.sd.load_clip.reset_mock()
        comfy.sd.VAE.reset_mock()
        comfy.sd.load_controlnet.reset_mock()
        
        # Default mock returns
        self.mock_model = MagicMock()
        self.mock_clip = MagicMock()
        self.mock_vae = MagicMock()
        self.mock_controlnet = MagicMock()
        
        comfy.sd.load_diffusion_model.return_value = self.mock_model
        comfy.sd.load_clip.return_value = self.mock_clip
        # VAE is instantiated
        comfy.sd.VAE.return_value = self.mock_vae
        comfy.sd.load_controlnet.return_value = self.mock_controlnet

    def test_preset_application(self):
        """Verify that presets override user inputs correctly."""
        print("\n--- Testing Preset Application ---")
        
        # Test: Flux Dev preset should force dual_clip=True and specific dtypes
        model, clip, vae, controlnet, info = self.loader.load_radiance_stack(
            preset="→ Flux Dev",
            unet_name="flux_dev.sft",
            weight_dtype="default", # Should be overridden to fp8_e4m3fn
            clip_name1="t5xxl.fp16.safetensors",
            model_type="sdxl", # Should be overridden to flux
            clip_dtype="default", # Should be overridden to fp16
            device="cpu", # Should be overridden to default
            vae_name="ae.sft",
            clip_name2="clip_l.safetensors"
        )
        
        # Verify Mocks called with correct args
        # UNET load options
        # flux dev preset -> weight_dtype="fp8_e4m3fn"
        call_args = comfy.sd.load_diffusion_model.call_args
        print(f"UNET Load Args: {call_args}")
        self.assertEqual(call_args[1]['model_options']['dtype'], torch.float8_e4m3fn)
        
        # CLIP load options
        # flux dev preset -> clip_dtype="fp16", model_type="flux"
        call_args = comfy.sd.load_clip.call_args
        print(f"CLIP Load Args: {call_args}")
        self.assertEqual(call_args[1]['clip_type'], MockCLIPType.FLUX)
        self.assertEqual(call_args[1]['model_options']['dtype'], torch.float16)

    def test_controlnet_params(self):
        """Verify ControlNet load and parameter handling."""
        print("\n--- Testing ControlNet Parameters ---")
        
        # Load with ControlNet params
        model, clip, vae, cnet, info = self.loader.load_radiance_stack(
            preset="None (Manual)",
            unet_name="model.sft",
            weight_dtype="default",
            clip_name1="clip.sft",
            model_type="sdxl",
            clip_dtype="default",
            device="default",
            vae_name="vae.sft",
            controlnet_name="canny.sft",
            controlnet_strength=0.8,
            controlnet_start=0.1,
            controlnet_end=0.9
        )
        
        self.assertIsNotNone(cnet)
        
        # We expect Radiance to attach these parameters to the object
        # so a downstream node can use them.
        print(f"Checking for 'radiance_strength' attribute...")
        has_metadata = hasattr(cnet, "radiance_strength")
        
        if not has_metadata:
             print("FAIL: ControlNet object missing expected 'radiance_strength' metadata.")
        
        self.assertTrue(has_metadata, "ControlNet params were ignored!")
        self.assertEqual(cnet.radiance_strength, 0.8)
        self.assertEqual(cnet.radiance_start, 0.1)
        self.assertEqual(cnet.radiance_end, 0.9)

    def test_cache_leaks(self):
        """Verify cache growth."""
        print("\n--- Testing Cache Growth ---")
        
        for i in range(5):
            # Mock different paths to force caching
            with patch('comfy.sd.load_diffusion_model', return_value=MagicMock()) as mock_load:
                self.loader.load_radiance_stack(
                    preset="None (Manual)",
                    unet_name=f"model_{i}.sft",
                    weight_dtype="default",
                    clip_name1="clip.sft",
                    model_type="sdxl",
                    clip_dtype="default",
                    device="default",
                    vae_name="vae.sft"
                )
        
        print(f"Cache size: {_cache.size}")
        # If cache is simple dict, it should be 5 + clips + vaes
        # We want to enforce a limit, say 3.
        # If it's > 10, it implies unbounded growth.
        # Currently we expect > 5.
        
        # We assert that it *should* be limited (e.g. <= 4 for models)
        # But for this debug script, we just want to see it fail or print the size.
        if _cache.size > 10:
             print("FAIL: Cache size is growing unbounded!")
             
        # Ideally we want an LRU cache.
        # Unbounded cache is dangerous.
        
    def test_lora_ignores(self):
        """Verify LoRA parameters."""
        # TODO: LoRA application logic uses comfy.sd.load_lora_for_models
        # which returns (model, clip). So strength IS applied there.
        pass

if __name__ == '__main__':
    unittest.main()
