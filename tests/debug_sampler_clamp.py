
import unittest
import sys
import os
import torch
from unittest.mock import MagicMock

# ═══════════════════════════════════════════════════════════════════════════════
#                         MOCKING COMFYUI MODULES
# ═══════════════════════════════════════════════════════════════════════════════
# Mock structure must be set up BEFORE importing nodes_sampler
mock_comfy = MagicMock()
mock_samplers = MagicMock()
mock_sample = MagicMock()
mock_model_management = MagicMock()
mock_utils = MagicMock()

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.samplers"] = mock_samplers
sys.modules["comfy.sample"] = mock_sample
sys.modules["comfy.model_management"] = mock_model_management
sys.modules["comfy.utils"] = mock_utils

mock_comfy.samplers = mock_samplers
mock_comfy.sample = mock_sample
mock_comfy.model_management = mock_model_management
mock_comfy.utils = mock_utils

# CONSTANTS
mock_samplers.KSampler.SAMPLERS = ["euler", "dpmpp_2m"]
mock_samplers.KSampler.SCHEDULERS = ["simple", "normal", "karras", "sgm_uniform"]

# Mock folder_paths just in case
sys.modules["folder_paths"] = MagicMock()

# Import AFTER mocking
# We need to add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nodes_sampler

class TestRadianceSamplerClamping(unittest.TestCase):
    
    def setUp(self):
        self.sampler_node = nodes_sampler.RadianceSamplerPro()
        
        # Mock Valid Sigmas
        self.valid_sigmas = torch.linspace(1.0, 0.0, 21)
        mock_samplers.calculate_sigmas = MagicMock(return_value=self.valid_sigmas)
        mock_samplers.sampler_object = MagicMock(return_value=MagicMock())
        
        mock_model_management.get_torch_device = MagicMock(return_value=torch.device("cpu"))
        
        # Mock Model
        self.mock_model = MagicMock()
        model_sampling = MagicMock()
        model_sampling.sigma_max = torch.tensor(1.0)
        # Prevent flux detection
        del model_sampling.shift
        del model_sampling.flux_shift
        self.mock_model.get_model_object = MagicMock(return_value=model_sampling)
        
        # Input Latent
        self.mock_latent = {
            "samples": torch.zeros((1, 4, 16, 16)), # Small 16x16 latent
            "batch_index": None
        }

    def test_sampler_passthrough_range(self):
        """
        Verify that RadianceSamplerPro does NOT clamp output latents.
        We mock the internal sample_custom to return extreme values.
        """
        print("\n--- Testing Radiance Sampler for Clamping ---")
        
        # 1. Setup Mock Sampler to return EXTREME values
        # e.g. -100.0 (Super Black) and +100.0 (Super White) and NaN checks
        
        extreme_output = torch.zeros((1, 4, 16, 16))
        extreme_output[..., 0] = -100.0 # Extreme negative
        extreme_output[..., 1] = 100.0  # Extreme positive
        extreme_output[..., 2] = 0.0
        
        mock_sample.sample_custom = MagicMock(return_value=extreme_output)
        
        # 2. Call Sample
        out_latent, out_sigmas, out_string = self.sampler_node.sample(
            model=self.mock_model,
            positive=[],
            negative=[],
            latent_image=self.mock_latent,
            preset="None (Custom)",
            steps=20,
            start_step=0,
            end_step=0,
            cfg=1.0,
            sampler="euler",
            sampler_mode="Standard",
            phase_split=0.4,
            scheduler="simple",
            scheduler_mode="Manual",
            denoise=1.0,
            flux_shift=1.0,
            flux_guidance=3.5,
            flux_guidance_profile="Static",
            add_noise=False, # Don't add noise to simplify input check
            return_with_leftover_noise=False,
            seed=0
        )
        
        # 3. Verify Output
        samples = out_latent["samples"]
        min_val = samples.min().item()
        max_val = samples.max().item()
        
        print(f"Sampler Output Range: Min={min_val}, Max={max_val}")
        
        self.assertLess(min_val, -99.0, "Sampler should preserve extreme negatives")
        self.assertGreater(max_val, 99.0, "Sampler should preserve extreme positives")
        
        print("Confirmed: Radiance Sampler does NOT clamp output values.")

    def test_flux_sigmas_no_clamp(self):
        """
        Verify that sigmas calculation does not introduce weird clamping.
        """
        # Test huge shift
        shift = 10.0
        sigmas = nodes_sampler.get_flux_sigmas(
            self.mock_model, "simple", steps=10, denoise=1.0, shift=shift
        )
        print(f"Shifted Sigmas (Shift={shift}): Max={sigmas.max()}, Min={sigmas.min()}")
        
        self.assertGreater(sigmas.max(), 0.0)
        # Should be monotonic decreasing
        is_monotonic = torch.all(sigmas[:-1] >= sigmas[1:])
        self.assertTrue(is_monotonic, "Sigmas should be monotonically decreasing")

if __name__ == '__main__':
    unittest.main()
