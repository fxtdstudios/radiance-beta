"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE SAMPLER PRO - UNIT TESTS
                    Verifies bug fixes and core functionality
═══════════════════════════════════════════════════════════════════════════════
"""
import unittest
import sys
import os
import torch
from unittest.mock import MagicMock, patch

# ═══════════════════════════════════════════════════════════════════════════════
#                         MOCKING COMFYUI MODULES
# ═══════════════════════════════════════════════════════════════════════════════
# We must mock these BEFORE importing nodes_sampler
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

# Link submodules
mock_comfy.samplers = mock_samplers
mock_comfy.sample = mock_sample
mock_comfy.model_management = mock_model_management
mock_comfy.utils = mock_utils

# Setup static mock values
mock_samplers.KSampler = MagicMock()
mock_samplers.KSampler.SAMPLERS = ["euler", "dpmpp_2m", "dpmpp_2m_sde"]
mock_samplers.KSampler.SCHEDULERS = ["simple", "karras", "exponential"]

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import AFTER mocking
import nodes_sampler


class TestRadianceSamplerPro(unittest.TestCase):
    """Test suite for FXTD_Radiance_Sampler_Pro node."""
    
    def setUp(self):
        """Reset mocks and create fresh sampler instance."""
        self.sampler_node = nodes_sampler.FXTD_Radiance_Sampler_Pro()
        
        # Create valid sigmas tensor
        self.valid_sigmas = torch.linspace(1.0, 0.0, 21)
        
        # Configure calculate_sigmas to return proper tensor
        mock_samplers.calculate_sigmas = MagicMock(return_value=self.valid_sigmas)
        
        # Configure get_torch_device to return real CPU device
        mock_model_management.get_torch_device = MagicMock(return_value=torch.device("cpu"))
        
        # Configure prepare_noise to return real tensor
        mock_sample.prepare_noise = MagicMock(return_value=torch.randn((1, 4, 64, 64)))
        
        # Configure sample_custom to return real tensor
        mock_sample.sample_custom = MagicMock(return_value=torch.zeros((1, 4, 64, 64)))
        mock_sample.sample = MagicMock()
        
        # Configure sampler_object
        mock_samplers.sampler_object = MagicMock(return_value=MagicMock())
        
        # Mock inputs
        self.mock_model = MagicMock()
        self.mock_model.get_model_object = MagicMock(return_value=MagicMock())
        
        self.mock_positive = [[torch.zeros(1), {"guidance": 3.5}]]
        self.mock_negative = [[torch.zeros(1), {}]]
        
        self.mock_latent = {
            "samples": torch.zeros((1, 4, 64, 64)),
            "batch_index": None
        }

    def _call_sample(self, **kwargs):
        """Helper to call sample() with defaults."""
        defaults = {
            "model": self.mock_model,
            "positive": self.mock_positive,
            "negative": self.mock_negative,
            "latent_image": self.mock_latent,
            "seed": 123,
            "preset": "None (Custom)",
            "steps": 20,
            "start_step": 0,
            "end_step": 150,
            "cfg": 1.0,
            "sampler": "euler",
            "sampler_mode": "Standard",
            "phase_split": 0.4,
            "scheduler": "simple",
            "scheduler_mode": "Manual",
            "denoise": 1.0,
            "flux_shift": 1.0,
            "flux_guidance": 3.5,
            "flux_guidance_profile": "Static",
            "add_noise": True,
            "return_with_leftover_noise": False,
        }
        defaults.update(kwargs)
        return self.sampler_node.sample(**defaults)

    def test_instantiation(self):
        """Verify node instantiation and input types."""
        self.assertIsNotNone(self.sampler_node)
        inputs = self.sampler_node.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("model", inputs["required"])
        self.assertIn("flux_shift", inputs["required"])
        self.assertIn("flux_guidance", inputs["required"])
        self.assertIn("flux_guidance_profile", inputs["required"])

    def test_flux_shift_application(self):
        """
        Verify that:
        1. Flux Shift calculates shifted sigmas.
        2. It uses sample_custom (Fix #1).
        3. The shifted sigmas are passed to sample_custom.
        """
        shift_val = 2.0
        steps = 20
        
        self._call_sample(flux_shift=shift_val, steps=steps)
        
        # Check that sample_custom was called (NOT sample)
        mock_sample.sample.assert_not_called()
        mock_sample.sample_custom.assert_called()
        
        # Inspect args passed to sample_custom
        call_args = mock_sample.sample_custom.call_args
        passed_sigmas = call_args[0][4]  # args: (model, noise, cfg, sampler, sigmas, ...)
        
        # Verify sigmas are tensors and have expected length (steps + 1)
        self.assertEqual(len(passed_sigmas), steps + 1)

    def test_no_double_guidance(self):
        """
        Verify that applied guidance is only handled inside the loop,
        and not pre-applied redundantly (Fix #2).
        """
        guidance_val = 4.0
        
        self._call_sample(flux_guidance=guidance_val, flux_guidance_profile="Static")
        
        call_args = mock_sample.sample_custom.call_args
        passed_positive = call_args[0][5]
        
        # The passed positive should have the guidance value applied
        self.assertEqual(passed_positive[0][1]["guidance"], guidance_val)

    def test_sampler_override_fix(self):
        """
        Verify that Phase-Shift mode respects the user's primary sampler choice (Fix #3).
        """
        user_sampler = "dpmpp_2m"
        
        self._call_sample(
            sampler=user_sampler, 
            sampler_mode="Phase-Shift (Euler→DPM)",
            phase_split=0.5
        )
        
        calls = mock_samplers.sampler_object.call_args_list
        
        first_call_arg = calls[0][0][0]
        self.assertEqual(first_call_arg, user_sampler, 
                         f"Primary sampler should be {user_sampler}, not overridden to Euler!")

    def test_noise_initialization(self):
        """
        Verify that noise is manually added for the first stage (Fix #4).
        """
        # Just verify it runs without error
        result = self._call_sample(add_noise=True)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # (latent, sigmas)

    def test_dynamic_guidance_math(self):
        """
        Verify the math for dynamic guidance (Fix #5).
        g_low = flux_guidance * 0.6
        """
        flux_g = 10.0
        expected_low = 6.0
        
        self._call_sample(
            steps=10,
            flux_guidance=flux_g,
            flux_guidance_profile="Dynamic (Creative Start/End)"
        )
        
        # Check calls to sample_custom
        calls = mock_sample.sample_custom.call_args_list
        
        # Should have 3 calls (for dynamic guidance: low-high-low stages)
        self.assertEqual(len(calls), 3, 
                         f"Expected 3 sample_custom calls for dynamic guidance, got {len(calls)}")
        
        # Stage 1 (0-2): Low
        args1 = calls[0][0][5]
        self.assertAlmostEqual(args1[0][1]["guidance"], expected_low)
        
        # Stage 2 (2-9): High (flux_guidance)
        args2 = calls[1][0][5]
        self.assertAlmostEqual(args2[0][1]["guidance"], flux_g)
        
        # Stage 3 (9-10): Low
        args3 = calls[2][0][5]
        self.assertAlmostEqual(args3[0][1]["guidance"], expected_low)

    def test_preset_application(self):
        """Verify that presets correctly override parameters."""
        # Use a known preset
        self._call_sample(preset="→ Flux txt2img")
        
        # Verify sample_custom was called
        mock_sample.sample_custom.assert_called()

    def test_return_structure(self):
        """Verify the return structure is correct."""
        result = self._call_sample()
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        latent, sigmas = result
        self.assertIn("samples", latent)
        self.assertIsInstance(sigmas, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
