"""
radiance/tests/test_sampler_audit.py
New test suite verifying specific bug fixes from the v3.2 Quality Audit.
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

mock_comfy.samplers = mock_samplers
mock_comfy.sample = mock_sample
mock_comfy.model_management = mock_model_management
mock_comfy.utils = mock_utils

mock_samplers.KSampler = MagicMock()
mock_samplers.KSampler.SAMPLERS = ["euler", "dpmpp_2m", "dpmpp_2m_sde"]
mock_samplers.KSampler.SCHEDULERS = ["simple", "karras", "exponential", "sgm_uniform"]

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import AFTER mocking
import nodes_sampler

class TestRadianceSamplerAudit(unittest.TestCase):
    
    def setUp(self):
        self.sampler_node = nodes_sampler.RadianceSamplerPro()
        
        # Valid sigmas
        self.valid_sigmas = torch.linspace(1.0, 0.0, 21)
        mock_samplers.calculate_sigmas = MagicMock(return_value=self.valid_sigmas)
        
        mock_model_management.get_torch_device = MagicMock(return_value=torch.device("cpu"))
        mock_sample.prepare_noise = MagicMock(return_value=torch.randn((1, 4, 64, 64)))
        mock_sample.sample_custom = MagicMock(return_value=torch.zeros((1, 4, 64, 64)))
        mock_samplers.sampler_object = MagicMock(return_value=MagicMock())
        
        # Mock Model with proper structure for PAG/Auto-Config
        self.mock_model = MagicMock()
        self.mock_model.get_model_object = MagicMock(return_value=MagicMock())
        # For PAG:
        self.mock_model.clone = MagicMock(return_value=self.mock_model) # returns self for chaining
        self.mock_model.set_model_attn1_patch = MagicMock()
        
        # for detect_model_type
        model_sampling = MagicMock()
        model_sampling.sigma_max = torch.tensor(1.0)
        # Ensure shift/flux_shift don't trigger Flux detection unless intended
        del model_sampling.shift 
        del model_sampling.flux_shift
        self.mock_model.get_model_object = MagicMock(return_value=model_sampling)
        
        self.mock_positive = [[torch.zeros(1), {"guidance": 3.5}]]
        self.mock_negative = [[torch.zeros(1), {}]]
        self.mock_latent = {"samples": torch.zeros((1, 4, 64, 64)), "batch_index": None}

    def _call_sample(self, **kwargs):
        defaults = {
            "model": self.mock_model,
            "positive": self.mock_positive,
            "negative": self.mock_negative,
            "latent_image": self.mock_latent,
            "seed": 123,
            "preset": "None (Custom)",
            "steps": 20,
            "start_step": 0,
            "end_step": 0,
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
            "pag_scale": 0.0,
            "model_type": "auto"
        }
        defaults.update(kwargs)
        return self.sampler_node.sample(**defaults)

    def test_bug1_pag_layer_targeting(self):
        """BUG-1: Ensure PAG patch only targets middle blocks."""
        # Enable PAG
        self._call_sample(pag_scale=1.5)
        
        # Verify set_model_attn1_patch was called
        # The mock model returns itself on clone(), so we check self.mock_model
        self.mock_model.set_model_attn1_patch.assert_called()
        
        # Get the patch function passed
        patch_fn = self.mock_model.set_model_attn1_patch.call_args[0][0]
        
        # Test the patch function with different blocks
        q = torch.randn(1, 8, 32) # batch, seq, dim
        k = torch.randn(1, 8, 32)
        v = torch.randn(1, 8, 32)
        
        # Case 1: Input block (Should be ignored -> normal attention)
        # We can't easily check "normal attention" vs "perturbed" without inspecting output values,
        # but we can check if it tries to access cond_or_uncond logic only for middle.
        # Actually, best way is to see if it injects identity.
        
        # We need a stable way to detect identity injection.
        # If we pass all zeros for Q, K, V, normal attention (softmax(0)*0) = 0.
        # If identity is injected (on uncond), output will be non-zero (identity*0 still 0 though).
        # Let's use Q=Identity, K=Identity, V=Ones.
        # With normal attention: Q*K^T = Identity. Softmax -> peak at diagonal. * V(Ones) -> Ones.
        # With Perturbed/Identity attention: We REPLACE attn scores with Identity*100.
        # So "Normal" Attention is actually functionally similar to Identity if Q,K are Identity.
        
        # Let's rely on the implementation detail: Does it check block_type?
        extra_options_input = {"block_type": "input", "cond_or_uncond": [1]}
        
        # We can spy on torch.eye to see if it's called.
        with patch('torch.eye', side_effect=torch.eye) as mock_eye:
             patch_fn(q, k, v, extra_options_input)
             mock_eye.assert_not_called()
             
             # Case 2: Middle block (Should be perturbed)
             extra_options_middle = {"block_type": "middle", "cond_or_uncond": [1]}
             patch_fn(q, k, v, extra_options_middle)
             mock_eye.assert_called()

    def test_bug2_auto_config_apply_scheduler(self):
        """BUG-2: Ensure Auto-Config applies optimal scheduler for SDXL."""
        # Mock detection as SDXL
        # SDXL defaults: cfg=7.0, scheduler='karras', sampler='euler_ancestral'
        with patch('nodes_sampler.detect_model_type', return_value="sdxl"):
            # Pass wrong defaults (simple/euler)
            self._call_sample(
                model_type="auto",
                scheduler="simple",
                sampler="euler",
                cfg=1.0,
                flux_guidance=3.5
            )
            
            # Verify sample_custom called with SDXL defaults
            args = mock_sample.sample_custom.call_args
            # args: (model, noise, cfg, sampler_obj, sigmas, ...)
            
            # Check CFG
            passed_cfg = args[0][2]
            self.assertEqual(passed_cfg, 7.0)
            
            # Check Sampler/Scheduler indirectly via sigmas calculation (scheduler used there)
            # nodes_sampler uses get_flux_sigmas logic which calls calculate_sigmas(model, SCHEDULER, steps)
            # So we check mock_samplers.calculate_sigmas call args
            calc_args = mock_samplers.calculate_sigmas.call_args
            passed_scheduler = calc_args[0][1]
            self.assertEqual(passed_scheduler, "karras", "Auto-config should set 'karras' for SDXL")

    def test_bug5_flux_presets(self):
        """BUG-5: Flux txt2img preset should use 25 steps."""
        self._call_sample(preset="→ Flux txt2img")
        
        # Verify steps passed to calculate_sigmas
        call_args = mock_samplers.calculate_sigmas.call_args
        steps_arg = call_args[0][2]
        self.assertEqual(steps_arg, 25)

    def test_bug3_dynamic_guidance_interpolation(self):
        """BUG-3: Verify dynamic guidance interpolation."""
        # Setup: Flux model, 10 steps.
        # Early threshold 20% (step 2), Late 90% (step 9).
        # Ramp 5% -> 0.5 steps range? No, 5% of 1.0 progress.
        # At step 2 (20%), we are exactly at threshold.
        # Ramp is +/- 0.05. Window: 0.15 to 0.25.
        # Step 2 is 0.20. t = (0.20 - 0.15)/0.10 = 0.5.
        # Cosine blend at t=0.5 is 0.5 * (1 - cos(pi*0.5)) = 0.5 * (1 - 0) = 0.5.
        # Guidance should be mid-point between low(2.1) and high(3.5) = 2.8.
        
        flux_g = 3.5
        g_low = 3.5 * 0.6 # 2.1
        
        # We assume flux detection works (mocked)
        with patch('nodes_sampler.detect_model_type', return_value="flux"):
            # Set end_step=3 so we run steps 0, 1, 2. But we only care about step 2 (start of stage 3? no).
            # If we run steps=10. step 0 (0.0), step 1 (0.1), step 2 (0.2).
            # We want to check guidance at step 2.
            # We can just run the whole sampling, but we need to spy on 'apply_guidance'.
            
            # Since apply_guidance is internal, verify via mocked sample_custom args.
            # sample_custom is called PER STAGE.
            # Dynamic Guidance splits the schedule at 20% and 90%.
            # So for 10 steps: splits at 2 (20%) and 9 (90%).
            # Stages: 0-2 (Low), 2-9 (High), 9-10 (Low).
            # Wait, my logic for bug 3 was adding RAMPS.
            # The stage splits are HARD splits. The interpolation happens inside the loop?
            # NO, the loop calculates guidance based on `s_start`.
            # `effective_guidance = ... (logic based on progress)`
            # So for the stage starting at 2 (progress 0.2), we compute interpolated guidance.
            # Yes.
            
            # So we expect 3 stages.
            # Stage 2 (starts at step 2): progress 0.2. Induces Blend.
            
            self._call_sample(
                steps=10,
                flux_guidance=3.5,
                flux_guidance_profile="Dynamic (Creative Start/End)"
            )
            
            # Verify sample_custom was called 3 times
            self.assertEqual(mock_sample.sample_custom.call_count, 3)
            
            # Check guidance for Stage 2 (call index 1)
            # args[5] is positive conditioning
            stage2_args = mock_sample.sample_custom.call_args_list[1][0]
            guidance_used = stage2_args[5][0][1]["guidance"]
            
            # Expected: 2.8 (midpoint)
            self.assertAlmostEqual(guidance_used, 2.8, delta=0.1)

    def test_bug4_phase_shift_offset(self):
        """BUG-4: Verify correct sigma offset when denoise < 1.0."""
        # Setup: denoise=0.5, steps=20.
        # calculate_sigmas returns 21 steps.
        # nodes_sampler trims it to 10 steps (plus terminal).
        # base_sigmas has 11 items.
        
        # Mock calculate_sigmas to return 21 items (0..20)
        # Note: calculate_sigmas result is reversed (high to low).
        # But here we just tracking steps count.
        sigmas_21 = torch.linspace(1.0, 0.0, 21) # 21 items
        mock_samplers.calculate_sigmas.return_value = sigmas_21
        
        # Call sample with denoise=0.5
        # The code will:
        # 1. Calc sigmas (21 items).
        # 2. Trim to last 11 items (indices 10..20?).
        #    n = 20. start = 20 * (1-0.5) = 10.
        #    bs = sigmas[10:]. Length 11.
        # 3. Splits: {0, 20} (default).
        # 4. EFFECTIVE START: 0.
        #    BUT wait, denoise < 1.0 in standard KSampler means we usually start at step 10?
        #    Radiance sampler has separate `start_step` and `denoise`.
        #    If `denoise=0.5`, typical workflow sets start_step=0 (of global? or relative?).
        #    Actually, standard Comfy KSampler uses `denoise` to set start_step.
        
        #    Radiance Sampler: `denoise` parameter trims the sigmas.
        #    But `start_step` parameter (from widget) is separate.
        #    If user sets `denoise=0.5`, they expect image to be 50% denoised.
        #    If they set `start_step=0`, `end_step=0` (defaults),
        #    `effective_start` = 0.
        #    `splits` = {0, 20}.
        
        #    Inside loop:
        #    Stage 1: 0-20.
        #    `bs_start_step` = 20 - (11-1) = 10.
        #    `local_start` = 0 - 10 = -10.
        #    `local_end` = 20 - 10 = 10.
        
        #    `local_end` >= 0. Safe to proceed.
        #    `safe_start` = max(0, -10) = 0.
        #    `safe_end` = max(0, min(11, 11)) = 11.
        #    `stage_sigmas` = base_sigmas[0:11].
        
        #    So it uses the FULL available sigmas (which represent the 50% denoising range).
        #    This is CORRECT behavior for denoise parameter usage.
        
        self._call_sample(steps=20, denoise=0.5, start_step=0, end_step=20)
        
        mock_sample.sample_custom.assert_called()
        args = mock_sample.sample_custom.call_args
        passed_sigmas = args[0][4]
        
        # Should have ~11 items
        self.assertEqual(len(passed_sigmas), 11)

if __name__ == '__main__':
    unittest.main()
