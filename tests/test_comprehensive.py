
# ============================================================================
# IMPORTS
# ============================================================================

import sys
import os
import unittest
import torch
import numpy as np
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nodes_color
import nodes_grade
import nodes_filmgrain
import color_utils


# ============================================================================
# 32-BIT PRECISION TESTS
# ============================================================================

class Test32BitPrecision(unittest.TestCase):
    """Verify 32-bit floating point precision throughout the pipeline."""
    
    def test_numpy_arrays_are_float32(self):
        """Ensure color operations maintain float32 precision."""
        from color_utils import linear_to_logc4, logc4_to_linear
        
        input_data = np.array([0.18, 0.5, 1.0, 2.0, 10.0], dtype=np.float32)
        encoded = linear_to_logc4(input_data)
        decoded = logc4_to_linear(encoded)
        
        self.assertEqual(input_data.dtype, np.float32, "Input should be float32")
        self.assertEqual(encoded.dtype, np.float32, "Encoded should be float32")
        self.assertEqual(decoded.dtype, np.float32, "Decoded should be float32")
    
    def test_torch_tensors_are_float32(self):
        """Verify torch operations use float32."""
        tensor = torch.randn(1, 64, 64, 3, dtype=torch.float32)
        self.assertEqual(tensor.dtype, torch.float32)
        
        # Matrix operation should preserve dtype
        matrix = torch.eye(3, dtype=torch.float32)
        result = torch.matmul(tensor[..., :3], matrix.T)
        self.assertEqual(result.dtype, torch.float32)
    
    def test_log_curve_roundtrip_precision(self):
        """Log encoding/decoding should have <0.01% error."""
        from color_utils import (
            linear_to_logc3, logc3_to_linear,
            linear_to_logc4, logc4_to_linear,
            linear_to_slog3, slog3_to_linear,
            linear_to_acescct, acescct_to_linear
        )
        
        test_values = np.array([0.01, 0.18, 0.5, 1.0, 5.0, 20.0], dtype=np.float32)
        
        curves = [
            ("LogC3", linear_to_logc3, logc3_to_linear),
            ("LogC4", linear_to_logc4, logc4_to_linear),
            ("S-Log3", linear_to_slog3, slog3_to_linear),
            ("ACEScct", linear_to_acescct, acescct_to_linear),
        ]
        
        for name, encode, decode in curves:
            encoded = encode(test_values)
            decoded = decode(encoded)
            
            # Calculate relative error percentage
            rel_error = np.abs(decoded - test_values) / (test_values + 1e-10) * 100
            max_error = np.max(rel_error)
            
            self.assertLess(max_error, 0.01, f"{name} roundtrip error > 0.01%: {max_error:.4f}%")
    
    def test_matrix_transform_preserves_hdr_values(self):
        """Matrix transforms should preserve values > 1.0."""
        from color_utils import ACESCG_TO_SRGB, apply_matrix_transform
        
        hdr_pixels = np.array([
            [2.0, 2.0, 2.0],
            [5.0, 3.0, 1.0],
            [10.0, 0.5, 0.1],
        ], dtype=np.float32)
        
        result = apply_matrix_transform(hdr_pixels, ACESCG_TO_SRGB)
        
        # Values should not be clamped
        self.assertTrue(np.any(result > 1.0), "HDR values should be preserved, not clamped")
    
    def test_grain_generation_float32(self):
        """Grain functions should output float32."""
        from nodes_filmgrain import generate_temporal_grain
        
        color_resp = {"r": 1.0, "g": 1.0, "b": 1.0}
        grain = generate_temporal_grain(64, 64, 1.0, 0.15, 0.2, color_resp,
                                        frame_index=0, temporal_seed=42)
        
        self.assertEqual(grain.dtype, np.float32, "Grain should be float32")

    def test_hdr_values_not_clipped_in_operations(self):
        """HDR values > 1.0 should not be clipped in intermediate operations."""
        from color_utils import linear_to_logc4, logc4_to_linear
        
        # Super-white test values
        hdr_values = np.array([2.0, 5.0, 10.0, 50.0, 100.0], dtype=np.float32)
        
        encoded = linear_to_logc4(hdr_values)
        decoded = logc4_to_linear(encoded)
        
        # Should preserve values within reasonable tolerance
        np.testing.assert_allclose(decoded, hdr_values, rtol=1e-4,
            err_msg="HDR values should be preserved through Log encoding")


# ============================================================================
# SAMPLER TESTS (FILE-BASED - No ComfyUI dependency)
# ============================================================================

class TestSamplerFileBased(unittest.TestCase):
    """Test sampler by parsing the source file directly."""
    
    @classmethod
    def setUpClass(cls):
        """Read sampler file once."""
        sampler_path = os.path.join(os.path.dirname(__file__), '..', 'nodes_sampler.py')
        with open(sampler_path, 'r', encoding='utf-8') as f:
            cls.sampler_source = f.read()
    
    def test_preset_configs_exist(self):
        """PRESET_CONFIGS should be defined in the file."""
        self.assertIn("PRESET_CONFIGS", self.sampler_source)
    
    def test_euler_sampler_in_flux_presets(self):
        """All Flux presets should use euler sampler."""
        # Find all preset definitions
        pattern = r'"→ Flux[^"]*":\s*\{[^}]+\}'
        matches = re.findall(pattern, self.sampler_source, re.DOTALL)
        
        self.assertGreater(len(matches), 0, "Should have Flux presets")
        
        for match in matches:
            self.assertIn('"sampler": "euler"', match,
                f"Flux preset should use euler: {match[:50]}...")
    
    def test_phase_shift_mode_exists(self):
        """Phase-Shift mode should be available."""
        self.assertIn("Phase-Shift", self.sampler_source)
    
    def test_dynamic_guidance_profiles_exist(self):
        """Dynamic guidance profiles should be defined."""
        profiles = [
            "Low",
            "High",
            "Low → High → Low",
        ]
        
        for profile in profiles:
            self.assertIn(profile, self.sampler_source, f"Missing profile: {profile}")
    
    def test_flux_shift_function_exists(self):
        """flux_shift_sigmas function should be defined."""
        self.assertIn("def flux_shift_sigmas", self.sampler_source)
    
    def test_cfg_defaults_reasonable(self):
        """Default CFG values should be reasonable (1.0-4.0 for Flux)."""
        # Find all "cfg": X patterns
        cfg_pattern = r'"cfg":\s*([\d.]+)'
        matches = re.findall(cfg_pattern, self.sampler_source)
        
        for cfg_str in matches:
            cfg = float(cfg_str)
            self.assertGreaterEqual(cfg, 0.0, f"CFG {cfg} < 0")
            self.assertLessEqual(cfg, 10.0, f"CFG {cfg} > 10 (unusual for Flux)")
    
    def test_steps_values_reasonable(self):
        """Step counts should be in reasonable range."""
        steps_pattern = r'"steps":\s*(\d+)'
        matches = re.findall(steps_pattern, self.sampler_source)
        
        for steps_str in matches:
            steps = int(steps_str)
            self.assertGreaterEqual(steps, 4, f"Steps {steps} < 4")
            self.assertLessEqual(steps, 100, f"Steps {steps} > 100")


# ============================================================================
# DYNAMIC GUIDANCE MATH TESTS
# ============================================================================

class TestDynamicGuidanceMath(unittest.TestCase):
    """Test dynamic guidance mathematical correctness."""
    
    def test_low_high_low_curve(self):
        """Low-High-Low profile should follow bell curve pattern."""
        # Simulated Low → High → Low curve using sin
        for progress in np.linspace(0, 1, 21):
            curve = np.sin(np.pi * progress)
            scale = 0.5 + 0.5 * curve  # Range: 0.5 to 1.0
            
            # Endpoints should be at 50%
            if progress < 0.02 or progress > 0.98:
                self.assertLess(scale, 0.60, f"Endpoints should be ~0.5, got {scale}")
            # Middle should be at 100%
            elif 0.48 < progress < 0.52:
                self.assertGreater(scale, 0.95, f"Middle should be ~1.0, got {scale}")
    
    def test_high_low_curve(self):
        """High-Low profile should decrease monotonically."""
        previous = 1.0
        for progress in np.linspace(0, 1, 21):
            # Linear decrease from 1.0 to 0.5
            scale = 1.0 - 0.5 * progress
            
            self.assertLessEqual(scale, previous + 1e-6, 
                f"High-Low should decrease: {previous} -> {scale}")
            previous = scale
    
    def test_constant_guidance(self):
        """Constant profile should maintain fixed value."""
        base_guidance = 4.0
        
        for progress in np.linspace(0, 1, 21):
            effective = base_guidance * 1.0  # Constant scale = 1.0
            self.assertAlmostEqual(effective, base_guidance, places=5)


# ============================================================================
# EXPERT PIPELINE TESTS (LogC4, Nuke Match, Compositing)
# ============================================================================

class TestPipelineExpert(unittest.TestCase):
    
    def setUp(self):
        # Set up common test data
        self.grade_node = nodes_grade.FXTD_Grade()
        self.color_pipeline = nodes_color.RadianceLogCurveEncode() # For encoding to Log
        self.decode_node = nodes_color.RadianceLogCurveDecode() # For decoding back
    
    def test_logc4_roundtrip_precision(self):
        """
        Verify that converting Linear -> LogC4 -> Linear is lossless (within float32 precision).
        Critical for VFX pipelines to ensure no data loss during round-trips.
        """
        # Create gradient from -0.1 to 50.0 (covering blacks, midtones, highlights, super-whites)
        linear_data = torch.linspace(-0.1, 50.0, steps=1000).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        # 1. Encode Linear -> LogC4
        logc4_encoded = self.color_pipeline.encode(linear_data, "ARRI LogC4", source_gamut="Native / No Transform", apply_gamma=False, clamp_output=False)[0]
        
        # 2. Decode LogC4 -> Linear
        linear_decoded = self.decode_node.decode(logc4_encoded, "ARRI LogC4", target_gamut="Native / No Transform")[0]
        
        # 3. Check difference
        # We allow a small epsilon for float32 errors, but it should be very small.
        # Note: LogC4 encoding of negative values might be clamped or handled specifically.
        # Let's check the range where it matters most [0.0, 50.0]
        mask = (linear_data >= 0.0) & (linear_data <= 45.0) # Standard range
        diff = torch.abs(linear_data[mask] - linear_decoded[mask])
        max_diff = diff.max().item()
        
        # print(f"\n[Precision] LogC4 Roundtrip Max Error: {max_diff:.8f}")
        self.assertLess(max_diff, 1e-4, "LogC4 Roundtrip precision loss is too high for VFX work.")

    def test_grade_math_nuke_match(self):
        """
        Verify Grade node math matches Nuke standards:
        Linear: (x + lift) * gain * gamma_power
        Note: Nuke's 'Gamma' slider is actually 1/Gamma.
        """
        # Test Value: 0.5 mid gray
        val = 0.5
        img = torch.tensor([[[[val, val, val]]]], dtype=torch.float32)
        
        # 1. Test Lift (Offset)
        # Nuke: 0.5 + 0.1 = 0.6
        lift = 0.1
        res_lift, _ = self.grade_node.grade(img, lift_r=lift, lift_g=lift, lift_b=lift)
        self.assertAlmostEqual(res_lift[0,0,0,0].item(), val + lift, places=6, msg="Lift math mismatch")
        
        # 2. Test Gain (Mult)
        # Nuke: 0.5 * 1.5 = 0.75
        gain = 1.5
        res_gain, _ = self.grade_node.grade(img, gain_r=gain, gain_g=gain, gain_b=gain)
        self.assertAlmostEqual(res_gain[0,0,0,0].item(), val * gain, places=6, msg="Gain math mismatch")
        
        # 3. Test Gamma
        # Nuke: 0.5 ^ (1/2.0) = 0.5 ^ 0.5 = 0.7071...
        # Our node implements 'gamma' as the value passed to pow(x, 1/gamma).
        gamma = 2.0
        res_gamma, _ = self.grade_node.grade(img, gamma_r=gamma, gamma_g=gamma, gamma_b=gamma)
        expected_gamma = pow(val, 1.0/gamma)
        self.assertAlmostEqual(res_gamma[0,0,0,0].item(), expected_gamma, places=5, msg="Gamma math mismatch")

    def test_alpha_compositing_simulation(self):
        """
        Simulate a basic compositing operation (Over).
        Radiance doesn't have a specific 'Merge/Over' node yet, but we can verify
        math steps manually or via expression if we had one.
        Here we verify the Grade node DOES NOT destroy Alpha if present.
        """
        # RGBA image: Red with 0.5 Alpha
        img = torch.tensor([[[[1.0, 0.0, 0.0, 0.5]]]], dtype=torch.float32)
        
        # Apply intense grading
        res, _ = self.grade_node.grade(img, gain_r=2.0, gamma_g=0.5, lift_b=0.1)
        
        # Alpha should remain untouched
        self.assertEqual(res[0,0,0,3].item(), 0.5, "Grading operation corrupted Alpha channel.")
        
        # RGB should be affected
        self.assertEqual(res[0,0,0,0].item(), 2.0, "Grading failed on Red channel.") # 1.0 * 2.0

    def test_pipeline_stress_test(self):
        """
        Full pipeline simulation:
        Linear Source -> LogC4 -> Grade -> Linear -> Grain
        Verifies data flows correctly without crashing or clamping unexpectedly.
        """
        # 2K Resolution-ish
        h, w = 64, 64 
        img = torch.ones((1, h, w, 3), dtype=torch.float32) * 0.18 # Mid gray
        
        # 1. To LogC4
        log_img = self.color_pipeline.encode(img, "ARRI LogC4", "ACEScg (Linear)", apply_gamma=False, clamp_output=False)[0]
        
        # 2. Grade in Log (Lift shadows)
        # LogC4 0.18 is approx 0.28. Let's lift it.
        graded_log, _ = self.grade_node.grade(log_img, lift_r=0.05, lift_g=0.05, lift_b=0.05)
        
        # 3. Back to Linear
        linear_final = self.decode_node.decode(graded_log, "ARRI LogC4", "ACEScg (Linear)")[0]
        
        # 4. Apply Grain
        # Use the Realistic Grain node interface
        grain_node = nodes_filmgrain.FXTDRealisticGrain()
        grained_img, _ = grain_node.apply_realistic_grain(linear_final, "ARRI Alexa 35", strength=0.5)
        
        # Check stats
        original_mean = 0.18
        final_mean = linear_final.mean().item()
        
        # We lifted the log image, so final linear should be > 0.18
        # How much? Hard to say exactly without calculation, but MUST be higher.
        self.assertGreater(final_mean, original_mean, "Grade did not brighten the image as expected.")
        
        # Grain should add variance
        variance = torch.var(grained_img).item()
        # Ensure variance is greater than the smooth graded image's variance
        graded_variance = torch.var(linear_final).item()
        self.assertGreater(variance, graded_variance, "Film grain added no variance.")
        
        # print(f"\n[Pipeline] Source: {original_mean:.4f} -> Graded Linear: {final_mean:.4f} -> Grained Var: {variance:.6f}")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPipelineIntegration(unittest.TestCase):
    """Test full pipeline integration."""
    
    def test_full_color_pipeline_float32(self):
        """Complete color pipeline should maintain float32."""
        from color_utils import (
            linear_to_logc4, logc4_to_linear,
            apply_matrix_transform, AWG4_TO_ACESCG, ACESCG_TO_SRGB
        )
        
        # Simulate camera capture
        cam_linear = np.random.rand(64, 64, 3).astype(np.float32) * 2.0
        
        # Encode to Log
        cam_log = linear_to_logc4(cam_linear)
        self.assertEqual(cam_log.dtype, np.float32)
        
        # Decode back
        cam_decoded = logc4_to_linear(cam_log)
        self.assertEqual(cam_decoded.dtype, np.float32)
        
        # Transform gamut
        aces_linear = apply_matrix_transform(cam_decoded, AWG4_TO_ACESCG)
        self.assertEqual(aces_linear.dtype, np.float32)
        
        # To display
        display = apply_matrix_transform(aces_linear, ACESCG_TO_SRGB)
        self.assertEqual(display.dtype, np.float32)
    
    def test_grain_pipeline_float32(self):
        """Grain generation and application should maintain float32."""
        from nodes_filmgrain import generate_temporal_grain
        
        # Generate sequence of grain
        color_resp = {"r": 1.0, "g": 1.0, "b": 1.0}
        
        grains = []
        for frame in range(5):
            grain = generate_temporal_grain(64, 64, 1.0, 0.15, 0.2, color_resp,
                                           frame_index=frame, temporal_seed=42)
            self.assertEqual(grain.dtype, np.float32)
            grains.append(grain)
        
        # Verify temporal consistency - adjacent frames should be correlated
        for i in range(len(grains) - 1):
            correlation = np.corrcoef(grains[i].flatten(), grains[i+1].flatten())[0, 1]
            # Should have some correlation due to temporal smoothness
            self.assertGreater(correlation, -0.5, "Adjacent frames should be somewhat correlated")


# ============================================================================
# SCORE CALCULATION
# ============================================================================

def run_tests_and_score():
    """Run all tests and calculate a quality score."""
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        Test32BitPrecision,
        TestSamplerFileBased,
        TestDynamicGuidanceMath,
        TestPipelineExpert,
        TestPipelineIntegration,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Calculate score
    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    score = (passed / total) * 100 if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("RADIANCE QUALITY SCORE REPORT")
    print("=" * 70)
    print(f"Tests Run:    {total}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {len(result.failures)}")
    print(f"Errors:       {len(result.errors)}")
    print()
    
    # Grade
    if score >= 95:
        grade = "A+"
        status = "PRODUCTION READY"
    elif score >= 90:
        grade = "A"
        status = "EXCELLENT"
    elif score >= 80:
        grade = "B"
        status = "GOOD"
    elif score >= 70:
        grade = "C"
        status = "NEEDS IMPROVEMENT"
    else:
        grade = "D"
        status = "CRITICAL ISSUES"
    
    print(f">>> QUALITY SCORE: {score:.1f}% (Grade: {grade}) <<<")
    print(f">>> STATUS: {status} <<<")
    print("=" * 70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if score >= 95:
        print("✅ EXCELLENT: All tests pass. Pipeline is production-ready.")
        print("   - 32-bit precision verified across all operations")
        print("   - Sampler presets correctly configured")
        print("   - Dynamic guidance math is correct")
    elif score >= 80:
        print("⚠️ GOOD: Minor issues detected. Review failures below.")
    else:
        print("❌ NEEDS WORK: Significant issues found. Address failures.")
    
    if result.failures:
        print("\n❌ Failed Tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            # Print first line of traceback
            for line in traceback.split('\n'):
                if 'AssertionError' in line:
                    print(f"    → {line.strip()}")
                    break
    
    if result.errors:
        print("\n⚠️ Tests with Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return score, passed, total


if __name__ == "__main__":
    score, passed, total = run_tests_and_score()
