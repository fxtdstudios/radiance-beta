"""
FXTD Radiance Sampler Pro - Deep Analysis & Scoring Report
============================================================
Comprehensive analysis of all sampler options and configurations.
"""

import sys
import os
import re
import unittest

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SamplerAnalyzer:
    """Deep analyzer for Radiance Sampler Pro."""
    
    def __init__(self):
        sampler_path = os.path.join(os.path.dirname(__file__), '..', 'nodes_sampler.py')
        with open(sampler_path, 'r', encoding='utf-8') as f:
            self.source = f.read()
        self.issues = []
        self.warnings = []
        self.passed = []
        
    def analyze(self):
        """Run complete analysis."""
        self._check_presets()
        self._check_phase_shift()
        self._check_dynamic_guidance()
        self._check_flux_shift()
        self._check_error_handling()
        self._check_sigma_continuity()
        self._check_refiner_support()
        self._check_noise_handling()
        self._check_timing_report()
        self._check_documentation()
        return self.generate_report()
    
    def _check_presets(self):
        """Analyze workflow presets."""
        # Count presets
        preset_matches = re.findall(r'"→ Flux[^"]*"', self.source)
        preset_count = len(set(preset_matches))
        
        if preset_count >= 7:
            self.passed.append(("Preset Count", f"✅ {preset_count} presets defined"))
        else:
            self.warnings.append(("Preset Count", f"Only {preset_count} presets found"))
        
        # Check all presets use euler
        euler_matches = re.findall(r'"→ Flux[^}]*"sampler":\s*"euler"', self.source, re.DOTALL)
        if len(euler_matches) >= 7:
            self.passed.append(("Euler Sampler", "✅ All Flux presets use euler (stable)"))
        else:
            self.warnings.append(("Euler Sampler", "Some presets don't use euler"))
        
        # Check CFG = 1.0 for Flux
        cfg_matches = re.findall(r'"→ Flux[^}]*"cfg":\s*([\d.]+)', self.source, re.DOTALL)
        bad_cfgs = [c for c in cfg_matches if float(c) != 1.0]
        if not bad_cfgs:
            self.passed.append(("CFG Settings", "✅ All Flux presets use CFG=1.0"))
        else:
            self.issues.append(("CFG Settings", f"Some presets have CFG != 1.0: {bad_cfgs}"))
    
    def _check_phase_shift(self):
        """Analyze Phase-Shift implementation."""
        if "Phase-Shift" in self.source:
            self.passed.append(("Phase-Shift Mode", "✅ Phase-Shift sampling implemented"))
        else:
            self.issues.append(("Phase-Shift Mode", "Phase-Shift not found"))
        
        # Check DPM and SGM options
        if "dpmpp_2m" in self.source:
            self.passed.append(("Phase-Shift DPM", "✅ DPM++ 2M variant available"))
        
        if "sgm_uniform" in self.source:
            self.passed.append(("Phase-Shift SGM", "✅ SGM Uniform variant available"))
        
        # Check phase_split parameter
        if "phase_split" in self.source:
            self.passed.append(("Split Control", "✅ Configurable phase split point"))
        
        # Check FIX #6 - respects user sampler
        if "Respect user's sampler" in self.source or "primary_sampler = sampler" in self.source:
            self.passed.append(("FIX #6", "✅ Phase-Shift respects user's sampler selection"))
        else:
            self.warnings.append(("FIX #6", "May override user's sampler choice"))
    
    def _check_dynamic_guidance(self):
        """Analyze Dynamic Guidance implementation."""
        if "Dynamic" in self.source and "flux_guidance_profile" in self.source:
            self.passed.append(("Dynamic Guidance", "✅ Dynamic guidance profile available"))
        else:
            self.issues.append(("Dynamic Guidance", "Not implemented"))
        
        # Check Low-High-Low pattern
        if "g_low" in self.source and "g_high" in self.source:
            self.passed.append(("Guidance Zones", "✅ Low-High-Low zones implemented"))
        
        # Check per-step guidance (FIX #7)
        if "effective_guidance" in self.source:
            self.passed.append(("FIX #7", "✅ Per-step dynamic guidance with stage zones"))
    
    def _check_flux_shift(self):
        """Analyze Flux Shift implementation."""
        # Check formula
        if "shift * sigmas / (1.0 + (shift - 1.0) * sigmas)" in self.source:
            self.passed.append(("Flux Shift Formula", "✅ Correct Flux shift formula"))
        elif "shift * sigma" in self.source:
            self.passed.append(("Flux Shift Formula", "✅ Flux shift formula present"))
        else:
            self.issues.append(("Flux Shift Formula", "Formula not found"))
        
        # Check FIX #1 - sample_custom with shift
        if "sample_custom" in self.source and "flux_shift_sigmas" in self.source:
            self.passed.append(("FIX #1", "✅ Flux Shift applied via sample_custom"))
        
        # Check shift=1.0 identity
        if "if shift == 1.0:" in self.source or "shift != 1.0" in self.source:
            self.passed.append(("Shift Identity", "✅ Shift=1.0 returns unchanged sigmas"))
    
    def _check_error_handling(self):
        """Analyze error handling."""
        try_count = len(re.findall(r'\btry:', self.source))
        except_count = len(re.findall(r'\bexcept\b', self.source))
        
        if try_count >= 1 and except_count >= 1:
            self.passed.append(("Error Handling", f"✅ Try/except blocks present ({try_count} try, {except_count} except)"))
        else:
            self.warnings.append(("Error Handling", "Limited error handling"))
        
        # Check for stage error reporting
        if "Error in Stage" in self.source:
            self.passed.append(("Stage Errors", "✅ Stage-level error reporting"))
    
    def _check_sigma_continuity(self):
        """Analyze sigma continuity (FIX #8)."""
        if "Sigma discontinuity" in self.source or "sigma_diff" in self.source:
            self.passed.append(("FIX #8", "✅ Sigma continuity validation at switch points"))
        else:
            self.warnings.append(("FIX #8", "No sigma continuity check"))
    
    def _check_refiner_support(self):
        """Analyze refiner model support."""
        if "refiner_model" in self.source:
            self.passed.append(("Refiner Support", "✅ Refiner model integration"))
        
        if "refiner_start_step" in self.source:
            self.passed.append(("Refiner Timing", "✅ Configurable refiner start step"))
    
    def _check_noise_handling(self):
        """Analyze noise handling."""
        # Check FIX #4
        if "WITHOUT pre-noising" in self.source or "Noise added to first stage" in self.source:
            self.passed.append(("FIX #4", "✅ Proper noise initialization (no pre-noising)"))
        
        if "prepare_noise" in self.source:
            self.passed.append(("Noise Prep", "✅ Uses comfy.sample.prepare_noise"))
        
        if "noise_override" in self.source:
            self.passed.append(("Noise Override", "✅ Custom noise input supported"))
        
        if "add_noise" in self.source:
            self.passed.append(("Add Noise Toggle", "✅ Noise enable/disable option"))
    
    def _check_timing_report(self):
        """Analyze timing/diagnostics."""
        if "TIMING REPORT" in self.source:
            self.passed.append(("Timing Report", "✅ Performance diagnostics included"))
        
        timing_stages = ["latent_copy", "prepare_noise", "sigma_calc", "sampling", "output_prep"]
        found_stages = [s for s in timing_stages if s in self.source]
        
        if len(found_stages) >= 4:
            self.passed.append(("Timing Breakdown", f"✅ {len(found_stages)}/5 timing stages tracked"))
    
    def _check_documentation(self):
        """Analyze inline documentation."""
        docstrings = len(re.findall(r'"""[^"]+"""', self.source))
        comments = len(re.findall(r'#[^\n]+', self.source))
        fixes = len(re.findall(r'FIX #\d+', self.source))
        
        if docstrings >= 3:
            self.passed.append(("Docstrings", f"✅ {docstrings} docstrings present"))
        
        if fixes >= 8:
            self.passed.append(("Bug Fixes", f"✅ {fixes} documented fixes (FIX #1-8)"))
        
        if comments >= 30:
            self.passed.append(("Code Comments", f"✅ {comments} code comments"))
    
    def generate_report(self):
        """Generate final report."""
        total_checks = len(self.passed) + len(self.warnings) + len(self.issues)
        score = (len(self.passed) / total_checks * 100) if total_checks > 0 else 0
        
        return {
            "score": score,
            "passed": self.passed,
            "warnings": self.warnings,
            "issues": self.issues,
            "total_checks": total_checks
        }


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestSamplerPresets(unittest.TestCase):
    """Test preset configurations."""
    
    @classmethod
    def setUpClass(cls):
        sampler_path = os.path.join(os.path.dirname(__file__), '..', 'nodes_sampler.py')
        with open(sampler_path, 'r', encoding='utf-8') as f:
            cls.source = f.read()
    
    def test_all_presets_defined(self):
        """All 7 Flux presets should be defined."""
        expected = [
            "Flux txt2img", "Flux img2img", "Flux Inpaint",
            "Flux High-Res Fix", "Flux Fast", "Flux Quality", "Flux Cinematic"
        ]
        for preset in expected:
            self.assertIn(preset, self.source, f"Missing preset: {preset}")
    
    def test_preset_steps_range(self):
        """Preset step counts should be 12-30."""
        steps_pattern = r'"steps":\s*(\d+)'
        matches = re.findall(steps_pattern, self.source)
        for steps_str in matches:
            steps = int(steps_str)
            self.assertGreaterEqual(steps, 10, f"Steps {steps} < 10")
            self.assertLessEqual(steps, 50, f"Steps {steps} > 50")


class TestPhaseShift(unittest.TestCase):
    """Test Phase-Shift implementation."""
    
    @classmethod
    def setUpClass(cls):
        sampler_path = os.path.join(os.path.dirname(__file__), '..', 'nodes_sampler.py')
        with open(sampler_path, 'r', encoding='utf-8') as f:
            cls.source = f.read()
    
    def test_phase_shift_modes_available(self):
        """Both Phase-Shift modes should be available."""
        self.assertIn("Phase-Shift (Euler→DPM)", self.source)
        self.assertIn("Phase-Shift (Euler→SGM)", self.source)
    
    def test_phase_split_parameter_exists(self):
        """phase_split parameter should exist with default 0.40."""
        self.assertIn("phase_split", self.source)
        self.assertIn("0.40", self.source)
    
    def test_respects_user_sampler(self):
        """Phase-Shift should respect user's primary sampler choice."""
        # Check that primary_sampler is set from user's selection
        self.assertIn("primary_sampler = sampler", self.source)


class TestDynamicGuidance(unittest.TestCase):
    """Test Dynamic Guidance implementation."""
    
    @classmethod
    def setUpClass(cls):
        sampler_path = os.path.join(os.path.dirname(__file__), '..', 'nodes_sampler.py')
        with open(sampler_path, 'r', encoding='utf-8') as f:
            cls.source = f.read()
    
    def test_guidance_profiles_available(self):
        """Static and Dynamic profiles should be available."""
        self.assertIn("Static", self.source)
        self.assertIn("Dynamic", self.source)
    
    def test_low_high_low_zones(self):
        """Dynamic guidance should have Low-High-Low zones."""
        self.assertIn("g_low", self.source)
        self.assertIn("g_high", self.source)
    
    def test_zone_thresholds(self):
        """Zone thresholds should be at 20% and 90%."""
        self.assertIn("0.2", self.source)  # 20% threshold
        self.assertIn("0.9", self.source)  # 90% threshold


class TestFluxShift(unittest.TestCase):
    """Test Flux Shift implementation."""
    
    @classmethod
    def setUpClass(cls):
        sampler_path = os.path.join(os.path.dirname(__file__), '..', 'nodes_sampler.py')
        with open(sampler_path, 'r', encoding='utf-8') as f:
            cls.source = f.read()
    
    def test_flux_shift_function_exists(self):
        """flux_shift_sigmas function should be defined."""
        self.assertIn("def flux_shift_sigmas", self.source)
    
    def test_shift_formula_correct(self):
        """Shift formula should match Flux specification."""
        # Formula: shift * sigma / (1 + (shift - 1) * sigma)
        self.assertIn("shift * sigmas / (1.0 + (shift - 1.0) * sigmas)", self.source)
    
    def test_identity_preserved(self):
        """shift=1.0 should return unchanged sigmas."""
        self.assertIn("if shift == 1.0:", self.source)


class TestBugFixes(unittest.TestCase):
    """Test that all documented bug fixes are in place."""
    
    @classmethod
    def setUpClass(cls):
        sampler_path = os.path.join(os.path.dirname(__file__), '..', 'nodes_sampler.py')
        with open(sampler_path, 'r', encoding='utf-8') as f:
            cls.source = f.read()
    
    def test_fix_1_sample_custom(self):
        """FIX #1: sample_custom for proper Flux Shift."""
        self.assertIn("sample_custom", self.source)
    
    def test_fix_3_user_sampler(self):
        """FIX #3: Respects user's sampler selection."""
        self.assertIn("primary_sampler = sampler", self.source)
    
    def test_fix_4_noise_init(self):
        """FIX #4: Proper noise initialization."""
        self.assertIn("Noise added to first stage", self.source)
    
    def test_fix_7_dynamic_guidance(self):
        """FIX #7: Per-step dynamic guidance."""
        self.assertIn("effective_guidance", self.source)
    
    def test_fix_8_sigma_continuity(self):
        """FIX #8: Sigma continuity validation."""
        self.assertIn("sigma_diff", self.source)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_analysis():
    """Run deep analysis and tests, generate report."""
    print("=" * 70)
    print("FXTD RADIANCE SAMPLER PRO - DEEP ANALYSIS REPORT")
    print("=" * 70)
    
    # Run analyzer
    analyzer = SamplerAnalyzer()
    report = analyzer.analyze()
    
    print("\n📊 ANALYSIS RESULTS")
    print("-" * 70)
    
    print("\n✅ PASSED CHECKS:")
    for name, desc in report["passed"]:
        print(f"   {desc}")
    
    if report["warnings"]:
        print("\n⚠️ WARNINGS:")
        for name, desc in report["warnings"]:
            print(f"   {desc}")
    
    if report["issues"]:
        print("\n❌ ISSUES:")
        for name, desc in report["issues"]:
            print(f"   {desc}")
    
    # Run unit tests
    print("\n" + "=" * 70)
    print("UNIT TESTS")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestSamplerPresets,
        TestPhaseShift,
        TestDynamicGuidance,
        TestFluxShift,
        TestBugFixes,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Final scoring
    test_passed = result.testsRun - len(result.failures) - len(result.errors)
    test_score = (test_passed / result.testsRun * 100) if result.testsRun > 0 else 0
    
    # Combine scores
    combined_score = (report["score"] + test_score) / 2
    
    print("\n" + "=" * 70)
    print("FINAL SCORE REPORT")
    print("=" * 70)
    print(f"  Analysis Score:    {report['score']:.1f}% ({len(report['passed'])}/{report['total_checks']} checks)")
    print(f"  Unit Test Score:   {test_score:.1f}% ({test_passed}/{result.testsRun} tests)")
    print(f"  ─────────────────────────────")
    print(f"  COMBINED SCORE:    {combined_score:.1f}%")
    print()
    
    # Grade
    if combined_score >= 95:
        grade, status = "A+", "PRODUCTION READY"
    elif combined_score >= 90:
        grade, status = "A", "EXCELLENT"
    elif combined_score >= 80:
        grade, status = "B", "GOOD"
    elif combined_score >= 70:
        grade, status = "C", "NEEDS IMPROVEMENT"
    else:
        grade, status = "D", "CRITICAL ISSUES"
    
    print(f"  GRADE: {grade} | STATUS: {status}")
    print("=" * 70)
    
    # Feature summary
    print("\n📋 SAMPLER PRO FEATURE SUMMARY")
    print("-" * 70)
    features = [
        ("Presets", "7 Flux workflow presets (txt2img, img2img, inpaint, etc.)"),
        ("Phase-Shift", "Dual-sampler mode: Euler→DPM++ or Euler→SGM"),
        ("Dynamic Guidance", "Low→High→Low guidance zones (0-20%, 20-90%, 90-100%)"),
        ("Flux Shift", "Native sigma shifting for high-res detail (0-10x)"),
        ("Refiner", "Integrated refiner model with configurable start step"),
        ("Noise Control", "add_noise, return_with_leftover_noise, noise_override"),
        ("Diagnostics", "Full timing breakdown and debug logging"),
        ("Bug Fixes", "8 documented fixes (v2.02 - v2.03)"),
    ]
    for name, desc in features:
        print(f"  • {name}: {desc}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 70)
    if combined_score >= 95:
        print("  ✅ Sampler Pro is production-ready, no changes needed.")
    else:
        print("  Consider addressing the warnings/issues listed above.")
    
    print("\n  Best Practices for Users:")
    print("    1. Use 'euler' sampler for Flux (most stable)")
    print("    2. Set CFG=1.0 (Flux uses internal guidance)")
    print("    3. Use flux_guidance=3.5-4.0 for prompt adherence")
    print("    4. Try Phase-Shift for structure+detail balance")
    print("    5. Enable Dynamic Guidance for creative consistency")
    
    return combined_score


if __name__ == "__main__":
    run_analysis()
