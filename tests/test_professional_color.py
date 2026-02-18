"""
═══════════════════════════════════════════════════════════════════════════════
              PROFESSIONAL COLOR PIPELINE ANALYSIS
              DoP / Camera Manufacturer / Colorist Perspective
              
              Analysis of LogC Decode and 32-bit Raw Workflow
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ProfessionalColorAnalyzer:
    """
    Professional analysis from DoP/Colorist perspective.
    
    Standard Reference Values (Camera Manufacturer Specs):
    - ARRI LogC3 EI800: 18% gray → 0.391 code value
    - ARRI LogC4: 18% gray → 0.32 code value
    - Sony S-Log3: 18% gray → 0.406 (10-bit: 420)
    - Panasonic V-Log: 18% gray → 0.423
    - Canon Log3: 18% gray → 0.343
    """
    
    def __init__(self):
        from color_utils import (
            linear_to_logc3, logc3_to_linear,
            linear_to_logc4, logc4_to_linear,
            linear_to_slog3, slog3_to_linear,
            linear_to_vlog, vlog_to_linear,
            linear_to_canonlog3, canonlog3_to_linear,
            linear_to_acescct, acescct_to_linear,
            AWG3_TO_ACESCG, AWG4_TO_ACESCG, SGAMUT3_CINE_TO_ACESCG,
            VGAMUT_TO_ACESCG, CINEMA_GAMUT_TO_ACESCG
        )
        
        self.curves = {
            "ARRI LogC3": {
                "encode": linear_to_logc3,
                "decode": logc3_to_linear,
                "matrix": AWG3_TO_ACESCG,
                "mid_gray_target": 0.391,  # Official ARRI spec EI800
                "tolerance": 0.01,
                "camera": "ARRI ALEXA Mini, ALEXA LF, AMIRA",
            },
            "ARRI LogC4": {
                "encode": linear_to_logc4,
                "decode": logc4_to_linear,
                "matrix": AWG4_TO_ACESCG,
                "mid_gray_target": 0.32,   # ALEXA 35 spec
                "tolerance": 0.01,
                "camera": "ARRI ALEXA 35",
            },
            "Sony S-Log3": {
                "encode": linear_to_slog3,
                "decode": slog3_to_linear,
                "matrix": SGAMUT3_CINE_TO_ACESCG,
                "mid_gray_target": 0.410,  # 420/1023
                "tolerance": 0.01,
                "camera": "Sony VENICE, FX9, FX6, a7S III",
            },
            "Panasonic V-Log": {
                "encode": linear_to_vlog,
                "decode": vlog_to_linear,
                "matrix": VGAMUT_TO_ACESCG,
                "mid_gray_target": 0.423,
                "tolerance": 0.015,
                "camera": "Panasonic VariCam, S1H, GH6",
            },
            "Canon Log3": {
                "encode": linear_to_canonlog3,
                "decode": canonlog3_to_linear,
                "matrix": CINEMA_GAMUT_TO_ACESCG,
                "mid_gray_target": 0.343,
                "tolerance": 0.02,
                "camera": "Canon C70, C300 III, R5C",
            },
        }
        
        self.results = []
        self.issues = []
        self.passed = []
    
    def analyze_mid_gray(self, name, curve_info):
        """Test 18% gray encoding accuracy."""
        mid_gray = np.array([0.18], dtype=np.float32)
        encoded = curve_info["encode"](mid_gray)
        target = curve_info["mid_gray_target"]
        tolerance = curve_info["tolerance"]
        
        error = abs(encoded[0] - target)
        
        if error <= tolerance:
            self.passed.append((name, f"18% Gray → {encoded[0]:.4f} (target: {target}, error: {error:.4f})"))
            return True
        else:
            self.issues.append((name, f"18% Gray → {encoded[0]:.4f} (target: {target}, ERROR: {error:.4f})"))
            return False
    
    def analyze_roundtrip(self, name, curve_info):
        """Test encode→decode precision."""
        test_values = np.array([0.001, 0.01, 0.18, 0.5, 1.0, 5.0, 20.0, 60.0], dtype=np.float32)
        
        encoded = curve_info["encode"](test_values)
        decoded = curve_info["decode"](encoded)
        
        # Calculate relative error
        rel_error = np.abs(decoded - test_values) / (test_values + 1e-10) * 100
        max_error = np.max(rel_error)
        avg_error = np.mean(rel_error)
        
        if max_error < 0.01:  # 0.01% tolerance
            self.passed.append((f"{name} Roundtrip", f"Max error: {max_error:.6f}% ✓"))
            return True
        elif max_error < 0.1:  # 0.1% warning
            self.passed.append((f"{name} Roundtrip", f"Max error: {max_error:.4f}% (acceptable)"))
            return True
        else:
            self.issues.append((f"{name} Roundtrip", f"Max error: {max_error:.4f}% EXCEEDS TOLERANCE"))
            return False
    
    def analyze_dynamic_range(self, name, curve_info):
        """Test dynamic range encoding."""
        # Camera stops above/below 18% gray
        stops = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14]  # 22+ stops
        linear_values = 0.18 * np.power(2.0, np.array(stops, dtype=np.float32))
        
        encoded = curve_info["encode"](linear_values)
        decoded = curve_info["decode"](encoded)
        
        # Check if all values decode correctly
        rel_error = np.abs(decoded - linear_values) / (linear_values + 1e-10) * 100
        max_error = np.max(rel_error)
        
        # Count usable stops (where error < 1%)
        usable_stops = np.sum(rel_error < 1.0)
        
        if usable_stops >= 10:
            self.passed.append((f"{name} DR", f"{usable_stops}/{len(stops)} stops accurate (<1% error)"))
        else:
            self.issues.append((f"{name} DR", f"Only {usable_stops}/{len(stops)} stops accurate"))
        
        return usable_stops
    
    def analyze_all(self):
        """Run complete analysis."""
        print("═" * 70)
        print("PROFESSIONAL COLOR PIPELINE ANALYSIS")
        print("Perspective: DoP / Camera Manufacturer / Colorist")
        print("═" * 70)
        
        for name, curve_info in self.curves.items():
            print(f"\n▶ {name} ({curve_info['camera']})")
            print("-" * 60)
            
            self.analyze_mid_gray(name, curve_info)
            self.analyze_roundtrip(name, curve_info)
            stops = self.analyze_dynamic_range(name, curve_info)
        
        return self._generate_report()
    
    def _generate_report(self):
        """Generate final report."""
        total = len(self.passed) + len(self.issues)
        score = (len(self.passed) / total * 100) if total > 0 else 0
        
        return {
            "score": score,
            "passed": self.passed,
            "issues": self.issues,
        }


def test_32bit_raw_workflow():
    """Test complete 32-bit raw image workflow."""
    from color_utils import (
        logc4_to_linear, linear_to_logc4,
        apply_matrix_transform, AWG4_TO_ACESCG, ACESCG_TO_SRGB
    )
    
    print("\n" + "═" * 70)
    print("32-BIT RAW WORKFLOW VERIFICATION")
    print("═" * 70)
    
    # Simulate ARRI ProRes 4444 LogC4 input (typical cinema workflow)
    # Values represent a high dynamic range scene
    h, w = 1080, 1920
    
    # Simulated LogC4 encoded data (0.0 - 1.0 range)
    np.random.seed(42)
    logc4_encoded = np.random.uniform(0.1, 0.9, (h, w, 3)).astype(np.float32)
    
    # Add some super-brights (HDR highlights)
    logc4_encoded[500:520, 900:1000, :] = 0.95  # Bright window
    # Add some deep shadows
    logc4_encoded[800:900, 100:200, :] = 0.08   # Dark corner
    
    print(f"\n1️⃣ Input: LogC4 Encoded (simulated ProRes 4444)")
    print(f"   Shape: {logc4_encoded.shape}")
    print(f"   Range: [{logc4_encoded.min():.4f}, {logc4_encoded.max():.4f}]")
    print(f"   Dtype: {logc4_encoded.dtype}")
    
    # Step 1: Decode LogC4 → Linear
    linear = logc4_to_linear(logc4_encoded)
    print(f"\n2️⃣ LogC4 → Linear Decode")
    print(f"   Range: [{linear.min():.4f}, {linear.max():.4f}]")
    print(f"   Dtype: {linear.dtype}")
    print(f"   HDR Values > 1.0: {np.sum(linear > 1.0)} pixels")
    
    # Step 2: Gamut Transform (AWG4 → ACEScg)
    acescg = apply_matrix_transform(linear, AWG4_TO_ACESCG)
    print(f"\n3️⃣ AWG4 → ACEScg Gamut Transform")
    print(f"   Range: [{acescg.min():.4f}, {acescg.max():.4f}]")
    print(f"   Dtype: {acescg.dtype}")
    
    # Step 3: Apply grade (simulate exposure adjustment)
    exposure_adj = 1.2  # +1/4 stop
    graded = acescg * exposure_adj
    print(f"\n4️⃣ Grade Applied (Exposure +0.25 stop)")
    print(f"   Range: [{graded.min():.4f}, {graded.max():.4f}]")
    
    # Step 4: Transform to display (ACEScg → sRGB Linear)
    display_linear = apply_matrix_transform(graded, ACESCG_TO_SRGB)
    print(f"\n5️⃣ ACEScg → sRGB Linear")
    print(f"   Range: [{display_linear.min():.4f}, {display_linear.max():.4f}]")
    
    # Verify 32-bit maintained throughout
    all_float32 = all([
        logc4_encoded.dtype == np.float32,
        linear.dtype == np.float32,
        acescg.dtype == np.float32,
        graded.dtype == np.float32,
        display_linear.dtype == np.float32,
    ])
    
    if all_float32:
        print(f"\n✅ 32-BIT PRECISION MAINTAINED THROUGHOUT PIPELINE")
    else:
        print(f"\n❌ PRECISION LOST - CHECK PIPELINE")
    
    # Check for HDR preservation
    if np.max(display_linear) > 1.0:
        print(f"✅ HDR VALUES PRESERVED (max: {np.max(display_linear):.2f})")
    
    return all_float32


def generate_best_workflow():
    """Generate best practice workflow documentation."""
    print("\n" + "═" * 70)
    print("RECOMMENDED 32-BIT RAW WORKFLOW")
    print("Professional Cinema Pipeline")
    print("═" * 70)
    
    workflow = """
┌─────────────────────────────────────────────────────────────────────────┐
│                    RADIANCE 32-BIT RAW WORKFLOW                         │
│                   (Cinema / VFX Standard Practice)                      │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: IMPORT
──────────────
  📥 Load Image/EXR → ImageToFloat32
  
  Input Formats Supported:
  • EXR (32-bit linear)
  • ProRes 4444 (decoded via ffmpeg)
  • TIFF 16/32-bit
  • PNG/JPG (will be converted to float32)


STEP 2: DECODE LOG (Camera Native → Linear)
───────────────────────────────────────────
  Choose decoder based on camera:
  
  ┌─────────────────┬──────────────────┬─────────────────────────┐
  │ Camera System   │ Log Curve        │ Gamut                   │
  ├─────────────────┼──────────────────┼─────────────────────────┤
  │ ARRI ALEXA 35   │ LogC4 Decode     │ AWG4 → ACEScg           │
  │ ARRI ALEXA Mini │ LogC3 Decode     │ AWG3 → ACEScg           │
  │ Sony VENICE     │ S-Log3 Decode    │ S-Gamut3.Cine → ACEScg  │
  │ Panasonic S1H   │ V-Log Decode     │ V-Gamut → ACEScg        │
  │ Canon C70       │ Canon Log3       │ Cinema Gamut → ACEScg   │
  │ RED DSMC2       │ Log3G10 Decode   │ REDWideGamut → ACEScg   │
  └─────────────────┴──────────────────┴─────────────────────────┘

  Node Chain: LogC4 Decode → GPU Color Matrix (AWG4→ACEScg)


STEP 3: GRADE IN ACESCG (32-bit Linear Scene-Referred)
──────────────────────────────────────────────────────
  ► Float32 Color Correct
    • Exposure, Contrast, Saturation
    • CDL (Slope/Offset/Power)
    • Lift/Gamma/Gain
  
  ► Apply LUT (Optional - creative looks)
    • Radiance LUT Apply (Tetrahedral interpolation recommended)
    • Log-space LUT application for matching DI workflow
  
  ► Film Grain (Optional)
    • FXTDTemporalGrain for video
    • Per-channel R/G/B control
    • Temporal smoothness for flicker-free


STEP 4: OUTPUT TRANSFORM
────────────────────────
  Based on delivery target:

  ┌──────────────────┬────────────────────────────────────────────┐
  │ Delivery         │ Transform                                  │
  ├──────────────────┼────────────────────────────────────────────┤
  │ Cinema DCI-P3    │ ACES 2.0 Output (P3-D65, SDR)              │
  │ HDR10 Master     │ ACES 2.0 Output (Rec.2020, PQ 1000 nits)   │
  │ Dolby Vision     │ ACES 2.0 Output (P3-D65, PQ 4000 nits)     │
  │ Web/Broadcast    │ ACEScg → sRGB + sRGB gamma                 │
  │ VFX Handoff      │ Keep ACEScg Linear (EXR export)            │
  └──────────────────┴────────────────────────────────────────────┘


STEP 5: EXPORT
──────────────
  For VFX/DI Handoff:
    ► Save EXR (ACEScg Linear, 32-bit)
    
  For Review/Delivery:
    ► Save Image (sRGB, 16-bit PNG/TIFF)
    
  For Web:
    ► Save Image (sRGB, 8-bit JPG)


═══════════════════════════════════════════════════════════════════════════
                           EXAMPLE NODE CHAIN
═══════════════════════════════════════════════════════════════════════════

  ┌────────────┐    ┌───────────────┐    ┌──────────────────┐
  │ Load Image │───►│ ImageToFloat32│───►│ RadianceLogDecode│
  └────────────┘    └───────────────┘    │  (LogC4→Linear)  │
                                          └────────┬─────────┘
                                                   │
                    ┌──────────────────────────────▼─────────────────┐
                    │         RadianceGPUColorMatrix                 │
                    │            (AWG4 → ACEScg)                     │
                    └────────────────────┬───────────────────────────┘
                                         │
     ┌───────────────────────────────────▼────────────────────────────┐
     │                    Float32ColorCorrect                         │
     │       (Exposure, Contrast, Lift/Gamma/Gain in ACEScg)          │
     └───────────────────────────────────┬────────────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────┐
                    │         RadianceLUTApply               │
                    │    (Optional - Tetrahedral mode)       │
                    └────────────────────┬───────────────────┘
                                         │
                    ┌────────────────────▼───────────────────┐
                    │       FXTDTemporalGrain                │
                    │    (Optional - for filmic texture)     │
                    └────────────────────┬───────────────────┘
                                         │
     ┌───────────────────────────────────▼────────────────────────────┐
     │                   ACES 2.0 Output Transform                    │
     │             (Target: P3-D65/Rec.2020, SDR/HDR)                 │
     └───────────────────────────────────┬────────────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────┐
                    │            Save EXR                    │ ──► VFX
                    │    (32-bit Linear ACEScg)              │
                    └────────────────────────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────┐
                    │          Save Image                    │ ──► Delivery
                    │       (sRGB, 16-bit PNG)               │
                    └────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
                              KEY TIPS
═══════════════════════════════════════════════════════════════════════════

  1. ALWAYS work in 32-bit float for grading
     → Prevents banding and preserves highlight/shadow detail

  2. Grade in ACEScg (Scene-Referred Linear)
     → Industry standard for cross-camera matching
     → Widest working gamut for compositing

  3. Use Log-space LUTs with Log Input enabled
     → Matches DaVinci Resolve / Baselight workflow
     → Preserves dynamic range during LUT application

  4. Choose Tetrahedral interpolation for critical color work
     → More accurate than trilinear, especially for skin tones

  5. Apply grain AFTER grading, BEFORE output transform
     → Grain responds correctly to scene brightness
     → Temporal grain prevents video flicker

  6. For VFX handoff: Export 32-bit EXR in ACEScg Linear
     → No baked-in look, maximum flexibility downstream

═══════════════════════════════════════════════════════════════════════════
"""
    print(workflow)
    return True


def run_full_analysis():
    """Run complete professional analysis."""
    analyzer = ProfessionalColorAnalyzer()
    report = analyzer.analyze_all()
    
    # Print results
    print("\n" + "─" * 70)
    print("ANALYSIS SUMMARY")
    print("─" * 70)
    
    print("\n✅ PASSED:")
    for name, result in report["passed"]:
        print(f"   {name}: {result}")
    
    if report["issues"]:
        print("\n⚠️ ISSUES:")
        for name, result in report["issues"]:
            print(f"   {name}: {result}")
    
    score = report["score"]
    
    # Grade
    if score >= 95:
        grade, status = "A+", "PRODUCTION READY"
    elif score >= 90:
        grade, status = "A", "EXCELLENT"
    elif score >= 80:
        grade, status = "B", "GOOD"
    else:
        grade, status = "C", "NEEDS REVIEW"
    
    print(f"\n{'═' * 70}")
    print(f"COLOR PIPELINE SCORE: {score:.1f}% (Grade: {grade})")
    print(f"STATUS: {status}")
    print('═' * 70)
    
    # Run 32-bit workflow test
    test_32bit_raw_workflow()
    
    # Generate best workflow
    generate_best_workflow()
    
    return score


if __name__ == "__main__":
    run_full_analysis()
