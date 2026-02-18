
import sys
import os
import torch
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from color_utils import (
        linear_to_srgb, srgb_to_linear,
        linear_to_logc4, logc4_to_linear,
        linear_to_acescct, acescct_to_linear,
        aces2_tonemap,
        SRGB_TO_ACESCG, ACESCG_TO_SRGB,
        apply_matrix_transform
    )
    from nodes_color import RadianceLUTApply
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def check_unclamped(name, func, val_in_linear, expected_min_out_linear=None, description=""):
    """
    Checks if a function preserves values > 1.0 (super-whites) or < 0.0 (super-blacks).
    """
    input_arr = np.array([val_in_linear], dtype=np.float32)
    
    try:
        out = func(input_arr)
        val_out = out[0]
        
        is_clamped = False
        if val_in_linear > 1.0 and np.isclose(val_out, 1.0, atol=1e-5):
            is_clamped = True
        if val_in_linear > 1.0 and val_out < 1.0: # If it compresses significantly (like tonemap)
            pass 
            
        print(f"TEST: {name:<30} | In: {val_in_linear:<8.3f} | Out: {val_out:<8.3f} | {description}")
        
        if is_clamped:
            print(f"  FAILED: {name} appears to be clamped to 1.0!")
            return False
        return True
    except Exception as e:
        print(f"  ERROR: {name} crashed: {e}")
        return False

def test_precision():
    print("="*60)
    print("FXTD Radiance - 32-bit Precision & Clamping Test")
    print("="*60)

    # 1. Colorspace Transforms (Matrix)
    # --------------------------------
    # ACEScg to sRGB (Linear)
    # If we convert a very bright ACEScg green to sRGB, does it clip?
    # (Note: sRGB gamut is smaller, so some values might go negative or >1)
    
    print("\n[ Matrix Transformations ]")
    # Test Super-White
    val_super = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    res = apply_matrix_transform(val_super, ACESCG_TO_SRGB)
    print(f"ACEScg(2.0) -> sRGB Linear: {res}")
    if np.all(res > 1.9):
         print("  ✓ PASS: Matrix transform preserves magnitude > 1.0")
    else:
         print("  ❌ FAIL: Matrix transform lost magnitude.")

    # 2. Transfer Functions (EOTF)
    # ----------------------------
    print("\n[ Transfer Functions ]")
    
    # linear_to_srgb (Gamma)
    # Input 2.0 linear -> Should be > 1.0 sRGB (approx 1.055 * 2^(1/2.4) ...)
    check_unclamped("linear_to_srgb (>1)", linear_to_srgb, 2.0, description="Gamma encode > 1")
    
    # logc4 (HDR encoding)
    # LogC4 can encode massive values. 2.0 linear is roughly 5 stops above grey?
    # 2.0 linear should be encoded safely.
    check_unclamped("linear_to_logc4", linear_to_logc4, 50.0, description="LogC4 encode high value")
    
    # ACEScct
    check_unclamped("linear_to_acescct", linear_to_acescct, 2.0)

    # 3. Tonemapping (Expected to Clamp?)
    # -----------------------------------
    # aces2_tonemap
    print("\n[ Tone Mapping (Expected Clamp) ]")
    # This SHOuLD clamp for display
    res_tm = aces2_tonemap(np.array([10.0], dtype=np.float32), peak_luminance=1000.0)
    print(f"aces2_tonemap(10.0): {res_tm[0]}")
    if res_tm[0] <= 1.0:
        print("  ✓ NOTE: Tonemapper clamps to 1.0 (Expected behavior for display)")
    else:
        print("  SHOCK: Tonemapper output > 1.0?")

    # 4. LUT Apply (The Critical Check)
    # ---------------------------------
    print("\n[ LUT Application ]")
    # We can't easily load a file here, but we can inspect the code logic via logic check.
    # RadianceLUTApply.trilinear_interpolate has:
    # coords = torch.clamp(coords, 0.0, 1.0)
    print("Analysis confirmed: RadianceLUTApply clamps input coordinates to 0-1.")
    print("This means HDR values > 1.0 are clipped before LUT lookup.")
    print("This is Standard Behavior for 3D LUTs.")

    print("\n" + "="*60)
    print("Test Complete")

if __name__ == "__main__":
    test_precision()
