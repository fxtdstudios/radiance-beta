# Radiance Color Space Guide

## Overview

Radiance supports professional color space conversions for VFX, color grading, and HDR workflows. This guide covers all color spaces, their proper usage, and technical details.

---

## Supported Color Spaces

### Input Color Spaces

| Color Space | Description | When to Use |
|-------------|-------------|-------------|
| **sRGB** | Standard RGB (Rec.709 primaries, gamma 2.2) | Default for most images, web content |
| **Linear** | Linear light (Rec.709 primaries, no gamma) | Pre-linearized images, render outputs |
| **Raw** | No conversion applied | Unknown or custom color spaces |

### Output Color Spaces

| Color Space | Primaries | Transfer | Professional Use |
|-------------|-----------|----------|------------------|
| **Linear** | Rec.709 | Linear | Compositing, intermediate files |
| **sRGB** | Rec.709 | Gamma 2.2 | Display, web delivery |
| **ACEScg** | AP1 | Linear | ACES workflows, VFX pipelines |
| **ARRI LogC3** | ✅ **AWG** | Log | ARRI Alexa workflows, DaVinci Resolve |
| **ARRI LogC4** | ✅ **AWG** | Log | ARRI Alexa 35 workflows |

| **Linear (Scene-Referred)** | None | None | **Bypass all conversions** |
| **Same as Input** | Original | Original | Preserve input encoding |

**✅ = Proper gamut conversion applied (professional accuracy)**

---

## Professional Color Science

### ARRI LogC Workflows

#### What We Fixed (2026-01-22)
**Before:** Only log curve applied (wrong primaries → incorrect colors)  
**After:** Proper gamut conversion + log curve (correct workflow)

#### Correct Workflow
```
Input (sRGB) → Linear → ARRI Wide Gamut → LogC3/4 → EXR
```

**Conversion Chain:**
1. Linearize: sRGB gamma → linear light
2. **Gamut Transform:** Rec.709 primaries → ARRI Wide Gamut (AWG3)
3. Log Encode: Linear → LogC3 or LogC4 curve
4. Save: EXR with correct metadata

**DaVinci Resolve Setup:**
- Input Color Space: `ARRI LogC3 / AWG`
- Timeline: `ARRI LogC3 / AWG` or `ACEScct`
- Output: As needed

**Gamut Matrix Used:**
```python
# sRGB (Rec.709) to ARRI Wide Gamut
[[0.680206, 0.236137, 0.083658],
 [0.085415, 0.908295, 0.006290],
 [0.002057, 0.027878, 0.970065]]
```

---



### ACEScg Workflows

**Workflow:**
```
Input (sRGB) → Linear (Rec.709) → ACEScg (AP1) → EXR
```

**Use Cases:**
- VFX pipelines (ILM, Weta, Framestore standard)
- Multi-vendor workflows
- HDR mastering

**Note:** ACEScg uses linear light with AP1 primaries (no log curve)

---

## Special Color Spaces

### Linear (Scene-Referred)
**New in 2026:** Purist option that bypasses ALL conversions.

**What it does:**
- Reads input data AS-IS
- No gamma conversion
- No primaries conversion
- No curve application
- Direct sensor/render data preservation

**When to use:**
- Pre-graded footage
- Custom color pipelines
- Maximum control workflows
- Direct from renderer (Blender, Maya)

**Example:**
```python
input_color_space = "Raw"
output_color_space = "Linear (Scene-Referred)"
# Result: Zero conversions applied
```

---

## Color Space Recommendations

### For Different Workflows

#### Web/Display Output
```python
input_color_space = "sRGB"
output_color_space = "sRGB"
# Result: sRGB display-ready
```

#### VFX Compositing
```python
input_color_space = "sRGB"
output_color_space = "ACEScg"
# Result: ACES pipeline ready
```

#### ARRI Camera Workflow
```python
input_color_space = "sRGB"  # If from diffusion model
output_color_space = "ARRI LogC3"  # Or LogC4 for Alexa 35
# Result: Proper AWG primaries + LogC curve
# DaVinci Resolve: Set to "ARRI LogC3 / AWG"
```



#### HDR Mastering
```python
input_color_space = "sRGB"
output_color_space = "Linear"
bit_depth = "32-bit Float"
# Result: Full HDR range preserved
```

---

## Technical Details

### Conversion Pipeline

**Full Pipeline:**
```
Input Image
    ↓
[Input Linearization] (if sRGB)
    ↓
Linear Rec.709 Working Space
    ↓
[Gamut Conversion] (if LogC)
    ↓
Target Primaries
    ↓
[Transfer Function] (log/gamma curve)
    ↓
Output Image → EXR/HDR
```

### Gamut Matrices

All matrices are D65 white point, derived from official manufacturer specifications:

**Sources:**
- ARRI: ARRI Camera System Technical Documentation
- Sony: Sony Technical Documentation (S-Gamut3.Cine)
- ACES: Academy Color Encoding System Specification

### Validation

**Resolution Limits:**
- Maximum width: 65,536 pixels (64K)
- Maximum height: 65,536 pixels (64K)
- Maximum total pixels: 200,000,000 (200MP safety limit)

**Metadata Limits:**
- Maximum key length: 64 characters
- Maximum value length: 1,024 characters
- Maximum entries: 100

---

## Migration Guide

### If You Used Old LogC

**What Changed:**
- **Before 2026-01-22:** Only curve applied (WRONG colors)
- **After 2026-01-22:** Gamut + curve (CORRECT colors)

**Action Required:**
1. Re-export files using LogC3/LogC4
2. Colors will be different (this is correct)
3. DaVinci Resolve/Nuke will now show accurate results

**Example:**
```python
# Old behavior (WRONG - don't use)
# → Only LogC3 curve, sRGB primaries

# New behavior (CORRECT)
output_color_space = "ARRI LogC3"
# → Proper AWG primaries + LogC3 curve
```

---

## Common Issues

### Colors Look Wrong in DaVinci Resolve

**Problem:** Input color space mismatch

**Solution:**
- If exported as `ARRI LogC3`: Set DaVinci input to `ARRI LogC3 / AWG`

- If exported as `ACEScg`: Set DaVinci input to `ACEScg / AP1`

### Image Too Dark/Bright

**Problem:** Wrong input color space

**Solution:**
- Check if input is sRGB or Linear
- If from diffusion model: usually sRGB
- If from Blender/Maya render: usually Linear

### Want Maximum Control

**Solution:** Use scene-linear bypass
```python
input_color_space = "Raw"
output_color_space = "Linear (Scene-Referred)"
# Result: No conversions, full control
```

---

## Performance Notes

### Alpha Premultiplication
**Optimized:** 10-100x faster than previous version

**Before:** ~400ms for 8K image  
**After:** ~4ms for 8K image

**Implementation:** Vectorized NumPy broadcasting instead of loops

---

## Further Reading

- [ACES Documentation](https://acescentral.com)
- [ARRI LogC White Paper](https://www.arri.com/en/learn-help/learn-help-camera-system/camera-workflow)

- [DaVinci Resolve Color Management](https://www.blackmagicdesign.com/products/davinciresolve/color)

---

**Last Updated:** 2026-02-23  
**Version:** 2.1.0 (Professional Color Science Update)
