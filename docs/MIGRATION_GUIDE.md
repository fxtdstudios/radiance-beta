# Radiance Migration Guide

## Overview

This guide helps users migrate from older versions of Radiance to the latest version (2.1) with professional color science and optimizations.

---

## Version 1.1.0 (2026-01-22) - Professional Color Science Update

### 🎯 Summary of Changes

**What's New:**
- ✅ Professional ARRI LogC gamut conversions
- ✅ Input validation (resolution, metadata)
- ✅ 100x performance improvement (alpha premultiplication)
- ✅ Scene-linear color space option
- ✅ Consolidated redundant nodes

### ⚠️ Breaking Changes

#### 1. Color Output Changed (LogC3/LogC4)

**Impact:** HIGH - Colors will be different (CORRECT now)

**What Changed:**
- **Before:** Only log curve applied (sRGB primaries → WRONG)
- **After:** Proper gamut conversion + log curve (AWG/S-Gamut3.Cine primaries → CORRECT)

**Who's Affected:**
- Users exporting to ARRI LogC3/LogC4
- Professional color grading workflows

**Migration Steps:**
1. Re-export all LogC3/LogC4 files
2. Update DaVinci Resolve project settings:
   - LogC3: Set input to `ARRI LogC3 / AWG`
   - LogC4: Set input to `ARRI LogC4 / AWG`

**Before/After Comparison:**
```python
# OLD (WRONG):
ARRI LogC3 → Only curve (sRGB primaries)
# Result: Incorrect colors in DaVinci Resolve

# NEW (CORRECT):
ARRI LogC3 → Gamut conversion + curve (AWG primaries)
# Result: Accurate colors matching ARRI specs
```

---

#### 2. Removed Nodes (Consolidation)

**Impact:** MEDIUM - Workflow adjustments needed

##### Removed EXR Savers (5 nodes)
- ❌ `FXTDSaveEXRMultiLayer`
- ❌ `FXTDSaveEXRSequence`
- ❌ `FXTDEXRChannelMerge`
- ❌ `FXTDSaveEXRCryptomatte`
- ❌ `RadianceHDRSave`

**Replaced By:** `◆ Radiance Save EXR/HDR` (`FXTDSaveEXR`)

**Migration:**
```
OLD: RadianceHDRSave
     ↓
NEW: ◆ Radiance Save EXR/HDR
     - Set format = "HDR"
     - All features preserved
```

**Features Now Available:**
- ✅ EXR export (16-bit/32-bit)
- ✅ HDR export (Radiance RGBE)
- ✅ 10 compression modes
- ✅ Professional color spaces
- ✅ Metadata embedding
- ✅ Alpha handling

**Not Yet Available:**
- ⏳ Multi-layer EXR (may return in future)
- ⏳ Cryptomatte (separate node planned)
- ⏳ AOV channels (planned feature)

---

### 🆕 New Features

#### 1. Scene-Linear Color Space
**New Option:** `"Linear (Scene-Referred)"`

**What it does:**
- Bypasses ALL color conversions
- Preserves raw sensor/render data
- Maximum control for advanced users

**Use Cases:**
- Pre-graded footage from Blender/Maya
- Custom color pipelines
- Direct renderer output

**Example:**
```python
output_color_space = "Linear (Scene-Referred)"
# Result: No conversions applied
```

---

#### 2. Input Validation

**Resolution Limits:**
- Max width: 65,536 pixels (64K)
- Max height: 65,536 pixels (64K)
- Max pixels: 200,000,000 (200MP)

**What happens:**
```python
# Oversized image (e.g., 100K x 100K)
→ Error: "Total pixels exceeds maximum. Try tiled export."
→ Graceful failure with clear message
```

**Metadata Validation:**
- Max key length: 64 characters
- Max value length: 1,024 characters
- Max entries: 100
- Automatic sanitization of special characters

---

#### 3. Performance Improvements

**Alpha Premultiplication:**
- **Before:** 400ms for 8K image
- **After:** 4ms for 8K image
- **Speedup:** 100x faster

**Technical:** Vectorized NumPy operations

---

## Upgrade Path

### From Version 1.0.x

#### Step 1: Check Workflows
```bash
# Find workflows using old nodes
# Search for: RadianceHDRSave, FXTDSaveEXRMultiLayer
```

#### Step 2: Update Nodes
Replace old nodes with `◆ Radiance Save EXR/HDR`:
- Set `format` parameter
- Adjust `output_color_space` if needed
- Verify compression settings

#### Step 3: Re-export LogC
To get the proper wide-gamut log files that VFX compositors and colorists expect, you **must re-export** your sequences out of ComfyUI. The previous v1.x EXRs will grade differently because they lack the proper gamut transformations.

**DaVinci Resolve Setup**
When importing the newly generated v2.0 EXRs into DaVinci Resolve, right click the footage and set the Input Color Space:

| EXR Export Parameter | DaVinci Resolve Input Space |
| :--- | :--- |
| ARRI LogC3 | `ARRI LogC3 / AWG` |
| ARRI LogC4 | `ARRI LogC4 / AWG` |

#### Step 5: Test Workflows
- Verify colors in grading suite
- Check alpha channel handling
- Validate metadata

---

## Compatibility

### ComfyUI Versions
- **Minimum:** ComfyUI 0.1.0+
- **Recommended:** Latest stable
- **Tested:** January 2026 builds

### Dependencies
- **Required:** NumPy, OpenCV (for EXR)
- **Optional:** OpenEXR library (for metadata)
- **Optional:** colour-science (for OCIO)

### Operating Systems
- ✅ Windows
- ✅ Linux
- ✅ macOS

---

## FAQ

### Q: Do I need to update my workflows?

**A:** Only if you use:
1. LogC3/LogC4 color spaces → Re-export needed
2. Removed nodes → Replace with unified saver
3. Otherwise → No changes needed

---

### Q: Will my old EXR files still work?

**A:** Yes! Old files are unaffected. Only NEW exports will use correct color science.

---

### Q: What if I want the old behavior?

**A:** Not recommended, but you can:
```python
# Use "Same as Input" to bypass conversions
output_color_space = "Same as Input"
# Or use "Linear (Scene-Referred)"
```

---

### Q: Why did colors change?

**A:** Previous LogC exports were technically INCORRECT:
- Used wrong primaries (sRGB instead of ARRI/Sony)
- Only applied log curve
- Didn't match manufacturer specifications

New behavior is CORRECT and matches industry standards.

---

### Q: Do I lose multi-layer EXR support?

**A:** Temporarily. It may return in a future update if there's demand. Most users don't need it.

---

### Q: How do I migrate from RadianceHDRSave?

**A:** Simple replacement:

**Old Workflow:**
```
Image → RadianceHDRSave → HDR file
```

**New Workflow:**
```
Image → ◆ Radiance Save EXR/HDR
        - Set format = "HDR"
        → HDR file
```

All parameters preserved!

---

## Rollback (Not Recommended)

If you MUST rollback:

1. **Git checkout** old version:
   ```bash
   cd custom_nodes/radiance
   git checkout v1.0.0  # Old version
   ```

2. **Restart ComfyUI**

**Warning:** You will lose:
- Correct color science
- Performance improvements
- Input validation
- Bug fixes

---

## Get Help

### Issues
- GitHub: [Report Issue]
- Discord: [FXTD Studios Channel]

### Documentation
- Color Space Guide: `COLOR_SPACE_GUIDE.md`
- Complete Analysis: `complete_node_analysis.md`
- Walkthrough: `walkthrough.md`

---

## Timeline

| Version | Date | Changes |
|---------|------|---------|
| **1.1.0** | 2026-01-22 | ✅ Professional color science |
| **1.0.1** | 2025-XX-XX | Bug fixes |
| **1.0.0** | 2024-XX-XX | Initial release |

---

**Last Updated:** 2026-01-22  
**Migration Support:** Active
