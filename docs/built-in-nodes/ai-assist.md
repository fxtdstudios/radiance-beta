[← Back to Radiance docs](../README.md)

# AI Assist

> **Moved.** The scene-cut nodes now live under the VFX menu and are documented
> in [VFX, Masks, Optics, and Multipass](vfx.md). This page is kept so existing
> links do not break.

- `RadianceSceneCutDetect` — [in the VFX reference](vfx.md)
- `RadianceSceneCutSplit` — [in the VFX reference](vfx.md)

## Before you use them

The detector normalises its scores by the batch maximum, so `threshold` has no
absolute meaning: the largest inter-frame difference in any clip is always 1.0
and therefore always exceeds the threshold. On footage with no cuts it will still
report cuts, spaced at `min_shot_frames`. Treat the output as a starting point to
check by eye, not as a decision. See [Known Limitations](../limitations.md).
