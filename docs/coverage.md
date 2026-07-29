[← Back to Radiance docs](README.md)

# Node Coverage Ledger

Generated from the live `NODE_CLASS_MAPPINGS` by `tools/generate_docs.py`.
Dynamic `.gizmo` nodes are created at runtime and are not counted here.

`tests/test_docs_coverage.py` fails if this drifts from the catalog — it
did drift before, in both directions at once: nine node classes existed
in the source and were never registered, and four registered nodes had
never been added to this ledger.

## Summary

| Group | Count |
| :--- | ---: |
| [Color](built-in-nodes/color.md) | 13 |
| [Generate, Loaders, and Sampling](built-in-nodes/generate.md) | 11 |
| [HDR and ACES](built-in-nodes/hdr-aces.md) | 19 |
| [IO and Delivery](built-in-nodes/io-delivery.md) | 5 |
| [Pipeline and Studio](built-in-nodes/pipeline.md) | 4 |
| [Review, Viewer, and Preview](built-in-nodes/review.md) | 10 |
| [Upscale](built-in-nodes/upscale.md) | 4 |
| [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) | 29 |
| [Video](built-in-nodes/video.md) | 14 |
| **Total** | **109** |

## Color

- `RadianceApplyGradeInfo`
- `RadianceCDLTransform`
- `RadianceCDLExport`
- `RadianceCDLImport`
- `RadianceCurves`
- `RadianceGrade`
- `RadianceGradeMatch`
- `RadianceHueCurves`
- `RadianceLUTApply`
- `RadianceColorSpaceConvert`
- `RadianceOCIOContext`
- `RadianceLUTBlend`
- `RadianceWhiteBalance`

## Generate, Loaders, and Sampling

- `RadianceControlNetApply`
- `RadianceDenoise`
- `RadianceLoraStack`
- `RadianceUnifiedLoader`
- `RadianceCinematicPromptEncoder`
- `RadianceRegionalGrid`
- `RadianceRegionalPrompt`
- `RadianceResolution`
- `RadianceSamplerPro`
- `RadianceHDRVAEDecode`
- `RadianceHDRLatentEncoder`

## HDR and ACES

- `RadianceACES2ReachGamutCompress`
- `RadianceACES2OutputTransformFull`
- `RadianceACES2Tonescale`
- `RadianceACESTransform`
- `RadianceClipDetector`
- `RadianceHDRAutoLogSelect`
- `RadianceHDRColorPipeline`
- `RadianceHDRDiagnostics`
- `RadianceHDREncode`
- `RadianceHDRExpandDynamicRange`
- `RadianceHDRHighlightComposite`
- `RadianceHDRLoRAApply`
- `RadianceHDRLoRALoader`
- `RadianceHDRSynthesisEngine`
- `RadianceSDRtoHDRExpand`
- `RadianceSDRToHDRPrepare`
- `RadianceSDRToHDRRecover`
- `RadianceSDRToHDRUniversal`
- `RadianceHDRToneMap`

## IO and Delivery

- `RadianceDigitalCinemaRead`
- `RadianceDigitalCinemaWrite`
- `RadianceRead`
- `RadianceWrite`
- `RadianceEXRMultiPart`

## Pipeline and Studio

- `RadianceNukeSend`
- `RadianceDaVinciSend`
- `RadianceMCP`
- `RadianceProjectManager`

## Review, Viewer, and Preview

- `RadianceFrameStamp`
- `RadianceContactSheet`
- `RadianceFlipbookGIF`
- `RadianceFocusPeaking`
- `RadianceHDRMonitor`
- `RadiancePreviewServer`
- `RadianceQC`
- `RadiancePolicyGuard`
- `RadianceViewer`
- `RadianceLiteViewer`

## Upscale

- `RadianceUpscaleFaceRestore`
- `RadianceUpscaleImage`
- `RadianceUpscaleTiler`
- `RadianceUpscaleVideo`

## VFX, Masks, Optics, and Multipass

- `RadianceChromaticAberration`
- `RadianceAnamorphicStreaks`
- `RadianceBlendComposite`
- `RadianceDepthMapGenerator`
- `RadianceFilmGrain`
- `RadianceHDRGrainMatcher`
- `RadianceHDRCrop`
- `RadianceHDRStitch`
- `RadianceSAMGenerator`
- `RadianceLensDistortion`
- `RadianceLinearMatting`
- `RadianceMotionBlur`
- `RadianceOpticalFlow`
- `RadianceMultipassAOVReader`
- `RadianceMultipassComposite`
- `RadianceMultipassMaster`
- `RadianceMultipassRelight`
- `RadianceBitDepthDegrade`
- `RadianceLoadImageMask`
- `RadianceRelightEngine`
- `RadianceVectorMaskDraw`
- `RadianceSAMModelLoader`
- `RadianceMultiMaskVisualPicker`
- `RadianceSceneCutDetect`
- `RadianceSceneCutSplit`
- `RadianceSubpixelStabilizer`
- `RadianceTemporalStitchStabilizer`
- `RadianceVignette`
- `RadianceEXRPassesWriter`

## Video

- `RadianceI2VPipeline`
- `RadianceT2VPipeline`
- `RadianceVideoAssembler`
- `RadianceVideoBatchDecode`
- `RadianceVideoCondMerge`
- `RadianceVideoExport`
- `RadianceVideoFrameRouter`
- `RadianceVideoHDRConditioner`
- `RadianceVideoHDRDecode`
- `RadianceVideoLatentNoise`
- `RadianceVideoLoader`
- `RadianceVideoMaskPropagator`
- `RadianceVideoModelInfo`
- `RadianceVideoSampler`

## Documentation gaps

33 of 109 nodes are missing a `DESCRIPTION`, an
input tooltip, or both. Those appear as blank cells in the reference
tables above and as an empty hover in ComfyUI itself, so this list is
the honest backlog rather than a silent gap.

| Node | Has description | Tooltips |
| :--- | :---: | ---: |
| `RadianceApplyGradeInfo` | yes | 0 of 3 |
| `RadianceBitDepthDegrade` | yes | 0 of 6 |
| `RadianceCDLExport` | yes | 0 of 12 |
| `RadianceColorSpaceConvert` | yes | 0 of 6 |
| `RadianceCurves` | yes | 0 of 7 |
| `RadianceDigitalCinemaRead` | **no** | 0 of 6 |
| `RadianceDigitalCinemaWrite` | **no** | 0 of 4 |
| `RadianceEXRMultiPart` | yes | 0 of 15 |
| `RadianceEXRPassesWriter` | yes | 0 of 9 |
| `RadianceHDRCrop` | **no** | 0 of 4 |
| `RadianceHDRExpandDynamicRange` | yes | 0 of 6 |
| `RadianceHDRGrainMatcher` | **no** | 0 of 7 |
| `RadianceHDRStitch` | **no** | 0 of 6 |
| `RadianceHueCurves` | yes | 0 of 5 |
| `RadianceLinearMatting` | **no** | 0 of 5 |
| `RadianceLoadImageMask` | yes | 0 of 1 |
| `RadianceMultiMaskVisualPicker` | **no** | 0 of 2 |
| `RadianceMultipassComposite` | **no** | 0 of 14 |
| `RadianceMultipassMaster` | yes | 0 of 33 |
| `RadianceMultipassRelight` | **no** | 0 of 26 |
| `RadianceOCIOContext` | yes | 0 of 2 |
| `RadiancePolicyGuard` | yes | 0 of 15 |
| `RadianceQC` | yes | 0 of 14 |
| `RadianceSAMGenerator` | **no** | 0 of 6 |
| `RadianceSAMModelLoader` | **no** | 0 of 4 |
| `RadianceSubpixelStabilizer` | **no** | 0 of 3 |
| `RadianceTemporalStitchStabilizer` | **no** | 0 of 2 |
| `RadianceVectorMaskDraw` | **no** | 1 of 5 |
| `RadianceVideoCondMerge` | yes | 0 of 7 |
| `RadianceVideoExport` | yes | 0 of 7 |
| `RadianceVideoMaskPropagator` | **no** | 1 of 3 |
| `RadianceVideoModelInfo` | yes | 0 of 5 |
| `RadianceWhiteBalance` | yes | 0 of 12 |
