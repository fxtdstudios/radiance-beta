# Documentation Coverage Ledger

This ledger is generated from the grouped Radiance node catalog used by the documentation. Dynamic `.gizmo` nodes are runtime-generated and are not counted as fixed catalog nodes.

## Summary

| Group | Count |
| :--- | ---: |
| [IO and Delivery](built-in-nodes/io-delivery.md) | 4 |
| [Generate, Loaders, and Sampling](built-in-nodes/generate.md) | 13 |
| [Color](built-in-nodes/color.md) | 15 |
| [HDR and ACES](built-in-nodes/hdr-aces.md) | 15 |
| [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) | 24 |
| [Pipeline and Studio](built-in-nodes/pipeline.md) | 8 |
| [Review, Viewer, and Preview](built-in-nodes/review.md) | 7 |
| [Upscale](built-in-nodes/upscale.md) | 4 |
| [Video](built-in-nodes/video.md) | 12 |
| [AI Assist](built-in-nodes/ai-assist.md) | 2 |
| **Total** | **104** |

## IO and Delivery

- `RadianceRead`
- `RadianceWrite`
- `RadianceEXRMultiPart`
- `RadianceLoadImageMask`

## Generate, Loaders, and Sampling

- `RadianceSamplerPro`
- `RadianceHDRVAEDecode`
- `RadianceLoraStack`
- `RadianceUnifiedLoader`
- `RadianceVideoLoader`
- `RadianceControlNetApply`
- `RadianceHDRLoRALoader`
- `RadianceHDRLoRAApply`
- `RadianceCinematicPromptEncoder`
- `RadianceRegionalPrompt`
- `RadianceRegionalGrid`
- `RadianceResolution`
- `RadianceDenoise`

## Color

- `RadianceCDLTransform`
- `RadianceCDLImport`
- `RadianceCDLExport`
- `RadianceWhiteBalance`
- `RadianceColorSpaceConvert`
- `RadianceACESTransform`
- `RadianceBitDepthDegrade`
- `RadianceHueCurves`
- `RadianceCurves`
- `RadianceGrade`
- `RadianceApplyGradeInfo`
- `RadianceGradeMatch`
- `RadianceOCIOContext`
- `RadianceQC`
- `RadiancePolicyGuard`

## HDR and ACES

- `RadianceACES2Tonescale`
- `RadianceACES2ReachGamutCompress`
- `RadianceACES2OutputTransformFull`
- `RadianceHDRColorPipeline`
- `RadianceHDREncode`
- `RadianceHDRMonitor`
- `RadianceHDRAutoLogSelect`
- `RadianceHDRDiagnostics`
- `RadianceClipDetector`
- `RadianceSDRToHDRPrepare`
- `RadianceHDRHighlightComposite`
- `RadianceSDRtoHDRExpand`
- `RadianceHDRSynthesisEngine`
- `RadianceRelightEngine`
- `RadianceHDRLatentEncoder`

## VFX, Masks, Optics, and Multipass

- `RadianceDepthMapGenerator`
- `RadianceOpticalFlow`
- `RadianceMotionBlur`
- `RadianceLensDistortion`
- `RadianceChromaticAberration`
- `RadianceAnamorphicStreaks`
- `RadianceFilmGrain`
- `RadianceVignette`
- `RadianceSAMModelLoader`
- `RadianceSAMGenerator`
- `RadianceMultiMaskVisualPicker`
- `RadianceLinearMatting`
- `RadianceHDRGrainMatcher`
- `RadianceSubpixelStabilizer`
- `RadianceHDRCrop`
- `RadianceHDRStitch`
- `RadianceTemporalStitchStabilizer`
- `RadianceVectorMaskDraw`
- `RadianceVideoMaskPropagator`
- `RadianceMultipassMaster`
- `RadianceMultipassAOVReader`
- `RadianceEXRPassesWriter`
- `RadianceMultipassRelight`
- `RadianceMultipassComposite`

## Pipeline and Studio

- `RadianceAudioCut`
- `RadianceProjectManager`
- `RadianceBlendComposite`
- `RadianceCinemaStudio`
- `RadianceMCP`
- `RadianceNukeSend`
- `RadianceDaVinciSend`
- `RadianceParamHistoryTracker`

## Review, Viewer, and Preview

- `RadianceLiteViewer`
- `RadianceViewer`
- `RadianceFocusPeaking`
- `RadianceContactSheet`
- `RadianceFlipbookGIF`
- `RadianceFrameStamp`
- `RadiancePreviewServer`

## Upscale

- `RadianceUpscaleTiler`
- `RadianceUpscaleImage`
- `RadianceUpscaleVideo`
- `RadianceUpscaleFaceRestore`

## Video

- `RadianceVideoModelInfo`
- `RadianceVideoLatentNoise`
- `RadianceVideoCondMerge`
- `RadianceVideoSampler`
- `RadianceT2VPipeline`
- `RadianceI2VPipeline`
- `RadianceVideoBatchDecode`
- `RadianceVideoExport`
- `RadianceVideoHDRConditioner`
- `RadianceVideoHDRDecode`
- `RadianceVideoFrameRouter`
- `RadianceVideoAssembler`

## AI Assist

- `RadianceSceneCutDetect`
- `RadianceSceneCutSplit`
