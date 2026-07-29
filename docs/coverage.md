[← Back to Radiance docs](README.md)

# Documentation Coverage Ledger

This ledger is generated from the grouped Radiance node catalog used by the documentation. Dynamic `.gizmo` nodes are runtime-generated and are not counted as fixed catalog nodes.

> Nine of these node classes existed in the source from v3.0 but were never
> listed in a group's mapping dict, so ComfyUI never showed them. They were
> registered in 3.2.0. Four more (the tone-map pair and the SDR→HDR pair)
> were live but missing from this ledger. The totals now match
> `NODE_CLASS_MAPPINGS` exactly — `tests/test_docs_coverage.py` checks it.

## Summary

| Group | Count |
| :--- | ---: |
| [IO and Delivery](built-in-nodes/io-delivery.md) | 6 |
| [Generate, Loaders, and Sampling](built-in-nodes/generate.md) | 13 |
| [Color](built-in-nodes/color.md) | 15 |
| [HDR and ACES](built-in-nodes/hdr-aces.md) | 19 |
| [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) | 25 |
| [Pipeline and Studio](built-in-nodes/pipeline.md) | 5 |
| [Review, Viewer, and Preview](built-in-nodes/review.md) | 8 |
| [Upscale](built-in-nodes/upscale.md) | 4 |
| [Video](built-in-nodes/video.md) | 12 |
| [AI Assist](built-in-nodes/ai-assist.md) | 2 |
| **Total** | **109** |

## IO and Delivery

- `RadianceRead`
- `RadianceWrite`
- `RadianceEXRMultiPart`
- `RadianceLoadImageMask`
- `RadianceDigitalCinemaRead`
- `RadianceDigitalCinemaWrite`


## Generate, Loaders, and Sampling

- `RadianceSamplerPro`
- `RadianceHDRVAEDecode`
- `RadianceLoraStack`
- `RadianceUnifiedLoader`
- `RadianceVideoLoader`
- `RadianceHDRLoRALoader`
- `RadianceHDRLoRAApply`
- `RadianceCinematicPromptEncoder`
- `RadianceRegionalPrompt`
- `RadianceRegionalGrid`
- `RadianceResolution`
- `RadianceDenoise`
- `RadianceControlNetApply`


## Color

- `RadianceCDLTransform`
- `RadianceCDLImport`
- `RadianceCDLExport`
- `RadianceWhiteBalance`
- `RadianceColorSpaceConvert`
- `RadianceACESTransform`
- `RadianceHueCurves`
- `RadianceCurves`
- `RadianceGrade`
- `RadianceApplyGradeInfo`
- `RadianceGradeMatch`
- `RadianceOCIOContext`
- `RadianceQC`
- `RadianceLUTApply`
- `RadianceLUTBlend`


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
- `RadianceHDRToneMap`
- `RadianceHDRExpandDynamicRange`
- `RadianceSDRToHDRUniversal`
- `RadianceSDRToHDRRecover`


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
- `RadianceBitDepthDegrade`


## Pipeline and Studio

- `RadianceProjectManager`
- `RadianceBlendComposite`
- `RadianceMCP`
- `RadianceNukeSend`
- `RadianceDaVinciSend`

## Review, Viewer, and Preview

- `RadianceLiteViewer`
- `RadianceViewer`
- `RadianceFocusPeaking`
- `RadianceContactSheet`
- `RadianceFrameStamp`
- `RadiancePolicyGuard`
- `RadianceFlipbookGIF`
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
