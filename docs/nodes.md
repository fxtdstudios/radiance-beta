[← Back to Radiance docs](README.md)

# Radiance Built-in Nodes

Radiance nodes are installed under `FXTD STUDIOS/Radiance` in ComfyUI. This reference follows the same style as the ComfyUI built-in node documentation: start with the node group, then open the page for the node family you are using.

Each group page includes the node purpose, when to use it, source category, input sockets, output sockets, practical wiring notes, and common gotchas.

## Node groups

| Section | Nodes | Start here when you need to... |
| :--- | ---: | :--- |
| [IO and Delivery](built-in-nodes/io-delivery.md) | 4 | Load, inspect, save, and package production media. These nodes are the safest entry and exit points for EXR, sequences, masks, and delivery files. |
| [Generate, Loaders, and Sampling](built-in-nodes/generate.md) | 13 | Model loading, LoRA stacks, prompt conditioning, resolution setup, denoising, and sampling controls for Radiance generation workflows. |
| [Color](built-in-nodes/color.md) | 15 | Primary grading, CDL exchange, curves, white balance, color-space conversion, OCIO context, and QC policy checks. |
| [HDR and ACES](built-in-nodes/hdr-aces.md) | 15 | HDR analysis, tone mapping, ACES 2.0 transforms, SDR-to-HDR preparation, highlight recovery, relighting, and HDR latent support. |
| [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) | 23 | Depth, optical flow, motion blur, lens effects, SAM masking, matting, inpaint crop/stitch, roto, video mask propagation, multipass extraction, relight, composite, and EXR pass writing. |
| [Pipeline and Studio](built-in-nodes/pipeline.md) | 8 | Project containers, audio cut data, blend composites, cinema prompt setup, local MCP bridge, Nuke send, Resolve handoff, and parameter history tracking. |
| [Review, Viewer, and Preview](built-in-nodes/review.md) | 7 | Interactive viewer, lightweight viewer, focus peaking, contact sheets, flipbook GIFs, frame stamps, and local preview server outputs. |
| [Upscale](built-in-nodes/upscale.md) | 4 | Image and video upscaling, tiling, confidence outputs, and face restoration for high-resolution finishing workflows. |
| [Video](built-in-nodes/video.md) | 12 | Video model inspection, latent noise, conditioning merge, sampling, T2V/I2V pipelines, batch decode, HDR video decode, frame routing, assembly, and export. |
| [AI Assist](built-in-nodes/ai-assist.md) | 2 | Scene cut detection and shot splitting for per-shot processing and grade routing. |

## Recommended reading order

1. [Quickstart](quickstart.md) for installation and the first graph.
2. [Concepts](concepts.md) for HDR, EXR, ACES/OCIO, masks, video batches, and DCC handoff.
3. The node group page that matches your graph area.
4. [Workflows](workflows.md) for end-to-end recipes.
5. [Troubleshooting](troubleshooting.md) when output is missing, clipped, wrong color, or not reaching a DCC.

## Catalog coverage

This reference covers **103 registered nodes** from the grouped Radiance catalog. User-generated `.gizmo` nodes are dynamic and should be documented with the studio workflow that creates them.

## Fast lookup

| Node | Section |
| :--- | :--- |
| `◎ Radiance Read` / `RadianceRead` | [IO and Delivery](built-in-nodes/io-delivery.md) |
| `◎ Radiance Write` / `RadianceWrite` | [IO and Delivery](built-in-nodes/io-delivery.md) |
| `◎ Radiance EXR Multi-Part` / `RadianceEXRMultiPart` | [IO and Delivery](built-in-nodes/io-delivery.md) |
| `◎ Radiance Load Image Mask` / `RadianceLoadImageMask` | [IO and Delivery](built-in-nodes/io-delivery.md) |
| `◎ Radiance Sampler Pro` / `RadianceSamplerPro` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ HDR VAE Decode` / `RadianceHDRVAEDecode` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ LoRA Stack` / `RadianceLoraStack` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ Radiance Read Models` / `RadianceUnifiedLoader` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ Video Loader` / `RadianceVideoLoader` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ ControlNet Apply` / `RadianceControlNetApply` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ HDR LoRA Loader` / `RadianceHDRLoRALoader` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ HDR LoRA Apply` / `RadianceHDRLoRAApply` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ Cinematic Prompt Encoder` / `RadianceCinematicPromptEncoder` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ Regional Prompt` / `RadianceRegionalPrompt` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ Regional Grid` / `RadianceRegionalGrid` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ Resolution` / `RadianceResolution` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ Denoise` / `RadianceDenoise` | [Generate, Loaders, and Sampling](built-in-nodes/generate.md) |
| `◎ Radiance CDL Transform` / `RadianceCDLTransform` | [Color](built-in-nodes/color.md) |
| `◎ Radiance CDL Import` / `RadianceCDLImport` | [Color](built-in-nodes/color.md) |
| `◎ Radiance CDL Export` / `RadianceCDLExport` | [Color](built-in-nodes/color.md) |
| `◎ Radiance White Balance` / `RadianceWhiteBalance` | [Color](built-in-nodes/color.md) |
| `◎ Radiance Colorspace Convert` / `RadianceColorSpaceConvert` | [Color](built-in-nodes/color.md) |
| `◎ Radiance ACES Transform` / `RadianceACESTransform` | [Color](built-in-nodes/color.md) |
| `◎ Radiance Bit Depth Degrade` / `RadianceBitDepthDegrade` | [Color](built-in-nodes/color.md) |
| `◎ Radiance Hue Curves` / `RadianceHueCurves` | [Color](built-in-nodes/color.md) |
| `◎ Radiance Curves` / `RadianceCurves` | [Color](built-in-nodes/color.md) |
| `◎ Radiance Grade` / `RadianceGrade` | [Color](built-in-nodes/color.md) |
| `◎ Radiance Apply Grade Info` / `RadianceApplyGradeInfo` | [Color](built-in-nodes/color.md) |
| `◎ Radiance Grade Match` / `RadianceGradeMatch` | [Color](built-in-nodes/color.md) |
| `◎ Radiance OCIO Context` / `RadianceOCIOContext` | [Color](built-in-nodes/color.md) |
| `◎ Radiance QC` / `RadianceQC` | [Color](built-in-nodes/color.md) |
| `◎ Policy Guard` / `RadiancePolicyGuard` | [Color](built-in-nodes/color.md) |
| `◎ ACES 2.0 Tonescale` / `RadianceACES2Tonescale` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ ACES 2.0 Gamut Compress` / `RadianceACES2ReachGamutCompress` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ ACES 2.0 Output Transform` / `RadianceACES2OutputTransformFull` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ HDR Color Pipeline` / `RadianceHDRColorPipeline` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ HDR Encode` / `RadianceHDREncode` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ HDR Monitor` / `RadianceHDRMonitor` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ HDR Auto Log Select` / `RadianceHDRAutoLogSelect` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ HDR Diagnostics` / `RadianceHDRDiagnostics` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ Clip Detector` / `RadianceClipDetector` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ SDR to HDR Prepare` / `RadianceSDRToHDRPrepare` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ HDR Highlight Composite` / `RadianceHDRHighlightComposite` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ SDR to HDR Expand` / `RadianceSDRtoHDRExpand` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ HDR Synthesis Engine` / `RadianceHDRSynthesisEngine` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ Relight Engine` / `RadianceRelightEngine` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ HDR Latent Encoder` / `RadianceHDRLatentEncoder` | [HDR and ACES](built-in-nodes/hdr-aces.md) |
| `◎ Depth Map Generator` / `RadianceDepthMapGenerator` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Optical Flow` / `RadianceOpticalFlow` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Physical Motion Blur` / `RadianceMotionBlur` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Lens Distortion` / `RadianceLensDistortion` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Chromatic Aberration` / `RadianceChromaticAberration` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Anamorphic Streaks` / `RadianceAnamorphicStreaks` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Film Grain (Simple)` / `RadianceFilmGrain` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Vignette` / `RadianceVignette` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ SAM Model Loader` / `RadianceSAMModelLoader` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ SAM Mask Generator` / `RadianceSAMGenerator` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ SAM Multi-Mask Picker` / `RadianceMultiMaskVisualPicker` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Linear Alpha Matting` / `RadianceLinearMatting` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ HDR Grain Matcher` / `RadianceHDRGrainMatcher` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Subpixel Plate Stabilizer` / `RadianceSubpixelStabilizer` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ HDR Inpaint Crop` / `RadianceHDRCrop` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ HDR Inpaint Stitch` / `RadianceHDRStitch` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Temporal Stitch Stabilizer` / `RadianceTemporalStitchStabilizer` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Vector Mask Draw (Roto)` / `RadianceVectorMaskDraw` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Video Mask Propagator` / `RadianceVideoMaskPropagator` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Multipass: Master VFX Extractor` / `RadianceMultipassMaster` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Radiance EXR Passes Writer` / `RadianceEXRPassesWriter` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Multipass: Real PBR Relight` / `RadianceMultipassRelight` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Multipass: VFX Composite` / `RadianceMultipassComposite` | [VFX, Masks, Optics, and Multipass](built-in-nodes/vfx.md) |
| `◎ Audio Cut` / `RadianceAudioCut` | [Pipeline and Studio](built-in-nodes/pipeline.md) |
| `◎ Project Manager` / `RadianceProjectManager` | [Pipeline and Studio](built-in-nodes/pipeline.md) |
| `◎ Blend Composite` / `RadianceBlendComposite` | [Pipeline and Studio](built-in-nodes/pipeline.md) |
| `◎ Cinema Studio` / `RadianceCinemaStudio` | [Pipeline and Studio](built-in-nodes/pipeline.md) |
| `◎ Radiance MCP Bridge` / `RadianceMCP` | [Pipeline and Studio](built-in-nodes/pipeline.md) |
| `◎ Radiance Send to Nuke` / `RadianceNukeSend` | [Pipeline and Studio](built-in-nodes/pipeline.md) |
| `◎ Radiance Send to DaVinci Resolve` / `RadianceDaVinciSend` | [Pipeline and Studio](built-in-nodes/pipeline.md) |
| `◎ Parameter History Tracker` / `RadianceParamHistoryTracker` | [Pipeline and Studio](built-in-nodes/pipeline.md) |
| `◎ Radiance Lite Viewer` / `RadianceLiteViewer` | [Review, Viewer, and Preview](built-in-nodes/review.md) |
| `◎ Radiance Viewer` / `RadianceViewer` | [Review, Viewer, and Preview](built-in-nodes/review.md) |
| `◎ Focus Peaking` / `RadianceFocusPeaking` | [Review, Viewer, and Preview](built-in-nodes/review.md) |
| `◎ Contact Sheet` / `RadianceContactSheet` | [Review, Viewer, and Preview](built-in-nodes/review.md) |
| `◎ Flipbook GIF` / `RadianceFlipbookGIF` | [Review, Viewer, and Preview](built-in-nodes/review.md) |
| `◎ Frame Stamp` / `RadianceFrameStamp` | [Review, Viewer, and Preview](built-in-nodes/review.md) |
| `◎ Preview Server` / `RadiancePreviewServer` | [Review, Viewer, and Preview](built-in-nodes/review.md) |
| `◎ Upscale Tiler` / `RadianceUpscaleTiler` | [Upscale](built-in-nodes/upscale.md) |
| `◎ Upscale Image` / `RadianceUpscaleImage` | [Upscale](built-in-nodes/upscale.md) |
| `◎ Upscale Video` / `RadianceUpscaleVideo` | [Upscale](built-in-nodes/upscale.md) |
| `◎ Upscale Face Restore` / `RadianceUpscaleFaceRestore` | [Upscale](built-in-nodes/upscale.md) |
| `◎ Video Model Info` / `RadianceVideoModelInfo` | [Video](built-in-nodes/video.md) |
| `◎ Video Latent Noise` / `RadianceVideoLatentNoise` | [Video](built-in-nodes/video.md) |
| `◎ Video Cond Merge` / `RadianceVideoCondMerge` | [Video](built-in-nodes/video.md) |
| `◎ Video Sampler` / `RadianceVideoSampler` | [Video](built-in-nodes/video.md) |
| `◎ T2V Pipeline` / `RadianceT2VPipeline` | [Video](built-in-nodes/video.md) |
| `◎ I2V Pipeline` / `RadianceI2VPipeline` | [Video](built-in-nodes/video.md) |
| `◎ Video Batch Decode` / `RadianceVideoBatchDecode` | [Video](built-in-nodes/video.md) |
| `◎ Video Export` / `RadianceVideoExport` | [Video](built-in-nodes/video.md) |
| `◎ Video HDR Conditioner` / `RadianceVideoHDRConditioner` | [Video](built-in-nodes/video.md) |
| `◎ Video HDR Decode` / `RadianceVideoHDRDecode` | [Video](built-in-nodes/video.md) |
| `◎ Video Frame Router` / `RadianceVideoFrameRouter` | [Video](built-in-nodes/video.md) |
| `◎ Video Assembler` / `RadianceVideoAssembler` | [Video](built-in-nodes/video.md) |
| `◎ Scene Cut Detect` / `RadianceSceneCutDetect` | [AI Assist](built-in-nodes/ai-assist.md) |
| `◎ Scene Cut Split` / `RadianceSceneCutSplit` | [AI Assist](built-in-nodes/ai-assist.md) |
