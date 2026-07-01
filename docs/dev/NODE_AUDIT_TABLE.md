# Radiance Node Audit Table

Date: 2026-07-01

This report reviews the Radiance node catalog from a ComfyUI and VFX artist perspective.

## Count

| Count | Meaning |
|---:|---|
| 104 | Expected full Radiance node catalog from `tests/node_keys_snapshot.json`. |
| 57 | Nodes loaded in a headless audit import on this machine. The local shell import is missing ComfyUI runtime modules such as `folder_paths` and `aiohttp`, so UI/server and Comfy-dependent nodes are skipped outside a real ComfyUI session. |

## Rating Key

| Rating | Meaning |
|---|---|
| Strong | Core production value, clear VFX/ComfyUI use, worth keeping and polishing. |
| Useful | Good supporting node, workflow helper, or practical artist utility. |
| Niche | Valuable in specific workflows, but not needed by every artist. |
| Weak | Low production value, duplicate behavior, debug-only, or needs stronger reason to exist. |
| Risky | Powerful but needs careful security, dependency, path, or network handling. |

## Full Node Table

| # | Node | Area | Artist explanation | Value |
|---:|---|---|---|---|
| 1 | `RadianceACES2OutputTransformFull` | HDR / Color | Full ACES 2 output transform for scene/display conversion. | Strong |
| 2 | `RadianceACES2ReachGamutCompress` | HDR / Color | Gamut compression to control out-of-gamut saturated colors. | Strong |
| 3 | `RadianceACES2Tonescale` | HDR / Color | ACES-style tonescale for controlled highlight rolloff. | Strong |
| 4 | `RadianceACESTransform` | HDR / Color | General ACES transform utility for color-managed workflows. | Strong |
| 5 | `RadianceAnamorphicStreaks` | Film / VFX | Adds horizontal lens streaks and bloom-style flare accents. | Useful |
| 6 | `RadianceApplyGradeInfo` | Color | Applies a saved or passed grade description to images. | Useful |
| 7 | `RadianceAudioCut` | Video / Editorial | Uses audio timing/cuts as workflow metadata. | Niche |
| 8 | `RadianceBitDepthDegrade` | Lookdev / QC | Simulates lower bit depth, banding, and delivery damage. | Niche |
| 9 | `RadianceBlendComposite` | Composite | Blends foreground/background with artist controls. | Useful |
| 10 | `RadianceCDLExport` | Color | Exports ASC CDL-style grades for handoff. | Strong |
| 11 | `RadianceCDLImport` | Color | Imports CDL values from grading pipelines. | Strong |
| 12 | `RadianceCDLTransform` | Color | Applies slope/offset/power/saturation transforms. | Strong |
| 13 | `RadianceChromaticAberration` | Film / VFX | Adds lens color fringing. Best used lightly. | Useful |
| 14 | `RadianceCinemaStudio` | Workflow | Higher-level studio/workflow wrapper. Needs clear UX to shine. | Niche |
| 15 | `RadianceCinematicPromptEncoder` | Generate | Builds more cinematic prompt conditioning. | Useful |
| 16 | `RadianceClipDetector` | HDR / QC | Finds clipped highlights for SDR-to-HDR repair. | Strong |
| 17 | `RadianceColorSpaceConvert` | Color | Converts between common color spaces. | Strong |
| 18 | `RadianceContactSheet` | Review | Creates contact sheets for comparison and client review. | Useful |
| 19 | `RadianceControlNetApply` | Generate | Applies ControlNet-style conditioning. May duplicate common Comfy nodes. | Weak |
| 20 | `RadianceCurves` | Color | Curve adjustment for contrast and channel shaping. | Strong |
| 21 | `RadianceDaVinciSend` | Pipeline | Sends frames/grades to DaVinci-oriented workflows. | Risky |
| 22 | `RadianceDenoise` | Generate / Cleanup | 32-bit-aware denoise for image cleanup. | Useful |
| 23 | `RadianceDepthMapGenerator` | VFX | Generates depth maps for relight, haze, lens, or comp tasks. | Strong |
| 24 | `RadianceEXRMultiPart` | IO / EXR | Multi-part EXR handling for production passes. | Strong |
| 25 | `RadianceEXRPassesWriter` | IO / EXR | Writes AOV/pass EXRs for Nuke and comp handoff. | Strong |
| 26 | `RadianceFilmGrain` | Film / VFX | Adds controlled film grain. Important for final integration. | Strong |
| 27 | `RadianceFlipbookGIF` | Review | Quick GIF flipbooks for preview. Limited for high-end review. | Weak |
| 28 | `RadianceFocusPeaking` | QC / Display | Visualizes sharpness/focus areas. | Useful |
| 29 | `RadianceFrameStamp` | Review / Editorial | Adds frame/time/version metadata stamps. | Useful |
| 30 | `RadianceGrade` | Color | Primary color grade controls. | Strong |
| 31 | `RadianceGradeMatch` | Color | Matches source image to reference look. | Useful |
| 32 | `RadianceHDRAutoLogSelect` | HDR | Chooses suitable HDR/log handling automatically. | Useful |
| 33 | `RadianceHDRColorPipeline` | HDR / Color | Central HDR color management pipeline. | Strong |
| 34 | `RadianceHDRCrop` | HDR / Utility | HDR-safe crop/framing utility. | Useful |
| 35 | `RadianceHDRDiagnostics` | HDR / QC | Reports HDR range, clipping, and technical health. | Strong |
| 36 | `RadianceHDREncode` | HDR / Generate | Encodes images into HDR latent/representation paths. | Strong |
| 37 | `RadianceHDRGrainMatcher` | HDR / Film | Matches grain across HDR/SDR material. | Useful |
| 38 | `RadianceHDRHighlightComposite` | HDR / Composite | Composites reconstructed highlight regions. | Strong |
| 39 | `RadianceHDRLatentEncoder` | HDR / Generate | Encodes HDR imagery into latent space. | Strong |
| 40 | `RadianceHDRLoRAApply` | Generate / HDR | Applies HDR LoRA deltas to a model. | Niche |
| 41 | `RadianceHDRLoRALoader` | Generate / HDR | Loads HDR LoRA weights. | Niche |
| 42 | `RadianceHDRMonitor` | Display / QC | HDR monitor/scopes style output checking. | Strong |
| 43 | `RadianceHDRStitch` | HDR / Panorama | Stitches HDR views or tiles. | Useful |
| 44 | `RadianceHDRSynthesisEngine` | HDR / Generate | Higher-level HDR synthesis engine. | Strong |
| 45 | `RadianceHDRVAEDecode` | HDR / Generate | Decodes HDR latents to image. | Strong |
| 46 | `RadianceHueCurves` | Color | Hue-based curve controls for targeted color shaping. | Useful |
| 47 | `RadianceI2VPipeline` | Video / Generate | Image-to-video pipeline wrapper. | Strong |
| 48 | `RadianceLensDistortion` | Film / VFX | Adds or corrects lens distortion. | Strong |
| 49 | `RadianceLinearMatting` | VFX / Masking | Extracts mattes using linear image assumptions. | Strong |
| 50 | `RadianceLiteViewer` | Display / Review | Lightweight viewer node for quick inspection. | Useful |
| 51 | `RadianceLoadImageMask` | IO / Masking | Loads image and mask pairs. | Useful |
| 52 | `RadianceLoraStack` | Generate | Combines multiple LoRAs in one workflow. | Useful |
| 53 | `RadianceMCP` | Pipeline / Control | External control bridge. Powerful but security-sensitive. | Risky |
| 54 | `RadianceMotionBlur` | Film / VFX | Adds motion blur for integration and temporal realism. | Strong |
| 55 | `RadianceMultiMaskVisualPicker` | VFX / Masking | Visual tool for choosing masks. | Useful |
| 56 | `RadianceMultipassAOVReader` | IO / EXR | Reads render passes/AOVs for comp workflows. | Strong |
| 57 | `RadianceMultipassComposite` | Composite | Combines passes into a final image. | Strong |
| 58 | `RadianceMultipassMaster` | Composite | Master multipass comp workflow node. | Strong |
| 59 | `RadianceMultipassRelight` | VFX / Relight | Relights rendered passes. High artist value. | Strong |
| 60 | `RadianceNukeSend` | Pipeline | Sends material to Nuke pipeline tooling. | Risky |
| 61 | `RadianceOCIOContext` | Color | Provides OCIO config/context awareness. | Strong |
| 62 | `RadianceOpticalFlow` | Video / VFX | Calculates motion vectors for retime, masks, and stabilization. | Strong |
| 63 | `RadianceParamHistoryTracker` | Debug / Utility | Tracks parameter history. More dev/debug than artist-facing. | Weak |
| 64 | `RadiancePolicyGuard` | Pipeline / Safety | Enforces policy/compliance rules. Studio-only value. | Niche |
| 65 | `RadiancePreviewServer` | Review / Server | Local preview daemon. Useful, but viewer nodes are cleaner. | Niche |
| 66 | `RadianceProjectManager` | Pipeline | Manages project paths/state. Good if hardened. | Useful |
| 67 | `RadianceQC` | QC | General quality-control and diagnostic node. | Strong |
| 68 | `RadianceRead` | IO | Reads production images/sequences. Core pipeline node. | Strong |
| 69 | `RadianceRegionalGrid` | Generate | Builds regional layout/grid conditioning. | Useful |
| 70 | `RadianceRegionalPrompt` | Generate | Assigns prompts to image regions. | Useful |
| 71 | `RadianceRelightEngine` | VFX / Relight | AI/vision relighting engine. High creative value. | Strong |
| 72 | `RadianceResolution` | Generate / Utility | Resolution helper for aspect ratios and targets. | Useful |
| 73 | `RadianceSAMGenerator` | VFX / Masking | Generates masks with SAM. Excellent when model is installed. | Strong |
| 74 | `RadianceSAMModelLoader` | VFX / Masking | Loads SAM model for segmentation nodes. | Useful |
| 75 | `RadianceSDRToHDRPrepare` | HDR | Prepares SDR material for HDR expansion. | Strong |
| 76 | `RadianceSDRtoHDRExpand` | HDR | Expands SDR dynamic range into HDR. | Strong |
| 77 | `RadianceSamplerPro` | Generate | Advanced sampler wrapper. Strong if it adds real workflow value. | Useful |
| 78 | `RadianceSceneCutDetect` | Video / Editorial | Detects scene cuts in video. | Strong |
| 79 | `RadianceSceneCutSplit` | Video / Editorial | Splits video by detected scene cuts. | Useful |
| 80 | `RadianceSubpixelStabilizer` | Video / VFX | Stabilizes tiny motion offsets. | Strong |
| 81 | `RadianceT2VPipeline` | Video / Generate | Text-to-video pipeline wrapper. | Strong |
| 82 | `RadianceTemporalStitchStabilizer` | Video / VFX | Stabilizes temporal seams/flicker across clips. | Strong |
| 83 | `RadianceUnifiedLoader` | Generate | Central model/workflow loader. | Strong |
| 84 | `RadianceUpscaleFaceRestore` | Upscale | Face restoration for upscaled outputs. | Useful |
| 85 | `RadianceUpscaleImage` | Upscale | Main still-image upscale node. | Strong |
| 86 | `RadianceUpscaleTiler` | Upscale | Tile preparation/control for large images. | Strong |
| 87 | `RadianceUpscaleVideo` | Upscale / Video | Video upscale workflow. | Strong |
| 88 | `RadianceVectorMaskDraw` | VFX / Masking | Draws/vectorizes masks for art direction. | Useful |
| 89 | `RadianceVideoAssembler` | Video / IO | Assembles frames into video output. | Strong |
| 90 | `RadianceVideoBatchDecode` | Video / Generate | Batch decodes video frames/latents. | Useful |
| 91 | `RadianceVideoCondMerge` | Video / Generate | Merges video conditioning streams. | Useful |
| 92 | `RadianceVideoExport` | Video / IO | Exports generated video. | Strong |
| 93 | `RadianceVideoFrameRouter` | Video / Utility | Routes frames through different paths. | Useful |
| 94 | `RadianceVideoHDRConditioner` | Video / HDR | Conditions video for HDR-aware generation. | Strong |
| 95 | `RadianceVideoHDRDecode` | Video / HDR | Decodes HDR video representations. | Strong |
| 96 | `RadianceVideoLatentNoise` | Video / Generate | Creates video latent noise. | Useful |
| 97 | `RadianceVideoLoader` | Video / IO | Loads videos/frame sequences. | Strong |
| 98 | `RadianceVideoMaskPropagator` | Video / Masking | Propagates masks over time. Very useful for roto-like workflows. | Strong |
| 99 | `RadianceVideoModelInfo` | Video / Utility | Reports video model capabilities/settings. | Useful |
| 100 | `RadianceVideoSampler` | Video / Generate | Video sampling node. | Strong |
| 101 | `RadianceViewer` | Display / Review | Main viewer UI for inspecting outputs. | Strong |
| 102 | `RadianceVignette` | Film / VFX | Adds lens vignette. Simple but artist-useful. | Useful |
| 103 | `RadianceWhiteBalance` | Color | Temperature/tint balancing. | Strong |
| 104 | `RadianceWrite` | IO | Writes production images/sequences. Core pipeline node. | Strong |

## Strongest Areas

| Area | Why it is strong |
|---|---|
| HDR and ACES pipeline | This is the plugin identity: HDR encode/decode, ACES transforms, SDR-to-HDR tools, diagnostics, and monitor nodes make Radiance more than a generic ComfyUI pack. |
| VFX comp and multipass | AOV reader, multipass composite/master/relight, EXR writing, lens distortion, motion blur, optical flow, matting, and mask propagation are real VFX production concepts. |
| Review and QC | Viewer, Lite Viewer, HDR Monitor, QC, focus peaking, contact sheets, and frame stamps help artists judge output instead of only generating it. |
| Video pipeline | T2V/I2V, video sampler, HDR conditioning/decode, loader/export, scene cuts, and stabilization form a coherent video workflow. |
| Upscale | Image/video upscale plus tiling and face restore are practical and easy for artists to understand. |

## Weak Or Low-Value Nodes

| Node | Why it is weak or less useful | Recommendation |
|---|---|---|
| `RadianceParamHistoryTracker` | Mostly developer/debug value; not a natural artist node. | Hide behind debug category or remove from normal menus. |
| `RadianceFlipbookGIF` | GIF preview is convenient but weak for HDR, color accuracy, and serious review. | Keep as quick-share utility, not core selling point. |
| `RadianceControlNetApply` | Likely duplicates established ComfyUI ControlNet nodes. | Keep only if Radiance adds HDR/video-specific behavior. |
| `RadiancePreviewServer` | A server node is heavier than viewer nodes and needs maintenance/security care. | Keep local-only and secondary to `RadianceViewer`. |
| `RadianceAudioCut` | Useful only for video/editorial cases and not central to image/VFX workflows. | Keep as niche video utility. |
| `RadianceBitDepthDegrade` | Mostly testing/lookdev; not a daily finishing tool. | Keep under QC/lookdev, not primary. |
| `RadiancePolicyGuard` | Studio compliance value, but weak for independent artists. | Make optional/studio-profile only. |
| `RadianceCinemaStudio` | Sounds broad; value depends on whether it is polished and clearly better than smaller nodes. | Simplify or document the exact workflow it owns. |

## Risky But Valuable Nodes

| Node | Risk |
|---|---|
| `RadianceMCP` | External control surfaces can become security-sensitive. Needs authentication, path restrictions, and clear local-only defaults. |
| `RadianceNukeSend` | Pipeline bridge can touch files/processes outside ComfyUI. Needs safe paths and auth/token behavior if networked. |
| `RadianceDaVinciSend` | Same class of risk as Nuke handoff: useful in studio workflows, but should be explicit and local-first. |
| `RadianceProjectManager` | Project/path managers can become file-system risk points. Needs strict path validation. |
| `RadiancePreviewServer` | Server behavior must stay `127.0.0.1` by default and avoid broad LAN exposure. |

## Overall Verdict

Radiance is strongest when it behaves like a ComfyUI-native VFX finishing toolkit: HDR color science, ACES/OCIO, EXR/AOV handling, relight, mask propagation, video HDR, upscale, and review/QC.

The weakest parts are broad helper nodes that do not clearly belong to the HDR/VFX mission, duplicated generation wrappers, and server/control bridge features that increase maintenance or security cost.
