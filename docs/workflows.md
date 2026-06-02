# Workflows

These recipes are written for end users who want a working graph pattern first, then tuning.

## HDR EXR Roundtrip

Use this for source plates or generated images where highlights must survive processing.

```mermaid
flowchart LR
    A["◎ Radiance Read"] --> B["◎ HDR Auto Log Select"]
    A --> C["◎ HDR Color Pipeline"]
    B --> C
    C --> D["Generate / VFX / Upscale"]
    D --> E["◎ HDR Diagnostics"]
    D --> F["◎ HDR Monitor"]
    D --> G["◎ Radiance Write"]
```

Recommended settings:

| Stage | Recommendation |
| :--- | :--- |
| Read | Prefer EXR for scene-linear plates. |
| Analyze | Use auto log or diagnostics before committing to a delivery transform. |
| Process | Keep the same metadata or compression assumptions across the graph. |
| Review | Inspect both image and report outputs. |
| Write | Use EXR for master output and PNG/JPEG only for review. |

## ACES Review Transform

Use this when you need a controlled display rendering from an ACES-managed image.

```mermaid
flowchart LR
    A["Scene-linear or ACES image"] --> B["◎ ACES 2.0 Gamut Compress"]
    B --> C["◎ ACES 2.0 Tonescale"]
    C --> D["◎ ACES 2.0 Output Transform"]
    D --> E["◎ Radiance Viewer"]
```

This is a viewing/delivery path. Keep a scene-linear master if you still need to composite or relight.

## VFX Plate Prep

Use this when preparing a shot for masks, depth, motion, or comp.

```mermaid
flowchart LR
    A["◎ Radiance Read"] --> B["◎ Subpixel Plate Stabilizer"]
    B --> C["◎ Depth Map Generator"]
    B --> D["◎ Optical Flow"]
    B --> E["◎ SAM Mask Generator"]
    C --> F["◎ Multipass: VFX Composite"]
    D --> F
    E --> F
```

Tips:

| Task | Useful nodes |
| :--- | :--- |
| Stabilize a plate | `RadianceSubpixelStabilizer` |
| Generate depth | `RadianceDepthMapGenerator` |
| Estimate motion | `RadianceOpticalFlow`, `RadianceMotionBlur` |
| Build masks | `RadianceSAMModelLoader`, `RadianceSAMGenerator`, `RadianceLinearMatting` |
| Composite | `RadianceBlendComposite`, `RadianceMultipassComposite` |

## Multipass Relight

Use this when you have or want AOV-like data for relighting and comp.

```mermaid
flowchart LR
    A["Beauty / supporting images"] --> B["◎ Multipass: Master VFX Extractor"]
    B --> C["◎ Multipass: Real PBR Relight"]
    C --> D["◎ Multipass: VFX Composite"]
    B --> E["◎ Radiance EXR Passes Writer"]
```

Use EXR output for passes. Keep pass names stable if another DCC reads them.

## Video Generation

Use this for text-to-video or image-to-video workflows.

```mermaid
flowchart LR
    A["◎ Video Model Info"] --> B["◎ Video Latent Noise"]
    C["◎ Cinematic Prompt Encoder"] --> D["◎ Video Cond Merge"]
    E["◎ Video HDR Conditioner"] --> D
    B --> F["◎ Video Sampler"]
    D --> F
    F --> G["◎ Video Batch Decode"]
    G --> H["◎ Video Export"]
```

If you want a single high-level node, start with `RadianceT2VPipeline` or `RadianceI2VPipeline`.

## Review and Approval

Use this when the goal is review, not final delivery.

```mermaid
flowchart LR
    A["Image batch"] --> B["◎ Radiance Viewer"]
    A --> C["◎ Contact Sheet"]
    A --> D["◎ Frame Stamp"]
    D --> E["◎ Flipbook GIF"]
    A --> F["◎ Preview Server"]
```

Use contact sheets and GIFs for fast approval. Use EXR or source sequences for final comp review.

## Nuke Handoff

```mermaid
flowchart LR
    A["Final image or sequence"] --> B["◎ Radiance Write"]
    B --> C["◎ Radiance Send to Nuke"]
```

Start the Nuke listener inside Nuke:

```python
exec(open("/path/to/ComfyUI/custom_nodes/radiance/scripts/start_nuke_server.py").read())
```

Use token auth with `RADIANCE_DCC_AUTH_TOKEN` when a studio policy requires it.

## Resolve Handoff

```mermaid
flowchart LR
    A["Final image or sequence"] --> B["◎ Radiance Write"]
    B --> C["◎ Radiance Send to DaVinci Resolve"]
    C --> D["Resolve media folder import"]
```

Resolve support is a folder handoff by default. Live Resolve scripting requires the helper to run inside Resolve Studio.

