# Troubleshooting

Use this page when a Radiance graph loads but the output is wrong, missing, clipped, or hard to interpret.

## Install And Import

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| Radiance appears with very few nodes | Required dependencies are missing. | Install the platform requirement file inside the ComfyUI Python environment. |
| Viewer node does not appear | Viewer dependencies failed to import. | Check `numpy`, `Pillow`, `aiohttp`, and ComfyUI startup logs. |
| EXR save fails | `OpenEXR` or `Imath` is missing. | Install the matching requirements file and restart ComfyUI. |
| Optional model node fails | Optional ML dependency or model weights are missing. | Install the dependency named in the node/log output and confirm model paths. |

## Color And HDR

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| Highlights are flat | HDR data was clipped or encoded/decoded with mismatched settings. | Use `RadianceHDRDiagnostics`, keep HDR metadata connected, and export EXR. |
| Image looks correct in viewer but wrong in file | Display transform was mistaken for master data. | Save the intended color space explicitly and use a known output transform. |
| Colors shift after Nuke/Resolve import | Source, working, and display spaces do not match. | Use `RadianceOCIOContext`, `RadianceColorSpaceConvert`, or ACES nodes consistently. |
| Banding appears | Output bit depth or delivery format is too low. | Save float EXR/TIFF for masters; use `RadianceBitDepthDegrade` to diagnose. |
| Gamut clips harshly | Wide-gamut values were hard clipped. | Try `RadianceACES2ReachGamutCompress` before output transform. |

## Masks, VFX, And Inpaint

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| Mask is inverted | Alpha convention differs between nodes. | Preview the mask, then invert or adjust before crop/composite. |
| Inpaint crop does not stitch cleanly | Crop/stitch data was not kept from the same node run. | Keep `STITCHER_DATA` from `RadianceHDRCrop` connected to the matching stitch step. |
| Video mask flickers | Per-frame masks are inconsistent. | Use `RadianceTemporalStitchStabilizer` or `RadianceVideoMaskPropagator`. |
| Multipass writer errors | Beauty pass is missing from the pass dictionary. | Feed `RadianceEXRPassesWriter` from `RadianceMultipassMaster`. |

## Video

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| Latent shape mismatch | Noise dimensions do not match model config. | Start with `RadianceVideoModelInfo`, then feed config into `RadianceVideoLatentNoise`. |
| Frame count is wrong | Router/assembler or decode path dropped frames. | Check `frame_count`, `frames_accumulated`, and `is_complete` outputs. |
| Video export looks SDR | HDR decode or monitor path was used only for preview. | Use `RadianceVideoHDRDecode` and write the HDR output to a capable format. |

## DCC Handoff

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| Nuke send cannot connect | Listener is not running or host/port does not match. | Run `scripts/start_nuke_server.py` inside Nuke and confirm localhost/port settings. |
| Nuke rejects actions | Token or dev-mode settings differ. | Match `RADIANCE_DCC_AUTH_TOKEN`; enable dev-only behavior only when intentionally testing. |
| Resolve does not receive media live | Resolve path is folder handoff by default. | Export to the configured folder and import manually, or run the Resolve helper inside Resolve Studio. |

## Debug Checklist

1. Confirm the node appears under `FXTD STUDIOS/Radiance`.
2. Confirm all required dependencies import in the same Python environment ComfyUI uses.
3. Add `RadianceQC` or `RadianceHDRDiagnostics` near the failure point.
4. Preview both image and mask outputs.
5. Save a small EXR test frame before running a full sequence.
6. Check ComfyUI console logs for the exact node key and exception.

