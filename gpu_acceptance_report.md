# Radiance GPU Acceptance Report

## Environment

- torch 2.12.1+cu130, CUDA available: True
- NVIDIA GeForce RTX 4080 SUPER, 17.2 GB VRAM, capability 8.9

## Node sweep (CUDA tensors)

- 97 nodes registered
- SLOW RadianceUpscaleImage: 7441 ms, peak 5004 MB
- executed OK: 71   env-skipped: 24   FAILED: 2
  - **FAIL** RadianceUnifiedLoader: ValueError: ❌ No CLIP encoders provided for architecture 'wan'. Fill the required slot(s): t5xxl
  - **FAIL** RadianceVideoLoader: ValueError: ❌ No CLIP encoders provided for architecture 'wan'. Fill the required slot(s): t5xxl

## RUDRA model scoring

- checkpoint dir: D:\A.I\ComfyUI\models\radiance
- found: rudra_full_decoder_flux_ema.safetensors, rudra_full_decoder_ltx-video_ema.safetensors, rudra_full_decoder_sdxl_ema.safetensors, rudra_full_decoder_wan_ema.safetensors, rudra_full_decoder_zimage_ema.safetensors, rudra_turbo_decoder_flux2-klein_ema.safetensors, rudra_turbo_decoder_flux2_ema.safetensors, rudra_turbo_decoder_flux_ema.safetensors, rudra_turbo_decoder_ltx-video_ema.safetensors, rudra_turbo_decoder_ltx_ema.safetensors, rudra_turbo_decoder_qwen_ema.safetensors, rudra_turbo_decoder_sdxl_ema.safetensors, rudra_turbo_decoder_wan_ema.safetensors, rudra_turbo_decoder_zimage_ema.safetensors
- turbo/flux: resolve signature differs; inspect resolve_rudra_model_type() and adjust
- turbo/ltx: resolve signature differs; inspect resolve_rudra_model_type() and adjust
- full/flux: resolve signature differs; inspect resolve_rudra_model_type() and adjust
- full/ltx: resolve signature differs; inspect resolve_rudra_model_type() and adjust

## TemporalRUDRA static-scene flicker

- no trained temporal checkpoint loaded — flicker UNVERIFIED
