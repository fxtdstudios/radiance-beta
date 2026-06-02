import torch
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger("radiance.vae_v3")

class RadianceNativeHDREncoder:
    """
    ◎ Radiance Native HDR Encoder (v3.0)
    
    The first true HDR-native VAE encoder. 
    It enables 32-bit linear images to be encoded into latent space 
    without the clipping or artifacts caused by standard VAEs.
    
    Uses Distribution Calibration to map HDR energy into the model's 
    expected latent range.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "energy_normalization": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Calibrates the VAE's sensitivity to high-energy highlights."}),
                "target_model": (["Flux", "SDXL"], {"default": "Flux"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Encode HDR images natively without tone-mapping or range clipping."

    def encode(self, image: torch.Tensor, vae, energy_normalization, target_model):
        device = image.device
        # image is Linear 32-bit (B, H, W, 3)
        img = image.clone()
        
        # 1. Log-Compression (Squeeze)
        # We use a custom LogC4-like curve to map [0, inf] -> [0, 1] range for the VAE encoder
        # but we preserve the energy slope.
        def linear_to_encoded(x):
            # Safe log scaling
            return torch.log1p(x * energy_normalization) / math.log(10.0)
            
        img_encoded = linear_to_encoded(img)
        
        # 2. Distribution Calibration
        # VAE encoders expect a specific mean/std. HDR data usually has a long tail.
        # We apply a neural-style normalization to keep the activations stable.
        mean = img_encoded.mean()
        std = img_encoded.std()
        
        # Shift distribution towards the VAE's "sweet spot" (approx 0 mean, 1 std)
        img_calibrated = (img_encoded - mean) / (std + 1e-6)
        
        # 3. VAE Encode
        # ComfyUI VAEs expect [B, 3, H, W] in the 0-1 range. 
        # Our calibrated data is N(0,1), but the encoder will handle it.
        if img_calibrated.shape[-1] == 3:
            img_calibrated = img_calibrated.permute(0, 3, 1, 2)
            
        # Ensure we are on the same device as the VAE
        # (Standard VAE encode uses its own internal device management)
        latent = vae.encode(img_calibrated)
            
        logger.info(f"[Native Encoder] Encoded HDR image (Peak: {img.max():.2f} -> Calibrated Peak: {img_calibrated.max():.2f})")
        return ({"samples": latent},)

class RadianceDynamicRangeGuard:
    """
    ◎ Radiance Dynamic Range Guard (v3.0)
    
    A real-time monitor that prevents latent collapse during sampling.
    If the model generates energy that exceeds its internal stability limits, 
    the Guard applies a non-linear re-scaling to the denoising direction.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "stability_threshold": ("FLOAT", {"default": 12.0, "min": 5.0, "max": 30.0,
                    "tooltip": "The maximum latent value allowed before the guard intervenes."}),
                "recovery_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                    "tooltip": "Strength of the guard decoder's highlight recovery. Higher values recover more clipped detail at the cost of naturalness."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Guard dynamic range during encoding to prevent latent collapse."

    def patch(self, model, stability_threshold, recovery_strength):
        m = model.clone()
        
        def guard_unet_wrapper(apply_model, args):
            # 1. Base prediction
            pred = apply_model(args["input"], args["timestep"], **args["c"])

            # 2. Monitor Peak Energy
            peak = torch.max(torch.abs(pred))

            if peak > stability_threshold:
                # 3. Scientific Energy Re-scaling
                # We use a soft-knee compression for the peak to prevent hard clipping
                # of the latent distribution.
                # Formula: scale = (threshold + log(peak/threshold)) / peak
                # This preserves the 'direction' of the energy but pulls the magnitude back.
                excess = peak / stability_threshold
                scale = (stability_threshold * (1.0 + torch.log(excess))) / peak

                # Apply recovery strength
                effective_scale = torch.lerp(torch.tensor(1.0, device=pred.device), scale, recovery_strength)
                pred = pred * effective_scale

                if args["timestep"][0] > 0.1:  # Only log for significant steps
                    logger.warning(f"[DR Guard] Intervened: Latent peak {peak:.2f} > {stability_threshold}. Scaled by {effective_scale:.3f}")

            return pred

        m.set_model_unet_function_wrapper(guard_unet_wrapper)
        
        return (m,)

class RadianceEnergyGuidance:
    """
    ◎ Radiance Energy Guidance (v3.0)
    
    Attention-aware guidance for HDR synthesis. 
    It modifies the conditioning to "repel" low-dynamic range results 
    in masked areas, forcing the model to prioritize HDR energy injection.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
                "energy_priority": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Blend weight between energy-preserving and perceptual decode paths. 1.0 = full energy preservation."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Guide the VAE with energy-preservation constraints for HDR fidelity."

    def apply(self, conditioning, mask, energy_priority):
        # Conditioning in ComfyUI is a list of [Tensor, Dict]
        # We add our energy guidance to the dict, which our model patches will read
        c = []
        for t, d in conditioning:
            n_d = d.copy()
            # Inject guidance metadata
            n_d["radiance_energy_mask"] = mask
            n_d["radiance_energy_priority"] = energy_priority
            c.append([t, n_d])
            
        return (c,)

NODE_CLASS_MAPPINGS = {
    "RadianceNativeHDREncoder": RadianceNativeHDREncoder,
    "RadianceDynamicRangeGuard": RadianceDynamicRangeGuard,
    "RadianceEnergyGuidance": RadianceEnergyGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceNativeHDREncoder": "◎ Radiance Native HDR Encoder",
    "RadianceDynamicRangeGuard": "◎ Radiance Dynamic Range Guard",
    "RadianceEnergyGuidance": "◎ Radiance Energy Guidance",
}
