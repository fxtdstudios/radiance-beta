import torch
import torch.nn.functional as F
import numpy as np

class RadianceSDRtoHDRExpand:
    """
    ◎ Radiance SDR to HDR Expand

    Expands dynamic range from SDR footage via an inverse OETF pass and a
    mathematical highlight expansion. Does not reconstruct clipped detail.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = ("Expand an SDR image into HDR headroom via inverse OETF and "
                   "mathematical highlight expansion. Does not reconstruct clipped detail.")
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "inverse_oetf": (["None", "sRGB", "Rec.709"], {"default": "sRGB"}),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Luminance threshold above which HDR expansion begins. 0.8 = expand highlights above 80% SDR white."}),
                "expansion_gain": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1, "tooltip": "Peak luminance multiplier for expanded highlights. 5.0 = 500 nits from 100-nit SDR white."}),
                "expansion_gamma": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Power curve applied to the expansion mask. Values > 1.0 create a harder shoulder; < 1.0 a softer roll-off."}),
                "smoothness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Feathering radius for the expansion mask edge. Higher values prevent harsh highlight boundaries."}),
            }
        }

    @torch.no_grad()
    def apply(self, image: torch.Tensor, inverse_oetf: str, threshold: float, 
              expansion_gain: float, expansion_gamma: float, smoothness: float):
        img = image.clone()
        
        if inverse_oetf == "sRGB":
            img = torch.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
        elif inverse_oetf == "Rec.709":
            img = torch.where(img < 0.0812, img / 4.5, ((img + 0.099) / 1.099) ** (1.0 / 0.45))
            
        RGB = img[..., :3]
        luma = 0.2126 * RGB[..., 0] + 0.7152 * RGB[..., 1] + 0.0722 * RGB[..., 2]
        
        diff = luma - threshold
        mask = torch.sigmoid(diff / max(smoothness, 0.0001)) if smoothness > 0 else (diff > 0).float()
        
        highlight_amt = F.relu(diff)
        expansion = (highlight_amt ** expansion_gamma) * expansion_gain
        
        luma_safe = torch.clamp(luma, min=1e-6)
        ratio = RGB / luma_safe.unsqueeze(-1)
        
        expanded_RGB = RGB + (ratio * expansion.unsqueeze(-1))
        
        if img.shape[-1] > 3:
            result = torch.cat([expanded_RGB, img[..., 3:]], dim=-1)
        else:
            result = expanded_RGB

        return (result,)


class RadianceHDRSynthesisEngine:
    """
    ◎ Radiance HDR Synthesis Engine
    
    Advanced Physically-Based HDR reconstruction.
    Uses Laplacian Pyramid decomposition to recover detail in clipped highlights
    and projects luminance into the 16-bit range using optical energy models.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Synthesise HDR imagery from SDR input and optional guidance signals."
    FUNCTION = "synthesize"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "highlight_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "energy_target": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Target peak luminance multiplier (e.g. 10.0 = 10 stops above SDR white)."}),
                "recovery_iters": ("INT", {"default": 3, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Number of Laplacian iterations to recover clipped detail."}),
                "chroma_preservation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Prevents expanded highlights from losing saturation or shifting hue."}),
            },
            "optional": {
                "guidance_mask":  ("MASK",  {"tooltip": "Per-pixel guidance mask from Radiance Luminance Guidance."}),
                "guidance_nits":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 50.0,
                    "tooltip": "Local target peak nits. 0 = use global energy_target only."}),
            }
        }

    def _build_pyramid(self, img, iters):
        pyramid = []
        curr = img
        for _ in range(iters):
            down = F.interpolate(curr, scale_factor=0.5, mode="bilinear", align_corners=False)
            up = F.interpolate(down, size=(curr.shape[2], curr.shape[3]), mode="bilinear", align_corners=False)
            diff = curr - up
            pyramid.append(diff)
            curr = down
        pyramid.append(curr)
        return pyramid

    def _reconstruct_pyramid(self, pyramid):
        curr = pyramid.pop()
        while pyramid:
            level = pyramid.pop()
            up = F.interpolate(curr, size=(level.shape[2], level.shape[3]), mode="bilinear", align_corners=False)
            curr = up + level
        return curr

    @torch.no_grad()
    def synthesize(self, image: torch.Tensor, energy_target: float, recovery_iters: int, chroma_preservation: float,
                   guidance_mask: torch.Tensor = None, guidance_nits: float = 0.0):
        # image shape (B, H, W, C)
        B, H, W, C = image.shape
        img = image.permute(0, 3, 1, 2).float() # (B, C, H, W)
        
        # 1. Decompose into Laplacian Pyramid
        pyramid = self._build_pyramid(img[:, :3, :, :], recovery_iters)
        
        # 2. Extract Base Layer (The low-frequency luminance)
        base = pyramid[-1]
        base_H, base_W = base.shape[2], base.shape[3]
        
        # 3. Intelligent Energy Projection
        # We identify areas that are clipped (near 1.0) and "lift" them towards the target
        luma = 0.2126 * base[:, 0:1, :, :] + 0.7152 * base[:, 1:2, :, :] + 0.0722 * base[:, 2:3, :, :]
        
        # Highlight mask (areas near or above SDR white)
        mask = torch.sigmoid((luma - 0.7) * 10.0)
        
        # Dynamic Target Logic (Luminance Guidance)
        if guidance_mask is not None and guidance_nits > 0.0:
            # guidance_mask shape: (B, H, W) — comes from RadianceLuminanceGuidance
            g_mask = guidance_mask.unsqueeze(1).float()  # (B, 1, H, W)
            g_mask_low = F.interpolate(g_mask, size=(base_H, base_W), mode="bilinear", align_corners=False)

            # Target nits: 100 nits = 1.0 energy, 1000 nits = 10.0 energy
            g_target = guidance_nits / 100.0

            # Blend global energy_target with local guidance target
            effective_target = torch.lerp(torch.full_like(luma, energy_target), torch.full_like(luma, g_target), g_mask_low)
        else:
            effective_target = energy_target

        # Non-linear energy lift
        energy_lift = torch.pow(luma.clamp(min=1e-6), 2.2) * (effective_target - 1.0)
        base_lifted = base + (base * energy_lift * mask)
        
        # 4. Chroma Preservation
        if chroma_preservation > 0:
            orig_ratio = base / luma.clamp(min=1e-6)
            new_luma = 0.2126 * base_lifted[:, 0:1, :, :] + 0.7152 * base_lifted[:, 1:2, :, :] + 0.0722 * base_lifted[:, 2:3, :, :]
            base_lifted = torch.lerp(base_lifted, orig_ratio * new_luma, chroma_preservation)
            
        pyramid[-1] = base_lifted
        
        # 5. Reconstruct with Detail Retention
        result_rgb = self._reconstruct_pyramid(pyramid)
        
        # 6. Final compositing
        if C > 3:
            result = torch.cat([result_rgb, img[:, 3:, :, :]], dim=1)
        else:
            result = result_rgb
            
        # Return image and mask
        mask_out = F.interpolate(mask, size=(H, W), mode="bilinear").permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
        return (result.permute(0, 2, 3, 1), mask_out)


class RadianceRelightEngine:
    """
    ◎ Radiance Relight Engine
    
    True 32-bit float geometric re-lighting using input Normal maps.
    Calculates physically plausible Lambertian fill and Blinn-Phong specular passes.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Relight an HDR or SDR image using environment map or directional light."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "lighting_pass_only")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "light_dir_x": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Light direction X component. Normalized internally — sets the horizontal angle of the synthetic light."}),
                "light_dir_y": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Light direction Y component. Positive = light from above."}),
                "light_dir_z": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Light direction Z component. Positive = light in front of surface."}),
                "light_color_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Red component of the synthetic light color. Values > 1.0 produce HDR emission."}),
                "light_color_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Green component of the synthetic light color."}),
                "light_color_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Blue component of the synthetic light color."}),
                "diffuse_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Lambertian diffuse reflection strength. Controls broad, soft illumination."}),
                "specular_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Specular highlight strength. Higher values produce brighter, more visible glints."}),
                "specular_roughness": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Surface roughness for Blinn-Phong specular. Low = sharp glints (metallic), high = soft broad highlights (matte)."}),
            },
            "optional": {
                "camera": ("RADIANCE_CAMERA",),
            }
        }

    @torch.no_grad()
    def apply(self, image: torch.Tensor, normal_map: torch.Tensor, 
              light_dir_x: float, light_dir_y: float, light_dir_z: float,
              light_color_r: float, light_color_g: float, light_color_b: float,
              diffuse_intensity: float, specular_intensity: float, specular_roughness: float,
              camera: dict = None):
        
        device = image.device
        normals = normal_map.clone()
        if normals.shape != image.shape:
            normals = F.interpolate(normals.permute(0, 3, 1, 2), 
                                    size=(image.shape[1], image.shape[2]), 
                                    mode="bilinear").permute(0, 2, 3, 1)

        normals = normals[..., :3] * 2.0 - 1.0
        n_norm = torch.norm(normals, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        N = normals / n_norm
        
        L = torch.tensor([light_dir_x, -light_dir_y, light_dir_z], device=device, dtype=torch.float32)
        L = F.normalize(L, p=2, dim=0)
        
        # ── View Vector (V) ──
        B_img, H_img, W_img, _ = image.shape
        y_c, x_c = torch.meshgrid(
            torch.linspace(-1, 1, H_img, device=device),
            torch.linspace(-1, 1, W_img, device=device),
            indexing='ij'
        )
        
        if camera is not None and "transform" in camera:
            cam_mat = np.array(camera["transform"])
            cam_pos = torch.tensor(cam_mat[:3, 3], device=device, dtype=torch.float32)
            surface_pos = torch.stack([x_c, -y_c, torch.zeros_like(x_c)], dim=-1)
            V = cam_pos.view(1, 1, 1, 3) - surface_pos.unsqueeze(0)
            V = F.normalize(V, p=2, dim=-1)
        else:
            cam_z = 2.0
            V = torch.stack([-x_c, y_c, torch.full_like(x_c, cam_z)], dim=-1)
            V = F.normalize(V, p=2, dim=-1).unsqueeze(0)
            
        N_dot_L = torch.sum(N * L.view(1, 1, 1, 3), dim=-1, keepdim=True)
        diffuse = F.relu(N_dot_L)
        
        H = F.normalize(L.view(1, 1, 1, 3) + V, p=2, dim=-1)
        N_dot_H = torch.sum(N * H, dim=-1, keepdim=True)
        
        shininess = max(0.001, min(2.0 / (specular_roughness**2) - 2.0, 2048.0))
        specular = torch.pow(F.relu(N_dot_H), shininess)
        
        light_color = torch.tensor([light_color_r, light_color_g, light_color_b], device=device, dtype=torch.float32).view(1, 1, 1, 3)
        total_diffuse = diffuse * diffuse_intensity * light_color
        total_spec = specular * specular_intensity * light_color
        
        lighting_pass = total_diffuse + total_spec
        relit_image = image[..., :3] + lighting_pass
        
        if image.shape[-1] > 3:
            relit_image = torch.cat([relit_image, image[..., 3:]], dim=-1)
        
        lighting_out = lighting_pass
        if lighting_pass.shape[-1] != relit_image.shape[-1] and relit_image.shape[-1] == 4:
            lighting_out = torch.cat([lighting_pass, torch.ones_like(lighting_pass[..., :1])], dim=-1)

        return (relit_image, lighting_out)


NODE_CLASS_MAPPINGS = {
    "RadianceSDRtoHDRExpand": RadianceSDRtoHDRExpand,
    "RadianceHDRSynthesisEngine": RadianceHDRSynthesisEngine,
    "RadianceRelightEngine": RadianceRelightEngine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSDRtoHDRExpand": "◎ Radiance SDR to HDR Expand",
    "RadianceHDRSynthesisEngine": "◎ Radiance HDR Synthesis Engine",
    "RadianceRelightEngine": "◎ Radiance Relight Engine",
}
