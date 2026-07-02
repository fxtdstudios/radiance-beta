import json
import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ....performance import perf_finish, perf_start

_NORMAL_INPUTS = ["OpenGL (Y-Up)", "DirectX (Y-Down)"]
_LIGHT_TYPES = ["Directional", "Point"]
logger = logging.getLogger("radiance.vfx.multipass.relight_comp")


def _resize_bhwc(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if x.shape[1] == height and x.shape[2] == width:
        return x
    return F.interpolate(
        x.float().permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)


def _match_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.shape[0] == batch:
        return x
    if x.shape[0] > batch:
        return x[:batch]
    return x[:1].expand(batch, -1, -1, -1)


def _match_image(
    x: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    channels: int = 3,
) -> torch.Tensor:
    out = _match_batch(_resize_bhwc(x.float(), height, width), batch)
    if out.shape[-1] == channels:
        return out.contiguous()
    if channels == 1:
        return out[..., :1].contiguous()
    if out.shape[-1] > channels:
        return out[..., :channels].contiguous()
    if out.shape[-1] == 1:
        return out.expand(-1, -1, -1, channels).contiguous()
    pad = out[..., -1:].expand(-1, -1, -1, channels - out.shape[-1])
    return torch.cat([out, pad], dim=-1).contiguous()


def _scalar_pass(
    x: Optional[torch.Tensor],
    batch: int,
    height: int,
    width: int,
    default: float,
    device: torch.device,
    clamp: bool = True,
) -> torch.Tensor:
    if x is None:
        out = torch.full((batch, height, width), default, device=device, dtype=torch.float32)
    else:
        out = _match_image(x, batch, height, width, channels=1)[..., 0].to(device=device)
    return out.clamp(0.0, 1.0) if clamp else out


def _color_tensor(r: float, g: float, b: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([r, g, b], device=device, dtype=torch.float32).view(1, 1, 1, 3)


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.sqrt((v * v).sum(dim=-1, keepdim=True).clamp(min=eps))


def _decode_normal_map(normal_map: torch.Tensor, convention: str) -> torch.Tensor:
    n = normal_map.float()[..., :3]
    if float(n.detach().min()) >= -0.001 and float(n.detach().max()) <= 1.001:
        n = n * 2.0 - 1.0
    if convention == "DirectX (Y-Down)":
        n = torch.stack([n[..., 0], -n[..., 1], n[..., 2]], dim=-1)
    return _normalize(n)


def _view_positions(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    depth_map: Optional[torch.Tensor],
    depth_scale: float,
) -> torch.Tensor:
    y = torch.linspace(1.0, -1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    aspect = float(width) / max(float(height), 1.0)
    if depth_map is None:
        z = torch.zeros((batch, height, width), device=device, dtype=torch.float32)
    else:
        z = _scalar_pass(depth_map, batch, height, width, 0.5, device, clamp=False)
        z = (z - 0.5) * float(depth_scale)
    return torch.stack(
        [
            xx.view(1, height, width).expand(batch, -1, -1) * aspect,
            yy.view(1, height, width).expand(batch, -1, -1),
            z,
        ],
        dim=-1,
    )


def _blur_bhwc(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    b, h, w, c = x.shape
    radius = min(int(radius), max(0, h - 1), max(0, w - 1))
    if radius <= 0:
        return x
    k = radius * 2 + 1
    x4 = x.permute(0, 3, 1, 2).reshape(b * c, 1, h, w)
    x4 = F.pad(x4, (radius, radius, radius, radius), mode="reflect")
    x4 = F.avg_pool2d(x4, kernel_size=k, stride=1)
    return x4.reshape(b, c, h, w).permute(0, 2, 3, 1)


class RadianceMultipassRelight:
    CATEGORY = "FXTD STUDIOS/Radiance/VFX"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "albedo": ("IMAGE",),
                "normal_map": ("IMAGE",),
            },
            "optional": {
                "beauty": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "specular": ("IMAGE",),
                "ao": ("IMAGE",),
                "alpha": ("IMAGE",),
                "shadow_mask": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "normal_convention": (_NORMAL_INPUTS, {"default": "OpenGL (Y-Up)"}),
                "light_type": (_LIGHT_TYPES, {"default": "Directional"}),
                "light_x": ("FLOAT", {"default": -0.35, "min": -10.0, "max": 10.0, "step": 0.01}),
                "light_y": ("FLOAT", {"default": 0.45, "min": -10.0, "max": 10.0, "step": 0.01}),
                "light_z": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "light_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "light_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "light_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
                "ambient": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 4.0, "step": 0.01}),
                "specular_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "depth_scale": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "mix_with_beauty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("relit", "diffuse_light", "specular_light", "lighting", "alpha", "relight_info")
    FUNCTION = "relight"

    def relight(
        self,
        albedo: torch.Tensor,
        normal_map: torch.Tensor,
        beauty: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
        metallic: Optional[torch.Tensor] = None,
        specular: Optional[torch.Tensor] = None,
        ao: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        shadow_mask: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        normal_convention: str = "OpenGL (Y-Up)",
        light_type: str = "Directional",
        light_x: float = -0.35,
        light_y: float = 0.45,
        light_z: float = 1.0,
        light_r: float = 1.0,
        light_g: float = 1.0,
        light_b: float = 1.0,
        intensity: float = 1.0,
        ambient: float = 0.03,
        specular_intensity: float = 1.0,
        depth_scale: float = 10.0,
        mix_with_beauty: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        batch, height, width, _ = albedo.shape
        device = albedo.device
        _perf = perf_start(device)
        base = _match_image(albedo, batch, height, width, 3).to(device=device).clamp(min=0.0)
        normals = _decode_normal_map(
            _match_image(normal_map, batch, height, width, 3).to(device=device),
            normal_convention,
        )

        rough = _scalar_pass(roughness, batch, height, width, 0.5, device).clamp(0.045, 1.0)
        metal = _scalar_pass(metallic, batch, height, width, 0.0, device)
        spec = _scalar_pass(specular, batch, height, width, 0.5, device)
        occlusion = _scalar_pass(ao, batch, height, width, 1.0, device)
        alpha_s = _scalar_pass(alpha, batch, height, width, 1.0, device)
        shadow = _scalar_pass(shadow_mask, batch, height, width, 0.0, device)
        visibility = (1.0 - shadow).clamp(0.0, 1.0)

        if light_type == "Point":
            positions = _view_positions(batch, height, width, device, depth_map, depth_scale)
            light_pos = _color_tensor(light_x, light_y, light_z, device)
            l_vec = light_pos - positions
            dist2 = (l_vec * l_vec).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            light_dir = _normalize(l_vec)
            attenuation = 1.0 / (1.0 + dist2 * 0.08)
        else:
            light_dir = _normalize(_color_tensor(light_x, light_y, light_z, device))
            light_dir = light_dir.expand(batch, height, width, 3)
            attenuation = 1.0

        view_dir = _normalize(_color_tensor(0.0, 0.0, 1.0, device)).expand(batch, height, width, 3)
        half_vec = _normalize(light_dir + view_dir)

        ndotl = (normals * light_dir).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        ndotv = (normals * view_dir).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        ndoth = (normals * half_vec).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        vdoth = (view_dir * half_vec).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)

        light_color = _color_tensor(light_r, light_g, light_b, device)
        direct_scalar = ndotl * visibility.unsqueeze(-1) * attenuation * float(intensity)
        diffuse_light = light_color * direct_scalar

        rough_v = rough.unsqueeze(-1)
        alpha_ggx = (rough_v * rough_v).clamp(0.002, 1.0)
        alpha2 = alpha_ggx * alpha_ggx
        denom = (ndoth * ndoth * (alpha2 - 1.0) + 1.0)
        d_ggx = alpha2 / (math.pi * denom * denom + 1e-8)
        k = ((rough_v + 1.0) * (rough_v + 1.0)) / 8.0
        g_l = ndotl / (ndotl * (1.0 - k) + k + 1e-8)
        g_v = ndotv / (ndotv * (1.0 - k) + k + 1e-8)
        f0_dielectric = 0.04 * spec.unsqueeze(-1)
        f0 = f0_dielectric * (1.0 - metal.unsqueeze(-1)) + base * metal.unsqueeze(-1)
        fresnel = f0 + (1.0 - f0) * torch.pow((1.0 - vdoth).clamp(0.0, 1.0), 5.0)
        spec_brdf = (d_ggx * g_l * g_v * fresnel) / (4.0 * ndotl * ndotv + 1e-6)
        specular_light = (
            spec_brdf
            * ndotl
            * light_color
            * visibility.unsqueeze(-1)
            * attenuation
            * float(intensity)
            * float(specular_intensity)
        ).clamp(min=0.0)

        ambient_light = light_color * float(ambient) * occlusion.unsqueeze(-1)
        diffuse = base * (1.0 - metal.unsqueeze(-1)) * (diffuse_light + ambient_light)
        relit = (diffuse + specular_light).clamp(min=0.0)
        relit = relit * alpha_s.unsqueeze(-1)

        if beauty is not None and mix_with_beauty > 0.0:
            src = _match_image(beauty, batch, height, width, 3).to(device=device).clamp(min=0.0)
            mix = float(max(0.0, min(1.0, mix_with_beauty)))
            src = src * alpha_s.unsqueeze(-1)
            relit = relit * (1.0 - mix) + src * mix

        lighting = (diffuse_light + ambient_light + specular_light).clamp(min=0.0)
        alpha_img = alpha_s.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

        info = {
            "mode": "real_pbr_relight",
            "required_passes": ["albedo", "normal_map"],
            "optional_passes_used": {
                "roughness": roughness is not None,
                "metallic": metallic is not None,
                "specular": specular is not None,
                "ao": ao is not None,
                "alpha": alpha is not None,
                "shadow_mask": shadow_mask is not None,
                "depth_map": depth_map is not None,
            },
            "missing_optional_defaults": {
                "roughness": 0.5 if roughness is None else None,
                "metallic": 0.0 if metallic is None else None,
                "specular": 0.5 if specular is None else None,
                "ao": 1.0 if ao is None else None,
                "alpha": 1.0 if alpha is None else None,
                "shadow_mask": 0.0 if shadow_mask is None else None,
            },
            "light_type": light_type,
            "normal_convention": normal_convention,
            "note": "This node consumes supplied utility/PBR passes; it does not extract or hallucinate missing passes from beauty.",
        }

        perf_finish(logger, "Multipass Relight", _perf, device)
        return (
            relit.contiguous(),
            diffuse_light.contiguous(),
            specular_light.contiguous(),
            lighting.contiguous(),
            alpha_img.contiguous(),
            json.dumps(info, indent=2),
        )


class RadianceMultipassComposite:
    CATEGORY = "FXTD STUDIOS/Radiance/VFX"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "foreground": ("IMAGE",),
                "alpha": ("IMAGE",),
            },
            "optional": {
                "background": ("IMAGE",),
                "relit_foreground": ("IMAGE",),
                "foreground_depth": ("IMAGE",),
                "background_depth": ("IMAGE",),
                "shadow_mask": ("IMAGE",),
                "alpha_invert": ("BOOLEAN", {"default": False}),
                "premultiplied_input": ("BOOLEAN", {"default": False}),
                "depth_near_is_white": ("BOOLEAN", {"default": True}),
                "depth_bias": ("FLOAT", {"default": 0.01, "min": -0.25, "max": 0.25, "step": 0.001}),
                "shadow_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_wrap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_wrap_radius": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("composite", "premultiplied_foreground", "holdout_mask", "depth_matte", "comp_info")
    FUNCTION = "composite"

    def composite(
        self,
        foreground: torch.Tensor,
        alpha: torch.Tensor,
        background: Optional[torch.Tensor] = None,
        relit_foreground: Optional[torch.Tensor] = None,
        foreground_depth: Optional[torch.Tensor] = None,
        background_depth: Optional[torch.Tensor] = None,
        shadow_mask: Optional[torch.Tensor] = None,
        alpha_invert: bool = False,
        premultiplied_input: bool = False,
        depth_near_is_white: bool = True,
        depth_bias: float = 0.01,
        shadow_strength: float = 0.35,
        light_wrap: float = 0.0,
        light_wrap_radius: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        batch, height, width, _ = foreground.shape
        device = foreground.device
        _perf = perf_start(device)
        fg_src = relit_foreground if relit_foreground is not None else foreground
        fg = _match_image(fg_src, batch, height, width, 3).to(device=device).clamp(min=0.0)
        matte = _scalar_pass(alpha, batch, height, width, 1.0, device)
        if alpha_invert:
            matte = 1.0 - matte

        if background is None:
            bg = torch.zeros((batch, height, width, 3), device=device, dtype=torch.float32)
        else:
            bg = _match_image(background, batch, height, width, 3).to(device=device).clamp(min=0.0)

        if shadow_mask is not None and shadow_strength > 0.0:
            sh = _scalar_pass(shadow_mask, batch, height, width, 0.0, device)
            bg = bg * (1.0 - sh.unsqueeze(-1) * float(shadow_strength)).clamp(0.0, 1.0)

        front_mask = torch.ones((batch, height, width), device=device, dtype=torch.float32)
        if foreground_depth is not None and background_depth is not None:
            fg_z = _scalar_pass(foreground_depth, batch, height, width, 0.5, device, clamp=False)
            bg_z = _scalar_pass(background_depth, batch, height, width, 0.5, device, clamp=False)
            if depth_near_is_white:
                front_mask = (fg_z >= (bg_z - float(depth_bias))).float()
            else:
                front_mask = (fg_z <= (bg_z + float(depth_bias))).float()

        visible_alpha = (matte * front_mask).clamp(0.0, 1.0)

        if light_wrap > 0.0 and background is not None:
            soft_bg = _blur_bhwc(bg, int(light_wrap_radius))
            edge = (_blur_bhwc(visible_alpha.unsqueeze(-1), max(1, int(light_wrap_radius) // 2))[..., 0] - visible_alpha).clamp(0.0, 1.0)
            fg = fg + soft_bg * edge.unsqueeze(-1) * float(light_wrap)

        premult = fg if premultiplied_input else fg * visible_alpha.unsqueeze(-1)
        composite = premult + bg * (1.0 - visible_alpha).unsqueeze(-1)
        holdout = visible_alpha.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        depth_matte = front_mask.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

        info = {
            "mode": "alpha_over_depth_comp",
            "relit_foreground_used": relit_foreground is not None,
            "background_used": background is not None,
            "depth_holdout_used": foreground_depth is not None and background_depth is not None,
            "shadow_mask_used": shadow_mask is not None,
            "light_wrap_used": bool(light_wrap > 0.0 and background is not None),
            "note": "Composite uses supplied alpha/depth/shadow data only; it does not infer hidden mattes or object IDs.",
        }

        perf_finish(logger, "Multipass Composite", _perf, device)
        return (
            composite.contiguous(),
            premult.contiguous(),
            holdout.contiguous(),
            depth_matte.contiguous(),
            json.dumps(info, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "RadianceMultipassRelight": RadianceMultipassRelight,
    "RadianceMultipassComposite": RadianceMultipassComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMultipassRelight": "Multipass Relight",
    "RadianceMultipassComposite": "Multipass Composite",
}
