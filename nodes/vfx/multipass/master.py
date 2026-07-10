import os
import json
import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from ....performance import perf_finish, perf_start
from ....core.system.path_utils import get_safe_output_dir

# Core library imports
from .core import (
    _to_3ch_image, _luminance, _LUMA_WEIGHTS, _threshold_mask,
    _guided_filter_diffuse, _scharr_edges, _colorfulness,
    _ssao_multisampled, _reflection_mask, _albedo_retinex,
    _emission_glow, _roughness_from_specular, _transmission_mask,
    _optical_flow_lk, _flow_to_hsv_image, _object_id_matte,
    _depth_anything_v2_infer, _normal_from_dsine, _surface_normals_gradient,
    _curvature_from_normals, _world_position_from_depth,
    _AUTO_DEPTH_CHOICES, _NORMAL_CONVENTIONS, _DA_CHOICE_TO_KEY,
    _highpass_filter, _metallic_mask
)

logger = logging.getLogger("radiance.vfx.multipass.master")

# Try importing EXR writer utilities from io/hdr
try:
    from ....hdr.io import write_exr_multipart, write_exr_openexr, check_openexr_available
    _HAS_EXR = True
except ImportError:
    try:
        from hdr.io import write_exr_multipart, write_exr_openexr, check_openexr_available  # type: ignore[import]
        _HAS_EXR = True
    except ImportError:
        _HAS_EXR = False

# Try importing ComfyUI folder_paths
try:
    import folder_paths  # type: ignore
    _HAS_FOLDER_PATHS = True
except ImportError:
    _HAS_FOLDER_PATHS = False


def _workflow_metadata(prompt=None, extra_pnginfo=None) -> Dict[str, Any]:
    """
    Build the ComfyUI provenance metadata dict ({"prompt": json, "workflow":
    json, ...}) exactly as core SaveImage embeds into PNG tEXt chunks and as
    RadianceWrite embeds into EXR headers, so multipass/AOV EXRs carry the
    workflow like a saved PNG does too. Values are JSON strings; write_exr_openexr
    / write_exr_multipart store "workflow"/"prompt" as unprefixed EXR string
    attributes (see hdr/io.py).
    """
    meta: Dict[str, Any] = {}
    try:
        if prompt is not None:
            meta["prompt"] = json.dumps(prompt)
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                meta[str(key)] = json.dumps(value)
    except (TypeError, ValueError) as exc:
        logger.warning("[EXR Passes Writer] workflow metadata not serialisable: %s", exc)
    return meta


def _generate_cryptomatte_manifest(K: int) -> str:
    import json
    phi = 1.6180339887
    manifest = {}
    for k in range(K):
        hue = (k * phi) % 1.0
        sat, val = 0.85, 0.90
        h6  = hue * 6.0
        idx = int(h6) % 6
        f   = h6 - int(h6)
        p_  = val * (1 - sat)
        q_  = val * (1 - f * sat)
        t_  = val * (1 - (1 - f) * sat)
        lut = [(val,t_,p_),(q_,val,p_),(p_,val,t_),(p_,q_,val),(t_,p_,val),(val,p_,q_)]
        r, g, b = lut[idx]
        
        # Convert float RGB to 8-bit hex color string
        hex_color = f"{int(round(r*255)):02x}{int(round(g*255)):02x}{int(round(b*255)):02x}"
        manifest[f"cluster_{k}"] = hex_color
    return json.dumps(manifest)


def _image_to_3ch(image: torch.Tensor) -> torch.Tensor:
    x = image.float()
    if x.shape[-1] == 3:
        return x.contiguous()
    if x.shape[-1] > 3:
        return x[..., :3].contiguous()
    if x.shape[-1] == 1:
        return x.expand(-1, -1, -1, 3).contiguous()
    pad = x[..., -1:].expand(-1, -1, -1, 3 - x.shape[-1])
    return torch.cat([x, pad], dim=-1).contiguous()


def _pass_channels_for_multilayer(name: str, image: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]

    channels: Dict[str, np.ndarray] = {}
    count = arr.shape[2] if arr.ndim == 3 else 1
    if name == "beauty":
        names = ["R", "G", "B", "A"][:count]
    elif name == "depth":
        # ALBABIT-FIX: was ["depth.R","depth.G","depth.B"] since extract() always
        # triples depth to 3 identical channels for ComfyUI's IMAGE type -- no
        # compositor auto-recognizes "depth.R/G/B" as a Z-depth buffer. Bare "Z"
        # (matching write_exr_multipart()'s already-correct convention) is what
        # Nuke/Fusion look for; the duplicate channels carry no extra data anyway.
        names = ["Z"]
    elif name == "normal":
        names = ["normal.NX", "normal.NY", "normal.NZ", "normal.A"][:count]
    else:
        names = [f"{name}.{ch}" for ch in ["R", "G", "B", "A"][:count]]

    for idx, channel_name in enumerate(names):
        channels[channel_name] = arr[..., idx if idx < count else 0]
    return channels


def _write_exr_singlepart_multilayer(
    filepath: str,
    parts: Dict[str, np.ndarray],
    bit_depth: str,
    compression: str,
    metadata: Dict[str, Any],
) -> bool:
    channels: Dict[str, np.ndarray] = {}
    for name, image in parts.items():
        if image is None:
            continue
        channels.update(_pass_channels_for_multilayer(name, image))

    if not channels:
        return False

    pixel_type = "HALF" if "16" in bit_depth else "FLOAT"
    write_exr_openexr(filepath, channels, compression, pixel_type, metadata)
    return True


class RadianceMultipassMaster:
    CATEGORY = "FXTD STUDIOS/Radiance/VFX"
    DESCRIPTION = "Master VFX Multipass Extractor: extracts 18+ high-fidelity physical, utility, and lighting passes from a single beauty image."

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "beauty": ("IMAGE",),
            },
            "optional": {
                "depth_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "prev_frame": ("IMAGE",),
                
                # General Settings
                "luma_weights": (list(_LUMA_WEIGHTS.keys()), {"default": "Rec.709 / sRGB"}),
                
                # Depth (Auto-download) Settings
                "auto_depth_model": (_AUTO_DEPTH_CHOICES, {"default": "disabled"}),
                "depth_near_is_white": ("BOOLEAN", {"default": True}),
                "depth_scale": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "fov_degrees": ("FLOAT", {"default": 60.0, "min": 10.0, "max": 120.0, "step": 1.0}),
                
                # Normals Settings
                "dsine_model_path": ("STRING", {"default": "auto"}),
                "normal_strength": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "normal_convention": (_NORMAL_CONVENTIONS, {"default": "OpenGL (Y-Up)"}),
                
                # PBR Settings
                "albedo_shading_radius": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 300.0, "step": 5.0}),
                "albedo_eps": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0005}),
                "specular_floor": ("BOOLEAN", {"default": True}),
                "roughness_fine_radius": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "roughness_coarse_radius": ("FLOAT", {"default": 15.0, "min": 3.0, "max": 60.0, "step": 1.0}),
                "transmission_sensitivity": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.25}),
                
                # Highpass Settings
                "highpass_radius": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.5}),
                "highpass_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "highpass_contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                
                # Lighting & Mask Settings
                "shadow_threshold": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.60, "step": 0.01}),
                "highlight_threshold": ("FLOAT", {"default": 0.75, "min": 0.40, "max": 1.0, "step": 0.01}),
                "mask_softness": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.50, "step": 0.01}),
                "ao_radius": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "ao_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
                "ao_samples": ("INT", {"default": 8, "min": 4, "max": 32, "step": 4}),
                
                # Motion & ID Settings
                "lk_window_radius": ("INT", {"default": 7, "min": 1, "max": 32}),
                "motion_coherence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05}),
                "object_id_segments": ("INT", {"default": 16, "min": 2, "max": 64}),
                "object_id_spatial_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = (
        "RADIANCE_PASSES", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", 
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", 
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE"
    )
    RETURN_NAMES = (
        "passes", "beauty", "albedo", "normal_map", "depth", "roughness", "specular", 
        "metallic", "ao", "emission", "transmission", "highpass", "world_position", 
        "curvature", "shadow_mask", "midtone_mask", "highlight_mask", "reflection_mask",
        "motion_vector", "segmentation_id"
    )
    
    FUNCTION = "extract"

    def extract(
        self,
        beauty: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        normal_map: Optional[torch.Tensor] = None,
        prev_frame: Optional[torch.Tensor] = None,
        luma_weights: str = "Rec.709 / sRGB",
        auto_depth_model: str = "disabled",
        depth_near_is_white: bool = True,
        depth_scale: float = 10.0,
        fov_degrees: float = 60.0,
        dsine_model_path: str = "auto",
        normal_strength: float = 2.0,
        normal_convention: str = "OpenGL (Y-Up)",
        albedo_shading_radius: float = 80.0,
        albedo_eps: float = 0.001,
        specular_floor: bool = True,
        roughness_fine_radius: float = 2.0,
        roughness_coarse_radius: float = 15.0,
        transmission_sensitivity: float = 2.0,
        highpass_radius: float = 8.0,
        highpass_strength: float = 1.0,
        highpass_contrast: float = 1.0,
        shadow_threshold: float = 0.20,
        highlight_threshold: float = 0.75,
        mask_softness: float = 0.15,
        ao_radius: float = 15.0,
        ao_strength: float = 1.0,
        ao_samples: int = 8,
        lk_window_radius: int = 7,
        motion_coherence: float = 0.5,
        object_id_segments: int = 16,
        object_id_spatial_weight: float = 0.25,
    ) -> Tuple:
        img = _image_to_3ch(beauty)
        B, H, W, _ = img.shape
        device = beauty.device
        _perf = perf_start(device)
        weights = _LUMA_WEIGHTS.get(luma_weights, _LUMA_WEIGHTS["Rec.709 / sRGB"])
        luma = _luminance(img, weights)

        # ── 1. Depth Map ──────────────────────────────────────────────────────
        depth_scalar: Optional[torch.Tensor] = None
        if depth_map is not None:
            d_in = depth_map.float()
            if d_in.shape[1] != H or d_in.shape[2] != W:
                d_in = _to_3ch_image(
                    F.interpolate(d_in[..., 0].unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
                )
            if d_in.shape[0] != B:
                d_in = d_in[:B] if d_in.shape[0] >= B else d_in.expand(B, -1, -1, -1)
            pass_depth = d_in.contiguous()
            depth_scalar = pass_depth[..., 0]
        elif auto_depth_model != "disabled":
            da_key = _DA_CHOICE_TO_KEY.get(auto_depth_model)
            if da_key:
                auto_d = _depth_anything_v2_infer(img, model_key=da_key)
                if auto_d is not None:
                    pass_depth = _to_3ch_image(auto_d.to(device))
                    depth_scalar = auto_d.to(device)
                else:
                    pass_depth = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)
            else:
                pass_depth = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)
        else:
            pass_depth = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)

        # ── 2. Normals ────────────────────────────────────────────────────────
        if normal_map is not None:
            n_in = normal_map.float()
            if n_in.shape[1] != H or n_in.shape[2] != W:
                n_in = F.interpolate(n_in.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            if n_in.shape[0] != B:
                n_in = n_in[:B] if n_in.shape[0] >= B else n_in.expand(B, -1, -1, -1)
            pass_normal = n_in.contiguous()
        else:
            dsine_result = _normal_from_dsine(img, dsine_model_path, normal_convention)
            if dsine_result is not None:
                pass_normal = dsine_result
            else:
                pass_normal = _surface_normals_gradient(luma, normal_strength, normal_convention)

        # ── 3. Geometry Utilities ─────────────────────────────────────────────
        pass_curvature = _curvature_from_normals(pass_normal)

        if depth_scalar is not None:
            pass_world_pos = _world_position_from_depth(depth_scalar, fov_degrees, depth_scale, depth_near_is_white)
        else:
            pass_world_pos = torch.full((B, H, W, 3), 0.5, device=device, dtype=torch.float32)

        # ── 4. Diffuse & Specular ─────────────────────────────────────────────
        pass_diffuse = _guided_filter_diffuse(img, 20.0, eps=0.01)
        pass_specular = img - pass_diffuse
        if specular_floor:
            pass_specular = pass_specular.clamp(min=0.0)

        # ── 5. Edge & Colorfulness ────────────────────────────────────────────
        edge_map = _scharr_edges(luma)
        colorf_map = _colorfulness(img, weights)

        # ── 6. Tone Masks ─────────────────────────────────────────────────────
        shadow_m = _threshold_mask(luma, shadow_threshold, mask_softness, above=False)
        highlight_m = _threshold_mask(luma, highlight_threshold, mask_softness, above=True)
        midtone_m = (1.0 - shadow_m - highlight_m).clamp(min=0.0)
        _sum = (shadow_m + midtone_m + highlight_m).clamp(min=1e-8)
        
        pass_shadow = _to_3ch_image(shadow_m / _sum)
        pass_midtone = _to_3ch_image(midtone_m / _sum)
        pass_highlight = _to_3ch_image(highlight_m / _sum)

        # ── 7. Lighting & Occlusion ───────────────────────────────────────────
        if depth_scalar is not None and ao_strength > 0.0:
            ao_map = _ssao_multisampled(depth_scalar, pass_normal, ao_radius, ao_strength, ao_samples, depth_near_is_white)
            pass_ao = _to_3ch_image(ao_map)
        else:
            pass_ao = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)

        pass_reflection = _reflection_mask(pass_specular, colorf_map)

        # ── 8. Material & PBR Passes ──────────────────────────────────────────
        pass_metallic = _metallic_mask(img, pass_specular, colorf_map)
        pass_albedo = _albedo_retinex(img, luma, shading_radius=albedo_shading_radius, eps=albedo_eps)
        pass_emission = _emission_glow(luma, colorf_map, radius=albedo_shading_radius * 0.375, boost=1.5)
        pass_roughness = _roughness_from_specular(pass_specular, fine_radius=roughness_fine_radius, coarse_radius=roughness_coarse_radius)
        pass_transmission = _transmission_mask(img, luma, edge_map, highlight_m, colorf_map, sensitivity=transmission_sensitivity)

        # ── 9. Highpass Filter ────────────────────────────────────────────────
        pass_highpass = _highpass_filter(img, highpass_radius, highpass_strength, highpass_contrast)

        # ── 10. Motion Vectors ────────────────────────────────────────────────
        if prev_frame is not None:
            pf_in = prev_frame.float()
            if pf_in.shape[1] != H or pf_in.shape[2] != W:
                pf_in = F.interpolate(pf_in.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            if pf_in.shape[0] != B:
                pf_in = pf_in[:B] if pf_in.shape[0] >= B else pf_in.expand(B, -1, -1, -1)
            prev_luma = _luminance(pf_in, weights)
            flow_u, flow_v = _optical_flow_lk(luma, prev_luma, window_radius=lk_window_radius)
            
            # Apply temporal coherence / EMA smoothing to reduce sub-pixel jitter
            if motion_coherence > 0.0:
                smooth_u = torch.zeros_like(flow_u)
                smooth_v = torch.zeros_like(flow_v)
                
                for b in range(B):
                    if b == 0:
                        smooth_u[0] = flow_u[0]
                        smooth_v[0] = flow_v[0]
                    else:
                        smooth_u[b] = flow_u[b] * (1.0 - motion_coherence) + smooth_u[b-1] * motion_coherence
                        smooth_v[b] = flow_v[b] * (1.0 - motion_coherence) + smooth_v[b-1] * motion_coherence
                
                flow_u = smooth_u
                flow_v = smooth_v
                
            pass_motion = _flow_to_hsv_image(flow_u, flow_v)
        else:
            pass_motion = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)

        # ── 11. Object ID / Cryptomatte ───────────────────────────────────────
        pass_object_id = _object_id_matte(img, luma, n_segments=object_id_segments, spatial_weight=object_id_spatial_weight)

        out_beauty = img.contiguous()
        out_albedo = pass_albedo.contiguous()
        out_normal = pass_normal.contiguous()
        out_depth = pass_depth.contiguous()
        out_roughness = pass_roughness.contiguous()
        out_specular = pass_specular.contiguous()
        out_metallic = pass_metallic.contiguous()
        out_ao = pass_ao.contiguous()
        out_emission = pass_emission.contiguous()
        out_transmission = pass_transmission.contiguous()
        out_highpass = pass_highpass.contiguous()
        out_world_pos = pass_world_pos.contiguous()
        out_curvature = pass_curvature.contiguous()
        out_shadow = pass_shadow.contiguous()
        out_midtone = pass_midtone.contiguous()
        out_highlight = pass_highlight.contiguous()
        out_reflection = pass_reflection.contiguous()
        out_motion = pass_motion.contiguous()
        out_object_id = pass_object_id.contiguous()

        # ── Bundle dictionary ──
        passes_dict = {
            "beauty": out_beauty,
            "albedo": out_albedo,
            "normal": out_normal,
            "depth": out_depth,
            "roughness": out_roughness,
            "specular": out_specular,
            "metallic": out_metallic,
            "ao": out_ao,
            "emission": out_emission,
            "transmission": out_transmission,
            "highpass": out_highpass,
            "world_position": out_world_pos,
            "curvature": out_curvature,
            "shadow_mask": out_shadow,
            "midtone_mask": out_midtone,
            "highlight_mask": out_highlight,
            "reflection_mask": out_reflection,
            "motion_vector": out_motion,
            "object_id": out_object_id,
        }

        perf_finish(logger, "Multipass Extract", _perf, device)
        return (
            passes_dict, out_beauty, out_albedo, out_normal, out_depth, out_roughness, out_specular,
            out_metallic, out_ao, out_emission, out_transmission, out_highpass, out_world_pos,
            out_curvature, out_shadow, out_midtone, out_highlight, out_reflection, out_motion, out_object_id
        )


class RadianceEXRPassesWriter:
    CATEGORY = "FXTD STUDIOS/Radiance/Load & Save"
    DESCRIPTION = "Write all passes inside the RADIANCE_PASSES bundle into a single-part multilayer or true multi-part OpenEXR file."
    FUNCTION = "write_passes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        compressions = [
            "ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed",
            "PXR24", "B44", "B44A", "DWAA", "DWAB"
        ]
        return {
            "required": {
                "passes": ("RADIANCE_PASSES",),
                "filename_prefix": ("STRING", {"default": "radiance_vfx_passes"}),
                "bit_depth": (["16-bit Half Float", "32-bit Float"], {"default": "16-bit Half Float"}),
                "compression": (compressions, {"default": "ZIP"}),
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}),
                "remote_path": ("STRING", {"default": ""}),
                "frame_index": ("INT", {"default": 1001, "min": 0, "max": 999999}),
                "exr_layout": (["Single-part multilayer", "Multi-part"], {"default": "Single-part multilayer"}),
                "custom_metadata": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def write_passes(
        self,
        passes: Dict[str, torch.Tensor],
        filename_prefix: str,
        bit_depth: str = "16-bit Half Float",
        compression: str = "ZIP",
        output_path: str = "",
        remote_path: str = "",
        frame_index: int = 1001,
        exr_layout: str = "Single-part multilayer",
        custom_metadata: str = "",
        prompt: Optional[Any] = None,
        extra_pnginfo: Optional[Any] = None,
    ) -> Tuple[str]:
        import datetime
        import tempfile
        import shutil

        # Determine target directory
        if _HAS_FOLDER_PATHS:
            base_dir = folder_paths.get_output_directory()
        else:
            base_dir = tempfile.gettempdir()

        # ALBABIT-FIX: was hand-rolled (os.path.isabs check + manual join),
        # with no anti-traversal protection for relative output_path values.
        # get_safe_output_dir() is the same helper RadianceWrite/RadianceEXRMultiPart
        # already use for this exact pattern.
        out_dir = get_safe_output_dir(base_dir, output_path.strip(), allow_absolute=True)

        # Standard Metadata dictionary
        meta: Dict[str, Any] = {
            "software": "Radiance VFX Multipass v3.1",
            "created": datetime.datetime.now().isoformat(),
        }

        # Custom user-defined metadata lines
        for line in custom_metadata.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()

        # ComfyUI workflow/prompt provenance — same convention as RadianceWrite,
        # so multipass EXRs can be dragged back into ComfyUI to restore the graph.
        meta.update(_workflow_metadata(prompt, extra_pnginfo))

        # Retrieve beauty shape to detect batch size
        beauty = passes.get("beauty")
        if beauty is None:
            raise ValueError("[EXR Passes Writer] passes dictionary must contain a 'beauty' tensor.")

        B = beauty.shape[0]

        # Extract frames as float32 numpy arrays (H, W, C)
        def _get_frame(tensor: torch.Tensor, idx: int) -> np.ndarray:
            frame_tensor = tensor[idx] if tensor.dim() == 4 else tensor
            return frame_tensor.float().cpu().numpy()

        saved_paths: List[str] = []

        # Iterate over each frame in batch
        for b in range(B):
            frame_num = str(frame_index + b).zfill(4)
            filepath = os.path.join(out_dir, f"{filename_prefix}.{frame_num}.exr")
            
            parts: Dict[str, np.ndarray] = {}
            for name, tensor in passes.items():
                parts[name] = _get_frame(tensor, b)

            # Normalise compression label
            comp = "None" if compression.lower() == "uncompressed" else compression

            # Write the EXR file
            if _HAS_EXR and exr_layout == "Single-part multilayer":
                ok = _write_exr_singlepart_multilayer(filepath, parts, bit_depth, comp, meta)
            elif _HAS_EXR and write_exr_multipart is not None:
                ok = write_exr_multipart(filepath, parts, bit_depth, comp, meta)
            else:
                raise RuntimeError(
                    "[EXR Passes Writer] EXR writing modules not found or import failed."
                )

            if not ok:
                raise RuntimeError(f"[EXR Passes Writer] Failed to write EXR file: {filepath}")

            logger.info("[EXR Passes Writer] Saved %d layers → %s", len(parts), filepath)
            saved_paths.append(filepath)

            # Best effort copy to remote/NAS path
            if remote_path.strip():
                try:
                    dest = os.path.join(remote_path.strip(), f"{filename_prefix}.{frame_num}.exr")
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(filepath, dest)
                    logger.info("[EXR Passes Writer] Copied to remote → %s", dest)
                except Exception as ex:
                    logger.warning("[EXR Passes Writer] Remote copy failed: %s", ex)

        return (saved_paths[0] if saved_paths else "",)


NODE_CLASS_MAPPINGS = {
    "RadianceMultipassMaster": RadianceMultipassMaster,
    "RadianceEXRPassesWriter": RadianceEXRPassesWriter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMultipassMaster": "Multipass Extract",
    "RadianceEXRPassesWriter": "Write EXR Passes"
}
