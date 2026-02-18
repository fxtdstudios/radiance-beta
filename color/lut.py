
import torch
import numpy as np
import os
import hashlib
import threading
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

# Setup logger
logger = logging.getLogger("radiance.color.lut")

@dataclass
class LUTData:
    """Container for parsed LUT information with metadata."""
    lut_tensor: torch.Tensor
    size: int
    domain_min: List[float]
    domain_max: List[float]
    filepath: str
    file_hash: str


class LUTCache:
    """Thread-safe LUT cache with bounded size to prevent memory leaks."""
    _cache: Dict[str, LUTData] = {}
    _max_cache_size: int = 32
    _lock = threading.RLock()  # Reentrant lock for thread safety
    
    @classmethod
    def get_file_hash(cls, filepath: str) -> str:
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.debug(f"Could not hash LUT file {filepath}: {e}")
            return ""
    
    @classmethod
    def get(cls, filepath: str) -> Optional[LUTData]:
        with cls._lock:
            if filepath in cls._cache:
                cached = cls._cache[filepath]
                current_hash = cls.get_file_hash(filepath)
                if current_hash == cached.file_hash:
                    return cached
                else:
                    del cls._cache[filepath]
            return None
    
    @classmethod
    def set(cls, filepath: str, lut_data: LUTData):
        with cls._lock:
            # Evict oldest if at capacity (FIFO eviction)
            while len(cls._cache) >= cls._max_cache_size:
                oldest_key = next(iter(cls._cache))
                del cls._cache[oldest_key]
                logger.debug(f"LUT cache evicted: {oldest_key}")
            cls._cache[filepath] = lut_data
    
    @classmethod
    def clear(cls):
        with cls._lock:
            cls._cache.clear()
            logger.debug("LUT cache cleared")


class RadianceLUTApply:
    """
    Apply a 3D LUT (.cube) to an image with professional-grade color transformation.
    """
    
    LOG_ENCODINGS = ["Log10", "Log2", "Natural Log (Ln)"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (cls.get_lut_files(),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "log_space": ("BOOLEAN", {"default": False, "label_on": "Log Input", "label_off": "Linear/sRGB Input"}),
            },
            "optional": {
                "log_encoding": (cls.LOG_ENCODINGS, {"default": "Log10"}),
                "clamp_output": ("BOOLEAN", {"default": False, "tooltip": "Clamp to 0-1. Disable for HDR."}),
                "interpolation": (["Trilinear", "Tetrahedral"], {"default": "Trilinear",
                                  "tooltip": "Tetrahedral is more accurate but slightly slower"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lut"
    CATEGORY = "FXTD Studios/Radiance/Color"
    
    @staticmethod
    def get_lut_files() -> List[str]:
        try:
            import folder_paths
            # Safe check avoiding KeyError if 'luts' isn't registered
            if "luts" not in folder_paths.folder_names_and_paths:
                # Attempt to register standard models/luts path
                luts_path = os.path.join(folder_paths.models_dir, "luts")
                if os.path.exists(luts_path):
                    folder_paths.add_model_folder_path("luts", luts_path)
                else:
                    # Create if it doesn't exist to prevent future errors
                    try:
                        os.makedirs(luts_path, exist_ok=True)
                        folder_paths.add_model_folder_path("luts", luts_path)
                    except Exception:
                        return ["No LUTs found"]

            luts = folder_paths.get_filename_list("luts")
            return luts if luts else ["No LUTs found"]
        except Exception as e:
            logger.debug(f"Could not load LUT file list: {e}")
            return ["No LUTs found"]

    @staticmethod
    def parse_cube_file(filepath: str) -> LUTData:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LUT file not found: {filepath}")
        
        lut_data = []
        lut_size = None
        domain_min = [0.0, 0.0, 0.0]
        domain_max = [1.0, 1.0, 1.0]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('TITLE') or line.startswith('LUT_1D_SIZE'):
                    continue
                
                if line.startswith('LUT_3D_SIZE'):
                    lut_size = int(line.split()[1])
                    continue
                if line.startswith('DOMAIN_MIN'):
                    domain_min = [float(x) for x in line.split()[1:4]]
                    continue
                if line.startswith('DOMAIN_MAX'):
                    domain_max = [float(x) for x in line.split()[1:4]]
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        lut_data.append([float(x) for x in parts[:3]])
                    except ValueError:
                        continue
        
        if not lut_data:
            raise ValueError("No valid data in LUT")

        if lut_size is None:
            lut_size = round(len(lut_data) ** (1/3))
        
        lut_array = np.array(lut_data, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3)
        lut_tensor = torch.from_numpy(lut_array)
        file_hash = LUTCache.get_file_hash(filepath)
        
        return LUTData(lut_tensor, lut_size, domain_min, domain_max, filepath, file_hash)

    @staticmethod
    def trilinear_interpolate(image, lut, lut_size, domain_min, domain_max):
        # Scale to domain
        domain_range = domain_max - domain_min
        coords = (image[..., :3] - domain_min) / domain_range
        coords = torch.clamp(coords, 0.0, 1.0) * (lut_size - 1)
        
        coords_floor = torch.floor(coords).long()
        coords_floor = torch.clamp(coords_floor, 0, lut_size - 2)
        coords_ceil = coords_floor + 1
        coords_frac = coords - coords_floor.float()
        
        x0, y0, z0 = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]
        x1, y1, z1 = coords_ceil[..., 0], coords_ceil[..., 1], coords_ceil[..., 2]
        
        c000 = lut[x0, y0, z0]
        c001 = lut[x0, y0, z1]
        c010 = lut[x0, y1, z0]
        c011 = lut[x0, y1, z1]
        c100 = lut[x1, y0, z0]
        c101 = lut[x1, y0, z1]
        c110 = lut[x1, y1, z0]
        c111 = lut[x1, y1, z1]
        
        xd, yd, zd = coords_frac[..., 0:1], coords_frac[..., 1:2], coords_frac[..., 2:3]
        
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        
        return c0 * (1 - zd) + c1 * zd

    @staticmethod
    def tetrahedral_interpolate(image, lut, lut_size, domain_min, domain_max):
        """
        Tetrahedral interpolation for 3D LUTs - more accurate than trilinear.
        
        Divides each cube into 6 tetrahedra and interpolates within the correct one.
        This provides better color accuracy, especially for large color changes.
        """
        # Scale to LUT domain
        domain_range = domain_max - domain_min
        coords = (image[..., :3] - domain_min) / domain_range
        coords = torch.clamp(coords, 0.0, 1.0) * (lut_size - 1)
        
        coords_floor = torch.floor(coords).long()
        coords_floor = torch.clamp(coords_floor, 0, lut_size - 2)
        coords_frac = coords - coords_floor.float()
        
        x0, y0, z0 = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        xd, yd, zd = coords_frac[..., 0], coords_frac[..., 1], coords_frac[..., 2]
        
        # Get corner values
        c000 = lut[x0, y0, z0]
        c001 = lut[x0, y0, z1]
        c010 = lut[x0, y1, z0]
        c011 = lut[x0, y1, z1]
        c100 = lut[x1, y0, z0]
        c101 = lut[x1, y0, z1]
        c110 = lut[x1, y1, z0]
        c111 = lut[x1, y1, z1]
        
        # Determine which tetrahedron we're in and interpolate accordingly
        # There are 6 tetrahedra in each cube, determined by the relative ordering of x, y, z
        result = torch.zeros_like(c000)
        
        # Case 1: x >= y >= z
        mask = (xd >= yd) & (yd >= zd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(m, 
                c000 + (c100 - c000) * xd.unsqueeze(-1) + 
                       (c110 - c100) * yd.unsqueeze(-1) + 
                       (c111 - c110) * zd.unsqueeze(-1),
                result)
        
        # Case 2: x >= z >= y
        mask = (xd >= zd) & (zd >= yd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(m,
                c000 + (c100 - c000) * xd.unsqueeze(-1) + 
                       (c101 - c100) * zd.unsqueeze(-1) + 
                       (c111 - c101) * yd.unsqueeze(-1),
                result)
        
        # Case 3: z >= x >= y
        mask = (zd >= xd) & (xd >= yd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(m,
                c000 + (c001 - c000) * zd.unsqueeze(-1) + 
                       (c101 - c001) * xd.unsqueeze(-1) + 
                       (c111 - c101) * yd.unsqueeze(-1),
                result)
        
        # Case 4: y >= x >= z
        mask = (yd >= xd) & (xd >= zd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(m,
                c000 + (c010 - c000) * yd.unsqueeze(-1) + 
                       (c110 - c010) * xd.unsqueeze(-1) + 
                       (c111 - c110) * zd.unsqueeze(-1),
                result)
        
        # Case 5: y >= z >= x
        mask = (yd >= zd) & (zd >= xd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(m,
                c000 + (c010 - c000) * yd.unsqueeze(-1) + 
                       (c011 - c010) * zd.unsqueeze(-1) + 
                       (c111 - c011) * xd.unsqueeze(-1),
                result)
        
        # Case 6: z >= y >= x
        mask = (zd >= yd) & (yd >= xd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(m,
                c000 + (c001 - c000) * zd.unsqueeze(-1) + 
                       (c011 - c001) * yd.unsqueeze(-1) + 
                       (c111 - c011) * xd.unsqueeze(-1),
                result)
        
        return result

    def apply_lut(self, image, lut_file, strength, log_space, log_encoding="Log10", clamp_output=False, interpolation="Trilinear"):
        if strength == 0.0: return (image,)
        
        try:
            import folder_paths
            lut_path = folder_paths.get_full_path("luts", lut_file)
        except Exception as e:
            logger.error(f"Failed to resolve LUT path for '{lut_file}': {e}")
            return (image,)
            
        if not lut_path or not os.path.exists(lut_path):
            return (image,)
            
        # Cache lookup
        lut_data = LUTCache.get(lut_path)
        if lut_data is None:
            try:
                lut_data = self.parse_cube_file(lut_path)
                LUTCache.set(lut_path, lut_data)
            except Exception as e:
                logger.warning(f"Error parsing LUT: {e}")
                return (image,)
        
        # Process
        device = image.device
        img_proc = image.clone()
        
        if log_space:
            if log_encoding == "Log10": img_proc = torch.pow(10.0, img_proc)
            elif log_encoding == "Log2": img_proc = torch.pow(2.0, img_proc)
            elif log_encoding == "Natural Log (Ln)": img_proc = torch.exp(img_proc)
            
        lut_tensor = lut_data.lut_tensor.to(device)
        d_min = torch.tensor(lut_data.domain_min, device=device)
        d_max = torch.tensor(lut_data.domain_max, device=device)
        
        # Choose interpolation method
        if interpolation == "Tetrahedral":
            result = self.tetrahedral_interpolate(img_proc, lut_tensor, lut_data.size, d_min, d_max)
        else:
            result = self.trilinear_interpolate(img_proc, lut_tensor, lut_data.size, d_min, d_max)
        
        if image.shape[-1] > 3:
            result = torch.cat([result, img_proc[..., 3:]], dim=-1)
            
        if strength < 1.0:
            result = image * (1.0 - strength) + result * strength
            
        if clamp_output:
            result = torch.clamp(result, 0.0, 1.0)
        # No min=0 clamp when disabled — inverse LUTs and gamut maps produce valid negatives
            
        return (result,)

class RadianceLUTBlend:
    """
    Blend two LUTs together with various blend modes. Allows creative combination of color grades and seamless transitions between different looks.
    """
    
    BLEND_MODES = ["Linear", "Luminosity", "Saturation", "Hue"]
    
    @classmethod
    def INPUT_TYPES(cls):
        lut_files = RadianceLUTApply.get_lut_files()
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_a": (lut_files,),
                "lut_b": (lut_files,),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "0.0 = LUT A only, 1.0 = LUT B only"}),
                "blend_mode": (cls.BLEND_MODES, {"default": "Linear"}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clamp_output": ("BOOLEAN", {"default": False, "tooltip": "Clamp to 0-1. Disable for HDR."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_luts"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Blend two LUTs with various blend modes for creative color grading."
    
    def blend_luts(self, image, lut_a, lut_b, blend_factor, blend_mode, strength=1.0, clamp_output=False):
        if strength == 0.0:
            return (image,)
        
        # Apply each LUT separately
        lut_applier = RadianceLUTApply()
        result_a = lut_applier.apply_lut(image, lut_a, 1.0, False)[0]
        result_b = lut_applier.apply_lut(image, lut_b, 1.0, False)[0]
        
        if blend_mode == "Linear":
            # Simple linear interpolation
            blended = result_a * (1.0 - blend_factor) + result_b * blend_factor
        
        elif blend_mode == "Luminosity":
            # Take luminosity from blend, color from A
            luma_a = 0.2126 * result_a[..., 0] + 0.7152 * result_a[..., 1] + 0.0722 * result_a[..., 2]
            luma_b = 0.2126 * result_b[..., 0] + 0.7152 * result_b[..., 1] + 0.0722 * result_b[..., 2]
            luma_blend = luma_a * (1.0 - blend_factor) + luma_b * blend_factor
            
            # Adjust A to match blended luminosity
            luma_a = luma_a.unsqueeze(-1)
            luma_blend = luma_blend.unsqueeze(-1)
            blended = result_a * (luma_blend / (luma_a + 1e-10))
        
        elif blend_mode == "Saturation":
            # Take saturation from blend
            luma_a = (0.2126 * result_a[..., 0] + 0.7152 * result_a[..., 1] + 0.0722 * result_a[..., 2]).unsqueeze(-1)
            luma_b = (0.2126 * result_b[..., 0] + 0.7152 * result_b[..., 1] + 0.0722 * result_b[..., 2]).unsqueeze(-1)
            
            sat_a = result_a - luma_a
            sat_b = result_b - luma_b
            sat_blend = sat_a * (1.0 - blend_factor) + sat_b * blend_factor
            
            blended = luma_a + sat_blend
        
        elif blend_mode == "Hue":
            # Take hue from B, saturation and luminosity from A
            luma_a = (0.2126 * result_a[..., 0] + 0.7152 * result_a[..., 1] + 0.0722 * result_a[..., 2]).unsqueeze(-1)
            luma_b = (0.2126 * result_b[..., 0] + 0.7152 * result_b[..., 1] + 0.0722 * result_b[..., 2]).unsqueeze(-1)
            
            # Normalize to get hue direction
            sat_a_mag = torch.sqrt(torch.sum((result_a - luma_a) ** 2, dim=-1, keepdim=True) + 1e-10)
            sat_b_dir = (result_b - luma_b) / (torch.sqrt(torch.sum((result_b - luma_b) ** 2, dim=-1, keepdim=True)) + 1e-10)
            
            # Blend hue direction
            sat_a_dir = (result_a - luma_a) / (sat_a_mag + 1e-10)
            hue_blend = sat_a_dir * (1.0 - blend_factor) + sat_b_dir * blend_factor
            hue_blend = hue_blend / (torch.sqrt(torch.sum(hue_blend ** 2, dim=-1, keepdim=True)) + 1e-10)
            
            blended = luma_a + hue_blend * sat_a_mag
        
        else:
            blended = result_a * (1.0 - blend_factor) + result_b * blend_factor
        
        # Apply overall strength
        if strength < 1.0:
            blended = image * (1.0 - strength) + blended * strength
        
        if clamp_output:
            blended = torch.clamp(blended, 0.0, 1.0)
        
        return (blended,)
