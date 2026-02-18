
import torch
import unittest
import numpy as np
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock ComfyUI dependencies
from unittest.mock import MagicMock
sys.modules["folder_paths"] = MagicMock()
sys.modules["folder_paths"].get_input_directory = MagicMock(return_value="/tmp")
sys.modules["folder_paths"].get_output_directory = MagicMock(return_value="/tmp")
sys.modules["folder_paths"].get_temp_directory = MagicMock(return_value="/tmp")
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.utils"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()

sys.modules["comfy.sd"] = MagicMock()
sys.modules["comfy.samplers"] = MagicMock()

# Now import radiance modules
from radiance.hdr.color import Float32ColorCorrect
from radiance.nodes_grade import RadianceGrade
from radiance.film.grain import RadianceFilmGrain
from radiance.film.camera import RadianceDepthOfField, RadianceMotionBlur, RadianceRollingShutter
from radiance.image.upscale import RadianceProUpscale

class TestSuperRange(unittest.TestCase):
    def setUp(self):
        # Create a test pattern with extreme values
        # [Super Black (-5), Black (0), Mid (0.5), White (1), Super White (5)]
        self.device = torch.device("cpu")
        self.super_tensor = torch.tensor([
            [-5.0, 0.0, 0.5, 1.0, 5.0],
            [-5.0, 0.0, 0.5, 1.0, 5.0],
            [-5.0, 0.0, 0.5, 1.0, 5.0]
        ], dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).unsqueeze(-1).repeat(1, 1, 1, 3)
        # Shape: (1, 5, 3, 3) -> Batch, Height, Width, Channels
        
    def check_range(self, output, node_name):
        min_val = output.min().item()
        max_val = output.max().item()
        
        print(f"\nNode: {node_name}")
        print(f"  Input Range: [-5.0, 5.0]")
        print(f"  Output Range: [{min_val:.4f}, {max_val:.4f}]")
        
        has_super_white = max_val > 1.1
        has_super_black = min_val < -0.1
        
        if has_super_white:
            print("  ✅ Preserves Super White")
        else:
            print("  ⚠️ CLAMPS Super White")
            
        if has_super_black:
            print("  ✅ Preserves Super Black")
        else:
            print("  ⚠️ CLAMPS Super Black (Expected for some nodes)")
            
        return min_val, max_val

    def test_grade_node(self):
        node = RadianceGrade()
        # Default settings should be passthrough
        out, _ = node.grade(self.super_tensor, 
                            preset="None (Custom)", preset_strength=0.0,
                            lift_r=0, lift_g=0, lift_b=0,
                            gamma_r=1, gamma_g=1, gamma_b=1,
                            gain_r=1, gain_g=1, gain_b=1,
                            offset_r=0, offset_g=0, offset_b=0,
                            contrast=1, pivot=0.5, saturation=1)
        self.check_range(out, "RadianceGrade (Default)")
        
        # Test with Gain up (simulated exposure)
        out_exp, _ = node.grade(self.super_tensor, 
                                preset="None (Custom)", preset_strength=0.0,
                                gain_r=2.0, gain_g=2.0, gain_b=2.0)
        self.check_range(out_exp, "RadianceGrade (Gain x2)")
        self.assertTrue(out_exp.max() > 9.0, "Gain x2 should increase super whites (5.0 -> 10.0)")

    def test_film_grain(self):
        node = RadianceFilmGrain()
        # HDR Safe = True
        out, = node.apply_grain(self.super_tensor, preset="Custom", intensity=0.5, size=1.0, seed=0, 
                               blend_mode="Overlay", halation_strength=0.0, hdr_safe=True)
        self.check_range(out, "FilmGrain (HDR Safe)")
        self.assertTrue(out.max() > 1.0, "Film grain should preserve super whites in HDR mode")



    def test_color_correct(self):
        node = Float32ColorCorrect()
        # Correct method name is 'correct'
        out, = node.correct(self.super_tensor, exposure=0, contrast=1, brightness=0, 
                           saturation=1, gamma=1, lift_r=0, lift_g=0, lift_b=0, 
                           gain_r=1, gain_g=1, gain_b=1, luma_space="Rec.709 / sRGB", clamp_output=False)
        self.check_range(out, "Float32ColorCorrect (Pass)")
        
        # Test gamma with super blacks (should sign-preserve)
        out_gamma, = node.correct(self.super_tensor, gamma=2.0)
        self.check_range(out_gamma, "Float32ColorCorrect (Gamma 2.0)")
        self.assertTrue(out_gamma.min() < -0.1, "Gamma should preserve negative values via sign-preserving power")

    def test_camera_nodes(self):
        # 1. Depth of Field
        dof_node = RadianceDepthOfField()
        out_dof, = dof_node.apply_dof(self.super_tensor, blur_amount=10.0, 
                                     focus_distance=0.5, focus_range=0.1, 
                                     bokeh_shape="Circle", highlight_boost=1.0, 
                                     foreground_blur=True, use_gpu=False)
        self.check_range(out_dof, "RadianceDepthOfField")
        self.assertTrue(out_dof.max() > 1.0, "DOF should preserve highlights")

        # 2. Motion Blur
        mb_node = RadianceMotionBlur()
        out_mb, = mb_node.apply_motion_blur(self.super_tensor, blur_type="Directional", amount=10.0,
                                           angle=0, center_x=0.5, center_y=0.5, samples=4, use_gpu=False)
        self.check_range(out_mb, "RadianceMotionBlur")
        
        # 3. Rolling Shutter
        rs_node = RadianceRollingShutter()
        out_rs, = rs_node.apply_rolling_shutter(self.super_tensor, skew_amount=10.0, 
                                               shutter_direction="Vertical", wobble_frequency=0, 
                                               wobble_amplitude=0, flash_band_position=-1, 
                                               flash_band_width=0.1, use_gpu=False)
        self.check_range(out_rs, "RadianceRollingShutter")

    def test_upscale_node(self):
        node = RadianceProUpscale()
        # Scale 1.0 to check pixel values directly
        out, w, h, info = node.upscale(self.super_tensor, scale_factor=1.0, preset="Custom", 
                                      method="nearest", sharpening=0.0, sharpen_radius=1.0, 
                                      detail_enhancement=0.0, antialiasing=0.0, 
                                      input_color_space="sRGB", process_in_linear=False, 
                                      use_tiles=False, tile_size=512, tile_overlap=64)
        self.check_range(out, "RadianceProUpscale (Nearest)")


if __name__ == '__main__':
    unittest.main()
